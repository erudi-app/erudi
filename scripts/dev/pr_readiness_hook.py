#!/usr/bin/env python3
"""
PreToolUse hook (Bash) for Claude Code -- double-checks a `gh pr create`
before the pull request is opened, looking for the gaps that are easy to miss:
missing documentation, missing tests, an incomplete migration, a missing
translation, a requirements file left out of sync, or leftover internal jargon
in the title/body. It is wired up in `.claude/settings.json`; nothing else
reads it, and a contributor who does not use Claude Code never sees it.

This is a double-check, not a wall. The first `gh pr create` on a branch, in a
given session, that shows a likely gap is BLOCKED (exit 2, questions on
stderr) so the gap can be fixed or dismissed as a false positive, and the same
command re-run. Every question already asked is remembered per branch, per
session, so re-running goes through -- but a gap that only *appears* after a
fix is a question that has not been asked yet, and it is asked once too. Set
`ERUDI_SKIP_PR_HOOKS=1` to turn the whole thing off.

What the checks look at is the union of the committed branch diff against the
base branch and the state of the working tree (staged, unstaged and
untracked). That union is deliberate: the most common way a pull request is
opened is one compound command -- `git add -A && git commit -m x && git push
&& gh pr create --fill` -- and the hook runs before any of it, against a HEAD
that does not yet contain the work. Looking only at commits would make the
hook blind in exactly the shape it exists for. Over-inclusion is the safe
direction here: a spurious question costs one re-run, a missed gap defeats the
hook.

Each of the six path-based checks depends on a directory that must exist in
the repository (e.g. `backend/tests/`); when that directory is absent the
check silently disables itself, which is what lets this same file be dropped
into a global hook install and behave sanely on a repository that has no
`backend/` tree at all. The seventh check (internal jargon in the PR
title/body) is a text check with no directory dependency.

Fail-open discipline: any exception, any git failure, a `cwd` that is not
inside a git repository, no resolvable base branch, an unreadable state file
-- and, above all, a state file that cannot be *written* -- all result in exit
0, silently. A marker that cannot be persisted would otherwise mean the hook
asks the same question on every single attempt while promising that the next
one goes through. A hook that blocks a contributor from opening a pull request
because of its own bug is worse than no hook at all, so `main()` is wrapped so
that no traceback can ever propagate to exit 2.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gh_pr_command import base_from_segment, find_pr_create_segment, flag_value  # noqa: E402

# Internal enumeration labels that mean something inside a working session and
# nothing to a contributor reading the pull request. Kept as a single constant
# so adding a new label is a one-line change.
#
# The bar for adding one is high, because this repository's ordinary prose is
# full of numbered technical nouns: "run it 3 times", "--batch 2048", "on
# iteration 0 the KV cache is empty", "Actions run 18234567". A pattern that
# catches those teaches contributors to ignore the hook, which costs more than
# the jargon it was meant to catch.
#
# `\bPR[0-9]\b` deliberately requires the digit to sit right against "PR" with
# no separator, so it catches internal refactor phase names like "PR1" without
# ever matching a legitimate cross-reference such as "PR #123" or "PR 456".
# `Wave`/`Phase` are matched case-sensitively, for the same reason: capitalised
# they are labels, lowercase they are ordinary words.
JARGON_PATTERNS = [
    re.compile(r"\bP[0-9]\b"),
    re.compile(r"\bPR[0-9]\b"),
    re.compile(r"\b(?:Wave|Phase)\s+\d+\b"),
    re.compile(r"\bvague\s+\d+\b", re.IGNORECASE),
    re.compile(r"\bbug[-\s]?bash\b", re.IGNORECASE),
]

# An issue reference in the PR title/body: a `#123` cross-reference, or a bare
# GitHub issues URL. Either one is enough to consider the pull request linked;
# the `issue-link` check fires only when neither appears.
ISSUE_REFERENCE_PATTERNS = [
    re.compile(r"#\d+"),
    re.compile(r"https://github\.com/[^/\s]+/[^/\s]+/issues/\d+"),
]

SKIP_ENV_VAR = "ERUDI_SKIP_PR_HOOKS"

STATE_DIR_NAME = "erudi-pr-readiness"
MARKER_MAX_AGE_SECONDS = 24 * 60 * 60

HEADER = "PR readiness double-check (each question is asked once per branch, per session):"
FOOTER = (
    "Answer these, then re-run the exact same command: none of the questions "
    "above is asked again for this branch in this session."
)


class Context(object):
    def __init__(self, repo_root, changed, added, pr_text):
        self.repo_root = repo_root
        self.changed = changed
        self.added = added
        self.pr_text = pr_text


def run_git(args, cwd):
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def git_lines(args, cwd):
    out = run_git(args, cwd)
    if out is None:
        return []
    return [line for line in out.splitlines() if line]


def resolve_repo_root(cwd):
    out = run_git(["rev-parse", "--show-toplevel"], cwd)
    if out is None:
        return None
    root = out.strip()
    return root or None


def ref_exists(ref, cwd):
    return run_git(["rev-parse", "--quiet", "--verify", ref], cwd) is not None


def distance_from_merge_base(ref, cwd):
    """Commits between merge-base(ref, HEAD) and HEAD, or None."""
    merge_base = run_git(["merge-base", ref, "HEAD"], cwd)
    if merge_base is None:
        return None
    merge_base = merge_base.strip()
    if not merge_base:
        return None
    count = run_git(["rev-list", "--count", merge_base + "..HEAD"], cwd)
    if count is None:
        return None
    try:
        return int(count.strip())
    except ValueError:
        return None


def resolve_base(cwd, explicit=None):
    """The branch this pull request would be opened against.

    An explicit `--base`/`-B` wins outright -- it is what the contributor
    typed. Otherwise the candidates are tried and the one whose merge base is
    *closest to HEAD* is chosen, which is what makes the hook usable from a
    fork: the standard contributor clones their fork, adds `upstream`,
    branches off `upstream/main` and never refreshes `origin/main`, so taking
    `origin/main` first drags every unrelated upstream commit into the diff
    and produces absurd questions on a one-line change.
    """
    if explicit and ref_exists(explicit, cwd):
        return explicit

    best = None
    best_distance = None
    for candidate in ("upstream/main", "origin/main", "main"):
        if not ref_exists(candidate, cwd):
            continue
        distance = distance_from_merge_base(candidate, cwd)
        if distance is None:
            continue
        if best_distance is None or distance < best_distance:
            best = candidate
            best_distance = distance
    return best


def current_branch(cwd):
    out = run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    if out is None:
        return None
    branch = out.strip()
    return branch or None


def changed_files(base, cwd, added_only=False):
    """Every path this pull request would carry, committed or not.

    Four sources, unioned:

    * `<base>...HEAD` -- what the branch's commits change relative to the
      merge base. Three dots, not two: with two, a base branch that advanced
      after the branch point drags every commit made on it since into the
      diff.
    * `--cached` -- staged, not yet committed.
    * the plain worktree diff -- modified, not yet staged.
    * `ls-files --others` -- untracked, which for these checks is an addition.

    `--no-renames` reports a rename as a deletion plus an addition rather than
    a single modified path, so a migration moved into place still counts as an
    added migration.
    """
    filter_args = ["--diff-filter=A"] if added_only else []
    paths = set()
    paths.update(
        git_lines(
            ["diff", "--name-only", "--no-renames"] + filter_args + [base + "...HEAD"],
            cwd,
        )
    )
    paths.update(git_lines(["diff", "--name-only", "--no-renames", "--cached"] + filter_args, cwd))
    paths.update(git_lines(["diff", "--name-only", "--no-renames"] + filter_args, cwd))
    paths.update(git_lines(["ls-files", "--others", "--exclude-standard"], cwd))
    return sorted(paths)


def dir_exists(repo_root, rel_path):
    return os.path.isdir(os.path.join(repo_root, rel_path))


def under(path, prefix):
    prefix = prefix.rstrip("/") + "/"
    return path.startswith(prefix)


def read_file_if_real(path, cwd):
    path = path.strip("'\"")
    if path in ("-", ""):
        return ""
    full = path if os.path.isabs(path) else os.path.join(cwd, path)
    try:
        with open(full, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError:
        return ""


TEXT_FLAGS = ("--title", "-t", "--body", "-b")
FILE_FLAGS = ("--body-file", "-F")
FILL_FLAGS = ("--fill", "-f", "--fill-first", "--fill-verbose")


def extract_pr_text(tokens, cwd, base):
    """Pull the title/body text this `gh pr create` would submit.

    `tokens` are the invocation's own tokens, so nothing that merely shares
    the command line is read. Handles --title/-t, --body/-b, --body-file/-F,
    their `--flag=value` and `-fvalue` spellings, and falls back to the commit
    messages `--fill` would fill from.
    """
    parts = []
    fill = False
    index = 0
    while index < len(tokens):
        value, consumed = flag_value(tokens, index, TEXT_FLAGS)
        if consumed:
            if value:
                parts.append(value)
            index += consumed
            continue
        value, consumed = flag_value(tokens, index, FILE_FLAGS)
        if consumed:
            if value:
                parts.append(read_file_if_real(value, cwd))
            index += consumed
            continue
        if tokens[index] in FILL_FLAGS:
            fill = True
        index += 1

    text = "\n".join(p for p in parts if p)
    if fill and not text.strip():
        commits = run_git(["log", "--format=%B", base + "..HEAD"], cwd)
        text = commits or ""
    return text


# --- checks ------------------------------------------------------------
# Each check takes a Context and returns None when it does not fire, or a
# (possibly empty) dict of details used to render its question.


def check_docs(ctx):
    trigger_dirs = ("backend/src", "frontend/src", "scripts")
    if not any(dir_exists(ctx.repo_root, d) for d in trigger_dirs):
        return None
    # frontend/src/locales/ is covered by its own `i18n` check below; without
    # this exclusion a translation-only change (the four locale files kept in
    # sync, exactly the work done right) would also trip `docs`, which is the
    # kind of false positive that teaches contributors to ignore the hook.
    touches_code = any(
        under(p, d) and not under(p, "frontend/src/locales")
        for p in ctx.changed
        for d in trigger_dirs
    )
    if not touches_code:
        return None
    has_docs = any(p.endswith(".md") or under(p, "docs") for p in ctx.changed)
    if has_docs:
        return None
    return {}


def check_backend_tests(ctx):
    if not dir_exists(ctx.repo_root, "backend/tests"):
        return None
    touches_src = any(under(p, "backend/src") and p.endswith(".py") for p in ctx.changed)
    if not touches_src:
        return None
    has_tests = any(under(p, "backend/tests") for p in ctx.changed)
    if has_tests:
        return None
    return {}


FRONTEND_TEST_SUFFIXES = (".test.js", ".test.jsx", ".test.ts", ".test.tsx")


def check_frontend_tests(ctx):
    if not dir_exists(ctx.repo_root, "frontend/src"):
        return None
    # Same exclusion as `docs`: a translation-only change under
    # frontend/src/locales/ is covered by `i18n`, not this check.
    touches_non_test = any(
        under(p, "frontend/src")
        and not under(p, "frontend/src/locales")
        and not p.endswith(FRONTEND_TEST_SUFFIXES)
        for p in ctx.changed
    )
    if not touches_non_test:
        return None
    has_test_file = any(p.endswith(FRONTEND_TEST_SUFFIXES) for p in ctx.changed)
    if has_test_file:
        return None
    return {}


def check_migration(ctx):
    if not dir_exists(ctx.repo_root, "backend/alembic/versions"):
        return None
    entities_changed = any(under(p, "backend/src/entities") for p in ctx.changed)
    migration_added = any(under(p, "backend/alembic/versions") for p in ctx.added)
    tests_changed = any(under(p, "backend/tests") for p in ctx.changed)

    reasons = []
    if entities_changed and not migration_added:
        reasons.append(
            "a path under backend/src/entities changed but no file was added "
            "under backend/alembic/versions"
        )
    if migration_added and not tests_changed:
        reasons.append(
            "a file was added under backend/alembic/versions but no path "
            "under backend/tests changed"
        )
    if not reasons:
        return None
    return {"reasons": reasons}


def check_i18n(ctx):
    if not dir_exists(ctx.repo_root, "frontend/src/locales/en"):
        return None
    en_changed = any(under(p, "frontend/src/locales/en") for p in ctx.changed)
    if not en_changed:
        return None
    missing = []
    for lang in ("fr", "es", "zh"):
        prefix = "frontend/src/locales/" + lang
        if not any(under(p, prefix) for p in ctx.changed):
            missing.append(lang)
    if not missing:
        return None
    return {"missing": missing}


def check_requirements(ctx):
    if not dir_exists(ctx.repo_root, "backend/requirements/meta"):
        return None
    entry_changed = any(under(p, "backend/requirements/entrypoints") for p in ctx.changed)
    if not entry_changed:
        return None
    meta_changed = any(under(p, "backend/requirements/meta") for p in ctx.changed)
    if meta_changed:
        return None
    return {}


def check_jargon(ctx):
    text = ctx.pr_text
    if not text:
        return None
    # Every match, not the first one: a title carrying two different labels
    # would otherwise surface one of them, and once the marker records the
    # check as asked, the second is never surfaced at all.
    matches = []
    for pattern in JARGON_PATTERNS:
        for match in pattern.finditer(text):
            if match.group(0) not in matches:
                matches.append(match.group(0))
    if not matches:
        return None
    return {"matches": matches}


def check_issue_link(ctx):
    # A text check, like `jargon`: it reads only the PR title/body the hook has
    # already extracted, makes no network call, and has no directory
    # dependency. When no text is available at all (a heredoc body the parser
    # cannot see, or an empty --fill) there is nothing to assess, so it stays
    # silent -- the fail-open direction for a double-check.
    text = ctx.pr_text
    if not text:
        return None
    for pattern in ISSUE_REFERENCE_PATTERNS:
        if pattern.search(text):
            return None
    return {}


def msg_docs(_result):
    return (
        "This change touches code under backend/src, frontend/src or "
        "scripts, but no .md file and nothing under docs/ changed. Does any "
        "documentation now describe behaviour this change alters? If not, "
        "re-run the command."
    )


def msg_backend_tests(_result):
    return (
        "This change touches backend/src/**.py but no path under "
        "backend/tests changed. Does existing coverage already exercise "
        "this, or is there a gap? If existing coverage is enough, re-run "
        "the command."
    )


def msg_frontend_tests(_result):
    return (
        "This change touches frontend/src (outside of test files) but no "
        "*.test.js, *.test.jsx, *.test.ts or *.test.tsx file changed. Does "
        "existing coverage already exercise this, or is there a gap? If "
        "existing coverage is enough, re-run the command."
    )


def msg_migration(result):
    return (
        "; ".join(result["reasons"]) + ". Is the migration complete (migration "
        "file, test coverage, and model/service/endpoint/UI alignment)? If "
        "so, re-run the command."
    )


def msg_i18n(result):
    missing = ", ".join("frontend/src/locales/" + lang for lang in result["missing"])
    return (
        "frontend/src/locales/en changed but no changed path under "
        + missing
        + ". Do these languages need the same update? If not, re-run the "
        "command."
    )


def msg_requirements(_result):
    return (
        "backend/requirements/entrypoints changed but no path under "
        "backend/requirements/meta changed. Is this platform-specific only, "
        "or should the shared meta files change too? If platform-specific "
        "is correct, re-run the command."
    )


def msg_jargon(result):
    labels = ", ".join("'" + m + "'" for m in result["matches"])
    return (
        "The PR title/body matches internal jargon: " + labels + ". Is this "
        "wording meaningful to a contributor reading the pull request, or "
        "should it be rephrased? If it's fine as-is, re-run the command."
    )


def msg_issue_link(_result):
    return (
        "This pull request references no issue. If it closes or relates to "
        "one, add it (a '#123', or 'Closes #123'). If standing alone is "
        "intentional, re-run the command."
    )


CHECKS = [
    ("docs", check_docs, msg_docs),
    ("backend-tests", check_backend_tests, msg_backend_tests),
    ("frontend-tests", check_frontend_tests, msg_frontend_tests),
    ("migration", check_migration, msg_migration),
    ("i18n", check_i18n, msg_i18n),
    ("requirements", check_requirements, msg_requirements),
    ("jargon", check_jargon, msg_jargon),
    ("issue-link", check_issue_link, msg_issue_link),
]


# --- one-shot state marker ----------------------------------------------


def state_dir():
    try:
        directory = os.path.join(tempfile.gettempdir(), STATE_DIR_NAME)
        os.makedirs(directory, exist_ok=True)
    except OSError:
        return None
    return directory


def marker_key(session_id, repo_root, branch):
    # The repository root is part of the key on purpose: two checkouts of the
    # same project, on the same branch name, in the same session, are two
    # different pull requests and each deserves its own question.
    raw = (session_id or "") + "|" + repo_root + "|" + branch
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def marker_path(directory, key):
    return os.path.join(directory, key + ".json")


def load_marker(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raised = data.get("raised") or []
        return set(raised)
    except (OSError, ValueError):
        return set()


def save_marker(path, raised_ids):
    data = {"raised": sorted(raised_ids), "timestamp": time.time()}
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp_path, path)


def purge_old_markers(directory):
    try:
        cutoff = time.time() - MARKER_MAX_AGE_SECONDS
        for name in os.listdir(directory):
            if not name.endswith(".json"):
                continue
            full = os.path.join(directory, name)
            try:
                if os.path.getmtime(full) < cutoff:
                    os.remove(full)
            except OSError:
                pass
    except OSError:
        pass


def format_message(fired):
    lines = [HEADER]
    for check_id, text in fired:
        lines.append("  [%s] %s" % (check_id, text))
    lines.append(FOOTER)
    return "\n".join(lines)


def skip_requested():
    value = (os.environ.get(SKIP_ENV_VAR) or "").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def main():
    if skip_requested():
        return 0

    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    if not isinstance(payload, dict):
        return 0

    if payload.get("tool_name") != "Bash":
        return 0

    command = (payload.get("tool_input") or {}).get("command") or ""
    segment = find_pr_create_segment(command)
    if segment is None:
        return 0

    cwd = payload.get("cwd") or os.getcwd()

    repo_root = resolve_repo_root(cwd)
    if not repo_root:
        return 0
    base = resolve_base(cwd, base_from_segment(segment))
    if not base:
        return 0
    branch = current_branch(cwd)
    if not branch:
        return 0

    changed = changed_files(base, cwd)
    added = changed_files(base, cwd, added_only=True)
    pr_text = extract_pr_text(segment, cwd, base)

    ctx = Context(repo_root=repo_root, changed=changed, added=added, pr_text=pr_text)

    fired = []
    for check_id, check_fn, msg_fn in CHECKS:
        result = check_fn(ctx)
        if result is not None:
            fired.append((check_id, msg_fn(result)))

    if not fired:
        return 0

    directory = state_dir()
    if directory is None:
        return 0
    purge_old_markers(directory)

    session_id = payload.get("session_id") or ""
    key = marker_key(session_id, repo_root, branch)
    path = marker_path(directory, key)
    raised = load_marker(path)

    fired_ids = set(check_id for check_id, _ in fired)
    new_ids = fired_ids - raised
    if not new_ids:
        return 0

    try:
        save_marker(path, raised | fired_ids)
    except OSError:
        # The question cannot be recorded, so re-running would ask it again,
        # and again. A hook that cannot remember it asked must not ask.
        return 0

    remaining = [(check_id, text) for check_id, text in fired if check_id in new_ids]
    print(format_message(remaining), file=sys.stderr)
    return 2


def safe_main():
    try:
        return main()
    except Exception:
        return 0


if __name__ == "__main__":
    sys.exit(safe_main())
