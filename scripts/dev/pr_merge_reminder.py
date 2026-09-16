#!/usr/bin/env python3
"""
PreToolUse hook (Bash) for Claude Code -- double-checks a `gh pr merge` before
the pull request is merged, on one question only: will this merge actually
close the issues the pull request references? An issue that is mentioned in the
body but is not set to auto-close stays open after the branch is gone, which is
the quiet way a "fixed" issue lingers. It is wired up in `.claude/settings.json`
alongside the readiness double-check; nothing else reads it. Set
`ERUDI_SKIP_PR_HOOKS=1` to turn it off.

This is a double-check, not a wall. The first `gh pr merge` in a session, on a
branch whose pull request references an issue that will not be closed, is
BLOCKED (exit 2, the question on stderr) so a closing keyword can be added or
the omission dismissed as intentional, and the same command re-run. The
question is remembered per branch, per session, so re-running goes through --
and an issue that is only referenced-without-closing *after* a fix is a
question not yet asked, so it is asked once too.

This is the one hook here that reaches the network. It resolves the pull
request through `gh pr view <n-or-nothing> --json
number,body,closingIssuesReferences`: the merge command may name the pull
request (`gh pr merge 625`, or its URL) or omit it, in which case `gh` resolves
the current branch's pull request. `closingIssuesReferences` is exactly the set
of issues GitHub will auto-close on merge; the body is scanned for `#<n>` and
bare issue URLs, and the check fires on the references that are not in the
closing set.

Fail-open discipline, without exception: `gh` missing from PATH, `gh` not
authenticated, no network, a non-zero exit, unparseable JSON, a detached HEAD
with no pull request, a command that names no pull request when none exists for
the current branch, a state file that cannot be written -- every one of these
results in exit 0, silently. `gh` is given a short timeout so a hang cannot
wedge the merge, and `main()` is wrapped so no traceback can ever propagate to
exit 2. A hook that blocks a merge because of its own bug is worse than no hook
at all.
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

from gh_pr_command import find_pr_merge_segment, pr_argument_from_segment  # noqa: E402

SKIP_ENV_VAR = "ERUDI_SKIP_PR_HOOKS"

STATE_DIR_NAME = "erudi-pr-merge-reminder"
MARKER_MAX_AGE_SECONDS = 24 * 60 * 60

# Seconds before `gh` is abandoned. A hung network call must never hold up the
# merge; abandoning it is just another fail-open path.
GH_TIMEOUT_SECONDS = 10

# Issue references in the pull-request body: a `#123` cross-reference, or a
# bare GitHub issues URL. The digits are captured so the numbers can be
# compared against the set of issues the merge will close.
ISSUE_NUMBER_PATTERNS = [
    re.compile(r"#(\d+)"),
    re.compile(r"https://github\.com/[^/\s]+/[^/\s]+/issues/(\d+)"),
]


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


def resolve_repo_root(cwd):
    out = run_git(["rev-parse", "--show-toplevel"], cwd)
    if out is None:
        return None
    root = out.strip()
    return root or None


def current_branch(cwd):
    out = run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    if out is None:
        return None
    branch = out.strip()
    return branch or None


def gh_pr_view(pr_arg, cwd):
    """The pull request as a dict, or None on any failure (fail open).

    `pr_arg` is the number/URL/branch the merge command named, or None to let
    `gh` resolve the current branch's pull request.
    """
    args = ["gh", "pr", "view"]
    if pr_arg:
        args.append(pr_arg)
    args += ["--json", "number,body,closingIssuesReferences"]
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=GH_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def referenced_issue_numbers(body):
    numbers = set()
    for pattern in ISSUE_NUMBER_PATTERNS:
        for match in pattern.finditer(body):
            try:
                numbers.add(int(match.group(1)))
            except (ValueError, IndexError):
                pass
    return numbers


def closing_issue_numbers(closing):
    numbers = set()
    if not isinstance(closing, list):
        return numbers
    for entry in closing:
        if isinstance(entry, dict) and isinstance(entry.get("number"), int):
            numbers.add(entry["number"])
    return numbers


# --- one-shot state marker ----------------------------------------------
# A private copy of the readiness hook's marker helpers: each file has to stand
# on its own as a hook, and the two use separate state directories so their
# questions never collide.


def state_dir():
    try:
        directory = os.path.join(tempfile.gettempdir(), STATE_DIR_NAME)
        os.makedirs(directory, exist_ok=True)
    except OSError:
        return None
    return directory


def marker_key(session_id, repo_root, branch):
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


def format_message(numbers):
    listed = ", ".join("#%d" % n for n in numbers)
    return (
        "PR merge double-check (asked once per branch, per session): these "
        "issues are referenced by the pull request but will not be closed by "
        "this merge: " + listed + ". If they should close, add a closing "
        "keyword ('Closes #123') to the PR body. If leaving them open is "
        "intentional, re-run the exact same command."
    )


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
    segment = find_pr_merge_segment(command)
    if segment is None:
        return 0

    cwd = payload.get("cwd") or os.getcwd()

    data = gh_pr_view(pr_argument_from_segment(segment), cwd)
    if data is None:
        return 0

    body = data.get("body")
    if not isinstance(body, str) or not body:
        return 0

    referenced = referenced_issue_numbers(body)
    if not referenced:
        return 0

    not_closing = referenced - closing_issue_numbers(data.get("closingIssuesReferences"))
    if not not_closing:
        return 0

    directory = state_dir()
    if directory is None:
        return 0
    purge_old_markers(directory)

    session_id = payload.get("session_id") or ""
    repo_root = resolve_repo_root(cwd) or cwd
    branch = current_branch(cwd) or ""
    key = marker_key(session_id, repo_root, branch)
    path = marker_path(directory, key)
    raised = load_marker(path)

    fired_ids = set(str(n) for n in not_closing)
    new_ids = fired_ids - raised
    if not new_ids:
        return 0

    try:
        save_marker(path, raised | fired_ids)
    except OSError:
        # The question cannot be recorded, so re-running would ask it again,
        # and again. A hook that cannot remember it asked must not ask.
        return 0

    print(format_message(sorted(int(x) for x in new_ids)), file=sys.stderr)
    return 2


def safe_main():
    try:
        return main()
    except Exception:
        return 0


if __name__ == "__main__":
    sys.exit(safe_main())
