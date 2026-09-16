"""Tests for the pull-request readiness double-check hook.

The hook (``scripts/dev/pr_readiness_hook.py``) is a Claude Code PreToolUse
hook, not backend application code, but it lives close to the rest of the
project's Python and is exercised the same way: through pytest. It is driven
as a real subprocess (payload JSON on stdin, exit code + stderr as the
contract) against a throwaway git repository built per test, which is what
lets these tests exercise the real `git diff` the hook relies on instead of
mocking it away.

``TMPDIR`` is pinned to a pytest ``tmp_path`` in every test so the one-shot
state marker never leaks between tests or onto the developer's machine, and
git is run with ``GIT_CONFIG_GLOBAL``/``GIT_CONFIG_NOSYSTEM`` neutralised so
that a contributor's own git configuration -- ``commit.gpgsign = true`` being
the one that actually bites -- cannot turn this file red on their machine.
"""

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK = REPO_ROOT / "scripts" / "dev" / "pr_readiness_hook.py"
SHARED_MODULE = REPO_ROOT / "scripts" / "dev" / "gh_pr_command.py"
SETTINGS = REPO_ROOT / ".claude" / "settings.json"


@pytest.fixture(autouse=True)
def _hook_script_must_exist():
    # CPython exits with status 2 when handed a path it cannot open, which is
    # the same status this hook uses to signal a block, and a missing script
    # writes its traceback to stderr rather than the hook's own messages. Left
    # unchecked, a deleted or renamed script would make every "must block"
    # assertion below pass for the wrong reason, and a weak "tag not in
    # stderr" assertion would pass against a Python traceback instead of the
    # hook's real "does not fire" output. Fail every test individually,
    # loudly, before any of that can happen.
    assert HOOK.is_file(), "expected hook script at %s" % HOOK
    assert SHARED_MODULE.is_file(), "expected shared module at %s" % SHARED_MODULE


# Neutralise the developer's own git configuration for every git invocation,
# the ones that build the fixtures and the ones the hook makes.
GIT_ENV = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}


# A directory tree that satisfies every check's dependency at once, so a
# single scaffold can serve most tests; dependency-absent tests start from a
# filtered copy instead of this one.
FULL_SCAFFOLD = {
    "README.md": "# Test repo\n",
    "backend/src/existing.py": "# existing module\n",
    "backend/src/entities/existing.py": "# existing entity\n",
    "backend/tests/existing_test.py": "def test_existing():\n    assert True\n",
    "backend/alembic/versions/0000_baseline.py": "# baseline migration\n",
    "backend/requirements/meta/base.txt": "requests\n",
    "backend/requirements/entrypoints/dev/existing.txt": "-r ../../meta/base.txt\n",
    "frontend/src/existing.jsx": "export default function Existing() { return null; }\n",
    "frontend/src/locales/en/common.json": '{"hello": "hello"}\n',
    "frontend/src/locales/fr/common.json": '{"hello": "bonjour"}\n',
    "frontend/src/locales/es/common.json": '{"hello": "hola"}\n',
    "frontend/src/locales/zh/common.json": '{"hello": "you hao"}\n',
    "scripts/existing.sh": "#!/usr/bin/env bash\necho existing\n",
    "docs/existing.md": "# Existing docs\n",
}

# A change that trips nothing at all: backend code with both a test and a doc.
CLEAN_BRANCH_FILES = {
    "backend/src/clean_module.py": "x = 1\n",
    "backend/tests/test_clean_module.py": "def test_x():\n    assert True\n",
    "docs/clean_module.md": "# Clean module\n",
}


def scaffold_without(*prefixes):
    """A copy of FULL_SCAFFOLD with any path under the given prefixes dropped."""
    result = {}
    for path, content in FULL_SCAFFOLD.items():
        if any(path == p or path.startswith(p.rstrip("/") + "/") for p in prefixes):
            continue
        result[path] = content
    return result


def _run(cmd, cwd):
    env = dict(os.environ)
    env.update(GIT_ENV)
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, env=env)
    assert result.returncode == 0, "%s failed: %s" % (cmd, result.stderr)
    return result.stdout


def _write_files(repo, files):
    for rel, content in files.items():
        full = repo / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)


def _commit(repo, files, message):
    _write_files(repo, files)
    _run(["git", "add", "-A"], repo)
    _run(["git", "commit", "-q", "-m", message], repo)


def make_repo(
    tmp_path,
    scaffold=None,
    branch_files=None,
    branch_name="feature",
    commit_message=None,
    name="repo",
):
    """Build a throwaway git repo: a `main` branch with `scaffold` committed,
    then a `branch_name` branch with `branch_files` committed on top (when
    given). `branch_name=None` leaves the repo on `main` for tests that build
    their own branch topology. Returns the repo directory.
    """
    repo = tmp_path / name
    repo.mkdir()
    _run(["git", "init", "-q", "-b", "main"], repo)
    _run(["git", "config", "user.email", "test@example.com"], repo)
    _run(["git", "config", "user.name", "Test"], repo)

    scaffold = FULL_SCAFFOLD if scaffold is None else scaffold
    _commit(repo, scaffold, "base")

    if branch_name:
        _run(["git", "checkout", "-q", "-b", branch_name], repo)
        if branch_files:
            _commit(repo, branch_files, commit_message or "feature")

    return repo


def delete_and_commit(repo, paths):
    _run(["git", "rm", "-q"] + list(paths), repo)
    _run(["git", "commit", "-q", "-m", "delete"], repo)


def run_hook(repo, command, state_dir, session_id="sess-1", cwd=None, env_overrides=None):
    # tempfile.gettempdir() validates each TMPDIR candidate by actually
    # writing into it; a directory that does not exist yet is silently
    # skipped in favour of the real system /tmp, which would make the
    # one-shot marker leak across tests (and across pytest's own recycled
    # temp-dir numbering between separate runs) instead of staying isolated
    # per test as intended. Create it before pointing TMPDIR at it.
    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "session_id": session_id,
        "tool_name": "Bash",
        "cwd": str(cwd if cwd is not None else repo),
        "tool_input": {"command": command},
    }
    env = dict(os.environ)
    env.update(GIT_ENV)
    env["TMPDIR"] = str(state_dir)
    env.pop("ERUDI_SKIP_PR_HOOKS", None)
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
    )


PR_CREATE = 'gh pr create --title "feat: something" --body "does a thing"'

# The shape that actually opens most pull requests: one compound command in
# which the commit does not exist yet when the hook runs.
COMPOUND_PR_CREATE = 'git add -A && git commit -m "feat: something" && git push && ' + PR_CREATE


# --- each of the seven ids fires -----------------------------------------


def test_docs_fires(tmp_path):
    repo = make_repo(tmp_path, branch_files={"scripts/new_tool.sh": "echo hi\n"})
    result = run_hook(repo, PR_CREATE, tmp_path / "state")
    assert result.returncode == 2
    assert "[docs]" in result.stderr


def test_backend_tests_fires(tmp_path):
    repo = make_repo(tmp_path, branch_files={"backend/src/new_module.py": "x = 1\n"})
    result = run_hook(repo, PR_CREATE, tmp_path / "state")
    assert result.returncode == 2
    assert "[backend-tests]" in result.stderr


def test_frontend_tests_fires(tmp_path):
    repo = make_repo(
        tmp_path,
        branch_files={"frontend/src/NewThing.jsx": "export default function NewThing() {}\n"},
    )
    result = run_hook(repo, PR_CREATE, tmp_path / "state")
    assert result.returncode == 2
    assert "[frontend-tests]" in result.stderr


def test_migration_fires_case_a_entities_without_migration(tmp_path):
    repo = make_repo(
        tmp_path, branch_files={"backend/src/entities/new_entity.py": "class New: pass\n"}
    )
    result = run_hook(repo, PR_CREATE, tmp_path / "state")
    assert result.returncode == 2
    assert "[migration]" in result.stderr
    assert "no file was added under backend/alembic/versions" in result.stderr


def test_migration_fires_case_b_migration_without_tests(tmp_path):
    repo = make_repo(
        tmp_path,
        branch_files={"backend/alembic/versions/0001_add_thing.py": "# new migration\n"},
    )
    result = run_hook(repo, PR_CREATE, tmp_path / "state")
    assert result.returncode == 2
    assert "[migration]" in result.stderr
    assert "no path under backend/tests changed" in result.stderr


def test_i18n_fires(tmp_path):
    repo = make_repo(
        tmp_path,
        branch_files={"frontend/src/locales/en/common.json": '{"hello": "hi"}\n'},
    )
    result = run_hook(repo, PR_CREATE, tmp_path / "state")
    assert result.returncode == 2
    assert "[i18n]" in result.stderr
    assert "frontend/src/locales/fr" in result.stderr
    assert "frontend/src/locales/es" in result.stderr
    assert "frontend/src/locales/zh" in result.stderr


def test_requirements_fires(tmp_path):
    repo = make_repo(
        tmp_path,
        branch_files={"backend/requirements/entrypoints/dev/new.txt": "somepkg==1.0\n"},
    )
    result = run_hook(repo, PR_CREATE, tmp_path / "state")
    assert result.returncode == 2
    assert "[requirements]" in result.stderr


def test_jargon_fires_on_pr_number_label(tmp_path):
    repo = make_repo(tmp_path, branch_files={"NOTES.txt": "just a note\n"})
    command = 'gh pr create --title "PR1: quick internal update" --body "cleanup"'
    result = run_hook(repo, command, tmp_path / "state")
    assert result.returncode == 2
    assert "[jargon]" in result.stderr
    assert "PR1" in result.stderr


# --- only a real invocation counts (not a mention of one) ------------------


def test_grep_quoting_the_command_neither_blocks_nor_spends_the_one_shot(tmp_path):
    # The consequence that made the raw-string trigger unshippable: a grep
    # over a file that quotes the command blocked, and burned the branch's
    # single question, after which the real invocation passed unasked.
    repo = make_repo(tmp_path, branch_files={"scripts/new_tool.sh": "echo hi\n"})
    state_dir = tmp_path / "state"

    grep = 'grep -rn "gh pr create" scripts/'
    first = run_hook(repo, grep, state_dir)
    assert first.returncode == 0
    assert first.stderr == ""

    real = run_hook(repo, PR_CREATE, state_dir)
    assert real.returncode == 2
    assert "[docs]" in real.stderr


def test_echo_of_the_command_does_not_fire(tmp_path):
    repo = make_repo(tmp_path, branch_files={"scripts/new_tool.sh": "echo hi\n"})
    result = run_hook(repo, 'echo "gh pr create --fill" >> notes.md', tmp_path / "state")
    assert result.returncode == 0
    assert result.stderr == ""


@pytest.mark.parametrize("flag", ["--help", "--web", "--dry-run"])
def test_non_creating_flags_do_not_fire(tmp_path, flag):
    repo = make_repo(tmp_path, branch_files={"scripts/new_tool.sh": "echo hi\n"})
    result = run_hook(repo, "gh pr create " + flag, tmp_path / "state")
    assert result.returncode == 0
    assert result.stderr == ""


def test_another_gh_subcommand_does_not_fire(tmp_path):
    repo = make_repo(tmp_path, branch_files={"scripts/new_tool.sh": "echo hi\n"})
    result = run_hook(repo, 'gh pr list --search "gh pr create"', tmp_path / "state")
    assert result.returncode == 0
    assert result.stderr == ""


def test_fires_behind_a_leading_cd_and_environment_assignment(tmp_path):
    repo = make_repo(tmp_path, branch_files={"scripts/new_tool.sh": "echo hi\n"})
    command = "cd backend && GH_PAGER=cat " + PR_CREATE
    result = run_hook(repo, command, tmp_path / "state")
    assert result.returncode == 2
    assert "[docs]" in result.stderr


def test_arguments_are_read_off_the_invocation_not_the_whole_line(tmp_path):
    # The `--title PR1` belongs to a different program on the same line; the
    # pull request being opened has a clean title and must not be accused.
    repo = make_repo(tmp_path, branch_files=CLEAN_BRANCH_FILES)
    command = 'some-other-tool --title "PR1 internal" && ' + PR_CREATE
    result = run_hook(repo, command, tmp_path / "state")
    assert result.returncode == 0
    assert result.stderr == ""


# --- the work is usually not committed yet when the hook runs --------------


def test_untracked_worktree_change_is_seen(tmp_path):
    repo = make_repo(tmp_path)
    _write_files(repo, {"backend/src/new_module.py": "x = 1\n"})
    result = run_hook(repo, COMPOUND_PR_CREATE, tmp_path / "state")
    assert result.returncode == 2
    assert "[backend-tests]" in result.stderr
    assert "[docs]" in result.stderr


def test_staged_but_uncommitted_change_is_seen(tmp_path):
    repo = make_repo(tmp_path)
    _write_files(repo, {"backend/src/new_module.py": "x = 1\n"})
    _run(["git", "add", "-A"], repo)
    result = run_hook(repo, COMPOUND_PR_CREATE, tmp_path / "state")
    assert result.returncode == 2
    assert "[backend-tests]" in result.stderr


def test_unstaged_modification_of_a_tracked_file_is_seen(tmp_path):
    repo = make_repo(tmp_path)
    _write_files(repo, {"backend/src/existing.py": "# modified module\nx = 1\n"})
    result = run_hook(repo, COMPOUND_PR_CREATE, tmp_path / "state")
    assert result.returncode == 2
    assert "[backend-tests]" in result.stderr


def test_untracked_migration_counts_as_an_added_file(tmp_path):
    # The migration check asks about *added* migrations. An untracked one is
    # about to be added by the `git add -A` in the same command line.
    repo = make_repo(
        tmp_path,
        branch_files={
            "backend/src/entities/new_entity.py": "class New: pass\n",
            "backend/tests/test_new_entity.py": "def test_x():\n    assert True\n",
            "docs/new_entity.md": "# New entity\n",
        },
    )
    _write_files(repo, {"backend/alembic/versions/0001_add_thing.py": "# new migration\n"})
    result = run_hook(repo, COMPOUND_PR_CREATE, tmp_path / "state")
    assert result.returncode == 0
    assert result.stderr == ""


def test_modified_migration_alongside_a_changed_entity_still_fires(tmp_path):
    # Modifying an existing migration is not adding one. A check that matched
    # the migration against the *changed* set instead of the *added* set would
    # go quiet here, which is the whole point of keeping the two sets apart.
    repo = make_repo(
        tmp_path,
        branch_files={
            "backend/alembic/versions/0000_baseline.py": "# baseline migration, edited\n",
            "backend/src/entities/existing.py": "# existing entity, edited\n",
            "backend/tests/test_entity.py": "def test_x():\n    assert True\n",
            "docs/entity.md": "# Entity\n",
        },
    )
    result = run_hook(repo, PR_CREATE, tmp_path / "state")
    assert result.returncode == 2
    assert "[migration]" in result.stderr
    assert "no file was added under backend/alembic/versions" in result.stderr


# --- base branch resolution ------------------------------------------------


def test_a_base_that_advanced_after_the_branch_point_is_excluded(tmp_path):
    # Three dots, not two. With two, everything committed on the base branch
    # after this branch was cut lands in the diff and gets asked about.
    repo = make_repo(tmp_path, branch_files=CLEAN_BRANCH_FILES)

    _run(["git", "checkout", "-q", "main"], repo)
    _commit(repo, {"frontend/src/Unrelated.jsx": "export default function U() {}\n"}, "upstream")
    _run(["git", "checkout", "-q", "feature"], repo)

    result = run_hook(repo, PR_CREATE, tmp_path / "state")
    assert result.returncode == 0
    assert result.stderr == ""


def test_upstream_main_wins_over_a_stale_origin_main(tmp_path):
    # The standard fork workflow: clone the fork, add `upstream`, branch off
    # `upstream/main`, never refresh `origin/main`. Diffing against the stale
    # `origin/main` drags unrelated upstream commits into a one-line change.
    repo = make_repo(tmp_path, branch_name=None)
    stale = _run(["git", "rev-parse", "main"], repo).strip()
    _run(["git", "update-ref", "refs/remotes/origin/main", stale], repo)

    _commit(repo, {"frontend/src/Unrelated.jsx": "export default function U() {}\n"}, "upstream")
    fresh = _run(["git", "rev-parse", "main"], repo).strip()
    _run(["git", "update-ref", "refs/remotes/upstream/main", fresh], repo)

    _run(["git", "checkout", "-q", "-b", "feature"], repo)
    _commit(repo, CLEAN_BRANCH_FILES, "feature")
    # Remove the local `main` so the only candidates left are the two remote
    # refs, and only picking `upstream/main` can produce a clean diff.
    _run(["git", "branch", "-q", "-D", "main"], repo)

    result = run_hook(repo, PR_CREATE, tmp_path / "state")
    assert result.returncode == 0
    assert result.stderr == ""


def test_without_an_explicit_base_a_side_branch_diff_is_noisy(tmp_path):
    # The control for the next test: branched off `develop`, which `main` does
    # not contain, so diffing against `main` sees `develop`'s work too.
    repo = make_repo(tmp_path, branch_name=None)
    _run(["git", "checkout", "-q", "-b", "develop"], repo)
    _commit(repo, {"frontend/src/DevOnly.jsx": "export default function D() {}\n"}, "develop")
    _run(["git", "checkout", "-q", "-b", "feature"], repo)
    _commit(repo, CLEAN_BRANCH_FILES, "feature")

    result = run_hook(repo, PR_CREATE, tmp_path / "state")
    assert result.returncode == 2
    assert "[frontend-tests]" in result.stderr


def test_explicit_base_flag_is_honoured(tmp_path):
    repo = make_repo(tmp_path, branch_name=None)
    _run(["git", "checkout", "-q", "-b", "develop"], repo)
    _commit(repo, {"frontend/src/DevOnly.jsx": "export default function D() {}\n"}, "develop")
    _run(["git", "checkout", "-q", "-b", "feature"], repo)
    _commit(repo, CLEAN_BRANCH_FILES, "feature")

    result = run_hook(repo, PR_CREATE + " --base develop", tmp_path / "state")
    assert result.returncode == 0
    assert result.stderr == ""


def test_explicit_base_flag_equals_spelling_is_honoured(tmp_path):
    repo = make_repo(tmp_path, branch_name=None)
    _run(["git", "checkout", "-q", "-b", "develop"], repo)
    _commit(repo, {"frontend/src/DevOnly.jsx": "export default function D() {}\n"}, "develop")
    _run(["git", "checkout", "-q", "-b", "feature"], repo)
    _commit(repo, CLEAN_BRANCH_FILES, "feature")

    result = run_hook(repo, PR_CREATE + " --base=develop", tmp_path / "state")
    assert result.returncode == 0
    assert result.stderr == ""


# --- jargon pattern edge cases ---------------------------------------------


def test_jargon_does_not_fire_on_real_pr_cross_reference(tmp_path):
    repo = make_repo(tmp_path)  # no branch diff at all
    command = 'gh pr create --title "chore: follow-up to PR #123" --body "fixes remaining issue"'
    result = run_hook(repo, command, tmp_path / "state")
    assert result.returncode == 0
    assert result.stderr == ""


@pytest.mark.parametrize(
    "body",
    [
        "run 3 times and compare the outputs",
        "raises the default to --batch 2048",
        "on iteration 0 the KV cache is still empty",
        "reproduced in Actions run 18234567",
        "round 2 of the tokenizer benchmark",
    ],
)
def test_jargon_ignores_ordinary_numbered_technical_prose(tmp_path, body):
    # This is an inference repository: numbered technical nouns are the house
    # style, not internal labels. A pattern that fires on these teaches
    # contributors to ignore the hook.
    repo = make_repo(tmp_path)
    command = 'gh pr create --title "perf: tokenizer" --body ' + json.dumps(body)
    result = run_hook(repo, command, tmp_path / "state")
    assert result.returncode == 0
    assert result.stderr == ""


def test_lowercase_wave_is_not_jargon(tmp_path):
    repo = make_repo(tmp_path)
    command = 'gh pr create --title "fix: audio" --body "the wave 2 harmonic was clipped"'
    result = run_hook(repo, command, tmp_path / "state")
    assert result.returncode == 0
    assert result.stderr == ""


def test_capitalised_wave_label_is_jargon(tmp_path):
    repo = make_repo(tmp_path)
    command = 'gh pr create --title "chore: tidy up" --body "closes out Wave 3 follow-ups"'
    result = run_hook(repo, command, tmp_path / "state")
    assert result.returncode == 2
    assert "[jargon]" in result.stderr
    assert "Wave 3" in result.stderr


def test_jargon_fires_on_bug_bash(tmp_path):
    repo = make_repo(tmp_path)
    command = 'gh pr create --title "fix: leftover bug-bash items" --body "n/a"'
    result = run_hook(repo, command, tmp_path / "state")
    assert result.returncode == 2
    assert "[jargon]" in result.stderr


def test_jargon_reports_every_match_not_only_the_first(tmp_path):
    # Reporting one label and then recording the check as asked means the
    # second label is never surfaced at all.
    repo = make_repo(tmp_path)
    command = (
        'gh pr create --title "chore: PR2 leftovers" --body "the rest of the bug-bash, and P0"'
    )
    result = run_hook(repo, command, tmp_path / "state")
    assert result.returncode == 2
    assert "PR2" in result.stderr
    assert "bug-bash" in result.stderr
    assert "P0" in result.stderr


def test_jargon_reads_body_file(tmp_path):
    repo = make_repo(tmp_path)
    body_file = repo / "body.txt"
    body_file.write_text("This addresses PR2 cleanup from last week.\n")
    command = 'gh pr create --title "chore: cleanup" --body-file body.txt'
    result = run_hook(repo, command, tmp_path / "state")
    assert result.returncode == 2
    assert "[jargon]" in result.stderr
    assert "PR2" in result.stderr


def test_jargon_reads_flag_equals_value_spellings(tmp_path):
    repo = make_repo(tmp_path)
    command = 'gh pr create --title="chore: PR3 leftovers" --body="n/a"'
    result = run_hook(repo, command, tmp_path / "state")
    assert result.returncode == 2
    assert "[jargon]" in result.stderr
    assert "PR3" in result.stderr


def test_jargon_reads_body_file_equals_value_spelling(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "body.txt").write_text("Finishes Phase 2 of the cleanup.\n")
    command = 'gh pr create --title="chore: cleanup" --body-file=body.txt'
    result = run_hook(repo, command, tmp_path / "state")
    assert result.returncode == 2
    assert "[jargon]" in result.stderr
    assert "Phase 2" in result.stderr


def test_jargon_reads_commit_messages_under_fill(tmp_path):
    repo = make_repo(
        tmp_path,
        branch_files={"NOTES.txt": "note\n"},
        commit_message="Wave 3: internal cleanup",
    )
    command = "gh pr create --fill"
    result = run_hook(repo, command, tmp_path / "state")
    assert result.returncode == 2
    assert "[jargon]" in result.stderr
    assert "Wave 3" in result.stderr


# --- each of the six path-based ids stays silent when its dependency dir is absent


def test_docs_silent_without_any_trigger_directory(tmp_path):
    scaffold = scaffold_without("backend/src", "frontend/src", "scripts")
    repo = make_repo(tmp_path, scaffold=scaffold, branch_files={"misc/data.txt": "value\n"})
    result = run_hook(repo, PR_CREATE, tmp_path / "state")
    assert result.returncode == 0
    assert result.stderr == ""


def test_backend_tests_silent_without_backend_tests_dir(tmp_path):
    # Without this guard, "[backend-tests] not in stderr" would also pass
    # against a Python traceback (see _hook_script_must_exist) or against a
    # run where something else fired and merely printed a different tag. The
    # branch change also adds a test and a doc file so that, dependency
    # absence aside, nothing else in the diff would fire either -- the only
    # way to legitimately expect a fully silent, exit-0 run.
    scaffold = scaffold_without("backend/tests")
    repo = make_repo(
        tmp_path,
        scaffold=scaffold,
        branch_files={
            "backend/src/new_module.py": "x = 1\n",
            "docs/new_module.md": "# New module\n",
        },
    )
    result = run_hook(repo, PR_CREATE, tmp_path / "state")
    assert result.returncode == 0
    assert result.stderr == ""


def test_frontend_tests_silent_without_frontend_src_dir(tmp_path):
    scaffold = {
        "README.md": "# minimal\n",
        "frontend/src/existing.jsx": "export default function Existing() { return null; }\n",
    }
    repo = make_repo(tmp_path, scaffold=scaffold)
    # Deletes the only file under frontend/src, so it appears in the diff
    # (satisfying "touches frontend/src" at the string level) while the
    # directory genuinely no longer exists at HEAD -- the only way to prove
    # the check reads the checkout, not the diff, since frontend/src is both
    # the trigger path and the dependency here.
    delete_and_commit(repo, ["frontend/src/existing.jsx"])
    result = run_hook(repo, PR_CREATE, tmp_path / "state")
    assert result.returncode == 0
    assert result.stderr == ""


def test_migration_silent_without_alembic_versions_dir(tmp_path):
    scaffold = scaffold_without("backend/alembic/versions")
    repo = make_repo(
        tmp_path,
        scaffold=scaffold,
        branch_files={
            "backend/src/entities/new_entity.py": "class New: pass\n",
            "backend/tests/test_new_entity.py": "def test_x():\n    assert True\n",
            "docs/new_entity.md": "# New entity\n",
        },
    )
    result = run_hook(repo, PR_CREATE, tmp_path / "state")
    assert result.returncode == 0
    assert result.stderr == ""


def test_i18n_silent_without_locales_en_dir(tmp_path):
    scaffold = {
        "README.md": "# minimal\n",
        "frontend/src/locales/en/common.json": '{"hello": "hi"}\n',
    }
    repo = make_repo(tmp_path, scaffold=scaffold)
    delete_and_commit(repo, ["frontend/src/locales/en/common.json"])
    result = run_hook(repo, PR_CREATE, tmp_path / "state")
    assert result.returncode == 0
    assert result.stderr == ""


def test_requirements_silent_without_requirements_meta_dir(tmp_path):
    scaffold = scaffold_without("backend/requirements/meta")
    repo = make_repo(
        tmp_path,
        scaffold=scaffold,
        branch_files={"backend/requirements/entrypoints/dev/new.txt": "somepkg==1.0\n"},
    )
    result = run_hook(repo, PR_CREATE, tmp_path / "state")
    assert result.returncode == 0
    assert result.stderr == ""


# --- translation-only changes must not trip docs/frontend-tests -----------


def test_locale_change_across_all_four_languages_fires_nothing(tmp_path):
    # Every value here must differ from FULL_SCAFFOLD's existing content --
    # otherwise git sees no real change for that file and it silently drops
    # out of the diff, which is exactly what broke this test the first time.
    repo = make_repo(
        tmp_path,
        branch_files={
            "frontend/src/locales/en/common.json": '{"hello": "hi"}\n',
            "frontend/src/locales/fr/common.json": '{"hello": "salut"}\n',
            "frontend/src/locales/es/common.json": '{"hello": "hola que tal"}\n',
            "frontend/src/locales/zh/common.json": '{"hello": "ni hao"}\n',
        },
    )
    result = run_hook(repo, PR_CREATE, tmp_path / "state")
    assert result.returncode == 0
    assert result.stderr == ""


def test_locale_change_in_english_only_fires_i18n_alone(tmp_path):
    repo = make_repo(
        tmp_path,
        branch_files={"frontend/src/locales/en/common.json": '{"hello": "hi"}\n'},
    )
    result = run_hook(repo, PR_CREATE, tmp_path / "state")
    assert result.returncode == 2
    assert "[i18n]" in result.stderr
    assert "[docs]" not in result.stderr
    assert "[frontend-tests]" not in result.stderr


# --- general behaviour -----------------------------------------------------


def test_clean_pr_exits_zero_with_no_output(tmp_path):
    repo = make_repo(tmp_path, branch_files=CLEAN_BRANCH_FILES)
    command = 'gh pr create --title "feat: add module" --body "Adds a module with tests and docs."'
    result = run_hook(repo, command, tmp_path / "state")
    assert result.returncode == 0
    assert result.stderr == ""


def test_non_pr_create_command_exits_zero(tmp_path):
    repo = make_repo(tmp_path, branch_files={"backend/src/new_module.py": "x = 1\n"})
    result = run_hook(repo, "gh pr list", tmp_path / "state")
    assert result.returncode == 0
    assert result.stderr == ""


def test_one_shot_contract_blocks_then_passes(tmp_path):
    repo = make_repo(tmp_path, branch_files={"scripts/new_tool.sh": "echo hi\n"})
    state_dir = tmp_path / "state"

    first = run_hook(repo, PR_CREATE, state_dir)
    assert first.returncode == 2
    assert "[docs]" in first.stderr

    second = run_hook(repo, PR_CREATE, state_dir)
    assert second.returncode == 0
    assert second.stderr == ""


def test_only_newly_appearing_gaps_are_named(tmp_path):
    # The second run has two gaps open at once, one of them already asked
    # about. Re-printing everything that fired would repeat the first
    # question, and the contributor learns that re-running is a lottery.
    repo = make_repo(tmp_path, branch_files={"scripts/new_tool.sh": "echo hi\n"})
    state_dir = tmp_path / "state"

    first = run_hook(repo, PR_CREATE, state_dir)
    assert first.returncode == 2
    assert "[docs]" in first.stderr

    _commit(repo, {"backend/src/new_mod.py": "x = 1\n"}, "more code, still no docs")
    second = run_hook(repo, PR_CREATE, state_dir)
    assert second.returncode == 2
    assert "[backend-tests]" in second.stderr
    assert "[docs]" not in second.stderr


def test_marker_merges_and_only_new_gaps_are_named(tmp_path):
    """Three runs, one branch, one session.

    Run 1 asks about the missing documentation. The fix lands, and opens a
    different gap: run 2 must name *that* one and not repeat the first. The
    doc is then removed again so both gaps are present at once: run 3 must be
    silent, which it only can be if the marker accumulated both ids rather
    than being overwritten with the last one.
    """
    repo = make_repo(tmp_path, branch_files={"scripts/new_tool.sh": "echo hi\n"})
    state_dir = tmp_path / "state"

    first = run_hook(repo, PR_CREATE, state_dir)
    assert first.returncode == 2
    assert "[docs]" in first.stderr
    assert "[backend-tests]" not in first.stderr

    _commit(
        repo,
        {"docs/new_tool.md": "# New tool\n", "backend/src/new_mod.py": "x = 1\n"},
        "docs and code",
    )
    second = run_hook(repo, PR_CREATE, state_dir)
    assert second.returncode == 2
    assert "[backend-tests]" in second.stderr
    assert "[docs]" not in second.stderr

    delete_and_commit(repo, ["docs/new_tool.md"])
    third = run_hook(repo, PR_CREATE, state_dir)
    assert third.returncode == 0
    assert third.stderr == ""


def test_different_branch_after_block_still_blocks(tmp_path):
    state_dir = tmp_path / "state"
    repo = make_repo(
        tmp_path, branch_files={"scripts/new_tool.sh": "echo hi\n"}, branch_name="feature-a"
    )
    first = run_hook(repo, PR_CREATE, state_dir)
    assert first.returncode == 2

    _run(["git", "checkout", "-q", "main"], repo)
    _run(["git", "checkout", "-q", "-b", "feature-b"], repo)
    _commit(repo, {"scripts/other_tool.sh": "echo hi\n"}, "feature-b")

    second = run_hook(repo, PR_CREATE, state_dir)
    assert second.returncode == 2
    assert "[docs]" in second.stderr


def test_different_session_after_block_still_blocks(tmp_path):
    state_dir = tmp_path / "state"
    repo = make_repo(tmp_path, branch_files={"scripts/new_tool.sh": "echo hi\n"})

    first = run_hook(repo, PR_CREATE, state_dir, session_id="sess-a")
    assert first.returncode == 2

    second = run_hook(repo, PR_CREATE, state_dir, session_id="sess-b")
    assert second.returncode == 2
    assert "[docs]" in second.stderr


def test_two_checkouts_of_the_same_branch_are_asked_separately(tmp_path):
    # Same branch name, same session, two working copies: two pull requests,
    # so two questions. Dropping the repository root from the marker key would
    # silence the second one.
    state_dir = tmp_path / "state"
    first_repo = make_repo(tmp_path, branch_files={"scripts/a.sh": "echo a\n"}, name="repo-a")
    second_repo = make_repo(tmp_path, branch_files={"scripts/b.sh": "echo b\n"}, name="repo-b")

    first = run_hook(first_repo, PR_CREATE, state_dir)
    assert first.returncode == 2
    assert "[docs]" in first.stderr

    second = run_hook(second_repo, PR_CREATE, state_dir)
    assert second.returncode == 2
    assert "[docs]" in second.stderr


# --- fail-open doctrine ----------------------------------------------------


def test_fail_open_when_cwd_is_not_a_git_repo(tmp_path):
    not_a_repo = tmp_path / "not_a_repo"
    not_a_repo.mkdir()
    result = run_hook(not_a_repo, PR_CREATE, tmp_path / "state", cwd=not_a_repo)
    assert result.returncode == 0
    assert result.stderr == ""


def test_fail_open_when_git_itself_fails(tmp_path):
    # The doctrine the file claims to follow, exercised rather than asserted:
    # every git call the hook makes returns non-zero, and the command still
    # goes through, silently.
    repo = make_repo(tmp_path, branch_files={"scripts/new_tool.sh": "echo hi\n"})

    fake_bin = tmp_path / "fakebin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text("#!/bin/sh\necho 'broken' >&2\nexit 1\n")
    fake_git.chmod(fake_git.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    result = run_hook(
        repo,
        PR_CREATE,
        tmp_path / "state",
        env_overrides={"PATH": str(fake_bin) + os.pathsep + os.environ.get("PATH", "")},
    )
    assert result.returncode == 0
    assert result.stderr == ""


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores directory permissions"
)
def test_fail_open_when_the_marker_cannot_be_written(tmp_path):
    # A shared /tmp where another account created the state directory 0755 is
    # the realistic case. A hook that cannot record the question it asked must
    # not ask it, or it asks forever while promising the opposite.
    repo = make_repo(tmp_path, branch_files={"scripts/new_tool.sh": "echo hi\n"})
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    locked = state_dir / "erudi-pr-readiness"
    locked.mkdir()
    locked.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        result = run_hook(repo, PR_CREATE, state_dir)
        assert result.returncode == 0
        assert result.stderr == ""
    finally:
        locked.chmod(stat.S_IRWXU)


def test_skip_env_var_disables_the_hook(tmp_path):
    repo = make_repo(tmp_path, branch_files={"scripts/new_tool.sh": "echo hi\n"})
    result = run_hook(
        repo, PR_CREATE, tmp_path / "state", env_overrides={"ERUDI_SKIP_PR_HOOKS": "1"}
    )
    assert result.returncode == 0
    assert result.stderr == ""


# --- the settings wrapper --------------------------------------------------


def _settings_hook_commands():
    settings = json.loads(SETTINGS.read_text(encoding="utf-8"))
    commands = []
    for event in settings.get("hooks", {}).values():
        for matcher in event:
            for hook in matcher.get("hooks", []):
                commands.append(hook["command"])
    return commands


def test_settings_hook_commands_are_guarded_by_a_file_test():
    # `python3 missing.py` exits 2, which is exactly the code that means
    # "block this tool call". Without the guard, any checkout that predates
    # these scripts -- an old tag, a bisect step -- refuses every single Bash
    # command, including the checkout that would undo it.
    commands = _settings_hook_commands()
    assert len(commands) == 2
    for command in commands:
        assert command.startswith("if [ -f "), command
        assert "python3" in command


def test_guarded_command_shape_exits_zero_when_the_script_is_missing(tmp_path):
    missing = tmp_path / "definitely" / "missing.py"

    unguarded = subprocess.run(
        [sys.executable, str(missing)], capture_output=True, text=True, input=""
    )
    assert unguarded.returncode == 2, "the premise: a missing script exits 2, i.e. 'block'"

    guarded = 'if [ -f "%s" ]; then python3 "%s"; fi' % (missing, missing)
    result = subprocess.run(["sh", "-c", guarded], capture_output=True, text=True, input="")
    assert result.returncode == 0
    assert result.stderr == ""


def test_guarded_command_shape_still_runs_the_hook_when_present(tmp_path):
    repo = make_repo(tmp_path, branch_files={"scripts/new_tool.sh": "echo hi\n"})
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "session_id": "sess-guard",
        "tool_name": "Bash",
        "cwd": str(repo),
        "tool_input": {"command": PR_CREATE},
    }
    env = dict(os.environ)
    env.update(GIT_ENV)
    env["TMPDIR"] = str(state_dir)
    env.pop("ERUDI_SKIP_PR_HOOKS", None)

    guarded = 'if [ -f "%s" ]; then %s "%s"; fi' % (HOOK, sys.executable, HOOK)
    result = subprocess.run(
        ["sh", "-c", guarded],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 2
    assert "[docs]" in result.stderr
