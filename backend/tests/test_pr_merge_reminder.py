"""Tests for the pull-request merge double-check hook.

The hook (``scripts/dev/pr_merge_reminder.py``) is a Claude Code PreToolUse
hook, exercised the same way as its readiness sibling: as a real subprocess
with a payload JSON on stdin and the exit code plus stderr as the contract.

Unlike the other hooks, this one shells out to ``gh`` -- so every test fakes
``gh`` on PATH, returning canned JSON. Two fakes are written for every case:
an extension-less ``gh`` shell script AND a ``gh.bat``. Windows resolves a
program through PATHEXT and ignores the extension-less script, so a test that
wrote only the shell script would silently exercise nothing there (the same
lesson baked into ``test_pr_readiness_hook``'s failing-git test). The fakes
print the contents of a file named by ``GH_FAKE_FILE`` and exit with
``GH_FAKE_EXIT``, which lets each test hand ``gh`` a different response without
a different script.

``TMPDIR`` is pinned to a pytest ``tmp_path`` so the one-shot marker never
leaks between tests or onto the developer's machine.
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
HOOK = REPO_ROOT / "scripts" / "dev" / "pr_merge_reminder.py"
SHARED_MODULE = REPO_ROOT / "scripts" / "dev" / "gh_pr_command.py"


@pytest.fixture(autouse=True)
def _hook_script_must_exist():
    # See the analogous fixture in test_pr_readiness_hook.py: a missing script
    # exits 2 (the same code that means "block") with a traceback on stderr,
    # which would make a weak "[issue] not in stderr" assertion pass for the
    # wrong reason. Fail every test individually, loudly, first.
    assert HOOK.is_file(), "expected hook script at %s" % HOOK
    assert SHARED_MODULE.is_file(), "expected shared module at %s" % SHARED_MODULE


MERGE = "gh pr merge 625 --squash"


def _write_fake_gh(bin_dir):
    """Write both a `gh` shell script and a `gh.bat` into bin_dir.

    Each prints the file named by GH_FAKE_FILE (when set) and exits with
    GH_FAKE_EXIT (default 0).
    """
    bin_dir.mkdir(parents=True, exist_ok=True)

    gh = bin_dir / "gh"
    gh.write_text(
        '#!/bin/sh\nif [ -n "$GH_FAKE_FILE" ]; then cat "$GH_FAKE_FILE"; fi\nexit ${GH_FAKE_EXIT:-0}\n'
    )
    gh.chmod(gh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    gh_bat = bin_dir / "gh.bat"
    gh_bat.write_text(
        "@echo off\r\n"
        'if defined GH_FAKE_FILE type "%GH_FAKE_FILE%"\r\n'
        "if not defined GH_FAKE_EXIT set GH_FAKE_EXIT=0\r\n"
        "exit /b %GH_FAKE_EXIT%\r\n"
    )


def run_hook(
    tmp_path,
    command=MERGE,
    gh_json=None,
    gh_exit=0,
    gh_present=True,
    session_id="sess-1",
    cwd=None,
):
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    bin_dir = tmp_path / "fakebin"

    env = dict(os.environ)
    env["TMPDIR"] = str(state_dir)
    env.pop("ERUDI_SKIP_PR_HOOKS", None)

    if gh_present:
        _write_fake_gh(bin_dir)
        env["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    else:
        # An empty bin dir as the *only* PATH entry: `gh` resolves nowhere, so
        # subprocess raises FileNotFoundError, the fail-open path under test.
        bin_dir.mkdir(parents=True, exist_ok=True)
        env["PATH"] = str(bin_dir)

    if gh_json is not None:
        fake_file = tmp_path / "gh_output.json"
        fake_file.write_text(gh_json)
        env["GH_FAKE_FILE"] = str(fake_file)
    env["GH_FAKE_EXIT"] = str(gh_exit)

    cwd_dir = cwd if cwd is not None else (tmp_path / "work")
    Path(cwd_dir).mkdir(parents=True, exist_ok=True)

    payload = {
        "session_id": session_id,
        "tool_name": "Bash",
        "cwd": str(cwd_dir),
        "tool_input": {"command": command},
    }
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
    )


def _view_json(body, closing_numbers):
    return json.dumps(
        {
            "number": 625,
            "body": body,
            "closingIssuesReferences": [{"number": n} for n in closing_numbers],
        }
    )


# --- the check fires and stays silent --------------------------------------


def test_referenced_but_not_closing_fires(tmp_path):
    gh_json = _view_json("Fixes #12 and relates to #34.", closing_numbers=[12])
    result = run_hook(tmp_path, gh_json=gh_json)
    assert result.returncode == 2
    # The listed set is exactly the referenced-but-not-closing issue: #34
    # alone, never the #12 that this merge does close. (A bare "#12 not in
    # stderr" would trip on the '#123' inside the message's own example.)
    assert "this merge: #34." in result.stderr


def test_all_referenced_issues_are_closing_is_silent(tmp_path):
    gh_json = _view_json("Closes #12 and closes #34.", closing_numbers=[12, 34])
    result = run_hook(tmp_path, gh_json=gh_json)
    assert result.returncode == 0
    assert result.stderr == ""


def test_nothing_referenced_is_silent(tmp_path):
    gh_json = _view_json("General cleanup, references no issue.", closing_numbers=[])
    result = run_hook(tmp_path, gh_json=gh_json)
    assert result.returncode == 0
    assert result.stderr == ""


def test_issues_url_reference_not_closing_fires(tmp_path):
    gh_json = _view_json(
        "Related to https://github.com/erudi-app/erudi/issues/77", closing_numbers=[]
    )
    result = run_hook(tmp_path, gh_json=gh_json)
    assert result.returncode == 2
    assert "#77" in result.stderr


# --- fail-open doctrine -----------------------------------------------------


def test_gh_absent_is_silent(tmp_path):
    result = run_hook(tmp_path, gh_present=False)
    assert result.returncode == 0
    assert result.stderr == ""


def test_gh_non_zero_exit_is_silent(tmp_path):
    # Valid JSON, but gh failed (not authenticated, no such PR, no network):
    # the non-zero exit alone must silence the hook.
    gh_json = _view_json("Fixes #12 and relates to #34.", closing_numbers=[12])
    result = run_hook(tmp_path, gh_json=gh_json, gh_exit=1)
    assert result.returncode == 0
    assert result.stderr == ""


def test_gh_unparseable_json_is_silent(tmp_path):
    result = run_hook(tmp_path, gh_json="not json at all {{{")
    assert result.returncode == 0
    assert result.stderr == ""


def test_non_merge_command_is_silent(tmp_path):
    # A real referenced-but-not-closing PR, but the command does not merge.
    gh_json = _view_json("Fixes #12 and relates to #34.", closing_numbers=[12])
    result = run_hook(tmp_path, command="gh pr view 625 --json body", gh_json=gh_json)
    assert result.returncode == 0
    assert result.stderr == ""


def test_skip_env_var_disables_the_hook(tmp_path):
    gh_json = _view_json("Fixes #12 and relates to #34.", closing_numbers=[12])
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    bin_dir = tmp_path / "fakebin"
    _write_fake_gh(bin_dir)
    fake_file = tmp_path / "gh_output.json"
    fake_file.write_text(gh_json)
    work = tmp_path / "work"
    work.mkdir()

    env = dict(os.environ)
    env["TMPDIR"] = str(state_dir)
    env["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    env["GH_FAKE_FILE"] = str(fake_file)
    env["GH_FAKE_EXIT"] = "0"
    env["ERUDI_SKIP_PR_HOOKS"] = "1"

    payload = {
        "session_id": "sess-skip",
        "tool_name": "Bash",
        "cwd": str(work),
        "tool_input": {"command": MERGE},
    }
    result = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert result.stderr == ""


# --- one-shot contract ------------------------------------------------------


def test_one_shot_contract_blocks_then_passes(tmp_path):
    gh_json = _view_json("Fixes #12 and relates to #34.", closing_numbers=[12])
    # Same session, same cwd across both runs, so the marker key is stable.
    work = tmp_path / "work"

    first = run_hook(tmp_path, gh_json=gh_json, cwd=work)
    assert first.returncode == 2
    assert "#34" in first.stderr

    second = run_hook(tmp_path, gh_json=gh_json, cwd=work)
    assert second.returncode == 0
    assert second.stderr == ""
