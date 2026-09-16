"""Tests for the CI-watch reminder hook.

The hook (``scripts/dev/ci_watch_reminder.py``) is a Claude Code PostToolUse
hook. Like the readiness double-check hook, it is exercised as a real
subprocess: a payload JSON on stdin, stdout/stderr/exit code as the contract.
It never blocks, so every test here is about *what* it prints (or stays silent
about), not about exit codes, which are always 0.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK = REPO_ROOT / "scripts" / "dev" / "ci_watch_reminder.py"
SHARED_MODULE = REPO_ROOT / "scripts" / "dev" / "gh_pr_command.py"


@pytest.fixture(autouse=True)
def _hook_script_must_exist():
    # See the analogous fixture in test_pr_readiness_hook.py: a missing
    # script would make every "stays silent" assertion below pass against an
    # empty stdout produced by a Python traceback on stderr, not against the
    # hook's real logic. Fail every test individually instead.
    assert HOOK.is_file(), "expected hook script at %s" % HOOK
    assert SHARED_MODULE.is_file(), "expected shared module at %s" % SHARED_MODULE


def run_hook(payload):
    env = dict(os.environ)
    env.pop("ERUDI_SKIP_PR_HOOKS", None)
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
    )


def bash_payload(command, tool_response=None):
    return {
        "tool_name": "Bash",
        "tool_input": {"command": command},
        "tool_response": tool_response,
    }


PR_CREATE = 'gh pr create --title "feat: something" --body "does a thing"'
PR_URL = "https://github.com/erudi-app/erudi/pull/123"


def test_fires_with_pr_url_on_success():
    payload = bash_payload(
        PR_CREATE,
        tool_response={"stdout": PR_URL + "\n", "stderr": "", "returnCode": 0},
    )
    result = run_hook(payload)
    assert result.returncode == 0
    assert result.stderr == ""

    output = json.loads(result.stdout)
    context = output["hookSpecificOutput"]["additionalContext"]
    assert output["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
    assert PR_URL in context
    assert "backend suite" in context
    assert "frontend suite" in context
    assert "full-app build smoke" in context
    assert "advisory" in context


def test_reminder_names_no_runner_image():
    # Runner images are retired and job matrices change; a reminder that names
    # them is wrong the day after, in a file nobody thinks to update.
    payload = bash_payload(PR_CREATE, tool_response={"stdout": PR_URL + "\n", "returnCode": 0})
    context = json.loads(run_hook(payload).stdout)["hookSpecificOutput"]["additionalContext"]
    for runner in ("macos-14", "ubuntu-latest", "windows-latest"):
        assert runner not in context


def test_silent_on_non_pr_create_command():
    payload = bash_payload(
        "gh pr list", tool_response={"stdout": "some/repo#999  a title\n", "returnCode": 0}
    )
    result = run_hook(payload)
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_silent_when_the_command_only_mentions_the_words():
    # A grep whose output happens to contain a pull request URL opened
    # nothing. This is the shape that fired the reminder twice in one day.
    payload = bash_payload(
        'grep -rn "gh pr create" docs/',
        tool_response={"stdout": "docs/notes.md:4: see " + PR_URL + "\n", "returnCode": 0},
    )
    result = run_hook(payload)
    assert result.stdout == ""
    assert result.stderr == ""


@pytest.mark.parametrize("flag", ["--help", "--web", "--dry-run"])
def test_silent_on_non_creating_flags(flag):
    payload = bash_payload(
        "gh pr create " + flag,
        tool_response={"stdout": PR_URL + "\n", "returnCode": 0},
    )
    result = run_hook(payload)
    assert result.stdout == ""
    assert result.stderr == ""


def test_silent_on_return_code_failure_signal():
    payload = bash_payload(PR_CREATE, tool_response={"stdout": PR_URL + "\n", "returnCode": 1})
    result = run_hook(payload)
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_silent_on_is_error_flag():
    payload = bash_payload(PR_CREATE, tool_response={"stdout": PR_URL + "\n", "is_error": True})
    result = run_hook(payload)
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_silent_on_malformed_payload():
    result = subprocess.run(
        [sys.executable, str(HOOK)],
        input="not valid json {{{",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_silent_when_no_url_and_empty_stdout():
    payload = bash_payload(PR_CREATE, tool_response={"stdout": "", "returnCode": 0})
    result = run_hook(payload)
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_silent_without_a_url_on_stdout():
    # No degraded, URL-less form of the reminder: without a URL there is no
    # evidence a pull request exists, and the CLI prints one when it does.
    payload = bash_payload(
        PR_CREATE,
        tool_response={
            "stdout": "Creating pull request for feature-branch into main\n",
            "returnCode": 0,
        },
    )
    result = run_hook(payload)
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_silent_when_the_url_is_only_on_stderr():
    # The shape that matters: when a pull request for the branch already
    # exists, the CLI says so on stderr and puts the existing URL on a line of
    # its own. Reading stderr would announce a pull request this command did
    # not open, on a line indistinguishable from the success output.
    payload = bash_payload(
        PR_CREATE,
        tool_response={
            "stdout": "",
            "stderr": (
                'a pull request for branch "feature" into branch "main" '
                "already exists:\n" + PR_URL + "\n"
            ),
            "returnCode": 0,
        },
    )
    result = run_hook(payload)
    assert result.stdout == ""
    assert result.stderr == ""


@pytest.mark.parametrize(
    "line",
    [
        "warning: a similar pull request exists at " + PR_URL,
        PR_URL + " (draft)",
        "see " + PR_URL + " for details",
    ],
)
def test_silent_when_the_url_is_embedded_in_a_line(line):
    # The CLI prints the URL and nothing else on the line. Anything else is
    # prose that happens to quote a URL, including prose the URL starts.
    payload = bash_payload(PR_CREATE, tool_response={"stdout": line + "\n", "returnCode": 0})
    result = run_hook(payload)
    assert result.stdout == ""
    assert result.stderr == ""


def test_silent_when_the_url_is_only_in_another_response_field():
    # No fallback to scanning a serialisation of the whole response.
    payload = bash_payload(
        PR_CREATE,
        tool_response={"result": PR_URL, "returnCode": 0},
    )
    result = run_hook(payload)
    assert result.stdout == ""
    assert result.stderr == ""


def test_silent_for_non_bash_tool():
    payload = {
        "tool_name": "Write",
        "tool_input": {"file_path": "notes.txt", "content": PR_CREATE},
        "tool_response": {"stdout": PR_URL + "\n"},
    }
    result = run_hook(payload)
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_skip_env_var_disables_the_hook():
    payload = bash_payload(PR_CREATE, tool_response={"stdout": PR_URL + "\n", "returnCode": 0})
    env = dict(os.environ)
    env["ERUDI_SKIP_PR_HOOKS"] = "1"
    result = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
