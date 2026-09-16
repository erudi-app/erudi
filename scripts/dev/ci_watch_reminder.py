#!/usr/bin/env python3
"""
PostToolUse hook (Bash) for Claude Code -- once a pull request has actually
been opened, injects a reminder that its checks still have to be watched. It
is wired up in `.claude/settings.json`. Set `ERUDI_SKIP_PR_HOOKS=1` to turn it
off.

This hook never blocks. It writes a JSON object on stdout carrying
`hookSpecificOutput.additionalContext`, which the harness feeds back into the
session's context, and always exits 0. Nothing is ever written to stderr.

It speaks on exactly one condition: the command invoked `gh pr create` -- as a
program and its subcommand, not as three words somewhere on the line -- and
the command's **stdout** carries, as a complete line, a GitHub pull request
URL. That is precisely what the GitHub CLI prints on success and nothing else
does. The two negatives matter as much as the positive:

* stderr is never read. The CLI writes its progress chatter and its errors
  there, and a URL quoted in an error message is not a pull request that
  exists.
* there is no fallback to scanning a serialisation of the whole response, and
  no URL-less form of the reminder. A reminder that fires without a URL fires
  on commands that opened nothing, which is how it ended up interrupting work
  twice in one day.

Fails open and silent on any internal error.
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gh_pr_command import find_pr_create_segment  # noqa: E402

# Anchored: the whole line is the URL, which is how `gh pr create` prints it.
PR_URL_LINE = re.compile(r"^https://github\.com/[^/]+/[^/]+/pull/\d+$")

SKIP_ENV_VAR = "ERUDI_SKIP_PR_HOOKS"

# Deliberately free of runner names and job labels. Those change -- a runner
# image is retired, a leg is added -- and a reminder that names them is wrong
# the day after, in a file nobody thinks to update. Naming the suites keeps it
# true for as long as the suites exist.
REMINDER = (
    "A pull request was just opened: %s. It is not done yet. Watch every "
    "required check on it until all of them have concluded: the backend "
    "suite on each platform it runs on, the frontend suite, and the full-app "
    "build smoke, whose Linux leg is advisory rather than blocking while its "
    "other legs block. Do not report the work as finished, ready to merge, or "
    "complete while any required check is still queued or running. If a check "
    "fails, read the failing job's log first, find the actual cause, and push "
    "a fix to the same branch -- do not re-run the job hoping it turns green, "
    "and do not hand back a success summary over a red check."
)

FAILURE_KEYS = ("returnCode", "return_code", "exitCode", "exit_code", "code", "status")
ERROR_FLAGS = ("is_error", "isError", "error")


def response_failed(response):
    for key in FAILURE_KEYS:
        value = response.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value != 0:
            return True
        if isinstance(value, str) and value.isdigit() and int(value) != 0:
            return True
    for key in ERROR_FLAGS:
        if response.get(key) is True:
            return True
    return response.get("interrupted") is True


def pr_url_from_stdout(stdout):
    for line in stdout.splitlines():
        candidate = line.strip()
        if PR_URL_LINE.match(candidate):
            return candidate
    return None


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

    tool_input = payload.get("tool_input")
    command = tool_input.get("command") if isinstance(tool_input, dict) else ""
    if find_pr_create_segment(command) is None:
        return 0

    response = payload.get("tool_response")
    if not isinstance(response, dict):
        return 0
    if response_failed(response):
        return 0

    stdout = response.get("stdout")
    if not isinstance(stdout, str):
        return 0

    url = pr_url_from_stdout(stdout)
    if url is None:
        return 0

    output = {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": REMINDER % url,
        }
    }
    sys.stdout.write(json.dumps(output))
    return 0


def safe_main():
    try:
        return main()
    except Exception:
        return 0


if __name__ == "__main__":
    sys.exit(safe_main())
