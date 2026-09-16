"""Recognise a real `gh pr create` invocation inside a Bash command string.

Shared by the two hooks in this directory, which both have to answer the same
question before doing anything: does this command actually open a pull
request?

Matching the raw command string against `gh pr create` is not an answer, it is
a bug. `grep -rn "gh pr create" scripts/` contains the words and opens
nothing; so does `echo "gh pr create" >> notes.md`, and so does the test file
that exercises these very hooks. A hook that fires on those is worse than
noise for the readiness check, which has a one-shot budget per branch: a
grep spends the budget, and the real invocation that follows sails through
unasked.

So the command is tokenized the way a shell would, split into segments on
`;`, `&&`, `||`, `|`, `&` and parentheses, and each segment is examined as an
invocation: leading environment assignments are skipped, the program name has
to be `gh` (or a path ending in `/gh`), and the words before the first flag
have to be `pr create`. That accepts `cd frontend && gh pr create --fill` and
`GH_TOKEN=x gh pr create`, and rejects any occurrence of the three words
inside an argument, a filename or a quoted string -- because in those shapes
they are not the program and its subcommand.

Segmentation with `shlex.shlex(punctuation_chars=True)` is the same approach
the repository's other command-inspecting hook uses; keeping one
implementation here means the two hooks cannot drift apart in what they
consider a pull request being opened.

Known limitation: a body supplied through a heredoc (`--body-file -` with
`<<EOF`) is not extracted. The invocation is still recognised, so the
path-based checks run; only the text checks see nothing.
"""

import os
import re
import shlex

# Tokens shlex emits for shell control operators. `<` and `>` are punctuation
# characters too, but they are redirections rather than separators: a segment
# keeps them, and they simply never look like a program name.
SEPARATOR = re.compile(r"^[;&|()]+$")

ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

SUBCOMMAND = ("pr", "create")

# Spellings of `gh pr create` that open no pull request: help text, a hand-off
# to the browser, and a rehearsal.
NON_CREATING_FLAGS = ("--help", "-h", "--web", "-w", "--dry-run")

BASE_FLAGS = ("--base", "-B")


def tokenize(text):
    """Shell-tokenize, keeping ; && || | & ( ) as their own tokens.

    Raises ValueError when the quoting is unbalanced. Callers treat that as
    "this is not a recognisable invocation" and stay silent, which is the
    fail-open direction for a double-check.
    """
    lexer = shlex.shlex(text, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    return list(lexer)


def split_segments(tokens):
    segments = []
    current = []
    for token in tokens:
        if SEPARATOR.match(token):
            if current:
                segments.append(current)
            current = []
            continue
        current.append(token)
    if current:
        segments.append(current)
    return segments


def flag_value(tokens, index, names):
    """Return (value, tokens_consumed) if tokens[index] carries one of names.

    Handles `--name value`, `--name=value`, `-n value` and `-nvalue`.
    """
    token = tokens[index]
    for name in names:
        if token == name:
            if index + 1 < len(tokens):
                return tokens[index + 1], 2
            return None, 1
        if token.startswith(name + "="):
            return token[len(name) + 1 :], 1
        if len(name) == 2 and name.startswith("-") and not name.startswith("--"):
            if token.startswith(name) and len(token) > 2 and not token.startswith("--"):
                return token[2:], 1
    return None, 0


def flag_present(tokens, names):
    for token in tokens:
        for name in names:
            if token == name or token.startswith(name + "="):
                return True
    return False


def segment_creates_pull_request(tokens):
    index = 0
    while index < len(tokens) and ENV_ASSIGNMENT.match(tokens[index]):
        index += 1
    if index >= len(tokens):
        return False
    if os.path.basename(tokens[index]) != "gh":
        return False

    # The subcommand path in `gh` always comes before any flag, so the words
    # up to the first flag are the whole of it. Stopping at the first flag is
    # what keeps `gh pr list --search "gh pr create"` out.
    words = []
    for token in tokens[index + 1 :]:
        if token.startswith("-"):
            break
        words.append(token)
    if tuple(words[: len(SUBCOMMAND)]) != SUBCOMMAND:
        return False

    return not flag_present(tokens, NON_CREATING_FLAGS)


def find_pr_create_segment(command):
    """Return the tokens of the first segment that opens a pull request.

    None when the command opens none. Returning the segment rather than a
    boolean means a caller reading `--title`/`--body`/`--base` reads them off
    the invocation itself, never off whatever else shares the command line.
    """
    if not isinstance(command, str) or "gh" not in command:
        return None
    try:
        tokens = tokenize(command)
    except ValueError:
        return None
    for segment in split_segments(tokens):
        if segment_creates_pull_request(segment):
            return segment
    return None


def base_from_segment(tokens):
    """The branch named by an explicit --base/-B, or None."""
    index = 0
    while index < len(tokens):
        value, consumed = flag_value(tokens, index, BASE_FLAGS)
        if consumed:
            return value or None
        index += 1
    return None
