"""Recognise a real `gh pr create` or `gh pr merge` invocation in a command.

Shared by the hooks in this directory, which all have to answer the same kind
of question before doing anything: does this command actually open (or merge)
a pull request?

Matching the raw command string against `gh pr create` is not an answer, it is
a bug. `grep -rn "gh pr create" scripts/` contains the words and opens
nothing; so does `echo "gh pr create" >> notes.md`, and so does the test file
that exercises these very hooks. A hook that fires on those is worse than
noise for the readiness check, which has a one-shot budget per branch: a
grep spends the budget, and the real invocation that follows sails through
unasked. `gh pr merge` is recognised the same way, for the same reason.

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

MERGE_SUBCOMMAND = ("pr", "merge")

# Spellings of `gh pr create` that open no pull request: help text, a hand-off
# to the browser, and a rehearsal.
NON_CREATING_FLAGS = ("--help", "-h", "--web", "-w", "--dry-run")

# Spellings of `gh pr merge` that merge nothing: only the help text.
NON_MERGING_FLAGS = ("--help", "-h")

BASE_FLAGS = ("--base", "-B")

# `gh pr merge` flags that consume the following token as their value, so the
# positional pull-request argument is never confused with one of their values.
MERGE_VALUE_FLAGS = (
    "--body",
    "-b",
    "--body-file",
    "-F",
    "--subject",
    "-t",
    "--match-head-commit",
    "--author-email",
)


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


def _segment_invokes(tokens, subcommand, excluded_flags):
    """True when the segment is a real `gh <subcommand>` invocation.

    Skips leading environment assignments, requires the program to be `gh`
    (or a path ending in `/gh`), and requires the words before the first flag
    to be exactly `subcommand`. `excluded_flags` names the spellings that turn
    the invocation into something other than the action being recognised
    (help text, a browser hand-off, a rehearsal); their presence disqualifies
    the segment.
    """
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
    if tuple(words[: len(subcommand)]) != subcommand:
        return False

    return not flag_present(tokens, excluded_flags)


def segment_creates_pull_request(tokens):
    return _segment_invokes(tokens, SUBCOMMAND, NON_CREATING_FLAGS)


def segment_merges_pull_request(tokens):
    return _segment_invokes(tokens, MERGE_SUBCOMMAND, NON_MERGING_FLAGS)


def find_pr_create_segment(command):
    """Return the tokens of the first segment that opens a pull request.

    None when the command opens none. Returning the segment rather than a
    boolean means a caller reading `--title`/`--body`/`--base` reads them off
    the invocation itself, never off whatever else shares the command line.
    """
    return _find_segment(command, segment_creates_pull_request)


def find_pr_merge_segment(command):
    """Return the tokens of the first segment that merges a pull request.

    None when the command merges none. Same discipline as its `create`
    sibling: a real `gh pr merge` invocation, never the three words quoted in
    an argument, a filename or another program's search string.
    """
    return _find_segment(command, segment_merges_pull_request)


def _find_segment(command, predicate):
    if not isinstance(command, str) or "gh" not in command:
        return None
    try:
        tokens = tokenize(command)
    except ValueError:
        return None
    for segment in split_segments(tokens):
        if predicate(segment):
            return segment
    return None


def pr_argument_from_segment(tokens):
    """The positional pull-request argument of a `gh pr merge` segment.

    A number, a URL or a branch name, or None when the segment names no pull
    request and the caller should fall back to the current branch. Value-
    taking flags (`--body`, `--subject`, ...) are stepped over so their values
    are never mistaken for the positional argument.
    """
    index = 0
    while index < len(tokens) and ENV_ASSIGNMENT.match(tokens[index]):
        index += 1
    # Step over the program name and the subcommand words.
    index += 1
    seen = 0
    while (
        index < len(tokens) and seen < len(MERGE_SUBCOMMAND) and not tokens[index].startswith("-")
    ):
        index += 1
        seen += 1
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("-"):
            _, consumed = flag_value(tokens, index, MERGE_VALUE_FLAGS)
            index += consumed if consumed else 1
            continue
        return token
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
