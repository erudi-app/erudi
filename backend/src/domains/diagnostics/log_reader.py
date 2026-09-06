"""Bounded, level-filtered reader for ``backend.log``.

Two constraints shape this module.

**It reads a window, never a file.** ``backend.log`` rotates at 10 MB, so the
live file is routinely megabytes long. The reader seeks to the last
``DEFAULT_TAIL_BYTES`` and decodes only that, then drops the first line of the
window, which is almost certainly cut in half.

**It keeps WARNING and above, and nothing else.** Erudi logs what it processes
on purpose -- the question asked, a preview of the answer, document names --
because nothing leaves the machine (see ``docs/privacy.md``). Those lines are
INFO. Showing them in a panel whose whole purpose is to be copied into a public
issue would turn a deliberate local-logging decision into a leak, so the filter
is a privacy property of this feature and its test is written as one.

A log *record* is not a log *line*. ``AppBaseException`` writes a header line
followed by three continuation lines carrying the status code, the error code
and the message. A record therefore starts at a line matching ``RECORD_RE`` and
absorbs every following line that does not. Continuations inherit the level of
their header: those under a kept record are kept, those under a filtered record
are dropped, and continuations with no header at all -- the top of a window that
started mid-record -- are dropped too, because their level is unknown.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# Read window. Comfortably more than 200 records of a few hundred bytes, and
# small enough that reading it costs nothing on any disk.
DEFAULT_TAIL_BYTES = 512 * 1024

# Records returned by default, newest last.
DEFAULT_LIMIT = 200

# Per-record cap. One stack trace should not be able to dominate the payload.
MAX_MESSAGE_CHARS = 4000

# Levels worth reporting. DEBUG and INFO are excluded; see the module docstring.
KEPT_LEVELS = ("WARNING", "ERROR", "CRITICAL")

# The head of a record as written by ``FILE_LOG_FORMAT`` in src/core/logging.py:
#     [ERROR] 2026-09-05T23:03:15.036Z [be-1f2e3d4c] - erudi - file.py:42 - text
# The request id is "-" outside a request.
RECORD_RE = re.compile(
    r"^\[(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL)\] "
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}T[0-9:.]+Z) "
    r"\[(?P<request_id>[^\]]*)\] - "
    r"(?P<logger>[^ ]+) - (?P<location>[^ ]+) - "
    r"(?P<message>.*)$"
)


def read_tail(path: Path, max_bytes: int = DEFAULT_TAIL_BYTES) -> str:
    """Return at most the last ``max_bytes`` of ``path`` as newline-normalised text.

    The file is opened in binary so the window can be a byte offset, which means
    no newline translation happens on the way in. ``logging``'s ``FileHandler``
    opens in text mode, so on Windows every record really is terminated with
    CRLF, and every parsed record would otherwise carry a trailing ``\\r`` into
    the panel and into the text a user pastes into an issue. ``\\r\\n`` is
    therefore collapsed to ``\\n`` here, so the reader's contract is the same
    text on every OS.

    A lone ``\\r`` is left alone. No writer we have ends a line with one; a bare
    carriage return in a log file is progress-bar output captured mid-line, and
    turning it into a newline would split one record into several and invent
    continuation lines that were never written.

    The first line of the window is dropped when the window does not start at
    the beginning of the file, because a byte offset lands mid-line.

    Args:
        path: File to read. A missing or unreadable file is not an error here.
        max_bytes: Size of the window to read from the end of the file.

    Returns:
        str: The decoded, newline-normalised window, or "" when the file cannot
        be read. It is a bound, not a length: the returned string is shorter
        than ``max_bytes`` whenever the partial first line is dropped, whenever
        the window contains multi-byte characters, and on Windows by one
        character per line, because the window is sized in bytes before any of
        that happens.
    """
    try:
        with open(path, "rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            start = max(0, size - max_bytes)
            handle.seek(start)
            window = handle.read()
    except OSError:
        return ""

    text = window.decode("utf-8", errors="replace").replace("\r\n", "\n")
    if start > 0:
        # The window began mid-line; that fragment belongs to a record whose
        # header we did not read.
        _, _, text = text.partition("\n")
    return text


def parse_records(
    text: str,
    limit: int = DEFAULT_LIMIT,
    levels: tuple = KEPT_LEVELS,
) -> List[Dict[str, Any]]:
    """Group ``text`` into records and keep the last ``limit`` at ``levels``.

    Args:
        text: Raw log text, typically the output of :func:`read_tail`.
        limit: Maximum number of records to return.
        levels: Levels to keep. Everything else, header and continuations
            alike, is discarded.

    Returns:
        list[dict]: Records ordered oldest first, each with ``timestamp``,
        ``level``, ``request_id`` (None outside a request) and ``message``.
    """
    records: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    keeping = False

    # `split("\n")`, never `splitlines()`. A record boundary is a newline and
    # nothing else, but `splitlines()` also breaks on \r, \v, \f, \x1c-\x1e,
    # \x85, \u2028 and \u2029 -- any of which can sit inside a logged prompt,
    # because Erudi logs content at INFO deliberately. Splitting there would
    # let the tail of an INFO message be re-read as a fresh record header and
    # promoted into the panel at whatever level it claims, defeating the filter
    # this module exists to enforce. `read_tail` has already normalised CRLF.
    lines = text.split("\n")
    if lines and lines[-1] == "":
        # A well-formed log file ends with a record terminator, and `split`
        # turns that final newline into a trailing empty element. It is the
        # terminator, not a blank line, so it must not become a continuation.
        lines.pop()

    for line in lines:
        match = RECORD_RE.match(line)
        if match:
            level = match.group("level")
            keeping = level in levels
            if not keeping:
                current = None
                continue
            request_id = match.group("request_id")
            current = {
                "timestamp": match.group("timestamp"),
                "level": level,
                "request_id": request_id if request_id and request_id != "-" else None,
                "message": match.group("message"),
            }
            records.append(current)
        elif keeping and current is not None:
            current["message"] = f"{current['message']}\n{line}"
        # else: a continuation of a filtered record, or of a record whose
        # header fell outside the window. Both are dropped.

    for record in records:
        if len(record["message"]) > MAX_MESSAGE_CHARS:
            dropped = len(record["message"]) - MAX_MESSAGE_CHARS
            record["message"] = f"{record['message'][:MAX_MESSAGE_CHARS]}... [+{dropped}]"

    return records[-limit:] if limit else records


def recent_errors(
    path: Path,
    limit: int = DEFAULT_LIMIT,
    max_bytes: int = DEFAULT_TAIL_BYTES,
) -> List[Dict[str, Any]]:
    """Return the last ``limit`` WARNING-or-worse records of ``path``.

    Never raises: a missing, locked or unreadable log yields an empty list, so
    the caller can still report everything else it knows.
    """
    return parse_records(read_tail(path, max_bytes=max_bytes), limit=limit)
