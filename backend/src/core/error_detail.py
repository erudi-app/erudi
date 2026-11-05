"""The cleaned cause of a failed turn, shown under the generic error headline.

``error_detail`` answers one question for the user (and for a bug report): what
did the exception that ended the turn actually say? It is the message of the
leaf cause, on one line, with every machine-specific or secret fragment removed
and at most ``MAX_DETAIL_CHARS`` characters. It is never a traceback and never
translated: it is quoted as the backend produced it.

Pure and dependency-free so the three catch sites (agent construction, agent
streaming, the conversation service's last-resort handler) share one rule.
"""

import re
from typing import Optional

from src.core.exceptions import AppBaseException

MAX_DETAIL_CHARS = 300

_MASK = "***"

# Characters that end a path segment on each family of separators. Directory
# segments may hold spaces ("Application Support", "John Doe"); the last
# segment stops at whitespace so the prose after a path survives.
_POSIX_DIR = r"[^/\r\n'\"<>:|*?]+"
_POSIX_LEAF = r"[^\s/'\"<>:|*?]*"
_WIN_DIR = r"[^\\/:*?\"<>|\r\n]+"
_WIN_LEAF = r"[^\s\\/:*?\"<>|]*"

_DB_URL = re.compile(
    r"\b(postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|redis|sqlite|mssql|oracle)"
    r"(?:\+\w+)?://[^\s'\"<>]+",
    re.IGNORECASE,
)
_URL_USERINFO = re.compile(r"(\b[a-z][a-z0-9+.\-]*://)[^/\s@'\"<>]+@", re.IGNORECASE)
_BEARER = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=\-]+", re.IGNORECASE)
_API_KEY_FLAG = re.compile(r"(--api-key)(=|\s+|['\"],\s*['\"])[^\s'\",\]]+")
_API_KEY_PAIR = re.compile(
    r"\b(api[_\-]?key)([\"']?\s*[=:]\s*[\"']?)[^\s&'\",}]+",
    re.IGNORECASE,
)
_FILE_URL = re.compile(r"file://[^\s'\"<>]*", re.IGNORECASE)
_UNC_PATH = re.compile(rf"\\\\[^\s\\]+\\(?:{_WIN_DIR}\\)*{_WIN_LEAF}")
_WIN_PATH = re.compile(rf"(?<![A-Za-z0-9])[A-Za-z]:[\\/](?:{_WIN_DIR}[\\/])*{_WIN_LEAF}")
_POSIX_PATH = re.compile(rf"(?<![\w:/.~\-])/(?:{_POSIX_DIR}/)+{_POSIX_LEAF}")


def _basename(path: str) -> str:
    return re.split(r"[\\/]", path.rstrip("\\/"))[-1]


def sanitize(text: str) -> str:
    """One line with paths reduced to file names and credentials masked.

    Credentials go first (a database URL or a URL's user name is masked whole,
    before any path rule can reshape it); paths follow.
    """
    text = " ".join(text.split())
    text = _DB_URL.sub(lambda m: f"{m.group(1).lower()}://{_MASK}", text)
    text = _URL_USERINFO.sub(lambda m: f"{m.group(1)}{_MASK}@", text)
    text = _BEARER.sub(f"Bearer {_MASK}", text)
    text = _API_KEY_FLAG.sub(lambda m: f"{m.group(1)}{m.group(2)}{_MASK}", text)
    text = _API_KEY_PAIR.sub(lambda m: f"{m.group(1)}{m.group(2)}{_MASK}", text)
    text = _FILE_URL.sub(lambda m: _basename(m.group(0)), text)
    text = _UNC_PATH.sub(lambda m: _basename(m.group(0)), text)
    text = _WIN_PATH.sub(lambda m: _basename(m.group(0)), text)
    text = _POSIX_PATH.sub(lambda m: _basename(m.group(0)), text)
    return text


def _leaf(exc: BaseException) -> BaseException:
    """The innermost exception of the chain whose message says something.

    ``__cause__`` is followed first, then ``__context__`` unless the raiser
    suppressed it (``raise ... from None``). An ``AppBaseException`` ends the
    walk: its message is written for the reader, what it wraps is not.
    """
    seen: set[int] = set()
    current: BaseException = exc
    while id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, AppBaseException):
            return current
        nxt: Optional[BaseException] = current.__cause__
        if nxt is None and not current.__suppress_context__:
            nxt = current.__context__
        if nxt is None:
            return current
        current = nxt
    return current


def error_detail(exc: BaseException) -> str:
    """The sanitised, single-line, at most ``MAX_DETAIL_CHARS`` cause of ``exc``."""
    leaf = _leaf(exc)
    message = sanitize(str(leaf)).strip() or type(leaf).__name__
    if len(message) > MAX_DETAIL_CHARS:
        message = message[: MAX_DETAIL_CHARS - 3].rstrip() + "..."
    return message
