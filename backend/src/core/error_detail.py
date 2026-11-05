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

# A path segment is capped so a path cannot run on into the next sentence.
_SEG_MAX = 64

# POSIX segments. A segment with spaces ("Application Support", "John Doe")
# is accepted only when each of its words starts with a capital, a digit or a
# parenthesis, so prose after a path ("a.gguf and/or ...") is never absorbed.
_P_WORD = rf"[^\s/'\"<>:|*?]{{1,{_SEG_MAX}}}"
_P_SPACED = rf"(?=[A-Z0-9(]){_P_WORD}(?: (?=[A-Z0-9(]){_P_WORD}){{1,4}}"
_P_SEG = rf"(?:{_P_SPACED}|{_P_WORD})"
_P_LEAF = r"[^\s/'\"<>:|*?]*"

# Windows segments: the separator is one backslash, two (the form ``repr``
# gives inside an exception message) or a slash.
_W_SEP = r"(?:\\{1,2}|/)"
_W_SEG = rf"[^\\/:*?\"<>|\r\n]{{1,{_SEG_MAX}}}"
_W_LEAF = r"[^\s\\/:*?\"<>|']*"

_DB_URL = re.compile(
    r"\b(postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|redis|sqlite|mssql|oracle)"
    r"(?:\+\w+)?://[^\s'\"<>]+",
    re.IGNORECASE,
)
# Everything up to the LAST "@" before the host: a password may hold a raw "@".
_URL_USERINFO = re.compile(r"(\b[a-z][a-z0-9+.\-]{0,30}://)[^/\s'\"<>]+@", re.IGNORECASE)
_BEARER = re.compile(r"\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=\-]+", re.IGNORECASE)
_API_KEY_FLAG = re.compile(r"(--api-key)(=|\s+|['\"],\s*['\"])[^\s'\",\]]+")
_SECRET_PAIR = re.compile(
    r"\b([\w\-]{0,40}(?:api[_\-]?key|password|passwd|pwd|token|secret|authorization))"
    r"([\"']?\s*[=:]\s*)(?!(?:Bearer|Basic)\b)(\"[^\"]*\"|'[^']*'|[^\s&'\",}]+)",
    re.IGNORECASE,
)
_HF_TOKEN = re.compile(r"\bhf_[A-Za-z0-9]{30,}")
_FILE_URL = re.compile(r"file://[^\s'\"<>]*", re.IGNORECASE)
_WIN_PATH = re.compile(
    rf"(?<![A-Za-z0-9])[A-Za-z]:{_W_SEP}(?:{_W_SEG}{_W_SEP})*{_W_LEAF}"
)
_UNC_PATH = re.compile(
    rf"(?<![\\\w])\\{{2,4}}[^\s\\'\"]+\\{{1,2}}(?:{_W_SEG}\\{{1,2}})*{_W_LEAF}"
)
_POSIX_PATH = re.compile(rf"(?<![\w/.~\-])/(?:{_P_SEG}/)+{_P_LEAF}")

# The first segment of a URL path the app itself serves or calls.
_URL_ROOTS = re.compile(r"v\d+|api|erudi|health|openai")
_HOME_ROOTS = {"users", "home"}


def _segments(path: str) -> list[str]:
    return [s for s in re.split(r"[\\/]+", path) if s]


def _reduce_path(match: "re.Match[str]") -> str:
    """A path becomes its file name; a home directory alone becomes ``~``."""
    path = match.group(0)
    parts = _segments(path)
    if not parts:
        return path
    if re.match(r"[A-Za-z]:", parts[0]):
        parts = parts[1:]
    if not parts:
        return path
    if len(parts) == 2 and parts[0].lower() in _HOME_ROOTS:
        return "~"
    if _URL_ROOTS.fullmatch(parts[0]) and path.startswith("/"):
        return path
    return parts[-1]


def _mask_pair(match: "re.Match[str]") -> str:
    value = match.group(3)
    quote = value[0] if value[0] in "\"'" else ""
    return f"{match.group(1)}{match.group(2)}{quote}{_MASK}{quote}"


def sanitize(text: str) -> str:
    """One line with paths reduced to file names and credentials masked.

    Credentials go first (a database URL or a URL's user name is masked whole,
    before any path rule can reshape it); paths follow, on the text as it was
    written (a path never spans a line break); the whitespace collapse is last.
    """
    text = _DB_URL.sub(lambda m: f"{m.group(1).lower()}://{_MASK}", text)
    text = _URL_USERINFO.sub(lambda m: f"{m.group(1)}{_MASK}@", text)
    text = _BEARER.sub(lambda m: f"{m.group(1).capitalize()} {_MASK}", text)
    text = _API_KEY_FLAG.sub(lambda m: f"{m.group(1)}{m.group(2)}{_MASK}", text)
    text = _SECRET_PAIR.sub(_mask_pair, text)
    text = _HF_TOKEN.sub(f"hf_{_MASK}", text)
    text = _FILE_URL.sub(lambda m: _segments(m.group(0))[-1] if _segments(m.group(0)) else "", text)
    text = _WIN_PATH.sub(_reduce_path, text)
    text = _UNC_PATH.sub(_reduce_path, text)
    text = _POSIX_PATH.sub(_reduce_path, text)
    return " ".join(text.split())


def _chain(exc: BaseException) -> list[BaseException]:
    """The exception and what led to it, outermost first.

    ``__cause__`` is followed first, then ``__context__`` unless the raiser
    suppressed it (``raise ... from None``). An ``AppBaseException`` ends the
    walk: its message is written for the reader, what it wraps is not.
    """
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        if isinstance(current, AppBaseException):
            break
        nxt = current.__cause__
        if nxt is None and not current.__suppress_context__:
            nxt = current.__context__
        current = nxt
    return chain


def error_detail(exc: BaseException) -> str:
    """The sanitised, single-line, at most ``MAX_DETAIL_CHARS`` cause of ``exc``.

    The innermost exception of the chain whose sanitised message says
    something; when none does, the outermost exception's type name.
    """
    chain = _chain(exc)
    message = ""
    for candidate in reversed(chain):
        message = sanitize(str(candidate)).strip()
        if message:
            break
    else:
        message = type(exc).__name__
    if len(message) > MAX_DETAIL_CHARS:
        message = message[: MAX_DETAIL_CHARS - 3].rstrip() + "..."
    return message
