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
from urllib.parse import unquote

from src.core.exceptions import AppBaseException

MAX_DETAIL_CHARS = 300

# What is read of a message: a cut path or a cut secret pair is still reduced or
# masked, and the cost stays bounded whatever the exception carries.
_MAX_INPUT_CHARS = 16 * 1024

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
# gives inside an exception message) or a slash. A segment followed by a
# backslash may hold spaces and apostrophes (O'Brien); one followed by a slash
# is a plain word or a capitalised word run, so a "/" in the prose after a path
# ("and/or", "docs/x.md") never extends it.
_W_SEP = r"(?:\\{1,2}|/)"
_W_WORD = rf"[^\s\\/:*?\"<>|']{{1,{_SEG_MAX}}}"
_W_SPACED = rf"(?=[A-Z0-9(]){_W_WORD}(?: (?=[A-Z0-9(]){_W_WORD}){{1,4}}"
_W_SEG = rf"(?:[^\\/:*?\"<>|\r\n']|'(?![\s,\])])){{1,{_SEG_MAX}}}"
_W_BODY = rf"(?:{_W_WORD}{_W_SEP}|{_W_SPACED}/|{_W_SEG}\\{{1,2}})*"
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
_SK_KEY = re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}")
_SECRET_PAIR = re.compile(
    r"\b([\w\-]{0,40}(?:api[_\-]?key|password|passwd|pwd|token|secret|authorization))"
    r"([\"']?\s*[=:]\s*)(?!(?:Bearer|Basic)\b)(\"[^\"]*\"|'[^']*'|[^\s&'\",}]+)",
    re.IGNORECASE,
)
_HF_TOKEN = re.compile(r"\bhf_[A-Za-z0-9]{30,}")
_FILE_URL = re.compile(r"file://([^\s'\"<>]*)", re.IGNORECASE)
# A path written percent-encoded ("%2FUsers%2Fjane%2Fx.gguf").
_ENCODED_PATH = re.compile(r"(?<![^\s'\"(=])%2F[^\s'\"<>]*", re.IGNORECASE)
_WIN_PATH = re.compile(rf"(?<![A-Za-z0-9])[A-Za-z]:{_W_SEP}{_W_BODY}{_W_LEAF}")
_UNC_PATH = re.compile(rf"(?<![\\\w])\\{{2,4}}[^\s\\'\"]+\\{{1,2}}{_W_BODY}{_W_LEAF}")
# A Windows path whose drive was dropped ("\\Users\\jane\\x.gguf").
_WIN_NODRIVE = re.compile(rf"(?<![\w\\:.\-])\\{{1,2}}Users\\{{1,2}}{_W_BODY}{_W_LEAF}")
_POSIX_PATH = re.compile(rf"(?<![\w/.~\-])/(?:{_P_SEG}/)+{_P_LEAF}")

# The first segment of a URL path the app itself serves or calls.
_URL_ROOTS = re.compile(r"v\d+|api|erudi|health|openai")
_HOME_ROOTS = {"users", "home"}


def _segments(path: str) -> list[str]:
    return [s for s in re.split(r"[\\/]+", path) if s]


def _reduce_path(match: "re.Match[str]") -> str:
    return _reduce(match.group(0))


def _reduce(path: str) -> str:
    """A path becomes its file name; a home directory alone becomes ``~``."""
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


_SECRET_LIKE = re.compile(r"[A-Za-z0-9._~+/=\-]{12,}")
_ALWAYS_SECRET_TOKENS = ("access_token", "refresh_token", "id_token", "access-token")


def _mask_pair(match: "re.Match[str]") -> str:
    key, value = match.group(1).lower(), match.group(3)
    quote = value[0] if value[0] in "\"'" else ""
    if key.endswith("token") and not key.endswith(_ALWAYS_SECRET_TOKENS):
        # A bare "token" is also tokenizer vocabulary (``<|im_end|>``,
        # ``eos_token='</s>'``): only a value shaped like a secret is masked.
        if not _SECRET_LIKE.fullmatch(value.strip("\"'")):
            return match.group(0)
    return f"{match.group(1)}{match.group(2)}{quote}{_MASK}{quote}"


def _decode_file_url(match: "re.Match[str]") -> str:
    path = unquote(match.group(1))
    return path[1:] if re.match(r"/[A-Za-z]:", path) else path


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
    text = _SK_KEY.sub(f"sk-{_MASK}", text)
    # Decoded to the path they spell, then reduced like any other path.
    text = _FILE_URL.sub(_decode_file_url, text)
    text = _ENCODED_PATH.sub(lambda m: unquote(m.group(0)), text)
    text = _WIN_PATH.sub(_reduce_path, text)
    text = _UNC_PATH.sub(_reduce_path, text)
    text = _WIN_NODRIVE.sub(_reduce_path, text)
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


_ONLY_A_QUOTE = re.compile(r"'[^']*'|\"[^\"]*\"")


def _message_of(exc: BaseException) -> str:
    """The sanitised message of one exception, or "" when it says nothing.

    Never raises: an exception whose ``str`` fails says nothing. A message that
    is only a quoted key (``KeyError('choices')``) names its type; empty quotes
    say nothing.
    """
    try:
        raw = str(exc)[:_MAX_INPUT_CHARS]
    except Exception:
        return ""
    message = sanitize(raw).strip()
    if message in ("''", '""'):
        return ""
    if _ONLY_A_QUOTE.fullmatch(message):
        return f"{type(exc).__name__}: {message}"
    return message


def error_detail(exc: BaseException) -> str:
    """The sanitised, single-line, at most ``MAX_DETAIL_CHARS`` cause of ``exc``.

    The innermost exception of the chain whose sanitised message says
    something; when none does, the outermost exception's type name.
    """
    chain = _chain(exc)
    message = ""
    for candidate in reversed(chain):
        message = _message_of(candidate)
        if message:
            break
    else:
        message = type(exc).__name__
    if len(message) > MAX_DETAIL_CHARS:
        message = message[: MAX_DETAIL_CHARS - 3].rstrip() + "..."
    return message
