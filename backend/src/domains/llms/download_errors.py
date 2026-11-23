"""The stable code of a failed model download (#780).

A failed download stores two things on its job row: ``error_message``, the
English text of the exception, for the log and Diagnostics; and
``error_code``, one of ``DOWNLOAD_ERROR_CODES``, which the interface turns
into a sentence in its own language (``downloads:errors.codes.<code>``,
``frontend/src/utils/downloadErrors.js``). The backend never translates.

The code is read off the exception where the failure is recorded
(``download_error_code``). A refusal Erudi raises itself carries its code
(``with_download_code``, at the raise site); a failure that comes from below
(the network, the disk, the Hub's HTTP status) is recognised from its type,
on the exception or on its explicit ``__cause__`` chain (``raise ... from``).
The implicit ``__context__`` is never followed: an unrelated defect raised
while another error was being handled (an ``AttributeError`` in an
``except ConnectError`` block) is a defect, not that other error.

The ``environmental`` codes describe the user's machine or the Hub, not Erudi
being wrong: the interface's bug counter does not count them
(``frontend/src/utils/bugCounter.js``).
"""

import errno
from typing import Optional

from src.core.logging import logger

# Every code a download job can store. The interface has a sentence for each;
# one it does not know shows its generic sentence.
DOWNLOAD_ERROR_CODES = (
    # No network reaching the Hub.
    "offline",
    # The disk the model is written to is full (ENOSPC, or a quota: EDQUOT).
    "disk_full",
    # The repository needs a Hugging Face account (gated: 401/403).
    "gated",
    # The repository, or a file of it, does not exist or is private (404,
    # and the 401 the Hub answers for a repository it does not show).
    "not_found",
    # The Hub refused for too many requests (429).
    "rate_limited",
    # The Hub answered with a server error (5xx).
    "hub_unavailable",
    # The app stopped during the download (recorded by the next start).
    "interrupted",
    # The repository ships nothing this machine's engine runs.
    "no_engine_artifact",
    # The GGUF weights are published as raw chunks that must be joined.
    "raw_gguf_chunks",
    # The listing misses a part of the split quant it would download.
    "incomplete_split",
    # A file name of the repository would be written outside the model folder.
    "unsafe_file_name",
    # The quant is known not to run on this machine's engine.
    "not_runnable",
    # The files arrived but the engine's integrity check refused them.
    "integrity_failed",
    # Anything else: a defect, logged with its traceback.
    "unexpected",
)

# The codes that are the environment's doing, mirrored by
# ENVIRONMENTAL_DOWNLOAD_CODES in frontend/src/utils/bugCounter.js.
ENVIRONMENTAL_DOWNLOAD_CODES = (
    "offline",
    "disk_full",
    "gated",
    "not_found",
    "rate_limited",
    "hub_unavailable",
    "interrupted",
)

# ENOSPC everywhere; EDQUOT where the platform defines it (122 on Linux, 69 on
# macOS). Windows' ERROR_DISK_FULL reaches Python as ENOSPC.
_DISK_FULL_ERRNOS = frozenset(
    code for code in (errno.ENOSPC, getattr(errno, "EDQUOT", None)) if code is not None
)

_ATTRIBUTE = "download_error_code"


def with_download_code(exc: BaseException, code: str) -> BaseException:
    """``exc``, marked with ``code`` (raise sites that know the cause)."""
    if code not in DOWNLOAD_ERROR_CODES:
        raise ValueError(f"unknown download error code: {code}")
    setattr(exc, _ATTRIBUTE, code)
    return exc


def _cause_chain(exc: BaseException):
    """``exc``, then each explicit ``__cause__`` (never ``__context__``)."""
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__


def _http_status(exc: BaseException) -> Optional[int]:
    return getattr(getattr(exc, "response", None), "status_code", None)


def download_error_code(exc: BaseException) -> str:
    """The code of the download failure ``exc`` (``"unexpected"`` if unknown)."""
    from huggingface_hub.errors import (
        EntryNotFoundError,
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
    )

    from src.core.exceptions import HuggingFaceAPIException
    from src.core.hf_http import NETWORK_ERRORS
    from src.domains.llms.services import OFFLINE_DOWNLOAD_MESSAGE

    for current in _cause_chain(exc):
        marked = getattr(current, _ATTRIBUTE, None)
        if marked:
            return marked
        if isinstance(current, HuggingFaceAPIException) and (
            current.message == OFFLINE_DOWNLOAD_MESSAGE
        ):
            return "offline"
        # A network failure that reached here without download_llm's rename
        # (the progress thread, a finalization step).
        if isinstance(current, NETWORK_ERRORS):
            return "offline"
        if isinstance(current, OSError) and current.errno in _DISK_FULL_ERRNOS:
            return "disk_full"
        # GatedRepoError is a RepositoryNotFoundError: checked first.
        if isinstance(current, GatedRepoError):
            return "gated"
        if isinstance(current, (RepositoryNotFoundError, EntryNotFoundError)):
            return "not_found"
        if isinstance(current, HfHubHTTPError):
            status = _http_status(current)
            if status in (401, 403):
                return "gated"
            if status == 404:
                return "not_found"
            if status == 429:
                return "rate_limited"
            if status is not None and status >= 500:
                return "hub_unavailable"
    return "unexpected"


def recorded_download_error_code(exc: BaseException) -> str:
    """``download_error_code`` for a failure being written to its job row.

    The classification never stops the job from being marked failed: if it
    raises, the job gets ``"unexpected"`` (the interface's generic sentence)
    and one WARNING with the traceback says the code could not be derived.
    """
    try:
        return download_error_code(exc)
    except Exception:
        # Degraded, not broken: the failure is still recorded, generically.
        logger.warning(
            f"Could not derive the code of a download failure ({type(exc).__name__}); "
            f"recording it as unexpected",
            exc_info=True,
        )
        return "unexpected"
