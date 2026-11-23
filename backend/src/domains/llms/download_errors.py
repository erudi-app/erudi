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
anywhere in the exception chain.
"""

import errno
from typing import Optional

# Every code a download job can store. The interface has a sentence for each;
# one it does not know shows its generic sentence.
DOWNLOAD_ERROR_CODES = (
    # No network reaching the Hub.
    "offline",
    # The disk the model is written to is full.
    "disk_full",
    # The repository needs a Hugging Face account (gated, private: 401/403).
    "gated",
    # The repository, or a file of it, does not exist (404).
    "not_found",
    # The Hub refused for too many requests (429).
    "rate_limited",
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

_ATTRIBUTE = "download_error_code"


def with_download_code(exc: BaseException, code: str) -> BaseException:
    """``exc``, marked with ``code`` (raise sites that know the cause)."""
    if code not in DOWNLOAD_ERROR_CODES:
        raise ValueError(f"unknown download error code: {code}")
    setattr(exc, _ATTRIBUTE, code)
    return exc


def _chain(exc: BaseException):
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


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
    from src.core.hf_http import is_offline_error
    from src.domains.llms.services import OFFLINE_DOWNLOAD_MESSAGE

    for current in _chain(exc):
        marked = getattr(current, _ATTRIBUTE, None)
        if marked:
            return marked
        if isinstance(current, HuggingFaceAPIException) and (
            current.message == OFFLINE_DOWNLOAD_MESSAGE
        ):
            return "offline"
        if isinstance(current, OSError) and current.errno == errno.ENOSPC:
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
    # A network failure that reached here without download_llm's rename (the
    # progress thread, a finalization step): the Hub downloads' one predicate.
    if is_offline_error(exc):
        return "offline"
    return "unexpected"
