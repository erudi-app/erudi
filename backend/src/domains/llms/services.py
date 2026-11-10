"""Business logic for LLM download orchestration with progress tracking.

This module manages the complete lifecycle of downloading LLM models from HuggingFace
and tracking progress in real-time. Every catalog link is a pre-built engine-format
quant, so downloads are moved into place with no local conversion.

Architecture:
    1. download_llm() orchestrates the full pipeline (select files → download → move).
    2. _select_download_files() picks the exact files to fetch (single best GGUF quant).
    3. DownloadTracker monitors progress and ETA across concurrent file downloads.
    4. make_callback() creates fsspec hooks to update DownloadTracker on each chunk
       and to abort the transfer (DownloadCancelled) once the tracker is cancelled.
    5. update_db_with_progress() runs in background thread to persist status to DB.
    6. download_files_concurrent() parallelizes .safetensors shard downloads.

Engine Integration:
    - Every catalog link is a pre-built engine-format quant; no local conversion.
    - GGUF engines (USES_GGUF) download one best quant file + mmproj + small aux files.

Download Flow:
    ┌─────────────┐
    │ HuggingFace │
    │  Repository │
    └──────┬──────┘
           │ (1) Select files & compute total bytes
           ↓
    ┌──────────────┐
    │ temp_save_dir│ ← Download model weights + config files
    └──────┬───────┘
           │ (2) shutil.move
           ↓
    ┌──────────────┐
    │final_save_dir│ ← Ready for inference
    └──────────────┘

Example:
    from src.domains.llms.services import download_llm
    import asyncio

    # Download a pre-built quant of Llama-3-8B
    final_path = asyncio.run(download_llm(
        model_link="meta-llama/Llama-3-8B-Instruct",
        model_id=42,
        temp_save_dir=config.LLM_DIR / "temp_42",
        final_save_dir=config.LLM_DIR / "42",
        job_id=15
    ))
    # → Downloads to temp, moves to final, updates job #15
"""

import os
import re
import time
import threading
import shutil
import asyncio
from contextlib import contextmanager
from types import SimpleNamespace
from pathlib import Path, PurePosixPath, PureWindowsPath
from threading import Lock
from typing import NamedTuple, Optional, List, Tuple, Union

from huggingface_hub import HfApi, HfFileSystem
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError
from fsspec.callbacks import Callback

from src.domains.llms.repository import remove_tree_reporting, update_db_with_progress

from src.core.config import HF_TOKEN
from src.core import config
from src.core.logging import logger
from src.core.exceptions import (
    DownloadJobNotFoundException,
    ModelNotFoundException,
    InvalidInputException,
    StateConflictException,
    UnsupportedPlatformException,
    HuggingFaceAPIException,
)

# Environment setup
FILES_TO_EXCLUDE = ["consolidated.safetensors"]

# Friendly, ASCII-only wording surfaced when a download fails because the
# machine has no network -- mirrors the e5 embedding-download offline path
# (ingestion/embedding_model.OFFLINE_ERROR_MESSAGE) so a connectivity blip
# reads as "offline" instead of the raw hf_hub "Cannot reach https://..."
# string (#109). Kept ASCII (no em dash) for log-safety on cp1252 consoles.
OFFLINE_DOWNLOAD_MESSAGE = "you appear to be offline - check your connection and retry"

# Lowercased substrings marking a no-network failure anywhere in the exception
# chain. hf_hub/transformers re-wrap connection errors, so matching by message
# (defensively, alongside the requests ConnectionError type) is deliberate.
_OFFLINE_MESSAGE_MARKERS = (
    "cannot reach https://huggingface.co",
    "couldn't connect to 'https://huggingface.co'",
    "we couldn't connect to 'https://huggingface.co'",
    "offline mode is enabled",
    "outgoing traffic has been disabled",
    "getaddrinfo",
    "name or service not known",
    "temporary failure in name resolution",
    "network is unreachable",
    "max retries exceeded",
    "connection error",
)


def _is_offline_download_error(exc: BaseException) -> bool:
    """True if the exception chain looks like a no-network download failure (#109)."""
    from huggingface_hub.errors import OfflineModeIsEnabled
    from requests.exceptions import ConnectionError as RequestsConnectionError

    seen: set = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (OfflineModeIsEnabled, RequestsConnectionError)):
            return True
        message = str(current).lower()
        if any(marker in message for marker in _OFFLINE_MESSAGE_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


# Registry of active download trackers keyed by job_id for cancellation support
_active_trackers: dict = {}
_trackers_lock = threading.Lock()
# One event per download task still running (``download_task_running``), set
# when the task ends: what the shutdown waits on before Postgres stops.
_running_downloads: set = set()
# Set once the backend began stopping: a download registered after that is
# interrupted at once instead of starting a transfer nobody will wait for.
_downloads_closed = False

# How long the cancel endpoint waits for the transfer to let go of its files
# before deleting them. A transfer sees the cancel at its next chunk (4 MiB
# through HfFileSystem: about 2.6 s at 1.6 MB/s); a transfer slower than this
# deletes its own files once it stops (``DownloadTracker.hand_over_cleanup``).
CANCEL_STOP_WAIT_SECONDS = 10.0
# How long a stopped download waits for its progress writer, which looks at
# the tracker once a second, so it never commits after the job's end.
PROGRESS_WRITER_JOIN_SECONDS = 2.0


def _register_tracker(job_id: int, tracker: "DownloadTracker") -> None:
    with _trackers_lock:
        _active_trackers[job_id] = tracker
        if _downloads_closed:
            tracker.interrupt()


def _unregister_tracker(job_id: int) -> None:
    with _trackers_lock:
        _active_trackers.pop(job_id, None)


def get_active_download_tracker(job_id: int) -> Optional["DownloadTracker"]:
    with _trackers_lock:
        return _active_trackers.get(job_id)


def track_download(job_id: int) -> "DownloadTracker":
    """Register a tracker for ``job_id`` and return it, for ``download_llm``.

    The download task does this before it claims its job row (``pending`` to
    ``running``). A cancel that found no tracker writes ``cancelled`` over a
    ``pending`` row only: then either the claim fails and the task stops, or
    the claim won and the cancel's write matches nothing, and the cancel
    looks again, finds this tracker and signals it (``cancel_download_job``).
    """
    tracker = DownloadTracker()
    _register_tracker(job_id, tracker)
    return tracker


@contextmanager
def download_task_running(job_id: Optional[int] = None):
    """Mark a download task as running for as long as the block runs.

    ``_run_download_task`` wraps its whole body in it, finalization included,
    so ``stop_active_downloads`` waits for the task's last database write and
    not only for its transfer. On the way out it unregisters the job's
    tracker, which a finalizing download leaves registered.
    """
    done = threading.Event()
    with _trackers_lock:
        _running_downloads.add(done)
    try:
        yield
    finally:
        if job_id is not None:
            _unregister_tracker(job_id)
        done.set()
        with _trackers_lock:
            _running_downloads.discard(done)


def interrupt_active_downloads() -> int:
    """Stop every download because the backend is stopping; how many were running.

    Signal only, like a cancel (#372): each transfer sees it at its next chunk
    and its task raises ``DownloadInterrupted``. Downloads registered later are
    interrupted on registration. Safe to call from any thread and more than
    once: ``backend/run.py`` calls it the moment a shutdown is requested, the
    lifespan again before it stops the database.
    """
    global _downloads_closed
    with _trackers_lock:
        _downloads_closed = True
        trackers = list(_active_trackers.items())
    for _job_id, tracker in trackers:
        tracker.interrupt()
    if trackers:
        jobs = ", ".join(str(job_id) for job_id, _ in trackers)
        logger.info(f"Interrupting download job(s) {jobs}: the backend is stopping")
    return len(trackers)


async def stop_active_downloads(timeout: float) -> None:
    """Interrupt every download and wait, at most ``timeout`` seconds, for their tasks.

    Called by the lifespan shutdown before anything a download task writes
    to (the database above all) is stopped. A task still running after the
    bound is a transfer parked in a network read; the shutdown goes on
    without it and ``backend/run.py`` bounds the process exit.
    """
    interrupt_active_downloads()
    deadline = time.monotonic() + timeout
    while True:
        with _trackers_lock:
            running = [done for done in _running_downloads if not done.is_set()]
        if not running:
            return
        if time.monotonic() >= deadline:
            logger.warning(
                f"{len(running)} download task(s) still running {timeout:g}s after the "
                f"interrupt; stopping without them (the next start resolves their jobs)"
            )
            return
        await asyncio.sleep(0.05)


GGUF_QUANT_PRIORITY = ["q4_k_m", "q4_0", "q5_k_m", "q8_0", "f16"]

# GGUF weights cut into raw byte pieces instead of llama.cpp splits:
#   <name>-chunk-001-of-018.gguf   <name>.gguf.part1of2
#   <name>.gguf.a / .b             <name>.gguf-split-a / -b
# Each piece is a byte range of ONE file: only the first carries a GGUF header,
# so llama.cpp cannot load any of them until they are concatenated, and Erudi
# does not concatenate downloads. llama.cpp's own splits
# (<name>-00001-of-00003.gguf) are different: every part is a complete GGUF file
# the loader stitches itself. The one ambiguous shape is a llama.cpp split whose
# name before the part numbers ends in "chunk" (<name>-chunk-00001-of-00003.gguf):
# it is read as raw chunks on purpose. Refusing that rare name costs a clear
# message; accepting it would let five-digit raw chunks through as a split, and
# a download of every piece would only fail once the loader opens them.
_BYTE_SPLIT_GGUF_RE = re.compile(
    r"(?:-chunk-\d+-of-\d+\.gguf|\.gguf\.part\d+of\d+|\.gguf\.[a-z]|\.gguf-split-[a-z])$",
    re.IGNORECASE,
)


def is_byte_split_gguf(filename: str) -> bool:
    """True when ``filename`` is one raw byte piece of a GGUF file (see _BYTE_SPLIT_GGUF_RE)."""
    return _BYTE_SPLIT_GGUF_RE.search(filename) is not None


def pick_best_gguf(filenames: list[str]) -> str | None:
    """From a list of filenames in a GGUF repo, return the best quantization.

    Uses the same priority order as CUDA_Engine._select_gguf so behavior is
    consistent between download-time selection and load-time selection.

    Args:
        filenames: All filenames returned by HfApi.list_repo_files().

    Returns:
        Filename of the chosen GGUF, or None if no loadable .gguf file is listed.
    """
    # Exclude multimodal projection files (mmproj-*.gguf) — these are vision
    # encoder weights, not the main text model, and must not be loaded as LLM —
    # and raw byte pieces of a GGUF file, which nothing loads without joining them.
    ggufs = [
        f
        for f in filenames
        if f.lower().endswith(".gguf")
        and not f.lower().startswith("mmproj")
        and not is_byte_split_gguf(f)
    ]
    if not ggufs:
        return None
    for quant in GGUF_QUANT_PRIORITY:
        for name in ggufs:
            if quant in name.lower():
                logger.info(f"Selected GGUF: {name} (quant={quant})")
                return name
    # Fall back to the first one alphabetically
    chosen = sorted(ggufs)[0]
    logger.warning(f"No preferred quant found; falling back to {chosen}")
    return chosen


# llama.cpp's standard split-GGUF naming convention: <name>-NNNNN-of-MMMMM.gguf.
# The loader auto-stitches sibling parts together when pointed at part 1, but
# only if every part is actually present on disk next to it.
_GGUF_SPLIT_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<part>\d{5})-of-(?P<total>\d{5})\.gguf$", re.IGNORECASE
)


def gguf_split_siblings(chosen: str, all_repo_files: List[str]) -> List[str]:
    """All part filenames of a split GGUF `chosen` belongs to (chosen included).

    A quant like Qwen2.5-7B's Q4_K_M ships as two files,
    ``…-00001-of-00002.gguf`` and ``…-00002-of-00002.gguf``. Downloading only
    the file `pick_best_gguf` happened to name left the model on disk with
    part 1 present and part 2 missing: llama-server correctly detects the
    split (it looks for the sibling next to part 1) and crashes on load with
    "failed to open GGUF file …-00002-of-00002.gguf" -- a real QA reproduction,
    not a hypothetical.

    Returns `[chosen]` unchanged when `chosen` isn't a split filename, or if
    no siblings with a matching prefix/total are found in the repo listing
    (nothing to add — download proceeds with just `chosen`, same as before).
    Order is by part number so the download list reads naturally in logs.
    """
    match = _GGUF_SPLIT_RE.match(chosen)
    if not match:
        return [chosen]
    prefix, total = match.group("prefix"), match.group("total")
    siblings = []
    for name in all_repo_files:
        sibling_match = _GGUF_SPLIT_RE.match(name)
        if (
            sibling_match
            and sibling_match.group("prefix") == prefix
            and sibling_match.group("total") == total
        ):
            siblings.append((int(sibling_match.group("part")), name))
    if not siblings:
        return [chosen]
    return [name for _part, name in sorted(siblings)]


def _assert_split_is_complete(chosen: str, parts: List[str]) -> None:
    """Refuse a split quant the repo listing cannot supply in full.

    `gguf_split_siblings` never invents a filename, so a listing with a hole
    comes back with fewer parts than the `-of-MMMMM` suffix promises. Downloading
    that set would report success and leave a model llama-server cannot load --
    the same failure #360 removed, reached from a rarer direction. Fail here,
    where it costs seconds, rather than after a multi-gigabyte transfer.

    No-op for a quant that does not ship split.
    """
    match = _GGUF_SPLIT_RE.match(chosen)
    if not match:
        return
    expected = int(match.group("total"))
    found = set()
    for name in parts:
        part_match = _GGUF_SPLIT_RE.match(name)
        if part_match:
            found.add(int(part_match.group("part")))
    if len(found) == expected:
        return
    missing = sorted(set(range(1, expected + 1)) - found)
    raise HuggingFaceAPIException(
        message=(
            f"{chosen} is part of a {expected}-part split quant, but the "
            f"repository listing only exposes {len(found)} of them (missing "
            f"part(s): {', '.join(str(p) for p in missing)}). llama.cpp needs "
            f"every part to load the model, so this download would produce an "
            f"unusable model."
        ),
    )


class _DownloadSelection(NamedTuple):
    """Files chosen for download, with the GGUF picks kept for logging."""

    files: List[str]
    best_gguf: Optional[str]
    mmproj_files: List[str]
    small_aux: List[str]


def _select_download_files(
    all_repo_files: List[str],
    file_sizes: dict,
    uses_gguf: bool,
) -> _DownloadSelection:
    """Pick the exact files to download from a repo listing (pure, no I/O).

    GGUF repos: the single best quantization (pick_best_gguf) + mmproj gguf
    files + auxiliary non-gguf files with a known size under 10 MB. Raw byte
    pieces of a GGUF file (is_byte_split_gguf) are never selected, in any of the
    three roles. When the repo has no loadable .gguf, best_gguf is None and files
    is empty (the download path never gets here in that case:
    `_assert_repo_has_engine_artifact` refuses the repo first;
    `_chosen_artifact_bytes` falls back to the whole-repo sum).

    Non-GGUF repos: every repo file with a known size (exclusions were already
    applied when building file_sizes).

    Args:
        all_repo_files: All filenames returned by HfApi.list_repo_files().
        file_sizes: Filename → size in bytes, exclusions already removed.
        uses_gguf: Whether the active engine consumes GGUF repos.

    Returns:
        _DownloadSelection with the files to download and the GGUF picks.
    """
    if not uses_gguf:
        return _DownloadSelection(
            files=[f for f in all_repo_files if f in file_sizes],
            best_gguf=None,
            mmproj_files=[],
            small_aux=[],
        )
    best_gguf = pick_best_gguf(all_repo_files)
    if not best_gguf:
        return _DownloadSelection(files=[], best_gguf=None, mmproj_files=[], small_aux=[])
    gguf_parts = gguf_split_siblings(best_gguf, all_repo_files)
    _assert_split_is_complete(best_gguf, gguf_parts)
    if len(gguf_parts) > 1:
        logger.info(
            f"Split GGUF detected for {best_gguf!r}: downloading all {len(gguf_parts)} parts"
        )
    mmproj_files = [
        f
        for f in all_repo_files
        if "mmproj" in f.lower() and f.lower().endswith(".gguf") and not is_byte_split_gguf(f)
    ]
    small_aux = [
        f
        for f in all_repo_files
        if not f.lower().endswith(".gguf")
        and not is_byte_split_gguf(f)
        and f in file_sizes
        and file_sizes.get(f, 0) < 10 * 1024 * 1024  # < 10 MB
    ]
    return _DownloadSelection(
        files=gguf_parts + mmproj_files + small_aux,
        best_gguf=best_gguf,
        mmproj_files=mmproj_files,
        small_aux=small_aux,
    )


def has_downloadable_gguf(repo_id: str, hf_api) -> bool:
    """Whether the downloader would select a loadable GGUF artefact from ``repo_id``.

    Runs the download path's own selection (`_select_download_files`) over the
    repo's file listing, so a catalog build never offers a repo the downloader
    refuses: one whose only GGUF weights are raw byte pieces, or a llama.cpp split
    whose listing misses a part.

    The listing is ``model_info(repo_id, expand=["siblings"])``: one request, file
    names only, and the call the catalog's retrying client paces and retries on
    HTTP 429. A listing failure propagates; the caller decides what it means.
    """
    info = hf_api.model_info(repo_id, expand=["siblings"])
    all_repo_files = [s.rfilename for s in (getattr(info, "siblings", None) or [])]
    try:
        selection = _select_download_files(all_repo_files, {}, uses_gguf=True)
    except HuggingFaceAPIException:
        # _assert_split_is_complete: the listing misses a part of the chosen
        # split, which the downloader refuses before any transfer.
        return False
    return selection.best_gguf is not None


class DownloadCancelled(Exception):
    """Raised from inside a transfer once its DownloadTracker has been cancelled.

    Cancellation is signal-and-check (#372): `cancel_download_job` flips the
    tracker flag and returns. This exception is how the download side reacts to
    that flag from where it is looked at -- before each shard starts and on every
    progress chunk -- so a cancelled job stops transferring instead of running
    every remaining shard to the end (#377). `download_llm` raises it to the
    download task, which takes it as the user's cancel: nothing to finalize,
    the cancel endpoint owns the job row and the cleanup.
    """


class DownloadInterrupted(Exception):
    """Raised by ``download_llm`` when the backend stopped the download.

    Not the user's cancel: the job row stays ``running`` and the partial files
    stay on disk, which is what the next start's cleanup resolves (it
    finalizes a complete artifact, otherwise removes it and records the job
    as interrupted by the shutdown). Nothing is written to the database on
    the way out, since it is being stopped.
    """


class DownloadTracker:
    """Thread-safe progress tracker for multi-file downloads with ETA estimation.

    Aggregates downloaded bytes across concurrent file transfers and periodically
    computes ETA based on moving average of download rate. Used by fsspec callbacks
    and background DB updater thread.

    Attributes:
        total_bytes: Total download size in bytes (set before download starts).
        downloaded_bytes: Bytes downloaded so far (updated by callbacks).
        eta_seconds: Estimated seconds remaining (None until first ETA computation).

    Example:
        >>> tracker = DownloadTracker()
        >>> tracker.total_bytes = 1_000_000_000
        >>> tracker.update(50_000_000)
        >>> print(tracker.percent)
        5.0
    """

    def __init__(self) -> None:
        self.total_bytes: int = 0
        self.downloaded_bytes: int = 0
        self.eta_seconds: Optional[float] = None
        self._lock = Lock()
        # Why the transfers must stop, first reason wins: "cancelled" (the
        # user), "interrupted" (the backend is stopping) or "failed" (a
        # sibling shard failed). None while the download runs.
        self._stop_reason: Optional[str] = None
        # Set once the files are complete and being promoted: from then on the
        # download ends as installed and no stop is taken any more.
        self._finalizing: bool = False
        # Set by the download side once none of its transfers runs any more.
        self._stopped = threading.Event()
        self._cleanup_handed_over: bool = False
        self.progress_writer: Optional[threading.Thread] = None
        logger.info("DownloadJob initialized")

    def update(self, bytes_downloaded: int) -> None:
        """Atomically increment downloaded byte count (thread-safe).

        Args:
            bytes_downloaded: Number of bytes downloaded in this chunk transfer.
        """
        with self._lock:
            self.downloaded_bytes += bytes_downloaded

    @property
    def percent(self) -> float:
        """Compute download completion percentage with zero-division guard.

        Returns:
            Progress percentage in range [0.0, 100.0]. Returns 0.0 if total_bytes is 0.
        """
        if self.total_bytes == 0:
            return 0.0
        return (self.downloaded_bytes / self.total_bytes) * 100

    def _stop(self, reason: str) -> bool:
        with self._lock:
            if self._finalizing:
                return False
            if self._stop_reason is None:
                self._stop_reason = reason
            return True

    def cancel(self) -> bool:
        """Signal the user's cancellation (thread-safe).

        False when the download is already finalizing (``begin_finalizing``):
        the cancel is refused and the download ends as installed.
        """
        return self._stop("cancelled")

    def interrupt(self) -> None:
        """Signal that the backend is stopping (thread-safe)."""
        self._stop("interrupted")

    def fail(self) -> None:
        """Stop the other transfers of a job one of whose files failed."""
        self._stop("failed")

    @property
    def finalizing(self) -> bool:
        """True once the download is promoting its files (no stop taken)."""
        with self._lock:
            return self._finalizing

    @property
    def cancelled(self) -> bool:
        """True when the user cancelled."""
        with self._lock:
            return self._stop_reason == "cancelled"

    @property
    def interrupted(self) -> bool:
        """True when the backend's shutdown stopped the download."""
        with self._lock:
            return self._stop_reason == "interrupted"

    def should_continue(self) -> bool:
        """Return False once a stop was signalled, True otherwise (thread-safe)."""
        with self._lock:
            return self._stop_reason is None

    def begin_finalizing(self) -> bool:
        """Every file is transferred: from now on, refuse every stop.

        False when a stop was signalled first (the caller then ends the
        download as stopped). True: the files are promoted and the job
        finalized, and a cancel arriving meanwhile is refused (409) rather
        than deleting a model the download task is about to mark installed.
        """
        with self._lock:
            if self._stop_reason is not None:
                return False
            self._finalizing = True
            return True

    def mark_stopped(self) -> bool:
        """The download side: none of this job's transfers runs any more.

        Returns True when the cancel endpoint handed the file cleanup over to
        the download (its wait ran out first), i.e. the caller must now remove
        the job's files itself.
        """
        with self._lock:
            handed_over = self._cleanup_handed_over
            self._stopped.set()
        return handed_over

    def hand_over_cleanup(self, timeout: float) -> bool:
        """The cancel side: wait up to ``timeout`` seconds for the transfers to stop.

        True: they stopped, no file is open any more and the caller removes
        the files. False: a transfer is still running; the download removes
        the files itself once it stops (``mark_stopped`` returns True), so a
        file it still holds open is never deleted under it.
        """
        if self._stopped.wait(timeout):
            return True
        with self._lock:
            if self._stopped.is_set():
                return True
            self._cleanup_handed_over = True
            return False

    async def monitor_eta(self, interval: float = 20.0) -> None:
        """Continuously estimate remaining time based on download rate (runs until 100%).

        Samples downloaded_bytes at regular intervals, computes delta rate, and updates
        eta_seconds. Runs as asyncio task in background until download completes.

        Args:
            interval: Seconds between ETA recalculations (default 20s).

        Example:
            >>> tracker = DownloadTracker()
            >>> eta_task = asyncio.create_task(tracker.monitor_eta(interval=5.0))
            >>> # ... download happens in parallel ...
            >>> await eta_task  # Wait for completion
        """
        logger.info("Starting ETA monitoring")
        last_time = time.time()
        last_downloaded = 0
        while True:
            await asyncio.sleep(interval)
            with self._lock:
                current = self.downloaded_bytes
                total = self.total_bytes
            if current >= total:
                break

            now = time.time()
            delta_bytes = current - last_downloaded
            delta_time = now - last_time

            if delta_time > 0 and delta_bytes > 0:
                rate = delta_bytes / delta_time
                self.eta_seconds = (total - current) / rate

            last_time = now
            last_downloaded = current

        logger.info("ETA monitoring complete")


def make_callback(job: DownloadTracker) -> Callback:
    """Create fsspec Callback that updates DownloadTracker on each file transfer chunk.

    The callback's transfer-chunk hook is invoked by fsspec after each network read.
    Computes delta bytes since last update to avoid double-counting (guards against
    negative deltas from fsspec resets).

    Args:
        job: DownloadTracker instance to update with progress.

    Returns:
        Configured fsspec Callback with total size and chunk hook.

    Example:
        >>> tracker = DownloadTracker()
        >>> tracker.total_bytes = 1_000_000
        >>> callback = make_callback(tracker)
        >>> fs.get_file("repo/file.safetensors", "local.safetensors", callback)
    """
    # prev tracks the last cumulative value seen for the current file.
    # `value` is per-file cumulative (resets to 0 at the start of each file),
    # so when it drops below prev we know a new file started and reset.
    state = {"prev": 0}

    def after_chunk(size: int, value: int, **kwargs) -> None:
        if not job.should_continue():
            # Abort the transfer from inside fsspec's read loop (#377): raising here
            # unwinds fs.get_file in its worker thread, so the shard stops now
            # instead of streaming to the end after the user cancelled.
            raise DownloadCancelled("download cancelled during transfer")
        if value is None:
            return
        prev = state["prev"]
        delta = value if value < prev else value - prev
        if delta > 0:
            job.update(delta)
        state["prev"] = value

    return Callback(size=job.total_bytes, hooks={"transfer-chunk": after_chunk})


# ============ Destination containment (#405) ============

_MEMBER_DISPLAY_LIMIT = 120


def _display_member(member: str) -> str:
    """ASCII, bounded rendering of a Hub file name for an error message."""
    shown = member.encode("ascii", "backslashreplace").decode("ascii")
    return shown if len(shown) <= _MEMBER_DISPLAY_LIMIT else shown[:_MEMBER_DISPLAY_LIMIT] + "..."


def resolve_member_path(root: Union[str, Path], member: str) -> Path:
    """Map a Hub-listed file name onto ``root`` or refuse it.

    Every destination the downloader writes is derived from names LISTED BY THE
    HUB (``list_repo_files``), and any repository can be downloaded by link. The
    rule is refuse, never normalise: the loaders (mlx-vlm, llama-server) expect
    the repo's own names and subfolders, so a rewritten name would silently
    produce a broken model and hide a hostile repo.

    Rejected (``InvalidInputException``, ASCII message naming the member):
      - empty name, or one carrying a NUL byte;
      - any backslash (the Hub only ever lists ``/``-separated names);
      - absolute forms on either family: POSIX (``/x``), Windows drive or
        drive-relative (``C:/x``, ``C:x``) and UNC (``//server/share/x``),
        checked with ``PurePosixPath``/``PureWindowsPath`` explicitly so the
        outcome is identical on the three CI legs;
      - any ``..``, ``.`` or empty segment (``a/../x``, ``./x``, ``a//b``, ``a/``);
      - anything that, once resolved, does not land under ``root.resolve()``
        (belt and braces after the syntactic checks).

    Returns ``root / member`` unchanged otherwise. Pure: nothing is created.
    """
    root = Path(root)
    shown = _display_member(member) if isinstance(member, str) else repr(member)

    def refuse(reason: str) -> InvalidInputException:
        return InvalidInputException(
            f"Hub file name '{shown}' ({reason}; refusing to write outside the model directory)"
        )

    if not isinstance(member, str) or not member:
        raise refuse("empty name")
    if "\x00" in member:
        raise refuse("NUL byte")
    if "\\" in member:
        raise refuse("backslash")
    windows_form = PureWindowsPath(member)
    if (
        PurePosixPath(member).is_absolute()
        or windows_form.is_absolute()
        or windows_form.drive
        or windows_form.anchor
    ):
        raise refuse("absolute path")
    if any(segment in ("", ".", "..") for segment in member.split("/")):
        raise refuse("'.', '..' or empty path segment")

    root_resolved = root.resolve()
    if not (root_resolved / member).resolve().is_relative_to(root_resolved):
        raise refuse("resolves outside the model directory")
    return root / member


async def download_files_concurrent(
    fs: HfFileSystem,
    callback: Callback,
    tasks: List[Tuple[str, str]],
    local_dir: str,
    tracker: Optional[DownloadTracker] = None,
) -> None:
    """Download multiple HuggingFace files in parallel using asyncio executor pool.

    Wraps synchronous fs.get_file() calls in run_in_executor to achieve I/O concurrency.
    Useful for downloading .safetensors shards simultaneously to maximize bandwidth.

    Args:
        fs: HfFileSystem instance with auth token.
        callback: fsspec Callback for progress tracking (shared across all files).
        tasks: List of (repo_id, file_path) tuples to download.
        local_dir: Base directory to save files (subdirectories created as needed).
        tracker: When given, each shard checks `tracker.should_continue()` right
            before its transfer starts and raises DownloadCancelled otherwise, so a
            cancelled job never opens the shards still queued behind the in-flight
            ones (#377). The in-flight ones abort through the callback. A shard
            that fails stops its siblings the same way (``tracker.fail()``).

    Returns only once every shard's thread has ended, whatever the outcome: a
    cancel's cleanup deletes these files next, and Windows refuses to delete a
    file a sibling shard still holds open (#690).

    Raises:
        DownloadCancelled: the tracker was cancelled before or during a transfer.
        Exception: the first failure of a shard, once its siblings have stopped.

    Example:
        >>> fs = HfFileSystem(token=HF_TOKEN)
        >>> callback = make_callback(tracker)
        >>> tasks = [("meta-llama/Llama-3-8B", "model-00001-of-00004.safetensors"), ...]
        >>> await download_files_concurrent(fs, callback, tasks, config.LLM_DIR / "temp_42")
    """
    loop = asyncio.get_running_loop()

    def get_file_unless_cancelled(remote: str, dest: str) -> None:
        if tracker is not None and not tracker.should_continue():
            raise DownloadCancelled(f"download cancelled before {remote} started")
        fs.get_file(remote, dest, callback)

    # Contain every destination BEFORE scheduling anything (#405): a hostile
    # member anywhere in the batch refuses the whole batch with no transfer
    # started and no directory created.
    dests = [(f"{repo_id}/{path}", resolve_member_path(local_dir, path)) for repo_id, path in tasks]
    futures = []
    for remote, dest in dests:
        dest.parent.mkdir(parents=True, exist_ok=True)
        futures.append(loop.run_in_executor(None, get_file_unless_cancelled, remote, str(dest)))
    if not futures:
        return
    done, pending = await asyncio.wait(futures, return_when=asyncio.FIRST_EXCEPTION)
    if pending:
        # A shard failed or saw a stop: the others stop at their next chunk,
        # and nothing returns before their threads have let go of the files.
        if tracker is not None:
            tracker.fail()
        await asyncio.wait(pending)
    errors = [f.exception() for f in futures if f.exception() is not None]
    if not errors:
        return
    if tracker is not None and (tracker.cancelled or tracker.interrupted):
        # Whatever else a shard raised on its way out, the job was stopped.
        raise DownloadCancelled("download cancelled during transfer")
    failures = [e for e in errors if not isinstance(e, DownloadCancelled)]
    raise (failures or errors)[0]


def _assert_runnable(model_link: str) -> None:
    """Reject a download for a KNOWN_BROKEN quant up front.

    Every catalog link is already a public engine-format quant (built from a
    filter=FORMAT_TAG search), so the only thing to reject is a quant flagged as
    crash-on-load for this engine — fail fast with a clear message.
    """
    if not config.LLM_Engine.is_runnable(model_link):
        raise UnsupportedPlatformException(
            feature=model_link,
            reason=f"this model is known not to run on {config.LLM_Engine.__name__}",
        )


def _assert_repo_has_engine_artifact(model_link: str, repo_info, all_repo_files: List[str]) -> None:
    """Refuse, before any byte is fetched, a repo that ships nothing this engine runs.

    Erudi never converts weights locally (#408): it only downloads pre-built
    artefacts, so a repo must already carry the active engine's format. Catalog
    and HF-search links do by construction (``filter=FORMAT_TAG``); a repo id the
    user pastes by hand may not -- e.g. a vendor's plain safetensors checkpoint.
    Downloading that would report success and leave a model on disk the engine
    cannot load, so fail here with the reason instead.

    - llama.cpp engines (``USES_GGUF``): at least one text-model ``.gguf`` must be
      listed (``mmproj-*.gguf`` vision projectors and raw byte pieces of a GGUF
      file do not count). A repo whose GGUF weights are only raw byte pieces gets
      its own message: the model exists, but as chunks that must be joined into
      one file before anything can load it, and Erudi does not join them.
    - Tag-based engines (MLX): the repo must be tagged ``FORMAT_TAG`` or declare
      it as its ``library_name``, exactly what the catalog search filters on.

    Raises:
        InvalidInputException: The repo has no ``<FORMAT_TAG>`` artefact for this engine.
    """
    engine = config.LLM_Engine
    tag = getattr(engine, "FORMAT_TAG", None)
    if getattr(engine, "USES_GGUF", False):
        present = pick_best_gguf(all_repo_files) is not None
        byte_pieces = [f for f in all_repo_files if is_byte_split_gguf(f)]
        if not present and byte_pieces:
            raise InvalidInputException(
                f"{model_link} publishes its GGUF weights as raw chunks "
                f"(for example {_display_member(byte_pieces[0])}) that must be joined "
                f"into one file before the model can run, and Erudi does not join them. "
                f"Pick another repository of this model that ships complete .gguf files."
            )
    else:
        tags = set(getattr(repo_info, "tags", None) or ())
        present = bool(tag) and (tag in tags or getattr(repo_info, "library_name", None) == tag)
    if present:
        return
    engine_name = getattr(engine, "__name__", type(engine).__name__)
    raise InvalidInputException(
        f"{model_link} has no {tag} artefact for this machine's engine ({engine_name}): "
        f"Erudi only downloads pre-built {tag} models and does not convert weights locally. "
        f"Pick a repository that ships {tag} files (for example one found through the "
        f"Hugging Face search in the app)."
    )


async def download_llm(
    model_link: str,
    model_id: int,
    temp_save_dir: str,
    final_save_dir: str,
    job_id: Optional[int] = None,
    tracker: Optional["DownloadTracker"] = None,
) -> str:
    """Download a pre-built HuggingFace quant with progress tracking.

    Complete pipeline: list repo files → select files to download → download to
    temp_save_dir → move to final_save_dir. Every catalog link is a pre-built quant,
    so no local conversion happens. Progress updates are persisted to
    DownloadJobModel if job_id provided.

    Args:
        model_link: HuggingFace repo ID (e.g., "meta-llama/Llama-3-8B-Instruct").
        model_id: Database ID of the LLM entry (used by the DB progress updater).
        temp_save_dir: Temp directory for the download (moved to final_save_dir on success).
        final_save_dir: Final directory for the model (ready for inference).
        job_id: Optional DownloadJobModel ID for progress tracking (spawns background thread).

    Returns:
        Path to temp_save_dir (note: moved to final_save_dir on success).

    Raises:
        DownloadCancelled: The job's tracker was cancelled (the user cancelled)
            before the files were moved: nothing was promoted.
        InvalidInputException: The repo ships no artefact in this engine's format
            (checked before any byte is transferred, #408).
        Exception: If HuggingFace API fails or the download fails.

    Example:
        >>> final_path = await download_llm(
        ...     model_link="meta-llama/Llama-3-8B-Instruct",
        ...     model_id=42,
        ...     temp_save_dir=config.LLM_DIR / "temp_42",
        ...     final_save_dir=config.LLM_DIR / "42",
        ...     job_id=15
        ... )
        >>> # Progress tracked in DownloadJobModel(id=15), final model in backend/data/models/42
    """
    # The catalog link IS already a public engine-format quant (resolved at seed
    # time from a filter=FORMAT_TAG search), so we download it directly — no mapping.
    actual_download_link = model_link
    _assert_runnable(model_link)

    # Everything we download is a pre-built quant → no local conversion. GGUF repos
    # additionally need only the single best quant picked (pick_best_gguf), plus
    # every sibling part if that quant ships split (gguf_split_siblings).
    _uses_gguf = getattr(config.LLM_Engine, "USES_GGUF", False)

    # Prepare local path
    os.makedirs(temp_save_dir, exist_ok=True)
    os.makedirs(final_save_dir, exist_ok=True)
    logger.info(f"Starting download for {model_link} -> {temp_save_dir}")
    logger.info(f"Downloading pre-built quant directly: {actual_download_link}")

    # Initialize HF API & filesystem
    api = HfApi(token=HF_TOKEN)
    fs = HfFileSystem(token=HF_TOKEN)

    # Initialize tracking and register for cancellation support
    # The download task registers the tracker itself, before it claims its
    # job row, so a cancel can never land where no tracker is registered yet
    # (``track_download``).
    job = tracker if tracker is not None else DownloadTracker()
    if job_id is not None and tracker is None:
        _register_tracker(job_id, job)

    try:
        # Gather file sizes, then select the exact files to download BEFORE
        # computing the total, so the logged size and progress reflect what is
        # actually fetched (a single GGUF quant, not the whole repo).
        info = api.repo_info(actual_download_link, files_metadata=True)
        file_sizes = {
            s.rfilename: s.size
            for s in info.siblings
            if s.size and s.rfilename not in FILES_TO_EXCLUDE
        }
        all_repo_files = list(api.list_repo_files(actual_download_link))
        # Refuse a repo with nothing this engine runs BEFORE any transfer (#408).
        _assert_repo_has_engine_artifact(actual_download_link, info, all_repo_files)
        selection = _select_download_files(all_repo_files, file_sizes, _uses_gguf)
        all_files = selection.files
        # Contain the WHOLE selection before a single byte is fetched (#405):
        # the names come from the Hub listing, so a hostile repo fails here
        # with a clear message and the job-failure path reports it.
        for path in all_files:
            resolve_member_path(temp_save_dir, path)
        if _uses_gguf:
            if selection.mmproj_files:
                logger.info(
                    f"GGUF download: {selection.best_gguf} + mmproj ({selection.mmproj_files[0]}) "
                    f"+ {len(selection.small_aux)} aux files"
                )
            else:
                logger.info(
                    f"GGUF download: {selection.best_gguf} + {len(selection.small_aux)} aux files"
                )

        job.total_bytes = sum(file_sizes.get(f, 0) for f in all_files)
        logger.info(f"Total size: {job.total_bytes} bytes")

        # Create progress callback
        callback = make_callback(job)
        callback.set_size(job.total_bytes)

        # Start DB updater if requested
        if job_id is not None:
            job.progress_writer = threading.Thread(
                target=update_db_with_progress, args=(job, job_id, model_id), daemon=True
            )
            job.progress_writer.start()

        # Start ETA monitoring
        eta_task = asyncio.create_task(job.monitor_eta(interval=5.0))

        misc = [f for f in all_files if not f.endswith(".safetensors")]
        shards = [f for f in all_files if f.endswith(".safetensors")]

        try:
            # Download misc sequentially
            for path in misc:
                if not job.should_continue():
                    raise DownloadCancelled(f"download cancelled before {path} started")
                dest = resolve_member_path(temp_save_dir, path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(
                    fs.get_file, f"{actual_download_link}/{path}", str(dest), callback
                )
                logger.info(f"Downloaded {path}")

            # Download shards concurrently
            shard_tasks = [(actual_download_link, path) for path in shards]
            await download_files_concurrent(fs, callback, shard_tasks, temp_save_dir, tracker=job)
            logger.info("All shards downloaded")
        except DownloadCancelled:
            # Raised from the pre-shard check or from inside a transfer (#377);
            # the download task says it, once.
            eta_task.cancel()
            raise

        # From here on the download ends as installed: a cancel is refused
        # (409) instead of deleting files the task is about to finalize. A
        # stop that landed between the last chunk and here wins: same
        # outcome as a stop during the transfer, no finalization.
        if not job.begin_finalizing():
            eta_task.cancel()
            raise DownloadCancelled("download cancelled after the transfer")
        # The ETA means nothing any more; its monitor sleeps 5 s per tick and
        # must not hold the finish for it.
        eta_task.cancel()

        # Everything is a pre-built quant: move files straight to the final dir
        logger.info("Moving downloaded files to final location")
        if not os.path.exists(temp_save_dir):
            logger.warning(f"temp dir {temp_save_dir} missing (cancelled?), skipping move")
            return temp_save_dir
        if os.path.exists(final_save_dir):
            shutil.rmtree(final_save_dir, ignore_errors=True)
        shutil.move(temp_save_dir, final_save_dir)

        try:
            await eta_task
        except asyncio.CancelledError:
            pass  # cancelled just above, on purpose
        logger.info("Download complete")

        return temp_save_dir

    except (GatedRepoError, HfHubHTTPError) as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if isinstance(e, GatedRepoError) or status in (401, 403):
            raise UnsupportedPlatformException(
                feature=model_link,
                reason="requires HuggingFace authentication and cannot be downloaded anonymously",
            )
        raise
    except DownloadCancelled:
        if job.interrupted:
            # The backend is stopping, not the user cancelling.
            raise DownloadInterrupted(f"download of {model_link} interrupted by the shutdown")
        # The user's cancel, not an error to rename.
        raise
    except Exception as e:
        # Re-raised in both branches: the download task that owns the job
        # writes the record, with the traceback. Named here so the offline
        # case still says which model it was about.
        if _is_offline_download_error(e):
            raise HuggingFaceAPIException(
                OFFLINE_DOWNLOAD_MESSAGE, trace=f"{actual_download_link}: {e}"
            )
        raise
    finally:
        # A finalizing download stays registered until its task has finished
        # finalizing (``download_task_running`` unregisters it), so a cancel
        # meanwhile is refused instead of finding no transfer and deleting
        # the model being installed.
        if job_id is not None and not job.finalizing:
            _unregister_tracker(job_id)
        # Every transfer has ended here (each one was awaited). A stopped job's
        # progress writer goes too, before anyone deletes the files or stops
        # the database under it.
        writer = job.progress_writer
        if writer is not None and not job.should_continue():
            writer.join(PROGRESS_WRITER_JOIN_SECONDS)
        if job.mark_stopped():
            _remove_cancelled_files(job_id, temp_save_dir, final_save_dir)


def _remove_cancelled_files(job_id: Optional[int], *paths) -> None:
    """Delete a cancelled job's files, once its transfers have stopped.

    Only reached when the cancel endpoint handed the cleanup over because the
    transfer outlived its wait (``DownloadTracker.hand_over_cleanup``). No file
    is open any more, so what still cannot be removed is reported.
    """
    all_removed = True
    for path in paths:
        if not path or not os.path.exists(str(path)):
            continue
        left = remove_tree_reporting(path)
        if left:
            all_removed = False
            logger.warning(
                f"Could not remove {path} of cancelled download job {job_id} "
                f"({left} bytes still on disk; startup cleanup will reclaim it)"
            )
    if all_removed:
        logger.info(
            f"Removed the files of cancelled download job {job_id} once its transfer stopped"
        )


def cancel_download_job(job_id: int, job_repo, llm_repo, db) -> dict:
    """Cancel an active download job, signal the tracker, and clean up partial files.

    The cancel is committed at once (job ``cancelled``, placeholder model
    deleted), so a status poll meanwhile finds nothing left to clean and the
    download task's own write finds nothing to do. The files are deleted once
    the transfer has stopped (at most ``CANCEL_STOP_WAIT_SECONDS`` of
    waiting); a transfer still running after that deletes them itself when it
    stops. A download already finalizing refuses the cancel (409): it ends
    installed. A cancel is the user's choice, not a failure: nothing here is
    logged above ``INFO``.

    Args:
        job_id: ID of the download job to cancel.
        job_repo: Download_Job_Repository instance.
        llm_repo: Llm_Repository instance.
        db: SQLAlchemy session for transaction control.

    Returns:
        dict with cancellation status.

    Raises:
        DownloadJobNotFoundException: no job with this id.
        ModelNotFoundException: the job's model row is gone.
        InvalidInputException: the job is not in a cancellable state.
        StateConflictException: the job already ended (400), or the download
            is finalizing and ends installed (409).
    """
    job = job_repo.get_by_id(job_id)
    if not job:
        raise DownloadJobNotFoundException(job_id)
    if job.status in ["completed", "failed", "cancelled"]:
        raise StateConflictException(
            "Cannot cancel a job that is already completed, failed, or cancelled"
        )

    llm = llm_repo.get_by_id(job.local_model_id)
    if not llm:
        raise ModelNotFoundException(f"LLM {job.local_model_id}")
    if llm.local != 2:
        raise InvalidInputException("Download ended - cannot be cancelled, please delete the model")

    tracker = get_active_download_tracker(job_id)
    if tracker is not None and not tracker.cancel():
        raise StateConflictException(
            f"Download job {job_id} is completing and can no longer be cancelled",
            status_code=409,
            detail={"reason": "completing", "job_id": job_id},
        )
    if tracker is not None:
        logger.info(f"Signaled cancellation for download job {job_id}")
    else:
        # Not started yet, or the transfer already ended: nothing to wait for.
        logger.info(f"No transfer running for download job {job_id}; cleaning up")
    # The paths, read now: the rows change below.
    files = SimpleNamespace(
        id=job.id,
        local_model_id=job.local_model_id,
        temp_local_model_link=job.temp_local_model_link,
        final_local_model_link=job.final_local_model_link,
    )

    # Conditional: a job that ended since the check above (completed,
    # failed) is never turned into a cancel, and nothing of it is touched.
    # With no tracker found, only a still-pending row is cancelled: a task
    # that claimed the row meanwhile has registered its tracker first, which
    # a second look finds and signals.
    rows = job_repo.mark_cancelled_if_active(
        job_id, ("pending", "running") if tracker is not None else ("pending",)
    )
    if not rows and tracker is None:
        tracker = get_active_download_tracker(job_id)
        if tracker is not None:
            if not tracker.cancel():
                db.rollback()
                raise StateConflictException(
                    f"Download job {job_id} is completing and can no longer be cancelled",
                    status_code=409,
                    detail={"reason": "completing", "job_id": job_id},
                )
            logger.info(f"Signaled cancellation for download job {job_id} (it started meanwhile)")
            rows = job_repo.mark_cancelled_if_active(job_id)
    if not rows:
        db.rollback()
        raise StateConflictException(
            "Cannot cancel a job that is already completed, failed, or cancelled"
        )
    try:
        # Commit first: the job reads cancelled and the placeholder model is
        # gone before anything waits.
        llm_repo.delete(llm)
        db.commit()
    finally:
        # Then the files, committed or not, once the transfer has released
        # them: Windows refuses to delete a file still open, and the
        # transfer only sees the flag at its next chunk (#690).
        if tracker is not None and not tracker.hand_over_cleanup(CANCEL_STOP_WAIT_SECONDS):
            logger.info(
                f"Download job {job_id} is still stopping after {CANCEL_STOP_WAIT_SECONDS:g}s; "
                f"its files are removed once the transfer has stopped"
            )
        else:
            job_repo.cleanup_job_files(files)

    logger.info(f"Cancelled download job {job_id} and deleted temp LLM {llm.id}")
    return {"message": "Download cancellation initiated", "job_id": job_id, "status": "cancelled"}
