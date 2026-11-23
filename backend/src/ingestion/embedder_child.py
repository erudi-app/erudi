"""Dedicated Knowledge Base embedding child — shared lifecycle manager (#594).

The KB embedder runs in ITS OWN child process, deliberately outside the chat
engines: it never enters `BaseEngine`/`BaseChatServerEngine` class state, the
CUDA gauges, the memory brake or the Settings idle delay. What it shares with
the chat child is the SHAPE of the lifecycle, re-implemented here as plain
module functions so the Windows/Linux `llama-server --embedding` child (F3 of
#594) plugs in a second adapter module without touching the chat path:

    port pick (27400-27499, above every chat pool) → per-spawn ``--api-key``
    (``secrets.token_hex(32)``, never logged) → two-stage readiness probe →
    FIFO-serialized ``POST /v1/embeddings`` batches → fixed 5-minute idle
    reap → bounded terminate, with the crash report quoting the child's log.

Two-stage probe, and why ``/health`` is not enough: `mlx_vlm.server` records
a failed ``--embedding-model`` preload in ``runtime.preload_failures`` and
still answers ``/health`` 200 — only a real 1-input ``POST /v1/embeddings``
proves the model loaded, pooled and answered. Every request carries
``"model": <local snapshot path>`` because mlx_vlm POPS the preload env var
at startup: a request without the field answers 400 on a healthy server.

Locking (three locks, three jobs):

- ``_request_lock`` (:class:`_FifoLock`) — one embeddings request in flight,
  held PER BATCH and FIFO: a search query asked while an ingestion streams a
  file's batches waits for at most ONE batch, never the file. FIFO matters:
  a plain ``threading.Lock`` lets the releasing thread barge back in, which
  starves the query for the whole file.
- ``_spawn_lock`` — one spawn at a time. The readiness probe runs OUTSIDE
  ``_state_lock`` (under this lock only), so a shutdown never waits out a
  probe: it flags ``_closed``, terminates the pending process, and the
  prober refuses to publish.
- ``_state_lock`` — the module state (handle, leases, idle clock, closed).

Lifecycle policy (product decisions on #594):

- **Ingestion lease.** `ingestion_lease` brackets a whole ingestion job: the
  idle reaper never terminates the child mid-ingestion, however long the job
  runs; releasing the lease restarts the idle clock.
- **Fixed idle.** `EMBEDDER_IDLE_SECONDS` (5 minutes, no setting — decided on
  #594; independent of the chat model's ``idle_unload_minutes``). A daemon
  reaper thread samples every `REAP_CHECK_INTERVAL_S`.
- **Memory pressure.** `release_now` lets the MLX memory brake give the
  child's memory back under CRITICAL pressure before it acts on the chat
  model — only when no lease is held and no request is in flight; the next
  KB use respawns in a few seconds.
- **Pre-start.** `prewarm_embedder` is called when a conversation with a KB
  opens (``conversations`` endpoints): a background thread pays the spawn
  ahead of the first KB use. It never raises, skips a live child or a spawn
  in flight without touching the idle clock, backs off
  `PREWARM_BACKOFF_S` after a failure, and an absent artifact is one INFO
  per state (the gate/banner owns the state), anything else one WARNING.
- **Start beside the job.** An ingestion job calls `start_in_background`
  once its first file has been read and holds text to index: the spawn runs
  while the job chunks that file (and loads the e5 tokenizer), instead of
  after. A caller that waited on ``_spawn_lock`` while a spawn failed gets
  that spawn's failure (a fresh exception of the same kind, chained to it
  and flagged ``joined_spawn``) instead of spawning again; the background
  starts (pre-start, the job's start) never log a failure they only
  joined. A spawn that failed BEFORE a caller arrived is not shared: that
  caller spawns again (the job then logs the first failure as a WARNING
  when its own start succeeds).
- **Shutdown is terminal.** `shutdown_embedder` sets ``_closed``; any later
  ``ensure_embedder_running`` refuses — a background thread that outlives
  the lifespan must never respawn a child into a stopping backend.

Platform dispatch: `_get_adapter` serves `src.ingestion.mlx_embedder` on
macOS ARM and `src.ingestion.llama_embedder` (the bundled CPU ``llama-server
--embedding``, F3) everywhere else — every platform serves KB embeddings
from a dedicated child. ``ERUDI_FORCE_CPU`` is deliberately ignored: it
picks the CHAT engine, and the embedder is not a chat engine (the llama
adapter always runs the CPU binary anyway).
"""

from __future__ import annotations

import atexit
import collections
import contextlib
import platform
import secrets
import socket
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests

from src.core.exceptions import (
    AppBaseException,
    EmbeddingModelMissingException,
    EngineException,
)
from src.core.logging import logger
from src.ingestion import embedding_model

# The embedder's own port slice: above the llama.cpp pools (27200-27299), the
# MLX chat pool (27300-27399) and the backend's 27182-27199 scan.
EMBEDDER_PORT_START = 27400
EMBEDDER_PORT_COUNT = 100

#: Released after this long unused — fixed, no setting (#594 decision).
EMBEDDER_IDLE_SECONDS = 300.0

#: How often the reaper thread samples the idle clock.
REAP_CHECK_INTERVAL_S = 15.0

#: After a failed pre-start, no new pre-start before this long: a banner or a
#: conversation list re-render must not retry-spam a broken spawn.
PREWARM_BACKOFF_S = 60.0

PROBE_TIMEOUT_S = 120.0
PROBE_POLL_INTERVAL_S = 0.3

#: Texts per ``/v1/embeddings`` request. The server pads a request's texts
#: into ONE batch; 32 bounds that batch (sentence-transformers' default
#: ``encode`` batch size, the reference e5 is measured against).
EMBED_BATCH_SIZE = 32
EMBED_REQUEST_TIMEOUT_S = 120.0


class _FifoLock:
    """Mutual exclusion granted in ARRIVAL order.

    ``threading.Lock`` wakes a waiter on release but lets the releasing
    thread re-acquire first (barging). Between two batches of a long
    ingestion that would starve a waiting search query for the whole file;
    FIFO hands the gap to the query. ``notify_all`` is fine: two or three
    threads at most ever wait here.
    """

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._waiters: collections.deque = collections.deque()
        self._locked = False

    def acquire(self) -> None:
        token = object()
        with self._cv:
            self._waiters.append(token)
            while self._locked or self._waiters[0] is not token:
                self._cv.wait()
            self._waiters.popleft()
            self._locked = True

    def try_acquire(self) -> bool:
        with self._cv:
            if self._locked or self._waiters:
                return False
            self._locked = True
            return True

    def release(self) -> None:
        with self._cv:
            self._locked = False
            self._cv.notify_all()

    def __enter__(self) -> None:
        self.acquire()

    def __exit__(self, *exc: Any) -> None:
        self.release()


#: Sentinel: "resolve the adapter from the platform". Tests override
#: ``_adapter_override`` with a fake module, or with ``None`` to disarm the
#: manager (no child: nothing spawns, the artifact reads as present).
_PLATFORM = object()
_adapter_override: Any = _PLATFORM

_state_lock = threading.RLock()
_spawn_lock = threading.Lock()
_request_lock = _FifoLock()
_handle: Optional[Dict[str, Any]] = None
#: A child spawned but not yet probed/published: ``(adapter, proc)``. The
#: shutdown terminates it so a probe in flight cannot outlive the backend.
_pending: Optional[Tuple[Any, Any]] = None
_closed = False
_leases = 0
_last_used = 0.0
_reaper_started = False
_prewarm_failed_at: Optional[float] = None
_prewarm_missing_logged = False
#: Failed spawns so far, and the last one's error: a caller that waited on
#: ``_spawn_lock`` while a spawn failed raises that spawn's error instead of
#: paying a second spawn (and a second probe timeout) on its own.
_spawn_failures = 0
_last_spawn_error: Optional[BaseException] = None


def _get_adapter() -> Any:
    """The platform's adapter module (every platform has one), or the test
    seam's override (``None``: the suite's disarmed manager)."""
    if _adapter_override is not _PLATFORM:
        return _adapter_override
    if platform.system().lower() == "darwin" and "arm" in platform.machine().lower():
        from src.ingestion import mlx_embedder

        return mlx_embedder
    from src.ingestion import llama_embedder

    return llama_embedder


def artifact_available() -> bool:
    """Whether the adapter's embedding artifact is on disk — the e5 snapshot
    on macOS, the bundled GGUF on Windows/Linux. True when the manager is
    disarmed (the test seam set the adapter to ``None``): nothing to miss."""
    adapter = _get_adapter()
    if adapter is None:
        return True
    return bool(adapter.artifact_available())


def artifact_downloadable() -> bool:
    """Whether the #146 download gate can produce the adapter's artifact
    (True on macOS — the child serves the downloadable snapshot; False on
    Windows/Linux — the GGUF ships inside the installers)."""
    adapter = _get_adapter()
    if adapter is None:
        return True
    return bool(adapter.artifact_downloadable)


def remove_obsolete_e5_snapshot() -> bool:
    """Delete the fp32 e5 snapshot where nothing reads it anymore (#594).

    Called once per start (idempotent: an absent snapshot is a no-op). The
    snapshot is obsolete only where the platform's artifact ships with the
    installation (Windows/Linux: the bundled GGUF) AND that artifact is on
    disk; on macOS it IS the child's artifact and is never touched.
    Returns whether a snapshot was deleted; never raises.
    """
    adapter = _get_adapter()
    if adapter is None or adapter.artifact_downloadable:
        return False
    if not adapter.artifact_available():
        return False
    return embedding_model.remove_snapshot()


def _require_adapter() -> Any:
    adapter = _get_adapter()
    if adapter is None:
        raise EngineException(
            message="The embedding child manager is disarmed (no adapter)",
        )
    return adapter


def _pick_free_port() -> int:
    """A free loopback port in the embedder's slice (same TOCTOU caveat as the
    chat engines' pick: the probe reports a hint if a race steals it)."""
    for offset in range(EMBEDDER_PORT_COUNT):
        port = EMBEDDER_PORT_START + offset
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", port))
            return port
        except OSError:
            continue
    raise EngineException(
        message=(
            f"No free port for the embedding child in range "
            f"{EMBEDDER_PORT_START}-{EMBEDDER_PORT_START + EMBEDDER_PORT_COUNT - 1}"
        ),
    )


def _touch() -> None:
    global _last_used
    _last_used = time.monotonic()


def _read_tail(adapter: Any, handle: Dict[str, Any]) -> str:
    try:
        return adapter.read_tail(handle.get("log_path"), handle.get("started_at"))
    except Exception:
        return "No child output could be read."


def _probe_ready(adapter: Any, handle: Dict[str, Any]) -> None:
    """Two-stage readiness: ``/health`` 200, then one real 1-input embedding.

    Runs with NO module lock held (only the caller's ``_spawn_lock``), so a
    shutdown can act meanwhile: it terminates the pending process, the
    liveness check below fails, and the caller never publishes. Raises
    :class:`EngineException` (with the child's log tail) on an early death,
    a timeout, or a failing embeddings probe — the caller terminates the
    child.
    """
    headers = {"Authorization": f"Bearer {handle['api_key']}"}
    base_url = handle["base_url"]
    deadline = time.monotonic() + PROBE_TIMEOUT_S
    last_status: Optional[int] = None
    last_err: Optional[Exception] = None
    while time.monotonic() < deadline:
        if not adapter.is_alive(handle["proc"]):
            raise EngineException(
                message=(
                    f"{adapter.server_name} exited before becoming ready "
                    f"(pid {handle.get('pid')}, port {handle['port']}). "
                    f"{_read_tail(adapter, handle)}"
                ),
            )
        try:
            response = requests.get(f"{base_url}/health", timeout=2.0, headers=headers)
            last_status = response.status_code
            if response.status_code == 200:
                break
        except requests.RequestException as exc:
            last_err = exc
        time.sleep(PROBE_POLL_INTERVAL_S)
    else:
        raise EngineException(
            message=(
                f"{adapter.server_name} did not become ready within "
                f"{PROBE_TIMEOUT_S:.0f}s (last status: {last_status}, last err: "
                f"{last_err}; port {handle['port']}). {_read_tail(adapter, handle)}"
            ),
        )
    # Stage 2 — one real embedding. mlx_vlm answers /health 200 even when the
    # preload failed; only this proves the model loaded and pools vectors.
    try:
        response = requests.post(
            f"{base_url}/v1/embeddings",
            json={"model": handle["model_value"], "input": ["ping"]},
            timeout=EMBED_REQUEST_TIMEOUT_S,
            headers=headers,
        )
    except requests.RequestException as exc:
        raise EngineException(
            message=(
                f"{adapter.server_name} embeddings probe failed on port "
                f"{handle['port']}: {exc}. {_read_tail(adapter, handle)}"
            ),
        )
    if response.status_code >= 400:
        raise EngineException(
            message=(
                f"{adapter.server_name} embeddings probe returned HTTP "
                f"{response.status_code} on port {handle['port']}: "
                f"{response.text[:200]}. {_read_tail(adapter, handle)}"
            ),
        )


def _start_reaper_locked() -> None:
    global _reaper_started
    if _reaper_started:
        return
    thread = threading.Thread(target=_reaper_loop, name="kb-embedder-reaper", daemon=True)
    thread.start()
    _reaper_started = True


def _reaper_loop() -> None:
    while True:
        time.sleep(REAP_CHECK_INTERVAL_S)
        try:
            reap_if_idle()
        except Exception as exc:
            # Nobody joins this thread: this record is the only trace.
            logger.error(f"Embedding child idle reap failed: {exc}", exc_info=True)


def _closed_error() -> EngineException:
    return EngineException(message="The embedding child manager is closed (backend shutdown)")


def _spawn_and_publish(adapter: Any) -> Dict[str, Any]:
    """Spawn, probe (unlocked), then publish — or terminate and raise.

    Caller holds ``_spawn_lock`` (one spawn at a time), never ``_state_lock``.
    """
    global _pending, _handle, _prewarm_missing_logged
    model_path = adapter.resolve_model()
    port = _pick_free_port()
    api_key = secrets.token_hex(32)
    started_at = time.time()
    log_path = adapter.prepare_log(port)
    spawn_start = time.monotonic()
    proc = adapter.spawn(port=port, api_key=api_key, model_path=model_path, log_path=log_path)
    with _state_lock:
        if _closed:
            adapter.terminate(proc)
            raise _closed_error()
        _pending = (adapter, proc)
    handle: Dict[str, Any] = {
        "proc": proc,
        "pid": getattr(proc, "pid", None),
        "port": port,
        "base_url": f"http://127.0.0.1:{port}",
        # Read by the probe and every embeddings request; never log the
        # handle wholesale.
        "api_key": api_key,
        "model_path": model_path,
        "model_value": adapter.model_value(model_path),
        "log_path": log_path,
        "started_at": started_at,
        # The adapter that owns this process: the stop paths (reaper,
        # shutdown, release_now) use it instead of re-resolving the platform.
        "_adapter": adapter,
    }
    try:
        _probe_ready(adapter, handle)
    except Exception:
        with _state_lock:
            _pending = None
        adapter.terminate(proc)
        raise
    with _state_lock:
        _pending = None
        if _closed:
            adapter.terminate(proc)
            raise _closed_error()

        def _kill_at_exit() -> None:
            adapter.terminate(proc)

        atexit.register(_kill_at_exit)
        handle["_atexit"] = _kill_at_exit
        _handle = handle
        _prewarm_missing_logged = False
        _touch()
        _start_reaper_locked()
    logger.info(
        f"Embedding child ready: {adapter.server_name}, pid={handle['pid']}, "
        f"port={port}, ready in {time.monotonic() - spawn_start:.1f}s"
    )
    return handle


def _unregister_atexit(handle: Dict[str, Any]) -> None:
    """Unregister exactly THIS handle's atexit handler (never another's)."""
    handler = handle.pop("_atexit", None)
    if handler is not None:
        try:
            atexit.unregister(handler)
        except Exception:
            pass


def _stop_locked(reason: str) -> None:
    """Terminate the current child and clear the handle (state lock held)."""
    global _handle
    handle = _handle
    if handle is None:
        return
    logger.info(
        f"Stopping the embedding child ({reason}): pid={handle.get('pid')}, "
        f"port={handle.get('port')}"
    )
    adapter = handle["_adapter"]
    adapter.terminate(handle.get("proc"))
    adapter.discard_log(handle.get("log_path"))
    _unregister_atexit(handle)
    _handle = None


def _drop_dead_locked(handle: Dict[str, Any]) -> None:
    """Forget a child found dead (its log is kept: it is the only account)."""
    global _handle
    _unregister_atexit(handle)
    if _handle is handle:
        _handle = None


def ensure_embedder_running() -> Dict[str, Any]:
    """The live child's handle — spawning, probing and registering it first
    when none runs (or the previous one died). Refuses after the shutdown.

    Raises:
        EmbeddingModelMissingException: The e5 artifact is not on disk.
        EngineException: The child could not be spawned or probed (the
            message quotes its log tail), or the manager is closed.
    """
    global _spawn_failures, _last_spawn_error
    adapter = _require_adapter()
    with _state_lock:
        if _closed:
            raise _closed_error()
        failures_seen = _spawn_failures
        handle = _handle
        if handle is not None:
            if adapter.is_alive(handle.get("proc")):
                _touch()
                return handle
            logger.warning(
                f"The embedding child is no longer running (pid "
                f"{handle.get('pid')}, port {handle.get('port')}, exit code "
                f"{adapter.exit_code(handle.get('proc'))}); respawning. "
                f"{_read_tail(adapter, handle)}"
            )
            _drop_dead_locked(handle)
    with _spawn_lock:
        # Another thread may have spawned while this one waited.
        with _state_lock:
            if _closed:
                raise _closed_error()
            handle = _handle
            if handle is not None and adapter.is_alive(handle.get("proc")):
                _touch()
                return handle
            if _spawn_failures != failures_seen and _last_spawn_error is not None:
                # A spawn ran and failed while this caller waited (the job's
                # background start, another request): its failure is this
                # caller's answer too. The next caller tries again.
                failure = _last_spawn_error
                raise _joined_failure(failure) from failure
        try:
            handle = _spawn_and_publish(adapter)
        except BaseException as exc:
            with _state_lock:
                _spawn_failures += 1
                _last_spawn_error = exc
            raise
        with _state_lock:
            # Nobody can join a failure once a spawn succeeded: drop it (and
            # the frames its traceback holds).
            _last_spawn_error = None
        return handle


def _joined_failure(failure: BaseException) -> BaseException:
    """What a caller that waited on a failed spawn raises: a FRESH exception
    of the same kind (each waiter's traceback is its own), chained to the
    spawn's failure by the ``raise ... from`` at the call site, and flagged
    ``joined_spawn`` so the caller knows the spawn's owner reports it."""
    if isinstance(failure, EmbeddingModelMissingException):
        fresh: AppBaseException = EmbeddingModelMissingException(trace=failure.trace)
    elif isinstance(failure, EngineException):
        fresh = EngineException(
            message=failure.message,
            trace=failure.trace,
            engine_code=getattr(failure, "engine_code", None),
        )
    else:
        fresh = EngineException(message=f"{type(failure).__name__}: {failure}")
    fresh.joined_spawn = True
    return fresh


def joined_spawn(exc: BaseException) -> bool:
    """Whether ``exc`` is a failure this caller only joined (another caller
    owned the spawn and reports it)."""
    return bool(getattr(exc, "joined_spawn", False))


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed ``texts`` through the child, one bounded batch per lock hold.

    The transport layer only: prefixes and the 512-token bound belong to
    ``ChildE5Embeddings``. The FIFO request lock is taken PER BATCH, so a
    search query asked during a long ingestion interleaves between batches
    instead of waiting out the file. Raises :class:`EngineException` when
    the child died mid-request (the report quotes its log tail) or answered
    an error; a child that died BETWEEN two batches is respawned and the
    call goes on.
    """
    if not texts:
        return []
    adapter = _require_adapter()
    vectors: List[List[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        with _request_lock:
            handle = ensure_embedder_running()
            headers = {"Authorization": f"Bearer {handle['api_key']}"}
            try:
                response = requests.post(
                    f"{handle['base_url']}/v1/embeddings",
                    json={"model": handle["model_value"], "input": batch},
                    timeout=EMBED_REQUEST_TIMEOUT_S,
                    headers=headers,
                )
            except requests.RequestException as exc:
                raise _request_error(adapter, handle, cause=str(exc))
            if response.status_code >= 400:
                raise _request_error(
                    adapter,
                    handle,
                    cause=f"HTTP {response.status_code}: {response.text[:200]}",
                )
            payload = response.json()
            rows = sorted(payload["data"], key=lambda item: item["index"])
            vectors.extend([row["embedding"] for row in rows])
            _touch()
    return vectors


def _request_error(adapter: Any, handle: Dict[str, Any], cause: str) -> EngineException:
    """The error for a failed embeddings request: a child found dead is
    dropped (next call respawns) and reported with its log tail; a live
    child's HTTP error keeps the child and reports the status."""
    if not adapter.is_alive(handle.get("proc")):
        with _state_lock:
            _drop_dead_locked(handle)
        return EngineException(
            message=(
                f"{adapter.server_name} died during an embedding request "
                f"(pid {handle.get('pid')}, port {handle.get('port')}, exit code "
                f"{adapter.exit_code(handle.get('proc'))}): {cause}. "
                f"{_read_tail(adapter, handle)}"
            ),
        )
    return EngineException(
        message=(
            f"{adapter.server_name} embedding request failed on port "
            f"{handle.get('port')}: {cause}"
        ),
    )


@contextlib.contextmanager
def ingestion_lease() -> Iterator[None]:
    """Hold the child across one ingestion job: the idle reaper never
    terminates it while a lease is held; releasing restarts the idle clock."""
    global _leases
    with _state_lock:
        _leases += 1
    try:
        yield
    finally:
        with _state_lock:
            _leases -= 1
            _touch()


def reap_if_idle(now: Optional[float] = None) -> bool:
    """Terminate the child once it sat unused past the fixed idle delay.

    ``now`` (a ``time.monotonic`` value) exists for tests; production passes
    nothing. Never reaps under a lease. Returns whether it reaped.
    """
    with _state_lock:
        if _handle is None or _leases > 0:
            return False
        moment = time.monotonic() if now is None else now
        if moment - _last_used < EMBEDDER_IDLE_SECONDS:
            return False
        _stop_locked(reason="idle for 5 minutes")
        return True


def release_now(reason: str) -> bool:
    """Terminate the child NOW, idle delay notwithstanding — the MLX memory
    brake's hook under critical pressure.

    Refused when a lease is held (an ingestion is running) or a request is
    in flight (the FIFO lock is busy): interrupting either would fail a KB
    operation to save memory the next sample can still reclaim. Returns
    whether the child was released.
    """
    if not _request_lock.try_acquire():
        return False
    try:
        with _state_lock:
            if _handle is None or _leases > 0:
                return False
            _stop_locked(reason=reason)
            return True
    finally:
        _request_lock.release()


def prewarm_embedder() -> None:
    """Start the child ahead of the first KB use, in the background.

    Called when a conversation with a Knowledge Base opens. Fire-and-forget:
    nothing is awaited, nothing raises into the caller. Skips — without a
    thread and without touching the idle clock — a live child, a spawn in
    flight, a closed manager, and the backoff window after a failure.
    """
    adapter = _get_adapter()
    if adapter is None:
        return
    with _state_lock:
        if _closed or _pending is not None:
            return
        if _handle is not None and adapter.is_alive(_handle.get("proc")):
            return
        if (
            _prewarm_failed_at is not None
            and time.monotonic() - _prewarm_failed_at < PREWARM_BACKOFF_S
        ):
            return
    threading.Thread(target=_prewarm, name="kb-embedder-prewarm", daemon=True).start()


class BackgroundStart:
    """One `start_in_background` call: when it ends, how long it took and
    how it failed.

    Attributes:
        seconds: The time ``ensure_embedder_running`` took (a spawn and its
            probe, or about nothing when the child was already running);
            ``None`` until it returns.
        error: What it raised, ``None`` on success. Not logged here: an
            embedding call that waited for this attempt gets its failure
            and reports it where it is handled; a later call spawns again,
            and the caller (the job) logs this error as a WARNING when that
            second start succeeds.
    """

    def __init__(self) -> None:
        self.seconds: Optional[float] = None
        self.error: Optional[BaseException] = None
        self._done = threading.Event()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for the start to end; returns whether it ended."""
        return self._done.wait(timeout)

    def _run(self) -> None:
        start = time.perf_counter()
        try:
            ensure_embedder_running()
        except BaseException as exc:
            # Kept for the caller, never logged here: the caller reports it
            # (see the class docstring) or, when it embedded nothing, did
            # not need the child.
            self.error = exc
        finally:
            self.seconds = time.perf_counter() - start
            self._done.set()


def start_in_background() -> Optional[BackgroundStart]:
    """Start the child on a background thread, beside the caller's own work.

    An ingestion job calls it once its first file has been read and holds
    text to index, so the spawn runs while the job chunks that file (and
    loads the e5 tokenizer) instead of after. The job holds its ingestion lease, so
    the idle reaper leaves the child alone. ``None`` when the manager is
    disarmed (tests); otherwise the start's record. Its failure is not
    reported here (see `BackgroundStart`).
    """
    if _get_adapter() is None:
        return None
    start = BackgroundStart()
    threading.Thread(target=start._run, name="kb-embedder-start", daemon=True).start()
    return start


def _prewarm() -> None:
    """The prewarm thread's body. An absent artifact is an interface state
    (the gate/banner owns it): one INFO per state, then silence until a
    spawn succeeds. Anything else degraded the pre-start only — the first
    KB use retries and reports properly: one WARNING, then the backoff."""
    global _prewarm_failed_at, _prewarm_missing_logged
    try:
        ensure_embedder_running()
    except EmbeddingModelMissingException:
        _prewarm_failed_at = time.monotonic()
        if not _prewarm_missing_logged:
            _prewarm_missing_logged = True
            logger.info("Embedding child pre-start skipped: the embedding model is not downloaded")
    except Exception as exc:
        _prewarm_failed_at = time.monotonic()
        if joined_spawn(exc):
            # Another caller (an ingestion job) owned this spawn and reports
            # its failure: a second record here would count it twice.
            return
        logger.warning(f"Embedding child pre-start failed (the first KB use will retry): {exc}")


def shutdown_embedder() -> None:
    """Terminate the child and CLOSE the manager (lifespan shutdown).

    Terminal: a background thread that outlives the lifespan can never
    respawn a child into a stopping backend. A spawn whose probe is in
    flight is terminated here (bounded) and refused at publication; the
    shutdown never waits out the probe timeout. Idempotent, never raises.
    """
    global _closed, _pending
    with _state_lock:
        _closed = True
        if _pending is not None:
            adapter, proc = _pending
            adapter.terminate(proc)
            # The prober clears _pending itself and sees _closed.
        _stop_locked(reason="shutdown")


def _reset_for_tests() -> None:
    """Clear module state between tests (terminates any live child)."""
    global _handle, _pending, _closed, _leases, _last_used
    global _prewarm_failed_at, _prewarm_missing_logged, _spawn_failures, _last_spawn_error
    with _state_lock:
        _stop_locked(reason="test reset")
        if _pending is not None:
            adapter, proc = _pending
            try:
                adapter.terminate(proc)
            except Exception:
                pass
        _handle = None
        _pending = None
        _closed = False
        _leases = 0
        _last_used = 0.0
        _prewarm_failed_at = None
        _prewarm_missing_logged = False
        _spawn_failures = 0
        _last_spawn_error = None
