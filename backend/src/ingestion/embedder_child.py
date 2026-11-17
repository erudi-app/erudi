"""Dedicated Knowledge Base embedding child — shared lifecycle manager (#594).

The KB embedder runs in ITS OWN child process, deliberately outside the chat
engines: it never enters `BaseEngine`/`BaseChatServerEngine` class state, the
CUDA gauges, the memory brake or the Settings idle delay. What it shares with
the chat child is the SHAPE of the lifecycle, re-implemented here as plain
module functions so the Windows/Linux `llama-server --embedding` child (F3 of
#594) plugs in a second adapter module without touching the chat path:

    port pick (27400-27499, above every chat pool) → per-spawn ``--api-key``
    (``secrets.token_hex(32)``, never logged) → two-stage readiness probe →
    serialized ``POST /v1/embeddings`` batches → fixed 5-minute idle reap →
    bounded terminate, with the crash report quoting the child's own log.

Two-stage probe, and why ``/health`` is not enough: `mlx_vlm.server` records
a failed ``--embedding-model`` preload in ``runtime.preload_failures`` and
still answers ``/health`` 200 — only a real 1-input ``POST /v1/embeddings``
proves the model loaded, pooled and answered. Every request carries
``"model": <local snapshot path>`` because mlx_vlm POPS the preload env var
at startup: a request without the field answers 400 on a healthy server.

Lifecycle policy (product decisions on #594):

- **Own lock.** A single ``threading.RLock`` serializes spawn/stop/reap (the
  KB business layer is sync); one ``threading.Lock`` serializes the HTTP
  requests — one embedding batch in flight at a time, which is what the GIL
  gave the in-process embedder.
- **Ingestion lease.** `ingestion_lease` brackets a whole ingestion job: the
  idle reaper never terminates the child mid-ingestion, however long the job
  runs; releasing the lease restarts the idle clock.
- **Fixed idle.** `EMBEDDER_IDLE_SECONDS` (5 minutes, no setting — decided on
  #594; independent of the chat model's ``idle_unload_minutes``). A daemon
  reaper thread samples every `REAP_CHECK_INTERVAL_S`.
- **Pre-start.** `prewarm_embedder` is called when a conversation with a KB
  opens (``conversations`` endpoints): a background thread pays the 1-2 s
  spawn ahead of the first KB use. It never raises; an absent artifact is
  one INFO (an interface state, not a failure), anything else one WARNING.

Platform dispatch: `_get_adapter` serves `src.ingestion.mlx_embedder` on
macOS ARM and ``None`` elsewhere — Windows and Linux keep the in-process
embedder until F3. ``ERUDI_FORCE_CPU`` is deliberately ignored: it picks the
CHAT engine, and the embedder is not a chat engine (mlx_vlm is the only
embedding server shipped on macOS).
"""

from __future__ import annotations

import atexit
import contextlib
import platform
import secrets
import socket
import threading
import time
from typing import Any, Dict, Iterator, List, Optional

import requests

from src.core.exceptions import EmbeddingModelMissingException, EngineException
from src.core.logging import logger

# The embedder's own port slice: above the llama.cpp pools (27200-27299), the
# MLX chat pool (27300-27399) and the backend's 27182-27199 scan.
EMBEDDER_PORT_START = 27400
EMBEDDER_PORT_COUNT = 100

#: Released after this long unused — fixed, no setting (#594 decision).
EMBEDDER_IDLE_SECONDS = 300.0

#: How often the reaper thread samples the idle clock.
REAP_CHECK_INTERVAL_S = 15.0

PROBE_TIMEOUT_S = 120.0
PROBE_POLL_INTERVAL_S = 0.3

#: Texts per ``/v1/embeddings`` request. The server pads a request's texts
#: into ONE batch; 32 bounds that batch the way sentence-transformers'
#: internal ``encode`` batching did.
EMBED_BATCH_SIZE = 32
EMBED_REQUEST_TIMEOUT_S = 120.0

#: Sentinel: "resolve the adapter from the platform". Tests override
#: ``_adapter_override`` with a fake module (or ``None`` for "unsupported").
_PLATFORM = object()
_adapter_override: Any = _PLATFORM

_state_lock = threading.RLock()
_request_lock = threading.Lock()
_handle: Optional[Dict[str, Any]] = None
_leases = 0
_last_used = 0.0
_reaper_started = False
_atexit_handler = None


def _get_adapter() -> Any:
    """The platform's adapter module, or ``None`` where no child exists."""
    if _adapter_override is not _PLATFORM:
        return _adapter_override
    if platform.system().lower() == "darwin" and "arm" in platform.machine().lower():
        from src.ingestion import mlx_embedder

        return mlx_embedder
    return None


def embedder_child_supported() -> bool:
    """Whether this platform serves KB embeddings from a dedicated child."""
    return _get_adapter() is not None


def _require_adapter() -> Any:
    adapter = _get_adapter()
    if adapter is None:
        raise EngineException(
            message="No dedicated embedding child exists on this platform",
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

    Raises :class:`EngineException` (with the child's log tail) on an early
    death, a timeout, or a failing embeddings probe — the caller terminates
    the child.
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


def _spawn_locked(adapter: Any) -> Dict[str, Any]:
    model_path = adapter.resolve_model()
    port = _pick_free_port()
    api_key = secrets.token_hex(32)
    started_at = time.time()
    log_path = adapter.prepare_log(port)
    spawn_start = time.monotonic()
    proc = adapter.spawn(port=port, api_key=api_key, model_path=model_path, log_path=log_path)
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
    }
    try:
        _probe_ready(adapter, handle)
    except Exception:
        adapter.terminate(proc)
        raise
    global _handle, _atexit_handler

    def _kill_at_exit() -> None:
        adapter.terminate(proc)

    atexit.register(_kill_at_exit)
    _atexit_handler = _kill_at_exit
    _handle = handle
    _touch()
    _start_reaper_locked()
    logger.info(
        f"Embedding child ready: {adapter.server_name}, pid={handle['pid']}, "
        f"port={port}, ready in {time.monotonic() - spawn_start:.1f}s"
    )
    return handle


def _stop_locked(adapter: Any, reason: str) -> None:
    """Terminate the current child and clear the handle (lock held)."""
    global _handle, _atexit_handler
    handle = _handle
    if handle is None:
        return
    logger.info(
        f"Stopping the embedding child ({reason}): pid={handle.get('pid')}, "
        f"port={handle.get('port')}"
    )
    adapter.terminate(handle.get("proc"))
    adapter.discard_log(handle.get("log_path"))
    if _atexit_handler is not None:
        try:
            atexit.unregister(_atexit_handler)
        except Exception:
            pass
        _atexit_handler = None
    _handle = None


def _drop_dead_locked(adapter: Any, handle: Dict[str, Any]) -> None:
    """Forget a child found dead (its log is kept: it is the only account)."""
    global _handle, _atexit_handler
    if _atexit_handler is not None:
        try:
            atexit.unregister(_atexit_handler)
        except Exception:
            pass
        _atexit_handler = None
    if _handle is handle:
        _handle = None


def ensure_embedder_running() -> Dict[str, Any]:
    """The live child's handle — spawning, probing and registering it first
    when none runs (or the previous one died).

    Raises:
        EmbeddingModelMissingException: The e5 artifact is not on disk.
        EngineException: The child could not be spawned or probed; the
            message quotes its log tail.
    """
    adapter = _require_adapter()
    with _state_lock:
        handle = _handle
        if handle is not None:
            if adapter.is_alive(handle.get("proc")):
                _touch()
                return handle
            logger.warning(
                f"The embedding child is no longer running (pid "
                f"{handle.get('pid')}, port {handle.get('port')}, exit code "
                f"{getattr(handle.get('proc'), 'exitcode', None)}); respawning. "
                f"{_read_tail(adapter, handle)}"
            )
            _drop_dead_locked(adapter, handle)
        return _spawn_locked(adapter)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed ``texts`` through the child, in bounded batches, serialized.

    The transport layer only: prefixes and the 512-token bound belong to
    ``ChildE5Embeddings``. Raises :class:`EngineException` when the child
    died mid-request (the report quotes its log tail) or answered an error.
    """
    if not texts:
        return []
    with _request_lock:
        handle = ensure_embedder_running()
        adapter = _require_adapter()
        headers = {"Authorization": f"Bearer {handle['api_key']}"}
        vectors: List[List[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
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
            _drop_dead_locked(adapter, handle)
        return EngineException(
            message=(
                f"{adapter.server_name} died during an embedding request "
                f"(pid {handle.get('pid')}, port {handle.get('port')}, exit code "
                f"{getattr(handle.get('proc'), 'exitcode', None)}): {cause}. "
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
    adapter = _get_adapter()
    if adapter is None:
        return False
    with _state_lock:
        if _handle is None or _leases > 0:
            return False
        moment = time.monotonic() if now is None else now
        if moment - _last_used < EMBEDDER_IDLE_SECONDS:
            return False
        _stop_locked(adapter, reason="idle for 5 minutes")
        return True


def prewarm_embedder() -> None:
    """Start the child ahead of the first KB use, in the background.

    Called when a conversation with a Knowledge Base opens. Fire-and-forget:
    nothing is awaited, nothing raises into the caller.
    """
    if _get_adapter() is None:
        return
    threading.Thread(target=_prewarm, name="kb-embedder-prewarm", daemon=True).start()


def _prewarm() -> None:
    """The prewarm thread's body. An absent artifact is an interface state
    (the gate/banner owns it): one INFO. Anything else degraded the
    pre-start only — the first KB use retries and reports properly: WARNING."""
    try:
        ensure_embedder_running()
    except EmbeddingModelMissingException:
        logger.info("Embedding child pre-start skipped: the embedding model is not downloaded")
    except Exception as exc:
        logger.warning(f"Embedding child pre-start failed (the first KB use will retry): {exc}")


def shutdown_embedder() -> None:
    """Terminate the child (lifespan shutdown). Idempotent, never raises."""
    adapter = _get_adapter()
    if adapter is None:
        return
    with _state_lock:
        _stop_locked(adapter, reason="shutdown")


def _reset_for_tests() -> None:
    """Clear module state between tests (terminates any live child)."""
    global _handle, _leases, _last_used, _atexit_handler
    adapter = _get_adapter()
    with _state_lock:
        if _handle is not None and adapter is not None:
            _stop_locked(adapter, reason="test reset")
        _handle = None
        _leases = 0
        _last_used = 0.0
        if _atexit_handler is not None:
            try:
                atexit.unregister(_atexit_handler)
            except Exception:
                pass
            _atexit_handler = None
