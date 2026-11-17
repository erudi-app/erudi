"""macOS adapter for the dedicated KB embedding child (#594): `mlx_vlm.server`
started with ONLY an embedding model.

This module implements the platform-adapter protocol `embedder_child`
consumes (``server_name``, ``resolve_model``, ``model_value``,
``prepare_log``, ``spawn``, ``terminate``, ``is_alive``, ``read_tail``,
``discard_log``). F3 adds the Windows/Linux sibling around the bundled CPU
``llama-server --embedding``.

What is deliberately REUSED from the chat child, and what is not:

- The child entry point IS the chat one (``run_mlx_vlm_server``): it is the
  only picklable target a PyInstaller frozen build can spawn, it redirects
  the child's stdout/stderr into the per-spawn file the parent resolves
  (``mlx_child_log.redirect_stdio_to``), and bounds MLX's buffer cache. Its
  chat-only knobs (APC, token-queue timeout, the Gemma patches) are inert in
  a server that loads no chat model. The argv differs: ``--embedding-model
  <local snapshot path>`` and no ``--model``, so the child serves
  ``POST /v1/embeddings`` only from the snapshot the #146 gate downloaded.
- The engine CLASSES are not touched: this child never enters
  ``MLX_Engine``'s state, locks, gauges, memory brake or idle delay. Its
  lifecycle (port slice 27400-27499, per-spawn key, two-stage probe, lease,
  fixed 5-minute idle) is `embedder_child`'s.

The child's log files are ``embedder-child-<port>.log`` beside
``backend.log``, rolled, read and discarded through `mlx_child_log`'s
path-generic helpers — same retention and crash-report semantics as the
chat child's ``mlx-child-<port>.log`` (see docs/logging.md).
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any, Optional

from src.core.exceptions import EmbeddingModelMissingException, EngineException
from src.core.logging import logger
from src.engines import mlx_child_log as child_log
from src.engines._mlx_vlm_server_runner import run_mlx_vlm_server
from src.ingestion.embedding_model import embedding_snapshot_path
from src.launcher import ensure_runtime_paths_initialized

SERVER_NAME = "mlx_vlm.server (embedding child)"
server_name = SERVER_NAME

#: One file per spawn beside ``backend.log``, named after the child's port —
#: distinct from the chat child's ``mlx-child-<port>.log`` prefix so the two
#: processes never share a file.
EMBEDDER_LOG_PREFIX = "embedder-child-"


def resolve_model() -> Path:
    """The LOCAL e5 snapshot directory the child will serve.

    Raises:
        EmbeddingModelMissingException: When the #146 gate says the artifact
            is absent or incomplete — the caller surfaces the state instead
            of spawning a child that cannot load.
    """
    path = embedding_snapshot_path()
    if path is None:
        raise EmbeddingModelMissingException()
    return path


def model_value(model_path: Path) -> str:
    """The ``model`` field every request must carry.

    mlx_vlm's startup POPS ``MLX_VLM_PRELOAD_EMBEDDING_MODEL`` from the
    child's environment once the preload ran, so a request without a
    ``model`` field answers 400 — the preloaded snapshot path is the cache
    key ``get_cached_model`` resolves.
    """
    return str(model_path)


def _log_path(port: int) -> Path:
    """Beside ``backend.log``, same directory the chat child captures into."""
    return ensure_runtime_paths_initialized().log_dir / f"{EMBEDDER_LOG_PREFIX}{port}.log"


def prepare_log(port: int) -> Optional[str]:
    """Roll the previous spawn's file and return this one's path, or ``None``.

    Best effort, exactly like the chat child's capture: a log directory that
    cannot be written costs the tail, never the spawn.
    """
    try:
        path = _log_path(port)
        path.parent.mkdir(parents=True, exist_ok=True)
        child_log.roll_child_log(path)
        return str(path)
    except OSError as exc:
        logger.warning(
            f"Cannot capture the {SERVER_NAME} output on port {port}: "
            f"{type(exc).__name__}: {exc}"
        )
        return None


def spawn(*, port: int, api_key: str, model_path: Path, log_path: Optional[str]) -> mp.Process:
    """Spawn the embedding child. The argv travels by pickle (never a command
    line), so the key is invisible to ``ps`` — same property as the chat
    child's spawn."""
    argv = [
        "mlx_vlm.server",
        "--embedding-model",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "INFO",
        "--api-key",
        api_key,
    ]
    # (argv, log_path, apc_num_blocks, token_queue_timeout_s, watch_parent):
    # the chat knobs stay at their defaults; watch_parent makes the child
    # exit if the backend dies without its orderly stops (os._exit paths),
    # so nothing orphans a ~1.2 GB process.
    proc = mp.Process(
        target=run_mlx_vlm_server, args=(argv, log_path, None, None, True), daemon=False
    )
    try:
        proc.start()
    except Exception as exc:
        raise EngineException(
            message=(f"Could not start the {SERVER_NAME} for {model_path} on port {port}: {exc}"),
            trace=f"{type(exc).__name__}: {exc}",
        ) from exc
    logger.info(
        f"[mlx_embedder] Spawned {SERVER_NAME}: pid={proc.pid}, port={port}, "
        f"model={model_path}, output={log_path or 'not captured'}"
    )
    return proc


def terminate(proc: Any) -> None:
    """Idempotently terminate the child (SIGTERM, then SIGKILL after 5 s).

    The only place the embedding child's exit code is recorded, so the level
    says what happened: INFO for an orderly stop (idle reap, shutdown, model
    swap), WARNING when it had to be SIGKILLed or had already died with a
    nonzero code.
    """
    if not proc:
        return
    try:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=2)
                outcome = "killed (SIGKILL after SIGTERM timeout)"
                degraded = True
            else:
                outcome = "terminated (SIGTERM)"
                degraded = False
        else:
            outcome = "already exited"
            exitcode = getattr(proc, "exitcode", None)
            degraded = isinstance(exitcode, int) and exitcode != 0
        record = logger.warning if degraded else logger.info
        record(
            f"[mlx_embedder] Embedding child terminated: {outcome}, "
            f"exitcode={getattr(proc, 'exitcode', None)}"
        )
    except Exception:
        # Best-effort teardown; never let it mask the failure that caused it.
        pass


def is_alive(proc: Any) -> bool:
    """Whether the spawned ``mp.Process`` still runs (False for ``None``)."""
    if proc is None:
        return False
    try:
        return bool(proc.is_alive())
    except Exception:
        return False


def read_tail(log_path: Optional[str], since: Optional[float]) -> str:
    """Tail of the child's own output file — the only account of a death."""
    if not log_path:
        return "No child output was captured."
    tail = child_log.read_child_log_tail(log_path, since=since)
    if not tail:
        return "The child produced no output."
    return f"Child output (last {len(tail)} chars, from {log_path}):\n{tail}"


def discard_log(log_path: Optional[str]) -> None:
    """Delete an orderly-stopped child's files (a crashed child keeps its)."""
    if log_path:
        child_log.discard_child_log(log_path)
