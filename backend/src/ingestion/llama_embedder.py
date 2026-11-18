"""Windows/Linux adapter for the dedicated KB embedding child (#594, F3):
the bundled CPU ``llama-server`` started with ``--embedding`` around the e5
GGUF that ships INSIDE the installers.

This module implements the platform-adapter protocol `embedder_child`
consumes (``server_name``, ``artifact_available``, ``artifact_downloadable``,
``resolve_model``, ``model_value``, ``prepare_log``, ``spawn``,
``terminate``, ``is_alive``, ``exit_code``, ``read_tail``, ``discard_log``)
— the sibling of the macOS `mlx_embedder`.

The artifact is ``artifacts/kb-embedder/multilingual-e5-small-q8_0.gguf``,
produced at build time from the pinned e5 snapshot by
``scripts/dev/backend/build_e5_gguf.py`` (community e5 GGUFs carry a wrong
WordPiece vocab — spike S4) and bundled by the PyInstaller specs. It is
**never downloaded at runtime** (``artifact_downloadable = False``): absent —
a dev tree where the script has not run — the F2 missing-model state applies
(:class:`EmbeddingModelMissingException`), and the #146 gate offers no
download for it.

The serving command is the spike-S4 one:

    llama-server -m e5.gguf --embedding --pooling mean -c 512 -b 512 -ub 512
                 -np 1 --poll 0 -ngl 0 --no-slots --no-webui

- ``-np 1`` is mandatory: llama-server sizes its logits buffer per slot, and
  the default 4 slots reserve ~560 MB where one slot holds ~315 MB.
- ``--poll 0``: ggml's spin-wait worker threads collapse under CPU
  contention, and this child shares the machine with the chat model.
- ``--pooling mean`` is defensive — the GGUF already carries MEAN pooling.
- No ``--threads``: llama-server's default counts the physical cores (see
  ``CPU_Engine._build_spawn_argv``); a user's ``LLAMA_ARG_THREADS`` reaches
  the child through the inherited environment.
- Inputs past 512 tokens are a hard HTTP 500 that fails the WHOLE batch
  (llama-server refuses, never truncates), so ``ChildE5Embeddings`` bounds
  every input Erudi-side (``chunking.truncate_to_tokens``) before the call.

**This child never touches a graphics card.** It always runs the CPU
``llama-server`` (the CUDA installers ship it beside the CUDA one for the
processor engine; ``cuda/bin`` is only the dev fallback for a tree where one
flavour was compiled, with a WARNING) under ``CUDA_VISIBLE_DEVICES=-1`` and
``-ngl 0``. With every device masked it never registers as a CUDA compute
process, and the CUDA gauges (spill, eviction, ``gpu_contention``) sample
the pid of the CHAT handle (``CUDA_Engine._model``) only — this child lives
in `embedder_child` module state, outside every engine class, their brakes
and the Settings idle delay (its own fixed idle is `embedder_child`'s).

The per-spawn API key travels by ENVIRONMENT (``LLAMA_API_KEY``, bound to
``--api-key`` in the pinned build's ``common/arg.cpp``), never on the
command line — the key is invisible to ``ps``, the same property the MLX
child gets from its pickled argv. The child's own output goes to
``embedder-child-<port>.log`` beside ``backend.log`` through `mlx_child_log`'s
path-generic helpers, exactly like the macOS embedding child (docs/logging.md).
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Any, List, Optional

from src.core import config
from src.core.exceptions import EmbeddingModelMissingException, EngineException
from src.core.logging import logger
from src.core.subprocess_flags import hidden_console_creationflags
from src.engines import mlx_child_log as child_log
from src.launcher import ensure_runtime_paths_initialized

SERVER_NAME = "llama-server (embedding child)"
server_name = SERVER_NAME

#: The GGUF ships with the installation; nothing can download it at runtime.
artifact_downloadable = False

#: The file ``scripts/dev/backend/build_e5_gguf.py`` produces and the
#: PyInstaller specs bundle (Q8_0, 132 MB; worst cosine vs
#: sentence-transformers 0.99981 — spike S4).
E5_GGUF_NAME = "multilingual-e5-small-q8_0.gguf"

#: The ``model`` field of every request and the server's ``--alias``.
E5_ALIAS = "multilingual-e5-small"

#: Same prefix as the macOS embedding child: one platform serves at a time,
#: and the retention/crash-report semantics are shared (docs/logging.md).
EMBEDDER_LOG_PREFIX = "embedder-child-"

#: e5's trained window; the context, batch and ubatch are all pinned to it
#: (position embeddings exist for 512 positions only).
EMBED_WINDOW_TOKENS = 512


def gguf_path() -> Path:
    """Where the bundled e5 GGUF lives, packaged and dev alike."""
    return config.ROOT_DIR / "artifacts" / "kb-embedder" / E5_GGUF_NAME


def artifact_available() -> bool:
    """Whether the bundled GGUF is on disk (the gate/banner's answer)."""
    return gguf_path().is_file()


def resolve_model() -> Path:
    """The GGUF file the child will serve.

    Raises:
        EmbeddingModelMissingException: The GGUF is not on disk — a dev tree
            where ``scripts/dev/backend/build_e5_gguf.py`` has not run (a
            packaged install always ships it). The caller surfaces the state
            instead of spawning a child that cannot load.
    """
    path = gguf_path()
    if not path.is_file():
        raise EmbeddingModelMissingException()
    return path


def model_value(model_path: Path) -> str:
    """The ``model`` field every request carries. llama-server serves its one
    model whatever the field says; the alias keeps requests and responses
    legible."""
    return E5_ALIAS


def _find_llama_server() -> Path:
    """The CPU ``llama-server`` beside the app (`BaseLlamaCppEngine` layout).

    The CPU flavour ships in EVERY Windows/Linux installer (the CUDA ones
    carry it for the processor engine), so the primary path always exists in
    a packaged install. The cuda/bin fallback is a development safety net for
    a tree where only that flavour was compiled — said out loud, and safe:
    the spawn masks every GPU and passes ``-ngl 0``.
    """
    exe = "llama-server.exe" if os.name == "nt" else "llama-server"
    primary = config.ROOT_DIR / "artifacts" / "llama-cpp" / "cpu" / "bin" / exe
    if primary.exists():
        return primary
    fallback = config.ROOT_DIR / "artifacts" / "llama-cpp" / "cuda" / "bin" / exe
    if fallback.exists():
        logger.warning(
            f"[llama_embedder] llama-server not found at {primary}; "
            f"falling back to the cuda build at {fallback} (GPUs stay masked)"
        )
        return fallback
    raise EngineException(
        message=(
            f"llama-server binary not found at {primary} or {fallback}. "
            f"Build llama.cpp first (see scripts/dev/backend/build-llamacpp-*)."
        ),
    )


def _server_command() -> List[str]:
    """The command the spawn prepends to the server flags. A seam: the HTTP
    double of the CI suite substitutes ``[python, fake_server.py]`` here and
    everything downstream (Popen, log redirect, env, key) stays the real
    code path."""
    return [str(_find_llama_server())]


def _log_path(port: int) -> Path:
    """Beside ``backend.log``, same directory as every inference child's."""
    return ensure_runtime_paths_initialized().log_dir / f"{EMBEDDER_LOG_PREFIX}{port}.log"


def prepare_log(port: int) -> Optional[str]:
    """Roll the previous spawn's file and return this one's path, or ``None``.

    Best effort: a log directory that cannot be written costs the tail,
    never the spawn.
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


def spawn(
    *, port: int, api_key: str, model_path: Path, log_path: Optional[str]
) -> subprocess.Popen:
    """Spawn the embedding ``llama-server`` (see module docstring for every
    flag's reason). The key goes through ``LLAMA_API_KEY`` in the child's
    environment, never argv; the backend's own environment is untouched."""
    argv = _server_command() + [
        "-m",
        str(model_path),
        "--embedding",
        "--pooling",
        "mean",
        "-c",
        str(EMBED_WINDOW_TOKENS),
        "-b",
        str(EMBED_WINDOW_TOKENS),
        "-ub",
        str(EMBED_WINDOW_TOKENS),
        "-np",
        "1",
        "--poll",
        "0",
        "-ngl",
        "0",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--alias",
        E5_ALIAS,
        "--no-slots",
        "--no-webui",
    ]
    env = os.environ.copy()
    env["LLAMA_API_KEY"] = api_key
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    stdout: Any = subprocess.DEVNULL
    log_handle = None
    if log_path:
        try:
            log_handle = open(log_path, "ab")
            stdout = log_handle
        except OSError as exc:
            logger.warning(
                f"Cannot open the {SERVER_NAME} log file {log_path}: "
                f"{type(exc).__name__}: {exc}"
            )
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            env=env,
            # llama-server is a console exe: without this it opens its own
            # terminal window on Windows. No-op (0) on POSIX.
            creationflags=hidden_console_creationflags(),
        )
    except OSError as exc:
        raise EngineException(
            message=(
                f"Could not start the {SERVER_NAME} at {argv[0]} on port {port}: "
                f"{exc.strerror or exc}"
            ),
            trace=f"{type(exc).__name__}: {exc}",
        ) from exc
    finally:
        # The child holds its own descriptor; the parent's copy would only
        # leak (one per spawn).
        if log_handle is not None:
            log_handle.close()
    logger.info(
        f"[llama_embedder] Spawned {SERVER_NAME}: pid={proc.pid}, port={port}, "
        f"model={model_path}, output={log_path or 'not captured'}"
    )
    return proc


def terminate(proc: Any) -> None:
    """Idempotently terminate the child — the `BaseLlamaCppEngine` shape:
    SIGINT (terminate on Windows), wait 5 s, then kill. The only place this
    child's exit code is recorded, so the level says what happened: INFO for
    an orderly stop, WARNING when it had to be killed or had already died
    with a nonzero code."""
    if not proc:
        return
    try:
        if proc.poll() is None:
            if platform.system() == "Windows":
                proc.terminate()
            else:
                import signal

                proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=5)
                outcome = "terminated"
                degraded = False
            except Exception:
                proc.kill()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    pass
                outcome = "killed (after the stop signal timed out)"
                degraded = True
        else:
            outcome = "already exited"
            code = exit_code(proc)
            degraded = isinstance(code, int) and code != 0
        record = logger.warning if degraded else logger.info
        record(
            f"[llama_embedder] Embedding child terminated: {outcome}, "
            f"exitcode={exit_code(proc)}"
        )
    except Exception:
        # Best-effort teardown (the handle may already be gone); never let it
        # mask the failure that caused the stop.
        pass


def is_alive(proc: Any) -> bool:
    """Whether the Popen child still runs (False for ``None``)."""
    if proc is None:
        return False
    try:
        return proc.poll() is None
    except Exception:
        return False


def exit_code(proc: Any) -> Optional[int]:
    """The dead child's exit code (``Popen`` spells it ``returncode``)."""
    return getattr(proc, "returncode", None)


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
