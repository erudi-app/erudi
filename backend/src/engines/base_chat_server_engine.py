"""Abstract base for engines that wrap an OpenAI-compatible HTTP server child.

Currently shared by:
- `MLX_Engine`        spawns `mlx_vlm.server` via `multiprocessing.Process`
- `BaseLlamaCppEngine` → `CPU_Engine` / `CUDA_Engine`  spawn `llama-server`
  via `subprocess.Popen`

The pattern:

1. Pick a free port in a configurable range.
2. Spawn the child via the abstract `_spawn_child` hook.
3. Two-stage probe: `GET /health` (poll until 200, the upstream `503 loading`
   contract handles model warm-up), then a single `POST /v1/chat/completions`
   with `max_tokens=1` to validate chat template + tokenizer + sampling.
4. Register an `atexit` handler stored on the class so we can unregister it
   before a model switch (fixes the original closure leak — without this,
   every swap would leak a stale handler holding a dead `proc`).
5. Hand back the child's `base_url` + a per-engine `_translate_payload_kwargs`
   hook (mlx_vlm.server uses HF/transformers names, llama-server uses its own).
   The agent layer streams tokens over this `base_url` via `ChatOpenAI`; the
   engine no longer parses SSE itself.

Generation serialization + idle-cleanup suppression live in
`BaseEngine.generation_guard`, which the agent layer wraps around model
resolution + the whole token stream. The idle-cleanup tick shares the guard's
asyncio lock, so the child is never terminated mid-generation.
"""

from __future__ import annotations

import atexit
import socket
import time
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Optional, Tuple, Union

import requests

from src.core.exceptions import EngineException
from src.engines.cuda_compatibility import classify_cuda_failure
from src.core.logging import logger
from src.engines.base_engine import BaseEngine


class BaseChatServerEngine(BaseEngine):
    """Subprocess + OpenAI-compat HTTP + SSE common pattern.

    Subclasses must override the class attributes marked `# must override` and
    implement the four `@abstractmethod` hooks. The rest of the lifecycle (port
    pick, probe, atexit, SSE parsing, idle cleanup) is provided here.
    """

    # ====================== Overridable class attributes ======================
    _port_range_start: ClassVar[int] = 0  # must override
    _port_range_count: ClassVar[int] = 100
    _server_name: ClassVar[str] = ""  # must override — used in error messages
    _server_alias_prefix: ClassVar[str] = "erudi-"
    _tokenizer_provider: ClassVar[str] = ""  # must override
    _probe_timeout_s: ClassVar[float] = 120.0
    _probe_poll_interval_s: ClassVar[float] = 0.4

    # ====================== Per-class state ======================
    # Stored separately from `_model` so we can unregister the atexit handler
    # before swapping models — see `_stop_server_if_running`.
    _atexit_handler: ClassVar[Optional[Callable[[], None]]] = None

    # ====================== Abstract hooks ======================
    @classmethod
    @abstractmethod
    def _spawn_child(cls, *, model_path: Path, alias: str, port: int, **ctx: Any) -> Dict[str, Any]:
        """Spawn the OpenAI-compat child and return the handle dict.

        The handle MUST contain at least: `pid`, `proc`, `port`, `base_url`,
        `alias`, `model_path`. Extra keys (e.g., `threads`, `gpu_layers`) are
        allowed and preserved.

        Implementations may receive subclass-specific context via `**ctx`
        (populated by `_prepare_spawn_context`). For example, CUDA injects
        `gpu_layers` here.
        """

    @classmethod
    @abstractmethod
    def _terminate_process(cls, proc: Any) -> None:
        """Idempotently terminate the child process.

        API differs by spawn type: `mp.Process` for MLX (`terminate`, `kill`,
        `join`); `subprocess.Popen` for llama-cpp engines (`send_signal`,
        `terminate`, `kill`, `wait`). Must accept `None` as a no-op.
        """

    @classmethod
    @abstractmethod
    def _proc_is_alive(cls, proc: Any) -> bool:
        """Whether the spawned child is still alive.

        `mp.Process.is_alive()` vs `subprocess.Popen.poll() is None`. Used by
        `_probe_ready` to catch early subprocess crashes.
        """

    @classmethod
    def _read_child_output(cls, proc: Any) -> str:
        """Best-effort tail of what the child printed before it died.

        Default: nothing to show. MLX spawns an `mp.Process`, which has no
        output pipe to read. `BaseLlamaCppEngine` overrides this to return the
        tail its drainer collected (#360, #361).
        """
        return "No child output is captured for this engine."

    @classmethod
    def _child_exit_code(cls, proc: Any) -> Optional[int]:
        """The child's exit status, or None while it runs or when unknown.

        `subprocess.Popen` exposes `returncode`, `multiprocessing.Process`
        exposes `exitcode`; both are read so the record of a death names the
        code whichever child type the subclass spawns.
        """
        for attr in ("returncode", "exitcode"):
            code = getattr(proc, attr, None)
            if isinstance(code, int) and not isinstance(code, bool):
                return code
        return None

    @classmethod
    def _describe_child(cls, proc: Any, port: Any = None) -> str:
        """`pid 1234, port 27200, exit code 139` -- what identifies a dead child."""
        parts = []
        pid = getattr(proc, "pid", None)
        if isinstance(pid, int):
            parts.append(f"pid {pid}")
        if port is not None:
            parts.append(f"port {port}")
        code = cls._child_exit_code(proc)
        parts.append(f"exit code {code}" if code is not None else "exit code unknown")
        return ", ".join(parts)

    @classmethod
    def child_crash_report(cls) -> Optional[str]:
        """Describe the loaded child if it is dead; None when it runs or when
        nothing is loaded.

        The agent layer calls this when a stream fails: a generation that
        breaks because llama-server aborted mid-request surfaces as an HTTP
        connection error, and the exit code and the child's last lines are
        the only diagnostic there is. The engine does not notice the death by
        itself; the next request for the same model does (see
        `_should_not_reload_model`).
        """
        proc = cls._model.get("proc") if isinstance(cls._model, dict) else None
        if proc is None or cls._proc_is_alive(proc):
            return None
        port = cls._model.get("port") if isinstance(cls._model, dict) else None
        return (
            f"{cls._server_name} child is dead ({cls._describe_child(proc, port)}). "
            f"{cls._read_child_output(proc)}"
        )

    @classmethod
    @abstractmethod
    def _resolve_model_artifact(cls, llm_local_path: Union[str, Path]) -> Path:
        """Resolve the artifact handed to `_spawn_child`.

        MLX returns a directory containing weights + tokenizer; llama-cpp
        engines return a single `.gguf` file (picked by quant-priority
        heuristic in `_select_gguf`).
        """

    @classmethod
    def validate_local_artifact(cls, llm_local_path: Union[str, Path]) -> None:
        """Integrity gate for the on-disk artifact (#88).

        Raise :class:`EngineException` with an explicit, user-facing message if
        the model's ESSENTIAL files are missing, empty, or corrupt; return None
        when the artifact looks loadable. Concrete engine families override this
        to check exactly what they consume (a valid GGUF container for
        llama.cpp; a config + tokenizer + weights snapshot for MLX).

        Called BOTH after a download completes (before the model can become
        selectable) and once more just before spawn (so a broken local model
        yields a precise load error instead of an opaque child crash). The
        default is a no-op so a bare ``BaseChatServerEngine`` subclass (tests)
        needs no override.
        """
        return None

    @staticmethod
    def _payload_model_value(handle: Dict[str, Any]) -> str:
        """Value to send as the `"model"` field in `/v1/chat/completions`.

        Default: use the handle's `alias` (llama-server convention). MLX
        overrides to return the real preloaded model path, since mlx_vlm.server
        resolves every request's model via `get_cached_model(request.model)`.
        """
        return handle["alias"]

    @classmethod
    def _prepare_spawn_context(cls) -> Dict[str, Any]:
        """Build per-spawn context passed to `_spawn_child` as `**ctx`.

        Default: empty. CUDA overrides to inject `{"gpu_layers": ...}` from
        NVML at spawn time.
        """
        return {}

    @classmethod
    def _translate_payload_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Translate engine-agnostic kwarg names (HF/transformers vocabulary)
        into the names the upstream server expects.

        Default: identity. LlamaCpp engines override to translate
        `repetition_penalty → repeat_penalty` and `repetition_context_size →
        repeat_last_n`; MLX keeps the HF names (mlx_vlm.server reads them
        natively) but stamps a fresh random `seed` per request, since
        mlx_vlm.server otherwise replays a fixed default seed.
        """
        return kwargs

    # ====================== Shared concrete methods ======================
    @classmethod
    def _assert_requests(cls) -> None:
        """Confirm the `requests` library is importable."""
        try:
            import requests  # noqa: F401
        except ImportError as e:
            raise EngineException(
                message=f"`requests` is required to talk to {cls._server_name}",
                trace=str(e),
            )

    @classmethod
    def _assert_required_attrs(cls) -> None:
        """Raise if a subclass forgot to override required class attrs."""
        missing = []
        if cls._port_range_start == 0:
            missing.append("_port_range_start")
        if not cls._server_name:
            missing.append("_server_name")
        if not cls._tokenizer_provider:
            missing.append("_tokenizer_provider")
        if missing:
            raise EngineException(
                message=f"{cls.__name__} did not override required class attrs: {missing}",
            )

    @classmethod
    def _pick_free_port(cls) -> int:
        """Find a free TCP port in `[start, start+count)`.

        TOCTOU caveat: the socket is closed before we hand the port to the
        child, so a racing process could grab it between. `_probe_ready`
        surfaces a hint in the error message when this happens.
        """
        cls._assert_required_attrs()
        for offset in range(cls._port_range_count):
            port = cls._port_range_start + offset
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(("127.0.0.1", port))
                logger.info(f"[{cls.__name__}] Picked free port {port} for {cls._server_name}")
                return port
            except OSError:
                continue
        raise EngineException(
            message=(
                f"No free port for {cls._server_name} in range "
                f"{cls._port_range_start}-{cls._port_range_start + cls._port_range_count - 1}"
            ),
        )

    @classmethod
    def _wait_port_closed(cls, port: int, timeout_s: float = 3.0) -> None:
        """Block until the OS releases `port` (TIME_WAIT) or `timeout_s` elapses."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    s.bind(("127.0.0.1", port))
                return  # port free
            except OSError:
                time.sleep(0.1)
        # If we time out, downstream port pick will skip this port anyway.

    @classmethod
    def _probe_ready(
        cls,
        base_url: str,
        proc: Any = None,
        model_field: str = "default_model",
        api_key: Optional[str] = None,
    ) -> None:
        """Two-stage readiness probe.

        Stage 1: poll `GET /health` until 200 OK. Readiness contract:
        - a non-200 (e.g. 503) means "still loading" → keep polling;
        - 200 means ready (mlx-vlm returns `{"status": "healthy", ...}` once the
          preloaded model is up; llama-server returns its own 200 body).
        Only the status code is read, so the exact body is irrelevant.
        If `proc` is provided, `_proc_is_alive(proc)` is checked each iteration
        so a child that crashes early (e.g., DLL_NOT_FOUND on Windows) is
        reported immediately rather than timing out at `_probe_timeout_s`.

        Stage 2: send a single `POST /v1/chat/completions` with `max_tokens=1`.
        This validates the chat template + tokenizer + sampling chain — a
        broken model (missing chat template) returns 200 on `/health` but 400
        on the first chat call. `model_field` is the value each subclass sends
        for real inference (`_start_server` computes it via
        `_payload_model_value(handle)`: llama-cpp returns the alias, MLX returns
        the preloaded model path mlx_vlm.server resolves with `get_cached_model`).

        `api_key` is the child's own credential, minted at spawn: llama-server
        is started with `--api-key` so nothing else on the loopback interface
        can drive it, and the probe would otherwise get a 401 from a perfectly
        healthy server. `/health` stays public in llama-server, so the header is
        redundant on stage 1, but sending it uniformly keeps the two calls
        symmetric. MLX has no key (mlx_vlm.server has no such option), passes
        None, and sends no header at all.
        """
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        port = base_url.rsplit(":", 1)[-1]
        probe_start = time.monotonic()
        deadline = probe_start + cls._probe_timeout_s
        last_status: Optional[int] = None
        last_err: Optional[Exception] = None
        while time.monotonic() < deadline:
            if proc is not None and not cls._proc_is_alive(proc):
                # The tail is the only diagnostic there is, so it is shown to
                # the user AND classified: a CUDA failure the app can propose a
                # remedy for gets a code, everything else stays generic. The
                # exit code, pid and port identify the process in the log.
                child_output = cls._read_child_output(proc)
                raise EngineException(
                    message=(
                        f"{cls._server_name} child exited before becoming ready "
                        f"(early crash; {cls._describe_child(proc, port)}, "
                        f"model {model_field}). {child_output}"
                    ),
                    trace=child_output,
                    engine_code=classify_cuda_failure(child_output),
                )
            try:
                resp = requests.get(f"{base_url}/health", timeout=2.0, headers=headers)
                last_status = resp.status_code
                if resp.status_code == 200:
                    health_ms = (time.monotonic() - probe_start) * 1000
                    logger.info(
                        f"[{cls.__name__}] {cls._server_name} /health ok "
                        f"after {health_ms:.0f}ms"
                    )
                    break
                # 503 (or anything else) → keep polling.
            except requests.RequestException as e:
                last_err = e
            time.sleep(cls._probe_poll_interval_s)
        else:
            # The child is alive but never answered: its output is the only
            # clue to what it was doing (a model still loading, a bind that
            # went to the wrong interface), and the caller kills it next.
            raise EngineException(
                message=(
                    f"{cls._server_name} did not become ready within "
                    f"{cls._probe_timeout_s:.0f}s (last status: {last_status}, "
                    f"last err: {last_err}; {cls._describe_child(proc, port)}, "
                    f"model {model_field}). If another process bound the port "
                    f"between pick and spawn, the request may be hitting the wrong "
                    f"server — check `lsof -i :{port}`."
                ),
                trace=cls._read_child_output(proc) if proc is not None else None,
            )
        # Stage 2 — cheap chat-completions ping (1 token). `model_field` is the
        # real per-subclass model identifier computed by `_start_server`.
        ping_start = time.monotonic()
        try:
            resp = requests.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": model_field,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "stream": False,
                },
                timeout=30.0,
                headers=headers,
            )
        except requests.RequestException as e:
            raise EngineException(
                message=(
                    f"{cls._server_name} chat-completions probe failed "
                    f"({cls._describe_child(proc, port)}, model {model_field}): {e}"
                ),
                trace=cls._read_child_output(proc) if proc is not None else str(e),
            )
        if resp.status_code >= 400:
            raise EngineException(
                message=(
                    f"{cls._server_name} chat-completions probe returned "
                    f"HTTP {resp.status_code} ({cls._describe_child(proc, port)}, "
                    f"model {model_field}): {resp.text[:200]}"
                ),
                trace=cls._read_child_output(proc) if proc is not None else None,
            )
        ping_ms = (time.monotonic() - ping_start) * 1000
        logger.info(f"[{cls.__name__}] {cls._server_name} chat-ping ok after {ping_ms:.0f}ms")

    @classmethod
    def _stop_server_if_running(cls) -> None:
        """Tear down the cached subprocess (if any) and unregister its atexit handler."""
        model = cls._model
        if isinstance(model, dict):
            proc = model.get("proc")
            port = model.get("port")
            if proc is not None:
                logger.info(f"[{cls.__name__}] Stopping {cls._server_name} on port {port}")
                cls._terminate_process(proc)
                if port is not None:
                    cls._wait_port_closed(port)
        # Always clear the atexit handler — even if the proc was already dead,
        # the registered lambda still holds a reference to it.
        if cls._atexit_handler is not None:
            try:
                atexit.unregister(cls._atexit_handler)
            except Exception:
                pass
            cls._atexit_handler = None

    @classmethod
    def cleanup(cls) -> None:
        """Terminate the child and reset cached engine state.

        Called by `BaseEngine._cleanup_monitor` after 300s of idle, or
        explicitly when switching models.
        """
        cls._stop_server_if_running()
        return super().cleanup()

    @classmethod
    def _should_not_reload_model(cls, llm_id: str) -> bool:
        """Extend the base cache check with child-process liveness.

        `BaseEngine`'s version only compares `llm_id` against cached state --
        it never notices when the child behind that state has died. A
        `llama-server` that crashes mid-generation (native abort, OOM, driver
        reset) leaves `_model`/`_model_id` untouched, so every following
        request for the same `llm_id` would keep reusing the dead handle and
        fail with a connection error forever, until a *different* model is
        loaded or the app is restarted. Reproduced on a real packaged build
        during QA: llama-server hard-crashed (0xc0000409 in ucrtbase.dll)
        while emitting a tool call, and the conversation stayed broken for
        every subsequent turn.
        """
        if not super()._should_not_reload_model(llm_id):
            return False
        proc = cls._model.get("proc") if isinstance(cls._model, dict) else None
        if not cls._proc_is_alive(proc):
            port = cls._model.get("port") if isinstance(cls._model, dict) else None
            logger.warning(
                f"[{cls.__name__}] Cached child for model {llm_id} is no "
                f"longer running ({cls._describe_child(proc, port)}); forcing a "
                f"respawn instead of reusing a dead handle. "
                f"{cls._read_child_output(proc)}"
            )
            return False
        return True

    # ====================== Template methods ======================
    @classmethod
    def _start_server(cls, *, model_path: Path, alias: str, port: int) -> Dict[str, Any]:
        """Spawn the child, probe, and register a stable atexit handler.

        The atexit handler is stored on `cls._atexit_handler` so a subsequent
        `_stop_server_if_running` (during a model swap) can unregister it.
        """
        cls._assert_required_attrs()
        cls._assert_requests()
        ctx = cls._prepare_spawn_context()
        handle = cls._spawn_child(model_path=model_path, alias=alias, port=port, **ctx)
        try:
            cls._probe_ready(
                handle["base_url"],
                proc=handle.get("proc"),
                model_field=cls._payload_model_value(handle),
                api_key=handle.get("api_key"),
            )
        except Exception:
            cls._terminate_process(handle.get("proc"))
            raise
        proc = handle.get("proc")

        def _atexit_handler() -> None:
            cls._terminate_process(proc)

        atexit.register(_atexit_handler)
        cls._atexit_handler = _atexit_handler
        return handle

    @classmethod
    def get_model_and_tokenizer(
        cls,
        llm_id: str,
        llm_local_path: Union[str, Path],
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Spawn (or reuse) the child server for `llm_id`.

        Singleton semantics: a second call with the same `llm_id` reuses the
        existing subprocess; a call with a different `llm_id` terminates the
        old child (including unregistering its atexit handler) before spawning.
        """
        logger.info(
            f"[{cls.__name__}] Loading model '{llm_id}' from {llm_local_path} "
            f"via {cls._server_name}..."
        )
        # Serialized by the caller's generation_guard (same asyncio lock as the
        # idle-cleanup tick), so no threading lock is needed around the swap.
        if cls._should_not_reload_model(llm_id):
            return cls._return_cached_model_and_tokenizer()
        cls._assert_requests()
        # Pre-spawn integrity gate (#88): surface a precise "the files are
        # incomplete/corrupt" load error here, before the expensive spawn +
        # probe would fail with an opaque child crash.
        cls.validate_local_artifact(llm_local_path)
        resolved = cls._resolve_model_artifact(llm_local_path)
        # Stop previous child (also unregisters its stale atexit handler).
        cls._stop_server_if_running()
        alias = f"{cls._server_alias_prefix}{llm_id}"
        port = cls._pick_free_port()
        handle = cls._start_server(model_path=resolved, alias=alias, port=port)
        cls._model = handle
        cls._tokenizer = {"type": "remote", "provider": cls._tokenizer_provider}
        cls._model_id = llm_id
        cls._last_used = datetime.now()
        logger.info(f"[{cls.__name__}] Model loaded on {handle['base_url']} alias={alias}")
        return cls._model, cls._tokenizer
