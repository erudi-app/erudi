"""MLX engine for Apple Silicon inference via `mlx_vlm.server` subprocess.

Inference is delegated to an out-of-process `mlx_vlm.server` HTTP server
(OpenAI-compatible), spawned by `multiprocessing.Process` and reached over
loopback HTTP — exactly aligned with the CPU/CUDA pattern that wraps the
`llama-server` binary. mlx-vlm is a superset of mlx-lm (it depends on it): it
serves plain text models, carries a working tool-calling parser (no
mlx_lm.server EOS-flush drop, so agentic tools fire), and accepts image input.
Apple Silicon hardware detection remains in-process: it doesn't need the
server and would only inflate cold-start time if it did.

Architecture:
    MLX_Engine (singleton)
    ┌───────────────────────────────────────────────────────────────┐
    │ get_model_and_tokenizer(llm_id, path)                         │
    │  1. Pick free TCP port (27300+)                              │
    │  2. Spawn child: mp.Process(target=run_mlx_vlm_server)        │
    │  3. Poll GET /health until 200 (≤120s)                        │
    │  4. atexit.register(terminate)                                │
    │  5. Cache handle {pid,proc,port,base_url,alias,model_path}    │
    └───────────────────────────────────────────────────────────────┘
                                  ↓
    ┌───────────────────────────────────────────────────────────────┐
    │ token streaming lives in the agent layer, not the engine:     │
    │   AgentRunner → ChatOpenAI(base_url) → POST /v1/chat/...      │
    │   ChatOpenAI yields delta.content, ignores delta.reasoning    │
    └───────────────────────────────────────────────────────────────┘
                                  ↓
    ┌───────────────────────────────────────────────────────────────┐
    │ cleanup() — override                                          │
    │  └─> SIGTERM child → join(5s) → SIGKILL if needed             │
    │      → wait_port_closed → super().cleanup()                   │
    └───────────────────────────────────────────────────────────────┘

Where the child's output goes:
    An `mp.Process` has no output pipe, so the child redirects its own stdout
    and stderr into `logs/mlx-child-<port>.log` (`mlx_child_log`, wired in
    `_mlx_vlm_server_runner`). `_read_child_output` quotes the tail of that
    file in every crash report and probe timeout, exactly where the llama-cpp
    engines quote their drainer.

Why multiprocessing instead of subprocess.Popen([sys.executable, "-m", ...])?
    In a PyInstaller frozen build, `sys.executable` is the launcher binary,
    not a Python interpreter, so the `-m` flag is a no-op. `mp.spawn`
    (already configured in `backend/run.py:143-160` via `mp.freeze_support()`
    + `set_start_method("spawn", force=True)`) re-executes the binary in
    child mode and reconstitutes the import graph — the same `run_mlx_vlm_server`
    target works in dev (real Python) and in prod (frozen).

Why the `<|channel>thought ... <channel|>` manual filter is gone:
    Reasoning stays INLINE in `delta.content` as raw `<think>...</think>` (#90):
    the child spawns with `--enable-thinking` (thinking on by default) and
    neutralizes mlx-vlm's server-side reasoning split before the server starts
    (`_patch_inline_thinking` in `_mlx_vlm_server_runner.py`), because the
    dedicated `delta.reasoning` field it would otherwise emit is silently
    dropped by ChatOpenAI. The runner's single streaming ThinkSplitter then
    separates thinking from answer — identical to llama-server with
    `--reasoning-format none` on the CPU/CUDA path.

Why the `_MLX_EXECUTOR` thread bottleneck is gone:
    Generation now runs in a separate OS process; the GPU stream is
    initialised inside that child, fully isolated from the FastAPI parent.
    The Stream(gpu, 0) crash that motivated the persistent thread executor
    (commits cefdc7a, 40fb55e) is structurally impossible from the parent.

Model format:
    The engine only ever loads pre-built MLX repos (HF library tag ``mlx``,
    see ``FORMAT_TAG``): the catalog and the HF search are filtered on that tag
    and a by-link download is refused up front when the repo lacks it. There is
    no local HF -> MLX conversion (#408).

Example:
    ::

        from src.engines.mlx_engine import MLX_Engine

        model, tokenizer = MLX_Engine.get_model_and_tokenizer(
            llm_id="mistral-7b",
            llm_local_path="/path/to/mlx/model"
        )
        # Token streaming is driven by the agent layer (ChatOpenAI(base_url=...)),
        # not the engine; ``model["base_url"]`` is what it connects to.
        MLX_Engine.cleanup()

Warning:
    Only use on Apple Silicon. On other platforms, BaseEngine.get_engine()
    will select CUDA_Engine or CPU_Engine instead.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import platform
import secrets
import subprocess
import time  # used by hardware warm-up loop
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.engines.base_chat_server_engine import BaseChatServerEngine
from src.engines._mlx_vlm_server_runner import run_mlx_vlm_server
from src.engines import mlx_child_log as child_log
from src.core.logging import logger
from src.core.exceptions import (
    EngineException,
    FileSystemException,
    HardwareException,
)


class MLX_Engine(BaseChatServerEngine):
    """Singleton Engine for MLX models and tokenizers runtimes.
    Built for Apple Silicon Backends.
    """

    # MLX models on HF carry the "mlx" library tag; the catalog is built by
    # searching filter="mlx" (any author), so no hand-maintained mapping is needed.
    FORMAT_TAG = "mlx"

    # Stored links that download but crash at load. Empty since the 0.6.13 bump:
    # gemma-4 E2B (the 0.6.2 140-weight ValueError) was re-probed on real
    # mlx-vlm 0.6.13 during the #273 hardware pass — loads via the native
    # gemma4 module and generates cleanly, so its entry was removed.
    KNOWN_BROKEN = frozenset()

    # Where the spawn stored the child's own log file, on the process object.
    # The llama-cpp engines keep their drainer the same way, and for the same
    # reason: the lifetime is the child's.
    _CHILD_LOG_ATTR = "erudi_child_log_path"

    # How much of the tail a crash message quotes. mlx-vlm's startup banner and
    # per-request lines are long; the reason it died is in the last lines.
    _CHILD_OUTPUT_TAIL_CHARS = child_log.DEFAULT_TAIL_CHARS

    # ======================= SUBPROCESS HTTP SERVER (mlx_vlm.server) =======================
    #
    # Inference goes through a subprocess `mlx_vlm.server` (OpenAI-compatible HTTP),
    # spawned via `multiprocessing.Process(target=run_mlx_vlm_server, args=([argv],))`.
    # mlx-vlm is a superset of mlx-lm: it serves plain text models through the same
    # endpoint, carries its own tool-calling parser (no mlx_lm.server EOS-flush drop,
    # so agentic tool use works on Apple Silicon), and accepts image input. The
    # shared lifecycle (port pick, /health + chat-ping probe, SSE parsing, atexit,
    # idle cleanup) lives in `BaseChatServerEngine`. Only the hooks below are
    # MLX-specific.
    #
    # Why `multiprocessing` and not `subprocess.Popen([sys.executable, "-m", ...])`:
    # in a PyInstaller frozen build, `sys.executable` is the launcher binary, not
    # a Python interpreter. `mp.spawn` (configured in `backend/run.py`) re-executes
    # the binary in child mode and reconstitutes the import graph, so the same
    # `run_mlx_vlm_server` target works in dev (real Python) and in prod (frozen).

    # --- BaseChatServerEngine config overrides ---
    # MLX binds the top slice of Erudi's canonical 271xx–273xx block: 27300–27399,
    # collision-free against llama.cpp (27200–27299) and the backend (27182–27199).
    # (Was 9080; see backend/run.py for the 271xx rationale — digits of e, below
    # every OS ephemeral range, IANA-unassigned.)
    _port_range_start = 27300
    _server_name = "mlx_vlm.server"
    _tokenizer_provider = "mlx-vlm-server"

    # mlx_vlm masks the request seed to 32 bits (`generation.py:_position_seed`);
    # staying below 2**31 also keeps it a valid signed int32 for any consumer.
    _SEED_UPPER_BOUND = 2**31

    @classmethod
    def _translate_payload_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """mlx_vlm.server accepts HF/transformers kwarg names natively
        (repetition_penalty, repetition_context_size, top_k, enable_thinking,
        ...), so the names pass through unchanged. What MLX adds is a fresh
        random ``seed`` per request: mlx_vlm.server (0.6.x,
        ``server/generation.py``) seeds its sampler from ``DEFAULT_SEED``
        whenever the request carries none, so without it every generation
        replayed byte-for-byte and the creativity controls (temperature /
        top_p / top_k) had no run-to-run effect on Apple Silicon. Greedy
        decoding is untouched: at temperature 0 the sampler is an argmax
        whatever the seed. llama-server samples randomly by default and gets
        no seed (see ``BaseLlamaCppEngine``)."""
        out = dict(kwargs)
        if "seed" not in out:
            out["seed"] = secrets.randbelow(cls._SEED_UPPER_BOUND)
        return out

    @staticmethod
    def _payload_model_value(handle: Dict[str, Any]) -> str:
        """mlx_vlm.server resolves every request's `model` field through
        `get_cached_model(request.model)` (there is no `default_model`
        sentinel), so the chat payload must carry the real model path that was
        preloaded with `--model`. The same value is used by the chat-ping probe
        and by ChatOpenAI for real inference (model_factory)."""
        return handle["model_path"]

    @classmethod
    def _resolve_model_artifact(cls, llm_local_path: Union[str, Path]) -> Path:
        """MLX models are directories containing weights + tokenizer."""
        path = Path(llm_local_path).resolve()
        if not path.exists():
            raise FileSystemException(f"MLX model path not found: {path}")
        return path

    @classmethod
    def validate_local_artifact(cls, llm_local_path: Union[str, Path]) -> None:
        """Integrity gate for an MLX snapshot (#88).

        An MLX model is loadable iff its directory ships ``config.json`` + a
        tokenizer file + at least one weights file, each non-empty. Raises
        :class:`EngineException` with a curated, user-facing message on the
        first missing/empty essential. Backs both the download-completion gate
        and the pre-spawn load gate.
        """
        from src.engines import integrity

        integrity.validate_hf_snapshot(llm_local_path)

    @classmethod
    def _load_capability_tokenizer(cls, llm_local_path: Union[str, Path]):
        """Load the HF tokenizer from the MLX model directory (no weights, no server).

        Used by ``compute_supports_tools`` for static tool-calling detection.
        """
        from transformers import AutoTokenizer

        model_dir = cls._resolve_model_artifact(llm_local_path)
        # local_files_only=True (#164 pattern): everything needed lives in the
        # already-downloaded directory. Without it, from_pretrained reaches out to
        # the Hub to check for updates and can hang for minutes on a slow/blocked
        # connection, stalling the whole download finalization (#291).
        return AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=False, local_files_only=True
        )

    @classmethod
    def compute_wire_tools(cls, llm_local_path: Union[str, Path]) -> Optional[bool]:
        """Verified tool-call wire capability on the mlx-vlm server (#298).

        Runs mlx-vlm's OWN parser inference (``_infer_tool_parser``) on the
        model's chat template: the server picks its tool parser from exactly
        that function at load time, so this executes the same code the server
        executes — the verdict is exact by construction for the pinned mlx-vlm
        (0.6.13). A template matching no parser entry means the server streams
        the model's tool-call output as raw text (#295), hence wire False.

        Tokenizer-level only (``_load_capability_tokenizer``: local files, no
        weights, no server). Any failure to load mlx-vlm or the tokenizer
        returns ``None`` (unverified) rather than a wrong verdict.
        """
        try:
            from mlx_vlm.tool_parsers import _infer_tool_parser
        except Exception:
            logger.warning(
                f"[MLX_Engine] wire tool detection unavailable (mlx_vlm import failed) "
                f"for {llm_local_path}"
            )
            return None
        try:
            tokenizer = cls._load_capability_tokenizer(llm_local_path)
        except Exception:
            logger.warning(
                f"[MLX_Engine] wire tool detection: could not load a tokenizer "
                f"for {llm_local_path}"
            )
            return None
        try:
            template = getattr(tokenizer, "chat_template", None)
            parser = _infer_tool_parser(template)
        except Exception:
            logger.warning(
                f"[MLX_Engine] wire tool detection: parser inference failed "
                f"for {llm_local_path}"
            )
            return None
        if parser is None:
            logger.info(
                f"[MLX_Engine] wire tools NOT verified for {llm_local_path}: "
                f"chat template matches no mlx-vlm tool parser"
            )
            return False
        logger.info(
            f"[MLX_Engine] wire tools verified for {llm_local_path}: "
            f"mlx-vlm inferred parser {parser}"
        )
        return True

    @classmethod
    def model_supports_vision(cls, llm_local_path: Union[str, Path]) -> Optional[bool]:
        """An MLX model is vision-capable iff its ``config.json`` declares vision.

        Reads the directory's ``config.json`` only (no weights, no server) and
        defers the decision to ``config_declares_vision``. Unreadable/absent
        config -> ``None`` (permissive).
        """
        import json

        from src.engines.vision_capability import config_declares_vision

        try:
            model_dir = cls._resolve_model_artifact(llm_local_path)
            config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
            return config_declares_vision(config)
        except Exception as exc:
            # Permissive fallback (the model stays usable, vision unverified):
            # a WARNING, with what went wrong -- an absent config and a corrupt
            # one are the same verdict here but not the same defect.
            logger.warning(
                f"[MLX_Engine] vision detection failed for {llm_local_path}: "
                f"{type(exc).__name__}: {exc}"
            )
            return None

    @classmethod
    def _spawn_child(
        cls,
        *,
        model_path: Path,
        alias: str,
        port: int,
        **ctx: Any,
    ) -> Dict[str, Any]:
        """Spawn `mlx_vlm.server` as an mp.Process. Returns the handle dict.

        The child is started with a per-spawn `--api-key` (mlx-vlm's own flag,
        exported in the child as `MLX_VLM_SERVER_API_KEY`) and the key travels
        in the handle under `"api_key"`, where `_probe_ready` and the
        ChatOpenAI factory read it. The argv goes to the child by pickle, not
        on a command line, so the key is not visible in `ps` arguments.
        """
        # Close the loopback port to everything but us. Spawned without
        # `--api-key`, mlx_vlm.server authenticates NOTHING: every endpoint
        # answers any caller that can reach 127.0.0.1 -- another local process,
        # or a web page the user has open, since a browser can POST across
        # origins to a loopback port -- and `/v1/chat/completions` lets it run
        # its own inference on the user's machine. The key is minted per spawn
        # so a disclosure dies with the child. mlx-vlm applies the key to every
        # route it registers, `/health` included, so `_probe_ready` sends it
        # from the handle on both probe stages.
        api_key = secrets.token_urlsafe(32)
        # The child captures its own output into this file (an mp.Process has
        # no pipe to drain); the path is resolved HERE because a frozen child
        # re-executes the binary with uninitialized runtime paths. Best
        # effort: a log directory we cannot write costs the tail, not the
        # model.
        try:
            log_path: Optional[str] = str(child_log.prepare_child_log(port))
        except OSError as exc:
            logger.warning(
                f"[MLX_Engine] Cannot capture the {cls._server_name} child's output "
                f"on port {port}: {type(exc).__name__}: {exc}"
            )
            log_path = None
        argv = [
            "mlx_vlm.server",
            "--model",
            str(model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "INFO",
            # Thinking on by default for requests that don't set enable_thinking
            # (Erudi's runner never does). Without it, mlx-vlm 0.6.13 renders the
            # chat template with enable_thinking=False and a thinking model
            # (e.g. Qwen3) answers directly — no reasoning ever exists (#90).
            # Safe for non-thinking models ONLY because the child neutralizes
            # the server-side thinking split (`_patch_inline_thinking` in
            # _mlx_vlm_server_runner.py): unpatched, a prompt whose template
            # opens a thinking block starts the stream in reasoning mode, and
            # any emitted <think> marker routes text to the delta.reasoning
            # channel that ChatOpenAI drops. With the patch, the flag's only
            # remaining effect is the template kwarg, which non-thinking
            # templates ignore (hardware-verified on Qwen2.5-0.5B on 0.6.2:
            # byte-identical prompts and answers; re-check in the 0.6.13
            # hardware pass).
            "--enable-thinking",
            "--api-key",
            api_key,
        ]
        proc = mp.Process(target=run_mlx_vlm_server, args=(argv, log_path), daemon=False)
        try:
            proc.start()
        except Exception as exc:
            # Spawning can fail before the child exists at all: a process or
            # file-descriptor limit, a sandbox that refuses the fork, a
            # freeze whose bootstrap cannot re-exec the binary. Named here,
            # because a bare OSError from start() says nothing about what was
            # being spawned -- the same reason the llama-server Popen is
            # wrapped in base_llama_cpp_engine.py.
            raise EngineException(
                message=(
                    f"Could not start the {cls._server_name} child for model "
                    f"{model_path} on port {port}: {exc}"
                ),
                trace=f"{type(exc).__name__}: {exc}",
            ) from exc
        # Carried on the process object, exactly like the llama-cpp drainer:
        # its lifetime is then the child's, with no registry to clean up and
        # no chance of a recycled pid handing out another child's output.
        if log_path:
            setattr(proc, cls._CHILD_LOG_ATTR, log_path)
        logger.info(
            f"[MLX_Engine] Spawned mlx_vlm.server child: pid={proc.pid}, "
            f"port={port}, model={model_path}, output={log_path or 'not captured'}"
        )
        return {
            "pid": proc.pid,
            "proc": proc,
            "port": port,
            "base_url": f"http://127.0.0.1:{port}",
            "alias": alias,
            "model_path": str(model_path),
            # Every caller that talks to this child reaches it through the
            # handle: the readiness probe and the ChatOpenAI inference client
            # both read the key from here. Never log the handle wholesale.
            "api_key": api_key,
        }

    @classmethod
    def _terminate_process(cls, proc) -> None:
        """Idempotently terminate an `mp.Process`, escalating to SIGKILL if needed.

        Accepts `None` (no-op) and `MagicMock`-like proxies for testability.
        Mirrors the bounded-time semantics of cpu_engine.py:226-240, but uses
        `mp.Process` API (`terminate` = SIGTERM, `kill` = SIGKILL).

        This is the ONLY place an mlx-vlm child's exit code is ever recorded,
        so the level says what happened: `INFO` for an orderly stop (a model
        swap, the idle reap), `WARNING` when the child had to be SIGKILLed or
        had already died on its own with a nonzero code -- the two shapes a
        maintainer wants to see on the Diagnostics page.
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
                f"[MLX_Engine] Child terminated: {outcome}, "
                f"exitcode={getattr(proc, 'exitcode', None)}"
            )
            # An orderly stop leaves nothing behind. A child that died on its
            # own keeps its file: that output is the only account of the
            # death, and the crash report is read after this call.
            path = cls._child_log_path_of(proc)
            if path is not None and not degraded:
                child_log.discard_child_log(path)
        except Exception:
            # Best-effort cleanup; never let teardown errors mask the real
            # failure that triggered termination in the first place.
            pass

    @classmethod
    def _child_log_path_of(cls, proc: Any) -> Optional[str]:
        """The file this child captured its output into, when there is one.

        Defensive about the type: `MagicMock` proxies answer any attribute
        with another mock, and a path that is not a string would turn a crash
        report into a second crash.
        """
        path = getattr(proc, cls._CHILD_LOG_ATTR, None) if proc is not None else None
        return path if isinstance(path, str) and path else None

    @classmethod
    def _read_child_output(cls, proc: Any) -> str:
        """Tail of the child's own stdout+stderr file (#361 for llama-server).

        The child redirects its descriptors into a per-spawn file at startup
        (`mlx_child_log`), because an `mp.Process` gives the parent no pipe to
        drain. Works whether or not the child has exited, so a probe timeout
        quotes what the server was doing and a crash quotes why it stopped.
        """
        path = cls._child_log_path_of(proc)
        if path is None:
            return "No child output was captured."
        tail = child_log.read_child_log_tail(path, max_chars=cls._CHILD_OUTPUT_TAIL_CHARS)
        if not tail:
            return "The child produced no output."
        return f"Child output (last {len(tail)} chars, from {path}):\n{tail}"

    @classmethod
    def _proc_is_alive(cls, proc: Any) -> bool:
        """Whether the spawned `mp.Process` is still running.

        Used by `BaseChatServerEngine._probe_ready` to detect early child
        crashes (otherwise the probe would time out at 120s with no hint
        about why the child went away).
        """
        if proc is None:
            return False
        try:
            return bool(proc.is_alive())
        except Exception:
            return False

    # ======================= HARDWARE DETECTION & EVALUATION =======================

    # Apple Silicon specifications database (official Apple specs)
    _APPLE_SILICON_SPECS = {
        "M1": {
            "gpu_cores": 8,
            "memory_bandwidth": 68.25,
            "neural_engine_tops": 11.0,
            "cpu_cores": {"performance": 4, "efficiency": 4},
            "max_memory": 16,
            "architecture": "5nm",
        },
        "M1 Pro": {
            "gpu_cores": 16,
            "memory_bandwidth": 200,
            "neural_engine_tops": 11.0,
            "cpu_cores": {"performance": 8, "efficiency": 2},
            "max_memory": 32,
            "architecture": "5nm",
        },
        "M1 Max": {
            "gpu_cores": 32,
            "memory_bandwidth": 400,
            "neural_engine_tops": 11.0,
            "cpu_cores": {"performance": 8, "efficiency": 2},
            "max_memory": 64,
            "architecture": "5nm",
        },
        "M1 Ultra": {
            "gpu_cores": 64,
            "memory_bandwidth": 800,
            "neural_engine_tops": 22.0,
            "cpu_cores": {"performance": 16, "efficiency": 4},
            "max_memory": 128,
            "architecture": "5nm",
        },
        "M2": {
            "gpu_cores": 10,
            "memory_bandwidth": 100,
            "neural_engine_tops": 15.8,
            "cpu_cores": {"performance": 4, "efficiency": 4},
            "max_memory": 24,
            "architecture": "5nm",
        },
        "M2 Pro": {
            "gpu_cores": 19,
            "memory_bandwidth": 200,
            "neural_engine_tops": 15.8,
            "cpu_cores": {"performance": 8, "efficiency": 4},
            "max_memory": 32,
            "architecture": "5nm",
        },
        "M2 Max": {
            "gpu_cores": 38,
            "memory_bandwidth": 400,
            "neural_engine_tops": 15.8,
            "cpu_cores": {"performance": 8, "efficiency": 4},
            "max_memory": 96,
            "architecture": "5nm",
        },
        "M2 Ultra": {
            "gpu_cores": 76,
            "memory_bandwidth": 800,
            "neural_engine_tops": 31.6,
            "cpu_cores": {"performance": 16, "efficiency": 8},
            "max_memory": 192,
            "architecture": "5nm",
        },
        "M3": {
            "gpu_cores": 10,
            "memory_bandwidth": 100,
            "neural_engine_tops": 18.0,
            "cpu_cores": {"performance": 4, "efficiency": 4},
            "max_memory": 24,
            "architecture": "3nm",
        },
        "M3 Pro": {
            "gpu_cores": 18,
            "memory_bandwidth": 150,
            "neural_engine_tops": 18.0,
            "cpu_cores": {"performance": 6, "efficiency": 6},
            "max_memory": 36,
            "architecture": "3nm",
        },
        "M3 Max": {
            "gpu_cores": 40,
            "memory_bandwidth": 400,
            "neural_engine_tops": 18.0,
            "cpu_cores": {"performance": 8, "efficiency": 4},
            "max_memory": 128,
            "architecture": "3nm",
        },
        "M4": {
            "gpu_cores": 10,
            "memory_bandwidth": 120,
            "neural_engine_tops": 38.0,
            "cpu_cores": {"performance": 4, "efficiency": 6},
            "max_memory": 32,
            "architecture": "3nm",
        },
        "M4 Pro": {
            "gpu_cores": 20,
            "memory_bandwidth": 273,
            "neural_engine_tops": 38.0,
            "cpu_cores": {"performance": 10, "efficiency": 4},
            "max_memory": 64,
            "architecture": "3nm",
        },
        "M4 Max": {
            "gpu_cores": 40,
            "memory_bandwidth": 546,
            "neural_engine_tops": 38.0,
            "cpu_cores": {"performance": 12, "efficiency": 4},
            "max_memory": 128,
            "architecture": "3nm",
        },
    }

    @classmethod
    def _detect_apple_silicon_chip(cls) -> Optional[str]:
        """Detect specific Apple Silicon chip model (M1, M2, M3, M4, etc.).

        Uses system_profiler command to identify the exact chip variant.

        Returns:
            Optional[str]: Chip model (e.g., "M3 Max") or None if not detected.

        Note:
            Internal method. Called by get_hardware_info().
        """
        try:
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                import json

                data = json.loads(result.stdout)
                hardware_data = data.get("SPHardwareDataType", [{}])[0]
                chip_name = hardware_data.get("chip_type", "")

                if chip_name:
                    for model_key in cls._APPLE_SILICON_SPECS.keys():
                        if model_key.replace(" ", "").lower() in chip_name.replace(" ", "").lower():
                            return model_key

            return None

        except Exception as e:
            logger.warning(f"Failed to detect Apple Silicon chip: {e}")
            return None

    @classmethod
    def _mps_available(cls) -> bool:
        """Check if Metal Performance Shaders (MPS) is available.

        Returns:
            bool: True if MPS backend is available in PyTorch.
        """
        # Import required modules for hardware detection
        try:
            import torch
        except ImportError as e:
            logger.warning(f"Optional hardware detection dependency missing: {e}")
            return False

        try:
            return torch.backends.mps.is_available()
        except Exception:
            return False

    @classmethod
    def get_hardware_info(cls) -> Dict[str, Any]:
        """Get comprehensive hardware information for Apple Silicon.

        Returns detailed hardware specifications including Apple chip model,
        unified memory, MPS availability, and Neural Engine specs.

        Returns:
            Dict containing hardware specifications following BaseEngine contract:
            {
                "system": {"platform": "Darwin", ...},
                "cpu": {"model": str, "is_apple_silicon": True, ...},
                "memory": {"total_memory_gb": float, "memory_type": "unified", ...},
                "gpu": {"gpu_name": str, "mlx_gpu_cores": int, "unified_memory": True, ...},
                "accelerator": {"neural_engine_tops": float, "architecture": str},
                "storage": {"total_gb": float, "available_gb": float},
                "backend_type": "mlx",
                "timestamp": float
            }

        Raises:
            HardwareException: If critical hardware detection fails.

        Note:
            Returns fallback values on non-critical failures rather than raising.

        Examples:
            >>> hw_info = MLX_Engine.get_hardware_info()
            >>> print(f"Chip: {hw_info['cpu']['model']}")
            >>> print(f"GPU Cores: {hw_info['gpu']['mlx_gpu_cores']}")
        """
        try:
            # Import required modules for hardware detection
            try:
                import psutil
                import cpuinfo
            except ImportError as e:
                logger.warning(f"Optional hardware detection dependency missing: {e}")

            # Detect chip model
            chip_model = cls._detect_apple_silicon_chip()

            # Get unified memory info
            vm = psutil.virtual_memory()
            total_memory_gb = vm.total / (1024**3)
            available_memory_gb = vm.available / (1024**3)
            memory_pressure = 1.0 - (vm.available / vm.total)

            # Get storage info
            disk = psutil.disk_usage(os.path.abspath(os.sep))
            disk_total_gb = disk.total / (1024**3)
            disk_available_gb = disk.free / (1024**3)
            disk_usage_pct = disk.percent

            # Get CPU info
            cpu_info_data = cpuinfo.get_cpu_info()
            cpu_model = cpu_info_data.get("brand_raw", "Apple Silicon CPU")
            total_cores = psutil.cpu_count(logical=False)
            logical_cores = psutil.cpu_count(logical=True)

            # Get chip specifications
            specs = cls._APPLE_SILICON_SPECS.get(chip_model, {}) if chip_model else {}
            gpu_cores = specs.get("gpu_cores", 0)
            memory_bandwidth = specs.get("memory_bandwidth", 0.0)
            neural_engine_tops = specs.get("neural_engine_tops", 0.0)
            architecture = specs.get("architecture", "Unknown")
            max_memory = specs.get("max_memory", 0)
            cpu_cores_breakdown = specs.get("cpu_cores", {})

            # Estimate TFLOPS (Apple doesn't publish official values)
            estimated_tflops = gpu_cores * 0.35 if gpu_cores else 0.0

            # Check MPS availability
            mps_supported = cls._mps_available()

            # Build hardware info dictionary
            hardware_info = {
                "system": {
                    "platform": platform.system(),
                    "platform_version": platform.version(),
                    "machine": platform.machine(),
                    "processor": platform.processor(),
                },
                "cpu": {
                    "model": cpu_model,
                    "architecture": platform.machine(),
                    "total_cores": total_cores,
                    "logical_cores": logical_cores,
                    "is_apple_silicon": True,
                    "performance_cores": cpu_cores_breakdown.get("performance"),
                    "efficiency_cores": cpu_cores_breakdown.get("efficiency"),
                },
                "memory": {
                    "total_memory_gb": round(total_memory_gb, 2),
                    "available_memory_gb": round(available_memory_gb, 2),
                    "memory_pressure": round(memory_pressure, 3),
                    "memory_type": "unified",
                },
                "gpu": {
                    "gpu_name": f"Apple {chip_model} GPU" if chip_model else "Apple GPU",
                    "mlx_gpu_cores": gpu_cores,
                    "memory_bandwidth_gbs": memory_bandwidth,
                    "mps_supported": mps_supported,
                    "unified_memory": True,
                    "estimated_tflops": round(estimated_tflops, 2),
                },
                "accelerator": {
                    "neural_engine_tops": neural_engine_tops,
                    "architecture": architecture,
                },
                "storage": {
                    "total_gb": round(disk_total_gb, 2),
                    "available_gb": round(disk_available_gb, 2),
                    "usage_percentage": round(disk_usage_pct, 2),
                },
                "backend_type": "mlx",
                "mlx_chip_model": chip_model,
                "timestamp": time.time(),
            }

            logger.info(
                f"MLX hardware detected: {chip_model}, {gpu_cores} GPU cores, {total_memory_gb:.1f}GB unified memory"
            )
            return hardware_info

        except Exception as e:
            logger.exception(f"MLX hardware detection failed: {e}")
            raise HardwareException("Failed to detect Apple Silicon hardware", trace=str(e))

    @classmethod
    def warm_up_accelerator(cls, duration_seconds: float = 1.0) -> bool:
        """Warm up Apple Silicon GPU using Metal Performance Shaders.

        Runs matrix operations on MPS device to bring GPU to optimal performance
        state before benchmarking or inference.

        Args:
            duration_seconds: How long to run warm-up operations (default: 1.0).

        Returns:
            bool: True if warm-up completed successfully, False otherwise.

        Note:
            Particularly important for Apple Silicon due to dynamic clock management.

        Examples:
            >>> success = MLX_Engine.warm_up_accelerator(1.5)
            >>> if success:
            ...     print("MPS device ready")
        """
        if not cls._mps_available():
            logger.warning("MPS not available, skipping GPU warm-up")
            return False

        try:
            # Import required modules for hardware detection
            try:
                import torch
            except ImportError as e:
                logger.warning(f"Optional hardware detection dependency missing: {e}")

            logger.info(f"Warming up MPS device for {duration_seconds}s...")
            start_time = time.time()

            # Create tensors on MPS device
            device = torch.device("mps")
            size = 4096

            while (time.time() - start_time) < duration_seconds:
                # Matrix multiplication on GPU
                a = torch.randn(size, size, device=device)
                b = torch.randn(size, size, device=device)
                c = torch.matmul(a, b)

                # Small sleep to prevent CPU overload
                time.sleep(0.05)

            logger.info("MPS warm-up completed successfully")
            return True

        except Exception as e:
            logger.exception(f"MPS warm-up failed: {e}")
            return False

    @classmethod
    def get_performance_evaluation(cls) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics for Apple Silicon.

        Evaluates hardware capabilities and returns performance scores for
        inference workloads. Scoring optimized for Apple Silicon unified memory
        architecture.

        Scoring methodology:
            - Inference: GPU compute (35%), memory bandwidth (30%), memory (20%),
              Neural Engine (10%), CPU (5%)

        Returns:
            Dict containing performance metrics and scores (0-100 scale):
            {
                "backend_type": "mlx",
                "gpu_name": str,
                "cpu_model": str,
                "total_memory_gb": float,
                "available_memory_gb": float,
                "memory_bandwidth_gbs": float,
                "disk_total_gb": float,
                "disk_available_gb": float,
                "estimated_tflops": float,
                "mlx_gpu_cores": int,
                "cpu_performance_units": float,
                "neural_engine_tops": float,
                "architecture": str,
                "global_inference_score": float,
                "global_inference_label": str,
                "gpu_score": float,
                "cpu_score": float,
                "memory_score": float,
                "unified_memory": True,
                "mps_available": bool,
                "system_platform": "Darwin",
                "mlx_chip_model": str,
                "performance_breakdown": dict
            }

        Raises:
            HardwareException: If evaluation fails critically.

        Examples:
            >>> eval_result = MLX_Engine.get_performance_evaluation()
            >>> print(f"Inference: {eval_result['global_inference_score']}/100")
            >>> print(f"Label: {eval_result['global_inference_label']}")
        """
        try:
            # Get base hardware info
            hw_info = cls.get_hardware_info()

            # Extract key metrics
            chip_model = hw_info.get("mlx_chip_model")
            gpu_cores = hw_info["gpu"]["mlx_gpu_cores"]
            estimated_tflops = hw_info["gpu"]["estimated_tflops"]
            memory_bandwidth = hw_info["gpu"]["memory_bandwidth_gbs"]
            neural_engine_tops = hw_info["accelerator"]["neural_engine_tops"]
            total_memory_gb = hw_info["memory"]["total_memory_gb"]
            available_memory_gb = hw_info["memory"]["available_memory_gb"]
            total_cores = hw_info["cpu"]["total_cores"]
            perf_cores = hw_info["cpu"].get("performance_cores", 4)

            # Calculate component scores (0-100 scale)

            # GPU/Accelerator score based on TFLOPS and GPU cores
            gpu_score = min(100, (estimated_tflops / 20.0) * 100)  # Normalize to 20 TFLOPS

            # Memory bandwidth score
            mem_bandwidth_score = min(
                100, (memory_bandwidth / 400.0) * 100
            )  # Normalize to 400 GB/s

            # Memory capacity score
            memory_capacity_score = min(100, (total_memory_gb / 64.0) * 100)  # Normalize to 64GB

            # Neural Engine score
            neural_score = min(100, (neural_engine_tops / 20.0) * 100)  # Normalize to 20 TOPS

            # CPU score based on performance cores
            cpu_performance_units = perf_cores * 2.5  # Weight performance cores higher
            cpu_score = min(100, (cpu_performance_units / 20.0) * 100)  # Normalize to 20 units

            # Calculate weighted inference score
            inference_score = (
                gpu_score * 0.35
                + mem_bandwidth_score * 0.30
                + memory_capacity_score * 0.20
                + neural_score * 0.10
                + cpu_score * 0.05
            )

            # Generate labels based on scores
            def get_label(score: float) -> str:
                if score >= 80:
                    return "Excellent"
                elif score >= 60:
                    return "Good"
                elif score >= 40:
                    return "Fair"
                elif score >= 20:
                    return "Poor"
                else:
                    return "Weak"

            inference_label = get_label(inference_score)

            # Build performance breakdown
            performance_breakdown = {
                "compute_score": round(gpu_score, 2),
                "memory_bandwidth_score": round(mem_bandwidth_score, 2),
                "memory_capacity_score": round(memory_capacity_score, 2),
                "neural_engine_score": round(neural_score, 2),
                "cpu_performance_score": round(cpu_score, 2),
                "weights_inference": {
                    "gpu_compute": 0.35,
                    "memory_bandwidth": 0.30,
                    "memory_capacity": 0.20,
                    "neural_engine": 0.10,
                    "cpu": 0.05,
                },
            }

            # Build complete evaluation result
            eval_result = {
                # Hardware identification
                "backend_type": "mlx",
                "gpu_name": hw_info["gpu"]["gpu_name"],
                "cpu_model": hw_info["cpu"]["model"],
                # Memory metrics
                "total_memory_gb": total_memory_gb,
                "available_memory_gb": available_memory_gb,
                "memory_bandwidth_gbs": memory_bandwidth,
                # Storage metrics
                "disk_total_gb": hw_info["storage"]["total_gb"],
                "disk_available_gb": hw_info["storage"]["available_gb"],
                # Compute metrics
                "estimated_tflops": estimated_tflops,
                "mlx_gpu_cores": gpu_cores,
                "cpu_performance_units": cpu_performance_units,
                # Apple Silicon specific
                "neural_engine_tops": neural_engine_tops,
                "architecture": hw_info["accelerator"]["architecture"],
                "mlx_chip_model": chip_model,
                # Performance scores (0-100)
                "global_inference_score": round(inference_score, 2),
                "global_inference_label": inference_label,
                "gpu_score": round(gpu_score, 2),
                "cpu_score": round(cpu_score, 2),
                "memory_score": round(memory_capacity_score, 2),
                # Technical details
                "unified_memory": True,
                "mps_available": hw_info["gpu"]["mps_supported"],
                "system_platform": hw_info["system"]["platform"],
                # Performance breakdown
                "performance_breakdown": performance_breakdown,
            }

            logger.info(
                f"Performance evaluation: Inference={inference_score:.1f}/100 ({inference_label})"
            )
            return eval_result

        except Exception as e:
            logger.exception(f"Performance evaluation failed: {e}")
            raise HardwareException("Failed to evaluate Apple Silicon performance", trace=str(e))

    @classmethod
    def get_flat_hardware_data(cls) -> Dict[str, Any]:
        """Get hardware data in flat format compatible with HardwareProfile entity.

        Returns hardware specifications as a flat dictionary ready for database
        insertion. For MLX backend, get_performance_evaluation() already returns
        data in the correct flat format.

        Returns:
            Flat dict with all fields matching HardwareProfile columns.

        Raises:
            HardwareException: If hardware data collection fails.
        """
        # get_performance_evaluation() already returns flat structure
        return cls.get_performance_evaluation()
