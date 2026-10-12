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
    │   AgentRunner → Erudi_Chat_OpenAI(base_url) → POST /v1/...    │
    │   carries delta.content AND delta.reasoning (#554)            │
    └───────────────────────────────────────────────────────────────┘
                                  ↓
    ┌───────────────────────────────────────────────────────────────┐
    │ cleanup() — override                                          │
    │  └─> SIGTERM child → join(5s) → SIGKILL if needed             │
    │      → wait_port_closed → super().cleanup()                   │
    └───────────────────────────────────────────────────────────────┘

The memory brake (``brake_point``, see the section below) gives memory back
under macOS memory pressure, at turn boundaries and from an idle monitor.

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
    mlx-vlm's native server-side split owns the reasoning markers (#554): the
    child spawns with `--enable-thinking` (thinking on by default) and streams
    chain-of-thought in the dedicated `delta.reasoning` field (mirrored into
    `delta.reasoning_content` on the pinned 0.6.17), which `Erudi_Chat_OpenAI`
    re-attaches to the message chunks and the runner emits as `thinking`
    events — symmetric to llama-server's default `--reasoning-format auto` on
    the CPU/CUDA path. The runner's ThinkSplitter stays as the inline fallback
    for families whose markers the server split does not know.

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

import math
import multiprocessing as mp
import os
import platform
import re
import secrets
import subprocess
import time  # used by hardware warm-up loop
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import requests

from src.engines.base_chat_server_engine import BaseChatServerEngine
from src.engines._mlx_vlm_server_runner import run_mlx_vlm_server
from src.engines import memory_brake
from src.engines.memory_brake import BrakeOutcome, BrakeState
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

    # Stored links that download but crash at load.
    # gemma-4 E2B (the 0.6.2 140-weight ValueError) was re-probed on real
    # mlx-vlm 0.6.13 during the #273 hardware pass — loads via the native
    # gemma4 module and generates cleanly, so its entry was removed.
    KNOWN_BROKEN = frozenset(
        {
            # multi_modality/vision.py (DeepSeek-VL v1) hard-imports cv2,
            # which the macOS PyInstaller build does not bundle (#512).
            "mlx-community/deepseek-vl-1.3b-chat-4bit",
            "mlx-community/deepseek-vl-1.3b-chat-8bit",
            "mlx-community/deepseek-vl-7b-chat-4bit",
            "mlx-community/deepseek-vl-7b-chat-8bit",
            # tokenizer_config.json declares the abstract PreTrainedTokenizer,
            # which transformers cannot instantiate (#506). The v1_5 sibling
            # declares PreTrainedTokenizerFast and stays runnable.
            "mlx-community/Llama-3_3-Nemotron-Super-49B-v1-mlx-4bit",
            "mlx-community/Llama-3_3-Nemotron-Super-49B-v1-mlx-6bit",
        }
    )

    # Where the spawn stored the child's own log file, on the process object.
    # The llama-cpp engines keep their drainer the same way, and for the same
    # reason: the lifetime is the child's.
    _CHILD_LOG_ATTR = "erudi_child_log_path"

    # When this child was spawned. Ports are reused and a crashed child keeps
    # its file, so this is what tells one spawn's rolled output from the dead
    # predecessor's -- see `mlx_child_log.read_child_log_tail`.
    _CHILD_LOG_STARTED_ATTR = "erudi_child_log_started_at"

    # How much of the tail a crash message quotes. mlx-vlm's startup banner and
    # per-request lines are long; the reason it died is in the last lines.
    _CHILD_OUTPUT_TAIL_CHARS = child_log.DEFAULT_TAIL_CHARS

    # ======================= SUBPROCESS HTTP SERVER (mlx_vlm.server) =======================
    #
    # Inference goes through a subprocess `mlx_vlm.server` (OpenAI-compatible HTTP),
    # spawned via `mp.Process(target=run_mlx_vlm_server,
    # args=(argv, log_path, apc_num_blocks, token_queue_timeout_s))`.
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
        # from the handle on both probe stages. Hex, never URL-safe base64:
        # argparse reads a value starting with "-" as an option, and the child
        # would exit with "argument --api-key: expected one argument".
        api_key = secrets.token_hex(32)
        # The MLX window mechanism, deliberately different from llama-server's:
        # mlx_vlm.server allocates nothing upfront (its KV cache grows lazily,
        # RoPE positions are computed on the fly), so an unbounded child accepts
        # a prompt beyond the model's trained window with HTTP 200 and silently
        # degenerate output. `--max-kv-size W` is a clean PREFLIGHT VALIDATOR
        # (an oversized request gets a 400 naming the exact budget: prompt +
        # max generation vs W; it is NOT a rotating cache), so the bound is the
        # trained window read from the LOCAL artifact's own config.json --
        # stamped here at spawn, the one moment the artifact and the child are
        # both in hand. No derivable window: no flag (today's unbounded
        # behaviour), never an invented number. The bound is a per-request
        # validator mlx_vlm re-reads at call time, so when the machine's memory
        # binds below it, `_read_server_properties` lowers it through
        # `PATCH /v1/settings` right after the base footprint is measured (no
        # respawn), and the memory brake moves it again under memory pressure;
        # when the model window binds, nothing runs.
        context_tokens = cls._trained_window_of(model_path)
        # Automatic Prefix Caching pool + the aligned child token-queue timeout,
        # both derived from the window here (the one moment the artifact and the
        # child are in hand) and passed to the child as explicit args. Deferred
        # imports keep the engine layer off the agent stack at boot and match
        # this file's other function-scoped imports; `chat_model` is
        # langchain-free at import (it defers `langchain_openai`), so there is
        # no import cycle.
        from src.agents.chat_model import FIRST_CHUNK_ABSOLUTE_MAX_S, first_chunk_ceiling_s
        from src.engines._mlx_vlm_server_runner import APC_BLOCK_SIZE_TOKENS

        # Size the APC pool to exactly cover the TRAINED window (the spawn's
        # --max-kv-size above), so a 2nd+ turn reuses the cached prefix
        # instead of re-prefilling the whole history. A memory bound applied
        # after load leaves the pool as is: it fills lazily, so over-sizing it
        # costs nothing, and mlx-vlm's own guards bound it; no derivable
        # window leaves mlx-vlm's 2048-block default. The pool counts toward the
        # child's footprint: the memory prior accounts it in its per-token
        # multiplier (``memory_budget.PRIOR_KV_MULTIPLIER``: the KV cache plus
        # its prefix-cache copy and MLX's buffer cache), and every turn's real
        # peak is measured against it (``end_memory_window``).
        apc_num_blocks: Optional[int] = (
            math.ceil(context_tokens / APC_BLOCK_SIZE_TOKENS)
            if context_tokens is not None
            else None
        )
        # Lift the child's per-token wait above the parent's first-chunk
        # watchdog ceiling for this window (the SAME `first_chunk_ceiling_s` the
        # parent clamps its budget to), so a legitimately long cold prefill is
        # ended by the parent's curated turn (src.agents.runner) -- never by the
        # child's raw "Increase MLX_VLM_TOKEN_QUEUE_TIMEOUT" error. +60s margin
        # so the child comfortably outlives the ceiling. A memory bound applied
        # after load only LOWERS a known window, hence the parent's ceiling.
        # Without a trained window, though, the bound becomes the parent's
        # first window and can raise its ceiling up to the absolute max: the
        # child then waits the absolute max + 60s (the timeout only fires on a
        # wedged worker; an abandoned request is cancelled at HTTP close).
        token_queue_timeout_s: float = (
            first_chunk_ceiling_s(context_tokens) + 60.0
            if context_tokens is not None
            else FIRST_CHUNK_ABSOLUTE_MAX_S + 60.0
        )
        # Before the roll below, so a file the PREVIOUS child on this port left
        # behind is older than this mark and cannot be read as ours.
        started_at = time.time()
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
            # (Erudi's chat turns never do; one-shot titles send
            # enable_thinking=False). Without it, the chat template renders
            # with enable_thinking=False and a thinking model (e.g. Qwen3)
            # answers directly — no reasoning ever exists (#90). The native
            # server-side split then routes the reasoning to delta.reasoning,
            # which Erudi_Chat_OpenAI carries to the runner (#554). Safe for
            # non-thinking models: their templates ignore the kwarg, their
            # prompts never open a thinking block, so the stream starts (and
            # stays) on the content channel.
            "--enable-thinking",
        ]
        # The prefill step the memory prior was measured at (its logits term
        # and t0 depend on it): on the command line, because mlx_vlm's CLI
        # rewrites PREFILL_STEP_SIZE from this option (server/cli.py).
        from src.engines.memory_budget import MLX_PREFILL_STEP_TOKENS

        argv += ["--prefill-step-size", str(MLX_PREFILL_STEP_TOKENS)]
        if context_tokens is not None:
            argv += ["--max-kv-size", str(context_tokens)]
        argv += [
            "--api-key",
            api_key,
        ]
        proc = mp.Process(
            target=run_mlx_vlm_server,
            args=(argv, log_path, apc_num_blocks, token_queue_timeout_s),
            daemon=False,
        )
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
        setattr(proc, cls._CHILD_LOG_STARTED_ATTR, started_at)
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
            # The ALLOCATED window (`BaseEngine.effective_context_tokens`):
            # at spawn it equals the preflight bound passed above -- nothing
            # to read back, unlike llama-server, which allocates its window at
            # load. `_read_server_properties` measures the child's base
            # footprint and, when memory binds, lowers the child's bound and
            # rewrites this to it. None = unbounded child.
            "context_tokens": context_tokens,
            # The model's trained window, kept as stamped at spawn: the
            # memory bound is judged against it (`working_window.kv_bound`).
            "trained_context_tokens": context_tokens,
            # Set by `_read_server_properties` when the machine's memory, not
            # the model, bounds the child's window (`memory_bound_tokens`).
            "memory_bound_tokens": None,
            # Who the child's prefix cache belongs to (see `claim_prefix`). A
            # new child starts with an EMPTY pool, so the first claim takes it
            # without a reset. No `abandoned` flag either: nothing ran yet.
            "prefix_owner": cls._PREFIX_OWNER_FRESH,
        }

    # ======================= PREFIX CACHE OWNERSHIP =======================
    #
    # The child's Automatic Prefix Cache pool holds ONLY the current
    # conversation: it is reset when another conversation (or the arena)
    # claims the child and when compaction rewrites the history, so a reused
    # prefix can only ever be this conversation's own and the pool never fills
    # with dead prefixes. The owner lives on the handle: a respawn, a swap or
    # an idle reap hands out a fresh handle, hence a fresh owner. One-shot
    # titles claim nothing: their few blocks go with the next reset.
    #
    # The reset is `POST /v1/cache/reset` (mlx_vlm `APCManager.clear()`, behind
    # the same Bearer key as inference). `clear()` re-pushes every block on the
    # free list with no guard against a double push, so it must never overlap a
    # request still holding blocks. An abandoned stream is cancelled
    # server-side before the next GPU step, but one cancelled during its LAST
    # prefill step finishes that step and releases its blocks AFTER the
    # cancellation -- possibly after a reset issued meanwhile. Hence the
    # barrier: after an abandon, a 1-token request goes first. The proof rests
    # on the child running ONE sequence at a time (`MLX_VLM_MAX_NUM_SEQS=1`,
    # set by `_mlx_vlm_server_runner._apply_child_runtime_env`): mlx-vlm admits
    # new requests before it drains cancellations, so with a wider batch the
    # barrier could share a step with the cancelled request and finish first.
    # With one sequence, the barrier is admitted only once the cancelled
    # request has left the batch -- its completion proves that request settled.

    _PREFIX_OWNER_FRESH = "<fresh>"
    _PREFIX_OWNER_DIRTY = "<dirty>"
    _CACHE_RESET_TIMEOUT_S = 5.0

    @classmethod
    def claim_prefix(cls, owner: str) -> None:
        """Make the prefix cache belong to ``owner``, resetting it if needed.

        Same owner: nothing to do. A fresh child: its pool is empty, so the
        owner is recorded without a reset. Anything else (another owner, a
        previous reset that failed -- ``<dirty>`` -- or no owner at all): reset,
        then record the owner. A failed reset leaves the owner ``<dirty>`` so
        the next claim retries; it never fails the turn.
        """
        handle = cls._model
        if not isinstance(handle, dict):
            return
        current = handle.get("prefix_owner")
        if current == owner:
            return
        if current == cls._PREFIX_OWNER_FRESH:
            handle["prefix_owner"] = owner
            return
        reason = "arena" if owner == "arena" else "conversation_change"
        if cls._reset_prefix_cache(handle, reason=reason):
            handle["prefix_owner"] = owner

    @classmethod
    def on_history_rewritten(cls) -> None:
        """Compaction rewrote the history: the cached prefix is garbage, and
        the turn's memory measurement restarts (the summary call's peak and
        the hops before it never count)."""
        handle = cls._model
        if not isinstance(handle, dict):
            return
        cls._reset_prefix_cache(handle, reason="compaction")
        cls._restart_memory_window(handle)

    @classmethod
    def note_stream_abandoned(cls, handle: Any) -> None:
        """Flag ``handle`` -- the child the stream was BUILT against, never the
        one loaded now -- so its next reset sends a barrier first."""
        if isinstance(handle, dict):
            handle["abandoned"] = True

    @classmethod
    def _reset_prefix_cache(
        cls,
        handle: Dict[str, Any],
        *,
        reason: str,
        barrier_timeout_s: Optional[float] = None,
    ) -> bool:
        """Reset the child's prefix cache; True on success. Never raises.

        Acts on ``handle`` only (the caller read ``cls._model`` once). After
        an abandoned stream, a barrier request goes first; a failed barrier
        skips the reset (deferring only costs memory) and keeps the flag. Any
        failure is ONE warning and the owner ``<dirty>``, so the next claim
        retries. A child without a prefix cache answers 200
        ``{"enabled": false}``: a success with nothing to clear.
        ``barrier_timeout_s`` bounds that barrier below the first-chunk
        ceiling (the memory brake's release: a shielded release must never
        hold a shutdown that long).
        """
        api_key = handle.get("api_key")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        if handle.get("abandoned"):
            if not cls._send_barrier(handle, headers, reason=reason, timeout_s=barrier_timeout_s):
                handle["prefix_owner"] = cls._PREFIX_OWNER_DIRTY
                return False
            handle["abandoned"] = False
        try:
            resp = requests.post(
                f"{handle.get('base_url')}/v1/cache/reset",
                headers=headers,
                timeout=cls._CACHE_RESET_TIMEOUT_S,
            )
        except requests.RequestException as exc:
            # The class name only: the exception text can quote the request.
            logger.warning(
                f"[MLX_Engine] Prefix cache reset failed ({type(exc).__name__}); "
                f"deferred to the next claim: reason={reason}"
            )
            handle["prefix_owner"] = cls._PREFIX_OWNER_DIRTY
            return False
        if resp.status_code != 200:
            logger.warning(
                f"[MLX_Engine] Prefix cache reset failed (HTTP {resp.status_code}); "
                f"deferred to the next claim: reason={reason}"
            )
            handle["prefix_owner"] = cls._PREFIX_OWNER_DIRTY
            return False
        logger.info(f"Prefix cache reset: reason={reason}")
        return True

    # What a failed barrier costs, by purpose: a claim's barrier defers the
    # reset to the next claim; the memory brake's release barrier leaves
    # MLX's buffer cache to a later release.
    _BARRIER_FAILURE_OUTCOMES = {
        "claim": "reset deferred to the next claim",
        "release": "memory release incomplete, a later release returns it",
    }

    @classmethod
    def _send_barrier(
        cls,
        handle: Dict[str, Any],
        headers: Optional[Dict[str, str]],
        *,
        reason: str,
        purpose: str = "claim",
        timeout_s: Optional[float] = None,
    ) -> bool:
        """One 1-token request, the readiness ping's shape; True on success.

        It proves a cancelled request has settled only because the child runs
        one sequence at a time (``MLX_VLM_MAX_NUM_SEQS=1``, see the section
        comment above). Bounded by default by the first-chunk ceiling for
        this child's window: the barrier may queue behind the last step of a
        cancelled prefill, and that step can legitimately last as long as a
        long prefill does. ``timeout_s`` bounds it tighter (the memory brake).

        It is NON-streamed on purpose: mlx_vlm's non-streamed chat handler
        clears MLX's buffer cache after generation, which makes this request
        the second half of the memory brake's release (``purpose="release"``).
        """
        from src.agents.chat_model import first_chunk_ceiling_s

        outcome = cls._BARRIER_FAILURE_OUTCOMES[purpose]
        timeout = (
            timeout_s
            if timeout_s is not None
            else first_chunk_ceiling_s(handle.get("context_tokens"))
        )
        try:
            resp = requests.post(
                f"{handle.get('base_url')}/v1/chat/completions",
                json={
                    "model": cls._payload_model_value(handle),
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "stream": False,
                },
                headers=headers,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            logger.warning(
                f"[MLX_Engine] Prefix cache barrier failed ({type(exc).__name__}); "
                f"{outcome}: reason={reason}"
            )
            return False
        if resp.status_code >= 400:
            logger.warning(
                f"[MLX_Engine] Prefix cache barrier failed (HTTP {resp.status_code}); "
                f"{outcome}: reason={reason}"
            )
            return False
        return True

    # ======================= MEASURED MEMORY =======================
    #
    # The memory prior (``src.engines.memory_budget``) is checked against what
    # the child really uses. The child's footprint right after its readiness
    # probe is its BASE (measured at every spawn). Around every conversation
    # and Arena turn (never a title) a measurement window is open: the
    # footprint interval is reset through the macOS SPI
    # (``process_footprint.begin_peak_window``), the conversation and Arena
    # clients push each call's usage (``note_call_usage``, bound to the handle
    # by the factory), and the end of the turn records one observation --
    # ``n = max over calls of prompt + completion`` (the decode KV is in the
    # peak; mixing the largest prompt of one hop with the longest answer of
    # another would over-size n), the peak above base, and ``predict(n)``.
    # A compaction restarts the window (a new sequence number: the hops before
    # it and the summary call never count). Recorded only when the window
    # really started on a measured base, and usage was reported -- or the
    # turn was abandoned with a call in flight (recorded at the client's
    # estimate, ``n_est``, announced before the call: ``note_call_start``), or
    # the child died. A peak above the prediction on a turn without images,
    # for a measured ``n``, is ONE WARNING per child and key; nothing is
    # corrected automatically.

    @classmethod
    def _read_server_properties(cls, handle: Dict[str, Any]) -> None:
        """The child is up: seed its memory brake, measure its base footprint,
        then bound its window by memory when memory binds. Degrades, never
        raises.

        The brake is seeded FIRST and unconditionally (the steps after it can
        stop early): a respawn of the model the brake unloaded carries its
        episode and its cap, so it resumes with the reduced window. A
        successful spawn always forgets the unload's record, whichever model
        it loads."""
        from src.engines import process_footprint

        handle["brake"] = cls._carried_brake(handle) or BrakeState()
        cls._brake_reload = None
        pid = handle.get("pid")
        base_failed = False
        try:
            handle["base_footprint_bytes"] = process_footprint.footprint(pid) if pid else None
        except Exception:
            # Costs the measured base only: the weights-on-disk fallback applies.
            logger.warning("[MLX_Engine] Base footprint unreadable after spawn", exc_info=True)
            handle["base_footprint_bytes"] = None
            base_failed = True
        cls._bound_window_by_memory(handle, base_failed=base_failed)

    # ======================= THE MEMORY BOUND =======================
    #
    # The working window is min(model window, memory ceiling). When the
    # memory ceiling is the binding side, the child itself must enforce it:
    # mlx_vlm re-reads `max_kv_size` per request (a live knob, outside the
    # reload fingerprint), so the bound is set through `PATCH /v1/settings`,
    # first right after the base is measured -- before anything but the
    # readiness ping reached the child -- then again whenever the memory brake
    # moves its cap (`_apply_memory_bound` takes min(ceiling, brake cap), and
    # brings the trained window back once memory no longer binds). The bound
    # IS the allocated window (`context_tokens`), so every consumer of the
    # allocated window (model info, first-chunk watchdog, preflight retry,
    # compaction) is right without knowing about it, exactly as with a
    # llama.cpp window its fit reduced at load. When the PATCH answer is lost
    # or unreadable, one `GET /v1/settings` reads the child's live value back,
    # so the handle never disagrees with what the child enforces. The ceiling
    # itself is constant for the life of a child (its inputs are the base and
    # static facts measured at load); the bound moves only with the brake's
    # cap. In the degraded corners the child keeps the window it runs with
    # and no new bound is recorded: a total that resolves only after load, a
    # PATCH the child did not apply, an unreadable base, unreadable static
    # facts.

    _SETTINGS_TIMEOUT_S = 2.0

    @classmethod
    def memory_bound_tokens(cls) -> Optional[int]:
        """The memory bound applied to the loaded child, or ``None``."""
        handle = cls._model
        if not isinstance(handle, dict):
            return None
        bound = handle.get("memory_bound_tokens")
        if isinstance(bound, int) and not isinstance(bound, bool) and bound > 0:
            return bound
        return None

    @classmethod
    def _stop_server_if_running(cls) -> None:
        """The shared stop, then the MLX window keys of the stopped handle:
        its trained window and its memory bound die with the child."""
        handle = cls._model
        super()._stop_server_if_running()
        if isinstance(handle, dict):
            handle["trained_context_tokens"] = None
            handle["memory_bound_tokens"] = None

    @classmethod
    def _bound_window_by_memory(cls, handle: Dict[str, Any], *, base_failed: bool) -> None:
        """At spawn: lower the child's window to the memory bound when memory
        binds -- min(ceiling, brake cap), the cap a brake reload carried.

        One guard around the whole step (computing the budget reads the
        disk): any failure is one WARNING with the exception's class name,
        the handle unchanged, and the load goes on. Never logs the key, the
        request headers or a response body. ``base_failed``: the base
        measurement raised and already wrote its WARNING. Without a measured
        base the ceiling is not used, but a carried brake cap still applies,
        so the reload turn already runs with the reduced window.
        """
        try:
            state = handle.get("brake")
            carried_cap = state.cap if isinstance(state, BrakeState) else None
            if handle.get("base_footprint_bytes") is None and carried_cap is None:
                # The budget would price the weights on disk instead: not a
                # number to bound a live child on. One record per failure: a
                # measurement that raised has written its own.
                if not base_failed:
                    logger.info(
                        "[MLX_Engine] Base footprint unknown; the child keeps its spawn window"
                    )
                return
            applied = cls._apply_memory_bound(handle)
            if applied is None or not applied["ok"] or applied["bound"] is None:
                return
            cap = applied["cap"]
            logger.info(
                f"[MLX_Engine] Window bounded by memory: trained={applied['trained']}, "
                f"ceiling={applied['ceiling']}, bound={applied['bound']}"
                + (f", brake_cap={cap}" if cap is not None else "")
            )
        except Exception as exc:
            logger.warning(
                f"[MLX_Engine] Memory bound of the window not applied "
                f"({type(exc).__name__}); the child keeps its spawn window"
            )

    @classmethod
    def _memory_ceiling(cls, handle: Dict[str, Any]) -> Optional[int]:
        """The memory ceiling of ``handle``'s child, in tokens -- only on a
        MEASURED base (the weights-on-disk fallback is no number to bound a
        live child on). Reads the disk once per child (static facts)."""
        from src.engines.memory_budget import MemoryBudget

        if handle.get("base_footprint_bytes") is None:
            return None
        return MemoryBudget.from_handle(cls, handle).tokens_at_ceiling()

    @classmethod
    def _apply_memory_bound(cls, handle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make the child enforce min(memory ceiling, brake cap), or its
        trained window once memory no longer binds.

        Returns ``None`` when there is nothing to change, else what it
        tried: ``bound`` (None: back to the model's window) and ``ok`` --
        False when the child did not apply it (``_apply_max_kv_size`` wrote
        the WARNING; the handle is unchanged). A bound equal to the window the
        child already enforces needs no HTTP call (a trained window at or
        below the floor: memory binds, and the notice must be able to say
        so). An unbounded child (no trained window) cannot be given its
        unbounded window back: it keeps its last bound, no longer recorded as
        a memory bound. Without a measured base the ceiling is ignored and the
        cap alone applies. May raise (the budget reads the disk): the callers
        guard it.
        """
        from src.engines.working_window import kv_bound

        trained = handle.get("trained_context_tokens")
        ceiling = cls._memory_ceiling(handle)
        state = handle.get("brake")
        cap = state.cap if isinstance(state, BrakeState) else None
        limits = [
            value
            for value in (ceiling, cap)
            if isinstance(value, int) and not isinstance(value, bool)
        ]
        bound = kv_bound(trained, min(limits) if limits else None)
        tried = {"trained": trained, "ceiling": ceiling, "cap": cap, "bound": bound}
        if bound is None:
            if handle.get("memory_bound_tokens") is None:
                return None
            if trained is not None:
                if handle.get("context_tokens") != trained and not cls._apply_max_kv_size(
                    handle, trained
                ):
                    return {**tried, "ok": False}
                handle["context_tokens"] = trained
            handle["memory_bound_tokens"] = None
            return {**tried, "ok": True}
        if bound != handle.get("context_tokens") and not cls._apply_max_kv_size(handle, bound):
            return {**tried, "ok": False}
        handle["context_tokens"] = bound
        handle["memory_bound_tokens"] = bound
        return {**tried, "ok": True}

    @classmethod
    def _apply_max_kv_size(cls, handle: Dict[str, Any], bound: int) -> bool:
        """``PATCH /v1/settings`` ``{"max_kv_size": bound}`` on the child;
        True iff the child enforces ``bound`` afterwards.

        A 200 that names ``bound`` as applied settles it. Anything else -- an
        error status, a lost answer (a timeout can fire after the child
        applied the change), an unreadable or unexpected body -- is checked
        with one ``GET /v1/settings`` read-back of the live value: the child
        enforcing ``bound`` is a success, anything else a failure. Either way
        it is ONE WARNING, with the HTTP status or the exception's class name
        only.
        """
        url = f"{handle.get('base_url')}/v1/settings"
        api_key = handle.get("api_key")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        try:
            resp = requests.patch(
                url,
                json={"max_kv_size": bound},
                headers=headers,
                timeout=cls._SETTINGS_TIMEOUT_S,
            )
            failure = cls._unconfirmed(resp, bound)
        except Exception as exc:
            # The answer is lost, not necessarily the change: the read-back
            # below says which, and the WARNING names the class.
            failure = type(exc).__name__
        if failure is None:
            return True
        live, read_back = cls._live_max_kv_size(url, headers)
        if live == bound:
            logger.warning(
                f"[MLX_Engine] Memory bound of the window unconfirmed ({failure}); "
                f"the child reports max_kv_size={bound}, so it applies"
            )
            return True
        logger.warning(
            f"[MLX_Engine] Memory bound of the window not applied ({failure}; "
            f"read back: {read_back}); the child keeps its current window"
        )
        return False

    @staticmethod
    def _unconfirmed(resp: Any, bound: int) -> Optional[str]:
        """``None`` when the PATCH answer confirms ``bound``, else why not
        (never the body itself). ``resp.json()`` may raise: the caller's
        handler names it."""
        if resp.status_code != 200:
            return f"HTTP {resp.status_code}"
        body = resp.json()
        applied = body.get("applied") if isinstance(body, dict) else None
        if isinstance(applied, dict) and applied.get("max_kv_size") == bound:
            return None
        return "max_kv_size not confirmed"

    @classmethod
    def _live_max_kv_size(
        cls, url: str, headers: Optional[Dict[str, str]]
    ) -> Tuple[Optional[int], str]:
        """The child's live ``max_kv_size`` (``GET /v1/settings``) and a
        loggable description of the read-back (never the body)."""
        try:
            resp = requests.get(url, headers=headers, timeout=cls._SETTINGS_TIMEOUT_S)
            if resp.status_code != 200:
                return None, f"HTTP {resp.status_code}"
            body = resp.json()
            current = body.get("current") if isinstance(body, dict) else None
            value = current.get("max_kv_size") if isinstance(current, dict) else None
        except Exception as exc:
            # Reported in the caller's single WARNING, by class name: a
            # read-back that fails leaves the handle at its current window.
            return None, type(exc).__name__
        if isinstance(value, int) and not isinstance(value, bool):
            return value, f"max_kv_size={value}"
        return None, "no max_kv_size"

    @classmethod
    def begin_memory_window(cls) -> Any:
        """Open the turn's measurement window on the loaded child; returns the
        ``(handle, pid, window)`` token ``end_memory_window`` closes it with
        (``window`` numbers the turn's window on this child: a token of an
        earlier window never closes a later one)."""
        from src.engines import process_footprint

        handle = cls._model
        if not isinstance(handle, dict) or not handle.get("pid"):
            return None
        pid = handle["pid"]
        # Every turn opens a window, abandoned ones included: the brake sees
        # the turn (no unload across it). The previous turn's largest request
        # stands until this one's window closes with a size of its own.
        state = handle.get("brake")
        if isinstance(state, BrakeState):
            state.turn_happened = True
        previous = handle.get("memory_window") or {}
        base = handle.get("base_footprint_bytes")
        current = process_footprint.footprint(pid)
        window_id = int(previous.get("window", 0)) + 1
        handle["memory_window"] = {
            "window": window_id,
            "seq": int(previous.get("seq", 0)) + 1,
            "open": True,
            # What the previous request left behind (MLX's buffer cache):
            # it can raise this turn's peak.
            "residue_at_start_bytes": (
                current - base if current is not None and base is not None else None
            ),
            "swapouts_start": process_footprint.swapouts(),
            "pressure_start": process_footprint.pressure_level(),
            "spi_started": process_footprint.begin_peak_window(pid),
            "calls": [],
        }
        return (handle, pid, window_id)

    @classmethod
    def _restart_memory_window(cls, handle: Dict[str, Any]) -> None:
        """A new measurement window inside the same turn (after a compaction)."""
        from src.engines import process_footprint

        window = handle.get("memory_window")
        if not isinstance(window, dict) or not window.get("open") or not handle.get("pid"):
            return
        window["seq"] = int(window.get("seq", 0)) + 1
        window["spi_started"] = process_footprint.begin_peak_window(handle["pid"])

    @classmethod
    def note_call_usage(
        cls,
        handle: Any,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        has_images: bool = False,
    ) -> None:
        """One call's usage, with the window's sequence number at push time."""
        window = handle.get("memory_window") if isinstance(handle, dict) else None
        if not isinstance(window, dict) or not window.get("open"):
            return
        window.pop("in_flight", None)
        window["calls"].append(
            {
                "seq": window["seq"],
                "n_in": int(input_tokens or 0),
                "n_out": int(output_tokens or 0),
                "cached": int(cached_tokens or 0),
                "has_images": bool(has_images),
            }
        )

    @classmethod
    def note_call_start(cls, handle: Any, estimated_prompt_tokens: int) -> None:
        """A call is in flight: its estimated size, until its usage arrives."""
        window = handle.get("memory_window") if isinstance(handle, dict) else None
        if not isinstance(window, dict) or not window.get("open"):
            return
        window["in_flight"] = {"seq": window["seq"], "n_est": int(estimated_prompt_tokens or 0)}

    @classmethod
    def end_memory_window(
        cls, token: Any, *, abandoned: bool = False, compaction_threshold: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Close the window and record the turn's observation; returns it, or
        ``None`` when nothing could be recorded.

        Recorded only on a MEASURED base (never the weights-on-disk fallback)
        and a window that really started. A child that died during the window
        is recorded too, ``child_died`` and no peak: it is the ground truth of
        an under-prediction. An ABANDONED turn whose call was cut before its
        usage arrived is recorded with ``n_est`` (the client's estimate of the
        call in flight) and no measured ``n``; an estimate never raises the
        exceeded-prediction WARNING.

        Whatever is recorded, the turn's largest request is written to
        ``handle["last_request_tokens"]`` first: the measured ``n``, else the
        estimate of the call in flight (``n_est``: an abandoned turn, or one
        that recovered from an error before any usage arrived, such as a
        first-chunk timeout); a turn with neither keeps the previous value.
        ``compaction_threshold`` -- the threshold of the conversation that
        sent it, ``None`` for an arena turn -- is written beside it, to
        ``handle["last_request_threshold"]``, and only then: the pair stays
        the request of one turn and the trigger it is judged against.
        The memory brake shrinks the window only when the next turn would
        compact under it, and a missing SPI or base must not hide that."""
        import time
        import uuid

        from src.engines import memory_observations, process_footprint
        from src.engines._mlx_vlm_server_runner import APC_BLOCK_SIZE_TOKENS
        from src.engines.memory_budget import MLX_BUFFER_CACHE_LIMIT, MemoryBudget, prior_scale

        if not token:
            return None
        handle, pid, window_id = token
        window = handle.get("memory_window") if isinstance(handle, dict) else None
        if (
            not isinstance(window, dict)
            or not window.get("open")
            or handle.get("pid") != pid
            or window.get("window") != window_id
        ):
            return None
        window["open"] = False
        calls = [call for call in window["calls"] if call["seq"] == window["seq"]]
        in_flight = window.get("in_flight")
        n_est = (
            in_flight.get("n_est")
            if isinstance(in_flight, dict) and in_flight.get("seq") == window["seq"]
            else None
        )
        if calls:
            largest = max(calls, key=lambda call: call["n_in"] + call["n_out"])
        else:
            largest = {"n_in": None, "n_out": None, "cached": None}
        n = (
            largest["n_in"] + largest["n_out"]
            if largest["n_in"] is not None and largest["n_out"] is not None
            else None
        )
        largest_request = n if n is not None else n_est
        if largest_request is not None:
            handle["last_request_tokens"] = largest_request
            handle["last_request_threshold"] = compaction_threshold
        base = handle.get("base_footprint_bytes")
        if not window.get("spi_started") or base is None:
            return None
        peak = process_footprint.peak_since(pid, True)
        child_died = peak is None and not cls._proc_is_alive(handle.get("proc"))
        measured = bool(calls) or child_died or (abandoned and n_est is not None)
        if (peak is None and not child_died) or not measured:
            return None
        budget = MemoryBudget.from_handle(cls, handle)
        scale = prior_scale()
        size = n if n is not None else n_est
        predicted = budget.predict(size) if size is not None else None
        if predicted is not None and scale != 1.0:
            predicted = int(predicted * scale)
        swap_start, swap_end = window.get("swapouts_start"), process_footprint.swapouts()
        pressures = [
            level
            for level in (window.get("pressure_start"), process_footprint.pressure_level())
            if level is not None
        ]
        has_images = any(call["has_images"] for call in calls)
        observation = {
            "id": uuid.uuid4().hex,
            "at": time.time(),
            "n": n,
            "n_est": n_est,
            "n_in": largest["n_in"],
            "n_out": largest["n_out"],
            "y_bytes": peak - base if peak is not None else None,
            "predicted_bytes": predicted,
            "cache_read": largest["cached"],
            "has_images": has_images,
            "abandoned": bool(abandoned),
            "residue_at_start_bytes": window.get("residue_at_start_bytes"),
            "pressure_level": max(pressures) if pressures else None,
            "swapouts": (
                swap_end - swap_start if swap_start is not None and swap_end is not None else None
            ),
            "spi_started": True,
            "child_died": child_died,
        }
        if scale != 1.0:
            observation["prior_scale"] = scale
        facts = handle.get("memory_facts") or {}
        components = memory_observations.key_components(
            model=facts.get("model_type"),
            artifact_bytes=facts.get("weights_bytes"),
            working_set_bytes=facts.get("working_set_bytes"),
            runtime={
                "apc_block_size": APC_BLOCK_SIZE_TOKENS,
                "prefill_step": facts.get("prefill_step"),
                "buffer_cache_limit": MLX_BUFFER_CACHE_LIMIT,
            },
        )
        try:
            memory_observations.record(components, base, observation)
        except Exception:
            # The observation is lost, nothing else: the turn already ended.
            logger.warning("[MLX_Engine] Recording a memory observation failed", exc_info=True)
        if (
            predicted is not None
            and n is not None
            and observation["y_bytes"] is not None
            and not has_images
            and observation["y_bytes"] > predicted
        ):
            warned = handle.setdefault("memory_exceeded_warned", set())
            key = memory_observations.entry_key(components)
            if key not in warned:
                warned.add(key)
                model = str(facts.get("model_type")).encode("ascii", "backslashreplace").decode()
                logger.warning(
                    f"Memory prediction exceeded on this machine (the app's memory estimate "
                    f"was wrong here; nothing is corrected automatically): model={model}, "
                    f"n_in={largest['n_in']}, n_out={largest['n_out']}, "
                    f"peak_above_base_bytes={observation['y_bytes']}, "
                    f"predicted_bytes={predicted}, "
                    f"residue_bytes={observation['residue_at_start_bytes']}"
                )
        return observation

    # ======================= THE MEMORY BRAKE =======================
    #
    # Under macOS memory pressure the child gives memory back, at SAFE POINTS
    # only: the start of a turn (for the child that turn will use), its end
    # (conversation and Arena turns alike), and idle (``_pressure_monitor``,
    # under the generation lock). Never during a generation. The control law
    # (``src.engines.memory_brake``) decides from the pressure level, the
    # turn's swap-outs, the footprint above base and the last turn's largest
    # request; this section executes it:
    #
    # * shrink / grow: the brake's cap moves, and ``_apply_memory_bound`` makes
    #   the child enforce min(memory ceiling, cap) -- the next turn compacts
    #   under the smaller window and frees its KV;
    # * release: the prefix cache is reset, then a NON-streamed 1-token
    #   request runs, after which mlx_vlm clears MLX's buffer cache (its
    #   streamed path never does, and the reset alone frees nothing: measured
    #   on Qwen2.5-0.5B, 1.833 GiB after a streamed 22.9k-token turn, 1.832
    #   after the reset, 1.054 after the 1-token request, for a 0.727 base).
    #   Both barriers it may send are bounded by ``RELEASE_BARRIER_TIMEOUT_S``
    #   so a shielded release never holds a shutdown for long. The prefix
    #   owner is kept;
    # * unload (idle only): the reap path, under the lock the idle monitor
    #   holds; the episode and the cap go to ``_brake_reload``, keyed by the
    #   model's alias, and the next spawn of that model is seeded with them
    #   (``_read_server_properties``) -- it resumes with the reduced window
    #   and takes no second unload in the same episode.
    #
    # Every sample runs through ``run_reset_shielded`` (it may talk to the
    # child): the next lock holder drains it. A sample never raises.

    RELEASE_BARRIER_TIMEOUT_S = 30.0

    @classmethod
    def brake_point(cls, point: str, *, llm_id: Any = None) -> BrakeOutcome:
        """Take the brake's sample at ``point`` and execute its decision.

        ``turn_start`` samples only when the loaded child is the one the turn
        for ``llm_id`` will use. Never raises: a failure is ONE WARNING with
        the exception's class name, and the sample is lost.
        """
        try:
            return cls._brake_point(point, llm_id)
        except Exception as exc:
            logger.warning(
                f"[MLX_Engine] Memory brake sample failed ({type(exc).__name__}): point={point}"
            )
            return BrakeOutcome()

    @classmethod
    def _brake_point(cls, point: str, llm_id: Any) -> BrakeOutcome:
        from src.engines import process_footprint
        from src.engines.memory_budget import total_memory_bytes
        from src.engines.working_window import canonical_working_window

        if point == memory_brake.POINT_TURN_START and not cls.loaded_child_for(llm_id):
            return BrakeOutcome()
        handle = cls._model
        if not isinstance(handle, dict) or not cls._proc_is_alive(handle.get("proc")):
            return BrakeOutcome()
        level = process_footprint.pressure_level()
        if level is None:
            if not MLX_Engine._brake_sensor_warned:
                MLX_Engine._brake_sensor_warned = True
                logger.warning(
                    "[MLX_Engine] Memory pressure level unreadable "
                    "(kern.memorystatus_vm_pressure_level); the memory brake is off"
                )
            return BrakeOutcome()
        pid = handle.get("pid")
        base = handle.get("base_footprint_bytes")
        footprint = process_footprint.footprint(pid) if pid else None
        excess = footprint - base if footprint is not None and base is not None else None
        swapped = None
        if point == memory_brake.POINT_TURN_END:
            window = handle.get("memory_window") or {}
            start, end = window.get("swapouts_start"), process_footprint.swapouts()
            swapped = end - start if start is not None and end is not None else None
        ceiling = cls._memory_ceiling(handle)
        total = total_memory_bytes(cls)
        state = handle.get("brake")
        decision = memory_brake.step(
            state if isinstance(state, BrakeState) else BrakeState(),
            memory_brake.Sample(
                point=point,
                now=time.time(),
                level=level,
                swapped_pages=swapped,
                excess_bytes=excess,
                last_request_tokens=handle.get("last_request_tokens"),
                compaction_threshold=handle.get("last_request_threshold"),
                last_used=(
                    cls._last_used.timestamp() if isinstance(cls._last_used, datetime) else None
                ),
            ),
            memory_brake.Context(
                window=canonical_working_window(handle.get("context_tokens"), ceiling),
                ceiling=ceiling,
                static_window=canonical_working_window(
                    handle.get("trained_context_tokens"), ceiling
                ),
                footprint_share=footprint / total if footprint is not None and total else None,
            ),
        )
        previous = state if isinstance(state, BrakeState) else BrakeState()
        handle["brake"] = decision.state
        if not decision.actions:
            return BrakeOutcome()
        if decision.unload:
            cls._brake_unload(handle, decision, point, excess)
            return BrakeOutcome(acted=True, action="unload")
        done = []
        if decision.release:
            # Settling the footprint takes up to RELEASE_SETTLE_S: only at
            # idle. At turn_end the lock is held and the turn's last events
            # wait, so the residual is one read.
            after = cls._brake_release(handle, settle=point == memory_brake.POINT_IDLE)
            # A release that failed (or whose residual is unreadable) leaves
            # ``excess_after_release`` as it was: the next eligible sample
            # retries, and no ``external`` notice is drawn from it. The
            # reset's or the barrier's own WARNING is the one record.
            if after is not None:
                decision.state.excess_after_release = after
                done.append("release")
                cls._log_brake(
                    logger.info,
                    point,
                    decision.level,
                    "release",
                    handle.get("context_tokens"),
                    handle.get("context_tokens"),
                    f"{cls._mib(excess)}->{cls._mib(after)}",
                )
        if decision.cap_change is not None:
            before = handle.get("context_tokens")
            if cls._brake_apply_cap(handle, decision, previous, point):
                done.append(decision.cap_change)
                cls._log_brake(
                    logger.info,
                    point,
                    decision.level,
                    decision.cap_change,
                    before,
                    handle.get("context_tokens"),
                    cls._mib(excess),
                )
        if not done:
            return BrakeOutcome()
        return BrakeOutcome(acted=True, action="+".join(done))

    @classmethod
    def _brake_apply_cap(
        cls,
        handle: Dict[str, Any],
        decision: "memory_brake.Decision",
        previous: BrakeState,
        point: str,
    ) -> bool:
        """Make the child enforce the decision's new cap; False when it did
        not. Then the decision never happened for the brake: a shrink's cap,
        window that hurt, episode count, turn guard and notice -- or a grow's
        cap -- go back to what they were, and the applier's WARNING (or this
        one, for an exception) is the one record."""
        try:
            applied = cls._apply_memory_bound(handle)
        except Exception as exc:
            logger.warning(
                f"[MLX_Engine] Memory brake {decision.cap_change} not applied "
                f"({type(exc).__name__}): point={point}"
            )
            applied = {"ok": False}
        if applied is None or applied["ok"]:
            return True
        state = decision.state
        state.cap = previous.cap
        if decision.cap_change == "shrink":
            state.faulty = previous.faulty
            state.shrinks_this_episode = previous.shrinks_this_episode
            state.shrunk_since_turn = previous.shrunk_since_turn
            state.notice = previous.notice
        return False

    @classmethod
    def _brake_release(cls, handle: Dict[str, Any], *, settle: bool) -> Optional[int]:
        """Reset the prefix cache, then send the non-streamed barrier that
        makes mlx_vlm clear MLX's buffer cache; returns what the child keeps
        above base afterwards -- only when both succeeded and the footprint
        is readable, else None. A failure is the reset's or the barrier's own
        WARNING. ``settle``: read the residual over ``RELEASE_SETTLE_S`` (idle
        only), else once."""
        from src.engines import process_footprint

        if not cls._reset_prefix_cache(
            handle, reason="memory_pressure", barrier_timeout_s=cls.RELEASE_BARRIER_TIMEOUT_S
        ):
            return None
        api_key = handle.get("api_key")
        if not cls._send_barrier(
            handle,
            {"Authorization": f"Bearer {api_key}"} if api_key else None,
            reason="memory_pressure",
            purpose="release",
            timeout_s=cls.RELEASE_BARRIER_TIMEOUT_S,
        ):
            return None
        pid, base = handle.get("pid"), handle.get("base_footprint_bytes")
        if not pid:
            return None
        after = cls._settled_footprint(pid) if settle else process_footprint.footprint(pid)
        return max(0, after - base) if after is not None and base is not None else None

    # mlx_vlm clears MLX's buffer cache synchronously, before the barrier's
    # answer is sent (``server/openai.py``: ``mx.clear_cache()`` then
    # ``gc.collect()`` ahead of building the response), but the process
    # footprint the kernel reports catches up a moment later (measured:
    # 656 MiB above base right after the answer, the ~0.33 GiB residual a few
    # seconds on). So at idle the residual is the lowest footprint over a
    # short, bounded window (under the lock, nobody waiting); at turn_end it
    # is one read -- the turn's last events must not wait, and a residual
    # read too early only makes the next release need a little more.
    RELEASE_SETTLE_S = 2.0
    _RELEASE_SETTLE_POLL_S = 0.25

    @classmethod
    def _settled_footprint(cls, pid: int) -> Optional[int]:
        """The child's lowest footprint over ``RELEASE_SETTLE_S`` (None:
        unreadable)."""
        from src.engines import process_footprint

        lowest = process_footprint.footprint(pid)
        deadline = time.monotonic() + cls.RELEASE_SETTLE_S
        while time.monotonic() < deadline:
            time.sleep(cls._RELEASE_SETTLE_POLL_S)
            current = process_footprint.footprint(pid)
            if current is not None and (lowest is None or current < lowest):
                lowest = current
        return lowest

    @classmethod
    def _brake_unload(
        cls,
        handle: Dict[str, Any],
        decision: "memory_brake.Decision",
        point: str,
        excess: Optional[int],
    ) -> None:
        """Reap the child (the caller holds the generation lock) and leave
        the episode for the next spawn of the same model. ONE WARNING: the
        app degraded on its own."""
        record = {
            "alias": handle.get("alias"),
            **decision.state.carry(),
            "unloaded_at": time.time(),
        }
        window = handle.get("context_tokens")
        cls._log_brake(
            logger.warning,
            point,
            decision.level,
            "unload",
            window,
            window,
            cls._mib(excess),
            "; the model reloads at the next message",
        )
        cls.cleanup()
        cls._brake_reload = record

    @staticmethod
    def _mib(value: Optional[int]) -> Any:
        return value // (1024 * 1024) if isinstance(value, int) else None

    @staticmethod
    def _log_brake(record, point, level, action, before, after, excess, suffix="") -> None:
        record(
            f"[MLX_Engine] Memory brake: point={point}, level={level}, action={action}, "
            f"window={before}->{after}, excess_mib={excess}{suffix}"
        )

    @classmethod
    def _trained_window_of(cls, model_path: Path) -> Optional[int]:
        """The trained context window the local artifact declares, or ``None``.

        Read through ``read_local_generation_hints`` (the offline capture over
        the artifact's own config.json, with the full key/container aliases),
        so the spawn bound and the catalog's ``context_length`` agree by
        construction. ``None`` -- no config, or none of the known keys -- is
        answered with one WARNING and the child spawns unbounded, exactly
        today's behaviour; a number is never invented.
        """
        from src.database.generation_hints import read_local_generation_hints

        hints = read_local_generation_hints(model_path)
        window = (hints or {}).get("context_length")
        if isinstance(window, int) and not isinstance(window, bool) and window > 0:
            return window
        logger.warning(
            f"[MLX_Engine] No trained context window derivable from {model_path}; "
            f"spawning mlx_vlm.server without a --max-kv-size bound"
        )
        return None

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
        started_at = getattr(proc, cls._CHILD_LOG_STARTED_ATTR, None)
        tail = child_log.read_child_log_tail(
            path,
            max_chars=cls._CHILD_OUTPUT_TAIL_CHARS,
            since=started_at if isinstance(started_at, (int, float)) else None,
        )
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
        # M4 Max also ships in a 14-core CPU / 32-core GPU / 410GB/s
        # configuration (support.apple.com/en-us/121553), but system_profiler
        # reports both configurations as the same "Apple M4 Max" chip_type,
        # so this table cannot hold a second "M4 Max" key without a way to
        # tell them apart at runtime. Not added; see #503.
        #
        # M3 Ultra (support.apple.com/en-us/122211): both the 28-core CPU /
        # 60-core GPU and 32-core CPU / 80-core GPU configurations report
        # 819GB/s, so -- unlike M4 Max -- the ambiguity does not affect
        # bandwidth. Entry uses the 32-core CPU / 80-core GPU configuration,
        # matching this table's existing M2 Ultra entry (also the higher-end
        # GPU config of its two). Neural Engine TOPS confirmed by Apple
        # Newsroom, "Apple reveals M3 Ultra, taking Apple silicon to a new
        # extreme" (2025-03-05): "32 Neural Engine cores capable of up to 36
        # trillion operations per second".
        "M3 Ultra": {
            "gpu_cores": 80,
            "memory_bandwidth": 819,
            "neural_engine_tops": 36.0,
            "cpu_cores": {"performance": 24, "efficiency": 8},
            "max_memory": 256,
            "architecture": "3nm",
        },
        # M5 family (support.apple.com/en-us/125405, /en-us/126318, and
        # apple.com/mac-studio/specs/). M5 Pro and M5 Max drop the
        # performance/efficiency split for a "super core" / "performance
        # core" pair instead -- both tiers are full-power cores, there is no
        # low-power efficiency core on these two chips. Folded into this
        # table's existing performance/efficiency shape as
        # performance = super + performance cores, efficiency = 0; that is
        # the CPU split Apple publishes, not an estimate.
        # Apple has not published a Neural Engine TOPS figure for M5 (the
        # generation moved to per-GPU-core "Neural Accelerators" instead), so
        # neural_engine_tops is left out of every M5 entry below and falls
        # back to this table's existing 0.0 "unknown" default rather than
        # a guessed number.
        "M5": {
            "gpu_cores": 10,
            "memory_bandwidth": 153,
            "cpu_cores": {"performance": 4, "efficiency": 6},
            "max_memory": 32,
            "architecture": "3nm",
        },
        # M5 Pro ships in two CPU/GPU configurations (15-core CPU/16-core GPU
        # and 18-core CPU/20-core GPU) that both report 307GB/s. Entry below
        # is the higher-end 18-core CPU (6 super + 12 performance) / 20-core
        # GPU configuration, its max_memory (64GB).
        "M5 Pro": {
            "gpu_cores": 20,
            "memory_bandwidth": 307,
            "cpu_cores": {"performance": 18, "efficiency": 0},
            "max_memory": 64,
            "architecture": "3nm",
        },
        # M5 Max has the same 32-core/410GB/s vs. 40-core/546-style ambiguity
        # as M4 Max (32-core GPU/460GB/s vs. 40-core GPU/614GB/s, both
        # reported as plain "Apple M5 Max" by system_profiler). Entry below
        # uses the higher-end 18-core CPU (6 super + 12 performance) /
        # 40-core GPU / 614GB/s configuration, matching this table's M4 Max
        # precedent; the 32-core GPU configuration is not added, same as the
        # M4 Max case above.
        "M5 Max": {
            "gpu_cores": 40,
            "memory_bandwidth": 614,
            "cpu_cores": {"performance": 18, "efficiency": 0},
            "max_memory": 128,
            "architecture": "3nm",
        },
        # M5 Ultra (apple.com/mac-studio/specs/, announced 2026-08-25,
        # shipping from 2026-09-22) ships as a 30-core CPU/64-core GPU chip,
        # configurable to 36-core CPU/80-core GPU; both report 1.2TB/s.
        # Unlike the other Ultra entries in this table, this entry uses the
        # LOWER-end 30-core CPU (10 super + 20 performance) / 64-core GPU
        # configuration: Apple's spec page gives the CPU core split for the
        # 30-core configuration but not for the 36-core one, and this table
        # does not guess a split it cannot cite. max_memory (256GB) is this
        # configuration's ceiling; the 36-core/80-core GPU config goes to
        # 512GB.
        "M5 Ultra": {
            "gpu_cores": 64,
            "memory_bandwidth": 1200,
            "cpu_cores": {"performance": 20, "efficiency": 0},
            "max_memory": 256,
            "architecture": "3nm",
        },
    }

    # Matches the chip variant out of a system_profiler `chip_type` string
    # (e.g. "Apple M4 Pro" -> number="4", variant="Pro"). Used to build an
    # exact "M4 Pro" lookup key instead of substring-matching the table (#503:
    # "M4" is a substring of "Apple M4 Pro", so the old loop returned the
    # base chip's numbers for every Pro/Max/Ultra machine).
    _CHIP_VARIANT_PATTERN = re.compile(r"\bM(\d+)(?:\s+(Pro|Max|Ultra))?\b", re.IGNORECASE)

    @classmethod
    def _detect_apple_silicon_chip(cls) -> Optional[str]:
        """Detect specific Apple Silicon chip model (M1, M2, M3, M4, etc.).

        Uses system_profiler command to identify the exact chip variant.

        Returns:
            Optional[str]: Chip model (e.g., "M3 Max") or None if not detected
                or if the detected variant has no entry in
                `_APPLE_SILICON_SPECS`. Never falls back to a different
                variant's key.

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
                    match = cls._CHIP_VARIANT_PATTERN.search(chip_name)
                    if match:
                        number, variant = match.groups()
                        canonical_key = f"M{number}" + (
                            f" {variant.capitalize()}" if variant else ""
                        )
                        if canonical_key in cls._APPLE_SILICON_SPECS:
                            return canonical_key
                        logger.warning(
                            f"Detected Apple Silicon chip '{chip_name}' (parsed as "
                            f"'{canonical_key}') has no entry in the specs table"
                        )

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

    @classmethod
    def max_recommended_working_set_bytes(cls) -> Optional[int]:
        """The GPU's usable working set in bytes, or ``None`` if unavailable.

        On Apple Silicon Metal caps the GPU at a recommended working set well
        below total unified memory (~74 % of RAM; mlx-vlm pins the wired limit
        to it on every generation). That working set -- not total RAM -- is the
        hard ceiling the memory accounting plans against, so
        ``memory_budget.total_memory_bytes`` reads it here (#601), and the
        recorded memory observations carry it as part of their key.

        Kept deliberately off ``get_flat_hardware_data`` /
        ``get_performance_evaluation``: this fact does not belong in the
        persisted HardwareProfile schema; a dedicated method keeps the hardware
        knowledge in the engine without polluting the profile.

        The ``mlx`` import is lazy (nothing at module import time may require
        mlx) and any failure answers ``None`` -- this may be called on a host
        where the query is unavailable.
        """
        try:
            import mlx.core as mx

            return int(mx.device_info()["max_recommended_working_set_size"])
        except Exception:
            # No record here on purpose: the caller
            # (memory_budget.total_memory_bytes) writes the single degraded
            # record and simply leaves the memory accounting off when this is None.
            return None
