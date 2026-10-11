"""Abstract sub-base for engines that wrap the `llama-server` binary.

Sits between `BaseChatServerEngine` and the concrete `CPU_Engine` /
`CUDA_Engine` classes. Factors the bits CPU and CUDA share but MLX does
not:

- Where the binary lives (`backend/artifacts/llama-cpp/<cpu|cuda>/bin/llama-server`)
- How to find / pick the GGUF file in a model directory
- The `subprocess.Popen` lifecycle (terminate, alive check, output draining)
- The flags every spawn carries whatever the flavour: `--api-key`, `--no-slots`,
  `--no-webui`, and `-np 1 --cache-ram 0` (one slot, no prompt cache: the child
  keeps the current conversation's KV only)
- Kwarg-name translation from Erudi's vocabulary (HF/transformers) to the
  llama.cpp wire names (`repetition_penalty` → `repeat_penalty`,
  `repetition_context_size` → `repeat_last_n`).

Subclasses choose:
- `_use_cuda_build` (False for CPU, True for CUDA — selects artifact dir)
- `_build_spawn_argv` (CPU forces `-ngl 0`; CUDA injects computed `-ngl`)
- `_build_spawn_env` (CUDA prepends the CUDA toolkit to `PATH`)
- `_tokenizer_provider` (just for the placeholder dict)
"""

from __future__ import annotations

import os
import platform
import secrets
import signal
import subprocess
import time
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Union

import requests

from src.core.config import ROOT_DIR
from src.core.exceptions import EngineException
from src.core.logging import logger
from src.engines import gguf_memory, memory_brake, memory_pressure
from src.engines.base_chat_server_engine import BaseChatServerEngine
from src.engines.child_output import ChildOutputDrainer
from src.engines.memory_brake import BrakeOutcome, BrakeState
from src.core.subprocess_flags import hidden_console_creationflags

# How long a memory-brake respawn may take to answer its readiness probe. It
# runs under the generation lock (the turn's last events or the shutdown drain
# wait for it): a warm reload answers in 1-2 s; past this bound the respawn
# fails and degrades into an unload.
LLAMA_RESPAWN_PROBE_TIMEOUT_S = 30.0

# The spawn-context items `_spawn_child` keeps on the handle: observability
# (threads, GPU layers, the window passed) and the window facts the memory
# brake and the post-load check read back.
PRESERVED_SPAWN_KEYS = (
    "threads",
    "gpu_layers",
    "ctx_size",
    "window_pinned",
    "trained_context_tokens",
    "static_window_tokens",
    "kv_token_bytes",
    "static_anonymous_bytes",
    "weights_bytes",
    "spawn_memory_bound",
    "mmproj_loaded",
)


# Mirror of llama.cpp's SPECIALIZED tool-format dispatch, for LOGS ONLY (#298).
#
# Provenance: backend/forks/llama-cpp/common/chat.cpp (b10883),
# common_chat_try_specialized_template, lines 1080-1204. Each entry is
# (format_name, required markers, forbidden markers, any-of markers), in the
# same order chat.cpp tests them: a template matches when every required marker
# appears, no forbidden one does, and at least one any-of does (an empty group
# is vacuously satisfied). The negative and either-or groups are not decoration
# -- three of these formats are told apart from a sibling by exactly that.
#
# A template matching none of them still gets structured tool handling: with
# `--jinja` (which both GGUF engines pass at spawn) llama-server hands it to the
# differential autoparser (chat.cpp:1311-1345), which reads the template and
# generates a PEG parser for it. That is why this table never gates the wire
# verdict; it only names which specialized handler would match, for the log.
LLAMA_NATIVE_TOOL_FORMATS = (
    # Ministral is Mistral Large 3 minus the [CALL_ID] Mistral Small 3.2 carries.
    ("ministral_3", ("[SYSTEM_PROMPT]", "[TOOL_CALLS]", "[ARGS]"), ("[CALL_ID]",), ()),
    ("gpt_oss", ("<|channel|>",), (), ()),  # chat.cpp:1093-1096
    ("muse_glimmer", ("<atem:function_calls>", "<|eom|>"), (), ()),  # chat.cpp:1099-1102
    ("functionary_v3_2", (">>>all", ">>>${recipient}"), (), ()),  # chat.cpp:1106-1109
    (
        "kimi_k2",
        ("<|tool_calls_section_begin|>", "<|tool_call_begin|>"),
        (),
        (),
    ),  # chat.cpp:1113-1117
    ("kimi_k3", ("<|open|>", "<|close|>", "<|end_of_msg|>"), (), ()),  # chat.cpp:1120-1124
    # Cohere2 MoE: <|START_TEXT|> is what separates it from older Command-R.
    ("cohere2moe", ("<|START_TEXT|>", "<|START_ACTION|>"), (), ()),  # chat.cpp:1129-1133
    ("lfm2", ("<|tool_list_start|>", "<|tool_list_end|>"), (), ()),  # parsers/lfm2.cpp:5-8
    # LFM2.5 spells its tool list out instead of using LFM2's markers.
    ("lfm2_5", ("List of tools: [",), ("<|tool_list_start|>",), ()),  # chat.cpp:1141-1145
    (
        "gigachat_v3",
        ("<|role_sep|>", "<|message_sep|>"),
        ("<|function_call|>",),
        (),
    ),  # chat.cpp:1148-1153
    (
        "minimax_m3",
        ("]<]minimax[>[", "<tool_call>", "<invoke name="),
        (),
        (),
    ),  # chat.cpp:1157-1162
    # DeepSeek V3.2 names the block function_calls, V4 names it tool_calls.
    (
        "deepseek_v3_2",
        ("dsml_token", "DSML"),
        (),
        ("function_calls", "tool_calls"),
    ),  # chat.cpp:1165-1173
    ("gemma4", ("'<|tool_call>call:'",), (), ()),  # chat.cpp:1176-1184
    (
        "minicpm5",
        ("Tool usage guidelines:", '<function name="', '<param name="'),
        (),
        (),
    ),  # chat.cpp:1187-1192
    # Qwen3-Coder, also used by Nemotron Nano 3, Qwen3.5 and StepFun-3.5-Flash.
    (
        "qwen3_coder",
        ("<tool_call>", "<function=", "<parameter="),
        (),
        (),
    ),  # chat.cpp:1196-1201
)

# What llama.cpp falls back to when no specialized handler matches: it analyses
# the template and generates a parser from it (chat.cpp:1311-1345).
LLAMA_AUTOPARSER_FORMAT = "autoparser"


def native_tool_format_for_template(template: str) -> str:
    """Name the llama.cpp tool handler a template would match (logs only).

    First entry of ``LLAMA_NATIVE_TOOL_FORMATS`` the template satisfies, or
    ``"autoparser"`` -- chat.cpp's own last resort -- when none does. Purely
    informational: the wire verdict never reads this.
    """
    for format_name, required, forbidden, any_of in LLAMA_NATIVE_TOOL_FORMATS:
        if any(marker in template for marker in forbidden):
            continue
        if any_of and not any(marker in template for marker in any_of):
            continue
        if all(marker in template for marker in required):
            return format_name
    return LLAMA_AUTOPARSER_FORMAT


class BaseLlamaCppEngine(BaseChatServerEngine):
    """Shared scaffolding for engines that spawn `llama-server` via Popen."""

    # ====================== Overridable class attrs ======================
    # llama-server binds inside Erudi's canonical 271xx–273xx block: 27200–27299,
    # collision-free against MLX (27300–27399) and the backend HTTP server
    # (27182–27199). Deliberately off the historic 8080 default, which is the most
    # contested port around (Tomcat, and llama.cpp's own default). See backend/run.py
    # for why 271xx (digits of e, below every OS ephemeral range, IANA-unassigned).
    _port_range_start: ClassVar[int] = 27200

    # Subclass selects which artifact directory to look in.
    # False → `artifacts/llama-cpp/cpu/bin`, True → `artifacts/llama-cpp/cuda/bin`.
    _use_cuda_build: ClassVar[bool] = False

    # Every llama-cpp engine (CPU + CUDA) consumes pre-built **public** GGUF repos.
    # The catalog is built by searching filter="gguf" (any author) and resolving each
    # base id to its public GGUF repo — no hand-maintained mapping, token-free by
    # construction (the gated first-party safetensors is never a GGUF, so never seen).
    USES_GGUF: ClassVar[bool] = True
    FORMAT_TAG = "gguf"

    # ====================== Concrete shared methods ======================
    @classmethod
    def max_context_tokens(cls) -> Optional[int]:
        """The DECLARED CEILING for llama-server: ``ERUDI_CTX`` when the user
        pinned one (then passed verbatim as ``-c`` -- llama-server honours an
        explicit ``-c`` unchanged, "no change" contract), else ``None``.

        ``None`` means no window is pinned: the CPU engine then sizes ``-c``
        itself from the GGUF and the RAM budget (``gguf_memory``), and CUDA --
        or the CPU engine when the GGUF cannot be read -- passes no ``-c``, so
        the pinned llama.cpp build resolves the window at load (``--fit``, ON
        by default -- the model's trained window, continuously reduced against
        measured free memory down to a 4096 floor). The window the child
        allocated is read back through ``effective_context_tokens()`` after
        boot, never guessed here.

        Shared by the spawn context and the sampling resolver's
        ``max_tokens_cap`` (#388), both of which tolerate ``None``."""
        raw = os.environ.get("ERUDI_CTX", "").strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            # Degraded on purpose: a typo in a QA env var must not crash the
            # spawn; the engine falls back to its own fit.
            logger.warning(f"[{cls.__name__}] ERUDI_CTX is not an integer; ignoring it")
            return None

    @classmethod
    def _default_install_dir(cls) -> Path:
        """Resolve the directory that holds `llama-server` for this engine."""
        flavour = "cuda" if cls._use_cuda_build else "cpu"
        return ROOT_DIR / "artifacts" / "llama-cpp" / flavour / "bin"

    @classmethod
    def _find_llama_server(cls, install_dir: Optional[Path] = None) -> Path:
        """Return the absolute path of the `llama-server` binary, or raise.

        Tries the configured flavour first, then falls back to the other
        flavour (a CUDA-built artifact runs CPU inference fine; the CPU
        artifact would just refuse to use the GPU). This preserves the
        existing fallback behaviour from cpu_engine.py.
        """
        install = install_dir or cls._default_install_dir()
        exe = "llama-server.exe" if os.name == "nt" else "llama-server"
        primary = install / exe
        if primary.exists():
            return primary
        # Fallback to the other flavour. Said out loud: a CUDA engine that
        # silently starts the CPU binary runs every generation on the
        # processor with nothing in the log to explain the speed.
        other = "cpu" if cls._use_cuda_build else "cuda"
        fallback = ROOT_DIR / "artifacts" / "llama-cpp" / other / "bin" / exe
        if fallback.exists():
            logger.warning(
                f"[{cls.__name__}] llama-server not found at {primary}; "
                f"falling back to the {other} build at {fallback}"
            )
            return fallback
        raise EngineException(
            message=(
                f"llama-server binary not found at {primary} or {fallback}. "
                f"Build llama.cpp first (see scripts/dev/backend/build-llamacpp-*)."
            ),
        )

    @classmethod
    def _select_gguf(cls, llm_local_path: Union[str, Path]) -> Path:
        """Pick the best GGUF file from `llm_local_path` (file or directory).

        Priority when a directory contains multiple GGUFs:
        `q4_k_m > q4_0 > q5_k_m > q8_0 > f16`, then smallest file.
        """
        p = Path(llm_local_path).resolve()
        if not p.exists():
            raise EngineException(message=f"Model path not found: {p}")
        if p.is_file():
            if p.suffix.lower() != ".gguf":
                raise EngineException(
                    message=f"Expected a .gguf file. Got: {p}",
                )
            return p
        ggufs = [g for g in p.glob("*.gguf") if "mmproj" not in g.name.lower()]
        if not ggufs:
            raise EngineException(
                message=f"No .gguf found in {p}. Convert or quantize first.",
            )
        if len(ggufs) == 1:
            return ggufs[0]
        QUANT_PRIORITY = ["q4_k_m", "q4_0", "q5_k_m", "q8_0", "f16"]
        for quant in QUANT_PRIORITY:
            for gguf in ggufs:
                if quant in gguf.stem.lower():
                    logger.info(f"[{cls.__name__}] Selected {gguf.name} (quant: {quant})")
                    return gguf
        smallest = min(ggufs, key=lambda x: x.stat().st_size)
        logger.warning(f"[{cls.__name__}] No known quant pattern; using smallest: {smallest.name}")
        return smallest

    @classmethod
    def _find_mmproj(cls, model_gguf: Path) -> Optional[Path]:
        """Return the mmproj GGUF in the same directory as model_gguf, or None."""
        candidates = list(model_gguf.parent.glob("mmproj-*.gguf"))
        if not candidates:
            return None
        if len(candidates) > 1:
            logger.warning(
                f"[{cls.__name__}] Multiple mmproj files found, using {candidates[0].name}"
            )
        return candidates[0]

    @classmethod
    def _resolve_model_artifact(cls, llm_local_path: Union[str, Path]) -> Path:
        """For llama-cpp engines the artifact is a single GGUF file."""
        return cls._select_gguf(llm_local_path)

    @classmethod
    def validate_local_artifact(cls, llm_local_path: Union[str, Path]) -> None:
        """Integrity gate for a GGUF artifact (#88).

        A llama.cpp model is loadable iff it exposes one non-``mmproj`` ``.gguf``
        that is non-empty and carries the GGUF magic (``llama-server`` reads the
        tokenizer + chat template out of that container). Validates the exact
        file the engine would pick (``_select_gguf`` quant-priority), so the
        download gate and the load gate agree. Raises :class:`EngineException`
        with a curated, user-facing message on the first problem.
        """
        from src.engines import integrity

        # The message is curated for the user; the path goes in the trace so
        # the log record says which folder failed the check.
        path = Path(llm_local_path)
        if not path.exists():
            raise EngineException(
                message=integrity.incomplete_message("the model folder is missing"),
                trace=str(path),
            )
        if path.is_file():
            chosen = path
        else:
            ggufs = [g for g in path.glob("*.gguf") if not g.name.lower().startswith("mmproj")]
            if not ggufs:
                raise EngineException(
                    message=integrity.incomplete_message("no GGUF weights file was found"),
                    trace=str(path),
                )
            chosen = cls._select_gguf(path)
        integrity.validate_gguf_file(chosen)

    @classmethod
    def _load_capability_tokenizer(cls, llm_local_path: Union[str, Path]):
        """Chat-template view of the GGUF, for the static capability probes.

        Reads the template straight out of the GGUF key-value header and renders
        it with plain ``jinja2`` (see ``engines.gguf_chat_template``). It used to
        build a ``transformers.AutoTokenizer``, which pulled the whole
        ``modeling_auto`` import graph -- sklearn, scipy BLAS and their native
        DLLs -- and DEADLOCKED in the frozen Windows build whenever it ran off the
        main thread: the first chat turn against any GGUF model hung forever
        (#313), as did download finalization (#291). Nothing here imports
        transformers, so no native extension is loaded on a request path.

        Returns None when the artifact carries no readable template; every caller
        already treats that as "unknown" and keeps its graceful default.
        """
        from src.engines.gguf_chat_template import load_gguf_chat_template

        return load_gguf_chat_template(cls._select_gguf(llm_local_path))

    @classmethod
    def compute_wire_tools(cls, llm_local_path: Union[str, Path]) -> Optional[bool]:
        """Verified tool-call wire capability on llama-server (#298).

        Both GGUF engines spawn ``llama-server`` with ``--jinja``
        (cpu_engine.py / cuda_engine.py), so llama.cpp's chat dispatch applies:
        a chat template matched by a specialized handler gets that handler, and
        ANY other usable template is handed to the differential autoparser,
        which reads it and generates a parser (forks/llama-cpp/common/chat.cpp,
        common_chat_templates_apply_jinja). Structured tool handling is
        therefore available whenever the model has a usable chat template at
        all: template present -> True.

        The verdict stays permissive on purpose. The autoparser can refuse a
        template it cannot analyse, and that surfaces as an error on the request
        rather than as a capability we could read here. Answering False on a
        template we merely failed to recognise would disable tools for a model
        that works, which is the worse failure of the two.

        The mirrored specialized-format table (``LLAMA_NATIVE_TOOL_FORMATS``) is
        consulted for the LOG only — which handler would match — never for the
        verdict. No template -> False (llama-server would fall back to its
        legacy non-jinja path); unreadable artifact -> None (unverified).
        """
        try:
            tokenizer = cls._load_capability_tokenizer(llm_local_path)
        except Exception:
            logger.warning(
                f"[{cls.__name__}] wire tool detection: could not load a "
                f"tokenizer for {llm_local_path}",
                exc_info=True,
            )
            return None
        template = getattr(tokenizer, "chat_template", None)
        if not template or not isinstance(template, str):
            logger.info(
                f"[{cls.__name__}] wire tools NOT verified for {llm_local_path}: "
                f"no chat template in the GGUF"
            )
            return False
        native_format = native_tool_format_for_template(template)
        logger.info(
            f"[{cls.__name__}] wire tools verified for {llm_local_path}: "
            f"--jinja tool handler={native_format}"
        )
        return True

    @classmethod
    def model_supports_vision(cls, llm_local_path: Union[str, Path]) -> Optional[bool]:
        """A llama.cpp model is vision-capable iff it ships an ``mmproj`` projector.

        That is exactly the file the engine passes to ``llama-server --mmproj``
        (#130). No artifact / unreadable directory -> ``None`` (permissive).
        """
        try:
            gguf_path = cls._select_gguf(llm_local_path)
            return cls._find_mmproj(gguf_path) is not None
        except Exception:
            logger.warning(
                f"[{cls.__name__}] vision detection failed for {llm_local_path}", exc_info=True
            )
            return None

    @classmethod
    def _terminate_process(cls, proc: Any) -> None:
        """Idempotent terminate for `subprocess.Popen`.

        macOS/Linux: SIGINT → wait 5s → SIGKILL.
        Windows: terminate → wait 5s → kill.
        """
        if not proc:
            return
        try:
            if proc.poll() is None:
                if platform.system() == "Windows":
                    proc.terminate()
                else:
                    proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=5)
                except Exception:
                    # It ignored the polite signal for 5s: a server stuck in a
                    # native call. Killed, and said so -- a child that has to
                    # be killed on every swap is a symptom worth a record.
                    logger.warning(
                        f"[{cls.__name__}] llama-server pid {getattr(proc, 'pid', '?')} "
                        f"did not exit within 5s of the stop signal; killing it"
                    )
                    proc.kill()
        except Exception as exc:
            # poll/signal/kill raising means the handle is already gone (the
            # process exited and was reaped, or the pid is invalid): there is
            # nothing left to terminate, so this is not a failure.
            logger.debug(f"[{cls.__name__}] terminate skipped: {type(exc).__name__}: {exc}")

    @classmethod
    def _proc_is_alive(cls, proc: Any) -> bool:
        """Whether the Popen child is still running."""
        if proc is None:
            return False
        try:
            return proc.poll() is None
        except Exception:
            return False

    # The drainer is stored on the Popen object itself rather than in a
    # module-level registry: its lifetime is then exactly the child's, with no
    # cleanup to forget and no chance of a recycled pid handing out another
    # child's output.
    _DRAINER_ATTR: ClassVar[str] = "erudi_output_drainer"

    # How much of the tail to quote in a crash message. llama-server's banner
    # and GGUF metadata dump are long; the reason it died is in the last lines.
    _CHILD_OUTPUT_TAIL_CHARS: ClassVar[int] = 2000

    @classmethod
    def _attach_output_drainer(cls, proc: Any, drainer: ChildOutputDrainer) -> None:
        setattr(proc, cls._DRAINER_ATTR, drainer)

    @classmethod
    def _output_drainer_for(cls, proc: Any) -> Optional[ChildOutputDrainer]:
        return getattr(proc, cls._DRAINER_ATTR, None)

    @classmethod
    def _read_child_output(cls, proc: Any) -> str:
        """Tail of the child's merged stdout+stderr, as collected by the drainer.

        Unlike reading `proc.stdout` here directly, this works whether or not
        the child has exited -- the drainer has been consuming the pipe since
        the moment the child was spawned (#361).
        """
        drainer = cls._output_drainer_for(proc) if proc is not None else None
        if drainer is None:
            return "No child output was captured."
        tail = drainer.tail(max_chars=cls._CHILD_OUTPUT_TAIL_CHARS)
        if not tail:
            return "The child produced no output."
        return f"Child output (last {len(tail)} chars):\n{tail}"

    @classmethod
    def _translate_payload_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Translate from Erudi vocabulary (HF/transformers) to llama-server names."""
        renames = {
            "repetition_penalty": "repeat_penalty",
            "repetition_context_size": "repeat_last_n",
        }
        out = {renames.get(k, k): v for k, v in kwargs.items() if k != "enable_thinking"}
        # llama-server has no top-level ``enable_thinking`` field (#266): it
        # forwards ``chat_template_kwargs`` to the Jinja chat template instead,
        # and templates without the kwarg ignore it harmlessly.
        if "enable_thinking" in kwargs:
            out["chat_template_kwargs"] = {"enable_thinking": kwargs["enable_thinking"]}
        return out

    # Bound on the one post-probe `/props` read. The server just answered the
    # readiness probe, so 2s of silence here means something is wrong enough
    # that degrading (window unknown) beats stalling the load.
    _PROPS_TIMEOUT_S: ClassVar[float] = 2.0

    @classmethod
    def _read_server_properties(cls, handle: Dict[str, Any]) -> None:
        """One bounded `GET /props` after the probe: what llama-server ALLOCATED.

        llama-server allocates its context window AT LOAD -- the ``-c`` the
        engine passed (the CPU engine's computed window, or ``ERUDI_CTX``,
        kept unchanged), or, with no ``-c``, its own fit: the model's trained
        window, continuously reduced against measured free memory (pinned
        b10883, ``common/fit.cpp``). The allocated value is read back from
        the running child:

        * ``context_tokens`` = ``default_generation_settings.n_ctx`` -- the
          allocated window every percentage, compaction threshold and watchdog
          budget downstream must use (a value other than the ``-c`` passed is
          one WARNING);
        * ``chat_template_caps`` -- notably ``supports_reasoning_effort``,
          read by the reasoning-effort mapping;
        * ``chat_template`` -- the template the server actually applies
          (per-spawn memory, possibly large; never persisted).

        ANY failure (timeout, non-200, unreadable body, missing keys) degrades:
        one WARNING, and the model finishes loading; ``context_tokens`` is
        then the ``-c`` passed, or ``None`` when llama-server's fit chose it.

        An engine with a memory brake (CPU) first seeds the child's brake --
        the episode an unload of the same model carried, else the cap and
        faulty window remembered for its alias, else a fresh state -- and
        after the read stamps its memory bound and checks its anonymous
        memory against the window budget (`_llama_check_footprint`).
        MLX overrides this hook -- its window is a preflight bound stamped at
        spawn, there is nothing to read back (see `MLX_Engine`).
        """
        braked = cls._brake_bound()
        if braked:
            cls._llama_seed(handle)
        passed = handle.get("ctx_size")
        handle["context_tokens"] = None
        handle["chat_template_caps"] = {}
        handle["chat_template"] = None
        api_key = handle.get("api_key")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        try:
            resp = requests.get(
                f"{handle['base_url']}/props", timeout=cls._PROPS_TIMEOUT_S, headers=headers
            )
            if resp.status_code != 200:
                raise ValueError(f"HTTP {resp.status_code}")
            props = resp.json()
            settings = props.get("default_generation_settings")
            n_ctx = settings.get("n_ctx") if isinstance(settings, dict) else None
            if not isinstance(n_ctx, int) or isinstance(n_ctx, bool) or n_ctx <= 0:
                raise ValueError("default_generation_settings.n_ctx missing or invalid")
            caps = props.get("chat_template_caps")
            template = props.get("chat_template")
            handle["context_tokens"] = n_ctx
            handle["chat_template_caps"] = caps if isinstance(caps, dict) else {}
            handle["chat_template"] = template if isinstance(template, str) else None
            logger.info(
                f"[{cls.__name__}] {cls._server_name} allocated context window: "
                f"{n_ctx} tokens (port {handle.get('port')})"
            )
            if isinstance(passed, int) and n_ctx != passed:
                logger.warning(
                    f"[{cls.__name__}] {cls._server_name} allocated {n_ctx} tokens "
                    f"where -c {passed} was passed (port {handle.get('port')})"
                )
        except Exception as exc:
            # Degraded, not failed: the model is loaded and usable. With a -c
            # the window is the one we asked for; without one (llama-server's
            # fit) it is unknown, which downstream treats as unknown.
            if isinstance(passed, int):
                handle["context_tokens"] = passed
            logger.warning(
                f"[{cls.__name__}] Could not read {cls._server_name} /props on "
                f"port {handle.get('port')}; "
                + (
                    f"using the window passed at spawn, {passed} tokens "
                    if isinstance(passed, int)
                    else "allocated context window unknown "
                )
                + f"({type(exc).__name__}: {exc})"
            )
        if braked:
            handle["memory_bound_tokens"] = (
                handle["context_tokens"] if handle.get("spawn_memory_bound") else None
            )
            cls._llama_check_footprint(handle)

    @classmethod
    def _spawn_child(
        cls,
        *,
        model_path: Path,
        alias: str,
        port: int,
        **ctx: Any,
    ) -> Dict[str, Any]:
        """Spawn `llama-server` via Popen. Subclasses inject CLI/env via hooks.

        Hooks called by this method:
        - `_build_spawn_argv(*, llama_server, model_gguf, alias, port, **ctx)`
        - `_build_spawn_env()`
        """
        install_dir = cls._default_install_dir()
        llama_server = cls._find_llama_server(install_dir)
        argv = cls._build_spawn_argv(
            llama_server=llama_server,
            model_gguf=model_path,
            alias=alias,
            port=port,
            **ctx,
        )
        mmproj = cls._find_mmproj(model_path)
        if mmproj:
            argv += ["--mmproj", str(mmproj)]
            logger.info(f"[{cls.__name__}] Vision projector found: {mmproj.name}")
        # Close the loopback port to everything but us. Spawned without
        # `--api-key`, llama-server authenticates NOTHING: every endpoint answers
        # any caller that can reach 127.0.0.1 -- another local process, or a web
        # page the user has open, since a browser can POST across origins to a
        # loopback port. `/slots` (on by default) then hands that caller the
        # prompt of every in-flight request, i.e. what the user is asking the
        # model right now, and `/v1/chat/completions` lets it run its own
        # inference on the user's machine. The key is minted per spawn so a
        # disclosure dies with the child; `/slots` and the bundled web UI are
        # switched off outright because Erudi calls neither (only `/health` and
        # `/v1/chat/completions`). Hex, never URL-safe base64: a key that starts
        # with "-" would be read as an option by the server's argument parser.
        api_key = secrets.token_hex(32)
        argv += ["--api-key", api_key, "--no-slots", "--no-webui"]
        # One slot and no prompt cache: the child holds a single conversation's
        # KV at a time, like the MLX engine's single-conversation prefix cache.
        # Upstream defaults are four slots and `--cache-ram 8192`, a host-RAM
        # cache of past prompts' full KV state of up to 8 GiB that nothing in
        # Erudi counts (each distinct long conversation added about 1.3 GiB to
        # the child and pushed the machine into swap). The current
        # conversation's follow-up still reuses its prefix inside the slot; going
        # back to an older conversation re-reads it once. `-np 1` turns unified
        # KV off; with one sequence the slot's window is the whole `n_ctx`
        # (`n_ctx_seq = n_ctx / 1`), so the window and `/props` `n_ctx` are
        # unchanged; the sliding-window cache of models such as Gemma 2/3
        # shrinks, since it no longer reserves room for four sequences.
        # `--no-cache-idle-slots` states the intent outright: idle-slot caching
        # needs the RAM cache, and llama-server otherwise warns at every spawn
        # that it disables it.
        argv += ["-np", "1", "--cache-ram", "0", "--no-cache-idle-slots"]
        env = cls._build_spawn_env()
        try:
            proc = subprocess.Popen(
                [str(a) for a in argv],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                # A byte the child writes that is not UTF-8 (a GGUF metadata
                # dump can carry one) must degrade to U+FFFD, not raise inside
                # the drainer and leave the pipe unread (#361).
                errors="replace",
                bufsize=1,
                env=env,
                # llama-server is a console exe: without this it opens its own
                # terminal window on Windows when the backend's console isn't
                # inheritable (#175). No-op (0) on POSIX.
                creationflags=hidden_console_creationflags(),
            )
        except OSError as exc:
            # The binary is missing, lost its executable bit in the freeze, or
            # a DLL beside it is absent (Windows reports that as an OSError
            # too). Named here, because a bare OSError from Popen does not say
            # which file it was trying to run.
            raise EngineException(
                message=(
                    f"Could not start llama-server at {llama_server}: " f"{exc.strerror or exc}"
                ),
                trace=f"{type(exc).__name__}: {exc}",
            ) from exc
        # Start draining immediately: llama-server writes its banner and the
        # GGUF metadata dump before it is ever ready, and an unread pipe would
        # wedge it mid-startup once full (#361).
        cls._attach_output_drainer(
            proc, ChildOutputDrainer(proc.stdout, name=f"{cls._server_name}:{proc.pid}")
        )
        handle: Dict[str, Any] = {
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
        # Preserve subclass-relevant context items in the handle: observability
        # (how many threads / GPU layers, which window) and the window facts the
        # memory brake reads back (`PRESERVED_SPAWN_KEYS`).
        for k in PRESERVED_SPAWN_KEYS:
            if k in ctx:
                handle[k] = ctx[k]
        return handle

    # ======================= THE WINDOW AND THE MEMORY BRAKE =======================
    #
    # Bound on CPU_Engine only (CUDA's KV sits partly or wholly in VRAM, which
    # a RAM budget would mis-size): every hook the runner gates is a `_llama_*`
    # body here and a binding on CPU_Engine, so CUDA inherits nothing the
    # runner would call.
    #
    # * The window: at spawn, `_llama_spawn_context` asks `gguf_memory` for
    #   min(trained window, what 0.75 x RAM affords) folded with the cap the
    #   brake remembers for the alias, and passes it as `-c`.
    # * The brake (`memory_brake.llama_step`) samples at the start of a turn,
    #   its end and idle (the shared monitor), under the generation lock:
    #   - respawn: the same model restarts in place with `-c = min(budget
    #     window, cap)`, the live brake state carried whole; the respawn's
    #     probe is bounded by `LLAMA_RESPAWN_PROBE_TIMEOUT_S`, and a respawn
    #     that fails degrades into an unload (the record below);
    #   - unload: the reap path, the episode carried to the next spawn of the
    #     same model through `_brake_reload` (the MLX shape);
    #   - the per-alias window memory: every change of the brake's cap or
    #     faulty window is written through to `_window_memory`, read (TTL
    #     checked) at every spawn of that alias -- a brake reload, a swap
    #     back, a reap's reload, a crash respawn -- so a grow or a shrink the
    #     brake decided applies at the next spawn. A reap forgets
    #     `_brake_reload`, never the window memory.

    # alias -> {"cap", "faulty", "at"}; written by CPU only.
    _window_memory: ClassVar[Dict[str, Dict[str, Any]]] = {}

    @classmethod
    def _window_memory_entry(cls, alias: str) -> Optional[Dict[str, Any]]:
        """The remembered cap and faulty window of ``alias``, younger than
        ``BRAKE_RELOAD_TTL_S`` (an expired entry is deleted here)."""
        entry = cls._window_memory.get(alias)
        if not isinstance(entry, dict):
            return None
        at = entry.get("at")
        if not isinstance(at, (int, float)) or time.time() - at >= memory_brake.BRAKE_RELOAD_TTL_S:
            cls._window_memory.pop(alias, None)
            return None
        return entry

    @classmethod
    def _remember_window(cls, alias: Any, state: BrakeState) -> None:
        """Write ``state``'s cap and faulty window through for ``alias``; an
        entry with neither is deleted."""
        if not isinstance(alias, str):
            return
        if state.cap is None and state.faulty is None:
            cls._window_memory.pop(alias, None)
            return
        cls._window_memory[alias] = {"cap": state.cap, "faulty": state.faulty, "at": time.time()}

    @classmethod
    def _llama_spawn_context(cls, model_path: Path, alias: str) -> Dict[str, Any]:
        """The window items of the spawn context: `-c` (`ctx_size`) and the
        facts the brake and the post-load check read back; `ctx_size=None`
        (llama-server's own fit) when the window cannot be sized --
        `gguf_memory.spawn_window` wrote the one WARNING."""
        entry = cls._window_memory_entry(alias)
        mmproj = cls._find_mmproj(model_path)
        window = gguf_memory.spawn_window(
            model_path,
            total_ram_bytes=memory_pressure.effective_total_ram_bytes(),
            brake_cap=entry.get("cap") if entry else None,
            mmproj_path=mmproj,
        )
        if window is None:
            return {"ctx_size": None}
        logger.info(
            f"[{cls.__name__}] Window sized at spawn: window={window.window}, "
            f"trained={window.trained}, memory_bound={window.memory_bound}"
            + (f", brake_cap={entry.get('cap')}" if entry and entry.get("cap") else "")
        )
        return {
            "ctx_size": window.window,
            "trained_context_tokens": window.trained,
            "static_window_tokens": window.static_window,
            "kv_token_bytes": window.kv_token_bytes,
            "static_anonymous_bytes": window.static_anonymous_bytes,
            "weights_bytes": window.weights_bytes,
            "spawn_memory_bound": window.memory_bound,
            "mmproj_loaded": mmproj is not None,
        }

    @classmethod
    def _llama_seed(cls, handle: Dict[str, Any]) -> None:
        """The new child's brake -- the episode an unload of the same model
        carried, else its remembered cap and faulty window, else a fresh
        state -- and its first paging reading. A successful spawn forgets the
        unload's record, whichever model it loads."""
        state = cls._carried_brake(handle)
        if state is None:
            entry = cls._window_memory_entry(handle.get("alias"))
            state = (
                BrakeState(cap=entry.get("cap"), faulty=entry.get("faulty"))
                if entry
                else BrakeState()
            )
        handle["brake"] = state
        cls._brake_reload = None
        handle["damage_last"] = memory_pressure.read_pressure().damage_total_bytes

    @classmethod
    def _llama_check_footprint(cls, handle: Dict[str, Any]) -> None:
        """Right after load: the child's anonymous memory against what the
        window predicts (KV, the compute buffer, one weights copy for the
        repack), ONE WARNING above it -- a Flash-Attention fallback or a
        mis-sized model. Never an action; never raises."""
        try:
            if handle.get("mmproj_loaded"):
                logger.debug(
                    f"[{cls.__name__}] Post-load memory check skipped: a vision projector "
                    f"is loaded (its compute buffer is not modelled)"
                )
                return
            window = handle.get("context_tokens")
            kv = handle.get("kv_token_bytes")
            weights = handle.get("weights_bytes")
            if not (window and kv and weights):
                return
            measured = memory_pressure.child_anonymous_bytes(handle.get("pid"))
            if measured is None:
                return
            limit = gguf_memory.post_load_limit_bytes(
                window=window, kv_token_bytes=kv, weights_bytes=weights
            )
            if measured > limit:
                logger.warning(
                    f"[{cls.__name__}] llama-server allocated more than the window budget "
                    f"predicted: anonymous_bytes={measured}, predicted_limit_bytes={limit}, "
                    f"window={window} (port {handle.get('port')})"
                )
        except Exception as exc:
            # The check is advisory: the model is loaded and serves.
            logger.warning(f"[{cls.__name__}] Post-load memory check failed ({type(exc).__name__})")

    @classmethod
    def _llama_memory_bound_tokens(cls) -> Optional[int]:
        """The window the machine's memory limits the loaded child to (it is
        its allocated window), or None."""
        handle = cls._model
        if not isinstance(handle, dict) or handle.get("context_tokens") is None:
            return None
        bound = handle.get("memory_bound_tokens")
        if isinstance(bound, int) and not isinstance(bound, bool) and bound > 0:
            return bound
        return None

    @classmethod
    def _llama_memory_bound_footprint_bytes(cls) -> Optional[int]:
        """What a memory-bound child holds: its window's KV and compute, and
        its weights -- the number the ``bound`` warning quotes."""
        if cls._llama_memory_bound_tokens() is None:
            return None
        handle = cls._model
        anonymous, weights = handle.get("static_anonymous_bytes"), handle.get("weights_bytes")
        if not isinstance(anonymous, int) or not isinstance(weights, int):
            return None
        return anonymous + weights

    @classmethod
    def _llama_begin_memory_window(cls) -> None:
        """A turn starts on the loaded child: the brake sees it, and the flag
        of a brake respawn is consumed (the turn that shows the reload state
        for it has begun). Nothing is measured: no token."""
        handle = cls._model
        if not isinstance(handle, dict):
            return None
        state = handle.get("brake")
        if isinstance(state, BrakeState):
            state.turn_happened = True
        handle.pop("brake_respawned", None)
        return None

    @classmethod
    def _llama_respawn_pending(cls, llm_id: Any) -> bool:
        """The loaded child for ``llm_id`` was respawned by the brake since
        its last turn began: that turn's reload state is still owed."""
        return cls.loaded_child_for(llm_id) and bool(cls._model.get("brake_respawned"))

    @classmethod
    def _llama_brake_point(cls, point: str, llm_id: Any) -> BrakeOutcome:
        """Take the brake's sample at ``point`` and execute its decision
        (the caller holds the generation lock). May raise: the binding
        guards it."""
        if point == memory_brake.POINT_TURN_START and not cls.loaded_child_for(llm_id):
            return BrakeOutcome()
        handle = cls._model
        if not isinstance(handle, dict) or not cls._proc_is_alive(handle.get("proc")):
            return BrakeOutcome()
        reading = memory_pressure.read_pressure()
        if reading.level is None:
            if not cls._brake_sensor_warned:
                cls._brake_sensor_warned = True
                logger.warning(
                    f"[{cls.__name__}] Memory pressure unreadable "
                    f"({reading.unavailable}); the memory brake is off"
                )
            return BrakeOutcome()
        total = memory_pressure.effective_total_ram_bytes()
        # What the child holds, without the mapped weights' clean pages.
        footprint = memory_pressure.child_anonymous_bytes(handle.get("pid"))
        window = handle.get("context_tokens")
        kv = handle.get("kv_token_bytes")
        current, last = reading.damage_total_bytes, handle.get("damage_last")
        state = handle.get("brake")
        previous = state if isinstance(state, BrakeState) else BrakeState()
        decision = memory_brake.llama_step(
            previous,
            memory_brake.Sample(
                point=point,
                now=time.time(),
                level=reading.level,
                last_used=(
                    cls._last_used.timestamp() if isinstance(cls._last_used, datetime) else None
                ),
                damage_bytes=(
                    max(0, current - last) if current is not None and last is not None else None
                ),
                damage_available=current is not None,
            ),
            memory_brake.Context(
                window=window,
                ceiling=None,
                static_window=handle.get("static_window_tokens"),
                footprint_share=footprint / total if footprint is not None and total else None,
                resizable=not handle.get("window_pinned"),
                kv_share=window * kv / total if window and kv and total else None,
            ),
        )
        handle["brake"] = decision.state
        if not decision.skipped:
            # A paced-out sample keeps the paging of its interval for the next.
            handle["damage_last"] = current
        if (decision.state.cap, decision.state.faulty) != (previous.cap, previous.faulty):
            cls._remember_window(handle.get("alias"), decision.state)
        if decision.unload:
            cls._llama_brake_unload(handle, decision, point, footprint)
            return BrakeOutcome(acted=True, action="unload")
        if decision.respawn:
            if not cls._llama_brake_respawn(handle, decision, point, footprint):
                return BrakeOutcome(acted=True, action="respawn-failed")
            return BrakeOutcome(acted=True, action="+".join(decision.actions))
        if decision.cap_change is not None:
            cls._log_llama_brake(
                logger.info,
                point,
                decision.level,
                decision.cap_change,
                window,
                window,
                footprint,
                f", cap={decision.state.cap}",
            )
            return BrakeOutcome(acted=True, action=decision.cap_change)
        return BrakeOutcome()

    @classmethod
    def _llama_brake_respawn(
        cls,
        handle: Dict[str, Any],
        decision: "memory_brake.Decision",
        point: str,
        footprint: Optional[int],
    ) -> bool:
        """Restart the same model in place with the cap as its window (the
        window memory hands it to the spawn context). The live state is
        carried whole; ``_model_id``, the tokenizer and the idle clock are
        untouched (the reap must still see an abandoned child). False when
        the respawn failed: the child is stopped, the engine state cleared
        and the episode recorded for the next message, like an unload -- ONE
        WARNING."""
        live = decision.state
        before = handle.get("context_tokens")
        try:
            cls._stop_server_if_running()
            port = cls._pick_free_port()
            new = cls._start_server(
                model_path=Path(handle["model_path"]),
                alias=handle["alias"],
                port=port,
                probe_timeout_s=LLAMA_RESPAWN_PROBE_TIMEOUT_S,
            )
        except Exception as exc:
            record = {"alias": handle.get("alias"), **live.carry(), "unloaded_at": time.time()}
            cls._log_llama_brake(
                logger.warning,
                point,
                decision.level,
                f"respawn-failed ({type(exc).__name__})",
                before,
                None,
                footprint,
                "; the model reloads at the next message",
            )
            cls.cleanup()
            cls._brake_reload = record
            return False
        new["brake"] = BrakeState.respawned(live)
        new["brake_respawned"] = True
        cls._model = new
        cls._log_llama_brake(
            logger.info,
            point,
            decision.level,
            "+".join(decision.actions),
            before,
            new.get("context_tokens"),
            footprint,
        )
        return True

    @classmethod
    def _llama_brake_unload(
        cls,
        handle: Dict[str, Any],
        decision: "memory_brake.Decision",
        point: str,
        footprint: Optional[int],
    ) -> None:
        """Reap the child (the caller holds the generation lock) and leave
        the episode for the next spawn of the same model. ONE WARNING: the
        app degraded on its own."""
        record = {
            "alias": handle.get("alias"),
            **decision.state.carry(),
            "unloaded_at": time.time(),
        }
        cls._remember_window(handle.get("alias"), decision.state)
        window = handle.get("context_tokens")
        cls._log_llama_brake(
            logger.warning,
            point,
            decision.level,
            "unload",
            window,
            window,
            footprint,
            "; the model reloads at the next message",
        )
        cls.cleanup()
        cls._brake_reload = record

    @classmethod
    def _log_llama_brake(
        cls, record, point, level, action, before, after, footprint, suffix=""
    ) -> None:
        mib = footprint // (1024 * 1024) if isinstance(footprint, int) else None
        record(
            f"[{cls.__name__}] Memory brake: point={point}, level={level}, action={action}, "
            f"window={before}->{after}, footprint_mib={mib}{suffix}"
        )

    # ====================== Abstract subclass hooks ======================
    @classmethod
    @abstractmethod
    def _build_spawn_argv(
        cls,
        *,
        llama_server: Path,
        model_gguf: Path,
        alias: str,
        port: int,
        **ctx: Any,
    ) -> List[Any]:
        """Build the CLI for `llama-server`. CPU forces `-ngl 0`, CUDA injects
        `-ngl <gpu_layers>` from `_prepare_spawn_context`."""

    @classmethod
    def _build_spawn_env(cls) -> Dict[str, str]:
        """Per-spawn environment. Default: inherit the parent env unchanged.
        CUDA overrides to prepend the CUDA toolkit bin to `PATH` so the
        runtime DLLs resolve.
        """
        return os.environ.copy()
