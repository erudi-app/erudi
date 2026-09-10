"""Abstract sub-base for engines that wrap the `llama-server` binary.

Sits between `BaseChatServerEngine` and the concrete `CPU_Engine` /
`CUDA_Engine` classes. Factors the bits CPU and CUDA share but MLX does
not:

- Where the binary lives (`backend/artifacts/llama-cpp/<cpu|cuda>/bin/llama-server`)
- How to find / pick the GGUF file in a model directory
- The `subprocess.Popen` lifecycle (terminate, alive check, output draining)
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
from abc import abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Union

from src.core.config import ROOT_DIR
from src.core.exceptions import EngineException
from src.core.logging import logger
from src.engines.base_chat_server_engine import BaseChatServerEngine
from src.engines.child_output import ChildOutputDrainer
from src.core.subprocess_flags import hidden_console_creationflags


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
    def max_context_tokens(cls) -> int:
        """llama-server runs with a fixed context window (``-c``): 4096 tokens by
        default, ``ERUDI_CTX`` to override. Shared by the spawn context and the
        sampling resolver's ``max_tokens_cap`` (#388)."""
        return int(os.environ.get("ERUDI_CTX", "4096"))

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
        # `/v1/chat/completions`).
        api_key = secrets.token_urlsafe(32)
        argv += ["--api-key", api_key, "--no-slots", "--no-webui"]
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
        # Preserve subclass-relevant context items in the handle so observability
        # (logs, debug endpoints) shows e.g. how many threads / GPU layers were used.
        for k in ("threads", "gpu_layers", "ctx_size"):
            if k in ctx:
                handle[k] = ctx[k]
        return handle

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
