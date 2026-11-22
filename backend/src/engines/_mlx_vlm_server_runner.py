"""Picklable child-process entry point for `mlx_vlm.server`.

Why a dedicated module
----------------------
The MLX engine spawns the OpenAI-compatible `mlx_vlm.server` HTTP server in a
separate process via `multiprocessing.Process(target=..., args=([argv],))`.
The `target` argument MUST be a top-level, importable, picklable function:

  - A lambda or a bound classmethod cannot be pickled by the `spawn` start
    method.
  - `spawn` is the only start method that works inside a PyInstaller frozen
    binary (where `sys.executable` is the launcher itself, not a Python
    interpreter). On the parent side, `mp.freeze_support()` and
    `set_start_method("spawn", force=True)` are configured in
    `backend/run.py` and `backend/tests/conftest.py`; the child reconstitutes
    the import graph and calls this function.

The two-function split (`_import_mlx_vlm_server_main` + `run_mlx_vlm_server`)
keeps the heavy `mlx_vlm.server` import lazy and — critically — patchable from
unit tests that run on Linux CI where `mlx-vlm` is not installed.

In-child patches (pinned mlx-vlm 0.6.17)
----------------------------------------
Five monkeypatches are applied before the server starts (`_IN_CHILD_PATCHES`),
each with the upstream version that makes it unnecessary:

  - `_patch_gemma3_tied_lm_head_quant`: quantized tied-embedding Gemma3
    checkpoints load (#273). Still needed.
  - `_patch_turn_end_stop_tokens`: generation stops on the tokenizer's eos and
    on the named end-of-turn tokens (`TURN_END_TOKENS`), never on gpt-oss's
    `<|end|>` (#249, #721). The eos half is upstream from 0.7.0
    (Neywa-Labs/mlx-vlm PR #2112); the named tokens are still needed on 0.8.0.
  - `_patch_json_tool_call_tolerance`: a tool call written with the doubled
    braces the Qwen2.5 template shows parses (#721). Still needed on 0.8.0.
  - `_patch_granite_decode_embedding_multiplier`: Granite multiplies its
    embeddings while decoding too (#721). Still needed on 0.8.0.
  - `_patch_internlm2_output_head_sanitize`: InternLM2 checkpoints load their
    `output.*` head (#721). Still needed on 0.8.0.

The server's native reasoning split is deliberately NOT patched: mlx-vlm
streaming chain-of-thought in the dedicated `delta.reasoning` field is the
wanted behavior (#554) —
`Erudi_Chat_OpenAI` (`src.agents.chat_model`) carries the field to the
runner's `thinking` events, so the old `_patch_inline_thinking` (which forced
reasoning inline into `delta.content` for the runner's ThinkSplitter) is gone.
Two 0.6.2-era patches were dropped with the 0.6.13 bump because upstream now
runs weight sanitization unconditionally in `mlx_vlm.utils.load_model` (the
0.6.2 `format == "mlx"` sanitize skip is gone) — but hardware validation
showed the static evidence for the first drop was incomplete:

  - `_patch_text_only_tied_embeddings` (dropped): the `text_only` route IS
    fixed upstream (`models/text_only.py` `Model.sanitize` delegates to the
    inner mlx-lm model). BUT Gemma3 text-only checkpoints no longer take that
    route: 0.6.13 ships a native `models/gemma3_text` module that shadows the
    `text_only` fallback, and its tied-head sanitize is quantization-unaware.
    The adapted `_patch_gemma3_tied_lm_head_quant` (below) covers that
    regression, proven on hardware with `mlx-community/gemma-3-270m-it-4bit`.
  - `_patch_gemma_shared_kv_sanitize` (dropped, #193): `models/gemma4/language.py`
    `LanguageModel.sanitize` drops the `_is_unused_shared_kv_weight` tensors
    and is invoked on every load via the unconditional class-level pass
    (`utils.py:717-721`): `gemma4/__init__.py` exports `LanguageModel` and
    `ModelConfig.text_config` exists, so the pass fires for gemma4-family
    checkpoints. Unlike gemma3, gemma4 heads are tied by construction
    (`embed_tokens.as_linear`, no `lm_head` parameter), so the gemma3_text
    quantized-head blind spot has no analogue there.

Output capture
--------------
An `mp.Process` gives the parent no pipe to drain, so the child captures
itself: `run_mlx_vlm_server` redirects its stdout and stderr into the
per-spawn file the parent resolved (`mlx_child_log`) before anything else
runs. mlx-vlm's `logging.basicConfig`, uvicorn's handlers and every native
write from mlx/Metal then land there, and `MLX_Engine._read_child_output`
quotes its tail in a crash report -- the same diagnostic `llama-server` gets
from its drainer. The patches below report a failure to apply into that
same file: each compensates for a defect the user meets head-on, so an
mlx-vlm bump that moves what they patch must not pass unnoticed.

Contract
--------
`run_mlx_vlm_server(argv, log_path, apc_num_blocks, token_queue_timeout_s)`
replaces `sys.argv` with the supplied list and calls the real
`mlx_vlm.server.cli.main()`. The first element of `argv` is the conventional
program name; the rest are the CLI flags that mlx-vlm's argparse expects
(`--model`, `--host`, `--port`, `--log-level`, ...). `main()` parses them,
exports the matching env vars (e.g. `MLX_VLM_PRELOAD_MODEL` from `--model`), and
launches `uvicorn.run("mlx_vlm.server:app", ...)`.

The last two parameters are the runtime knobs this build turns on, applied to
the child's own `os.environ` BEFORE the server is imported (mlx-vlm reads them
at import/init): `apc_num_blocks` sizes the Automatic Prefix Caching pool so a
2nd+ turn prefills only the suffix of the conversation, and
`token_queue_timeout_s` lifts the child's per-token wait above the parent's
first-chunk watchdog ceiling so a long cold prefill is owned by the parent's
curated turn. Both default to `None` (mlx-vlm's own defaults) for callers and
tests that only pass `argv`/`log_path`. The same helper always caps the running
batch at one sequence (`MLX_VLM_MAX_NUM_SEQS=1`), which the prefix-cache
barrier relies on (see `_apply_child_runtime_env`).

The child also bounds MLX's buffer cache at 0 (``MLX_BUFFER_CACHE_LIMIT_BYTES``,
see ``_bound_buffer_cache``) before the server is imported.

Once invoked, this function blocks for the lifetime of the HTTP server,
exiting only when the child process is terminated by the parent.
"""

from __future__ import annotations

import functools
import inspect
import json
import os
import time
from typing import List, Optional

from src.engines.mlx_child_log import child_warning, redirect_stdio_to, write_child_record

# Automatic Prefix Caching (APC) block size, in tokens. The parent sizes the
# APC pool to cover the child's allocated window as ``ceil(window / this)``
# (``src.engines.mlx_engine._spawn_child``), so the two MUST agree -- this is
# the single source of truth for the block size, exported to the parent.
APC_BLOCK_SIZE_TOKENS = 16

# Ceiling, in bytes, on the pool of freed GPU buffers MLX keeps for reuse (its
# "buffer cache"). MLX's default cache limit is its memory limit, 1.5 x the
# device's recommended working set (mlx/include/mlx/memory.h), so in practice
# nothing bounds it in this child, and a long prefill leaves gigabytes of freed
# buffers in the pool. The memory predictor derives its configuration key from
# this constant (``src.engines.memory_budget``), which is why it lives here: the
# child must not import ``src.core.logging``. Measured on an M4 16 GB (mlx 0.32.2, mlx-vlm
# 0.6.17), prefix cache reset after each request, limit 0 against the default:
# no measurable speed cost (Qwen3-4B: a ~6k-token prompt 24.0 s to first token
# and ~29 tok/s both ways, a short one 0.33 s and ~40 tok/s; Qwen3-0.6B: 4.92 s
# and 80 tok/s, short 237 tok/s, identical over 12 requests each), the peak of
# a long prefill lower by 1.25-1.8 GiB (4B: 7.19 -> 5.37 GiB; 0.6B: 4.37 ->
# 3.12 GiB) and the footprint at rest after a conversation is reset lower by
# 0.7-0.85 GiB (4B: 4.37 -> 3.53 GiB). Bounding beats clearing at idle because
# it also lowers the peak, not only the resting size.
MLX_BUFFER_CACHE_LIMIT_BYTES = 0


def _patch_gemma3_tied_lm_head_quant() -> bool:
    """Complete the tied-lm_head sanitize for quantized Gemma3 checkpoints (#273).

    Hardware-found regression on the pinned mlx-vlm 0.6.13:
    `mlx-community/gemma-3-270m-it-4bit` (tied-embeddings Gemma3 text-only,
    MLX-format 4-bit checkpoint, ships NO ``lm_head.*`` tensors) fails to load:

        File ".../mlx_vlm/utils.py", line 966, in load
        File ".../mlx_vlm/utils.py", line 842, in load_model
            model.load_weights(list(weights.items()), strict=strict)
        ValueError: Expected shape (262144, 640) but received shape (262144, 80)
        for parameter language_model.lm_head.weight

    Note this DIFFERS from 0.6.2's failure ("Missing 1 parameters:
    lm_head.weight", caused by the ``format == "mlx"`` sanitize skip). The
    0.6.13 path:

      1. ``get_model_and_args`` (``utils.py:517``) routes
         ``model_type == "gemma3_text"`` to the NEW native
         ``models/gemma3_text`` module — the ``text_only`` fallback
         (``utils.py:549-551``), whose sanitize correctly delegates to the
         inner mlx-lm model, is never reached.
      2. ``gemma3_text.Model`` wraps ``gemma3.LanguageModel``
         (``gemma3/language.py:245``) which instantiates
         ``self.lm_head = nn.Linear(hidden, vocab)`` UNCONDITIONALLY — no
         ``tie_word_embeddings`` branch.
      3. The (now unconditional) sanitize at ``utils.py:713`` runs
         ``gemma3_text.Model.sanitize`` (prefixes every key with
         ``language_model.``) then ``gemma3.LanguageModel.sanitize``
         (``gemma3/language.py:259-266``): its tied fallback copies ONLY
         ``model.embed_tokens.weight`` — for a 4-bit checkpoint the PACKED
         uint32 ``(vocab, hidden/8)`` tensor — to ``lm_head.weight``, without
         the ``.scales``/``.biases`` sidecars.
      4. ``nn.quantize``'s class predicate (``utils.py:824``) requires
         ``f"{p}.scales" in weights``; ``language_model.lm_head.scales`` is
         absent, so ``lm_head`` stays an UNQUANTIZED ``nn.Linear`` expecting
         the full ``(vocab, hidden)`` float shape.
      5. The strict ``model.load_weights`` (``utils.py:842``) then rejects the
         packed tensor with the ValueError above.

    Fix: wrap ``gemma3.LanguageModel.sanitize`` and, exactly when the upstream
    tied fallback fired (the head weight IS the embed weight, by identity) on
    a quantized embedding whose head lacks sidecars, copy
    ``embed_tokens.scales``/``.biases`` to ``lm_head.*`` too. The predicate
    then converts ``lm_head`` to a ``QuantizedLinear`` sharing the tied
    tensors — numerically identical to ``embed_tokens.as_linear``. A no-op
    for unquantized checkpoints (no sidecars to copy), for checkpoints that
    ship their own head, and on every re-run (sidecars already present).

    This hooks the shared ``gemma3.LanguageModel`` class, so both the
    ``gemma3_text`` route (270m/1b text-only) and the multimodal ``gemma3``
    route (4b+/quantized, same weight-only tied fallback) are covered.

    Returns:
        True if the patch was applied (or already present), False if mlx-vlm's
        gemma3 language module could not be imported (non-MLX hosts, CI).
        Idempotent.
    """
    try:
        from mlx_vlm.models.gemma3 import language as _gemma3_language
    except Exception:
        return False

    lm_cls = getattr(_gemma3_language, "LanguageModel", None)
    if lm_cls is None or not hasattr(lm_cls, "sanitize"):
        return False
    if getattr(lm_cls, "_erudi_tied_lm_head_quant_patch", False):
        return True

    _orig_sanitize = lm_cls.sanitize

    def _sanitize(self, weights):
        weights = _orig_sanitize(self, weights)
        prefix = "language_model."
        head_w = weights.get(f"{prefix}lm_head.weight")
        embed_w = weights.get(f"{prefix}model.embed_tokens.weight")
        if head_w is not None and head_w is embed_w and f"{prefix}lm_head.scales" not in weights:
            for sidecar in ("scales", "biases"):
                src = weights.get(f"{prefix}model.embed_tokens.{sidecar}")
                if src is not None:
                    weights[f"{prefix}lm_head.{sidecar}"] = src
        return weights

    lm_cls.sanitize = _sanitize
    lm_cls._erudi_tied_lm_head_quant_patch = True
    return True


#: Tokens that close a message in a chat template while the tokenizer's eos may
#: be something else: Gemma, ChatML (Yi, InternLM2/3, Olmo-3, Qwen2.5-Coder),
#: Llama 3, Cohere2 (tiny-aya). Never ``<|end|>``: see ``_HARMONY_MESSAGE_END``.
TURN_END_TOKENS = ("<end_of_turn>", "<|im_end|>", "<|eot_id|>", "<|END_RESPONSE|>")

#: gpt-oss (Harmony) closes its ``analysis`` message with ``<|end|>`` and only
#: then writes the ``final`` answer, so stopping there would cut every answer
#: after its reasoning. Phi-3 declares the same token as its eos and does end
#: its turn on it; the Harmony ``<|channel|>`` token tells the two apart.
_HARMONY_MESSAGE_END = "<|end|>"
_HARMONY_CHANNEL = "<|channel|>"


def _is_token_id(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _turn_end_stop_ids(tok) -> set:
    """The ids a turn ends on per the tokenizer: its eos and the named turn ends it knows."""
    ids = set()
    for attr in ("eos_token_id", "eos_token_ids"):
        value = getattr(tok, attr, None)
        values = value if isinstance(value, (list, tuple, set, frozenset)) else (value,)
        ids.update(v for v in values if _is_token_id(v))

    unk = getattr(tok, "unk_token_id", None)

    def known(name):
        # A token the tokenizer lacks converts to its unk id (or None).
        tid = tok.convert_tokens_to_ids(name)
        return tid if _is_token_id(tid) and tid != unk else None

    ids.update(tid for tid in map(known, TURN_END_TOKENS) if tid is not None)
    if known(_HARMONY_CHANNEL) is not None:
        ids.discard(known(_HARMONY_MESSAGE_END))
    return ids


def _patch_turn_end_stop_tokens() -> bool:
    """Stop where the turn ends: the tokenizer's eos and the named turn ends (#249, #721).

    mlx-vlm 0.6.17 (``server/generation.py:_initialize_model``) builds its
    stop-token set from ``config.eos_token_id`` alone, overridden by a
    checkpoint's ``generation_config.json``. It ignores the tokenizer's own
    ``eos_token`` and the token the chat template closes a message with, so
    sampling runs past the answer and streams that token as text -- then, on
    some models, a made-up next turn until ``max_tokens``. Measured on an M4
    without the patch: Meta-Llama-3-8B-Instruct (config eos
    ``<|end_of_text|>``, turn end ``<|eot_id|>``) loops on invented turns,
    Yi-1.5-6B (``<|im_end|>``) invents a ``user`` turn, Phi-3-mini shows
    ``<|end|>``, tiny-aya shows ``<|END_RESPONSE|>``, Gemma streams
    ``<end_of_turn>``. A static scan of the MLX catalog snapshot flags 48
    entries. The OpenAI ``stop`` request field does not help: the generation
    loop halts on token ids in ``stop_tokens``, not on decoded strings.

    The wrapped ``_initialize_model`` adds, after the model loads:

      - the tokenizer's eos ids (``eos_token_id``, and ``eos_token_ids`` when
        present) -- what upstream does from 0.7.0 on (Neywa-Labs/mlx-vlm PR
        #2112);
      - the ids of the ``TURN_END_TOKENS`` the tokenizer knows (an unknown one
        converts to ``unk`` or ``None`` and is skipped). tiny-aya needs this
        even on 0.8.0: its eos is ``<|END_OF_TURN_TOKEN|>`` but the message
        closes on ``<|END_RESPONSE|>`` first.

    ``<|end|>`` is never a named turn end, and on a Harmony tokenizer it is
    taken out of what is added even when declared the eos: gpt-oss writes its
    answer after it. Nothing upstream put in the set is removed. A failure
    while reading the tokenizer leaves upstream's set and one warning in the
    child's log; the model still loads. Idempotent.

    Returns:
        True if the patch was applied (or already present), False if mlx-vlm's
        server generation module or its ``ResponseGenerator._initialize_model``
        is not where 0.6.17 has it (non-MLX hosts, an mlx-vlm bump).
    """
    try:
        from mlx_vlm.server import generation as _gen
    except Exception:
        return False

    rg = getattr(_gen, "ResponseGenerator", None)
    if rg is None or not hasattr(rg, "_initialize_model"):
        return False
    if getattr(rg, "_erudi_turn_end_stop_patch", False):
        return True

    _orig_initialize_model = rg._initialize_model

    @functools.wraps(_orig_initialize_model)
    def _initialize_model(self):
        _orig_initialize_model(self)
        tok = getattr(self, "tokenizer", None)
        stop = getattr(self, "stop_tokens", None)
        if tok is None or stop is None:
            return
        try:
            added = _turn_end_stop_ids(tok) - stop
        except Exception as exc:
            child_warning(
                f"Turn-end stop tokens not added: {type(exc).__name__}: {exc}. "
                f"Generation stops on the config eos only."
            )
            return
        stop.update(added)
        write_child_record("INFO", f"Stop tokens: {sorted(stop)} (added {sorted(added)})")

    rg._initialize_model = _initialize_model
    rg._erudi_turn_end_stop_patch = True
    return True


def _patch_json_tool_call_tolerance() -> bool:
    """Parse the doubled-brace tool call the Qwen2.5 template teaches (#721).

    The official Qwen2.5 tool template (also in Nemotron-Nano-9B-v2 and
    VoiceChat-11B) shows the call format as
    ``{{"name": <function-name>, "arguments": <args-json-object>}}``, a
    Jinja-escaping slip that renders the braces doubled. Qwen2.5-0.5B copies
    them: ``{{"name": "get_weather", "arguments": {"city": "Paris"}}}}``.
    mlx-vlm 0.6.17's ``tool_parsers/json_tools.parse_tool_call`` is a strict
    ``json.loads``, so ``process_tool_calls`` logs "Invalid tool call" and drops
    it; the streamed text after ``<tool_call>`` is already suppressed, so the
    turn ends with no content and no call. Unchanged in 0.8.0; no upstream
    issue found.

    The wrapper tries the strict parse first, so a valid call is parsed
    exactly as before. Only when it fails on text that opens with ``{{``, it
    drops the extra leading braces, decodes one JSON object and accepts it
    when nothing but closing braces follows and the object is a call (a
    string ``name``, ``arguments`` an object or a string). Anything else
    raises, as today: a malformed call is never executed. Idempotent.

    Returns:
        True if the patch was applied (or already present), False if mlx-vlm's
        ``json_tools`` parser is not where 0.6.17 has it.
    """
    try:
        from mlx_vlm.tool_parsers import json_tools as _json_tools
    except Exception:
        return False

    strict = getattr(_json_tools, "parse_tool_call", None)
    if strict is None:
        return False
    if getattr(_json_tools, "_erudi_tolerant_parse_patch", False):
        return True

    @functools.wraps(strict)
    def parse_tool_call(text, tools=None):
        try:
            return strict(text, tools)
        except ValueError:  # json.JSONDecodeError
            candidate = text.strip()
            if not candidate.startswith("{{"):
                raise
        while candidate.startswith("{{"):
            candidate = candidate[1:]
        call, end = json.JSONDecoder().raw_decode(candidate)
        if candidate[end:].strip("} \t\r\n"):
            raise ValueError(f"Tool call has trailing text: {text!r}")
        if not (
            isinstance(call, dict)
            and isinstance(call.get("name"), str)
            and isinstance(call.get("arguments"), (dict, str))
        ):
            raise ValueError(f"Tool call has no name or arguments: {text!r}")
        return call

    _json_tools.parse_tool_call = parse_tool_call
    _json_tools._erudi_tolerant_parse_patch = True
    return True


def _patch_granite_decode_embedding_multiplier() -> bool:
    """Apply Granite's embedding multiplier while decoding, not only at prefill (#721).

    mlx-vlm 0.6.17 ``models/granite/language.py`` (``GraniteModel.__call__``)
    multiplies the embeddings by ``embedding_multiplier`` only on the
    ``inputs_embeds`` path. The prefill takes it (``Model.get_input_embeddings``),
    but every decode step passes token ids, so the multiplier -- 12 on
    granite-3.3 and granite-4.1 -- is skipped: the first token is right and the
    rest is noise ("The Par thees the,: France"). mlx-lm multiplies on both
    paths. Unchanged in 0.8.0; no upstream issue found. The ``granitemoehybrid``
    models (granite-4.0) multiply on both paths and are not affected.

    The wrapper embeds the ids itself when no embeddings are given, so the
    upstream body multiplies exactly once on both paths. Idempotent.

    Returns:
        True if the patch was applied (or already present), False if
        ``GraniteModel`` or its ``inputs_embeds`` parameter is not where 0.6.17
        has it.
    """
    try:
        from mlx_vlm.models.granite import language as _granite_language
    except Exception:
        return False

    cls = getattr(_granite_language, "GraniteModel", None)
    orig = vars(cls).get("__call__") if cls is not None else None
    if orig is None or "inputs_embeds" not in inspect.signature(orig).parameters:
        return False
    if getattr(cls, "_erudi_granite_multiplier_patch", False):
        return True

    @functools.wraps(orig)
    def __call__(self, inputs, cache=None, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(inputs)
        return orig(self, inputs, cache, inputs_embeds=inputs_embeds)

    cls.__call__ = __call__
    cls._erudi_granite_multiplier_patch = True
    return True


def _patch_internlm2_output_head_sanitize() -> bool:
    """Load InternLM2's ``output.*`` head (#721).

    mlx-vlm 0.6.17 ``models/internlm2/internlm2.py`` (``Model.sanitize``)
    prefixes ``model.*`` and ``lm_head.*`` with ``language_model.``, but
    InternLM2 names its head ``output.*``, so the strict load rejects every
    InternLM2 checkpoint: "Received 3 parameters not in model: output.biases,
    output.scales, output.weight". Unchanged in 0.8.0; no upstream issue found.
    internlm3 names its head ``lm_head`` and is not affected.

    The wrapper prefixes ``output.*`` before the upstream sanitize runs, which
    leaves an already-prefixed key alone. Idempotent.

    Returns:
        True if the patch was applied (or already present), False if the
        InternLM2 ``Model.sanitize`` is not where 0.6.17 has it.
    """
    try:
        from mlx_vlm.models.internlm2 import internlm2 as _internlm2
    except Exception:
        return False

    cls = getattr(_internlm2, "Model", None)
    if cls is None or not hasattr(cls, "sanitize"):
        return False
    if getattr(cls, "_erudi_internlm2_head_patch", False):
        return True

    _orig_sanitize = cls.sanitize

    @functools.wraps(_orig_sanitize)
    def sanitize(self, weights):
        weights = {
            (f"language_model.{k}" if k.startswith("output.") else k): v for k, v in weights.items()
        }
        return _orig_sanitize(self, weights)

    cls.sanitize = sanitize
    cls._erudi_internlm2_head_patch = True
    return True


#: The in-child patches, applied in this order before the server starts.
_IN_CHILD_PATCHES = (
    # Quantized tied-embedding Gemma3 checkpoints load (#273).
    "_patch_gemma3_tied_lm_head_quant",
    # Generation stops at the end of the turn instead of streaming the
    # end-of-turn token and invented turns (#249, #721).
    "_patch_turn_end_stop_tokens",
    # A tool call written with the Qwen2.5 template's doubled braces parses (#721).
    "_patch_json_tool_call_tolerance",
    # Granite keeps its embedding multiplier while decoding (#721).
    "_patch_granite_decode_embedding_multiplier",
    # InternLM2 checkpoints load their `output.*` head (#721).
    "_patch_internlm2_output_head_sanitize",
)


def _bound_buffer_cache() -> None:
    """Cap MLX's buffer cache at ``MLX_BUFFER_CACHE_LIMIT_BYTES`` in this child.

    A failure (``mlx.core`` missing, ``set_cache_limit`` absent or raising) is
    one warning in the child's log and the server starts anyway: the bound is a
    memory saving, never a precondition for inference.
    """
    try:
        import mlx.core as mx

        previous = mx.set_cache_limit(MLX_BUFFER_CACHE_LIMIT_BYTES)
    except Exception as exc:
        child_warning(
            f"MLX buffer cache not bounded: {type(exc).__name__}: {exc}. "
            f"The child keeps MLX's default cache."
        )
        return
    write_child_record(
        "INFO",
        f"MLX buffer cache limit set to {MLX_BUFFER_CACHE_LIMIT_BYTES} bytes (was {previous})",
    )


def _import_mlx_vlm_server_main():
    """Import and return `mlx_vlm.server.cli.main`.

    Extracted as a separate function so tests can patch this seam without
    requiring `mlx-vlm` to be installed on the CI runner. The real import is
    deferred until first call, both for test isolation and to keep the
    parent-process import time low.
    """
    from mlx_vlm.server.cli import main as _main

    return _main


def _apply_child_runtime_env(
    apc_num_blocks: Optional[int], token_queue_timeout_s: Optional[float]
) -> None:
    """Set this build's mlx-vlm runtime knobs into the child's own environment.

    mlx-vlm reads these at import/init, so they must land in ``os.environ``
    BEFORE ``mlx_vlm.server`` is imported. Applied as an explicit dict (values
    computed by the parent and passed in, never inherited) so a frozen child
    that re-execs the binary still gets them, and so the mapping is unit-testable
    without a spawn.

    - ``APC_ENABLED`` / ``APC_BLOCK_SIZE`` / ``APC_NUM_BLOCKS`` turn on Automatic
      Prefix Caching, so a 2nd+ turn prefills only the suffix of the
      conversation instead of the whole history. The pool is sized by the parent
      to cover the model's trained window, the spawn's ``--max-kv-size``
      (``ceil(window / APC_BLOCK_SIZE_TOKENS)``; a memory bound applied after
      load leaves it as is, since the pool fills lazily); a ``None`` count leaves
      mlx-vlm's own default (2048 blocks), for a child spawned without a window.
      The pool fills lazily and mlx-vlm's own ``APC_MAX_POOL_TENSORS`` + Metal
      memory-pressure guard bound it. The pool counts toward the child's
      footprint, which ``memory_budget``'s prior accounts per token and the
      engine measures every turn (``MLX_Engine.end_memory_window``).
    - ``MLX_VLM_TOKEN_QUEUE_TIMEOUT`` lifts the child's per-token wait above the
      parent's first-chunk watchdog ceiling so a long cold prefill trips the
      parent's curated timeout, not the child's raw error; ``None`` leaves
      mlx-vlm's own default (600 s).
    - ``MLX_VLM_MAX_NUM_SEQS=1``: one sequence in the running batch at a time.
      The prefix-cache barrier (``MLX_Engine._send_barrier``) relies on it.
      mlx-vlm's GPU loop admits new requests BEFORE it drains cancellations
      (``server/generation.py``: ``_collect_pending_requests``, then
      ``_drain_cancellations``), and its batch is unbounded by default, so a
      barrier could run in the same step as a cancelled request whose
      cancellation is not registered yet, finish, and let the reset clear
      blocks that request still holds. With a capacity of
      ``max(0, 1 - len(active))`` the barrier is only admitted once the
      cancelled request has left the batch. No concurrency is lost: Erudi
      already serializes every request to the child behind the generation
      guard. The cost is a wait right after an abandon: any request sent then
      (not only the barrier) waits for the cancelled request to leave the
      batch, typically one prefill step.

    The prefill step is NOT set here: mlx-vlm's CLI rewrites ``PREFILL_STEP_SIZE``
    from its own ``--prefill-step-size`` option, so the parent passes it on the
    command line (``MLX_Engine._spawn_child``).

    Nothing here may import ``src.core.logging`` (directly or through another
    ``src`` module): it would install the backend's handlers in the child
    before mlx-vlm's ``logging.basicConfig``, which would then do nothing.

    Uses ``os.environ.update`` (not ``os.environ[name] = ...``) on purpose: these
    are knobs the parent WRITES for the mlx-vlm child, not configuration the
    backend itself reads, so they do not belong in ``backend/.env.example``.
    """
    env = {
        "APC_ENABLED": "1",
        "APC_BLOCK_SIZE": str(APC_BLOCK_SIZE_TOKENS),
        "MLX_VLM_MAX_NUM_SEQS": "1",
    }
    if apc_num_blocks is not None:
        env["APC_NUM_BLOCKS"] = str(int(apc_num_blocks))
    if token_queue_timeout_s is not None:
        env["MLX_VLM_TOKEN_QUEUE_TIMEOUT"] = str(float(token_queue_timeout_s))
    os.environ.update(env)


#: Seconds between the parent-watch thread's ppid reads.
PARENT_WATCH_INTERVAL_S = 2.0


def _parent_watch_loop(initial_ppid, *, getppid, hard_exit, sleep, interval_s=None) -> None:
    """Exit this process once its parent is gone (ppid changed).

    The chat child is stopped by the backend's orderly paths (lifespan
    cleanup, atexit); but a backend ended by ``os._exit`` -- run.py's
    ``arm_exit_deadline`` -- runs no atexit, and a non-daemon ``mp.Process``
    then survives, reparented. For the embedding child that would orphan
    ~1.2 GB with nothing left to reap it, so the child watches its parent
    itself. On POSIX a dead parent shows as a CHANGED ppid (reparenting to
    launchd/init). Injected callables keep the loop unit-testable.
    """
    interval = PARENT_WATCH_INTERVAL_S if interval_s is None else interval_s
    while True:
        sleep(interval)
        if getppid() != initial_ppid:
            child_warning("parent process is gone; the embedding child exits")
            hard_exit(0)
            return


def _start_parent_watch() -> None:
    """Daemon thread running :func:`_parent_watch_loop` on the real process."""
    import threading

    initial = os.getppid()
    threading.Thread(
        target=_parent_watch_loop,
        args=(initial,),
        kwargs={"getppid": os.getppid, "hard_exit": os._exit, "sleep": time.sleep},
        name="parent-watch",
        daemon=True,
    ).start()


def run_mlx_vlm_server(
    argv: List[str],
    log_path: Optional[str] = None,
    apc_num_blocks: Optional[int] = None,
    token_queue_timeout_s: Optional[float] = None,
    watch_parent: bool = False,
) -> None:
    """Child-process entry: set `sys.argv = argv` then run `mlx_vlm.server`'s main().

    Args:
        argv: Full argument vector. `argv[0]` is the program name
            (conventionally ``"mlx_vlm.server"``); the rest are CLI flags
            consumed by argparse inside `main()`.
        log_path: File this child sends its stdout and stderr to, resolved by
            the parent (only the parent knows the log directory -- see
            ``mlx_child_log.prepare_child_log``). ``None`` leaves the
            descriptors inherited from the backend, which is what unit tests
            and any caller without a writable log directory get.
        apc_num_blocks: Automatic Prefix Caching pool size in blocks, sized by
            the parent to cover the model's trained window; ``None`` leaves mlx-vlm's
            2048-block default. See ``_apply_child_runtime_env``.
        token_queue_timeout_s: The child's per-token wait, sized by the parent
            above its own first-chunk watchdog ceiling; ``None`` leaves
            mlx-vlm's 600 s default. See ``_apply_child_runtime_env``.
        watch_parent: Exit when the parent process dies (ppid changes). The
            EMBEDDING child (#594) opts in: a backend ended by ``os._exit``
            (run.py's exit deadline) runs no atexit and would orphan it.
            The chat child keeps today's behaviour (``False``).

    Returns:
        None. This call blocks for the lifetime of the HTTP server.
    """
    import sys

    sys.argv = list(argv)
    if log_path:
        try:
            # Before anything else: mlx-vlm's `logging.basicConfig` and
            # uvicorn's handlers bind to whatever `sys.stderr` is when they
            # run, and a model that fails to load does so during `main()`.
            redirect_stdio_to(log_path)
        except Exception as exc:
            # Capture is a diagnostic, never a precondition. A child that
            # cannot open its log file still serves inference; it just has no
            # last words. Reported on the inherited stderr, which the Electron
            # main process writes to the app log.
            print(f"[WARNING] mlx-vlm child output capture disabled: {exc}", file=sys.stderr)
    # Before the server blocks: the embedding child must not outlive a parent
    # that could not run its orderly stops (see _parent_watch_loop).
    if watch_parent:
        _start_parent_watch()
    # Before the server is imported, so no buffer is pooled under the default.
    _bound_buffer_cache()
    # APC on, the aligned token-queue timeout and the one-sequence batch,
    # BEFORE the server is imported: mlx-vlm reads them from the environment
    # (the APC pool in `apc.from_env`, the timeout in `server.runtime_config`,
    # the batch cap when its generation loop starts).
    _apply_child_runtime_env(apc_num_blocks, token_queue_timeout_s)
    # Applied in-child before the server loads any model: each one fixes a
    # defect the user would otherwise meet head-on, so a patch that did not
    # apply is recorded rather than silently tolerated -- the record lands in
    # the captured file, which the parent quotes. Looked up by name at call
    # time, so a test can stub any of them.
    for name in _IN_CHILD_PATCHES:
        _record_unapplied_patch(name, globals()[name]())
    main = _import_mlx_vlm_server_main()
    main()


def _record_unapplied_patch(name: str, applied: bool) -> None:
    """Warn (in the child's own log) when a monkeypatch could not be applied.

    Every patch in ``_IN_CHILD_PATCHES`` compensates for a pinned-mlx-vlm
    defect with a visible consequence: a checkpoint that refuses to load, an
    answer that runs past its end-of-turn token, an empty tool turn, Granite
    noise. Each returns ``False`` when its
    target module or class is not where it expects it -- an mlx-vlm bump,
    typically. Unrecorded, that ships a degraded product with nothing to
    explain it.
    """
    if not applied:
        child_warning(
            f"{name} did not apply: the mlx-vlm internals it patches moved. "
            f"The defect it works around is back."
        )
