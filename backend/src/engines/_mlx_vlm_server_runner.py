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

In-child patches (pinned mlx-vlm 0.6.13)
----------------------------------------
Three monkeypatches are applied before the server starts. Two 0.6.2-era
patches were dropped with the 0.6.13 bump because upstream now runs weight
sanitization unconditionally in `mlx_vlm.utils.load_model` (the 0.6.2
`format == "mlx"` sanitize skip is gone) — but hardware validation showed the
static evidence for the first drop was incomplete:

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

Contract
--------
`run_mlx_vlm_server(argv)` replaces `sys.argv` with the supplied list and calls
the real `mlx_vlm.server.cli.main()`. The first element of `argv` is the
conventional program name; the rest are the CLI flags that mlx-vlm's argparse
expects (`--model`, `--host`, `--port`, `--log-level`, ...). `main()` parses
them, exports the matching env vars (e.g. `MLX_VLM_PRELOAD_MODEL` from
`--model`), and launches `uvicorn.run("mlx_vlm.server:app", ...)`.

Once invoked, this function blocks for the lifetime of the HTTP server,
exiting only when the child process is terminated by the parent.
"""

from __future__ import annotations

from typing import List


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


def _patch_gemma_end_of_turn_stop() -> bool:
    """Register Gemma's ``<end_of_turn>`` as a stop token in the mlx_vlm server (#249).

    mlx_vlm (still on 0.6.13, ``server/generation.py:_initialize_model``) builds
    its stop-token set solely from ``config.eos_token_id``. Gemma checkpoints
    declare ``eos_token`` = ``<eos>`` (id 1), but their chat template ends
    *every turn* with ``<end_of_turn>`` (id 106) — which is therefore NOT in
    the stop set. Sampling runs past the answer and streams the literal
    ``<end_of_turn>`` token text plus multilingual garbage to the user. (The
    OpenAI ``stop`` request field does not help: mlx_vlm's generation loop
    halts on token *ids* in ``stop_tokens``, not on decoded strings.)

    0.6.13 additionally merges a checkpoint's ``generation_config.json`` eos
    ids into the config (``utils._merge_generation_config``), which covers
    Gemma checkpoints that ship ``eos_token_id: [1, 106]`` there — but that is
    checkpoint metadata, not a server guarantee. This patch stays as the
    checkpoint-independent belt: derived from the tokenizer, and a no-op when
    the merge already put id 106 in the stop set.

    We wrap ``_initialize_model`` to add the tokenizer's ``<end_of_turn>`` id to
    ``stop_tokens`` after the model loads — derived from the tokenizer, no
    hardcoded id. A no-op for tokenizers that don't define the token (the id then
    resolves to ``unk``), so non-Gemma checkpoints are untouched. Idempotent.

    Returns:
        True if the patch was applied (or already present), False if mlx-vlm's
        server generation module could not be imported (non-MLX hosts, CI).
    """
    try:
        from mlx_vlm.server import generation as _gen
    except Exception:
        return False

    rg = getattr(_gen, "ResponseGenerator", None)
    if rg is None or not hasattr(rg, "_initialize_model"):
        return False
    if getattr(rg, "_erudi_end_of_turn_patch", False):
        return True

    _orig_initialize_model = rg._initialize_model

    def _initialize_model(self):
        _orig_initialize_model(self)
        try:
            tok = getattr(self, "tokenizer", None)
            stop = getattr(self, "stop_tokens", None)
            if tok is None or stop is None:
                return
            unk = getattr(tok, "unk_token_id", None)
            tid = tok.convert_tokens_to_ids("<end_of_turn>")
            if tid is not None and tid >= 0 and tid != unk:
                stop.add(tid)
        except Exception:
            # Stop-token augmentation must never break model load.
            pass

    rg._initialize_model = _initialize_model
    rg._erudi_end_of_turn_patch = True
    return True


# Unmatchable thinking markers injected by `_patch_inline_thinking`. Model text
# can never contain a NUL byte, so these never match a marker (no split) and
# never partially match a chunk suffix (no `_split_partial` holdback latency).
_NEVER_OPEN_MARKER = "\x00erudi:no-thinking-split\x00"
_NEVER_CLOSE_MARKER = "\x00/erudi:no-thinking-split\x00"


def _patch_inline_thinking() -> bool:
    """Keep model reasoning INLINE in ``delta.content`` (#90).

    mlx-vlm 0.6.13 splits streamed thinking into a dedicated
    ``delta.reasoning`` field via ``ThinkingStreamState`` — a channel that
    ChatOpenAI silently drops, so the reasoning never reaches the runner. The
    design (#90) wants the raw ``<think>...</think>`` INLINE in
    ``delta.content`` so the runner's single streaming ThinkSplitter handles
    MLX exactly like llama-server with ``--reasoning-format none``.

    Why a monkeypatch and not configuration — on the pinned 0.6.13:

      - ``--thinking-start-token`` / ``MLX_VLM_THINKING_START_TOKEN`` exist
        but cannot disable the split: ``_build_open_close_markers`` always
        APPENDS the built-in marker families (``<think>``,
        ``<|channel>thought``, ``<|START_THINKING|>``) after any custom pair,
        and a custom pair only registers when BOTH start and end tokens are
        set. There is no native "reasoning inline / no split" control.
      - ``ThinkingStreamState.__init__`` still sets ``in_thinking =
        bool(enable_thinking)``: the route passes ``prompt_has_open_thinking``
        there, so a prompt whose template opens a thinking block starts the
        stream in reasoning mode regardless of any marker.

    So the first choke point is the class itself: force every instance to
    start OUTSIDE thinking with unmatchable markers. ``feed()`` then falls
    through to its plain-content branch, preserving upstream
    ``<|START_TEXT|>`` content-marker stripping and the downstream tool-call
    suppression untouched. The class object is mutated in place (never
    rebound), so it is irrelevant whether callers imported it before or after
    the patch.

    0.6.13 adds a second choke point: the ``make_response_stream_state``
    factory prefers a ``ResponseTemplateStreamState`` (a transformers
    response-template parser that ALSO routes reasoning to
    ``delta.reasoning``) whenever the tokenizer exposes a
    ``response_template`` — bypassing ``ThinkingStreamState`` entirely. The
    factory resolves its ``_response_template_tokenizer`` helper through the
    module globals at call time, so neutralizing that helper disables the
    bypass even though the route modules from-import the factory at package
    import time. Every stream then goes through the neutralized
    ``ThinkingStreamState``.

    Returns:
        True if the patch was applied (or already present), False if
        mlx-vlm's server module could not be imported (non-MLX hosts, CI).
        Idempotent.
    """
    try:
        from mlx_vlm.server import responses_state
    except Exception:
        return False

    state_cls = getattr(responses_state, "ThinkingStreamState", None)
    if state_cls is None:
        return False

    if not getattr(state_cls, "_erudi_inline_thinking_patch", False):
        _orig_init = state_cls.__init__

        def _init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            self.in_thinking = False
            self.open_close_markers = ((_NEVER_OPEN_MARKER, _NEVER_CLOSE_MARKER),)
            self.open_markers = (_NEVER_OPEN_MARKER,)
            self.close_markers = (_NEVER_CLOSE_MARKER,)

        state_cls.__init__ = _init
        state_cls._erudi_inline_thinking_patch = True

    # Disable the 0.6.13 template-parser bypass in `make_response_stream_state`.
    if hasattr(responses_state, "_response_template_tokenizer") and not getattr(
        responses_state, "_erudi_template_bypass_patch", False
    ):
        responses_state._response_template_tokenizer = lambda processor: None
        responses_state._erudi_template_bypass_patch = True

    return True


def _import_mlx_vlm_server_main():
    """Import and return `mlx_vlm.server.cli.main`.

    Extracted as a separate function so tests can patch this seam without
    requiring `mlx-vlm` to be installed on the CI runner. The real import is
    deferred until first call, both for test isolation and to keep the
    parent-process import time low.
    """
    from mlx_vlm.server.cli import main as _main

    return _main


def run_mlx_vlm_server(argv: List[str]) -> None:
    """Child-process entry: set `sys.argv = argv` then run `mlx_vlm.server`'s main().

    Args:
        argv: Full argument vector. `argv[0]` is the program name
            (conventionally ``"mlx_vlm.server"``); the rest are CLI flags
            consumed by argparse inside `main()`.

    Returns:
        None. This call blocks for the lifetime of the HTTP server.
    """
    import sys

    sys.argv = list(argv)
    # Applied in-child before the server loads any model so quantized
    # tied-embeddings Gemma3 checkpoints (270m/1b text-only via the native
    # gemma3_text route, and multimodal gemma3) load cleanly on 0.6.13 (#273).
    _patch_gemma3_tied_lm_head_quant()
    # Register Gemma's <end_of_turn> as a stop token so generation halts at the
    # end of the answer instead of streaming the literal token + garbage (#249).
    _patch_gemma_end_of_turn_stop()
    # Applied in-child before the server starts so every ThinkingStreamState it
    # builds keeps reasoning inline in delta.content (#90) — see the patch's
    # docstring for why 0.6.13 offers no configuration path for this.
    _patch_inline_thinking()
    main = _import_mlx_vlm_server_main()
    main()
