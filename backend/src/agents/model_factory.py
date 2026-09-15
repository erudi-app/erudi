"""Build a ``ChatOpenAI`` pointed at the local engine's OpenAI-compatible server.

The engine (MLX/CUDA/CPU) spawns a child server and exposes its
``/v1/chat/completions`` endpoint; ``ChatOpenAI(base_url=...)`` talks to it.
``get_model_and_tokenizer`` is the authority that spawns/selects the child and
hands back the ``base_url``; the engine no longer parses SSE itself — token
streaming is owned by this ``ChatOpenAI`` layer.

The client is an ``Erudi_Chat_OpenAI`` (``src.agents.chat_model``): a
``ChatOpenAI`` whose async stream carries the two #573 wall-clock budgets
(prompt-sized before the first token, fixed between tokens) in place of
langchain's single uniform ``stream_chunk_timeout``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from src.agents.chat_model import erudi_chat_openai_class
from src.core import config
from src.core.logging import logger
from src.database.generation_hints import (
    FALLBACK_REPETITION_CONTEXT_SIZE,
    FALLBACK_REPETITION_PENALTY,
    SamplingDefaults,
)

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

# Sampling controls the bare OpenAI wire schema lacks but local models need to
# avoid degenerate repetition loops. These mirror the pre-LangChain engine
# defaults: the hand-rolled path passed repetition_penalty=1.2 +
# repetition_context_size=5 to EVERY generation. Dropping them on the ChatOpenAI
# path made even tiny models (e.g. Gemma-270M) loop on trivial prompts, so they
# are restored here. The values themselves live with the other sampling
# fallbacks in src.database.generation_hints (#388); the #129 rationale (1.1
# over 64 tokens) is documented there.
DEFAULT_REPETITION_PENALTY = FALLBACK_REPETITION_PENALTY
DEFAULT_REPETITION_CONTEXT_SIZE = FALLBACK_REPETITION_CONTEXT_SIZE


def build_chat_model(
    llm,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    repetition_context_size: int = DEFAULT_REPETITION_CONTEXT_SIZE,
    disable_thinking: bool = False,
    auto_output_budget: bool = True,
    sampling: Optional[SamplingDefaults] = None,
) -> ChatOpenAI:
    """Resolve the engine child for ``llm`` and wrap it as a ``ChatOpenAI``.

    SYNC and potentially slow: ``get_model_and_tokenizer`` spawns/probes the
    child server under the engine lock, so call this via ``run_in_threadpool``
    from async code (the agent runner does).

    The ``model`` field MUST go through the engine's ``_payload_model_value`` —
    mlx_vlm.server resolves it via ``get_cached_model(request.model)`` so MLX
    sends the real preloaded model path, while llama.cpp sends the alias;
    hardcoding either would break the other.

    Params are set on the constructor (NOT via ``.bind`` — LangChain v1 rejects
    pre-bound models passed to ``create_agent``).

    ``max_tokens`` is normally only the FALLBACK output budget: the client
    recomputes a real one per model call from the window it is running in
    (``src.agents.output_budget``). ``auto_output_budget=False`` turns that off
    for the caller whose small budget is deliberate -- the one-shot title path.
    """
    # Deferred (#160): langchain_openai only loads on the first turn, not at
    # boot -- the subclass that inherits from it is built on the same first call.
    chat_openai_class = erudi_chat_openai_class()

    engine = config.LLM_Engine
    handle, _tokenizer = engine.get_model_and_tokenizer(llm.id, llm.link)
    model_field = engine._payload_model_value(handle)

    # The ALLOCATED window of the child just resolved (llama-server: read from
    # /props after its fit; MLX: the --max-kv-size bound stamped at spawn).
    # Handed to the client so the first-chunk watchdog ceiling scales with the
    # window a full-window prompt can now legitimately fill -- without it, the
    # dynamic context window would recreate the #573 kill (a healthy long
    # prefill cut at the fixed 900 s ceiling). Fresh per turn: this factory
    # runs on every turn, after any model swap. None = window unknown.
    window_probe = getattr(engine, "effective_context_tokens", None)
    effective_window = window_probe() if callable(window_probe) else None

    # Whether this engine's child rejects an over-budget request instead of
    # clamping it (MLX's preflight validator vs llama-server's n_predict
    # clamp). The output budget holds itself under a provable prompt bound on
    # the ones that reject, so the app can never reject its own turn.
    # (getattr keeps test stubs and non-server engines working.)
    preflight_probe = getattr(engine, "preflight_counts_output_tokens", None)
    preflight_counts_output = bool(preflight_probe()) if callable(preflight_probe) else False

    # Extra sampling params absent from the OpenAI wire schema. mlx_vlm.server reads
    # the HF names natively; llama.cpp engines translate them to their wire names
    # (repeat_penalty / repeat_last_n) via ``_translate_payload_kwargs``. Sent via
    # ChatOpenAI.extra_body so they land in the local server's chat-completions
    # body. (getattr keeps non-server engines / test stubs working via identity.)
    # The MLX translation also stamps a fresh random ``seed`` per request: this
    # factory runs once per turn, so every generation gets its own, which is what
    # makes temperature / top_p / top_k actually vary the output on Apple Silicon
    # (mlx_vlm.server replays a fixed default seed when the request has none).
    # #388: a resolved per-model profile supplies the repetition controls and,
    # ONLY when it defines them, top_k / min_p / presence_penalty. Without a
    # profile (or for a model without hints, whose profile is the fallback)
    # the body is byte-identical to the #129-validated one above.
    if sampling is not None:
        raw_kwargs = sampling.wire_kwargs()
    else:
        raw_kwargs = {
            "repetition_penalty": repetition_penalty,
            "repetition_context_size": repetition_context_size,
        }
    # Suppress reasoning at the chat-template level (#266): one-shot utility
    # calls (e.g. conversation titles) run on a ~12-token budget that a thinking
    # model would burn entirely inside <think>. mlx_vlm.server reads the
    # per-request ``enable_thinking`` field natively (it overrides the server
    # default); llama.cpp engines translate it to ``chat_template_kwargs`` in
    # their ``_translate_payload_kwargs``. Chat paths never pass
    # ``disable_thinking``, so their request body stays byte-identical to today.
    if disable_thinking:
        raw_kwargs["enable_thinking"] = False
    translate = getattr(engine, "_translate_payload_kwargs", lambda kw: kw)
    extra_body = translate(raw_kwargs)

    # Log the extra_body AS SENT (post-translation, so llama.cpp's wire names
    # show up), one key=value per entry on the same line: the optional profile
    # keys (top_k / min_p / presence_penalty, #388) are otherwise invisible in
    # the INFO log and a QA pass cannot confirm they reached the server.
    extra_body_desc = ", ".join(f"{key}={value}" for key, value in extra_body.items())
    logger.info(
        f"ChatOpenAI built: model={model_field}, base_url={handle['base_url']}/v1, "
        f"temperature={temperature}, top_p={top_p}, max_tokens={max_tokens}, "
        f"extra_body=[{extra_body_desc}]"
    )
    return chat_openai_class(
        base_url=f"{handle['base_url']}/v1",
        # Every inference child (llama-server, mlx_vlm.server) is spawned with a
        # per-spawn `--api-key` so nothing else on the loopback interface can
        # drive it, and the handle is the only place that key lives. A handle
        # without one keeps the literal: an empty api_key would make the OpenAI
        # client fall back to reading OPENAI_API_KEY from the environment.
        # Deliberately absent from the log line above.
        api_key=handle.get("api_key") or "not-needed",
        model=model_field,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_body=extra_body,  # restore small-model coherence (repetition controls)
        timeout=None,  # cold model load can stall several seconds before first token
        max_retries=0,  # don't silently double-submit a slow local generation
        # langchain's ONE uniform per-chunk budget (120 s) cannot tell a long
        # prefill from a hang and killed healthy long-history turns (#573).
        # Off here; ``Erudi_Chat_OpenAI._astream`` enforces the two budgets that
        # replace it.
        stream_chunk_timeout=None,
        effective_context_tokens=effective_window,
        auto_output_budget=auto_output_budget,
        preflight_counts_output=preflight_counts_output,
        streaming=True,
        stream_usage=False,  # local servers may not emit usage in SSE; summarization triggers on count
    )
