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

import functools
from typing import TYPE_CHECKING, Optional

from src.agents.chat_model import erudi_chat_openai_class
from src.agents.reasoning_effort import EffortPlan
from src.engines.working_window import working_context_tokens
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
    effort_plan: Optional[EffortPlan] = None,
    prompt_ratio: Optional[float] = None,
    preflight_retry: bool = True,
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
    for the callers whose budget is deliberate -- the one-shot title path and
    the capped compaction summary.

    ``prompt_ratio`` is the first-hop ratio the runner read from the raw
    checkpoint state at turn start (``src.agents.token_accounting``); the
    output budget scales its estimate by it on a first hop. ``None``: nothing
    measured, the script-aware fallback applies. ``preflight_retry=False``
    turns off the single retry with a smaller cap after a context-check
    rejection (the summary client: a smaller cap would truncate the summary).

    Every client asks the server for usage (``stream_usage=True``): the last
    chunk then carries the real prompt size, which the client stamps with its
    own estimate of the request (``Erudi_Chat_OpenAI._astream``).

    ``effort_plan`` carries the turn's reasoning effort (1.1.2). Its
    ``wire_effort`` rides the NATIVE ``reasoning_effort`` request field, not
    ``extra_body``: it is a first-class field of all three layers (the
    ``ChatOpenAI`` constructor, llama-server, mlx_vlm). ``None`` (no plan, or a
    level no native lever carries) leaves it unset, and langchain drops unset
    fields from the payload -- so the request body stays byte-identical to
    today's.
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

    # The MEMORY value for the SAME client: the ONE canonical working window
    # (min of the allocated window and the memory ceiling, over the known
    # candidates -- src.engines.working_window). Only the output budget reads
    # it; the first-chunk watchdog and preflight retry keep the raw allocated
    # window above. ``working_context_tokens`` calls ``MemoryBudget.from_engine``
    # (disk I/O to read the loaded artifact + its config.json), which is safe
    # here because this factory runs in a threadpool per turn.
    working_window = working_context_tokens(engine)

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

    # A stream that does not end normally flags the child it ran against
    # (``Erudi_Chat_OpenAI.abandon_hook``; the MLX engine then sends a barrier
    # before its next prefix-cache reset). Bound NOW to this handle, never
    # re-read from the engine later: a stream finalized after a model swap
    # must flag its own child, not whichever child is loaded by then.
    note_abandoned = getattr(engine, "note_stream_abandoned", None)
    abandon_hook = functools.partial(note_abandoned, handle) if callable(note_abandoned) else None

    # Log the extra_body AS SENT (post-translation, so llama.cpp's wire names
    # show up), one key=value per entry on the same line: the optional profile
    # keys (top_k / min_p / presence_penalty, #388) are otherwise invisible in
    # the INFO log and a QA pass cannot confirm they reached the server.
    extra_body_desc = ", ".join(f"{key}={value}" for key, value in extra_body.items())
    wire_effort = effort_plan.wire_effort if effort_plan is not None else None
    logger.info(
        f"ChatOpenAI built: model={model_field}, base_url={handle['base_url']}/v1, "
        f"temperature={temperature}, top_p={top_p}, max_tokens={max_tokens}, "
        f"reasoning_effort={wire_effort or 'unset'}, "
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
        # Native field of all three layers (1.1.2): llama-server reads "none"
        # as enable_thinking=false and forwards any other value to the chat
        # template; mlx_vlm normalizes it the same way. None = unset, dropped
        # from the payload.
        reasoning_effort=wire_effort,
        extra_body=extra_body,  # restore small-model coherence (repetition controls)
        timeout=None,  # cold model load can stall several seconds before first token
        max_retries=0,  # don't silently double-submit a slow local generation
        # langchain's ONE uniform per-chunk budget (120 s) cannot tell a long
        # prefill from a hang and killed healthy long-history turns (#573).
        # Off here; ``Erudi_Chat_OpenAI._astream`` enforces the two budgets that
        # replace it.
        stream_chunk_timeout=None,
        effective_context_tokens=effective_window,
        working_context_tokens=working_window,
        auto_output_budget=auto_output_budget,
        prompt_ratio=prompt_ratio,
        preflight_retry=preflight_retry,
        abandon_hook=abandon_hook,
        streaming=True,
        # ``stream_options.include_usage``: both local servers then end the
        # stream with a usage chunk (mlx_vlm 0.6.17 and llama-server alike);
        # a server that sends none leaves every consumer on its estimate.
        stream_usage=True,
    )
