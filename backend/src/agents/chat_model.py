"""The ChatOpenAI seam: what Erudi changes about the stock client, and where.

``Erudi_Chat_OpenAI`` (built lazily by :func:`erudi_chat_openai_class`) carries
four behaviours, each on the narrowest hook that expresses it:

1. **The #573 two-phase streaming watchdog**, in ``_astream`` -- the single
   place ``ChatOpenAI`` routes async streaming through, and the only hook that
   receives the messages positionally (the first-chunk budget is computed from
   what is actually being sent). Details below.
2. **The automatic output budget**, also in ``_astream``, and for the same
   reason: it is the one hook holding the FINAL message list of a model call,
   so the budget is recomputed per model hop of a tool turn, each one against
   the history that hop actually sends. The arithmetic and the reasoning live
   in ``src.agents.output_budget``.
3. **The #554 reasoning extraction**, in ``_convert_chunk_to_generation_chunk``
   -- the single place every raw streamed chunk dict is converted to a
   LangChain chunk, so it is the last point where the dedicated reasoning
   field the local servers emit (``delta.reasoning_content`` from llama-server
   under its default ``--reasoning-format auto``, ``delta.reasoning`` from
   mlx_vlm.server) is still visible: upstream's ``_convert_delta_to_message_chunk``
   drops it. The override re-attaches it as
   ``additional_kwargs["reasoning_content"]`` on the message chunk, which the
   runner turns into ``thinking`` events. ``finish_reason`` needs no help: the
   base method folds it into ``generation_info`` and langchain-core's stream
   loop folds that into the yielded message's ``response_metadata``, where the
   runner reads it (pinned in ``tests/test_stream_watchdog.py``).
4. **The legacy token-cap key**, in ``_get_request_payload`` -- see
   ``The wire name of the cap`` below.

The hooks are disjoint -- the conversion runs INSIDE the budgeted stream, so
extraction never loosens the watchdog -- and all of them rest on upstream
assumptions pinned by ``tests/test_stream_watchdog.py`` and
``tests/test_output_budget.py`` so a langchain-openai bump fails loudly instead
of silently restoring the old behavior.

The wire name of the cap
------------------------
OpenAI deprecated ``max_tokens`` in favour of ``max_completion_tokens`` in
2024, and stock ``ChatOpenAI`` renames the field on its way into the payload.
Erudi does not talk to OpenAI: it talks to two local servers, and only one of
them followed. llama-server accepts both names (``n_predict`` aliases each),
but mlx_vlm.server 0.6.17 reads ``max_tokens`` alone -- and because its request
schema DEFAULTS that field, the modern name is not rejected, it is silently
replaced by the server's own default (2048). ``_get_request_payload`` therefore
puts the cap back on the legacy key, the only one both children honour.

Why the watchdog replaces the uniform timeout (#573)
----------------------------------------------------
``langchain-openai`` (pinned 1.2.2) bounds every async SSE stream with ONE
uniform ``stream_chunk_timeout`` (default 120 s, overridable through
``LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S``), enforced by
``_astream_with_chunk_timeout`` in ``langchain_openai/chat_models/_client_utils.py``.
A local model has two silences with nothing in common:

- **Before the first chunk** the child server is prefilling the prompt. That
  silence grows with the conversation and has no upper bound the library can
  guess: a 6878-token turn on a Qwen3.5 9B (Apple M4 base, 16 GB) prefilled at
  51.8 tok/s, i.e. 132.9 s of nothing -- and the uniform 120 s budget killed the
  turn 13 s before its first token, with the child logging
  ``stream_closed_before_completion`` at 0 generated tokens. Retrying is
  structurally doomed: the history is longer every time.
- **Between chunks** the model is decoding, one token every few milliseconds.
  Two minutes of silence there is a genuine hang.

So the uniform knob is switched off (``stream_chunk_timeout=None``) and replaced
by a two-phase watchdog: the first chunk gets a budget computed from the prompt
actually being sent, every later chunk gets the fixed inter-chunk budget.

This is engine-agnostic on purpose. A slow CPU machine can exceed 120 s of
prefill on a few thousand tokens just as an Apple Silicon machine does at 6878;
nothing here branches on MLX / CUDA / CPU. The only engine fact that enters is
the loaded child's ALLOCATED context window (stamped on the client by the
factory), which raises the first-chunk ceiling so a legitimately full window
is never mistaken for a hang.

Both budgets are per MODEL CALL, not per turn: every hop of a tool-calling turn
pays its own prefill, and each one gets its own first-chunk budget. ``_astream``
is also where langchain routes a non-streaming ``ainvoke`` on a client built
with ``streaming=True`` (summarization, titles), so those calls are covered by
the same two budgets instead of the uniform one.

The watchdog raises ``GenerationTimeoutException`` and logs NOTHING: per
docs/logging.md the record belongs where the failure is handled, which is
``src.agents.runner`` (one WARNING carrying the phase, the budget and the
estimated prompt size, then an honest error turn).
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any, AsyncIterator, Iterable, Optional

from src.agents.output_budget import compute_output_budget, output_budget_override
from src.agents.reasoning_stream import REASONING_KWARG, extract_reasoning_delta
from src.core.exceptions import GenerationTimeoutException
from src.core.logging import logger

# Which silence was too long. Carried by the exception, logged by the handler.
PHASE_FIRST_CHUNK = "first-chunk"
PHASE_INTER_CHUNK = "inter-chunk"

# --- The first-chunk budget ------------------------------------------------
#
# budget = clamp(FLOOR, BASE + estimated_prompt_tokens / RATE, CEILING)
#
# All four constants are derived from the measured incident (#573): 6878 tokens
# prefilled at 51.8 tok/s = 132.9 s on the weakest chip currently in use (Apple
# M4 base, 16 GB) running a 9B model.
#
# - RATE is deliberately HALF the measured rate: the machine that matters is the
#   slowest one, not the one that was measured, and a CPU-only laptop prefills
#   far below 51.8 tok/s. Under-estimating the rate over-estimates the budget,
#   and over-budgeting only delays the detection of a true hang -- it never
#   kills a healthy turn. Under-budgeting does exactly what #573 reports.
# - BASE covers what is not proportional to the prompt: connecting, the child
#   server picking up the request, the chat template being applied.
# - FLOOR keeps short prompts at the old, field-proven 120 s.
# - CEILING stops the budget growing without limit. Its constant (900 s) was
#   sized against the old fixed 4096 window; with the dynamic context window
#   it is only the FLOOR OF THE CEILING: a full-window prompt on the child's
#   ALLOCATED window W legitimately needs BASE + W/RATE seconds of prefill,
#   so the effective ceiling is max(900, BASE + W/RATE). Keeping 900 s flat
#   would recreate the #573 kill on every long-window model the dynamic
#   window now allows -- the exact bug this watchdog exists to prevent. With
#   no known window (engine handle without one) 900 s stands.
#
# The incident's own turn lands at 30 + 6878/25 = 305 s, more than twice the
# 132.9 s it actually needed.
FIRST_CHUNK_FLOOR_S = 120.0
FIRST_CHUNK_BASE_S = 30.0
CONSERVATIVE_PREFILL_TOKENS_PER_SEC = 25.0
FIRST_CHUNK_CEILING_S = 900.0

# Absolute wall-clock backstop on the first-chunk budget, whatever the window.
# The window-scaled ceiling exists so a legitimately full ALLOCATED window is
# never mistaken for a hang -- but the catalog carries million-token windows,
# and scaling alone would let a genuinely hung child sit undetected for hours
# while ``generation_guard`` holds the engine's global lock (no other
# conversation, no model swap). A prefill that has produced nothing after an
# hour is not an experience worth waiting for on any machine this app targets:
# machines large enough to hold such prompts prefill far above the
# conservative rate, and machines that cannot hold them never reach prompts
# that size (the KV alone exceeds their memory). An honest timeout beats an
# eleven-hour lock.
FIRST_CHUNK_ABSOLUTE_MAX_S = 3600.0

# Once tokens flow, two minutes of silence is a hang: decode emits a token every
# few milliseconds on every engine. This is langchain's default, kept on purpose
# -- it was never wrong for THIS phase.
INTER_CHUNK_BUDGET_S = 120.0

# --- Estimating the prompt size -------------------------------------------
#
# The local server owns the tokenizer; the backend does not tokenize here (it
# would cost more than the budget it informs, on every hop of every turn). What
# it needs is not an average but an UPPER BOUND -- under-counting hands the
# budget back to the bug -- and the UTF-8 byte length of the text is a provable
# one: these models tokenize bytes (byte-level BPE, with byte fallback for
# anything unknown), and no token consumes less than one byte, so text of N
# bytes can never produce more than N tokens.
#
# A character-based heuristic is NOT a bound. Chinese and Japanese run near one
# token per character, and each of those characters is 3 UTF-8 bytes; code and
# punctuation-dense text sit at 1-2 characters per token. Something like
# "characters / 3" therefore under-counts CJK threefold -- 7000 Chinese
# characters would estimate ~2333 tokens and buy a ~123 s budget for ~135 s of
# real prefill, restoring the #573 kill for exactly the users the zh locale
# exists for.
#
# The bound is loose on English (~4 bytes per token), which only means a bigger
# budget, and long histories now land on FIRST_CHUNK_CEILING_S more often. That
# is the accepted trade: a too-large budget merely delays the detection of a
# true first-chunk hang, a too-small one ends a healthy turn.
#
# Same reasoning for the flat allowances below: per-message chat template
# overhead, and image parts whose true cost the backend cannot know. What the
# estimate cannot see at all (bound tool schemas, a system prompt injected
# downstream) is one more reason to keep every fudge factor pessimistic.
PER_MESSAGE_OVERHEAD_TOKENS = 8
NON_TEXT_PART_TOKENS = 1024


def _content_cost(content: Any) -> tuple[int, int]:
    """``(UTF-8 bytes of text, extra tokens)`` carried by one message's content.

    Handles the three shapes LangChain hands a chat model: a plain string, a
    list of content parts (``{"type": "text", ...}`` /
    ``{"type": "image_url", ...}``), or anything else (stringified).
    """
    if content is None:
        return 0, 0
    if isinstance(content, str):
        return len(content.encode("utf-8")), 0
    if isinstance(content, (list, tuple)):
        text_bytes = 0
        extra = 0
        for part in content:
            if isinstance(part, str):
                text_bytes += len(part.encode("utf-8"))
                continue
            text = part.get("text") if isinstance(part, dict) else None
            if isinstance(text, str):
                text_bytes += len(text.encode("utf-8"))
                continue
            # An image (or any part the backend cannot read): charge the flat
            # allowance rather than the base64 length, which says nothing about
            # how many tokens the vision encoder will produce.
            extra += NON_TEXT_PART_TOKENS
        return text_bytes, extra
    return len(str(content).encode("utf-8")), 0


def estimate_prompt_tokens(messages: Optional[Iterable[Any]]) -> int:
    """Upper bound on the tokens the messages about to be sent can produce.

    One token per UTF-8 byte of text (see above: a byte-level tokenizer cannot
    do better than one token per byte), plus the flat per-message and
    non-text-part allowances.

    Accepts ``BaseMessage`` objects (``.content``) and raw dicts
    (``{"role": ..., "content": ...}``) so it never fails on the shape it is
    handed. An unreadable message contributes its overhead and nothing else.
    """
    text_bytes = 0
    tokens = 0
    for message in messages or ():
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        part_bytes, part_tokens = _content_cost(content)
        text_bytes += part_bytes
        tokens += part_tokens + PER_MESSAGE_OVERHEAD_TOKENS
    return tokens + text_bytes


def first_chunk_budget_s(
    estimated_prompt_tokens: int, effective_window_tokens: Optional[int] = None
) -> float:
    """Seconds of silence allowed before the first chunk, for that prompt size.

    ``effective_window_tokens`` is the ALLOCATED window of the loaded child
    (``BaseEngine.effective_context_tokens``): it raises the ceiling to a
    full-window prefill (``max(900, BASE + W/RATE)``) so a legitimate
    window-filling prompt is never killed mid-prefill, while ``None`` (window
    unknown) keeps the field-proven 900 s. The ceiling only ever rises with
    the window -- a small window never cuts below 900 s.
    """
    ceiling = FIRST_CHUNK_CEILING_S
    if effective_window_tokens is not None and effective_window_tokens > 0:
        ceiling = max(
            ceiling,
            FIRST_CHUNK_BASE_S + effective_window_tokens / CONSERVATIVE_PREFILL_TOKENS_PER_SEC,
        )
    ceiling = min(ceiling, FIRST_CHUNK_ABSOLUTE_MAX_S)
    raw = FIRST_CHUNK_BASE_S + estimated_prompt_tokens / CONSERVATIVE_PREFILL_TOKENS_PER_SEC
    return min(max(FIRST_CHUNK_FLOOR_S, raw), ceiling)


async def stream_with_two_phase_budget(
    source: AsyncIterator[Any],
    *,
    first_budget_s: float,
    inter_budget_s: float,
    estimated_prompt_tokens: int = 0,
    model_name: Optional[str] = None,
) -> AsyncIterator[Any]:
    """Yield from ``source``, bounding the wait before each chunk.

    The FIRST chunk gets ``first_budget_s`` (prefill), every later chunk gets
    ``inter_budget_s`` (decode). Either budget expiring raises
    ``GenerationTimeoutException`` carrying the phase; the source is closed on
    the way out so the child server's HTTP stream is released instead of being
    left to the garbage collector.
    """
    iterator = source.__aiter__()
    chunks_received = 0
    try:
        while True:
            budget = first_budget_s if chunks_received == 0 else inter_budget_s
            try:
                chunk = await asyncio.wait_for(iterator.__anext__(), timeout=budget)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError as exc:
                phase = PHASE_FIRST_CHUNK if chunks_received == 0 else PHASE_INTER_CHUNK
                raise GenerationTimeoutException(
                    f"No streaming chunk after {budget:.0f}s ({phase}); "
                    f"model={model_name or '?'}, chunks_received={chunks_received}",
                    phase=phase,
                    budget_s=budget,
                    estimated_prompt_tokens=estimated_prompt_tokens,
                ) from exc
            chunks_received += 1
            yield chunk
    finally:
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:
                # Best-effort release of the HTTP stream: a cleanup failure must
                # never replace the timeout (or the answer) the caller is after.
                logger.debug("Closing the model stream raised; ignoring", exc_info=True)


@lru_cache(maxsize=1)
def erudi_chat_openai_class():
    """The ``ChatOpenAI`` subclass every chat client is built from.

    Built lazily, and cached: ``langchain_openai`` must not load at boot
    (issue #160, guarded by ``tests/test_lazy_langchain_imports.py``), so the
    class that inherits from it cannot exist at module import time.

    ``_astream`` is the narrowest hook that survives a patch bump: it is where
    ``ChatOpenAI`` routes async streaming (Chat Completions or Responses API),
    it receives the messages positionally -- which is what the first-chunk
    budget is computed from -- and wrapping it leaves every other path
    (``ainvoke``, the sync ``stream``) untouched. Same logic for
    ``_convert_chunk_to_generation_chunk``: it sees every raw streamed chunk
    dict exactly once, before upstream throws the reasoning field away. The
    assumptions both rest on are pinned by ``tests/test_stream_watchdog.py``.
    """
    from langchain_openai import ChatOpenAI

    class Erudi_Chat_OpenAI(ChatOpenAI):
        """``ChatOpenAI`` with the #573 watchdog and the #554 reasoning carry."""

        # The ALLOCATED window of the child this client points at, stamped by
        # the factory at build time (the client is rebuilt every turn, right
        # after the engine resolved the model, so the value is fresh across
        # model swaps). It raises the first-chunk ceiling to a full-window
        # prefill; None (unknown window) keeps the 900 s constant.
        effective_context_tokens: Optional[int] = None

        # Whether this client's ``max_tokens`` is a fallback the automatic
        # budget may replace (chat turns and the summarization calls that ride
        # the same client) or a DELIBERATE budget it must leave alone. The
        # one-shot utility path sets this False: a conversation title runs on
        # ~12 tokens on purpose (#266), and handing it the whole window would
        # make it ramble for thousands of tokens before the sanitizer took its
        # first four words. ``ainvoke`` on a ``streaming=True`` client routes
        # through ``_astream``, so the distinction has to live here.
        auto_output_budget: bool = True

        # Whether the child this client points at REJECTS a request whose
        # prompt plus requested generation overflows the window (MLX's
        # preflight validator) rather than clamping it (llama-server). Stamped
        # by the factory from ``BaseEngine.preflight_counts_output_tokens()``.
        # When it does, the budget below is additionally held under the byte
        # bound, which is provably >= the real prompt -- otherwise a budget
        # sized from the chars/4 estimate could make the app reject its own
        # turn on text that estimate under-counts (CJK).
        preflight_counts_output: bool = False

        async def _astream(self, messages, *args, **kwargs):
            # One byte-bound estimate, two consumers: the first-chunk watchdog
            # budget below, and -- on a preflighting child -- the safety cap on
            # the output budget. Both need an UPPER bound; only the output
            # budget's SIZE comes from the chars/4 counter instead (the
            # duality is spelled out in src.agents.output_budget).
            estimated = estimate_prompt_tokens(messages)
            # What this call may generate: the window minus what the turn
            # already occupies. Per model call, not per turn -- every hop of a
            # tool turn sends a longer history and gets a smaller budget. A
            # kwarg wins over the constructor's ``max_tokens`` in
            # ``_get_request_payload`` (pinned); ``None`` leaves that value
            # alone, which is what an engine with no reportable window gets.
            budget = (
                compute_output_budget(
                    messages,
                    self.effective_context_tokens,
                    override=output_budget_override(),
                    prompt_upper_bound_tokens=(estimated if self.preflight_counts_output else None),
                )
                if self.auto_output_budget
                else None
            )
            if budget is not None:
                kwargs["max_tokens"] = budget
            source = super()._astream(messages, *args, **kwargs)
            async for chunk in stream_with_two_phase_budget(
                source,
                # Read from the module (not captured) so the budgets stay one
                # source of truth -- and patchable in tests.
                first_budget_s=first_chunk_budget_s(
                    estimated, effective_window_tokens=self.effective_context_tokens
                ),
                inter_budget_s=INTER_CHUNK_BUDGET_S,
                estimated_prompt_tokens=estimated,
                model_name=self.model_name,
            ):
                yield chunk

        def _get_request_payload(self, input_, *, stop=None, **kwargs):
            """Send the token cap as ``max_tokens`` (see the module docstring).

            Stock ``ChatOpenAI`` renames it to ``max_completion_tokens``, which
            mlx_vlm.server silently replaces with its own default. Renaming it
            back here covers every path that builds a payload -- streaming and
            non-streaming alike -- and leaves the value itself untouched.
            """
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
            if "max_completion_tokens" in payload:
                payload["max_tokens"] = payload.pop("max_completion_tokens")
            return payload

        def _convert_chunk_to_generation_chunk(
            self, chunk, default_chunk_class, base_generation_info
        ):
            """Carry the servers' dedicated reasoning field to the runner (#554).

            The base conversion drops ``delta.reasoning_content`` /
            ``delta.reasoning``; copy the raw delta's reasoning onto the
            message chunk's ``additional_kwargs`` so the runner can emit it as
            ``thinking`` events. Everything else -- content, tool_call_chunks,
            ``finish_reason`` into ``generation_info`` -- is the base method's
            result, untouched.
            """
            generation_chunk = super()._convert_chunk_to_generation_chunk(
                chunk, default_chunk_class, base_generation_info
            )
            if generation_chunk is None:
                return None
            reasoning = extract_reasoning_delta(chunk)
            if reasoning:
                generation_chunk.message.additional_kwargs[REASONING_KWARG] = reasoning
            return generation_chunk

    return Erudi_Chat_OpenAI
