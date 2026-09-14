"""Two streaming budgets per model call: prompt-sized prefill, then decode (#573).

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
prefill at 4096 context just as an Apple Silicon machine does at 6878 tokens;
nothing here branches on MLX / CUDA / CPU.

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
# - CEILING stops the budget growing without limit: a turn that has not started
#   after 15 minutes is not worth waiting for, whatever the arithmetic says.
#
# The incident's own turn lands at 30 + 6878/25 = 305 s, more than twice the
# 132.9 s it actually needed.
FIRST_CHUNK_FLOOR_S = 120.0
FIRST_CHUNK_BASE_S = 30.0
CONSERVATIVE_PREFILL_TOKENS_PER_SEC = 25.0
FIRST_CHUNK_CEILING_S = 900.0

# Once tokens flow, two minutes of silence is a hang: decode emits a token every
# few milliseconds on every engine. This is langchain's default, kept on purpose
# -- it was never wrong for THIS phase.
INTER_CHUNK_BUDGET_S = 120.0

# --- Estimating the prompt size -------------------------------------------
#
# The local server owns the tokenizer; the backend does not tokenize here (it
# would cost more than the budget it informs, on every hop of every turn).
# 3 characters per token deliberately OVER-estimates -- real tokenizers average
# closer to 4 on English and ~2-3 on CJK -- and over-estimating only buys a
# bigger budget. Same reasoning for the flat allowances below: per-message chat
# template overhead, and image parts whose true cost the backend cannot know.
# What the estimate CANNOT see (bound tool schemas, a system prompt injected
# downstream) is one more reason to keep every fudge factor pessimistic.
CHARS_PER_ESTIMATED_TOKEN = 3.0
PER_MESSAGE_OVERHEAD_TOKENS = 8
NON_TEXT_PART_TOKENS = 1024


def _content_cost(content: Any) -> tuple[int, int]:
    """``(characters, extra tokens)`` carried by one message's content.

    Handles the three shapes LangChain hands a chat model: a plain string, a
    list of content parts (``{"type": "text", ...}`` /
    ``{"type": "image_url", ...}``), or anything else (stringified).
    """
    if content is None:
        return 0, 0
    if isinstance(content, str):
        return len(content), 0
    if isinstance(content, (list, tuple)):
        chars = 0
        extra = 0
        for part in content:
            if isinstance(part, str):
                chars += len(part)
                continue
            text = part.get("text") if isinstance(part, dict) else None
            if isinstance(text, str):
                chars += len(text)
                continue
            # An image (or any part the backend cannot read): charge the flat
            # allowance rather than the base64 length, which says nothing about
            # how many tokens the vision encoder will produce.
            extra += NON_TEXT_PART_TOKENS
        return chars, extra
    return len(str(content)), 0


def estimate_prompt_tokens(messages: Optional[Iterable[Any]]) -> int:
    """Pessimistic token count for the messages about to be sent.

    Accepts ``BaseMessage`` objects (``.content``) and raw dicts
    (``{"role": ..., "content": ...}``) so it never fails on the shape it is
    handed. An unreadable message contributes its overhead and nothing else.
    """
    chars = 0
    tokens = 0
    for message in messages or ():
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        part_chars, part_tokens = _content_cost(content)
        chars += part_chars
        tokens += part_tokens + PER_MESSAGE_OVERHEAD_TOKENS
    return int(tokens + chars / CHARS_PER_ESTIMATED_TOKEN)


def first_chunk_budget_s(estimated_prompt_tokens: int) -> float:
    """Seconds of silence allowed before the first chunk, for that prompt size."""
    raw = FIRST_CHUNK_BASE_S + estimated_prompt_tokens / CONSERVATIVE_PREFILL_TOKENS_PER_SEC
    return min(max(FIRST_CHUNK_FLOOR_S, raw), FIRST_CHUNK_CEILING_S)


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
    (``ainvoke``, the sync ``stream``) untouched. The assumptions it rests on
    are pinned by ``tests/test_stream_watchdog.py``.
    """
    from langchain_openai import ChatOpenAI

    class Erudi_Chat_OpenAI(ChatOpenAI):
        """``ChatOpenAI`` with the #573 two-phase streaming watchdog."""

        async def _astream(self, messages, *args, **kwargs):
            estimated = estimate_prompt_tokens(messages)
            source = super()._astream(messages, *args, **kwargs)
            async for chunk in stream_with_two_phase_budget(
                source,
                first_budget_s=first_chunk_budget_s(estimated),
                # Read from the module (not captured) so the budgets stay one
                # source of truth -- and patchable in tests.
                inter_budget_s=INTER_CHUNK_BUDGET_S,
                estimated_prompt_tokens=estimated,
                model_name=self.model_name,
            ):
                yield chunk

    return Erudi_Chat_OpenAI
