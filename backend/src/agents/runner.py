"""AgentRunner — the shared conversation/arena streaming primitive.

One ``create_agent`` per turn. The turn is captured as STRUCTURED EVENTS (#90):
``_astream_events`` yields dicts — ``{"t":"answer",...}``, ``{"t":"thinking",...}``,
``{"t":"tool_call",...}``, ``{"t":"tool_result",...}`` — so thinking and tool
activity are surfaced instead of dropped. ``astream_text`` is a thin projection
over those events with two modes selected by ``emit_events``:

  - ``emit_events=True`` (conversations): yields the event dicts unchanged; the
    conversation service frames them as NDJSON and persists a replayable trace.
  - ``emit_events=False`` (arena / default): yields ONLY answer text as ``str``,
    dropping thinking + tool events — byte-for-byte the old plain-text contract,
    so arena and its wire stay untouched. Reasoning stays hidden there because
    the same splitter strips inline ``<think>`` before the text is yielded.

  - Conversation: ``thread_id`` set + ``summarize=True`` + a checkpointer →
    history is restored from the checkpointer (only the new message is sent),
    and old turns are summarized in the agent state.
  - Arena: ``thread_id=None`` + ``summarize=False`` + no checkpointer → a
    stateless single-model call.

Thinking events come from two sources (#554). The primary one is the dedicated
reasoning channel: both local servers extract each family's chain-of-thought
server-side (llama-server's default ``--reasoning-format auto``, mlx_vlm's
native split) and ``Erudi_Chat_OpenAI`` re-attaches it to every streamed chunk
as ``additional_kwargs["reasoning_content"]``. The fallback is the streaming
ThinkSplitter on the content channel, for families whose markers the server
parser does not know -- and Arena's plain-text projection depends on it to keep
inline ``<think>`` out of its wire. A turn that ends with reasoning but no
answer text (and no tool result to fall back to, #90) yields a curated
empty-answer line -- a NORMAL answer picked by ``finish_reason``, never the
ERROR sentinel, so the trace survives persistence -- and, on stateful runs,
writes that line into the thread state in place of the empty assistant message.

On tool-carrying turns (#297), text a model hop streams BEFORE its tool call is
not the answer — it is pre-answer narration (often hallucinated guessing on
small local models: an invented payload figure narrated at length, THEN the
``search_knowledge_base`` call, THEN the grounded answer). The capture loop
therefore BUFFERS each hop's post-splitter answer text instead of yielding it:
the moment the hop's first ``tool_call_chunk`` arrives, the buffer is re-emitted
as ``thinking`` events (before the ``tool_call`` event) and further text in that
hop streams as thinking too; a hop that ends WITHOUT a tool call flushes its
buffer as ``answer`` at stream end. The accepted cost is that on agentic turns
the final answer's text appears at hop end rather than token-by-token; plain
and systematic (zero-tool) turns are untouched and still stream live.

Everything runs inside ``engine.generation_guard()`` so model resolution + the
whole stream serialize on the single-model engine and the idle-cleanup monitor
never reaps the model mid-stream.

LangChain imports are deferred to the methods that use them (issue #160):
this module is imported at boot by the conversation/arena services, but the
agent stack is only needed on the FIRST turn, so keeping the imports
function-scoped keeps ``import src.main`` fast. ``build_chat_model`` stays a
module-level name — tests monkeypatch ``runner.build_chat_model`` — and is
itself LangChain-free at import time (``ChatOpenAI`` is deferred inside it).
"""

from __future__ import annotations

import contextlib
import functools
import json
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

from fastapi.concurrency import run_in_threadpool

from src.agents.chat_model import (
    INTER_CHUNK_BUDGET_S,
    PHASE_FIRST_CHUNK,
    first_chunk_ceiling_s,
    is_child_prefill_timeout,
)
from src.agents.isolated_stream import isolated_stream
from src.agents.model_factory import build_chat_model
from src.agents.output_budget import OUTPUT_BUDGET_FLOOR_TOKENS
from src.agents.overflow import ContextOverflow, parse_context_overflow
from src.agents.reasoning_effort import NO_REASONING_PLAN, EffortPlan
from src.agents.token_accounting import (
    COUNTER_DENSE_TOKENS,
    COUNTER_WEIGHT_FLOOR,
    STALE_TOOL_RESULT_MARKERS,
    SUMMARY_SOURCE_MARKER,
    RequestOverhead,
    fresh_weight,
    last_human_index,
    message_weights,
    overhead_tokens,
    request_overhead,
    stale_result_copy,
    weighted_cost,
)
from src.database.generation_hints import resolve_sampling_defaults
from src.agents.think_splitter import ThinkSplitter
from src.core import config
from src.core.exceptions import EngineException, GenerationTimeoutException
from src.core.logging import logger
from src.engines.base_engine import BaseEngine, run_reset_shielded
from src.engines.memory_budget import MemoryBudget
from src.engines.working_window import canonical_working_window

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


# Auto-summarization (compaction) fires at 80 % of the WORKING window, the
# smaller of two values recomputed PER TURN (``_build_middleware`` runs after
# the child spawned, so both are fresh):
#   1. the ALLOCATED context window (``BaseEngine.effective_context_tokens``);
#   2. the MEMORY ceiling (Apple Silicon) -- the conversation token count at
#      which the child's predicted footprint reaches the memory budget
#      (``MemoryBudget.tokens_at_ceiling``, a measured prior; no margin is
#      stacked on top of it).
# The 20-message floor stays as the trigger when no window is readable
# (``W_eff=None``), and always rides along with OR semantics. Once triggered,
# older turns are summarized by the same local model and replaced in the
# checkpointer state; the Message table keeps the full history for display.
# The summary prompt handed to the compaction middleware. The library default
# is a generic extraction prompt, and the release recette proved it LOSSY on
# small local models: summarizing 11 messages of a 4B conversation, it kept
# the dominant interaction pattern and dropped the one user-stated fact (a
# name planted early on), which a follow-up then answered wrongly. Compaction
# is this app's memory resolver, so the prompt puts concrete facts FIRST.
# ASCII, addressed to the summarizing model.
SUMMARY_PROMPT = (
    "Summarize the conversation below for the assistant's own memory. "
    "FIRST, list every concrete fact the user stated about themselves or "
    "their world (names, numbers, dates, preferences, decisions), each on "
    "its own line, exactly as stated -- these lines are the summary's most "
    "important content and must never be dropped or generalized. "
    "THEN describe in a few sentences what was discussed and what the "
    "assistant did. Do not invent anything; omit nothing the user asked to "
    "remember.\n\nConversation:\n{messages}"
)

SUMMARY_TRIGGER_MESSAGES = 20
SUMMARY_KEEP_MESSAGES = 10
COMPACTION_WINDOW_FRACTION = 0.8
# The amber warning's threshold: it shows when, even after compacting to the
# kept budget, the model plus the kept conversation would use more than 85 %
# of the memory budget (see ``_memory_warning_event``). It is the warning's
# threshold ONLY: the compaction ceiling is the honest one, with no margin
# stacked on top (the prior carries its own margin, ``memory_budget``).
MEMORY_WARNING_MARGIN = 0.15
# Token allowance for the summary a compaction would insert, used when
# projecting the post-compaction size the warning is judged on -- ONLY when no
# working window is known. With a window, the projection uses the summary's
# real cap, ``summary_cap(W)``.
SUMMARY_TOKEN_ALLOWANCE = 512

# What a compaction keeps (#611). The kept tail is bounded TWICE: at most
# ``SUMMARY_KEEP_MESSAGES`` messages AND at most ``KEEP_FRACTION`` of the
# working window, in tokens -- whichever keeps less. One bound alone loops:
# the last 10 messages can by themselves exceed the 80 % token trigger (a
# message keep would re-fire compaction on every turn), and a token keep of
# many short messages can leave >= 20 messages, which re-fires the message
# floor (OR'd into the trigger). The token bound also leaves room for the
# summary and a margin under the trigger, so with ordinary messages the
# post-compaction state stays under ``0.8 W - 256`` for every window (the
# exceptions ``compaction_cutoff`` documents keep more): ``max(1, min(0.4 W, 0.8 W -
# summary_cap(W) - 256))`` (non-positive below ~480 tokens, hence the floor of
# one token). 0.4 is a design value, not a measurement: it trades kept detail
# for room to grow before the next compaction.
KEEP_FRACTION = 0.4
POST_COMPACTION_MARGIN_TOKENS = 256

# The summary's own length is BOUNDED: the summary client is built with
# ``max_tokens = summary_cap(W)`` and no automatic output budget (which would
# otherwise hand it the whole window minus the prompt). An eighth of the
# window, clamped to [128, 1024].
SUMMARY_CAP_FLOOR_TOKENS = 128
SUMMARY_CAP_CEILING_TOKENS = 1024

# How much history the summarizer reads. LangChain trims what it summarizes to
# 4000 tokens by default (strategy "last"), so with a big window the previous
# summary would fall out of the summarizer's input and its facts would be lost
# at every compaction. With a known window the budget is
#   max(256, min(max(4000, 0.8 W - cap),
#                W - 2 cap - 256,
#                (W_alloc - 2 cap - 512) // 2))
# -- the second term keeps the summarizer prompt (input + the prepended
# previous summary + the summary it writes, each up to ``cap``) inside the
# MEMORY window, the third inside the ALLOCATED window. The counter is in real
# tokens now (a measured ratio), but that ratio is a whole-REQUEST average: an
# English system prompt and tool schemas around a CJK history (O_est 1500 at
# r ~ 1.1, history at r ~ 3: r = 2.37) under-count the history by ~1.3x. The
# factor 2 covers that heterogeneity; with no factor the summarizer prompt at
# W = 8192 would leave ~9 % of slack in the allocated window. The previous
# summary is always prepended on top of this budget, never trimmed away.
SUMMARY_TRIM_DEFAULT_TOKENS = 4000
SUMMARY_TRIM_FLOOR_TOKENS = 256
SUMMARY_TRIM_MEMORY_MARGIN_TOKENS = 256
SUMMARY_TRIM_ALLOCATED_MARGIN_TOKENS = 512
SUMMARY_TRIM_HETEROGENEITY_FACTOR = 2

# chars per token of ``count_tokens_approximately`` (its default), used to
# truncate an oversized message to a token budget (divided by the measured
# ratio, so the cut is in real tokens).
_APPROX_CHARS_PER_TOKEN = 4
# Room left for the role and per-message overhead when truncating a message.
_TRUNCATION_OVERHEAD_TOKENS = 16
# LangChain's fallback when ``trim_messages`` itself fails.
_TRIM_FALLBACK_MESSAGE_COUNT = 15

# The summary message LangChain inserts (``_build_new_messages``, pinned in
# tests/test_compaction_keep.py) and the marker it carries.
SUMMARY_MESSAGE_PREFIX = "Here is a summary of the conversation to date:\n\n"
# The placeholder a compaction writes when the summary could not be produced
# (see ``_summary_placeholder``). ASCII, addressed to the model.
SUMMARY_LATER_LOST = "Later messages of this conversation could not be summarized."
SUMMARY_EARLIER_LOST = "Earlier messages of this conversation could not be summarized."


def summarization_triggers(working_window: Optional[int]) -> list:
    """The OR-semantics trigger list for ``SummarizationMiddleware``.

    ``working_window`` is the ONE canonical working context window
    (``src.engines.working_window.canonical_working_window`` -- the minimum of
    the allocated window and the memory ceiling, over whichever candidates are
    known). Its 80 % (``COMPACTION_WINDOW_FRACTION``) becomes the token
    threshold, OR'd with the ``SUMMARY_TRIGGER_MESSAGES`` message floor. The
    memory fold happens at the call site, so this function sees a single value:
    ``None`` (no window and no memory ceiling) leaves the message floor alone.

    Deliberately never ``("fraction", ...)``: that form needs a
    ``model.profile`` our local chat clients do not carry (the middleware's
    ``__init__`` would raise). The middleware tests the token clause against
    ``count + O`` in real tokens (``_should_summarize``).
    """
    triggers: list = []
    if (
        isinstance(working_window, int)
        and not isinstance(working_window, bool)
        and working_window > 0
    ):
        triggers.append(("tokens", max(1, int(COMPACTION_WINDOW_FRACTION * working_window))))
    triggers.append(("messages", SUMMARY_TRIGGER_MESSAGES))
    return triggers


def _close_memory_window(engine: Any, token: Any, *, abandoned: bool) -> None:
    """Close the turn's memory measurement window; a failure costs the
    observation, never the turn (one WARNING)."""
    if token is None:
        return
    try:
        engine.end_memory_window(token, abandoned=abandoned)
    except Exception:
        logger.warning("Closing the memory measurement window failed", exc_info=True)


def _track_memory_warning(engine: Any, thread_id: Optional[str], active: bool) -> None:
    """Per-conversation warning state on the loaded child's handle, with ONE
    INFO line at each transition (start, end). Lost when the child is reaped
    or swapped: the next warning simply starts again."""
    handle = getattr(engine, "_model", None)
    if not isinstance(handle, dict):
        return
    state = handle.setdefault("warning_state", {})
    if active and thread_id not in state:
        state[thread_id] = True
        logger.info(f"Memory warning started: thread_id={thread_id}")
    elif not active and thread_id in state:
        state.pop(thread_id, None)
        logger.info(f"Memory warning ended: thread_id={thread_id}")


def _flag_abandoned(abandon_hook: Any) -> None:
    """Flag the child a turn or title ran against, from an ``except
    BaseException`` that re-raises.

    A failure of the hook must never replace the exit in flight (the
    consumer's GeneratorExit or cancellation): it is one WARNING with its
    traceback instead, like ``Erudi_Chat_OpenAI._notify_abandoned``.
    """
    if abandon_hook is None:
        return
    try:
        abandon_hook()
    except Exception:
        # The exit in flight is what propagates; this record says the
        # runner-level flag was lost (the next reset then skips its barrier).
        logger.warning("Flagging an abandoned turn failed", exc_info=True)


def _engine_overrides(engine: Any, hook_name: str) -> bool:
    """Whether ``engine`` implements the prefix-cache hook ``hook_name``
    itself, rather than inheriting ``BaseEngine``'s no-op.

    Only MLX does. The llama.cpp engines (CPU, CUDA) must stay a strict
    no-op: no reset thread, no ``BaseEngine._pending_reset`` -- so the runner
    calls ``run_reset_shielded`` only for an engine that overrides the hook.
    """
    hook = getattr(engine, hook_name, None)
    if not callable(hook):
        return False
    base = getattr(BaseEngine, hook_name)
    return getattr(hook, "__func__", hook) is not getattr(base, "__func__", base)


def _is_marker(message: Any) -> bool:
    """A tool result already sent as its stale-result marker."""
    return getattr(message, "type", None) == "tool" and getattr(
        message, "content", None
    ) == STALE_TOOL_RESULT_MARKERS.get(getattr(message, "name", None))


def counter_weight(message: Any) -> float:
    """The counter's weight for a message nothing was frozen for (rule 2 with
    the counter's settings: CJK one token per character, floor 1.2)."""
    return fresh_weight(message, dense=COUNTER_DENSE_TOKENS, weight_floor=COUNTER_WEIGHT_FLOOR)


def frozen_weights(messages: Any) -> tuple[dict, Optional[float]]:
    """``({message id: weight}, r)`` for a list of messages AS SENT, decided
    once from the whole list (``token_accounting.message_weights`` with the
    counter's settings)."""
    messages = list(messages)
    weights, ratio = message_weights(
        messages, dense=COUNTER_DENSE_TOKENS, weight_floor=COUNTER_WEIGHT_FLOOR
    )
    by_id = {
        message.id: weight
        for message, weight in zip(messages, weights)
        if getattr(message, "id", None) is not None
    }
    return by_id, ratio


def real_token_count(
    messages: Any, weights_by_id: Optional[dict] = None, past_ids: frozenset = frozenset()
) -> int:
    """THE token counter of compaction, in real tokens, over the messages AS
    SENT: each message costs ``ceil(weight * chars/4 of that copy)``.

    ``weights_by_id`` is frozen once per call from the full state
    (``frozen_weights``), so the count works on any list LangChain hands it --
    suffixes, the reversed pool of ``trim_messages``, partial copies (same id,
    shorter content: they cost their share). A ToolMessage whose id is in
    ``past_ids`` is sent as its marker, and a marker costs the weight of its
    OWN text, never the frozen weight of the result it replaces. A message
    nothing was frozen for (a new summary, a re-ided copy) gets the counter's
    rule 2 on the fly. Additive: ``count(a + b) == count(a) + count(b)``.

    A distinct function on purpose: handed ``count_tokens_approximately``
    itself, LangChain's ``SummarizationMiddleware`` swaps in a variant that
    rescales the count with the last AI message's reported usage -- a stale
    total that counts messages compaction already removed.
    """
    weights_by_id = weights_by_id or {}
    total = 0
    for message in messages:
        message_id = getattr(message, "id", None)
        if past_ids and message_id in past_ids:
            message = stale_result_copy(message)
        if _is_marker(message):
            weight = counter_weight(message)
        else:
            weight = weights_by_id.get(message_id) if message_id is not None else None
            if weight is None:
                weight = counter_weight(message)
        total += weighted_cost(message, weight)
    return total


def _known_window(window: Any) -> bool:
    """A real positive int (``bool`` rejected), like the trigger's own test."""
    return isinstance(window, int) and not isinstance(window, bool) and window > 0


def summary_cap(working_window: int) -> int:
    """The summary's token cap for this working window: ``W // 8`` clamped
    to [128, 1024]. Also the summary size the keep budget and the amber
    warning projection reserve."""
    return max(SUMMARY_CAP_FLOOR_TOKENS, min(SUMMARY_CAP_CEILING_TOKENS, working_window // 8))


def keep_token_budget(working_window: int) -> int:
    """Tokens a compaction may keep: ``max(1, min(0.4 W, 0.8 W - cap - 256))``."""
    room_under_trigger = (
        int(COMPACTION_WINDOW_FRACTION * working_window)
        - summary_cap(working_window)
        - POST_COMPACTION_MARGIN_TOKENS
    )
    return max(1, min(int(KEEP_FRACTION * working_window), room_under_trigger))


def summarize_trim_budget(working_window: Optional[int], allocated_window: Optional[int]) -> int:
    """Tokens of history the summarizer reads (see the constants above).

    No known working window: LangChain's own 4000. The allocated-window term
    is dropped when the allocation is unknown.
    """
    if not _known_window(working_window):
        return SUMMARY_TRIM_DEFAULT_TOKENS
    cap = summary_cap(working_window)
    terms = [
        max(SUMMARY_TRIM_DEFAULT_TOKENS, int(COMPACTION_WINDOW_FRACTION * working_window) - cap),
        working_window - 2 * cap - SUMMARY_TRIM_MEMORY_MARGIN_TOKENS,
    ]
    if _known_window(allocated_window):
        terms.append(
            (allocated_window - 2 * cap - SUMMARY_TRIM_ALLOCATED_MARGIN_TOKENS)
            // SUMMARY_TRIM_HETEROGENEITY_FACTOR
        )
    return max(SUMMARY_TRIM_FLOOR_TOKENS, min(terms))


def compaction_cutoff(
    messages: list,
    working_window: Optional[int],
    counter: Any,
    *,
    overhead: int = 0,
    allocated_window: Optional[int] = None,
) -> int:
    """Index of the first message a compaction keeps (0: nothing to compact).

    THE cutoff rule, pure: the summarization middleware uses it, and so does
    the amber-warning projection (which has no middleware instance).
    ``counter`` counts real tokens as sent; ``overhead`` is O, what the
    request carries beyond the state (system prompt, KB block, tool schemas),
    already scaled.

    1. With a known window, the later of two cutoffs -- the earliest suffix
       that fits ``max(1, keep_token_budget(W) - O)`` tokens and the one
       keeping ``SUMMARY_KEEP_MESSAGES`` messages -- so the kept tail
       respects BOTH bounds. Without a window, the message cutoff alone.
    2. Never past the LAST user message: the question the current turn
       answers is never summarized away mid-turn (a tool round after it can
       be large on its own).
    3. Never ON a user message: LangChain inserts the summary as a user
       message, and two user messages in a row make the strict chat
       templates (Gemma, Mistral) reject every later turn, so the kept tail
       always starts with an assistant message and the state alternates
       ``[summary, answer, question, ...]``. On an OLDER user message the
       cutoff moves FORWARD to the next assistant message (keeping less, so
       the budget holds); only on the current question -- or when no
       assistant message lies between the cutoff and it -- does it move BACK
       to the answer before it.
    4. Tool-pair safe (LangChain's ``_find_safe_cutoff_point``): an AI
       message is never separated from its tool results -- moving back can
       land in a tool round, and the cutoff then moves back to its call.
    5. If only the previous summary would be summarized (cutoff <= 1 after a
       summary), nothing is: re-summarizing a summary alone adds nothing and
       would reset the prefix cache for nothing.
    6. Futility guard -- only with a known window and only when the TOKEN
       trigger fired (``count + O >= 0.8 W``; the 20-message clause alone
       keeps the rules above): a compaction whose gain
       ``count(all) - count(kept) - summary_cap(W)`` is below
       ``POST_COMPACTION_MARGIN_TOKENS`` (256) -- nothing at all included --
       is skipped unless the request would overflow without it: against the
       allocated window (``O + count + 512 > W_alloc``, the output budget's
       floor) or the working window (``O + count > W``). A small gain is not
       worth a summary call and a prefix-cache reset; any compaction that may
       save an overflowing turn is taken (the gain is reckoned with the
       summary's CAP, the real summary is often shorter).

    Termination: every compaction strictly lowers the number of messages
    before the last user message; steps 2-3 bound the cutoff at
    the answer before the current question, and there step 5 returns 0 on
    the next hop -- the kept tail can stay above the trigger, so termination
    rests on step 5, not on a margin under the trigger.

    With ordinary messages the kept tail stays inside the token budget. It
    exceeds it only by the answer before the current question when the cut
    falls ON that question (steps 2-3), by a last message larger than the
    budget on its own (kept whole, summarized on a later turn, never
    truncated), and by a tool round kept with its call (step 4) -- accepted,
    the alternatives break the conversation.
    """
    from langchain.agents.middleware import SummarizationMiddleware
    from langchain_core.messages import HumanMessage, ToolMessage

    safe_point = SummarizationMiddleware._find_safe_cutoff_point
    if len(messages) <= SUMMARY_KEEP_MESSAGES:
        cutoff = 0
    else:
        cutoff = safe_point(messages, len(messages) - SUMMARY_KEEP_MESSAGES)
    if _known_window(working_window):
        keep_budget = max(1, keep_token_budget(working_window) - overhead)
        token_cut = _token_cutoff(messages, keep_budget, counter)
        cutoff = max(token_cut, cutoff)
    if cutoff <= 0:
        return 0
    last_human = last_human_index(messages)
    if last_human is not None:
        cutoff = min(cutoff, last_human)
    if (
        last_human is not None
        and cutoff < last_human
        and isinstance(messages[cutoff], HumanMessage)
    ):
        # An OLDER user message: move FORWARD to the next assistant message
        # (tool results after a call are fine, they follow it). Moving back
        # would keep the answer before it -- one that did not fit the budget.
        forward = cutoff
        while forward <= last_human and isinstance(messages[forward], (HumanMessage, ToolMessage)):
            forward += 1
        if forward <= last_human:
            cutoff = forward
    # Only on the current question (or with no assistant message between the
    # cutoff and it): every step moves strictly back, so the loop ends.
    while cutoff > 0:
        message = messages[cutoff]
        if isinstance(message, HumanMessage):
            cutoff -= 1
        elif isinstance(message, ToolMessage):
            paired = safe_point(messages, cutoff)
            # LangChain answers an orphaned tool result (no matching call)
            # by moving FORWARD, which could land back on a user message:
            # step back past it instead.
            cutoff = paired if paired < cutoff else cutoff - 1
        else:
            break
    if cutoff <= 1 and messages and _is_previous_summary(messages[0]):
        return 0
    if cutoff > 0 and _known_window(working_window):
        total = counter(messages)
        threshold = max(1, int(COMPACTION_WINDOW_FRACTION * working_window))
        if total + overhead >= threshold:
            gain = total - counter(messages[cutoff:]) - summary_cap(working_window)
            overflows = overhead + total > working_window or (
                _known_window(allocated_window)
                and overhead + total + OUTPUT_BUDGET_FLOOR_TOKENS > allocated_window
            )
            if gain < POST_COMPACTION_MARGIN_TOKENS and not overflows:
                return 0
    return cutoff


def _token_cutoff(messages: list, budget: int, counter: Any) -> int:
    """LangChain's ``_find_token_based_cutoff``, for a given budget: the
    earliest index whose suffix fits, the last message kept whole when it
    alone exceeds the budget, then tool-pair safe."""
    from langchain.agents.middleware import SummarizationMiddleware

    if not messages or counter(messages) <= budget:
        return 0
    low, high = 0, len(messages)
    while low < high:
        mid = (low + high) // 2
        if counter(messages[mid:]) <= budget:
            high = mid
        else:
            low = mid + 1
    cutoff = low
    if cutoff >= len(messages):
        if len(messages) == 1:
            return 0
        cutoff = len(messages) - 1
    return SummarizationMiddleware._find_safe_cutoff_point(messages, cutoff)


def _is_previous_summary(message: Any) -> bool:
    kwargs = getattr(message, "additional_kwargs", None) or {}
    return kwargs.get("lc_source") == SUMMARY_SOURCE_MARKER


def _previous_summary_text(previous: Any) -> str:
    """The previous summary's own text: LangChain's prefix and a closing
    placeholder sentence stripped, so neither is ever repeated."""
    if previous is None:
        return ""
    text = getattr(previous, "text", None)
    if not isinstance(text, str):
        text = str(getattr(previous, "content", "") or "")
    if text.startswith(SUMMARY_MESSAGE_PREFIX):
        text = text[len(SUMMARY_MESSAGE_PREFIX) :]
    text = text.strip()
    for tail in (SUMMARY_LATER_LOST, SUMMARY_EARLIER_LOST):
        if text.endswith(tail):
            text = text[: -len(tail)].rstrip()
    return text


def _summary_placeholder(previous: Any, max_tokens: Optional[int] = None) -> str:
    """What a compaction writes when no summary could be produced: the
    previous summary, carried over, plus one honest sentence. Bounded loss --
    the earlier memory survives -- and no compaction dead-lock.

    ``max_tokens`` (``summary_cap(W)`` when the window is known) caps the
    carried text in REAL tokens -- ``cap * 4 / w`` characters, ``w`` the
    counter's weight of that text, head kept: a summary written under a larger
    window would otherwise keep the state above a smaller window's trigger
    forever, and a CJK summary capped at ``cap * 4`` characters would carry
    several times ``cap`` real tokens while the keep reserves ``cap``.
    """
    text = _previous_summary_text(previous)
    if text and max_tokens is not None:
        from langchain_core.messages import HumanMessage

        weight = counter_weight(HumanMessage(text))
        text = text[: int(max_tokens * _APPROX_CHARS_PER_TOKEN / weight)].rstrip()
    if text:
        return f"{text}\n\n{SUMMARY_LATER_LOST}"
    return SUMMARY_EARLIER_LOST


def _truncate_for_summary(messages: list, budget: int, counter: Any, weight_of: Any) -> list:
    """Copies of ``messages`` whose content alone exceeds ``budget`` real
    tokens, cut to fit it (the head is kept): ``(budget / w - 16) * 4``
    characters, ``w`` the message's own weight (``weight_of``; the role and
    per-message overhead are weighed too). Halving the input cannot fix a
    single oversized answer; cutting it can."""
    out = []
    for message in messages:
        if counter([message]) <= budget:
            out.append(message)
            continue
        unscaled = budget / max(weight_of(message), 1e-6)
        keep_chars = max(0, int((unscaled - _TRUNCATION_OVERHEAD_TOKENS) * _APPROX_CHARS_PER_TOKEN))
        text = message.text if isinstance(getattr(message, "text", None), str) else ""
        out.append(message.model_copy(update={"content": text[:keep_chars]}))
    return out


# HTTP statuses of the summary call that are worth retrying on the NEXT turn.
_TRANSIENT_HTTP_STATUSES = frozenset({408, 409, 429})


def _is_prefill_timeout(exc: BaseException) -> bool:
    """The summary call timed out READING its input: the parent watchdog's
    first-chunk budget, or mlx_vlm.server's raw token-queue timeout."""
    if isinstance(exc, GenerationTimeoutException):
        return exc.phase == PHASE_FIRST_CHUNK
    return is_child_prefill_timeout(exc)


def _is_transient_summary_error(exc: BaseException) -> bool:
    """A failure of the summary call that the next turn can expect not to
    meet again: connection refused/reset, a client timeout, a timeout while
    DECODING, 408/409/429/5xx, a dead child. The turn then fails with the
    history intact, and the next turn retries the compaction.

    Everything else would fail every later turn identically while the main
    model is fine, so it degrades instead (``_acreate_summary``): a context
    overflow, any other 4xx, a timeout while reading the input (deterministic
    for that input size -- a smaller input is the cure), a bug.
    """
    import httpx
    import openai

    if _is_prefill_timeout(exc):
        return False
    if isinstance(exc, (EngineException, GenerationTimeoutException)):
        return True
    if parse_context_overflow(exc) is not None:
        return False
    if isinstance(exc, openai.APIConnectionError):  # includes APITimeoutError
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in _TRANSIENT_HTTP_STATUSES or exc.status_code >= 500
    # A connection that breaks mid-stream (the child died while answering)
    # surfaces from the SSE iterator as a raw httpx transport error.
    return isinstance(exc, httpx.TransportError)


def _is_rejection(exc: BaseException) -> bool:
    """A deterministic failure the app expects (an overflow, another 4xx, a
    prefill timeout), as opposed to an unexpected one."""
    import openai

    return (
        parse_context_overflow(exc) is not None
        or isinstance(exc, openai.APIStatusError)
        or _is_prefill_timeout(exc)
    )


@lru_cache(maxsize=1)
def _summarization_middleware_class():
    """The per-turn compaction middleware class, built lazily (LangChain is
    not imported at boot, #160) and cached."""
    from langchain.agents.middleware import SummarizationMiddleware
    from langchain_core.messages.utils import get_buffer_string, trim_messages

    class _Logged_Summarization_Middleware(SummarizationMiddleware):
        """The stock middleware, with what Erudi changes about compaction:

        * every count is in REAL tokens, as sent (``real_token_count``): a
          weight per message, frozen once per call from the full state
          (``_freeze``: measured ratio before the anchor, exact output tokens,
          script weights), with past KB/web results counted as their markers
          through the frozen set of their message ids;
        * the request overhead O (system prompt, tool schemas, KB block --
          not in the state) enters the trigger (``_should_summarize``) and the
          keep and futility guard (``compaction_cutoff``), never the counter
          itself (an O larger than the summarizer's trim budget would empty
          the trim);
        * the cutoff is ``compaction_cutoff`` (token AND message bound);
        * a performed compaction resets the engine's prefix cache, shielded
          (``run_reset_shielded``): the old prefix is garbage once the
          history is rewritten;
        * the summarizer input always carries the previous summary, reads
          past tool results as their markers, and a failed summary call never
          erases memory (``_acreate_summary``);
        * the ONE aggregate log line the QA spec promises ("backend.log
          records the summarization"). ASCII, INFO: the app did its job.
        """

        def __init__(
            self,
            *,
            working_window: Optional[int],
            engine: Any,
            allocated_window: Optional[int] = None,
            overhead: Optional[RequestOverhead] = None,
            **kwargs: Any,
        ):
            # The frozen per-call context, initialised before ``super()``
            # binds the counter; ``_freeze`` replaces it at every call.
            self.overhead = overhead
            self._weights: dict = {}
            self._past_ids: frozenset = frozenset()
            self._overhead = 0
            super().__init__(token_counter=self._real_count, **kwargs)
            self.working_window = working_window
            self.allocated_window = allocated_window
            self.engine = engine

        def _real_count(self, messages) -> int:
            return real_token_count(messages, self._weights, self._past_ids)

        def _weight_of(self, message) -> float:
            weight = self._weights.get(getattr(message, "id", None))
            return weight if weight is not None else counter_weight(message)

        def _freeze(self, messages) -> None:
            """The per-call context, from the FULL raw state as sent: one
            weight per message id, the past tool results' ids, and O."""
            from src.agents.middleware import past_tool_result_ids, strip_stale_tool_results

            self._ensure_message_ids(messages)
            self._past_ids = past_tool_result_ids(messages)
            self._weights, ratio = frozen_weights(strip_stale_tool_results(messages))
            self._overhead = overhead_tokens(
                self.overhead,
                ratio,
                dense=COUNTER_DENSE_TOKENS,
                weight_floor=COUNTER_WEIGHT_FLOOR,
            )

        def _determine_cutoff_index(self, messages):
            return compaction_cutoff(
                messages,
                self.working_window,
                self._real_count,
                overhead=self._overhead,
                allocated_window=self.allocated_window,
            )

        def _should_summarize(self, messages, total_tokens):
            # The token clause judges the whole request: the state plus O.
            return super()._should_summarize(messages, total_tokens + self._overhead)

        def _should_summarize_based_on_reported_tokens(self, messages, threshold):
            # Never: the usage a preserved AI message reports is the stale
            # total of the call that produced it, counting messages that are
            # no longer there. The trigger counts what IS there.
            return False

        def before_model(self, state, runtime):
            # The runner is async-only: no half-implemented sync path.
            raise NotImplementedError("compaction runs through abefore_model only")

        async def abefore_model(self, state, runtime):
            self._freeze(state["messages"])
            # Non-None exactly when the history was rewritten.
            result = await super().abefore_model(state, runtime)
            if result is not None:
                if _engine_overrides(self.engine, "on_history_rewritten"):
                    await run_reset_shielded(self.engine.on_history_rewritten)
            return result

        async def _acreate_summary(self, messages_to_summarize):
            """Summarize, never losing what the previous summary knew.

            The previous summary is set aside from the pool, the pool is
            trimmed to the budget, and the previous summary is prepended
            back. A transient failure of the summary call is RAISED: the
            turn fails with the history intact and the next turn retries. A
            deterministic one -- or a pool where no user turn fits the budget
            because one answer alone exceeds it -- takes the size path: one
            retry with oversized messages truncated (the budget halved after
            a rejection), then a placeholder carrying the previous summary.
            One WARNING per failed attempt; an unexpected error is one ERROR
            with its traceback.
            """
            from src.agents.middleware import strip_stale_tool_results

            if not messages_to_summarize:
                return "No previous conversation history."
            previous = next((m for m in messages_to_summarize if _is_previous_summary(m)), None)
            # The pool holds only past turns: their KB/web results are read as
            # the markers the model last saw, so what the summarizer reads is
            # what its trim counts.
            pool = strip_stale_tool_results(
                [m for m in messages_to_summarize if m is not previous], all_past=True
            )
            budget = self.trim_tokens_to_summarize or SUMMARY_TRIM_DEFAULT_TOKENS
            retry_budget = budget
            trimmed = self._trim_for_summary(pool, budget, start_on="human")
            if pool and not trimmed:
                logger.warning(
                    f"Compaction summary skipped its first attempt: no user turn fits "
                    f"the {budget}-token summarizer budget; retrying with oversized "
                    f"messages truncated"
                )
            else:
                summary = await self._summarize(previous, trimmed, attempt="first")
                if summary is not None:
                    return self._compacted(messages_to_summarize, summary)
                retry_budget = max(1, budget // 2)
            truncated = _truncate_for_summary(
                pool, retry_budget, self._partial_token_counter, self._weight_of
            )
            trimmed = self._trim_for_summary(truncated, retry_budget, start_on=None)
            summary = await self._summarize(previous, trimmed, attempt="retry")
            if summary is None:
                cap = (
                    summary_cap(self.working_window) if _known_window(self.working_window) else None
                )
                summary = _summary_placeholder(previous, cap)
            return self._compacted(messages_to_summarize, summary)

        def _trim_for_summary(self, messages, budget, *, start_on):
            if not messages:
                return []
            try:
                return trim_messages(
                    messages,
                    max_tokens=budget,
                    token_counter=self.token_counter,
                    start_on=start_on,
                    strategy="last",
                    allow_partial=True,
                    include_system=True,
                )
            except Exception:
                # LangChain's own fallback when trimming fails; the summary
                # still runs, on the most recent messages.
                logger.warning(
                    "Trimming the summarizer input failed; using the last messages",
                    exc_info=True,
                )
                return messages[-_TRIM_FALLBACK_MESSAGE_COUNT:]

        async def _summarize(self, previous, trimmed, *, attempt):
            """One summary call; the text, or None after a deterministic
            failure (logged). A transient failure propagates."""
            messages = ([previous] if previous is not None else []) + list(trimmed)
            prompt = self.summary_prompt.format(messages=get_buffer_string(messages)).rstrip()
            try:
                response = await self.model.ainvoke(
                    prompt, config={"metadata": {"lc_source": SUMMARY_SOURCE_MARKER}}
                )
            except Exception as exc:
                if _is_transient_summary_error(exc):
                    raise
                if _is_rejection(exc):
                    logger.warning(
                        f"Compaction summary call rejected ({attempt} attempt): "
                        f"{type(exc).__name__}, status={getattr(exc, 'status_code', None)}"
                    )
                else:
                    logger.error(
                        f"Compaction summary call failed ({attempt} attempt): "
                        f"{type(exc).__name__}",
                        exc_info=True,
                    )
                return None
            summary = response.text.strip()
            if not summary:
                # An empty summary would replace the whole history with
                # nothing: a failure, never a success.
                logger.warning(f"Compaction summary call returned no text ({attempt} attempt)")
                return None
            return summary

        @staticmethod
        def _compacted(messages_to_summarize, summary):
            logger.info(
                f"Conversation compacted: {len(messages_to_summarize)} message(s) "
                f"summarized into {len(summary or '')} chars"
            )
            return summary

    return _Logged_Summarization_Middleware


# Hard cap on LangGraph super-steps per turn (#277). Without it the graph
# defaults leave a runaway agent unbounded: a small model that keeps issuing the
# identical tool call gets the identical result back and never converges (Qwen3
# 0.6B repeated one KB search 53 times, holding the turn for ~16 minutes before
# emitting garbage). Each tool round is roughly two super-steps (model -> tools),
# so this allows ~7 legitimate tool rounds before the graph raises
# GraphRecursionError, which we then handle gracefully below instead of letting
# the user stare at a spinner.
AGENT_RECURSION_LIMIT = 15

# Frontend detects this prefix (substring match) to render an error turn in red.
# Keep it; the message intentionally carries NO traceback (avoids info leak).
ERROR_SENTINEL = "[ERROR_MESSAGE_SYSTEM]"
ERROR_MESSAGE = (
    f"{ERROR_SENTINEL} I apologize, but I encountered an error while generating "
    "a response. Please try asking your question again."
)

# Curated turn for a runaway agent that hit AGENT_RECURSION_LIMIT with nothing
# usable to fall back to (#277). Carries the ERROR sentinel so the frontend
# renders it in red and no traceback leaks -- there is genuinely no answer, so an
# honest error turn beats a raw tool dump or a silent stop.
LOOP_LIMIT_MESSAGE = (
    f"{ERROR_SENTINEL} I couldn't reach a final answer for this request "
    "(the model kept retrying without making progress). Please try rephrasing "
    "your question."
)

# Curated turns for a stream that ended with NO answer text and nothing to
# fall back to (#554). Deliberately NOT the ERROR sentinel: a sentinel turn is
# rendered red by the frontend and persisted WITHOUT its trace (the
# conversation service drops the trace on sentinel answers), which would throw
# away exactly the reasoning that explains what happened. One impersonal ASCII
# line each, actionable by the user AND by the model -- the persisted line is
# replayed to the model on the next turn. Two wordings because the two
# finish_reasons are different failures: ``length`` means generation was cut
# mid-reasoning (the length line never names the removed Max Tokens control);
# ``stop`` means the model closed its turn without writing an answer.
EMPTY_ANSWER_LENGTH_MESSAGE = (
    "Generation stopped during reasoning, before an answer was written. "
    "Send a follow-up asking for the final answer directly, or lower the reasoning effort."
)
EMPTY_ANSWER_STOP_MESSAGE = (
    "The model finished its turn without writing an answer. "
    "Send a follow-up asking it to continue."
)

# Curated turns for a stream that ran out of wall-clock budget (#573). The two
# silences are not the same failure and must not read the same: one says the
# model never started on a prompt this size (retrying makes it WORSE -- the
# history only grows), the other says it started and then stopped.
PREFILL_TIMEOUT_MESSAGE_TEMPLATE = (
    "{sentinel} This model did not start answering within {minutes} minutes on this "
    "machine. Before writing a word it has to read everything this turn sends it -- "
    "the whole conversation so far -- and it did not get through that in time. "
    "Sending the same thing again will take longer, not less: start a new "
    "conversation, send less at once, or pick a smaller model."
)
DECODE_TIMEOUT_MESSAGE = (
    f"{ERROR_SENTINEL} This model started answering, then went silent for "
    f"{INTER_CHUNK_BUDGET_S:.0f} seconds, so the turn was stopped. Anything it had "
    "already written is kept above. Please try asking your question again."
)


# Curated turn for a context-window overflow (PR-G). Both local engines
# reject an over-budget prompt with a precise 400 that names the real numbers
# (src.agents.overflow.parse_context_overflow discriminates it from a generic
# 400). Surfacing those numbers beats the catch-all apology below -- the user
# learns exactly why and what to do -- and the app never silently truncates
# or shifts the conversation to make it fit.
CONTEXT_OVERFLOW_MESSAGE_TEMPLATE = (
    f"{ERROR_SENTINEL} This conversation no longer fits the model's context window "
    "(the request needs about {prompt_tokens} tokens; the window holds {context_tokens}). "
    "Start a new conversation or send less at once."
)
CONTEXT_OVERFLOW_MESSAGE = (
    f"{ERROR_SENTINEL} This conversation no longer fits the model's context window. "
    "Start a new conversation or send less at once."
)


def _context_overflow_message(overflow: ContextOverflow) -> str:
    """The curated turn a parsed ``ContextOverflow`` becomes: the numbers
    when the wire error carried them, a numberless variant when it matched
    but didn't parse (still an honest overflow message, just without figures)."""
    if overflow.prompt_tokens is None or overflow.context_tokens is None:
        return CONTEXT_OVERFLOW_MESSAGE
    return CONTEXT_OVERFLOW_MESSAGE_TEMPLATE.format(
        prompt_tokens=overflow.prompt_tokens,
        context_tokens=overflow.context_tokens,
    )


def _stream_timeout_message(exc: GenerationTimeoutException) -> str:
    """The curated turn a ``GenerationTimeoutException`` becomes."""
    if exc.phase != PHASE_FIRST_CHUNK:
        return DECODE_TIMEOUT_MESSAGE
    # The estimated prompt size stays in the WARNING and out of the turn: it is
    # a deliberate UPPER BOUND (one token per UTF-8 byte, #573), so quoting it
    # to the user as a token count would overstate the real prompt several-fold.
    return PREFILL_TIMEOUT_MESSAGE_TEMPLATE.format(
        sentinel=ERROR_SENTINEL,
        minutes=max(1, round(exc.budget_s / 60)),
    )


# Prepended (and persisted with the assistant message) by the conversation and
# arena services when the CURRENT turn carries images but the model is not
# positively vision-capable (#212): ``_StripImagesForTextModel`` drops the image
# parts, and the user is told explicitly instead of silently. Markdown italics —
# the frontend renders streamed answers as markdown.
IMAGES_IGNORED_NOTICE = "*This model doesn't support images — your image was ignored.*\n\n"


def _construction_error_message(exc: Exception) -> str:
    """Curated, traceback-free error turn for a failed agent construction.

    Agent construction is where the model is loaded (``build_chat_model`` ->
    ``engine.get_model_and_tokenizer``). When that load fails with an
    ``EngineException`` -- a specific, already-curated diagnostic like a missing
    model folder, no ``.gguf`` found, a corrupt GGUF, or a child server that
    died on spawn (#88) -- surface its message so the user learns what is
    actually wrong and can act (re-download / pick another model). Any other
    failure keeps the generic message. Neither path leaks a traceback:
    ``EngineException`` messages are hand-written, not stringified stack traces.
    """
    if isinstance(exc, EngineException):
        return f"{ERROR_SENTINEL} {exc}"
    return ERROR_MESSAGE


def _construction_error_event(exc: Exception) -> dict:
    """The answer event a failed agent construction becomes.

    Beyond the curated text, an ``EngineException`` that identified its cause
    carries it: ``code`` (one of
    ``src.engines.cuda_compatibility.CUDA_FAILURE_CODES``) and ``raw`` (the
    child's captured output). The conversation service copies both onto the
    wire ``error`` event, which is where the renderer picks them up to offer
    the matching remedy -- switching to the CPU engine, updating the driver --
    instead of a generic apology.

    Riding on the existing answer event rather than a new event type keeps the
    persistence path untouched: the text still accumulates into the assistant
    message exactly as before, and a client that ignores the extra keys sees
    the stream it has always seen.
    """
    event: dict = {"t": "answer", "text": _construction_error_message(exc)}
    code = getattr(exc, "engine_code", None)
    if code:
        event["code"] = code
        raw = getattr(exc, "engine_trace", None)
        if raw:
            event["raw"] = raw
    return event


# ===================== Tool-call accumulation (#90) =====================
# Tool-call args stream as JSON fragments across ``AIMessageChunk.tool_call_chunks``
# (keyed by call index). Fragments are NEVER emitted raw: they accumulate here and
# a single complete ``tool_call`` event is emitted per call once assembled (on the
# tools node's ToolMessage, or at final flush).


def _chunk_get(chunk: Any, key: str) -> Any:
    """Read a field from a ``tool_call_chunk`` (a TypedDict at runtime, but be
    defensive about object-shaped chunks from other langchain versions)."""
    if isinstance(chunk, dict):
        return chunk.get(key)
    return getattr(chunk, key, None)


def _accumulate_tool_call(pending: dict, chunk: Any) -> None:
    """Fold one streamed ``tool_call_chunk`` into the per-index buffer.

    ``name`` and ``id`` arrive once (kept on first sight); ``args`` arrive as
    string fragments and are concatenated in order.
    """
    index = _chunk_get(chunk, "index")
    if index is None:
        index = 0
    slot = pending.setdefault(index, {"name": None, "args": "", "id": None})
    name = _chunk_get(chunk, "name")
    if name:
        slot["name"] = name
    call_id = _chunk_get(chunk, "id")
    if call_id:
        slot["id"] = call_id
    frag = _chunk_get(chunk, "args")
    if frag:
        slot["args"] += frag


def _parse_tool_args(raw: str) -> dict:
    """Accumulated args JSON -> dict when it parses to an object, else
    ``{"raw": <string>}``. Empty args -> ``{}``. Never returns raw fragments."""
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return {"raw": raw}
    return parsed if isinstance(parsed, dict) else {"raw": raw}


def _drain_tool_calls(pending: dict) -> list:
    """Emit one complete ``tool_call`` event per accumulated call (index order),
    then clear the buffer so the next agent step accumulates fresh."""
    events = []
    for index in sorted(pending):
        slot = pending[index]
        events.append(
            {
                "t": "tool_call",
                "name": slot["name"] or "",
                "args": _parse_tool_args(slot["args"]),
            }
        )
    pending.clear()
    return events


@dataclass
class GenParams:
    """Per-request generation parameters (resolved from payload-or-conversation)."""

    temperature: float
    top_p: float
    max_tokens: int


def _child_crash_suffix(engine) -> str:
    """`; llama-server child is dead (...)` when the engine's child died, else ""."""
    report = getattr(engine, "child_crash_report", None)
    if report is None:
        return ""
    try:
        text = report()
    except Exception:  # a diagnostic must never mask the failure it describes
        return ""
    return f"; {text}" if text else ""


class AgentRunner:
    """Streams an agent turn as structured events. Shared by conversation and arena.

    ``_astream_events`` is the single capture loop (answer / thinking / tool_call /
    tool_result); ``astream_text`` projects it to either event dicts or plain
    answer text via ``emit_events`` (see the module docstring). Pass a
    ``checkpointer`` (the app-wide ``AsyncPostgresSaver``) for stateful
    conversations; arena constructs it with ``checkpointer=None``.
    """

    def __init__(self, checkpointer: Optional[BaseCheckpointSaver] = None):
        self.checkpointer = checkpointer

    async def astream_text(
        self,
        *,
        llm,
        user_message: str | list,
        system_prompt: str,
        params: GenParams,
        thread_id: Optional[str] = None,
        summarize: bool = False,
        kb_context_block: Optional[str] = None,
        kb_language_line: str = "",
        tools: Optional[list] = None,
        context: Optional[Any] = None,
        supports_vision: Optional[bool] = None,
        effort_plan: Optional[EffortPlan] = None,
        emit_events: bool = False,
    ) -> AsyncIterator:
        """Project the turn's event stream (:meth:`_astream_events`).

        ``emit_events=True`` yields the event dicts unchanged (conversations frame
        them as NDJSON). ``emit_events=False`` (arena / default) yields ONLY answer
        text as ``str`` -- thinking and tool events are dropped and inline
        ``<think>`` is stripped, preserving the old plain-text wire byte-for-byte.
        Error paths ride an ``answer`` event carrying the ERROR sentinel string, so
        in str mode the sentinel is yielded exactly as before (the conversation
        service maps it to an ``error`` event on the wire; DB persistence
        unchanged).
        """
        # ``aclosing``: closing THIS generator closes the capture loop (and,
        # through it, LangGraph) right away, inside the generation guard.
        async with contextlib.aclosing(
            self._astream_events(
                llm=llm,
                user_message=user_message,
                system_prompt=system_prompt,
                params=params,
                thread_id=thread_id,
                summarize=summarize,
                kb_context_block=kb_context_block,
                kb_language_line=kb_language_line,
                tools=tools,
                context=context,
                supports_vision=supports_vision,
                effort_plan=effort_plan,
            )
        ) as events:
            async for event in events:
                if emit_events:
                    yield event
                elif event["t"] == "answer":
                    yield event["text"]

    async def _astream_events(
        self,
        *,
        llm,
        user_message: str | list,
        system_prompt: str,
        params: GenParams,
        thread_id: Optional[str] = None,
        summarize: bool = False,
        kb_context_block: Optional[str] = None,
        kb_language_line: str = "",
        tools: Optional[list] = None,
        context: Optional[Any] = None,
        supports_vision: Optional[bool] = None,
        effort_plan: Optional[EffortPlan] = None,
    ) -> AsyncIterator[dict]:
        """The single capture loop: structured events for the whole turn (#90).

        Yields ``{"t":"answer","text":...}`` (the content channel, outside any
        inline ``<think>``), ``{"t":"thinking","text":...}`` (the servers'
        dedicated reasoning channel, plus inline ``<think>`` content caught by
        the fallback splitter, #554), one
        ``{"t":"tool_call","name":...,"args":{...}}`` per call, and
        ``{"t":"tool_result","name":...,"text":...}`` per ToolMessage.

        Error paths (#252 construction failure, streaming failure, #573 stream
        budget expiry) yield a curated ERROR sentinel as an ``answer`` event --
        each one saying what actually happened. Callers map it: the
        conversation service turns a sentinel-prefixed answer into an ``error``
        wire event while still accumulating the sentinel STRING for persistence
        (DB behavior unchanged per #225-D4); arena yields it as plain text.
        """
        # Deferred (#160): first turn pays the agent-stack import, boot doesn't.
        from langchain.agents import create_agent
        from langchain_core.messages import HumanMessage
        from langgraph.errors import GraphRecursionError

        from src.agents.middleware import (
            _FoldSystemIntoUserMiddleware,
            _KbContextMiddleware,
            _StripImagesForTextModel,
        )
        from src.engines.system_role_capability import model_supports_system_role

        engine = config.LLM_Engine
        stateful = thread_id is not None and self.checkpointer is not None
        # recursion_limit bounds a runaway agent (#277); it rides at the top level
        # of the run config, alongside (not inside) ``configurable``.
        run_config: dict = {"recursion_limit": AGENT_RECURSION_LIMIT}
        if stateful:
            run_config["configurable"] = {"thread_id": thread_id}

        async with engine.generation_guard():
            try:
                sampling = resolve_sampling_defaults(llm)
                # No implicit tools (#129): callers own the tool list (built by
                # ``plan_turn``); ``tools=None`` means a zero-tool agent.
                effective_tools = tools if tools is not None else []
                # What every request of this turn carries beyond the state:
                # the system prompt and the tool schemas, and the KB additions.
                # The compaction middleware and the warning projection cost it
                # in real tokens; the client stamps the KB part's size so the
                # ratio it measures excludes a block the next request drops.
                overhead = request_overhead(
                    system_prompt, kb_context_block, kb_language_line, effective_tools
                )
                model = await run_in_threadpool(
                    build_chat_model,
                    llm,
                    temperature=params.temperature,
                    top_p=params.top_p,
                    max_tokens=params.max_tokens,
                    # Per-model extra sampling keys (#388); the user-facing three
                    # above still come from the conversation row / arena panel.
                    sampling=sampling,
                    # The turn's reasoning effort (1.1.2): its wire value, when
                    # the artifact has a native lever for it.
                    effort_plan=effort_plan,
                    kb_additions=overhead.added_text,
                    # Each call's usage feeds the turn's memory measurement
                    # (MLX); the summary and title clients push nothing.
                    record_usage=True,
                )
                # Memory accounting for the compaction ceiling and the amber
                # warning. Derived AFTER build_chat_model so the child is up
                # and the handle carries the loaded artifact and its measured
                # base; file I/O and hardware probes -> threadpool.
                # ``from_engine`` never raises; an unaccountable model just
                # carries None facts.
                budget = (
                    await run_in_threadpool(MemoryBudget.from_engine, engine) if summarize else None
                )
                allocated_window, working_window = (
                    self._compaction_windows(budget) if summarize else (None, None)
                )
                # The compaction summary ALWAYS runs at effort "none" (1.1.2):
                # summarizing is machine work, and a reasoning model would spend
                # the call deliberating about it instead of writing it. Same
                # child, same everything else -- only the reasoning field and
                # the length differ, so the second client costs a cached
                # handle lookup. Its length is BOUNDED to ``summary_cap(W)``,
                # with no automatic output budget (which would hand it the
                # whole window) and no preflight retry (a smaller cap would
                # silently truncate the summary; a rejection takes the
                # summarizer's size path instead): the keep arithmetic reserves
                # exactly that much for it. No known window: today's budget.
                summary_model = (
                    await run_in_threadpool(
                        build_chat_model,
                        llm,
                        temperature=params.temperature,
                        top_p=params.top_p,
                        max_tokens=(
                            summary_cap(working_window)
                            if _known_window(working_window)
                            else params.max_tokens
                        ),
                        sampling=sampling,
                        effort_plan=NO_REASONING_PLAN,
                        auto_output_budget=not _known_window(working_window),
                        preflight_retry=False,
                    )
                    if summarize
                    else None
                )
                middleware = (
                    self._build_middleware(summary_model, budget, overhead=overhead)
                    if summarize
                    else []
                )
                if kb_context_block:
                    # After summarization: the merge must see the final
                    # message list that actually reaches the model.
                    middleware = [
                        *middleware,
                        _KbContextMiddleware(kb_context_block, kb_language_line),
                    ]
                if supports_vision is not True:
                    # Unless the model is POSITIVELY vision-capable, strip images
                    # (#212): unknown capability (None) is treated like False, so
                    # a maybe-text-only model never breaks on an attachment (the
                    # services prepend a user-facing notice). Outermost, so images
                    # are gone before the KB merge re-reads the last user message.
                    middleware = [_StripImagesForTextModel(), *middleware]
                if not await run_in_threadpool(
                    model_supports_system_role, getattr(llm, "link", None)
                ):
                    # Model's chat template rejects a system role (Gemma): fold the
                    # system prompt into the first user turn instead of 500ing every
                    # turn. Innermost (added last), so it folds the FINAL messages
                    # after the KB merge has shaped the last user message.
                    middleware = [*middleware, _FoldSystemIntoUserMiddleware()]
                agent = create_agent(
                    model,
                    tools=effective_tools,
                    system_prompt=system_prompt,
                    checkpointer=self.checkpointer if stateful else None,
                    middleware=middleware,
                    context_schema=type(context) if context is not None else None,
                )
                logger.info(
                    f"Agent built: llm={getattr(llm, 'id', '?')} "
                    f"({getattr(llm, 'name', '?')}), "
                    f"tools={[getattr(t, 'name', str(t)) for t in effective_tools]}, "
                    f"stateful={stateful}, summarize={summarize}, "
                    f"kb_context={'yes' if kb_context_block else 'no'}"
                )
            except Exception as exc:
                logger.exception(
                    f"Agent construction failed: llm={getattr(llm, 'id', '?')} "
                    f"({getattr(llm, 'name', '?')}), thread_id={thread_id}"
                )
                # #252: construction failed (model load / spawn). Emit the curated
                # sentinel as an answer event; callers map it to an error turn.
                yield _construction_error_event(exc)
                return

            # A turn that does not end normally -- the client went away
            # (GeneratorExit at a yield, or a cancellation), or an exception
            # escaped -- flags the child it ran against, through the hook the
            # factory bound to the handle captured at BUILD time (never
            # ``cls._model`` read now): the MLX engine then sends a barrier
            # before its next prefix-cache reset. Deterministic, at guard
            # release, in addition to the client-level flag.
            abandon_hook = getattr(model, "abandon_hook", None)
            memory_token = None
            try:
                # The prefix cache belongs to ONE conversation (MLX; a no-op
                # on llama.cpp): claim it for this thread -- or for the arena --
                # now that the child is resolved and before the agent sends
                # anything. A reset it needs runs shielded: the guard is never
                # left while it still talks to the child.
                if _engine_overrides(engine, "claim_prefix"):
                    owner = f"conv:{thread_id}" if stateful else "arena"
                    await run_reset_shielded(functools.partial(engine.claim_prefix, owner))
                # What this turn really uses is measured on the child (MLX
                # only): the window opens after the claim (its reset is not
                # this turn's memory) and closes at the end of the turn,
                # inside the guard.
                if _engine_overrides(engine, "begin_memory_window"):
                    memory_token = await run_in_threadpool(engine.begin_memory_window)

                # Aggregate-only stream accounting (never log per token): start,
                # first-token latency, then one completion line with totals.
                # ``char_count`` counts ANSWER text only -- reasoning is counted
                # apart (``reasoning_chars``) and must not inflate the answer
                # accounting nor the empty-final signal below. ``finish_reason``
                # keeps the LAST finish_reason a model hop reported (#554): it
                # picks the curated empty-answer wording and lands in the
                # completion log for field-report attribution.
                stream_start_s = time.perf_counter()
                first_token_s: Optional[float] = None
                chunk_count = 0
                char_count = 0
                reasoning_chars = 0
                finish_reason: Optional[str] = None
                # Empty-final fallback bookkeeping (#90): some agentic models call a
                # tool successfully, then emit an EMPTY final ANSWER (observed with
                # Gemma: calculator("1240 + 1378 + 1456") -> ToolMessage "4074" ->
                # empty AIMessage, finish_reason=stop). ``emitted_model_text`` tracks
                # non-blank ANSWER text ONLY (post-splitter), so a model that only
                # thinks then calls a tool and returns nothing still triggers the
                # fallback -- thinking must never mask an empty answer.
                emitted_model_text = False
                last_tool_result: Optional[str] = None
                splitter = ThinkSplitter()
                pending_tool_calls: dict = {}
                # Pre-tool narration reclassification (#297), tool-carrying turns
                # only: each model hop's post-splitter ANSWER text is buffered here.
                # The hop's first tool_call_chunk re-emits the buffer as thinking
                # (the text was narration, not the answer) and flips
                # ``hop_has_tool_call`` so the rest of the hop streams as thinking;
                # a ToolMessage resets the flag for the next hop; stream end flushes
                # whatever is buffered as the real answer. ``emitted_model_text``
                # and ``char_count`` are only touched on that final ANSWER flush --
                # reclassified narration must not defeat the #90 fallback.
                agentic = bool(effective_tools)
                hop_text_buffer: list[str] = []
                hop_has_tool_call = False
                logger.info(
                    f"Agent stream started: llm={getattr(llm, 'id', '?')}, "
                    f"thread_id={thread_id}"
                )
                # LangGraph is iterated in a child task (``isolated_stream``)
                # under ``aclosing``: whether this generator is closed by its
                # consumer or cancelled by a client disconnect (anyio
                # re-delivers that cancellation at every await), LangGraph
                # receives ONE cancellation, its exit cancels and awaits the
                # in-flight node, and all of it completes before the guard is
                # released -- never later from a finalizer or a stray task.
                async with contextlib.aclosing(
                    isolated_stream(
                        lambda: agent.astream(
                            {"messages": [HumanMessage(user_message)]},
                            config=run_config,
                            context=context,
                            stream_mode="messages",
                        )
                    )
                ) as agent_stream:
                    try:
                        async for token, meta in agent_stream:
                            if getattr(token, "type", None) == "tool":
                                # ToolMessage from the tools node: the model node has
                                # finished streaming this step's tool_call_chunks, so emit
                                # the complete tool_call event(s) first, then the result.
                                # Keep the latest non-blank result for the #90 fallback.
                                tool_text = getattr(token, "text", "") or ""
                                if tool_text.strip():
                                    last_tool_result = tool_text
                                # Defensive (#297): narration not yet reclassified (tool
                                # calls that arrived without streamed chunks) goes out
                                # as thinking BEFORE the tool_call events.
                                for buffered in hop_text_buffer:
                                    yield {"t": "thinking", "text": buffered}
                                hop_text_buffer.clear()
                                for tc_event in _drain_tool_calls(pending_tool_calls):
                                    yield tc_event
                                yield {
                                    "t": "tool_result",
                                    "name": getattr(token, "name", "") or "",
                                    "text": tool_text,
                                }
                                # The hop ended with tools: the next model hop buffers
                                # fresh (#297).
                                hop_has_tool_call = False
                                continue
                            if meta.get("langgraph_node") == "model":
                                tc_chunks = getattr(token, "tool_call_chunks", None) or []
                                for tc_chunk in tc_chunks:
                                    _accumulate_tool_call(pending_tool_calls, tc_chunk)
                                if agentic and tc_chunks and not hop_has_tool_call:
                                    # First tool_call_chunk of this hop (#297): the text
                                    # streamed so far was pre-tool narration -- re-emit
                                    # it as thinking NOW (before the tool_call event),
                                    # preserving stream liveness.
                                    hop_has_tool_call = True
                                    for buffered in hop_text_buffer:
                                        yield {"t": "thinking", "text": buffered}
                                    hop_text_buffer.clear()
                                # Dedicated reasoning channel (#554): both servers
                                # extract chain-of-thought server-side and the chat
                                # client re-attaches it to the chunk. A reasoning-only
                                # chunk has EMPTY ``.text`` but is stream activity all
                                # the same: it must start the first-token clock and
                                # count in the chunk total, or an all-reasoning turn
                                # would look like a silent hang in the logs.
                                reasoning_delta = (
                                    getattr(token, "additional_kwargs", None) or {}
                                ).get("reasoning_content") or ""
                                text = getattr(token, "text", "")
                                hop_finish = (getattr(token, "response_metadata", None) or {}).get(
                                    "finish_reason"
                                )
                                if hop_finish:
                                    finish_reason = hop_finish
                                if text or reasoning_delta:
                                    if first_token_s is None:
                                        first_token_s = time.perf_counter()
                                        logger.info(
                                            f"Agent first token: llm={getattr(llm, 'id', '?')}, "
                                            f"latency_ms={(first_token_s - stream_start_s) * 1000:.0f}"
                                        )
                                    chunk_count += 1
                                if reasoning_delta:
                                    # Never buffered by the #297 narration logic:
                                    # reasoning is thinking by definition, on every hop.
                                    reasoning_chars += len(reasoning_delta)
                                    yield {"t": "thinking", "text": reasoning_delta}
                                if text:
                                    for event in splitter.feed(text):
                                        if event["t"] != "answer":
                                            # Real <think> content (fallback splitter
                                            # families): flows immediately.
                                            reasoning_chars += len(event["text"])
                                            yield event
                                        elif agentic and hop_has_tool_call:
                                            # Post-tool-call text in a narrating hop
                                            # (#297): also narration -> thinking.
                                            yield {"t": "thinking", "text": event["text"]}
                                        elif agentic:
                                            # Tool-carrying turn, no tool call yet this
                                            # hop: hold the text (#297) -- it is either
                                            # narration (a tool call follows) or the
                                            # final answer (flushed at stream end).
                                            hop_text_buffer.append(event["text"])
                                        else:
                                            if event["text"].strip():
                                                emitted_model_text = True
                                            char_count += len(event["text"])
                                            yield event
                        # The final hop ended without a tool call: its buffered text IS
                        # the final answer (#297) -- flush it as ANSWER events (this is
                        # the only place buffered text counts as emitted answer).
                        for text in hop_text_buffer:
                            if text.strip():
                                emitted_model_text = True
                            char_count += len(text)
                            yield {"t": "answer", "text": text}
                        hop_text_buffer.clear()
                        # Flush any buffered splitter text (a trailing partial tag, or an
                        # unclosed <think> -> thinking) BEFORE the empty-final decision.
                        for event in splitter.flush():
                            if event["t"] == "answer":
                                if event["text"].strip():
                                    emitted_model_text = True
                                char_count += len(event["text"])
                            else:
                                reasoning_chars += len(event["text"])
                            yield event
                        # Empty/blank final answer, but a tool produced a result this
                        # turn: deliver that last tool result AS THE ANSWER (#90) so a
                        # correct value is streamed and persisted instead of crashing the
                        # empty-content guard. No tool ran -> the curated empty-answer
                        # turn below (#554).
                        if not emitted_model_text and last_tool_result is not None:
                            logger.info(
                                f"Empty final answer with a tool result; falling back to "
                                f"the last tool result: llm={getattr(llm, 'id', '?')}, "
                                f"tool_result_chars={len(last_tool_result)}"
                            )
                            char_count += len(last_tool_result)
                            yield {"t": "answer", "text": last_tool_result}
                        # A tool call that never produced a ToolMessage this turn (rare):
                        # emit it now so the trace still records the attempt.
                        for tc_event in _drain_tool_calls(pending_tool_calls):
                            yield tc_event
                        # No answer text and nothing to fall back to (#554): deliver
                        # the curated empty-answer turn as a NORMAL answer instead of
                        # yielding nothing (which crashed the downstream empty-content
                        # guard into a generic error that also dropped the trace). The
                        # wording follows finish_reason: ``length`` = cut mid-reasoning,
                        # anything else = the model closed its turn without an answer.
                        if not emitted_model_text and last_tool_result is None:
                            curated = (
                                EMPTY_ANSWER_LENGTH_MESSAGE
                                if finish_reason == "length"
                                else EMPTY_ANSWER_STOP_MESSAGE
                            )
                            logger.info(
                                f"Turn ended with no answer text: llm={getattr(llm, 'id', '?')}, "
                                f"finish_reason={finish_reason or 'unknown'}, "
                                f"reasoning_chars={reasoning_chars}; yielding the curated turn"
                            )
                            char_count += len(curated)
                            if stateful:
                                # State BEFORE the yield: a client that disconnects
                                # right after receiving the curated event closes this
                                # generator at the yield, and code after it never runs
                                # -- while the conversation service's finally still
                                # persists the curated line to SQL. Writing first keeps
                                # the checkpointer consistent with what a reload shows;
                                # the reverse window (state written, client already
                                # gone) is covered by the service's interrupted-turn
                                # handling. Without the write at all, the checkpointer
                                # keeps the EMPTY AIMessage the model node committed
                                # and the next turn replays an empty assistant turn.
                                await self._write_curated_empty_turn(agent, run_config, curated)
                            yield {"t": "answer", "text": curated}
                        # Amber warning check (1.1.2), at end of turn on the
                        # POST-turn thread state: one ``memory_warning`` event goes
                        # out ONLY when even a compaction down to the keep-tail would
                        # leave the model and the conversation above 85 % of the
                        # memory budget (see ``_memory_warning_event`` for the
                        # projection). The conversation service forwards it to the
                        # wire and never persists it (it is a statement about NOW,
                        # on THIS machine). Its start and end per conversation are
                        # one INFO line each (``_track_memory_warning``).
                        if stateful and budget is not None:
                            warning = await self._memory_warning_event(
                                agent,
                                run_config,
                                budget,
                                working_window,
                                # The KB block of the turn just answered is not
                                # in the next request: the fixed part only.
                                overhead=RequestOverhead(
                                    fixed_est=overhead.fixed_est,
                                    fixed_text=overhead.fixed_text,
                                ),
                                allocated_window=allocated_window,
                            )
                            if budget.used_fraction(0) is not None:
                                _track_memory_warning(engine, thread_id, warning is not None)
                            if warning is not None:
                                yield warning
                        duration_ms = (time.perf_counter() - stream_start_s) * 1000
                        logger.info(
                            f"Agent stream completed: llm={getattr(llm, 'id', '?')}, "
                            f"duration_ms={duration_ms:.0f}, chunks={chunk_count} (~tokens), "
                            f"reasoning_chars={reasoning_chars}, answer_chars={char_count}, "
                            f"finish_reason={finish_reason or 'unknown'}"
                        )
                    except GraphRecursionError:
                        # #277: the agent hit AGENT_RECURSION_LIMIT (a small model looping
                        # on the same tool call). Degrade gracefully instead of surfacing a
                        # generic error after a long hang: flush any buffered answer text,
                        # then, if the model never produced usable text, fall back to the
                        # last tool result (like the #90 empty-final path) or a curated
                        # loop-limit turn.
                        logger.warning(
                            "Agent hit recursion limit (%s) -- likely a tool-call loop: "
                            "llm=%s, thread_id=%s",
                            AGENT_RECURSION_LIMIT,
                            getattr(llm, "id", "?"),
                            thread_id,
                        )
                        # Deliver what was gathered: an interrupted hop's buffered text
                        # (#297) had no tool call yet, so it flushes as answer.
                        for text in hop_text_buffer:
                            if text.strip():
                                emitted_model_text = True
                            char_count += len(text)
                            yield {"t": "answer", "text": text}
                        hop_text_buffer.clear()
                        for event in splitter.flush():
                            if event["t"] == "answer":
                                if event["text"].strip():
                                    emitted_model_text = True
                                char_count += len(event["text"])
                            yield event
                        if not emitted_model_text:
                            if last_tool_result is not None:
                                char_count += len(last_tool_result)
                                yield {"t": "answer", "text": last_tool_result}
                            else:
                                yield {"t": "answer", "text": LOOP_LIMIT_MESSAGE}
                        if stateful:
                            await self._repair_alternation(agent, run_config)
                    except GenerationTimeoutException as exc:
                        # #573: the stream stayed silent past its budget. Nothing
                        # crashed -- the watchdog ended the turn on a budget WE chose --
                        # so this is a degradation, logged at WARNING with the numbers
                        # behind the decision and no traceback (docs/logging.md), and
                        # the user gets a turn that names the actual cause.
                        logger.warning(
                            f"Agent stream timed out: llm={getattr(llm, 'id', '?')} "
                            f"({getattr(llm, 'name', '?')}), thread_id={thread_id}, "
                            f"phase={exc.phase}, budget_s={exc.budget_s:.0f}, "
                            f"est_prompt_tokens={exc.estimated_prompt_tokens}"
                        )
                        if stateful:
                            await self._repair_alternation(agent, run_config)
                        # Same parity as the generic failure below: text buffered before
                        # the timeout is delivered ahead of the curated turn.
                        for text in hop_text_buffer:
                            yield {"t": "answer", "text": text}
                        hop_text_buffer.clear()
                        yield {"t": "answer", "text": _stream_timeout_message(exc)}
                    except Exception as exc:
                        overflow = parse_context_overflow(exc)
                        if overflow is not None:
                            # PR-G: the engine's own 400 already named the real
                            # numbers -- nothing crashed, the app degraded on its
                            # own, so WARNING (not exception) with the numbers and no
                            # traceback (docs/logging.md).
                            logger.warning(
                                f"Agent stream hit a context-window overflow: llm={getattr(llm, 'id', '?')} "
                                f"({getattr(llm, 'name', '?')}), thread_id={thread_id}, "
                                f"prompt_tokens={overflow.prompt_tokens}, "
                                f"context_tokens={overflow.context_tokens}"
                            )
                            if stateful:
                                await self._repair_alternation(agent, run_config)
                            for text in hop_text_buffer:
                                yield {"t": "answer", "text": text}
                            hop_text_buffer.clear()
                            yield {"t": "answer", "text": _context_overflow_message(overflow)}
                        elif is_child_prefill_timeout(exc):
                            # Defense in depth (#573 alignment): the MLX child's
                            # token-queue timeout is sized ABOVE the parent's first-chunk
                            # ceiling (src.engines.mlx_engine), so this branch should
                            # never fire -- but if mlx_vlm.server's raw "Increase
                            # MLX_VLM_TOKEN_QUEUE_TIMEOUT ..." error ever surfaces, the
                            # child out-waited the parent (worth a WARNING) and the user
                            # must get the SAME curated prefill turn, never the internal
                            # env var name. Reported at WARNING with no traceback: the
                            # app degraded on its own (docs/logging.md).
                            window_probe = getattr(engine, "effective_context_tokens", None)
                            window = window_probe() if callable(window_probe) else None
                            ceiling = first_chunk_ceiling_s(window)
                            logger.warning(
                                f"Child token-queue timeout surfaced past the aligned budget: "
                                f"llm={getattr(llm, 'id', '?')} ({getattr(llm, 'name', '?')}), "
                                f"thread_id={thread_id}, ceiling_s={ceiling:.0f}"
                            )
                            if stateful:
                                await self._repair_alternation(agent, run_config)
                            for text in hop_text_buffer:
                                yield {"t": "answer", "text": text}
                            hop_text_buffer.clear()
                            # Reuse the parent watchdog's curated first-chunk turn.
                            yield {
                                "t": "answer",
                                "text": PREFILL_TIMEOUT_MESSAGE_TEMPLATE.format(
                                    sentinel=ERROR_SENTINEL, minutes=max(1, round(ceiling / 60))
                                ),
                            }
                        else:
                            # A stream that breaks because the inference child died shows
                            # up here as a connection error; the engine knows the exit
                            # code and the child's last lines, so ask it.
                            logger.exception(
                                f"Agent streaming failed: llm={getattr(llm, 'id', '?')} "
                                f"({getattr(llm, 'name', '?')}), thread_id={thread_id}"
                                f"{_child_crash_suffix(engine)}"
                            )
                            if stateful:
                                await self._repair_alternation(agent, run_config)
                            # Parity with the pre-#297 live stream: text buffered before the
                            # failure would already have been yielded, so flush it ahead of
                            # the sentinel instead of dropping it.
                            for text in hop_text_buffer:
                                yield {"t": "answer", "text": text}
                            hop_text_buffer.clear()
                            yield {"t": "answer", "text": ERROR_MESSAGE}
            except BaseException:
                _flag_abandoned(abandon_hook)
                # Synchronous on purpose: the exit in flight (a closed
                # consumer, a cancellation) must not wait on a thread. It
                # reads two process counters and writes one small file.
                _close_memory_window(engine, memory_token, abandoned=True)
                raise
            if memory_token is not None:
                await run_in_threadpool(_close_memory_window, engine, memory_token, abandoned=False)

    async def astream_oneshot(
        self,
        *,
        llm,
        prompt_text: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """Stateless one-shot stream (no agent/checkpointer), e.g. title generation.

        Still routed through ``engine.generation_guard`` so it serializes with
        conversation/arena generations and keeps the model pinned while streaming.
        Failures are swallowed (the caller falls back to a default).

        Yields ANSWER text only (#266): inline ``<think>`` reasoning is stripped
        by a ``ThinkSplitter``, mirroring the two other streaming paths.
        """
        from langchain_core.messages import HumanMessage

        engine = config.LLM_Engine
        async with engine.generation_guard():
            try:
                model = await run_in_threadpool(
                    build_chat_model,
                    llm,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    # One-shot calls (titles) never want reasoning -- it would
                    # burn the whole tiny token budget inside <think>. The
                    # splitter below is the safety net for models/engines where
                    # chat-template-level suppression does not apply.
                    disable_thinking=True,
                    # ...and the native effort field says the same thing to the
                    # templates ``enable_thinking`` does not reach (1.1.2): a
                    # template that grades its reasoning reads "none" here and
                    # nothing at all from the chat-template kwarg.
                    effort_plan=NO_REASONING_PLAN,
                    # ...and that tiny budget is the point: the caller sized it
                    # for a 2-4 word title, so the automatic window-sized budget
                    # must not replace it.
                    auto_output_budget=False,
                    sampling=resolve_sampling_defaults(llm),
                )
            except Exception:
                # The caller falls back to a default title: recovered, but the
                # cause is worth the traceback.
                logger.warning(
                    f"One-shot model construction failed: llm={getattr(llm, 'id', '?')} "
                    f"({getattr(llm, 'name', '?')}); using the default",
                    exc_info=True,
                )
                return
            splitter = ThinkSplitter()
            # Same rule as a conversation turn: a title that does not end
            # normally (its caller went away) flags the child it ran against,
            # through the hook bound to the build-time handle. ``aclosing``
            # closes ``BaseChatModel.astream`` here, but that generator does
            # not close the client's own stream deterministically, so the
            # HTTP response may close -- and the client-level flag may land --
            # only after the guard is released. This flag makes the next
            # prefix reset send a barrier first, which waits for that request.
            abandon_hook = getattr(model, "abandon_hook", None)
            try:
                try:
                    async with contextlib.aclosing(
                        model.astream([HumanMessage(prompt_text)])
                    ) as chunks:
                        async for chunk in chunks:
                            text = getattr(chunk, "text", "")
                            if not text:
                                continue
                            for event in splitter.feed(text):
                                if event["t"] == "answer":
                                    yield event["text"]
                except Exception:
                    logger.warning(
                        f"One-shot streaming failed: llm={getattr(llm, 'id', '?')} "
                        f"({getattr(llm, 'name', '?')}); using the default"
                        f"{_child_crash_suffix(engine)}",
                        exc_info=True,
                    )
                    return
            except BaseException:
                _flag_abandoned(abandon_hook)
                raise
            # Stream completed normally: drain the splitter. An unclosed
            # <think> flushes as thinking and is dropped on purpose -- the
            # caller then falls back to its default title.
            for event in splitter.flush():
                if event["t"] == "answer":
                    yield event["text"]

    def _build_middleware(
        self, model, memory_budget=None, *, overhead: Optional[RequestOverhead] = None
    ):
        """Auto-summarization using the SAME local model, on the two-signal trigger.

        Runs per turn, after the child spawned, so the allocated window
        (``effective_context_tokens``) and the memory accounting are fresh for
        THIS model on THIS machine. The middleware rewrites the checkpointer
        state (drops old turns, inserts a summary) so the agent's context stays
        bounded; the Message table is untouched, so the UI still shows the full
        conversation.

        ``model`` here is the summary client -- the same child served by a
        second ``ChatOpenAI`` pinned to ``reasoning_effort="none"`` (1.1.2), so
        the summary is written rather than deliberated about, and capped at
        ``summary_cap(W)`` tokens.

        With a known working window the middleware keeps a TOKEN budget
        (``compaction_cutoff``) and reads up to ``summarize_trim_budget``
        tokens of history; without one it keeps the last
        ``SUMMARY_KEEP_MESSAGES`` messages and LangChain's 4000-token trim.
        Every count -- trigger, cutoff, trim, truncation -- goes through ONE
        counter in real tokens (``real_token_count``), and ``overhead`` (the
        request's system prompt, tool schemas and KB additions) enters the
        trigger and the keep.
        """
        from src.agents.middleware import (
            _StripStaleImagesMiddleware,
            _StripStaleToolResults,
        )

        allocated_window, working_window = self._compaction_windows(memory_budget)
        keep = (
            ("tokens", keep_token_budget(working_window))
            if _known_window(working_window)
            else ("messages", SUMMARY_KEEP_MESSAGES)
        )
        return [
            _StripStaleImagesMiddleware(),
            _StripStaleToolResults(),
            _summarization_middleware_class()(
                model=model,
                trigger=summarization_triggers(working_window),
                keep=keep,
                summary_prompt=SUMMARY_PROMPT,
                trim_tokens_to_summarize=summarize_trim_budget(working_window, allocated_window),
                working_window=working_window,
                allocated_window=allocated_window,
                overhead=overhead,
                engine=config.LLM_Engine,
            ),
        ]

    @staticmethod
    def _compaction_windows(memory_budget=None) -> tuple[Optional[int], Optional[int]]:
        """``(allocated window, working window)`` for this turn's compaction.

        The allocated window is the loaded child's (``effective_context_tokens``);
        the working window folds it with the memory ceiling into the ONE
        canonical value the memory consumers share.
        """
        engine = config.LLM_Engine
        window_probe = getattr(engine, "effective_context_tokens", None)
        effective_window = window_probe() if callable(window_probe) else None
        memory_token_ceiling = (
            memory_budget.tokens_at_ceiling() if memory_budget is not None else None
        )
        # Compact-ASAP safety net: a non-positive ceiling means the child's
        # fixed part alone already fills the memory budget. Clamp it up to 1 so
        # the working window is 1 and compaction fires as early as it can --
        # replacing N messages with one summary shrinks the live KV, which is
        # exactly what relieves such a machine. This clamp is DELIBERATELY only
        # on the compaction path: the output budget (working_context_tokens,
        # which does NOT clamp) keeps sizing from the allocated window in this
        # corner, precisely what it did before PR3.1. Both views get unified
        # when the reactive memory brake lands (PR3.4).
        if memory_token_ceiling is not None:
            memory_token_ceiling = max(1, memory_token_ceiling)
        # Fold the allocated window and the memory ceiling into the ONE
        # canonical working window (the memory consumer's value); the raw
        # allocated window stays with the time/bound consumers elsewhere.
        return effective_window, canonical_working_window(effective_window, memory_token_ceiling)

    async def _memory_warning_event(
        self,
        agent,
        run_config,
        budget,
        working_window: Optional[int] = None,
        *,
        overhead: Optional[RequestOverhead] = None,
        allocated_window: Optional[int] = None,
    ) -> Optional[dict]:
        """The ``memory_warning`` event for this turn, or ``None``.

        Warn ONLY IF compaction cannot save this conversation. The
        summarization middleware runs in ``before_model``, so the growth of
        THIS turn is compacted on the NEXT turn -- judging the warning on the
        current size would flag every conversation for exactly one turn and
        then flicker off once compaction ran. Instead the post-turn thread
        state is projected the way the NEXT request will see it, in real
        tokens: the turn just answered is past (its KB/web results count as
        markers, ``strip_stale_tool_results(all_past=True)``), an empty next
        question is appended (so the cutoff is the one the next request's
        compaction computes), every message is weighed as the counter weighs
        it, and the request overhead O is on top -- the system prompt and the
        tool schemas only (``overhead``; the KB block of the turn just
        answered is not in the next request). When a compaction would happen
        (``compaction_cutoff`` > 0, futility guard included):
        ``O + count(kept) + summary_cap(W)`` (or ``SUMMARY_TOKEN_ALLOWANCE``
        without a window); otherwise ``O + count(all)`` -- no summary is added
        when nothing would be compacted. The warning goes out only when even
        THAT projection leaves the margin strictly under
        ``MEMORY_WARNING_MARGIN``: the model plus the kept conversation would
        use more than 85 % of the memory budget (``MemoryBudget.used_fraction``,
        a measured prior with a fixed part -- so it can show for a model whose
        weights alone are well below 85 %, and from the first turn on a tight
        machine). The payload quotes the projection: ``footprint_bytes``
        (model and conversation, what the copy shows) and
        ``conversation_bytes`` (what the conversation adds). Advisory only: a
        failure here is logged and never sinks the turn.
        """
        from langchain_core.messages import HumanMessage

        from src.agents.middleware import strip_stale_tool_results

        try:
            state = await agent.aget_state(run_config)
            messages = (state.values or {}).get("messages", []) if state else []
            if not messages:
                return None
            projected = strip_stale_tool_results(messages, all_past=True) + [
                HumanMessage(content="", id="erudi-next-request")
            ]
            weights, ratio = frozen_weights(projected)

            def counter(part):
                return real_token_count(part, weights)

            fixed = overhead_tokens(
                overhead,
                ratio,
                dense=COUNTER_DENSE_TOKENS,
                weight_floor=COUNTER_WEIGHT_FLOOR,
            )
            cutoff = compaction_cutoff(
                projected,
                working_window,
                counter,
                overhead=fixed,
                allocated_window=allocated_window,
            )
            conversation_tokens = fixed + counter(projected)
            if cutoff > 0:
                summary_tokens = (
                    summary_cap(working_window)
                    if _known_window(working_window)
                    else SUMMARY_TOKEN_ALLOWANCE
                )
                projected_tokens = fixed + counter(projected[cutoff:]) + summary_tokens
            else:
                projected_tokens = conversation_tokens
            projected_margin = budget.memory_margin_fraction(projected_tokens)
            if projected_margin is None or projected_margin >= MEMORY_WARNING_MARGIN:
                return None
            logger.warning(
                f"Memory budget over 85 % even after a projected compaction: "
                f"projected_margin={projected_margin:.3f}, "
                f"projected_tokens={projected_tokens}, "
                f"conversation_tokens={conversation_tokens}"
            )
            return {
                "t": "memory_warning",
                "used_fraction": round(1.0 - projected_margin, 4),
                "conversation_bytes": budget.conversation_bytes(projected_tokens),
                # What the warning copy quotes: the conversation AND its
                # loaded model together (the two things the user can act on).
                "footprint_bytes": budget.footprint_bytes(projected_tokens),
            }
        except Exception:
            # Advisory signal: losing it costs one warning, never the answer.
            logger.exception("Memory-margin evaluation failed; skipping the warning")
            return None

    async def _write_curated_empty_turn(self, agent, run_config, text: str) -> None:
        """Write the curated empty-answer line into the thread state (#554).

        A turn that ended with no answer text has committed an EMPTY
        ``AIMessage`` to the checkpointer (the model node aggregates the
        streamed chunks, reasoning excluded). Mirror of ``_repair_alternation``:
        update the state as the ``model`` node -- replacing the trailing empty
        assistant message in place (same id, LangGraph's ``add_messages``
        replaces on id match) so the next turn's template sees the curated
        line, not an empty turn; append instead when the last message is
        something else (defensive: the state is then already well-formed).
        """
        from langchain_core.messages import AIMessage

        try:
            state = await agent.aget_state(run_config)
            messages = (state.values or {}).get("messages", []) if state else []
            curated = AIMessage(content=text)
            if messages:
                last = messages[-1]
                if (
                    last.type == "ai"
                    and not getattr(last, "tool_calls", None)
                    and not str(getattr(last, "text", "") or "").strip()
                ):
                    # A copy of the empty message: same id (replaced in
                    # place), and its usage and stamp survive -- the request
                    # it answered is still a measurement. Its output count is
                    # the replaced generation's (a whitespace loop can be 20k
                    # tokens), not the curated line's: none is kept.
                    usage = dict(last.usage_metadata or {})
                    if usage:
                        usage["output_tokens"] = 0
                        usage["total_tokens"] = usage.get("input_tokens", 0)
                    curated = last.model_copy(
                        update={"content": text, "usage_metadata": usage or None}
                    )
            await agent.aupdate_state(run_config, {"messages": [curated]}, as_node="model")
        except Exception:
            # Accepted trade: a failed state write leaves SQL ahead of the
            # thread state (the user still gets the curated line; the next
            # turn replays an empty assistant message instead of it). There is
            # no better recovery than proceeding -- retrying here would block
            # the turn on a checkpointer that just failed.
            logger.exception("Failed to write the curated empty-answer turn into the thread state")

    async def _repair_alternation(self, agent, run_config) -> None:
        """Preserve role alternation in the checkpointer after a failed turn.

        If the failed super-step left a dangling ``HumanMessage`` as the last
        message, the next turn would send two consecutive user messages and the
        local chat template would 400 ("roles must alternate"). Append an error
        ``AIMessage`` so the thread stays well-formed. If the super-step never
        committed (last message is not a human, or state is empty), do nothing.
        """
        from langchain_core.messages import AIMessage

        try:
            state = await agent.aget_state(run_config)
            messages = (state.values or {}).get("messages", []) if state else []
            if messages and messages[-1].type == "human":
                # as_node="model" is required: updating a non-empty thread is
                # otherwise "ambiguous". "model" is the create_agent node name.
                await agent.aupdate_state(
                    run_config,
                    {"messages": [AIMessage(content=ERROR_MESSAGE)]},
                    as_node="model",
                )
        except Exception:
            logger.exception("Failed to repair conversation alternation after error")
