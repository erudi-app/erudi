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

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

from fastapi.concurrency import run_in_threadpool

from src.agents.chat_model import INTER_CHUNK_BUDGET_S, PHASE_FIRST_CHUNK
from src.agents.model_factory import build_chat_model
from src.database.generation_hints import resolve_sampling_defaults
from src.agents.think_splitter import ThinkSplitter
from src.core import config
from src.core.exceptions import EngineException, GenerationTimeoutException
from src.core.logging import logger
from src.engines.memory_budget import MemoryBudget

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


# Auto-summarization (compaction) fires on TWO signals, whichever comes first,
# recomputed PER TURN (``_build_middleware`` runs after the child spawned, so
# the allocated window is fresh):
#   1. the WINDOW signal -- the conversation reaches 80 % of the ALLOCATED
#      context window (``BaseEngine.effective_context_tokens``);
#   2. the MEMORY signal -- the machine's deterministic memory margin
#      (``src.engines.memory_budget``) would drop under 15 %, expressed as the
#      conversation token count at which that happens.
# The 20-message floor stays as the trigger when no window is readable
# (``W_eff=None``), and always rides along with OR semantics. Once triggered,
# older turns are summarized by the same local model and replaced in the
# checkpointer state; the Message table keeps the full history for display.
SUMMARY_TRIGGER_MESSAGES = 20
SUMMARY_KEEP_MESSAGES = 10
COMPACTION_WINDOW_FRACTION = 0.8
# Shared floor of the memory signal AND the amber warning: under 15 % of
# deterministic margin, compaction fires; if the margin is STILL under 15 %
# at end of turn (compaction had its chance), the warning is emitted.
MEMORY_MARGIN_FLOOR = 0.15


def summarization_triggers(
    effective_window: Optional[int], memory_token_ceiling: Optional[int] = None
) -> list:
    """The OR-semantics trigger list for ``SummarizationMiddleware``.

    ``effective_window`` is the ALLOCATED window (80 % of it becomes the token
    threshold); ``memory_token_ceiling`` is the conversation token count at
    which the memory margin hits its floor (``MemoryBudget.tokens_at_margin``,
    possibly negative when the weights alone blow it -- clamped to 1 so
    compaction fires as early as it can). The two fold into ONE ``tokens``
    entry (min: whichever signal comes first) so the middleware sees at most
    one token threshold plus the message floor.

    Deliberately never ``("fraction", ...)``: that form needs a
    ``model.profile`` our local chat clients do not carry (the middleware's
    ``__init__`` would raise). The token counter stays
    ``count_tokens_approximately`` -- the same counter the output budget uses,
    so the two never disagree.
    """
    token_thresholds = []
    if isinstance(effective_window, int) and effective_window > 0:
        token_thresholds.append(int(COMPACTION_WINDOW_FRACTION * effective_window))
    if memory_token_ceiling is not None:
        token_thresholds.append(max(1, memory_token_ceiling))
    triggers: list = []
    if token_thresholds:
        triggers.append(("tokens", max(1, min(token_thresholds))))
    triggers.append(("messages", SUMMARY_TRIGGER_MESSAGES))
    return triggers

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
        async for event in self._astream_events(
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
        ):
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
                model = await run_in_threadpool(
                    build_chat_model,
                    llm,
                    temperature=params.temperature,
                    top_p=params.top_p,
                    max_tokens=params.max_tokens,
                    # Per-model extra sampling keys (#388); the user-facing three
                    # above still come from the conversation row / arena panel.
                    sampling=resolve_sampling_defaults(llm),
                )
                # Memory accounting for the compaction signal and the amber
                # warning (1.1.2). Derived AFTER build_chat_model so the child
                # is up and the handle carries the loaded artifact; file I/O
                # and hardware probes -> threadpool. ``from_engine`` never
                # raises; an unaccountable model just carries None facts.
                budget = (
                    await run_in_threadpool(MemoryBudget.from_engine, engine)
                    if summarize
                    else None
                )
                middleware = self._build_middleware(model, budget) if summarize else []
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
                # No implicit tools (#129): callers own the tool list (built by
                # ``plan_turn``); ``tools=None`` means a zero-tool agent.
                effective_tools = tools if tools is not None else []
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
                f"Agent stream started: llm={getattr(llm, 'id', '?')}, " f"thread_id={thread_id}"
            )
            try:
                async for token, meta in agent.astream(
                    {"messages": [HumanMessage(user_message)]},
                    config=run_config,
                    context=context,
                    stream_mode="messages",
                ):
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
                        reasoning_delta = (getattr(token, "additional_kwargs", None) or {}).get(
                            "reasoning_content"
                        ) or ""
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
                # Compact first, warn second (1.1.2): the summarization
                # middleware ran INSIDE this turn, so by stream end it has had
                # its chance -- evaluate the memory margin HERE, at end of
                # turn, on the POST-turn thread state (the compacted state a
                # follow-up will actually replay). If the margin is still
                # under the floor, one ``memory_warning`` event goes out; the
                # conversation service forwards it to the wire and never
                # persists it (it is a statement about NOW, on THIS machine).
                if stateful and budget is not None:
                    warning = await self._memory_warning_event(agent, run_config, budget)
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
            except Exception:
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
            try:
                async for chunk in model.astream([HumanMessage(prompt_text)]):
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
            # Stream completed normally: drain the splitter. An unclosed
            # <think> flushes as thinking and is dropped on purpose -- the
            # caller then falls back to its default title.
            for event in splitter.flush():
                if event["t"] == "answer":
                    yield event["text"]

    def _build_middleware(self, model, memory_budget=None):
        """Auto-summarization using the SAME local model, on the two-signal trigger.

        Runs per turn, after the child spawned, so the allocated window
        (``effective_context_tokens``) and the memory accounting are fresh for
        THIS model on THIS machine. The middleware rewrites the checkpointer
        state (drops old turns, inserts a summary) so the agent's context stays
        bounded; the Message table is untouched, so the UI still shows the full
        conversation.
        """
        from langchain.agents.middleware import SummarizationMiddleware
        from langchain_core.messages.utils import count_tokens_approximately

        from src.agents.middleware import (
            _StripStaleImagesMiddleware,
            _StripStaleToolResults,
        )

        engine = config.LLM_Engine
        window_probe = getattr(engine, "effective_context_tokens", None)
        effective_window = window_probe() if callable(window_probe) else None
        memory_token_ceiling = (
            memory_budget.tokens_at_margin(MEMORY_MARGIN_FLOOR)
            if memory_budget is not None
            else None
        )
        return [
            _StripStaleImagesMiddleware(),
            _StripStaleToolResults(),
            SummarizationMiddleware(
                model=model,
                trigger=summarization_triggers(effective_window, memory_token_ceiling),
                keep=("messages", SUMMARY_KEEP_MESSAGES),
                token_counter=count_tokens_approximately,
            ),
        ]

    async def _memory_warning_event(self, agent, run_config, budget) -> Optional[dict]:
        """The ``memory_warning`` event for this turn, or ``None``.

        Reads the POST-turn thread state (what the next turn will replay,
        summarization included), counts it with the SAME
        ``count_tokens_approximately`` the compaction trigger uses, and asks
        the deterministic accounting for the margin. Strictly under the shared
        floor -> the event; anything else (fine margin, unaccountable model,
        empty state) -> ``None``. Advisory only: a failure here is logged and
        never sinks the turn.
        """
        from langchain_core.messages.utils import count_tokens_approximately

        try:
            state = await agent.aget_state(run_config)
            messages = (state.values or {}).get("messages", []) if state else []
            if not messages:
                return None
            conversation_tokens = count_tokens_approximately(messages)
            margin = budget.memory_margin_fraction(conversation_tokens)
            if margin is None or margin >= MEMORY_MARGIN_FLOOR:
                return None
            logger.warning(
                f"Memory margin still under the floor after compaction: "
                f"margin={margin:.3f}, conversation_tokens={conversation_tokens}"
            )
            return {
                "t": "memory_warning",
                "used_fraction": round(1.0 - margin, 4),
                "conversation_bytes": budget.conversation_bytes(conversation_tokens),
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
            replace_id = None
            if messages:
                last = messages[-1]
                if (
                    last.type == "ai"
                    and not getattr(last, "tool_calls", None)
                    and not str(getattr(last, "text", "") or "").strip()
                ):
                    replace_id = last.id
            await agent.aupdate_state(
                run_config,
                {"messages": [AIMessage(content=text, id=replace_id)]},
                as_node="model",
            )
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
