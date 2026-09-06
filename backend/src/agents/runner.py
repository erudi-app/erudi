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

from src.agents.model_factory import build_chat_model
from src.database.generation_hints import resolve_sampling_defaults
from src.agents.think_splitter import ThinkSplitter
from src.core import config
from src.core.exceptions import EngineException
from src.core.logging import logger

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


# Auto-summarization thresholds (message-count based — token triggers need a
# model profile the local server doesn't expose). Once a conversation's agent
# state exceeds the trigger, older turns are summarized by the same local model
# and replaced in the checkpointer state; the Message table keeps the full
# history for display.
SUMMARY_TRIGGER_MESSAGES = 20
SUMMARY_KEEP_MESSAGES = 10

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

        Yields ``{"t":"answer","text":...}`` (text outside ``<think>``),
        ``{"t":"thinking","text":...}`` (text inside ``<think>``), one
        ``{"t":"tool_call","name":...,"args":{...}}`` per call, and
        ``{"t":"tool_result","name":...,"text":...}`` per ToolMessage.

        Error paths (#252 construction failure, streaming failure) yield the
        curated ERROR sentinel as an ``answer`` event -- callers map it: the
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
                middleware = self._build_middleware(model) if summarize else []
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
            # first-token latency, then one completion line with totals. Counts
            # ANSWER text only -- thinking is separated out and must not inflate
            # the answer accounting nor the empty-final signal below.
            stream_start_s = time.perf_counter()
            first_token_s: Optional[float] = None
            chunk_count = 0
            char_count = 0
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
                        text = getattr(token, "text", "")
                        if text:
                            if first_token_s is None:
                                first_token_s = time.perf_counter()
                                logger.info(
                                    f"Agent first token: llm={getattr(llm, 'id', '?')}, "
                                    f"latency_ms={(first_token_s - stream_start_s) * 1000:.0f}"
                                )
                            chunk_count += 1
                            for event in splitter.feed(text):
                                if event["t"] != "answer":
                                    # Real <think> content: flows immediately.
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
                    yield event
                # Empty/blank final answer, but a tool produced a result this
                # turn: deliver that last tool result AS THE ANSWER (#90) so a
                # correct value is streamed and persisted instead of crashing the
                # empty-content guard. No tool ran -> nothing to fall back to;
                # keep today's behavior (a genuine empty-answer failure).
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
                duration_ms = (time.perf_counter() - stream_start_s) * 1000
                logger.info(
                    f"Agent stream completed: llm={getattr(llm, 'id', '?')}, "
                    f"duration_ms={duration_ms:.0f}, chunks={chunk_count} (~tokens), "
                    f"chars={char_count}"
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

    def _build_middleware(self, model):
        """Auto-summarization using the SAME local model, triggered by message count.

        The middleware rewrites the checkpointer state (drops old turns, inserts a
        summary) so the agent's context stays bounded; the Message table is
        untouched, so the UI still shows the full conversation.
        """
        from langchain.agents.middleware import SummarizationMiddleware
        from langchain_core.messages.utils import count_tokens_approximately

        from src.agents.middleware import (
            _StripStaleImagesMiddleware,
            _StripStaleToolResults,
        )

        return [
            _StripStaleImagesMiddleware(),
            _StripStaleToolResults(),
            SummarizationMiddleware(
                model=model,
                trigger=("messages", SUMMARY_TRIGGER_MESSAGES),
                keep=("messages", SUMMARY_KEEP_MESSAGES),
                token_counter=count_tokens_approximately,
            ),
        ]

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
