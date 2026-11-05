"""Services for conversation management and streaming AI generation.

Business logic for the conversation lifecycle and streaming generation. Since the
LangChain refactor, generation goes through ``AgentRunner`` (one ``create_agent``
per turn, history persisted in the LangGraph checkpointer) instead of the
hand-rolled prompt/engine loop. The SQLAlchemy ``Message``/``Conversation`` tables
remain the source of truth for display (fetch_messages, starred, timestamps); the
checkpointer holds the agent's working state and rolling summary.

Layering: streaming methods are async generators (the FastAPI ``StreamingResponse``
consumes them directly on the event loop); all synchronous SQLAlchemy work is
wrapped in ``run_in_threadpool`` so DB commits never block the loop.
"""

import contextlib
import json
import re
import time
from typing import AsyncGenerator, Optional

from fastapi.concurrency import run_in_threadpool

from src.core.error_detail import error_detail
from src.core.logging import logger
from src.core.logutils import truncate_for_log
from src.agents.kb_mode import plan_turn
from src.agents.reasoning_effort import plan_reasoning_effort
from src.agents.runner import (
    AgentRunner,
    COMPACTION_EVENT,
    ERROR_DETAIL_EVENT,
    GenParams,
    ERROR_MESSAGE,
    ERROR_SENTINEL,
    IMAGES_IGNORED_NOTICE,
)
from src.database.generation_hints import FALLBACK_MAX_TOKENS, resolve_sampling_defaults
from src.domains.conversations.repository import ConversationRepository, MessageRepository
from src.domains.llms.conversation_defaults import SOURCE_SAVED, read_conversation_defaults
from src.domains.llms.repository import detect_supports_vision
from src.domains.conversations.schemas import ConversationQuery
from src.entities.Conversation import Conversation
from src.entities.Llm import Llm
from src.entities.Message import Message
from src.core.exceptions import (
    CompactionSeparatorException,
    DatabaseException,
    ModelNotFoundException,
    StateConflictException,
)
from src.core.request_context import get_request_id
from src.utils.attachment_utils import (
    build_attachment_notice,
    prepend_attachment_block,
    resolve_attachments,
)
from src.utils.kb_utils import KbExcerpt, retrieve_kb_excerpts
from src.utils.prompt_utils import get_prompting_strategy


# Serialized-trace cap (#90): persisted traces are bounded so a runaway
# thinking/tool turn can't bloat the messages row. Drop-oldest past the cap and
# prepend a ``{"t":"truncated"}`` marker so the UI can show reasoning was elided.
TRACE_MAX_BYTES = 32 * 1024

# A title needs the head of the message, not its whole body; the prefill cost
# otherwise doubles on a long first message.
TITLE_QUESTION_MAX_CHARS = 2000


def _ndjson(event: dict) -> str:
    """Serialize one stream event as a single NDJSON line.

    ``ensure_ascii`` defaults to True, so non-ASCII answer/thinking text is
    ``\\uXXXX``-escaped on the wire (ASCII-safe) and decoded by the client's
    ``JSON.parse``.
    """
    return json.dumps(event) + "\n"


def build_stream_error_event(answer_event: dict) -> dict:
    """The wire ``error`` event for a sentinel-carrying answer event.

    The renderer has always read ``{"t": "error", "text": ...}`` and renders it
    as the red turn. A failure the engine could identify adds two keys the
    renderer acts on rather than displays: ``code`` (one of
    ``src.engines.cuda_compatibility.CUDA_FAILURE_CODES``) and ``raw`` (the
    child's captured output, shown in the copyable block of the decision
    dialog). Both are omitted for every other failure, so the untyped path
    keeps exactly the shape it had.
    """
    error_event: dict = {"t": "error", "text": answer_event["text"]}
    for key in ("code", "raw", "detail", "request_id"):
        value = answer_event.get(key)
        if value:
            error_event[key] = value
    if answer_event.get("curated"):
        # Words the app chose (a prompt that does not fit, a timeout, the loop
        # limit): the renderer does not count the turn as a bug.
        error_event["curated"] = True
    return error_event


def _is_compaction(event: dict) -> bool:
    return event.get("t") == COMPACTION_EVENT


def _is_pinned(event: dict) -> bool:
    """What an error turn keeps of its trace and what a cap never drops: its
    compactions and the cleaned cause of its failure."""
    return _is_compaction(event) or event.get("t") == ERROR_DETAIL_EVENT


def _append_failure(partial: str, failure: str) -> str:
    """The red line of a failed turn goes below what streamed before it, after
    exactly one blank line (a notice already ends with one)."""
    head = partial.rstrip("\n")
    return f"{head}\n\n{failure}" if head else failure


def _error_detail_event(detail: str, request_id: Optional[str]) -> dict:
    return {"t": ERROR_DETAIL_EVENT, "text": detail, "request_id": request_id}


def _cap_trace(events: list) -> Optional[list]:
    """Cap the serialized trace at ``TRACE_MAX_BYTES`` (drop-oldest).

    Returns ``None`` for an empty trace (nothing to persist). When the trace fits,
    it is returned unchanged. Otherwise the oldest events are dropped -- with the
    ``{"t":"truncated"}`` marker counted in the budget -- until the marker plus the
    remaining events fit, always keeping at least the newest event.

    Compaction events (#614) and the failure's ``error_detail`` (#615) are
    pinned: a reasoning turn emits one event per token, so a compaction
    announced at the first model call would be the first dropped. They lead the
    capped trace, in their order, and are never dropped;
    the other events are capped against what they leave -- all of them may go,
    and the marker follows the compactions only when one of them did. With a
    known window a summary is capped at ``summary_cap`` tokens and a turn's
    compactions fit easily; without one the summary client runs on the
    conversation's output budget, so compactions alone can exceed the cap --
    they are kept whole all the same.
    """
    if not events:
        return None
    if len(json.dumps(events)) <= TRACE_MAX_BYTES:
        return list(events)
    marker = {"t": "truncated"}
    pinned = [e for e in events if _is_pinned(e)]
    others = [e for e in events if not _is_pinned(e)]
    capped = list(others)
    floor = 0 if pinned else 1
    while len(capped) > floor and len(json.dumps([*pinned, marker, *capped])) > TRACE_MAX_BYTES:
        capped.pop(0)
    if len(capped) == len(others):
        # Nothing elided: only the compactions themselves exceed the cap.
        return [*pinned, *capped]
    return [*pinned, marker, *capped]


def _sanitize_title(raw: str, *, max_words: int = 6, max_chars: int = 48) -> str:
    """Clean a model-generated conversation title.

    Tiny local models occasionally emit markdown noise on instruction-style first
    messages (e.g. a repeated ```json fence). Strip code fences / backticks /
    wrapping quotes / list markers, keep the first line, collapse consecutive
    duplicate words, and cap length. Returns "" when nothing usable remains so the
    caller can fall back to the default name.
    """
    if not raw:
        return ""
    # Remove fenced-code markers (```lang) and stray backticks.
    text = re.sub(r"```[a-zA-Z0-9]*", " ", raw).replace("`", " ")
    # Titles are single-line; keep the first non-empty line.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    text = lines[0] if lines else ""
    # Drop wrapping quotes and leading list/heading punctuation.
    text = text.strip(" \"'#*-•.").strip()
    # Collapse consecutive duplicate words ("json json json" -> "json").
    out: list[str] = []
    for word in text.split():
        if not out or out[-1].lower() != word.lower():
            out.append(word)
    text = " ".join(out[:max_words])[:max_chars].strip()
    # Reject leftovers that are only a fence language tag / generic noise.
    if text.lower() in {"", "json", "code", "markdown", "text", "title"}:
        return ""
    return text


class ConversationService:
    """Service for managing conversations and message processing."""

    def __init__(self, db, checkpointer=None):
        """Initialize the service.

        Args:
            db: SQLAlchemy session for repository operations.
            checkpointer: LangGraph checkpointer (app-wide ``AsyncPostgresSaver``)
                for stateful conversations. ``None`` runs the agent statelessly
                (used by tests that don't exercise persistence).
        """
        logger.debug("Initializing ConversationService")
        self.conversation_repo = ConversationRepository(db)
        self.message_repo = MessageRepository(db)
        self.checkpointer = checkpointer
        self.runner = AgentRunner(checkpointer)
        self.db = db

    # ===================== Synchronous CRUD =====================
    def create_conversation(
        self,
        llm_id: int,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        custom_prompt: Optional[str] = None,
        web_search_enabled: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        compaction_threshold: Optional[int] = None,
    ) -> Conversation:
        """Create a new conversation with the given LLM and generation params.

        Every value left ``None`` is resolved for the model by
        ``resolve_conversation_defaults`` (#609): the value saved for it with
        ``/config global``, else the global user setting (effort, web search,
        threshold) or the model's recommendation (temperature, top_p), else a
        constant; custom instructions default to empty. ``max_tokens`` comes
        from the model's own defaults (#388). An explicit value always wins.
        The conversation owns its values afterwards -- later changes to the
        model's saved values or to the settings never retro-affect it.

        Raises:
            ModelNotFoundException: If ``llm_id`` does not exist.
        """
        logger.info(f"Creating new conversation with LLM {llm_id}")
        llm = self.conversation_repo.get_llm_by_id(llm_id)
        resolved = read_conversation_defaults(self.db, llm)
        sent = {
            "reasoning_effort": reasoning_effort,
            "temperature": temperature,
            "top_p": top_p,
            "compaction_threshold": compaction_threshold,
            "web_search_enabled": web_search_enabled,
            "custom_prompt": custom_prompt,
        }
        values = {
            field: resolved[field].value if value is None else value
            for field, value in sent.items()
        }
        if max_tokens is None:
            max_tokens = resolve_sampling_defaults(llm).max_tokens
        return self.conversation_repo.create_conversation(
            llm_id=llm_id,
            name="New Conversation",
            max_tokens=max_tokens,
            **values,
        )

    def update_conversation(
        self,
        conversation_id: int,
        name: Optional[str] = None,
        llm_id: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        custom_prompt: Optional[str] = None,
        web_search_enabled: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        compaction_threshold: Optional[int] = None,
    ) -> Conversation:
        """Partial update of conversation metadata (only non-None fields).

        Switching model (#609, the ``/model`` command): when ``llm_id`` differs
        from the stored one (an unbound conversation's included), the new
        model brings its own values in the same transaction:

        - every value saved for it with ``/config global`` replaces the
          conversation's, whatever this update carries for that field;
        - ``temperature`` and ``top_p`` it has not saved take its
          recommendation (``resolve_sampling_defaults``), and the
          ``temperature`` / ``top_p`` of this update are ignored;
        - ``reasoning_effort``, ``compaction_threshold``,
          ``web_search_enabled`` and ``custom_prompt`` it has not saved take
          this update's value when it carries one, else stay as they are;
        - ``max_tokens``, when this update does not carry one, takes the new
          model's fallback budget (``resolve_sampling_defaults``), as at
          creation.

        An update whose ``llm_id`` equals the stored one is an ordinary
        partial update: nothing of the model is re-applied.

        Once the repository has written the update, one INFO line names the
        fields written (``Conversation updated``), and a switch adds one
        naming what the new model brought.

        Raises:
            ConversationNotFoundException: If the conversation does not exist.
            ModelNotFoundException: If the new ``llm_id`` does not exist
                (nothing is changed).
        """
        logger.info(f"Updating conversation {conversation_id}")
        conversation = self.conversation_repo.get_conversation_by_id(conversation_id)
        resolved = None
        if llm_id is not None and llm_id != conversation.llm_id:
            new_llm = self.conversation_repo.get_llm_by_id(llm_id)
            resolved = read_conversation_defaults(self.db, new_llm)
            temperature = resolved["temperature"].value
            top_p = resolved["top_p"].value
            if max_tokens is None:
                max_tokens = resolve_sampling_defaults(new_llm).max_tokens
            if resolved["reasoning_effort"].source == SOURCE_SAVED:
                reasoning_effort = resolved["reasoning_effort"].value
            if resolved["compaction_threshold"].source == SOURCE_SAVED:
                compaction_threshold = resolved["compaction_threshold"].value
            if resolved["web_search_enabled"].source == SOURCE_SAVED:
                web_search_enabled = resolved["web_search_enabled"].value
            if resolved["custom_prompt"].source == SOURCE_SAVED:
                custom_prompt = resolved["custom_prompt"].value
        written = {
            "name": name,
            "llm_id": llm_id,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "custom_prompt": custom_prompt,
            "web_search_enabled": web_search_enabled,
            "reasoning_effort": reasoning_effort,
            "compaction_threshold": compaction_threshold,
        }
        updated = self.conversation_repo.update_conversation(
            conversation_id=conversation_id, **written
        )
        logger.info(
            f"Conversation updated: id={conversation_id}, "
            f"fields={[field for field, value in written.items() if value is not None]}"
        )
        if resolved is not None:
            saved = [f for f, r in resolved.items() if r.source == SOURCE_SAVED]
            logger.info(
                f"Conversation {conversation_id} switched to model {llm_id}: "
                f"saved=[{', '.join(saved)}], "
                f"temperature={temperature} ({resolved['temperature'].source}), "
                f"top_p={top_p} ({resolved['top_p'].source}), "
                f"max_tokens={max_tokens}"
            )
        return updated

    # ===================== Deletion (DB + checkpointer thread) =====================
    async def delete_conversation(self, conversation_id: int) -> None:
        """Delete a conversation and purge its checkpointer thread.

        Both stores must be cleared: ids can be reused after a fresh start or a
        sequence reset, so a stale checkpointer thread would otherwise leak a deleted
        conversation's agent context into a future one with the same id (BLOCKER B3).
        """
        logger.info(f"Deleting conversation {conversation_id}")
        await run_in_threadpool(self._delete_conversation_db, conversation_id)
        await self._purge_thread(conversation_id)

    def _delete_conversation_db(self, conversation_id: int) -> None:
        self.conversation_repo.delete_conversation(conversation_id)
        self.db.commit()

    async def _purge_thread(self, conversation_id: int) -> None:
        if self.checkpointer is None:
            return
        try:
            await self.checkpointer.adelete_thread(str(conversation_id))
        except Exception:
            logger.exception(
                f"Failed to purge checkpointer thread for conversation {conversation_id}"
            )

    def store_error_message(self, conversation_id: int) -> int:
        """Persist a fallback error message (called by the frontend on failure)."""
        logger.info(f"Storing error message for conversation {conversation_id}")
        message = self.message_repo.create_message(
            conversation_id=conversation_id,
            content=ERROR_MESSAGE,
            sender="llm",
        )
        self.conversation_repo.update_last_message_time(conversation_id)
        return message.id

    # ===================== Streaming generation =====================
    def _retrieve_kb_excerpts(self, llm: Llm, query: str) -> list[KbExcerpt]:
        """Adaptive KB excerpts for this question, or [] (no KB / disabled / error).

        Same contract as the arena's retrieval, with one difference: a
        retrieval failure degrades to no-context instead of erroring — a
        broken vector store must not sink an ongoing conversation.
        """
        if not llm.is_attached_to_kb:
            return []

        param_size = llm.param_size if llm.param_size is not None else 2
        strategy = get_prompting_strategy(param_size)
        if not strategy.get("use_kb_context", False):
            return []

        try:
            return retrieve_kb_excerpts(query, llm.kb_id, token_budget=strategy["kb_token_budget"])
        except Exception as e:
            # Degraded, not failed: the turn is answered without the excerpts.
            logger.warning(
                f"KB retrieval failed for kb {llm.kb_id} (llm {getattr(llm, 'id', '?')}); "
                f"continuing without context: {e}",
                exc_info=True,
            )
            return []

    async def query_and_respond_stream(
        self,
        conversation_id: int,
        payload: ConversationQuery,
    ) -> AsyncGenerator[str, None]:
        """Stream the agent's turn as NDJSON events (``application/x-ndjson``).

        One JSON object per line: ``answer`` (streamed answer text), ``thinking``
        (streamed reasoning), ``tool_call`` / ``tool_result`` (agent activity),
        ``compaction`` (#614: ``phase: "start"`` while earlier messages are being
        summarized, ``phase: "done"`` with ``summarized``, ``reason``, ``summary``
        and ``summary_lost`` once they were), ``memory_reload``
        (the model reloads; ``reason`` says why: ``memory`` the memory brake
        unloaded or restarted it, ``idle`` the idle delay unloaded it,
        ``manual`` the user did) / ``memory_warning`` (this
        machine's memory), ``error`` (the mapped ERROR
        sentinel), then a terminal ``done`` (#90).

        The answer text is accumulated exactly as before and persisted as the
        assistant message ``content``; thinking, tool and compaction-``done``
        events are collected into an ordered ``trace`` (capped, drop-oldest,
        compactions pinned) and persisted alongside so the panel and the
        compaction marker replay on reload. ``compaction`` ``start`` and the
        memory events reach the wire only. Persists the user message up front (so
        it shows immediately); the agent restores prior history from the
        checkpointer.

        Error mapping (#252/#225-D4): the runner yields its curated error as an
        ``answer`` event carrying the ERROR sentinel. Here that answer is (a) still
        appended to ``assistant_response`` so the persisted content is byte-for-byte
        the pre-#90 error turn, and (b) re-framed on the wire as an ``error`` event
        (the frontend renders the red bubble from ``error`` as it did from the
        sentinel substring). Error turns persist only their compaction and error_detail events
        (the compaction is checkpointed even when the model call then fails).
        """
        start_s = time.perf_counter()
        # Local-first policy: log the question content (bounded), never image
        # bytes — only their count and encoded size.
        image_note = ""
        if payload.images:
            total_b64_chars = sum(len(url) for url in payload.images)
            image_note = f", images={len(payload.images)} ({total_b64_chars} base64 chars)"
        if payload.attachments:
            image_note += f", attachments={len(payload.attachments)}"
        logger.info(
            f"Processing query for conversation {conversation_id}{image_note}: "
            f"{truncate_for_log(payload.question, 2000)}"
        )

        try:
            conversation, llm = await run_in_threadpool(
                self._load_conversation_and_llm, conversation_id
            )
        except Exception as exc:
            # The response has started (an NDJSON stream), so no exception
            # handler will see this: the record is written here.
            logger.exception(f"Failed to load conversation {conversation_id} or its LLM for query")
            # No conversation loaded -> nothing to persist; just surface the error.
            yield _ndjson(
                {
                    "t": "error",
                    "text": ERROR_MESSAGE,
                    "detail": error_detail(exc),
                    "request_id": get_request_id(),
                }
            )
            yield _ndjson({"t": "done"})
            return

        assistant_response = ""
        trace: list = []
        persisted = False

        async def _persist_assistant_once() -> None:
            # The ``done`` event is the client's refetch signal (#303): the
            # assistant row must be committed before that event goes out, or
            # the refetch races the insert and returns rows without the answer.
            # Idempotent so the finally-block safety net (client disconnect
            # mid-stream) never double-writes.
            nonlocal persisted
            if persisted:
                return
            persisted = True
            # An error turn persists only its compactions (checkpointed before
            # the model call failed, so their marker must survive a reload) and
            # the cleaned cause of an unexpected failure.
            if ERROR_SENTINEL in assistant_response:
                persist_trace = _cap_trace([e for e in trace if _is_pinned(e)])
            else:
                persist_trace = _cap_trace(trace)
            await run_in_threadpool(
                self._persist_assistant_message,
                conversation_id,
                assistant_response,
                persist_trace,
            )

        try:
            # Documents attached to this question (#492): read on this machine
            # through the KB's DocumentReader and injected as delimited blocks
            # ahead of the user's words. Blocking parsing -> threadpool.
            attachments = await run_in_threadpool(resolve_attachments, payload.attachments)
            user_message = self._build_user_message(
                prepend_attachment_block(attachments.block, payload.question), payload.images
            )
            # Read BEFORE the user message is persisted: a new conversation
            # shows its first-reply hint, not the idle-reload notice.
            has_turns = await run_in_threadpool(self.message_repo.has_messages, conversation_id)
            await run_in_threadpool(
                self._persist_user_message,
                conversation_id,
                self._user_display_content(
                    payload.question,
                    payload.images,
                    payload.image_paths or [],
                    attachments.paths,
                ),
            )

            starred = await run_in_threadpool(
                self.message_repo.get_starred_messages, conversation_id
            )
            # Derive the turn's mode (plain / systematic-KB / agentic-KB) from
            # the model's tool-calling capability and build the runner bundle
            # (#84). Retrieval is injected so it runs only in systematic mode and
            # keeps the conversation's degrade-to-no-context policy.
            # How much this conversation lets the model deliberate (1.1.2). The
            # conversation OWNS its level, like its web toggle; the plan is
            # resolved against the ARTIFACT's lever, so the same level reaches
            # a gpt-oss and a Qwen2.5 through different mechanisms. Blocking
            # (a template probe may read the tokenizer) -> threadpool.
            effort_plan = await run_in_threadpool(
                plan_reasoning_effort, llm, conversation.reasoning_effort
            )
            # The turn's instructions: the payload's when it carries them (an
            # explicit "" runs without any), else the conversation's stored
            # ones, so a client that sends none never drops them.
            custom_prompt = (
                payload.custom_prompt
                if payload.custom_prompt is not None
                else conversation.custom_prompt or ""
            )
            plan = await run_in_threadpool(
                plan_turn,
                llm,
                question=payload.question,
                retrieve=lambda: self._retrieve_kb_excerpts(llm, payload.question),
                custom_prompt=custom_prompt,
                starred_messages=starred,
                # #310: the conversation OWNS its web toggle (set at creation from
                # the model's saved defaults or the global default, #609).
                web_search_enabled=bool(conversation.web_search_enabled),
                effort_section=effort_plan.prompt_section,
            )
            params = GenParams(
                temperature=payload.temperature
                if payload.temperature is not None
                else conversation.temperature,
                top_p=payload.top_p if payload.top_p is not None else conversation.top_p,
                # Only the FALLBACK output budget. What the model actually gets
                # is computed per model call from the window it is running in
                # (``src.agents.output_budget``), and that overrides this value
                # whenever the engine reports a window. This one is what an
                # engine that reports none still runs with -- today's behaviour,
                # unchanged. ``payload.max_new_tokens`` is deliberately not read:
                # the control it came from is gone (the field is still accepted
                # so an older client is not rejected).
                max_tokens=conversation.max_tokens or FALLBACK_MAX_TOKENS,
            )
            logger.info(
                f"Turn settings: conversation={conversation_id}, model={llm.id}, "
                f"effort={conversation.reasoning_effort}, temperature={params.temperature}, "
                f"top_p={params.top_p}, threshold={conversation.compaction_threshold}, "
                f"web_search_toggle={bool(conversation.web_search_enabled)}, "
                f"custom_prompt_chars={len(custom_prompt)}"
            )

            # Safety net (#133/#212): unless the model is positively vision-
            # capable, the runner strips images before inference — surface that
            # up front instead of silently dropping the attachment. The notice
            # is persisted with the assistant message (deliberate).
            supports_vision = await run_in_threadpool(detect_supports_vision, llm.link)
            if payload.images and supports_vision is not True:
                assistant_response += IMAGES_IGNORED_NOTICE
                yield _ndjson({"t": "answer", "text": IMAGES_IGNORED_NOTICE})

            # Same contract for documents (#492): a file the reader could not
            # turn into text, or text cut to fit the budget, is named up front
            # rather than dropped in silence. Persisted with the answer.
            attachment_notice = build_attachment_notice(attachments)
            if attachment_notice:
                assistant_response += attachment_notice
                yield _ndjson({"t": "answer", "text": attachment_notice})

            # ``aclosing``: whenever THIS generator is closed, the runner is
            # closed with it, inside the guard. Starlette never closes a body
            # iterator itself: a disconnect that cancels the stream unwinds
            # the chain through the cancellation (the runner then closes
            # LangGraph inside the guard, see ``src.agents.isolated_stream``),
            # but a disconnect noticed while ``send()`` is in progress leaves
            # this generator suspended at its yield, and closing it then still
            # depends on its finalization.
            async with contextlib.aclosing(
                self.runner.astream_text(
                    llm=llm,
                    user_message=user_message,
                    system_prompt=plan.system_prompt,
                    params=params,
                    thread_id=str(conversation_id),
                    summarize=True,
                    kb_context_block=plan.kb_context_block,
                    kb_language_line=plan.kb_language_line,
                    tools=plan.tools,
                    context=plan.context,
                    supports_vision=supports_vision,
                    effort_plan=effort_plan,
                    # #607: the conversation OWNS its compaction threshold
                    # (set at creation from the model's saved defaults or the global
                    # default, #609); a lowered
                    # one compacts on this very turn.
                    compaction_threshold=conversation.compaction_threshold,
                    emit_events=True,
                    announce_idle_reload=has_turns,
                )
            ) as events:
                async for event in events:
                    if event["t"] == "answer":
                        text = event["text"]
                        # Persistence is unchanged: the full answer text
                        # (including a sentinel error string, if any) is
                        # accumulated for the DB.
                        if text.startswith(ERROR_SENTINEL):
                            # Whatever streamed before the failure is kept; the
                            # red line starts a paragraph of its own.
                            assistant_response = _append_failure(assistant_response, text)
                            if event.get("detail"):
                                trace.append(
                                    _error_detail_event(event["detail"], event.get("request_id"))
                                )
                            yield _ndjson(build_stream_error_event(event))
                        else:
                            assistant_response += text
                            yield _ndjson(event)
                    elif event["t"] == "memory_warning":
                        # Wire only, NEVER the persisted trace (1.1.2): the
                        # warning describes the machine's memory NOW -- an old
                        # conversation reopened on a bigger machine must not
                        # replay it as if it still applied.
                        yield _ndjson(event)
                    elif event["t"] == "memory_reload":
                        # Wire only, never persisted, for the same reason: the
                        # memory brake, the idle delay or the user (``reason``)
                        # unloaded the model, and this turn reloads it -- a
                        # fact about this machine at this moment, not part of
                        # the conversation. ``reason`` passes through as is.
                        yield _ndjson(event)
                    elif event["t"] == COMPACTION_EVENT and event.get("phase") == "start":
                        # Wire only (#614): the pending answer says the
                        # conversation is being summarized; ``done`` (below)
                        # is what the trace keeps.
                        yield _ndjson(event)
                    else:
                        # thinking / tool_call / tool_result / compaction done
                        # -> wire AND replay trace.
                        trace.append(event)
                        yield _ndjson(event)
            await _persist_assistant_once()
            yield _ndjson({"t": "done"})
        except Exception as exc:
            logger.exception(f"Query streaming failed for conversation {conversation_id}")
            if ERROR_SENTINEL not in assistant_response:
                # The turn ends red whether or not text had streamed: a partial
                # reply stays above the headline, and the cleaned cause rides
                # the error event and the persisted trace.
                detail, request_id = error_detail(exc), get_request_id()
                assistant_response = _append_failure(assistant_response, ERROR_MESSAGE)
                trace.append(_error_detail_event(detail, request_id))
                yield _ndjson(
                    {
                        "t": "error",
                        "text": ERROR_MESSAGE,
                        "detail": detail,
                        "request_id": request_id,
                    }
                )
            await _persist_assistant_once()
            yield _ndjson({"t": "done"})
        finally:
            duration_ms = (time.perf_counter() - start_s) * 1000
            logger.info(
                f"Query completed for conversation {conversation_id}: "
                f"duration_ms={duration_ms:.0f}, "
                f"response_chars={len(assistant_response)}, "
                f"preview={truncate_for_log(assistant_response, 500)}"
            )
            # Safety net: a client that disconnects mid-stream closes the
            # generator before either ``done`` path ran — persist here so the
            # partial answer is never lost (no-op when done already persisted).
            await _persist_assistant_once()

    async def generate_title_stream(
        self,
        conversation_id: int,
        question: str,
    ) -> AsyncGenerator[str, None]:
        """Stream an auto-generated 2–4 word title (stateless one-shot)."""
        start_s = time.perf_counter()
        logger.info(
            f"Generating title for conversation {conversation_id}: "
            f"{truncate_for_log(question, 2000)}"
        )

        try:
            _conversation, llm = await run_in_threadpool(
                self._load_conversation_and_llm, conversation_id
            )
        except Exception:
            # Recovered with the default title; the traceback says why.
            logger.warning(
                f"Title generation: could not load conversation {conversation_id} or its LLM; "
                f"keeping the default title",
                exc_info=True,
            )
            await run_in_threadpool(self._save_title, conversation_id, "New Conversation")
            return

        if not question or not question.strip():
            await run_in_threadpool(self._save_title, conversation_id, "New Conversation")
            return

        prompt_text = self._build_title_prompt_text(question, llm.type)
        temperature = 0.5 if llm.type == "mistral" else 1.0
        top_p = 0.9 if llm.type == "mistral" else 0.95

        generated_title = ""
        try:
            async with contextlib.aclosing(
                self.runner.astream_oneshot(
                    llm=llm,
                    prompt_text=prompt_text,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=12,
                )
            ) as chunks:
                async for chunk in chunks:
                    generated_title += chunk
                    yield chunk
        finally:
            final_title = _sanitize_title(generated_title) or "New Conversation"
            duration_ms = (time.perf_counter() - start_s) * 1000
            logger.info(
                f"Title generation completed for conversation {conversation_id}: "
                f"duration_ms={duration_ms:.0f}, title={final_title!r}"
            )
            await run_in_threadpool(self._save_title, conversation_id, final_title)

    # ===================== Sync DB helpers (run in threadpool) =====================
    def _load_conversation_and_llm(self, conversation_id: int):
        """Load the conversation + its LLM, auto-repairing a stale ``llm_id``."""
        conversation = self.conversation_repo.get_conversation_by_id(conversation_id)
        try:
            llm = self.conversation_repo.get_llm_by_id(conversation.llm_id)
        except ModelNotFoundException:
            logger.warning(
                f"LLM id={conversation.llm_id} not found for conversation "
                f"{conversation_id}, attempting auto-repair"
            )
            local_llm = self.db.query(Llm).filter(Llm.local == 1).first()
            if local_llm is None:
                raise ModelNotFoundException(
                    f"LLM {conversation.llm_id} not found and no local models available"
                )
            logger.info(
                f"Auto-repairing conversation {conversation_id}: "
                f"llm_id {conversation.llm_id} -> {local_llm.id} ({local_llm.name})"
            )
            conversation.llm_id = local_llm.id
            self.db.commit()
            llm = local_llm
        return conversation, llm

    @staticmethod
    def _build_user_message(question: str, images):
        """Multimodal content (text + ``image_url`` parts) when images are
        attached, else the plain question string. The base64 data-URLs ride the
        live turn only; ``_StripStaleImagesMiddleware`` drops them on follow-ups."""
        if not images:
            return question
        return [
            {"type": "text", "text": question},
            *[{"type": "image_url", "image_url": {"url": url}} for url in images],
        ]

    @staticmethod
    def _encode_marker_path(path: str) -> str:
        """Percent-encode what would break a ``[...]`` marker.

        The frontend parses these markers with a bracket-delimited regex, so a
        path holding ``]`` (perfectly legal on every platform) would end its own
        marker and leak the rest into the readable text. ``%`` is encoded FIRST
        so decoding is unambiguous: a literal ``%5D`` in a path survives the
        round trip as ``%255D``.
        """
        return path.replace("%", "%25").replace("]", "%5D")

    @staticmethod
    def _user_display_content(
        question: str, images, image_paths=None, attachment_paths=None
    ) -> str:
        """Short text persisted in the Message table: the question plus one
        marker per attachment. When a local filesystem path is known an image is
        stored as ``[image_path:/abs/path]`` so the frontend can reload the file
        on revisit. Falls back to ``[image]`` for clipboard/unknown-origin
        images. Documents are stored as ``[file_path:/abs/path]`` (#492): the
        extracted text rides the live turn only, so a reloaded conversation
        shows WHAT was attached without carrying the whole document forever.

        Both marker paths are percent-encoded by ``_encode_marker_path``; the
        frontend decodes them in ``messageContent.js``."""
        markers = []
        encode = ConversationService._encode_marker_path
        paths = list(image_paths or [])
        for i in range(len(images or [])):
            p = paths[i] if i < len(paths) else ""
            markers.append(f"[image_path:{encode(p)}]" if p else "[image]")
        for p in attachment_paths or []:
            markers.append(f"[file_path:{encode(p)}]")
        if not markers:
            return question
        marker_str = " ".join(markers)
        return f"{question} {marker_str}".strip() if question.strip() else marker_str

    def _persist_user_message(self, conversation_id: int, content: str) -> None:
        self.message_repo.create_message(
            conversation_id=conversation_id, sender="user", content=content
        )
        self.conversation_repo.update_last_message_time(conversation_id)
        self.db.commit()

    def _persist_assistant_message(
        self, conversation_id: int, content: str, trace: Optional[list] = None
    ) -> None:
        self.message_repo.create_message(
            conversation_id=conversation_id,
            sender="llm",
            content=content.strip(),
            trace=trace,
        )
        self.conversation_repo.update_last_message_time(conversation_id)
        self.db.commit()

    def _save_title(self, conversation_id: int, title: str) -> None:
        conversation = self.conversation_repo.get_conversation_by_id(conversation_id)
        conversation.name = title
        self.db.commit()

    def _build_title_prompt_text(self, question: str, model_type: str) -> str:
        """Single merged user-message prompt for title generation (2–4 words)."""
        question = question[:TITLE_QUESTION_MAX_CHARS]
        if model_type == "mistral":
            return (
                "You are a TITLE generator. Produce ONLY a very short title "
                "(2–4 words maximum).\n"
                "Rules: only the title text; Title Case; no question mark; "
                "no quotes; no emojis; no hashtags; no code; no trailing "
                "punctuation; never answer the question; if empty/URL/noise => "
                "output nothing.\n"
                "Write the title in the same language as the user's message "
                "(a French message gets a French title).\n"
                f"User message: {question}\n"
                "Do not answer the question, only create a relevant title. "
                "Do NOT add quotes around the title.\n"
                "Examples (user question -> title):\n"
                "give me pizza recipe -> Pizza Recipe\n"
                "google founding team members -> Google Founding Team\n"
                "what's the capital of japan -> Japan Capital\n"
                "female of the pig -> Pig Female Name\n"
                "combien de pommes me reste-t-il -> Compte De Pommes\n"
            )
        system_prompt = (
            "You are a very-short-title generator. Return ONLY a concise "
            "title. No punctuation (except apostrophes in possessives), "
            "no quotes, no hashtags, no emojis, no trailing filler words. "
            "Capitalize important words. The title shouldn't be a question.\n"
            "Do not answer the question, only create a relevant title.\n"
            "Write the title in the same language as the user's message "
            "(a French message gets a French title).\n"
            "If the message is empty or meaningless, return nothing.\n"
            "Examples (user question -> title):\n"
            "give me pizza recipe -> Pizza Recipe\n"
            "google founding team members -> Google Founding Team\n"
            "what's the capital of japan -> Japan Capital\n"
            "female of the pig -> Pig Female Name\n"
            "combien de pommes me reste-t-il -> Compte De Pommes\n"
            "Format: just the title, nothing else."
        )
        user_prompt = f"Create a 2-to-4-word title for:\n{question}"
        return f"{system_prompt}\n\n{user_prompt}"

    # ===================== Manual compaction and context usage =====================
    async def compact_conversation(self, conversation_id: int) -> dict:
        """Summarize the conversation's history now (``/compact``, #585).

        Returns ``{"summarized", "reason", "summary"}``, or ``{"summarized":
        0}`` when there is nothing to compact (a single exchange, or nothing
        new since the last manual compaction: nothing is loaded or stored).
        Refused with a 409 when the conversation has no model (it is never
        re-assigned one) or, read under the generation guard, ends with a
        question that has no answer (an answer of any kind, a failed one
        included, is accepted). The runner compacts (``AgentRunner.acompact``:
        busy -> 409, a summary or a thread write that fails ->
        ``CompactionFailedException``, the conversation deleted meanwhile ->
        ``ConversationNotFoundException``); the separator -- the compaction's
        ``done`` event, ``manual: True`` -- is appended to the trace of the
        last answer read under the guard, once the checkpoint is rewritten. A
        separator that cannot be saved then raises
        ``CompactionSeparatorException`` (the history stays summarized).
        """
        conversation, llm = await run_in_threadpool(
            self._load_conversation_and_llm_read_only, conversation_id
        )
        event = await self.runner.acompact(
            llm=llm,
            thread_id=str(conversation_id),
            params=GenParams(
                temperature=conversation.temperature,
                top_p=conversation.top_p,
                max_tokens=conversation.max_tokens or FALLBACK_MAX_TOKENS,
            ),
            compaction_threshold=conversation.compaction_threshold,
            store=_ConversationCompactionStore(self, conversation_id),
        )
        return {key: event[key] for key in ("summarized", "reason", "summary") if key in event}

    async def context_usage(self, conversation_id: int) -> dict:
        """The conversation's context usage for the panel (``/context``, #608).

        The NEXT request is planned the way a turn plans it -- the system
        prompt with the conversation's own instructions (its column), its
        starred messages and its reasoning-effort section, the tools its
        mode carries -- without retrieving anything (``plan_turn``'s
        ``context_count``): an assistant that searches its knowledge base on
        every question is counted in that mode with no excerpts, which
        ``kb_excerpts_excluded`` says. The runner counts it
        (``AgentRunner.acontext_usage``). Read only: the conversation is
        never re-assigned a model (409 asking to pick one first), nothing is
        committed, the model is never loaded.
        """
        conversation, llm = await run_in_threadpool(
            self._load_conversation_and_llm_read_only, conversation_id
        )
        starred = await run_in_threadpool(self.message_repo.get_starred_messages, conversation_id)
        effort_plan = await run_in_threadpool(
            plan_reasoning_effort, llm, conversation.reasoning_effort
        )
        plan = await run_in_threadpool(
            plan_turn,
            llm,
            question="",
            # Never called: the count retrieves nothing.
            retrieve=list,
            custom_prompt=conversation.custom_prompt,
            starred_messages=starred,
            web_search_enabled=bool(conversation.web_search_enabled),
            effort_section=effort_plan.prompt_section,
            context_count=True,
        )
        usage = await self.runner.acontext_usage(
            llm=llm,
            thread_id=str(conversation_id),
            system_prompt=plan.system_prompt,
            tools=plan.tools,
            compaction_threshold=conversation.compaction_threshold,
        )
        return {**usage, "kb_excerpts_excluded": plan.systematic_kb}

    def _load_conversation_and_llm_read_only(self, conversation_id: int):
        """The conversation and its model, read only: never re-assigned, never
        committed (unlike ``_load_conversation_and_llm``). A conversation
        whose model was deleted is a 409 asking to pick a model first."""
        conversation = self.conversation_repo.get_conversation_by_id(conversation_id)
        if conversation.llm_id is not None:
            try:
                return conversation, self.conversation_repo.get_llm_by_id(conversation.llm_id)
            except ModelNotFoundException:
                pass  # the 409 below says what to do about it
        raise StateConflictException(
            "This conversation has no model; pick a model first.",
            status_code=409,
            detail={"reason": "no_model"},
        )

    def _append_to_answer_trace(self, answer_id: Optional[int], event: dict) -> None:
        """Append ``event`` to the trace of the answer ``answer_id`` and
        commit. Not re-capped (``_cap_trace``): the event is small, and a
        compaction event is pinned by the cap anyway; a full trace may exceed
        ``TRACE_MAX_BYTES`` by it."""
        try:
            answer = self.db.get(Message, answer_id) if answer_id is not None else None
            if answer is None:
                raise DatabaseException(f"No answer {answer_id} to hold the separator")
            answer.trace = [*(answer.trace or []), event]
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def _conversation_exists(self, conversation_id: int) -> bool:
        return (
            self.db.query(Conversation.id).filter(Conversation.id == conversation_id).first()
            is not None
        )


class _ConversationCompactionStore:
    """The messages table, as a manual compaction needs it
    (``src.agents.runner.CompactionStore``). The conversation's last row is
    read under the generation guard (``refuse_if_open``); the separator goes
    on the answer read then."""

    def __init__(self, service: ConversationService, conversation_id: int):
        self._service = service
        self._conversation_id = conversation_id
        self._answer_id: Optional[int] = None

    async def refuse_if_open(self) -> None:
        last = await run_in_threadpool(
            self._service.message_repo.get_last_message, self._conversation_id
        )
        if last is not None and last.sender == "user":
            raise StateConflictException(
                "The last question has no answer yet; it must be answered before the "
                "conversation can be summarized.",
                status_code=409,
                detail={"reason": "unanswered_question"},
            )
        self._answer_id = last.id if last is not None else None

    async def still_exists(self) -> bool:
        return await run_in_threadpool(self._service._conversation_exists, self._conversation_id)

    async def save_separator(self, event: dict) -> None:
        try:
            await run_in_threadpool(self._service._append_to_answer_trace, self._answer_id, event)
        except Exception as exc:
            raise CompactionSeparatorException() from exc
