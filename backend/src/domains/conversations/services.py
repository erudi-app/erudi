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

import json
import re
import time
from typing import AsyncGenerator, Optional

from fastapi.concurrency import run_in_threadpool

from src.core.logging import logger
from src.core.logutils import truncate_for_log
from src.agents.kb_mode import plan_turn
from src.agents.runner import (
    AgentRunner,
    GenParams,
    ERROR_MESSAGE,
    ERROR_SENTINEL,
    IMAGES_IGNORED_NOTICE,
)
from src.database.generation_hints import resolve_sampling_defaults
from src.domains.conversations.repository import ConversationRepository, MessageRepository
from src.domains.llms.repository import detect_supports_vision
from src.domains.user_settings.repository import User_Settings_Repository
from src.domains.conversations.schemas import ConversationQuery
from src.entities.Conversation import Conversation
from src.entities.Llm import Llm
from src.core.exceptions import ModelNotFoundException
from src.utils.kb_utils import KbExcerpt, retrieve_kb_excerpts
from src.utils.prompt_utils import get_prompting_strategy


# Serialized-trace cap (#90): persisted traces are bounded so a runaway
# thinking/tool turn can't bloat the messages row. Drop-oldest past the cap and
# prepend a ``{"t":"truncated"}`` marker so the UI can show reasoning was elided.
TRACE_MAX_BYTES = 32 * 1024


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
    for key in ("code", "raw"):
        value = answer_event.get(key)
        if value:
            error_event[key] = value
    return error_event


def _cap_trace(events: list) -> Optional[list]:
    """Cap the serialized trace at ``TRACE_MAX_BYTES`` (drop-oldest).

    Returns ``None`` for an empty trace (nothing to persist). When the trace fits,
    it is returned unchanged. Otherwise the oldest events are dropped -- with the
    ``{"t":"truncated"}`` marker counted in the budget -- until the marker plus the
    remaining events fit, always keeping at least the newest event.
    """
    if not events:
        return None
    if len(json.dumps(events)) <= TRACE_MAX_BYTES:
        return list(events)
    marker = {"t": "truncated"}
    capped = list(events)
    while len(capped) > 1 and len(json.dumps([marker, *capped])) > TRACE_MAX_BYTES:
        capped.pop(0)
    return [marker, *capped]


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
        custom_prompt: str = "",
        web_search_enabled: Optional[bool] = None,
    ) -> Conversation:
        """Create a new conversation with the given LLM and generation params.

        ``temperature`` / ``top_p`` / ``max_tokens`` left ``None`` resolve to the
        MODEL's defaults (#388: the captured generation_config, else the neutral
        constants, see ``src.database.generation_hints``); an explicit value
        wins. Mirrors
        ``web_search_enabled=None``, which copies the GLOBAL user-settings
        default at creation (#310). The conversation owns its values afterwards
        -- later model/global changes never retro-affect it.
        """
        logger.info(f"Creating new conversation with LLM {llm_id}")
        if temperature is None or top_p is None or max_tokens is None:
            defaults = resolve_sampling_defaults(self.conversation_repo.get_llm_by_id(llm_id))
            temperature = defaults.temperature if temperature is None else temperature
            top_p = defaults.top_p if top_p is None else top_p
            max_tokens = defaults.max_tokens if max_tokens is None else max_tokens
        if web_search_enabled is None:
            web_search_enabled = User_Settings_Repository(self.db).get_web_search_enabled()
        return self.conversation_repo.create_conversation(
            llm_id=llm_id,
            name="New Conversation",
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            custom_prompt=custom_prompt,
            web_search_enabled=web_search_enabled,
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
    ) -> Conversation:
        """Partial update of conversation metadata (only non-None fields)."""
        logger.info(f"Updating conversation {conversation_id}")
        return self.conversation_repo.update_conversation(
            conversation_id=conversation_id,
            name=name,
            llm_id=llm_id,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            custom_prompt=custom_prompt,
            web_search_enabled=web_search_enabled,
        )

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
        ``error`` (the mapped ERROR sentinel), then a terminal ``done`` (#90).

        The answer text is accumulated exactly as before and persisted as the
        assistant message ``content``; non-answer events are collected into an
        ordered ``trace`` (capped, drop-oldest) and persisted alongside so the
        panel can replay on reload. Persists the user message up front (so it
        shows immediately); the agent restores prior history from the checkpointer.

        Error mapping (#252/#225-D4): the runner yields its curated error as an
        ``answer`` event carrying the ERROR sentinel. Here that answer is (a) still
        appended to ``assistant_response`` so the persisted content is byte-for-byte
        the pre-#90 error turn, and (b) re-framed on the wire as an ``error`` event
        (the frontend renders the red bubble from ``error`` as it did from the
        sentinel substring). Error turns persist no trace.
        """
        start_s = time.perf_counter()
        # Local-first policy: log the question content (bounded), never image
        # bytes — only their count and encoded size.
        image_note = ""
        if payload.images:
            total_b64_chars = sum(len(url) for url in payload.images)
            image_note = f", images={len(payload.images)} ({total_b64_chars} base64 chars)"
        logger.info(
            f"Processing query for conversation {conversation_id}{image_note}: "
            f"{truncate_for_log(payload.question, 2000)}"
        )

        try:
            conversation, llm = await run_in_threadpool(
                self._load_conversation_and_llm, conversation_id
            )
        except Exception:
            # The response has started (an NDJSON stream), so no exception
            # handler will see this: the record is written here.
            logger.exception(f"Failed to load conversation {conversation_id} or its LLM for query")
            # No conversation loaded -> nothing to persist; just surface the error.
            yield _ndjson({"t": "error", "text": ERROR_MESSAGE})
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
            persist_trace = None
            if trace and ERROR_SENTINEL not in assistant_response:
                persist_trace = _cap_trace(trace)
            await run_in_threadpool(
                self._persist_assistant_message,
                conversation_id,
                assistant_response,
                persist_trace,
            )

        try:
            user_message = self._build_user_message(payload.question, payload.images)
            await run_in_threadpool(
                self._persist_user_message,
                conversation_id,
                self._user_display_content(
                    payload.question, payload.images, payload.image_paths or []
                ),
            )

            starred = await run_in_threadpool(
                self.message_repo.get_starred_messages, conversation_id
            )
            # Derive the turn's mode (plain / systematic-KB / agentic-KB) from
            # the model's tool-calling capability and build the runner bundle
            # (#84). Retrieval is injected so it runs only in systematic mode and
            # keeps the conversation's degrade-to-no-context policy.
            plan = await run_in_threadpool(
                plan_turn,
                llm,
                question=payload.question,
                retrieve=lambda: self._retrieve_kb_excerpts(llm, payload.question),
                custom_prompt=payload.custom_prompt,
                starred_messages=starred,
                # #310: the conversation OWNS its web toggle (copied from the
                # global default at creation).
                web_search_enabled=bool(conversation.web_search_enabled),
            )
            params = GenParams(
                temperature=payload.temperature
                if payload.temperature is not None
                else conversation.temperature,
                top_p=payload.top_p if payload.top_p is not None else conversation.top_p,
                max_tokens=payload.max_new_tokens or conversation.max_tokens or 1024,
            )

            # Safety net (#133/#212): unless the model is positively vision-
            # capable, the runner strips images before inference — surface that
            # up front instead of silently dropping the attachment. The notice
            # is persisted with the assistant message (deliberate).
            supports_vision = await run_in_threadpool(detect_supports_vision, llm.link)
            if payload.images and supports_vision is not True:
                assistant_response += IMAGES_IGNORED_NOTICE
                yield _ndjson({"t": "answer", "text": IMAGES_IGNORED_NOTICE})

            async for event in self.runner.astream_text(
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
                emit_events=True,
            ):
                if event["t"] == "answer":
                    text = event["text"]
                    # Persistence is unchanged: the full answer text (including a
                    # sentinel error string, if any) is accumulated for the DB.
                    assistant_response += text
                    if text.startswith(ERROR_SENTINEL):
                        yield _ndjson(build_stream_error_event(event))
                    else:
                        yield _ndjson(event)
                else:
                    # thinking / tool_call / tool_result -> wire AND replay trace.
                    trace.append(event)
                    yield _ndjson(event)
            await _persist_assistant_once()
            yield _ndjson({"t": "done"})
        except Exception:
            logger.exception(f"Query streaming failed for conversation {conversation_id}")
            if not assistant_response:
                assistant_response = ERROR_MESSAGE
                yield _ndjson({"t": "error", "text": ERROR_MESSAGE})
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
            async for chunk in self.runner.astream_oneshot(
                llm=llm,
                prompt_text=prompt_text,
                temperature=temperature,
                top_p=top_p,
                max_tokens=12,
            ):
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
    def _user_display_content(question: str, images, image_paths=None) -> str:
        """Short text persisted in the Message table: the question plus one
        marker per attachment. When a local filesystem path is known it is stored
        as ``[image_path:/abs/path]`` so the frontend can reload the file on
        revisit. Falls back to ``[image]`` for clipboard/unknown-origin images."""
        if not images:
            return question
        paths = list(image_paths or [])
        markers = []
        for i in range(len(images)):
            p = paths[i] if i < len(paths) else ""
            markers.append(f"[image_path:{p}]" if p else "[image]")
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
