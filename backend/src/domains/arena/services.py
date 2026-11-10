"""Business logic for stateless arena LLM queries with KB-aware prompting.

The arena is "conversation minus state": a single-model, no-history streaming
query. Since the LangChain refactor it shares the conversation ``AgentRunner``
but runs it statelessly (``thread_id=None``, no checkpointer, no summarization).

Pipeline:
1. Validate the question and fetch the LLM.
2. Pick a prompting strategy (param_size-based) and, if the model has a KB
   attached, retrieve top-k chunks (hybrid pgvector search via ``kb_utils``).
3. Build a real system prompt (size-adaptive + custom + KB context) and stream
   the answer through the shared agent runner as NDJSON: ``answer`` events
   (text) and, for a failed turn, one ``error`` event shaped like a
   conversation's (``build_stream_error_event``: the red line, and a curated
   turn's ``curated_error`` the panel translates it from). Reasoning, tool and
   memory events are not sent: a panel shows the answer only.

The A/B "duel" is orchestrated entirely on the frontend (N parallel calls to
this endpoint); the runner's engine guard serializes them on the single-model
engine, so model swaps don't thrash the subprocess.
"""

import contextlib
import json
import time
from typing import AsyncGenerator, Dict, Any

from fastapi.concurrency import run_in_threadpool

from src.core.logging import logger
from src.core.logutils import truncate_for_log
from src.utils.prompt_utils import get_prompting_strategy
from src.utils.attachment_utils import (
    build_attachment_notice,
    prepend_attachment_block,
    resolve_attachments,
)
from src.utils.kb_utils import KbExcerpt, retrieve_kb_excerpts
from src.agents.kb_mode import plan_turn
from src.agents.reasoning_effort import plan_reasoning_effort
from src.agents.runner import AgentRunner, ERROR_SENTINEL, GenParams, IMAGES_IGNORED_NOTICE
from src.database.generation_hints import resolve_sampling_defaults
from src.domains.arena.repository import ArenaRepository
from src.domains.llms.conversation_defaults import read_conversation_defaults
from src.domains.llms.repository import detect_supports_vision
from src.domains.user_settings.repository import User_Settings_Repository
from src.domains.arena.schemas import ArenaQueryPayload
from src.domains.conversations.services import build_stream_error_event
from src.entities.Llm import Llm
from src.core.exceptions import (
    KnowledgeBaseNotFoundException,
    KnowledgeBaseCorruptedException,
    InvalidInputException,
)


def _line(event: dict) -> str:
    """One NDJSON line (non-ASCII ``\\uXXXX``-escaped, like a conversation's)."""
    return json.dumps(event) + "\n"


class ArenaService:
    """Service layer for arena stateless query processing."""

    def __init__(self, db):
        """Initialize arena service with database session.

        Args:
            db: SQLAlchemy database session for repository access.
        """
        logger.debug("Initializing ArenaService")
        self.arena_repo = ArenaRepository(db)
        self.runner = AgentRunner(checkpointer=None)  # stateless: no history persisted
        self.db = db

    def _get_llm(self, llm_id: int) -> Llm:
        """Retrieve LLM entity by ID via repository (raises 404 if missing)."""
        return self.arena_repo.get_llm_by_id(llm_id)

    @staticmethod
    def _build_user_message(question: str, images):
        """Multimodal content (text + ``image_url`` parts) when images are
        attached, else the plain question string (mirrors conversations; the
        arena is stateless so images live for this turn only)."""
        if not images:
            return question
        return [
            {"type": "text", "text": question},
            *[{"type": "image_url", "image_url": {"url": url}} for url in images],
        ]

    def _retrieve_kb_excerpts(
        self,
        llm: Llm,
        query: str,
        strategy: Dict[str, Any],
    ) -> list[KbExcerpt]:
        """Adaptive KB excerpts if the LLM has a KB attached and strategy allows.

        Adaptive selection through the hybrid pgvector search (``kb_utils``
        façade). Returns an empty list when KB is unavailable/disabled.
        """
        if not llm.is_attached_to_kb or not strategy.get("use_kb_context", False):
            return []

        try:
            return retrieve_kb_excerpts(query, llm.kb_id, token_budget=strategy["kb_token_budget"])
        except (KnowledgeBaseNotFoundException, KnowledgeBaseCorruptedException):
            raise
        except Exception as e:
            logger.exception("Failed to retrieve Knowledge Base context")
            raise KnowledgeBaseCorruptedException(
                llm.kb_id,
                f"Knowledge Base retrieval error: {e}",
                trace=str(e),
            )

    async def query_llm_stream(
        self,
        llm_id: int,
        payload: ArenaQueryPayload,
    ) -> AsyncGenerator[str, None]:
        """Query an LLM in stateless arena mode and stream its answer as NDJSON lines.

        Raises:
            InvalidInputException: empty question.
            ModelNotFoundException: ``llm_id`` not found (via ``_get_llm``; the
                endpoint also validates eagerly before opening the stream).
            KnowledgeBase*Exception: KB retrieval failure.

        Generation/model-load failures are NOT raised here — the runner yields the
        ``[ERROR_MESSAGE_SYSTEM]`` sentinel inline (unified with conversation),
        sent as an ``error`` event.
        """
        if not payload.question.strip() and not payload.images and not payload.attachments:
            raise InvalidInputException("question")

        start_s = time.perf_counter()
        # Local-first policy: log the question content (bounded), never image
        # bytes — only their count and encoded size (mirrors conversations).
        image_note = ""
        if payload.images:
            total_b64_chars = sum(len(url) for url in payload.images)
            image_note = f", images={len(payload.images)} ({total_b64_chars} base64 chars)"
        if payload.attachments:
            image_note += f", attachments={len(payload.attachments)}"
        logger.info(
            f"Arena query started for LLM {llm_id}{image_note}: "
            f"{truncate_for_log(payload.question, 2000)}"
        )
        llm = self._get_llm(llm_id)

        param_size = llm.param_size if getattr(llm, "param_size", None) else 2
        strategy = get_prompting_strategy(param_size)

        # Derive the turn's mode (plain / systematic-KB / agentic-KB) from the
        # model's tool-calling capability (#84). Retrieval is injected so it runs
        # only in systematic mode and keeps arena's raise-on-failure policy.
        # #310: arena panels have no conversation row, so the web toggle follows
        # the GLOBAL user setting directly (kept deliberately simple).
        settings_repo = User_Settings_Repository(self.db)
        web_search_enabled = await run_in_threadpool(settings_repo.get_web_search_enabled)
        # #609: effort, temperature and top_p a panel leaves out (None) follow
        # the model like a new conversation does: the values saved for it with
        # ``/config global``, else the Settings effort and the model's
        # recommendation (#388), else a constant. Explicit values win.
        resolved = await run_in_threadpool(read_conversation_defaults, self.db, llm)
        effort_level = (
            resolved["reasoning_effort"].value
            if payload.reasoning_effort is None
            else payload.reasoning_effort
        )
        effort_plan = await run_in_threadpool(plan_reasoning_effort, llm, effort_level)
        plan = await run_in_threadpool(
            plan_turn,
            llm,
            question=payload.question,
            retrieve=lambda: self._retrieve_kb_excerpts(llm, payload.question, strategy),
            custom_prompt=payload.custom_prompt,
            web_search_enabled=web_search_enabled,
            effort_section=effort_plan.prompt_section,
        )

        params = GenParams(
            temperature=resolved["temperature"].value
            if payload.temperature is None
            else payload.temperature,
            top_p=resolved["top_p"].value if payload.top_p is None else payload.top_p,
            # Only the FALLBACK output budget (same contract as a conversation):
            # the per-call budget computed from the running window
            # (``src.agents.output_budget``) overrides it whenever the engine
            # reports one, and this value is what an engine that reports none
            # runs with. ``payload.max_new_tokens`` is deliberately not read --
            # the arena panel no longer offers the control, and the field stays
            # accepted only so an older client is not rejected.
            max_tokens=resolve_sampling_defaults(llm).max_tokens,
        )

        # Safety net (#133/#212): unless the model is positively vision-capable,
        # the runner strips images before inference — tell the user up front
        # instead of silently dropping the attachment.
        supports_vision = await run_in_threadpool(detect_supports_vision, llm.link)

        # Documents attached to this question (#492): the same resolver the
        # conversation uses, so both paths read files exactly once, the same way.
        attachments = await run_in_threadpool(resolve_attachments, payload.attachments)

        response = ""
        if payload.images and supports_vision is not True:
            response += IMAGES_IGNORED_NOTICE
            yield _line({"t": "answer", "text": IMAGES_IGNORED_NOTICE})
        attachment_notice = build_attachment_notice(attachments)
        if attachment_notice:
            response += attachment_notice
            yield _line({"t": "answer", "text": attachment_notice})
        # ``aclosing``: whenever THIS generator is closed, the runner is closed
        # with it, inside the guard. Starlette never closes a body iterator
        # itself: a disconnect noticed while ``send()`` is in progress leaves
        # this generator suspended at its yield, and closing it then still
        # depends on its finalization.
        async with contextlib.aclosing(
            self.runner.astream_text(
                llm=llm,
                user_message=self._build_user_message(
                    prepend_attachment_block(attachments.block, payload.question), payload.images
                ),
                system_prompt=plan.system_prompt,
                params=params,
                thread_id=None,
                summarize=False,
                kb_context_block=plan.kb_context_block,
                kb_language_line=plan.kb_language_line,
                tools=plan.tools,
                context=plan.context,
                supports_vision=supports_vision,
                effort_plan=effort_plan,
            )
        ) as events:
            async for event in events:
                if event["t"] != "answer":
                    continue
                response += event["text"]
                if event["text"].startswith(ERROR_SENTINEL):
                    yield _line(build_stream_error_event(event))
                else:
                    yield _line(event)

        duration_ms = (time.perf_counter() - start_s) * 1000
        logger.info(
            f"Arena query completed for LLM {llm_id} "
            f"(model={getattr(llm, 'name', '?')}): duration_ms={duration_ms:.0f}, "
            f"response_chars={len(response)}, "
            f"preview={truncate_for_log(response, 500)}"
        )
