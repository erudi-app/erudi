"""Business logic for stateless arena LLM queries with KB-aware prompting.

The arena is "conversation minus state": a single-model, no-history streaming
query. Since the LangChain refactor it shares the conversation ``AgentRunner``
but runs it statelessly (``thread_id=None``, no checkpointer, no summarization).

Pipeline:
1. Validate the question and fetch the LLM.
2. Pick a prompting strategy (param_size-based) and, if the model has a KB
   attached, retrieve top-k chunks (hybrid pgvector search via ``kb_utils``).
3. Build a real system prompt (size-adaptive + custom + KB context) and stream
   the answer as raw token text via the shared agent runner.

The A/B "duel" is orchestrated entirely on the frontend (N parallel calls to
this endpoint); the runner's engine guard serializes them on the single-model
engine, so model swaps don't thrash the subprocess.
"""

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
from src.agents.runner import AgentRunner, GenParams, IMAGES_IGNORED_NOTICE
from src.database.generation_hints import resolve_sampling_defaults
from src.domains.arena.repository import ArenaRepository
from src.domains.llms.repository import detect_supports_vision
from src.domains.user_settings.repository import User_Settings_Repository
from src.domains.arena.schemas import ArenaQueryPayload
from src.entities.Llm import Llm
from src.core.exceptions import (
    KnowledgeBaseNotFoundException,
    KnowledgeBaseCorruptedException,
    InvalidInputException,
)


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
        """Query an LLM in stateless arena mode and stream response tokens (raw text).

        Raises:
            InvalidInputException: empty question.
            ModelNotFoundException: ``llm_id`` not found (via ``_get_llm``; the
                endpoint also validates eagerly before opening the stream).
            KnowledgeBase*Exception: KB retrieval failure.

        Generation/model-load failures are NOT raised here — the runner yields the
        ``[ERROR_MESSAGE_SYSTEM]`` sentinel inline (unified with conversation).
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
        web_search_enabled = await run_in_threadpool(
            User_Settings_Repository(self.db).get_web_search_enabled
        )
        plan = await run_in_threadpool(
            plan_turn,
            llm,
            question=payload.question,
            retrieve=lambda: self._retrieve_kb_excerpts(llm, payload.question, strategy),
            custom_prompt=payload.custom_prompt,
            web_search_enabled=web_search_enabled,
        )

        # Omitted values resolve to the MODEL's defaults (#388); explicit wins.
        defaults = resolve_sampling_defaults(llm)
        params = GenParams(
            temperature=defaults.temperature
            if payload.temperature is None
            else payload.temperature,
            top_p=defaults.top_p if payload.top_p is None else payload.top_p,
            max_tokens=defaults.max_tokens
            if payload.max_new_tokens is None
            else payload.max_new_tokens,
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
            yield IMAGES_IGNORED_NOTICE
        attachment_notice = build_attachment_notice(attachments)
        if attachment_notice:
            response += attachment_notice
            yield attachment_notice
        async for token in self.runner.astream_text(
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
        ):
            response += token
            yield token

        duration_ms = (time.perf_counter() - start_s) * 1000
        logger.info(
            f"Arena query completed for LLM {llm_id} "
            f"(model={getattr(llm, 'name', '?')}): duration_ms={duration_ms:.0f}, "
            f"response_chars={len(response)}, "
            f"preview={truncate_for_log(response, 500)}"
        )
