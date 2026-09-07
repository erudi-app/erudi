"""Comprehensive tests for conversations domain (chat management).

Tests cover:
- Repository layer (conversations and messages CRUD)
- Service layer (chat logic, streaming, title generation with mocked engine)
- Endpoint layer (REST API with streaming responses)

All LLM_Engine operations are mocked for fast, isolated testing.
"""

import json

import pytest
from unittest.mock import patch
from fastapi import status

from tests._helpers import ToolableFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

import src.agents.runner as agent_runner
from src.core import config
from src.engines.base_engine import BaseEngine
from src.domains.conversations.repository import ConversationRepository, MessageRepository
from src.domains.conversations.services import (
    ConversationService,
    _sanitize_title,
    _cap_trace,
    _ndjson,
    TRACE_MAX_BYTES,
)
from src.utils.kb_utils import KbExcerpt
from src.domains.conversations.schemas import ConversationQuery

# ERROR sentinel (hardcoded on purpose: the tests pin the wire/DB contract, not
# the constant's name).
_SENTINEL = "[ERROR_MESSAGE_SYSTEM]"


def _parse_ndjson(body):
    """Parse an NDJSON stream body (a str or a list of yielded chunks) into a
    list of event dicts. Each non-empty line must be a complete JSON object."""
    if isinstance(body, (list, tuple)):
        body = "".join(body)
    return [json.loads(line) for line in body.splitlines() if line.strip()]


def _answer_text(body):
    """Concatenate the ``answer`` event text from an NDJSON stream body."""
    return "".join(e["text"] for e in _parse_ndjson(body) if e.get("t") == "answer")


class _FakeEngine(BaseEngine):
    """Engine stub exposing generation_guard without spawning a real model.

    Inherits the ``BaseEngine.model_supports_vision`` default, so
    ``detect_supports_vision`` resolves to ``None`` (unknown capability).
    """


# Exact user-facing line prepended when the current turn carries images but the
# model is not positively vision-capable (#212). Hardcoded on purpose: the test
# pins the wire/persistence contract, not the constant's name.
_VISION_NOTICE = "*This model doesn't support images — your image was ignored.*\n\n"


def _fake_chat_model(*texts):
    """Return a ``build_chat_model`` replacement yielding a scripted fake model.

    ``GenericFakeChatModel`` streams the content word-by-word, so assertions
    compare the concatenated stream (the text/plain wire contract), not the
    original token list.
    """
    msgs = [AIMessage(content=t) for t in texts]
    return lambda llm, **kw: ToolableFakeChatModel(messages=iter(msgs))


# ============ Repository Tests ============


class TestConversationRepository:
    """Test suite for ConversationRepository database operations."""

    def test_create_conversation(self, test_db_session, mock_llm):
        """Test conversation creation in database.

        Args:
            test_db_session: Database session fixture.
            mock_llm: LLM entity fixture.
        """
        repo = ConversationRepository(test_db_session)

        conversation = repo.create_conversation(
            llm_id=mock_llm.id, name="Test Chat", temperature=0.7, top_p=0.9, max_tokens=1024
        )

        assert conversation.id is not None
        assert conversation.name == "Test Chat"
        assert conversation.llm_id == mock_llm.id
        assert conversation.temperature == 0.7
        assert conversation.top_p == 0.9
        assert conversation.max_tokens == 1024

    def test_get_all_conversations(self, test_db_session, mock_llm):
        """Test retrieving all conversations.

        Args:
            test_db_session: Database session fixture.
            mock_llm: LLM entity fixture.
        """
        repo = ConversationRepository(test_db_session)

        # Create multiple conversations
        repo.create_conversation(
            llm_id=mock_llm.id, name="Chat 1", temperature=0.5, top_p=0.8, max_tokens=512
        )
        repo.create_conversation(
            llm_id=mock_llm.id, name="Chat 2", temperature=0.7, top_p=0.9, max_tokens=1024
        )

        conversations = repo.get_all_conversations()

        assert len(conversations) == 2
        assert conversations[0].name == "Chat 1"
        assert conversations[1].name == "Chat 2"

    def test_get_conversation_by_id(self, test_db_session, mock_llm):
        """Test retrieving specific conversation by ID.

        Args:
            test_db_session: Database session fixture.
            mock_llm: LLM entity fixture.
        """
        repo = ConversationRepository(test_db_session)

        created = repo.create_conversation(
            llm_id=mock_llm.id, name="Test", temperature=0.5, top_p=0.8, max_tokens=512
        )
        retrieved = repo.get_conversation_by_id(created.id)

        assert retrieved.id == created.id
        assert retrieved.name == "Test"

    def test_get_conversation_by_id_not_found(self, test_db_session):
        """Test 404 error when conversation doesn't exist.

        Args:
            test_db_session: Database session fixture.
        """
        repo = ConversationRepository(test_db_session)

        with pytest.raises(Exception) as exc_info:
            repo.get_conversation_by_id(999)

        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND

    def test_update_conversation(self, test_db_session, mock_llm):
        """Test updating conversation properties.

        Args:
            test_db_session: Database session fixture.
            mock_llm: LLM entity fixture.
        """
        repo = ConversationRepository(test_db_session)

        conversation = repo.create_conversation(
            llm_id=mock_llm.id, name="Original", temperature=0.5, top_p=0.8, max_tokens=512
        )

        updated = repo.update_conversation(
            conversation_id=conversation.id, name="Updated", temperature=0.9
        )

        assert updated.name == "Updated"
        assert updated.temperature == 0.9
        assert updated.top_p == 0.8  # Unchanged

    def test_delete_conversation(self, test_db_session, mock_llm):
        """Test conversation deletion.

        Args:
            test_db_session: Database session fixture.
            mock_llm: LLM entity fixture.
        """
        repo = ConversationRepository(test_db_session)

        conversation = repo.create_conversation(
            llm_id=mock_llm.id, name="To Delete", temperature=0.5, top_p=0.8, max_tokens=512
        )

        repo.delete_conversation(conversation.id)

        with pytest.raises(Exception):
            repo.get_conversation_by_id(conversation.id)


class TestMessageRepository:
    """Test suite for MessageRepository database operations."""

    def test_create_message(self, test_db_session, mock_llm):
        """Test message creation in database.

        Args:
            test_db_session: Database session fixture.
            mock_llm: LLM entity fixture.
        """
        conv_repo = ConversationRepository(test_db_session)
        msg_repo = MessageRepository(test_db_session)

        conversation = conv_repo.create_conversation(
            llm_id=mock_llm.id, name="Test", temperature=0.5, top_p=0.8, max_tokens=512
        )

        message = msg_repo.create_message(
            conversation_id=conversation.id, sender="user", content="Hello AI"
        )

        assert message.id is not None
        assert message.conversation_id == conversation.id
        assert message.sender == "user"
        assert message.content == "Hello AI"
        assert message.starred == 0

    def test_get_messages_by_conversation_id(self, test_db_session, mock_llm):
        """Test retrieving all messages for a conversation.

        Args:
            test_db_session: Database session fixture.
            mock_llm: LLM entity fixture.
        """
        conv_repo = ConversationRepository(test_db_session)
        msg_repo = MessageRepository(test_db_session)

        conversation = conv_repo.create_conversation(
            llm_id=mock_llm.id, name="Test", temperature=0.5, top_p=0.8, max_tokens=512
        )

        msg_repo.create_message(conversation.id, "Question 1", "user")
        msg_repo.create_message(conversation.id, "Answer 1", "llm")
        msg_repo.create_message(conversation.id, "Question 2", "user")

        messages = msg_repo.get_messages_by_conversation(conversation.id)

        assert len(messages) == 3
        assert messages[0].sender == "user"
        assert messages[1].sender == "llm"
        assert messages[2].content == "Question 2"

    def test_star_message(self, test_db_session, mock_llm):
        """Test starring/bookmarking a message.

        Args:
            test_db_session: Database session fixture.
            mock_llm: LLM entity fixture.
        """
        conv_repo = ConversationRepository(test_db_session)
        msg_repo = MessageRepository(test_db_session)

        conversation = conv_repo.create_conversation(
            llm_id=mock_llm.id, name="Test", temperature=0.5, top_p=0.8, max_tokens=512
        )
        message = msg_repo.create_message(conversation.id, "Important answer", "llm")

        msg_repo.star_message(message.id)
        test_db_session.refresh(message)

        assert message.starred == 1

    def test_unstar_message(self, test_db_session, mock_llm):
        """Test unstarring a message.

        Args:
            test_db_session: Database session fixture.
            mock_llm: LLM entity fixture.
        """
        conv_repo = ConversationRepository(test_db_session)
        msg_repo = MessageRepository(test_db_session)

        conversation = conv_repo.create_conversation(
            llm_id=mock_llm.id, name="Test", temperature=0.5, top_p=0.8, max_tokens=512
        )
        message = msg_repo.create_message(conversation.id, "Answer", "llm")

        # Star then unstar
        msg_repo.star_message(message.id)
        msg_repo.unstar_message(message.id)
        test_db_session.refresh(message)

        assert message.starred == 0


# ============ Service Tests ============


class TestConversationService:
    """Test suite for ConversationService business logic with mocked engine."""

    def test_create_conversation_service(self, test_db_session, mock_llm):
        """Test conversation creation via service layer.

        Args:
            test_db_session: Database session fixture.
            mock_llm: LLM entity fixture.
        """
        service = ConversationService(test_db_session)

        conversation = service.create_conversation(
            llm_id=mock_llm.id, temperature=0.6, top_p=0.85, max_tokens=2048
        )

        assert conversation.name  # Auto-generated
        assert conversation.llm_id == mock_llm.id
        assert conversation.llm_id == mock_llm.id

    async def test_delete_conversation_service(self, test_db_session, mock_llm):
        """Test conversation deletion via service (now async; purges checkpointer)."""
        service = ConversationService(test_db_session)

        conversation = service.create_conversation(
            llm_id=mock_llm.id, temperature=0.5, top_p=0.8, max_tokens=512
        )

        await service.delete_conversation(conversation.id)

        # Verify deleted
        with pytest.raises(Exception):
            service.conversation_repo.get_conversation_by_id(conversation.id)

    def test_store_error_message(self, test_db_session, mock_llm):
        """Test storing error message when generation fails.

        Args:
            test_db_session: Database session fixture.
            mock_llm: LLM entity fixture.
        """
        service = ConversationService(test_db_session)

        conversation = service.create_conversation(
            llm_id=mock_llm.id, temperature=0.5, top_p=0.8, max_tokens=512
        )

        message_id = service.store_error_message(conversation.id)

        assert message_id is not None
        messages = service.message_repo.get_messages_by_conversation(conversation.id)
        assert len(messages) == 1
        assert "[ERROR_MESSAGE_SYSTEM]" in messages[0].content

    async def test_generate_title_stream(self, test_db_session, mock_llm, monkeypatch):
        """Title generation streams via a stateless one-shot model."""
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(agent_runner, "build_chat_model", _fake_chat_model("AI Basics"))
        service = ConversationService(test_db_session)
        conversation = service.create_conversation(
            llm_id=mock_llm.id, temperature=0.5, top_p=0.8, max_tokens=512
        )

        result = [t async for t in service.generate_title_stream(conversation.id, "What is AI?")]

        assert "".join(result) == "AI Basics"
        updated_conv = service.conversation_repo.get_conversation_by_id(conversation.id)
        assert updated_conv.name == "AI Basics"

    async def test_generate_title_prompt_instructs_conversation_language(
        self, test_db_session, mock_llm, monkeypatch
    ):
        """The title prompt tells the model to title in the USER's language.

        Without the instruction, French conversations get English titles
        ("Pomme Count", "Aster X2 Max Capacity") — #303. The capturing fake
        records the exact messages sent to the model so the assertion pins the
        prompt content, not the sanitized output.
        """
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        captured = {}

        class _CapturingFake(ToolableFakeChatModel):
            def _stream(self, messages, stop=None, run_manager=None, **kwargs):
                captured["messages"] = messages
                return super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)

        monkeypatch.setattr(
            agent_runner,
            "build_chat_model",
            lambda llm, **kw: _CapturingFake(
                messages=iter([AIMessage(content="Compte De Pommes")])
            ),
        )
        service = ConversationService(test_db_session)
        conversation = service.create_conversation(
            llm_id=mock_llm.id, temperature=0.5, top_p=0.8, max_tokens=512
        )

        async for _ in service.generate_title_stream(
            conversation.id, "Combien de pommes me reste-t-il ?"
        ):
            pass

        prompt = "\n".join(str(m.content) for m in captured["messages"])
        assert "same language as the user's message" in prompt

    async def test_generate_title_stream_empty_question(
        self, test_db_session, mock_llm, monkeypatch
    ):
        """Empty question short-circuits to the default title (no streaming)."""
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        service = ConversationService(test_db_session)
        conversation = service.create_conversation(
            llm_id=mock_llm.id, temperature=0.5, top_p=0.8, max_tokens=512
        )

        async for _ in service.generate_title_stream(conversation.id, ""):
            pass

        updated_conv = service.conversation_repo.get_conversation_by_id(conversation.id)
        assert updated_conv.name == "New Conversation"

    async def test_generate_title_strips_inline_thinking(
        self, test_db_session, mock_llm, monkeypatch
    ):
        """#266: a thinking model's inline <think>...</think> block never reaches
        the persisted title; only the answer text does."""
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(
            agent_runner, "build_chat_model", _fake_chat_model("<think>hmm</think>AI Basics")
        )
        service = ConversationService(test_db_session)
        conversation = service.create_conversation(
            llm_id=mock_llm.id, temperature=0.5, top_p=0.8, max_tokens=512
        )

        result = [t async for t in service.generate_title_stream(conversation.id, "What is AI?")]

        assert "".join(result) == "AI Basics"
        updated_conv = service.conversation_repo.get_conversation_by_id(conversation.id)
        assert updated_conv.name == "AI Basics"

    async def test_generate_title_unclosed_think_falls_back_to_default(
        self, test_db_session, mock_llm, monkeypatch
    ):
        """#266: a thinking model that burns the whole token budget inside an
        unclosed <think> must NOT persist the literal tag as the title."""
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(
            agent_runner, "build_chat_model", _fake_chat_model("<think>planning a title")
        )
        service = ConversationService(test_db_session)
        conversation = service.create_conversation(
            llm_id=mock_llm.id, temperature=0.5, top_p=0.8, max_tokens=512
        )

        async for _ in service.generate_title_stream(conversation.id, "What is AI?"):
            pass

        updated_conv = service.conversation_repo.get_conversation_by_id(conversation.id)
        assert updated_conv.name == "New Conversation"

    async def test_generate_title_sanitizes_junk_to_default(
        self, test_db_session, mock_llm, monkeypatch
    ):
        """A junk title (markdown fence repetition from a tiny model) is sanitized
        away, so the conversation keeps the default name instead of '```json…'."""
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(
            agent_runner, "build_chat_model", _fake_chat_model("```json\n```json\n```json")
        )
        service = ConversationService(test_db_session)
        conversation = service.create_conversation(
            llm_id=mock_llm.id, temperature=0.5, top_p=0.8, max_tokens=512
        )

        async for _ in service.generate_title_stream(
            conversation.id, "Remember this word: SUN. Reply OK."
        ):
            pass

        updated_conv = service.conversation_repo.get_conversation_by_id(conversation.id)
        assert updated_conv.name == "New Conversation"

    async def test_query_and_respond_stream(self, test_db_session, mock_llm, monkeypatch):
        """Query streams the agent response (raw text) and persists both messages."""
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(
            agent_runner, "build_chat_model", _fake_chat_model("Decorators are functions.")
        )
        service = ConversationService(test_db_session, InMemorySaver())
        conversation = service.create_conversation(
            llm_id=mock_llm.id, temperature=0.7, top_p=0.9, max_tokens=1024
        )

        payload = ConversationQuery(
            question="Explain Python decorators",
            temperature=0.7,
        )

        result = [t async for t in service.query_and_respond_stream(conversation.id, payload)]

        assert _answer_text(result) == "Decorators are functions."
        # NDJSON framing: every line parses, and the terminal event is `done`.
        events = _parse_ndjson(result)
        assert events[-1] == {"t": "done"}

        messages = service.message_repo.get_messages_by_conversation(conversation.id)
        assert len(messages) == 2
        assert messages[0].sender == "user"
        assert messages[0].content == "Explain Python decorators"
        assert messages[1].sender == "llm"
        assert messages[1].content == "Decorators are functions."
        # No thinking/tools this turn -> no trace persisted.
        assert messages[1].trace is None

    async def test_query_stream_persists_assistant_before_done_event(
        self, test_db_session, mock_llm, monkeypatch
    ):
        """#303: the ``done`` event is the client's signal to refetch messages.
        The assistant row must therefore be committed BEFORE ``done`` is
        yielded — otherwise the refetch races the insert, returns rows without
        the answer, and the frontend's list replacement erases the streamed
        bubble (observed live: fetch_messages 200 at t+7ms vs threadpool
        persist still in flight)."""
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(agent_runner, "build_chat_model", _fake_chat_model("The answer."))
        service = ConversationService(test_db_session, InMemorySaver())
        conversation = service.create_conversation(
            llm_id=mock_llm.id, temperature=0.7, top_p=0.9, max_tokens=1024
        )

        payload = ConversationQuery(question="A question", temperature=0.7)

        messages_at_done = None
        agen = service.query_and_respond_stream(conversation.id, payload)
        async for chunk in agen:
            events = _parse_ndjson(chunk)
            if any(e.get("t") == "done" for e in events):
                # The client acts HERE (refetch on ``done``), not after the
                # generator's finally block has run.
                messages_at_done = service.message_repo.get_messages_by_conversation(
                    conversation.id
                )
                break
        await agen.aclose()

        assert messages_at_done is not None, "stream never yielded a done event"
        senders = [m.sender for m in messages_at_done]
        assert senders == [
            "user",
            "llm",
        ], f"assistant row missing when done was emitted (got {senders})"
        assert messages_at_done[1].content == "The answer."

    async def test_query_stream_with_images_multimodal_and_placeholder(
        self, test_db_session, mock_llm, monkeypatch
    ):
        """An attached image rides the model call as multimodal content, but the
        DB stores only a short ``[image]`` placeholder (never the base64).
        Vision detection is pinned to True: this test covers the VLM
        pass-through path, not the #212 strip/notice."""
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(agent_runner, "build_chat_model", _fake_chat_model("A red square."))
        monkeypatch.setattr(
            "src.domains.conversations.services.detect_supports_vision", lambda _link: True
        )
        service = ConversationService(test_db_session, InMemorySaver())
        conversation = service.create_conversation(
            llm_id=mock_llm.id, temperature=0.7, top_p=0.9, max_tokens=1024
        )

        captured = {}
        original = service.runner.astream_text

        def spy(**kwargs):
            captured.update(kwargs)
            return original(**kwargs)

        monkeypatch.setattr(service.runner, "astream_text", spy)

        data_url = "data:image/png;base64," + "A" * 4000  # large: must NOT be persisted
        payload = ConversationQuery(question="What is this?", images=[data_url])

        result = [t async for t in service.query_and_respond_stream(conversation.id, payload)]
        assert _answer_text(result) == "A red square."

        # The model received multimodal content carrying the image.
        um = captured["user_message"]
        assert isinstance(um, list)
        assert any(p.get("type") == "image_url" and p["image_url"]["url"] == data_url for p in um)

        # The persisted user message is a short placeholder, not the base64.
        messages = service.message_repo.get_messages_by_conversation(conversation.id)
        assert messages[0].sender == "user"
        assert messages[0].content == "What is this? [image]"
        assert "base64" not in messages[0].content
        assert len(messages[0].content) < 100

    async def test_query_stream_unknown_vision_prepends_notice(
        self, test_db_session, mock_llm, monkeypatch
    ):
        """Current-turn images + unknown vision capability (None): the stream
        starts with the italic notice and the notice is persisted with the
        assistant message (#212)."""
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(agent_runner, "build_chat_model", _fake_chat_model("A red square."))
        service = ConversationService(test_db_session, InMemorySaver())
        conversation = service.create_conversation(
            llm_id=mock_llm.id, temperature=0.7, top_p=0.9, max_tokens=1024
        )

        payload = ConversationQuery(question="What is this?", images=["data:image/png;base64,AAA"])
        result = [t async for t in service.query_and_respond_stream(conversation.id, payload)]

        assert _answer_text(result) == _VISION_NOTICE + "A red square."

        messages = service.message_repo.get_messages_by_conversation(conversation.id)
        assert messages[1].sender == "llm"
        assert messages[1].content == _VISION_NOTICE + "A red square."

    async def test_query_stream_no_images_no_notice(self, test_db_session, mock_llm, monkeypatch):
        """No images this turn: no notice, even with unknown vision capability."""
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(agent_runner, "build_chat_model", _fake_chat_model("Hello."))
        service = ConversationService(test_db_session, InMemorySaver())
        conversation = service.create_conversation(
            llm_id=mock_llm.id, temperature=0.7, top_p=0.9, max_tokens=1024
        )

        payload = ConversationQuery(question="Say hello")
        result = [t async for t in service.query_and_respond_stream(conversation.id, payload)]

        assert _answer_text(result) == "Hello."
        assert not _answer_text(result).startswith(_VISION_NOTICE)

    async def test_query_stream_vision_true_no_notice(self, test_db_session, mock_llm, monkeypatch):
        """Images + a positively vision-capable model: no notice."""
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(agent_runner, "build_chat_model", _fake_chat_model("A red square."))
        monkeypatch.setattr(
            "src.domains.conversations.services.detect_supports_vision", lambda _link: True
        )
        service = ConversationService(test_db_session, InMemorySaver())
        conversation = service.create_conversation(
            llm_id=mock_llm.id, temperature=0.7, top_p=0.9, max_tokens=1024
        )

        payload = ConversationQuery(question="What is this?", images=["data:image/png;base64,AAA"])
        result = [t async for t in service.query_and_respond_stream(conversation.id, payload)]

        assert _answer_text(result) == "A red square."

    async def test_query_stream_history_images_do_not_trigger_notice(
        self, test_db_session, mock_llm, monkeypatch
    ):
        """Turn 1 carries an image (notice fires); turn 2 is text-only: the
        notice must NOT reappear — only CURRENT-turn images count (#212)."""
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        # One iterator SHARED across turns (``_fake_chat_model`` would restart
        # from the first message on the second ``build_chat_model`` call).
        msgs = iter([AIMessage(content="A red square."), AIMessage(content="It was red.")])
        monkeypatch.setattr(
            agent_runner,
            "build_chat_model",
            lambda llm, **kw: ToolableFakeChatModel(messages=msgs),
        )
        service = ConversationService(test_db_session, InMemorySaver())
        conversation = service.create_conversation(
            llm_id=mock_llm.id, temperature=0.7, top_p=0.9, max_tokens=1024
        )

        turn1 = ConversationQuery(question="What is this?", images=["data:image/png;base64,AAA"])
        result1 = [t async for t in service.query_and_respond_stream(conversation.id, turn1)]
        assert _answer_text(result1).startswith(_VISION_NOTICE)

        turn2 = ConversationQuery(question="What color was it?")
        result2 = [t async for t in service.query_and_respond_stream(conversation.id, turn2)]
        assert _answer_text(result2) == "It was red."

    def test_user_display_content_placeholder(self):
        assert ConversationService._user_display_content("hi", None) == "hi"
        assert ConversationService._user_display_content("hi", ["x"]) == "hi [image]"
        assert ConversationService._user_display_content("", ["x", "y"]) == "[image] [image]"

    def test_user_display_content_records_attached_documents(self):
        """#492 - a document attachment persists as a compact [file_path:...]
        marker, so a reloaded conversation shows what was attached without
        re-storing the extracted text."""
        assert (
            ConversationService._user_display_content("hi", None, None, ["/docs/report.pdf"])
            == "hi [file_path:/docs/report.pdf]"
        )
        assert (
            ConversationService._user_display_content("", None, None, ["/a.txt", "/b.md"])
            == "[file_path:/a.txt] [file_path:/b.md]"
        )
        # Images and documents coexist on the same turn.
        assert (
            ConversationService._user_display_content("hi", ["x"], ["/p.png"], ["/a.txt"])
            == "hi [image_path:/p.png] [file_path:/a.txt]"
        )

    def test_marker_paths_encode_the_closing_bracket(self):
        """A real path can hold ``]`` and ``%``. Markers are parsed on the
        frontend with a bracket-delimited regex, so both are percent-encoded
        here (``%`` first, so decoding is unambiguous) and the marker stays a
        single unambiguous token."""
        content = ConversationService._user_display_content(
            "look",
            ["x"],
            ["/docs/[2026] 100%/photo.png"],
            ["/docs/[2026] 100%/report.pdf"],
        )
        assert content == (
            "look [image_path:/docs/[2026%5D 100%25/photo.png] "
            "[file_path:/docs/[2026%5D 100%25/report.pdf]"
        )
        # Nothing between the marker prefix and its closing bracket can end it
        # early any more.
        assert content.count("]") == 2

    async def test_query_stream_injects_attachment_blocks_and_marker(
        self, test_db_session, mock_llm, monkeypatch, tmp_path
    ):
        """#492 - the model turn carries the delimited attachment block ahead of
        the question, and the persisted user message carries only the marker."""
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(agent_runner, "build_chat_model", _fake_chat_model("It is a report."))
        service = ConversationService(test_db_session, InMemorySaver())
        conversation = service.create_conversation(
            llm_id=mock_llm.id, temperature=0.7, top_p=0.9, max_tokens=1024
        )

        doc = tmp_path / "report.txt"
        doc.write_text("Revenue grew by twelve percent.", encoding="utf-8")

        captured = {}
        original = service.runner.astream_text

        def spy(**kwargs):
            captured.update(kwargs)
            return original(**kwargs)

        monkeypatch.setattr(service.runner, "astream_text", spy)

        payload = ConversationQuery(question="What does it say?", attachments=[str(doc)])
        result = [t async for t in service.query_and_respond_stream(conversation.id, payload)]

        assert _answer_text(result) == "It is a report."
        user_message = captured["user_message"]
        assert "[Attached file: report.txt]" in user_message
        assert "Revenue grew by twelve percent." in user_message
        assert "[End of attached file]" in user_message
        # The question stays last, after the attachment blocks.
        assert user_message.index("[End of attached file]") < user_message.index(
            "What does it say?"
        )

        messages = service.message_repo.get_messages_by_conversation(conversation.id)
        assert messages[0].sender == "user"
        assert messages[0].content == f"What does it say? [file_path:{doc}]"
        # The extracted text is NOT persisted as visible content.
        assert "Revenue grew" not in messages[0].content

    async def test_query_stream_reports_an_unreadable_attachment(
        self, test_db_session, mock_llm, monkeypatch, tmp_path
    ):
        """#492 - a file the reader cannot parse is named in the answer instead
        of failing the turn."""
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(agent_runner, "build_chat_model", _fake_chat_model("Answer."))
        service = ConversationService(test_db_session, InMemorySaver())
        conversation = service.create_conversation(
            llm_id=mock_llm.id, temperature=0.7, top_p=0.9, max_tokens=1024
        )

        bad = tmp_path / "archive.zip"
        bad.write_bytes(b"PK\x03\x04")

        payload = ConversationQuery(question="Read this", attachments=[str(bad)])
        result = [t async for t in service.query_and_respond_stream(conversation.id, payload)]

        answer = _answer_text(result)
        assert "archive.zip" in answer
        assert answer.endswith("Answer.")

    def test_build_user_message_shape(self):
        assert ConversationService._build_user_message("hi", None) == "hi"
        msg = ConversationService._build_user_message("hi", ["data:image/png;base64,AAA"])
        assert msg[0] == {"type": "text", "text": "hi"}
        assert msg[1]["type"] == "image_url"
        assert msg[1]["image_url"]["url"] == "data:image/png;base64,AAA"

    async def test_query_and_respond_stream_with_context(
        self, test_db_session, mock_llm, monkeypatch
    ):
        """Query with prior messages persists a 4th message (2 prior + user + llm)."""
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(
            agent_runner, "build_chat_model", _fake_chat_model("Python is versatile.")
        )
        service = ConversationService(test_db_session, InMemorySaver())
        conversation = service.create_conversation(
            llm_id=mock_llm.id, temperature=0.7, top_p=0.9, max_tokens=1024
        )

        service.message_repo.create_message(conversation.id, "What is Python?", "user")
        service.message_repo.create_message(conversation.id, "Python is a language.", "llm")

        payload = ConversationQuery(
            question="Tell me more",
            temperature=0.7,
        )

        result = [t async for t in service.query_and_respond_stream(conversation.id, payload)]

        assert _answer_text(result) == "Python is versatile."

        messages = service.message_repo.get_messages_by_conversation(conversation.id)
        assert len(messages) == 4

    async def test_query_and_respond_stream_with_kb(
        self, test_db_session, mock_llm_with_kb, monkeypatch
    ):
        """A KB-attached assistant retrieves chunks per question and injects
        them into the system prompt (same contract as the arena)."""
        llm, _kb = mock_llm_with_kb
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(agent_runner, "build_chat_model", _fake_chat_model("27 jours."))
        service = ConversationService(test_db_session, InMemorySaver())
        conversation = service.create_conversation(
            llm_id=llm.id, temperature=0.7, top_p=0.9, max_tokens=1024
        )

        captured = {}
        original = service.runner.astream_text

        def spy(**kwargs):
            captured.update(kwargs)
            return original(**kwargs)

        monkeypatch.setattr(service.runner, "astream_text", spy)

        payload = ConversationQuery(question="Combien de jours de congés payés ?")
        with patch("src.domains.conversations.services.retrieve_kb_excerpts") as mock_kb:
            mock_kb.return_value = [
                KbExcerpt(
                    source_file="convention.pdf",
                    text="Chaque employé dispose de 27 jours de congés payés par an.",
                )
            ]
            result = [t async for t in service.query_and_respond_stream(conversation.id, payload)]

        assert _answer_text(result) == "27 jours."
        mock_kb.assert_called_once()
        # param_size=7 → medium tier → its KB token budget reaches retrieval.
        assert mock_kb.call_args.kwargs["token_budget"] == 1000
        # #129: the KB system prompt composes the tier persona with an
        # appended excerpts contract, and the per-turn block carries the
        # excerpts + grounding reminder to the runner's request-time
        # middleware.
        prompt = captured["system_prompt"]
        assert "You are Erudi" in prompt
        assert "excerpts from the user's documents" in prompt
        block = captured["kb_context_block"]
        assert "[Document: convention.pdf]" in block
        assert "27 jours de congés payés" in block
        # FR question → localized scaffolding (English structural strings
        # around the question feed the English drift — eval runs 3-5).
        assert "Si les extraits ci-dessus sont pertinents" in block

    async def test_query_stream_kb_retrieval_failure_degrades_gracefully(
        self, test_db_session, mock_llm_with_kb, monkeypatch
    ):
        """A broken vector store must not sink the conversation: the stream
        continues without KB context instead of erroring out."""
        llm, _kb = mock_llm_with_kb
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(
            agent_runner, "build_chat_model", _fake_chat_model("Réponse sans contexte.")
        )
        service = ConversationService(test_db_session, InMemorySaver())
        conversation = service.create_conversation(
            llm_id=llm.id, temperature=0.7, top_p=0.9, max_tokens=1024
        )

        payload = ConversationQuery(question="Une question quelconque ?")
        with patch("src.domains.conversations.services.retrieve_kb_excerpts") as mock_kb:
            mock_kb.side_effect = RuntimeError("vector store unreachable")
            result = [t async for t in service.query_and_respond_stream(conversation.id, payload)]

        assert _answer_text(result) == "Réponse sans contexte."

    async def test_query_stream_agentic_mode_offers_kb_tool(
        self, test_db_session, mock_llm_with_kb, monkeypatch
    ):
        """A tool-capable model gets the KB as a TOOL (agentic), not a systematic
        injection: the runner receives search_knowledge_base + a TurnToolContext
        and NO kb_context_block, and there is no up-front retrieval."""
        llm, kb = mock_llm_with_kb
        llm.supports_tools = True
        test_db_session.commit()
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        # Agentic KB is opt-in (#288); this test asserts the tool path.
        monkeypatch.setattr(config, "KB_AGENTIC_MODE", True)
        monkeypatch.setattr(agent_runner, "build_chat_model", _fake_chat_model("ok"))
        service = ConversationService(test_db_session, InMemorySaver())
        conversation = service.create_conversation(
            llm_id=llm.id, temperature=0.7, top_p=0.9, max_tokens=1024
        )

        captured = {}
        original = service.runner.astream_text

        def spy(**kwargs):
            captured.update(kwargs)
            return original(**kwargs)

        monkeypatch.setattr(service.runner, "astream_text", spy)

        payload = ConversationQuery(question="Que dit le contrat ?")
        with patch("src.domains.conversations.services.retrieve_kb_excerpts") as mock_kb:
            result = [t async for t in service.query_and_respond_stream(conversation.id, payload)]

        assert _answer_text(result) == "ok"
        # Agentic mode does NOT retrieve up front — the model decides via the tool.
        mock_kb.assert_not_called()
        tool_names = [getattr(t, "name", None) for t in captured["tools"]]
        assert "search_knowledge_base" in tool_names
        assert captured["kb_context_block"] is None
        assert captured["context"] is not None and captured["context"].kb_id == kb.id

    async def test_query_stream_empty_final_persists_last_tool_result(
        self, test_db_session, mock_llm_with_kb, monkeypatch
    ):
        """#90/#84: agentic KB path — the model calls search_knowledge_base, the
        tool returns grounded text, then the model emits an EMPTY final answer
        (the Gemma pattern). The turn must NOT crash the empty-content guard in
        ``_persist_assistant_message``: the last tool result is streamed and
        persisted as the assistant message so the correct answer still reaches
        the user."""
        llm, _kb = mock_llm_with_kb
        llm.supports_tools = True
        test_db_session.commit()
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        # Agentic KB is opt-in (#288); this test exercises the tool path.
        monkeypatch.setattr(config, "KB_AGENTIC_MODE", True)

        tool_call = AIMessage(
            content="",
            tool_calls=[
                {"name": "search_knowledge_base", "args": {"query": "preavis"}, "id": "k1"}
            ],
        )
        # Successful tool call, then an EMPTY final answer.
        msgs = iter([tool_call, AIMessage(content="")])
        monkeypatch.setattr(
            agent_runner, "build_chat_model", lambda llm, **kw: ToolableFakeChatModel(messages=msgs)
        )
        service = ConversationService(test_db_session, InMemorySaver())
        conversation = service.create_conversation(
            llm_id=llm.id, temperature=0.7, top_p=0.9, max_tokens=1024
        )

        payload = ConversationQuery(question="Quel est le preavis ?")
        with patch("src.agents.tools.retrieve_kb_excerpts") as mock_kb:
            mock_kb.return_value = [
                KbExcerpt(source_file="contrat.pdf", text="Le preavis est de 90 jours.")
            ]
            result = [t async for t in service.query_and_respond_stream(conversation.id, payload)]

        events = _parse_ndjson(result)
        # The fallback lands as answer text (#90) and no error event is emitted.
        assert "90 jours" in _answer_text(result)
        assert not any(e.get("t") == "error" for e in events)
        # The tool step is captured in the wire trace (and persisted, below).
        assert any(e.get("t") == "tool_result" for e in events)

        # No crash: the user message + the fallback assistant message are persisted.
        messages = service.message_repo.get_messages_by_conversation(conversation.id)
        assert len(messages) == 2
        assert messages[1].sender == "llm"
        assert "90 jours" in messages[1].content
        # The tool activity is persisted as a replayable trace on the assistant turn.
        assert messages[1].trace is not None
        assert any(e.get("t") == "tool_result" for e in messages[1].trace)

    async def test_delete_conversation_purges_checkpointer_thread(
        self, test_db_session, mock_llm, monkeypatch
    ):
        """IT4 (BLOCKER B3): deleting a conversation purges its checkpointer thread,
        not just the DB rows. SQLite reuses autoincrement ids, so a stale thread
        would otherwise leak a deleted conversation's agent context into a new one.
        """
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(agent_runner, "build_chat_model", _fake_chat_model("An answer."))
        checkpointer = InMemorySaver()
        service = ConversationService(test_db_session, checkpointer)
        conversation = service.create_conversation(
            llm_id=mock_llm.id, temperature=0.7, top_p=0.9, max_tokens=512
        )

        payload = ConversationQuery(question="Hello there", temperature=0.7)
        async for _ in service.query_and_respond_stream(conversation.id, payload):
            pass

        cfg = {"configurable": {"thread_id": str(conversation.id)}}
        assert await checkpointer.aget_tuple(cfg) is not None  # thread created by the turn

        await service.delete_conversation(conversation.id)

        assert await checkpointer.aget_tuple(cfg) is None  # B3: thread purged

    async def test_query_failure_persists_sentinel_without_traceback(
        self, test_db_session, mock_llm, monkeypatch
    ):
        """IT9 (G1): a generation failure persists an llm message carrying the error
        sentinel and NO traceback / internal detail (no info leak), so the UI can
        render an error turn while the user message stays persisted.
        """
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)

        def _boom(llm, **kw):
            raise RuntimeError("model load failed: /secret/leak/path")

        monkeypatch.setattr(agent_runner, "build_chat_model", _boom)
        service = ConversationService(test_db_session, InMemorySaver())
        conversation = service.create_conversation(
            llm_id=mock_llm.id, temperature=0.7, top_p=0.9, max_tokens=512
        )

        payload = ConversationQuery(question="Trigger failure", temperature=0.7)
        result = [t async for t in service.query_and_respond_stream(conversation.id, payload)]

        # On the wire the sentinel is mapped to an `error` event (not an answer).
        events = _parse_ndjson(result)
        assert any(e.get("t") == "error" and _SENTINEL in e["text"] for e in events)
        assert _answer_text(result) == ""
        assert events[-1] == {"t": "done"}
        messages = service.message_repo.get_messages_by_conversation(conversation.id)
        # user message persisted up-front + the llm error message
        assert len(messages) == 2
        assert messages[0].sender == "user"
        assert messages[1].sender == "llm"
        # DB persistence UNCHANGED (#225-D4): the assistant content still carries
        # the sentinel string, and an error turn persists no trace.
        assert _SENTINEL in messages[1].content
        assert "Traceback" not in messages[1].content
        assert "/secret/leak/path" not in messages[1].content
        assert messages[1].trace is None

    async def test_query_stream_persists_thinking_trace(
        self, test_db_session, mock_llm, monkeypatch
    ):
        """A thinking model's ``<think>`` reasoning streams as ``thinking`` events
        and is persisted as the assistant turn's replayable trace; the answer text
        stays clean and is stored as the message content (#90)."""
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(
            agent_runner,
            "build_chat_model",
            _fake_chat_model("<think>reasoning here</think>The answer."),
        )
        service = ConversationService(test_db_session, InMemorySaver())
        conversation = service.create_conversation(
            llm_id=mock_llm.id, temperature=0.7, top_p=0.9, max_tokens=1024
        )

        payload = ConversationQuery(question="Think then answer")
        result = [t async for t in service.query_and_respond_stream(conversation.id, payload)]

        events = _parse_ndjson(result)
        assert events[-1] == {"t": "done"}
        # Thinking surfaced on the wire; the answer is clean (no <think> leakage).
        assert any(e["t"] == "thinking" for e in events)
        assert _answer_text(result) == "The answer."
        assert "<think>" not in _answer_text(result)

        messages = service.message_repo.get_messages_by_conversation(conversation.id)
        assert messages[1].content == "The answer."
        # Trace persisted: the ordered thinking events, replayable on reload.
        assert messages[1].trace is not None
        assert all(e["t"] == "thinking" for e in messages[1].trace)
        assert "".join(e["text"] for e in messages[1].trace) == "reasoning here"


class TestTraceHelpers:
    """Unit tests for the NDJSON framing + trace-cap helpers (#90)."""

    def test_ndjson_is_one_ascii_line(self):
        line = _ndjson({"t": "thinking", "text": "café"})
        assert line.endswith("\n")
        assert line.count("\n") == 1
        # ensure_ascii (default): non-ASCII is escaped on the wire but round-trips.
        assert "caf\\u00e9" in line
        assert json.loads(line) == {"t": "thinking", "text": "café"}

    def test_cap_trace_empty_is_none(self):
        assert _cap_trace([]) is None

    def test_cap_trace_small_is_unchanged(self):
        events = [
            {"t": "thinking", "text": "a"},
            {"t": "tool_result", "name": "x", "text": "y"},
        ]
        assert _cap_trace(events) == events

    def test_cap_trace_drops_oldest_and_marks_truncated(self):
        big = "x" * 1000
        events = [{"t": "thinking", "text": f"{i}-{big}"} for i in range(50)]

        capped = _cap_trace(events)

        assert len(json.dumps(capped)) <= TRACE_MAX_BYTES
        assert capped[0] == {"t": "truncated"}
        assert capped[-1] == events[-1]  # newest survives
        assert events[0] not in capped  # oldest dropped


# ============ Endpoint Tests ============


class TestConversationEndpoints:
    """Test suite for conversation REST API endpoints with mocked engine."""

    def test_create_conversation_endpoint(self, client, mock_llm):
        """Test conversation creation via REST API.

        Args:
            client: FastAPI test client.
            mock_llm: LLM entity fixture.
        """
        payload = {"llm_id": mock_llm.id, "temperature": 0.7, "top_p": 0.9, "max_tokens": 1024}

        response = client.post("/erudi/conversations/", json=payload)

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["name"] == "New Conversation"  # Default name when not specified
        assert data["llm_id"] == mock_llm.id

    def test_get_all_conversations_endpoint(self, client, mock_llm):
        """Test retrieving all conversations via REST API.

        Args:
            client: FastAPI test client.
            mock_llm: LLM entity fixture.
        """
        # Create conversations
        client.post("/erudi/conversations/", json={"llm_id": mock_llm.id})
        client.post("/erudi/conversations/", json={"llm_id": mock_llm.id})

        response = client.get("/erudi/conversations/")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 2

    def test_get_conversation_by_id_endpoint(self, client, mock_llm):
        """Test retrieving specific conversation via REST API.

        Args:
            client: FastAPI test client.
            mock_llm: LLM entity fixture.
        """
        create_response = client.post("/erudi/conversations/", json={"llm_id": mock_llm.id})
        conversation_id = create_response.json()["id"]

        response = client.get(f"/erudi/conversations/{conversation_id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == conversation_id
        assert data["name"] == "New Conversation"  # Default name when not specified

    def test_update_conversation_endpoint(self, client, mock_llm):
        """Test updating conversation via REST API.

        Args:
            client: FastAPI test client.
            mock_llm: LLM entity fixture.
        """
        create_response = client.post(
            "/erudi/conversations/", json={"llm_id": mock_llm.id, "name": "Original"}
        )
        conversation_id = create_response.json()["id"]

        update_payload = {"name": "Updated Name", "temperature": 0.9}
        response = client.patch(f"/erudi/conversations/{conversation_id}", json=update_payload)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["name"] == "Updated Name"
        assert data["temperature"] == 0.9

    def test_delete_conversation_endpoint(self, client, mock_llm):
        """Test deleting conversation via REST API.

        Args:
            client: FastAPI test client.
            mock_llm: LLM entity fixture.
        """
        create_response = client.post(
            "/erudi/conversations/", json={"llm_id": mock_llm.id, "name": "To Delete"}
        )
        conversation_id = create_response.json()["id"]

        response = client.delete(f"/erudi/conversations/{conversation_id}")

        assert response.status_code == status.HTTP_200_OK

        # Verify deleted
        get_response = client.get(f"/erudi/conversations/{conversation_id}")
        assert get_response.status_code == status.HTTP_404_NOT_FOUND

    def test_query_endpoint_streaming(self, client, mock_llm):
        """Test streaming query endpoint with mocked engine.

        Args:
            client: FastAPI test client.
            mock_llm: LLM entity fixture.
        """
        create_response = client.post(
            "/erudi/conversations/", json={"llm_id": mock_llm.id, "name": "Stream Test"}
        )
        conversation_id = create_response.json()["id"]

        query_payload = {"question": "What is Python?", "temperature": 0.7}

        # Patch model construction; the client fixture already set a real engine
        # (for generation_guard) and an in-memory checkpointer on app.state.
        with patch.object(agent_runner, "build_chat_model", _fake_chat_model("Python is awesome.")):
            response = client.post(
                f"/erudi/conversations/{conversation_id}/query", json=query_payload
            )

        assert response.status_code == status.HTTP_200_OK
        # NDJSON wire contract (#90): one event per line, `done` last.
        assert response.headers["content-type"].startswith("application/x-ndjson")
        assert _answer_text(response.text) == "Python is awesome."
        assert _parse_ndjson(response.text)[-1] == {"t": "done"}

    def test_generate_title_endpoint(self, client, mock_llm):
        """Test title generation endpoint with mocked engine.

        Args:
            client: FastAPI test client.
            mock_llm: LLM entity fixture.
        """
        create_response = client.post(
            "/erudi/conversations/", json={"llm_id": mock_llm.id, "name": "New"}
        )
        conversation_id = create_response.json()["id"]

        title_payload = {"question": "What is machine learning?"}

        with patch.object(agent_runner, "build_chat_model", _fake_chat_model("ML Intro")):
            response = client.post(
                f"/erudi/conversations/{conversation_id}/generate_title", json=title_payload
            )

        assert response.status_code == status.HTTP_200_OK
        assert response.text == "ML Intro"

    def test_get_messages_endpoint(self, client, mock_llm, test_db_session):
        """Test retrieving messages for a conversation.

        Args:
            client: FastAPI test client.
            mock_llm: LLM entity fixture.
            test_db_session: Database session fixture.
        """
        from src.domains.conversations.repository import MessageRepository

        create_response = client.post(
            "/erudi/conversations/", json={"llm_id": mock_llm.id, "name": "Message Test"}
        )
        conversation_id = create_response.json()["id"]

        # Add messages directly
        msg_repo = MessageRepository(test_db_session)
        msg_repo.create_message(conversation_id, "Hello", "user")
        msg_repo.create_message(conversation_id, "Hi there", "llm")

        response = client.get(f"/erudi/conversations/{conversation_id}/fetch_messages")

        assert response.status_code == status.HTTP_200_OK
        messages = response.json()
        assert len(messages) == 2
        assert messages[0]["content"] == "Hello"
        assert messages[1]["content"] == "Hi there"


class TestTitleSanitization:
    """Unit tests for _sanitize_title (strip markdown noise / repetition / caps)."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("```json\n```json\n```json\n```json", ""),  # fenced repetition -> empty
            ("Paris, City of Light", "Paris, City of Light"),
            ('"Pizza Recipe"', "Pizza Recipe"),  # wrapping quotes stripped
            ("json json json", ""),  # dedupe -> 'json' -> rejected
            ("# Google Founding Team", "Google Founding Team"),  # heading marker stripped
            ("```\ntext\n```", ""),  # fence + generic noise -> empty
            ("", ""),
        ],
    )
    def test_sanitize_title(self, raw, expected):
        assert _sanitize_title(raw) == expected

    def test_sanitize_title_caps_word_count(self):
        assert (
            _sanitize_title("one two three four five six seven eight")
            == "one two three four five six"
        )
