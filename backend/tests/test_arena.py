"""Comprehensive tests for arena domain (stateless LLM queries).

Tests cover:
- Repository layer (database operations)
- Service layer (business logic with a fake chat model)
- Endpoint layer (REST API with streaming)

Generation goes through the shared AgentRunner (stateless: no checkpointer),
so the fake chat model is injected by patching ``build_chat_model``.
"""

import pytest
from unittest.mock import patch
from fastapi import status

from tests._helpers import ToolableFakeChatModel
from langchain_core.messages import AIMessage

import src.agents.runner as agent_runner
from src.core import config
from src.engines.base_engine import BaseEngine
from src.agents.runner import ERROR_SENTINEL
from src.domains.arena.repository import ArenaRepository
from src.domains.arena.services import ArenaService
from src.domains.arena.schemas import ArenaQueryPayload
from src.utils.kb_utils import KbExcerpt


class _FakeEngine(BaseEngine):
    """Engine stub exposing generation_guard without spawning a real model.

    Inherits the ``BaseEngine.model_supports_vision`` default, so
    ``detect_supports_vision`` resolves to ``None`` (unknown capability).
    """


# Exact user-facing line prepended when the current turn carries images but the
# model is not positively vision-capable (#212). Hardcoded on purpose: the test
# pins the wire contract, not the constant's name.
_VISION_NOTICE = "*This model doesn't support images — your image was ignored.*\n\n"


def _fake_chat_model(*texts):
    """Return a build_chat_model replacement yielding a scripted fake model."""
    msgs = [AIMessage(content=t) for t in texts]
    return lambda llm, **kw: ToolableFakeChatModel(messages=iter(msgs))


# ============ Repository Tests ============


class TestArenaRepository:
    """Test suite for ArenaRepository database operations."""

    def test_get_llm_by_id_success(self, test_db_session, mock_llm):
        repo = ArenaRepository(test_db_session)
        llm = repo.get_llm_by_id(mock_llm.id)
        assert llm is not None
        assert llm.id == mock_llm.id
        assert llm.name == "Test Base Model"
        assert llm.param_size == 7.0

    def test_get_llm_by_id_not_found(self, test_db_session):
        repo = ArenaRepository(test_db_session)
        with pytest.raises(Exception) as exc_info:
            repo.get_llm_by_id(999)
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND


# ============ Service Tests ============


class TestArenaService:
    """Test suite for ArenaService business logic (fake chat model)."""

    async def test_query_llm_stream_basic(self, test_db_session, mock_llm, monkeypatch):
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(
            agent_runner, "build_chat_model", _fake_chat_model("AI is artificial intelligence.")
        )
        service = ArenaService(test_db_session)
        payload = ArenaQueryPayload(
            question="What is AI?", temperature=0.7, top_p=0.9, max_new_tokens=512
        )

        result = [t async for t in service.query_llm_stream(mock_llm.id, payload)]

        assert "".join(result) == "AI is artificial intelligence."

    async def test_query_llm_stream_empty_question_rejected_by_pydantic(
        self, test_db_session, mock_llm
    ):
        # Empty question with no images is rejected at the schema level -> 422.
        with pytest.raises(Exception):
            ArenaQueryPayload(question="", temperature=0.5)

    def test_payload_attachment_only_is_valid(self):
        # A document-only ask ("what does this say?") is a valid turn (#492).
        payload = ArenaQueryPayload(question="  ", attachments=["/docs/report.pdf"])
        assert payload.question == ""
        assert payload.attachments == ["/docs/report.pdf"]

    async def test_query_llm_stream_injects_attachment_blocks(
        self, test_db_session, mock_llm, monkeypatch, tmp_path
    ):
        """#492 - the arena shares the conversation resolver: the model turn
        carries the delimited attachment block ahead of the question."""
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(agent_runner, "build_chat_model", _fake_chat_model("It is a report."))
        service = ArenaService(test_db_session)

        doc = tmp_path / "report.txt"
        doc.write_text("Revenue grew by twelve percent.", encoding="utf-8")

        captured = {}
        original = service.runner.astream_text

        def spy(**kwargs):
            captured.update(kwargs)
            return original(**kwargs)

        monkeypatch.setattr(service.runner, "astream_text", spy)

        payload = ArenaQueryPayload(question="What does it say?", attachments=[str(doc)])
        result = [t async for t in service.query_llm_stream(mock_llm.id, payload)]

        assert "".join(result) == "It is a report."
        user_message = captured["user_message"]
        assert "[Attached file: report.txt]" in user_message
        assert "Revenue grew by twelve percent." in user_message
        assert user_message.index("[End of attached file]") < user_message.index(
            "What does it say?"
        )

    async def test_query_llm_stream_reports_an_unreadable_attachment(
        self, test_db_session, mock_llm, monkeypatch, tmp_path
    ):
        """#492 - an unreadable attachment is named in the stream, not fatal."""
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(agent_runner, "build_chat_model", _fake_chat_model("Answer."))
        service = ArenaService(test_db_session)

        bad = tmp_path / "archive.zip"
        bad.write_bytes(b"PK\x03\x04")

        payload = ArenaQueryPayload(question="Read this", attachments=[str(bad)])
        result = [t async for t in service.query_llm_stream(mock_llm.id, payload)]

        joined = "".join(result)
        assert "archive.zip" in joined
        assert joined.endswith("Answer.")

    def test_payload_image_only_is_valid(self):
        # An image-only ask (no text) is a legitimate vision-model turn,
        # mirroring conversations (#136 C).
        payload = ArenaQueryPayload(question="  ", images=["data:image/png;base64,AAA"])
        assert payload.question == ""
        assert payload.images == ["data:image/png;base64,AAA"]

    async def test_query_llm_stream_with_images_multimodal(
        self, test_db_session, mock_llm, monkeypatch
    ):
        """Attached images ride the model call as multimodal content (#136 C).
        Vision detection is pinned to True: this test covers the VLM
        pass-through path, not the #212 strip/notice."""
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(agent_runner, "build_chat_model", _fake_chat_model("A red square."))
        monkeypatch.setattr("src.domains.arena.services.detect_supports_vision", lambda _link: True)
        service = ArenaService(test_db_session)

        captured = {}
        original = service.runner.astream_text

        def spy(**kwargs):
            captured.update(kwargs)
            return original(**kwargs)

        monkeypatch.setattr(service.runner, "astream_text", spy)

        data_url = "data:image/png;base64," + "A" * 4000
        payload = ArenaQueryPayload(question="What is this?", images=[data_url])

        result = [t async for t in service.query_llm_stream(mock_llm.id, payload)]
        assert "".join(result) == "A red square."

        # The model received multimodal content carrying the image.
        um = captured["user_message"]
        assert isinstance(um, list)
        assert any(p.get("type") == "image_url" and p["image_url"]["url"] == data_url for p in um)

    async def test_query_llm_stream_unknown_vision_prepends_notice(
        self, test_db_session, mock_llm, monkeypatch
    ):
        """Current-turn images + unknown vision capability (None): the stream
        starts with the italic notice (#212)."""
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(agent_runner, "build_chat_model", _fake_chat_model("A red square."))
        service = ArenaService(test_db_session)

        payload = ArenaQueryPayload(question="What is this?", images=["data:image/png;base64,AAA"])
        result = [t async for t in service.query_llm_stream(mock_llm.id, payload)]

        assert "".join(result) == _VISION_NOTICE + "A red square."

    async def test_query_llm_stream_vision_true_no_notice(
        self, test_db_session, mock_llm, monkeypatch
    ):
        """Images + a positively vision-capable model: no notice."""
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(agent_runner, "build_chat_model", _fake_chat_model("A red square."))
        monkeypatch.setattr("src.domains.arena.services.detect_supports_vision", lambda _link: True)
        service = ArenaService(test_db_session)

        payload = ArenaQueryPayload(question="What is this?", images=["data:image/png;base64,AAA"])
        result = [t async for t in service.query_llm_stream(mock_llm.id, payload)]

        assert "".join(result) == "A red square."

    async def test_query_llm_stream_plain_is_zero_tool(
        self, test_db_session, mock_llm, monkeypatch
    ):
        """#129: an arena turn on a model without a KB attached (plain mode)
        reaches the runner with NO tools at all — not even the calculator."""
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(agent_runner, "build_chat_model", _fake_chat_model("Hello."))
        service = ArenaService(test_db_session)

        captured = {}
        original = service.runner.astream_text

        def spy(**kwargs):
            captured.update(kwargs)
            return original(**kwargs)

        monkeypatch.setattr(service.runner, "astream_text", spy)

        payload = ArenaQueryPayload(question="Say hello")
        result = [t async for t in service.query_llm_stream(mock_llm.id, payload)]

        assert "".join(result) == "Hello."
        assert captured["tools"] == []

    async def test_query_llm_stream_no_images_no_notice(
        self, test_db_session, mock_llm, monkeypatch
    ):
        """No images this turn: no notice, even with unknown vision capability."""
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(agent_runner, "build_chat_model", _fake_chat_model("Hello."))
        service = ArenaService(test_db_session)

        payload = ArenaQueryPayload(question="Say hello")
        result = [t async for t in service.query_llm_stream(mock_llm.id, payload)]

        assert "".join(result) == "Hello."
        assert not "".join(result).startswith(_VISION_NOTICE)

    async def test_query_llm_stream_model_not_found(self, test_db_session):
        service = ArenaService(test_db_session)
        payload = ArenaQueryPayload(question="Test question")
        with pytest.raises(Exception) as exc_info:
            async for _ in service.query_llm_stream(999, payload):
                pass
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND

    async def test_query_llm_stream_with_kb(self, test_db_session, mock_llm_with_kb, monkeypatch):
        llm, _kb = mock_llm_with_kb
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(agent_runner, "build_chat_model", _fake_chat_model("Answer from KB."))
        service = ArenaService(test_db_session)
        payload = ArenaQueryPayload(question="What is in the KB?", temperature=0.5)

        with patch("src.domains.arena.services.retrieve_kb_excerpts") as mock_kb:
            mock_kb.return_value = [KbExcerpt(source_file="notes.md", text="Relevant KB context")]
            result = [t async for t in service.query_llm_stream(llm.id, payload)]

        assert "".join(result) == "Answer from KB."
        mock_kb.assert_called_once()
        # param_size=7 → medium tier → its KB token budget reaches retrieval.
        assert mock_kb.call_args.kwargs["token_budget"] == 1000

    async def test_query_llm_stream_empty_final_falls_back_to_tool_result(
        self, test_db_session, mock_llm_with_kb, monkeypatch
    ):
        """#90/#84: the arena shares the conversation ``AgentRunner``, so the
        empty-final fallback applies here too. A tool-capable model calls
        search_knowledge_base, the tool returns grounded text, then the model
        emits an EMPTY final answer — the last tool result must be streamed to
        the user instead of nothing."""
        llm, _kb = mock_llm_with_kb
        llm.supports_tools = True
        test_db_session.commit()
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        # Agentic KB is opt-in (#288); this test exercises the tool path.
        monkeypatch.setattr(config, "KB_AGENTIC_MODE", True)

        tool_call = AIMessage(
            content="",
            tool_calls=[{"name": "search_knowledge_base", "args": {"query": "x"}, "id": "k1"}],
        )
        msgs = iter([tool_call, AIMessage(content="")])
        monkeypatch.setattr(
            agent_runner, "build_chat_model", lambda llm, **kw: ToolableFakeChatModel(messages=msgs)
        )
        service = ArenaService(test_db_session)
        payload = ArenaQueryPayload(question="What is in the KB?", temperature=0.5)

        with patch("src.agents.tools.retrieve_kb_excerpts") as mock_kb:
            mock_kb.return_value = [KbExcerpt(source_file="notes.md", text="Relevant KB context")]
            result = [t async for t in service.query_llm_stream(llm.id, payload)]

        streamed = "".join(result)
        assert "Relevant KB context" in streamed
        assert ERROR_SENTINEL not in streamed

    async def test_query_llm_stream_custom_params(self, test_db_session, mock_llm, monkeypatch):
        # Per-request generation params must reach the model factory.
        captured = {}

        def _capture(llm, **kw):
            captured.update(kw)
            return ToolableFakeChatModel(messages=iter([AIMessage(content="Custom response.")]))

        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(agent_runner, "build_chat_model", _capture)
        service = ArenaService(test_db_session)
        payload = ArenaQueryPayload(
            question="Test custom params",
            temperature=1.5,
            top_p=0.95,
            max_new_tokens=2048,
            custom_prompt="Be concise",
        )

        result = [t async for t in service.query_llm_stream(mock_llm.id, payload)]

        assert "".join(result) == "Custom response."
        assert captured["temperature"] == 1.5
        assert captured["top_p"] == 0.95
        assert captured["max_tokens"] == 2048

    async def test_query_llm_stream_engine_failure_yields_sentinel(
        self, test_db_session, mock_llm, monkeypatch
    ):
        # Unified error policy: model-load failure yields the sentinel inline
        # (the old code raised, which was lost after the 200 response started).
        def _boom(llm, **kw):
            raise RuntimeError("Model load failed")

        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(agent_runner, "build_chat_model", _boom)
        service = ArenaService(test_db_session)
        payload = ArenaQueryPayload(question="Trigger error")

        result = [t async for t in service.query_llm_stream(mock_llm.id, payload)]
        assert any(ERROR_SENTINEL in t for t in result)


# ============ Endpoint Tests ============


class TestArenaEndpoints:
    """Test suite for arena REST API endpoints (fake chat model)."""

    def test_query_endpoint_success(self, client, test_db_session, mock_llm):
        payload = {
            "question": "What is machine learning?",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 512,
        }
        with patch.object(
            agent_runner, "build_chat_model", _fake_chat_model("Machine learning is AI.")
        ):
            response = client.post(f"/erudi/arena/{mock_llm.id}/query", json=payload)
        assert response.status_code == status.HTTP_200_OK
        assert response.text == "Machine learning is AI."

    def test_query_endpoint_invalid_payload(self, client, test_db_session, mock_llm):
        payload = {"question": "", "temperature": 3.0}  # empty + out-of-range
        response = client.post(f"/erudi/arena/{mock_llm.id}/query", json=payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_query_endpoint_model_not_found(self, client):
        response = client.post("/erudi/arena/999/query", json={"question": "Test question"})
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_query_endpoint_image_only(self, client, mock_llm):
        # A question-less ask carrying an image streams normally (#136 C).
        # The mock LLM's path resolves to no real model, so vision capability
        # is unknown (None) and the stream starts with the #212 notice.
        payload = {"question": "", "images": ["data:image/png;base64,AAA"]}
        with patch.object(agent_runner, "build_chat_model", _fake_chat_model("A tiny image.")):
            response = client.post(f"/erudi/arena/{mock_llm.id}/query", json=payload)
        assert response.status_code == status.HTTP_200_OK
        assert response.text == _VISION_NOTICE + "A tiny image."

    def test_query_endpoint_with_custom_prompt(self, client, mock_llm):
        payload = {
            "question": "Explain quantum physics",
            "custom_prompt": "Use simple language for a child",
            "temperature": 0.8,
        }
        with patch.object(
            agent_runner, "build_chat_model", _fake_chat_model("Quantum is tiny stuff.")
        ):
            response = client.post(f"/erudi/arena/{mock_llm.id}/query", json=payload)
        assert response.status_code == status.HTTP_200_OK
        assert "Quantum" in response.text
