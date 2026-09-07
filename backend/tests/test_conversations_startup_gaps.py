"""Gap coverage for the conversations HTTP layer, service fallbacks and the
startup endpoints, on top of `test_conversations.py` / `test_startup.py`.

Pins: error-message persistence, star/unstar routes with their rollback
paths, the create/delete/update error branches, the streaming generators'
load-failure degradation (error event instead of a 500 mid-stream), the
stale-llm auto-repair (#), the checkpointer purge tolerance, title-gen
fallbacks, and the welcome-popup endpoint.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.domains.conversations.services import ConversationService, ERROR_MESSAGE
from src.entities.Conversation import Conversation
from src.entities.Llm import Llm
from src.entities.Message import Message


def _conversation(db, llm_id, **overrides):
    conv = Conversation(llm_id=llm_id, name="Conv", **overrides)
    db.add(conv)
    db.commit()
    return conv


# =====================================================================
# INTEGRATION - conversations endpoints
# =====================================================================


@pytest.mark.integration
class TestConversationEndpointGaps:
    def test_create_with_unknown_llm_rolls_back(self, client):
        # FK violation surfaces as a wrapped DatabaseException (500), and the
        # endpoint's rollback branch must leave the session usable afterwards.
        # Explicit sampling values keep the request off the model lookup (#388:
        # omitted values resolve from the Llm row first, which is a clean 404 --
        # see test_sampling_contracts) so the FK path itself stays exercised.
        resp = client.post(
            "/erudi/conversations/",
            json={
                "llm_id": 987654,
                "temperature": 0.2,
                "top_p": 0.95,
                "max_tokens": 64,
            },
        )
        assert resp.status_code == 500
        assert client.get("/erudi/conversations/").status_code == 200
        assert client.post("/erudi/conversations/", json={"llm_id": 987654}).status_code == 404

    def test_delete_unknown_conversation_rolls_back(self, client):
        assert client.delete("/erudi/conversations/987654").status_code == 404

    def test_update_unknown_conversation_rolls_back(self, client):
        resp = client.patch("/erudi/conversations/987654", json={"name": "X"})
        assert resp.status_code == 404

    def test_store_error_message_persists_fallback_row(self, client, test_db_session, mock_llm):
        conv = _conversation(test_db_session, mock_llm.id)
        resp = client.post(f"/erudi/conversations/{conv.id}/store_error_message")
        assert resp.status_code == 200
        body = resp.json()
        message = test_db_session.get(Message, body["error_message_id"])
        assert message.sender == "llm"
        assert message.content == ERROR_MESSAGE

    def test_store_error_message_unknown_conversation_rolls_back(self, client):
        # No existence check before the insert: the FK violation is wrapped as
        # a DatabaseException (500) after the endpoint's rollback.
        resp = client.post("/erudi/conversations/987654/store_error_message")
        assert resp.status_code == 500

    def test_star_and_unstar_roundtrip(self, client, test_db_session, mock_llm):
        conv = _conversation(test_db_session, mock_llm.id)
        message = Message(conversation_id=conv.id, content="hi", sender="llm")
        test_db_session.add(message)
        test_db_session.commit()

        resp = client.post("/erudi/conversations/star_message", json={"message_id": message.id})
        assert resp.status_code == 200
        assert resp.json()["state"] == "success"
        test_db_session.refresh(message)
        assert message.starred is True

        resp = client.post("/erudi/conversations/unstar_message", json={"message_id": message.id})
        assert resp.status_code == 200
        test_db_session.refresh(message)
        assert message.starred is False

    def test_star_unknown_message_rolls_back(self, client):
        resp = client.post("/erudi/conversations/star_message", json={"message_id": 987654})
        assert resp.status_code == 404

    def test_unstar_unknown_message_rolls_back(self, client):
        resp = client.post("/erudi/conversations/unstar_message", json={"message_id": 987654})
        assert resp.status_code == 404


# =====================================================================
# INTEGRATION - service fallbacks
# =====================================================================


@pytest.mark.integration
class TestConversationServiceGaps:
    async def test_purge_thread_without_checkpointer_is_noop(self, test_db_session):
        service = ConversationService(test_db_session, checkpointer=None)
        await service._purge_thread(1)

    async def test_purge_thread_swallows_checkpointer_failure(self, test_db_session):
        checkpointer = MagicMock()

        async def broken_delete(thread_id):
            raise RuntimeError("checkpointer db gone")

        checkpointer.adelete_thread = broken_delete
        service = ConversationService(test_db_session, checkpointer=checkpointer)
        await service._purge_thread(1)  # must not raise

    async def test_query_stream_degrades_when_conversation_missing(self, test_db_session):
        service = ConversationService(test_db_session)
        payload = SimpleNamespace(question="hello", images=None, attachments=None)
        events = [chunk async for chunk in service.query_and_respond_stream(987654, payload)]
        assert any('"error"' in e and ERROR_MESSAGE in e for e in events)
        assert '"done"' in events[-1]

    async def test_title_stream_saves_default_when_conversation_missing(self, test_db_session):
        service = ConversationService(test_db_session)
        with patch.object(service, "_save_title") as save:
            chunks = [c async for c in service.generate_title_stream(987654, "any question")]
        assert chunks == []
        save.assert_called_once_with(987654, "New Conversation")

    async def test_title_stream_saves_default_on_empty_question(self, test_db_session, mock_llm):
        conv = _conversation(test_db_session, mock_llm.id)
        service = ConversationService(test_db_session)
        with patch.object(service, "_save_title") as save:
            chunks = [c async for c in service.generate_title_stream(conv.id, "   ")]
        assert chunks == []
        save.assert_called_once_with(conv.id, "New Conversation")

    def test_build_title_prompt_has_mistral_variant(self, test_db_session):
        service = ConversationService(test_db_session)
        mistral = service._build_title_prompt_text("What is the sun?", "mistral")
        generic = service._build_title_prompt_text("What is the sun?", "qwen")
        assert mistral != generic
        assert "TITLE" in mistral

    def test_load_auto_repairs_stale_llm_id(self, test_db_session, mock_llm):
        stale = Llm(name="Doomed", local=0, link="x/y", type="qwen")
        test_db_session.add(stale)
        test_db_session.flush()
        conv = _conversation(test_db_session, stale.id)
        test_db_session.delete(stale)
        test_db_session.commit()

        service = ConversationService(test_db_session)
        conversation, llm = service._load_conversation_and_llm(conv.id)

        # mock_llm is the only local=1 model: the conversation is repointed to it
        assert llm.id == mock_llm.id
        assert conversation.llm_id == mock_llm.id

    def test_load_raises_when_no_local_model_available(self, test_db_session, mock_llm):
        from src.core.exceptions import ModelNotFoundException

        stale = Llm(name="Doomed", local=0, link="x/y", type="qwen")
        test_db_session.add(stale)
        test_db_session.flush()
        conv = _conversation(test_db_session, stale.id)
        test_db_session.delete(stale)
        # Remove the only local model so auto-repair has no target
        test_db_session.delete(mock_llm)
        test_db_session.commit()

        service = ConversationService(test_db_session)
        with pytest.raises(ModelNotFoundException):
            service._load_conversation_and_llm(conv.id)


# =====================================================================
# INTEGRATION - startup endpoints
# =====================================================================


@pytest.mark.integration
class TestStartupEndpoints:
    def test_welcome_popup_flips_on_first_call(self, client):
        first = client.get("/erudi/startup/welcome-popup")
        assert first.status_code == 200
        assert first.json()["has_already_displayed"] is False

        second = client.get("/erudi/startup/welcome-popup")
        assert second.status_code == 200
        assert second.json()["has_already_displayed"] is True

    def test_welcome_popup_failure_maps_to_database_exception(self, client):
        from src.domains.startup.repository import Startup_Variables_Repository

        with patch.object(
            Startup_Variables_Repository,
            "get_or_create",
            side_effect=RuntimeError("db down"),
        ):
            resp = client.get("/erudi/startup/welcome-popup")
        assert resp.status_code == 500
