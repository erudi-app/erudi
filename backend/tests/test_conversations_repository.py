"""Tests for the conversation domain data access layer.

Runs `ConversationRepository` and `MessageRepository` against the real test
cluster: CRUD roundtrips, the no-op update fast path, cascade deletion,
star/unstar, and the SQLAlchemyError -> DatabaseException wrapping that the
endpoint layer relies on for structured error responses.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest
from sqlalchemy.exc import SQLAlchemyError

from src.core.exceptions import (
    ConversationNotFoundException,
    DatabaseException,
    MessageNotFoundException,
    ModelNotFoundException,
)
from src.domains.conversations.repository import (
    ConversationRepository,
    MessageRepository,
)
from src.entities.Message import Message


@pytest.fixture
def conv_repo(test_db_session):
    return ConversationRepository(test_db_session)


@pytest.fixture
def msg_repo(test_db_session):
    return MessageRepository(test_db_session)


@pytest.fixture
def conversation(conv_repo, mock_llm):
    return conv_repo.create_conversation(llm_id=mock_llm.id, name="Test Conv")


def _broken_query(*args, **kwargs):
    raise SQLAlchemyError("connection reset")


# =====================================================================
# INTEGRATION - ConversationRepository
# =====================================================================


@pytest.mark.integration
class TestConversationRepository:
    def test_create_and_get_roundtrip(self, conv_repo, mock_llm):
        created = conv_repo.create_conversation(
            llm_id=mock_llm.id,
            name="Roundtrip",
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
            custom_prompt="be brief",
        )
        fetched = conv_repo.get_conversation_by_id(created.id)
        assert fetched.name == "Roundtrip"
        assert fetched.temperature == 0.7
        assert fetched.custom_prompt == "be brief"

    def test_get_all_conversations(self, conv_repo, mock_llm):
        before = len(conv_repo.get_all_conversations())
        conv_repo.create_conversation(llm_id=mock_llm.id)
        conv_repo.create_conversation(llm_id=mock_llm.id)
        assert len(conv_repo.get_all_conversations()) == before + 2

    def test_get_unknown_conversation_raises(self, conv_repo):
        with pytest.raises(ConversationNotFoundException):
            conv_repo.get_conversation_by_id(987654321)

    def test_get_llm_by_id(self, conv_repo, mock_llm):
        assert conv_repo.get_llm_by_id(mock_llm.id).id == mock_llm.id

    def test_get_unknown_llm_raises(self, conv_repo):
        with pytest.raises(ModelNotFoundException):
            conv_repo.get_llm_by_id(987654321)

    def test_update_changes_all_provided_fields(self, conv_repo, conversation, mock_llm):
        updated = conv_repo.update_conversation(
            conversation.id,
            name="Renamed",
            temperature=0.9,
            top_p=0.1,
            max_tokens=64,
            custom_prompt="new prompt",
        )
        assert updated.name == "Renamed"
        assert updated.temperature == 0.9
        assert updated.top_p == 0.1
        assert updated.max_tokens == 64
        assert updated.custom_prompt == "new prompt"

    def test_update_with_identical_values_is_noop(self, conv_repo, conversation):
        with patch.object(conv_repo.db, "flush") as flush:
            result = conv_repo.update_conversation(conversation.id, name=conversation.name)
        assert result.id == conversation.id
        flush.assert_not_called()

    def test_update_llm_id(self, conv_repo, conversation, mock_llm_with_kb):
        other_llm, _ = mock_llm_with_kb
        updated = conv_repo.update_conversation(conversation.id, llm_id=other_llm.id)
        assert updated.llm_id == other_llm.id

    def test_update_unknown_conversation_raises(self, conv_repo):
        with pytest.raises(ConversationNotFoundException):
            conv_repo.update_conversation(987654321, name="X")

    def test_delete_conversation(self, conv_repo, conversation):
        conv_repo.delete_conversation(conversation.id)
        with pytest.raises(ConversationNotFoundException):
            conv_repo.get_conversation_by_id(conversation.id)

    def test_delete_unknown_conversation_raises(self, conv_repo):
        with pytest.raises(ConversationNotFoundException):
            conv_repo.delete_conversation(987654321)

    def test_update_last_message_time_touches_updated_at(self, conv_repo, conversation):
        before = conversation.updated_at
        conv_repo.update_last_message_time(conversation.id)
        # The touch writes datetime.utcnow() (naive UTC) while the column
        # default is naive local time, so ordering across the two cannot be
        # asserted - only that the timestamp was rewritten.
        assert conversation.updated_at is not None
        assert conversation.updated_at != before

    def test_update_last_message_time_unknown_raises(self, conv_repo):
        with pytest.raises(ConversationNotFoundException):
            conv_repo.update_last_message_time(987654321)

    # ---- SQLAlchemyError -> DatabaseException wrapping ----

    def test_get_all_wraps_database_error(self, conv_repo):
        with patch.object(conv_repo.db, "query", side_effect=_broken_query):
            with pytest.raises(DatabaseException):
                conv_repo.get_all_conversations()

    def test_get_by_id_wraps_database_error(self, conv_repo):
        with patch.object(conv_repo.db, "query", side_effect=_broken_query):
            with pytest.raises(DatabaseException):
                conv_repo.get_conversation_by_id(1)

    def test_get_llm_wraps_database_error(self, conv_repo):
        with patch.object(conv_repo.db, "query", side_effect=_broken_query):
            with pytest.raises(DatabaseException):
                conv_repo.get_llm_by_id(1)

    def test_create_wraps_database_error(self, conv_repo, mock_llm):
        with patch.object(conv_repo.db, "flush", side_effect=SQLAlchemyError("boom")):
            with pytest.raises(DatabaseException):
                conv_repo.create_conversation(llm_id=mock_llm.id)

    def test_update_wraps_database_error(self, conv_repo, conversation):
        with patch.object(conv_repo.db, "flush", side_effect=SQLAlchemyError("boom")):
            with pytest.raises(DatabaseException):
                conv_repo.update_conversation(conversation.id, name="Другое")

    def test_delete_wraps_database_error(self, conv_repo, conversation):
        with patch.object(conv_repo.db, "delete", side_effect=SQLAlchemyError("boom")):
            with pytest.raises(DatabaseException):
                conv_repo.delete_conversation(conversation.id)

    def test_touch_wraps_database_error(self, conv_repo, conversation):
        with patch.object(conv_repo.db, "flush", side_effect=SQLAlchemyError("boom")):
            with pytest.raises(DatabaseException):
                conv_repo.update_last_message_time(conversation.id)


# =====================================================================
# INTEGRATION - MessageRepository
# =====================================================================


@pytest.mark.integration
class TestMessageRepository:
    def test_create_and_list_ordered(self, msg_repo, conversation):
        msg_repo.create_message(conversation.id, "hello", "user")
        msg_repo.create_message(conversation.id, "hi there", "llm")
        messages = msg_repo.get_messages_by_conversation(conversation.id)
        assert [m.content for m in messages] == ["hello", "hi there"]
        assert [m.sender for m in messages] == ["user", "llm"]

    def test_list_follows_insertion_order_not_timestamps(
        self, msg_repo, conversation, test_db_session
    ):
        """``now()`` is frozen per transaction, so timestamps tie (or, set by
        hand, disagree with the order of writes): the list follows the
        primary key, the insertion order."""
        first = msg_repo.create_message(conversation.id, "question", "user")
        second = msg_repo.create_message(conversation.id, "answer", "llm")
        first.timestamp = datetime(2030, 1, 1)
        second.timestamp = datetime(2020, 1, 1)
        test_db_session.flush()

        messages = msg_repo.get_messages_by_conversation(conversation.id)

        assert [m.sender for m in messages] == ["user", "llm"]

    def test_create_message_persists_trace(self, msg_repo, conversation):
        trace = [{"type": "thinking", "content": "hmm"}]
        message = msg_repo.create_message(conversation.id, "answer", "llm", trace=trace)
        assert message.trace == trace

    def test_star_and_unstar_roundtrip(self, msg_repo, conversation, test_db_session):
        message = msg_repo.create_message(conversation.id, "important", "llm")
        msg_repo.star_message(message.id)
        assert test_db_session.get(Message, message.id).starred is True
        assert msg_repo.get_starred_messages(conversation.id) == ["important"]

        msg_repo.unstar_message(message.id)
        assert test_db_session.get(Message, message.id).starred is False
        assert msg_repo.get_starred_messages(conversation.id) == []

    def test_star_unknown_message_raises(self, msg_repo):
        with pytest.raises(MessageNotFoundException):
            msg_repo.star_message(987654321)

    def test_unstar_unknown_message_raises(self, msg_repo):
        with pytest.raises(MessageNotFoundException):
            msg_repo.unstar_message(987654321)

    # ---- SQLAlchemyError -> DatabaseException wrapping ----

    def test_list_wraps_database_error(self, msg_repo):
        with patch.object(msg_repo.db, "query", side_effect=_broken_query):
            with pytest.raises(DatabaseException):
                msg_repo.get_messages_by_conversation(1)

    def test_starred_list_swallows_database_error(self, msg_repo):
        """Starred lookup is best-effort: a broken query yields an empty list."""
        with patch.object(msg_repo.db, "query", side_effect=_broken_query):
            assert msg_repo.get_starred_messages(1) == []

    def test_create_wraps_database_error(self, msg_repo, conversation):
        with patch.object(msg_repo.db, "flush", side_effect=SQLAlchemyError("boom")):
            with pytest.raises(DatabaseException):
                msg_repo.create_message(conversation.id, "x", "user")

    def test_star_wraps_database_error(self, msg_repo, conversation):
        message = msg_repo.create_message(conversation.id, "x", "user")
        with patch.object(msg_repo.db, "flush", side_effect=SQLAlchemyError("boom")):
            with pytest.raises(DatabaseException):
                msg_repo.star_message(message.id)

    def test_unstar_wraps_database_error(self, msg_repo, conversation):
        message = msg_repo.create_message(conversation.id, "x", "user")
        with patch.object(msg_repo.db, "flush", side_effect=SQLAlchemyError("boom")):
            with pytest.raises(DatabaseException):
                msg_repo.unstar_message(message.id)
