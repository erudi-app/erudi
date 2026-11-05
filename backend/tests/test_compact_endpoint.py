"""``POST /conversations/{id}/compact`` and its service (#585).

The conversation is read without ever re-assigning a model (a conversation
whose model was deleted gets "pick a model first"), and refused while its
last row is a question without an answer. The runner compacts (see
``test_manual_compaction``); the service saves the separator -- a manual
``compaction`` event appended to the LAST answer's trace, never re-capped.
A summary that cannot be written is a 503 and one WARNING; a separator that
cannot be saved after the history was summarized is a 500 and one ERROR.
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.agents.runner import ERROR_MESSAGE
from src.core.exceptions import (
    CompactionFailedException,
    CompactionSeparatorException,
    EngineBusyException,
    StateConflictException,
)
from src.domains.conversations.services import TRACE_MAX_BYTES, ConversationService
from src.entities.Conversation import Conversation
from src.entities.Message import Message
from tests.test_compaction_keep import _ScriptedSummaryModel, _seed, _state
from tests.test_manual_compaction import _BrakeEngine, _Builds, _engine  # noqa: F401

pytestmark = pytest.mark.unit


def _exchanges(n: int) -> list:
    out = []
    for i in range(1, n + 1):
        out += [HumanMessage(f"q{i}", id=f"h{i}"), AIMessage(f"a{i}", id=f"a{i}")]
    return out


def _rows(db, conversation_id, n: int, *, last_content: str = "a", last_trace=None):
    """``n`` persisted exchanges; the last answer carries ``last_content``
    and ``last_trace``."""
    for i in range(1, n + 1):
        db.add(Message(conversation_id=conversation_id, sender="user", content=f"q{i}"))
        last = i == n
        db.add(
            Message(
                conversation_id=conversation_id,
                sender="llm",
                content=last_content if last else f"a{i}",
                trace=last_trace if last else [{"t": "thinking", "text": "earlier"}],
            )
        )
    db.commit()


def _answers(db, conversation_id) -> list:
    return (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id, Message.sender == "llm")
        .order_by(Message.id)
        .all()
    )


@pytest.fixture
def setup(test_db_session, mock_llm, monkeypatch):
    """A conversation with two exchanges, in SQL and in the checkpoint."""
    checkpointer = InMemorySaver()
    service = ConversationService(test_db_session, checkpointer)
    conversation = service.create_conversation(llm_id=mock_llm.id)
    test_db_session.commit()
    _rows(test_db_session, conversation.id, 2)
    builds = _Builds(
        monkeypatch, _ScriptedSummaryModel(messages=iter([]), script=["Ada lives in Lyon."])
    )
    return service, conversation, checkpointer, builds


# ===================== the service =====================


async def test_a_compaction_saves_its_separator_on_the_last_answer(setup, test_db_session):
    service, conversation, cp, _builds = setup
    await _seed(cp, str(conversation.id), _exchanges(2))

    result = await service.compact_conversation(conversation.id)

    assert result == {"summarized": 1, "reason": "length", "summary": "Ada lives in Lyon."}
    first, last = _answers(test_db_session, conversation.id)
    assert first.trace == [{"t": "thinking", "text": "earlier"}]
    assert last.trace == [
        {
            "t": "compaction",
            "phase": "done",
            "manual": True,
            "reason": "length",
            "summarized": 1,
            "summary": "Ada lives in Lyon.",
            "summary_lost": False,
        }
    ]
    state = await _state(cp, str(conversation.id))
    assert [m.id for m in state[1:]] == ["a1", "h2", "a2"]


async def test_the_separator_is_appended_without_recapping_the_trace(
    test_db_session, mock_llm, monkeypatch
):
    cp = InMemorySaver()
    service = ConversationService(test_db_session, cp)
    conversation = service.create_conversation(llm_id=mock_llm.id)
    test_db_session.commit()
    full = [{"t": "thinking", "text": "x" * 1000} for _ in range(33)]
    assert len(json.dumps(full)) > TRACE_MAX_BYTES
    _rows(test_db_session, conversation.id, 2, last_trace=full)
    _Builds(monkeypatch, _ScriptedSummaryModel(messages=iter([])))
    await _seed(cp, str(conversation.id), _exchanges(2))

    await service.compact_conversation(conversation.id)

    last = _answers(test_db_session, conversation.id)[-1]
    assert last.trace[:-1] == full
    assert last.trace[-1]["manual"] is True


async def test_after_a_failed_turn_the_separator_goes_on_the_error_turn(
    test_db_session, mock_llm, monkeypatch
):
    cp = InMemorySaver()
    service = ConversationService(test_db_session, cp)
    conversation = service.create_conversation(llm_id=mock_llm.id)
    test_db_session.commit()
    _rows(test_db_session, conversation.id, 2, last_content=ERROR_MESSAGE)
    _Builds(monkeypatch, _ScriptedSummaryModel(messages=iter([])))
    # The thread state as the runner's repair closed the failed turn.
    await _seed(
        cp,
        str(conversation.id),
        [*_exchanges(1), HumanMessage("q2", id="h2"), AIMessage(ERROR_MESSAGE, id="err")],
    )

    result = await service.compact_conversation(conversation.id)

    assert result["summarized"] == 1
    assert _answers(test_db_session, conversation.id)[-1].trace[-1]["manual"] is True


async def test_nothing_to_compact_loads_nothing_and_stores_nothing(
    test_db_session, mock_llm, monkeypatch
):
    cp = InMemorySaver()
    service = ConversationService(test_db_session, cp)
    conversation = service.create_conversation(llm_id=mock_llm.id)
    test_db_session.commit()
    _rows(test_db_session, conversation.id, 1)
    builds = _Builds(monkeypatch, _ScriptedSummaryModel(messages=iter([])))
    await _seed(cp, str(conversation.id), _exchanges(1))

    assert await service.compact_conversation(conversation.id) == {"summarized": 0}

    assert builds.calls == []
    assert _answers(test_db_session, conversation.id)[-1].trace is None


async def test_an_unanswered_last_question_is_refused(setup, test_db_session):
    service, conversation, cp, builds = setup
    test_db_session.add(Message(conversation_id=conversation.id, sender="user", content="q3"))
    test_db_session.commit()
    await _seed(cp, str(conversation.id), [*_exchanges(2), HumanMessage("q3", id="h3")])

    with pytest.raises(StateConflictException) as raised:
        await service.compact_conversation(conversation.id)

    assert raised.value.status_code == 409
    assert raised.value.detail == {"reason": "unanswered_question"}
    assert builds.calls == []


async def test_a_conversation_without_a_model_is_refused_and_never_rebound(
    setup, test_db_session, monkeypatch
):
    service, conversation, cp, builds = setup
    test_db_session.query(Conversation).filter(Conversation.id == conversation.id).update(
        {"llm_id": None}
    )
    test_db_session.commit()
    await _seed(cp, str(conversation.id), _exchanges(2))
    commits: list = []
    monkeypatch.setattr(test_db_session, "commit", lambda: commits.append(1))

    with pytest.raises(StateConflictException) as raised:
        await service.compact_conversation(conversation.id)

    assert raised.value.status_code == 409
    assert raised.value.detail == {"reason": "no_model"}
    assert "pick a model" in raised.value.message
    assert commits == []
    test_db_session.expire_all()
    assert test_db_session.get(Conversation, conversation.id).llm_id is None
    assert builds.calls == []


async def test_a_summary_that_cannot_be_written_changes_nothing(
    test_db_session, mock_llm, monkeypatch
):
    cp = InMemorySaver()
    service = ConversationService(test_db_session, cp)
    conversation = service.create_conversation(llm_id=mock_llm.id)
    test_db_session.commit()
    _rows(test_db_session, conversation.id, 2)
    _Builds(
        monkeypatch,
        _ScriptedSummaryModel(messages=iter([]), script=[ValueError("x"), ValueError("y")]),
    )
    await _seed(cp, str(conversation.id), _exchanges(2))

    with pytest.raises(CompactionFailedException):
        await service.compact_conversation(conversation.id)

    assert _answers(test_db_session, conversation.id)[-1].trace is None
    assert [m.id for m in await _state(cp, str(conversation.id))] == ["h1", "a1", "h2", "a2"]


async def test_a_separator_that_cannot_be_saved_after_the_checkpoint(
    setup, test_db_session, monkeypatch
):
    service, conversation, cp, _builds = setup
    await _seed(cp, str(conversation.id), _exchanges(2))

    def _broken_commit():
        raise RuntimeError("database gone")

    monkeypatch.setattr(test_db_session, "commit", _broken_commit)

    with pytest.raises(CompactionSeparatorException) as raised:
        await service.compact_conversation(conversation.id)

    assert isinstance(raised.value.__cause__, RuntimeError)
    state = await _state(cp, str(conversation.id))
    assert state[0].additional_kwargs.get("lc_source") == "summarization"


async def test_a_busy_engine_is_refused(setup, monkeypatch):
    from src.engines import base_engine

    service, conversation, cp, builds = setup
    monkeypatch.setattr(base_engine, "GENERATION_BUSY_WAIT_S", 0.05)
    await _seed(cp, str(conversation.id), _exchanges(2))

    async with _BrakeEngine.generation_guard():
        with pytest.raises(EngineBusyException):
            await service.compact_conversation(conversation.id)

    assert builds.calls == []


# ===================== the endpoint =====================


def _client_conversation(client, test_db_session, mock_llm, exchanges=2):
    conversation_id = client.post("/erudi/conversations/", json={"llm_id": mock_llm.id}).json()[
        "id"
    ]
    _rows(test_db_session, conversation_id, exchanges)
    asyncio.run(_seed(client.app.state.checkpointer, str(conversation_id), _exchanges(exchanges)))
    return conversation_id


def test_the_endpoint_answers_with_what_was_summarized(client, test_db_session, mock_llm, setup):
    conversation_id = _client_conversation(client, test_db_session, mock_llm)

    response = client.post(f"/erudi/conversations/{conversation_id}/compact")

    assert response.status_code == 200
    assert response.json() == {
        "summarized": 1,
        "reason": "length",
        "summary": "Ada lives in Lyon.",
    }


def test_the_endpoint_with_nothing_to_compact(client, test_db_session, mock_llm, setup):
    conversation_id = _client_conversation(client, test_db_session, mock_llm, exchanges=1)

    response = client.post(f"/erudi/conversations/{conversation_id}/compact")

    assert response.status_code == 200
    assert response.json() == {"summarized": 0}


def test_the_endpoint_on_an_unknown_conversation_is_404(client):
    assert client.post("/erudi/conversations/999999/compact").status_code == 404


def test_the_endpoint_says_to_pick_a_model_first(client, test_db_session, mock_llm, setup):
    conversation_id = _client_conversation(client, test_db_session, mock_llm)
    test_db_session.query(Conversation).filter(Conversation.id == conversation_id).update(
        {"llm_id": None}
    )
    test_db_session.commit()

    response = client.post(f"/erudi/conversations/{conversation_id}/compact")

    assert response.status_code == 409
    error = response.json()["error"]
    assert error["type"] == "STATE_CONFLICT"
    assert error["detail"] == {"reason": "no_model"}


def test_a_busy_engine_is_a_409_logged_at_info(client, monkeypatch, caplog):
    async def _busy(self, conversation_id):
        raise EngineBusyException()

    monkeypatch.setattr(ConversationService, "compact_conversation", _busy)

    with caplog.at_level(logging.INFO):
        response = client.post("/erudi/conversations/1/compact")

    assert response.status_code == 409
    assert response.json()["error"]["type"] == "ENGINE_BUSY"
    assert response.json()["error"]["message"] == "The model is busy; try again in a moment."
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_failed_summary_is_a_503_with_one_warning(client, monkeypatch, caplog):
    async def _failed(self, conversation_id):
        raise CompactionFailedException(trace="APIConnectionError")

    monkeypatch.setattr(ConversationService, "compact_conversation", _failed)

    with caplog.at_level(logging.INFO):
        response = client.post("/erudi/conversations/5/compact")

    assert response.status_code == 503
    assert response.json()["error"]["type"] == "COMPACTION_FAILED"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "APIConnectionError" in warnings[0].getMessage()
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_a_lost_separator_is_a_500_with_one_error_and_its_traceback(client, monkeypatch, caplog):
    async def _lost(self, conversation_id):
        try:
            raise RuntimeError("database gone")
        except RuntimeError as exc:
            raise CompactionSeparatorException() from exc

    monkeypatch.setattr(ConversationService, "compact_conversation", _lost)

    with caplog.at_level(logging.INFO):
        response = client.post("/erudi/conversations/5/compact")

    assert response.status_code == 500
    error = response.json()["error"]
    assert error["type"] == "COMPACTION_SEPARATOR_NOT_SAVED"
    assert "summarized" in error["message"]
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert errors[0].exc_info is not None
