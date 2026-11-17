"""Pre-starting the KB embedding child when a conversation opens (#594).

The decision: the embedder is started ahead when a conversation with a
Knowledge Base opens, so the first KB use of the session does not pay the
spawn. The hook sits on the two endpoints the frontend calls at that moment
— ``POST /conversations/`` (a new KB-assistant conversation) and
``GET /conversations/{id}`` (opening an existing one) — and must stay
invisible: fire-and-forget, never a request failure.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def prewarm_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "src.domains.conversations.endpoints.prewarm_embedder", lambda: calls.append(1)
    )
    return calls


def _create_conversation(client, llm_id):
    response = client.post(
        "/erudi/conversations/",
        json={"llm_id": llm_id, "temperature": 0.7, "top_p": 0.9, "max_tokens": 512},
    )
    assert response.status_code == 201
    return response.json()["id"]


class TestPrewarmOnConversationOpen:
    def test_creating_a_kb_conversation_prewarms(self, client, mock_llm_with_kb, prewarm_calls):
        llm, _kb = mock_llm_with_kb
        _create_conversation(client, llm.id)
        assert len(prewarm_calls) == 1

    def test_opening_a_kb_conversation_prewarms(self, client, mock_llm_with_kb, prewarm_calls):
        llm, _kb = mock_llm_with_kb
        conversation_id = _create_conversation(client, llm.id)
        prewarm_calls.clear()
        response = client.get(f"/erudi/conversations/{conversation_id}")
        assert response.status_code == 200
        assert len(prewarm_calls) == 1

    def test_a_conversation_without_kb_does_not_prewarm(self, client, mock_llm, prewarm_calls):
        conversation_id = _create_conversation(client, mock_llm.id)
        response = client.get(f"/erudi/conversations/{conversation_id}")
        assert response.status_code == 200
        assert prewarm_calls == []

    def test_a_prewarm_failure_never_fails_the_request(self, client, mock_llm_with_kb, monkeypatch):
        def boom():
            raise RuntimeError("prewarm exploded")

        monkeypatch.setattr("src.domains.conversations.endpoints.prewarm_embedder", boom)
        llm, _kb = mock_llm_with_kb
        conversation_id = _create_conversation(client, llm.id)
        response = client.get(f"/erudi/conversations/{conversation_id}")
        assert response.status_code == 200
