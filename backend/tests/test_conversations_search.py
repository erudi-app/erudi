"""Search across the conversation history (#429).

The route matches the title or the text of a message, case-insensitively, with
the user's ``%`` / ``_`` / ``\\`` taken literally; the model's reasoning (the
``trace`` column) and the failed turns never match; the answer is bounded.
"""

import pytest
from fastapi import status

from src.domains.conversations.repository import (
    SEARCH_LIMIT,
    ConversationRepository,
    MessageRepository,
)

_SENTINEL = "[ERROR_MESSAGE_SYSTEM]"
SEARCH = "/erudi/conversations/search"


@pytest.fixture
def make_conversation(test_db_session, mock_llm):
    """Create a conversation with a title and (sender, content[, trace]) messages."""
    conversations = ConversationRepository(test_db_session)
    messages = MessageRepository(test_db_session)

    def _make(name, *turns):
        conv = conversations.create_conversation(llm_id=mock_llm.id, name=name)
        for sender, content, *rest in turns:
            messages.create_message(conv.id, content, sender, rest[0] if rest else None)
        test_db_session.flush()
        return conv

    return _make


def _ids(response):
    assert response.status_code == status.HTTP_200_OK
    return [c["id"] for c in response.json()]


class TestSearchRoute:
    def test_route_is_not_swallowed_by_the_id_route(self, client):
        """`/search` is declared before `/{conversation_id}`: no 422 on the int path param."""
        response = client.get(SEARCH, params={"q": "anything"})
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == []

    def test_matches_a_title_case_insensitively(self, client, make_conversation):
        hit = make_conversation("Quantum Computing Notes")
        make_conversation("Cooking")
        assert _ids(client.get(SEARCH, params={"q": "qUANTUM"})) == [hit.id]

    def test_matches_message_text_of_either_side(self, client, make_conversation):
        by_user = make_conversation("A", ("user", "tell me about zeppelins"))
        by_model = make_conversation("B", ("user", "hi"), ("llm", "Zeppelins are airships"))
        make_conversation("C", ("user", "unrelated"))
        found = _ids(client.get(SEARCH, params={"q": "zeppelin"}))
        assert sorted(found) == sorted([by_user.id, by_model.id])

    def test_a_conversation_appears_once_however_many_messages_match(
        self, client, make_conversation
    ):
        conv = make_conversation("Dup", ("user", "kiwi"), ("llm", "kiwi kiwi"), ("user", "kiwi"))
        assert _ids(client.get(SEARCH, params={"q": "kiwi"})) == [conv.id]

    def test_reasoning_only_text_finds_nothing(self, client, make_conversation):
        make_conversation(
            "Plain title",
            ("user", "What is 2+2?"),
            ("llm", "4", [{"t": "thinking", "text": "secretreasoningtoken here"}]),
        )
        assert _ids(client.get(SEARCH, params={"q": "secretreasoningtoken"})) == []

    def test_failed_turns_never_match(self, client, make_conversation):
        make_conversation(
            "Plain title",
            ("user", "hello"),
            ("llm", f"{_SENTINEL} I apologize, but the zorblax failed"),
        )
        assert _ids(client.get(SEARCH, params={"q": "zorblax"})) == []

    def test_the_failed_turn_filter_takes_the_sentinel_literally(self, client, make_conversation):
        """`_` in the sentinel is not a wildcard: a look-alike text is a real message."""
        lookalike = make_conversation("Plain title", ("llm", "[ERRORXMESSAGEXSYSTEM] zorblax"))
        assert _ids(client.get(SEARCH, params={"q": "zorblax"})) == [lookalike.id]

    @pytest.mark.parametrize(
        "needle, literal, decoy",
        [
            ("100%", "we reached 100% today", "we reached 1000 today"),
            ("a_b", "the a_b variable", "the aXb variable"),
            ("c:\\dir", "open c:\\dir now", "open c:/dir now"),
        ],
    )
    def test_user_wildcards_are_taken_literally(
        self, client, make_conversation, needle, literal, decoy
    ):
        hit = make_conversation("T1", ("user", literal))
        make_conversation("T2", ("user", decoy))
        assert _ids(client.get(SEARCH, params={"q": needle})) == [hit.id]

    def test_a_bare_percent_does_not_match_everything(self, client, make_conversation):
        make_conversation("No wildcard here", ("user", "plain"))
        assert _ids(client.get(SEARCH, params={"q": "%"})) == []

    @pytest.mark.parametrize("q", ["", "   "])
    def test_a_blank_query_matches_nothing(self, client, make_conversation, q):
        make_conversation("Anything", ("user", "text"))
        assert _ids(client.get(SEARCH, params={"q": q})) == []

    def test_a_missing_query_is_a_422(self, client):
        assert client.get(SEARCH).status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_an_oversized_query_is_a_422(self, client):
        response = client.get(SEARCH, params={"q": "x" * 201})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_results_are_bounded_newest_first(self, client, make_conversation):
        made = [make_conversation(f"match {i}") for i in range(SEARCH_LIMIT + 5)]
        ids = _ids(client.get(SEARCH, params={"q": "match"}))
        assert len(ids) == SEARCH_LIMIT
        assert ids == [c.id for c in reversed(made)][:SEARCH_LIMIT]

    def test_the_payload_is_the_history_row_shape(self, client, make_conversation):
        make_conversation("Shape", ("user", "x"))
        row = client.get(SEARCH, params={"q": "shape"}).json()[0]
        assert {"id", "name", "llm_id", "last_message_time"} <= set(row)


class TestHiddenMarkers:
    """A user message stores its attachments as markers the chat never shows
    (``[file_path:...]``, ``[image_path:...]``, ``[image]``): they are not text
    the user can search for."""

    CONTENT = "hello world[file_path:/Users/me/my_report.pdf][image_path:/Users/me/pic.png][image]"

    @pytest.mark.parametrize("needle", ["file", "path", "Users", "pdf", "png", "image", "_", "me/"])
    def test_a_marker_only_term_finds_nothing(self, client, make_conversation, needle):
        make_conversation("Plain title", ("user", self.CONTENT))
        assert _ids(client.get(SEARCH, params={"q": needle})) == []

    def test_the_visible_text_still_matches(self, client, make_conversation):
        conv = make_conversation("Plain title", ("user", self.CONTENT))
        assert _ids(client.get(SEARCH, params={"q": "hello"})) == [conv.id]
        assert _ids(client.get(SEARCH, params={"q": "world"})) == [conv.id]

    def test_a_term_visible_elsewhere_still_matches(self, client, make_conversation):
        conv = make_conversation(
            "Plain title", ("user", "see my file[file_path:/a/report.pdf]"), ("llm", "ok")
        )
        assert _ids(client.get(SEARCH, params={"q": "file"})) == [conv.id]

    def test_an_encoded_bracket_in_a_path_does_not_end_the_marker_early(
        self, client, make_conversation
    ):
        make_conversation("Plain title", ("user", "hi[file_path:/a/we%5Dird.pdf]"))
        assert _ids(client.get(SEARCH, params={"q": "ird"})) == []
