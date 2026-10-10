"""Stale tool-result placeholdering middleware (#84 KB, generalized for #310 web).

The checkpointer persists every ToolMessage; past turns' bulky results must
shrink to a short per-tool directive marker while the CURRENT turn's results
stay intact. Content is rewritten, never dropped, so the chat template's
``AIMessage(tool_calls) -> ToolMessage`` pairing invariant holds.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import pytest

from src.agents.middleware import _StripStaleToolResults

pytestmark = pytest.mark.unit


class _FakeRequest:
    def __init__(self, messages):
        self.messages = messages

    def override(self, messages):
        return _FakeRequest(messages)


def _turn(question, tool_name, tool_text, answer, call_id):
    return [
        HumanMessage(content=question),
        AIMessage(
            content="",
            tool_calls=[{"name": tool_name, "args": {"query": "q"}, "id": call_id}],
        ),
        ToolMessage(content=tool_text, name=tool_name, tool_call_id=call_id),
        AIMessage(content=answer),
    ]


class TestStripStaleToolResults:
    def test_past_kb_results_get_the_kb_marker(self):
        messages = [
            *_turn("q1", "search_knowledge_base", "bulky excerpts", "a1", "c1"),
            HumanMessage(content="q2"),
        ]
        out = _StripStaleToolResults()._strip(_FakeRequest(messages))
        stale = out.messages[2]
        assert stale.type == "tool"
        assert "knowledge base results from an earlier turn omitted" in stale.content
        assert "search_knowledge_base again" in stale.content

    def test_past_web_results_get_the_web_marker(self):
        messages = [
            *_turn("q1", "web_search", "bulky web snippets", "a1", "c1"),
            HumanMessage(content="q2"),
        ]
        out = _StripStaleToolResults()._strip(_FakeRequest(messages))
        stale = out.messages[2]
        assert stale.type == "tool"
        assert "web search results from an earlier turn omitted" in stale.content
        assert "call web_search again" in stale.content
        assert "fresh web facts" in stale.content

    def test_current_turn_results_stay_intact(self):
        messages = [
            *_turn("q1", "web_search", "old snippets", "a1", "c1"),
            *_turn("q2", "web_search", "fresh snippets", "a2", "c2"),
        ]
        out = _StripStaleToolResults()._strip(_FakeRequest(messages))
        assert out.messages[6].content == "fresh snippets"
        assert "omitted" in out.messages[2].content

    def test_mixed_tools_each_get_their_own_marker(self):
        messages = [
            *_turn("q1", "search_knowledge_base", "kb text", "a1", "c1"),
            *_turn("q2", "web_search", "web text", "a2", "c2"),
            HumanMessage(content="q3"),
        ]
        out = _StripStaleToolResults()._strip(_FakeRequest(messages))
        assert "knowledge base results" in out.messages[2].content
        assert "web search results" in out.messages[6].content

    def test_unknown_tools_untouched(self):
        messages = [
            *_turn("q1", "calculator", "4074", "a1", "c1"),
            HumanMessage(content="q2"),
        ]
        out = _StripStaleToolResults()._strip(_FakeRequest(messages))
        assert out.messages[2].content == "4074"

    def test_pairing_invariant_no_message_dropped(self):
        messages = [
            *_turn("q1", "web_search", "old", "a1", "c1"),
            HumanMessage(content="q2"),
        ]
        out = _StripStaleToolResults()._strip(_FakeRequest(messages))
        assert len(out.messages) == len(messages)
        assert [m.type for m in out.messages] == [m.type for m in messages]
        # The AIMessage carrying the tool_calls is untouched.
        assert out.messages[1].tool_calls[0]["id"] == "c1"

    def test_no_human_messages_is_a_noop(self):
        req = _FakeRequest([AIMessage(content="hello")])
        out = _StripStaleToolResults()._strip(req)
        assert out is req


# ===================== the pure function shared with the counters =====================


def _kb_history(call_id="functions.search_knowledge_base:0"):
    """Two KB turns whose parser reuses the SAME call id every turn (mlx_vlm's
    kimi_k2 parser emits ``functions.<name>:0``), then the current question."""
    past = [
        HumanMessage(content="q1", id="h1"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search_knowledge_base", "args": {}, "id": call_id}],
            id="a1",
        ),
        ToolMessage(
            content="PAST " * 100, name="search_knowledge_base", tool_call_id=call_id, id="t1"
        ),
        AIMessage(content="answer one", id="a2"),
    ]
    current = [
        HumanMessage(content="q2", id="h2"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search_knowledge_base", "args": {}, "id": call_id}],
            id="a3",
        ),
        ToolMessage(
            content="CURRENT " * 100, name="search_knowledge_base", tool_call_id=call_id, id="t2"
        ),
    ]
    return past + current


def test_the_pure_function_is_what_the_middleware_applies():
    from src.agents.middleware import strip_stale_tool_results

    messages = _kb_history()
    via_middleware = _StripStaleToolResults()._strip(_FakeRequest(messages)).messages
    assert [m.content for m in strip_stale_tool_results(messages)] == [
        m.content for m in via_middleware
    ]
    assert via_middleware[2].content == _StripStaleToolResults._MARKERS["search_knowledge_base"]
    assert via_middleware[6].content.startswith("CURRENT")


def test_all_past_strips_every_marked_result_including_the_last_turns():
    from src.agents.middleware import strip_stale_tool_results

    out = strip_stale_tool_results(_kb_history(), all_past=True)
    marker = _StripStaleToolResults._MARKERS["search_knowledge_base"]
    assert out[2].content == marker
    assert out[6].content == marker


def test_the_pure_function_never_mutates_its_input():
    from src.agents.middleware import strip_stale_tool_results

    messages = _kb_history()
    strip_stale_tool_results(messages, all_past=True)
    assert messages[2].content.startswith("PAST")
    assert messages[6].content.startswith("CURRENT")


def test_past_results_are_designated_by_message_id_not_call_id():
    """A parser with deterministic call ids reuses them every turn: the past
    set holds ToolMessage ids, so the current result is never marked."""
    from src.agents.middleware import past_tool_result_ids

    assert past_tool_result_ids(_kb_history()) == frozenset({"t1"})


def test_no_user_message_means_no_past_results():
    from src.agents.middleware import past_tool_result_ids

    assert past_tool_result_ids([AIMessage(content="a")]) == frozenset()
