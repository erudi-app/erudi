"""The projection of the next request, one pure function (#608).

The amber memory warning and the ``/context`` panel count the same thing:
what the NEXT request of a conversation carries, in real tokens -- the
system prompt and tool schemas (O), the previous summary and the messages,
with past knowledge-base and web results sent as their markers. The
projection is ``project_next_request``; ``AgentRunner._project_next_request``
only reads the thread state and calls it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agents.runner import (
    AgentRunner,
    compaction_cutoff,
    frozen_weights,
    project_next_request,
    real_token_count,
    summary_cap,
)
from src.agents.token_accounting import (
    COUNTER_DENSE_TOKENS,
    COUNTER_WEIGHT_FLOOR,
    STALE_TOOL_RESULT_MARKERS,
    overhead_tokens,
    request_overhead,
)
from tests.test_compaction_keep import _alternating, _previous_summary_message, _text

pytestmark = pytest.mark.unit

_OVERHEAD = request_overhead("You are a helpful assistant. " * 20)


def _agent_with_state(messages):
    async def aget_state(config):
        return SimpleNamespace(values={"messages": messages})

    return SimpleNamespace(aget_state=aget_state)


def _states():
    kb_round = [
        HumanMessage("what does the doc say?"),
        AIMessage(
            content="",
            tool_calls=[{"name": "search_knowledge_base", "args": {}, "id": "k1"}],
        ),
        ToolMessage(_text(2000), tool_call_id="k1", name="search_knowledge_base"),
        AIMessage("It says so."),
    ]
    return {
        "one-exchange": [HumanMessage("hi"), AIMessage("hello")],
        "long": _alternating(30, 300),
        "summary": [_previous_summary_message(), AIMessage(_text(300))]
        + [HumanMessage("q"), AIMessage(_text(400))],
        "kb-round": kb_round,
        "summary-long": [_previous_summary_message()] + _alternating(30, 400)[1:],
    }


@pytest.mark.parametrize("name", list(_states()))
async def test_the_warning_projection_is_the_pure_function(name):
    state = _states()[name]
    via_method = await AgentRunner()._project_next_request(
        _agent_with_state(state), {}, 10_000, overhead=_OVERHEAD, compaction_threshold=80
    )
    pure = project_next_request(
        state, 10_000, overhead=_OVERHEAD, compaction_threshold=80, allocated_window=None
    )
    assert via_method == (pure.projected_tokens, pure.conversation_tokens)


async def test_the_method_still_says_nothing_on_an_empty_thread():
    assert (
        await AgentRunner()._project_next_request(
            _agent_with_state([]), {}, 10_000, compaction_threshold=80
        )
        is None
    )


@pytest.mark.parametrize("name", list(_states()))
def test_the_categories_add_up_to_the_request(name):
    pure = project_next_request(
        _states()[name], 10_000, overhead=_OVERHEAD, compaction_threshold=80
    )
    assert (
        pure.overhead_tokens + pure.summary_tokens + pure.message_tokens == pure.conversation_tokens
    )


def test_the_summary_is_never_inside_the_messages():
    state = _states()["summary"]
    pure = project_next_request(state, 10_000, overhead=_OVERHEAD, compaction_threshold=80)
    projected = list(state) + [HumanMessage("", id="erudi-next-request")]
    weights, _ = frozen_weights(projected)
    assert pure.summary_tokens == real_token_count([state[0]], weights)
    assert pure.summary_tokens > 0
    assert pure.message_tokens == real_token_count(projected[1:], weights)


def test_without_a_summary_the_summary_counts_zero():
    pure = project_next_request(_states()["long"], 10_000, compaction_threshold=80)
    assert pure.summary_tokens == 0


def test_a_past_tool_result_counts_as_its_marker():
    state = _states()["kb-round"]
    pure = project_next_request(state, 10_000, compaction_threshold=80)
    marker = STALE_TOOL_RESULT_MARKERS["search_knowledge_base"]
    as_sent = [*state[:2], state[2].model_copy(update={"content": marker}), state[3]]
    as_sent.append(HumanMessage("", id="erudi-next-request"))
    weights, _ = frozen_weights(as_sent)
    assert pure.message_tokens == real_token_count(as_sent, weights)
    assert pure.message_tokens < 500


def test_the_overhead_is_o_in_real_tokens_at_the_states_ratio():
    state = _states()["long"]
    pure = project_next_request(state, 10_000, overhead=_OVERHEAD, compaction_threshold=80)
    projected = list(state) + [HumanMessage("", id="erudi-next-request")]
    _, ratio = frozen_weights(projected)
    assert pure.ratio == ratio
    assert pure.overhead_tokens == overhead_tokens(
        _OVERHEAD, ratio, dense=COUNTER_DENSE_TOKENS, weight_floor=COUNTER_WEIGHT_FLOOR
    )


def test_a_compaction_the_next_request_would_run_is_projected():
    state = _states()["summary-long"]
    pure = project_next_request(state, 10_000, overhead=_OVERHEAD, compaction_threshold=80)
    projected = list(state) + [HumanMessage("", id="erudi-next-request")]
    weights, _ = frozen_weights(projected)

    def counter(part):
        return real_token_count(part, weights)

    cut = compaction_cutoff(
        projected, 10_000, counter, pct=80, overhead=pure.overhead_tokens, allocated_window=None
    )
    assert cut > 0
    assert pure.projected_tokens == (
        pure.overhead_tokens + counter(projected[cut:]) + summary_cap(10_000)
    )


def test_an_unknown_window_projects_the_whole_request():
    pure = project_next_request(_states()["long"], None, compaction_threshold=80)
    assert pure.projected_tokens == pure.conversation_tokens


def test_an_empty_thread_costs_the_overhead_and_the_empty_question():
    pure = project_next_request([], 10_000, overhead=_OVERHEAD, compaction_threshold=80)
    assert pure.summary_tokens == 0
    assert pure.message_tokens == real_token_count([HumanMessage("", id="erudi-next-request")])
    assert pure.conversation_tokens == pure.overhead_tokens + pure.message_tokens
