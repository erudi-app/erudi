"""The cutoff of a manual compaction (``/compact``, #585).

The same pure rule as the automatic compaction (``compaction_cutoff``), with
a keep budget of ONE token, no futility guard and never the message-count
branch (an unknown window keeps the same one-token rule). The kept tail then
starts with the answer before the last question --
``[summary, previous answer, last question, last answer]`` -- because the
summary is inserted as a user message, and keeping the last question alone
would put two user messages in a row. Nothing to compact: a single exchange,
or nothing new since the last manual compaction.
"""

from __future__ import annotations

import inspect
import random

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately as approx_token_count

from src.agents.runner import (
    ERROR_MESSAGE,
    MANUAL_KEEP_BUDGET_TOKENS,
    _closing_messages,
    compaction_cutoff,
    manual_compaction_cutoff,
)
from tests.test_compaction_keep import _previous_summary_message, _text

pytestmark = pytest.mark.unit


def _call(call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": call_id}])


def _after(messages, cut):
    """The state a compaction at ``cut`` leaves: the new summary, then the kept tail."""
    return [_previous_summary_message("new")] + list(messages[cut:])


def _assert_strictly_alternates(state):
    """User and assistant turns alternate: tool results follow their call,
    and never two user messages (nor two assistant answers) in a row."""
    roles = [m.type for m in state if m.type != "tool"]
    assert roles[0] == "human"
    for first, second in zip(roles, roles[1:]):
        assert first != second or first == "ai", f"two {first} messages in a row: {roles}"
    for prev, current in zip(state, state[1:]):
        if current.type == "tool":
            assert prev.type == "tool" or getattr(prev, "tool_calls", None)


def _manual(messages):
    return manual_compaction_cutoff(messages, 80)


def test_the_manual_keep_budget_is_one_token():
    assert MANUAL_KEEP_BUDGET_TOKENS == 1


def test_the_default_callers_keep_todays_rule():
    params = inspect.signature(compaction_cutoff).parameters
    assert params["keep_budget"].default is None
    assert params["futility"].default is True


def test_two_exchanges_keep_the_answer_before_the_last_question():
    messages = [
        HumanMessage("h1", id="h1"),
        AIMessage("a1", id="a1"),
        HumanMessage("h2", id="h2"),
        AIMessage("a2", id="a2"),
    ]
    cut = _manual(messages)
    assert cut == 1
    assert [m.id for m in messages[cut:]] == ["a1", "h2", "a2"]
    _assert_strictly_alternates(_after(messages, cut))


def test_a_single_exchange_has_nothing_to_compact():
    assert _manual([HumanMessage("h1"), AIMessage("a1")]) == 0


def test_an_empty_thread_has_nothing_to_compact():
    assert _manual([]) == 0


def test_right_after_a_manual_compaction_there_is_nothing_to_compact():
    state = [_previous_summary_message(), AIMessage("a1"), HumanMessage("h2"), AIMessage("a2")]
    assert _manual(state) == 0


def test_a_summary_plus_two_exchanges_compacts_the_first_one():
    state = [
        _previous_summary_message(),
        AIMessage("a1"),
        HumanMessage("h2"),
        AIMessage("a2", id="a2"),
        HumanMessage("h3"),
        AIMessage("a3"),
    ]
    cut = _manual(state)
    assert cut == 3
    assert state[cut].id == "a2"
    _assert_strictly_alternates(_after(state, cut))


def test_a_tool_round_in_the_last_turn_stays_with_it():
    messages = [
        HumanMessage("h1"),
        AIMessage("a1", id="a1"),
        HumanMessage("h2"),
        _call("c2"),
        ToolMessage("result", tool_call_id="c2"),
        AIMessage("a2"),
    ]
    cut = _manual(messages)
    assert cut == 1
    assert messages[cut].id == "a1"
    _assert_strictly_alternates(_after(messages, cut))


def test_a_tool_round_of_an_earlier_turn_is_summarized_with_its_call():
    messages = [
        HumanMessage("h1"),
        _call("c1"),
        ToolMessage(_text(3000), tool_call_id="c1"),
        AIMessage("a1", id="a1"),
        HumanMessage("h2"),
        AIMessage("a2"),
    ]
    cut = _manual(messages)
    assert cut == 3
    assert messages[cut].id == "a1"
    _assert_strictly_alternates(_after(messages, cut))


def test_a_previous_summary_is_summarized_again_with_the_turns_after_it():
    state = [
        _previous_summary_message(),
        AIMessage("a1"),
        HumanMessage("h2"),
        _call("c2"),
        ToolMessage("result", tool_call_id="c2"),
        AIMessage("a2", id="a2"),
        HumanMessage("h3"),
        AIMessage("a3"),
    ]
    cut = _manual(state)
    assert state[cut].id == "a2"
    _assert_strictly_alternates(_after(state, cut))


@pytest.mark.parametrize(
    "open_turn",
    [
        [HumanMessage("h2")],
        [HumanMessage("h2"), _call("c2")],
        [HumanMessage("h2"), _call("c2"), ToolMessage("partial", tool_call_id="c2")],
    ],
)
def test_after_a_failed_turn_closed_by_the_repair_the_failed_turn_is_kept(open_turn):
    committed = [HumanMessage("h1"), AIMessage("a1", id="a1"), *open_turn]
    repaired = committed + _closing_messages(committed, ERROR_MESSAGE)
    cut = _manual(repaired)
    assert cut == 1
    assert repaired[cut].id == "a1"
    assert repaired[-1].content == ERROR_MESSAGE
    _assert_strictly_alternates(_after(repaired, cut))


def test_an_unknown_window_never_takes_the_message_count_branch():
    messages = []
    for i in range(15):
        messages += [HumanMessage(f"h{i}"), AIMessage(f"a{i}", id=f"a{i}")]
    # The window-less fallback keeps the last ten messages (moved forward off
    # the user message they start on); the manual rule keeps the answer
    # before the last question.
    automatic = compaction_cutoff(messages, None, approx_token_count, pct=80)
    assert automatic == len(messages) - 9
    cut = compaction_cutoff(
        messages, None, approx_token_count, pct=80, keep_budget=1, futility=False
    )
    assert cut == len(messages) - 3
    assert messages[cut].id == "a13"


def test_the_futility_guard_never_stands_in_the_way():
    # The automatic rule skips summarizing a tiny head over the trigger; the
    # user asked for it, so the manual rule takes it.
    messages = [
        HumanMessage("hi"),
        AIMessage("yo"),
        HumanMessage(_text(9000)),
        AIMessage("short"),
    ]
    assert compaction_cutoff(messages, 10_000, approx_token_count, pct=80) == 0
    cut = compaction_cutoff(
        messages, 10_000, approx_token_count, pct=80, keep_budget=1, futility=False
    )
    assert cut == 1


def test_a_large_last_exchange_is_kept_whole():
    messages = [
        HumanMessage("h1"),
        AIMessage("a1", id="a1"),
        HumanMessage(_text(20_000)),
        AIMessage(_text(20_000)),
    ]
    assert _manual(messages) == 1


def _closed_random_conversation(rng):
    """A thread as a closed turn leaves it: alternating exchanges with
    random tool rounds, ending with an answer, sometimes after a summary."""
    out = []
    if rng.random() < 0.5:
        out += [_previous_summary_message(), AIMessage(_text(rng.randint(1, 300)))]
    call = 0
    for _ in range(rng.randint(1, 12)):
        out.append(HumanMessage(_text(rng.randint(1, 300))))
        for _ in range(rng.choice([0, 0, 1, 2])):
            ids = [f"c{call + k}" for k in range(rng.randint(1, 3))]
            call += len(ids)
            out.append(
                AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": i} for i in ids])
            )
            out.extend(ToolMessage(_text(rng.randint(1, 600)), tool_call_id=i) for i in ids)
        out.append(AIMessage(_text(rng.randint(1, 300))))
    return out


def test_a_manual_compaction_always_leaves_strictly_alternating_roles():
    rng = random.Random(585)
    for _ in range(400):
        messages = _closed_random_conversation(rng)
        cut = _manual(messages)
        if cut == 0:
            continue
        assert messages[cut].type == "ai" and not messages[cut].tool_calls
        last_human = max(i for i, m in enumerate(messages) if m.type == "human")
        # The kept tail is exactly: the answer before the last question onward.
        before = [
            i
            for i, m in enumerate(messages[:last_human])
            if m.type == "ai" and not getattr(m, "tool_calls", None)
        ]
        assert cut == before[-1]
        _assert_strictly_alternates(_after(messages, cut))
