"""A compaction is visible in the conversation (#614).

The compaction middleware announces itself through LangGraph's custom
stream: ``{"t": "compaction", "phase": "start"}`` before the summary call,
then ``{"t": "compaction", "phase": "done", ...}`` once the history is
rewritten -- how many messages a user would recognise were summarized, why
(the conversation's length or the computer's memory), the summary itself and
whether it was lost. The runner turns those into turn events without touching
the message stream; a missing channel is silent and a failing one never fails
the turn.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from src.agents import runner as runner_module
from src.agents.runner import ERROR_SENTINEL, AgentRunner
from src.core import config
from tests.test_compaction_keep import (
    _PARAMS,
    _PREVIOUS,
    _Llm,
    _ScriptedSummaryModel,
    _WindowEngine,
    _after,
    _alternating,
    _conversation,
    _middleware,
    _models,
    _overflow_400,
    _previous_summary_message,
    _request,
    _seed,
    _seed_messages,
    _text,
    _turn,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _engine(monkeypatch):
    _WindowEngine.window = None
    _WindowEngine.claims = []
    _WindowEngine.rewrites = []
    monkeypatch.setattr(config, "LLM_Engine", _WindowEngine)
    yield
    _WindowEngine._last_used = None


class _ProbeSummaryModel(_ScriptedSummaryModel):
    """A summary client that records which announcements had been written
    when it was called."""

    written: list = Field(default_factory=list)
    seen_at_call: list = Field(default_factory=list)

    def _next(self, messages):
        self.seen_at_call.append([e["phase"] for e in self.written])
        return super()._next(messages)


def _runtime(written: list) -> Any:
    return SimpleNamespace(stream_writer=written.append)


def _kinds(events) -> list:
    """``(type, phase)`` per event, consecutive answer chunks collapsed."""
    out: list = []
    for e in events:
        kind = (e["t"], e.get("phase"))
        if not (out and kind == out[-1] == ("answer", None)):
            out.append(kind)
    return out


def _compacting_state() -> list:
    """A state over 80 % of a 10,000-token window: one call compacts it."""
    return _alternating(14, 950) + [HumanMessage("next question", id="q1")]


# ===================== the middleware announces =====================


async def test_a_compaction_announces_start_before_the_summary_call_then_done():
    summary = _ProbeSummaryModel(messages=iter([]), script=["The summary."])
    mw = _middleware(summary, window=10_000)

    await mw.abefore_model({"messages": _compacting_state()}, _runtime(summary.written))

    assert [e["phase"] for e in summary.written] == ["start", "done"]
    assert all(e["t"] == "compaction" for e in summary.written)
    # ``start`` reached the stream BEFORE the summary call ran.
    assert summary.seen_at_call == [["start"]]
    assert summary.written[0] == {"t": "compaction", "phase": "start"}


async def test_no_compaction_announces_nothing():
    summary = _ProbeSummaryModel(messages=iter([]))
    mw = _middleware(summary, window=10_000)
    state = [HumanMessage("hi", id="1"), AIMessage("yo", id="2"), HumanMessage("ok", id="3")]

    assert await mw.abefore_model({"messages": state}, _runtime(summary.written)) is None

    assert summary.written == []


async def test_without_a_runtime_a_compaction_runs_silently():
    summary = _ProbeSummaryModel(messages=iter([]), script=["The summary."])
    mw = _middleware(summary, window=10_000)

    after = _after(await mw.abefore_model({"messages": _compacting_state()}, None))

    assert "The summary." in after[0].content
    assert summary.written == []


def test_the_count_is_what_a_user_recognises_as_their_rows():
    messages = [
        _previous_summary_message(),
        HumanMessage("look it up", id="h1"),
        AIMessage(
            content="Let me check.",
            tool_calls=[{"name": "lookup", "args": {}, "id": "c1"}],
            id="a1",
        ),
        ToolMessage("result", tool_call_id="c1", id="t1"),
        AIMessage("Found it.", id="a2"),
        HumanMessage("and then?", id="h2"),
        AIMessage("", id="a3"),
    ]

    # Two user messages and two answers (one of them empty) -- never the
    # previous summary, the tool call or its result.
    assert runner_module.summarized_message_count(messages) == 4


async def test_done_carries_the_count_the_summary_and_no_loss():
    summary = _ProbeSummaryModel(messages=iter([]), script=["Ada lives in Lyon."])
    mw = _middleware(summary, window=None)
    # No window: 21 messages trigger and the last ten are kept, so eleven
    # plain messages are summarized.
    state = _conversation(10, 10)

    await mw.abefore_model({"messages": state}, _runtime(summary.written))

    done = summary.written[-1]
    assert done["summarized"] == 11
    assert done["summary"] == "Ada lives in Lyon."
    assert done["summary_lost"] is False


async def test_a_lost_summary_sends_the_carried_summary_without_the_closing_sentence():
    summary = _ProbeSummaryModel(messages=iter([]), script=[_overflow_400(), _overflow_400()])
    mw = _middleware(summary, window=10_000)
    state = [_previous_summary_message()] + _compacting_state()[1:]

    await mw.abefore_model({"messages": state}, _runtime(summary.written))

    done = summary.written[-1]
    assert done["summary_lost"] is True
    assert done["summary"] == _PREVIOUS


async def test_a_lost_summary_without_a_previous_one_sends_no_text():
    summary = _ProbeSummaryModel(messages=iter([]), script=[_overflow_400(), _overflow_400()])
    mw = _middleware(summary, window=10_000)

    await mw.abefore_model({"messages": _compacting_state()}, _runtime(summary.written))

    done = summary.written[-1]
    assert done["summary_lost"] is True
    assert done["summary"] == ""


async def test_two_compactions_on_one_middleware_announce_twice_each_with_its_own_facts():
    summary = _ProbeSummaryModel(messages=iter([]), script=["First.", "Second."])
    mw = _middleware(summary, window=10_000)
    runtime = _runtime(summary.written)

    after = _after(await mw.abefore_model({"messages": _compacting_state()}, runtime))
    growth = [
        (AIMessage if i % 2 == 0 else HumanMessage)(_text(950, "g"), id=f"g{i}") for i in range(14)
    ]
    await mw.abefore_model({"messages": after + growth}, runtime)

    assert [e["phase"] for e in summary.written] == ["start", "done", "start", "done"]
    assert [e["summary"] for e in summary.written if e["phase"] == "done"] == [
        "First.",
        "Second.",
    ]


async def test_a_failing_stream_writer_never_fails_the_turn_and_warns_once(caplog):
    summary = _ScriptedSummaryModel(messages=iter([]), script=["The summary."])
    mw = _middleware(summary, window=10_000)

    def broken(_event):
        raise RuntimeError("no stream")

    with caplog.at_level(logging.WARNING):
        after = _after(
            await mw.abefore_model(
                {"messages": _compacting_state()}, SimpleNamespace(stream_writer=broken)
            )
        )

    assert "The summary." in after[0].content
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].getMessage().isascii()


# ===================== why it happened =====================


class _CeilingBudget:
    def __init__(self, ceiling):
        self._ceiling = ceiling

    def tokens_at_ceiling(self):
        return self._ceiling


@pytest.mark.parametrize(
    ("window", "bound", "ceiling", "reason"),
    [
        # CPU / CUDA: -c sized from the RAM (or VRAM) budget; working == allocated.
        (10_000, 10_000, None, "memory"),
        # MLX: the child's window lowered to the memory ceiling after load.
        (10_000, 10_000, 10_000, "memory"),
        # MLX: the memory ceiling folds the working window under the allocated one.
        (32_768, None, 9_000, "memory"),
        # The model's own window binds.
        (10_000, None, None, "length"),
        # A ceiling above the window changes nothing.
        (10_000, None, 50_000, "length"),
    ],
    ids=["cpu-bound", "mlx-bound", "mlx-ceiling", "model-window", "roomy-ceiling"],
)
async def test_the_reason_says_memory_when_memory_bounds_the_window(
    monkeypatch, window, bound, ceiling, reason
):
    monkeypatch.setattr(_WindowEngine, "memory_bound_tokens", classmethod(lambda cls: bound))
    summary = _ProbeSummaryModel(messages=iter([]))
    budget = _CeilingBudget(ceiling) if ceiling is not None else None
    mw = _middleware(summary, window=window, budget=budget)
    state = _alternating(40, 950) + [HumanMessage("next question", id="q1")]

    await mw.abefore_model({"messages": state}, _runtime(summary.written))

    assert summary.written[-1]["reason"] == reason


async def test_the_window_less_fallback_says_length():
    summary = _ProbeSummaryModel(messages=iter([]))
    mw = _middleware(summary, window=None)
    state = _alternating(20, 10) + [HumanMessage("current", id="q")]

    await mw.abefore_model({"messages": state}, _runtime(summary.written))

    assert summary.written[-1]["reason"] == "length"


async def test_the_compact_asap_clamp_says_memory():
    """No allocated window but a memory ceiling the fixed part already
    fills: the working window is the clamped ceiling -- memory alone."""
    summary = _ProbeSummaryModel(messages=iter([]))
    mw = _middleware(summary, window=None, budget=_CeilingBudget(-5))
    state = [HumanMessage("a", id="1"), AIMessage("b", id="2"), HumanMessage("c", id="3")]

    await mw.abefore_model({"messages": state}, _runtime(summary.written))

    assert summary.written[-1]["reason"] == "memory"


# ===================== through the runner and the real graph =====================


async def test_a_compacting_turn_streams_start_done_then_the_answer(monkeypatch):
    _WindowEngine.window = 10_000
    summary = _ScriptedSummaryModel(messages=iter([]), script=["The new summary."])
    _models(monkeypatch, ["the answer"], summary)
    cp = InMemorySaver()
    await _seed(cp, "e1", _seed_messages())

    events = await _turn(AgentRunner(checkpointer=cp), "e1")

    assert _kinds(events) == [("compaction", "start"), ("compaction", "done"), ("answer", None)]
    done = events[1]
    assert done["reason"] == "length"
    assert done["summary"] == "The new summary."
    assert done["summary_lost"] is False
    assert done["summarized"] > 0
    # The summarizer's own tokens never reach the answer.
    assert "".join(e["text"] for e in events if e["t"] == "answer") == "the answer"


async def test_a_turn_without_compaction_streams_no_compaction_event(monkeypatch):
    _WindowEngine.window = 100_000
    _models(monkeypatch, ["the answer"], _ScriptedSummaryModel(messages=iter([])))

    events = await _turn(AgentRunner(checkpointer=InMemorySaver()), "e2")

    assert _kinds(events) == [("answer", None)]


async def test_thinking_tool_calls_and_answers_keep_their_order(monkeypatch):
    """The custom channel rides beside the message stream: an agentic turn
    still streams its narration, tool call, result and answer in order."""
    from tests._helpers import ToolableFakeChatModel
    from src.agents.tools import calculator

    _WindowEngine.window = 10_000
    main = ToolableFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "calculator", "args": {"expression": "2 + 2"}, "id": "k1"}
                    ],
                ),
                AIMessage(content="It is 4."),
            ]
        )
    )
    summary = _ScriptedSummaryModel(messages=iter([]))

    def _build(llm, **kw):
        from src.agents.reasoning_effort import NO_REASONING_PLAN

        return summary if kw.get("effort_plan") is NO_REASONING_PLAN else main

    monkeypatch.setattr(runner_module, "build_chat_model", _build)
    cp = InMemorySaver()
    await _seed(cp, "e3", _seed_messages())

    events = [
        e
        async for e in AgentRunner(checkpointer=cp).astream_text(
            llm=_Llm(),
            user_message="add",
            system_prompt="s",
            params=_PARAMS,
            thread_id="e3",
            summarize=True,
            tools=[calculator],
            emit_events=True,
        )
    ]

    assert _kinds(events) == [
        ("compaction", "start"),
        ("compaction", "done"),
        ("tool_call", None),
        ("tool_result", None),
        ("answer", None),
    ]
    assert "".join(e["text"] for e in events if e["t"] == "answer") == "It is 4."


async def test_a_transient_summary_failure_streams_start_and_no_done(monkeypatch):
    import openai

    _WindowEngine.window = 10_000
    summary = _ScriptedSummaryModel(
        messages=iter([]), script=[openai.APIConnectionError(request=_request())]
    )
    _models(monkeypatch, ["unused"], summary)
    cp = InMemorySaver()
    await _seed(cp, "e4", _seed_messages())

    events = await _turn(AgentRunner(checkpointer=cp), "e4")

    assert events[0] == {"t": "compaction", "phase": "start"}
    assert not any(e.get("phase") == "done" for e in events)
    assert events[-1]["t"] == "answer"
    assert events[-1]["text"].startswith(ERROR_SENTINEL)


async def test_an_arena_projection_carries_only_answer_text(monkeypatch):
    _WindowEngine.window = 10_000
    _models(monkeypatch, ["the answer"], _ScriptedSummaryModel(messages=iter([])))
    cp = InMemorySaver()
    await _seed(cp, "e5", _seed_messages())

    out = [
        t
        async for t in AgentRunner(checkpointer=cp).astream_text(
            llm=_Llm(),
            user_message="next",
            system_prompt="s",
            params=_PARAMS,
            thread_id="e5",
            summarize=True,
        )
    ]

    assert all(isinstance(t, str) for t in out)
    assert "".join(out) == "the answer"


class _ReloadEngine(_WindowEngine):
    pending: ClassVar[bool] = True

    @classmethod
    def brake_reload_pending(cls, llm_id):
        return cls.pending


async def test_a_reload_comes_before_the_compaction_and_the_compaction_before_the_tokens(
    monkeypatch,
):
    monkeypatch.setattr(config, "LLM_Engine", _ReloadEngine)
    _ReloadEngine.window = 10_000
    _models(monkeypatch, ["the answer"], _ScriptedSummaryModel(messages=iter([])))
    cp = InMemorySaver()
    await _seed(cp, "e6", _seed_messages())

    events = await _turn(AgentRunner(checkpointer=cp), "e6")

    assert _kinds(events) == [
        ("memory_reload", None),
        ("compaction", "start"),
        ("compaction", "done"),
        ("answer", None),
    ]


def test_unknown_custom_payloads_are_ignored():
    assert runner_module._compaction_event({"t": "compaction", "phase": "start"}) == {
        "t": "compaction",
        "phase": "start",
    }
    for payload in (None, "text", {"t": "other"}, {"t": "compaction", "phase": "later"}):
        assert runner_module._compaction_event(payload) is None


def test_a_long_summary_fits_the_trace_cap():
    """The longest summary the capped client writes stays far under the
    32 KiB trace cap, even JSON-escaped."""
    import json

    from src.domains.conversations.services import TRACE_MAX_BYTES

    cap = runner_module.summary_cap(10**9)
    event = {
        "t": "compaction",
        "phase": "done",
        "summarized": 999,
        "reason": "memory",
        "summary": _text(cap, "é"),
        "summary_lost": False,
    }
    assert len(json.dumps(event)) < TRACE_MAX_BYTES
