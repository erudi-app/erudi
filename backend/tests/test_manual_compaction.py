"""Manual compaction through the runner (``AgentRunner.acompact``, #585).

``/compact`` summarizes the history now, with the automatic compaction's
cutoff at a one-token keep. Under a non-blocking engine guard it takes the
brake's turn-start sample, loads the model, builds the summary client and
the compaction middleware through the same helper a turn uses, writes the
summary, then the checkpoint (as the model node), resets the prefix cache
like the middleware does, saves the separator through the caller's
callback, and takes the turn-end sample. A failed or overdue summary
changes nothing; a client that goes away mid-summary releases the lock with
nothing written, and one that goes away while the stores are written never
leaves them half-written.
"""

from __future__ import annotations

import asyncio
from typing import ClassVar

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import Field

from langgraph.checkpoint.memory import InMemorySaver

from src.agents import runner as runner_module
from src.agents.reasoning_effort import NO_REASONING_PLAN
from src.agents.runner import (
    COMPACTION_REASON_LENGTH,
    AgentRunner,
    GenParams,
    manual_summary_timeout_s,
    summary_cap,
)
from src.agents.chat_model import CONSERVATIVE_PREFILL_TOKENS_PER_SEC, first_chunk_ceiling_s
from src.core import config
from src.core.exceptions import (
    CompactionFailedException,
    CompactionSeparatorException,
    EngineBusyException,
    EngineException,
)
from src.engines import base_engine
from src.engines.base_engine import BaseEngine
from src.engines.memory_brake import POINT_TURN_END, POINT_TURN_START, BrakeOutcome
from tests._helpers import ToolableFakeChatModel
from tests.test_compaction_keep import (
    _Llm,
    _ScriptedSummaryModel,
    _WindowEngine,
    _previous_summary_message,
    _seed,
    _state,
    _turn,
)

pytestmark = pytest.mark.unit

_PARAMS = GenParams(temperature=0.5, top_p=0.9, max_tokens=64)


class _BrakeEngine(_WindowEngine):
    """A window engine that records its brake samples and every client build."""

    points: ClassVar[list] = []

    @classmethod
    def brake_point(cls, point, *, llm_id=None):
        cls.points.append((point, llm_id))
        return BrakeOutcome()


@pytest.fixture(autouse=True)
def _engine(monkeypatch):
    _BrakeEngine.window = 10_000
    _BrakeEngine.claims = []
    _BrakeEngine.rewrites = []
    _BrakeEngine.points = []
    monkeypatch.setattr(config, "LLM_Engine", _BrakeEngine)
    yield
    _BrakeEngine._last_used = None
    _BrakeEngine._model = None


class _Builds:
    """Patch the client factory: the main client is a fake that never
    answers (a manual compaction sends nothing to it); the summary client
    (effort "none") is ``summary``. Records each build and the brake samples
    taken before it."""

    def __init__(self, monkeypatch, summary):
        self.calls: list = []
        self.summary = summary

        def _build(llm, **kw):
            kind = "summary" if kw.get("effort_plan") is NO_REASONING_PLAN else "main"
            self.calls.append((kind, list(_BrakeEngine.points), kw))
            if kind == "summary":
                return summary
            return ToolableFakeChatModel(messages=iter([]))

        monkeypatch.setattr(runner_module, "build_chat_model", _build)


def _two_exchanges():
    return [
        HumanMessage("My name is Ada.", id="h1"),
        AIMessage("Nice to meet you, Ada.", id="a1"),
        HumanMessage("I live in Lyon.", id="h2"),
        AIMessage("Lyon is lovely.", id="a2"),
    ]


class _Marker:
    """The caller's separator store: records what it was handed."""

    def __init__(self, fail: BaseException | None = None, delay: float = 0.0):
        self.events: list = []
        self.fail = fail
        self.delay = delay

    async def __call__(self, event):
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail is not None:
            raise self.fail
        self.events.append(event)


async def _compact(runner, thread_id, marker=None):
    return await runner.acompact(
        llm=_Llm(),
        thread_id=thread_id,
        params=_PARAMS,
        compaction_threshold=80,
        persist_marker=marker or _Marker(),
    )


def _assert_alternates(state):
    roles = [m.type for m in state if m.type != "tool"]
    assert roles[0] == "human"
    for first, second in zip(roles, roles[1:]):
        assert not (first == second == "human"), roles


# ===================== success =====================


async def test_a_manual_compaction_summarizes_everything_before_the_previous_answer(
    monkeypatch,
):
    summary = _ScriptedSummaryModel(messages=iter([]), script=["Ada lives in Lyon."])
    _Builds(monkeypatch, summary)
    cp = InMemorySaver()
    await _seed(cp, "m1", _two_exchanges())
    marker = _Marker()

    event = await _compact(AgentRunner(checkpointer=cp), "m1", marker)

    state = await _state(cp, "m1")
    assert [m.id for m in state[1:]] == ["a1", "h2", "a2"]
    assert "Ada lives in Lyon." in state[0].content
    assert state[0].additional_kwargs.get("lc_source") == "summarization"
    _assert_alternates(state)
    assert event == {
        "t": "compaction",
        "phase": "done",
        "manual": True,
        "reason": COMPACTION_REASON_LENGTH,
        "summarized": 1,
        "summary": "Ada lives in Lyon.",
        "summary_lost": False,
    }
    assert marker.events == [event]


async def test_the_prefix_cache_is_claimed_then_reset_once_like_a_compacting_turn(monkeypatch):
    _Builds(monkeypatch, _ScriptedSummaryModel(messages=iter([])))
    cp = InMemorySaver()
    await _seed(cp, "m2", _two_exchanges())

    await _compact(AgentRunner(checkpointer=cp), "m2")

    assert _BrakeEngine.claims == ["conv:m2"]
    assert _BrakeEngine.rewrites == ["rewrite"]


async def test_the_brake_samples_surround_it_and_turn_start_comes_before_the_load(monkeypatch):
    builds = _Builds(monkeypatch, _ScriptedSummaryModel(messages=iter([])))
    cp = InMemorySaver()
    await _seed(cp, "m3", _two_exchanges())

    await _compact(AgentRunner(checkpointer=cp), "m3")

    assert _BrakeEngine.points == [(POINT_TURN_START, 7), (POINT_TURN_END, None)]
    first_kind, points_at_first_build, _kw = builds.calls[0]
    assert first_kind == "main"
    assert points_at_first_build == [(POINT_TURN_START, 7)]


async def test_the_summary_client_is_the_turns_one(monkeypatch):
    builds = _Builds(monkeypatch, _ScriptedSummaryModel(messages=iter([])))
    cp = InMemorySaver()
    await _seed(cp, "m4", _two_exchanges())

    await _compact(AgentRunner(checkpointer=cp), "m4")

    summary_kw = next(kw for kind, _p, kw in builds.calls if kind == "summary")
    assert summary_kw["max_tokens"] == summary_cap(10_000)
    assert summary_kw["auto_output_budget"] is False
    assert summary_kw["preflight_retry"] is False


async def test_the_compaction_is_logged_once_as_manual(monkeypatch, caplog):
    import logging

    _Builds(monkeypatch, _ScriptedSummaryModel(messages=iter([])))
    cp = InMemorySaver()
    await _seed(cp, "m5", _two_exchanges())

    with caplog.at_level(logging.INFO):
        await _compact(AgentRunner(checkpointer=cp), "m5")

    lines = [r.getMessage() for r in caplog.records if "Conversation compacted" in r.getMessage()]
    assert len(lines) == 1
    assert "manual" in lines[0]


async def test_the_next_turn_answers_on_the_compacted_history(monkeypatch):
    summary = _ScriptedSummaryModel(messages=iter([]), script=["Ada lives in Lyon."])
    _Builds(monkeypatch, summary)
    cp = InMemorySaver()
    await _seed(cp, "m6", _two_exchanges())
    runner = AgentRunner(checkpointer=cp)
    await _compact(runner, "m6")

    seen: list = []

    class _Main(ToolableFakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            seen.append(list(messages))
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    main = _Main(messages=iter([AIMessage(content="You live in Lyon.")]))

    def _build(llm, **kw):
        if kw.get("effort_plan") is NO_REASONING_PLAN:
            return _ScriptedSummaryModel(messages=iter([]))
        return main

    monkeypatch.setattr(runner_module, "build_chat_model", _build)
    events = await _turn(runner, "m6", question="Where do I live?")

    assert "".join(e["text"] for e in events if e["t"] == "answer") == "You live in Lyon."
    sent = [m for m in seen[0] if m.type != "system"]
    assert "Ada lives in Lyon." in sent[0].content
    _assert_alternates(sent)
    _assert_alternates(await _state(cp, "m6"))


# ===================== nothing to compact =====================


@pytest.mark.parametrize(
    "state",
    [
        [HumanMessage("hi", id="h1"), AIMessage("hello", id="a1")],
        [
            _previous_summary_message(),
            AIMessage("a1", id="a1"),
            HumanMessage("h2", id="h2"),
            AIMessage("a2", id="a2"),
        ],
        [],
    ],
    ids=["single-exchange", "right-after-a-manual-compaction", "empty"],
)
async def test_nothing_to_compact_loads_nothing_and_stores_nothing(monkeypatch, state):
    builds = _Builds(monkeypatch, _ScriptedSummaryModel(messages=iter([])))
    cp = InMemorySaver()
    if state:
        await _seed(cp, "n1", state)
    marker = _Marker()

    result = await _compact(AgentRunner(checkpointer=cp), "n1", marker)

    assert result == {"summarized": 0}
    assert builds.calls == []
    assert _BrakeEngine.points == []
    assert marker.events == []
    if state:
        assert [m.id for m in await _state(cp, "n1")] == [m.id for m in state]


async def test_nothing_to_compact_is_answered_even_while_the_engine_is_busy(monkeypatch):
    monkeypatch.setattr(base_engine, "GENERATION_BUSY_WAIT_S", 0.05)
    _Builds(monkeypatch, _ScriptedSummaryModel(messages=iter([])))
    cp = InMemorySaver()
    await _seed(cp, "n2", [HumanMessage("hi"), AIMessage("hello")])

    async with _BrakeEngine.generation_guard():
        result = await _compact(AgentRunner(checkpointer=cp), "n2")

    assert result == {"summarized": 0}


# ===================== busy =====================


async def test_a_busy_engine_refuses_and_changes_nothing(monkeypatch):
    monkeypatch.setattr(base_engine, "GENERATION_BUSY_WAIT_S", 0.05)
    builds = _Builds(monkeypatch, _ScriptedSummaryModel(messages=iter([])))
    cp = InMemorySaver()
    await _seed(cp, "b1", _two_exchanges())

    async with _BrakeEngine.generation_guard():
        with pytest.raises(EngineBusyException):
            await _compact(AgentRunner(checkpointer=cp), "b1")

    assert builds.calls == []
    assert [m.id for m in await _state(cp, "b1")] == ["h1", "a1", "h2", "a2"]


# ===================== failures =====================


@pytest.mark.parametrize(
    "script",
    [
        [ValueError("a bug"), ValueError("again")],
        ["", ""],
        [EngineException("The inference child died")],
    ],
    ids=["deterministic-twice", "empty-twice", "transient"],
)
async def test_a_summary_that_cannot_be_written_changes_nothing(monkeypatch, script):
    _Builds(monkeypatch, _ScriptedSummaryModel(messages=iter([]), script=list(script)))
    cp = InMemorySaver()
    await _seed(cp, "f1", _two_exchanges())
    marker = _Marker()

    with pytest.raises(CompactionFailedException):
        await _compact(AgentRunner(checkpointer=cp), "f1", marker)

    assert [m.id for m in await _state(cp, "f1")] == ["h1", "a1", "h2", "a2"]
    assert marker.events == []
    assert _BrakeEngine.rewrites == []
    # The turn-end sample is still taken, and the lock is free again.
    assert _BrakeEngine.points[-1] == (POINT_TURN_END, None)
    assert BaseEngine._generation_lock.demand == 0


class _SlowSummaryModel(_ScriptedSummaryModel):
    """A summary client that never finishes on its own."""

    started: asyncio.Event = Field(default_factory=asyncio.Event)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        self.started.set()
        await asyncio.sleep(3600)

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        self.started.set()
        await asyncio.sleep(3600)
        yield  # pragma: no cover


def test_the_summary_bound_is_the_first_chunk_ceiling_plus_its_decode():
    assert manual_summary_timeout_s(8192, 1024) == (
        first_chunk_ceiling_s(8192) + 1024 / CONSERVATIVE_PREFILL_TOKENS_PER_SEC
    )
    assert manual_summary_timeout_s(None, 64) == (
        first_chunk_ceiling_s(None) + 64 / CONSERVATIVE_PREFILL_TOKENS_PER_SEC
    )


async def test_a_summary_past_its_bound_changes_nothing(monkeypatch):
    monkeypatch.setattr(runner_module, "manual_summary_timeout_s", lambda *a: 0.05)
    _Builds(monkeypatch, _SlowSummaryModel(messages=iter([])))
    cp = InMemorySaver()
    await _seed(cp, "t1", _two_exchanges())

    with pytest.raises(CompactionFailedException):
        await _compact(AgentRunner(checkpointer=cp), "t1")

    assert [m.id for m in await _state(cp, "t1")] == ["h1", "a1", "h2", "a2"]
    assert not _BrakeEngine._generation_lock_for_running_loop().locked()


async def test_a_separator_that_cannot_be_saved_leaves_the_history_summarized(monkeypatch):
    _Builds(monkeypatch, _ScriptedSummaryModel(messages=iter([])))
    cp = InMemorySaver()
    await _seed(cp, "s1", _two_exchanges())

    with pytest.raises(CompactionSeparatorException):
        await _compact(
            AgentRunner(checkpointer=cp), "s1", _Marker(fail=CompactionSeparatorException())
        )

    state = await _state(cp, "s1")
    assert state[0].additional_kwargs.get("lc_source") == "summarization"
    assert _BrakeEngine.points[-1] == (POINT_TURN_END, None)


# ===================== the client goes away =====================


async def test_a_client_gone_during_the_summary_releases_the_lock_with_nothing_written(
    monkeypatch,
):
    summary = _SlowSummaryModel(messages=iter([]))
    _Builds(monkeypatch, summary)
    cp = InMemorySaver()
    await _seed(cp, "c1", _two_exchanges())
    marker = _Marker()

    task = asyncio.create_task(_compact(AgentRunner(checkpointer=cp), "c1", marker))
    await asyncio.wait_for(summary.started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert [m.id for m in await _state(cp, "c1")] == ["h1", "a1", "h2", "a2"]
    assert marker.events == []
    assert BaseEngine._generation_lock.demand == 0
    assert not _BrakeEngine._generation_lock_for_running_loop().locked()


async def test_a_client_gone_while_the_stores_are_written_never_half_writes(monkeypatch):
    _Builds(monkeypatch, _ScriptedSummaryModel(messages=iter([])))
    cp = InMemorySaver()
    await _seed(cp, "c2", _two_exchanges())
    marker = _Marker(delay=0.1)

    task = asyncio.create_task(_compact(AgentRunner(checkpointer=cp), "c2", marker))
    while not _BrakeEngine.rewrites:
        await asyncio.sleep(0.005)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Checkpoint and separator both written: the cancellation waited for them.
    state = await _state(cp, "c2")
    assert state[0].additional_kwargs.get("lc_source") == "summarization"
    assert len(marker.events) == 1
    assert BaseEngine._generation_lock.demand == 0


async def test_a_store_failing_after_the_client_went_away_is_logged_once(monkeypatch, caplog):
    import logging

    _Builds(monkeypatch, _ScriptedSummaryModel(messages=iter([])))
    cp = InMemorySaver()
    await _seed(cp, "c3", _two_exchanges())
    marker = _Marker(fail=CompactionSeparatorException(), delay=0.1)

    task = asyncio.create_task(_compact(AgentRunner(checkpointer=cp), "c3", marker))
    while not _BrakeEngine.rewrites:
        await asyncio.sleep(0.005)
    with caplog.at_level(logging.WARNING):
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert isinstance(errors[0].exc_info[1], CompactionSeparatorException)


async def test_a_tool_round_survives_a_manual_compaction(monkeypatch):
    _Builds(monkeypatch, _ScriptedSummaryModel(messages=iter([])))
    cp = InMemorySaver()
    await _seed(
        cp,
        "r1",
        [
            HumanMessage("h1", id="h1"),
            AIMessage("a1", id="a1"),
            HumanMessage("h2", id="h2"),
            AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "c2"}], id="call"),
            ToolMessage("result", tool_call_id="c2", id="res"),
            AIMessage("a2", id="a2"),
        ],
    )

    await _compact(AgentRunner(checkpointer=cp), "r1")

    assert [m.id for m in (await _state(cp, "r1"))[1:]] == ["a1", "h2", "call", "res", "a2"]
