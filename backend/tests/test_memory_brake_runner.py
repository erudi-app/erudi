"""The memory brake in a turn: where the runner samples it, what it tells the user.

The runner takes the brake's ``turn_start`` sample before it resolves the
model (so a shrink applies to the turn about to run), and its ``turn_end``
sample once per turn right after the memory window closes -- on the success
path before the warning check, or at the later close site for the turns that
recover from an error. A stateful turn end consumes the brake's notice: a
shrink is the ``pressure`` warning, pressure from other apps is
``pressure_external``, else the memory-bound criterion (``bound``). Before the
model is resolved, a turn that reloads a model the brake unloaded announces it
(``memory_reload``).
"""

from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace
from typing import ClassVar, Optional

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from src.agents import runner as runner_module
from src.agents.reasoning_effort import NO_REASONING_PLAN
from src.agents.runner import ERROR_SENTINEL, AgentRunner, GenParams
from src.core import config
from src.engines.base_engine import BaseEngine
from src.engines.memory_brake import NOTICE_ACTED, NOTICE_EXTERNAL, BrakeOutcome
from src.engines.mlx_engine import MLX_Engine
from tests._helpers import ToolableFakeChatModel
from tests.test_agent_runner import _TimingOutModel
from tests.test_compaction_keep import _measured, _text

pytestmark = pytest.mark.unit

_PARAMS = GenParams(temperature=0.5, top_p=0.9, max_tokens=64)


class _Llm:
    id = 7
    link = "/fake/path"
    name = "Test 7B"
    param_size = 7.0


class _BrakeEngine(BaseEngine):
    """An engine whose brake hooks record what the runner calls, in order."""

    calls: ClassVar[list] = []
    # The compaction threshold each closed measurement window was stamped with.
    thresholds: ClassVar[list] = []
    notice: ClassVar[Optional[str]] = None
    pending: ClassVar[set] = set()
    bound: ClassVar[Optional[int]] = None
    bound_footprint: ClassVar[Optional[int]] = None
    window: ClassVar[Optional[int]] = None
    turn_start_outcome: ClassVar[BrakeOutcome] = BrakeOutcome()

    @classmethod
    def brake_point(cls, point, *, llm_id=None):
        worker = threading.current_thread() is not threading.main_thread()
        cls.calls.append(("brake", point, llm_id, worker))
        return cls.turn_start_outcome if point == "turn_start" else BrakeOutcome()

    @classmethod
    def consume_brake_notice(cls):
        cls.calls.append(("consume",))
        notice, cls.notice = cls.notice, None
        return notice

    @classmethod
    def brake_reload_pending(cls, llm_id):
        cls.calls.append(("pending", llm_id))
        return llm_id in cls.pending

    @classmethod
    def take_idle_reload(cls, llm_id):
        cls.calls.append(("take", llm_id))
        return super().take_idle_reload(llm_id)

    @classmethod
    def begin_memory_window(cls):
        cls.calls.append(("begin",))
        return ("token",)

    @classmethod
    def end_memory_window(cls, token, *, abandoned=False, compaction_threshold=None):
        cls.calls.append(("end", abandoned))
        cls.thresholds.append(compaction_threshold)

    @classmethod
    def memory_bound_tokens(cls):
        return cls.bound

    @classmethod
    def memory_bound_footprint_bytes(cls):
        return cls.bound_footprint

    @classmethod
    def effective_context_tokens(cls):
        return cls.window


@pytest.fixture(autouse=True)
def _engine(monkeypatch):
    _BrakeEngine.calls = []
    _BrakeEngine.thresholds = []
    _BrakeEngine.notice = None
    _BrakeEngine.pending = set()
    _BrakeEngine.bound = None
    _BrakeEngine.bound_footprint = None
    _BrakeEngine.window = None
    _BrakeEngine.turn_start_outcome = BrakeOutcome()
    _BrakeEngine._idle_unload = None
    _BrakeEngine._model = None
    _BrakeEngine._model_id = None
    monkeypatch.setattr(config, "LLM_Engine", _BrakeEngine)
    yield
    _BrakeEngine._idle_unload = None
    _BrakeEngine._model = None
    _BrakeEngine._model_id = None
    _BrakeEngine._last_used = None


def _patch_model(monkeypatch, fake):
    def _build(llm, **kw):
        _BrakeEngine.calls.append(("build",))
        return fake

    monkeypatch.setattr(runner_module, "build_chat_model", _build)


def _answer(monkeypatch, text="hello"):
    _patch_model(monkeypatch, ToolableFakeChatModel(messages=iter([AIMessage(content=text)])))


async def _events(runner, *, thread_id="b1", summarize=True, emit_events=True, **kwargs):
    # A conversation turn carries its compaction threshold (#607); an arena turn none.
    kwargs.setdefault("compaction_threshold", 80 if summarize else None)
    return [
        e
        async for e in runner.astream_text(
            llm=_Llm(),
            user_message=kwargs.pop("user_message", "hi"),
            system_prompt="s",
            params=_PARAMS,
            thread_id=thread_id,
            summarize=summarize,
            emit_events=emit_events,
            **kwargs,
        )
    ]


def _brakes(point):
    return [c for c in _BrakeEngine.calls if c[0] == "brake" and c[1] == point]


class _StubBudget:
    def __init__(self, ceiling=10):
        self._ceiling = ceiling

    def footprint_bytes(self, conversation_tokens):
        return 4321

    def used_fraction(self, conversation_tokens):
        return 0.5

    def tokens_at_ceiling(self):
        return self._ceiling


def _patch_budget(monkeypatch, budget):
    monkeypatch.setattr(
        runner_module, "MemoryBudget", SimpleNamespace(from_engine=lambda engine: budget)
    )


# ===================== where the samples are taken =====================


async def test_turn_start_runs_before_the_model_is_resolved_and_turn_end_after_the_window(
    monkeypatch,
):
    _answer(monkeypatch)

    await _events(AgentRunner(checkpointer=InMemorySaver()))

    names = [c[:2] if c[0] == "brake" else c[:1] for c in _BrakeEngine.calls]
    assert names.index(("brake", "turn_start")) < names.index(("build",))
    assert names.index(("end",)) < names.index(("brake", "turn_end"))
    assert names.index(("brake", "turn_end")) < names.index(("consume",))
    (start,) = _brakes("turn_start")
    assert start[2] == 7
    # Both run shielded, in a worker thread (they may talk to the child).
    assert start[3] is True
    assert _brakes("turn_end")[0][3] is True


async def test_the_turn_end_sample_runs_once_on_the_success_path(monkeypatch):
    _answer(monkeypatch)

    await _events(AgentRunner(checkpointer=InMemorySaver()))

    assert len(_brakes("turn_end")) == 1
    assert [c for c in _BrakeEngine.calls if c[0] == "end"] == [("end", False)]


async def test_a_recovered_error_turn_takes_its_turn_end_sample_at_the_later_site(monkeypatch):
    _patch_model(monkeypatch, _TimingOutModel(messages=iter([])))

    events = await _events(AgentRunner(checkpointer=InMemorySaver()))

    assert ERROR_SENTINEL in "".join(e.get("text", "") for e in events)
    assert len(_brakes("turn_end")) == 1
    assert [c for c in _BrakeEngine.calls if c[0] == "end"] == [("end", False)]
    # No warning on an error turn: the notice waits for a turn that can show it.
    assert ("consume",) not in _BrakeEngine.calls


async def test_an_abandoned_turn_takes_no_turn_end_sample(monkeypatch):
    _answer(monkeypatch, "one two three")
    runner = AgentRunner(checkpointer=InMemorySaver())
    stream = runner.astream_text(
        llm=_Llm(),
        user_message="hi",
        system_prompt="s",
        params=_PARAMS,
        thread_id="ab",
        summarize=True,
        emit_events=True,
        compaction_threshold=80,
    )
    async for event in stream:
        if event["t"] == "answer":
            break
    await stream.aclose()

    assert _brakes("turn_end") == []
    assert [c for c in _BrakeEngine.calls if c[0] == "end"] == [("end", True)]


async def test_the_window_closes_with_the_conversations_threshold(monkeypatch):
    # The brake judges the turn's largest request against the trigger of the
    # conversation that sent it (#607).
    _answer(monkeypatch)

    await _events(AgentRunner(checkpointer=InMemorySaver()), compaction_threshold=55)

    assert _BrakeEngine.thresholds == [55]


async def test_an_abandoned_turn_closes_with_its_threshold(monkeypatch):
    _answer(monkeypatch, "one two three")
    stream = AgentRunner(checkpointer=InMemorySaver()).astream_text(
        llm=_Llm(),
        user_message="hi",
        system_prompt="s",
        params=_PARAMS,
        thread_id="ab",
        summarize=True,
        compaction_threshold=65,
        emit_events=True,
    )
    async for event in stream:
        if event["t"] == "answer":
            break
    await stream.aclose()

    assert [c for c in _BrakeEngine.calls if c[0] == "end"] == [("end", True)]
    assert _BrakeEngine.thresholds == [65]


async def test_an_arena_turn_closes_the_window_without_a_threshold(monkeypatch):
    _answer(monkeypatch, "duel")

    await _events(AgentRunner(checkpointer=None), thread_id=None, summarize=False)

    assert _BrakeEngine.thresholds == [None]


async def test_an_arena_turn_takes_its_samples_and_never_consumes_the_notice(monkeypatch):
    _answer(monkeypatch, "duel")
    _BrakeEngine.notice = NOTICE_ACTED

    out = await _events(AgentRunner(checkpointer=None), thread_id=None, summarize=False)

    assert "".join(e["text"] for e in out if e["t"] == "answer") == "duel"
    assert len(_brakes("turn_start")) == 1
    assert len(_brakes("turn_end")) == 1
    assert ("consume",) not in _BrakeEngine.calls
    assert _BrakeEngine.notice == NOTICE_ACTED
    assert not any(e["t"] == "memory_warning" for e in out)


async def test_an_engine_without_a_brake_runs_no_shielded_sample(monkeypatch):
    class _Plain(BaseEngine):
        pass

    monkeypatch.setattr(config, "LLM_Engine", _Plain)
    _answer(monkeypatch)
    seen = []
    real = runner_module.run_reset_shielded

    async def _spy(fn):
        seen.append(fn)
        return await real(fn)

    monkeypatch.setattr(runner_module, "run_reset_shielded", _spy)

    await _events(AgentRunner(checkpointer=InMemorySaver()))

    assert seen == []


# ===================== what the user is told =====================


async def test_a_shrink_is_the_pressure_warning_and_skips_the_bound_criterion(monkeypatch):
    _answer(monkeypatch)
    _patch_budget(monkeypatch, _StubBudget(ceiling=10))
    _BrakeEngine.bound = 4096
    _BrakeEngine.notice = NOTICE_ACTED
    evaluated = []

    async def _never(self, *args, **kwargs):
        evaluated.append(True)
        return {"t": "memory_warning", "reason": "bound", "footprint_bytes": 1}

    monkeypatch.setattr(AgentRunner, "_memory_warning_event", _never)

    events = await _events(AgentRunner(checkpointer=InMemorySaver()))

    warnings = [e for e in events if e["t"] == "memory_warning"]
    assert warnings == [{"t": "memory_warning", "reason": "pressure"}]
    assert evaluated == []
    assert _BrakeEngine.notice is None


async def test_pressure_from_other_apps_is_the_external_warning(monkeypatch):
    _answer(monkeypatch)
    _patch_budget(monkeypatch, _StubBudget(ceiling=16_000))
    _BrakeEngine.notice = NOTICE_EXTERNAL

    events = await _events(AgentRunner(checkpointer=InMemorySaver()))

    warnings = [e for e in events if e["t"] == "memory_warning"]
    assert warnings == [{"t": "memory_warning", "reason": "pressure_external"}]


async def test_without_a_notice_the_bound_criterion_decides(monkeypatch):
    _answer(monkeypatch)
    _patch_budget(monkeypatch, _StubBudget(ceiling=10))
    _BrakeEngine.bound = 4096

    events = await _events(AgentRunner(checkpointer=InMemorySaver()))

    warnings = [e for e in events if e["t"] == "memory_warning"]
    assert warnings == [{"t": "memory_warning", "reason": "bound", "footprint_bytes": 4321}]


async def test_the_warning_comes_after_the_answer(monkeypatch):
    _answer(monkeypatch)
    _patch_budget(monkeypatch, _StubBudget(ceiling=16_000))
    _BrakeEngine.notice = NOTICE_ACTED

    events = await _events(AgentRunner(checkpointer=InMemorySaver()))

    kinds = [e["t"] for e in events]
    assert kinds.index("memory_warning") > max(i for i, k in enumerate(kinds) if k == "answer")


# ===================== the reload =====================


async def test_a_pending_reload_is_announced_first(monkeypatch):
    _answer(monkeypatch)
    _BrakeEngine.pending = {7}

    events = await _events(AgentRunner(checkpointer=InMemorySaver()))

    assert events[0] == {"t": "memory_reload"}
    assert sum(1 for e in events if e["t"] == "memory_reload") == 1
    # Announced before the model is resolved (the reload IS the resolution).
    names = [c[0] for c in _BrakeEngine.calls]
    assert names.index("pending") < names.index("build")


async def test_no_reload_no_announcement(monkeypatch):
    _answer(monkeypatch)

    events = await _events(AgentRunner(checkpointer=InMemorySaver()))

    assert not any(e["t"] == "memory_reload" for e in events)
    assert ("pending", 7) in _BrakeEngine.calls


@pytest.mark.parametrize("action", ["shrink+respawn", "respawn", "respawn-failed"])
async def test_a_respawn_at_turn_start_is_announced_in_the_same_turn(monkeypatch, action):
    """llama.cpp: a critical turn_start respawns the child, whose prompt is
    re-prefilled by this very turn -- the reload state is said now, before
    the model is resolved."""
    _answer(monkeypatch)
    _BrakeEngine.turn_start_outcome = BrakeOutcome(acted=True, action=action)

    events = await _events(AgentRunner(checkpointer=InMemorySaver()))

    assert events[0] == {"t": "memory_reload"}
    assert sum(1 for e in events if e["t"] == "memory_reload") == 1
    names = [c[:2] if c[0] == "brake" else c[:1] for c in _BrakeEngine.calls]
    assert names.index(("brake", "turn_start")) < names.index(("build",))


async def test_a_respawn_after_an_announced_reload_is_not_announced_twice(monkeypatch):
    _answer(monkeypatch)
    _BrakeEngine.pending = {7}
    _BrakeEngine.turn_start_outcome = BrakeOutcome(acted=True, action="shrink+respawn")

    events = await _events(AgentRunner(checkpointer=InMemorySaver()))

    assert sum(1 for e in events if e["t"] == "memory_reload") == 1


# ===================== idle reload (#619) =====================


IDLE_RELOAD = {"t": "memory_reload", "reason": "idle"}


async def test_a_turn_after_an_idle_reap_announces_why_the_model_reloads(monkeypatch):
    """The record is consumed before the model is resolved, so a turn
    cancelled before the load loses the notice (accepted)."""
    _answer(monkeypatch)
    _BrakeEngine._idle_unload = {"model_id": 7}

    events = await _events(AgentRunner(checkpointer=InMemorySaver()), announce_idle_reload=True)

    assert events[0] == IDLE_RELOAD
    assert sum(1 for e in events if e["t"] == "memory_reload") == 1
    names = [c[0] for c in _BrakeEngine.calls]
    assert names.index("take") < names.index("build")
    assert _BrakeEngine._idle_unload is None


async def test_a_first_turn_consumes_the_idle_record_without_announcing_it(monkeypatch):
    """A new conversation shows its first-reply hint, not the idle copy: the
    record is taken (it must not leak to a later turn) but not announced."""
    _answer(monkeypatch)
    _BrakeEngine._idle_unload = {"model_id": 7}

    events = await _events(AgentRunner(checkpointer=InMemorySaver()))

    assert not any(e["t"] == "memory_reload" for e in events)
    assert ("take", 7) in _BrakeEngine.calls
    assert _BrakeEngine._idle_unload is None


async def test_a_record_left_by_another_model_is_dropped_when_one_is_loaded(monkeypatch):
    """A model switch does not go through ``cleanup()``: the protection is
    that a record found while a model is loaded is stale."""
    _answer(monkeypatch)
    _BrakeEngine._idle_unload = {"model_id": 7}
    _BrakeEngine._model = object()
    _BrakeEngine._model_id = 8

    events = await _events(AgentRunner(checkpointer=InMemorySaver()), announce_idle_reload=True)

    assert not any(e["t"] == "memory_reload" for e in events)
    assert _BrakeEngine._idle_unload is None


async def test_the_brake_reload_wins_over_an_idle_record(monkeypatch):
    _answer(monkeypatch)
    _BrakeEngine.pending = {7}
    _BrakeEngine._idle_unload = {"model_id": 7}

    events = await _events(AgentRunner(checkpointer=InMemorySaver()), announce_idle_reload=True)

    assert [e for e in events if e["t"] == "memory_reload"] == [{"t": "memory_reload"}]


async def test_an_idle_reload_is_not_announced_twice_by_a_turn_start_respawn(monkeypatch):
    _answer(monkeypatch)
    _BrakeEngine._idle_unload = {"model_id": 7}
    _BrakeEngine.turn_start_outcome = BrakeOutcome(acted=True, action="respawn")

    events = await _events(AgentRunner(checkpointer=InMemorySaver()), announce_idle_reload=True)

    assert [e for e in events if e["t"] == "memory_reload"] == [IDLE_RELOAD]


@pytest.mark.parametrize("action", ["shrink", "grow", None])
async def test_a_turn_start_without_a_respawn_announces_nothing(monkeypatch, action):
    _answer(monkeypatch)
    _BrakeEngine.turn_start_outcome = BrakeOutcome(acted=action is not None, action=action)

    events = await _events(AgentRunner(checkpointer=InMemorySaver()))

    assert not any(e["t"] == "memory_reload" for e in events)


async def test_the_bound_warning_on_llama_quotes_the_child_allocation(monkeypatch):
    """llama.cpp has no per-token accounting (the budget is unaccounted): the
    bound criterion runs on the working window alone and quotes what the
    memory-bound child allocated at load."""
    from src.engines.memory_budget import MemoryBudget

    _answer(monkeypatch)
    _patch_budget(monkeypatch, MemoryBudget.unaccounted())
    _BrakeEngine.window = 24
    _BrakeEngine.bound = 24
    _BrakeEngine.bound_footprint = 9_999

    events = await _events(AgentRunner(checkpointer=InMemorySaver()))

    warnings = [e for e in events if e["t"] == "memory_warning"]
    assert warnings == [{"t": "memory_warning", "reason": "bound", "footprint_bytes": 9_999}]


async def test_no_bound_warning_on_llama_when_memory_does_not_bind(monkeypatch):
    from src.engines.memory_budget import MemoryBudget

    _answer(monkeypatch)
    _patch_budget(monkeypatch, MemoryBudget.unaccounted())
    _BrakeEngine.window = 24
    _BrakeEngine.bound_footprint = 9_999

    events = await _events(AgentRunner(checkpointer=InMemorySaver()))

    assert not any(e["t"] == "memory_warning" for e in events)


async def test_the_arena_wire_never_carries_the_reload(monkeypatch):
    _answer(monkeypatch, "duel")
    _BrakeEngine.pending = {7}

    out = [
        t
        async for t in AgentRunner(checkpointer=None).astream_text(
            llm=_Llm(),
            user_message="hi",
            system_prompt="s",
            params=_PARAMS,
            thread_id=None,
            summarize=False,
        )
    ]

    assert out == ["duel"]


async def test_a_reload_that_fails_ends_with_the_error(monkeypatch):
    def _boom(llm, **kw):
        raise RuntimeError("spawn failed")

    monkeypatch.setattr(runner_module, "build_chat_model", _boom)
    _BrakeEngine.pending = {7}

    events = await _events(AgentRunner(checkpointer=InMemorySaver()))

    assert events[0] == {"t": "memory_reload"}
    assert events[-1]["t"] == "answer"
    assert ERROR_SENTINEL in events[-1]["text"]


@pytest.fixture
def mlx(monkeypatch):
    """The real MLX engine, nothing loaded (the model factory is faked)."""
    monkeypatch.setattr(config, "LLM_Engine", MLX_Engine)
    MLX_Engine._model = None
    MLX_Engine._model_id = None
    MLX_Engine._brake_reload = None
    yield MLX_Engine
    MLX_Engine._model = None
    MLX_Engine._model_id = None
    MLX_Engine._last_used = None
    MLX_Engine._brake_reload = None
    BaseEngine._pending_reset = None


def _record(alias="erudi-7", age_s=10.0):
    return {
        "alias": alias,
        "cap": 12_000,
        "faulty": None,
        "in_episode": True,
        "episode_level": 4,
        "shrinks_this_episode": 1,
        "unloaded_this_episode": True,
        "unloaded_at": time.time() - age_s,
    }


@pytest.mark.parametrize(
    "record,announced",
    [
        (_record(), True),
        (_record(alias="erudi-8"), False),
        (None, False),
    ],
)
async def test_the_mlx_engine_announces_only_its_own_pending_reload(
    monkeypatch, mlx, record, announced
):
    _answer(monkeypatch)
    MLX_Engine._brake_reload = record

    events = await _events(AgentRunner(checkpointer=InMemorySaver()))

    assert (events[0] == {"t": "memory_reload"}) is announced


async def test_no_reload_is_announced_after_an_idle_timeout_reap(monkeypatch, mlx):
    _answer(monkeypatch)
    MLX_Engine._brake_reload = _record()
    MLX_Engine.cleanup()  # the reap

    events = await _events(AgentRunner(checkpointer=InMemorySaver()))

    assert not any(e["t"] == "memory_reload" for e in events)


# ===================== M9: a shrink below the history =====================


class _WindowedModel(ToolableFakeChatModel):
    """The child's preflight: a request over ``_BrakeEngine.window`` (chars/4,
    plus the output cap) is refused with mlx_vlm's overflow detail -- and so
    is the first call when ``refuse_first`` (a counter that under-counted)."""

    cap: int = 0
    refuse_first: bool = False
    calls: list = Field(default_factory=list)

    def _check(self, messages):
        size = count_tokens_approximately(messages) + self.cap
        self.calls.append(size)
        window = _BrakeEngine.window
        refused = self.refuse_first and len(self.calls) == 1
        if refused or (window is not None and size > window):
            raise RuntimeError(
                f"Request needs {size} context tokens ({size - self.cap} prompt), "
                f"MAX_KV_SIZE is {window}"
            )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._check(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _history(answer_tokens):
    """Alternating, measured (the counter reads chars/4), ending with an answer."""
    out = []
    for i, tokens in enumerate(answer_tokens):
        out.append(HumanMessage(_text(100, "q"), id=f"h{i}"))
        out.append(_measured(_text(tokens, chr(ord("a") + i)), f"a{i}"))
    return out


async def _seed(checkpointer, thread_id, messages):
    from langchain.agents import create_agent

    probe = create_agent(
        ToolableFakeChatModel(messages=iter([])), tools=[], checkpointer=checkpointer
    )
    await probe.aupdate_state(
        {"configurable": {"thread_id": thread_id}}, {"messages": messages}, as_node="model"
    )


@pytest.mark.parametrize("refused", [False, True], ids=["summary-fits", "summary-size-path"])
async def test_after_a_shrink_below_the_history_the_next_turn_compacts_and_answers(
    monkeypatch, caplog, refused
):
    """M9: the brake shrank the window below a history the old window held
    without compacting. The next turn compacts under the new window and
    answers -- no error turn -- also when the child refuses the first summary
    call (the summarizer's size path)."""
    old_w, new_w = 20_000, 15_000
    history = _history([1_850] * 8)
    size = count_tokens_approximately(history)
    # Under the old window's trigger (no compaction there), over the new window.
    assert new_w < size < 0.8 * old_w
    _BrakeEngine.window = new_w
    main = _WindowedModel(messages=iter([AIMessage(content="still here")]))
    summary = _WindowedModel(
        messages=iter([AIMessage(content="The user asked about many things.")] * 3),
        cap=runner_module.summary_cap(new_w),
        refuse_first=refused,
    )

    def _build(llm, **kw):
        return summary if kw.get("effort_plan") is NO_REASONING_PLAN else main

    monkeypatch.setattr(runner_module, "build_chat_model", _build)
    checkpointer = InMemorySaver()
    await _seed(checkpointer, "m9", history)

    with caplog.at_level(logging.INFO):
        events = await _events(AgentRunner(checkpointer=checkpointer), thread_id="m9")

    text = "".join(e.get("text", "") for e in events if e["t"] == "answer")
    assert text == "still here"
    assert len(summary.calls) == (2 if refused else 1)
    assert all(call <= new_w for call in main.calls)
    assert any("Conversation compacted" in r.getMessage() for r in caplog.records)
    if refused:
        assert any("Compaction summary call rejected" in r.getMessage() for r in caplog.records)
