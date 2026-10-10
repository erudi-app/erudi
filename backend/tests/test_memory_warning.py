"""The amber memory warning: 85 % of the working window, only when memory binds.

Contract: the warning shows only when the loaded child's window is bounded by
memory (``engine.memory_bound_tokens()`` is set -- the memory ceiling is the
binding side of ``W = min(model window, memory ceiling)``) AND the projected
next request, even after a projected compaction to the kept budget, reaches
85 % of the working window (``MEMORY_WARNING_FRACTION``). When the model's own
window binds, memory is not the limiting factor and no memory warning applies,
however full the budget is in bytes. The payload is the one number the copy
quotes: ``footprint_bytes`` (the model and the conversation together).
"""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents import runner as runner_module
from src.agents.runner import (
    MEMORY_WARNING_FRACTION,
    AgentRunner,
    compaction_cutoff,
    frozen_weights,
    real_token_count,
    summary_cap,
)
from src.agents.token_accounting import RequestOverhead
from src.engines.base_engine import BaseEngine
from src.engines.memory_budget import MemoryBudget

pytestmark = pytest.mark.unit

GIB = 1024**3
KV = 114_688
VOCAB = 151_936


def _budget(base_gib, total_gib=10.0):
    return MemoryBudget(
        base_bytes=int(base_gib * GIB) if base_gib is not None else None,
        total_bytes=int(total_gib * GIB),
        kv_token_bytes=KV,
        vocab_size=VOCAB,
        prefill_step=2048,
        weights_bytes=GIB,
    )


class _Engine:
    """The loaded child's memory bound, as the engine reports it."""

    def __init__(self, bound):
        self._bound = bound

    def memory_bound_tokens(self):
        return self._bound


class _LlamaLikeEngine(BaseEngine):
    """An engine that never bounds its window by memory (BaseEngine's default)."""


def _agent_with_state(messages):
    async def aget_state(config):
        return SimpleNamespace(values={"messages": messages})

    return SimpleNamespace(aget_state=aget_state)


def _text(tokens, char="a"):
    return char * (tokens * 4)


def _long_thread(pairs=12, tokens_each=900):
    out = []
    for i in range(pairs):
        out.append(HumanMessage(_text(tokens_each), id=f"h{i}"))
        out.append(AIMessage(_text(tokens_each), id=f"a{i}"))
    return out


def _oversized_thread(tokens):
    # The last answer is oversized: the next request's compaction keeps it
    # whole (the answer before the next question is never summarized away).
    return [
        HumanMessage("hi", id="h0"),
        AIMessage("yo", id="a0"),
        HumanMessage("go", id="h1"),
        AIMessage(_text(tokens), id="huge"),
    ]


async def _warning(state, budget, window, engine):
    return await AgentRunner()._memory_warning_event(
        _agent_with_state(state),
        {},
        budget,
        window,
        overhead=RequestOverhead(fixed_est=200, fixed_text="system prompt"),
        allocated_window=window,
        engine=engine,
    )


def _projected(state, window):
    """The projection, computed independently: the state plus an empty next
    question, every message at the counter's floored script weight (nothing
    measured), the overhead too; kept tail + summary cap when a compaction
    would happen, else the whole request."""
    projected_state = state + [HumanMessage("", id="erudi-next-request")]
    weights, _ = frozen_weights(projected_state)

    def counter(part):
        return real_token_count(part, weights)

    overhead = math.ceil(1.2 * 200)
    cut = compaction_cutoff(
        projected_state, window, counter, overhead=overhead, allocated_window=window
    )
    if cut > 0:
        return overhead + counter(projected_state[cut:]) + summary_cap(window)
    return overhead + counter(projected_state)


def test_the_warning_threshold_is_85_percent_of_the_working_window():
    assert MEMORY_WARNING_FRACTION == 0.85


async def test_no_warning_when_the_model_window_binds_even_at_99_percent_of_the_budget():
    # The fixed part alone fills the budget in bytes, but nothing bounded the
    # child's window by memory: the model window binds, no memory notice.
    budget = _budget(base_gib=8.4)
    assert budget.used_fraction(0) >= 0.99
    window = 32_768
    state = _oversized_thread(26_000)
    assert _projected(state, window) >= MEMORY_WARNING_FRACTION * window

    assert await _warning(state, budget, window, _Engine(None)) is None


async def test_the_warning_fires_when_memory_binds_and_the_projection_reaches_85_percent():
    budget = _budget(base_gib=2.0)
    window = 32_768
    state = _oversized_thread(26_000)
    projected = _projected(state, window)
    assert projected >= MEMORY_WARNING_FRACTION * window

    event = await _warning(state, budget, window, _Engine(window))

    assert event == {"t": "memory_warning", "footprint_bytes": budget.footprint_bytes(projected)}


async def test_no_warning_under_85_percent_of_the_working_window():
    budget = _budget(base_gib=8.4)
    window = 32_768
    short = [HumanMessage("hello", id="h"), AIMessage("hi there", id="a")]
    assert _projected(short, window) < MEMORY_WARNING_FRACTION * window

    assert await _warning(short, budget, window, _Engine(window)) is None


async def test_the_warning_waits_for_what_compaction_cannot_save():
    # A long thread well past the window: a compaction to the kept budget
    # brings the next request back under 85 % -- no warning now.
    budget = _budget(base_gib=2.0)
    window = 16_000
    state = _long_thread(pairs=14, tokens_each=1800)
    assert _projected(state, window) < MEMORY_WARNING_FRACTION * window

    assert await _warning(state, budget, window, _Engine(window)) is None


async def test_no_warning_without_a_base_to_quote():
    # Nothing to quote in the copy: no notice.
    budget = _budget(base_gib=None)
    window = 32_768
    state = _oversized_thread(26_000)

    assert await _warning(state, budget, window, _Engine(window)) is None


async def test_a_trained_window_at_the_floor_with_a_tiny_ceiling_warns():
    # Trained 4096, ceiling 1000: the bound is the 4096 floor (no PATCH), the
    # working window is the 1000-token ceiling, and a next request past 85 %
    # of it warns although the child itself allows 4096.
    budget = _budget(base_gib=2.0)
    window = 1000
    state = _oversized_thread(1200)
    projected = _projected(state, window)
    assert projected >= MEMORY_WARNING_FRACTION * window

    event = await _warning(state, budget, window, _Engine(4096))

    assert event == {"t": "memory_warning", "footprint_bytes": budget.footprint_bytes(projected)}


async def test_a_llama_cpp_engine_never_warns():
    # A budget that WOULD quote a footprint: only the missing memory bound
    # keeps the engine silent.
    budget = _budget(base_gib=2.0)
    window = 32_768
    state = _oversized_thread(26_000)
    projected = _projected(state, window)
    assert projected >= MEMORY_WARNING_FRACTION * window
    assert budget.footprint_bytes(projected) is not None

    assert await _warning(state, budget, window, _LlamaLikeEngine) is None


@pytest.mark.parametrize(
    "projected,fires",
    [(850, True), (849, False)],
)
async def test_the_threshold_is_inclusive_at_85_percent_of_the_window(
    monkeypatch, projected, fires
):
    async def _projection(self, *args, **kwargs):
        return projected, projected

    monkeypatch.setattr(AgentRunner, "_project_next_request", _projection)
    budget = _budget(base_gib=2.0)

    event = await _warning([HumanMessage("hi", id="h")], budget, 1000, _Engine(1000))

    if fires:
        assert event == {"t": "memory_warning", "footprint_bytes": budget.footprint_bytes(850)}
    else:
        assert event is None


async def test_a_model_whose_prediction_is_off_never_warns():
    budget = MemoryBudget(
        base_bytes=9 * GIB,
        total_bytes=10 * GIB,
        kv_token_bytes=None,
        vocab_size=VOCAB,
        prefill_step=2048,
        weights_bytes=GIB,
    )
    state = _oversized_thread(26_000)
    assert await _warning(state, budget, 32_768, _Engine(32_768)) is None


async def test_the_warning_is_logged_with_the_projection_in_ascii(caplog):
    budget = _budget(base_gib=2.0)
    window = 32_768
    state = _oversized_thread(26_000)
    projected = _projected(state, window)

    with caplog.at_level(logging.WARNING, logger="erudi"):
        await _warning(state, budget, window, _Engine(window))

    lines = [
        r.getMessage() for r in caplog.records if "Memory-bound conversation" in r.getMessage()
    ]
    assert len(lines) == 1
    assert lines[0].startswith(
        "Memory-bound conversation over 85 % of the working window even after a projected "
        f"compaction: projected_tokens={projected}, working_window={window}, conversation_tokens="
    )
    assert lines[0].isascii()


# ===================== one INFO line per transition =====================


class _MlxLikeEngine:
    _model: dict = {}


def test_the_warning_state_logs_only_its_transitions(caplog):
    engine = _MlxLikeEngine()
    engine._model = {"pid": 1}
    with caplog.at_level(logging.INFO, logger="erudi"):
        runner_module._track_memory_warning(engine, "t1", True)
        runner_module._track_memory_warning(engine, "t1", True)
        runner_module._track_memory_warning(engine, "t1", False)
        runner_module._track_memory_warning(engine, "t1", False)

    lines = [r.getMessage() for r in caplog.records if "Memory warning" in r.getMessage()]
    assert lines == ["Memory warning started: thread_id=t1", "Memory warning ended: thread_id=t1"]
    assert engine._model["warning_state"] == {}
    assert all(line.isascii() for line in lines)


def test_the_warning_state_is_per_conversation():
    engine = _MlxLikeEngine()
    engine._model = {}
    runner_module._track_memory_warning(engine, "a", True)
    runner_module._track_memory_warning(engine, "b", True)
    runner_module._track_memory_warning(engine, "a", False)
    assert set(engine._model["warning_state"]) == {"b"}


def test_no_handle_means_no_state():
    engine = _MlxLikeEngine()
    engine._model = None
    runner_module._track_memory_warning(engine, "a", True)  # never raises


# ===================== the floor is gone from the ceiling =====================


def test_the_compaction_ceiling_is_tokens_at_ceiling_clamped_to_one(monkeypatch):
    from src.core import config

    class _Engine:
        @staticmethod
        def effective_context_tokens():
            return 32_768

    monkeypatch.setattr(config, "LLM_Engine", _Engine)
    budget = _budget(base_gib=2.0)
    assert AgentRunner._compaction_windows(budget) == (
        32_768,
        min(32_768, budget.tokens_at_ceiling()),
    )
    tight = _budget(base_gib=9.5)
    assert tight.tokens_at_ceiling() <= 0
    assert AgentRunner._compaction_windows(tight) == (32_768, 1)


def test_the_output_budget_reads_the_same_ceiling_unclamped(monkeypatch):
    from src.engines import working_window as ww

    class _Engine:
        @staticmethod
        def effective_context_tokens():
            return 32_768

    budget = _budget(base_gib=2.0)
    monkeypatch.setattr(ww.MemoryBudget, "from_engine", staticmethod(lambda engine: budget))
    assert ww.working_context_tokens(_Engine()) == min(32_768, budget.tokens_at_ceiling())
