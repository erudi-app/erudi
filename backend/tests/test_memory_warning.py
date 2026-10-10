"""The amber memory warning on honest numbers (a real ``MemoryBudget``).

Contract: the warning shows when, even after compacting to the kept budget,
the model plus the kept conversation would use more than 85 % of the memory
budget (``MEMORY_WARNING_MARGIN``). ``predict`` carries a fixed part, so the
warning can appear for a model whose weights alone are well below 85 %, and
from the first turn on a tight machine. The payload quotes the projection:
``footprint_bytes`` (model and conversation) and ``conversation_bytes``.
"""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents import runner as runner_module
from src.agents.runner import (
    MEMORY_WARNING_MARGIN,
    AgentRunner,
    compaction_cutoff,
    frozen_weights,
    real_token_count,
    summary_cap,
)
from src.agents.token_accounting import RequestOverhead
from src.engines.memory_budget import MemoryBudget

pytestmark = pytest.mark.unit

GIB = 1024**3
KV = 114_688
VOCAB = 151_936


def _budget(base_gib, total_gib=10.0):
    return MemoryBudget(
        base_bytes=int(base_gib * GIB),
        total_bytes=int(total_gib * GIB),
        kv_token_bytes=KV,
        vocab_size=VOCAB,
        prefill_step=2048,
        weights_bytes=GIB,
    )


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


async def _warning(state, budget, window):
    return await AgentRunner()._memory_warning_event(
        _agent_with_state(state),
        {},
        budget,
        window,
        overhead=RequestOverhead(fixed_est=200, fixed_text="system prompt"),
        allocated_window=window,
    )


def test_the_warning_threshold_is_15_percent():
    assert MEMORY_WARNING_MARGIN == 0.15


async def test_the_warning_fires_from_the_first_turn_when_the_fixed_part_alone_exceeds_85():
    # base + m (t0 + logits) is about 7.7 + 1.1 x (1 + 0.58) GiB = 9.4 GiB > 8.5.
    budget = _budget(base_gib=7.7)
    assert budget.used_fraction(0) > 1 - MEMORY_WARNING_MARGIN
    state = [HumanMessage("hello", id="h"), AIMessage("hi there", id="a")]

    event = await _warning(state, budget, 32_768)

    assert event is not None and event["t"] == "memory_warning"


async def test_the_warning_waits_for_the_compaction_point_when_base_is_high():
    budget = _budget(base_gib=5.0)
    window = 32_768
    short = [HumanMessage("hello", id="h"), AIMessage("hi there", id="a")]
    assert await _warning(short, budget, window) is None

    # A long thread that even a compaction to the kept budget cannot bring
    # under 85 %: kept tail + summary + O.
    long = _long_thread(pairs=14, tokens_each=1800)
    event = await _warning(long, budget, window)
    assert event is not None


async def test_the_payload_quotes_the_projection():
    budget = _budget(base_gib=5.0)
    window = 32_768
    state = _long_thread(pairs=14, tokens_each=1800)
    # Nothing measured: every message at the counter's floored script weight,
    # the overhead too; an empty next question closes the projection.
    projected_state = state + [HumanMessage("", id="erudi-next-request")]
    weights, _ = frozen_weights(projected_state)

    def counter(part):
        return real_token_count(part, weights)

    overhead = math.ceil(1.2 * 200)
    cut = compaction_cutoff(
        projected_state, window, counter, overhead=overhead, allocated_window=window
    )
    assert cut > 0
    projected = overhead + counter(projected_state[cut:]) + summary_cap(window)

    event = await _warning(state, budget, window)

    assert event["footprint_bytes"] == budget.footprint_bytes(projected)
    assert event["conversation_bytes"] == budget.conversation_bytes(projected)
    assert event["used_fraction"] == round(budget.used_fraction(projected), 4)
    assert event["footprint_bytes"] > (1 - MEMORY_WARNING_MARGIN) * budget.total_bytes


async def test_an_oversized_last_message_is_projected_whole():
    budget = _budget(base_gib=5.0)
    window = 32_768
    # The last answer is oversized: the next request's compaction keeps it
    # whole (the answer before the next question is never summarized away).
    huge = AIMessage(_text(26_000), id="huge")
    state = [
        HumanMessage("hi", id="h0"),
        AIMessage("yo", id="a0"),
        HumanMessage("go", id="h1"),
        huge,
    ]

    event = await _warning(state, budget, window)

    assert event is not None
    assert event["conversation_bytes"] >= budget.conversation_bytes(real_token_count([huge]))


async def test_a_model_whose_prediction_is_off_never_warns():
    budget = MemoryBudget(
        base_bytes=9 * GIB,
        total_bytes=10 * GIB,
        kv_token_bytes=None,
        vocab_size=VOCAB,
        prefill_step=2048,
        weights_bytes=GIB,
    )
    assert await _warning([HumanMessage("hello", id="h")], budget, 32_768) is None


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
