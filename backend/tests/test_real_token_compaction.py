"""Compaction in real tokens: per-message weights, the request overhead, as sent.

The counter costs every message at its own weight (``real_token_count``):
the measured ratio of the anchor for what was inside the measured request, the
exact ``output_tokens`` for an answer without reasoning, the script weight of
its own text for anything new (floor 1.2) -- frozen once per ``abefore_model``
from the full state, by message id, with past KB/web results counted as their
markers. The request overhead O (system prompt, tool schemas, KB block -- not
in the state) enters only where the whole request matters: the trigger, the
keep and the warning projection. A futility guard skips a compaction that
gains less than 256 tokens unless the request would overflow without it.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from langchain_core.messages.utils import count_tokens_approximately as approx_token_count
from src.agents import runner as runner_module
from src.agents.middleware import STALE_TOOL_RESULT_MARKERS, strip_stale_tool_results
from src.agents.runner import (
    AgentRunner,
    compaction_cutoff,
    counter_weight,
    frozen_weights,
    keep_token_budget,
    real_token_count,
    summarize_trim_budget,
    summary_cap,
)
from src.agents.token_accounting import (
    COUNTER_DENSE_TOKENS,
    COUNTER_WEIGHT_FLOOR,
    REQUEST_EST_KEY,
    REQUEST_FIRST_HOP_KEY,
    REQUEST_HAS_IMAGES_KEY,
    RequestOverhead,
    estimate,
    exact_tokens,
    overhead_tokens,
    script_weight,
)
from src.core import config
from src.engines.base_engine import BaseEngine
from tests.test_compaction_keep import (  # noqa: F401  (``_engine`` is an autouse fixture)
    _PARAMS,
    _Llm,
    _ScriptedSummaryModel,
    _WindowEngine,
    _engine,
    _seed,
    _text,
)

pytestmark = pytest.mark.unit

KB = "search_knowledge_base"
KB_MARKER = STALE_TOOL_RESULT_MARKERS[KB]
FLOOR = COUNTER_WEIGHT_FLOOR


def _stamped(content, input_tokens, est, *, first=True, msg_id=None, output_tokens=5, **kwargs):
    return AIMessage(
        content=content,
        id=msg_id,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        response_metadata={
            REQUEST_EST_KEY: est,
            REQUEST_HAS_IMAGES_KEY: False,
            REQUEST_FIRST_HOP_KEY: first,
        },
        **kwargs,
    )


def _summary(text="facts", msg_id="s0"):
    return HumanMessage(
        content=f"Here is a summary of the conversation to date:\n\n{text}",
        additional_kwargs={"lc_source": "summarization"},
        id=msg_id,
    )


def _kb_round(n, result_tokens, *, first_r=1.0, last_r=None, call_id="functions.kb:0"):
    """One finished KB turn: question, first hop (tool call, stamped first),
    the tool result, the answer (stamped last hop when ``last_r``)."""
    answer = (
        _stamped("answer", int(last_r * 1000), 1000, first=False, msg_id=f"ans{n}")
        if last_r is not None
        else AIMessage("answer", id=f"ans{n}")
    )
    return [
        HumanMessage(f"question {n}", id=f"q{n}"),
        _stamped(
            "",
            int(first_r * 1000),
            1000,
            first=True,
            msg_id=f"call{n}",
            tool_calls=[{"name": KB, "args": {}, "id": call_id}],
        ),
        ToolMessage(_text(result_tokens, "r"), name=KB, tool_call_id=call_id, id=f"res{n}"),
        answer,
    ]


def _fixed(est, text="You are a helpful assistant."):
    return RequestOverhead(fixed_est=est, fixed_text=text)


def _mw(window, *, overhead=None, budget=None, summary=None):
    from langchain.agents.middleware import SummarizationMiddleware

    _WindowEngine.window = window
    summary = summary or _ScriptedSummaryModel(messages=iter([]))
    if isinstance(overhead, int):
        overhead = _fixed(overhead)
    built = AgentRunner()._build_middleware(summary, budget, overhead=overhead)
    return next(m for m in built if isinstance(m, SummarizationMiddleware))


def _after(result) -> list:
    assert result is not None
    return list(result["messages"][1:])


def _cost(message, weight):
    return math.ceil(weight * estimate(message))


# ===================== the counter =====================


async def test_each_message_costs_its_own_weight_with_past_results_as_markers():
    mw = _mw(100_000)
    state = _kb_round(1, 3000, first_r=2.0) + [HumanMessage("next question", id="q2")]

    assert await mw.abefore_model({"messages": state}, None) is None

    as_sent = strip_stale_tool_results(state)
    marker = as_sent[2]
    assert marker.content == KB_MARKER
    # The anchor is the stamped first hop (index 1): the question before it
    # at r = 2.0, the hop at its exact output, the marker at its OWN weight,
    # the answer and the new question at their script weight (floor 1.2).
    expected = (
        _cost(state[0], 2.0)
        + exact_tokens(state[1])
        + _cost(marker, counter_weight(marker))
        + _cost(state[3], FLOOR)
        + _cost(state[4], FLOOR)
    )
    assert mw.token_counter(state) == expected
    assert mw.token_counter(state) < approx_token_count(state)


async def test_weights_are_frozen_by_id_so_a_suffix_counts_its_share():
    mw = _mw(100_000)
    state = [
        HumanMessage(_text(400), id="h0"),
        _stamped("ok", 2000, 1000, msg_id="a0"),
        HumanMessage("next", id="q"),
    ]
    await mw.abefore_model({"messages": state}, None)

    assert mw.token_counter(state[:1]) == _cost(state[0], 2.0)
    half = state[0].model_copy(update={"content": state[0].content[:800]})
    assert mw.token_counter([half]) == _cost(half, 2.0)


async def test_a_marker_never_takes_the_frozen_weight_of_the_result_it_replaces():
    mw = _mw(100_000)
    # Both turns' calls carry the SAME call id; only the past result is a marker.
    state = (
        _kb_round(1, 3000, first_r=1.0)
        + [HumanMessage("question 2", id="q2")]
        + [
            _stamped(
                "",
                3000,
                1000,
                msg_id="call2",
                tool_calls=[{"name": KB, "args": {}, "id": "functions.kb:0"}],
            ),
            ToolMessage(_text(3000, "c"), name=KB, tool_call_id="functions.kb:0", id="res2"),
        ]
    )
    await mw.abefore_model({"messages": state}, None)

    marker = state[2].model_copy(update={"content": KB_MARKER})
    assert mw.token_counter([state[2]]) == _cost(marker, counter_weight(marker))
    # The current result is not past: it counts in full at its own weight.
    assert mw.token_counter([state[-1]]) == _cost(state[-1], FLOOR)


async def test_without_a_measurement_new_text_counts_at_its_script_weight_floored():
    mw = _mw(100_000)
    english = [HumanMessage("hello there", id="q1")]
    await mw.abefore_model({"messages": english}, None)
    assert mw.token_counter(english) == _cost(english[0], FLOOR)

    cjk = [HumanMessage("这是一个很长的中文句子" * 30, id="q1")]
    await mw.abefore_model({"messages": cjk}, None)
    weight = script_weight(cjk[0].content, dense=COUNTER_DENSE_TOKENS)
    assert weight > 3.0
    assert mw.token_counter(cjk) == _cost(cjk[0], weight)


def test_the_counter_is_additive_and_monotone_for_the_binary_search():
    messages = [HumanMessage(_text(i * 10), id=f"m{i}") for i in range(1, 30)]
    weights, _ = frozen_weights(messages)
    counts = [real_token_count(messages[i:], weights) for i in range(len(messages))]
    assert counts == sorted(counts, reverse=True)
    assert real_token_count(messages, weights) == (
        real_token_count(messages[:10], weights) + real_token_count(messages[10:], weights)
    )


async def test_a_partial_trim_copy_keeps_its_id_and_lowers_the_count():
    mw = _mw(100_000)
    state = [
        HumanMessage(_text(4000), id="big"),
        AIMessage("a", id="a1"),
        HumanMessage("q", id="q"),
    ]
    await mw.abefore_model({"messages": state}, None)

    trimmed = mw._trim_for_summary(state[:2], 500, start_on=None)

    assert mw.token_counter(trimmed) <= 500 < mw.token_counter(state[:2])


def test_the_sync_path_is_refused():
    mw = _mw(10_000)
    with pytest.raises(NotImplementedError):
        mw.before_model({"messages": [HumanMessage("q", id="q")]}, None)


# ===================== the overhead O =====================


def _english_pair_state():
    """~6000 estimated tokens ending with the current question, measured on
    the last answer (r = 1.0)."""
    return [
        HumanMessage(_text(2000), id="h0"),
        AIMessage(_text(2000), id="a0"),
        HumanMessage(_text(1500), id="h1"),
        _stamped(_text(1000), 1000, 1000, msg_id="a1", output_tokens=1000),
        HumanMessage(_text(100), id="q"),
    ]


async def test_the_trigger_fires_on_count_plus_overhead():
    state = _english_pair_state()
    window = 10_000
    mw = _mw(window, overhead=0)
    assert await mw.abefore_model({"messages": state}, None) is None
    assert mw.token_counter(state) < int(0.8 * window)

    with_overhead = _mw(window, overhead=2000)
    after = _after(await with_overhead.abefore_model({"messages": state}, None))
    assert [m.id for m in after[1:]] == ["a1", "q"]


async def test_the_fixed_overhead_costs_the_anchors_ratio():
    mw = _mw(100_000, overhead=1000)
    state = [HumanMessage("q", id="q"), _stamped("a", 1500, 1000, msg_id="a"), HumanMessage("n")]
    await mw.abefore_model({"messages": state}, None)
    assert mw._overhead == 1500


async def test_the_kb_additions_cost_their_own_weight_not_the_ratio():
    block = "北京" * 400
    overhead = RequestOverhead(fixed_est=100, fixed_text="system", added_text=block)
    mw = _mw(100_000, overhead=overhead)
    state = [HumanMessage("q", id="q"), _stamped("a", 900, 1000, msg_id="a"), HumanMessage("n")]
    await mw.abefore_model({"messages": state}, None)
    assert mw._overhead == overhead_tokens(
        overhead, 0.9, dense=COUNTER_DENSE_TOKENS, weight_floor=FLOOR
    )
    assert mw._overhead == 90 + 800  # the block at 4 tokens per character, not at 0.9


def test_the_keep_budget_subtracts_the_overhead():
    window = 10_000
    state = _alternating_kept(window)
    no_overhead = compaction_cutoff(state, window, approx_token_count)
    with_overhead = compaction_cutoff(state, window, approx_token_count, overhead=2500)
    assert with_overhead > no_overhead
    assert approx_token_count(state[with_overhead:]) <= keep_token_budget(window) - 2500
    # Never below one token of keep.
    huge = compaction_cutoff(state, window, approx_token_count, overhead=10**6)
    assert huge == len(state) - 2  # the answer before the current question


def _alternating_kept(window):
    out = []
    for i in range(12):
        cls = HumanMessage if i % 2 == 0 else AIMessage
        out.append(cls(_text(window // 20, "k"), id=f"k{i}"))
    out.append(HumanMessage("current", id="cur"))
    return out


async def test_an_overhead_larger_than_the_trim_budget_never_empties_the_summary():
    summary = _ScriptedSummaryModel(messages=iter([]))
    mw = _mw(10_000, overhead=50_000, summary=summary)
    state = [
        HumanMessage(_text(1500), id="h0"),
        AIMessage(_text(1500), id="a0"),
        HumanMessage("my name is Ada", id="h1"),
        AIMessage(_text(1500), id="a1"),
        HumanMessage("q", id="q"),
    ]

    after = _after(await mw.abefore_model({"messages": state}, None))

    # One summary call, on a non-empty trim (an empty one would have taken
    # the truncation retry), and the summary written into the state.
    (prompt,) = summary.prompts
    assert "my name is Ada" in prompt
    assert "A short summary." in after[0].content


# ===================== character helpers by the message's own weight =====================


def test_the_truncated_copy_fits_the_trim_at_its_own_weight():
    huge = AIMessage("长城" * 20_000, id="huge")
    weights = {"huge": script_weight(huge.content, dense=COUNTER_DENSE_TOKENS)}

    def counter(messages):
        return real_token_count(messages, weights)

    out = runner_module._truncate_for_summary(
        [huge], 1000, counter, lambda m: weights.get(m.id, counter_weight(m))
    )
    assert counter(out) <= 1000


def test_the_placeholder_carries_at_most_the_cap_in_real_tokens():
    previous = _summary("长城" * 5000)
    text = runner_module._summary_placeholder(previous, 500)
    carried = text[: -len(runner_module.SUMMARY_LATER_LOST)].rstrip()
    weight = counter_weight(HumanMessage(carried))
    assert math.ceil(weight * len(carried) / 4) <= 500
    # chars/4 alone would have carried four times the cap in real tokens.
    assert len(carried) < 500 * 4 / 3


def test_the_trims_allocated_term_uses_the_heterogeneity_factor_2():
    assert runner_module.SUMMARY_TRIM_HETEROGENEITY_FACTOR == 2
    window = allocated = 10_000
    assert (
        summarize_trim_budget(window, allocated) == (allocated - 2 * summary_cap(window) - 512) // 2
    )


# ===================== counted as sent =====================


async def test_large_past_kb_results_count_as_markers_for_trigger_gain_and_keep():
    window = 10_000
    state = _kb_round(1, 6000) + _kb_round(2, 6000) + [HumanMessage("question 3", id="q3")]
    assert approx_token_count(state) > window
    mw = _mw(window)

    assert await mw.abefore_model({"messages": state}, None) is None


async def test_a_kb_pool_yields_a_summarizer_prompt_within_the_trim_budget():
    summary = _ScriptedSummaryModel(messages=iter([]))
    mw = _mw(10_000, summary=summary)
    await mw.abefore_model({"messages": [HumanMessage("q", id="q")]}, None)
    pool = _kb_round(1, 6000) + _kb_round(2, 6000) + _kb_round(3, 6000)

    await mw._acreate_summary(pool)

    (prompt,) = summary.prompts
    assert KB_MARKER in prompt
    assert "rrrrrrrr" not in prompt
    assert real_token_count([HumanMessage(prompt)]) <= mw.trim_tokens_to_summarize + 400


async def test_the_checkpoint_keeps_the_full_results_after_a_compaction():
    """The counter's markers never reach the state: a kept past result is
    written back whole (the request strips it again, at send time)."""
    window = 10_000
    state = (
        [HumanMessage(_text(3000), id="h0"), AIMessage(_text(3000), id="a0")]
        + _kb_round(1, 2000)
        + [HumanMessage(_text(2000), id="q2")]
    )
    mw = _mw(window, overhead=1000)

    after = _after(await mw.abefore_model({"messages": state}, None))

    kept_results = [m for m in after if isinstance(m, ToolMessage)]
    assert kept_results and kept_results[0].content.startswith("rrrr")


# ===================== the futility guard =====================


def _counter_r1(messages):
    return approx_token_count(messages)


def test_no_guard_without_a_window():
    state = [
        _summary(),
        AIMessage("a"),
        HumanMessage("b"),
        AIMessage(_text(7000)),
        HumanMessage("q"),
    ]
    assert compaction_cutoff(state, None, _counter_r1, overhead=5000) == compaction_cutoff(
        state, None, _counter_r1
    )


def _tiny_gain_state():
    return [
        _summary(_text(200)),
        AIMessage(_text(50), id="a0"),
        HumanMessage(_text(100), id="h1"),
        AIMessage(_text(7000), id="a1"),
        HumanMessage(_text(900), id="q"),
    ]


def test_a_tiny_gain_compaction_is_skipped():
    state = _tiny_gain_state()
    window = 10_000
    assert _counter_r1(state) >= int(0.8 * window)
    assert compaction_cutoff(state, window, _counter_r1, allocated_window=window) == 0


def test_a_small_gain_is_taken_when_the_request_would_overflow_the_allocated_window():
    state = [
        _summary(_text(200)),
        AIMessage(_text(400), id="a0"),
        HumanMessage(_text(400), id="h1"),
        AIMessage(_text(6600), id="a1"),
        HumanMessage(_text(900), id="q"),
    ]
    window = 10_000
    total = _counter_r1(state)
    kept = _counter_r1(state[3:])
    gain = total - kept - summary_cap(window)
    assert 0 < gain < 256
    assert compaction_cutoff(state, window, _counter_r1, allocated_window=10_000) == 0
    # The same state on an allocation it overflows (count + floor > W_alloc).
    tight = total + 512 - 1
    assert compaction_cutoff(state, window, _counter_r1, allocated_window=tight) == 3
    # ...or on the working window itself (O + count > W).
    assert compaction_cutoff(state, window, _counter_r1, overhead=window - total + 1) == 3


def test_a_compaction_with_no_gain_is_skipped_when_nothing_overflows():
    state = _tiny_gain_state()
    total = _counter_r1(state)
    assert total - _counter_r1(state[3:]) - summary_cap(10_000) <= 0
    assert compaction_cutoff(state, 10_000, _counter_r1, allocated_window=10_000) == 0


def test_the_20_message_clause_alone_keeps_3_2a_behaviour():
    window = 100_000
    state = [
        (HumanMessage if i % 2 == 0 else AIMessage)(_text(20), id=f"m{i}") for i in range(30)
    ] + [HumanMessage("q", id="q")]
    assert _counter_r1(state) < int(0.8 * window)
    assert compaction_cutoff(state, window, _counter_r1) > 0


async def test_the_8192_example_compacts_and_does_not_loop():
    """W = 8192, O = 2000: a summary, A1, H2, A2 (measured) and a long current
    question overflow the trigger; the cutoff keeps A2 and H3, and the next
    hop compacts nothing (rule 5) although the kept tail stays above the
    trigger."""
    summary = _ScriptedSummaryModel(messages=iter([]), script=[_text(1024, "s")])
    mw = _mw(8192, overhead=2000, summary=summary)
    state = [
        _summary(_text(880)),
        AIMessage(_text(495), id="A1"),
        HumanMessage(_text(495), id="H2"),
        _stamped(_text(595), 1000, 1000, msg_id="A2", output_tokens=600),
        HumanMessage(_text(3795), id="H3"),
    ]

    after = _after(await mw.abefore_model({"messages": state}, None))

    assert [m.id for m in after[1:]] == ["A2", "H3"]
    assert await mw.abefore_model({"messages": after}, None) is None
    assert len(summary.prompts) == 1


async def test_a_post_compaction_state_and_a_summary_less_state_do_not_loop_per_hop():
    summary = _ScriptedSummaryModel(messages=iter([]))
    mw = _mw(4096, overhead=500, summary=summary)
    state = [
        HumanMessage(_text(1500), id="h0"),
        AIMessage(_text(1500), id="a0"),
        HumanMessage(_text(1500), id="q"),
    ]
    first = await mw.abefore_model({"messages": state}, None)
    after = _after(first) if first is not None else state
    for _ in range(3):
        assert await mw.abefore_model({"messages": after}, None) is None


# ===================== the summary carries nothing; the turn wiring =====================


async def test_a_compaction_writes_the_stock_summary_message():
    mw = _mw(10_000)
    state = [
        HumanMessage(_text(3000), id="h0"),
        _stamped(_text(3000), 1000, 1000, msg_id="a0", output_tokens=3000),
        HumanMessage(_text(3000), id="h1"),
        AIMessage(_text(500), id="a1"),
        HumanMessage("q", id="q"),
    ]

    after = _after(await mw.abefore_model({"messages": state}, None))

    assert after[0].additional_kwargs == {"lc_source": "summarization"}


async def test_after_a_kb_turn_the_next_trigger_weighs_the_history_at_the_first_hop_ratio():
    mw = _mw(100_000)
    state = _kb_round(1, 3000, first_r=1.1, last_r=2.14) + [HumanMessage("next", id="q2")]
    await mw.abefore_model({"messages": state}, None)
    # The anchor is the turn's first hop: the question before it at 1.1, not
    # at the last hop's 2.14.
    assert mw.token_counter([state[0]]) == _cost(state[0], 1.1)


async def test_the_runner_hands_the_kb_text_to_the_client_and_reads_no_checkpoint(monkeypatch):
    _WindowEngine.window = 100_000
    captured: list = []
    main = _ScriptedSummaryModel(messages=iter([]), script=["the answer"])

    def _build(llm, **kw):
        captured.append(kw)
        return main

    monkeypatch.setattr(runner_module, "build_chat_model", _build)
    cp = InMemorySaver()
    reads = []
    real_get_tuple = cp.aget_tuple

    async def _counting_get_tuple(config):
        reads.append(config)
        return await real_get_tuple(config)

    monkeypatch.setattr(cp, "aget_tuple", _counting_get_tuple)

    events = [
        e
        async for e in AgentRunner(checkpointer=cp).astream_text(
            llm=_Llm(),
            user_message="next question",
            system_prompt="system",
            params=_PARAMS,
            thread_id="kb",
            summarize=True,
            kb_context_block="EXCERPTS",
            kb_language_line="Answer in English.",
            emit_events=True,
        )
    ]

    assert any(e["t"] == "answer" for e in events)
    main_kwargs = next(kw for kw in captured if kw.get("record_usage"))
    assert main_kwargs["kb_additions"] == "EXCERPTS\n\n\n\nAnswer in English."


# ===================== the warning projection =====================


class _RecordingBudget:
    weights_bytes = 10

    def __init__(self, margin=0.5):
        self.calls = []
        self.margin = margin

    def memory_margin_fraction(self, tokens):
        self.calls.append(tokens)
        return self.margin

    def conversation_bytes(self, tokens):
        return tokens

    def footprint_bytes(self, tokens):
        return tokens + self.weights_bytes

    def used_fraction(self, tokens):
        return 1.0 - self.margin


def _agent_with_state(messages):
    async def aget_state(config):
        return SimpleNamespace(values={"messages": messages})

    return SimpleNamespace(aget_state=aget_state)


def _projection(state):
    projected = strip_stale_tool_results(state, all_past=True) + [
        HumanMessage("", id="erudi-next-request")
    ]
    weights, ratio = frozen_weights(projected)
    return projected, (lambda part: real_token_count(part, weights)), ratio


async def test_the_projection_is_overhead_plus_all_when_nothing_would_be_compacted():
    state = [HumanMessage("hello", id="h"), AIMessage("hi", id="a")]
    budget = _RecordingBudget()

    await AgentRunner()._memory_warning_event(
        _agent_with_state(state), {}, budget, 10_000, overhead=_fixed(400), allocated_window=10_000
    )

    projected, counter, _ = _projection(state)
    # Nothing measured: O at its own script weight, floored.
    assert budget.calls[0] == math.ceil(FLOOR * 400) + counter(projected)


async def test_the_projection_is_overhead_plus_kept_plus_the_cap_otherwise():
    window = 10_000
    state = [
        HumanMessage(_text(3000), id="h0"),
        _stamped(_text(3000), 1000, 1000, msg_id="a0", output_tokens=3000),
        HumanMessage(_text(3000), id="h1"),
        AIMessage(_text(500), id="a1"),
    ]
    budget = _RecordingBudget()
    projected, counter, ratio = _projection(state)
    assert ratio == 1.0
    cut = compaction_cutoff(projected, window, counter, overhead=300, allocated_window=window)
    assert cut > 0

    await AgentRunner()._memory_warning_event(
        _agent_with_state(state), {}, budget, window, overhead=_fixed(300), allocated_window=window
    )

    assert budget.calls[0] == 300 + counter(projected[cut:]) + summary_cap(window)


async def test_the_projection_keeps_what_the_next_requests_compaction_keeps():
    """The empty next question is appended: the cutoff cannot stop on the
    question of the turn just answered as if it were still the current one."""
    window = 4096
    state = [
        HumanMessage(_text(900), id="h0"),
        AIMessage(_text(900), id="a0"),
        HumanMessage(_text(900), id="h1"),
        AIMessage(_text(900), id="a1"),
    ]
    projected, counter, _ = _projection(state)
    cut = compaction_cutoff(projected, window, counter, allocated_window=window)
    budget = _RecordingBudget()

    await AgentRunner()._memory_warning_event(
        _agent_with_state(state), {}, budget, window, allocated_window=window
    )

    assert cut == 3  # the last answer is kept; the sentinel is the "current question"
    assert budget.calls[0] == counter(projected[cut:]) + summary_cap(window)


async def test_the_projection_counts_the_turn_just_answered_as_past():
    window = 100_000
    state = _kb_round(1, 5000, first_r=1.0, last_r=2.14)
    budget = _RecordingBudget()

    await AgentRunner()._memory_warning_event(
        _agent_with_state(state), {}, budget, window, allocated_window=window
    )

    projected, counter, _ = _projection(state)
    assert projected[2].content == KB_MARKER
    assert budget.calls[0] == counter(projected)


async def test_the_warning_payload_includes_the_overhead():
    state = [HumanMessage("hello", id="h"), AIMessage("hi", id="a")]
    budget = _RecordingBudget(margin=0.01)

    event = await AgentRunner()._memory_warning_event(
        _agent_with_state(state), {}, budget, 10_000, overhead=_fixed(400), allocated_window=10_000
    )

    projected, counter, _ = _projection(state)
    expected = math.ceil(FLOOR * 400) + counter(projected)
    assert event["conversation_bytes"] == expected
    assert event["footprint_bytes"] == expected + budget.weights_bytes


async def test_the_runner_projects_without_the_kb_block_of_the_turn_just_answered(monkeypatch):
    seen = {}

    async def _spy(self, agent, run_config, budget, working_window=None, **kwargs):
        seen.update(kwargs)
        return None

    monkeypatch.setattr(AgentRunner, "_memory_warning_event", _spy)
    main = _ScriptedSummaryModel(messages=iter([]), script=["the answer"])
    monkeypatch.setattr(runner_module, "build_chat_model", lambda llm, **kw: main)

    _ = [
        e
        async for e in AgentRunner(checkpointer=InMemorySaver()).astream_text(
            llm=_Llm(),
            user_message="q",
            system_prompt="system",
            params=_PARAMS,
            thread_id="w",
            summarize=True,
            kb_context_block="EXCERPTS " * 100,
            kb_language_line="Answer in English.",
            emit_events=True,
        )
    ]

    assert seen["overhead"].added_text == ""
    assert seen["overhead"].fixed_est > 0


# ===================== llama.cpp engines =====================


class _GgufEngine(BaseEngine):
    FORMAT_TAG = "gguf"
    window: ClassVar[Any] = 8192

    @classmethod
    def effective_context_tokens(cls):
        return cls.window


async def test_llama_cpp_engines_trigger_in_real_tokens_without_a_memory_ceiling(monkeypatch):
    from src.engines.memory_budget import MemoryBudget

    monkeypatch.setattr(config, "LLM_Engine", _GgufEngine)
    budget = MemoryBudget.from_engine(_GgufEngine)
    assert AgentRunner._compaction_windows(budget) == (8192, 8192)
    built = AgentRunner()._build_middleware(_ScriptedSummaryModel(messages=iter([])), budget)
    mw = built[-1]
    # Unscaled, this state is under the 80 % trigger; with the question before
    # the measured hop at r = 2.0 it is over it.
    state = [
        HumanMessage(_text(3000), id="h0"),
        _stamped(_text(500), 2000, 1000, msg_id="a0", output_tokens=500),
        HumanMessage(_text(500), id="q"),
    ]
    assert approx_token_count(state) < int(0.8 * 8192)

    assert await mw.abefore_model({"messages": state}, None) is not None


# ===================== the private LangChain surface this rests on =====================


def test_the_extended_langchain_surface_is_pinned():
    import inspect

    from langchain.agents.middleware import SummarizationMiddleware

    assert list(inspect.signature(SummarizationMiddleware._should_summarize).parameters) == [
        "self",
        "messages",
        "total_tokens",
    ]
    assert list(inspect.signature(SummarizationMiddleware._ensure_message_ids).parameters) == [
        "messages"
    ]
    source = inspect.getsource(SummarizationMiddleware.abefore_model)
    assert "self._should_summarize(messages, total_tokens)" in source
