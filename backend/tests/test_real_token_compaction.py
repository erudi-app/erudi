"""Compaction in real tokens: a measured ratio, the request overhead, as sent.

The counter is ``ceil(r * chars/4)`` over the messages AS SENT (past KB/web
results as their markers, designated by message id), with r frozen once per
``abefore_model`` from the full state. The request overhead O (system prompt,
KB block, tool schemas -- not in the state) enters only where the whole
request matters: the trigger, the keep and the warning projection. A futility
guard skips a compaction that gains less than 256 tokens unless the request
would overflow without it.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.agents import runner as runner_module
from src.agents.middleware import STALE_TOOL_RESULT_MARKERS, strip_stale_tool_results
from src.agents.runner import (
    AgentRunner,
    approx_token_count,
    compaction_cutoff,
    keep_token_budget,
    real_token_count,
    summarize_trim_budget,
    summary_cap,
)
from src.agents.token_accounting import (
    FIRST_HOP_RATIO_KWARG,
    REQUEST_EST_KEY,
    REQUEST_FIRST_HOP_KEY,
    REQUEST_HAS_IMAGES_KEY,
)
from src.core import config
from src.engines.base_engine import BaseEngine
from tests.test_compaction_keep import (  # noqa: F401  (``_engine`` is an autouse fixture)
    _PARAMS,
    _Llm,
    _ScriptedSummaryModel,
    _WindowEngine,
    _engine,
    _models,
    _seed,
    _state,
    _text,
)

pytestmark = pytest.mark.unit

KB = "search_knowledge_base"
KB_MARKER = STALE_TOOL_RESULT_MARKERS[KB]


def _stamped(content, input_tokens, est, *, first=True, msg_id=None, **kwargs):
    return AIMessage(
        content=content,
        id=msg_id,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": 5,
            "total_tokens": input_tokens + 5,
        },
        response_metadata={
            REQUEST_EST_KEY: est,
            REQUEST_HAS_IMAGES_KEY: False,
            REQUEST_FIRST_HOP_KEY: first,
        },
        **kwargs,
    )


def _summary(text="facts", ratio=None, msg_id="s0"):
    kwargs = {"lc_source": "summarization"}
    if ratio is not None:
        kwargs[FIRST_HOP_RATIO_KWARG] = ratio
    return HumanMessage(
        content=f"Here is a summary of the conversation to date:\n\n{text}",
        additional_kwargs=kwargs,
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


def _mw(window, *, overhead_est=0, budget=None, summary=None):
    from langchain.agents.middleware import SummarizationMiddleware

    _WindowEngine.window = window
    summary = summary or _ScriptedSummaryModel(messages=iter([]))
    built = AgentRunner()._build_middleware(summary, budget, overhead_est=overhead_est)
    return next(m for m in built if isinstance(m, SummarizationMiddleware))


def _after(result) -> list:
    assert result is not None
    return list(result["messages"][1:])


# ===================== the counter =====================


async def test_the_counter_is_ceil_r_times_chars4_with_past_results_as_markers():
    mw = _mw(100_000)
    state = _kb_round(1, 3000, first_r=2.0) + [HumanMessage("next question", id="q2")]

    assert await mw.abefore_model({"messages": state}, None) is None

    as_sent = strip_stale_tool_results(state)
    assert as_sent[2].content == KB_MARKER
    assert mw.token_counter(state) == math.ceil(2.0 * approx_token_count(as_sent))
    assert mw.token_counter(state) < approx_token_count(state)


async def test_r_is_frozen_per_call_so_a_suffix_without_ai_message_counts_at_it():
    mw = _mw(100_000)
    state = _kb_round(1, 10, first_r=2.0) + [HumanMessage("next question", id="q2")]
    await mw.abefore_model({"messages": state}, None)

    suffix = [HumanMessage(_text(400), id="x")]
    assert mw.token_counter(suffix) == math.ceil(2.0 * approx_token_count(suffix))


async def test_past_results_count_as_markers_even_when_the_call_id_repeats_every_turn():
    mw = _mw(100_000)
    # Both turns' calls carry the SAME call id; only the past result is a marker.
    state = (
        _kb_round(1, 3000, first_r=1.0)
        + [HumanMessage("question 2", id="q2")]
        + [
            _stamped(
                "",
                1000,
                1000,
                msg_id="call2",
                tool_calls=[{"name": KB, "args": {}, "id": "functions.kb:0"}],
            ),
            ToolMessage(_text(3000, "c"), name=KB, tool_call_id="functions.kb:0", id="res2"),
        ]
    )
    await mw.abefore_model({"messages": state}, None)

    assert mw.token_counter([state[2]]) == approx_token_count(
        [state[2].model_copy(update={"content": KB_MARKER})]
    )
    assert mw.token_counter([state[-1]]) == approx_token_count([state[-1]])


async def test_without_a_measurement_the_counter_uses_max_1_5_and_the_script_ratio():
    mw = _mw(100_000)
    english = [HumanMessage("hello there", id="q1")]
    await mw.abefore_model({"messages": english}, None)
    assert mw.token_counter(english) == math.ceil(1.5 * approx_token_count(english))

    cjk = [HumanMessage("这是一个很长的中文句子" * 30, id="q1")]
    await mw.abefore_model({"messages": cjk}, None)
    assert mw.token_counter(cjk) > math.ceil(3.0 * approx_token_count(cjk))


def test_the_counter_is_monotone_for_the_binary_search():
    messages = [HumanMessage(_text(i * 10), id=f"m{i}") for i in range(1, 30)]
    counts = [real_token_count(messages[i:], 2.37) for i in range(len(messages))]
    assert counts == sorted(counts, reverse=True)


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


# ===================== the overhead O =====================


def _english_pair_state():
    """~7100 real tokens at a measured r = 1.0, ending with the current question."""
    return [
        HumanMessage(_text(2000), id="h0"),
        _stamped(_text(2000), 1000, 1000, msg_id="a0"),
        HumanMessage(_text(2000), id="h1"),
        AIMessage(_text(1000), id="a1"),
        HumanMessage(_text(100), id="q"),
    ]


async def test_the_trigger_fires_on_count_plus_overhead():
    state = _english_pair_state()
    window = 10_000
    assert approx_token_count(state) < int(0.8 * window)

    assert await _mw(window, overhead_est=0).abefore_model({"messages": state}, None) is None
    with_overhead = _mw(window, overhead_est=1500)
    after = _after(await with_overhead.abefore_model({"messages": state}, None))
    assert [m.id for m in after[1:]] == ["a1", "q"]


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
    mw = _mw(10_000, overhead_est=50_000, summary=summary)
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


# ===================== character helpers in real units =====================


def test_with_r_2_5_the_truncated_copy_fits_the_trim():
    counter = lambda messages: real_token_count(messages, 2.5)  # noqa: E731
    out = runner_module._truncate_for_summary(
        [AIMessage(_text(10_000, "z"), id="huge")], 1000, counter, 2.5
    )
    assert counter(out) <= 1000


def test_the_placeholder_carries_at_most_the_cap_in_real_tokens():
    previous = _summary(_text(5000, "x"))
    text = runner_module._summary_placeholder(previous, 500, 2.5)
    carried = text[: -len(runner_module.SUMMARY_LATER_LOST)].rstrip()
    assert math.ceil(2.5 * len(carried) / 4) <= 500
    # chars/4 alone would have carried 2.5x the cap in real tokens.
    assert len(carried) == int(500 * 4 / 2.5)


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
    assert real_token_count([HumanMessage(prompt)], mw._ratio) <= mw.trim_tokens_to_summarize + 300


async def test_the_checkpoint_keeps_the_full_results_after_a_compaction():
    """The counter's markers never reach the state: a kept past result is
    written back whole (the request strips it again, at send time)."""
    window = 10_000
    state = (
        [HumanMessage(_text(3000), id="h0"), AIMessage(_text(3000), id="a0")]
        + _kb_round(1, 2000)
        + [HumanMessage(_text(2000), id="q2")]
    )
    mw = _mw(window, overhead_est=1000)

    after = _after(await mw.abefore_model({"messages": state}, None))

    kept_results = [m for m in after if isinstance(m, ToolMessage)]
    assert kept_results and kept_results[0].content.startswith("rrrr")


# ===================== the futility guard =====================


def _counter_r1(messages):
    return real_token_count(messages, 1.0)


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
    """W = 8192, O = 2000: summary 900 + A1 500 + H2 500 + A2 600 + current
    question 3800 overflows (8300 > 0.8 W); the cutoff keeps A2 and H3,
    summarizing ~1900 for a gain of ~876, and the next hop compacts nothing
    (rule 5) although the kept tail stays above the trigger."""
    summary = _ScriptedSummaryModel(messages=iter([]), script=[_text(1024, "s")])
    mw = _mw(8192, overhead_est=2000, summary=summary)
    state = [
        _summary(_text(880), ratio=1.0),
        AIMessage(_text(495), id="A1"),
        HumanMessage(_text(495), id="H2"),
        AIMessage(_text(595), id="A2"),
        HumanMessage(_text(3795), id="H3"),
    ]

    after = _after(await mw.abefore_model({"messages": state}, None))

    assert [m.id for m in after[1:]] == ["A2", "H3"]
    assert await mw.abefore_model({"messages": after}, None) is None
    assert len(summary.prompts) == 1


async def test_a_post_compaction_state_and_a_summary_less_state_do_not_loop_per_hop():
    summary = _ScriptedSummaryModel(messages=iter([]))
    mw = _mw(4096, overhead_est=500, summary=summary)
    state = [
        HumanMessage(_text(1500), id="h0"),
        AIMessage(_text(1500), id="a0"),
        HumanMessage(_text(1500), id="q"),
    ]
    first = await mw.abefore_model({"messages": state}, None)
    after = _after(first) if first is not None else state
    for _ in range(3):
        assert await mw.abefore_model({"messages": after}, None) is None


# ===================== the summary carries the first-hop ratio =====================


async def test_a_mid_turn_compaction_carries_the_current_turns_first_hop_ratio():
    summary = _ScriptedSummaryModel(messages=iter([]))
    mw = _mw(10_000, summary=summary)
    state = (
        [HumanMessage(_text(3500), id="h0"), AIMessage(_text(3500), id="a0")]
        + _kb_round(1, 10, first_r=1.0, last_r=2.14)
        + [
            HumanMessage(_text(200), id="cur"),
            _stamped(
                "",
                1100,
                1000,
                msg_id="cur-call",
                tool_calls=[{"name": KB, "args": {}, "id": "c9"}],
            ),
            ToolMessage(_text(1500), name=KB, tool_call_id="c9", id="cur-res"),
        ]
    )

    after = _after(await mw.abefore_model({"messages": state}, None))

    assert after[0].additional_kwargs[FIRST_HOP_RATIO_KWARG] == pytest.approx(1.1)


async def test_a_second_compaction_without_a_stamped_first_hop_propagates_the_carried_value():
    mw = _mw(10_000)
    state = [
        _summary("old", ratio=1.3),
        AIMessage(_text(3000), id="a0"),
        HumanMessage(_text(3000), id="h1"),
        AIMessage(_text(3000), id="a1"),
        HumanMessage("q", id="q"),
    ]

    after = _after(await mw.abefore_model({"messages": state}, None))

    assert after[0].additional_kwargs[FIRST_HOP_RATIO_KWARG] == pytest.approx(1.3)


async def test_nothing_measured_leaves_the_key_absent():
    mw = _mw(10_000)
    state = [
        HumanMessage(_text(3000), id="h0"),
        AIMessage(_text(3000), id="a0"),
        HumanMessage(_text(3000), id="h1"),
        AIMessage(_text(500), id="a1"),
        HumanMessage("q", id="q"),
    ]

    after = _after(await mw.abefore_model({"messages": state}, None))

    assert FIRST_HOP_RATIO_KWARG not in after[0].additional_kwargs
    assert mw._ratio == 1.5


# ===================== counter and budget agree on the first-hop ratio =====================


async def test_after_a_kb_turn_the_next_trigger_uses_the_first_hop_ratio():
    mw = _mw(100_000)
    state = _kb_round(1, 3000, first_r=1.1, last_r=2.14) + [HumanMessage("next", id="q2")]
    await mw.abefore_model({"messages": state}, None)
    assert mw._ratio == pytest.approx(1.1)

    compacted = [_summary(ratio=1.1), state[3], HumanMessage("next", id="q2")]
    await mw.abefore_model({"messages": compacted}, None)
    assert mw._ratio == pytest.approx(1.1)


@pytest.mark.parametrize("system_role", [True, False], ids=["system-role", "gemma-fold"])
@pytest.mark.parametrize("compacted", [False, True], ids=["full", "after-compaction"])
async def test_the_next_turns_budget_reads_the_first_hop_ratio_from_the_raw_state(
    monkeypatch, system_role, compacted
):
    import src.engines.system_role_capability as role_module

    _WindowEngine.window = 100_000
    captured: list = []
    main = _ScriptedSummaryModel(messages=iter([]), script=["the answer"])

    def _build(llm, **kw):
        captured.append(kw)
        return main

    monkeypatch.setattr(runner_module, "build_chat_model", _build)
    monkeypatch.setattr(role_module, "model_supports_system_role", lambda link: system_role)
    cp = InMemorySaver()
    turn = _kb_round(1, 3000, first_r=1.1, last_r=2.14)
    seed = [_summary(ratio=1.1), turn[3]] if compacted else turn
    await _seed(cp, "ratio", seed)

    events = [
        e
        async for e in AgentRunner(checkpointer=cp).astream_text(
            llm=_Llm(),
            user_message="next question",
            system_prompt="system",
            params=_PARAMS,
            thread_id="ratio",
            summarize=True,
            emit_events=True,
        )
    ]

    assert any(e["t"] == "answer" for e in events)
    main_kwargs = next(kw for kw in captured if kw.get("preflight_retry", True))
    assert main_kwargs["prompt_ratio"] == pytest.approx(1.1)
    summary_kwargs = next(kw for kw in captured if kw.get("preflight_retry") is False)
    assert summary_kwargs.get("prompt_ratio") is None


async def test_a_stateless_turn_has_no_prompt_ratio(monkeypatch):
    captured: list = []
    main = _ScriptedSummaryModel(messages=iter([]), script=["the answer"])

    def _build(llm, **kw):
        captured.append(kw)
        return main

    monkeypatch.setattr(runner_module, "build_chat_model", _build)

    _ = [
        e
        async for e in AgentRunner().astream_text(
            llm=_Llm(), user_message="hi", system_prompt="s", params=_PARAMS
        )
    ]

    assert captured[0].get("prompt_ratio") is None


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


def _agent_with_state(messages):
    async def aget_state(config):
        return SimpleNamespace(values={"messages": messages})

    return SimpleNamespace(aget_state=aget_state)


async def test_the_projection_is_overhead_plus_all_when_nothing_would_be_compacted():
    state = [HumanMessage("hello", id="h"), AIMessage("hi", id="a")]
    budget = _RecordingBudget()

    await AgentRunner()._memory_warning_event(
        _agent_with_state(state), {}, budget, 10_000, overhead_est=400, allocated_window=10_000
    )

    ratio = 1.5  # nothing measured
    assert budget.calls[0] == math.ceil(ratio * 400) + real_token_count(state, ratio)


async def test_the_projection_is_overhead_plus_kept_plus_the_cap_otherwise():
    window = 10_000
    state = [
        HumanMessage(_text(3000), id="h0"),
        _stamped(_text(3000), 1000, 1000, msg_id="a0"),
        HumanMessage(_text(3000), id="h1"),
        AIMessage(_text(500), id="a1"),
    ]
    budget = _RecordingBudget()
    counter = lambda m: real_token_count(m, 1.0)  # noqa: E731
    cut = compaction_cutoff(state, window, counter, overhead=300, allocated_window=window)
    assert cut > 0

    await AgentRunner()._memory_warning_event(
        _agent_with_state(state), {}, budget, window, overhead_est=300, allocated_window=window
    )

    assert budget.calls[0] == 300 + counter(state[cut:]) + summary_cap(window)


async def test_the_projection_counts_the_turn_just_answered_as_past():
    window = 100_000
    state = _kb_round(1, 5000, first_r=1.0, last_r=2.14)
    budget = _RecordingBudget()

    await AgentRunner()._memory_warning_event(
        _agent_with_state(state), {}, budget, window, overhead_est=0, allocated_window=window
    )

    as_sent = strip_stale_tool_results(state, all_past=True)
    assert budget.calls[0] == real_token_count(as_sent, 1.0)


async def test_the_warning_payload_includes_the_overhead():
    state = [HumanMessage("hello", id="h"), AIMessage("hi", id="a")]
    budget = _RecordingBudget(margin=0.01)

    event = await AgentRunner()._memory_warning_event(
        _agent_with_state(state), {}, budget, 10_000, overhead_est=400, allocated_window=10_000
    )

    expected = math.ceil(1.5 * 400) + real_token_count(state, 1.5)
    assert event["conversation_bytes"] == expected
    assert event["footprint_bytes"] == expected + budget.weights_bytes


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
    # Unscaled, this state is under the 80 % trigger; at the measured r = 2.0
    # it is over it.
    state = [
        HumanMessage(_text(2000), id="h0"),
        _stamped(_text(1500), 2000, 1000, msg_id="a0"),
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
    assert list(inspect.signature(SummarizationMiddleware._build_new_messages).parameters) == [
        "summary"
    ]
    assert list(inspect.signature(SummarizationMiddleware._ensure_message_ids).parameters) == [
        "messages"
    ]
    source = inspect.getsource(SummarizationMiddleware.abefore_model)
    assert "self._build_new_messages(summary)" in source
    assert "self._should_summarize(messages, total_tokens)" in source


async def test_abefore_model_reaches_our_build_new_messages_through_self():
    mw = _mw(10_000)
    calls = []
    original = mw._build_new_messages

    def _spy(summary):
        calls.append(summary)
        return original(summary)

    mw._build_new_messages = _spy
    state = [
        HumanMessage(_text(3000), id="h0"),
        _stamped(_text(3000), 1000, 1000, msg_id="a0"),
        HumanMessage(_text(3000), id="h1"),
        AIMessage(_text(500), id="a1"),
        HumanMessage("q", id="q"),
    ]

    after = _after(await mw.abefore_model({"messages": state}, None))

    assert calls == ["A short summary."]
    assert after[0].additional_kwargs[FIRST_HOP_RATIO_KWARG] == 1.0
