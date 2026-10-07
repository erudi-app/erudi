"""Real-token accounting, message by message (plan 3.2b v17).

One ratio measured on a request of one composition is wrong on a request of
another, so a list of messages is costed per message: the messages before the
anchor k (the last stamped hop of the current turn, else the last stamped
first hop) at its measured r; an AI message without reasoning at its exact
``output_tokens``; everything else at the script weight of its own text. These
tests pin each rule; ``test_live_token_accounting.py`` pins them on the texts
and counts of the live run.
"""

from __future__ import annotations

import math

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import tool

from src.agents.token_accounting import (
    BUDGET_DENSE_TOKENS,
    COUNTER_DENSE_TOKENS,
    COUNTER_WEIGHT_FLOOR,
    RATIO_CEILING,
    RATIO_FLOOR,
    REQUEST_EST_KEY,
    REQUEST_FIRST_HOP_KEY,
    REQUEST_HAS_IMAGES_KEY,
    STALE_TOOL_RESULT_MARKERS,
    RequestOverhead,
    estimate,
    exact_tokens,
    first_hop_ratio,
    has_reasoning,
    hop_ratio,
    last_human_index,
    measured_anchor,
    message_weights,
    messages_have_images,
    overhead_tokens,
    real_tokens_est,
    request_overhead,
    request_tokens_est,
    script_weight,
    stale_result_copy,
    weighted_cost,
)

pytestmark = pytest.mark.unit


@tool
def search_knowledge_base(query: str) -> str:
    """Search the documents of the knowledge base."""
    return ""


def _hop(input_tokens, est, *, first=True, images=False, content="a", output_tokens=5, **kwargs):
    """An AI message as ``_astream`` leaves it: server usage + the stamp."""
    return AIMessage(
        content=content,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        response_metadata={
            REQUEST_EST_KEY: est,
            REQUEST_HAS_IMAGES_KEY: images,
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


def _budget_est(messages, **kwargs):
    return real_tokens_est(messages, dense=BUDGET_DENSE_TOKENS, **kwargs)


# ===================== the one estimator =====================


def test_the_request_estimator_is_count_tokens_approximately_with_its_defaults():
    messages = [SystemMessage("system"), HumanMessage("hello there")]
    assert request_tokens_est(messages) == count_tokens_approximately(messages)


def test_the_request_estimator_counts_the_tool_schemas_converted_alike():
    from langchain_core.utils.function_calling import convert_to_openai_tool

    messages = [HumanMessage("hello")]
    assert request_tokens_est(messages, [search_knowledge_base]) > request_tokens_est(messages)
    assert request_tokens_est(messages, [convert_to_openai_tool(search_knowledge_base)]) == (
        request_tokens_est(messages, [search_knowledge_base])
    )


def test_images_are_detected_in_content_parts():
    assert not messages_have_images([HumanMessage("text")])
    assert messages_have_images(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": "what is this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ]
            )
        ]
    )


def test_one_last_human_index_and_one_marker_helper():
    from src.agents import middleware

    messages = [HumanMessage("a"), AIMessage("b"), HumanMessage("c"), AIMessage("d")]
    assert last_human_index(messages) == 2
    assert last_human_index([AIMessage("x")]) is None
    result = ToolMessage("big", name="web_search", tool_call_id="c", id="t")
    marked = stale_result_copy(result)
    assert marked.content == STALE_TOOL_RESULT_MARKERS["web_search"] and marked.id == "t"
    assert stale_result_copy(ToolMessage("4", name="calculator", tool_call_id="c")).content == "4"
    assert middleware.STALE_TOOL_RESULT_MARKERS is STALE_TOOL_RESULT_MARKERS


# ===================== a hop's measured ratio =====================


def test_a_hop_ratio_is_input_tokens_over_the_stamped_estimate_clamped():
    assert hop_ratio(_hop(1100, 1000)) == pytest.approx(1.1)
    assert hop_ratio(_hop(100, 1000)) == RATIO_FLOOR
    assert hop_ratio(_hop(10_000, 1000)) == RATIO_CEILING


def test_an_image_or_unstamped_hop_has_no_ratio():
    assert hop_ratio(_hop(5000, 1000, images=True)) is None
    assert hop_ratio(AIMessage("a")) is None
    assert hop_ratio(_hop(1000, 0)) is None
    assert hop_ratio(HumanMessage("q")) is None


def test_first_hop_ratio_reads_the_last_stamped_first_hop_only():
    messages = [
        HumanMessage("q"),
        _hop(1100, 1000),
        HumanMessage("q2"),
        _hop(2140, 1000, first=False),
    ]
    assert first_hop_ratio(messages) == pytest.approx(1.1)
    assert first_hop_ratio([HumanMessage("q"), AIMessage("a")]) is None


# ===================== the anchor k =====================


def test_the_anchor_is_the_last_stamped_hop_of_the_current_turn():
    messages = [
        HumanMessage("q", id="q"),
        _hop(
            1100, 1000, msg_id="h1", content="", tool_calls=[{"name": "t", "args": {}, "id": "c"}]
        ),
        ToolMessage("result", tool_call_id="c", id="t"),
        _hop(2000, 1000, first=False, content=""),
    ]
    assert measured_anchor(messages) == (3, pytest.approx(2.0))


def test_without_a_hop_after_the_last_question_the_anchor_is_the_last_first_hop():
    messages = [
        HumanMessage("q1"),
        _hop(1100, 1000),
        ToolMessage("result", tool_call_id="c"),
        _hop(2140, 1000, first=False),
        HumanMessage("q2"),
    ]
    assert measured_anchor(messages) == (1, pytest.approx(1.1))


def test_nothing_measured_means_no_anchor():
    assert measured_anchor([HumanMessage("q"), AIMessage("a"), HumanMessage("q2")]) == (None, None)


# ===================== the script weight =====================


def test_english_prose_reads_one_and_digits_read_one_token_each():
    assert script_weight("The quick brown fox jumps over the lazy dog. " * 10, dense=1.0) == 1.0
    assert script_weight("2026", dense=1.0) == pytest.approx(4.0)  # four tokens, chars/4 says one
    assert script_weight("in 1854 near kilometre 42", dense=1.0) > 1.4


def test_french_russian_and_cjk():
    text = ("le cafe est servi dans la petite salle du chateau pres de la riviere " * 3)[:92]
    assert script_weight(text + "éèàç", dense=1.0) == pytest.approx(1.025, abs=0.01)
    russian = "Привет, как дела? Сегодня хорошая погода и мы идём гулять в парк. " * 10
    assert 1.35 <= script_weight(russian, dense=1.0) <= 1.6
    cjk = "这是一个很长的中文句子，用来测试预算的计算方式。" * 20
    assert 3.0 <= script_weight(cjk, dense=COUNTER_DENSE_TOKENS) <= 4.0
    assert 2.0 <= script_weight(cjk, dense=BUDGET_DENSE_TOKENS) <= 2.6


def test_an_empty_text_weighs_one():
    assert script_weight("", dense=1.0) == 1.0


# ===================== exact AI messages =====================


def test_an_ai_message_without_reasoning_costs_its_output_tokens_plus_overhead():
    message = _hop(500, 400, content="x" * 4000, output_tokens=900)
    overhead = estimate(AIMessage(content=""))
    assert exact_tokens(message) == 900 + overhead
    assert weighted_cost(message, 900 / 1) > 0


def test_a_thinking_model_answer_is_weighed_by_its_text_not_its_output_tokens():
    reasoning = _hop(500, 400, content="The answer is four.", output_tokens=6000)
    reasoning.additional_kwargs["reasoning_content"] = "let me think " * 500
    inline = _hop(500, 400, content="<think>long</think>The answer.", output_tokens=6000)
    assert has_reasoning(reasoning) and has_reasoning(inline)
    assert exact_tokens(reasoning) is None and exact_tokens(inline) is None
    cost = _budget_est([HumanMessage("q", id="q"), reasoning]).total
    assert cost < 100


# ===================== per-message weights =====================


def test_messages_before_the_anchor_cost_r_and_the_rest_their_own_weight():
    paste = HumanMessage("北京的秋天很美。" * 200, id="paste")
    messages = [
        HumanMessage("hello there, how are you?", id="h1"),
        _hop(90, 100, msg_id="a1", content="fine, thanks", output_tokens=4),
        paste,
    ]
    weights, ratio = message_weights(messages, dense=BUDGET_DENSE_TOKENS)
    assert ratio == pytest.approx(0.9)
    assert weights[0] == pytest.approx(0.9)
    assert weights[1] == pytest.approx(exact_tokens(messages[1]) / estimate(messages[1]))
    assert weights[2] == pytest.approx(script_weight(paste.content, dense=BUDGET_DENSE_TOKENS))


def test_a_summary_before_the_anchor_is_weighed_as_new_text():
    messages = [_summary("这是摘要" * 50), _hop(3000, 1000, msg_id="a"), HumanMessage("next")]
    weights, ratio = message_weights(messages, dense=COUNTER_DENSE_TOKENS)
    assert ratio == pytest.approx(3.0)
    assert weights[0] == pytest.approx(script_weight(messages[0].content, dense=1.0))


def test_the_counter_floors_script_weights_at_1_2():
    weights, _ = message_weights(
        [HumanMessage("plain english words", id="h")],
        dense=COUNTER_DENSE_TOKENS,
        weight_floor=COUNTER_WEIGHT_FLOOR,
    )
    assert weights == [COUNTER_WEIGHT_FLOOR]


def test_costs_are_additive():
    messages = [
        HumanMessage("hello 2026", id="h1"),
        _hop(90, 100, msg_id="a1", content="fine", output_tokens=4),
        HumanMessage("长城很长" * 30, id="h2"),
    ]
    weights, _ = message_weights(messages, dense=BUDGET_DENSE_TOKENS)
    total = sum(weighted_cost(m, w) for m, w in zip(messages, weights))
    assert total == _budget_est(messages).total


def test_a_partial_copy_costs_its_share():
    message = HumanMessage("北京" * 1000, id="big")
    weight = script_weight(message.content, dense=COUNTER_DENSE_TOKENS)
    half = message.model_copy(update={"content": message.content[:1000]})
    assert weighted_cost(half, weight) == pytest.approx(
        weighted_cost(message, weight) / 2, rel=0.02
    )


def test_the_exact_part_of_a_request():
    looping = _hop(7926, 6701, msg_id="loop", content="z" * 100_000, output_tokens=24_778)
    result = _budget_est([HumanMessage("summary", id="s"), looping, HumanMessage("next")])
    assert result.exact == 24_778 + estimate(AIMessage(content=""))
    assert result.total - result.exact < 100


# ===================== the request overhead =====================


def test_the_overhead_splits_the_fixed_part_from_the_kb_additions():
    overhead = request_overhead("You are helpful.", "excerpt " * 50, "Answer in French.", [])
    assert overhead.fixed_est == request_tokens_est([SystemMessage("You are helpful.")])
    assert overhead.added_text == "excerpt " * 50 + "\n\n\n\nAnswer in French."
    with_tools = request_overhead("You are helpful.", None, "", [search_knowledge_base])
    assert with_tools.fixed_est == request_tokens_est(
        [SystemMessage("You are helpful.")], [search_knowledge_base]
    )
    assert "search_knowledge_base" in with_tools.fixed_text


def test_the_fixed_overhead_costs_r_when_measured_and_its_own_weight_otherwise():
    overhead = RequestOverhead(fixed_est=1000, fixed_text="english system prompt")
    assert overhead_tokens(overhead, 0.9, dense=1.0) == 900
    assert overhead_tokens(overhead, None, dense=1.0) == 1000
    assert overhead_tokens(overhead, None, dense=1.0, weight_floor=1.2) == 1200


def test_the_kb_additions_are_new_text_every_turn():
    overhead = RequestOverhead(fixed_est=0, added_text="北京" * 400)
    expected = math.ceil(script_weight("北京" * 400, dense=1.0) * math.ceil(800 / 4))
    assert overhead_tokens(overhead, 0.9, dense=1.0) == expected


def test_the_request_tools_cost_r_or_their_own_weight():
    from langchain_core.utils.function_calling import convert_to_openai_tool

    tools = [convert_to_openai_tool(search_knowledge_base)]
    tools_est = count_tokens_approximately([], tools=tools)
    measured = [HumanMessage("q", id="q"), _hop(900, 1000, msg_id="a"), HumanMessage("q2")]
    without = _budget_est(measured).total
    assert _budget_est(measured, tools=tools).total == without + math.ceil(0.9 * tools_est)


# ===================== a KB block is not history =====================


def test_a_kb_turns_ratio_excludes_the_block_it_carried():
    """``_KbContextMiddleware`` adds the block to the request only: the next
    request no longer has it. CJK history (1000 estimated, real ratio 2.5)
    plus an English block (1500 estimated, real 1500) measured a plain 1.6;
    the history alone reads 2.5."""
    from src.agents.token_accounting import REQUEST_KB_EST_KEY, REQUEST_KB_REAL_KEY

    kb_turn = _hop(4000, 2500, msg_id="kb")
    assert hop_ratio(kb_turn) == pytest.approx(1.6)
    kb_turn.response_metadata[REQUEST_KB_EST_KEY] = 1500
    kb_turn.response_metadata[REQUEST_KB_REAL_KEY] = 1500
    assert hop_ratio(kb_turn) == pytest.approx(2.5)


def test_a_degenerate_kb_stamp_falls_back_to_the_plain_ratio():
    from src.agents.token_accounting import REQUEST_KB_EST_KEY, REQUEST_KB_REAL_KEY

    hop = _hop(4000, 2500)
    hop.response_metadata[REQUEST_KB_EST_KEY] = 2500  # nothing left of the request
    hop.response_metadata[REQUEST_KB_REAL_KEY] = 3000
    assert hop_ratio(hop) == pytest.approx(1.6)


# ===================== inline reasoning =====================


def test_inline_reasoning_costs_only_the_text_after_the_last_think_block():
    answer = "The answer is four, as computed."
    message = AIMessage(
        id="t",
        content="<think>" + "deliberating " * 400 + "</think>" + answer,
        usage_metadata={"input_tokens": 500, "output_tokens": 3000, "total_tokens": 3500},
    )
    tail = AIMessage(content=answer)
    cost = _budget_est([HumanMessage("q", id="q"), message]).total
    alone = _budget_est([HumanMessage("q", id="q")]).total
    assert cost - alone == math.ceil(
        script_weight(answer, dense=BUDGET_DENSE_TOKENS) * estimate(tail)
    )


def test_an_empty_exact_answer_never_divides_by_zero():
    from src.agents.token_accounting import fresh_weight

    empty = _hop(10, 10, content="", output_tokens=7)
    assert fresh_weight(empty, dense=1.0) == pytest.approx(exact_tokens(empty) / estimate(empty))


# ===================== the densities are tokenizer-dependent constants =====================


def test_the_script_densities_are_pinned():
    """Non-regression: the constants of the unmeasured tail. They are
    tokenizer-dependent (Japanese kana, small-vocabulary tokenizers and
    Devanagari or Thai can be under-estimated -- the budget is caught by the
    preflight retry, compaction may come late), so a change is a decision."""
    from src.agents.token_accounting import DIGIT_TOKENS, OTHER_LETTER_TOKENS

    assert (COUNTER_DENSE_TOKENS, COUNTER_WEIGHT_FLOOR, BUDGET_DENSE_TOKENS) == (1.0, 1.2, 0.65)
    assert (DIGIT_TOKENS, OTHER_LETTER_TOKENS) == (1.0, 0.4)
    assert script_weight("中文" * 50, dense=BUDGET_DENSE_TOKENS) == pytest.approx(2.6)
    assert script_weight("ひらがなカタカナ" * 20, dense=BUDGET_DENSE_TOKENS) == pytest.approx(2.6)
    assert script_weight("नमस्ते" * 30, dense=BUDGET_DENSE_TOKENS) == pytest.approx(1.6)
    assert script_weight("1234567890", dense=BUDGET_DENSE_TOKENS) == pytest.approx(4.0)
