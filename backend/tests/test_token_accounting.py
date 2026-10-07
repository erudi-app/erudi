"""Real-token accounting: one estimator, measured ratios, a script-aware fallback.

The local servers report the real prompt size of every call (``usage``); the
app keeps estimating with chars/4 and scales the estimate by the ratio the
server measured on a comparable request. These tests pin the estimator, the
two ratio readers (the last stamped FIRST hop, and a later hop of the current
turn), the summary-carried ratio, and the fallback used before anything is
measured.
"""

from __future__ import annotations

import math

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import tool

from src.agents.token_accounting import (
    COUNTER_FALLBACK_RATIO_FLOOR,
    FIRST_HOP_RATIO_KWARG,
    RATIO_CEILING,
    RATIO_FLOOR,
    REQUEST_EST_KEY,
    REQUEST_FIRST_HOP_KEY,
    REQUEST_HAS_IMAGES_KEY,
    counter_ratio,
    current_turn_ratio,
    first_hop_ratio,
    hop_ratio,
    messages_have_images,
    request_overhead_est,
    request_tokens_est,
    script_ratio,
)

pytestmark = pytest.mark.unit


@tool
def search_knowledge_base(query: str) -> str:
    """Search the documents of the knowledge base."""
    return ""


def _hop(input_tokens, est, *, first=True, images=False, content="a", **kwargs):
    """An AI message as ``_astream`` leaves it: server usage + the stamp."""
    return AIMessage(
        content=content,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": 10,
            "total_tokens": input_tokens + 10,
        },
        response_metadata={
            REQUEST_EST_KEY: est,
            REQUEST_HAS_IMAGES_KEY: images,
            REQUEST_FIRST_HOP_KEY: first,
        },
        **kwargs,
    )


def _summary(ratio=None):
    kwargs = {"lc_source": "summarization"}
    if ratio is not None:
        kwargs[FIRST_HOP_RATIO_KWARG] = ratio
    return HumanMessage(
        content="Here is a summary of the conversation to date:\n\nfacts",
        additional_kwargs=kwargs,
    )


# ===================== the one estimator =====================


def test_the_request_estimator_is_count_tokens_approximately_with_its_defaults():
    messages = [SystemMessage("system"), HumanMessage("hello there")]
    assert request_tokens_est(messages) == count_tokens_approximately(messages)


def test_the_request_estimator_counts_the_tool_schemas():
    messages = [HumanMessage("hello")]
    assert request_tokens_est(messages, [search_knowledge_base]) > request_tokens_est(messages)


def test_tools_are_converted_the_same_way_on_both_sides():
    """The stamp sees the request's dict schemas, the overhead sees BaseTool
    objects: both are converted to the OpenAI shape, so they count alike."""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    messages = [HumanMessage("hello")]
    as_dicts = [convert_to_openai_tool(search_knowledge_base)]
    assert request_tokens_est(messages, as_dicts) == request_tokens_est(
        messages, [search_knowledge_base]
    )


def test_the_overhead_counts_the_system_prompt_the_tools_and_the_kb_additions_unscaled():
    system = "You are helpful. " * 20
    bare = request_overhead_est(system, None, "", [])
    assert bare == request_tokens_est([SystemMessage(system)])
    with_tools = request_overhead_est(system, None, "", [search_knowledge_base])
    assert with_tools == request_tokens_est([SystemMessage(system)], [search_knowledge_base])
    block = "excerpt " * 100
    line = "Answer in French."
    with_kb = request_overhead_est(system, block, line, [])
    # The block, the two blank-line joins and the language line, exactly as
    # ``_KbContextMiddleware._merge`` adds them; the question is NOT counted.
    assert with_kb == bare + math.ceil((len(block) + 4 + len(line)) / 4)


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


# ===================== the measured ratio of one hop =====================


def test_a_hop_ratio_is_input_tokens_over_the_stamped_estimate():
    assert hop_ratio(_hop(1100, 1000)) == pytest.approx(1.1)


@pytest.mark.parametrize(
    "input_tokens, est, expected",
    [(100, 1000, RATIO_FLOOR), (10_000, 1000, RATIO_CEILING)],
)
def test_a_hop_ratio_is_clamped(input_tokens, est, expected):
    assert hop_ratio(_hop(input_tokens, est)) == expected


def test_an_image_hop_has_no_ratio():
    # The estimator counts 85 tokens per image, the server the real ones.
    assert hop_ratio(_hop(5000, 1000, images=True)) is None


def test_a_hop_without_usage_or_stamp_has_no_ratio():
    assert hop_ratio(AIMessage("a")) is None
    assert hop_ratio(AIMessage("a", response_metadata={REQUEST_EST_KEY: 1000})) is None
    assert (
        hop_ratio(
            AIMessage(
                "a", usage_metadata={"input_tokens": 10, "output_tokens": 1, "total_tokens": 11}
            )
        )
        is None
    )
    assert hop_ratio(_hop(1000, 0)) is None
    assert hop_ratio(HumanMessage("q")) is None


# ===================== first_hop_ratio =====================


def _kb_turn(first_r=1.1, last_r=2.14):
    """A finished KB/web turn: first hop (tool call), the tool result, then
    the last hop, whose request carried the large results."""
    return [
        HumanMessage("question", id="h1"),
        _hop(
            int(first_r * 1000),
            1000,
            first=True,
            content="",
            tool_calls=[{"name": "search_knowledge_base", "args": {}, "id": "c1"}],
            id="a1",
        ),
        ToolMessage("results " * 500, tool_call_id="c1", name="search_knowledge_base", id="t1"),
        _hop(int(last_r * 1000), 1000, first=False, content="the answer", id="a2"),
    ]


def test_first_hop_ratio_reads_the_last_stamped_first_hop_not_the_last_hop():
    assert first_hop_ratio(_kb_turn()) == pytest.approx(1.1)


def test_the_stamp_wins_over_the_position():
    """After a compaction the AI message right after the summary is often a
    turn's LAST hop: its position says nothing, the stamp does."""
    messages = [
        _summary(),
        _hop(2140, 1000, first=False, content="last hop kept by the compaction"),
        HumanMessage("next"),
    ]
    assert first_hop_ratio(messages) is None


def test_first_hop_ratio_falls_back_to_the_summary_carried_value():
    messages = [_summary(1.3), _hop(2140, 1000, first=False), HumanMessage("next")]
    assert first_hop_ratio(messages) == pytest.approx(1.3)


def test_a_stamped_first_hop_after_the_summary_wins_over_the_carried_value():
    messages = [_summary(1.3), _hop(2500, 1000, first=True), HumanMessage("next")]
    assert first_hop_ratio(messages) == pytest.approx(2.5)


def test_first_hop_ratio_skips_image_hops():
    messages = [
        HumanMessage("q1"),
        _hop(1200, 1000, first=True),
        HumanMessage("q2 with an image"),
        _hop(9000, 1000, first=True, images=True),
    ]
    assert first_hop_ratio(messages) == pytest.approx(1.2)


def test_first_hop_ratio_is_none_when_nothing_is_measured():
    assert first_hop_ratio([HumanMessage("q"), AIMessage("a"), HumanMessage("q2")]) is None
    assert first_hop_ratio([]) is None


# ===================== current_turn_ratio =====================


def test_current_turn_ratio_reads_a_later_hop_of_the_current_turn():
    messages = _kb_turn()[:3]  # mid-turn: question, first hop, tool result
    assert current_turn_ratio(messages) == pytest.approx(1.1)
    assert current_turn_ratio(_kb_turn()) == pytest.approx(2.14)


def test_current_turn_ratio_is_none_at_a_first_hop():
    messages = _kb_turn() + [HumanMessage("next question")]
    assert current_turn_ratio(messages) is None


# ===================== the script-aware fallback =====================


def test_english_and_code_read_about_one():
    assert script_ratio([HumanMessage("The quick brown fox jumps over the lazy dog. " * 20)]) == 1.0
    assert script_ratio([HumanMessage("def f(x):\n    return x * 2  # double\n" * 20)]) == 1.0


def test_french_with_a_few_accents_reads_about_1_02():
    text = ("le cafe est servi dans la petite salle du chateau pres de la riviere " * 3)[:96]
    text = text[:92] + "éèàç"  # 4 % accented letters
    assert script_ratio([HumanMessage(text)]) == pytest.approx(1.024, abs=0.01)


def test_russian_reads_about_1_5():
    text = "Привет, как дела? Сегодня хорошая погода и мы идём гулять в парк. " * 10
    assert 1.35 <= script_ratio([HumanMessage(text)]) <= 1.6


def test_hindi_counts_the_combining_marks_too():
    text = "नमस्ते, आप कैसे हैं? आज मौसम बहुत अच्छा है। " * 10
    assert script_ratio([HumanMessage(text)]) >= 1.4


def test_cjk_reads_three_to_four():
    text = "这是一个很长的中文句子，用来测试预算的计算方式。" * 20
    assert 3.0 <= script_ratio([HumanMessage(text)]) <= 4.0
    assert 3.0 <= script_ratio([HumanMessage("これは日本語の文章です。" * 20)]) <= 4.0
    assert 3.0 <= script_ratio([HumanMessage("이것은 한국어 문장입니다" * 20)]) <= 4.0


def test_script_ratio_of_nothing_is_one():
    assert script_ratio([]) == 1.0
    assert script_ratio([AIMessage("")]) == 1.0


def test_script_ratio_reads_text_parts_of_multimodal_content():
    message = HumanMessage(
        content=[
            {"type": "text", "text": "中文句子" * 20},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 4000}},
        ]
    )
    assert script_ratio([message]) == 4.0


# ===================== the counter's ratio =====================


def test_the_counter_prefers_the_current_turn_then_the_first_hop():
    assert counter_ratio(_kb_turn()[:3]) == pytest.approx(1.1)
    assert counter_ratio(_kb_turn() + [HumanMessage("next")]) == pytest.approx(1.1)


def test_the_counter_falls_back_to_max_1_5_and_the_script_ratio():
    english = [HumanMessage("hello there, how are you today?")]
    assert counter_ratio(english) == COUNTER_FALLBACK_RATIO_FLOOR == 1.5
    cjk = [HumanMessage("这是一个很长的中文句子" * 20)]
    assert counter_ratio(cjk) == script_ratio(cjk) > 1.5
