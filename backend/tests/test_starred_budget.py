"""Starred messages are bounded in the prompt (#659).

Later turns carry the beginning of each starred message only: about
``STARRED_MESSAGE_TOKENS`` per message and ``STARRED_TOTAL_TOKENS`` in all,
counted with the compaction gate's own counter (``counter_weight``), the
oldest stars first. A short star is carried word for word.
"""

from __future__ import annotations

import math

import pytest
from langchain_core.messages import HumanMessage

from src.agents.kb_mode import plan_turn
from src.agents.prompts import (
    STARRED_MESSAGE_TOKENS,
    STARRED_TOTAL_TOKENS,
    STARRED_TRUNCATION_MARK,
    bound_starred_messages,
)
from src.agents.token_accounting import counter_weight

pytestmark = pytest.mark.unit


def _tokens(text: str) -> int:
    """The gate's count of a text's content (its weight times chars/4)."""
    return math.ceil(counter_weight(HumanMessage(text)) * len(text) / 4)


def _body(text: str) -> str:
    return text.removesuffix(STARRED_TRUNCATION_MARK)


def test_the_budgets():
    assert STARRED_MESSAGE_TOKENS == 500
    assert STARRED_TOTAL_TOKENS == 2000


def test_a_short_star_is_carried_word_for_word():
    stars = ["Ada lives in Lyon.", "The meeting is on Tuesday at 10."]
    assert bound_starred_messages(stars) == stars


def test_a_long_star_keeps_its_beginning_within_its_budget():
    long = "START " + "word " * 20_000 + " END"

    (carried,) = bound_starred_messages([long])

    assert carried.endswith(STARRED_TRUNCATION_MARK)
    assert long.startswith(_body(carried))
    assert "START" in carried and "END" not in carried
    assert STARRED_MESSAGE_TOKENS - 10 <= _tokens(_body(carried)) <= STARRED_MESSAGE_TOKENS


def test_a_chinese_star_is_cut_in_tokens_not_characters():
    long = "开始" + "中文内容" * 5_000

    (carried,) = bound_starred_messages([long])

    assert _tokens(_body(carried)) <= STARRED_MESSAGE_TOKENS
    # About one token per character: far fewer characters than a Latin cut.
    assert len(_body(carried)) <= STARRED_MESSAGE_TOKENS + 1


def test_the_total_is_bounded_and_the_oldest_stars_come_first():
    stars = [f"STAR{i} " + "lorem ipsum " * 2_000 for i in range(6)]

    carried = bound_starred_messages(stars)

    assert [c.split()[0] for c in carried] == ["STAR0", "STAR1", "STAR2", "STAR3"]
    assert sum(_tokens(_body(c)) for c in carried) <= STARRED_TOTAL_TOKENS


def test_short_stars_after_the_budget_is_spent_are_cut_too():
    stars = ["x " * 10_000] * 4 + ["Ada lives in Lyon."]

    carried = bound_starred_messages(stars)

    assert len(carried) == 4
    assert "Ada lives in Lyon." not in carried


def test_a_star_partly_over_the_remaining_total_takes_what_is_left():
    stars = ["a " * 10_000] * 3 + ["b " * 300, "c " * 10_000]

    carried = bound_starred_messages(stars)

    assert carried[3] == stars[3]
    assert carried[4].startswith("c")
    left = STARRED_TOTAL_TOKENS - sum(_tokens(_body(c)) for c in carried[:4])
    assert left - 10 <= _tokens(_body(carried[4])) <= left < STARRED_MESSAGE_TOKENS


def test_a_turn_plan_carries_the_bounded_stars():
    class _Llm:
        id = 1
        name = "Test 7B"
        param_size = 7.0
        supports_tools = False

    long = "BEGIN " + "filler " * 20_000 + " FINISH"

    plan = plan_turn(_Llm(), question="hi", retrieve=list, starred_messages=["short star", long])

    assert "short star" in plan.system_prompt
    assert "BEGIN" in plan.system_prompt
    assert "FINISH" not in plan.system_prompt
    assert len(plan.system_prompt) < len(long) // 10


def test_a_mixed_script_star_keeps_its_bound():
    """2,200 Chinese characters then 48,000 Latin ones: the whole message's
    average weight would let the CJK head run far past 500 tokens."""
    star = "中文内容" * 550 + "latin text " * 4_364

    (carried,) = bound_starred_messages([star])

    assert _tokens(_body(carried)) <= STARRED_MESSAGE_TOKENS
    assert carried.endswith(STARRED_TRUNCATION_MARK)


def test_a_star_whose_cut_would_be_empty_is_skipped():
    stars = ["x " * 10_000] * 3 + ["y " * 990, "中" * 5_000]

    carried = bound_starred_messages(stars)

    assert all(_body(c).strip() for c in carried)
    assert sum(_tokens(_body(c)) for c in carried) <= STARRED_TOTAL_TOKENS


def test_a_latin_star_ahead_of_a_dense_tail_keeps_nearly_its_whole_budget():
    """The cut is the longest head within the budget, whatever follows it:
    a Latin beginning keeps its 500 tokens ahead of a Chinese tail."""
    star = "The plan for the week is simple. " * 40 + "这是一个很长的中文句子" * 2_000

    (carried,) = bound_starred_messages([star])

    body = _body(carried)
    assert star.startswith(body)
    assert 0.9 * STARRED_MESSAGE_TOKENS <= _tokens(body) <= STARRED_MESSAGE_TOKENS
