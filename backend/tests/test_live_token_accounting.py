"""The live run's requests, costed by the per-message rule (plan 3.2b v17).

Apple Silicon, Qwen2.5-0.5B-Instruct-4bit, MAX_KV_SIZE 32768. The texts are the
ones the scenarios sent (``fixtures/live_token_accounting.json``, regenerated
from their seeds); the stamps and usage are the ones read from the
conversations' checkpoints; the real prompt sizes are the server's
(``Request needs ... prompt``). With one ratio measured on the short turns
before them, the long pastes were under-estimated and the server rejected the
budget ("Output budget overshot"); costed per message, every budget below
passes the server's check with its margin.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.output_budget import compute_output_budget
from src.agents.token_accounting import (
    BUDGET_DENSE_TOKENS,
    REQUEST_EST_KEY,
    REQUEST_FIRST_HOP_KEY,
    REQUEST_HAS_IMAGES_KEY,
    real_tokens_est,
    request_tokens_est,
)

pytestmark = pytest.mark.unit

WINDOW = 32_768
LIVE = json.loads(
    (Path(__file__).parent / "fixtures" / "live_token_accounting.json").read_text(encoding="utf-8")
)

E1 = "What is the capital of France? Answer in one sentence."
E2 = "Name three large rivers in Europe and one country each crosses."
P1 = "Hello! In one sentence, what is a desktop application?"
P2 = "And in one sentence, what is a local language model?"
C1 = "请用两三句话介绍一下长城的历史。"


def _system(first_request_est: int, question: str) -> SystemMessage:
    """A system prompt the size of the live one: the first request (system
    prompt + first question) was stamped at ``first_request_est``."""
    size = 0
    while request_tokens_est([SystemMessage("x" * size), HumanMessage(question)]) < (
        first_request_est
    ):
        size += 1
    return SystemMessage("x" * size)


def _hop(chars, input_tokens, est, output_tokens, filler="a", msg_id=None):
    """An answer as its checkpoint holds it (content length, usage, stamp)."""
    return AIMessage(
        content=(filler * chars)[:chars],
        id=msg_id,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        response_metadata={
            REQUEST_EST_KEY: est,
            REQUEST_HAS_IMAGES_KEY: False,
            REQUEST_FIRST_HOP_KEY: True,
        },
    )


def _passes(messages, real_prompt):
    budget = compute_output_budget(messages, WINDOW)
    return budget, real_prompt + budget <= WINDOW


def _one_ratio_budget(messages, ratio):
    """What the v15 rule handed out: one measured ratio over chars/4."""
    import math

    prompt = math.ceil(ratio * request_tokens_est(messages))
    return max(512, WINDOW - prompt - max(256, int(0.10 * prompt)))


def test_english_7926_after_two_short_turns():
    system = _system(128, E1)
    request = [
        system,
        HumanMessage(E1),
        _hop(31, 115, 128, 8),
        HumanMessage(E2),
        _hop(207, 144, 161, 48),
        HumanMessage(LIVE["eng_paste"]),
    ]
    budget, ok = _passes(request, real_prompt=7926)
    assert ok, budget
    # The live budget was 26175 for the same request: rejected by the server.
    assert 7926 + _one_ratio_budget(request, 144 / 161) > WINDOW


def test_english_prose_7101_after_two_short_turns():
    system = _system(128, P1)
    request = [
        system,
        HumanMessage(P1),
        _hop(271, 115, 128, 47),
        HumanMessage(P2),
        _hop(384, 183, 218, 70),
        HumanMessage(LIVE["prose_paste"]),
    ]
    budget, ok = _passes(request, real_prompt=7101)
    assert ok, budget
    # 32768 - ceil(0.839 x 7187) - 603 = 26131, the logged budget: rejected.
    assert _one_ratio_budget(request, 183 / 218) == 26131
    assert 7101 + 26131 > WINDOW


def test_the_cjk_c2_paste_1921_after_one_short_turn():
    system = _system(118, C1)
    request = [
        system,
        HumanMessage(C1),
        _hop(125, 111, 118, 87, filler="长城是中国古代的防御工程"),
        HumanMessage(LIVE["cjk_c2_paste"]),
    ]
    budget, ok = _passes(request, real_prompt=1921)
    assert ok, budget
    assert 1921 + _one_ratio_budget(request, 111 / 118) > WINDOW


def test_the_cjk_24772_token_paste_after_four_turns():
    system = _system(118, C1)
    big = LIVE["cjk_big_paste"] + "\n\n请用三句话总结上面的文字。"
    request = [
        system,
        HumanMessage(C1),
        _hop(125, 111, 118, 87, filler="长城是"),
        HumanMessage(LIVE["cjk_c2_paste"]),
        _hop(137, 1921, 791, 100, filler="长城是"),
        HumanMessage("谢谢。请再用一句话说说北京的秋天。"),
        _hop(179, 2042, 840, 106, filler="北京的"),
        HumanMessage("请列出中国的四大发明，并各用一句话解释。"),
        _hop(454, 2168, 899, 282, filler="四大发明"),
        HumanMessage(big),
    ]
    # The paste alone is 24772 tokens for this tokenizer; the request before
    # it measured 2168, plus that answer's 282.
    real_prompt = 2168 + 282 + LIVE["cjk_big_paste_tokens"] + 15
    budget, ok = _passes(request, real_prompt=real_prompt)
    assert ok, budget
    assert budget > 512


def test_after_a_24_8k_token_looping_answer_the_budget_is_not_the_floor():
    """runB e3: the answer looped to its 24778-token cap; compaction kept it
    with the next question. Counted exactly it leaves W - real - margin; at
    the previous ratio over chars/4 (1.183 x 110547 / 4) it hit the floor."""
    system = _system(128, E1)
    looping = _hop(110_547, 7926, 6701, 24_778, filler="The river was measured. ")
    request = [
        system,
        HumanMessage(
            "Here is a summary of the conversation to date:\n\n" + "w" * 4998,
            additional_kwargs={"lc_source": "summarization"},
        ),
        looping,
        HumanMessage("Thanks. Now give me one sentence about Paris in winter."),
    ]
    prompt = real_tokens_est(request, dense=BUDGET_DENSE_TOKENS)
    budget = compute_output_budget(request, WINDOW)
    # The margin applies to the estimated part only.
    assert budget == WINDOW - prompt.total - max(256, int(0.10 * (prompt.total - prompt.exact)))
    assert budget > 5000
    assert _one_ratio_budget(request, 7926 / 6701) == 512


def test_a_turn_right_after_a_compaction_that_kept_only_a_last_hop_uses_rule_3():
    # No stamped first hop survives: everything is costed as new text.
    last_hop = AIMessage(
        content="the answer",
        usage_metadata={"input_tokens": 3000, "output_tokens": 4, "total_tokens": 3004},
        response_metadata={
            REQUEST_EST_KEY: 1000,
            REQUEST_HAS_IMAGES_KEY: False,
            REQUEST_FIRST_HOP_KEY: False,
        },
    )
    request = [
        HumanMessage(
            "Here is a summary of the conversation to date:\n\nfacts",
            additional_kwargs={"lc_source": "summarization"},
        ),
        last_hop,
        HumanMessage("next question"),
    ]
    from src.agents.token_accounting import estimate, script_weight

    expected = (
        sum(
            -(-script_weight(m.content, dense=BUDGET_DENSE_TOKENS) * estimate(m) // 1)
            for m in (request[0], request[2])
        )
        + 4
        + estimate(AIMessage(content=""))
    )
    assert real_tokens_est(request, dense=BUDGET_DENSE_TOKENS).total == expected
