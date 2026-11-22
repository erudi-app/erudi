"""No hopeless compaction, and a rejected summary keeps the history (#690).

* A new message that cannot fit the window on its own -- with the request's
  overhead, whatever a compaction would do -- refuses the turn at once, before
  any summary call: the curated ``message_too_long`` turn (or its memory
  variant) names the message, not the conversation, with its figures.
* A summary that cannot stand for what it replaces -- far too short for what
  the summarizer read, or a copy of the last message -- is rejected like a
  failed one. Automatic: the history is untouched, the turn goes ahead when
  the request still fits and is refused with ``summary_failed`` when it does
  not, and automatic compaction pauses for the next
  ``COMPACTION_PAUSE_MESSAGES`` user messages. Manual: the compaction fails
  and changes nothing.
"""

from __future__ import annotations

import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.agents import runner as runner_module
from src.agents.runner import (
    COMPACTION_PAUSE_KEY,
    COMPACTION_PAUSE_MESSAGES,
    ERROR_MESSAGE,
    ERROR_SENTINEL,
    AgentRunner,
    real_token_count,
    summary_rejection,
)
from src.agents.token_accounting import message_weights, request_overhead
from src.core import config
from src.core.exceptions import CompactionFailedException
from tests.test_compaction_keep import (
    _ScriptedSummaryModel,
    _WindowEngine,
    _alternating,
    _middleware,
    _models,
    _seed,
    _state,
    _text,
    _turn,
)

pytestmark = pytest.mark.unit

# A summary long enough for any pool these tests build.
_GOOD_SUMMARY = (
    "The user is called Ada and lives in Lyon. She asked about trains to Paris, "
    "the assistant listed the morning departures and the fares, and she chose the 8:04 train."
)


@pytest.fixture(autouse=True)
def _engine(monkeypatch):
    _WindowEngine.window = None
    _WindowEngine.claims = []
    _WindowEngine.rewrites = []
    monkeypatch.setattr(config, "LLM_Engine", _WindowEngine)
    yield
    _WindowEngine._last_used = None


def _curated(events) -> list:
    return [e["curated_error"] for e in events if e["t"] == "answer" and "curated_error" in e]


def _pause_of(cp, thread_id):
    tup = cp.get_tuple({"configurable": {"thread_id": thread_id}})
    return tup.checkpoint["channel_values"].get(COMPACTION_PAUSE_KEY)


def _budget_request_tokens(state, question="next question"):
    """The same request counted the way the refusals count it: the output
    budget's estimator (no 1.2 floor, CJK 0.65 token per character)."""
    from src.agents.token_accounting import BUDGET_DENSE_TOKENS, GROUPED_DIGIT_TOKENS

    messages = list(state) + [HumanMessage(question, id="probe")]
    weights, _ratio = message_weights(
        messages, dense=BUDGET_DENSE_TOKENS, digit_tokens=GROUPED_DIGIT_TOKENS
    )
    by_id = {m.id: w for m, w in zip(messages, weights)}
    return real_token_count(messages, by_id) + request_overhead("s").fixed_est


def _request_tokens(state, question="next question"):
    """The request a turn on ``state`` sends, as the gate counts it: the
    state plus the question, every message weighed by the counter, plus the
    system prompt's overhead (``"s"``, a few tokens)."""
    messages = list(state) + [HumanMessage(question, id="probe")]
    weights, _ratio = runner_module.frozen_weights(messages)
    return real_token_count(messages, weights) + request_overhead("s").fixed_est


# ===================== 1. a message too long on its own =====================


async def test_a_message_too_long_on_its_own_is_refused_without_compacting(monkeypatch, caplog):
    _WindowEngine.window = 10_000
    summary = _ScriptedSummaryModel(messages=iter([]), script=[_GOOD_SUMMARY])
    _models(monkeypatch, ["never sent"], summary)
    cp = InMemorySaver()
    # One short exchange, then a turn whose message alone exceeds the window.
    seed = [HumanMessage("hi", id="h0"), AIMessage("hello", id="a0")]
    await _seed(cp, "long1", seed)

    with caplog.at_level(logging.INFO):
        events = await _turn(AgentRunner(checkpointer=cp), "long1", _text(12_000, "x"))

    (curated,) = _curated(events)
    assert curated["kind"] == "message_too_long"
    assert curated["params"]["context_tokens"] == 10_000
    assert curated["params"]["prompt_tokens"] > 12_000
    assert any(e["t"] == "answer" and e["text"].startswith(ERROR_SENTINEL) for e in events)
    # No summary call, no rewrite, no compaction record.
    assert summary.prompts == []
    assert _WindowEngine.rewrites == []
    assert not any("Conversation compacted" in r.getMessage() for r in caplog.records)
    assert not any(e["t"] == "compaction" for e in events)
    # The history is unchanged; the refused question is closed like any failed turn.
    state = await _state(cp, "long1")
    assert [m.id for m in state[:2]] == ["h0", "a0"]
    assert state[-1].content == ERROR_MESSAGE
    refusals = [r for r in caplog.records if "refused" in r.getMessage()]
    assert len(refusals) == 1 and refusals[0].levelno == logging.WARNING
    assert refusals[0].getMessage().isascii()


_PROSE = "The quick brown fox jumps over the lazy dog near the river bank. "


@pytest.mark.parametrize(
    "message",
    [_PROSE * 500, "这是一个很长的中文句子" * 960],
    ids=["latin-32500-chars", "cjk-9600-chars"],
)
async def test_a_borderline_message_the_engine_would_serve_is_not_refused(monkeypatch, message):
    """The conservative compaction counter (1.2 floor, one token per CJK
    character) would refuse these; the output budget's estimator, which the
    refusals use, lets them through."""
    _WindowEngine.window = 10_000
    summary = _ScriptedSummaryModel(messages=iter([]))
    _models(monkeypatch, ["served"], summary)
    cp = InMemorySaver()

    events = await _turn(AgentRunner(checkpointer=cp), "border1", message)

    assert _curated(events) == []
    assert "".join(e["text"] for e in events if e["t"] == "answer") == "served"


async def test_a_message_clearly_over_the_window_is_still_refused(monkeypatch):
    _WindowEngine.window = 10_000
    _models(monkeypatch, ["never sent"], _ScriptedSummaryModel(messages=iter([])))

    events = await _turn(AgentRunner(checkpointer=InMemorySaver()), "over1", _PROSE * 700)

    (curated,) = _curated(events)
    assert curated["kind"] == "message_too_long"


async def test_a_message_too_long_for_a_memory_bound_window_names_memory(monkeypatch):
    _WindowEngine.window = 10_000
    monkeypatch.setattr(_WindowEngine, "memory_bound_tokens", classmethod(lambda cls: 10_000))
    summary = _ScriptedSummaryModel(messages=iter([]))
    _models(monkeypatch, ["never sent"], summary)
    cp = InMemorySaver()

    events = await _turn(AgentRunner(checkpointer=cp), "long2", _text(12_000, "x"))

    (curated,) = _curated(events)
    assert curated["kind"] == "message_too_long_memory"
    assert curated["params"]["context_tokens"] == 10_000
    assert summary.prompts == []


async def test_the_message_alone_refusal_counts_digits_as_the_loaded_tokenizer_does(monkeypatch):
    """A tokenizer that splits numbers digit by digit (Qwen on MLX) spends one
    token per digit: the refusal counts them that way, as the output budget
    does, not at the grouped 0.34."""
    _WindowEngine.window = 10_000
    monkeypatch.setattr(_WindowEngine, "memory_bound_tokens", classmethod(lambda cls: 10_000))
    monkeypatch.setattr(runner_module, "budget_digit_tokens", lambda engine: 1.0, raising=False)
    summary = _ScriptedSummaryModel(messages=iter([]))
    _models(monkeypatch, ["never sent"], summary)
    # 12 000 digits and 3 000 spaces: about 4 830 tokens at the grouped 0.34
    # (fits 10 000), about 12 750 one token per digit (does not).
    message = "4821 " * 3_000

    events = await _turn(AgentRunner(checkpointer=InMemorySaver()), "digits1", message)

    (curated,) = _curated(events)
    assert curated["kind"] == "message_too_long_memory"
    assert curated["params"]["context_tokens"] == 10_000
    assert curated["params"]["prompt_tokens"] > 12_000
    assert summary.prompts == []


class _OverflowError(Exception):
    """llama-server's context overflow as the openai SDK hands it over."""

    def __init__(self, prompt_tokens: int, context_tokens: int):
        super().__init__("Error code: 400")
        self.body = {
            "type": "exceed_context_size_error",
            "n_prompt_tokens": prompt_tokens,
            "n_ctx": context_tokens,
        }


def _overflowing_models(monkeypatch, prompt_tokens: int, context_tokens: int):
    """The REAL chat client, its HTTP layer replaced by the engine's overflow
    rejection: what the client adds to the error on its way to the runner is
    what the runner classifies."""
    from langchain_openai import ChatOpenAI

    from src.agents.chat_model import erudi_chat_openai_class
    from src.agents.reasoning_effort import NO_REASONING_PLAN

    async def _server(self, messages, *args, **kw):
        raise _OverflowError(prompt_tokens, context_tokens)
        yield  # pragma: no cover  (makes this an async generator function)

    monkeypatch.setattr(ChatOpenAI, "_astream", _server)
    main = erudi_chat_openai_class()(
        base_url="http://127.0.0.1:1/v1", api_key="k", model="m", streaming=True
    )
    summary = _ScriptedSummaryModel(messages=iter([]))

    def _build(llm, **kw):
        return summary if kw.get("effort_plan") is NO_REASONING_PLAN else main

    monkeypatch.setattr(runner_module, "build_chat_model", _build)
    return summary


async def test_an_engine_overflow_caused_by_the_new_message_is_the_message_alone_kind(
    monkeypatch,
):
    """The QA case of #759: the estimate let the request through, the engine
    counted 14 453 tokens for a memory bound of 11 886, and the new message is
    about 95 % of the request -- without the history it still does not fit."""
    _WindowEngine.window = 11_886
    monkeypatch.setattr(_WindowEngine, "memory_bound_tokens", classmethod(lambda cls: 11_886))
    summary = _overflowing_models(monkeypatch, 14_453, 11_886)
    cp = InMemorySaver()
    await _seed(
        cp, "eng1", [HumanMessage(_text(100, "h"), id="h0"), AIMessage(_text(100, "a"), id="a0")]
    )

    events = await _turn(AgentRunner(checkpointer=cp), "eng1", _text(3_800, "x"))

    (curated,) = _curated(events)
    assert curated["kind"] == "message_too_long_memory"
    assert curated["params"]["context_tokens"] == 11_886
    # About 95 % of the engine's 14 453, plus the answer's floor.
    assert 13_000 < curated["params"]["prompt_tokens"] < 14_453 + 512
    assert summary.prompts == []


async def test_an_engine_overflow_of_a_window_bound_model_is_the_window_kind(monkeypatch):
    _WindowEngine.window = 11_886
    _overflowing_models(monkeypatch, 14_453, 11_886)
    cp = InMemorySaver()
    await _seed(
        cp, "eng2", [HumanMessage(_text(100, "h"), id="h0"), AIMessage(_text(100, "a"), id="a0")]
    )

    events = await _turn(AgentRunner(checkpointer=cp), "eng2", _text(3_800, "x"))

    (curated,) = _curated(events)
    assert curated["kind"] == "message_too_long"


async def test_an_engine_overflow_caused_by_the_history_stays_the_conversation_kind(monkeypatch):
    """A long history and a short new message: without the history the
    request would fit, so the conversation is what no longer fits."""
    _WindowEngine.window = 11_886
    monkeypatch.setattr(_WindowEngine, "memory_bound_tokens", classmethod(lambda cls: 11_886))
    _overflowing_models(monkeypatch, 14_453, 11_886)
    cp = InMemorySaver()
    await _seed(cp, "eng3", _alternating(10, 380))

    events = await _turn(AgentRunner(checkpointer=cp), "eng3", _text(200, "x"))

    (curated,) = _curated(events)
    assert curated == {
        "kind": "memory_overflow",
        "params": {"prompt_tokens": 14_453, "context_tokens": 11_886},
        "text": curated["text"],
    }


async def test_a_message_that_fits_on_its_own_still_compacts_the_history(monkeypatch):
    """A long message that fits with the summary: the compaction runs, the
    turn is answered."""
    _WindowEngine.window = 10_000
    summary = _ScriptedSummaryModel(messages=iter([]), script=[_GOOD_SUMMARY])
    _models(monkeypatch, ["the answer"], summary)
    cp = InMemorySaver()
    await _seed(cp, "long3", _alternating(20, 300))

    events = await _turn(AgentRunner(checkpointer=cp), "long3", _text(3_000, "x"))

    assert _curated(events) == []
    assert len(summary.prompts) == 1
    assert _WindowEngine.rewrites == ["rewrite"]


async def test_without_a_known_window_nothing_is_refused_up_front():
    summary = _ScriptedSummaryModel(messages=iter([]))
    mw = _middleware(summary, window=None)
    state = [HumanMessage(_text(50_000, "x"), id="q")]

    assert await mw.abefore_model({"messages": state}, None) is None


# ===================== 2. the rejection rules =====================


def _pool(n: int) -> list:
    """``n`` alternating messages of ordinary chat."""
    out = []
    for i in range(n):
        if i % 2 == 0:
            out.append(HumanMessage(f"Question {i}: what about the train to Paris on day {i}?"))
        else:
            out.append(AIMessage(f"Answer {i}: the 8:04 train reaches Paris at 10:01 on day {i}."))
    return out


def test_an_eight_character_summary_of_nineteen_messages_is_too_short():
    pool = _pool(18) + [AIMessage("Bonjour ! Comment puis-je vous aider ?")]
    assert summary_rejection("Bonjour.", pool) == "too_short"


def test_a_degenerate_non_summary_is_too_short_for_what_it_read():
    # Measured on stored conversations: 41 characters for a 1,080-character pool.
    pool = _pool(20)  # about 1,150 characters
    assert summary_rejection("None of the above messages were provided.", pool) == "too_short"


def test_a_copy_of_the_last_message_is_rejected():
    pool = _pool(6) + [AIMessage("Bonjour ! Comment puis-je vous aider aujourd'hui ?")]
    assert summary_rejection("bonjour !  comment puis-je vous aider aujourd'hui", pool) == "copy"
    # Near-equality: one character off.
    assert summary_rejection("Bonjour ! Comment puis-je vous aider aujourd'hui.", pool) == "copy"


def test_a_copy_of_the_last_question_is_rejected():
    pool = _pool(5)
    last = pool[-1].content
    assert summary_rejection(last, pool) == "copy"


def test_a_full_recap_equal_to_the_last_answer_is_not_a_copy():
    """The user asked for a recap of everything; the model's last answer is
    that recap, and the summary repeats it: long for what it read, so kept."""
    recap = (
        "Recap: Ada lives in Lyon, works as an engineer, takes the 8:04 train to Paris on "
        "Mondays, prefers aisle seats, and wants the fares compared before Friday."
    )
    pool = [
        HumanMessage("My name is Ada, I live in Lyon."),
        AIMessage("Noted."),
        HumanMessage("Recap everything I told you."),
        AIMessage(recap),
    ]
    assert summary_rejection(recap, pool) is None


def test_a_legitimately_short_summary_of_a_two_message_pool_is_accepted():
    pool = [HumanMessage("Bonjour"), AIMessage("Bonjour ! Comment puis-je vous aider ?")]
    assert summary_rejection("The user greeted the assistant, who offered to help.", pool) is None


def test_a_short_summary_of_a_single_message_is_accepted():
    pool = [HumanMessage("My name is Ada.")]
    assert summary_rejection("Ada lives in Lyon.", pool) is None
    # A single message has nothing a copy of it drops.
    assert summary_rejection("My name is Ada.", pool) is None


def test_a_short_chinese_summary_is_counted_in_tokens_not_characters():
    pool = [HumanMessage("你好"), AIMessage("你好！有什么可以帮你的吗？")]
    assert summary_rejection("用户打招呼，助手回应并提供帮助。", pool) is None


def test_a_real_summary_of_a_long_pool_is_accepted():
    assert summary_rejection(_GOOD_SUMMARY, _pool(40)) is None


# ===================== 2. automatic: history kept, proceed or refuse, pause =====================


async def test_a_rejected_summary_keeps_the_history_and_pauses(caplog):
    summary = _ScriptedSummaryModel(messages=iter([]), script=["Bonjour.", "Bonjour."])
    mw = _middleware(summary, window=10_000)
    state = _alternating(22, 300) + [HumanMessage("next question", id="q1")]

    with caplog.at_level(logging.INFO):
        result = await mw.abefore_model({"messages": state}, None)

    # Nothing rewritten: only the pause is written, anchored on the question.
    assert result == {COMPACTION_PAUSE_KEY: "q1"}
    assert len(summary.prompts) == 2
    rejected = [r for r in caplog.records if "summary rejected" in r.getMessage()]
    assert len(rejected) == 2
    assert all(r.levelno == logging.WARNING for r in rejected)
    assert all("summary_chars=8" in r.getMessage() for r in rejected)
    assert all(r.getMessage().isascii() for r in caplog.records)
    assert not any("Conversation compacted" in r.getMessage() for r in caplog.records)


async def test_a_failed_summary_call_keeps_the_history_too():
    """No more placeholder: a summary call that fails twice deterministically
    leaves the history as it is, like a rejected summary."""
    from tests.test_compaction_keep import _overflow_400

    summary = _ScriptedSummaryModel(messages=iter([]), script=[_overflow_400(), _overflow_400()])
    mw = _middleware(summary, window=10_000)
    state = _alternating(22, 300) + [HumanMessage("next question", id="q1")]

    assert await mw.abefore_model({"messages": state}, None) == {COMPACTION_PAUSE_KEY: "q1"}


async def test_a_rejected_summary_lets_the_turn_go_ahead_when_it_fits(monkeypatch):
    _WindowEngine.window = 10_000
    summary = _ScriptedSummaryModel(messages=iter([]), script=["Bonjour.", "Bonjour."])
    _models(monkeypatch, ["the answer"], summary)
    cp = InMemorySaver()
    seed = _alternating(22, 300)
    # Over the 80 % trigger, under the window: compaction is attempted, the
    # request still fits.
    assert 8_000 <= _request_tokens(seed) < 10_000 - 512
    await _seed(cp, "fit1", seed)

    events = await _turn(AgentRunner(checkpointer=cp), "fit1")

    assert "".join(e["text"] for e in events if e["t"] == "answer") == "the answer"
    assert _WindowEngine.rewrites == []
    assert not any(e["t"] == "compaction" and e["phase"] == "done" for e in events)
    state = await _state(cp, "fit1")
    assert [m.id for m in state[: len(seed)]] == [m.id for m in seed]
    assert state[-1].content == "the answer"
    assert _pause_of(cp, "fit1") == state[-2].id


async def test_a_rejected_summary_refuses_the_turn_when_it_does_not_fit(monkeypatch, caplog):
    _WindowEngine.window = 10_000
    summary = _ScriptedSummaryModel(messages=iter([]), script=["Bonjour.", "Bonjour."])
    _models(monkeypatch, ["never sent"], summary)
    cp = InMemorySaver()
    seed = _alternating(36, 300)
    assert _budget_request_tokens(seed) > 10_000
    await _seed(cp, "nofit1", seed)

    with caplog.at_level(logging.INFO):
        events = await _turn(AgentRunner(checkpointer=cp), "nofit1")

    (curated,) = _curated(events)
    assert curated["kind"] == "summary_failed"
    assert curated["params"]["context_tokens"] == 10_000
    assert curated["params"]["prompt_tokens"] > 10_000
    state = await _state(cp, "nofit1")
    assert [m.id for m in state[: len(seed)]] == [m.id for m in seed]
    assert state[-1].content == ERROR_MESSAGE
    # The pause is written even though the turn was refused.
    assert _pause_of(cp, "nofit1") == state[-2].id
    assert not any("Conversation compacted" in r.getMessage() for r in caplog.records)


async def test_the_pause_skips_the_next_messages_then_compaction_is_retried(monkeypatch):
    _WindowEngine.window = 10_000
    summary = _ScriptedSummaryModel(
        messages=iter([]), script=["Bonjour.", "Bonjour.", _GOOD_SUMMARY]
    )
    answers = ["a"] * (COMPACTION_PAUSE_MESSAGES + 2)
    _models(monkeypatch, answers, summary)
    cp = InMemorySaver()
    await _seed(cp, "pause1", _alternating(22, 300))
    runner = AgentRunner(checkpointer=cp)

    await _turn(runner, "pause1", "q0")
    assert len(summary.prompts) == 2
    anchor = _pause_of(cp, "pause1")
    assert anchor is not None

    for i in range(COMPACTION_PAUSE_MESSAGES):
        events = await _turn(runner, "pause1", f"q{i + 1}")
        assert _curated(events) == []
        # Paused: no summary call, the history is not rewritten.
        assert len(summary.prompts) == 2
        assert _WindowEngine.rewrites == []
        assert _pause_of(cp, "pause1") == anchor

    events = await _turn(runner, "pause1", "q-retry")
    assert len(summary.prompts) == 3
    assert _WindowEngine.rewrites == ["rewrite"]
    # Released once a compaction runs again.
    assert _pause_of(cp, "pause1") is None
    state = await _state(cp, "pause1")
    assert _GOOD_SUMMARY in state[0].content


async def test_a_paused_conversation_that_no_longer_fits_tries_one_compaction(monkeypatch):
    _WindowEngine.window = 10_000
    summary = _ScriptedSummaryModel(
        messages=iter([]), script=["Bonjour.", "Bonjour.", _GOOD_SUMMARY]
    )
    _models(monkeypatch, ["a", "b"], summary)
    cp = InMemorySaver()
    await _seed(cp, "pause2", _alternating(22, 300))
    runner = AgentRunner(checkpointer=cp)
    await _turn(runner, "pause2", "q0")
    assert len(summary.prompts) == 2
    big = _text(3_000, "y")
    assert _budget_request_tokens(await _state(cp, "pause2"), big) + 512 > 10_000

    # Past the window during the pause: one compaction is tried, and it works.
    events = await _turn(runner, "pause2", big)

    assert _curated(events) == []
    assert len(summary.prompts) == 3
    assert _WindowEngine.rewrites == ["rewrite"]
    assert _pause_of(cp, "pause2") is None


async def test_a_paused_conversation_whose_compaction_fails_again_is_refused(monkeypatch):
    _WindowEngine.window = 10_000
    summary = _ScriptedSummaryModel(messages=iter([]), script=["Bonjour."] * 4)
    _models(monkeypatch, ["a"], summary)
    cp = InMemorySaver()
    await _seed(cp, "pause3", _alternating(22, 300))
    runner = AgentRunner(checkpointer=cp)
    await _turn(runner, "pause3", "q0")
    anchor = _pause_of(cp, "pause3")

    events = await _turn(runner, "pause3", _text(3_000, "y"))

    (curated,) = _curated(events)
    assert curated["kind"] == "summary_failed"
    assert len(summary.prompts) == 4
    assert _WindowEngine.rewrites == []
    # The pause keeps its anchor: it is not extended by the extra attempt.
    assert _pause_of(cp, "pause3") == anchor


async def test_a_paused_conversation_that_fits_makes_no_summary_call(monkeypatch):
    _WindowEngine.window = 10_000
    summary = _ScriptedSummaryModel(messages=iter([]), script=["Bonjour.", "Bonjour."])
    _models(monkeypatch, ["a", "b"], summary)
    cp = InMemorySaver()
    await _seed(cp, "pause4", _alternating(22, 300))
    runner = AgentRunner(checkpointer=cp)
    await _turn(runner, "pause4", "q0")

    events = await _turn(runner, "pause4", "short")

    assert _curated(events) == []
    assert len(summary.prompts) == 2


# 2. /compact ends the pause.
async def test_a_manual_compaction_clears_the_pause(monkeypatch):
    from tests.test_manual_compaction import _compact

    _WindowEngine.window = 10_000
    summary = _ScriptedSummaryModel(
        messages=iter([]), script=["Bonjour.", "Bonjour.", _GOOD_SUMMARY]
    )
    _models(monkeypatch, ["a"], summary)
    cp = InMemorySaver()
    await _seed(cp, "pause5", _alternating(22, 300))
    runner = AgentRunner(checkpointer=cp)
    await _turn(runner, "pause5", "q0")
    assert _pause_of(cp, "pause5") is not None

    await _compact(runner, "pause5")

    assert _pause_of(cp, "pause5") is None


def test_the_pause_lasts_a_few_messages():
    assert COMPACTION_PAUSE_MESSAGES == 4


def test_a_rewritten_history_ends_the_pause():
    """``/compact`` rewrites the history: the anchor is gone, so is the pause."""
    state = _alternating(6, 10) + [HumanMessage("q", id="q1")]

    assert runner_module._compaction_paused(state, "q1")
    assert not runner_module._compaction_paused(state[2:], "gone")
    assert not runner_module._compaction_paused(state, None)


async def test_an_overhead_that_alone_does_not_fit_is_not_blamed_on_the_message():
    """The message is not what is too long: no ``message_too_long``, the
    engine's own overflow speaks."""
    from tests.test_real_token_compaction import _mw

    mw = _mw(10_000, overhead=50_000, summary=_ScriptedSummaryModel(messages=iter([])))
    state = [HumanMessage("hi", id="h0"), AIMessage("yo", id="a0"), HumanMessage("q", id="q")]

    # Not refused up front: the compaction runs (the engine will refuse
    # the request itself, with its own overflow).
    result = await mw.abefore_model({"messages": state}, None)

    assert result is not None and "messages" in result
    assert [m.id for m in result["messages"][2:]] == ["a0", "q"]


# ===================== 2. manual: the error, the history kept =====================


@pytest.mark.parametrize("which", ["too-short", "copy"])
async def test_a_rejected_manual_summary_fails_and_keeps_the_history(monkeypatch, which):
    from tests.test_manual_compaction import _Builds, _compact, _Store

    _WindowEngine.window = 10_000
    # The manual cutoff keeps [answer 19, "ok", "done"]: the pool ends with
    # question 18, which the "copy" summary repeats.
    seed = _pool(20) + [HumanMessage("ok"), AIMessage("done")]
    script = ["Hi.", "Hi."] if which == "too-short" else [seed[18].content] * 2
    _Builds(monkeypatch, _ScriptedSummaryModel(messages=iter([]), script=script))
    cp = InMemorySaver()
    for i, m in enumerate(seed):
        m.id = f"s{i}"
    await _seed(cp, "man1", seed)
    marker = _Store()

    with pytest.raises(CompactionFailedException):
        await _compact(AgentRunner(checkpointer=cp), "man1", marker)

    assert [m.id for m in await _state(cp, "man1")] == [m.id for m in seed]
    assert marker.events == []


@pytest.mark.parametrize("language", ["en", "fr", "es", "zh"])
def test_the_summary_failed_copy_names_compact_never_a_slash_command(language):
    import json
    from pathlib import Path

    from src.agents.curated_turns import CURATED_KINDS

    locales = Path(__file__).resolve().parents[2] / "frontend" / "src" / "locales"
    chat = json.loads((locales / language / "chat.json").read_text(encoding="utf-8"))
    line = chat["curated"]["summary_failed"]
    assert chat["commands"]["compact"]["label"] in line
    assert "/" not in line
    template = CURATED_KINDS["summary_failed"].template
    assert "Compact" in template and "/" not in template
