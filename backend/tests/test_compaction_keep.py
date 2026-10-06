"""Compaction keeps a TOKEN budget, a bounded summary, and never loses memory.

What a compaction keeps (#611): at most ``SUMMARY_KEEP_MESSAGES`` messages AND
at most ``KEEP_FRACTION`` of the working window, so the state left behind can
re-fire neither the token trigger nor the 20-message floor on the next model
call. The summary client is capped at ``summary_cap(W)`` tokens; the
summarizer input always carries the previous summary; a failed summary call
either fails the turn with the history intact (transient) or degrades to a
smaller retry and then to a placeholder that carries the previous summary
(deterministic). The cutoff rule is one pure function shared by the
middleware and the amber-warning projection.
"""

from __future__ import annotations

import inspect
import logging
from types import SimpleNamespace
from typing import Any, ClassVar

import httpx
import openai
import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from src.agents import runner as runner_module
from src.agents.reasoning_effort import NO_REASONING_PLAN
from src.agents.runner import (
    AgentRunner,
    GenParams,
    ERROR_SENTINEL,
    KEEP_FRACTION,
    compaction_cutoff,
    keep_token_budget,
    summarize_trim_budget,
    summary_cap,
)
from src.core import config
from src.engines.base_engine import BaseEngine
from tests._helpers import ToolableFakeChatModel

pytestmark = pytest.mark.unit

_PARAMS = GenParams(temperature=0.5, top_p=0.9, max_tokens=64)
_PREVIOUS = "The user is called Ada and lives in Lyon."


def _text(tokens: int, char: str = "a") -> str:
    """Text the chars/4 estimator counts as roughly ``tokens`` tokens."""
    return char * (tokens * 4)


def _count(messages) -> int:
    return count_tokens_approximately(messages)


def _previous_summary_message(text: str = _PREVIOUS) -> HumanMessage:
    return HumanMessage(
        content=f"Here is a summary of the conversation to date:\n\n{text}",
        additional_kwargs={"lc_source": "summarization"},
    )


def _alternating(n: int, tokens_each: int) -> list:
    out = []
    for i in range(n):
        cls = HumanMessage if i % 2 == 0 else AIMessage
        out.append(cls(content=_text(tokens_each, chr(ord("a") + i % 26)), id=f"m{i}"))
    return out


class _Llm:
    id = 7
    link = "/fake/path"
    name = "Test 7B"
    param_size = 7.0


class _WindowEngine(BaseEngine):
    """A bare engine whose loaded child reports ``window`` and that records
    every prefix hook call."""

    window: ClassVar[Any] = None
    claims: ClassVar[list] = []
    rewrites: ClassVar[list] = []

    @classmethod
    def effective_context_tokens(cls):
        return cls.window

    @classmethod
    def claim_prefix(cls, owner):
        cls.claims.append(owner)

    @classmethod
    def on_history_rewritten(cls):
        cls.rewrites.append("rewrite")


@pytest.fixture(autouse=True)
def _engine(monkeypatch):
    _WindowEngine.window = None
    _WindowEngine.claims = []
    _WindowEngine.rewrites = []
    monkeypatch.setattr(config, "LLM_Engine", _WindowEngine)
    yield
    _WindowEngine._last_used = None


class _ScriptedSummaryModel(ToolableFakeChatModel):
    """A summary client answering from a script: a string is the summary, an
    exception is raised. Records every prompt it was sent."""

    script: list = Field(default_factory=list)
    prompts: list = Field(default_factory=list)

    def _next(self, messages):
        self.prompts.append("\n".join(str(m.content) for m in messages))
        outcome = self.script.pop(0) if self.script else "A short summary."
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        text = self._next(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        text = self._next(messages)
        yield ChatGenerationChunk(message=AIMessageChunk(content=text, chunk_position="last"))


def _middleware(summary_model, window=None, budget=None):
    from langchain.agents.middleware import SummarizationMiddleware

    _WindowEngine.window = window
    built = AgentRunner()._build_middleware(summary_model, budget)
    return next(m for m in built if isinstance(m, SummarizationMiddleware))


def _after(result) -> list:
    """The state a compaction leaves: everything after the RemoveMessage."""
    assert result is not None
    return list(result["messages"][1:])


# ===================== the pure arithmetic =====================


def test_summary_cap_scales_with_the_window_and_is_clamped():
    assert summary_cap(1) == 128
    assert summary_cap(800) == 128
    assert summary_cap(4000) == 500
    assert summary_cap(8000) == 1000
    assert summary_cap(200_000) == 1024


@pytest.mark.parametrize("window", [1, 300, 480, 481])
def test_the_keep_budget_is_at_least_one_token_for_tiny_windows(window):
    assert keep_token_budget(window) == 1


@pytest.mark.parametrize("window", [1000, 2000, 4000, 10_000, 32_768])
def test_the_keep_budget_leaves_room_for_the_summary_under_the_trigger(window):
    keep = keep_token_budget(window)
    assert keep <= int(KEEP_FRACTION * window)
    # kept suffix + the longest summary stays 256 under the 80 % trigger.
    assert keep + summary_cap(window) <= int(0.8 * window) - 256


def test_the_keep_budget_values():
    assert KEEP_FRACTION == 0.4
    assert keep_token_budget(10_000) == 4000
    assert keep_token_budget(2000) == 800
    assert keep_token_budget(600) == 96


def test_the_trim_budget_without_a_window_is_langchains_default():
    assert summarize_trim_budget(None, None) == 4000


def test_the_trim_budget_is_bounded_by_the_allocated_window():
    window = allocated = 10_000
    cap = summary_cap(window)
    trim = summarize_trim_budget(window, allocated)
    assert trim == (allocated - 2 * cap - 512) // 3


def test_the_trim_budget_is_bounded_by_a_small_memory_window():
    # A memory-bound working window far below the allocation: the summarizer
    # prompt must still fit the MEMORY window, the 4000 floor notwithstanding.
    window, allocated = 1500, 32_768
    cap = summary_cap(window)
    assert summarize_trim_budget(window, allocated) == window - 2 * cap - 256


def test_the_trim_budget_never_drops_under_256():
    assert summarize_trim_budget(1, 1) == 256
    assert summarize_trim_budget(1, None) == 256


def test_the_trim_budget_grows_with_a_big_window():
    window = allocated = 131_072
    cap = summary_cap(window)
    assert summarize_trim_budget(window, allocated) == min(
        max(4000, int(0.8 * window) - cap),
        window - 2 * cap - 256,
        (allocated - 2 * cap - 512) // 3,
    )


# ===================== the cutoff rule =====================


def test_without_a_window_the_cutoff_keeps_the_last_ten_messages():
    messages = _alternating(30, 10)
    assert compaction_cutoff(messages, None, count_tokens_approximately) == 20


def test_with_a_window_the_cutoff_is_the_later_of_the_token_and_message_cutoffs():
    window = 10_000
    budget = keep_token_budget(window)
    # Short messages: the token budget alone would keep far more than 10.
    short = _alternating(30, 50)
    assert _count(short[-10:]) < budget
    assert compaction_cutoff(short, window, count_tokens_approximately) == 20
    # Long messages: the token budget keeps fewer than 10.
    long = _alternating(30, 900)
    cut = compaction_cutoff(long, window, count_tokens_approximately)
    assert cut > 20
    assert _count(long[cut:]) <= budget
    assert _count(long[cut - 1 :]) > budget


def test_an_oversized_last_message_is_kept_whole():
    messages = [HumanMessage("hi", id="a"), AIMessage("yo", id="b"), HumanMessage(_text(9000))]
    assert compaction_cutoff(messages, 10_000, count_tokens_approximately) == 2


def test_the_degenerate_window_of_one_keeps_only_the_last_message():
    messages = _alternating(5, 10)
    assert compaction_cutoff(messages, 1, count_tokens_approximately) == 4


def test_many_parallel_tool_calls_are_kept_with_their_call_documented_over_keep():
    calls = [{"name": "t", "args": {}, "id": f"c{i}"} for i in range(12)]
    messages = [
        HumanMessage(_text(500), id="h0"),
        AIMessage(_text(500), id="a0"),
        HumanMessage("use the tools", id="h1"),
        AIMessage(content="", tool_calls=calls, id="a1"),
        *[ToolMessage(_text(70), tool_call_id=f"c{i}", id=f"t{i}") for i in range(12)],
        AIMessage("done", id="a2"),
    ]
    cut = compaction_cutoff(messages, 2000, count_tokens_approximately)
    # Never split the AI message from its tool results: the cutoff moves BACK
    # to the AI message, keeping more than 10 messages and more than the budget.
    assert messages[cut].id == "a1"
    assert len(messages) - cut > runner_module.SUMMARY_KEEP_MESSAGES


# ===================== the middleware wiring =====================


def test_the_middleware_keeps_tokens_with_a_window_and_messages_without():
    with_window = _middleware(_ScriptedSummaryModel(messages=iter([])), window=10_000)
    assert with_window.keep == ("tokens", keep_token_budget(10_000))
    without = _middleware(_ScriptedSummaryModel(messages=iter([])), window=None)
    assert without.keep == ("messages", runner_module.SUMMARY_KEEP_MESSAGES)


def test_the_middleware_cutoff_is_the_pure_rule():
    mw = _middleware(_ScriptedSummaryModel(messages=iter([])), window=10_000)
    messages = _alternating(30, 900)
    assert mw._determine_cutoff_index(messages) == compaction_cutoff(
        messages, 10_000, count_tokens_approximately
    )


def test_the_middleware_trims_the_summarizer_input_to_the_window():
    mw = _middleware(_ScriptedSummaryModel(messages=iter([])), window=10_000)
    assert mw.trim_tokens_to_summarize == summarize_trim_budget(10_000, 10_000)


def test_the_middleware_uses_the_working_window_and_the_allocated_one_for_the_trim():
    budget = SimpleNamespace(tokens_at_margin=lambda margin: 1500)
    mw = _middleware(_ScriptedSummaryModel(messages=iter([])), window=32_768, budget=budget)
    assert mw.keep == ("tokens", keep_token_budget(1500))
    assert mw.trim_tokens_to_summarize == summarize_trim_budget(1500, 32_768)


async def test_the_611_loop_compacts_once_and_not_on_the_next_turn():
    """The last 10 messages alone exceed the trigger: keeping 10 messages
    re-fired compaction on every turn. The token keep ends the loop."""
    window = 10_000
    summary = _ScriptedSummaryModel(messages=iter([]), script=[_text(1000, "s")])
    mw = _middleware(summary, window=window)
    state = _alternating(14, 950) + [HumanMessage("next question", id="q1")]
    assert _count(state[-10:]) > int(0.8 * window)

    after = _after(await mw.abefore_model({"messages": state}, None))
    next_call = after + [AIMessage("answer", id="r1"), HumanMessage("again", id="q2")]

    assert await mw.abefore_model({"messages": next_call}, None) is None
    assert len(summary.prompts) == 1


async def test_thirty_short_messages_compact_once_and_not_on_the_next_turn():
    """Between 0.4 W and 0.8 W of short messages, the 20-message floor fires;
    a pure token keep would leave >= 20 messages and re-fire it every turn."""
    window = 10_000
    summary = _ScriptedSummaryModel(messages=iter([]))
    mw = _middleware(summary, window=window)
    state = _alternating(29, 200) + [HumanMessage(_text(200), id="q1")]
    total = _count(state)
    assert int(0.4 * window) < total < int(0.8 * window)

    after = _after(await mw.abefore_model({"messages": state}, None))
    assert len(after) <= runner_module.SUMMARY_KEEP_MESSAGES + 1
    next_call = after + [AIMessage("answer", id="r1"), HumanMessage("again", id="q2")]

    assert await mw.abefore_model({"messages": next_call}, None) is None
    assert len(summary.prompts) == 1


@pytest.mark.parametrize("window", [2000, 4000])
async def test_small_windows_do_not_retrigger_on_the_next_call(window):
    cap = summary_cap(window)
    # The longest summary the capped client can write.
    summary = _ScriptedSummaryModel(messages=iter([]), script=[_text(cap, "s")])
    mw = _middleware(summary, window=window)
    state = _alternating(9, window // 10) + [HumanMessage(_text(window // 10), id="q1")]
    assert _count(state) >= int(0.8 * window)

    after = _after(await mw.abefore_model({"messages": state}, None))
    next_call = after + [AIMessage("ok", id="r1"), HumanMessage("next", id="q2")]

    assert await mw.abefore_model({"messages": next_call}, None) is None


async def test_the_degenerate_window_compacts_every_call_down_to_the_last_message():
    summary = _ScriptedSummaryModel(messages=iter([]))
    mw = _middleware(summary, window=None, budget=SimpleNamespace(tokens_at_margin=lambda m: -5))
    state = [HumanMessage("a", id="1"), AIMessage("b", id="2"), HumanMessage("c", id="3")]

    after = _after(await mw.abefore_model({"messages": state}, None))

    assert [m.content for m in after[1:]] == ["c"]


async def test_an_oversized_last_message_survives_compaction_whole():
    summary = _ScriptedSummaryModel(messages=iter([]))
    mw = _middleware(summary, window=10_000)
    huge = HumanMessage(_text(9000), id="huge")
    state = [HumanMessage("hi", id="1"), AIMessage("yo", id="2"), huge]

    after = _after(await mw.abefore_model({"messages": state}, None))

    assert after[-1].content == huge.content


# ===================== the summarizer input and failures =====================


async def test_the_previous_summary_stays_in_the_summarizer_input_past_the_trim():
    summary = _ScriptedSummaryModel(messages=iter([]))
    mw = _middleware(summary, window=10_000)
    to_summarize = [_previous_summary_message()] + _alternating(20, 400)
    assert _count(to_summarize) > mw.trim_tokens_to_summarize

    await mw._acreate_summary(to_summarize)

    (prompt,) = summary.prompts
    assert _PREVIOUS in prompt
    assert prompt.count(_PREVIOUS) == 1


async def test_a_size_failure_retries_with_the_oversized_message_truncated():
    summary = _ScriptedSummaryModel(messages=iter([]), script=["Merged summary."])
    mw = _middleware(summary, window=10_000)
    budget = mw.trim_tokens_to_summarize
    to_summarize = [
        _previous_summary_message(),
        HumanMessage("question", id="h"),
        AIMessage(_text(budget * 3, "z"), id="huge"),
    ]

    result = await mw._acreate_summary(to_summarize)

    assert result == "Merged summary."
    (prompt,) = summary.prompts
    assert _PREVIOUS in prompt
    assert "zzzz" in prompt
    # The oversized answer reached the summarizer TRUNCATED to the budget.
    assert prompt.count("z") <= budget * 4


async def test_a_size_failure_that_fails_again_writes_a_placeholder_carrying_the_previous_summary(
    caplog,
):
    overflow = _overflow_400()
    summary = _ScriptedSummaryModel(messages=iter([]), script=[overflow])
    mw = _middleware(summary, window=10_000)
    budget = mw.trim_tokens_to_summarize
    to_summarize = [
        _previous_summary_message(),
        HumanMessage("question", id="h"),
        AIMessage(_text(budget * 3, "z"), id="huge"),
    ]

    with caplog.at_level(logging.WARNING):
        result = await mw._acreate_summary(to_summarize)

    assert result == f"{_PREVIOUS}\n\nLater messages of this conversation could not be summarized."
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2
    assert all(r.getMessage().isascii() for r in caplog.records)


async def test_a_placeholder_never_repeats_its_closing_sentence():
    previous = f"{_PREVIOUS}\n\nLater messages of this conversation could not be summarized."
    summary = _ScriptedSummaryModel(messages=iter([]), script=[_overflow_400(), _overflow_400()])
    mw = _middleware(summary, window=10_000)

    result = await mw._acreate_summary(
        [_previous_summary_message(previous), HumanMessage("q", id="h"), AIMessage("a", id="a")]
    )

    assert result == previous


async def test_a_placeholder_without_a_previous_summary_says_so():
    summary = _ScriptedSummaryModel(messages=iter([]), script=[_overflow_400(), _overflow_400()])
    mw = _middleware(summary, window=10_000)

    result = await mw._acreate_summary([HumanMessage("q", id="h"), AIMessage("a", id="a")])

    assert result == "Earlier messages of this conversation could not be summarized."


async def test_an_unexpected_summary_error_logs_one_error_and_takes_the_size_path(caplog):
    summary = _ScriptedSummaryModel(
        messages=iter([]), script=[ValueError("a bug"), "Recovered summary."]
    )
    mw = _middleware(summary, window=10_000)

    with caplog.at_level(logging.WARNING):
        result = await mw._acreate_summary(
            [_previous_summary_message(), HumanMessage("q", id="h"), AIMessage("a", id="a")]
        )

    assert result == "Recovered summary."
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert errors[0].exc_info is not None


def _request():
    return httpx.Request("POST", "http://127.0.0.1:27300/v1/chat/completions")


def _overflow_400():
    detail = "Request needs 5037 context tokens (5029 prompt + 8 max generation), but MAX_KV_SIZE is 4096."
    response = httpx.Response(400, request=_request(), json={"detail": detail})
    return openai.BadRequestError(detail, response=response, body={"detail": detail})


@pytest.mark.parametrize(
    "exc",
    [
        openai.APIConnectionError(request=_request()),
        openai.APITimeoutError(request=_request()),
        openai.InternalServerError(
            "boom", response=httpx.Response(500, request=_request()), body=None
        ),
        openai.RateLimitError("slow", response=httpx.Response(429, request=_request()), body=None),
    ],
    ids=["connection", "timeout", "5xx", "429"],
)
async def test_a_transient_summary_error_is_raised(exc):
    summary = _ScriptedSummaryModel(messages=iter([]), script=[exc])
    mw = _middleware(summary, window=10_000)

    with pytest.raises(type(exc)):
        await mw._acreate_summary(
            [_previous_summary_message(), HumanMessage("q", id="h"), AIMessage("a", id="a")]
        )


async def test_engine_and_watchdog_failures_are_transient():
    from src.core.exceptions import EngineException, GenerationTimeoutException

    for exc in (
        EngineException(message="child died"),
        GenerationTimeoutException("silent", phase="first-chunk", budget_s=1.0),
    ):
        summary = _ScriptedSummaryModel(messages=iter([]), script=[exc])
        mw = _middleware(summary, window=10_000)
        with pytest.raises(type(exc)):
            await mw._acreate_summary([HumanMessage("q", id="h"), AIMessage("a", id="a")])


# ===================== through the runner and the real graph =====================


def _models(monkeypatch, answers, summary_model, captured=None):
    """Patch the factory: the main client answers from ``answers``; the
    summary client (built at effort "none") is ``summary_model``."""
    main = ToolableFakeChatModel(messages=iter([AIMessage(content=a) for a in answers]))

    def _build(llm, **kw):
        if kw.get("effort_plan") is NO_REASONING_PLAN:
            if captured is not None:
                captured.update(kw)
            return summary_model
        return main

    monkeypatch.setattr(runner_module, "build_chat_model", _build)


async def _seed(checkpointer, thread_id, messages):
    probe = create_agent(
        ToolableFakeChatModel(messages=iter([])), tools=[], checkpointer=checkpointer
    )
    await probe.aupdate_state(
        {"configurable": {"thread_id": thread_id}}, {"messages": messages}, as_node="model"
    )


async def _state(checkpointer, thread_id):
    probe = create_agent(
        ToolableFakeChatModel(messages=iter([])), tools=[], checkpointer=checkpointer
    )
    snap = await probe.aget_state({"configurable": {"thread_id": thread_id}})
    return snap.values["messages"]


async def _turn(runner, thread_id, question="next question"):
    return [
        e
        async for e in runner.astream_text(
            llm=_Llm(),
            user_message=question,
            system_prompt="s",
            params=_PARAMS,
            thread_id=thread_id,
            summarize=True,
            emit_events=True,
        )
    ]


def _seed_messages():
    return [_previous_summary_message()] + _alternating(14, 900)


async def test_the_summary_client_is_capped_and_never_auto_budgeted(monkeypatch):
    _WindowEngine.window = 10_000
    captured: dict = {}
    _models(monkeypatch, ["hello"], _ScriptedSummaryModel(messages=iter([])), captured)

    await _turn(AgentRunner(checkpointer=InMemorySaver()), "cap1")

    assert captured["max_tokens"] == summary_cap(10_000)
    assert captured["auto_output_budget"] is False


async def test_without_a_window_the_summary_client_keeps_todays_budget(monkeypatch):
    captured: dict = {}
    _models(monkeypatch, ["hello"], _ScriptedSummaryModel(messages=iter([])), captured)

    await _turn(AgentRunner(checkpointer=InMemorySaver()), "cap2")

    assert captured["max_tokens"] == _PARAMS.max_tokens
    assert captured.get("auto_output_budget", True) is True


async def test_a_compaction_resets_the_prefix_exactly_once(monkeypatch):
    _WindowEngine.window = 10_000
    _models(monkeypatch, ["first", "second"], _ScriptedSummaryModel(messages=iter([])))
    cp = InMemorySaver()
    await _seed(cp, "r1", _seed_messages())
    runner = AgentRunner(checkpointer=cp)

    await _turn(runner, "r1")
    assert _WindowEngine.rewrites == ["rewrite"]

    # No compaction on the next turn -> no reset.
    await _turn(runner, "r1")
    assert _WindowEngine.rewrites == ["rewrite"]


async def test_a_transient_summary_failure_fails_the_turn_and_keeps_the_history(monkeypatch):
    _WindowEngine.window = 10_000
    summary = _ScriptedSummaryModel(
        messages=iter([]), script=[openai.APIConnectionError(request=_request()), "Recovered."]
    )
    _models(monkeypatch, ["answer one", "answer two"], summary)
    cp = InMemorySaver()
    seed = _seed_messages()
    await _seed(cp, "t1", seed)
    runner = AgentRunner(checkpointer=cp)

    events = await _turn(runner, "t1")

    assert any(e["t"] == "answer" and e["text"].startswith(ERROR_SENTINEL) for e in events)
    state = await _state(cp, "t1")
    # Nothing was summarized away: every seeded message is still there.
    assert [m.content for m in state[: len(seed)]] == [m.content for m in seed]
    assert _WindowEngine.rewrites == []

    # The next turn retries the compaction, and it succeeds.
    events = await _turn(runner, "t1", "retry")
    assert not any(e["t"] == "answer" and e["text"].startswith(ERROR_SENTINEL) for e in events)
    assert _WindowEngine.rewrites == ["rewrite"]
    state = await _state(cp, "t1")
    assert "Recovered." in state[0].content


async def test_a_summary_overflow_degrades_to_a_placeholder_and_the_turn_succeeds(monkeypatch):
    _WindowEngine.window = 10_000
    summary = _ScriptedSummaryModel(messages=iter([]), script=[_overflow_400(), _overflow_400()])
    _models(monkeypatch, ["the answer"], summary)
    cp = InMemorySaver()
    await _seed(cp, "o1", _seed_messages())

    events = await _turn(AgentRunner(checkpointer=cp), "o1")

    answers = "".join(e["text"] for e in events if e["t"] == "answer")
    assert answers == "the answer"
    assert _WindowEngine.rewrites == ["rewrite"]
    # The retry sent a smaller summarizer input than the first attempt.
    first, second = summary.prompts
    assert len(second) < len(first)
    state = await _state(cp, "o1")
    assert _PREVIOUS in state[0].content
    assert "could not be summarized" in state[0].content


async def test_the_memory_warning_projects_the_kept_suffix_plus_the_summary_cap(monkeypatch):
    _WindowEngine.window = 10_000
    _models(monkeypatch, ["hello"], _ScriptedSummaryModel(messages=iter([])))
    calls = []

    class _Budget:
        weights_bytes = 10

        def tokens_at_margin(self, margin):
            return None

        def memory_margin_fraction(self, tokens):
            calls.append(tokens)
            return 0.5

        def conversation_bytes(self, tokens):
            return 1

    monkeypatch.setattr(
        runner_module, "MemoryBudget", SimpleNamespace(from_engine=lambda engine: _Budget())
    )
    cp = InMemorySaver()
    runner = AgentRunner(checkpointer=cp)

    await _turn(runner, "w1")

    state = await _state(cp, "w1")
    cut = compaction_cutoff(state, 10_000, count_tokens_approximately)
    assert calls[0] == _count(state[cut:]) + summary_cap(10_000)


# ===================== the private LangChain surface this rests on =====================


def test_the_langchain_surface_the_compaction_overrides_is_pinned():
    import langchain
    from langchain.agents.middleware import SummarizationMiddleware

    assert langchain.__version__ == "1.3.9"
    assert list(inspect.signature(SummarizationMiddleware._determine_cutoff_index).parameters) == [
        "self",
        "messages",
    ]
    assert list(inspect.signature(SummarizationMiddleware._acreate_summary).parameters) == [
        "self",
        "messages_to_summarize",
    ]
    assert inspect.iscoroutinefunction(SummarizationMiddleware._acreate_summary)
    assert list(inspect.signature(SummarizationMiddleware.abefore_model).parameters) == [
        "self",
        "state",
        "runtime",
    ]
    assert list(inspect.signature(SummarizationMiddleware._find_safe_cutoff_point).parameters) == [
        "messages",
        "cutoff_index",
    ]
    built = SummarizationMiddleware._build_new_messages("X")
    assert built[0].additional_kwargs == {"lc_source": "summarization"}
    assert built[0].content == "Here is a summary of the conversation to date:\n\nX"
    mw = SummarizationMiddleware(model=ToolableFakeChatModel(messages=iter([])))
    assert hasattr(mw, "_partial_token_counter")
    assert hasattr(mw, "trim_tokens_to_summarize")
