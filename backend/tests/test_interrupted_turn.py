"""A turn the client interrupts still leaves an alternating history.

A disconnect (Stop, a closed tab, the app's client timeout) cancels the model
node before it writes the turn's answer, so the checkpoint would hold the
turn's question with no answer after it -- and the next question would make
two user messages in a row, which strict chat templates (Gemma, Mistral)
reject on every later turn. The runner therefore closes an interrupted
stateful turn itself, inside the generation guard and shielded from the
cancellation that triggered it: one AIMessage carrying the answer text already
streamed (what the conversation service persists), or a short curated line
when nothing was streamed -- preceded by a closing tool result for every tool
call left unanswered. The decision reads the COMMITTED checkpoint: a node
that finished after the consumer stopped reading leaves its answer as a
pending write, which the next turn's input discards.
"""

from __future__ import annotations

import asyncio
import logging

import anyio
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langgraph.checkpoint.memory import InMemorySaver

from src.agents import runner as runner_module
from src.agents.runner import AgentRunner
from tests._helpers import ToolableFakeChatModel
from tests.test_prefix_cache_turns import (  # noqa: F401  (``_engine`` is an autouse fixture)
    _RecEngine,
    _SlowModel,
    _engine,
    _index,
    _patch_model,
    _stream,
)

pytestmark = pytest.mark.unit


class _SilentModel(ToolableFakeChatModel):
    """Called, then silent: the client goes away before any token."""

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        _RecEngine.events.append(("model-called", None))
        await asyncio.sleep(30)
        yield ChatGenerationChunk(message=AIMessageChunk(content="never"))


async def _state(checkpointer, thread_id="1"):
    tup = await checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
    return tup.checkpoint["channel_values"]["messages"]


def _alternates(messages):
    return not any(a.type == "human" and b.type == "human" for a, b in zip(messages, messages[1:]))


async def _cancel_after(runner, ready_event, *, thread_id="1"):
    """Consume a turn inside an anyio cancel scope (Starlette's disconnect
    path) and cancel it once ``ready_event`` fires."""
    holder: dict = {}

    async def _consume():
        with anyio.CancelScope() as scope:
            holder["scope"] = scope
            async for event in _stream(runner, thread_id=thread_id, summarize=False):
                if event.get("t") == "answer":
                    ready_event.set()

    task = asyncio.create_task(_consume())
    return task, holder


async def test_a_disconnect_mid_answer_closes_the_turn_with_the_streamed_text(monkeypatch):
    _patch_model(monkeypatch, _SlowModel(messages=iter([]), first="The partial answer "))
    checkpointer = InMemorySaver()
    runner = AgentRunner(checkpointer=checkpointer)
    got_first = asyncio.Event()
    task, holder = await _cancel_after(runner, got_first)

    await asyncio.wait_for(got_first.wait(), timeout=5)
    holder["scope"].cancel()
    await asyncio.wait_for(task, timeout=5)

    messages = await _state(checkpointer)
    assert [m.type for m in messages] == ["human", "ai"]
    assert messages[-1].content == "The partial answer "


async def test_a_disconnect_before_any_token_closes_the_turn_with_the_curated_line(monkeypatch):
    _patch_model(monkeypatch, _SilentModel(messages=iter([])))
    checkpointer = InMemorySaver()
    runner = AgentRunner(checkpointer=checkpointer)
    holder: dict = {}

    async def _consume():
        with anyio.CancelScope() as scope:
            holder["scope"] = scope
            async for _ in _stream(runner, summarize=False):
                pass

    task = asyncio.create_task(_consume())
    for _ in range(250):
        if ("model-called", None) in _RecEngine.events:
            break
        await asyncio.sleep(0.02)
    holder["scope"].cancel()
    await asyncio.wait_for(task, timeout=5)

    messages = await _state(checkpointer)
    assert [m.type for m in messages] == ["human", "ai"]
    assert messages[-1].content == runner_module.INTERRUPTED_ANSWER_MESSAGE
    assert runner_module.INTERRUPTED_ANSWER_MESSAGE.isascii()


async def test_a_closed_consumer_closes_the_turn_too(monkeypatch):
    """The GeneratorExit path: the service's stream is closed (``aclose``)."""
    _patch_model(monkeypatch, _SlowModel(messages=iter([]), first="Half "))
    checkpointer = InMemorySaver()
    gen = _stream(AgentRunner(checkpointer=checkpointer), summarize=False)
    while (await gen.__anext__()).get("t") != "answer":
        pass

    await asyncio.wait_for(gen.aclose(), timeout=5)

    messages = await _state(checkpointer)
    assert messages[-1].type == "ai"
    assert messages[-1].content == "Half "


async def test_the_write_happens_before_the_guard_is_released(monkeypatch):
    _patch_model(monkeypatch, _SlowModel(messages=iter([])))
    real = AgentRunner._append_closure

    async def _spy(self, agent, run_config, text):
        await real(self, agent, run_config, text)
        _RecEngine.events.append(("interrupted-written", None))

    monkeypatch.setattr(AgentRunner, "_append_closure", _spy)
    gen = _stream(AgentRunner(checkpointer=InMemorySaver()), summarize=False)
    while (await gen.__anext__()).get("t") != "answer":
        pass

    await asyncio.wait_for(gen.aclose(), timeout=5)

    assert _index(("interrupted-written", None)) < _index(("guard-released", None))


async def test_the_next_turn_alternates(monkeypatch):
    checkpointer = InMemorySaver()
    _patch_model(monkeypatch, _SlowModel(messages=iter([]), first="cut "))
    gen = _stream(AgentRunner(checkpointer=checkpointer), summarize=False)
    while (await gen.__anext__()).get("t") != "answer":
        pass
    await asyncio.wait_for(gen.aclose(), timeout=5)

    _patch_model(
        monkeypatch, ToolableFakeChatModel(messages=iter([AIMessage(content="next answer")]))
    )
    _ = [e async for e in _stream(AgentRunner(checkpointer=checkpointer), question="again")]

    messages = await _state(checkpointer)
    assert [m.type for m in messages] == ["human", "ai", "human", "ai"]
    assert _alternates(messages)


async def test_a_turn_already_answered_is_not_closed_twice(monkeypatch):
    """The node finished just before the disconnect: the state already ends
    with the answer -- check, don't assume."""
    checkpointer = InMemorySaver()
    _patch_model(monkeypatch, ToolableFakeChatModel(messages=iter([AIMessage(content="done")])))
    runner = AgentRunner(checkpointer=checkpointer)
    _ = [event async for event in _stream(runner, summarize=False)]
    from langchain.agents import create_agent

    probe = create_agent(
        ToolableFakeChatModel(messages=iter([])), tools=[], checkpointer=checkpointer
    )
    await runner._write_interrupted_turn(probe, {"configurable": {"thread_id": "1"}}, "late")

    messages = await _state(checkpointer)
    assert [m.content for m in messages] == ["hi", "done"]


async def test_a_failing_write_logs_one_warning_and_the_cancellation_still_propagates(
    monkeypatch, caplog
):
    _patch_model(monkeypatch, _SlowModel(messages=iter([])))

    def _boom(text):
        raise RuntimeError("checkpoint write failed")

    monkeypatch.setattr(runner_module, "_interrupted_answer", _boom)
    runner = AgentRunner(checkpointer=InMemorySaver())
    got_first = asyncio.Event()

    async def _consume():
        async for event in _stream(runner, summarize=False):
            if event.get("t") == "answer":
                got_first.set()

    with caplog.at_level(logging.WARNING):
        task = asyncio.create_task(_consume())
        await asyncio.wait_for(got_first.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)

    warnings = [r for r in caplog.records if "interrupted turn" in r.getMessage().lower()]
    assert len(warnings) == 1
    assert warnings[0].exc_info is not None
    assert ("guard-released", None) in _RecEngine.events


async def test_the_arena_writes_nothing(monkeypatch):
    _patch_model(monkeypatch, _SlowModel(messages=iter([])))
    calls = []

    async def _spy(self, *args, **kwargs):
        calls.append(args)

    monkeypatch.setattr(AgentRunner, "_write_interrupted_turn", _spy)
    from tests.test_prefix_cache_turns import _PARAMS, _Llm

    gen = AgentRunner().astream_text(
        llm=_Llm(), user_message="hi", system_prompt="s", params=_PARAMS, emit_events=True
    )
    while (await gen.__anext__()).get("t") != "answer":
        pass
    await asyncio.wait_for(gen.aclose(), timeout=5)

    assert calls == []


def test_an_interrupted_answer_is_costed_by_its_text():
    """No usage on the closing message: rule 2, its own script weight."""
    from src.agents.token_accounting import (
        BUDGET_DENSE_TOKENS,
        estimate,
        exact_tokens,
        real_tokens_est,
        script_weight,
    )

    closing = runner_module._interrupted_answer("partial answer text " * 20)
    assert exact_tokens(closing) is None
    cost = real_tokens_est([closing], dense=BUDGET_DENSE_TOKENS).total
    assert cost == -(
        -script_weight(closing.content, dense=BUDGET_DENSE_TOKENS) * estimate(closing) // 1
    )


# ===================== the decision reads the COMMITTED checkpoint =====================

_CFG = {"configurable": {"thread_id": "c"}}


class _QuickModel(ToolableFakeChatModel):
    """Streams many chunks without pausing, then finishes."""

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        for i in range(40):
            yield ChatGenerationChunk(message=AIMessageChunk(content=f"w{i} "))
        _RecEngine.events.append(("model-done", None))


async def _pending_messages(checkpointer, config):
    tup = await checkpointer.aget_tuple(config)
    return [w for w in (tup.pending_writes or []) if w[1] == "messages"] if tup else []


async def _committed(checkpointer, config=_CFG):
    return (await checkpointer.aget_tuple(config)).checkpoint["channel_values"]["messages"]


async def test_a_node_that_finished_after_the_consumer_stopped_reading_is_still_closed(
    monkeypatch,
):
    """The model node finished and saved its answer as a PENDING write while
    the consumer no longer read; the stream is then closed. ``aget_state``
    would show that answer, but the next turn's input discards it: the
    committed state is what decides."""
    _patch_model(monkeypatch, _QuickModel(messages=iter([])))
    checkpointer = InMemorySaver()
    config = {"configurable": {"thread_id": "1"}}
    gen = _stream(AgentRunner(checkpointer=checkpointer), summarize=False)
    while (await gen.__anext__()).get("t") != "answer":
        pass
    for _ in range(250):
        if ("model-done", None) in _RecEngine.events and await _pending_messages(
            checkpointer, config
        ):
            break
        await asyncio.sleep(0.02)
    assert await _pending_messages(checkpointer, config), "the answer is a pending write"

    await asyncio.wait_for(gen.aclose(), timeout=5)

    assert [m.type for m in await _committed(checkpointer, config)] == ["human", "ai"]
    _patch_model(
        monkeypatch, ToolableFakeChatModel(messages=iter([AIMessage(content="next answer")]))
    )
    _ = [e async for e in _stream(AgentRunner(checkpointer=checkpointer), question="again")]
    assert _alternates(await _state(checkpointer))


async def _seeded(checkpointer, messages):
    from langchain.agents import create_agent

    probe = create_agent(
        ToolableFakeChatModel(messages=iter([])), tools=[], checkpointer=checkpointer
    )
    await probe.aupdate_state(_CFG, {"messages": messages}, as_node="model")
    return probe


def _calls(*ids):
    return [{"name": "search_knowledge_base", "args": {}, "id": i} for i in ids]


@pytest.mark.parametrize("answered", [0, 1], ids=["no-result", "partial-results"])
async def test_unanswered_tool_calls_get_closing_results_then_the_answer(answered):
    from langchain_core.messages import HumanMessage, ToolMessage

    checkpointer = InMemorySaver()
    seed = [HumanMessage("q", id="h"), AIMessage("", tool_calls=_calls("c1", "c2"), id="a")]
    seed += [ToolMessage("result", tool_call_id="c1", name="search_knowledge_base", id="t1")][
        :answered
    ]
    agent = await _seeded(checkpointer, seed)

    await AgentRunner(checkpointer=checkpointer)._write_interrupted_turn(agent, _CFG, "")

    messages = await _committed(checkpointer)
    tail = messages[len(seed) :]
    unanswered = ["c1", "c2"][answered:]
    assert [m.type for m in tail] == ["tool"] * len(unanswered) + ["ai"]
    assert [m.tool_call_id for m in tail[:-1]] == unanswered
    assert all(m.content == runner_module.INTERRUPTED_TOOL_RESULT for m in tail[:-1])
    assert runner_module.INTERRUPTED_TOOL_RESULT.isascii()
    assert tail[-1].content == runner_module.INTERRUPTED_ANSWER_MESSAGE
    assert not any(a.type == "ai" and b.type == "ai" for a, b in zip(messages, messages[1:]))


async def test_a_tool_round_with_every_call_answered_gets_the_answer():
    from langchain_core.messages import HumanMessage, ToolMessage

    checkpointer = InMemorySaver()
    agent = await _seeded(
        checkpointer,
        [
            HumanMessage("q", id="h"),
            AIMessage("", tool_calls=_calls("c1"), id="a"),
            ToolMessage("result", tool_call_id="c1", name="search_knowledge_base", id="t1"),
        ],
    )

    await AgentRunner(checkpointer=checkpointer)._write_interrupted_turn(agent, _CFG, "partial")

    messages = await _committed(checkpointer)
    assert [m.type for m in messages] == ["human", "ai", "tool", "ai"]
    assert messages[-1].content == "partial"


async def test_a_committed_answer_is_left_alone():
    from langchain_core.messages import HumanMessage

    checkpointer = InMemorySaver()
    agent = await _seeded(checkpointer, [HumanMessage("q", id="h"), AIMessage("a", id="a")])

    await AgentRunner(checkpointer=checkpointer)._write_interrupted_turn(agent, _CFG, "late")

    assert [m.content for m in await _committed(checkpointer)] == ["q", "a"]


class _PendingViewAgent:
    """The compiled graph, but ``aget_state`` shows an answer the committed
    checkpoint does not hold (a pending write) -- the view to ignore."""

    def __init__(self, agent):
        self._agent = agent

    async def aget_state(self, config):
        from types import SimpleNamespace

        messages = list(await _committed(self._agent.checkpointer, config))
        return SimpleNamespace(values={"messages": messages + [AIMessage("pending")]})

    async def aupdate_state(self, *args, **kwargs):
        return await self._agent.aupdate_state(*args, **kwargs)


async def test_the_closure_decides_on_the_committed_state_not_the_pending_view():
    from langchain_core.messages import HumanMessage

    checkpointer = InMemorySaver()
    agent = _PendingViewAgent(await _seeded(checkpointer, [HumanMessage("q", id="h")]))

    await AgentRunner(checkpointer=checkpointer)._write_interrupted_turn(agent, _CFG, "cut")

    assert [m.type for m in await _committed(checkpointer)] == ["human", "ai"]


async def test_the_alternation_repair_decides_on_the_committed_state_too():
    from langchain_core.messages import HumanMessage

    checkpointer = InMemorySaver()
    agent = _PendingViewAgent(await _seeded(checkpointer, [HumanMessage("q", id="h")]))

    await AgentRunner(checkpointer=checkpointer)._repair_alternation(agent, _CFG)

    messages = await _committed(checkpointer)
    assert [m.type for m in messages] == ["human", "ai"]
    assert messages[-1].content == runner_module.ERROR_MESSAGE


async def test_the_alternation_repair_closes_unanswered_tool_calls():
    from langchain_core.messages import HumanMessage

    checkpointer = InMemorySaver()
    agent = await _seeded(
        checkpointer, [HumanMessage("q", id="h"), AIMessage("", tool_calls=_calls("c1"), id="a")]
    )

    await AgentRunner(checkpointer=checkpointer)._repair_alternation(agent, _CFG)

    assert [m.type for m in await _committed(checkpointer)] == ["human", "ai", "tool", "ai"]


# ===================== the write is bounded and never loses a cancellation =====================


async def test_a_native_cancellation_during_the_write_is_re_raised(monkeypatch):
    """The exit in flight is a GeneratorExit (``aclose``); a cancellation
    absorbed while the shielded write ran must not be lost."""
    _patch_model(monkeypatch, _SlowModel(messages=iter([])))
    entered, release = asyncio.Event(), asyncio.Event()

    async def _slow_closure(self, agent, run_config, text):
        entered.set()
        await release.wait()

    monkeypatch.setattr(AgentRunner, "_append_closure", _slow_closure)
    gen = _stream(AgentRunner(checkpointer=InMemorySaver()), summarize=False)
    while (await gen.__anext__()).get("t") != "answer":
        pass

    closing = asyncio.create_task(gen.aclose())
    await asyncio.wait_for(entered.wait(), timeout=5)
    closing.cancel()
    await asyncio.sleep(0.05)
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(closing, timeout=5)
    assert ("guard-released", None) in _RecEngine.events


async def test_a_write_that_hangs_is_cut_with_one_warning_and_the_guard_released(
    monkeypatch, caplog
):
    _patch_model(monkeypatch, _SlowModel(messages=iter([])))
    monkeypatch.setattr(runner_module, "INTERRUPTED_WRITE_TIMEOUT_S", 0.05)

    async def _hung_closure(self, agent, run_config, text):
        await asyncio.sleep(30)

    monkeypatch.setattr(AgentRunner, "_append_closure", _hung_closure)
    gen = _stream(AgentRunner(checkpointer=InMemorySaver()), summarize=False)
    while (await gen.__anext__()).get("t") != "answer":
        pass

    with caplog.at_level(logging.WARNING):
        await asyncio.wait_for(gen.aclose(), timeout=5)

    warnings = [r for r in caplog.records if "interrupted turn" in r.getMessage().lower()]
    assert len(warnings) == 1
    assert warnings[0].getMessage().isascii()
    assert ("guard-released", None) in _RecEngine.events


async def test_the_write_happens_before_the_guard_on_the_anyio_cancel_path(monkeypatch):
    _patch_model(monkeypatch, _SlowModel(messages=iter([])))
    real = AgentRunner._append_closure

    async def _spy(self, agent, run_config, text):
        await real(self, agent, run_config, text)
        _RecEngine.events.append(("interrupted-written", None))

    monkeypatch.setattr(AgentRunner, "_append_closure", _spy)
    got_first = asyncio.Event()
    task, holder = await _cancel_after(AgentRunner(checkpointer=InMemorySaver()), got_first)
    await asyncio.wait_for(got_first.wait(), timeout=5)
    holder["scope"].cancel()
    await asyncio.wait_for(task, timeout=5)

    assert _index(("interrupted-written", None)) < _index(("guard-released", None))


# ===================== against the app's checkpointer =====================


async def test_the_closure_against_the_postgres_checkpointer(monkeypatch, pg_test_cluster):
    from src.agents.checkpoint import open_checkpointer

    _patch_model(monkeypatch, _SlowModel(messages=iter([]), first="Postgres partial "))
    config = {"configurable": {"thread_id": "pg-interrupt"}}
    async with open_checkpointer(pg_test_cluster.psycopg_url) as saver:
        gen = _stream(AgentRunner(checkpointer=saver), thread_id="pg-interrupt", summarize=False)
        while (await gen.__anext__()).get("t") != "answer":
            pass
        await asyncio.wait_for(gen.aclose(), timeout=10)
        messages = await _committed(saver, config)

    assert [m.type for m in messages] == ["human", "ai"]
    assert messages[-1].content == "Postgres partial "
