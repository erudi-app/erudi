"""A turn the client interrupts still leaves an alternating history.

A disconnect (Stop, a closed tab, the app's client timeout) cancels the model
node before it writes the turn's answer, so the checkpoint would hold the
turn's question with no answer after it -- and the next question would make
two user messages in a row, which strict chat templates (Gemma, Mistral)
reject on every later turn. The runner therefore closes an interrupted
stateful turn itself, inside the generation guard and shielded from the
cancellation that triggered it: one AIMessage carrying the answer text already
streamed (what the conversation service persists), or a short curated line
when nothing was streamed, marked ``erudi_interrupted``.
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
    assert messages[-1].additional_kwargs == {"erudi_interrupted": True}


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
    assert messages[-1].additional_kwargs == {"erudi_interrupted": True}


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
    real = AgentRunner._write_interrupted_turn

    async def _spy(self, agent, run_config, text):
        await real(self, agent, run_config, text)
        _RecEngine.events.append(("interrupted-written", None))

    monkeypatch.setattr(AgentRunner, "_write_interrupted_turn", _spy)
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
