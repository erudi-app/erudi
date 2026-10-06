"""Prefix-cache ownership through real turns: claims, teardown, cancellation.

The runner claims the prefix for the conversation (``conv:<id>``) or the arena
before the agent runs; titles claim nothing. Every generator in the
services -> runner -> LangGraph chain is consumed under ``aclosing``, so a
closed or abandoned stream closes LangGraph (and its in-flight node) inside the
generation guard, and the runner flags the child abandoned. When a client
disconnect (anyio cancel scope, the Starlette path) tears the guard holder down
while a reset thread still runs, the next acquirer drains it first.
"""

from __future__ import annotations

import asyncio
import gc
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Callable, ClassVar, Optional

import anyio
import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk
from langgraph.checkpoint.memory import InMemorySaver

from src.agents import runner as runner_module
from src.agents.reasoning_effort import NO_REASONING_PLAN
from src.agents.runner import AgentRunner, GenParams
from src.core import config
from src.engines.base_engine import BaseEngine
from tests._helpers import ToolableFakeChatModel

_PARAMS = GenParams(temperature=0.5, top_p=0.9, max_tokens=64)


class _Llm:
    id = 7
    link = "/fake/path"
    name = "Test 7B"
    param_size = 7.0


class _RecEngine(BaseEngine):
    """Records every prefix hook and guard release, in order. A hook can be
    made to block its worker thread on ``blocker``."""

    events: ClassVar[list] = []
    window: ClassVar[Any] = None
    blocker: ClassVar[Optional["_Blocker"]] = None
    block_on: ClassVar[Optional[str]] = None

    @classmethod
    def effective_context_tokens(cls):
        return cls.window

    @classmethod
    def _maybe_block(cls, name):
        if cls.block_on == name and cls.blocker is not None:
            cls.block_on = None
            cls.blocker.wait()

    @classmethod
    def claim_prefix(cls, owner):
        cls.events.append(("claim-start", owner))
        cls._maybe_block("claim")
        cls.events.append(("claim-end", owner))

    @classmethod
    def on_history_rewritten(cls):
        cls.events.append(("rewrite-start", None))
        cls._maybe_block("rewrite")
        cls.events.append(("rewrite-end", None))

    @classmethod
    @asynccontextmanager
    async def generation_guard(cls):
        try:
            async with super().generation_guard():
                cls.events.append(("guard-acquired", None))
                yield
        finally:
            cls.events.append(("guard-released", None))


class _Blocker:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished_at: Optional[datetime] = None

    def wait(self):
        self.entered.set()
        self.release.wait(10)
        self.finished_at = datetime.now()


@pytest.fixture(autouse=True)
def _engine(monkeypatch):
    _RecEngine.events = []
    _RecEngine.window = None
    _RecEngine.blocker = None
    _RecEngine.block_on = None
    BaseEngine._pending_reset = None
    monkeypatch.setattr(config, "LLM_Engine", _RecEngine)
    yield
    if _RecEngine.blocker is not None:
        _RecEngine.blocker.release.set()
    BaseEngine._pending_reset = None
    _RecEngine._last_used = None


def _names(kind):
    return [e for e in _RecEngine.events if e[0] == kind]


def _index(event):
    return _RecEngine.events.index(event)


def _patch_model(monkeypatch, model, summary_model=None):
    def _build(llm, **kw):
        if kw.get("effort_plan") is NO_REASONING_PLAN and summary_model is not None:
            return summary_model
        _RecEngine.events.append(("model-built", None))
        return model

    monkeypatch.setattr(runner_module, "build_chat_model", _build)


class _SlowModel(ToolableFakeChatModel):
    """Streams one chunk, then hangs; records when its stream is closed and
    carries an ``abandon_hook`` like the real client."""

    abandon_hook: Optional[Callable[[], None]] = None
    first: str = "first "

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        _RecEngine.events.append(("model-called", None))
        try:
            yield ChatGenerationChunk(message=AIMessageChunk(content=self.first))
            await asyncio.sleep(30)
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="never", chunk_position="last")
            )
        finally:
            _RecEngine.events.append(("node-closed", None))


class _Flag:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1


def _stream(runner, *, thread_id="1", summarize=True, question="hi"):
    return runner.astream_text(
        llm=_Llm(),
        user_message=question,
        system_prompt="s",
        params=_PARAMS,
        thread_id=thread_id,
        summarize=summarize,
        emit_events=True,
    )


# ===================== who claims what =====================


async def test_a_conversation_turn_claims_its_thread_after_the_model_and_before_the_agent(
    monkeypatch,
):
    _patch_model(monkeypatch, ToolableFakeChatModel(messages=iter([AIMessage(content="ok")])))

    _ = [e async for e in _stream(AgentRunner(checkpointer=InMemorySaver()), thread_id="42")]

    assert _names("claim-start") == [("claim-start", "conv:42")]
    assert _index(("model-built", None)) < _index(("claim-start", "conv:42"))
    assert _index(("claim-end", "conv:42")) < _index(("guard-released", None))


async def test_the_arena_claims_the_arena(monkeypatch):
    _patch_model(monkeypatch, ToolableFakeChatModel(messages=iter([AIMessage(content="duel")])))
    runner = AgentRunner(checkpointer=None)

    out = [
        t
        async for t in runner.astream_text(
            llm=_Llm(),
            user_message="hi",
            system_prompt="s",
            params=_PARAMS,
            thread_id=None,
            summarize=False,
        )
    ]

    assert "".join(out) == "duel"
    assert _names("claim-start") == [("claim-start", "arena")]


async def test_a_title_claims_nothing(monkeypatch):
    from types import SimpleNamespace

    class _OneShot:
        async def astream(self, messages):
            yield SimpleNamespace(text="A Title")

    monkeypatch.setattr(runner_module, "build_chat_model", lambda llm, **kw: _OneShot())
    runner = AgentRunner(checkpointer=None)

    out = [
        t
        async for t in runner.astream_oneshot(
            llm=_Llm(), prompt_text="p", temperature=0.5, top_p=0.9, max_tokens=12
        )
    ]

    assert "".join(out) == "A Title"
    assert _names("claim-start") == []


# ===================== the generator chain =====================


async def test_closing_the_runner_mid_stream_closes_the_node_before_the_guard(monkeypatch):
    flag = _Flag()
    _patch_model(monkeypatch, _SlowModel(messages=iter([]), abandon_hook=flag))
    gen = _stream(AgentRunner(checkpointer=InMemorySaver()), summarize=False)

    first = await gen.__anext__()
    assert first == {"t": "answer", "text": "first "}
    await asyncio.wait_for(gen.aclose(), timeout=5)

    assert _index(("node-closed", None)) < _index(("guard-released", None))
    assert flag.calls >= 1


async def test_a_gc_finalized_runner_closes_the_node_before_the_guard(monkeypatch):
    flag = _Flag()
    _patch_model(monkeypatch, _SlowModel(messages=iter([]), abandon_hook=flag))
    runner = AgentRunner(checkpointer=InMemorySaver())

    async def _consume_one_then_drop():
        gen = _stream(runner, summarize=False)
        await gen.__anext__()

    await _consume_one_then_drop()
    gc.collect()
    for _ in range(250):
        if ("guard-released", None) in _RecEngine.events:
            break
        await asyncio.sleep(0.02)

    assert ("guard-released", None) in _RecEngine.events
    assert _index(("node-closed", None)) < _index(("guard-released", None))
    assert flag.calls >= 1


async def test_closing_the_conversation_service_closes_the_whole_chain_inside_the_guard(
    test_db_session, mock_llm, monkeypatch
):
    """The service iterates the runner under ``aclosing`` too: closing the
    service generator at its yield reaches LangGraph inside the guard, before
    ``aclose`` returns -- nothing is left to the garbage collector."""
    from src.domains.conversations.schemas import ConversationQuery
    from src.domains.conversations.services import ConversationService

    flag = _Flag()
    _patch_model(monkeypatch, _SlowModel(messages=iter([]), abandon_hook=flag))
    service = ConversationService(test_db_session, InMemorySaver())
    conversation = service.create_conversation(
        llm_id=mock_llm.id, temperature=0.7, top_p=0.9, max_tokens=64
    )
    gen = service.query_and_respond_stream(conversation.id, ConversationQuery(question="hi"))

    async for chunk in gen:
        if "first" in chunk:
            break
    await asyncio.wait_for(gen.aclose(), timeout=5)

    assert ("guard-released", None) in _RecEngine.events
    assert _index(("node-closed", None)) < _index(("guard-released", None))
    assert flag.calls >= 1


async def _anyio_cancelled_turn(monkeypatch, flag):
    """A turn consumed inside an anyio cancel scope -- Starlette's client
    disconnect path -- cancelled once its first answer chunk arrived."""
    _patch_model(monkeypatch, _SlowModel(messages=iter([]), abandon_hook=flag))
    runner = AgentRunner(checkpointer=InMemorySaver())
    got_first = asyncio.Event()
    holder: dict = {}

    async def _consume():
        with anyio.CancelScope() as scope:
            holder["scope"] = scope
            async for _ in _stream(runner, summarize=False):
                got_first.set()

    task = asyncio.create_task(_consume())
    await asyncio.wait_for(got_first.wait(), timeout=5)
    holder["scope"].cancel()
    await asyncio.wait_for(task, timeout=5)
    for _ in range(25):
        if ("node-closed", None) in _RecEngine.events:
            break
        await asyncio.sleep(0.02)


async def test_an_anyio_cancelled_consumer_releases_the_guard_and_flags_the_child(monkeypatch):
    flag = _Flag()
    await _anyio_cancelled_turn(monkeypatch, flag)

    assert ("guard-released", None) in _RecEngine.events
    assert flag.calls >= 1


@pytest.mark.xfail(
    strict=True,
    reason=(
        "langgraph 1.2.11: AsyncPregelLoop.__aexit__ unwinds its exit stack in a "
        "separate task and awaits it; anyio re-delivers the cancellation, which "
        "cancels that task before it runs, so AsyncBackgroundExecutor.__aexit__ "
        "never cancels the in-flight model node. aclosing cannot reach it: the "
        "LangGraph generator is already finished by the exception."
    ),
)
async def test_an_anyio_cancelled_consumer_closes_the_in_flight_node(monkeypatch):
    await _anyio_cancelled_turn(monkeypatch, _Flag())

    assert ("node-closed", None) in _RecEngine.events


# ===================== cancellation while a reset thread runs =====================


async def _seed(checkpointer, thread_id, messages):
    probe = create_agent(
        ToolableFakeChatModel(messages=iter([])), tools=[], checkpointer=checkpointer
    )
    await probe.aupdate_state(
        {"configurable": {"thread_id": thread_id}}, {"messages": messages}, as_node="model"
    )


def _long_history():
    out = []
    for i in range(14):
        cls = HumanMessage if i % 2 == 0 else AIMessage
        out.append(cls(content=chr(ord("a") + i) * 3600))
    return out


async def _next_acquirer(kind, runner):
    """Through the common acquire helper: a turn, a title, or the idle tick."""
    if kind == "turn":
        _ = [e async for e in _stream(runner, thread_id="next", summarize=False)]
    elif kind == "title":
        _ = [
            t
            async for t in runner.astream_oneshot(
                llm=_Llm(), prompt_text="p", temperature=0.5, top_p=0.9, max_tokens=12
            )
        ]
    else:
        await _RecEngine._cleanup_tick()
    _RecEngine.events.append(("next-proceeded", kind))


class _NextModel:
    """Builds a fresh scripted model per call for the follow-up acquirer."""

    def __init__(self, first_model):
        self.first = first_model
        self.used = False

    def __call__(self, llm, **kw):
        _RecEngine.events.append(("model-built", None))
        if not self.used:
            self.used = True
            return self.first
        if kw.get("disable_thinking"):
            from types import SimpleNamespace

            class _OneShot:
                async def astream(self, messages):
                    yield SimpleNamespace(text="Title")

            return _OneShot()
        return ToolableFakeChatModel(messages=iter([AIMessage(content="next answer")]))


async def _cancel_while_blocked(service_stream_factory, blocker):
    holder: dict = {}

    async def _consume():
        with anyio.CancelScope() as scope:
            holder["scope"] = scope
            async for _ in service_stream_factory():
                pass

    task = asyncio.create_task(_consume())
    await asyncio.wait_for(asyncio.to_thread(blocker.entered.wait, 5), timeout=6)
    assert blocker.entered.is_set()
    holder["scope"].cancel()
    return task


@pytest.mark.parametrize("acquirer", ["turn", "title", "idle_tick"])
async def test_a_disconnect_during_the_claim_reset_holds_every_next_acquirer(
    acquirer, test_db_session, mock_llm, monkeypatch
):
    from src.domains.conversations.schemas import ConversationQuery
    from src.domains.conversations.services import ConversationService

    blocker = _Blocker()
    _RecEngine.blocker = blocker
    _RecEngine.block_on = "claim"
    first = ToolableFakeChatModel(messages=iter([AIMessage(content="never sent")]))
    monkeypatch.setattr(runner_module, "build_chat_model", _NextModel(first))
    service = ConversationService(test_db_session, InMemorySaver())
    conversation = service.create_conversation(
        llm_id=mock_llm.id, temperature=0.7, top_p=0.9, max_tokens=64
    )
    payload = ConversationQuery(question="hello")

    task = await _cancel_while_blocked(
        lambda: service.query_and_respond_stream(conversation.id, payload), blocker
    )
    nxt = asyncio.create_task(_next_acquirer(acquirer, service.runner))
    await asyncio.sleep(0.2)
    assert ("next-proceeded", acquirer) not in _RecEngine.events

    blocker.release.set()
    await asyncio.wait_for(task, timeout=10)
    await asyncio.wait_for(nxt, timeout=10)

    owner = f"conv:{conversation.id}"
    assert _index(("claim-end", owner)) < _index(("next-proceeded", acquirer))
    if acquirer == "turn":
        # The orphan claim finished writing before the next turn claimed.
        assert _index(("claim-end", owner)) < _index(("claim-start", "conv:next"))


@pytest.mark.parametrize("acquirer", ["turn", "title", "idle_tick"])
async def test_a_disconnect_during_the_compaction_reset_holds_every_next_acquirer(
    acquirer, test_db_session, mock_llm, monkeypatch
):
    """The compaction reset runs in a LangGraph node task: anyio's repeated
    cancellation can unwind the guard holder while the node still waits on
    the reset thread. The next acquirer drains it before proceeding."""
    from src.domains.conversations.schemas import ConversationQuery
    from src.domains.conversations.services import ConversationService

    blocker = _Blocker()
    _RecEngine.blocker = blocker
    _RecEngine.block_on = "rewrite"
    _RecEngine.window = 10_000
    first = ToolableFakeChatModel(messages=iter([AIMessage(content="answer after compaction")]))
    summary = ToolableFakeChatModel(messages=iter([AIMessage(content="A summary.")]))
    builder = _NextModel(first)

    def _build(llm, **kw):
        if kw.get("effort_plan") is NO_REASONING_PLAN:
            return summary
        return builder(llm, **kw)

    monkeypatch.setattr(runner_module, "build_chat_model", _build)
    checkpointer = InMemorySaver()
    service = ConversationService(test_db_session, checkpointer)
    conversation = service.create_conversation(
        llm_id=mock_llm.id, temperature=0.7, top_p=0.9, max_tokens=64
    )
    await _seed(checkpointer, str(conversation.id), _long_history())
    payload = ConversationQuery(question="hello")

    task = await _cancel_while_blocked(
        lambda: service.query_and_respond_stream(conversation.id, payload), blocker
    )
    nxt = asyncio.create_task(_next_acquirer(acquirer, service.runner))
    await asyncio.sleep(0.2)
    assert ("next-proceeded", acquirer) not in _RecEngine.events

    blocker.release.set()
    await asyncio.wait_for(task, timeout=10)
    await asyncio.wait_for(nxt, timeout=10)

    # The guard holder was torn down while the reset thread still ran: only
    # the drain held the next acquirer.
    assert _index(("guard-released", None)) < _index(("rewrite-end", None))
    assert _index(("rewrite-end", None)) < _index(("next-proceeded", acquirer))
    if acquirer == "turn":
        assert _index(("rewrite-end", None)) < _index(("claim-start", "conv:next"))
