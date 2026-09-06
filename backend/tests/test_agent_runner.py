"""P3 — AgentRunner: happy path, multi-turn persistence, arena mode, error policy.

Uses a real ``GenericFakeChatModel`` (not an AsyncMock — ``create_agent``
validates the model type and runs it through the LangGraph runtime) injected by
patching ``build_chat_model``. The engine is a bare ``BaseEngine`` subclass so
``generation_guard`` works without spawning a real model.
"""

import logging
from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from tests._helpers import ToolableFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.agents import runner as runner_module
from src.agents.model_factory import build_chat_model
from src.agents.runner import AgentRunner, GenParams, ERROR_SENTINEL
from src.agents.tools import calculator
from src.core import config
from src.engines.base_engine import BaseEngine

pytestmark = pytest.mark.unit


class _FakeEngine(BaseEngine):
    """Supplies generation_guard without touching real engine state."""


class _Llm:
    id = 7
    link = "/fake/path"
    name = "Test 7B"
    param_size = 7.0


_PARAMS = GenParams(temperature=0.5, top_p=0.9, max_tokens=64)


@pytest.fixture(autouse=True)
def _engine(monkeypatch):
    monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
    yield
    _FakeEngine._last_used = None


def _patch_model(monkeypatch, fake_model):
    monkeypatch.setattr(runner_module, "build_chat_model", lambda llm, **kw: fake_model)


async def test_astream_yields_raw_token_text(monkeypatch):
    fake = ToolableFakeChatModel(messages=iter([AIMessage(content="Python is awesome")]))
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=InMemorySaver())
    out = [
        t
        async for t in runner.astream_text(
            llm=_Llm(),
            user_message="hi",
            system_prompt="sys",
            params=_PARAMS,
            thread_id="c1",
            summarize=False,
        )
    ]
    # raw concatenation of token text (the text/plain wire contract)
    assert "".join(out) == "Python is awesome"


async def test_multi_turn_restores_context_from_checkpointer(monkeypatch):
    fake = ToolableFakeChatModel(
        messages=iter([AIMessage(content="first"), AIMessage(content="second")])
    )
    _patch_model(monkeypatch, fake)
    cp = InMemorySaver()
    runner = AgentRunner(checkpointer=cp)
    cfg = {"configurable": {"thread_id": "c1"}}

    async for _ in runner.astream_text(
        llm=_Llm(), user_message="q1", system_prompt="s", params=_PARAMS, thread_id="c1"
    ):
        pass
    async for _ in runner.astream_text(
        llm=_Llm(), user_message="q2", system_prompt="s", params=_PARAMS, thread_id="c1"
    ):
        pass

    # Only the new message is sent each turn; the checkpointer restores + appends.
    probe = create_agent(ToolableFakeChatModel(messages=iter([])), tools=[], checkpointer=cp)
    snap = await probe.aget_state(cfg)
    assert [m.type for m in snap.values["messages"]] == ["human", "ai", "human", "ai"]


async def test_arena_mode_runs_without_checkpointer(monkeypatch):
    fake = ToolableFakeChatModel(messages=iter([AIMessage(content="duel answer")]))
    _patch_model(monkeypatch, fake)
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
    assert "".join(out) == "duel answer"


async def test_tools_none_builds_zero_tool_agent(monkeypatch, caplog):
    """#129: with no explicit tools the agent is built with NO tools at all.

    Every production path goes through ``plan_turn`` and passes an explicit
    list; ``tools=None`` must not silently sneak the calculator back in."""
    fake = ToolableFakeChatModel(messages=iter([AIMessage(content="ok")]))
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=None)

    with caplog.at_level(logging.INFO, logger="erudi"):
        async for _ in runner.astream_text(
            llm=_Llm(),
            user_message="hi",
            system_prompt="s",
            params=_PARAMS,
            thread_id=None,
            summarize=False,
        ):
            pass

    built = [r.message for r in caplog.records if "Agent built" in r.message]
    assert built, f"no 'Agent built' log found in: {[r.message for r in caplog.records]}"
    assert "tools=[]" in built[0]


async def test_construction_error_yields_sentinel(monkeypatch):
    def _boom(llm, **kw):
        raise RuntimeError("model load failed")

    monkeypatch.setattr(runner_module, "build_chat_model", _boom)
    runner = AgentRunner(checkpointer=InMemorySaver())
    out = [
        t
        async for t in runner.astream_text(
            llm=_Llm(), user_message="hi", system_prompt="s", params=_PARAMS, thread_id="c1"
        )
    ]
    assert any(ERROR_SENTINEL in t for t in out)
    # error message must NOT leak a traceback
    assert all("Traceback" not in t for t in out)


async def test_recursion_limit_loop_falls_back_to_last_tool_result(monkeypatch):
    """#277: a model that loops forever on the same tool call is bounded by the
    recursion limit and degrades to the last tool result instead of hanging."""
    import itertools

    from langchain_core.tools import tool

    @tool
    def loop_tool(query: str) -> str:
        """A tool that always returns the same result (drives the fixed point)."""
        return "stuck result"

    call = AIMessage(
        content="",
        tool_calls=[{"name": "loop_tool", "args": {"query": "x"}, "id": "c"}],
    )
    fake = ToolableFakeChatModel(messages=itertools.repeat(call))
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=InMemorySaver())

    out = [
        t
        async for t in runner.astream_text(
            llm=_Llm(),
            user_message="loop please",
            system_prompt="s",
            params=_PARAMS,
            thread_id="c1",
            tools=[loop_tool],
        )
    ]
    joined = "".join(out)
    # It terminated (no 16-minute hang, no unhandled GraphRecursionError) and
    # delivered what was gathered rather than the generic streaming error.
    assert "stuck result" in joined
    assert runner_module.ERROR_MESSAGE not in joined


async def test_recursion_limit_loop_no_result_yields_loop_message(monkeypatch, caplog):
    """#277: loop with nothing usable to fall back to -> curated loop-limit turn."""
    import itertools

    from langchain_core.tools import tool

    @tool
    def blank_tool(query: str) -> str:
        """A tool that returns nothing usable."""
        return ""

    call = AIMessage(
        content="",
        tool_calls=[{"name": "blank_tool", "args": {"query": "x"}, "id": "c"}],
    )
    fake = ToolableFakeChatModel(messages=itertools.repeat(call))
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=InMemorySaver())

    with caplog.at_level(logging.WARNING, logger="erudi"):
        out = [
            t
            async for t in runner.astream_text(
                llm=_Llm(),
                user_message="loop please",
                system_prompt="s",
                params=_PARAMS,
                thread_id="c1",
                tools=[blank_tool],
            )
        ]
    joined = "".join(out)
    assert ERROR_SENTINEL in joined
    assert "rephrasing" in joined
    assert any("recursion limit" in r.message for r in caplog.records)


async def test_repair_alternation_appends_ai_after_dangling_human(monkeypatch):
    # M2: a failed turn that left a dangling HumanMessage must be repaired so the
    # next turn doesn't send two consecutive user messages (local templates 400).
    cp = InMemorySaver()
    agent = create_agent(ToolableFakeChatModel(messages=iter([])), tools=[], checkpointer=cp)
    cfg = {"configurable": {"thread_id": "c1"}}
    await agent.aupdate_state(cfg, {"messages": [HumanMessage("orphan question")]})
    assert (await agent.aget_state(cfg)).values["messages"][-1].type == "human"

    runner = AgentRunner(checkpointer=cp)
    await runner._repair_alternation(agent, cfg)

    msgs = (await agent.aget_state(cfg)).values["messages"]
    assert msgs[-1].type == "ai"
    assert ERROR_SENTINEL in msgs[-1].content


def test_build_middleware_includes_strip_and_summarization():
    from langchain.agents.middleware import SummarizationMiddleware

    built = AgentRunner()._build_middleware(ToolableFakeChatModel(messages=iter([])))
    assert any(isinstance(m, SummarizationMiddleware) for m in built)
    assert any(type(m).__name__ == "_StripStaleImagesMiddleware" for m in built)


async def test_summarization_compacts_checkpointer_state(monkeypatch):
    import itertools

    # Lower the thresholds so summarization fires within a few turns.
    monkeypatch.setattr(runner_module, "SUMMARY_TRIGGER_MESSAGES", 6)
    monkeypatch.setattr(runner_module, "SUMMARY_KEEP_MESSAGES", 2)

    def infinite():
        for i in itertools.count():
            yield AIMessage(content=f"answer {i} with several words here")

    fake = ToolableFakeChatModel(messages=infinite())
    monkeypatch.setattr(runner_module, "build_chat_model", lambda llm, **kw: fake)

    cp = InMemorySaver()
    runner = AgentRunner(checkpointer=cp)
    for turn in range(5):
        async for _ in runner.astream_text(
            llm=_Llm(),
            user_message=f"q{turn}",
            system_prompt="s",
            params=_PARAMS,
            thread_id="c1",
            summarize=True,
        ):
            pass

    probe = create_agent(ToolableFakeChatModel(messages=iter([])), tools=[], checkpointer=cp)
    msgs = (await probe.aget_state({"configurable": {"thread_id": "c1"}})).values["messages"]
    # 5 turns = 10 messages un-summarized; compaction keeps the agent context bounded.
    assert len(msgs) < 10
    assert any("summary of the conversation" in m.content.lower() for m in msgs)


def test_build_chat_model_uses_engine_handle(monkeypatch):
    # build_chat_model must point ChatOpenAI at the engine base_url and use the
    # engine's _payload_model_value (MLX sentinel), not handle["alias"].
    class _Engine:
        @staticmethod
        def get_model_and_tokenizer(llm_id, link):
            return ({"base_url": "http://127.0.0.1:8080", "alias": f"erudi-{llm_id}"}, {})

        @staticmethod
        def _payload_model_value(handle):
            return "default_model"  # MLX-style sentinel

    monkeypatch.setattr(config, "LLM_Engine", _Engine)
    chat = build_chat_model(_Llm(), temperature=0.3, top_p=0.8, max_tokens=55)
    assert chat.model_name == "default_model"
    assert chat.openai_api_base == "http://127.0.0.1:8080/v1"
    assert chat.temperature == 0.3
    # Repetition controls restored on the ChatOpenAI path (regression: tiny models
    # looped without them). Identity engine (no _translate_payload_kwargs) => HF names.
    assert chat.extra_body == {"repetition_penalty": 1.1, "repetition_context_size": 64}


def test_build_chat_model_translates_extra_body_per_engine(monkeypatch):
    # llama.cpp engines rename repetition_penalty -> repeat_penalty (and
    # repetition_context_size -> repeat_last_n). build_chat_model must route the
    # repetition controls through the engine's _translate_payload_kwargs so each
    # local server receives its own wire names in extra_body.
    class _LlamaEngine:
        @staticmethod
        def get_model_and_tokenizer(llm_id, link):
            return ({"base_url": "http://127.0.0.1:9090", "alias": f"erudi-{llm_id}"}, {})

        @staticmethod
        def _payload_model_value(handle):
            return handle["alias"]

        @staticmethod
        def _translate_payload_kwargs(kw):
            rename = {
                "repetition_penalty": "repeat_penalty",
                "repetition_context_size": "repeat_last_n",
            }
            return {rename.get(k, k): v for k, v in kw.items()}

    monkeypatch.setattr(config, "LLM_Engine", _LlamaEngine)
    chat = build_chat_model(_Llm(), temperature=0.3, top_p=0.8, max_tokens=55)
    assert chat.extra_body == {"repeat_penalty": 1.1, "repeat_last_n": 64}


class _IdentityEngine:
    """Engine stub with NO _translate_payload_kwargs (identity translation),
    mirroring MLX: mlx_vlm.server reads the HF-named fields natively."""

    @staticmethod
    def get_model_and_tokenizer(llm_id, link):
        return ({"base_url": "http://127.0.0.1:8080", "alias": f"erudi-{llm_id}"}, {})

    @staticmethod
    def _payload_model_value(handle):
        return "default_model"


def test_build_chat_model_disable_thinking_sets_enable_thinking_false(monkeypatch):
    # #266: one-shot utility calls (titles) suppress reasoning at the chat
    # template level; the flag rides extra_body next to the repetition controls.
    monkeypatch.setattr(config, "LLM_Engine", _IdentityEngine)
    chat = build_chat_model(
        _Llm(), temperature=0.3, top_p=0.8, max_tokens=12, disable_thinking=True
    )
    assert chat.extra_body == {
        "repetition_penalty": 1.1,
        "repetition_context_size": 64,
        "enable_thinking": False,
    }


def test_build_chat_model_authenticates_with_the_handles_key(monkeypatch):
    # llama-server is spawned with a per-spawn `--api-key` so that nothing else
    # on the loopback interface can drive it. Inference goes through this
    # ChatOpenAI, so it has to present that key or every turn would 401 against
    # our own child server.
    class _KeyedEngine:
        @staticmethod
        def get_model_and_tokenizer(llm_id, link):
            return (
                {
                    "base_url": "http://127.0.0.1:27201",
                    "alias": f"erudi-{llm_id}",
                    "api_key": "per-spawn-secret",
                },
                {},
            )

        @staticmethod
        def _payload_model_value(handle):
            return handle["alias"]

    monkeypatch.setattr(config, "LLM_Engine", _KeyedEngine)
    chat = build_chat_model(_Llm(), temperature=0.3, top_p=0.8, max_tokens=55)
    assert chat.openai_api_key.get_secret_value() == "per-spawn-secret"


def test_build_chat_model_keeps_the_placeholder_key_without_one(monkeypatch):
    # A handle without a key keeps the literal, because an empty api_key makes
    # the OpenAI client fall back to reading OPENAI_API_KEY from the environment.
    monkeypatch.setattr(config, "LLM_Engine", _IdentityEngine)
    chat = build_chat_model(_Llm(), temperature=0.3, top_p=0.8, max_tokens=55)
    assert chat.openai_api_key.get_secret_value() == "not-needed"


def test_build_chat_model_never_logs_the_api_key(monkeypatch, caplog):
    # The factory logs the whole ChatOpenAI construction at INFO, and backend
    # logs are shipped in bug reports; the key must not ride along.
    class _KeyedEngine:
        @staticmethod
        def get_model_and_tokenizer(llm_id, link):
            return (
                {
                    "base_url": "http://127.0.0.1:27201",
                    "alias": f"erudi-{llm_id}",
                    "api_key": "per-spawn-secret",
                },
                {},
            )

        @staticmethod
        def _payload_model_value(handle):
            return handle["alias"]

    monkeypatch.setattr(config, "LLM_Engine", _KeyedEngine)
    with caplog.at_level(logging.DEBUG):
        build_chat_model(_Llm(), temperature=0.3, top_p=0.8, max_tokens=55)
    for record in caplog.records:
        assert "per-spawn-secret" not in record.getMessage()


def test_build_chat_model_default_omits_enable_thinking(monkeypatch):
    # Regression guard (#266): chat paths never pass disable_thinking, so their
    # request body must stay byte-identical to before (no enable_thinking key).
    monkeypatch.setattr(config, "LLM_Engine", _IdentityEngine)
    chat = build_chat_model(_Llm(), temperature=0.3, top_p=0.8, max_tokens=55)
    assert "enable_thinking" not in chat.extra_body


# ===== Integration (IT3 / IT5 / IT11) — PR1 E2E validation, runner level =====


async def test_thread_id_isolation_no_cross_bleed(monkeypatch):
    # IT3: two conversations -> two checkpointer threads; neither leaks into the
    # other (each thread's history holds only its own user messages).
    import itertools

    def infinite():
        for i in itertools.count():
            yield AIMessage(content=f"reply {i}")

    monkeypatch.setattr(
        runner_module,
        "build_chat_model",
        lambda llm, **kw: ToolableFakeChatModel(messages=infinite()),
    )
    cp = InMemorySaver()
    runner = AgentRunner(checkpointer=cp)

    async for _ in runner.astream_text(
        llm=_Llm(), user_message="alpha", system_prompt="s", params=_PARAMS, thread_id="conv-1"
    ):
        pass
    async for _ in runner.astream_text(
        llm=_Llm(), user_message="beta", system_prompt="s", params=_PARAMS, thread_id="conv-2"
    ):
        pass

    probe = create_agent(ToolableFakeChatModel(messages=iter([])), tools=[], checkpointer=cp)
    msgs_1 = (await probe.aget_state({"configurable": {"thread_id": "conv-1"}})).values["messages"]
    msgs_2 = (await probe.aget_state({"configurable": {"thread_id": "conv-2"}})).values["messages"]
    assert [m.content for m in msgs_1 if m.type == "human"] == ["alpha"]
    assert [m.content for m in msgs_2 if m.type == "human"] == ["beta"]


async def test_purged_thread_starts_fresh_no_resurrection(monkeypatch):
    # IT5 (BLOCKER B3): once a thread is purged (conversation deleted), reusing the
    # same thread_id — SQLite reuses autoincrement ids — must start a FRESH thread,
    # never resurrecting the deleted conversation's history.
    monkeypatch.setattr(
        runner_module,
        "build_chat_model",
        lambda llm, **kw: ToolableFakeChatModel(
            messages=iter([AIMessage(content="a"), AIMessage(content="b")])
        ),
    )
    cp = InMemorySaver()
    runner = AgentRunner(checkpointer=cp)
    cfg = {"configurable": {"thread_id": "5"}}

    async for _ in runner.astream_text(
        llm=_Llm(), user_message="old-secret", system_prompt="s", params=_PARAMS, thread_id="5"
    ):
        pass
    await cp.adelete_thread("5")
    assert await cp.aget_tuple(cfg) is None

    async for _ in runner.astream_text(
        llm=_Llm(), user_message="brand-new", system_prompt="s", params=_PARAMS, thread_id="5"
    ):
        pass

    probe = create_agent(ToolableFakeChatModel(messages=iter([])), tools=[], checkpointer=cp)
    msgs = (await probe.aget_state(cfg)).values["messages"]
    # Only the new turn — the deleted "old-secret" turn must NOT reappear.
    assert [m.type for m in msgs] == ["human", "ai"]
    assert [m.content for m in msgs if m.type == "human"] == ["brand-new"]


async def test_astream_holds_generation_lock_across_whole_stream(monkeypatch):
    # IT11: the runner wraps model resolution + the ENTIRE token stream in
    # engine.generation_guard, so the shared generation lock is held for every
    # token. The idle-cleanup tick takes that same lock, so it can never reap
    # the model mid-stream.
    monkeypatch.setattr(
        runner_module,
        "build_chat_model",
        lambda llm, **kw: ToolableFakeChatModel(
            messages=iter([AIMessage(content="one two three")])
        ),
    )
    runner = AgentRunner(checkpointer=InMemorySaver())

    observed_locked = []
    async for _ in runner.astream_text(
        llm=_Llm(), user_message="hi", system_prompt="s", params=_PARAMS, thread_id="c1"
    ):
        lock = _FakeEngine._generation_lock
        observed_locked.append(lock is not None and lock.locked())

    assert observed_locked and all(observed_locked)  # lock held for every token
    # Released once the stream completes (model reapable again).
    assert _FakeEngine._generation_lock is None or not _FakeEngine._generation_lock.locked()


# ===================== KB context middleware (PR3, issue #81) =====================

from pydantic import Field  # noqa: E402


class _RecordingModel(ToolableFakeChatModel):
    """Fake model that records the exact message lists it receives."""

    received: list = Field(default_factory=list)

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        self.received.append(list(messages))
        yield from super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


_BLOCK_1 = "[Document: a.md]\nLe préavis est de 90 jours.\n\nAnswer ONLY from the excerpts above."
_BLOCK_2 = "[Document: b.md]\nLe SLA est de 99,7 %.\n\nAnswer ONLY from the excerpts above."


async def test_kb_block_is_merged_into_the_model_request(monkeypatch):
    """The per-turn KB block rides the LAST user message of the model call
    (close to generation — system instructions dissolve over turn depth on
    small local models), with the real question kept last."""
    fake = _RecordingModel(messages=iter([AIMessage(content="90 jours.")]))
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=InMemorySaver())

    async for _ in runner.astream_text(
        llm=_Llm(),
        user_message="Quel est le préavis ?",
        system_prompt="sys",
        params=_PARAMS,
        thread_id="c-kb",
        kb_context_block=_BLOCK_1,
        kb_language_line="Réponds en français.",
    ):
        pass

    last_call = fake.received[-1]
    merged = last_call[-1]
    assert merged.type == "human"
    assert _BLOCK_1 in merged.text
    assert "Quel est le préavis ?" in merged.text
    # The user-voiced language request is the LAST thing before generation
    # (no English "Question:" label — structural English feeds the drift).
    assert "Question:" not in merged.text
    assert merged.text.strip().endswith("Réponds en français.")
    assert merged.text.find("Quel est le préavis ?") < merged.text.find("Réponds en français.")


async def test_kb_block_is_ephemeral_history_stays_clean(monkeypatch):
    """The merge happens in the model REQUEST only: the checkpointer keeps
    the clean question, so turn 2's history must show turn 1's question
    WITHOUT its excerpts (no context pollution, no parroting fuel)."""
    fake = _RecordingModel(messages=iter([AIMessage(content="r1"), AIMessage(content="r2")]))
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=InMemorySaver())

    async for _ in runner.astream_text(
        llm=_Llm(),
        user_message="q1",
        system_prompt="s",
        params=_PARAMS,
        thread_id="c-kb2",
        kb_context_block=_BLOCK_1,
    ):
        pass
    async for _ in runner.astream_text(
        llm=_Llm(),
        user_message="q2",
        system_prompt="s",
        params=_PARAMS,
        thread_id="c-kb2",
        kb_context_block=_BLOCK_2,
    ):
        pass

    second_call = fake.received[-1]
    history_humans = [m for m in second_call if m.type == "human"]
    # Turn 1's question is back to its clean form in the history…
    assert history_humans[0].text == "q1"
    assert _BLOCK_1 not in "".join(m.text for m in second_call)
    # …and only the current turn carries its own fresh block.
    assert _BLOCK_2 in history_humans[-1].text
    assert "q2" in history_humans[-1].text


async def test_no_kb_block_leaves_messages_untouched(monkeypatch):
    fake = _RecordingModel(messages=iter([AIMessage(content="hello")]))
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=InMemorySaver())

    async for _ in runner.astream_text(
        llm=_Llm(),
        user_message="hi",
        system_prompt="sys",
        params=_PARAMS,
        thread_id="c-plain",
    ):
        pass

    assert fake.received[-1][-1].text == "hi"


# ===================== Calculator tool in the agent loop (PR3) =====================


async def test_tool_call_round_trip_streams_only_final_text(monkeypatch):
    """Full agentic loop with the REAL calculator tool: the scripted model
    requests calculator(expression), the tool node executes it, and the
    model answers from the ToolMessage. The text/plain wire contract must
    only carry the FINAL answer (tool steps emit no text tokens).

    The calculator is passed explicitly (as ``plan_turn`` does on KB paths):
    since #129 there is no implicit default-tools fallback."""
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculator",
                "args": {"expression": "1240 + 1378 + 1456 + 1689"},
                "id": "call-1",
            }
        ],
    )
    fake = _RecordingModel(
        messages=iter([tool_call_msg, AIMessage(content="Le total est 5763 k€.")])
    )
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=InMemorySaver())

    out = [
        t
        async for t in runner.astream_text(
            llm=_Llm(),
            user_message="Additionne les quatre trimestres.",
            system_prompt="sys",
            params=_PARAMS,
            thread_id="c-calc",
            tools=[calculator],
        )
    ]

    assert "".join(out) == "Le total est 5763 k€."
    # The second model call must carry the REAL tool result (5763), proof
    # the calculator executed inside the loop.
    second_call = fake.received[-1]
    tool_messages = [m for m in second_call if m.type == "tool"]
    assert tool_messages and tool_messages[-1].text == "5763"


async def test_empty_final_answer_falls_back_to_last_tool_result(monkeypatch):
    """#90: the model calls the calculator (which returns 4074), then emits an
    EMPTY final answer (the Gemma pattern: successful tool call, then a
    ``finish_reason=stop`` message with no content). The runner must fall back
    to the LAST tool result so a correct answer is still delivered/persisted
    instead of yielding nothing (which would crash the empty-content guard)."""
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculator",
                "args": {"expression": "1240 + 1378 + 1456"},
                "id": "call-1",
            }
        ],
    )
    # Second model turn is EMPTY -> the fallback must kick in.
    fake = _RecordingModel(messages=iter([tool_call_msg, AIMessage(content="")]))
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=InMemorySaver())

    out = [
        t
        async for t in runner.astream_text(
            llm=_Llm(),
            user_message="1240 + 1378 + 1456 ?",
            system_prompt="sys",
            params=_PARAMS,
            thread_id="c-empty-final",
            tools=[calculator],
        )
    ]

    # The tool result (4074) is delivered as the answer, sober (no prefix/JSON).
    assert "".join(out).strip() == "4074"
    assert ERROR_SENTINEL not in "".join(out)


async def test_empty_final_answer_no_tool_yields_nothing(monkeypatch):
    """#90 boundary: an empty final answer with NO tool run this turn is a
    genuine failure — there is nothing to fall back to, so the runner must NOT
    fabricate content. Behavior is unchanged: the stream yields no text (the
    downstream empty-content guard still applies at persistence)."""
    fake = ToolableFakeChatModel(messages=iter([AIMessage(content="")]))
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=InMemorySaver())

    out = [
        t
        async for t in runner.astream_text(
            llm=_Llm(),
            user_message="hi",
            system_prompt="s",
            params=_PARAMS,
            thread_id="c-empty-notool",
            tools=[],
        )
    ]

    assert "".join(out) == ""
    assert ERROR_SENTINEL not in "".join(out)


async def test_non_empty_final_with_tool_does_not_append_tool_result(monkeypatch):
    """#90 guard: when the model DOES produce a real final answer after a tool
    call, the fallback must not fire — the raw tool result is never appended to
    a valid answer."""
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "calculator", "args": {"expression": "2 + 2"}, "id": "c1"}],
    )
    fake = _RecordingModel(messages=iter([tool_call_msg, AIMessage(content="The answer is four.")]))
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=InMemorySaver())

    out = "".join(
        [
            t
            async for t in runner.astream_text(
                llm=_Llm(),
                user_message="2 + 2 ?",
                system_prompt="s",
                params=_PARAMS,
                thread_id="c-nonempty",
                tools=[calculator],
            )
        ]
    )

    assert out == "The answer is four."
    # calculator("2 + 2") == "4"; if the fallback wrongly fired it would be
    # appended here. Its absence proves the fallback stayed dormant.
    assert "4" not in out


# ===================== Vision input (mlx-vlm swap, image content-parts) =====================

_IMG = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANS"}}


async def test_astream_accepts_multimodal_user_message(monkeypatch):
    """A list user_message (text + image_url parts) reaches a vision model as a
    HumanMessage whose content keeps the image part (supports_vision=True is
    required since #212: anything else strips images)."""
    fake = _RecordingModel(messages=iter([AIMessage(content="a red square")]))
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=InMemorySaver())

    async for _ in runner.astream_text(
        llm=_Llm(),
        user_message=[{"type": "text", "text": "what is this?"}, _IMG],
        system_prompt="sys",
        params=_PARAMS,
        thread_id="c-img",
        summarize=False,
        supports_vision=True,
    ):
        pass

    last = fake.received[-1][-1]
    assert last.type == "human"
    assert isinstance(last.content, list)
    assert any(p.get("type") == "image_url" for p in last.content)


async def test_kb_merge_preserves_image_parts(monkeypatch):
    """With a KB block AND an image, the merged last message carries the KB
    block in its text part and STILL keeps the image part."""
    fake = _RecordingModel(messages=iter([AIMessage(content="90 jours.")]))
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=InMemorySaver())

    async for _ in runner.astream_text(
        llm=_Llm(),
        user_message=[{"type": "text", "text": "Quel préavis ?"}, _IMG],
        system_prompt="sys",
        params=_PARAMS,
        thread_id="c-kb-img",
        kb_context_block=_BLOCK_1,
        kb_language_line="Réponds en français.",
        supports_vision=True,
    ):
        pass

    merged = fake.received[-1][-1]
    assert isinstance(merged.content, list)
    text_part = next(p for p in merged.content if p.get("type") == "text")
    assert _BLOCK_1 in text_part["text"]
    assert "Quel préavis ?" in text_part["text"]
    assert any(p.get("type") == "image_url" for p in merged.content)


async def test_latest_image_carried_forward_on_followup(monkeypatch):
    """Turn 1 sends an image; turn 2 is text-only. Turn 2's model call STILL
    carries turn 1's image (the most recent one), so a user can ask follow-ups
    about it without re-attaching — while context stays bounded to one turn's
    images (option B)."""
    fake = _RecordingModel(messages=iter([AIMessage(content="r1"), AIMessage(content="r2")]))
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=InMemorySaver())

    # supports_vision=True so ONLY the stale-image middleware is exercised
    # (anything else would add _StripImagesForTextModel too since #212).
    async for _ in runner.astream_text(
        llm=_Llm(),
        user_message=[{"type": "text", "text": "see this"}, _IMG],
        system_prompt="s",
        params=_PARAMS,
        thread_id="c-carry",
        summarize=True,
        supports_vision=True,
    ):
        pass
    async for _ in runner.astream_text(
        llm=_Llm(),
        user_message="and now?",
        system_prompt="s",
        params=_PARAMS,
        thread_id="c-carry",
        summarize=True,
        supports_vision=True,
    ):
        pass

    second_call = fake.received[-1]
    # Turn 1's image is carried into turn 2's request (kept, not flattened).
    past_human = [m for m in second_call if m.type == "human"][0]
    assert isinstance(past_human.content, list)
    assert any(p.get("type") == "image_url" for p in past_human.content)
    assert any(
        p.get("type") == "text" and "see this" in p.get("text", "") for p in past_human.content
    )


async def test_only_latest_image_kept_across_two_image_turns(monkeypatch):
    """Two image-bearing turns: only the MOST RECENT image survives; the older one
    collapses to an [image] marker, so context never accrues more than one turn's
    images (option B, bounded)."""
    fake = _RecordingModel(messages=iter([AIMessage(content="r1"), AIMessage(content="r2")]))
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=InMemorySaver())

    async for _ in runner.astream_text(
        llm=_Llm(),
        user_message=[{"type": "text", "text": "first pic"}, _IMG],
        system_prompt="s",
        params=_PARAMS,
        thread_id="c-two",
        summarize=True,
        supports_vision=True,
    ):
        pass
    async for _ in runner.astream_text(
        llm=_Llm(),
        user_message=[{"type": "text", "text": "second pic"}, _IMG],
        system_prompt="s",
        params=_PARAMS,
        thread_id="c-two",
        summarize=True,
        supports_vision=True,
    ):
        pass

    second_call = fake.received[-1]
    humans = [m for m in second_call if m.type == "human"]
    # Older image turn -> collapsed to an [image] marker (no image_url).
    first = humans[0]
    assert isinstance(first.content, str)
    assert "[image]" in first.content and "first pic" in first.content
    # Most recent image turn -> image kept.
    last = humans[-1]
    assert isinstance(last.content, list)
    assert any(p.get("type") == "image_url" for p in last.content)


async def test_images_stripped_for_non_vision_model(monkeypatch):
    """A text-only model (supports_vision=False) must never receive image parts:
    the CURRENT turn's image is flattened to an [image] marker so inference is
    clean text instead of broken/garbage output (#133)."""
    fake = _RecordingModel(messages=iter([AIMessage(content="ok")]))
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=None)

    async for _ in runner.astream_text(
        llm=_Llm(),
        user_message=[{"type": "text", "text": "what is this"}, _IMG],
        system_prompt="s",
        params=_PARAMS,
        supports_vision=False,
    ):
        pass

    sent = fake.received[-1]
    for m in sent:
        if isinstance(m.content, list):
            assert all(p.get("type") != "image_url" for p in m.content)
    human = [m for m in sent if m.type == "human"][0]
    assert isinstance(human.content, str)
    assert "[image]" in human.content
    assert "what is this" in human.content


async def test_images_kept_for_vision_model(monkeypatch):
    """A vision model (supports_vision=True) keeps the current image attached (#133)."""
    fake = _RecordingModel(messages=iter([AIMessage(content="ok")]))
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=None)

    async for _ in runner.astream_text(
        llm=_Llm(),
        user_message=[{"type": "text", "text": "what is this"}, _IMG],
        system_prompt="s",
        params=_PARAMS,
        supports_vision=True,
    ):
        pass

    last = fake.received[-1][-1]
    assert isinstance(last.content, list)
    assert any(p.get("type") == "image_url" for p in last.content)


async def test_images_stripped_when_vision_capability_unknown(monkeypatch):
    """Unknown vision capability (supports_vision=None) strips images too (#212):
    only a positively-detected vision model receives image parts, so a
    maybe-text-only model never breaks on an attachment."""
    fake = _RecordingModel(messages=iter([AIMessage(content="ok")]))
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=None)

    async for _ in runner.astream_text(
        llm=_Llm(),
        user_message=[{"type": "text", "text": "what is this"}, _IMG],
        system_prompt="s",
        params=_PARAMS,
        supports_vision=None,
    ):
        pass

    sent = fake.received[-1]
    for m in sent:
        if isinstance(m.content, list):
            assert all(p.get("type") != "image_url" for p in m.content)
    human = [m for m in sent if m.type == "human"][0]
    assert isinstance(human.content, str)
    assert "[image]" in human.content
    assert "what is this" in human.content


# ===================== Agentic KB tool (issue #84) =====================

from unittest.mock import MagicMock  # noqa: E402

from src.agents.tools import TurnToolContext, search_knowledge_base  # noqa: E402
from src.utils.kb_utils import KbExcerpt  # noqa: E402


def test_kb_tool_exposes_only_query_to_the_model():
    # The runtime context (kb_id, token_budget) must be hidden from the model;
    # only `query` is part of the tool schema the model sees.
    assert "query" in search_knowledge_base.args
    assert "runtime" not in search_knowledge_base.args


async def test_kb_tool_round_trip_searches_with_runtime_context(monkeypatch):
    """The model calls search_knowledge_base(query=...); the tool retrieves with
    the HIDDEN kb_id/token_budget from its runtime context, and its grounded
    result reaches the second model call as a ToolMessage."""
    excerpts = [KbExcerpt(source_file="contrat.pdf", text="Le préavis est de 90 jours.")]
    mock_retrieve = MagicMock(return_value=excerpts)
    monkeypatch.setattr("src.agents.tools.retrieve_kb_excerpts", mock_retrieve)

    tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_knowledge_base",
                "args": {"query": "préavis de résiliation"},
                "id": "k1",
            }
        ],
    )
    fake = _RecordingModel(messages=iter([tool_call, AIMessage(content="90 jours.")]))
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=InMemorySaver())

    out = [
        t
        async for t in runner.astream_text(
            llm=_Llm(),
            user_message="Quel est le préavis ?",
            system_prompt="sys",
            params=_PARAMS,
            thread_id="c-kbtool",
            tools=[search_knowledge_base],
            context=TurnToolContext(kb_id=7, kb_token_budget=1000),
        )
    ]

    assert "".join(out) == "90 jours."
    # query from the model + kb_id/budget from the hidden runtime context
    mock_retrieve.assert_called_once_with("préavis de résiliation", 7, 1000)
    tool_messages = [m for m in fake.received[-1] if m.type == "tool"]
    assert tool_messages
    assert "[Document: contrat.pdf]" in tool_messages[-1].text
    assert "90 jours" in tool_messages[-1].text


async def test_kb_tool_returns_not_found_message_on_empty_pool(monkeypatch):
    monkeypatch.setattr("src.agents.tools.retrieve_kb_excerpts", MagicMock(return_value=[]))
    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "search_knowledge_base", "args": {"query": "x"}, "id": "k2"}],
    )
    fake = _RecordingModel(
        messages=iter([tool_call, AIMessage(content="Ce n'est pas dans les documents.")])
    )
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=InMemorySaver())

    async for _ in runner.astream_text(
        llm=_Llm(),
        user_message="?",
        system_prompt="s",
        params=_PARAMS,
        thread_id="c-empty",
        tools=[search_knowledge_base],
        context=TurnToolContext(kb_id=7, kb_token_budget=1000),
    ):
        pass

    tool_messages = [m for m in fake.received[-1] if m.type == "tool"]
    assert "not in their documents" in tool_messages[-1].text


def test_build_middleware_includes_kb_tool_strip():
    built = AgentRunner()._build_middleware(ToolableFakeChatModel(messages=iter([])))
    assert any(type(m).__name__ == "_StripStaleToolResults" for m in built)


async def test_stale_kb_tool_results_placeholdered_on_followup(monkeypatch):
    """The checkpointer persists every KB ToolMessage; on a follow-up the model
    request must placeholder PAST turns' (bulky) excerpts to avoid multi-turn
    pollution, while keeping the CURRENT turn's result intact and the
    AIMessage(tool_calls) -> ToolMessage pairing valid."""
    ex1 = [KbExcerpt(source_file="d1.pdf", text="Le préavis est de 90 jours.")]
    ex2 = [KbExcerpt(source_file="d2.pdf", text="Le SLA est de 99,7 pourcent.")]
    monkeypatch.setattr("src.agents.tools.retrieve_kb_excerpts", MagicMock(side_effect=[ex1, ex2]))

    tc1 = AIMessage(
        content="",
        tool_calls=[{"name": "search_knowledge_base", "args": {"query": "préavis"}, "id": "a"}],
    )
    tc2 = AIMessage(
        content="",
        tool_calls=[{"name": "search_knowledge_base", "args": {"query": "sla"}, "id": "b"}],
    )
    fake = _RecordingModel(
        messages=iter([tc1, AIMessage(content="r1"), tc2, AIMessage(content="r2")])
    )
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=InMemorySaver())
    ctx = TurnToolContext(kb_id=7, kb_token_budget=1000)

    for q in ("q1", "q2"):
        async for _ in runner.astream_text(
            llm=_Llm(),
            user_message=q,
            system_prompt="s",
            params=_PARAMS,
            thread_id="c-kbstrip",
            summarize=True,
            tools=[search_knowledge_base],
            context=ctx,
        ):
            pass

    last_call = fake.received[-1]  # turn 2, post-tool model call
    tool_msgs = [m for m in last_call if m.type == "tool"]
    # Pairing preserved: both ToolMessages still present (none dropped).
    assert len(tool_msgs) == 2
    # Past turn's real excerpts are gone (placeholdered)…
    assert "90 jours" not in "".join(m.text for m in last_call)
    assert any("earlier turn omitted" in m.content for m in tool_msgs)
    # #304: the placeholder is DIRECTIVE - the model must be told to search
    # again rather than trust its memory of removed excerpts (observed live:
    # a 4B asserted "not in the documents" from a placeholdered turn while the
    # fact sat in the spec sheet).
    assert any(
        "call search_knowledge_base again" in m.content
        for m in tool_msgs
        if "earlier turn omitted" in m.content
    )
    # …and the current turn's KB result is intact.
    assert any("99,7" in m.text for m in tool_msgs)


# ===================== Structured event stream (issue #90) =====================


def _answers(events):
    return "".join(e["text"] for e in events if e["t"] == "answer")


def _thinking(events):
    return "".join(e["text"] for e in events if e["t"] == "thinking")


async def _events(runner, **kwargs):
    return [e async for e in runner.astream_text(emit_events=True, **kwargs)]


async def test_events_answer_only_stream(monkeypatch):
    """A plain answer surfaces as ``answer`` events only (no thinking/tool)."""
    fake = ToolableFakeChatModel(messages=iter([AIMessage(content="Python is awesome")]))
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=InMemorySaver())

    events = await _events(
        runner,
        llm=_Llm(),
        user_message="hi",
        system_prompt="s",
        params=_PARAMS,
        thread_id="e1",
        summarize=False,
    )

    assert _answers(events) == "Python is awesome"
    assert all(e["t"] == "answer" for e in events)


async def test_events_split_thinking_from_answer(monkeypatch):
    """Inline ``<think>...</think>`` is routed to ``thinking`` events; the answer
    text stays clean (no tag leakage)."""
    fake = ToolableFakeChatModel(
        messages=iter([AIMessage(content="<think>reasoning here</think>Answer text")])
    )
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=InMemorySaver())

    events = await _events(
        runner,
        llm=_Llm(),
        user_message="hi",
        system_prompt="s",
        params=_PARAMS,
        thread_id="e2",
        summarize=False,
    )

    assert _answers(events) == "Answer text"
    assert _thinking(events) == "reasoning here"
    assert "<think>" not in _answers(events) and "</think>" not in _answers(events)


async def test_str_mode_drops_thinking_keeps_answer(monkeypatch):
    """Default (str) mode -- used by arena -- yields ONLY answer text and strips
    inline thinking, preserving the plain-text wire."""
    fake = ToolableFakeChatModel(
        messages=iter([AIMessage(content="<think>reasoning here</think>Answer text")])
    )
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=InMemorySaver())

    out = [
        t
        async for t in runner.astream_text(
            llm=_Llm(),
            user_message="hi",
            system_prompt="s",
            params=_PARAMS,
            thread_id="e3",
            summarize=False,
        )
    ]

    assert "".join(out) == "Answer text"
    assert all(isinstance(t, str) for t in out)


async def test_events_tool_call_then_result_then_answer(monkeypatch):
    """A full agentic loop yields, in order: one complete ``tool_call`` (args as a
    parsed dict, never raw fragments), one ``tool_result``, then ``answer``."""
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculator",
                "args": {"expression": "1240 + 1378 + 1456 + 1689"},
                "id": "call-1",
            }
        ],
    )
    fake = _RecordingModel(
        messages=iter([tool_call_msg, AIMessage(content="Le total est 5763 k€.")])
    )
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=InMemorySaver())

    events = await _events(
        runner,
        llm=_Llm(),
        user_message="Additionne.",
        system_prompt="s",
        params=_PARAMS,
        thread_id="e4",
        tools=[calculator],
    )

    tool_calls = [e for e in events if e["t"] == "tool_call"]
    tool_results = [e for e in events if e["t"] == "tool_result"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "calculator"
    assert tool_calls[0]["args"] == {"expression": "1240 + 1378 + 1456 + 1689"}
    assert len(tool_results) == 1
    assert tool_results[0]["name"] == "calculator"
    assert tool_results[0]["text"] == "5763"
    assert _answers(events) == "Le total est 5763 k€."
    # Ordering: tool_call precedes its result, which precedes the final answer.
    kinds = [e["t"] for e in events]
    assert kinds.index("tool_call") < kinds.index("tool_result") < kinds.index("answer")


async def test_events_empty_final_fallback_arrives_as_answer(monkeypatch):
    """#90: the empty-final fallback (last tool result) is emitted as an
    ``answer`` event -- not a raw string, never an error."""
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculator",
                "args": {"expression": "1240 + 1378 + 1456"},
                "id": "call-1",
            }
        ],
    )
    fake = _RecordingModel(messages=iter([tool_call_msg, AIMessage(content="")]))
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=InMemorySaver())

    events = await _events(
        runner,
        llm=_Llm(),
        user_message="1240 + 1378 + 1456 ?",
        system_prompt="s",
        params=_PARAMS,
        thread_id="e5",
        tools=[calculator],
    )

    assert _answers(events).strip() == "4074"
    assert any(e["t"] == "tool_result" and e["text"] == "4074" for e in events)
    assert all(ERROR_SENTINEL not in e.get("text", "") for e in events)


async def test_events_construction_error_is_sentinel_answer(monkeypatch):
    """#252: a construction failure yields a single ``answer`` event carrying the
    curated sentinel (services later maps it to an ``error`` wire event)."""

    def _boom(llm, **kw):
        raise RuntimeError("model load failed: /secret/path")

    monkeypatch.setattr(runner_module, "build_chat_model", _boom)
    runner = AgentRunner(checkpointer=InMemorySaver())

    events = await _events(
        runner,
        llm=_Llm(),
        user_message="hi",
        system_prompt="s",
        params=_PARAMS,
        thread_id="e6",
    )

    assert len(events) == 1
    assert events[0]["t"] == "answer"
    assert ERROR_SENTINEL in events[0]["text"]
    assert "Traceback" not in events[0]["text"]
    assert "/secret/path" not in events[0]["text"]


# ===== Pre-tool narration reclassified as thinking (#297) =====
#
# On tool-carrying turns, text the model streams BEFORE its tool call is
# hallucinated guessing, not the answer. The runner buffers the hop's answer
# text and re-emits it as ``thinking`` events the moment the hop's first
# tool_call_chunk arrives (before the ``tool_call`` event); a hop that ends
# without a tool call flushes its buffer as ``answer`` at stream end.


_NARRATION_CALL = AIMessage(
    content="I believe the answer is around one thousand, let me verify.",
    tool_calls=[{"name": "calculator", "args": {"expression": "4 + 5"}, "id": "n1"}],
)


async def test_narration_before_tool_call_becomes_thinking(monkeypatch):
    """(a) narration + tool_call + tool_result + final answer: the narration
    surfaces as ``thinking`` events emitted BEFORE the ``tool_call`` event;
    only the final hop's text is ``answer``."""
    fake = ToolableFakeChatModel(
        messages=iter([_NARRATION_CALL, AIMessage(content="The result is 9.")])
    )
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=InMemorySaver())

    events = await _events(
        runner,
        llm=_Llm(),
        user_message="4 + 5 ?",
        system_prompt="s",
        params=_PARAMS,
        thread_id="n-a",
        tools=[calculator],
    )

    assert _answers(events) == "The result is 9."
    assert "let me verify" in _thinking(events)
    # Narration must never leak into answer events.
    assert "one thousand" not in _answers(events)
    # Order: every thinking event precedes the tool_call, which precedes the
    # tool_result, which precedes all answer events.
    kinds = [e["t"] for e in events]
    idx_call = kinds.index("tool_call")
    idx_result = kinds.index("tool_result")
    assert all(i < idx_call for i, k in enumerate(kinds) if k == "thinking")
    assert idx_call < idx_result
    assert all(i > idx_result for i, k in enumerate(kinds) if k == "answer")


async def test_tools_bound_no_tool_call_flushes_buffer_as_answer(monkeypatch):
    """(b) tools bound but the model answers directly: the buffered text is
    flushed as ``answer`` events with the exact content, nothing as thinking."""
    fake = ToolableFakeChatModel(
        messages=iter([AIMessage(content="Direct answer, no tool needed.")])
    )
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=InMemorySaver())

    events = await _events(
        runner,
        llm=_Llm(),
        user_message="hi",
        system_prompt="s",
        params=_PARAMS,
        thread_id="n-b",
        tools=[calculator],
    )

    assert _answers(events) == "Direct answer, no tool needed."
    assert _thinking(events) == ""
    assert all(e["t"] == "answer" for e in events)


async def test_narration_does_not_defeat_empty_final_fallback(monkeypatch):
    """(c) narration + tool result + EMPTY final answer: the narration went out
    as thinking, so it must NOT count as emitted answer text -- the #90
    fallback still delivers the tool result as the answer."""
    narrating_call = AIMessage(
        content="Maybe around one thousand kilograms.",
        tool_calls=[
            {
                "name": "calculator",
                "args": {"expression": "1240 + 1378 + 1456"},
                "id": "n2",
            }
        ],
    )
    fake = ToolableFakeChatModel(messages=iter([narrating_call, AIMessage(content="")]))
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=InMemorySaver())

    events = await _events(
        runner,
        llm=_Llm(),
        user_message="total ?",
        system_prompt="s",
        params=_PARAMS,
        thread_id="n-c",
        tools=[calculator],
    )

    assert _answers(events).strip() == "4074"
    assert "one thousand" in _thinking(events)
    assert all(ERROR_SENTINEL not in e.get("text", "") for e in events)


async def test_think_tags_inside_narration_no_double_wrapping(monkeypatch):
    """(d) inline ``<think>`` inside a narrating hop: splitter thinking flows
    through unchanged, the remaining narration is reclassified as thinking too,
    and no tag ever leaks into any event."""
    narrating_call = AIMessage(
        content="<think>internal reasoning</think>guessing before the call",
        tool_calls=[{"name": "calculator", "args": {"expression": "4 + 5"}, "id": "n3"}],
    )
    fake = ToolableFakeChatModel(messages=iter([narrating_call, AIMessage(content="Nine.")]))
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=InMemorySaver())

    events = await _events(
        runner,
        llm=_Llm(),
        user_message="4 + 5 ?",
        system_prompt="s",
        params=_PARAMS,
        thread_id="n-d",
        tools=[calculator],
    )

    thinking = _thinking(events)
    assert "internal reasoning" in thinking
    assert "guessing before the call" in thinking
    assert _answers(events) == "Nine."
    for e in events:
        assert "<think>" not in e.get("text", "")
        assert "</think>" not in e.get("text", "")


async def test_zero_tool_turn_event_stream_unchanged(monkeypatch):
    """(e) regression guard: a zero-tool turn streams answer events immediately,
    token by token, exactly as before -- no buffering side effects."""
    fake = ToolableFakeChatModel(messages=iter([AIMessage(content="One two three")]))
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=InMemorySaver())

    events = await _events(
        runner,
        llm=_Llm(),
        user_message="hi",
        system_prompt="s",
        params=_PARAMS,
        thread_id="n-e",
        tools=[],
    )

    assert events == [
        {"t": "answer", "text": "One"},
        {"t": "answer", "text": " "},
        {"t": "answer", "text": "two"},
        {"t": "answer", "text": " "},
        {"t": "answer", "text": "three"},
    ]


async def test_multi_hop_narration_reclassified_each_hop(monkeypatch):
    """(f) two tool rounds then a final answer: each hop's narration comes out
    as thinking (before that hop's tool_call), the final answer is intact."""
    call_1 = AIMessage(
        content="First guess, checking.",
        tool_calls=[{"name": "calculator", "args": {"expression": "1 + 1"}, "id": "m1"}],
    )
    call_2 = AIMessage(
        content="Second guess, checking again.",
        tool_calls=[{"name": "calculator", "args": {"expression": "2 + 2"}, "id": "m2"}],
    )
    fake = ToolableFakeChatModel(
        messages=iter([call_1, call_2, AIMessage(content="Final answer.")])
    )
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=InMemorySaver())

    events = await _events(
        runner,
        llm=_Llm(),
        user_message="sum things",
        system_prompt="s",
        params=_PARAMS,
        thread_id="n-f",
        tools=[calculator],
    )

    assert _answers(events) == "Final answer."
    assert "First guess" in _thinking(events)
    assert "Second guess" in _thinking(events)
    kinds = [e["t"] for e in events]
    assert kinds.count("tool_call") == 2
    assert kinds.count("tool_result") == 2
    # Hop 1's narration precedes tool_call #1; hop 2's narration sits between
    # tool_result #1 and tool_call #2; answers come after tool_result #2.
    idx_call_1 = kinds.index("tool_call")
    idx_result_1 = kinds.index("tool_result")
    idx_call_2 = kinds.index("tool_call", idx_call_1 + 1)
    idx_result_2 = kinds.index("tool_result", idx_result_1 + 1)
    thinking_idx = [i for i, k in enumerate(kinds) if k == "thinking"]
    hop1_thinking = [i for i in thinking_idx if i < idx_call_1]
    hop2_thinking = [i for i in thinking_idx if idx_result_1 < i < idx_call_2]
    assert hop1_thinking and hop2_thinking
    assert len(hop1_thinking) + len(hop2_thinking) == len(thinking_idx)
    assert all(i > idx_result_2 for i, k in enumerate(kinds) if k == "answer")


async def test_str_mode_drops_narration_keeps_final_answer(monkeypatch):
    """Arena projection (emit_events=False): the reclassified narration rides
    thinking events, which the projection drops -- the hallucinated guessing
    disappears from arena answers while the final answer is unchanged."""
    fake = ToolableFakeChatModel(
        messages=iter([_NARRATION_CALL, AIMessage(content="The result is 9.")])
    )
    _patch_model(monkeypatch, fake)
    runner = AgentRunner(checkpointer=None)

    out = [
        t
        async for t in runner.astream_text(
            llm=_Llm(),
            user_message="4 + 5 ?",
            system_prompt="s",
            params=_PARAMS,
            thread_id=None,
            tools=[calculator],
        )
    ]

    assert "".join(out) == "The result is 9."
    assert all("one thousand" not in t for t in out)


# ===== One-shot stream (#266) — thinking must never reach the title =====


class _OneShotStreamModel:
    """astream-only stub replaying EXACT scripted chunks.

    ``ToolableFakeChatModel`` re-tokenizes content on whitespace, which cannot
    pin a ``<think>`` tag split across chunk boundaries; this stub yields each
    scripted delta verbatim as a chunk object exposing ``.text``.
    """

    def __init__(self, chunks):
        self._chunks = chunks

    async def astream(self, messages):
        for text in self._chunks:
            yield SimpleNamespace(text=text)


async def _collect_oneshot(monkeypatch, chunks):
    monkeypatch.setattr(
        runner_module, "build_chat_model", lambda llm, **kw: _OneShotStreamModel(chunks)
    )
    runner = AgentRunner(checkpointer=None)
    out = [
        t
        async for t in runner.astream_oneshot(
            llm=_Llm(), prompt_text="Title this", temperature=0.5, top_p=0.9, max_tokens=12
        )
    ]
    return "".join(out)


async def test_oneshot_strips_think_block_spanning_chunks(monkeypatch):
    # #266: inline reasoning is stripped; only the answer (the title) streams out.
    collected = await _collect_oneshot(
        monkeypatch, ["<think>reasoning", " more</think>", "Nice Title"]
    )
    assert collected == "Nice Title"


async def test_oneshot_unclosed_think_collects_to_empty(monkeypatch):
    # #266: a thinking model that burns the whole 12-token budget inside an
    # unclosed <think> yields NOTHING (caller falls back to the default title)
    # instead of the literal tag.
    collected = await _collect_oneshot(monkeypatch, ["<think>budget burned entirely"])
    assert collected == ""


async def test_oneshot_strips_tag_split_across_chunk_boundary(monkeypatch):
    collected = await _collect_oneshot(monkeypatch, ["<th", "ink>hidden</think>Real"])
    assert collected == "Real"


async def test_oneshot_without_think_tags_passes_through(monkeypatch):
    collected = await _collect_oneshot(monkeypatch, ["A Plain", " Title"])
    assert collected == "A Plain Title"


async def test_oneshot_requests_thinking_suppression(monkeypatch):
    # #266: one-shot calls ask the engine to disable thinking at the chat
    # template level; the splitter is only the safety net.
    captured = {}

    def _capture(llm, **kw):
        captured.update(kw)
        return _OneShotStreamModel(["T"])

    monkeypatch.setattr(runner_module, "build_chat_model", _capture)
    runner = AgentRunner(checkpointer=None)
    async for _ in runner.astream_oneshot(
        llm=_Llm(), prompt_text="p", temperature=0.5, top_p=0.9, max_tokens=12
    ):
        pass
    assert captured.get("disable_thinking") is True


# ---------------------------------------------------------------------------
# #388: per-model sampling on the ChatOpenAI path. The zero-diff guard comes
# first: a row without hints must produce the EXACT request body the #129 eval
# campaign validated (kwargs + extra_body byte-identical to today's).
# ---------------------------------------------------------------------------
from src.database.generation_hints import resolve_sampling_defaults  # noqa: E402


class _HintedLlm(_Llm):
    def __init__(self, **generation_config):
        self.generation_hints = {
            "base_repo": "Qwen/Qwen3-0.6B",
            "generation_config": generation_config,
            "supports_thinking": True,
            "context_length": 40960,
            "captured_at": "d",
        }


def test_build_chat_model_zero_diff_for_a_row_without_hints(monkeypatch):
    monkeypatch.setattr(config, "LLM_Engine", _IdentityEngine)
    today = build_chat_model(_Llm(), temperature=0.3, top_p=0.8, max_tokens=55)
    resolved = build_chat_model(
        _Llm(),
        temperature=0.3,
        top_p=0.8,
        max_tokens=55,
        sampling=resolve_sampling_defaults(_Llm()),
    )
    assert (
        resolved.extra_body
        == today.extra_body
        == {"repetition_penalty": 1.1, "repetition_context_size": 64}
    )
    assert (
        (resolved.temperature, resolved.top_p, resolved.max_tokens)
        == (today.temperature, today.top_p, today.max_tokens)
        == (0.3, 0.8, 55)
    )
    assert resolved.model_kwargs == today.model_kwargs


def test_build_chat_model_sends_optional_keys_only_when_the_profile_defines_them(monkeypatch):
    monkeypatch.setattr(config, "LLM_Engine", _IdentityEngine)
    llm = _HintedLlm(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
    chat = build_chat_model(
        llm, temperature=0.6, top_p=0.95, max_tokens=55, sampling=resolve_sampling_defaults(llm)
    )
    assert chat.extra_body == {
        "repetition_penalty": 1.1,
        "repetition_context_size": 64,
        "top_k": 20,
        "min_p": 0.0,
    }

    # presence_penalty / repetition_penalty from the profile; no top_k -> absent.
    llm = _HintedLlm(temperature=0.7, presence_penalty=1.5, repetition_penalty=1.05)
    chat = build_chat_model(
        llm, temperature=0.7, top_p=0.95, max_tokens=55, sampling=resolve_sampling_defaults(llm)
    )
    assert chat.extra_body == {
        "repetition_penalty": 1.05,
        "repetition_context_size": 64,
        "presence_penalty": 1.5,
    }


def test_build_chat_model_logs_the_resolved_extra_body(monkeypatch, caplog):
    """QA could not verify from the INFO log that a profile's top_k reached the
    server: the "ChatOpenAI built" line must list every extra_body key as sent
    (post-translation, so the wire names appear), ASCII, on the one line."""
    import logging

    monkeypatch.setattr(config, "LLM_Engine", _IdentityEngine)
    llm = _HintedLlm(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
    with caplog.at_level(logging.INFO, logger="erudi"):
        build_chat_model(
            llm, temperature=0.6, top_p=0.95, max_tokens=55, sampling=resolve_sampling_defaults(llm)
        )
    (line,) = [r.message for r in caplog.records if r.message.startswith("ChatOpenAI built:")]
    assert "temperature=0.6" in line and "max_tokens=55" in line
    assert "extra_body=" in line
    for key in ("repetition_penalty=1.1", "repetition_context_size=64", "top_k=20", "min_p=0.0"):
        assert key in line
    assert "\n" not in line and line.isascii()


def test_build_chat_model_logs_translated_wire_names(monkeypatch, caplog):
    import logging

    from src.engines.base_llama_cpp_engine import BaseLlamaCppEngine

    class _LlamaEngine:
        @staticmethod
        def get_model_and_tokenizer(llm_id, link):
            return ({"base_url": "http://127.0.0.1:9090", "alias": f"erudi-{llm_id}"}, {})

        @staticmethod
        def _payload_model_value(handle):
            return handle["alias"]

        _translate_payload_kwargs = BaseLlamaCppEngine._translate_payload_kwargs

    monkeypatch.setattr(config, "LLM_Engine", _LlamaEngine)
    with caplog.at_level(logging.INFO, logger="erudi"):
        build_chat_model(_Llm(), temperature=0.3, top_p=0.8, max_tokens=55)
    (line,) = [r.message for r in caplog.records if r.message.startswith("ChatOpenAI built:")]
    assert "repeat_penalty=1.1" in line and "repeat_last_n=64" in line


def test_build_chat_model_llama_cpp_leaves_optional_names_untouched(monkeypatch):
    # top_k / min_p / presence_penalty ARE llama-server's own names: the
    # translation renames only the repetition controls.
    from src.engines.base_llama_cpp_engine import BaseLlamaCppEngine

    class _LlamaEngine:
        @staticmethod
        def get_model_and_tokenizer(llm_id, link):
            return ({"base_url": "http://127.0.0.1:9090", "alias": f"erudi-{llm_id}"}, {})

        @staticmethod
        def _payload_model_value(handle):
            return handle["alias"]

        _translate_payload_kwargs = BaseLlamaCppEngine._translate_payload_kwargs

    monkeypatch.setattr(config, "LLM_Engine", _LlamaEngine)
    llm = _HintedLlm(temperature=0.6, top_k=20, min_p=0.0, presence_penalty=1.5)
    chat = build_chat_model(
        llm, temperature=0.6, top_p=0.95, max_tokens=55, sampling=resolve_sampling_defaults(llm)
    )
    assert chat.extra_body == {
        "repeat_penalty": 1.1,
        "repeat_last_n": 64,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
    }


async def test_runner_passes_the_resolved_sampling_to_the_factory(monkeypatch):
    captured = {}

    def _capture(llm, **kw):
        captured.update(kw)
        return ToolableFakeChatModel(messages=iter([AIMessage(content="ok")]))

    monkeypatch.setattr(runner_module, "build_chat_model", _capture)
    llm = _HintedLlm(temperature=0.6, top_k=20)
    runner = AgentRunner(checkpointer=None)
    async for _ in runner.astream_text(
        llm=llm, user_message="q", system_prompt="s", params=_PARAMS
    ):
        pass
    assert captured["sampling"].top_k == 20
    assert captured["sampling"].source == "base_generation_config"
    # User-facing three still come from GenParams (the conversation row / slider).
    assert (captured["temperature"], captured["top_p"], captured["max_tokens"]) == (0.5, 0.9, 64)


async def test_oneshot_passes_the_resolved_sampling_too(monkeypatch):
    captured = {}

    def _capture(llm, **kw):
        captured.update(kw)
        return _OneShotStreamModel(["T"])

    monkeypatch.setattr(runner_module, "build_chat_model", _capture)
    runner = AgentRunner(checkpointer=None)
    async for _ in runner.astream_oneshot(
        llm=_Llm(), prompt_text="p", temperature=0.5, top_p=0.9, max_tokens=12
    ):
        pass
    assert captured["sampling"].source == "none"


# ---------------------------------------------------------------------------
# Per-request seed on the MLX path: mlx_vlm.server replays DEFAULT_SEED when a
# request carries no ``seed``, so "creativity" had no run-to-run effect on Apple
# Silicon. The factory must let the engine translation stamp a fresh seed into
# extra_body and the "ChatOpenAI built" line must show it.
# ---------------------------------------------------------------------------
from src.engines.mlx_engine import MLX_Engine  # noqa: E402


class _MlxLikeEngine(_IdentityEngine):
    _translate_payload_kwargs = MLX_Engine._translate_payload_kwargs


def test_build_chat_model_mlx_extra_body_carries_a_fresh_seed(monkeypatch):
    monkeypatch.setattr(config, "LLM_Engine", _MlxLikeEngine)
    chats = [build_chat_model(_Llm(), temperature=0.6, top_p=0.95, max_tokens=55) for _ in range(8)]
    assert all(isinstance(chat.extra_body["seed"], int) for chat in chats)
    assert len({chat.extra_body["seed"] for chat in chats}) > 1
    # The repetition controls still ride next to it, untouched.
    assert {k: v for k, v in chats[0].extra_body.items() if k != "seed"} == {
        "repetition_penalty": 1.1,
        "repetition_context_size": 64,
    }


def test_build_chat_model_oneshot_on_mlx_also_gets_a_seed(monkeypatch):
    # Titles run at temperature 1.0 with enable_thinking=False; a random seed
    # there is fine and the body shape stays identical apart from the seed.
    monkeypatch.setattr(config, "LLM_Engine", _MlxLikeEngine)
    chat = build_chat_model(
        _Llm(), temperature=1.0, top_p=0.95, max_tokens=12, disable_thinking=True
    )
    assert chat.extra_body["enable_thinking"] is False
    assert isinstance(chat.extra_body["seed"], int)


def test_build_chat_model_logs_the_seed_on_mlx(monkeypatch, caplog):
    monkeypatch.setattr(config, "LLM_Engine", _MlxLikeEngine)
    with caplog.at_level(logging.INFO, logger="erudi"):
        chat = build_chat_model(_Llm(), temperature=0.6, top_p=0.95, max_tokens=55)
    (line,) = [r.message for r in caplog.records if r.message.startswith("ChatOpenAI built:")]
    assert f"seed={chat.extra_body['seed']}" in line


def test_build_chat_model_llama_cpp_extra_body_has_no_seed(monkeypatch):
    from src.engines.base_llama_cpp_engine import BaseLlamaCppEngine

    class _LlamaEngine(_IdentityEngine):
        _translate_payload_kwargs = BaseLlamaCppEngine._translate_payload_kwargs

    monkeypatch.setattr(config, "LLM_Engine", _LlamaEngine)
    chat = build_chat_model(_Llm(), temperature=0.6, top_p=0.95, max_tokens=55)
    assert "seed" not in chat.extra_body
