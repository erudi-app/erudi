"""One INFO line per model call, with its prompt size (#504 measurement)."""

import logging

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_openai import ChatOpenAI

from src.agents.chat_model import erudi_chat_openai_class

pytestmark = pytest.mark.unit


def _client(monkeypatch, **kwargs):
    async def _server(self, messages, *args, **kw):
        for raw in (
            {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {"prompt_tokens": 120, "completion_tokens": 3, "total_tokens": 123},
            },
        ):
            yield self._convert_chunk_to_generation_chunk(raw, AIMessageChunk, {})

    monkeypatch.setattr(ChatOpenAI, "_astream", _server)
    return erudi_chat_openai_class()(
        base_url="http://127.0.0.1:1/v1", api_key="k", model="m", **kwargs
    )


def _lines(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("Model call")]


async def test_each_call_logs_one_line_with_its_hop_and_sizes(monkeypatch, caplog):
    client = _client(monkeypatch, call_log_label="thread_id=abc")

    with caplog.at_level(logging.INFO, logger="erudi"):
        for _ in range(2):
            _ = [c async for c in client._astream([HumanMessage("secret question")])]

    assert _lines(caplog) == [
        "Model call: thread_id=abc hop=1 prompt_tokens=120 completion_tokens=3",
        "Model call: thread_id=abc hop=2 prompt_tokens=120 completion_tokens=3",
    ]
    assert all(r.levelno == logging.INFO for r in caplog.records if "Model call" in r.getMessage())
    assert "secret question" not in caplog.text


async def test_the_hops_of_a_real_tool_loop_count_on_one_client(monkeypatch, caplog):
    from langchain.agents import create_agent
    from langchain_core.tools import tool

    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    calls = []

    async def _server(self, messages, *args, **kw):
        calls.append(len(messages))
        if len(calls) == 1:
            delta = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
                    }
                ]
            }
            finish, usage = "tool_calls", {"prompt_tokens": 100, "completion_tokens": 9}
        else:
            delta, finish = {"content": "3"}, "stop"
            usage = {"prompt_tokens": 140, "completion_tokens": 1}
        yield self._convert_chunk_to_generation_chunk(
            {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]},
            AIMessageChunk,
            {},
        )
        yield self._convert_chunk_to_generation_chunk(
            {"choices": [], "usage": {**usage, "total_tokens": 0}}, AIMessageChunk, {}
        )

    monkeypatch.setattr(ChatOpenAI, "_astream", _server)
    client = _client(monkeypatch, call_log_label="thread_id=loop")
    monkeypatch.setattr(ChatOpenAI, "_astream", _server)  # _client installed its own
    agent = create_agent(model=client, tools=[add])

    with caplog.at_level(logging.INFO, logger="erudi"):
        _ = [
            e
            async for e in agent.astream(
                {"messages": [HumanMessage("1+2?")]}, stream_mode="messages"
            )
        ]

    assert _lines(caplog) == [
        "Model call: thread_id=loop hop=1 prompt_tokens=100 completion_tokens=9",
        "Model call: thread_id=loop hop=2 prompt_tokens=140 completion_tokens=1",
    ]


# ===== A call that ends without usage still leaves its prompt size =====


def _failing_client(monkeypatch, *, after_text=False, **kwargs):
    async def _server(self, messages, *args, **kw):
        if after_text:
            yield self._convert_chunk_to_generation_chunk(
                {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]},
                AIMessageChunk,
                {},
            )
        raise RuntimeError("stream ended")

    monkeypatch.setattr(ChatOpenAI, "_astream", _server)
    return erudi_chat_openai_class()(
        base_url="http://127.0.0.1:1/v1", api_key="k", model="m", **kwargs
    )


@pytest.mark.parametrize("after_text", [False, True])
async def test_a_call_that_fails_before_its_usage_writes_one_estimated_line(
    monkeypatch, caplog, after_text
):
    client = _failing_client(monkeypatch, after_text=after_text, call_log_label="thread_id=abc")

    with caplog.at_level(logging.INFO, logger="erudi"):
        with pytest.raises(RuntimeError):
            _ = [c async for c in client._astream([HumanMessage("x" * 400)])]

    (line,) = _lines(caplog)
    assert line.startswith("Model call: thread_id=abc hop=1 failed est_prompt_tokens=")
    fields = dict(part.split("=", 1) for part in line.split() if "=" in part)
    assert int(fields["est_prompt_tokens"]) >= 100  # 400 chars at about 4 per token
    assert int(fields["est_budget_tokens"]) >= 100
    assert line.isascii()
    assert "x" * 20 not in caplog.text


async def test_the_failed_line_carries_the_estimate_at_the_loaded_tokenizers_digit_weight(
    monkeypatch, caplog
):
    """chars/4 reads 400 digits as about 100 tokens; a tokenizer that splits
    them one by one spends 400. The line carries both."""
    client = _failing_client(monkeypatch, call_log_label="thread_id=abc", digit_tokens=1.0)

    with caplog.at_level(logging.INFO, logger="erudi"):
        with pytest.raises(RuntimeError):
            _ = [c async for c in client._astream([HumanMessage("7" * 400)])]

    (line,) = _lines(caplog)
    fields = dict(part.split("=", 1) for part in line.split() if "=" in part)
    assert int(fields["est_prompt_tokens"]) < 150
    assert int(fields["est_budget_tokens"]) >= 400


# ===== An engine overflow carries the new message's share of the request =====


class _Overflow(Exception):
    def __init__(self):
        super().__init__("Error code: 400")
        self.body = {"type": "exceed_context_size_error", "n_prompt_tokens": 9000, "n_ctx": 8192}


def _rejecting_client(monkeypatch, exc, **kwargs):
    async def _server(self, messages, *args, **kw):
        raise exc
        yield  # pragma: no cover  (makes this an async generator function)

    monkeypatch.setattr(ChatOpenAI, "_astream", _server)
    return erudi_chat_openai_class()(
        base_url="http://127.0.0.1:1/v1", api_key="k", model="m", **kwargs
    )


async def test_an_overflow_carries_the_share_the_request_keeps_without_its_history(monkeypatch):
    from langchain_core.messages import AIMessage, SystemMessage

    from src.agents.overflow import parse_context_overflow

    exc = _Overflow()
    client = _rejecting_client(monkeypatch, exc)
    messages = [
        SystemMessage("s" * 400),
        HumanMessage("h" * 4_000),
        AIMessage("a" * 4_000),
        HumanMessage("q" * 11_600),
    ]

    with pytest.raises(_Overflow):
        _ = [c async for c in client._astream(messages)]

    overflow = parse_context_overflow(exc)
    # The system prompt and the last question: 12 000 of 20 000 characters.
    assert overflow.alone_share == pytest.approx(0.6, abs=0.01)


async def test_an_error_that_is_not_an_overflow_carries_no_share(monkeypatch):
    from src.agents.overflow import ALONE_SHARE_ATTR

    exc = RuntimeError("boom")
    client = _rejecting_client(monkeypatch, exc)

    with pytest.raises(RuntimeError):
        _ = [c async for c in client._astream([HumanMessage("q")])]

    assert not hasattr(exc, ALONE_SHARE_ATTR)


async def test_a_retried_preflight_rejection_logs_only_the_successful_call(monkeypatch, caplog):
    window = 32768
    attempts = []

    async def _server(self, messages, *args, **kw):
        attempts.append(kw.get("max_tokens"))
        if len(attempts) == 1:
            raise Exception(
                f"Request needs {30000 + kw['max_tokens']} context tokens "
                f"(30000 prompt + {kw['max_tokens']} max generation), but MAX_KV_SIZE is {window}."
            )
        yield self._convert_chunk_to_generation_chunk(
            {"choices": [], "usage": {"prompt_tokens": 30000, "completion_tokens": 2}},
            AIMessageChunk,
            {},
        )

    monkeypatch.setattr(ChatOpenAI, "_astream", _server)
    client = erudi_chat_openai_class()(
        base_url="http://127.0.0.1:1/v1",
        api_key="k",
        model="m",
        max_tokens=4000,
        effective_context_tokens=window,
        call_log_label="thread_id=abc",
    )

    with caplog.at_level(logging.INFO, logger="erudi"):
        _ = [c async for c in client._astream([HumanMessage("hello")])]

    assert len(attempts) == 2
    assert _lines(caplog) == [
        "Model call: thread_id=abc hop=1 prompt_tokens=30000 completion_tokens=2"
    ]


# ===== The label the runner binds =====


async def _label_of(monkeypatch, thread_id):
    from langgraph.checkpoint.memory import InMemorySaver

    from src.agents import runner as runner_module
    from src.agents.runner import AgentRunner, GenParams
    from src.core import config
    from src.engines.base_engine import BaseEngine
    from tests._helpers import ToolableFakeChatModel

    class _Llm:
        id = 7
        link = "/fake/path"
        name = "Test 7B"
        param_size = 7.0

    class _Raises(ToolableFakeChatModel):
        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
            raise RuntimeError("stop here")
            yield  # pragma: no cover

    captured = {}

    def _build(llm, **kw):
        captured.update(kw)
        return _Raises(messages=iter([]))

    monkeypatch.setattr(config, "LLM_Engine", BaseEngine)
    monkeypatch.setattr(runner_module, "build_chat_model", _build)
    runner = AgentRunner(checkpointer=InMemorySaver())
    _ = [
        e
        async for e in runner.astream_text(
            llm=_Llm(),
            user_message="hi",
            system_prompt="s",
            params=GenParams(temperature=0.5, top_p=0.9, max_tokens=64),
            thread_id=thread_id,
        )
    ]
    return captured["call_log_label"]


async def test_a_conversation_turn_is_labelled_with_its_thread(monkeypatch):
    assert await _label_of(monkeypatch, "conv-1") == "thread_id=conv-1"


async def test_an_arena_panel_is_labelled_with_its_model(monkeypatch):
    assert await _label_of(monkeypatch, None) == "thread_id=arena:7"


async def test_a_client_without_a_label_logs_nothing(monkeypatch, caplog):
    client = _client(monkeypatch)

    with caplog.at_level(logging.INFO, logger="erudi"):
        _ = [c async for c in client._astream([HumanMessage("hello")])]

    assert _lines(caplog) == []
