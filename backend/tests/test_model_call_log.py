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


async def test_a_client_without_a_label_logs_nothing(monkeypatch, caplog):
    client = _client(monkeypatch)

    with caplog.at_level(logging.INFO, logger="erudi"):
        _ = [c async for c in client._astream([HumanMessage("hello")])]

    assert _lines(caplog) == []
