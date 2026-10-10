"""Server usage and the request stamp: what a measured ratio is computed from.

Every client asks for usage (``stream_options.include_usage``). Both local
servers answer with a last chunk that carries ``usage`` and no choices
(mlx_vlm 0.6.17 ``_chat_usage_chunk``; llama-server the same OpenAI shape).
``Erudi_Chat_OpenAI._astream`` stamps THAT chunk -- and only that one -- with
the estimate of the request it sent, so the aggregated AI message carries both
sides of the ratio. These tests feed real server-shaped dicts through the
client's own conversion hook.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from src.agents.chat_model import erudi_chat_openai_class
from src.agents.token_accounting import (
    REQUEST_EST_KEY,
    REQUEST_FIRST_HOP_KEY,
    REQUEST_HAS_IMAGES_KEY,
    hop_ratio,
    request_tokens_est,
)

pytestmark = pytest.mark.unit


def _client(**kwargs):
    return erudi_chat_openai_class()(
        base_url="http://127.0.0.1:1/v1",
        api_key="not-needed",
        model="fake-model",
        streaming=True,
        **kwargs,
    )


def _content(text, finish_reason=None):
    return {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "model": "m",
        "choices": [{"index": 0, "finish_reason": finish_reason, "delta": {"content": text}}],
    }


def _usage(prompt, completion, cached=0):
    """mlx_vlm 0.6.17's usage-only last chunk (``choices: []``)."""
    return {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "model": "m",
        "choices": [],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    }


def _patch_server(monkeypatch, raws, seen=None):
    from langchain_openai import ChatOpenAI

    async def _server(self, messages, *args, **kwargs):
        if seen is not None:
            seen.append(kwargs)
        for raw in raws:
            chunk = self._convert_chunk_to_generation_chunk(raw, AIMessageChunk, {})
            if chunk is not None:
                yield chunk

    monkeypatch.setattr(ChatOpenAI, "_astream", _server)


_ANSWER = [_content("Hello"), _content(" there", finish_reason="stop"), _usage(1234, 2, 100)]


def test_the_usage_only_chunk_has_no_reasoning_and_no_text():
    chunk = _client()._convert_chunk_to_generation_chunk(_usage(10, 2), AIMessageChunk, {})
    assert chunk.message.content == ""
    assert "reasoning_content" not in chunk.message.additional_kwargs
    assert chunk.message.usage_metadata["input_tokens"] == 10
    assert chunk.message.usage_metadata["input_token_details"]["cache_read"] == 0


async def test_the_usage_chunk_passes_the_watchdog_and_carries_the_stamp(monkeypatch):
    _patch_server(monkeypatch, _ANSWER)
    messages = [HumanMessage("hi there")]

    chunks = [c async for c in _client()._astream(messages)]

    assert [c.message.content for c in chunks] == ["Hello", " there", ""]
    last = chunks[-1].message
    assert last.usage_metadata["input_tokens"] == 1234
    assert last.response_metadata[REQUEST_EST_KEY] == request_tokens_est(messages)
    assert last.response_metadata[REQUEST_HAS_IMAGES_KEY] is False
    assert last.response_metadata[REQUEST_FIRST_HOP_KEY] is True
    # Only the usage chunk is stamped.
    assert all(REQUEST_EST_KEY not in c.message.response_metadata for c in chunks[:-1])


async def test_the_stamp_survives_aggregation_on_the_async_invoke_path(monkeypatch):
    _patch_server(monkeypatch, _ANSWER)
    messages = [HumanMessage("hello there " * 100)]

    message = await _client().ainvoke(messages)

    assert message.text == "Hello there"
    assert message.usage_metadata["input_tokens"] == 1234
    assert message.response_metadata[REQUEST_EST_KEY] == request_tokens_est(messages)
    assert message.response_metadata[REQUEST_FIRST_HOP_KEY] is True
    assert message.response_metadata["finish_reason"] == "stop"
    assert hop_ratio(message) == pytest.approx(1234 / request_tokens_est(messages))


async def test_the_stamp_survives_aggregation_on_the_async_stream_path(monkeypatch):
    _patch_server(monkeypatch, _ANSWER)
    messages = [HumanMessage("hi there")]

    total = None
    async for chunk in _client().astream(messages):
        total = chunk if total is None else total + chunk

    assert total.response_metadata[REQUEST_EST_KEY] == request_tokens_est(messages)
    assert total.usage_metadata["input_tokens"] == 1234


async def test_a_second_usage_chunk_is_never_stamped(monkeypatch):
    """``merge_dicts`` sums two ints under one key and rejects two differing
    bools: a second stamp would double the estimate (or fail the turn)."""
    _patch_server(monkeypatch, [_content("a"), _usage(100, 1), _usage(100, 1)])
    messages = [HumanMessage("hi")]

    chunks = [c async for c in _client()._astream(messages)]

    stamped = [c for c in chunks if REQUEST_EST_KEY in c.message.response_metadata]
    assert len(stamped) == 1


def test_pinned_why_the_stamp_rides_one_chunk_only():
    # Two differing ints under one key are SUMMED (equal ones are kept as is).
    first = AIMessageChunk(content="", response_metadata={REQUEST_EST_KEY: 10, "flag": True})
    second = AIMessageChunk(content="", response_metadata={REQUEST_EST_KEY: 15, "flag": True})
    assert (first + second).response_metadata[REQUEST_EST_KEY] == 25
    third = AIMessageChunk(content="", response_metadata={"flag": False})
    with pytest.raises(Exception):
        _ = first + third


async def test_a_later_hop_and_an_image_request_are_stamped_as_such(monkeypatch):
    _patch_server(monkeypatch, _ANSWER)
    later = [
        HumanMessage("q"),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "c1"}]),
        ToolMessage("result", tool_call_id="c1"),
    ]
    message = await _client().ainvoke(later)
    assert message.response_metadata[REQUEST_FIRST_HOP_KEY] is False

    image = [
        HumanMessage(
            content=[
                {"type": "text", "text": "what is it"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ]
        )
    ]
    message = await _client().ainvoke(image)
    assert message.response_metadata[REQUEST_HAS_IMAGES_KEY] is True
    assert hop_ratio(message) is None


async def test_the_stamp_counts_the_tool_schemas_of_the_request(monkeypatch):
    from langchain_core.utils.function_calling import convert_to_openai_tool

    def web_search(query: str) -> str:
        """Search the web."""
        return ""

    _patch_server(monkeypatch, _ANSWER)
    tools = [convert_to_openai_tool(web_search)]
    messages = [HumanMessage("hi there")]

    chunks = [c async for c in _client()._astream(messages, tools=tools)]

    assert chunks[-1].message.response_metadata[REQUEST_EST_KEY] == request_tokens_est(
        messages, tools
    )


async def test_the_one_shot_title_stream_ignores_the_usage_chunk(monkeypatch):
    _patch_server(monkeypatch, _ANSWER)

    texts = [c.text async for c in _client().astream([HumanMessage("title please")])]

    assert "".join(texts) == "Hello there"


async def test_the_stamp_and_usage_survive_a_postgres_checkpoint_round_trip(
    monkeypatch, pg_test_cluster
):
    from langchain.agents import create_agent

    from src.agents.checkpoint import open_checkpointer
    from tests._helpers import ToolableFakeChatModel

    _patch_server(monkeypatch, _ANSWER)
    messages = [HumanMessage("hi there")]
    message = await _client().ainvoke(messages)

    async with open_checkpointer(pg_test_cluster.psycopg_url) as saver:
        agent = create_agent(ToolableFakeChatModel(messages=iter([])), tools=[], checkpointer=saver)
        config = {"configurable": {"thread_id": "stamp-round-trip"}}
        await agent.aupdate_state(config, {"messages": [*messages, message]}, as_node="model")
        restored = (await agent.aget_state(config)).values["messages"][-1]

    assert restored.usage_metadata["input_tokens"] == 1234
    assert restored.response_metadata[REQUEST_EST_KEY] == request_tokens_est(messages)
    assert restored.response_metadata[REQUEST_FIRST_HOP_KEY] is True
    assert hop_ratio(restored) == hop_ratio(message)


# ===================== the KB block's share of the request =====================

_BLOCK = "Excerpt from the user manual: the device restarts after a long press. " * 30
_KB_ADDITIONS = f"{_BLOCK}\n\n\n\nAnswer in Chinese."


async def test_a_request_carrying_the_kb_block_stamps_its_size(monkeypatch):
    from src.agents.token_accounting import (
        REQUEST_KB_EST_KEY,
        REQUEST_KB_REAL_KEY,
        kb_stamp,
    )

    _patch_server(monkeypatch, _ANSWER)
    merged = HumanMessage(f"{_BLOCK}\n\n设备怎么重启？\n\nAnswer in Chinese.")

    message = await _client(kb_additions=_KB_ADDITIONS).ainvoke([merged])

    expected = kb_stamp(_KB_ADDITIONS)
    assert message.response_metadata[REQUEST_KB_EST_KEY] == expected[REQUEST_KB_EST_KEY]
    assert message.response_metadata[REQUEST_KB_REAL_KEY] == expected[REQUEST_KB_REAL_KEY]


async def test_a_request_without_the_block_stamps_no_kb_size(monkeypatch):
    from src.agents.token_accounting import REQUEST_KB_EST_KEY

    _patch_server(monkeypatch, _ANSWER)

    message = await _client(kb_additions=_KB_ADDITIONS).ainvoke([HumanMessage("plain question")])

    assert REQUEST_KB_EST_KEY not in message.response_metadata


def test_cjk_history_and_an_english_kb_block_next_turn():
    """The previous KB turn's request: an English block (1500 estimated, real
    1500) around a CJK history (1000 estimated, real 2500). Its plain ratio is
    1.6; the next request carries the history alone, which reads 2.5."""
    from langchain_core.messages import AIMessage

    from src.agents.token_accounting import (
        BUDGET_DENSE_TOKENS,
        REQUEST_HAS_IMAGES_KEY,
        REQUEST_KB_EST_KEY,
        REQUEST_KB_REAL_KEY,
        message_weights,
    )

    kb_hop = AIMessage(
        content="答案",
        usage_metadata={"input_tokens": 4000, "output_tokens": 3, "total_tokens": 4003},
        response_metadata={
            REQUEST_EST_KEY: 2500,
            REQUEST_HAS_IMAGES_KEY: False,
            REQUEST_FIRST_HOP_KEY: True,
            REQUEST_KB_EST_KEY: 1500,
            REQUEST_KB_REAL_KEY: 1500,
        },
    )
    history = HumanMessage("长城的历史很长。" * 100)
    weights, ratio = message_weights(
        [history, kb_hop, HumanMessage("下一个问题")], dense=BUDGET_DENSE_TOKENS
    )
    assert ratio == pytest.approx(2.5)
    assert weights[0] == pytest.approx(2.5)
