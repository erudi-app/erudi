"""An abandoned model stream flags the child it was built against.

``Erudi_Chat_OpenAI._astream`` consumes its inner stream under
``contextlib.aclosing`` so the HTTP response is closed deterministically, and
when the stream did not end normally it calls the ``abandon_hook`` the factory
bound to the engine handle captured at build time. The MLX engine then sends a
barrier before its next prefix-cache reset (``src.engines.mlx_engine``).
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import HumanMessage

from src.agents import chat_model as chat_model_module
from src.agents.chat_model import erudi_chat_openai_class
from src.agents.model_factory import build_chat_model
from src.core import config
from src.core.exceptions import GenerationTimeoutException
from src.engines.mlx_engine import MLX_Engine

pytestmark = pytest.mark.unit


def _client(hook):
    return erudi_chat_openai_class()(
        base_url="http://127.0.0.1:1/v1",
        api_key="not-needed",
        model="fake-model",
        abandon_hook=hook,
    )


class _Recorder:
    def __init__(self):
        self.events: list[str] = []

    def hook(self):
        self.events.append("hook")


def _inner(recorder, script):
    """A stand-in for ``ChatOpenAI._astream`` that records when it is closed."""

    async def _astream(self, messages, *args, **kwargs):
        try:
            for step in script:
                if isinstance(step, BaseException):
                    raise step
                if step == "sleep":
                    await asyncio.sleep(10)
                yield step
        finally:
            recorder.events.append("inner-closed")

    return _astream


async def test_a_normal_completion_never_calls_the_hook(monkeypatch):
    from langchain_openai import ChatOpenAI

    rec = _Recorder()
    monkeypatch.setattr(ChatOpenAI, "_astream", _inner(rec, ["a", "b"]))

    chunks = [c async for c in _client(rec.hook)._astream([HumanMessage("hi")])]

    assert chunks == ["a", "b"]
    assert "hook" not in rec.events


async def test_a_consumer_that_closes_mid_stream_calls_the_hook_once_after_the_inner_close(
    monkeypatch,
):
    from langchain_openai import ChatOpenAI

    rec = _Recorder()
    monkeypatch.setattr(ChatOpenAI, "_astream", _inner(rec, ["a", "b", "c"]))

    stream = _client(rec.hook)._astream([HumanMessage("hi")])
    assert await stream.__anext__() == "a"
    await stream.aclose()  # GeneratorExit at the yield

    assert rec.events == ["inner-closed", "hook"]


async def test_an_exception_mid_stream_calls_the_hook_once(monkeypatch):
    from langchain_openai import ChatOpenAI

    rec = _Recorder()
    monkeypatch.setattr(ChatOpenAI, "_astream", _inner(rec, ["a", RuntimeError("reset")]))

    with pytest.raises(RuntimeError):
        _ = [c async for c in _client(rec.hook)._astream([HumanMessage("hi")])]

    assert rec.events == ["inner-closed", "hook"]


async def test_a_watchdog_timeout_calls_the_hook_once(monkeypatch):
    from langchain_openai import ChatOpenAI

    rec = _Recorder()
    monkeypatch.setattr(chat_model_module, "INTER_CHUNK_BUDGET_S", 0.05)
    monkeypatch.setattr(ChatOpenAI, "_astream", _inner(rec, ["a", "sleep"]))

    with pytest.raises(GenerationTimeoutException):
        _ = [c async for c in _client(rec.hook)._astream([HumanMessage("hi")])]

    assert rec.events.count("hook") == 1
    assert rec.events.index("inner-closed") < rec.events.index("hook")


async def test_a_cancelled_stream_calls_the_hook_once(monkeypatch):
    from langchain_openai import ChatOpenAI

    rec = _Recorder()
    monkeypatch.setattr(ChatOpenAI, "_astream", _inner(rec, ["a", "sleep"]))

    async def _consume():
        async for _ in _client(rec.hook)._astream([HumanMessage("hi")]):
            pass

    task = asyncio.create_task(_consume())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert rec.events == ["inner-closed", "hook"]


async def test_a_retried_preflight_rejection_also_flags(monkeypatch):
    """Documented cost: a preflight 400 that is then retried flags the child,
    which costs one barrier before the next reset."""
    from langchain_openai import ChatOpenAI

    rec = _Recorder()
    attempts = []

    class _Rejection(Exception):
        def __str__(self):
            return "Request needs 9000 context tokens (100 prompt + 8900 max generation), but MAX_KV_SIZE is 4096."

    async def _server(self, messages, *args, **kwargs):
        attempts.append(kwargs.get("max_tokens"))
        if len(attempts) == 1:
            raise _Rejection()
        yield "ok"

    monkeypatch.setattr(ChatOpenAI, "_astream", _server)
    client = erudi_chat_openai_class()(
        base_url="http://127.0.0.1:1/v1",
        api_key="not-needed",
        model="fake-model",
        effective_context_tokens=4096,
        abandon_hook=rec.hook,
    )

    assert [c async for c in client._astream([HumanMessage("hi")])] == ["ok"]
    assert len(attempts) == 2
    assert rec.events == ["hook"]


class _Llm:
    id = 7
    link = "/fake/path"
    name = "Test 7B"
    param_size = 7.0


class _HandleEngine(MLX_Engine):
    """MLX hooks over a scripted handle; no child is spawned."""

    @classmethod
    def get_model_and_tokenizer(cls, llm_id, link):
        return cls._model, {}

    @classmethod
    def _translate_payload_kwargs(cls, kwargs):
        return dict(kwargs)


def test_the_factory_binds_the_hook_to_the_handle_captured_at_build_time(monkeypatch):
    built_for = {"base_url": "http://127.0.0.1:27300", "model_path": "/m", "api_key": "k"}
    monkeypatch.setattr(_HandleEngine, "_model", built_for)
    monkeypatch.setattr(config, "LLM_Engine", _HandleEngine)

    client = build_chat_model(_Llm(), temperature=0.5, top_p=0.9, max_tokens=64)

    # The engine swapped children since the build: the late finalizer must
    # still flag the child the stream ran against, never the new one.
    swapped_in = {"base_url": "http://127.0.0.1:27301", "model_path": "/m2", "api_key": "k2"}
    _HandleEngine._model = swapped_in
    client.abandon_hook()

    assert built_for["abandoned"] is True
    assert "abandoned" not in swapped_in


def test_an_engine_without_the_hook_builds_a_client_without_one(monkeypatch):
    class _Plain:
        @staticmethod
        def get_model_and_tokenizer(llm_id, link):
            return ({"base_url": "http://127.0.0.1:8080", "alias": "erudi-7"}, {})

        @staticmethod
        def _payload_model_value(handle):
            return "default_model"

    monkeypatch.setattr(config, "LLM_Engine", _Plain)

    client = build_chat_model(_Llm(), temperature=0.5, top_p=0.9, max_tokens=64)

    assert client.abandon_hook is None
