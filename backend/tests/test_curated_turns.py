"""Curated turns carry a machine-readable trace, the model keeps English (#690.2).

A curated turn is the app's own sentence for a turn that could not be
answered normally (a prompt that does not fit, a model that did not start
answering in time, the tool-loop limit, an answer that never came). The
backend writes it in English, ASCII, for the database and for the model's next
request, and carries ``curated_error {kind, params}`` beside it so the
interface translates it at display, numbers formatted for its language.
"""

import itertools
import json
import re
from pathlib import Path
from typing import Any, ClassVar

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from src.agents import curated_turns
from src.agents import runner as runner_module
from src.agents.chat_model import INTER_CHUNK_BUDGET_S, PHASE_FIRST_CHUNK, PHASE_INTER_CHUNK
from src.agents.curated_turns import CURATED_KINDS, curated_answer, curated_text
from src.agents.runner import ERROR_MESSAGE, ERROR_SENTINEL, AgentRunner, GenParams
from src.core import config
from src.core.exceptions import GenerationTimeoutException
from src.engines.base_engine import BaseEngine
from tests._helpers import ToolableFakeChatModel

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


async def _turn(monkeypatch, model, *, thread_id, saver=None, tools=None):
    _patch_model(monkeypatch, model)
    runner = AgentRunner(checkpointer=saver or InMemorySaver())
    return [
        e
        async for e in runner.astream_text(
            llm=_Llm(),
            user_message="hi",
            system_prompt="s",
            params=_PARAMS,
            thread_id=thread_id,
            tools=tools,
        )
    ]


def _curated_events(events):
    """The curated answer events, their ``curated_error`` reduced to kind and
    params (its English ``text`` is pinned by the registry tests)."""
    found = []
    for e in events:
        if e["t"] == "answer" and "curated_error" in e:
            curated = e["curated_error"]
            assert e["text"].endswith(curated["text"])
            reduced = {"kind": curated["kind"], "params": curated["params"]}
            found.append({**e, "curated_error": reduced})
    return found


def _last_thread_message(saver, thread_id):
    tup = saver.get_tuple({"configurable": {"thread_id": thread_id}})
    return tup.checkpoint["channel_values"]["messages"][-1]


class _RaisingModel(ToolableFakeChatModel):
    exc: ClassVar[Exception] = RuntimeError("unset")

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        raise self.exc
        yield  # pragma: no cover  (makes this an async generator function)


class _FakeBadRequestError(Exception):
    def __init__(self, message: str, body: dict):
        super().__init__(message)
        self.body = body


def _raising(exc):
    return type("_M", (_RaisingModel,), {"exc": exc})(messages=iter([]))


def _raw_chunk(content="", reasoning=None, finish_reason=None):
    return ChatGenerationChunk(
        message=AIMessageChunk(
            content=content,
            additional_kwargs={"reasoning_content": reasoning} if reasoning is not None else {},
        ),
        generation_info={"finish_reason": finish_reason} if finish_reason else None,
    )


class _RawChunkModel(ToolableFakeChatModel):
    hops: Any = None

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        yield from next(self.hops)


# ===== The registry =====


def test_every_kind_is_english_ascii_and_formats_with_its_params():
    samples = {
        "context_overflow": {"prompt_tokens": 9030, "context_tokens": 8192},
        "context_overflow_unsized": {},
        "memory_overflow": {"prompt_tokens": 13531, "context_tokens": 8584},
        "prefill_timeout": {"minutes": 5},
        "decode_timeout": {"seconds": 120},
        "loop_limit": {},
        "empty_answer_length": {},
        "empty_answer_stop": {},
    }
    assert set(CURATED_KINDS) == set(samples)
    for kind, params in samples.items():
        text = curated_text(kind, **params)
        assert text.isascii(), kind
        assert "{" not in text and "}" not in text, kind
        for value in params.values():
            assert str(value) in text, kind


def test_error_kinds_carry_the_sentinel_and_empty_answer_kinds_do_not():
    for kind, spec in CURATED_KINDS.items():
        params = {name: 1 for name in spec.params}
        text = curated_text(kind, **params)
        assert text.startswith(f"{ERROR_SENTINEL} ") is spec.error, kind


def test_the_answer_event_carries_the_trace_beside_the_english_line():
    event = curated_answer("memory_overflow", prompt_tokens=13531, context_tokens=8584)
    assert event == {
        "t": "answer",
        "text": (
            f"{ERROR_SENTINEL} This conversation no longer fits what your machine's memory can "
            "hold for this model (the request needs about 13531 tokens; memory allows 8584). "
            "Start a new conversation, send less at once, or switch to a smaller model."
        ),
        "curated": True,
        "curated_error": {
            "kind": "memory_overflow",
            "params": {"prompt_tokens": 13531, "context_tokens": 8584},
            # The English line itself, without the sentinel: what the
            # interface finds in the stored content and replaces (#690).
            "text": (
                "This conversation no longer fits what your machine's memory can "
                "hold for this model (the request needs about 13531 tokens; memory allows 8584). "
                "Start a new conversation, send less at once, or switch to a smaller model."
            ),
        },
    }


def test_an_empty_answer_line_is_not_flagged_as_a_failure():
    event = curated_answer("empty_answer_stop")
    assert "curated" not in event
    assert event["curated_error"] == {
        "kind": "empty_answer_stop",
        "params": {},
        "text": event["text"],
    }


def test_a_kind_refuses_missing_or_unknown_params():
    with pytest.raises(KeyError):
        curated_answer("no_such_kind")
    with pytest.raises(ValueError):
        curated_answer("memory_overflow", prompt_tokens=1)
    with pytest.raises(ValueError):
        curated_answer("loop_limit", extra=1)


def test_the_trace_event_is_what_the_conversation_persists():
    event = curated_answer("prefill_timeout", minutes=5)
    assert curated_turns.curated_trace_event(event["curated_error"]) == {
        "t": "curated_error",
        "kind": "prefill_timeout",
        "params": {"minutes": 5},
        "text": event["curated_error"]["text"],
    }
    assert event["text"] == f"{ERROR_SENTINEL} {event['curated_error']['text']}"


# ===== Each curated turn of the runner writes its trace =====


async def test_a_window_overflow_carries_its_numbers(monkeypatch):
    saver = InMemorySaver()
    exc = _FakeBadRequestError(
        "Error code: 400",
        body={"type": "exceed_context_size_error", "n_prompt_tokens": 9030, "n_ctx": 8192},
    )
    events = await _turn(monkeypatch, _raising(exc), thread_id="c-window", saver=saver)

    (event,) = _curated_events(events)
    assert event["curated_error"] == {
        "kind": "context_overflow",
        "params": {"prompt_tokens": 9030, "context_tokens": 8192},
    }
    assert event["text"] == curated_text(
        "context_overflow", prompt_tokens=9030, context_tokens=8192
    )
    # The model's next request sees an English, ASCII assistant turn.
    last = _last_thread_message(saver, "c-window")
    assert last.type == "ai" and str(last.text) == ERROR_MESSAGE and ERROR_MESSAGE.isascii()


async def test_an_overflow_without_numbers_has_its_own_kind(monkeypatch):
    exc = _FakeBadRequestError("Error code: 400", body={"type": "exceed_context_size_error"})
    events = await _turn(monkeypatch, _raising(exc), thread_id="c-unsized")

    (event,) = _curated_events(events)
    assert event["curated_error"] == {"kind": "context_overflow_unsized", "params": {}}


async def test_a_memory_overflow_carries_its_numbers(monkeypatch):
    monkeypatch.setattr(_FakeEngine, "memory_bound_tokens", classmethod(lambda cls: 8584))
    exc = _FakeBadRequestError(
        "Request needs 13531 context tokens (13523 prompt + 8 max generation), "
        "but MAX_KV_SIZE is 8584.",
        body={},
    )
    events = await _turn(monkeypatch, _raising(exc), thread_id="c-memory")

    (event,) = _curated_events(events)
    assert event["curated_error"] == {
        "kind": "memory_overflow",
        "params": {"prompt_tokens": 13531, "context_tokens": 8584},
    }


async def test_a_first_chunk_timeout_carries_its_minutes(monkeypatch):
    exc = GenerationTimeoutException(
        "no first chunk", phase=PHASE_FIRST_CHUNK, budget_s=305.0, estimated_prompt_tokens=6878
    )
    events = await _turn(monkeypatch, _raising(exc), thread_id="c-prefill")

    (event,) = _curated_events(events)
    assert event["curated_error"] == {"kind": "prefill_timeout", "params": {"minutes": 5}}


async def test_a_silent_stream_carries_its_seconds(monkeypatch):
    exc = GenerationTimeoutException(
        "silent", phase=PHASE_INTER_CHUNK, budget_s=120.0, estimated_prompt_tokens=10
    )
    events = await _turn(monkeypatch, _raising(exc), thread_id="c-decode")

    (event,) = _curated_events(events)
    assert event["curated_error"] == {
        "kind": "decode_timeout",
        "params": {"seconds": round(INTER_CHUNK_BUDGET_S)},
    }


async def test_a_leaked_child_queue_timeout_is_the_prefill_kind(monkeypatch):
    exc = RuntimeError(
        "Timed out waiting for 600s for the next generated token. "
        "Increase MLX_VLM_TOKEN_QUEUE_TIMEOUT for long prefills, or reduce the prompt size."
    )
    events = await _turn(monkeypatch, _raising(exc), thread_id="c-queue")

    (event,) = _curated_events(events)
    assert event["curated_error"]["kind"] == "prefill_timeout"
    assert event["curated_error"]["params"]["minutes"] >= 1


async def test_the_tool_loop_limit_has_its_kind(monkeypatch):
    @tool
    def blank_tool(query: str) -> str:
        """A tool that returns nothing usable."""
        return ""

    call = AIMessage(
        content="", tool_calls=[{"name": "blank_tool", "args": {"query": "x"}, "id": "c"}]
    )
    saver = InMemorySaver()
    events = await _turn(
        monkeypatch,
        ToolableFakeChatModel(messages=itertools.repeat(call)),
        thread_id="c-loop",
        saver=saver,
        tools=[blank_tool],
    )

    (event,) = _curated_events(events)
    assert event["curated_error"] == {"kind": "loop_limit", "params": {}}
    assert event["curated"] is True


@pytest.mark.parametrize(
    "finish_reason, kind",
    [("length", "empty_answer_length"), ("stop", "empty_answer_stop")],
)
async def test_an_empty_answer_has_its_kind_and_the_thread_keeps_english(
    monkeypatch, finish_reason, kind
):
    saver = InMemorySaver()
    model = _RawChunkModel(
        messages=iter([]),
        hops=iter([[_raw_chunk(reasoning="hmm.", finish_reason=finish_reason)]]),
    )
    events = await _turn(monkeypatch, model, thread_id=f"c-{kind}", saver=saver)

    (event,) = _curated_events(events)
    assert event["curated_error"] == {"kind": kind, "params": {}}
    assert "curated" not in event
    english = curated_text(kind)
    assert event["text"] == english
    last = _last_thread_message(saver, f"c-{kind}")
    assert str(last.text) == english and english.isascii()


async def test_an_ordinary_answer_carries_no_curated_trace(monkeypatch):
    events = await _turn(
        monkeypatch,
        ToolableFakeChatModel(messages=iter([AIMessage(content="Fine.")])),
        thread_id="c-plain",
    )
    assert _curated_events(events) == []


# ===== Every kind has a sentence in every interface language =====

_LOCALES = Path(__file__).resolve().parents[2] / "frontend" / "src" / "locales"


@pytest.mark.parametrize("language", ["en", "fr", "es", "zh"])
def test_every_kind_has_its_sentence_with_exactly_its_placeholders(language):
    """A kind the interface has no key for shows its stored English line: a
    hole here is a curated turn left untranslated. The placeholders must be the
    template's own, or a number goes missing from the sentence."""
    chat = json.loads((_LOCALES / language / "chat.json").read_text(encoding="utf-8"))
    curated = chat["curated"]

    assert set(curated) == set(CURATED_KINDS)
    for kind, spec in CURATED_KINDS.items():
        placeholders = set(re.findall(r"\{\{\s*(\w+)\s*\}\}", curated[kind]))
        assert placeholders == set(spec.params), (language, kind)
