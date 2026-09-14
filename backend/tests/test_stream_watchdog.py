"""#573 -- two wall-clock budgets per model call instead of one uniform timeout.

langchain-openai (pinned 1.2.2) bounds EVERY async SSE stream with a single
``stream_chunk_timeout`` (default 120 s). A local model has two silences with
nothing in common: prefill, which grows with the prompt and produced 132.9 s of
silence on a 6878-token turn (Qwen3.5 9B, Apple M4 base, 51.8 tok/s), and the
gaps between decoded tokens, which stay in the millisecond range. The uniform
knob killed a healthy turn 13 s before its first token.

These tests pin the replacement: a first-chunk budget computed from the prompt
size, a 120 s inter-chunk budget once tokens flow, and the honest error the
first-chunk budget raises. No inference runs here -- the streams are fakes with
budgets in the tens of milliseconds, so the proof is in the phase boundaries,
not in wall-clock duration.
"""

import asyncio
import inspect

import pytest

from src.agents import chat_model as chat_model_module
from src.agents.chat_model import (
    CONSERVATIVE_PREFILL_TOKENS_PER_SEC,
    FIRST_CHUNK_BASE_S,
    FIRST_CHUNK_CEILING_S,
    FIRST_CHUNK_FLOOR_S,
    INTER_CHUNK_BUDGET_S,
    PHASE_FIRST_CHUNK,
    PHASE_INTER_CHUNK,
    erudi_chat_openai_class,
    estimate_prompt_tokens,
    first_chunk_budget_s,
    stream_with_two_phase_budget,
)
from src.agents.model_factory import build_chat_model
from src.core import config
from src.core.exceptions import GenerationTimeoutException

pytestmark = pytest.mark.unit


# ===================== fake streams =====================


async def _scripted(script, *, closed=None):
    """Yield each ``(delay, value)`` pair after sleeping ``delay``.

    ``closed`` (a list) records the close so a test can prove the watchdog
    releases the underlying stream instead of leaking the HTTP connection.
    """
    try:
        for delay, value in script:
            await asyncio.sleep(delay)
            yield value
    finally:
        if closed is not None:
            closed.append(True)


async def _drain(source, **kwargs):
    return [chunk async for chunk in stream_with_two_phase_budget(source, **kwargs)]


# ===================== two-phase wrapper =====================


async def test_chunks_pass_through_untouched_when_every_gap_fits():
    chunks = await _drain(
        _scripted([(0.0, "a"), (0.01, "b"), (0.01, "c")]),
        first_budget_s=0.3,
        inter_budget_s=0.3,
    )

    assert chunks == ["a", "b", "c"]


async def test_a_late_first_chunk_raises_with_the_first_chunk_phase():
    with pytest.raises(GenerationTimeoutException) as excinfo:
        await _drain(
            _scripted([(0.2, "a")]),
            first_budget_s=0.03,
            inter_budget_s=0.3,
            estimated_prompt_tokens=6878,
            model_name="qwen-9b",
        )

    exc = excinfo.value
    assert exc.phase == PHASE_FIRST_CHUNK
    assert exc.budget_s == 0.03
    assert exc.estimated_prompt_tokens == 6878
    assert exc.erudi_code == "GENERATION_TIMEOUT"


async def test_a_first_chunk_inside_a_long_budget_survives_a_short_inter_budget():
    # The incident in one assertion: a prefill longer than the inter-chunk
    # budget is NOT a hang, and must not be cut. A single uniform timeout of
    # ``inter_budget_s`` would have killed this stream.
    chunks = await _drain(
        _scripted([(0.15, "first"), (0.01, "second")]),
        first_budget_s=0.4,
        inter_budget_s=0.05,
    )

    assert chunks == ["first", "second"]


async def test_a_gap_after_the_first_chunk_raises_with_the_inter_chunk_phase():
    with pytest.raises(GenerationTimeoutException) as excinfo:
        await _drain(
            _scripted([(0.0, "a"), (0.2, "b")]),
            first_budget_s=0.3,
            inter_budget_s=0.03,
        )

    assert excinfo.value.phase == PHASE_INTER_CHUNK
    assert excinfo.value.budget_s == 0.03


async def test_a_gap_within_the_inter_chunk_budget_is_fine():
    chunks = await _drain(
        _scripted([(0.0, "a"), (0.05, "b")]),
        first_budget_s=0.3,
        inter_budget_s=0.3,
    )

    assert chunks == ["a", "b"]


async def test_the_source_stream_is_closed_when_the_watchdog_fires():
    closed: list = []

    with pytest.raises(GenerationTimeoutException):
        await _drain(
            _scripted([(0.2, "a")], closed=closed),
            first_budget_s=0.03,
            inter_budget_s=0.3,
        )

    assert closed == [True]


async def test_an_empty_stream_ends_cleanly():
    assert await _drain(_scripted([]), first_budget_s=0.3, inter_budget_s=0.3) == []


# ===================== the budget formula =====================


def test_a_small_prompt_gets_the_floor():
    assert first_chunk_budget_s(0) == FIRST_CHUNK_FLOOR_S
    assert first_chunk_budget_s(100) == FIRST_CHUNK_FLOOR_S


def test_the_budget_scales_with_the_prompt_past_the_floor():
    small = first_chunk_budget_s(5_000)
    large = first_chunk_budget_s(20_000)

    assert FIRST_CHUNK_FLOOR_S < small < large
    assert small == pytest.approx(FIRST_CHUNK_BASE_S + 5_000 / CONSERVATIVE_PREFILL_TOKENS_PER_SEC)


def test_a_huge_prompt_stops_at_the_ceiling():
    assert first_chunk_budget_s(10_000_000) == FIRST_CHUNK_CEILING_S


def test_the_incident_prompt_gets_more_than_the_measured_prefill():
    # 6878 tokens prefilled at 51.8 tok/s = 132.9 s of silence, which the old
    # uniform 120 s budget cut short. The conservative rate must leave room.
    budget = first_chunk_budget_s(6878)

    assert budget > 132.9
    assert budget > INTER_CHUNK_BUDGET_S


# ===================== the prompt-size estimate =====================


class _Msg:
    def __init__(self, content):
        self.content = content


def test_the_estimate_over_counts_rather_than_under_counts():
    # Deliberately pessimistic: ~3 chars per token (real tokenizers average
    # closer to 4), so the budget is never short because of the estimate.
    tokens = estimate_prompt_tokens([_Msg("x" * 3_000)])

    assert tokens >= 1_000


def test_the_estimate_grows_with_the_conversation():
    short = estimate_prompt_tokens([_Msg("hello")])
    long = estimate_prompt_tokens([_Msg("hello " * 500)])

    assert long > short


def test_the_estimate_reads_multimodal_parts():
    text_only = estimate_prompt_tokens([_Msg([{"type": "text", "text": "describe this"}])])
    with_image = estimate_prompt_tokens(
        [
            _Msg(
                [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ]
            )
        ]
    )

    assert with_image > text_only


def test_the_estimate_survives_odd_shapes():
    assert estimate_prompt_tokens([]) == 0
    assert estimate_prompt_tokens(None) == 0
    assert estimate_prompt_tokens([_Msg(None)]) >= 0
    assert estimate_prompt_tokens([{"role": "user", "content": "hi"}]) > 0


# ===================== the ChatOpenAI subclass =====================


class _IdentityEngine:
    """Engine stub: a base_url handle and an MLX-style payload model value."""

    @staticmethod
    def get_model_and_tokenizer(llm_id, link):
        return ({"base_url": "http://127.0.0.1:8080", "alias": f"erudi-{llm_id}"}, {})

    @staticmethod
    def _payload_model_value(handle):
        return "default_model"


class _Llm:
    id = 7
    link = "/fake/path"
    name = "Test 7B"
    param_size = 7.0


def test_build_chat_model_returns_the_watchdog_subclass_with_langchains_knob_off(monkeypatch):
    monkeypatch.setattr(config, "LLM_Engine", _IdentityEngine)

    chat = build_chat_model(_Llm(), temperature=0.3, top_p=0.8, max_tokens=55)

    assert isinstance(chat, erudi_chat_openai_class())
    # langchain's uniform per-chunk timeout must be OFF: ours replaces it.
    assert chat.stream_chunk_timeout is None


def _tiny_budgets(monkeypatch):
    monkeypatch.setattr(chat_model_module, "FIRST_CHUNK_FLOOR_S", 0.03)
    monkeypatch.setattr(chat_model_module, "FIRST_CHUNK_BASE_S", 0.01)
    monkeypatch.setattr(chat_model_module, "FIRST_CHUNK_CEILING_S", 0.05)
    monkeypatch.setattr(chat_model_module, "INTER_CHUNK_BUDGET_S", 0.03)


def _model():
    return erudi_chat_openai_class()(
        base_url="http://127.0.0.1:1/v1",
        api_key="not-needed",
        model="fake-model",
    )


async def test_the_subclass_stream_enforces_the_first_chunk_budget(monkeypatch):
    from langchain_openai import ChatOpenAI

    _tiny_budgets(monkeypatch)

    async def _slow_parent(self, messages, *args, **kwargs):
        await asyncio.sleep(0.3)
        yield "never reached"

    monkeypatch.setattr(ChatOpenAI, "_astream", _slow_parent)

    with pytest.raises(GenerationTimeoutException) as excinfo:
        async for _ in _model()._astream([_Msg("hi")]):
            pass

    assert excinfo.value.phase == PHASE_FIRST_CHUNK


async def test_the_subclass_stream_passes_healthy_chunks_through(monkeypatch):
    from langchain_openai import ChatOpenAI

    _tiny_budgets(monkeypatch)

    async def _fast_parent(self, messages, *args, **kwargs):
        for token in ("a", "b", "c"):
            yield token

    monkeypatch.setattr(ChatOpenAI, "_astream", _fast_parent)

    assert [chunk async for chunk in _model()._astream([_Msg("hi")])] == ["a", "b", "c"]


# ===================== pinned upstream assumptions =====================
#
# The override is the narrowest one that survives a patch bump, but it still
# rests on three facts about langchain-openai. A future upgrade that moves any
# of them must fail HERE, loudly, rather than silently restoring the uniform
# 120 s timeout on a machine nobody is watching.


def test_chat_openai_still_exposes_the_async_stream_hook_we_override():
    from langchain_openai import ChatOpenAI

    assert inspect.isasyncgenfunction(ChatOpenAI._astream)
    parameters = list(inspect.signature(ChatOpenAI._astream).parameters)
    # ``messages`` must still reach the override positionally: the budget is
    # computed from what is actually being sent.
    assert parameters[0] == "self"
    assert len(parameters) >= 2

    subclass = erudi_chat_openai_class()
    assert subclass._astream is not ChatOpenAI._astream
    assert inspect.isasyncgenfunction(subclass._astream)


def test_chat_openai_still_owns_the_uniform_chunk_timeout_field():
    from langchain_openai import ChatOpenAI

    field = ChatOpenAI.model_fields["stream_chunk_timeout"]
    # The public knob we switch off. If upstream drops or renames it, our
    # ``stream_chunk_timeout=None`` silently stops disabling anything.
    assert field.default_factory is not None
    assert field.default_factory() == 120.0
