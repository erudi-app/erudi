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

``Erudi_Chat_OpenAI`` carries a second, disjoint override since #554 (the
reasoning extraction in ``_convert_chunk_to_generation_chunk``); the last
section pins that hook, its upstream assumptions, and its composition with the
watchdog.
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


def test_the_incident_history_reaches_the_ceiling_through_the_estimate():
    # End to end on the reported turn: 6878 English-ish tokens is roughly 27.5k
    # characters, hence ~27.5k UTF-8 bytes, hence a 30 + 27500/25 = 1130 s raw
    # budget -- clamped to the 900 s ceiling. Far more than the 132.9 s the turn
    # actually needed, which is the accepted trade of a provable upper bound.
    budget = first_chunk_budget_s(estimate_prompt_tokens([_Msg("a" * 27_500)]))

    assert budget == FIRST_CHUNK_CEILING_S


# ===================== the window-aware ceiling =====================
#
# FIRST_CHUNK_CEILING_S (900 s) was sized against the old fixed 4096 window.
# With the dynamic context landing, a full-window prompt on a big window
# legitimately needs BASE + W/RATE seconds of prefill; a fixed 900 s ceiling
# there would recreate exactly the #573 kill the two-phase watchdog fixed. So
# the ceiling derives from the ALLOCATED window of the loaded child
# (effective_context_tokens) and 900 s stays the floor of the ceiling.


def test_the_ceiling_stays_at_900_without_an_effective_window():
    assert first_chunk_budget_s(10_000_000, effective_window_tokens=None) == FIRST_CHUNK_CEILING_S


def test_a_big_window_raises_the_ceiling_to_a_full_window_prefill():
    budget = first_chunk_budget_s(10_000_000, effective_window_tokens=60_000)

    assert budget == pytest.approx(
        FIRST_CHUNK_BASE_S + 60_000 / CONSERVATIVE_PREFILL_TOKENS_PER_SEC
    )
    assert budget > FIRST_CHUNK_CEILING_S


def test_the_absolute_backstop_caps_a_million_token_window():
    # The window-scaled ceiling must NOT scale without limit: a million-token
    # window would otherwise let a genuinely hung child hold the engine's
    # global generation lock for ~11 hours before the watchdog fires. One hour
    # is the honest maximum wait for a first token on any machine this app
    # targets.
    budget = first_chunk_budget_s(10_000_000, effective_window_tokens=1_048_576)

    assert budget == chat_model_module.FIRST_CHUNK_ABSOLUTE_MAX_S


def test_a_small_window_never_lowers_the_ceiling():
    # max(900, ...) by design: a 4096 window keeps the field-proven ceiling.
    assert first_chunk_budget_s(10_000_000, effective_window_tokens=4096) == FIRST_CHUNK_CEILING_S


def test_a_prompt_below_the_raised_ceiling_keeps_its_own_budget():
    # The raised ceiling is a cap, not a grant: a smaller prompt still gets
    # its prompt-sized budget.
    budget = first_chunk_budget_s(30_000, effective_window_tokens=40_960)

    assert budget == pytest.approx(
        FIRST_CHUNK_BASE_S + 30_000 / CONSERVATIVE_PREFILL_TOKENS_PER_SEC
    )
    assert budget < FIRST_CHUNK_BASE_S + 40_960 / CONSERVATIVE_PREFILL_TOKENS_PER_SEC


# ===================== the prompt-size estimate =====================


class _Msg:
    def __init__(self, content):
        self.content = content


def _utf8_bytes(*texts) -> int:
    return sum(len(text.encode("utf-8")) for text in texts)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("The quick brown fox jumps over the lazy dog. " * 50, id="english"),
        # ~1 token per character on a BPE tokenizer (3 UTF-8 bytes each): the
        # shape that a chars/N heuristic under-counts by 3x, cutting exactly the
        # users the zh locale exists for.
        pytest.param("这是一个很长的中文对话历史记录。" * 200, id="chinese"),
        pytest.param("こんにちは世界、これはテストです。" * 200, id="japanese"),
        # Punctuation-dense code sits near 1-2 characters per token.
        pytest.param("def f(x):\n    return [y**2 for y in x if y % 2 == 0]\n" * 40, id="code"),
        # 4-byte code points.
        pytest.param("🚀🎉🔥" * 300, id="emoji"),
    ],
)
def test_the_estimate_never_falls_below_the_utf8_byte_count(text):
    # A byte-level BPE tokenizer cannot emit more tokens than the text has
    # bytes: every token consumes at least one. So the byte count is a bound
    # that holds for EVERY language, not an average that holds for English.
    assert estimate_prompt_tokens([_Msg(text)]) >= _utf8_bytes(text)


def test_a_chinese_history_keeps_a_budget_longer_than_its_real_prefill():
    # 7000 Chinese characters ~= 7000 tokens ~= 135 s of prefill at the incident
    # machine's 51.8 tok/s. The budget must clear that.
    estimated = estimate_prompt_tokens([_Msg("文" * 7_000)])

    assert estimated >= 7_000
    assert first_chunk_budget_s(estimated) > 135.0


def test_the_estimate_covers_the_text_of_every_message():
    parts = ["première partie", "seconde partie", "troisième partie"]

    assert estimate_prompt_tokens([_Msg(part) for part in parts]) >= _utf8_bytes(*parts)


def test_the_estimate_grows_with_the_conversation():
    short = estimate_prompt_tokens([_Msg("hello")])
    long = estimate_prompt_tokens([_Msg("hello " * 500)])

    assert long > short


def test_the_estimate_counts_the_text_inside_content_parts():
    parts = [{"type": "text", "text": "décris cette image 描述这张图片"}]

    assert estimate_prompt_tokens([_Msg(parts)]) >= _utf8_bytes(parts[0]["text"])


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


class _WindowedEngine(_IdentityEngine):
    """Engine stub whose loaded child carries an allocated window."""

    @staticmethod
    def effective_context_tokens():
        return 40_960


def test_build_chat_model_returns_the_watchdog_subclass_with_langchains_knob_off(monkeypatch):
    monkeypatch.setattr(config, "LLM_Engine", _IdentityEngine)

    chat = build_chat_model(_Llm(), temperature=0.3, top_p=0.8, max_tokens=55)

    assert isinstance(chat, erudi_chat_openai_class())
    # langchain's uniform per-chunk timeout must be OFF: ours replaces it.
    assert chat.stream_chunk_timeout is None


def test_build_chat_model_carries_the_engines_allocated_window(monkeypatch):
    # The factory runs once per turn, AFTER get_model_and_tokenizer, so the
    # window it reads is the loaded child's -- fresh across model swaps.
    monkeypatch.setattr(config, "LLM_Engine", _WindowedEngine)

    chat = build_chat_model(_Llm(), temperature=0.3, top_p=0.8, max_tokens=55)

    assert chat.effective_context_tokens == 40_960


def test_build_chat_model_without_a_window_probe_keeps_the_window_unknown(monkeypatch):
    monkeypatch.setattr(config, "LLM_Engine", _IdentityEngine)

    chat = build_chat_model(_Llm(), temperature=0.3, top_p=0.8, max_tokens=55)

    assert chat.effective_context_tokens is None


async def test_the_subclass_stream_budgets_with_its_effective_window(monkeypatch):
    from langchain_openai import ChatOpenAI

    captured: dict = {}

    def _spy(estimated, effective_window_tokens=None):
        captured["window"] = effective_window_tokens
        return 0.3

    monkeypatch.setattr(chat_model_module, "first_chunk_budget_s", _spy)

    async def _fast_parent(self, messages, *args, **kwargs):
        yield "a"

    monkeypatch.setattr(ChatOpenAI, "_astream", _fast_parent)

    model = erudi_chat_openai_class()(
        base_url="http://127.0.0.1:1/v1",
        api_key="not-needed",
        model="fake-model",
        effective_context_tokens=40_960,
    )
    assert [chunk async for chunk in model._astream([_Msg("hi")])] == ["a"]
    assert captured["window"] == 40_960


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


# ===================== the reasoning extraction hook (#554) =====================
#
# The second, disjoint override on ``Erudi_Chat_OpenAI``:
# ``_convert_chunk_to_generation_chunk`` re-attaches the dedicated reasoning
# field that both local servers stream (llama-server ``delta.reasoning_content``
# under the default ``--reasoning-format auto``; mlx_vlm ``delta.reasoning``,
# mirrored into ``reasoning_content`` on the pinned 0.6.17) and that upstream
# ``_convert_delta_to_message_chunk`` drops on the floor. The raw chunk shapes
# below are copied from the design-phase captures of both engines.


def _llama_reasoning_chunk(text):
    return {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "model": "erudi-model",
        "choices": [{"index": 0, "finish_reason": None, "delta": {"reasoning_content": text}}],
    }


def _mlx_reasoning_chunk(text):
    return {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "model": "/path/to/model",
        "choices": [
            {
                "index": 0,
                "finish_reason": None,
                "delta": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": text,
                    "reasoning": text,
                    "tool_calls": None,
                },
            }
        ],
    }


def _content_chunk(text, finish_reason=None):
    return {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "model": "erudi-model",
        "choices": [{"index": 0, "finish_reason": finish_reason, "delta": {"content": text}}],
    }


def _convert(raw):
    from langchain_core.messages import AIMessageChunk

    return _model()._convert_chunk_to_generation_chunk(raw, AIMessageChunk, {})


def test_the_conversion_hook_carries_llama_reasoning_onto_the_message_chunk():
    generation_chunk = _convert(_llama_reasoning_chunk("Thinking Process"))
    assert generation_chunk.message.additional_kwargs["reasoning_content"] == "Thinking Process"
    assert generation_chunk.message.content == ""


def test_the_conversion_hook_carries_mlx_reasoning_onto_the_message_chunk():
    generation_chunk = _convert(_mlx_reasoning_chunk("step one"))
    assert generation_chunk.message.additional_kwargs["reasoning_content"] == "step one"
    assert generation_chunk.message.content == ""


def test_a_content_chunk_gets_no_reasoning_kwarg():
    generation_chunk = _convert(_content_chunk("The answer"))
    assert "reasoning_content" not in generation_chunk.message.additional_kwargs
    assert generation_chunk.message.content == "The answer"


def test_the_conversion_hook_preserves_upstreams_none_result():
    # ``{"type": "content.delta"}`` is upstream's beta-stream sentinel: the base
    # method returns None and the override must not resurrect it.
    assert _convert({"type": "content.delta"}) is None


def test_finish_reason_still_lands_in_generation_info_through_the_override():
    """#554 relies on finish_reason surviving WITHOUT stamping: the base method
    folds ``choice.finish_reason`` into ``generation_info``, and langchain-core's
    stream loop folds generation_info into the yielded message's
    ``response_metadata`` (where the runner reads it). Pin the first half here
    on the subclass; the runner tests pin the second half end to end."""
    generation_chunk = _convert(_content_chunk("", finish_reason="length"))
    assert generation_chunk.generation_info["finish_reason"] == "length"


async def test_a_late_first_chunk_still_raises_with_both_overrides_active(monkeypatch):
    """Composition pin: adding the #554 conversion hook must not loosen the
    #573 watchdog -- extraction happens INSIDE the budgeted stream."""
    from langchain_openai import ChatOpenAI

    _tiny_budgets(monkeypatch)

    async def _slow_parent(self, messages, *args, **kwargs):
        await asyncio.sleep(0.3)
        yield "never reached"

    monkeypatch.setattr(ChatOpenAI, "_astream", _slow_parent)

    subclass = erudi_chat_openai_class()
    # Both behaviours live on the same class, on two disjoint hooks.
    assert subclass._astream is not ChatOpenAI._astream
    assert (
        subclass._convert_chunk_to_generation_chunk
        is not ChatOpenAI._convert_chunk_to_generation_chunk
    )

    with pytest.raises(GenerationTimeoutException) as excinfo:
        async for _ in _model()._astream([_Msg("hi")]):
            pass

    assert excinfo.value.phase == PHASE_FIRST_CHUNK


def test_chat_openai_still_exposes_the_chunk_conversion_hook_we_override():
    """Pinned upstream assumption (langchain-openai 1.2.2): the sync
    ``_convert_chunk_to_generation_chunk(self, chunk, default_chunk_class,
    base_generation_info)`` is where every streamed Chat Completions chunk is
    converted. A bump that renames or reshapes it would silently drop the
    reasoning again -- fail HERE instead."""
    from langchain_openai import ChatOpenAI

    hook = ChatOpenAI._convert_chunk_to_generation_chunk
    assert callable(hook)
    assert not inspect.iscoroutinefunction(hook)
    assert not inspect.isasyncgenfunction(hook)
    parameters = list(inspect.signature(hook).parameters)
    assert parameters[:4] == ["self", "chunk", "default_chunk_class", "base_generation_info"]
