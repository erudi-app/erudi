"""The automatic output budget: what a model call is allowed to generate.

There is no user-facing Max Tokens control any more. Every model call is given
``max(512, W_eff - est_prompt - max(256, 10 % of est_prompt))`` tokens, so the
ceiling IS the context window: what the window still has free once the turn's
prompt is in it, minus a margin.

These tests pin the arithmetic (floor, margin, window, override) and the one
non-obvious design decision behind it -- that this module and
``src.agents.chat_model`` estimate the SAME prompt with two DIFFERENT
estimators on purpose, because their failure costs are opposite. The last
section is that pin.
"""

import pytest
from langchain_core.messages import HumanMessage

from src.agents.chat_model import estimate_prompt_tokens as byte_bound_estimate
from src.agents.output_budget import (
    MARGIN_FLOOR_TOKENS,
    MARGIN_FRACTION,
    MAX_TOKENS_ENV_VAR,
    OUTPUT_BUDGET_FLOOR_TOKENS,
    compute_output_budget,
    estimate_prompt_tokens,
    output_budget_override,
)

pytestmark = pytest.mark.unit


def _Msg(content):
    """One real user message -- the shape ``_astream`` is always handed."""
    return HumanMessage(content=content)


def _text(tokens: int) -> str:
    """ASCII text the chars/4 estimator counts as roughly ``tokens`` tokens."""
    return "a" * (tokens * 4)


# ===================== the constants =====================


def test_the_constants_are_the_decided_ones():
    assert OUTPUT_BUDGET_FLOOR_TOKENS == 512
    assert MARGIN_FLOOR_TOKENS == 256
    assert MARGIN_FRACTION == 0.10


# ===================== the formula =====================


def test_a_roomy_window_yields_the_whole_remainder_minus_the_margin():
    messages = [_Msg(_text(1000))]
    est = estimate_prompt_tokens(messages)

    budget = compute_output_budget(messages, 32768)

    assert budget == 32768 - est - max(MARGIN_FLOOR_TOKENS, int(MARGIN_FRACTION * est))


def test_a_small_prompt_pays_the_flat_margin_not_the_percentage():
    messages = [_Msg(_text(100))]  # 10 % of 100 is far below the 256 floor
    est = estimate_prompt_tokens(messages)

    assert compute_output_budget(messages, 8192) == 8192 - est - MARGIN_FLOOR_TOKENS


def test_a_large_prompt_pays_the_percentage_not_the_flat_margin():
    messages = [_Msg(_text(10_000))]  # 10 % of 10000 is far above the 256 floor
    est = estimate_prompt_tokens(messages)
    assert int(MARGIN_FRACTION * est) > MARGIN_FLOOR_TOKENS

    assert compute_output_budget(messages, 131_072) == 131_072 - est - int(MARGIN_FRACTION * est)


def test_a_window_the_prompt_nearly_fills_still_gets_the_floor():
    # Remainder alone would be negative; the floor is what the model gets, and
    # the engine's own overflow handling (llama-server truncates n_predict, MLX
    # rejects the request) owns what happens next -- never a zero-token call.
    messages = [_Msg(_text(4000))]

    assert compute_output_budget(messages, 4096) == OUTPUT_BUDGET_FLOOR_TOKENS


def test_an_exhausted_window_never_goes_negative():
    assert compute_output_budget([_Msg(_text(100_000))], 2048) == OUTPUT_BUDGET_FLOOR_TOKENS


def test_the_budget_grows_with_the_window_for_the_same_prompt():
    messages = [_Msg(_text(500))]

    assert compute_output_budget(messages, 131_072) > compute_output_budget(messages, 8192)


def test_the_budget_shrinks_as_the_conversation_grows():
    short = compute_output_budget([_Msg(_text(100))], 32768)
    long = compute_output_budget([_Msg(_text(10_000))], 32768)

    assert long < short


def test_there_is_no_fixed_ceiling_the_window_is_the_ceiling():
    # A million-token window hands out a million-token budget: runaway
    # generation is a runtime-intervention problem, not a budget one.
    budget = compute_output_budget([_Msg("hi")], 1_000_000)

    assert budget > 900_000


# ===================== no window reported =====================


def test_an_unreported_window_yields_no_budget():
    # None means "the engine could not tell us": the caller keeps whatever
    # max_tokens it already resolved, so behaviour is unchanged.
    assert compute_output_budget([_Msg("hi")], None) is None


def test_a_nonsensical_window_yields_no_budget():
    assert compute_output_budget([_Msg("hi")], 0) is None
    assert compute_output_budget([_Msg("hi")], -1) is None


def test_a_message_the_counter_cannot_read_costs_the_budget_not_the_turn():
    # The chars/4 counter coerces every message and raises on a shape it does
    # not know -- unlike the watchdog's byte bound, which tolerates anything.
    # A budget is an optimisation over a working default: it degrades to "no
    # budget", never to a failed turn.
    class _Unreadable:
        pass

    assert compute_output_budget([_Unreadable()], 32768) is None


# ===================== the escape hatch =====================


def test_the_override_wins_over_the_computed_budget():
    assert compute_output_budget([_Msg(_text(100))], 32768, override=64) == 64


def test_the_override_wins_even_with_no_window():
    assert compute_output_budget([_Msg("hi")], None, override=64) == 64


def test_the_override_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv(MAX_TOKENS_ENV_VAR, "4096")

    assert output_budget_override() == 4096


def test_no_environment_variable_means_no_override(monkeypatch):
    monkeypatch.delenv(MAX_TOKENS_ENV_VAR, raising=False)

    assert output_budget_override() is None


@pytest.mark.parametrize("value", ["", "   ", "abc", "0", "-5", "1.5"])
def test_an_unusable_override_is_ignored(monkeypatch, value):
    monkeypatch.setenv(MAX_TOKENS_ENV_VAR, value)

    assert output_budget_override() is None


# ===================== the estimator duality (pinned on purpose) =====================


def test_the_two_estimators_disagree_on_cjk_which_is_the_whole_point():
    """chars/4 here, one-token-per-UTF-8-byte in the watchdog -- deliberately.

    The two estimates have OPPOSITE failure costs. The watchdog needs an UPPER
    bound (under-counting kills a healthy turn mid-prefill, #573), and CJK is
    where a character heuristic under-counts worst: ~1 token per character, 3
    UTF-8 bytes each. The budget needs the counter the summarization middleware
    already uses, so the compaction trigger and the budget can never disagree
    about how full the window is -- and there under-counting is nearly free
    (llama-server truncates n_predict server-side, MLX's preflight has the
    margin). If these two ever return the same number, one of the modules has
    silently adopted the other's estimator and this reasoning is gone.
    """
    cjk = [_Msg("这是一个很长的中文句子" * 50)]

    assert byte_bound_estimate(cjk) > 3 * estimate_prompt_tokens(cjk)


def test_the_two_estimators_stay_comparable_on_english():
    # The duality is about CJK, not about a different order of magnitude
    # everywhere: on English the byte bound is only ~4x the chars/4 count.
    english = [_Msg("the quick brown fox jumps over the lazy dog " * 50)]

    assert byte_bound_estimate(english) < 6 * estimate_prompt_tokens(english)


def test_the_budget_estimator_is_the_summarization_middlewares_counter():
    from langchain_core.messages import AIMessage
    from langchain_core.messages.utils import count_tokens_approximately

    messages = [_Msg("hello world"), AIMessage("and a second turn")]

    assert estimate_prompt_tokens(messages) == count_tokens_approximately(messages)


# ===================== where the budget lands in the request =====================
#
# Pinned assumptions about langchain-openai 1.2.2. A bump that changes any of
# them fails here instead of silently sending a stale (or no) cap.


def _client(**kwargs):
    from src.agents.chat_model import erudi_chat_openai_class

    return erudi_chat_openai_class()(
        base_url="http://127.0.0.1:1/v1",
        api_key="not-needed",
        model="fake-model",
        **kwargs,
    )


def _payload(client, **kwargs):
    return client._get_request_payload([HumanMessage("hi")], stop=None, stream=True, **kwargs)


def test_pinned_a_max_tokens_kwarg_overrides_the_constructor_value():
    # `_get_request_payload` builds `{**self._default_params, **kwargs}`, so a
    # per-call kwarg wins over the field set at build time. This is what lets
    # `_astream` re-budget every hop of a tool turn.
    client = _client(max_tokens=1234)

    assert _payload(client)["max_tokens"] == 1234
    assert _payload(client, max_tokens=99)["max_tokens"] == 99


def test_pinned_the_cap_reaches_the_wire_as_max_tokens_not_max_completion_tokens():
    """Both local servers read ``max_tokens``; only one reads the modern name.

    Stock ``ChatOpenAI`` renames the cap to ``max_completion_tokens`` (OpenAI
    deprecated the old name in 2024). llama-server accepts both (its
    ``n_predict`` parameter aliases each), but mlx_vlm.server 0.6.17 reads only
    ``max_tokens`` -- and its request schema DEFAULTS that field, so the modern
    name is not rejected, it is silently replaced by the server's own default.
    ``Erudi_Chat_OpenAI`` therefore puts the cap back on the legacy key, which
    is the only one both children honour.
    """
    payload = _payload(_client(max_tokens=1234))

    assert payload["max_tokens"] == 1234
    assert "max_completion_tokens" not in payload


async def test_the_stream_budgets_the_call_from_the_window_and_the_messages(monkeypatch):
    from langchain_openai import ChatOpenAI

    captured: dict = {}

    async def _capture(self, messages, *args, **kwargs):
        captured.update(kwargs)
        yield "chunk"

    monkeypatch.setattr(ChatOpenAI, "_astream", _capture)
    messages = [HumanMessage(_text(200))]
    client = _client(max_tokens=1234, effective_context_tokens=32768)

    assert [c async for c in client._astream(messages)] == ["chunk"]
    assert captured["max_tokens"] == compute_output_budget(messages, 32768)
    assert captured["max_tokens"] != 1234


async def test_without_a_window_the_stream_keeps_the_resolved_value(monkeypatch):
    from langchain_openai import ChatOpenAI

    captured: dict = {}

    async def _capture(self, messages, *args, **kwargs):
        captured.update(kwargs)
        yield "chunk"

    monkeypatch.setattr(ChatOpenAI, "_astream", _capture)
    client = _client(max_tokens=1234, effective_context_tokens=None)

    assert [c async for c in client._astream([HumanMessage("hi")])] == ["chunk"]
    # No max_tokens kwarg at all: the constructor value stands, so an engine
    # that cannot report its window behaves exactly as it does today.
    assert "max_tokens" not in captured


async def test_a_client_with_a_deliberate_budget_keeps_it(monkeypatch):
    """One-shot utility calls own their budget; the window must not raise it.

    ``ainvoke`` on a ``streaming=True`` client routes through ``_astream`` too,
    so conversation titles (a ~12-token budget, #266) would otherwise be handed
    the whole window and ramble for thousands of tokens before the sanitizer
    took the first four words.
    """
    from langchain_openai import ChatOpenAI

    captured: dict = {}

    async def _capture(self, messages, *args, **kwargs):
        captured.update(kwargs)
        yield "chunk"

    monkeypatch.setattr(ChatOpenAI, "_astream", _capture)
    client = _client(max_tokens=12, effective_context_tokens=32768, auto_output_budget=False)

    assert [c async for c in client._astream([HumanMessage("hi")])] == ["chunk"]
    assert "max_tokens" not in captured


def test_the_factory_opts_a_client_out_of_the_budget(monkeypatch):
    from src.agents.model_factory import build_chat_model
    from src.core import config

    class _Engine:
        @staticmethod
        def get_model_and_tokenizer(llm_id, link):
            return ({"base_url": "http://127.0.0.1:8080", "alias": "a"}, {})

        @staticmethod
        def _payload_model_value(handle):
            return "m"

    class _Llm:
        id = 7
        link = "/fake"
        name = "Test"

    monkeypatch.setattr(config, "LLM_Engine", _Engine)

    assert build_chat_model(_Llm(), temperature=0.3, top_p=0.8, max_tokens=55).auto_output_budget
    assert not build_chat_model(
        _Llm(), temperature=0.3, top_p=0.8, max_tokens=12, auto_output_budget=False
    ).auto_output_budget


async def test_the_environment_override_wins_over_the_computed_budget(monkeypatch):
    from langchain_openai import ChatOpenAI

    captured: dict = {}

    async def _capture(self, messages, *args, **kwargs):
        captured.update(kwargs)
        yield "chunk"

    monkeypatch.setattr(ChatOpenAI, "_astream", _capture)
    monkeypatch.setenv(MAX_TOKENS_ENV_VAR, "77")
    client = _client(max_tokens=1234, effective_context_tokens=32768)

    assert [c async for c in client._astream([HumanMessage("hi")])] == ["chunk"]
    assert captured["max_tokens"] == 77


async def test_the_budget_composes_with_the_first_chunk_watchdog(monkeypatch):
    import asyncio

    from langchain_openai import ChatOpenAI

    from src.agents import chat_model as chat_model_module
    from src.core.exceptions import GenerationTimeoutException

    captured: dict = {}

    async def _silent(self, messages, *args, **kwargs):
        captured.update(kwargs)
        await asyncio.sleep(5)
        yield "never"

    monkeypatch.setattr(ChatOpenAI, "_astream", _silent)
    monkeypatch.setattr(chat_model_module, "FIRST_CHUNK_FLOOR_S", 0.03)
    monkeypatch.setattr(chat_model_module, "FIRST_CHUNK_BASE_S", 0.01)
    monkeypatch.setattr(chat_model_module, "FIRST_CHUNK_CEILING_S", 0.05)
    client = _client(max_tokens=1234, effective_context_tokens=32768)

    with pytest.raises(GenerationTimeoutException):
        async for _ in client._astream([HumanMessage("hi")]):
            pass
    # The budget was still applied to the call the watchdog then bounded.
    assert captured["max_tokens"] == compute_output_budget([HumanMessage("hi")], 32768)
