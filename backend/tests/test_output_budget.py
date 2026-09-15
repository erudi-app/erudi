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


# ===================== the exact-count retry =====================
#
# mlx_vlm.server validates `prompt + max_tokens <= window` BEFORE generating
# and answers 400 when it does not hold (`_check_configured_context_budget`).
# It counts the REAL tokenised prompt, which chars/4 under-counts threefold on
# CJK -- far past what the margin absorbs -- so a budget sized from the
# estimate can make the app reject its own turn.
#
# The fix is precision, not pessimism: that 400 NAMES the exact prompt count,
# so the call is retried ONCE with a budget computed from it. Capping the
# budget with the watchdog's byte bound instead would be provably safe but
# costs English dearly (the bound over-counts it fourfold: a turn of ~8000
# real tokens in a 32k window would drop from a ~24000-token budget to the
# 512 floor, silently). The retry costs nothing on English, which never
# triggers the 400, and one instant local round-trip on the rare turn that
# does. The estimators stay in their lane: chars/4 sizes, bytes bound the
# watchdog, and the SERVER supplies the only exact number anyone has.

_CJK = "这是一个很长的中文句子，用来测试预算的计算方式。" * 40

_MLX_400 = (
    "Error code: 400 - Request needs {needed} context tokens "
    "({prompt} prompt + {generation} max generation), but MAX_KV_SIZE is {window}."
)


class _PreflightRejection(Exception):
    """The mlx_vlm.server 400 as the openai client surfaces it."""

    def __init__(self, prompt, generation, window):
        super().__init__(
            _MLX_400.format(
                needed=prompt + generation, prompt=prompt, generation=generation, window=window
            )
        )


def _retry_budget(exc, window):
    from src.agents.chat_model import preflight_retry_budget

    return preflight_retry_budget(exc, window)


def test_the_retry_budget_comes_from_the_servers_own_prompt_count():
    from src.agents.chat_model import PREFLIGHT_RETRY_MARGIN_TOKENS

    window = 32768

    budget = _retry_budget(
        _PreflightRejection(prompt=30000, generation=5000, window=window), window
    )

    assert budget == window - 30000 - PREFLIGHT_RETRY_MARGIN_TOKENS
    assert 30000 + budget <= window


def test_the_retry_budget_can_fall_below_the_normal_floor():
    # A tiny honest budget beats an error: the turn still answers, briefly.
    window = 4096

    budget = _retry_budget(_PreflightRejection(prompt=4000, generation=900, window=window), window)

    assert 1 <= budget < OUTPUT_BUDGET_FLOOR_TOKENS
    assert 4000 + budget <= window


def test_a_genuine_overflow_is_not_retried():
    # The prompt ALONE fills the window: no budget makes this request fit, and
    # the runner's curated overflow turn is the honest answer.
    window = 4096

    assert (
        _retry_budget(_PreflightRejection(prompt=4096, generation=8, window=window), window) is None
    )
    assert (
        _retry_budget(_PreflightRejection(prompt=9000, generation=8, window=window), window) is None
    )


def test_an_unrelated_failure_is_not_retried():
    assert _retry_budget(Exception("connection reset by peer"), 32768) is None


def test_the_llama_overflow_shape_is_not_retried():
    # llama-server clamps `n_predict` instead of rejecting on this axis, so a
    # 400 from it is a genuine overflow, never a budget miscount.
    class _Llama(Exception):
        body = {"type": "exceed_context_size_error", "n_prompt_tokens": 9030, "n_ctx": 8192}

    assert _retry_budget(_Llama("Error code: 400"), 8192) is None


def test_no_window_means_no_retry():
    assert (
        _retry_budget(_PreflightRejection(prompt=30000, generation=5000, window=32768), None)
        is None
    )


async def test_a_cjk_turn_is_retried_once_with_the_exact_budget(monkeypatch):
    from langchain_openai import ChatOpenAI

    from src.agents.chat_model import PREFLIGHT_RETRY_MARGIN_TOKENS

    window = 32768
    real_prompt = 30000
    attempts = []

    async def _server(self, messages, *args, **kwargs):
        attempts.append(kwargs.get("max_tokens"))
        if real_prompt + kwargs["max_tokens"] > window:
            raise _PreflightRejection(real_prompt, kwargs["max_tokens"], window)
        yield "answer"

    monkeypatch.setattr(ChatOpenAI, "_astream", _server)
    client = _client(max_tokens=1234, effective_context_tokens=window)

    chunks = [c async for c in client._astream([HumanMessage(_CJK)])]

    assert chunks == ["answer"]
    assert len(attempts) == 2, "exactly one retry"
    assert attempts[0] == compute_output_budget([HumanMessage(_CJK)], window)
    assert attempts[1] == window - real_prompt - PREFLIGHT_RETRY_MARGIN_TOKENS


async def test_an_english_turn_is_never_retried(monkeypatch):
    from langchain_openai import ChatOpenAI

    window = 32768
    real_prompt = 8000
    attempts = []

    async def _server(self, messages, *args, **kwargs):
        attempts.append(kwargs.get("max_tokens"))
        if real_prompt + kwargs["max_tokens"] > window:
            raise _PreflightRejection(real_prompt, kwargs["max_tokens"], window)
        yield "answer"

    monkeypatch.setattr(ChatOpenAI, "_astream", _server)
    client = _client(max_tokens=1234, effective_context_tokens=window)

    chunks = [c async for c in client._astream([HumanMessage(_text(8000))])]

    assert chunks == ["answer"]
    assert len(attempts) == 1, "chars/4 is accurate on English: nothing to correct"


async def test_a_genuine_overflow_reaches_the_caller_untouched(monkeypatch):
    from langchain_openai import ChatOpenAI

    window = 4096
    attempts = []

    async def _server(self, messages, *args, **kwargs):
        attempts.append(kwargs.get("max_tokens"))
        raise _PreflightRejection(prompt=9000, generation=kwargs["max_tokens"], window=window)
        yield  # pragma: no cover

    monkeypatch.setattr(ChatOpenAI, "_astream", _server)
    client = _client(max_tokens=1234, effective_context_tokens=window)

    with pytest.raises(_PreflightRejection) as raised:
        async for _ in client._astream([HumanMessage(_CJK)]):
            pass

    assert len(attempts) == 1, "a prompt that alone overflows is never retried"
    # The runner parses THIS exception into its curated overflow turn.
    from src.agents.overflow import parse_context_overflow

    assert parse_context_overflow(raised.value).context_tokens == window


async def test_a_rejection_after_the_first_chunk_is_never_retried(monkeypatch):
    # Retrying mid-stream would replay text the user has already seen.
    from langchain_openai import ChatOpenAI

    attempts = []

    async def _server(self, messages, *args, **kwargs):
        attempts.append(kwargs.get("max_tokens"))
        yield "partial"
        raise _PreflightRejection(prompt=30000, generation=5000, window=32768)

    monkeypatch.setattr(ChatOpenAI, "_astream", _server)
    client = _client(max_tokens=1234, effective_context_tokens=32768)

    seen = []
    with pytest.raises(_PreflightRejection):
        async for chunk in client._astream([HumanMessage(_CJK)]):
            seen.append(chunk)

    assert seen == ["partial"]
    assert len(attempts) == 1


async def test_the_retry_gets_its_own_first_chunk_budget(monkeypatch):
    """The watchdog clock restarts on the retry, deliberately.

    The rejection arrives from the preflight BEFORE any prefill, so the first
    attempt consumed effectively none of its budget; the retry is the attempt
    that actually prefills, and it must get the full budget its prompt size
    earns. Sharing one clock would charge the retry for a wait that never
    happened.
    """
    from langchain_openai import ChatOpenAI

    from src.agents import chat_model as chat_model_module

    budgets = []
    real = chat_model_module.first_chunk_budget_s

    def _spy(estimated, effective_window_tokens=None):
        budgets.append(real(estimated, effective_window_tokens))
        return budgets[-1]

    monkeypatch.setattr(chat_model_module, "first_chunk_budget_s", _spy)

    window = 32768
    attempts = []

    async def _server(self, messages, *args, **kwargs):
        attempts.append(kwargs.get("max_tokens"))
        if len(attempts) == 1:
            raise _PreflightRejection(prompt=30000, generation=kwargs["max_tokens"], window=window)
        yield "answer"

    monkeypatch.setattr(ChatOpenAI, "_astream", _server)
    client = _client(max_tokens=1234, effective_context_tokens=window)

    assert [c async for c in client._astream([HumanMessage(_CJK)])] == ["answer"]
    assert len(budgets) == 2 and budgets[0] == budgets[1]


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


class _Engine:
    """Engine stub: a handle, an MLX-style payload model value, no preflight."""

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


def test_the_factory_opts_a_client_out_of_the_budget(monkeypatch):
    from src.agents.model_factory import build_chat_model
    from src.core import config

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
