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
    output_budget_override,
)
from src.agents.token_accounting import request_tokens_est as estimate_prompt_tokens

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


async def test_an_operator_pin_is_never_substituted_by_the_retry(monkeypatch):
    # ERUDI_MAX_TOKENS exists to reproduce EXACT budgets: if the pinned value
    # overflows the preflight, the honest answer is the overflow error, never
    # a silently different budget than the one the operator pinned.
    from langchain_openai import ChatOpenAI

    window = 32768
    attempts = []

    async def _server(self, messages, *args, **kwargs):
        attempts.append(kwargs.get("max_tokens"))
        raise _PreflightRejection(prompt=30000, generation=kwargs["max_tokens"], window=window)
        yield  # pragma: no cover

    monkeypatch.setenv("ERUDI_MAX_TOKENS", "30000")
    monkeypatch.setattr(ChatOpenAI, "_astream", _server)
    client = _client(max_tokens=1234, effective_context_tokens=window)

    with pytest.raises(_PreflightRejection):
        _ = [c async for c in client._astream([HumanMessage(_CJK)])]

    assert attempts == [30000], "no retry: the pin reached the wire once and the 400 re-raised"


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


def test_one_estimator_for_the_stamp_the_counter_and_the_budget():
    """chars/4 through ``count_tokens_approximately`` with its defaults: the
    budget, the request stamp and the compaction counter all read this one
    function, so the three can never disagree about the unscaled size."""
    import inspect

    from langchain_core.messages import AIMessage
    from langchain_core.messages.utils import count_tokens_approximately

    from src.agents import chat_model, output_budget

    messages = [_Msg("hello world"), AIMessage("and a second turn")]

    assert estimate_prompt_tokens(messages) == count_tokens_approximately(messages)
    assert "real_tokens_est" in inspect.getsource(output_budget.estimate_prompt)
    assert "estimate_prompt" in inspect.getsource(output_budget.compute_output_budget)
    assert "request_tokens_est" in inspect.getsource(chat_model)


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


# ===================== working window vs. allocated window (PR3.1) =====================
#
# The output budget is a MEMORY consumer: it is sized from the ONE canonical
# working window (min of the allocated window and the memory ceiling), NOT the
# raw allocated window. The watchdog and the preflight retry are TIME/BOUND
# consumers and keep reading the raw allocated window. These pin that split on
# a single client that carries BOTH values.


async def test_the_budget_prefers_the_working_window_over_the_allocated_window(monkeypatch):
    from langchain_openai import ChatOpenAI

    captured: dict = {}

    async def _capture(self, messages, *args, **kwargs):
        captured.update(kwargs)
        yield "chunk"

    monkeypatch.setattr(ChatOpenAI, "_astream", _capture)
    messages = [HumanMessage(_text(200))]
    client = _client(max_tokens=1234, effective_context_tokens=32768, working_context_tokens=8000)

    assert [c async for c in client._astream(messages)] == ["chunk"]
    # Sized from the 8000 working window, NOT the 32768 allocation.
    assert captured["max_tokens"] == compute_output_budget(messages, 8000)
    assert captured["max_tokens"] != compute_output_budget(messages, 32768)


async def test_the_budget_falls_back_to_the_allocated_window_without_a_working_window(monkeypatch):
    # A client built with only the allocated window stamped still budgets, from
    # that window -- identical to canonical_working_window(allocated, None).
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


async def test_the_watchdog_reads_the_allocated_window_not_the_working_one(monkeypatch):
    # #573 ceiling scales with the PHYSICAL prefill the child must do, which is
    # the raw allocation -- a memory-shrunk working window must not shorten it.
    from langchain_openai import ChatOpenAI

    from src.agents import chat_model as chat_model_module

    seen: dict = {}

    def _spy(estimated, effective_window_tokens=None):
        seen["window"] = effective_window_tokens
        return 0.3

    monkeypatch.setattr(chat_model_module, "first_chunk_budget_s", _spy)

    async def _fast(self, messages, *args, **kwargs):
        yield "a"

    monkeypatch.setattr(ChatOpenAI, "_astream", _fast)
    client = _client(max_tokens=1234, effective_context_tokens=32768, working_context_tokens=8000)

    assert [c async for c in client._astream([_Msg("hi")])] == ["a"]
    assert seen["window"] == 32768


async def test_the_preflight_retry_reads_the_allocated_window_not_the_working_one(monkeypatch):
    # The retry fits `prompt + max_tokens` against the child's REAL window, the
    # raw allocation -- sizing the retry from the smaller working window would
    # hand back a budget below what the child can actually take.
    from langchain_openai import ChatOpenAI

    from src.agents import chat_model as chat_model_module

    window = 32768
    seen: dict = {}
    real = chat_model_module.preflight_retry_budget

    def _spy(exc, window_tokens):
        seen["window"] = window_tokens
        return real(exc, window_tokens)

    monkeypatch.setattr(chat_model_module, "preflight_retry_budget", _spy)

    async def _server(self, messages, *args, **kwargs):
        if not seen.get("raised"):
            seen["raised"] = True
            raise _PreflightRejection(prompt=30000, generation=kwargs["max_tokens"], window=window)
        yield "answer"

    monkeypatch.setattr(ChatOpenAI, "_astream", _server)
    client = _client(max_tokens=1234, effective_context_tokens=window, working_context_tokens=8000)

    assert [c async for c in client._astream([HumanMessage(_CJK)])] == ["answer"]
    assert seen["window"] == window


def test_the_factory_stamps_both_the_allocated_and_working_windows(monkeypatch):
    from types import SimpleNamespace

    from src.agents.model_factory import build_chat_model
    from src.core import config
    from src.engines import working_window as ww

    class _WindowedEngine(_Engine):
        @staticmethod
        def effective_context_tokens():
            return 40_960

    monkeypatch.setattr(config, "LLM_Engine", _WindowedEngine)
    # Force a memory ceiling below the allocation: the working window folds to
    # it, the allocated window is left untouched.
    monkeypatch.setattr(
        ww.MemoryBudget,
        "from_engine",
        staticmethod(lambda engine: SimpleNamespace(tokens_at_ceiling=lambda: 8000)),
    )

    chat = build_chat_model(_Llm(), temperature=0.3, top_p=0.8, max_tokens=55)

    assert chat.effective_context_tokens == 40_960
    assert chat.working_context_tokens == 8000


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


# ===================== the budget in real tokens (plan 3.2b v17) =====================
#
# The prompt is costed message by message (``real_tokens_est``): what the
# server measured at its measured ratio, its answers at their exact output
# tokens, everything new at the script weight of its own text (CJK 0.65 token
# per character, no floor), the tool schemas included. The margin applies to
# the ESTIMATED part only. The live cases are pinned in
# ``test_live_token_accounting.py``.


def _stamped_hop(input_tokens, est, *, first=True, output_tokens=5, content="", **kwargs):
    from langchain_core.messages import AIMessage

    return AIMessage(
        content=content,
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        response_metadata={
            "erudi_request_est": est,
            "erudi_request_has_images": False,
            "erudi_request_first_hop": first,
        },
        **kwargs,
    )


def test_a_request_with_tool_schemas_gets_a_smaller_budget():
    from langchain_core.tools import tool

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the documents of the knowledge base for the passages that answer it."""
        return ""

    messages = [_Msg(_text(500))]

    assert compute_output_budget(messages, 8192, tools=[search_knowledge_base]) < (
        compute_output_budget(messages, 8192)
    )


def test_a_first_turn_english_paste_keeps_todays_budget():
    """Nothing measured yet: English prose weighs 1.0, so the budget is the
    chars/4 one -- with the compaction's floor (1.2) it would be ~1000
    tokens smaller."""
    messages = [_Msg(_text(4800))]
    est = estimate_prompt_tokens(messages)

    budget = compute_output_budget(messages, 8192)

    assert budget == 8192 - est - int(MARGIN_FRACTION * est)
    assert budget > 2800


def test_a_first_turn_cjk_paste_is_sized_at_the_budgets_density():
    from src.agents.token_accounting import (
        BUDGET_DENSE_TOKENS,
        estimate,
        script_weight,
    )

    messages = [_Msg(_CJK)]
    weight = script_weight(_CJK, dense=BUDGET_DENSE_TOKENS)
    assert 2.0 <= weight <= 2.6
    prompt = -(-weight * estimate(messages[0]) // 1)

    budget = compute_output_budget(messages, 32768)

    assert budget == 32768 - prompt - max(MARGIN_FLOOR_TOKENS, int(MARGIN_FRACTION * prompt))
    assert budget < compute_output_budget([_Msg(_text(len(_CJK) // 4))], 32768)


def test_the_margin_applies_to_the_estimated_part_only():
    from src.agents.token_accounting import BUDGET_DENSE_TOKENS, real_tokens_est

    looping = _stamped_hop(7926, 6701, output_tokens=24_778, content="z" * 100_000)
    messages = [_Msg("summary " * 200), looping, _Msg("next question")]
    prompt = real_tokens_est(messages, dense=BUDGET_DENSE_TOKENS)

    budget = compute_output_budget(messages, 32768)

    assert prompt.exact > 24_778
    assert budget == 32768 - prompt.total - max(256, int(0.10 * (prompt.total - prompt.exact)))


async def test_the_stream_budgets_the_request_as_sent_with_its_tools(monkeypatch):
    from langchain_core.messages import ToolMessage
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from langchain_openai import ChatOpenAI

    captured: dict = {}

    async def _capture(self, messages, *args, **kwargs):
        captured.update(kwargs)
        yield "chunk"

    def web_search(query: str) -> str:
        """Search the web for fresh facts about the query and return snippets."""
        return ""

    monkeypatch.setattr(ChatOpenAI, "_astream", _capture)
    tools = [convert_to_openai_tool(web_search)]
    messages = [
        HumanMessage(_text(300)),
        _stamped_hop(2000, 1000, tool_calls=[{"name": "web_search", "args": {}, "id": "c1"}]),
        ToolMessage(_text(2000), tool_call_id="c1"),
    ]
    client = _client(max_tokens=1234, effective_context_tokens=32768)

    assert [c async for c in client._astream(messages, tools=tools)] == ["chunk"]
    assert captured["max_tokens"] == compute_output_budget(messages, 32768, tools=tools)
    assert captured["max_tokens"] < compute_output_budget(messages, 32768)


async def test_the_tools_reach_the_budget_through_create_agents_tool_binding(monkeypatch):
    from langchain.agents import create_agent
    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGenerationChunk
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search the documents of the knowledge base."""
        return ""

    seen: dict = {}

    async def _server(self, messages, *args, **kwargs):
        seen["messages"] = list(messages)
        seen["kwargs"] = dict(kwargs)
        yield ChatGenerationChunk(message=AIMessageChunk(content="done"))

    monkeypatch.setattr(ChatOpenAI, "_astream", _server)
    client = _client(max_tokens=1234, effective_context_tokens=8192, streaming=True)
    agent = create_agent(client, tools=[search_knowledge_base], system_prompt="be brief")

    await agent.ainvoke({"messages": [HumanMessage(_text(200))]})

    assert seen["kwargs"]["tools"], "the request carried the tool schemas"
    assert seen["kwargs"]["max_tokens"] == compute_output_budget(
        seen["messages"], 8192, tools=seen["kwargs"]["tools"]
    )


# ===================== the preflight retry, per client =====================


async def test_the_summary_client_is_never_retried_with_a_smaller_cap(monkeypatch):
    """The retry substitutes a smaller ``max_tokens``; on the summary client
    (capped at ``summary_cap(W)``) that would silently truncate the summary.
    Its rejection reaches the summarizer instead, which takes the size path."""
    from langchain_openai import ChatOpenAI

    window = 8192
    attempts = []

    async def _server(self, messages, *args, **kwargs):
        attempts.append(kwargs.get("max_tokens"))
        raise _PreflightRejection(prompt=8000, generation=1024, window=window)
        yield  # pragma: no cover

    monkeypatch.setattr(ChatOpenAI, "_astream", _server)
    client = _client(
        max_tokens=1024,
        effective_context_tokens=window,
        auto_output_budget=False,
        preflight_retry=False,
    )

    with pytest.raises(_PreflightRejection):
        _ = [c async for c in client._astream([HumanMessage(_CJK)])]

    assert len(attempts) == 1


async def test_a_giant_first_message_still_gets_a_title_through_the_retry(monkeypatch):
    from langchain_openai import ChatOpenAI

    window = 8192
    real_prompt = 8185
    attempts = []

    async def _server(self, messages, *args, **kwargs):
        attempts.append(kwargs.get("max_tokens"))
        if real_prompt + (kwargs.get("max_tokens") or 12) > window:
            raise _PreflightRejection(real_prompt, kwargs.get("max_tokens") or 12, window)
        yield "A title"

    monkeypatch.setattr(ChatOpenAI, "_astream", _server)
    client = _client(max_tokens=12, effective_context_tokens=window, auto_output_budget=False)

    assert client.preflight_retry is True
    assert [c async for c in client._astream([HumanMessage(_CJK)])] == ["A title"]
    assert len(attempts) == 2


def test_the_factory_streams_usage_and_carries_the_kb_text_and_the_retry_flag(monkeypatch):
    from src.agents.model_factory import build_chat_model
    from src.core import config

    monkeypatch.setattr(config, "LLM_Engine", _Engine)

    default = build_chat_model(_Llm(), temperature=0.3, top_p=0.8, max_tokens=55)
    assert default.stream_usage is True
    assert default._should_stream_usage() is True
    assert default.kb_additions == ""
    assert default.preflight_retry is True

    summary = build_chat_model(
        _Llm(),
        temperature=0.3,
        top_p=0.8,
        max_tokens=55,
        kb_additions="BLOCK\n\n\n\nAnswer in English.",
        preflight_retry=False,
    )
    assert summary.kb_additions == "BLOCK\n\n\n\nAnswer in English."
    assert summary.preflight_retry is False
