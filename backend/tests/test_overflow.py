"""PR-G — parse_context_overflow: the two local engines' 400 wire shapes.

Pure module, no network/model dependencies: exercises the parser directly
against plain fake exceptions shaped like what each engine actually returns.
"""

import pytest

from src.agents.overflow import ContextOverflow, parse_context_overflow

pytestmark = pytest.mark.unit


class _FakeBadRequestError(Exception):
    """Stands in for ``openai.BadRequestError``: any object with a ``.body``
    dict reaches ``parse_context_overflow`` the same way."""

    def __init__(self, message: str, body: dict):
        super().__init__(message)
        self.body = body


def test_the_mlx_detail_also_yields_the_prompt_alone():
    # `prompt_tokens` carries what the REQUEST needed (prompt + generation),
    # which is what the user-facing message quotes. The prompt ALONE is a
    # different number and the one the output budget needs to recompute an
    # exact retry, so it is captured separately (PR-E).
    exc = Exception(
        "Request needs 5037 context tokens (5029 prompt + 8 max generation), "
        "but MAX_KV_SIZE is 4096."
    )

    overflow = parse_context_overflow(exc)

    assert overflow.prompt_tokens == 5037
    assert overflow.prompt_only_tokens == 5029
    assert overflow.context_tokens == 4096


def test_an_mlx_detail_without_the_breakdown_has_no_prompt_alone():
    exc = Exception("Something went wrong, but MAX_KV_SIZE is 4096.")

    overflow = parse_context_overflow(exc)

    assert overflow is not None
    assert overflow.prompt_only_tokens is None


def test_the_llama_shape_carries_no_prompt_alone():
    # llama-server clamps instead of rejecting on this axis, so nothing
    # downstream recomputes a budget from its body.
    exc = _FakeBadRequestError(
        "Error code: 400",
        body={"type": "exceed_context_size_error", "n_prompt_tokens": 9030, "n_ctx": 8192},
    )

    assert parse_context_overflow(exc).prompt_only_tokens is None


def test_llama_overflow_body_yields_both_numbers():
    # The PRODUCTION shape: the openai SDK unwraps the wire envelope before
    # storing it (openai/_client.py: ``data = body.get("error", body)``), so
    # ``exc.body`` is the FLAT error object, not ``{"error": {...}}``.
    exc = _FakeBadRequestError(
        "Error code: 400",
        body={
            "code": 400,
            "message": (
                "request (9030 tokens) exceeds the available context size "
                "(8192 tokens), try increasing it"
            ),
            "type": "exceed_context_size_error",
            "n_prompt_tokens": 9030,
            "n_ctx": 8192,
        },
    )

    overflow = parse_context_overflow(exc)

    assert overflow == ContextOverflow(prompt_tokens=9030, context_tokens=8192)


def test_llama_overflow_enveloped_body_still_matches():
    # Defensive fallback: a client path that skips the SDK's unwrap hands the
    # whole ``{"error": {...}}`` envelope. Both shapes must parse.
    exc = _FakeBadRequestError(
        "Error code: 400",
        body={
            "error": {
                "type": "exceed_context_size_error",
                "n_prompt_tokens": 9030,
                "n_ctx": 8192,
            }
        },
    )

    overflow = parse_context_overflow(exc)

    assert overflow == ContextOverflow(prompt_tokens=9030, context_tokens=8192)


def test_mlx_overflow_detail_yields_both_numbers():
    exc = _FakeBadRequestError(
        "Request needs 5037 context tokens (5029 prompt + 8 max generation), "
        "but MAX_KV_SIZE is 4096.",
        body={"error": "bad request"},
    )

    overflow = parse_context_overflow(exc)

    assert overflow == ContextOverflow(
        prompt_tokens=5037, context_tokens=4096, prompt_only_tokens=5029
    )


def test_mlx_overflow_unparseable_but_marked_yields_overflow_with_nones():
    # Wording drifted from the known template, but the MAX_KV_SIZE marker
    # (the stable part of the contract) is still there.
    exc = _FakeBadRequestError("Context window exceeded -- see MAX_KV_SIZE for details", body={})

    overflow = parse_context_overflow(exc)

    assert overflow == ContextOverflow(prompt_tokens=None, context_tokens=None)


def test_unrelated_400_returns_none():
    exc = _FakeBadRequestError(
        "Error code: 400",
        body={
            "error": {
                "code": 400,
                "message": "invalid 'temperature': must be between 0 and 2",
                "type": "invalid_request_error",
            }
        },
    )

    assert parse_context_overflow(exc) is None


def test_unrelated_exception_without_body_returns_none():
    assert parse_context_overflow(RuntimeError("connection reset")) is None


def test_non_exception_garbage_returns_none():
    assert parse_context_overflow(42) is None
    assert parse_context_overflow(None) is None
    assert parse_context_overflow(["not", "an", "exception"]) is None
