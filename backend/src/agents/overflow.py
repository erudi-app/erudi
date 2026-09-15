"""Context-overflow detection for the two local engines' wire errors (PR-G).

Both llama-server and mlx_vlm.server reject an over-budget prompt with an
HTTP 400 that reaches the LangChain client as an ``openai.BadRequestError``.
This module parses that specific failure out of the generic 400 space so the
runner can surface an honest, numbers-carrying turn instead of the catch-all
error sentinel -- never a silent truncation or shift of the conversation.

Two wire shapes (established evidence):

- llama-server: the error body is JSON,
  ``{"error": {"type": "exceed_context_size_error", "n_prompt_tokens": N,
  "n_ctx": W, ...}}``. Discriminate STRICTLY on ``type`` -- none of
  langchain's overflow-detection substrings match this message, and the
  message text itself is not a stable contract.
- mlx_vlm.server (spawned with the context-window chantier's
  ``--max-kv-size``): a detail string like ``"Request needs 5037 context
  tokens (5029 prompt + 8 max generation), but MAX_KV_SIZE is 4096."``,
  parsed defensively with regexes; an unparseable variant still counts as
  overflow when the ``MAX_KV_SIZE`` marker is present, since the marker is
  the stable part of the contract and the wording around it is not.

Import-light and duck-typed on purpose: this module never imports ``openai``
or ``langchain`` -- it reads ``exc.body`` (when present) and falls back to
``str(exc)``, so it stays a plain, fast, unit-testable module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_LLAMA_OVERFLOW_TYPE = "exceed_context_size_error"
_MLX_MARKER = "MAX_KV_SIZE"
_MLX_NEEDS_PATTERN = re.compile(r"Request needs (\d+) context tokens")
_MLX_LIMIT_PATTERN = re.compile(r"MAX_KV_SIZE is (\d+)")


@dataclass(frozen=True)
class ContextOverflow:
    """A parsed context-overflow failure.

    Either field may be ``None`` when the wire error matched (the failure IS
    a context overflow) but its numbers didn't parse -- callers degrade to a
    numberless message rather than treat that as "not an overflow".
    """

    prompt_tokens: Optional[int]
    context_tokens: Optional[int]


def parse_context_overflow(exc: object) -> Optional[ContextOverflow]:
    """``exc`` -> ``ContextOverflow`` iff it is one of the two known
    context-overflow wire shapes coming back from the local engines, else
    ``None``.

    Defensive by construction: never raises on a malformed or unexpected
    exception shape (bad ``.body``, non-exception input), and never matches
    the llama shape on message text alone -- only the ``type`` field is a
    stable contract there.
    """
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        overflow = _parse_llama_body(body)
        if overflow is not None:
            return overflow

    return _parse_mlx_detail(str(exc))


def _parse_llama_body(body: dict) -> Optional[ContextOverflow]:
    # The openai SDK UNWRAPS the wire envelope before storing it: for a body
    # of ``{"error": {...}}`` it sets ``exc.body`` to the INNER error object
    # (openai/_client.py: ``data = body.get("error", body)``). So the common
    # shape here is the flat error dict; the enveloped form is kept as a
    # defensive fallback for any client path that skips the unwrap.
    error = body if body.get("type") == _LLAMA_OVERFLOW_TYPE else body.get("error")
    if not isinstance(error, dict):
        return None
    if error.get("type") != _LLAMA_OVERFLOW_TYPE:
        return None
    return ContextOverflow(
        prompt_tokens=_as_int(error.get("n_prompt_tokens")),
        context_tokens=_as_int(error.get("n_ctx")),
    )


def _parse_mlx_detail(text: str) -> Optional[ContextOverflow]:
    if _MLX_MARKER not in text:
        return None
    needs_match = _MLX_NEEDS_PATTERN.search(text)
    limit_match = _MLX_LIMIT_PATTERN.search(text)
    return ContextOverflow(
        prompt_tokens=_as_int(needs_match.group(1)) if needs_match else None,
        context_tokens=_as_int(limit_match.group(1)) if limit_match else None,
    )


def _as_int(value: object) -> Optional[int]:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
