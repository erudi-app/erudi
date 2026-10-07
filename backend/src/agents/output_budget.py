"""How many tokens a model call is allowed to generate, computed per call.

There is no Max Tokens control. ``max_tokens`` is a server-side guillotine the
model never sees -- it cannot make an answer shorter, only cut it mid-sentence
-- so asking the user to pick a number only gave them a way to truncate their
own answers. What replaces it is arithmetic:

    max_tokens = max(512, W_eff - prompt - max(256, 10 % of (prompt - exact)))

- ``W_eff`` is the working window of the loaded child (the allocated window
  folded with the memory ceiling, ``src.engines.working_window``, stamped on
  the chat client by the factory). **The window is the ceiling**: there is no
  fixed upper bound on top of it. A model that will not stop is a runtime
  problem -- cancel the turn -- not something a smaller number fixes, and
  every fixed cap ever chosen truncated a legitimate long answer somewhere.
- ``prompt`` is what the request occupies in REAL tokens, costed message by
  message (``src.agents.token_accounting.real_tokens_est``): what the
  server already measured at its measured ratio, the answers it generated at
  their exact ``output_tokens``, everything new -- the question, a paste, the
  KB block, a tool result -- at the script weight of its own text, with the
  budget's settings (CJK at 0.65 token per character, no floor: here an
  over-estimate silently truncates the answer). The tool schemas the call
  carries are counted too. ``exact`` is the part costed from
  ``output_tokens``.
- The margin covers what the estimate still cannot see: the chat template's
  own tokens and the error left in the estimated part (never in the exact
  one: a 24.8k-token answer counted exactly does not eat 2.5k of margin).
  Flat 256 tokens for short prompts, 10 % of the estimated part once it is
  big enough that a percentage is the honest shape of the error.
- The 512-token floor keeps a window-filling turn from being handed a
  zero-token budget. What happens next belongs to the engine, which knows:
  llama-server truncates ``n_predict`` against its own remaining window, MLX's
  preflight validator answers 400 naming the exact budget.

``W_eff is None`` means the engine could not report a window. Then there is no
budget to compute and the caller keeps whatever it already resolved (the
conversation row's value, or the per-model fallback) -- an unreporting engine
must behave exactly as it did before this module existed.

``ERUDI_MAX_TOKENS`` wins over all of it, window or no window: a QA/dev escape
hatch for pinning a small budget while reproducing a truncation report.

One estimator for sizing, one bound for the watchdog
----------------------------------------------------
The budget, the request stamp (``Erudi_Chat_OpenAI._astream``) and the
compaction counter (``src.agents.runner``) all read ONE estimator,
``request_tokens_est``, so they can never disagree about the unscaled size.
``src.agents.chat_model.estimate_prompt_tokens`` stays a separate function on
purpose: it bounds the messages with one token per UTF-8 byte, a PROVABLE
UPPER bound the first-chunk watchdog needs (under-counting there ends a
healthy turn mid-prefill, #573). Sizing with it would over-count English
fourfold.

An under-estimate is still recovered: mlx_vlm.server validates
``prompt + max_tokens <= window`` against the REAL tokenised prompt and its
400 names the exact prompt count, so ``chat_model`` retries the call once with
a budget computed from it -- against the ALLOCATED window only; the memory
ceiling has no such net. llama-server truncates ``n_predict`` server-side.

``tests/test_output_budget.py`` asserts the two estimators still disagree on a
CJK string, so neither can silently adopt the other's.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional

from src.agents.token_accounting import BUDGET_DENSE_TOKENS, real_tokens_est
from src.core.logging import logger

# Never hand a model a budget below this, however full the window is.
OUTPUT_BUDGET_FLOOR_TOKENS = 512

# The margin between the prompt and the end of the window: whichever of these
# two is larger.
MARGIN_FLOOR_TOKENS = 256
MARGIN_FRACTION = 0.10

# QA/dev escape hatch. Documented in backend/.env.example.
MAX_TOKENS_ENV_VAR = "ERUDI_MAX_TOKENS"


def output_budget_override() -> Optional[int]:
    """``ERUDI_MAX_TOKENS`` as a positive int, or ``None`` when unusable.

    A value that is not a positive integer is ignored with one WARNING: the
    operator asked for something the app is not doing, and silence would send
    them hunting in the wrong place.
    """
    raw = os.getenv(MAX_TOKENS_ENV_VAR)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw.strip())
    except ValueError:
        value = 0
    if value <= 0:
        logger.warning(
            f"{MAX_TOKENS_ENV_VAR}={raw!r} is not a positive integer; "
            f"ignoring it and computing the output budget from the context window"
        )
        return None
    return value


def compute_output_budget(
    messages: Optional[Iterable[Any]],
    effective_window_tokens: Optional[int],
    override: Optional[int] = None,
    *,
    tools: Optional[Iterable[Any]] = None,
) -> Optional[int]:
    """Tokens this call may generate, or ``None`` to leave the caller's value.

    ``messages`` is the request as sent (the stamped AI messages keep their
    usage through the strippers and the fold) and ``tools`` the schemas it
    carries. Pure: the environment is read by :func:`output_budget_override`,
    which the caller passes in, so the arithmetic stays testable on its own.
    """
    if override is not None:
        return override
    if not effective_window_tokens or effective_window_tokens <= 0:
        return None
    try:
        prompt = real_tokens_est(list(messages or ()), dense=BUDGET_DENSE_TOKENS, tools=tools)
    except Exception:
        # The budget is an optimisation over a working default; it must never
        # be the reason a turn fails. A message shape the counter cannot read
        # costs the budget, not the answer: the caller keeps the max_tokens it
        # resolved, which is the behaviour from before this module existed.
        logger.warning(
            "Could not estimate the prompt size; this call keeps the resolved max_tokens",
            exc_info=True,
        )
        return None
    margin = max(MARGIN_FLOOR_TOKENS, int(MARGIN_FRACTION * (prompt.total - prompt.exact)))
    return max(OUTPUT_BUDGET_FLOOR_TOKENS, effective_window_tokens - prompt.total - margin)
