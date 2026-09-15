"""How many tokens a model call is allowed to generate, computed per call.

There is no Max Tokens control. ``max_tokens`` is a server-side guillotine the
model never sees -- it cannot make an answer shorter, only cut it mid-sentence
-- so asking the user to pick a number only gave them a way to truncate their
own answers. What replaces it is arithmetic:

    max_tokens = max(512, W_eff - est_prompt - max(256, 10 % of est_prompt))

- ``W_eff`` is the ALLOCATED context window of the loaded child
  (``BaseEngine.effective_context_tokens``, stamped on the chat client by the
  factory). **The window is the ceiling**: there is no fixed upper bound on
  top of it. A model that will not stop is a runtime problem -- cancel the
  turn -- not something a smaller number fixes, and every fixed cap ever
  chosen truncated a legitimate long answer somewhere.
- ``est_prompt`` is what the turn already occupies (see the estimator below).
- The margin covers what the estimate cannot see: the chat template's own
  tokens, tool schemas bound by the agent, a system prompt injected further
  down. Flat 256 tokens for short prompts, 10 % once the conversation is big
  enough that a percentage is the honest shape of the error.
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

Two estimators, two jobs -- and one of them does both
-----------------------------------------------------
``src.agents.chat_model.estimate_prompt_tokens`` bounds the messages with one
token per UTF-8 byte, and this module counts characters/4 through
``count_tokens_approximately``. That is not a duplication anyone forgot to
clean up -- the two estimates have OPPOSITE failure costs:

- The byte bound is a PROVABLE UPPER bound (byte-level tokenizers cannot emit
  a token per less than a byte). The watchdog needs one: under-counting there
  ends a healthy turn mid-prefill (#573), so it pays a loose over-count on
  English to stay honest on CJK.
- SIZING the budget needs the counter the summarization middleware already
  uses (``runner._build_middleware``), so the compaction trigger and the budget
  can never disagree about how full the window is. Using the byte bound to SIZE
  would over-count English ~4x and shrink real answer budgets by thousands of
  tokens -- a visible regression.

The byte bound has a second job here, though: a SAFETY CAP on the engine whose
own preflight counts ``prompt + max_tokens`` against the window and answers 400
when the sum does not fit (mlx_vlm.server's
``_check_configured_context_budget``). That preflight counts the REAL tokenised
prompt, so a budget sized from chars/4 can ask for more window than exists and
make the app reject its own turn -- badly on CJK, where chars/4 under-counts
threefold and the margin is nowhere near enough to absorb it. Capping the
budget at ``window - byte_bound`` makes that arithmetically impossible, because
``byte_bound >= real prompt`` by construction. It applies ONLY to the
preflighting engine: llama-server truncates ``n_predict`` server-side instead,
where the same cap would shrink long English budgets for nothing.

``tests/test_output_budget.py`` asserts the two estimators still disagree on a
CJK string, so neither can silently adopt the other's.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional

from langchain_core.messages.utils import count_tokens_approximately

from src.core.logging import logger

# Never hand a model a budget below this, however full the window is.
OUTPUT_BUDGET_FLOOR_TOKENS = 512

# The margin between the prompt and the end of the window: whichever of these
# two is larger.
MARGIN_FLOOR_TOKENS = 256
MARGIN_FRACTION = 0.10

# QA/dev escape hatch. Documented in backend/.env.example.
MAX_TOKENS_ENV_VAR = "ERUDI_MAX_TOKENS"


def estimate_prompt_tokens(messages: Optional[Iterable[Any]]) -> int:
    """Approximate tokens the outgoing messages occupy (chars/4).

    The summarization middleware's counter, on purpose -- see the module
    docstring. Unlike the watchdog's byte bound it is STRICT about shapes: it
    coerces every item and raises on one it cannot read, which is why
    :func:`compute_output_budget` treats a raised estimate as "no budget".
    """
    if not messages:
        return 0
    return count_tokens_approximately(messages)


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
    prompt_upper_bound_tokens: Optional[int] = None,
) -> Optional[int]:
    """Tokens this call may generate, or ``None`` to leave the caller's value.

    ``prompt_upper_bound_tokens`` is the PROVABLE upper bound on the prompt
    (the watchdog's byte bound), and passing it additionally caps the budget at
    ``window - bound``. Pass it when the engine's own preflight counts
    ``prompt + max_tokens`` against the window and rejects the request when the
    sum does not fit -- it is what keeps the app from rejecting its own turn on
    a prompt chars/4 under-counted. Leave it ``None`` where the server
    truncates instead (llama-server), so a long English turn keeps the full
    budget the formula gives it. See the module docstring.

    Pure: the environment is read by :func:`output_budget_override` and the
    bound by the caller (which already computes it for the watchdog), so the
    arithmetic stays testable on its own -- and this module never has to import
    back from ``chat_model``.
    """
    if override is not None:
        return override
    if not effective_window_tokens or effective_window_tokens <= 0:
        return None
    try:
        estimated = estimate_prompt_tokens(messages)
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
    margin = max(MARGIN_FLOOR_TOKENS, int(MARGIN_FRACTION * estimated))
    budget = effective_window_tokens - estimated - margin
    if prompt_upper_bound_tokens is not None:
        budget = min(budget, effective_window_tokens - prompt_upper_bound_tokens)
    # The floor stands even when the bound alone fills the window: a 0-token
    # call is never the right answer. The preflight then rejects only if the
    # REAL prompt plus 512 overflows, which is a genuinely full window -- and
    # reporting that honestly is the overflow chantier's job, not this one's.
    return max(OUTPUT_BUDGET_FLOOR_TOKENS, budget)
