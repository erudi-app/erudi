"""The ONE canonical working context window (PR3.1).

Two "context window" numbers serve different consumers:

* the raw ALLOCATED window (``BaseEngine.effective_context_tokens`` -- the
  window the loaded child actually runs with), a TIME/BOUND fact: the
  first-chunk watchdog ceiling and the preflight retry are sized from it, and a
  prefill-timeout log quotes it. Those consumers must keep reading it RAW.
* a MEMORY-bounded value the output budget and the compaction trigger share,
  composed in one place so the two cannot disagree. The compaction trigger is
  a share of it (the conversation's compaction threshold, ``trigger_tokens``).

This module owns the single MEMORY value the memory consumers share:

    working_window = min(allocated window, memory ceiling)

taken over the candidates that are actually KNOWN. It is deliberately SEPARATE
from the allocated window: the time/bound consumers keep the raw window, the
memory consumers (output budget, compaction) take this one.

It also owns the bound the MLX child ENFORCES (``kv_bound``): when the memory
ceiling (or the memory brake's cap, ``src.engines.memory_brake``) is the
binding side of the min, the engine lowers the child's live ``max_kv_size`` to
it once the child's base footprint is measured, and that bound then IS the
child's allocated window.

No import cycle: this module imports ``memory_budget`` only (``MemoryBudget``),
which imports nothing of the engines; inside ``engines/`` the memory brake
imports it at module level and ``mlx_engine`` function-scoped.
"""

from __future__ import annotations

from typing import Any, Optional

from src.engines.memory_budget import MemoryBudget

# The smallest window a memory bound may impose on the MLX child, and the
# floor of the CPU engine's spawn window (``gguf_memory``): the floor
# llama.cpp's ``--fit`` applies to its own memory-driven reduction
# (``base_llama_cpp_engine.max_context_tokens``). Below it the model would
# refuse nearly every request; on a machine that tight the amber notice speaks
# for the model instead.
MIN_KV_BOUND_TOKENS = 4096

# The compaction threshold (#607): the share of the working window, in whole
# percent, at which a conversation compacts. Each conversation carries its
# own (``Conversation.compaction_threshold``), copied at creation from the
# global default (``UserSettings.default_compaction_threshold``). It lives
# here, with the window it is a share of, because the engine layer reads it
# too: the memory brake shrinks the window only when the next turn would
# compact under the shrunk one (``src.engines.memory_brake``).
COMPACTION_THRESHOLD_MIN = 50
COMPACTION_THRESHOLD_MAX = 90
COMPACTION_THRESHOLD_STEP = 5
DEFAULT_COMPACTION_THRESHOLD = 80
COMPACTION_THRESHOLDS = tuple(
    range(COMPACTION_THRESHOLD_MIN, COMPACTION_THRESHOLD_MAX + 1, COMPACTION_THRESHOLD_STEP)
)

# What the trigger always leaves for the answer: the output budget's floor
# (512, ``src.agents.output_budget.OUTPUT_BUDGET_FLOOR_TOKENS``) and its margin
# floor (256, ``MARGIN_FLOOR_TOKENS``). Mirrored, not imported: the agent
# layer pulls LangChain in, and this module is imported at boot.
TRIGGER_ANSWER_ROOM_TOKENS = 768


def is_compaction_threshold(value: Any) -> bool:
    """A whole percent on the scale (``COMPACTION_THRESHOLDS``); a bool, a
    float or a string is not one."""
    return _is_int(value) and value in COMPACTION_THRESHOLDS


def trigger_tokens(working_window: int, pct: int) -> int:
    """The request size, in tokens, at which a conversation compacts.

    Pure, integer: ``pct`` percent of the working window, never closer to the
    window than ``TRIGGER_ANSWER_ROOM_TOKENS`` (a request at the trigger still
    leaves the output budget its 512-token floor plus the FLOOR of its margin,
    256 tokens -- the margin itself is ``max(256, 10 % of the prompt
    estimate)``, so a large prompt can take more), and at least 1. Every
    reader of the trigger goes through it: the summarization trigger, the
    room the keep budget leaves under it, the futility guard, the
    summarizer's trim budget and the memory brake's shrink test. At 80 % the
    answer-room bound only binds below 3836 tokens.
    """
    return max(1, min(working_window * pct // 100, working_window - TRIGGER_ANSWER_ROOM_TOKENS))


def _positive_int(value: Any) -> Optional[int]:
    """``value`` if it is a real positive int, else ``None`` (rejects bool)."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def kv_bound(trained: Optional[int], ceiling: Optional[int]) -> Optional[int]:
    """The window the MLX child must enforce when memory binds, else ``None``.

    Pure. ``trained`` is the model's trained window stamped at spawn (``None``:
    unknown, the child spawned unbounded); ``ceiling`` is the memory ceiling in
    tokens (``MemoryBudget.tokens_at_ceiling``, possibly non-positive).

    * no ceiling -> ``None`` (nothing to bound against);
    * a known trained window at or below the ceiling -> ``None``: the model
      window binds, memory is not the limiting factor;
    * otherwise the ceiling lifted to ``MIN_KV_BOUND_TOKENS`` (a non-positive
      ceiling is the floor), and never above a known trained window.

    The ceiling already reserves the output budget's 512-token floor, and the
    child's preflight counts ``prompt + max_tokens`` against the bound: that
    floor is counted twice, which keeps the child 512 tokens on the safe side
    and its bound equal to the ``W`` the output budget plans against. A bool,
    a non-int or a non-positive trained window is rejected (``None``).
    """
    if not _is_int(ceiling):
        return None
    if trained is not None and _positive_int(trained) is None:
        return None
    if trained is not None and ceiling >= trained:
        return None
    bound = max(MIN_KV_BOUND_TOKENS, ceiling)
    return min(bound, trained) if trained is not None else bound


def canonical_working_window(
    allocated: Optional[int], memory_ceiling: Optional[int]
) -> Optional[int]:
    """The working window: the minimum over the KNOWN positive candidates.

    Pure. A candidate counts only when it is a positive ``int`` (``bool`` and
    non-int are rejected). If both are known the answer is the smaller; if only
    one is known it IS the window; if neither, ``None``.
    """
    candidates = [
        c for c in (_positive_int(allocated), _positive_int(memory_ceiling)) if c is not None
    ]
    return min(candidates) if candidates else None


def working_context_tokens(engine: Any) -> Optional[int]:
    """The working window FROM an engine -- the ONE place it is composed.

    Folds the engine's allocated window
    (``effective_context_tokens``) with the memory ceiling
    (``MemoryBudget.from_engine(engine).tokens_at_ceiling()`` -- the
    conversation token count at which the child's predicted footprint reaches
    the memory budget, the output floor reserved; no margin stacked on top).
    The ceiling is not clamped here: a non-positive one (the fixed part alone
    fills the budget) is no candidate, and the output budget keeps sizing from
    the allocated window in that corner; the compaction path clamps it to 1
    (compact as early as it can). The window probe is read defensively: an engine (or
    test stub) that carries no ``effective_context_tokens`` simply has no
    allocated candidate, exactly as ``model_factory`` already treats it.
    ``MemoryBudget.from_engine`` never raises; a window probe that itself
    raises propagates, and fails the turn exactly as reading the raw allocated
    window already would -- the sole caller (``model_factory.build_chat_model``)
    treats both the same.
    """
    if engine is None:
        return None
    probe = getattr(engine, "effective_context_tokens", None)
    allocated = probe() if callable(probe) else None
    memory_ceiling = MemoryBudget.from_engine(engine).tokens_at_ceiling()
    return canonical_working_window(allocated, memory_ceiling)
