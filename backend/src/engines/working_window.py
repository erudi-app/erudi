"""The ONE canonical working context window (PR3.1).

Three "context window" numbers used to coexist, each computed at its own call
site:

* the raw ALLOCATED window (``BaseEngine.effective_context_tokens`` -- the
  window the loaded child actually runs with), a TIME/BOUND fact: the
  first-chunk watchdog ceiling and the preflight retry are sized from it, and a
  prefill-timeout log quotes it. Those consumers must keep reading it RAW.
* a MEMORY-bounded value the output budget and the compaction trigger each
  folded their own way -- the budget read the raw window, compaction folded
  ``min(0.8 * allocated, memory_ceiling)``. Two spellings of the same idea that
  could disagree.

This module owns the single MEMORY value the memory consumers share:

    working_window = min(allocated window, memory ceiling)

taken over the candidates that are actually KNOWN. It is deliberately SEPARATE
from the allocated window: the time/bound consumers keep the raw window, the
memory consumers (output budget, compaction) take this one.

No import cycle: this module imports ``base_engine``/``memory_budget`` shapes
only through ``MemoryBudget`` and is imported by nobody inside ``engines/``.
"""

from __future__ import annotations

from typing import Any, Optional

from src.engines.memory_budget import MemoryBudget


def _positive_int(value: Any) -> Optional[int]:
    """``value`` if it is a real positive int, else ``None`` (rejects bool)."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


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
