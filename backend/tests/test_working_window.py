"""The one canonical working context window (PR3.1).

Three "context window" numbers used to coexist: the raw ALLOCATED window (a
TIME/BOUND fact -- the watchdog ceiling and the preflight retry read it), and a
MEMORY-bounded value the output budget and the compaction trigger each folded
their own way. This module owns the single MEMORY value:

    working_window = min(allocated window, memory ceiling)

taken over the candidates that are actually known. These tests pin the pure
arithmetic; the memory consumers that route onto it are pinned in
``test_output_budget`` (budget) and ``test_agent_runner`` (compaction).
"""

import pytest

from src.engines.working_window import (
    MEMORY_MARGIN_FLOOR,
    canonical_working_window,
    working_context_tokens,
)

pytestmark = pytest.mark.unit


# ===================== the pure fold =====================


def test_both_candidates_present_returns_the_minimum():
    assert canonical_working_window(32768, 8000) == 8000
    assert canonical_working_window(8000, 32768) == 8000


def test_allocated_only_is_the_window():
    assert canonical_working_window(32768, None) == 32768


def test_memory_only_is_the_window():
    assert canonical_working_window(None, 5000) == 5000


def test_neither_present_is_none():
    assert canonical_working_window(None, None) is None


def test_equal_candidates_return_that_value():
    assert canonical_working_window(16384, 16384) == 16384


@pytest.mark.parametrize("bad", [0, -1, -12345])
def test_a_nonpositive_candidate_is_not_a_candidate(bad):
    # Rejected, so the OTHER candidate stands (or None when it is the only one).
    assert canonical_working_window(bad, 8000) == 8000
    assert canonical_working_window(32768, bad) == 32768
    assert canonical_working_window(bad, None) is None
    assert canonical_working_window(None, bad) is None
    assert canonical_working_window(bad, bad) is None


@pytest.mark.parametrize("flag", [True, False])
def test_a_bool_is_never_a_candidate(flag):
    # bool is an int subclass; a stray True/False must not masquerade as 1/0.
    assert canonical_working_window(flag, 8000) == 8000
    assert canonical_working_window(32768, flag) == 32768
    assert canonical_working_window(flag, None) is None
    assert canonical_working_window(None, flag) is None


@pytest.mark.parametrize("bad", [1.5, "8000", object()])
def test_a_non_int_is_never_a_candidate(bad):
    assert canonical_working_window(bad, 8000) == 8000
    assert canonical_working_window(32768, bad) == 32768
    assert canonical_working_window(bad, bad) is None


# ===================== composed from an engine =====================


class _MargingBudget:
    def __init__(self, ceiling):
        self._ceiling = ceiling

    def tokens_at_margin(self, margin):
        assert margin == MEMORY_MARGIN_FLOOR
        return self._ceiling


class _Engine:
    """Engine stub: an allocated window and a memory ceiling of choice."""

    def __init__(self, allocated, ceiling):
        self._allocated = allocated
        self._ceiling = ceiling

    def effective_context_tokens(self):
        return self._allocated


def _patch_budget(monkeypatch, ceiling):
    import src.engines.working_window as ww

    monkeypatch.setattr(
        ww.MemoryBudget, "from_engine", staticmethod(lambda e: _MargingBudget(ceiling))
    )


def test_working_context_tokens_folds_the_allocated_window_and_the_memory_ceiling(monkeypatch):
    _patch_budget(monkeypatch, ceiling=8000)
    assert working_context_tokens(_Engine(allocated=32768, ceiling=8000)) == 8000


def test_working_context_tokens_is_the_allocated_window_when_memory_is_off(monkeypatch):
    _patch_budget(monkeypatch, ceiling=None)
    assert working_context_tokens(_Engine(allocated=32768, ceiling=None)) == 32768


def test_working_context_tokens_is_the_memory_ceiling_when_the_window_is_unknown(monkeypatch):
    _patch_budget(monkeypatch, ceiling=5000)
    assert working_context_tokens(_Engine(allocated=None, ceiling=5000)) == 5000


def test_working_context_tokens_is_none_when_nothing_is_known(monkeypatch):
    _patch_budget(monkeypatch, ceiling=None)
    assert working_context_tokens(_Engine(allocated=None, ceiling=None)) is None


def test_working_context_tokens_tolerates_an_engine_without_a_window_probe(monkeypatch):
    _patch_budget(monkeypatch, ceiling=None)

    class _NoProbe:
        pass

    assert working_context_tokens(_NoProbe()) is None


def test_working_context_tokens_on_none_engine_is_none():
    assert working_context_tokens(None) is None


def test_the_memory_margin_floor_mirrors_the_runner_value():
    from src.agents import runner

    assert MEMORY_MARGIN_FLOOR == runner.MEMORY_MARGIN_FLOOR == 0.15
