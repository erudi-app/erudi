"""P2 / Wave C — engine generation guard + idle-cleanup serialization.

The agent layer wraps model resolution + the whole token stream in
``generation_guard``. Idle cleanup (``_cleanup_tick``, looped every 30 s by
``_cleanup_monitor``) shares the SAME asyncio lock as the guard, so:

  - the monitor can never reap a model mid-generation (it waits on the lock),
    which is the structural form of review BLOCKER B1;
  - the old reentrant ``threading.Lock`` deadlock (monitor held ``cls._lock``
    then called ``cleanup()`` which re-acquired it) is impossible — there is no
    ``threading.Lock`` left to re-enter.
"""

import asyncio
from datetime import datetime, timedelta

import pytest

from src.engines.base_engine import BaseEngine

pytestmark = pytest.mark.unit


class _GuardEngine(BaseEngine):
    """Minimal subclass to exercise BaseEngine classmethods in isolation
    (abstract methods stay unimplemented — we never instantiate)."""


def _reset():
    _GuardEngine._model = None
    _GuardEngine._tokenizer = None
    _GuardEngine._model_id = None
    _GuardEngine._last_used = None
    _GuardEngine._idle_unload = None


# ───────────────────────── generation_guard ─────────────────────────


async def test_guard_refreshes_idle_clock_on_exit():
    # A model idle long enough to be reaped right now.
    _GuardEngine._model = object()
    _GuardEngine._model_id = "x"
    _GuardEngine._last_used = datetime.now() - timedelta(seconds=10_000)
    try:
        assert _GuardEngine._should_cleanup() is True  # would be reaped right now
        async with _GuardEngine.generation_guard():
            pass
        # Exiting the guard refreshes _last_used -> no longer idle.
        assert _GuardEngine._last_used is not None
        assert _GuardEngine._should_cleanup() is False
    finally:
        _reset()


async def test_guard_refreshes_idle_clock_on_exception():
    _GuardEngine._model = object()
    _GuardEngine._model_id = "x"
    _GuardEngine._last_used = datetime.now() - timedelta(seconds=10_000)
    try:
        with pytest.raises(ValueError):
            async with _GuardEngine.generation_guard():
                raise ValueError("boom")
        # Clock refreshed despite the error: the model isn't pinned-idle forever
        # and the shared lock is released for the next generation.
        assert _GuardEngine._should_cleanup() is False
    finally:
        _reset()


async def test_guard_serializes_concurrent_generations():
    order: list[str] = []

    async def worker(name: str):
        async with _GuardEngine.generation_guard():
            order.append(f"{name}-enter")
            await asyncio.sleep(0.02)
            order.append(f"{name}-exit")

    try:
        await asyncio.gather(worker("A"), worker("B"))
        # M1: one generation fully completes before the other enters
        # (single-model engine — model swaps must not interleave).
        assert order in (
            ["A-enter", "A-exit", "B-enter", "B-exit"],
            ["B-enter", "B-exit", "A-enter", "A-exit"],
        )
    finally:
        _reset()


# ─────────────────────── idle cleanup (_cleanup_tick) ───────────────────────


async def test_idle_tick_reaps_when_idle():
    """The idle tick reaps an idle model — and crucially does NOT deadlock.

    The old reentrant ``cls._lock`` path (monitor held the lock then called a
    ``cleanup()`` that re-acquired it) would hang here forever; ``wait_for``
    asserts the tick completes promptly.
    """
    _GuardEngine._model = object()
    _GuardEngine._model_id = "x"
    _GuardEngine._last_used = datetime.now() - timedelta(seconds=10_000)
    try:
        assert _GuardEngine._should_cleanup() is True
        await asyncio.wait_for(_GuardEngine._cleanup_tick(), timeout=5)
        assert _GuardEngine._model is None  # reaped
    finally:
        _reset()


async def test_idle_tick_noop_when_recent():
    _GuardEngine._model = object()
    _GuardEngine._model_id = "x"
    _GuardEngine._last_used = datetime.now()
    try:
        assert _GuardEngine._should_cleanup() is False
        await asyncio.wait_for(_GuardEngine._cleanup_tick(), timeout=5)
        assert _GuardEngine._model is not None  # untouched
    finally:
        _reset()


async def test_idle_tick_cannot_reap_during_generation():
    """B1 (reinforced): while a generation holds the guard, a concurrent idle
    tick is blocked on the shared lock and cannot reap the model. After the
    guard exits, the refreshed idle clock prevents a reap anyway.
    """
    _GuardEngine._model = object()
    _GuardEngine._model_id = "x"
    _GuardEngine._last_used = datetime.now() - timedelta(seconds=10_000)
    try:
        async with _GuardEngine.generation_guard():
            tick = asyncio.create_task(_GuardEngine._cleanup_tick())
            await asyncio.sleep(0.05)  # let the tick try to acquire the lock
            assert _GuardEngine._model is not None  # still alive — tick is waiting
            assert not tick.done()
        # Guard released: the tick runs, but _last_used was refreshed -> no reap.
        await asyncio.wait_for(tick, timeout=5)
        assert _GuardEngine._model is not None
    finally:
        _reset()


# ───────────────── idle reload record (#619: say why a model reloads) ─────────────────


def _load_idle(model_id="x", idle_s=10_000):
    _GuardEngine._model = object()
    _GuardEngine._model_id = model_id
    _GuardEngine._last_used = datetime.now() - timedelta(seconds=idle_s)


async def test_idle_reap_records_which_model_it_unloaded():
    _load_idle("x")
    try:
        await asyncio.wait_for(_GuardEngine._cleanup_tick(), timeout=5)
        assert _GuardEngine._model is None
        assert _GuardEngine._idle_unload == {"model_id": "x"}
    finally:
        _reset()


async def test_a_tick_that_does_not_reap_leaves_no_record():
    _load_idle("x", idle_s=1)
    try:
        await asyncio.wait_for(_GuardEngine._cleanup_tick(), timeout=5)
        assert _GuardEngine._model is not None
        assert _GuardEngine._idle_unload is None
    finally:
        _reset()


def test_any_other_unload_forgets_the_record():
    """A delete, a shutdown or the brake's unload all call ``cleanup()``: none
    of them may leave an idle record behind. (A model switch does not; the
    stale-drop in ``take_idle_reload`` covers it.)"""
    _load_idle("x")
    _GuardEngine._idle_unload = {"model_id": "x"}
    try:
        _GuardEngine.cleanup()
        assert _GuardEngine._idle_unload is None
    finally:
        _reset()


def test_take_idle_reload_is_true_once_for_the_reaped_model():
    _GuardEngine._idle_unload = {"model_id": "x"}
    try:
        assert _GuardEngine.take_idle_reload("x") is True
        assert _GuardEngine.take_idle_reload("x") is False
        assert _GuardEngine._idle_unload is None
    finally:
        _reset()


def test_take_idle_reload_is_false_for_another_model_and_keeps_the_record():
    _GuardEngine._idle_unload = {"model_id": "x"}
    try:
        assert _GuardEngine.take_idle_reload("y") is False
        assert _GuardEngine._idle_unload == {"model_id": "x"}
    finally:
        _reset()


def test_take_idle_reload_is_false_and_clears_when_a_model_is_loaded():
    """The title request of a new conversation may have reloaded the model
    already: the record is stale, and the turn must not announce a reload."""
    _GuardEngine._idle_unload = {"model_id": "x"}
    _GuardEngine._model = object()
    _GuardEngine._model_id = "x"
    try:
        assert _GuardEngine.take_idle_reload("x") is False
        assert _GuardEngine._idle_unload is None
    finally:
        _reset()


def test_take_idle_reload_without_a_record_is_false():
    assert _GuardEngine.take_idle_reload("x") is False


# ───────────────────── idle cadence (30 s tick, 300 s threshold) ─────────────────────


def test_idle_release_contract_is_300_to_330_seconds():
    """A model is released 300 s after its last use, noticed within one 30 s tick."""
    assert BaseEngine._max_idle_time == 300
    assert BaseEngine._cleanup_interval_s == 30.0


async def test_idle_tick_does_not_queue_on_the_lock_when_nothing_to_reap():
    """A tick with nothing to reap returns at once, even during a generation.

    With a 30 s tick, an idle check must not wait behind a running generation.
    """
    _GuardEngine._model = object()
    _GuardEngine._model_id = "x"
    _GuardEngine._last_used = datetime.now()
    try:
        async with _GuardEngine._acquire_generation_lock():
            await asyncio.wait_for(_GuardEngine._cleanup_tick(), timeout=1)
        assert _GuardEngine._model is not None
    finally:
        _reset()


async def test_idle_tick_reap_waits_for_a_running_generation():
    """An idle model is reaped only once the generation holding the lock is done."""
    _GuardEngine._model = object()
    _GuardEngine._model_id = "x"
    _GuardEngine._last_used = datetime.now() - timedelta(seconds=10_000)
    try:
        async with _GuardEngine._acquire_generation_lock():
            tick = asyncio.create_task(_GuardEngine._cleanup_tick())
            await asyncio.sleep(0.05)
            assert not tick.done()
            assert _GuardEngine._model is not None
        await asyncio.wait_for(tick, timeout=5)
        assert _GuardEngine._model is None
    finally:
        _reset()


class _VanishingTimestamp:
    """A ``_last_used`` that reads as a datetime once, then as None.

    Stands for ``cleanup()`` running on another thread between two reads of the
    attribute made by the tick's lock-free pre-check.
    """

    def __init__(self):
        self._reads = 0

    def __get__(self, obj, objtype=None):
        self._reads += 1
        return datetime.now() - timedelta(seconds=10_000) if self._reads == 1 else None


def test_should_cleanup_reads_the_idle_clock_once(monkeypatch):
    """A concurrent cleanup() cannot turn the idle clock into None mid-check."""
    _GuardEngine._model = object()
    try:
        monkeypatch.setattr(_GuardEngine, "_last_used", _VanishingTimestamp())
        assert _GuardEngine._should_cleanup() in (True, False)
    finally:
        monkeypatch.undo()
        _reset()
