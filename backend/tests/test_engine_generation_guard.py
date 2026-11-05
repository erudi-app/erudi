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


# ──────────────── non-blocking guard (/compact, unload: busy -> 409) ────────────────


@pytest.fixture
def _short_busy_wait(monkeypatch):
    """The 2 s busy wait, shortened so a refusal is quick to observe."""
    from src.engines import base_engine

    monkeypatch.setattr(base_engine, "GENERATION_BUSY_WAIT_S", 0.05)


def test_the_busy_wait_is_two_seconds():
    from src.engines import base_engine

    assert base_engine.GENERATION_BUSY_WAIT_S == 2.0


def test_engine_busy_is_a_409_with_a_neutral_message():
    from src.core.exceptions import AppBaseException, EngineBusyException

    exc = EngineBusyException()
    assert isinstance(exc, AppBaseException)
    assert exc.status_code == 409
    assert exc.erudi_code == "ENGINE_BUSY"
    assert exc.message == "The model is busy; try again in a moment."


async def test_try_guard_enters_a_free_engine_and_holds_the_lock():
    try:
        async with _GuardEngine.try_generation_guard():
            assert _GuardEngine._generation_lock_for_running_loop().locked()
        assert not _GuardEngine._generation_lock_for_running_loop().locked()
        assert BaseEngine._generation_lock.demand == 0
    finally:
        _reset()


async def test_try_guard_refuses_when_the_lock_stays_held(_short_busy_wait):
    from src.core.exceptions import EngineBusyException

    entered = False
    try:
        async with _GuardEngine.generation_guard():
            with pytest.raises(EngineBusyException):
                async with _GuardEngine.try_generation_guard():
                    entered = True
        assert entered is False
        assert BaseEngine._generation_lock.demand == 0
    finally:
        _reset()


async def test_try_guard_waits_for_a_short_hold_to_end():
    """A sub-second hold (a contention or pressure tick) is absorbed."""
    released = asyncio.Event()

    async def short_holder():
        async with _GuardEngine._acquire_generation_lock():
            await asyncio.sleep(0.05)
        released.set()

    try:
        holder = asyncio.create_task(short_holder())
        await asyncio.sleep(0)
        async with _GuardEngine.try_generation_guard():
            assert released.is_set()
        await holder
    finally:
        _reset()


async def test_try_guard_never_jumps_ahead_of_a_queued_turn(_short_busy_wait):
    """Busy is the lock held OR an acquirer waiting: a turn queued behind a
    holder keeps the engine busy even at the instant the lock reads free."""
    from src.core.exceptions import EngineBusyException

    order: list = []

    async def queued_turn():
        async with _GuardEngine.generation_guard():
            order.append("turn")
            await asyncio.sleep(0.2)

    try:
        async with _GuardEngine._acquire_generation_lock():
            turn = asyncio.create_task(queued_turn())
            await asyncio.sleep(0.01)
            assert BaseEngine._generation_lock.demand == 2
        # Released: asyncio hands the lock to the queued turn, which has not
        # run yet -- the lock may read free, the demand does not.
        with pytest.raises(EngineBusyException):
            async with _GuardEngine.try_generation_guard():
                order.append("compact")
        await turn
        assert order == ["turn"]
        assert BaseEngine._generation_lock.demand == 0
    finally:
        _reset()


async def test_a_cancelled_queued_acquirer_leaves_no_demand_behind():
    async def queued():
        async with _GuardEngine.generation_guard():
            pass

    try:
        async with _GuardEngine._acquire_generation_lock():
            waiter = asyncio.create_task(queued())
            await asyncio.sleep(0.01)
            assert BaseEngine._generation_lock.demand == 2
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert BaseEngine._generation_lock.demand == 1
        assert BaseEngine._generation_lock.demand == 0
        async with _GuardEngine.try_generation_guard():
            pass
    finally:
        _reset()


async def test_the_demand_counter_is_recreated_with_the_lock_per_loop():
    old = _GuardEngine._generation_lock_for_running_loop()
    old.demand = 3
    BaseEngine._generation_lock_loop = object()  # a loop that is not this one
    try:
        new = _GuardEngine._generation_lock_for_running_loop()
        assert new is not old
        assert new.demand == 0
    finally:
        _reset()


def test_a_holder_of_an_older_loop_never_touches_the_new_loops_demand():
    """The lock (and its demand) is recreated for each event loop: a holder
    that unwinds on the old loop after the new lock exists lowers the old
    lock's counter, never the new one's -- which would mask a waiter."""
    loop_a = asyncio.new_event_loop()
    try:
        release = loop_a.run_until_complete(_make_event())
        holding = loop_a.run_until_complete(_make_event())

        async def hold():
            async with _GuardEngine._acquire_generation_lock():
                holding.set()
                await release.wait()

        holder = loop_a.create_task(hold())
        loop_a.run_until_complete(holding.wait())
        old_lock = BaseEngine._generation_lock
        assert old_lock.demand == 1

        async def on_loop_b():
            return _GuardEngine._generation_lock_for_running_loop()

        new_lock = asyncio.run(on_loop_b())
        assert new_lock is not old_lock and new_lock.demand == 0

        release.set()
        loop_a.run_until_complete(holder)
        assert old_lock.demand == 0
        assert new_lock.demand == 0
    finally:
        loop_a.close()
        _reset()


def test_the_demand_counter_never_goes_below_zero():
    async def unwind_twice():
        lock = _GuardEngine._generation_lock_for_running_loop()
        async with _GuardEngine._acquire_generation_lock():
            lock.demand = 0  # as if it had been recreated under the holder
        return lock.demand

    try:
        assert asyncio.run(unwind_twice()) == 0
    finally:
        _reset()


async def _make_event():
    return asyncio.Event()


async def test_try_guard_drains_a_pending_reset_before_entering():
    finished: list = []

    async def reset():
        await asyncio.sleep(0.05)
        finished.append("reset")

    BaseEngine._pending_reset = asyncio.get_running_loop().create_task(reset())
    try:
        async with _GuardEngine.try_generation_guard():
            assert finished == ["reset"]
        assert BaseEngine._pending_reset is None
    finally:
        _reset()


async def test_try_guard_stamps_the_idle_clock_when_a_model_is_loaded_at_exit():
    _GuardEngine._model = object()
    _GuardEngine._model_id = "x"
    _GuardEngine._last_used = datetime.now() - timedelta(seconds=10_000)
    try:
        async with _GuardEngine.try_generation_guard():
            pass
        assert _GuardEngine._should_cleanup() is False
    finally:
        _reset()


async def test_try_guard_does_not_stamp_the_idle_clock_after_an_unload():
    _GuardEngine._model = object()
    _GuardEngine._model_id = "x"
    try:
        async with _GuardEngine.try_generation_guard():
            _GuardEngine.cleanup()
        assert _GuardEngine._last_used is None
    finally:
        _reset()


async def test_try_guard_releases_the_lock_on_an_exception():
    try:
        with pytest.raises(ValueError):
            async with _GuardEngine.try_generation_guard():
                raise ValueError("boom")
        assert not _GuardEngine._generation_lock_for_running_loop().locked()
        assert BaseEngine._generation_lock.demand == 0
    finally:
        _reset()
