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
        assert _GuardEngine._idle_unload == {"model_id": "x", "reason": "idle"}
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


@pytest.mark.parametrize("reason", ["idle", "manual"])
def test_take_idle_reload_returns_the_reason_once_for_the_unloaded_model(reason):
    _GuardEngine._idle_unload = {"model_id": "x", "reason": reason}
    try:
        assert _GuardEngine.take_idle_reload("x") == reason
        assert _GuardEngine.take_idle_reload("x") is None
        assert _GuardEngine._idle_unload is None
    finally:
        _reset()


def test_take_idle_reload_is_none_for_another_model_and_keeps_the_record():
    _GuardEngine._idle_unload = {"model_id": "x", "reason": "idle"}
    try:
        assert _GuardEngine.take_idle_reload("y") is None
        assert _GuardEngine._idle_unload == {"model_id": "x", "reason": "idle"}
    finally:
        _reset()


def test_take_idle_reload_is_none_and_clears_when_a_model_is_loaded():
    """The title request of a new conversation may have reloaded the model
    already: the record is stale, and the turn must not announce a reload."""
    _GuardEngine._idle_unload = {"model_id": "x", "reason": "manual"}
    _GuardEngine._model = object()
    _GuardEngine._model_id = "x"
    try:
        assert _GuardEngine.take_idle_reload("x") is None
        assert _GuardEngine._idle_unload is None
    finally:
        _reset()


def test_take_idle_reload_without_a_record_is_none():
    assert _GuardEngine.take_idle_reload("x") is None


# ───────────────── manual unload (#619: the header's Unload button) ─────────────────


async def test_a_manual_unload_reaps_the_model_and_records_why():
    _load_idle("x", idle_s=1)
    try:
        assert await asyncio.wait_for(_GuardEngine.unload("x"), timeout=5) == "x"
        assert _GuardEngine._model is None
        assert _GuardEngine._last_used is None
        assert _GuardEngine._idle_unload == {"model_id": "x", "reason": "manual"}
        assert BaseEngine._generation_lock.demand == 0
        assert not _GuardEngine._generation_lock_for_running_loop().locked()
    finally:
        _reset()


async def test_a_manual_unload_with_nothing_loaded_does_nothing():
    _GuardEngine._idle_unload = {"model_id": "x", "reason": "idle"}
    try:
        assert await asyncio.wait_for(_GuardEngine.unload("x"), timeout=5) is None
        # An earlier idle record still says why that model reloads.
        assert _GuardEngine._idle_unload == {"model_id": "x", "reason": "idle"}
    finally:
        _reset()


async def test_a_manual_unload_is_refused_while_a_generation_runs(_short_busy_wait):
    from src.core.exceptions import EngineBusyException

    _load_idle("x", idle_s=1)
    try:
        async with _GuardEngine.generation_guard():
            with pytest.raises(EngineBusyException):
                await _GuardEngine.unload("x")
            assert _GuardEngine._model is not None
        assert _GuardEngine._idle_unload is None
        assert BaseEngine._generation_lock.demand == 0
    finally:
        _reset()


async def test_a_manual_unload_of_a_model_no_longer_loaded_is_refused():
    """The button named A; B was loaded since (a /model switch and a send):
    the press must not unload B."""
    from src.core.exceptions import LoadedModelChangedException

    _load_idle("b", idle_s=200)
    last_used = _GuardEngine._last_used
    _GuardEngine._generation_ended_at = None
    try:
        with pytest.raises(LoadedModelChangedException) as raised:
            await _GuardEngine.unload("a")
        assert raised.value.status_code == 409
        assert raised.value.erudi_code == "LOADED_MODEL_CHANGED"
        assert raised.value.message == "Another model is loaded now."
        assert _GuardEngine._model is not None
        assert _GuardEngine._model_id == "b"
        assert _GuardEngine._idle_unload is None
        # A refused press changes nothing: B's idle clock does not restart,
        # and no generation end is stamped (CUDA's contention settle).
        assert _GuardEngine._last_used == last_used
        assert _GuardEngine._generation_ended_at is None
        assert not _GuardEngine._generation_lock_for_running_loop().locked()
    finally:
        _reset()


async def test_a_cancelled_manual_unload_finishes_the_cleanup_under_the_lock(monkeypatch):
    """A request cancelled mid-unload (shutdown, a disconnect) never releases
    the lock while the cleanup thread still kills the child and resets the
    engine: the cleanup ends, the record is written, then the cancellation
    is raised."""
    import threading

    started = threading.Event()
    release = threading.Event()
    real_cleanup = _GuardEngine.cleanup

    def _slow_cleanup():
        started.set()
        release.wait(5)
        real_cleanup()

    monkeypatch.setattr(_GuardEngine, "cleanup", staticmethod(_slow_cleanup))
    _load_idle("x", idle_s=1)
    try:
        request = asyncio.create_task(_GuardEngine.unload("x"))
        while not started.is_set():
            await asyncio.sleep(0.01)
        request.cancel()
        await asyncio.sleep(0.05)
        # Still cleaning: the request has not ended and the lock is held.
        assert not request.done()
        assert _GuardEngine._generation_lock_for_running_loop().locked()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(request, timeout=5)
        assert _GuardEngine._model is None
        assert _GuardEngine._idle_unload == {"model_id": "x", "reason": "manual"}
        assert not _GuardEngine._generation_lock_for_running_loop().locked()
        assert BaseEngine._generation_lock.demand == 0
    finally:
        release.set()
        _reset()


# ───────────── the idle delay setting (#619: 5/15/30/60 minutes or never) ─────────────


@pytest.fixture
def _restore_idle_delay(monkeypatch):
    """The delay is a class attribute the setter writes: put it back."""
    monkeypatch.setattr(_GuardEngine, "_max_idle_time", BaseEngine._max_idle_time)


@pytest.mark.parametrize("minutes, seconds", [(5, 300), (15, 900), (30, 1800), (60, 3600)])
def test_the_idle_delay_is_set_in_minutes(_restore_idle_delay, minutes, seconds):
    _GuardEngine.set_idle_unload_minutes(minutes)
    assert _GuardEngine._max_idle_time == seconds
    # Only the engine it is applied to: the base class keeps its default.
    assert BaseEngine._max_idle_time == 300


def test_never_is_zero_minutes_and_no_idle_time(_restore_idle_delay):
    _GuardEngine.set_idle_unload_minutes(0)
    assert _GuardEngine._max_idle_time is None


def test_a_longer_delay_keeps_a_model_idle_past_five_minutes(_restore_idle_delay):
    _GuardEngine.set_idle_unload_minutes(15)
    _load_idle("x", idle_s=600)
    try:
        assert _GuardEngine._should_cleanup() is False
        _GuardEngine._last_used = datetime.now() - timedelta(seconds=905)
        assert _GuardEngine._should_cleanup() is True
    finally:
        _reset()


async def test_never_keeps_an_idle_model_loaded(_restore_idle_delay):
    _GuardEngine.set_idle_unload_minutes(0)
    _load_idle("x", idle_s=10 * 24 * 3600)
    try:
        assert _GuardEngine._should_cleanup() is False
        await asyncio.wait_for(_GuardEngine._cleanup_tick(), timeout=5)
        assert _GuardEngine._model is not None
        assert _GuardEngine._idle_unload is None
    finally:
        _reset()


async def test_never_still_lets_the_button_unload(_restore_idle_delay):
    _GuardEngine.set_idle_unload_minutes(0)
    _load_idle("x", idle_s=1)
    try:
        assert await _GuardEngine.unload("x") == "x"
        assert _GuardEngine._model is None
    finally:
        _reset()


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
