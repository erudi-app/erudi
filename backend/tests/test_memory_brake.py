"""The reactive memory brake's control law (pure, no I/O).

The law is built on PRESSURE EPISODES: an episode opens at the first non-calm
sample and ends after ``CALM_SAMPLES_TO_END`` calm ones, and every action is
bounded per episode -- a per-sample rule would keep shrinking or releasing
for ever under a permanent warning. These tests drive the law through whole
scenarios (a permanent warning with an idle or an active user, an escalation,
an unload and the reload that carries the episode) and pin each rule alone.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.engines import memory_brake as mb
from src.engines.memory_brake import (
    ATTRIBUTION_FLOOR_BYTES,
    BRAKE_IDLE_INTERVAL_S,
    CALM_SAMPLES_TO_END,
    CALM_SAMPLES_TO_FORGET,
    CALM_SAMPLES_TO_GROW,
    RELOAD_GRACE_S,
    SHRINK_FACTOR,
    SWAP_BURST_PAGES,
    UNLOAD_MIN_SHARE,
    WARNING_RELEASE_IDLE_S,
    BrakeState,
    Context,
    Sample,
    step,
)
from src.engines.working_window import MIN_KV_BOUND_TOKENS, trigger_tokens

pytestmark = pytest.mark.unit

GIB = 1024**3
MIB = 1024**2
W = 32_768
BIG_TURN = 30_000  # the largest request of the last turn: a shrink of W is useful
RESIDUAL = int(0.33 * GIB)  # what MLX keeps after a release (measured)
CALM, WARNING, CRITICAL = 1, 2, 4


class _Child:
    """One child under the law, as the engine drives it: the window follows
    the cap, a release leaves the measured residual, a turn re-allocates."""

    def __init__(self, *, window=W, static=W, ceiling=W, share=0.3, state=None):
        self.state = state or BrakeState()
        self.static = static
        self.ceiling = ceiling
        self.window = window
        self.share = share
        self.excess = GIB
        self.last_used = 0.0
        self.last_request_tokens = BIG_TURN
        self.decisions = []

    def sample(self, point, now, level, *, swapped=None):
        decision = step(
            self.state,
            Sample(
                point=point,
                now=now,
                level=level,
                swapped_pages=swapped,
                excess_bytes=self.excess,
                last_request_tokens=self.last_request_tokens,
                last_used=self.last_used,
            ),
            Context(
                window=self.window,
                ceiling=self.ceiling,
                static_window=self.static,
                footprint_share=self.share,
            ),
        )
        self.state = decision.state
        if decision.release:
            # The engine measures what the release left behind.
            self.excess = RESIDUAL
            self.state.excess_after_release = RESIDUAL
        if decision.cap_change is not None:
            self.window = self.state.cap if self.state.cap is not None else self.static
        self.decisions.append((point, now, decision))
        return decision

    def turn(self, start, end, level, *, request_tokens=BIG_TURN, abandoned=False):
        """turn_start, the turn itself (it re-allocates), turn_end."""
        self.sample(mb.POINT_TURN_START, start, level)
        self.state.turn_happened = True  # begin_memory_window
        self.excess = GIB
        self.last_request_tokens = request_tokens
        self.last_used = end
        if not abandoned:
            return self.sample(mb.POINT_TURN_END, end, level)
        return None

    def count(self, action):
        return sum(1 for _, _, d in self.decisions if action in d.actions)


def _one(state=None, *, point=mb.POINT_IDLE, now=1000.0, level=WARNING, **sample):
    """One sample on a fresh child, the window at W."""
    context = {
        "window": sample.pop("window", W),
        "ceiling": sample.pop("ceiling", W),
        "static_window": sample.pop("static_window", W),
        "footprint_share": sample.pop("footprint_share", 0.3),
    }
    fields = {
        "excess_bytes": GIB,
        "last_request_tokens": BIG_TURN,
        "last_used": 0.0,
        **sample,
    }
    return step(
        state or BrakeState(),
        Sample(point=point, now=now, level=level, **fields),
        Context(**context),
    )


# ===================== the constants =====================


def test_the_qa_tuned_starting_values():
    assert SHRINK_FACTOR == 0.75
    assert CALM_SAMPLES_TO_END == 2
    assert CALM_SAMPLES_TO_GROW == 3
    assert CALM_SAMPLES_TO_FORGET == 10
    assert BRAKE_IDLE_INTERVAL_S == 60
    assert WARNING_RELEASE_IDLE_S == 180
    assert RELOAD_GRACE_S == 600
    assert UNLOAD_MIN_SHARE == 0.05
    assert ATTRIBUTION_FLOOR_BYTES == 256 * MIB
    # 64 MiB of 16 KiB pages (Apple Silicon's page size).
    assert SWAP_BURST_PAGES == 64 * MIB // 16384


# ===================== episode bounds: whole scenarios =====================


def test_a_permanent_warning_with_an_idle_user_releases_once_and_shrinks_at_most_once():
    child = _Child()
    child.turn(0.0, 10.0, WARNING)
    for i in range(1, 61):
        child.sample(mb.POINT_IDLE, 10.0 + 60.0 * i, WARNING)

    assert child.count("release") == 1
    assert child.count("shrink") <= 1
    assert child.count("unload") == 0
    # The release came once the user had been away WARNING_RELEASE_IDLE_S.
    ((_, at, _),) = [d for d in child.decisions if "release" in d[2].actions]
    assert at - child.last_used >= WARNING_RELEASE_IDLE_S
    # The shrink's notice survives every idle sample until a stateful turn
    # end consumes it.
    assert child.state.notice == mb.NOTICE_ACTED
    assert child.state.take_notice() == mb.NOTICE_ACTED
    assert child.state.notice is None


def test_a_permanent_warning_with_an_active_user_never_releases():
    # 20 turns, each followed by a 90 s reading pause: the prefix cache is
    # kept (a reader is not a walk-away), at most one shrink.
    child = _Child()
    now = 0.0
    for _ in range(20):
        child.turn(now, now + 10.0, WARNING)
        child.sample(mb.POINT_IDLE, now + 10.0 + 60.0, WARNING)
        now += 10.0 + 90.0

    assert child.count("release") == 0
    assert child.count("shrink") <= 1
    assert child.count("unload") == 0


def test_an_escalation_to_critical_shrinks_at_most_twice():
    child = _Child()
    child.turn(0.0, 10.0, WARNING)
    now = 100.0
    for _ in range(10):
        child.turn(now, now + 10.0, CRITICAL)
        now += 100.0

    assert child.count("shrink") == 2
    assert child.state.shrinks_this_episode == 2


def test_never_two_shrinks_without_a_turn_between_them():
    child = _Child()
    child.sample(mb.POINT_TURN_START, 0.0, CALM)
    child.state.turn_happened = True
    child.last_used = 10.0
    child.sample(mb.POINT_TURN_END, 10.0, CRITICAL)
    assert child.count("shrink") == 1
    # Idle criticals with no turn: no second shrink although critical allows two.
    for i in range(1, 4):
        child.sample(mb.POINT_IDLE, 10.0 + 60.0 * i, CRITICAL)
    child.sample(mb.POINT_TURN_START, 300.0, CRITICAL)
    assert child.count("shrink") == 1
    # A turn ran: the second one is allowed.
    child.state.turn_happened = True
    child.sample(mb.POINT_TURN_END, 310.0, CRITICAL)
    assert child.count("shrink") == 2


def test_critical_through_a_brake_reload_takes_no_second_unload_and_no_extra_shrink():
    child = _Child()
    child.turn(0.0, 10.0, WARNING)
    child.turn(100.0, 110.0, CRITICAL)
    assert child.count("shrink") == 2
    # Walk away under critical: two idle samples unload.
    child.sample(mb.POINT_IDLE, 170.0, CRITICAL)
    unload = child.sample(mb.POINT_IDLE, 230.0, CRITICAL)
    assert unload.unload

    # The reload carries the episode.
    reloaded = BrakeState.reloaded(child.state.carry(), now=300.0)
    assert reloaded.in_episode
    assert reloaded.shrinks_this_episode == 2
    assert reloaded.unloaded_this_episode
    assert reloaded.reloaded_at == 300.0
    after = _Child(window=child.window, state=reloaded)
    after.last_used = 300.0
    after.turn(300.0, 310.0, CRITICAL)
    # Grace elapsed, two idle criticals and more: no second unload.
    for i in range(1, 6):
        after.sample(mb.POINT_IDLE, 310.0 + RELOAD_GRACE_S + 60.0 * i, CRITICAL)
    assert after.count("unload") == 0
    assert after.count("shrink") == 0


def test_the_measured_residual_never_retriggers_a_release():
    child = _Child()
    child.turn(0.0, 10.0, CRITICAL)
    assert child.count("release") == 1  # turn_end, critical
    for i in range(1, 30):
        child.sample(mb.POINT_IDLE, 10.0 + 60.0 * i, CRITICAL)
    assert child.count("release") == 1


def test_a_release_needs_a_turn_that_reallocated():
    child = _Child()
    child.turn(0.0, 10.0, CRITICAL)
    child.turn(100.0, 110.0, CRITICAL)
    assert child.count("release") == 2


# ===================== the unload =====================


def test_two_idle_criticals_with_no_turn_between_unload():
    child = _Child()
    child.turn(0.0, 10.0, WARNING)
    first = child.sample(mb.POINT_IDLE, 70.0, CRITICAL)
    assert not first.unload
    assert child.state.critical_idle_at == 70.0
    second = child.sample(mb.POINT_IDLE, 130.0, CRITICAL)
    assert second.unload
    assert second.actions == ("unload",)
    assert child.state.unloaded_this_episode


@pytest.mark.parametrize("abandoned", [False, True])
def test_a_turn_between_two_idle_criticals_prevents_the_unload(abandoned):
    child = _Child()
    child.turn(0.0, 10.0, WARNING)
    child.sample(mb.POINT_IDLE, 70.0, CRITICAL)
    child.turn(80.0, 90.0, CRITICAL, abandoned=abandoned)
    after = child.sample(mb.POINT_IDLE, 150.0, CRITICAL)
    assert not after.unload
    # It takes another idle critical.
    assert child.sample(mb.POINT_IDLE, 210.0, CRITICAL).unload


@pytest.mark.parametrize("between", [WARNING, CALM])
def test_an_unload_needs_two_consecutive_idle_criticals(between):
    child = _Child()
    child.turn(0.0, 10.0, WARNING)
    child.sample(mb.POINT_IDLE, 70.0, CRITICAL)
    child.sample(mb.POINT_IDLE, 130.0, between)
    assert child.state.critical_idle_at is None
    assert not child.sample(mb.POINT_IDLE, 190.0, CRITICAL).unload
    assert child.sample(mb.POINT_IDLE, 250.0, CRITICAL).unload


def test_an_unload_happens_once_per_episode():
    state = BrakeState(in_episode=True, critical_idle_at=900.0, unloaded_this_episode=True)
    decision = _one(state, level=CRITICAL)
    assert not decision.unload
    assert decision.state.critical_idle_at == 1000.0


def test_no_unload_under_the_minimum_share():
    state = BrakeState(in_episode=True, critical_idle_at=900.0)
    low = _one(state, level=CRITICAL, footprint_share=UNLOAD_MIN_SHARE / 2)
    assert not low.unload
    unknown = _one(state, level=CRITICAL, footprint_share=None)
    assert not unknown.unload
    enough = _one(state, level=CRITICAL, footprint_share=UNLOAD_MIN_SHARE)
    assert enough.unload


def test_the_reload_grace_blocks_an_unload():
    state = BrakeState(in_episode=True, critical_idle_at=900.0, reloaded_at=500.0)
    within = _one(state, level=CRITICAL, now=500.0 + RELOAD_GRACE_S - 1)
    assert not within.unload
    state = replace(state, critical_idle_at=500.0 + RELOAD_GRACE_S - 60)
    after = _one(state, level=CRITICAL, now=500.0 + RELOAD_GRACE_S + 1)
    assert after.unload


def test_never_an_unload_at_a_turn_point():
    state = BrakeState(in_episode=True, critical_idle_at=900.0)
    for point in (mb.POINT_TURN_START, mb.POINT_TURN_END):
        assert not _one(state, point=point, level=CRITICAL).unload


def test_the_carry_holds_the_episode_and_the_cap():
    state = BrakeState(
        cap=24_576,
        faulty=W,
        in_episode=True,
        episode_level=CRITICAL,
        shrinks_this_episode=2,
        unloaded_this_episode=True,
        notice=mb.NOTICE_ACTED,
        calm_count=1,
    )
    assert state.carry() == {
        "cap": 24_576,
        "faulty": W,
        "in_episode": True,
        "episode_level": CRITICAL,
        "shrinks_this_episode": 2,
        "unloaded_this_episode": True,
    }


# ===================== the release =====================


def test_a_release_at_idle_under_warning_waits_for_the_walk_away():
    assert not _one(last_used=1000.0 - WARNING_RELEASE_IDLE_S + 1).release
    assert _one(last_used=1000.0 - WARNING_RELEASE_IDLE_S).release


def test_a_release_at_idle_under_critical_needs_no_walk_away():
    assert _one(level=CRITICAL, last_used=1000.0 - BRAKE_IDLE_INTERVAL_S).release


def test_a_release_at_turn_end_only_under_critical():
    assert _one(point=mb.POINT_TURN_END, level=CRITICAL, last_used=1000.0).release
    assert not _one(point=mb.POINT_TURN_END, level=WARNING, last_used=0.0).release


def test_never_a_release_at_turn_start():
    assert not _one(point=mb.POINT_TURN_START, level=CRITICAL).release


def test_a_release_needs_what_it_would_give_back_to_reach_the_floor():
    state = BrakeState(excess_after_release=RESIDUAL)
    below = _one(state, level=CRITICAL, excess_bytes=RESIDUAL + ATTRIBUTION_FLOOR_BYTES - 1)
    assert not below.release
    at = _one(state, level=CRITICAL, excess_bytes=RESIDUAL + ATTRIBUTION_FLOOR_BYTES)
    assert at.release
    assert not _one(level=CRITICAL, excess_bytes=None).release


def test_a_release_alone_sets_no_notice():
    # A short conversation (no useful shrink) with a big footprint: the
    # memory is given back, the user loses nothing visible.
    decision = _one(level=CRITICAL, last_request_tokens=100)
    assert decision.actions == ("release",)
    assert decision.state.notice is None


# ===================== the shrink =====================


def test_a_shrink_is_one_step_of_the_working_window():
    decision = _one(point=mb.POINT_TURN_END, level=WARNING)
    assert decision.cap_change == "shrink"
    assert decision.state.cap == int(W * SHRINK_FACTOR)
    assert decision.state.faulty == W
    assert decision.state.notice == mb.NOTICE_ACTED


def test_a_shrink_never_goes_under_the_floor():
    decision = _one(point=mb.POINT_TURN_END, window=5000, last_request_tokens=5000)
    assert decision.state.cap == MIN_KV_BOUND_TOKENS
    at_floor = _one(point=mb.POINT_TURN_END, window=MIN_KV_BOUND_TOKENS)
    assert at_floor.cap_change is None


def test_a_shrink_only_when_the_next_turn_compacts():
    new = int(W * SHRINK_FACTOR)
    threshold = trigger_tokens(new, 80)
    under = _one(point=mb.POINT_TURN_END, last_request_tokens=int(threshold) - 1)
    assert under.cap_change is None
    assert under.state.cap is None
    unknown = _one(point=mb.POINT_TURN_END, last_request_tokens=None)
    assert unknown.cap_change is None
    useful = _one(point=mb.POINT_TURN_END, last_request_tokens=int(threshold) + 1)
    assert useful.cap_change == "shrink"


def test_a_shrink_judges_the_last_request_at_its_conversations_threshold():
    # The last request belongs to a conversation compacting at its own share
    # of the window (#607): the shrink is useful when THAT trigger fires
    # under the shrunk window. None -- an arena turn, a handle no turn has
    # stamped yet -- is the default, 80 %.
    new = int(W * SHRINK_FACTOR)
    between = (trigger_tokens(new, 50) + trigger_tokens(new, 80)) // 2
    at_default = _one(point=mb.POINT_TURN_END, last_request_tokens=between)
    assert at_default.cap_change is None
    unstamped = _one(
        point=mb.POINT_TURN_END, last_request_tokens=between, compaction_threshold=None
    )
    assert unstamped.cap_change is None
    low = _one(point=mb.POINT_TURN_END, last_request_tokens=between, compaction_threshold=50)
    assert low.cap_change == "shrink"
    over_80 = trigger_tokens(new, 80) + 1
    assert _one(point=mb.POINT_TURN_END, last_request_tokens=over_80).cap_change == "shrink"
    high = _one(point=mb.POINT_TURN_END, last_request_tokens=over_80, compaction_threshold=90)
    assert high.cap_change is None


def test_one_shrink_at_warning_then_one_more_at_critical():
    state = BrakeState(in_episode=True, shrinks_this_episode=1)
    assert _one(state, point=mb.POINT_TURN_END, level=WARNING).cap_change is None
    assert _one(state, point=mb.POINT_TURN_END, level=CRITICAL).cap_change == "shrink"
    state = replace(state, shrinks_this_episode=2)
    assert _one(state, point=mb.POINT_TURN_END, level=CRITICAL).cap_change is None


def test_a_shrink_at_turn_start_applies_to_the_turn_about_to_run():
    decision = _one(point=mb.POINT_TURN_START, level=WARNING)
    assert decision.cap_change == "shrink"


# ===================== the notice =====================


def test_external_when_the_conversation_uses_little_and_nothing_acted():
    decision = _one(level=WARNING, excess_bytes=ATTRIBUTION_FLOOR_BYTES - 1, last_request_tokens=10)
    assert decision.actions == ()
    assert decision.state.notice == mb.NOTICE_EXTERNAL


def test_acted_is_never_overwritten_by_external():
    state = BrakeState(notice=mb.NOTICE_ACTED, in_episode=True, shrinks_this_episode=1)
    decision = _one(state, excess_bytes=0, last_request_tokens=10)
    assert decision.state.notice == mb.NOTICE_ACTED


def test_a_shrink_overwrites_external():
    state = BrakeState(notice=mb.NOTICE_EXTERNAL)
    decision = _one(state, point=mb.POINT_TURN_END)
    assert decision.state.notice == mb.NOTICE_ACTED


def test_no_external_on_a_brake_reloaded_child_until_its_episode_ends():
    state = BrakeState.reloaded(
        {
            "cap": None,
            "faulty": None,
            "in_episode": True,
            "episode_level": CRITICAL,
            "shrinks_this_episode": 2,
            "unloaded_this_episode": True,
        },
        now=0.0,
    )
    child = _Child(state=state)
    child.excess = 0  # we just unloaded it: of course it uses little
    child.last_request_tokens = 10
    child.sample(mb.POINT_IDLE, 100.0, WARNING)
    assert child.state.notice is None
    for i in range(CALM_SAMPLES_TO_END):
        child.sample(mb.POINT_IDLE, 200.0 + 60.0 * i, CALM)
    assert not child.state.in_episode
    child.excess = 0
    child.sample(mb.POINT_IDLE, 400.0, WARNING)
    assert child.state.notice == mb.NOTICE_EXTERNAL


def test_take_notice_consumes_it():
    state = BrakeState(notice=mb.NOTICE_EXTERNAL)
    assert state.take_notice() == mb.NOTICE_EXTERNAL
    assert state.take_notice() is None


# ===================== the sensors =====================


def test_a_swap_burst_at_turn_end_counts_as_a_warning():
    burst = _one(point=mb.POINT_TURN_END, level=CALM, swapped_pages=SWAP_BURST_PAGES)
    assert burst.level == WARNING
    assert burst.state.in_episode
    assert burst.cap_change == "shrink"
    small = _one(point=mb.POINT_TURN_END, level=CALM, swapped_pages=SWAP_BURST_PAGES - 1)
    assert small.level == CALM
    assert not small.state.in_episode
    # Only the turn's own swap counts, at turn_end.
    idle = _one(point=mb.POINT_IDLE, level=CALM, swapped_pages=SWAP_BURST_PAGES)
    assert idle.level == CALM


def test_an_unreadable_sensor_does_nothing():
    state = BrakeState(turn_happened=True, critical_idle_at=10.0)
    decision = _one(state, level=None)
    assert decision.actions == ()
    assert decision.level is None
    assert not decision.state.in_episode
    # The turn was still seen.
    assert decision.state.turn_happened is False


# ===================== pacing and turns =====================


def test_an_idle_tick_right_after_a_turn_is_skipped():
    child = _Child()
    child.turn(0.0, 10.0, WARNING)
    skipped = child.sample(mb.POINT_IDLE, 10.0 + 30.0, CRITICAL)
    assert skipped.skipped
    assert skipped.actions == ()
    taken = child.sample(mb.POINT_IDLE, 10.0 + BRAKE_IDLE_INTERVAL_S, CRITICAL)
    assert not taken.skipped


def test_the_pacing_tolerates_a_tick_that_comes_a_little_early():
    # The monitor's period IS the interval: a tick a few seconds short of it
    # (scheduling jitter, the lock) is still the next sample, not a skip.
    assert mb.IDLE_PACING_TOLERANCE_S == 5
    state = BrakeState(last_sample_at=1000.0)
    early = 1000.0 + BRAKE_IDLE_INTERVAL_S - mb.IDLE_PACING_TOLERANCE_S
    assert not _one(state, now=early, last_used=0.0).skipped
    assert _one(state, now=early - 0.5, last_used=0.0).skipped


def test_idle_samples_are_paced_on_the_last_sample_too():
    state = BrakeState(last_sample_at=990.0)
    assert _one(state, now=1000.0, last_used=0.0).skipped
    assert not _one(state, now=990.0 + BRAKE_IDLE_INTERVAL_S, last_used=0.0).skipped


def test_turn_points_are_never_paced():
    state = BrakeState(last_sample_at=999.0)
    assert not _one(state, point=mb.POINT_TURN_END, last_used=999.0).skipped


def test_every_sample_clears_the_turn_it_saw():
    state = BrakeState(turn_happened=True, critical_idle_at=900.0)
    decision = _one(state, point=mb.POINT_TURN_END, level=CALM)
    assert decision.state.turn_happened is False
    assert decision.state.critical_idle_at is None


def test_the_law_never_mutates_the_state_it_is_given():
    state = BrakeState()
    _one(state, point=mb.POINT_TURN_END)
    assert state == BrakeState()


# ===================== episodes and growth =====================


def test_an_episode_ends_after_the_calm_samples_and_resets_its_counters():
    child = _Child()
    child.turn(0.0, 10.0, CRITICAL)
    assert child.state.in_episode
    assert child.state.episode_level == CRITICAL
    child.sample(mb.POINT_IDLE, 70.0, CALM)
    assert child.state.in_episode
    child.sample(mb.POINT_IDLE, 130.0, CALM)
    assert not child.state.in_episode
    assert child.state.episode_level is None
    assert child.state.shrinks_this_episode == 0
    assert child.state.critical_idle_at is None
    assert child.state.unloaded_this_episode is False


def test_opening_an_episode_resets_no_carried_field():
    state = BrakeState(shrinks_this_episode=2, unloaded_this_episode=True, episode_level=CRITICAL)
    decision = _one(state, point=mb.POINT_TURN_END, level=WARNING)
    assert decision.state.in_episode
    assert decision.state.shrinks_this_episode == 2
    assert decision.state.unloaded_this_episode
    assert decision.state.episode_level == CRITICAL


def _shrunk_child():
    child = _Child()
    child.turn(0.0, 10.0, WARNING)
    assert child.state.cap == int(W * SHRINK_FACTOR)
    return child


def _calm(child, n, start):
    for i in range(n):
        child.sample(mb.POINT_IDLE, start + 60.0 * i, CALM)
    return start + 60.0 * n


def test_calm_grows_the_window_back_one_step_at_a_time():
    child = _shrunk_child()
    now = _calm(child, CALM_SAMPLES_TO_GROW - 1, 100.0)
    assert child.count("grow") == 0
    _calm(child, 1, now)
    assert child.count("grow") == 1
    # Never back to the size that hurt while it is remembered.
    assert child.state.cap == int(0.9 * W)
    assert child.state.calm_since_grow == 0


def test_a_grow_that_changes_nothing_is_nothing():
    child = _shrunk_child()
    now = _calm(child, CALM_SAMPLES_TO_GROW, 100.0)
    assert child.count("grow") == 1
    _calm(child, CALM_SAMPLES_TO_GROW, now)
    # 0.9 x faulty again: no change, no action.
    assert child.count("grow") == 1
    assert child.state.cap == int(0.9 * W)


def test_the_faulty_window_is_forgotten_after_the_calm_samples():
    child = _shrunk_child()
    _calm(child, CALM_SAMPLES_TO_FORGET, 100.0)
    assert child.state.faulty is None


def test_the_cap_is_dropped_once_growth_reaches_the_static_window():
    child = _shrunk_child()
    now = _calm(child, CALM_SAMPLES_TO_FORGET, 100.0)
    _calm(child, CALM_SAMPLES_TO_GROW, now)
    assert child.state.cap is None
    assert child.window == W


def test_no_growth_inside_an_episode():
    state = BrakeState(cap=20_000, in_episode=True, calm_since_grow=5, calm_count=0)
    decision = _one(state, level=CALM)
    assert decision.cap_change is None
    assert decision.state.cap == 20_000


def test_a_non_calm_sample_restarts_every_calm_count():
    state = BrakeState(calm_count=1, calm_since_grow=2, calm_since_pressure=9)
    decision = _one(state, level=WARNING, last_request_tokens=10)
    assert decision.state.calm_count == 0
    assert decision.state.calm_since_grow == 0
    assert decision.state.calm_since_pressure == 0


# ===================== the window =====================


def test_no_known_window_falls_back_to_the_memory_ceiling():
    decision = _one(point=mb.POINT_TURN_END, window=None, ceiling=W)
    assert decision.cap_change == "shrink"
    assert decision.state.cap == int(W * SHRINK_FACTOR)


def test_neither_window_nor_ceiling_is_inert():
    decision = _one(point=mb.POINT_TURN_END, level=CRITICAL, window=None, ceiling=None)
    assert decision.actions == ()
    assert not decision.state.in_episode
