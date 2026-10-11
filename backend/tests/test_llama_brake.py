"""The llama.cpp memory brake's control law (pure, no I/O).

``memory_brake.llama_step`` shares the MLX law's episodes, pacing, turn
bookkeeping and calm rules; what differs is the lever. llama-server allocates
its KV in full at load, so nothing frees without a respawn with a smaller
``-c`` -- and a smaller ``-c`` ALWAYS frees. So: no release; a shrink moves
the cap (warning: only with paging evidence; one per episode, two once
critical), a RESPAWN applies it (critical: at any safe point; warning: at an
idle sample once the user has been away ``LLAMA_WARNING_RESPAWN_IDLE_S``), at
most one per turn; the window grows back lazily (state only, applied at the
next spawn); the attribution is the KV share of RAM. The MLX ``step`` and its
tests are untouched.
"""

from __future__ import annotations

import pytest

from src.engines import memory_brake as mb
from src.engines.memory_brake import (
    BRAKE_IDLE_INTERVAL_S,
    CALM_SAMPLES_TO_END,
    CALM_SAMPLES_TO_FORGET,
    CALM_SAMPLES_TO_GROW,
    FAULTY_GROW_LIMIT,
    LLAMA_KV_SHARE_MIN,
    LLAMA_WARNING_DAMAGE_MIN_BYTES,
    LLAMA_WARNING_RESPAWN_IDLE_S,
    SHRINK_FACTOR,
    SWAP_BURST_BYTES,
    UNLOAD_MIN_SHARE,
    BrakeState,
    Context,
    Sample,
    llama_step,
)
from src.engines.working_window import MIN_KV_BOUND_TOKENS

pytestmark = pytest.mark.unit

MIB = 1024**2
W = 32_768
CALM, WARNING, CRITICAL = 1, 2, 4
DAMAGE = LLAMA_WARNING_DAMAGE_MIN_BYTES


class _Child:
    """One llama child under the law, as the engine drives it: a respawn
    makes the allocated window the cap; an unload ends it (the record
    carries the episode to the next spawn)."""

    def __init__(self, *, window=W, static=W, kv_share=0.3, share=0.4, resizable=True, state=None):
        self.state = state or BrakeState()
        self.window = window
        self.static = static
        self.kv_share = kv_share
        self.share = share
        self.resizable = resizable
        self.last_used = 0.0
        self.respawns = 0
        self.unloads = 0
        self.decisions = []

    def sample(self, point, now, level, *, damage=None, available=True):
        decision = llama_step(
            self.state,
            Sample(
                point=point,
                now=now,
                level=level,
                damage_bytes=damage,
                damage_available=available,
                last_used=self.last_used,
            ),
            Context(
                window=self.window,
                ceiling=None,
                static_window=self.static,
                footprint_share=self.share,
                resizable=self.resizable,
                kv_share=self.kv_share,
            ),
        )
        self.state = decision.state
        if decision.respawn:
            self.respawns += 1
            self.window = min(self.static, self.state.cap or self.static)
            self.state = BrakeState.respawned(self.state)
        if decision.unload:
            self.unloads += 1
        self.decisions.append((point, now, decision))
        return decision

    def turn(self, start, end, level, *, damage=None, available=True):
        self.sample("turn_start", start, level, damage=damage, available=available)
        self.state.turn_happened = True
        decision = self.sample("turn_end", end, level, damage=damage, available=available)
        self.last_used = end
        return decision


def _idle_ticks(child, start, count, level, **kw):
    out = []
    for i in range(count):
        out.append(child.sample("idle", start + i * BRAKE_IDLE_INTERVAL_S, level, **kw))
    return out


# ===================== shared shape =====================


def test_the_shape_changes_are_additive_and_mlx_defaults_hold():
    assert Sample(point="idle", now=0, level=1).damage_bytes is None
    assert Sample(point="idle", now=0, level=1).damage_available is False
    context = Context(window=1, ceiling=None, static_window=1, footprint_share=None)
    assert context.resizable is True and context.kv_share is None
    assert mb.Decision(state=BrakeState()).respawn is False
    assert BrakeState().respawned_this_turn is False


def test_respawned_keeps_the_live_state_and_sets_no_grace():
    live = BrakeState(
        cap=24_576,
        faulty=W,
        in_episode=True,
        episode_level=CRITICAL,
        shrinks_this_episode=1,
        critical_idle_at=123.0,
        shrunk_since_turn=True,
        notice="acted",
        calm_count=1,
        respawned_this_turn=True,
    )
    carried = BrakeState.respawned(live)
    assert carried == live
    assert carried is not live
    assert carried.reloaded_at is None
    assert carried.reload_quiet is False


def test_sensor_none_and_no_window_are_inert():
    child = _Child()
    assert child.sample("idle", 100, None).actions == ()
    child = _Child(window=None)
    decision = child.sample("idle", 100, CRITICAL)
    assert not decision.respawn and decision.cap_change is None


def test_idle_samples_are_paced_like_mlx():
    child = _Child()
    child.sample("idle", 1000, CALM)
    assert child.sample("idle", 1010, CALM).skipped


# ===================== warning: deferred shrink, idle respawn =====================


def test_a_warning_shrink_at_turn_end_is_deferred_to_an_idle_respawn():
    child = _Child()
    child.sample("turn_start", 100, CALM, damage=0)
    child.state.turn_happened = True
    decision = child.sample("turn_end", 110, WARNING, damage=DAMAGE)
    child.last_used = 110
    assert decision.cap_change == "shrink"
    assert child.state.cap == int(W * SHRINK_FACTOR)
    assert child.state.notice == "acted"
    assert not decision.respawn
    # The user reads: an idle tick at 60 s respawns nothing.
    assert not child.sample("idle", 110 + 60, WARNING, damage=0).respawn
    # Two ticks' worth of user idle: the cap is applied.
    decision = child.sample("idle", 110 + LLAMA_WARNING_RESPAWN_IDLE_S, WARNING, damage=0)
    assert decision.respawn
    assert child.window == int(W * SHRINK_FACTOR)
    # The acted notice survives the respawn.
    assert child.state.notice == "acted"


def test_a_permanent_warning_with_an_idle_user_shrinks_and_respawns_once():
    child = _Child()
    child.turn(100, 110, WARNING, damage=DAMAGE)
    decisions = _idle_ticks(child, 110 + 60, 20, WARNING, damage=DAMAGE)
    assert child.respawns == 1
    assert sum(1 for d in decisions if d.cap_change == "shrink") == 0
    assert child.state.shrinks_this_episode == 1
    assert all(not d.unload for d in decisions)


@pytest.mark.parametrize("point", ["idle", "turn_start"])
def test_warning_without_a_counter_lets_the_level_decide(point):
    child = _Child()
    child.last_used = 0
    decision = child.sample(point, 1000, WARNING, damage=None, available=False)
    assert decision.cap_change == "shrink"


@pytest.mark.parametrize("point", ["idle", "turn_start"])
def test_warning_with_a_small_paging_delta_does_nothing_and_says_nothing(point):
    child = _Child()
    decision = child.sample(point, 1000, WARNING, damage=DAMAGE - 1)
    assert decision.cap_change is None
    assert not decision.respawn
    assert child.state.notice is None
    assert child.state.in_episode


@pytest.mark.parametrize("point", ["idle", "turn_start"])
def test_warning_with_paging_shrinks(point):
    child = _Child()
    decision = child.sample(point, 1000, WARNING, damage=DAMAGE)
    assert decision.cap_change == "shrink"
    assert child.state.notice == "acted"


def test_a_swap_burst_at_turn_end_counts_as_warning():
    child = _Child()
    child.sample("turn_start", 100, CALM, damage=0)
    child.state.turn_happened = True
    decision = child.sample("turn_end", 110, CALM, damage=SWAP_BURST_BYTES)
    assert decision.level == WARNING
    assert decision.cap_change == "shrink"
    # Only with a counter.
    child = _Child()
    child.sample("turn_start", 100, CALM)
    decision = child.sample("turn_end", 110, CALM, damage=SWAP_BURST_BYTES, available=False)
    assert decision.level == CALM


# ===================== critical =====================


@pytest.mark.parametrize("point", ["turn_start", "turn_end", "idle"])
def test_critical_respawns_at_any_safe_point(point):
    child = _Child()
    decision = child.sample(point, 1000, CRITICAL)
    assert decision.cap_change == "shrink"
    assert decision.respawn
    assert child.window == int(W * SHRINK_FACTOR)


def test_one_respawn_per_turn():
    child = _Child()
    first = child.sample("turn_start", 1000, CRITICAL)
    assert first.respawn
    child.state.turn_happened = True
    # Still critical at the same turn's end: the cap may move, no second respawn.
    second = child.sample("turn_end", 1010, CRITICAL)
    assert second.cap_change == "shrink"
    assert not second.respawn
    assert child.respawns == 1
    # The next turn re-arms it.
    child.last_used = 1010
    third = child.sample("turn_start", 1100, CRITICAL)
    assert third.respawn
    assert child.respawns == 2


def test_an_idle_sample_re_arms_the_respawn():
    child = _Child()
    child.sample("turn_start", 1000, CRITICAL)
    child.state.turn_happened = True
    child.sample("turn_end", 1010, CRITICAL)
    child.last_used = 1010
    decision = child.sample("idle", 1010 + BRAKE_IDLE_INTERVAL_S, CRITICAL)
    assert decision.respawn


def test_the_shrink_steps_from_the_cap_not_the_allocated_window():
    child = _Child()
    child.turn(100, 110, WARNING, damage=DAMAGE)
    assert child.state.cap == int(W * SHRINK_FACTOR)
    assert child.window == W
    child.last_used = 110
    decision = child.sample("turn_start", 200, CRITICAL)
    assert decision.cap_change == "shrink"
    assert child.state.cap == int(int(W * SHRINK_FACTOR) * SHRINK_FACTOR)
    assert decision.respawn
    assert child.window == child.state.cap
    assert child.state.faulty == int(W * SHRINK_FACTOR)


def test_bounds_per_episode_across_respawns_and_an_unload():
    child = _Child()
    t = 1000.0
    for _ in range(6):
        child.turn(t, t + 10, CRITICAL)
        child.last_used = t + 10
        t += 100
    assert child.state.shrinks_this_episode == 2
    assert child.respawns == 2
    # Then two consecutive idle criticals: the unload, once.
    decisions = _idle_ticks(child, t, 6, CRITICAL)
    assert sum(1 for d in decisions if d.unload) == 1
    record = child.state.carry()
    reloaded = _Child(window=child.window, state=BrakeState.reloaded(record, now=t + 1000))
    decisions = [reloaded.turn(t + 1000 + i * 100, t + 1010 + i * 100, CRITICAL) for i in range(3)]
    decisions += _idle_ticks(reloaded, t + 2000 + mb.RELOAD_GRACE_S, 6, CRITICAL)
    assert not any(d.unload for d in decisions)
    assert not any(d.cap_change == "shrink" for d in decisions)
    assert reloaded.respawns == 0


def test_a_respawn_then_two_idle_criticals_unloads_without_a_grace():
    child = _Child()
    child.last_used = 0
    first = child.sample("idle", 1000, CRITICAL)
    assert first.respawn
    second = child.sample("idle", 1000 + BRAKE_IDLE_INTERVAL_S, CRITICAL)
    assert second.unload


def test_the_unload_needs_a_share_worth_it():
    child = _Child(share=UNLOAD_MIN_SHARE / 2)
    _idle_ticks(child, 1000, 4, CRITICAL)
    assert child.unloads == 0


def test_never_two_respawns_without_a_turn_or_an_idle_between():
    child = _Child()
    seen = []
    t = 1000.0
    for point in ["turn_start", "turn_end", "turn_end", "turn_start", "idle", "idle"]:
        decision = child.sample(point, t, CRITICAL)
        if point == "turn_start":
            child.state.turn_happened = True
        seen.append((point, decision.respawn))
        t += BRAKE_IDLE_INTERVAL_S
    respawn_points = [i for i, (_, r) in enumerate(seen) if r]
    for a, b in zip(respawn_points, respawn_points[1:]):
        between = [p for p, _ in seen[a + 1 : b + 1]]
        assert "turn_start" in between or "idle" in between


# ===================== attribution =====================


def test_a_small_kv_share_is_external_never_a_shrink():
    child = _Child(kv_share=LLAMA_KV_SHARE_MIN / 2)
    decision = child.sample("idle", 1000, CRITICAL)
    assert decision.cap_change is None
    assert not decision.respawn
    assert child.state.notice == "external"


def test_a_big_kv_share_on_a_tiny_machine_still_shrinks():
    child = _Child(kv_share=0.6, share=0.9)
    assert child.sample("turn_end", 1000, CRITICAL).cap_change == "shrink"


def test_external_never_overrides_acted_nor_lands_on_a_reload_quiet_child():
    child = _Child()
    child.sample("turn_end", 1000, CRITICAL)
    child.kv_share = 0.01
    child.state.turn_happened = True
    child.sample("turn_end", 1100, CRITICAL)
    assert child.state.notice == "acted"
    quiet = _Child(kv_share=0.01, state=BrakeState(reload_quiet=True, reloaded_at=0))
    quiet.sample("turn_end", 1000, CRITICAL)
    assert quiet.state.notice is None


def test_an_unknown_kv_share_neither_shrinks_nor_notifies():
    child = _Child(kv_share=None)
    decision = child.sample("turn_end", 1000, CRITICAL)
    assert decision.cap_change is None and not decision.respawn
    assert child.state.notice is None


# ===================== calm =====================


def test_grow_is_lazy_and_never_respawns():
    child = _Child()
    child.sample("turn_end", 1000, CRITICAL)
    assert child.window == int(W * SHRINK_FACTOR)
    decisions = _idle_ticks(child, 2000, CALM_SAMPLES_TO_END + CALM_SAMPLES_TO_GROW, CALM)
    grows = [d for d in decisions if d.cap_change == "grow"]
    assert grows
    assert not any(d.respawn for d in decisions)
    assert child.respawns == 1
    # Growth stays under the window that hurt while it is remembered.
    assert child.state.cap is None or child.state.cap <= int(FAULTY_GROW_LIMIT * W)


def test_an_unapplied_cap_is_dropped_at_the_end_of_the_episode():
    child = _Child()
    child.turn(100, 110, WARNING, damage=DAMAGE)
    assert child.state.cap is not None and child.window == W
    child.last_used = 110
    _idle_ticks(child, 200, CALM_SAMPLES_TO_END, CALM)
    assert not child.state.in_episode
    assert child.state.cap is None
    assert child.state.faulty == W
    _idle_ticks(child, 200 + 10 * BRAKE_IDLE_INTERVAL_S, CALM_SAMPLES_TO_FORGET, CALM)
    assert child.state.faulty is None


def test_an_applied_cap_survives_the_end_of_the_episode():
    child = _Child()
    child.sample("turn_end", 1000, CRITICAL)
    _idle_ticks(child, 2000, CALM_SAMPLES_TO_END, CALM)
    assert child.state.cap == int(W * SHRINK_FACTOR)


def test_the_floor_is_never_crossed():
    child = _Child(window=MIN_KV_BOUND_TOKENS + 100, static=MIN_KV_BOUND_TOKENS + 100)
    child.sample("turn_end", 1000, CRITICAL)
    assert child.state.cap == MIN_KV_BOUND_TOKENS
    child.state.turn_happened = True
    child.last_used = 1000
    decision = child.sample("turn_start", 1100, CRITICAL)
    assert decision.cap_change is None


# ===================== ERUDI_CTX pinned =====================


def test_a_pinned_window_is_never_resized_but_can_unload():
    child = _Child(resizable=False)
    decisions = [child.turn(100 + 100 * i, 110 + 100 * i, CRITICAL) for i in range(3)]
    assert not any(d.cap_change or d.respawn for d in decisions)
    assert child.state.notice != "acted"
    child.last_used = 400
    decisions = _idle_ticks(child, 1000, 3, CRITICAL)
    assert any(d.unload for d in decisions)


def test_a_pinned_window_does_not_grow_a_carried_cap():
    child = _Child(resizable=False, state=BrakeState(cap=20_000))
    decisions = _idle_ticks(child, 1000, 8, CALM)
    assert not any(d.cap_change for d in decisions)
    assert child.state.cap == 20_000


# ===================== oscillation =====================


def test_a_persistent_external_consumer_is_bounded_by_the_remembered_window():
    child = _Child()
    t = 1000.0
    for _ in range(4):
        child.turn(t, t + 10, CRITICAL)
        child.last_used = t + 10
        t += 100
        for d in _idle_ticks(child, t, CALM_SAMPLES_TO_END + 2 * CALM_SAMPLES_TO_GROW, CALM):
            pass
        t += 20 * BRAKE_IDLE_INTERVAL_S
        if child.state.faulty is not None and child.state.cap is not None:
            assert child.state.cap <= int(FAULTY_GROW_LIMIT * child.state.faulty)
