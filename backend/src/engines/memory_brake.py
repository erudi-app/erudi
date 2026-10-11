"""The reactive memory brake's control law (Apple Silicon, MLX) -- pure.

The MLX child's memory grows with the conversation (its KV cache and prefix
cache fill lazily), so when the Mac comes under memory pressure the engine can
give some of it back. This module decides WHAT to do; the engine
(``MLX_Engine.brake_point``) reads the sensors, calls :func:`step` and
executes the decision. Nothing here does I/O, so every rule is table-tested.

Samples are taken at SAFE POINTS only -- the start of a turn, its end, and
idle -- never during a generation. A sample carries the system pressure level
(``kern.memorystatus_vm_pressure_level``: 1 calm, 2 warning, 4 critical), the
system's swap-outs during the turn (at ``turn_end`` only), what the child uses
above its base footprint, and the size of the last turn's largest request.

The law is built on PRESSURE EPISODES, so that every action is bounded per
episode instead of being taken again at every sample (a permanent warning
must not shrink the window to the floor nor release every minute):

* An episode opens at the first non-calm sample (a swap burst at ``turn_end``
  counts as a warning) and ends after ``CALM_SAMPLES_TO_END`` calm samples in
  a row. Opening one resets nothing; only its end resets the per-episode
  counters.
* **Release** (the engine resets the prefix cache, then sends a non-streamed
  1-token request, after which mlx_vlm clears MLX's buffer cache): at an idle
  sample under critical pressure, or under warning once the user has been away
  ``WARNING_RELEASE_IDLE_S`` (a reader keeps the prefix cache); at ``turn_end``
  under critical pressure; never at ``turn_start``. It always needs what it
  would give back NOW to reach ``ATTRIBUTION_FLOOR_BYTES``, measured above what
  the last release left -- so a release needs a turn to have re-allocated.
* **Shrink** (one step of the working window, ``SHRINK_FACTOR``, never under
  ``MIN_KV_BOUND_TOKENS``): one per episode under warning, one more once it is
  critical, never two without a turn between them, and only when it is useful
  -- the last turn's largest request reaches the compaction threshold of the
  shrunk window, so the next turn compacts and frees KV.
* **Unload** (idle only): critical at two consecutive idle samples with no
  turn between them (a calm or warning idle sample in between starts over),
  at most once per episode (a brake reload carries the
  episode), never within ``RELOAD_GRACE_S`` of a brake reload, and only when
  the child is worth it (``UNLOAD_MIN_SHARE`` of the memory total).
* **Grow** (outside an episode): every ``CALM_SAMPLES_TO_GROW`` calm samples,
  one step back up, never to ``0.9 x`` the window that hurt while it is
  remembered (forgotten after ``CALM_SAMPLES_TO_FORGET`` calm samples), and
  the cap is dropped once growth reaches the window without brake.
* **Notice**: ``"acted"`` when a shrink happens (the conversation is now
  summarized earlier); ``"external"`` when the pressure is not ours -- what a
  release would give back is under the floor and no useful shrink is left --
  never over ``"acted"``, and never on a child a brake reload seeded before
  its episode ends. A release alone sets no notice: the user loses nothing
  visible. The notice waits for a stateful turn end to consume it.

Idle samples are paced: one is skipped within ``BRAKE_IDLE_INTERVAL_S`` (less a
few seconds of tolerance, ``IDLE_PACING_TOLERANCE_S``) of the last sample or of
the end of the last turn.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

from src.engines.working_window import COMPACTION_WINDOW_FRACTION, MIN_KV_BOUND_TOKENS

# QA-tuned starting values.
SHRINK_FACTOR = 0.75
CALM_SAMPLES_TO_END = 2
CALM_SAMPLES_TO_GROW = 3
CALM_SAMPLES_TO_FORGET = 10
BRAKE_IDLE_INTERVAL_S = 60
# The idle monitor's period IS the pacing interval, so a tick that comes a
# few seconds short of it (scheduling jitter, the lock) is still the next
# sample, not one to skip.
IDLE_PACING_TOLERANCE_S = 5
# Below the 300 s idle reap, above a reading pause.
WARNING_RELEASE_IDLE_S = 180
RELOAD_GRACE_S = 600
UNLOAD_MIN_SHARE = 0.05
# 64 MiB of swap-outs during one turn, in pages (16 KiB on Apple Silicon).
SWAP_BURST_PAGES = 64 * 1024**2 // 16384
ATTRIBUTION_FLOOR_BYTES = 256 * 1024**2
# Growth never goes back to this fraction of the window that hurt while the
# brake remembers it.
FAULTY_GROW_LIMIT = 0.9
# How long the engine keeps what an unload carries to the next spawn.
BRAKE_RELOAD_TTL_S = 1800

POINT_TURN_START = "turn_start"
POINT_TURN_END = "turn_end"
POINT_IDLE = "idle"

LEVEL_CALM = 1
LEVEL_WARNING = 2
LEVEL_CRITICAL = 4

NOTICE_ACTED = "acted"
NOTICE_EXTERNAL = "external"

# The fields an unload carries to the next spawn of the same model.
_CARRIED = (
    "cap",
    "faulty",
    "in_episode",
    "episode_level",
    "shrinks_this_episode",
    "unloaded_this_episode",
)


@dataclass
class BrakeState:
    """The brake's memory of one child (kept on the engine handle)."""

    # The working window the brake imposes (None: no brake cap).
    cap: Optional[int] = None
    # The window that was under pressure at the last shrink (growth stays
    # under it while remembered).
    faulty: Optional[int] = None
    in_episode: bool = False
    episode_level: Optional[int] = None
    shrinks_this_episode: int = 0
    # Consecutive calm samples (the end of an episode).
    calm_count: int = 0
    # Calm samples since the last grow or the last non-calm sample.
    calm_since_grow: int = 0
    # Calm samples since the last non-calm one (forgetting ``faulty``).
    calm_since_pressure: int = 0
    # When the previous idle sample found critical pressure (None after a turn).
    critical_idle_at: Optional[float] = None
    last_sample_at: Optional[float] = None
    # What the child kept above base right after the last release.
    excess_after_release: int = 0
    # When a brake reload spawned this child.
    reloaded_at: Optional[float] = None
    unloaded_this_episode: bool = False
    # A brake reload seeded this child and its episode has not ended: no
    # ``external`` notice (the conversation uses little BECAUSE we unloaded it).
    reload_quiet: bool = False
    # A turn ran since the previous sample (set when the turn's memory window
    # opens, abandoned turns included; cleared by every sample).
    turn_happened: bool = False
    # A shrink happened and no turn ran since.
    shrunk_since_turn: bool = False
    notice: Optional[str] = None

    def carry(self) -> Dict[str, Any]:
        """What an unload hands to the next spawn of the same model."""
        return {name: getattr(self, name) for name in _CARRIED}

    @classmethod
    def reloaded(cls, carry: Dict[str, Any], *, now: float) -> "BrakeState":
        """The state of a child a brake reload spawned: the carried episode
        and cap, the reload time (the grace), and no ``external`` notice until
        the episode ends."""
        return cls(
            **{name: carry.get(name, getattr(cls, name)) for name in _CARRIED},
            reloaded_at=now,
            reload_quiet=True,
        )

    def take_notice(self) -> Optional[str]:
        """The pending notice, consumed (a stateful turn end shows it)."""
        notice, self.notice = self.notice, None
        return notice


@dataclass(frozen=True)
class Sample:
    """What the sensors said at one safe point."""

    point: str
    now: float
    # Pressure level (1/2/4), or None when unreadable.
    level: Optional[int]
    # System swap-outs during OUR turn, in pages (``turn_end`` only).
    swapped_pages: Optional[int] = None
    # The child's footprint above its base.
    excess_bytes: Optional[int] = None
    # The largest request of the last measured turn.
    last_request_tokens: Optional[int] = None
    # When the last turn ended (the engine's idle clock).
    last_used: Optional[float] = None


@dataclass(frozen=True)
class Context:
    """The child's windows, from the caller."""

    # The current working window (None: the memory ceiling stands in).
    window: Optional[int]
    # The memory ceiling, in tokens.
    ceiling: Optional[int]
    # The working window without brake: min(memory ceiling, trained window).
    static_window: Optional[int]
    # The child's footprint over the memory total.
    footprint_share: Optional[float]


@dataclass(frozen=True)
class Decision:
    """What one sample decided; ``state`` is the brake's new state."""

    state: BrakeState
    level: Optional[int] = None
    skipped: bool = False
    release: bool = False
    # "shrink" or "grow" (``state.cap`` holds the new cap), or None.
    cap_change: Optional[str] = None
    unload: bool = False

    @property
    def actions(self) -> tuple:
        """The actions to execute, in order."""
        if self.unload:
            return ("unload",)
        out = []
        if self.release:
            out.append("release")
        if self.cap_change is not None:
            out.append(self.cap_change)
        return tuple(out)


@dataclass(frozen=True)
class BrakeOutcome:
    """What one safe point did, as the engine reports it: whether it acted,
    and which action(s) (``"release+shrink"`` when a sample takes two)."""

    acted: bool = False
    action: Optional[str] = None


def _positive(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def _effective_level(sample: Sample) -> Optional[int]:
    """The sample's level on the 1/2/4 scale; a swap burst during the turn
    counts as a warning."""
    if sample.level is None:
        return None
    if sample.level >= LEVEL_CRITICAL:
        level = LEVEL_CRITICAL
    elif sample.level >= LEVEL_WARNING:
        level = LEVEL_WARNING
    else:
        level = LEVEL_CALM
    if (
        sample.point == POINT_TURN_END
        and sample.swapped_pages is not None
        and sample.swapped_pages >= SWAP_BURST_PAGES
    ):
        level = max(level, LEVEL_WARNING)
    return level


def step(state: BrakeState, sample: Sample, context: Context) -> Decision:
    """Apply one sample to ``state``; returns the decision and the new state
    (``state`` itself is never modified)."""
    state = replace(state)
    level = _effective_level(sample)
    # 1. Idle pacing.
    if sample.point == POINT_IDLE:
        marks = [mark for mark in (state.last_sample_at, sample.last_used) if mark is not None]
        if marks and sample.now - max(marks) < BRAKE_IDLE_INTERVAL_S - IDLE_PACING_TOLERANCE_S:
            return Decision(state=state, level=level, skipped=True)
    # 1b. The turn seen since the previous sample: no unload across it, and
    # the next shrink is allowed.
    if state.turn_happened:
        state.critical_idle_at = None
        state.shrunk_since_turn = False
        state.turn_happened = False
    state.last_sample_at = sample.now
    # 2. An unreadable sensor: nothing.
    if level is None:
        return Decision(state=state)
    # An unload needs two CONSECUTIVE idle criticals: any other idle sample
    # forgets the previous one.
    if sample.point == POINT_IDLE and level != LEVEL_CRITICAL:
        state.critical_idle_at = None
    window = _positive(context.window) or _positive(context.ceiling)
    if window is None:
        # Neither a working window nor a memory ceiling: inert.
        return Decision(state=state, level=level)
    if level == LEVEL_CALM:
        return _calm(state, context, level)
    return _pressure(state, sample, context, level, window)


def _calm(state: BrakeState, context: Context, level: int) -> Decision:
    state.calm_count += 1
    state.calm_since_grow += 1
    state.calm_since_pressure += 1
    if state.calm_count >= CALM_SAMPLES_TO_END and (state.in_episode or state.reload_quiet):
        state.in_episode = False
        state.episode_level = None
        state.shrinks_this_episode = 0
        state.critical_idle_at = None
        state.unloaded_this_episode = False
        state.reload_quiet = False
    if state.faulty is not None and state.calm_since_pressure >= CALM_SAMPLES_TO_FORGET:
        state.faulty = None
    if state.in_episode or state.cap is None or state.calm_since_grow < CALM_SAMPLES_TO_GROW:
        return Decision(state=state, level=level)
    state.calm_since_grow = 0
    target = int(state.cap / SHRINK_FACTOR)
    if state.faulty is not None:
        target = min(target, int(FAULTY_GROW_LIMIT * state.faulty))
    static = _positive(context.static_window)
    if static is not None and target >= static:
        state.cap = None
        return Decision(state=state, level=level, cap_change="grow")
    if target <= state.cap:
        return Decision(state=state, level=level)
    state.cap = target
    return Decision(state=state, level=level, cap_change="grow")


def _pressure(
    state: BrakeState, sample: Sample, context: Context, level: int, window: int
) -> Decision:
    state.calm_count = 0
    state.calm_since_grow = 0
    state.calm_since_pressure = 0
    state.in_episode = True
    state.episode_level = max(state.episode_level or 0, level)
    # What a release would give back NOW (the residual MLX keeps after a
    # release never counts twice).
    fresh = (
        sample.excess_bytes - state.excess_after_release
        if sample.excess_bytes is not None
        else None
    )
    if sample.point == POINT_IDLE and level == LEVEL_CRITICAL:
        grace_over = state.reloaded_at is None or sample.now - state.reloaded_at >= RELOAD_GRACE_S
        if (
            state.critical_idle_at is not None
            and not state.unloaded_this_episode
            and grace_over
            and context.footprint_share is not None
            and context.footprint_share >= UNLOAD_MIN_SHARE
        ):
            state.unloaded_this_episode = True
            state.critical_idle_at = None
            return Decision(state=state, level=level, unload=True)
        state.critical_idle_at = sample.now
    new_window = max(MIN_KV_BOUND_TOKENS, int(window * SHRINK_FACTOR))
    allowance = 2 if level == LEVEL_CRITICAL else 1
    shrink = (
        state.shrinks_this_episode < allowance
        and not state.shrunk_since_turn
        and new_window < window
        and sample.last_request_tokens is not None
        and sample.last_request_tokens >= COMPACTION_WINDOW_FRACTION * new_window
    )
    idle_for = sample.now - sample.last_used if sample.last_used is not None else None
    release = (
        fresh is not None
        and fresh >= ATTRIBUTION_FLOOR_BYTES
        and (
            (
                sample.point == POINT_IDLE
                and (
                    level == LEVEL_CRITICAL
                    or (idle_for is not None and idle_for >= WARNING_RELEASE_IDLE_S)
                )
            )
            or (sample.point == POINT_TURN_END and level == LEVEL_CRITICAL)
        )
    )
    if shrink:
        state.faulty = window
        state.cap = new_window
        state.shrinks_this_episode += 1
        state.shrunk_since_turn = True
        state.notice = NOTICE_ACTED
    elif (
        fresh is not None
        and fresh < ATTRIBUTION_FLOOR_BYTES
        and state.notice != NOTICE_ACTED
        and not state.reload_quiet
    ):
        state.notice = NOTICE_EXTERNAL
    return Decision(
        state=state,
        level=level,
        release=release,
        cap_change="shrink" if shrink else None,
    )
