"""The memory brake, executed by the MLX engine (sensors and HTTP faked).

``MLX_Engine.brake_point`` reads the sensors, applies the control law
(``src.engines.memory_brake``) and executes its decision: a shrink or a grow
moves the bound the child enforces (``PATCH /v1/settings``, through the same
applier as the memory bound set at spawn), a release resets the prefix cache
then sends a non-streamed 1-token request, an unload reaps the child and
leaves the episode for the next spawn of the same model. An idle monitor
samples every ``BRAKE_IDLE_INTERVAL_S`` under the generation lock. Nothing
here talks to a real child.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from src.engines import base_engine as base_engine_module
from src.engines import memory_brake as mb
from src.engines import memory_budget
from src.engines import mlx_engine as mlx_engine_module
from src.engines import process_footprint
from src.engines.base_engine import BaseEngine
from src.engines.cuda_engine import CUDA_Engine
from src.engines.memory_brake import BrakeOutcome, BrakeState
from src.engines.memory_budget import MemoryBudget
from src.engines.mlx_engine import MLX_Engine

pytestmark = pytest.mark.unit

GIB = 1024**3
MIB = 1024**2
KEY = "brake-key-never-logged-0123456789"
BASE = 2 * GIB
TOTAL = 10 * GIB


class _Proc:
    def __init__(self):
        self.alive = True
        self.exitcode = None
        self.pid = 4321

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.alive = False
        self.exitcode = 0

    def join(self, timeout=None):
        return None

    def kill(self):
        self.alive = False


class _Resp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = ""

    def json(self):
        return self._payload


class _Http:
    """Records every request; answers success unless told otherwise."""

    def __init__(self):
        self.calls = []
        self.fail = {}
        self.live_max_kv_size = None

    def _record(self, method, url, **kwargs):
        path = url.split("27399", 1)[-1]
        self.calls.append({"method": method, "path": path, **kwargs})
        outcome = self.fail.get(path)
        if isinstance(outcome, list):
            outcome = outcome.pop(0) if outcome else None
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, int):
            return _Resp(outcome)
        return None

    def post(self, url, json=None, headers=None, timeout=None, **kwargs):
        failed = self._record("POST", url, json=json, headers=headers, timeout=timeout)
        return failed or _Resp(200, {"enabled": True, "status": "cleared"})

    def patch(self, url, json=None, headers=None, timeout=None, **kwargs):
        failed = self._record("PATCH", url, json=json, headers=headers, timeout=timeout)
        if failed:
            return failed
        self.live_max_kv_size = json["max_kv_size"]
        return _Resp(200, {"applied": {"max_kv_size": json["max_kv_size"]}})

    def get(self, url, headers=None, timeout=None, **kwargs):
        failed = self._record("GET", url, headers=headers, timeout=timeout)
        return failed or _Resp(200, {"current": {"max_kv_size": self.live_max_kv_size}})

    def paths(self):
        return [(call["method"], call["path"]) for call in self.calls]


class _Sensors:
    def __init__(self):
        self.level = 2
        self.footprint = BASE + GIB
        self.after_release = BASE + int(0.33 * GIB)
        self.swapouts = 1000
        self.level_reads = 0

    def install(self, monkeypatch):
        monkeypatch.setattr(process_footprint, "pressure_level", self._level)
        monkeypatch.setattr(process_footprint, "footprint", lambda pid: self.footprint)
        monkeypatch.setattr(process_footprint, "swapouts", lambda: self.swapouts)
        monkeypatch.setattr(process_footprint, "begin_peak_window", lambda pid: True)
        monkeypatch.setattr(process_footprint, "peak_since", lambda pid, started: self.footprint)
        # The memory total comes from the GPU working set, which only a Mac
        # reports: pinned, so the unload's share check does not depend on the
        # host running the tests.
        monkeypatch.setattr(memory_budget, "total_memory_bytes", lambda engine: TOTAL)

    def _level(self):
        self.level_reads += 1
        return self.level


def _reset_class_state():
    MLX_Engine._model = None
    MLX_Engine._tokenizer = None
    MLX_Engine._model_id = None
    MLX_Engine._last_used = None
    MLX_Engine._brake_reload = None
    MLX_Engine._brake_sensor_warned = False
    BaseEngine._pending_reset = None


@pytest.fixture(autouse=True)
def _clean_engine():
    _reset_class_state()
    yield
    _reset_class_state()


@pytest.fixture
def sensors(monkeypatch):
    fake = _Sensors()
    fake.install(monkeypatch)
    return fake


@pytest.fixture
def http(monkeypatch):
    fake = _Http()
    monkeypatch.setattr(mlx_engine_module.requests, "post", fake.post)
    monkeypatch.setattr(mlx_engine_module.requests, "patch", fake.patch)
    monkeypatch.setattr(mlx_engine_module.requests, "get", fake.get)
    return fake


@pytest.fixture
def ceiling(monkeypatch):
    """The memory ceiling the budget reports, in tokens (mutable)."""
    value = {"tokens": 20_000}

    def _from_handle(cls, engine, handle):
        return SimpleNamespace(tokens_at_ceiling=lambda: value["tokens"], total_bytes=TOTAL)

    monkeypatch.setattr(MemoryBudget, "from_handle", classmethod(_from_handle))
    return value


def _handle(**extra):
    handle = {
        "pid": 4321,
        "proc": _Proc(),
        "port": 27399,
        "base_url": "http://127.0.0.1:27399",
        "alias": "erudi-7",
        "model_path": "/models/qwen",
        "api_key": KEY,
        # PR3.3: the ceiling (20 000) binds below the trained window.
        "context_tokens": 20_000,
        "trained_context_tokens": 32_768,
        "memory_bound_tokens": 20_000,
        "base_footprint_bytes": BASE,
        "prefix_owner": "conv:1",
        "brake": BrakeState(),
        "last_request_tokens": 19_000,
    }
    handle.update(extra)
    return handle


@pytest.fixture
def child(sensors, http, ceiling, monkeypatch):
    # The residual is read over a bounded settle window; tests run it at speed.
    monkeypatch.setattr(MLX_Engine, "RELEASE_SETTLE_S", 0.0)
    handle = _handle()
    MLX_Engine._model = handle
    MLX_Engine._model_id = 7
    MLX_Engine._tokenizer = {"type": "remote"}
    MLX_Engine._last_used = datetime.now() - timedelta(seconds=1000)
    return handle


def _brake_lines(caplog):
    return [r for r in caplog.records if "Memory brake" in r.getMessage()]


def _assert_clean(caplog):
    for record in caplog.records:
        message = record.getMessage()
        assert KEY not in message
        assert "Bearer" not in message
        assert message.isascii(), message


# ===================== shrink and grow =====================


def test_a_shrink_patches_the_child_to_the_brake_cap(child, http, caplog):
    with caplog.at_level(logging.INFO):
        outcome = MLX_Engine.brake_point("turn_end")

    assert outcome == BrakeOutcome(acted=True, action="shrink")
    patches = [c for c in http.calls if c["method"] == "PATCH"]
    assert [c["json"] for c in patches] == [{"max_kv_size": 15_000}]
    assert patches[0]["headers"] == {"Authorization": f"Bearer {KEY}"}
    assert child["context_tokens"] == 15_000
    assert MLX_Engine.memory_bound_tokens() == 15_000
    assert child["brake"].cap == 15_000
    (line,) = _brake_lines(caplog)
    assert line.levelno == logging.INFO
    assert line.getMessage() == (
        "[MLX_Engine] Memory brake: point=turn_end, level=2, action=shrink, "
        "window=20000->15000, excess_mib=1024"
    )
    _assert_clean(caplog)


def test_the_bound_is_the_smaller_of_the_ceiling_and_the_cap(child, http, ceiling):
    child["brake"] = BrakeState(cap=25_000)
    ceiling["tokens"] = 20_000
    MLX_Engine._apply_memory_bound(child)
    # The ceiling binds below the cap: PR3.3's bound, no PATCH (already there).
    assert child["context_tokens"] == 20_000
    assert http.calls == []
    child["brake"] = BrakeState(cap=12_000)
    MLX_Engine._apply_memory_bound(child)
    assert http.live_max_kv_size == 12_000
    assert child["context_tokens"] == 12_000


def test_a_grow_back_to_the_static_window_restores_the_spawn_bound(child, http, sensors, caplog):
    child["context_tokens"] = 15_000
    child["memory_bound_tokens"] = 15_000
    child["brake"] = BrakeState(cap=15_000, calm_since_grow=2, calm_count=5)
    http.live_max_kv_size = 15_000
    sensors.level = 1

    with caplog.at_level(logging.INFO):
        outcome = MLX_Engine.brake_point("idle")

    assert outcome.action == "grow"
    assert child["brake"].cap is None
    assert http.live_max_kv_size == 20_000
    assert child["context_tokens"] == 20_000
    assert MLX_Engine.memory_bound_tokens() == 20_000
    (line,) = _brake_lines(caplog)
    assert "action=grow, window=15000->20000" in line.getMessage()


def test_a_grow_back_where_the_model_binds_restores_the_trained_window(child, http, ceiling):
    # Memory no longer binds at all: the trained window comes back and no
    # memory bound is recorded.
    ceiling["tokens"] = 40_000
    child["context_tokens"] = 24_576
    child["memory_bound_tokens"] = 24_576
    child["brake"] = BrakeState(cap=None)

    MLX_Engine._apply_memory_bound(child)

    assert http.live_max_kv_size == 32_768
    assert child["context_tokens"] == 32_768
    assert MLX_Engine.memory_bound_tokens() is None


def test_an_unbounded_child_keeps_its_last_bound_when_memory_stops_binding(child, http, ceiling):
    ceiling["tokens"] = None
    child["trained_context_tokens"] = None
    child["context_tokens"] = 12_000
    child["memory_bound_tokens"] = 12_000
    child["brake"] = BrakeState(cap=None)

    MLX_Engine._apply_memory_bound(child)

    assert http.calls == []
    assert child["context_tokens"] == 12_000
    assert MLX_Engine.memory_bound_tokens() is None


# ===================== release =====================


def test_a_release_resets_then_sends_a_short_non_streamed_barrier(
    child, http, sensors, monkeypatch, caplog
):
    sensors.level = 4
    child["last_request_tokens"] = 10  # no useful shrink: the release alone
    child["abandoned"] = True

    def _footprint(pid):
        # Once the cache is reset and the barrier answered, MLX's buffer
        # cache is gone: the measured residual remains.
        if ("POST", "/v1/cache/reset") in http.paths():
            return sensors.after_release
        return sensors.footprint

    monkeypatch.setattr(process_footprint, "footprint", _footprint)
    with caplog.at_level(logging.INFO):
        outcome = MLX_Engine.brake_point("idle")

    assert outcome == BrakeOutcome(acted=True, action="release")
    assert http.paths() == [
        ("POST", "/v1/chat/completions"),
        ("POST", "/v1/cache/reset"),
        ("POST", "/v1/chat/completions"),
    ]
    for barrier in (http.calls[0], http.calls[2]):
        assert barrier["json"]["stream"] is False
        assert barrier["json"]["max_tokens"] == 1
        assert barrier["timeout"] == MLX_Engine.RELEASE_BARRIER_TIMEOUT_S == 30
    # The prefix owner is kept; what the release left is remembered.
    assert child["prefix_owner"] == "conv:1"
    assert child["brake"].excess_after_release == int(0.33 * GIB)
    lines = _brake_lines(caplog)
    assert len(lines) == 1
    assert lines[0].levelno == logging.INFO
    assert "action=release" in lines[0].getMessage()
    assert "excess_mib=1024->337" in lines[0].getMessage()
    _assert_clean(caplog)


def test_the_residual_is_read_once_the_footprint_has_settled(child, http, sensors, monkeypatch):
    # mlx_vlm clears its buffer cache before answering the barrier, but the
    # footprint the kernel reports catches up a moment later: the residual is
    # the lowest reading over the bounded settle window, not the first one.
    sensors.level = 4
    child["last_request_tokens"] = 10
    monkeypatch.setattr(MLX_Engine, "RELEASE_SETTLE_S", 0.2)
    monkeypatch.setattr(MLX_Engine, "_RELEASE_SETTLE_POLL_S", 0.01)
    readings = iter([BASE + 656 * MIB, BASE + 500 * MIB, BASE + 340 * MIB])

    def _footprint(pid):
        if ("POST", "/v1/cache/reset") not in http.paths():
            return sensors.footprint
        return next(readings, BASE + 340 * MIB)

    monkeypatch.setattr(process_footprint, "footprint", _footprint)
    started = time.monotonic()

    MLX_Engine.brake_point("idle")

    assert child["brake"].excess_after_release == 340 * MIB
    # Bounded: the window, plus one poll.
    assert time.monotonic() - started < 1.0


def test_a_failed_release_barrier_names_the_release_not_a_claim(child, http, sensors, caplog):
    sensors.level = 4
    child["last_request_tokens"] = 10
    http.fail["/v1/chat/completions"] = 500

    with caplog.at_level(logging.INFO):
        MLX_Engine.brake_point("idle")

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "release" in message
    assert "deferred to the next claim" not in message
    assert "HTTP 500" in message
    _assert_clean(caplog)


def test_a_claim_barrier_keeps_its_own_text(child, http, caplog):
    # The claim's barrier is unchanged by the release's wording.
    child["abandoned"] = True
    http.fail["/v1/chat/completions"] = 500
    with caplog.at_level(logging.INFO):
        MLX_Engine.claim_prefix("conv:2")
    (warning,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert "reset deferred to the next claim" in warning.getMessage()
    assert http.calls[0]["timeout"] != MLX_Engine.RELEASE_BARRIER_TIMEOUT_S


# ===================== unload and the carry =====================


def test_two_idle_criticals_unload_and_carry_the_episode(child, http, sensors, caplog):
    sensors.level = 4
    child["last_request_tokens"] = 10
    with caplog.at_level(logging.INFO):
        MLX_Engine.brake_point("idle")
        child["brake"].last_sample_at -= mb.BRAKE_IDLE_INTERVAL_S
        outcome = MLX_Engine.brake_point("idle")

    assert outcome == BrakeOutcome(acted=True, action="unload")
    assert MLX_Engine._model is None
    assert child["proc"].alive is False
    record = MLX_Engine._brake_reload
    assert record["alias"] == "erudi-7"
    assert record["unloaded_this_episode"] is True
    assert record["in_episode"] is True
    assert record["episode_level"] == 4
    assert set(record) == {
        "alias",
        "cap",
        "faulty",
        "in_episode",
        "episode_level",
        "shrinks_this_episode",
        "unloaded_this_episode",
        "unloaded_at",
    }
    assert time.time() - record["unloaded_at"] < 5
    unloads = [r for r in _brake_lines(caplog) if "action=unload" in r.getMessage()]
    assert len(unloads) == 1
    assert unloads[0].levelno == logging.WARNING
    _assert_clean(caplog)
    assert MLX_Engine.brake_reload_pending(7)
    assert not MLX_Engine.brake_reload_pending(8)


def _spawned_handle(alias="erudi-7"):
    handle = _handle(alias=alias, context_tokens=32_768, memory_bound_tokens=None)
    del handle["brake"]
    del handle["last_request_tokens"]
    return handle


def test_a_same_alias_spawn_within_the_ttl_is_seeded_with_the_carry(sensors, http, ceiling):
    MLX_Engine._brake_reload = {
        "alias": "erudi-7",
        "cap": 12_000,
        "faulty": 16_000,
        "in_episode": True,
        "episode_level": 4,
        "shrinks_this_episode": 2,
        "unloaded_this_episode": True,
        "unloaded_at": time.time() - 30,
    }
    handle = _spawned_handle()

    MLX_Engine._read_server_properties(handle)

    state = handle["brake"]
    assert state.cap == 12_000
    assert state.faulty == 16_000
    assert state.in_episode and state.episode_level == 4
    assert state.shrinks_this_episode == 2
    assert state.unloaded_this_episode
    assert time.time() - state.reloaded_at < 5
    # The bound is min(ceiling, cap): the reload resumes with the reduced window.
    assert http.live_max_kv_size == 12_000
    assert handle["context_tokens"] == 12_000
    assert MLX_Engine._brake_reload is None


def test_another_model_is_not_seeded_and_its_spawn_clears_the_record(sensors, http, ceiling):
    MLX_Engine._brake_reload = {
        "alias": "erudi-7",
        "cap": 12_000,
        "faulty": None,
        "in_episode": True,
        "episode_level": 4,
        "shrinks_this_episode": 1,
        "unloaded_this_episode": True,
        "unloaded_at": time.time(),
    }
    handle = _spawned_handle(alias="erudi-8")

    MLX_Engine._read_server_properties(handle)

    assert handle["brake"] == BrakeState()
    assert handle["context_tokens"] == 20_000
    assert MLX_Engine._brake_reload is None


def test_an_expired_record_is_not_seeded(sensors, http, ceiling):
    MLX_Engine._brake_reload = {
        "alias": "erudi-7",
        "cap": 12_000,
        "faulty": None,
        "in_episode": True,
        "episode_level": 4,
        "shrinks_this_episode": 1,
        "unloaded_this_episode": True,
        "unloaded_at": time.time() - mb.BRAKE_RELOAD_TTL_S - 1,
    }
    assert not MLX_Engine.brake_reload_pending(7)
    handle = _spawned_handle()

    MLX_Engine._read_server_properties(handle)

    assert handle["brake"] == BrakeState()


def test_a_pending_reload_needs_no_child_loaded(child):
    MLX_Engine._brake_reload = {"alias": "erudi-7", "unloaded_at": time.time()}
    assert not MLX_Engine.brake_reload_pending(7)
    MLX_Engine._model = None
    assert MLX_Engine.brake_reload_pending(7)


async def test_the_idle_timeout_reap_forgets_the_record(child):
    MLX_Engine._brake_reload = {"alias": "erudi-9", "unloaded_at": time.time()}
    MLX_Engine._last_used = datetime.now() - timedelta(seconds=MLX_Engine._max_idle_time + 10)

    await MLX_Engine._cleanup_tick()

    assert MLX_Engine._model is None
    assert MLX_Engine._brake_reload is None


def test_shutdown_forgets_the_record():
    MLX_Engine._brake_reload = {"alias": "erudi-7", "unloaded_at": time.time()}
    MLX_Engine.cleanup()
    assert MLX_Engine._brake_reload is None


def test_the_base_engine_never_has_a_reload_pending():
    assert BaseEngine.brake_reload_pending(7) is False
    assert CUDA_Engine.brake_reload_pending(7) is False
    assert CUDA_Engine.brake_point("idle") == BrakeOutcome()
    assert CUDA_Engine.consume_brake_notice() is None


# ===================== turn_start, robustness, sensors =====================


def test_turn_start_samples_only_the_child_this_turn_will_use(child, sensors):
    MLX_Engine.brake_point("turn_start", llm_id=8)
    assert sensors.level_reads == 0
    child["proc"].alive = False
    MLX_Engine.brake_point("turn_start", llm_id=7)
    assert sensors.level_reads == 0
    child["proc"].alive = True
    MLX_Engine.brake_point("turn_start", llm_id=7)
    assert sensors.level_reads == 1


def test_loaded_child_for_has_no_side_effect(child, caplog):
    child["proc"].alive = False
    with caplog.at_level(logging.INFO):
        assert MLX_Engine.loaded_child_for(7) is False
    assert caplog.records == []
    child["proc"].alive = True
    assert MLX_Engine.loaded_child_for(7) is True
    assert MLX_Engine.loaded_child_for(8) is False


def test_the_brake_never_raises(child, monkeypatch, caplog):
    def _boom():
        raise RuntimeError(f"sensor exploded {KEY}")

    monkeypatch.setattr(process_footprint, "pressure_level", _boom)
    with caplog.at_level(logging.INFO):
        assert MLX_Engine.brake_point("turn_end") == BrakeOutcome()
    (warning,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert "RuntimeError" in warning.getMessage()
    _assert_clean(caplog)


def test_an_unreadable_sensor_turns_the_brake_off_with_one_warning(child, sensors, http, caplog):
    sensors.level = None
    with caplog.at_level(logging.INFO):
        for _ in range(3):
            assert MLX_Engine.brake_point("turn_end") == BrakeOutcome()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "brake is off" in warnings[0].getMessage()
    assert http.calls == []


def test_no_child_no_sample(sensors):
    assert MLX_Engine.brake_point("idle") == BrakeOutcome()
    assert sensors.level_reads == 0


def test_the_turn_end_sample_reads_the_turns_own_swap(child, sensors, http):
    sensors.level = 1
    child["memory_window"] = {"swapouts_start": 1000}
    sensors.swapouts = 1000 + mb.SWAP_BURST_PAGES
    MLX_Engine.brake_point("turn_end")
    assert child["brake"].in_episode


def test_consume_brake_notice_takes_it_once(child):
    child["brake"] = BrakeState(notice=mb.NOTICE_ACTED)
    assert MLX_Engine.consume_brake_notice() == mb.NOTICE_ACTED
    assert MLX_Engine.consume_brake_notice() is None


# ===================== the turn's own measurement =====================


def test_the_window_records_the_last_request_even_without_the_spi(child, sensors, monkeypatch):
    monkeypatch.setattr(process_footprint, "begin_peak_window", lambda pid: False)
    child["brake"].turn_happened = False
    token = MLX_Engine.begin_memory_window()
    # The previous turn's value stands until this turn produces one.
    assert child["last_request_tokens"] == 19_000
    assert child["brake"].turn_happened is True
    MLX_Engine.note_call_usage(child, 9_000, 500)
    MLX_Engine.note_call_usage(child, 12_000, 300)

    assert MLX_Engine.end_memory_window(token) is None  # no SPI: nothing recorded
    assert child["last_request_tokens"] == 12_300


def test_an_abandoned_turn_records_its_estimate(child, sensors, monkeypatch):
    monkeypatch.setattr(process_footprint, "begin_peak_window", lambda pid: False)
    token = MLX_Engine.begin_memory_window()
    MLX_Engine.note_call_start(child, 18_000)

    MLX_Engine.end_memory_window(token, abandoned=True)

    assert child["last_request_tokens"] == 18_000


def test_a_base_without_measure_still_records_the_last_request(child, sensors):
    child["base_footprint_bytes"] = None
    token = MLX_Engine.begin_memory_window()
    MLX_Engine.note_call_usage(child, 7_000, 100)
    MLX_Engine.end_memory_window(token)
    assert child["last_request_tokens"] == 7_100


def test_a_recovered_error_turn_records_its_in_flight_estimate(child, sensors):
    # A first-chunk timeout under swap: no usage ever arrived, the turn was
    # NOT abandoned -- its estimate is still the size of what it sent.
    token = MLX_Engine.begin_memory_window()
    MLX_Engine.note_call_start(child, 21_000)

    MLX_Engine.end_memory_window(token, abandoned=False)

    assert child["last_request_tokens"] == 21_000


def test_a_turn_with_nothing_measured_keeps_the_previous_value(child, sensors):
    token = MLX_Engine.begin_memory_window()

    MLX_Engine.end_memory_window(token)

    assert child["last_request_tokens"] == 19_000


# ===================== review fixes: failures are not successes =====================


def _past_pacing(handle):
    handle["brake"].last_sample_at -= mb.BRAKE_IDLE_INTERVAL_S


@pytest.mark.parametrize("failing", ["/v1/chat/completions", "/v1/cache/reset"])
def test_a_failed_release_is_retried_and_posts_no_external_notice(
    child, http, sensors, caplog, failing
):
    # Warning, the user away: every idle sample is eligible for a release.
    sensors.level = 2
    child["last_request_tokens"] = 10  # no useful shrink
    http.fail[failing] = [500]

    with caplog.at_level(logging.INFO):
        first = MLX_Engine.brake_point("idle")
    assert first == BrakeOutcome()
    assert child["brake"].excess_after_release == 0
    assert child["brake"].notice is None
    # The reset's or the barrier's own WARNING is the one record.
    assert not [r for r in _brake_lines(caplog) if "action=release" in r.getMessage()]

    _past_pacing(child)
    second = MLX_Engine.brake_point("idle")

    assert second.action == "release"
    assert child["brake"].notice is None


def test_a_shrink_the_child_did_not_apply_leaves_no_trace(child, http, caplog):
    http.fail["/v1/settings"] = 500  # the PATCH and its read-back

    with caplog.at_level(logging.INFO):
        outcome = MLX_Engine.brake_point("turn_end")

    assert outcome == BrakeOutcome()
    state = child["brake"]
    assert (state.cap, state.faulty, state.shrinks_this_episode) == (None, None, 0)
    assert state.shrunk_since_turn is False
    assert state.notice is None
    assert child["context_tokens"] == 20_000
    assert _brake_lines(caplog) == []
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "not applied" in warnings[0].getMessage()


def test_a_shrink_whose_applier_raises_leaves_no_trace(child, monkeypatch, caplog):
    def _boom(cls, handle):
        raise OSError("disk gone")

    monkeypatch.setattr(MLX_Engine, "_apply_memory_bound", classmethod(_boom))

    with caplog.at_level(logging.INFO):
        MLX_Engine.brake_point("turn_end")

    state = child["brake"]
    assert (state.cap, state.shrinks_this_episode, state.notice) == (None, 0, None)
    assert [r.levelno for r in caplog.records if "MLX_Engine" in r.getMessage()] == [
        logging.WARNING
    ]
    assert "OSError" in caplog.records[-1].getMessage()
    assert "disk gone" not in caplog.records[-1].getMessage()


def test_a_grow_the_child_did_not_apply_keeps_the_previous_cap(child, http, sensors, caplog):
    child["context_tokens"] = 15_000
    child["memory_bound_tokens"] = 15_000
    child["brake"] = BrakeState(cap=15_000, calm_since_grow=2, calm_count=5)
    sensors.level = 1
    http.fail["/v1/settings"] = 500

    with caplog.at_level(logging.INFO):
        assert MLX_Engine.brake_point("idle") == BrakeOutcome()

    assert child["brake"].cap == 15_000
    assert child["context_tokens"] == 15_000
    assert _brake_lines(caplog) == []


def test_a_release_at_turn_end_reads_the_footprint_once_without_waiting(
    child, http, sensors, monkeypatch
):
    sensors.level = 4
    child["last_request_tokens"] = 10
    monkeypatch.setattr(MLX_Engine, "RELEASE_SETTLE_S", 1.0)
    reads = []

    def _footprint(pid):
        if ("POST", "/v1/cache/reset") in http.paths():
            reads.append(pid)
        return sensors.footprint

    monkeypatch.setattr(process_footprint, "footprint", _footprint)
    started = time.monotonic()

    outcome = MLX_Engine.brake_point("turn_end")

    assert outcome.action == "release"
    assert time.monotonic() - started < 0.5
    assert len(reads) == 1


def test_a_reload_spawn_applies_the_carried_cap_without_a_measured_base(
    sensors, http, ceiling, monkeypatch
):
    MLX_Engine._brake_reload = {
        "alias": "erudi-7",
        "cap": 12_000,
        "faulty": None,
        "in_episode": True,
        "episode_level": 4,
        "shrinks_this_episode": 1,
        "unloaded_this_episode": True,
        "unloaded_at": time.time(),
    }
    monkeypatch.setattr(process_footprint, "footprint", lambda pid: None)
    handle = _spawned_handle()

    MLX_Engine._read_server_properties(handle)

    # The ceiling stays ignored without a measured base; the cap applies.
    assert handle["base_footprint_bytes"] is None
    assert http.live_max_kv_size == 12_000
    assert handle["context_tokens"] == 12_000
    assert handle["memory_bound_tokens"] == 12_000


# ===================== the idle monitor =====================


async def test_the_idle_tick_samples_only_a_loaded_child(monkeypatch):
    points = []
    monkeypatch.setattr(
        MLX_Engine, "brake_point", classmethod(lambda cls, point, **kw: points.append(point))
    )
    await MLX_Engine._pressure_tick()
    assert points == []
    MLX_Engine._model = _handle()
    await MLX_Engine._pressure_tick()
    assert points == ["idle"]
    MLX_Engine._model["proc"].alive = False
    await MLX_Engine._pressure_tick()
    assert points == ["idle"]


async def test_the_idle_tick_waits_for_a_running_generation(monkeypatch):
    points = []
    monkeypatch.setattr(
        MLX_Engine, "brake_point", classmethod(lambda cls, point, **kw: points.append(point))
    )
    MLX_Engine._model = _handle()
    holding = asyncio.Event()
    release = asyncio.Event()

    async def _generation():
        async with MLX_Engine._acquire_generation_lock():
            holding.set()
            await release.wait()

    generation = asyncio.create_task(_generation())
    await holding.wait()
    tick = asyncio.create_task(MLX_Engine._pressure_tick())
    await asyncio.sleep(0.05)
    assert points == []
    release.set()
    await asyncio.wait_for(asyncio.gather(generation, tick), timeout=5)
    assert points == ["idle"]


async def test_the_idle_sample_is_a_shielded_reset_the_next_holder_drains(monkeypatch):
    entered = threading.Event()
    finish = threading.Event()
    done_at = {}

    def _slow_brake(cls, point, **kw):
        entered.set()
        finish.wait(10)
        done_at["t"] = datetime.now()
        return BrakeOutcome()

    monkeypatch.setattr(MLX_Engine, "brake_point", classmethod(_slow_brake))
    MLX_Engine._model = _handle()
    tick = asyncio.create_task(MLX_Engine._pressure_tick())
    await asyncio.to_thread(entered.wait, 5)
    assert BaseEngine._pending_reset is not None
    # The tick is torn down (a shutdown): the thread still runs, and the next
    # lock holder waits for it.
    tick.cancel()
    proceeded = {}

    async def _next():
        async with MLX_Engine.generation_guard():
            proceeded["t"] = datetime.now()

    nxt = asyncio.create_task(_next())
    await asyncio.sleep(0.1)
    assert "t" not in proceeded
    finish.set()
    await asyncio.wait_for(nxt, timeout=5)
    with pytest.raises(asyncio.CancelledError):
        await tick
    assert proceeded["t"] >= done_at["t"]


async def test_the_monitor_logs_a_failing_tick_and_keeps_running(monkeypatch, caplog):
    ticks = []
    resumed = asyncio.Event()

    async def _tick():
        ticks.append(1)
        if len(ticks) == 1:
            raise RuntimeError("pressure tick exploded")
        resumed.set()

    monkeypatch.setattr(MLX_Engine, "_pressure_tick", classmethod(lambda cls: _tick()))
    monkeypatch.setattr(MLX_Engine, "_pressure_interval_s", 0.01)
    with caplog.at_level(logging.INFO, logger="erudi"):
        MLX_Engine.start_pressure_task()
        try:
            await asyncio.wait_for(resumed.wait(), timeout=5)
        finally:
            await MLX_Engine.stop_pressure_task()
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert "pressure tick exploded" in errors[0].getMessage()
    assert errors[0].exc_info
    assert MLX_Engine._pressure_task is None


async def test_stopping_the_monitor_awaits_the_sample_in_flight(monkeypatch):
    entered = threading.Event()
    finish = threading.Event()

    def _slow_brake(cls, point, **kw):
        entered.set()
        finish.wait(10)
        return BrakeOutcome()

    monkeypatch.setattr(MLX_Engine, "brake_point", classmethod(_slow_brake))
    monkeypatch.setattr(MLX_Engine, "_pressure_interval_s", 0.01)
    MLX_Engine._model = _handle()
    MLX_Engine.start_pressure_task()
    await asyncio.to_thread(entered.wait, 5)
    stopping = asyncio.create_task(MLX_Engine.stop_pressure_task())
    await asyncio.sleep(0.05)
    assert not stopping.done()
    finish.set()
    await asyncio.wait_for(stopping, timeout=5)
    assert BaseEngine._pending_reset is None


async def test_the_base_engine_pressure_task_is_a_no_op():
    CUDA_Engine.start_pressure_task()
    await CUDA_Engine.stop_pressure_task()
    assert getattr(CUDA_Engine, "_pressure_task", None) is None


# ===================== the lifespan =====================


async def test_shutdown_stops_the_monitor_and_drains_before_cleanup(monkeypatch):
    from fastapi import FastAPI

    from src.core import api, config

    order = []
    pending_done = asyncio.Event()

    class _Engine:
        @staticmethod
        def start_cleanup_task():
            order.append("start_cleanup")

        @staticmethod
        def start_pressure_task():
            order.append("start_pressure")

        @staticmethod
        def stop_cleanup_task():
            order.append("stop_cleanup")

        @staticmethod
        async def stop_pressure_task():
            order.append("stop_pressure")

            async def _orphan():
                await asyncio.sleep(0.05)
                order.append("pending_reset_done")
                pending_done.set()

            BaseEngine._pending_reset = asyncio.create_task(_orphan())

        @staticmethod
        def cleanup():
            order.append("cleanup")

    class _Checkpointer:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *exc):
            return False

    handle = SimpleNamespace(sqlalchemy_url="x", psycopg_url="y")
    monkeypatch.setattr(api, "start_postgres", lambda *_: handle)
    monkeypatch.setattr(api, "stop_postgres", lambda *_: None)
    monkeypatch.setattr(api, "init_database", lambda *_: None)
    monkeypatch.setattr(api, "run_migrations", lambda *_: None)
    monkeypatch.setattr(api, "init_kb_store", lambda *_: None)
    monkeypatch.setattr(api, "close_kb_store", lambda *_a, **_k: None)
    monkeypatch.setattr(api, "open_checkpointer", lambda *_: _Checkpointer())

    async def _populate():
        return {}

    monkeypatch.setattr(api, "startup_populate_database", _populate)
    monkeypatch.setattr(api, "BaseEngine", SimpleNamespace(get_engine=lambda: _Engine))
    monkeypatch.setattr(config, "LLM_Engine", None, raising=False)

    async with api.lifespan(FastAPI()):
        assert order == ["start_cleanup", "start_pressure"]

    assert order[2:] == ["stop_cleanup", "stop_pressure", "pending_reset_done", "cleanup"]
    assert BaseEngine._pending_reset is None


def test_the_drain_is_public_for_the_lifespan():
    assert callable(base_engine_module.drain_pending_reset)
