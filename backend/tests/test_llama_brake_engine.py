"""The llama.cpp memory brake, executed by the CPU engine (child, HTTP and
sensors faked).

``CPU_Engine`` sizes the window it passes as ``-c`` from the GGUF and the
machine's RAM (``gguf_memory.spawn_window``), folding in the cap the brake
remembers for that model. ``brake_point`` reads the sensors
(``memory_pressure``), applies ``memory_brake.llama_step`` and executes it: a
respawn restarts the same model in place with the reduced window (under the
generation lock, its probe bounded by ``LLAMA_RESPAWN_PROBE_TIMEOUT_S``); an
unload reaps the child and leaves the episode for the next spawn. The CUDA
engine inherits the plumbing and stays inert. Nothing here starts a real
llama-server.
"""

from __future__ import annotations

import ast
import asyncio
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents.runner import _engine_overrides
from src.engines import base_llama_cpp_engine as base_mod
from src.engines import gguf_memory
from src.engines import memory_brake as mb
from src.engines import memory_pressure
from src.engines.base_engine import BaseEngine
from src.engines.base_llama_cpp_engine import LLAMA_RESPAWN_PROBE_TIMEOUT_S
from src.engines.cpu_engine import CPU_Engine
from src.engines.cuda_engine import CUDA_Engine
from src.engines.gguf_memory import shape_from_fields, window_for
from src.engines.memory_brake import BrakeOutcome, BrakeState
from src.engines.memory_pressure import PressureReading

pytestmark = pytest.mark.unit

GIB = 1024**3
MIB = 1024**2
KEY = "llama-key-never-logged-0123456789"
WEIGHTS = 601 * MIB
QWEN3 = shape_from_fields(
    {
        "architecture": "qwen3",
        "block_count": 28,
        "context_length": 40_960,
        "embedding_length": 1024,
        "head_count": 16,
        "head_count_kv": 8,
        "key_length": 128,
        "value_length": 128,
    }
)
KV = 114_688
W = 40_960


class _Popen:
    def __init__(self, pid):
        self.pid = pid
        self.alive = True
        self.returncode = None

    def poll(self):
        return None if self.alive else 0

    def send_signal(self, sig):
        self.alive = False
        self.returncode = 0

    terminate = send_signal

    def kill(self):
        self.alive = False

    def wait(self, timeout=None):
        return 0


class _Resp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class _World:
    """The fake machine: sensors, the spawned children, their /props."""

    def __init__(self, monkeypatch):
        self.level = 1
        self.damage = 0
        self.total = 16 * GIB
        self.anonymous = None
        self.spawns = []
        self.probe_timeouts = []
        self.probe_fail = []
        self.props_fail = False
        self.caps_seen = []
        self.reads = 0
        self._ports = iter(range(27201, 27299))
        self._pids = iter(range(5000, 5100))
        world = self

        def _read_pressure():
            world.reads += 1
            return PressureReading(world.level, world.damage)

        monkeypatch.setattr(memory_pressure, "read_pressure", _read_pressure)
        monkeypatch.setattr(
            memory_pressure,
            "child_anonymous_bytes",
            lambda pid: world.anonymous if world.anonymous is not None else world.expected(),
        )
        monkeypatch.setattr(memory_pressure, "effective_total_ram_bytes", lambda: world.total)

        def _spawn_window(path, *, total_ram_bytes, brake_cap=None, mmproj_path=None):
            world.caps_seen.append(brake_cap)
            return window_for(
                QWEN3,
                weights_bytes=WEIGHTS,
                total_ram_bytes=total_ram_bytes,
                brake_cap=brake_cap,
            )

        monkeypatch.setattr(gguf_memory, "spawn_window", _spawn_window)
        monkeypatch.setattr(CPU_Engine, "_find_mmproj", classmethod(lambda cls, p: None))

        def _spawn_child(cls, *, model_path, alias, port, **ctx):
            world.spawns.append({"port": port, "alias": alias, **ctx})
            handle = {
                "pid": next(world._pids),
                "proc": _Popen(4000 + len(world.spawns)),
                "port": port,
                "base_url": f"http://127.0.0.1:{port}",
                "alias": alias,
                "model_path": str(model_path),
                "api_key": KEY,
            }
            for key in base_mod.PRESERVED_SPAWN_KEYS:
                if key in ctx:
                    handle[key] = ctx[key]
            return handle

        monkeypatch.setattr(CPU_Engine, "_spawn_child", classmethod(_spawn_child))

        def _probe_ready(cls, base_url, proc=None, model_field="m", api_key=None, **kw):
            world.probe_timeouts.append(kw.get("probe_timeout_s"))
            if world.probe_fail:
                outcome = world.probe_fail.pop(0)
                if outcome is not None:
                    raise outcome

        monkeypatch.setattr(CPU_Engine, "_probe_ready", classmethod(_probe_ready))
        monkeypatch.setattr(
            CPU_Engine, "_pick_free_port", classmethod(lambda cls: next(world._ports))
        )
        monkeypatch.setattr(
            CPU_Engine, "_wait_port_closed", classmethod(lambda cls, port, timeout_s=3.0: None)
        )

        def _get(url, timeout=None, headers=None):
            if world.props_fail:
                raise base_mod.requests.ConnectionError("down")
            port = int(url.split(":")[-1].split("/")[0])
            spawn = [s for s in world.spawns if s["port"] == port][-1]
            return _Resp(
                200,
                {
                    "default_generation_settings": {"n_ctx": spawn.get("ctx_size") or W},
                    "chat_template_caps": {},
                    "chat_template": "t",
                },
            )

        monkeypatch.setattr(base_mod.requests, "get", _get)

    def expected(self):
        """What the child being checked really holds: its KV, a repacked
        weights copy, and some heap."""
        window = self.spawns[-1].get("ctx_size") or W
        return window * KV + WEIGHTS + 100 * MIB


def _reset():
    for engine in (CPU_Engine, CUDA_Engine):
        engine._model = None
        engine._tokenizer = None
        engine._model_id = None
        engine._last_used = None
        engine._brake_reload = None
        engine._brake_sensor_warned = False
    CPU_Engine._window_memory = {}
    BaseEngine._pending_reset = None


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("ERUDI_CTX", raising=False)
    _reset()
    yield
    for engine in (CPU_Engine, CUDA_Engine):
        handle = engine._model
        if isinstance(handle, dict):
            handle["proc"].alive = False
    _reset()


@pytest.fixture
def world(monkeypatch):
    return _World(monkeypatch)


def _load(world, *, llm_id=7):
    alias = f"erudi-{llm_id}"
    handle = CPU_Engine._start_server(model_path=Path("/models/q.gguf"), alias=alias, port=27200)
    CPU_Engine._model = handle
    CPU_Engine._model_id = llm_id
    CPU_Engine._tokenizer = {"type": "remote"}
    CPU_Engine._last_used = datetime.now() - timedelta(seconds=30)
    return handle


def _brake_records(caplog):
    return [r for r in caplog.records if "Memory brake" in r.getMessage()]


def _assert_no_key(caplog):
    for record in caplog.records:
        assert KEY not in record.getMessage()


# ===================== the spawn window =====================


def test_the_spawn_passes_the_computed_window_and_keeps_its_facts(world):
    handle = _load(world)
    spawn = world.spawns[0]
    assert spawn["ctx_size"] == W
    assert handle["context_tokens"] == W
    assert handle["trained_context_tokens"] == W
    assert handle["static_window_tokens"] == W
    assert handle["kv_token_bytes"] == KV
    assert handle["weights_bytes"] == WEIGHTS
    assert handle["spawn_memory_bound"] is False
    assert isinstance(handle["brake"], BrakeState)
    assert handle["damage_last"] == 0
    assert CPU_Engine.memory_bound_tokens() is None


def test_a_memory_bound_window_is_reported_with_its_footprint(world):
    world.total = 4 * GIB
    handle = _load(world)
    assert handle["spawn_memory_bound"] is True
    assert CPU_Engine.memory_bound_tokens() == handle["context_tokens"] < W
    assert CPU_Engine.memory_bound_footprint_bytes() == (
        handle["static_anonymous_bytes"] + handle["weights_bytes"]
    )


def test_erudi_ctx_is_passed_verbatim(world, monkeypatch):
    monkeypatch.setenv("ERUDI_CTX", "8192")
    _load(world)
    assert world.spawns[0]["ctx_size"] == 8192
    assert world.caps_seen == []


def test_an_unreadable_shape_spawns_without_c(world, monkeypatch):
    monkeypatch.setattr(gguf_memory, "spawn_window", lambda *a, **k: None)
    handle = _load(world)
    assert world.spawns[0]["ctx_size"] is None
    # llama-server's own fit: the window it reports.
    assert handle["context_tokens"] == W
    assert CPU_Engine.memory_bound_tokens() is None


def test_a_failed_props_read_keeps_the_window_we_passed(world, caplog):
    world.props_fail = True
    world.total = 4 * GIB
    with caplog.at_level(logging.WARNING, logger="erudi"):
        handle = _load(world)
    assert handle["context_tokens"] == world.spawns[0]["ctx_size"]
    assert CPU_Engine.memory_bound_tokens() == handle["context_tokens"]


def test_a_props_window_other_than_the_c_passed_is_a_warning(world, monkeypatch, caplog):
    real_get = base_mod.requests.get

    def _get(url, timeout=None, headers=None):
        resp = real_get(url, timeout=timeout, headers=headers)
        resp._payload["default_generation_settings"]["n_ctx"] = 1234
        return resp

    monkeypatch.setattr(base_mod.requests, "get", _get)
    with caplog.at_level(logging.WARNING, logger="erudi"):
        handle = _load(world)
    assert handle["context_tokens"] == 1234
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("1234" in r.getMessage() and str(W) in r.getMessage() for r in warnings)


def test_the_real_spawn_child_preserves_the_window_facts(tmp_path, monkeypatch):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"\0" * 16)
    server = tmp_path / "llama-server"
    server.write_bytes(b"\0")
    monkeypatch.setattr(CPU_Engine, "_find_llama_server", classmethod(lambda cls, d=None: server))
    monkeypatch.setattr(
        base_mod.subprocess, "Popen", lambda argv, **kw: SimpleNamespace(pid=1, stdout=None)
    )
    ctx = {
        "ctx_size": 30_720,
        "threads": 4,
        "gpu_layers": 0,
        "trained_context_tokens": W,
        "static_window_tokens": W,
        "kv_token_bytes": KV,
        "static_anonymous_bytes": 123,
        "weights_bytes": WEIGHTS,
        "spawn_memory_bound": True,
        "mmproj_loaded": False,
    }
    handle = CPU_Engine._spawn_child(model_path=model, alias="erudi-1", port=27201, **ctx)
    for key, value in ctx.items():
        assert handle[key] == value


# ===================== the post-load check =====================


def test_the_post_load_check_warns_on_an_inflated_child(world, caplog):
    world.anonymous = 20 * GIB
    with caplog.at_level(logging.WARNING, logger="erudi"):
        _load(world)
    warnings = [r for r in caplog.records if "window budget predicted" in r.getMessage()]
    assert len(warnings) == 1
    assert str(20 * GIB) in warnings[0].getMessage()


def test_the_post_load_check_is_quiet_within_margin_and_repack(world, caplog):
    with caplog.at_level(logging.WARNING, logger="erudi"):
        _load(world)
    assert not [r for r in caplog.records if "window budget predicted" in r.getMessage()]


def test_the_post_load_check_is_skipped_with_a_vision_projector(world, monkeypatch, caplog):
    monkeypatch.setattr(CPU_Engine, "_find_mmproj", classmethod(lambda cls, p: Path("/mm.gguf")))
    world.anonymous = 20 * GIB
    with caplog.at_level(logging.DEBUG, logger="erudi"):
        _load(world)
    assert not [r for r in caplog.records if "window budget predicted" in r.getMessage()]
    assert any("vision projector" in r.getMessage() for r in caplog.records)


# ===================== respawn in place =====================


def test_a_critical_turn_end_respawns_in_place_with_the_reduced_window(world, caplog):
    handle = _load(world)
    last_used = CPU_Engine._last_used
    old_proc = handle["proc"]
    world.level = 4
    with caplog.at_level(logging.INFO, logger="erudi"):
        outcome = CPU_Engine.brake_point("turn_end")
    assert outcome == BrakeOutcome(acted=True, action="shrink+respawn")
    new = CPU_Engine._model
    assert new is not handle
    assert old_proc.alive is False
    assert world.spawns[-1]["ctx_size"] == int(W * mb.SHRINK_FACTOR)
    assert new["context_tokens"] == int(W * mb.SHRINK_FACTOR)
    assert world.caps_seen[-1] == int(W * mb.SHRINK_FACTOR)
    assert CPU_Engine._model_id == 7
    assert CPU_Engine._last_used == last_used
    assert new["brake_respawned"] is True
    # The live state is carried: the notice, the episode.
    assert new["brake"].notice == "acted"
    assert new["brake"].in_episode
    assert new["brake"].reloaded_at is None
    # The probe of the respawn is bounded by a parameter, the class value untouched.
    assert world.probe_timeouts == [None, LLAMA_RESPAWN_PROBE_TIMEOUT_S]
    assert CPU_Engine._probe_timeout_s == 120.0
    lines = [r.getMessage() for r in _brake_records(caplog)]
    assert any(f"window={W}->{int(W * mb.SHRINK_FACTOR)}" in line for line in lines)
    _assert_no_key(caplog)


def test_a_failed_respawn_degrades_into_an_unload(world, caplog):
    _load(world)
    world.level = 4
    world.probe_fail = [RuntimeError("probe timed out " + KEY)]
    with caplog.at_level(logging.WARNING, logger="erudi"):
        outcome = CPU_Engine.brake_point("turn_end")
    assert outcome == BrakeOutcome(acted=True, action="respawn-failed")
    assert CPU_Engine._model is None
    record = CPU_Engine._brake_reload
    assert record["alias"] == "erudi-7"
    assert record["cap"] == int(W * mb.SHRINK_FACTOR)
    assert CPU_Engine.brake_reload_pending(7)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "RuntimeError" in warnings[0].getMessage()
    _assert_no_key(caplog)
    # The next message reloads with the reduced window, seeded with the episode.
    world.probe_fail = []
    handle = _load(world)
    assert handle["context_tokens"] == int(W * mb.SHRINK_FACTOR)
    assert handle["brake"].in_episode
    assert CPU_Engine._brake_reload is None


def test_two_idle_criticals_unload_and_carry_the_episode(world, caplog):
    _load(world)
    world.level = 4
    CPU_Engine._last_used = datetime.now() - timedelta(seconds=600)
    with caplog.at_level(logging.INFO, logger="erudi"):
        first = CPU_Engine.brake_point("idle")
        assert first.action == "shrink+respawn"
        CPU_Engine._model["brake"].last_sample_at -= mb.BRAKE_IDLE_INTERVAL_S
        second = CPU_Engine.brake_point("idle")
    assert second == BrakeOutcome(acted=True, action="unload")
    assert CPU_Engine._model is None
    record = CPU_Engine._brake_reload
    assert record["unloaded_this_episode"] is True
    assert CPU_Engine.brake_reload_pending(7)
    unloads = [r for r in _brake_records(caplog) if "action=unload" in r.getMessage()]
    assert len(unloads) == 1 and unloads[0].levelno == logging.WARNING
    # The window memory also keeps the cap for the reload.
    assert CPU_Engine._window_memory["erudi-7"]["cap"] == record["cap"]


def test_the_unload_share_is_the_child_anonymous_memory(world):
    _load(world)
    # A child holding little anonymous memory is not worth an unload.
    world.anonymous = 100 * MIB
    world.level = 4
    CPU_Engine._last_used = datetime.now() - timedelta(seconds=600)
    CPU_Engine.brake_point("idle")
    CPU_Engine._model["brake"].last_sample_at -= mb.BRAKE_IDLE_INTERVAL_S
    assert CPU_Engine.brake_point("idle") == BrakeOutcome()
    assert CPU_Engine._model is not None


# ===================== the window memory and the seeds =====================


def test_a_shrink_is_written_through_to_the_window_memory(world):
    _load(world)
    world.level = 2
    world.damage = 0
    CPU_Engine.brake_point("turn_start", llm_id=7)
    world.damage = 64 * MIB
    CPU_Engine._model["brake"].turn_happened = True
    outcome = CPU_Engine.brake_point("turn_end")
    assert outcome == BrakeOutcome(acted=True, action="shrink")
    entry = CPU_Engine._window_memory["erudi-7"]
    assert entry["cap"] == int(W * mb.SHRINK_FACTOR)
    assert entry["faulty"] == W
    assert time.time() - entry["at"] < 5


def test_the_window_memory_survives_a_reap_and_sizes_the_reload(world):
    _load(world)
    CPU_Engine._window_memory["erudi-7"] = {"cap": 20_480, "faulty": 30_720, "at": time.time()}
    CPU_Engine._brake_reload = {"alias": "erudi-9", "unloaded_at": time.time()}
    CPU_Engine.cleanup()
    assert CPU_Engine._brake_reload is None
    assert "erudi-7" in CPU_Engine._window_memory
    handle = _load(world)
    assert handle["context_tokens"] == 20_480
    assert handle["brake"].cap == 20_480
    assert handle["brake"].faulty == 30_720
    assert handle["brake"].reloaded_at is None


def test_the_unload_record_seeds_before_the_window_memory(world):
    CPU_Engine._window_memory["erudi-7"] = {"cap": 30_720, "faulty": None, "at": time.time()}
    CPU_Engine._brake_reload = {
        "alias": "erudi-7",
        "cap": 20_480,
        "faulty": 30_720,
        "in_episode": True,
        "episode_level": 4,
        "shrinks_this_episode": 2,
        "unloaded_this_episode": True,
        "unloaded_at": time.time() - 30,
    }
    handle = _load(world)
    state = handle["brake"]
    assert state.shrinks_this_episode == 2
    assert state.unloaded_this_episode
    assert time.time() - state.reloaded_at < 5
    assert CPU_Engine._brake_reload is None


def test_an_expired_window_memory_seeds_nothing_and_hands_no_cap(world):
    CPU_Engine._window_memory["erudi-7"] = {
        "cap": 20_480,
        "faulty": 30_720,
        "at": time.time() - mb.BRAKE_RELOAD_TTL_S - 1,
    }
    handle = _load(world)
    assert world.caps_seen == [None]
    assert handle["brake"] == BrakeState()
    assert "erudi-7" not in CPU_Engine._window_memory


def test_an_entry_back_to_nothing_is_deleted(world):
    _load(world)
    CPU_Engine._window_memory["erudi-7"] = {"cap": 20_480, "faulty": None, "at": time.time()}
    CPU_Engine._remember_window("erudi-7", BrakeState())
    assert "erudi-7" not in CPU_Engine._window_memory


# ===================== samples =====================


def test_damage_last_is_refreshed_at_every_sample_but_a_skipped_one(world):
    handle = _load(world)
    world.damage = 10 * MIB
    CPU_Engine.brake_point("turn_start", llm_id=7)
    assert CPU_Engine._model["damage_last"] == 10 * MIB
    world.damage = 30 * MIB
    handle["brake"].last_sample_at -= mb.BRAKE_IDLE_INTERVAL_S
    CPU_Engine._last_used = datetime.now() - timedelta(seconds=600)
    CPU_Engine.brake_point("idle")
    assert handle["damage_last"] == 30 * MIB
    # Paced out: the paging of its interval stays for the next sample.
    world.damage = 50 * MIB
    CPU_Engine.brake_point("idle")
    assert handle["damage_last"] == 30 * MIB


def test_turn_start_samples_only_the_child_this_turn_will_use(world):
    handle = _load(world)
    reads = world.reads
    CPU_Engine.brake_point("turn_start", llm_id=8)
    assert world.reads == reads
    handle["proc"].alive = False
    CPU_Engine.brake_point("turn_start", llm_id=7)
    assert world.reads == reads
    handle["proc"].alive = True
    CPU_Engine.brake_point("turn_start", llm_id=7)
    assert world.reads == reads + 1


def test_a_sample_never_raises(world, monkeypatch, caplog):
    _load(world)

    def _boom():
        raise RuntimeError("sensor exploded " + KEY)

    monkeypatch.setattr(memory_pressure, "read_pressure", _boom)
    with caplog.at_level(logging.WARNING, logger="erudi"):
        assert CPU_Engine.brake_point("idle") == BrakeOutcome()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "RuntimeError" in warnings[0].getMessage()
    _assert_no_key(caplog)


def test_an_unreadable_sensor_is_one_warning_and_the_brake_off(world, monkeypatch, caplog):
    _load(world)
    monkeypatch.setattr(
        memory_pressure,
        "read_pressure",
        lambda: PressureReading(None, None, unavailable="no PSI, no /proc/vmstat"),
    )
    with caplog.at_level(logging.WARNING, logger="erudi"):
        CPU_Engine.brake_point("turn_start", llm_id=7)
        CPU_Engine.brake_point("turn_end")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "no PSI" in warnings[0].getMessage()
    assert "brake is off" in warnings[0].getMessage()


def test_a_pinned_window_is_never_resized(world, monkeypatch):
    monkeypatch.setenv("ERUDI_CTX", "8192")
    _load(world)
    world.level = 4
    outcome = CPU_Engine.brake_point("turn_end")
    assert outcome == BrakeOutcome()
    assert len(world.spawns) == 1
    assert CPU_Engine._model["brake"].notice != "acted"


# ===================== the turn hooks =====================


def test_the_turn_window_marks_the_turn_and_consumes_the_respawn_flag(world):
    handle = _load(world)
    world.level = 4
    CPU_Engine.brake_point("turn_end")
    new = CPU_Engine._model
    assert new is not handle
    assert CPU_Engine.brake_reload_pending(7)
    assert not CPU_Engine.brake_reload_pending(8)
    assert CPU_Engine.begin_memory_window() is None
    assert new["brake"].turn_happened
    assert "brake_respawned" not in new
    assert not CPU_Engine.brake_reload_pending(7)


def test_consume_brake_notice_reads_the_loaded_child(world):
    _load(world)
    CPU_Engine._model["brake"].notice = "external"
    assert CPU_Engine.consume_brake_notice() == "external"
    assert CPU_Engine.consume_brake_notice() is None


# ===================== CUDA stays inert =====================


@pytest.mark.parametrize(
    "hook",
    [
        "brake_point",
        "begin_memory_window",
        "claim_prefix",
        "on_history_rewritten",
        "note_stream_abandoned",
        "memory_bound_tokens",
    ],
)
def test_cuda_overrides_none_of_the_runner_gated_hooks(hook):
    assert not _engine_overrides(CUDA_Engine, hook)


@pytest.mark.parametrize("hook", ["brake_point", "begin_memory_window", "memory_bound_tokens"])
def test_cpu_binds_the_brake_hooks(hook):
    assert _engine_overrides(CPU_Engine, hook)


def test_cuda_answers_the_base_values():
    assert CUDA_Engine.brake_reload_pending(7) is False
    assert CUDA_Engine.consume_brake_notice() is None
    assert CUDA_Engine.brake_point("idle") == BrakeOutcome()
    assert CUDA_Engine.memory_bound_tokens() is None


async def test_the_monitor_starts_on_cpu_and_not_on_cuda(monkeypatch):
    monkeypatch.setattr(CPU_Engine, "_pressure_interval_s", 3600.0)
    CUDA_Engine.start_pressure_task()
    assert CUDA_Engine._pressure_task is None
    CPU_Engine.start_pressure_task()
    try:
        assert CPU_Engine._pressure_task is not None
    finally:
        await CPU_Engine.stop_pressure_task()
    assert CPU_Engine._pressure_task is None


def test_the_spawn_falls_back_to_the_fit_without_gguf(tmp_path, monkeypatch, caplog):
    # The macOS build ships no gguf package: the spawn path must not raise.
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF" + b"\0" * 60)
    monkeypatch.setitem(sys.modules, "gguf", None)
    with caplog.at_level(logging.WARNING, logger="erudi"):
        ctx = CPU_Engine._prepare_spawn_context(model, "erudi-7")
    assert ctx["ctx_size"] is None
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_no_engine_module_imports_gguf_at_module_level():
    engines = Path(__file__).resolve().parents[1] / "src" / "engines"
    for path in engines.glob("*.py"):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(name.split(".")[0] == "gguf" for name in names), path.name


async def test_the_idle_tick_respawns_under_the_lock(world):
    _load(world)
    world.level = 4
    CPU_Engine._last_used = datetime.now() - timedelta(seconds=600)
    await CPU_Engine._pressure_tick()
    assert len(world.spawns) == 2
    assert CPU_Engine._model["context_tokens"] == int(W * mb.SHRINK_FACTOR)
    assert BaseEngine._pending_reset is None
    assert not CPU_Engine._generation_lock_for_running_loop().locked()
    await asyncio.sleep(0)
