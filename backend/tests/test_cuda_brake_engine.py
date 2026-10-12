"""The CUDA window and its memory brake, executed by ``CUDA_Engine`` (child,
HTTP, NVML, PDH and RAM sensors all faked).

``CUDA_Engine`` sizes ``-c`` for a full offload against the VRAM free at load
(``cuda_memory``) and hands the layers to llama.cpp's fit (``-ngl -1``). After
load it classifies the placement fit chose -- ``offloaded N/M`` when the line
reaches the drainer, the per-pid device bytes otherwise -- and a partial child
runs brake-off. ``brake_point`` fuses the WDDM spill gauge with the RAM sensors
into ``memory_brake.llama_step`` (unchanged) and executes it: a respawn reuses
the live handle's window facts folded with the cap and re-reads no VRAM; on
Windows it pins every layer (``-ngl -2``), on Linux it first waits for the
killed child's VRAM to come back. The real NVML and the host's GPU are never
touched: ``CUDA_Engine._read_vram_info`` and ``memory_pressure.read_vram`` are
faked in every test, and the fake child implements both ``terminate()`` and
``send_signal()`` (the stop path branches on the OS).
"""

from __future__ import annotations

import logging
import platform
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core.exceptions import EngineException
from src.engines import base_llama_cpp_engine as base_mod
from src.engines import cuda_engine as cuda_mod
from src.engines import cuda_memory, gguf_memory, memory_pressure
from src.engines import memory_brake as mb
from src.engines.base_engine import BaseEngine
from src.engines.cuda_compatibility import classify_cuda_failure
from src.engines.cuda_engine import CUDA_Engine
from src.engines.gguf_memory import TensorBytes, shape_from_fields
from src.engines.memory_brake import BrakeOutcome, BrakeState, llama_padded
from src.engines.memory_pressure import PressureReading, VramReading
from src.engines.working_window import MIN_KV_BOUND_TOKENS

pytestmark = pytest.mark.unit

GIB = 1024**3
MIB = 1024**2
KEY = "cuda-key-never-logged-0123456789"
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
WEIGHTS = int(2.5 * GIB)
TOK_EMBD = 300 * MIB
UUID = "GPU-8932f937-d72c-4106-c12f-20bd9faed9f6"
FIT_WARN = (
    "common_fit_params: failed to fit params to free device memory: "
    "n_gpu_layers already set by user\n"
)


class _Popen:
    """A fake child. BOTH stop paths are real methods: Windows stops the child
    with terminate(), the others with send_signal(SIGINT)."""

    def __init__(self, pid):
        self.pid = pid
        self.alive = True
        self.returncode = None
        self.stopped_by = None

    def poll(self):
        return None if self.alive else 0

    def send_signal(self, sig):
        self.alive = False
        self.returncode = 0
        self.stopped_by = "send_signal"

    def terminate(self):
        self.alive = False
        self.returncode = 0
        self.stopped_by = "terminate"

    def kill(self):
        self.alive = False

    def wait(self, timeout=None):
        return 0


class _Drainer:
    def __init__(self, world):
        self.world = world

    def tail(self, max_chars=2000):
        return self.world.tail[-max_chars:]


class _Resp:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload

    def json(self):
        return self._payload


class _World:
    """The fake machine: RAM sensors, the GPU (NVML and PDH), the children."""

    def __init__(self, monkeypatch, os_name):
        self.os_name = os_name
        self.level = mb.LEVEL_CALM
        self.damage = 0
        self.total_ram = 32 * GIB
        self.host_anonymous = 2 * GIB
        self.spill = 0
        self.vram_unavailable = None
        # Device bytes the child really holds over the floor prediction.
        self.extras = 700 * MIB
        self.device_override = None
        self.vram_total = 16 * GIB
        self.free = int(13.8 * GIB)
        self.processes_readable = True
        self.linger_reads = 0
        self.tail = ""
        self.inputs = (QWEN3, WEIGHTS)
        self.tensors = TensorBytes(tok_embd_bytes=TOK_EMBD, exps_bytes=0)
        self.spawns = []
        self.children = {}
        self.probe_fail = []
        self.reads = 0
        self.vram_reads = []
        self.nvml_reads = 0
        self.sized = 0
        self.now = 0.0
        self._ports = iter(range(27201, 27299))
        self._pids = iter(range(5000, 5100))
        world = self

        monkeypatch.setattr(platform, "system", lambda: os_name)

        def _read_pressure():
            world.reads += 1
            return PressureReading(world.level, world.damage)

        monkeypatch.setattr(memory_pressure, "read_pressure", _read_pressure)
        monkeypatch.setattr(
            memory_pressure, "child_anonymous_bytes", lambda pid: world.host_anonymous
        )
        monkeypatch.setattr(memory_pressure, "effective_total_ram_bytes", lambda: world.total_ram)

        def _read_vram(pid):
            # WDDM: the child's allocation is split between dedicated VRAM
            # and the shared memory it spilled to.
            world.vram_reads.append(pid)
            if world.vram_unavailable:
                return VramReading(None, None, unavailable=world.vram_unavailable)
            total = world.device_bytes(pid)
            return VramReading(world.spill, max(0, total - world.spill))

        monkeypatch.setattr(memory_pressure, "read_vram", _read_vram)

        def _read_vram_info(cls):
            world.nvml_reads += 1
            for pid in list(world.children):
                child = world.children[pid]
                if not child["proc"].alive:
                    if child["linger"] <= 0:
                        del world.children[pid]
                    else:
                        child["linger"] -= 1
            processes = (
                tuple((pid, world.device_bytes(pid)) for pid in world.children)
                if world.processes_readable
                else None
            )
            return cuda_memory.DeviceMemory(
                index=0,
                uuid=UUID,
                total_bytes=world.vram_total,
                free_bytes=world.free,
                processes=processes,
            )

        monkeypatch.setattr(CUDA_Engine, "_read_vram_info", classmethod(_read_vram_info))
        real_sizing = CUDA_Engine._device_for_sizing.__func__

        def _device_for_sizing(cls):
            world.sized += 1
            return real_sizing(cls)

        monkeypatch.setattr(CUDA_Engine, "_device_for_sizing", classmethod(_device_for_sizing))
        monkeypatch.setattr(
            gguf_memory, "read_sizing_inputs", lambda path, mmproj_path=None: world.inputs
        )
        monkeypatch.setattr(gguf_memory, "read_shape", lambda path: world.inputs[0])
        monkeypatch.setattr(gguf_memory, "read_tensor_bytes", lambda path: world.tensors)
        monkeypatch.setattr(CUDA_Engine, "_find_mmproj", classmethod(lambda cls, p: None))

        def _spawn_child(cls, *, model_path, alias, port, **ctx):
            pid = next(world._pids)
            proc = _Popen(pid)
            setattr(proc, CUDA_Engine._DRAINER_ATTR, _Drainer(world))
            world.spawns.append({"port": port, "alias": alias, "pid": pid, **ctx})
            world.children[pid] = {"proc": proc, "ctx": ctx, "linger": world.linger_reads}
            handle = {
                "pid": pid,
                "proc": proc,
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

        monkeypatch.setattr(CUDA_Engine, "_spawn_child", classmethod(_spawn_child))

        def _probe_ready(cls, base_url, proc=None, model_field="m", api_key=None, **kw):
            if world.probe_fail:
                outcome = world.probe_fail.pop(0)
                if outcome is not None:
                    proc.alive = False
                    raise outcome

        monkeypatch.setattr(CUDA_Engine, "_probe_ready", classmethod(_probe_ready))
        monkeypatch.setattr(
            CUDA_Engine, "_pick_free_port", classmethod(lambda cls: next(world._ports))
        )
        monkeypatch.setattr(
            CUDA_Engine, "_wait_port_closed", classmethod(lambda cls, port, timeout_s=3.0: None)
        )

        def _get(url, timeout=None, headers=None):
            port = int(url.split(":")[-1].split("/")[0])
            spawn = [s for s in world.spawns if s["port"] == port][-1]
            return _Resp(
                {
                    "default_generation_settings": {
                        "n_ctx": llama_padded(spawn.get("ctx_size") or W)
                    },
                    "chat_template_caps": {},
                    "chat_template": "t",
                }
            )

        monkeypatch.setattr(base_mod.requests, "get", _get)
        monkeypatch.setattr(
            cuda_mod,
            "time",
            SimpleNamespace(time=time.time, monotonic=lambda: world.now, sleep=world._sleep),
        )

    def _sleep(self, seconds):
        self.now += seconds

    def device_bytes(self, pid):
        """What child ``pid`` really holds on the device under a full
        offload: its weights without tok_embd, its KV, and some extras."""
        if self.device_override is not None:
            return self.device_override
        child = self.children.get(pid)
        window = (child["ctx"].get("ctx_size") if child else None) or W
        return WEIGHTS - TOK_EMBD + window * KV + self.extras


def _reset():
    CUDA_Engine._model = None
    CUDA_Engine._tokenizer = None
    CUDA_Engine._model_id = None
    CUDA_Engine._last_used = None
    CUDA_Engine._brake_reload = None
    CUDA_Engine._brake_sensor_warned = False
    CUDA_Engine._window_memory = {}
    BaseEngine._pending_reset = None


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("ERUDI_CTX", raising=False)
    monkeypatch.delenv("ERUDI_VRAM_OVERRIDE_GB", raising=False)
    monkeypatch.delenv("ERUDI_FORCE_MEMORY_PRESSURE", raising=False)
    _reset()
    yield
    handle = CUDA_Engine._model
    if isinstance(handle, dict):
        handle["proc"].alive = False
    _reset()


@pytest.fixture
def windows(monkeypatch):
    return _World(monkeypatch, "Windows")


@pytest.fixture
def linux(monkeypatch):
    return _World(monkeypatch, "Linux")


def _load(world, *, llm_id=7):
    alias = f"erudi-{llm_id}"
    handle = CUDA_Engine._start_server(model_path=Path("/models/q.gguf"), alias=alias, port=27200)
    CUDA_Engine._model = handle
    CUDA_Engine._model_id = llm_id
    CUDA_Engine._tokenizer = {"type": "remote"}
    CUDA_Engine._last_used = datetime.now() - timedelta(seconds=30)
    return handle


def _records(caplog, text, level=None):
    return [
        r
        for r in caplog.records
        if text in r.getMessage() and (level is None or r.levelno == level)
    ]


def _assert_no_key(caplog):
    for record in caplog.records:
        assert KEY not in record.getMessage()


def _away(seconds):
    CUDA_Engine._last_used = datetime.now() - timedelta(seconds=seconds)
    CUDA_Engine._model["brake"].last_sample_at -= mb.BRAKE_IDLE_INTERVAL_S


# ===================== the load =====================


def test_a_load_is_sized_classified_full_and_seeded(windows, caplog):
    windows.spill = 40 * MIB
    with caplog.at_level(logging.INFO, logger="erudi"):
        handle = _load(windows)
    spawn = windows.spawns[0]
    assert spawn["ctx_size"] == W
    assert spawn["gpu_layers"] == -1
    assert handle["window_sized"] is True
    assert handle["context_tokens"] == W
    assert isinstance(handle["brake"], BrakeState)
    assert handle["damage_last"] == 0
    # The spill baseline: the seed reading, bounded by the constant.
    assert handle["spill_baseline"] == 40 * MIB
    assert CUDA_Engine.memory_bound_tokens() is None
    assert _records(caplog, "full offload", logging.INFO)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    _assert_no_key(caplog)


def test_a_seed_spill_above_the_baseline_warns_on_a_first_load(windows, caplog):
    windows.spill = 400 * MIB
    with caplog.at_level(logging.INFO, logger="erudi"):
        handle = _load(windows)
    assert handle["spill_baseline"] == cuda_memory.CUDA_SPILL_BASELINE_BYTES
    assert len(_records(caplog, "Shared GPU memory", logging.WARNING)) == 1


def test_a_memory_bound_window_reports_its_device_footprint(windows):
    windows.vram_total, windows.free = 8 * GIB, int(6.5 * GIB)
    handle = _load(windows)
    assert handle["spawn_memory_bound"] is True
    assert CUDA_Engine.memory_bound_tokens() == handle["context_tokens"] == 20_736
    assert CUDA_Engine.memory_bound_footprint_bytes() == (
        handle["static_device_bytes"] + handle["weights_bytes"]
    )


def test_the_host_anonymous_check_is_skipped_on_a_full_offload(windows, caplog):
    windows.host_anonymous = 40 * GIB
    with caplog.at_level(logging.DEBUG, logger="erudi"):
        _load(windows)
    assert not _records(caplog, "window budget predicted")
    assert _records(caplog, "host memory check skipped", logging.DEBUG)


def test_the_device_check_warns_on_an_inflated_child(linux, caplog):
    linux.extras = 6 * GIB
    with caplog.at_level(logging.WARNING, logger="erudi"):
        handle = _load(linux)
    assert handle["window_sized"] is True
    warnings = _records(caplog, "device memory", logging.WARNING)
    assert len(warnings) == 1


def test_the_fa_disabled_warn_in_the_tail_is_one_warning(windows, caplog):
    windows.tail = "llama_context: Flash Attention not supported, set to disabled\n"
    with caplog.at_level(logging.WARNING, logger="erudi"):
        _load(windows)
    assert len(_records(caplog, "Flash Attention", logging.WARNING)) == 1


def test_the_fit_warn_of_a_pinned_respawn_is_neither_an_oom_nor_an_fa_miss(windows, caplog):
    """m-D: fit writes this WARN on every Windows brake respawn (-ngl -2); a
    LIVE child's tail is never classified, and the line matches neither the
    CUDA_OUT_OF_MEMORY signatures nor the FA scan."""
    windows.tail = FIT_WARN
    assert classify_cuda_failure(FIT_WARN) is None
    with caplog.at_level(logging.INFO, logger="erudi"):
        _load(windows)
        windows.spill = 2 * GIB
        outcome = CUDA_Engine.brake_point("turn_start", llm_id=7)
    assert outcome.action == "shrink+respawn"
    assert CUDA_Engine._model["window_sized"] is True
    assert not _records(caplog, "Flash Attention")
    assert not [
        r for r in caplog.records if r.levelno >= logging.WARNING and "Shared" not in r.getMessage()
    ]


# ===================== the placement classification =====================


def test_the_offloaded_line_decides_when_present(windows, caplog):
    windows.tail = "load_tensors: offloaded 20/29 layers to GPU\n"
    with caplog.at_level(logging.INFO, logger="erudi"):
        handle = _load(windows)
    assert handle["window_sized"] is False
    (info,) = _records(caplog, "partly on the GPU", logging.INFO)
    assert "20/29" in info.getMessage()
    reads = windows.reads
    assert CUDA_Engine.brake_point("turn_start", llm_id=7) == BrakeOutcome()
    assert CUDA_Engine.brake_point("idle") == BrakeOutcome()
    assert windows.reads == reads


def test_a_full_line_wins_over_the_bytes(windows):
    windows.tail = "load_tensors: offloaded 29/29 layers to GPU\n"
    windows.device_override = GIB
    assert _load(windows)["window_sized"] is True


def test_the_byte_measurement_classifies_a_partial_child(linux, caplog):
    # fit dropped a third of the layers: their weights and KV sit in RAM.
    linux.device_override = 3 * GIB
    CUDA_Engine._window_memory["erudi-7"] = {"cap": 30_720, "faulty": W, "at": time.time()}
    with caplog.at_level(logging.INFO, logger="erudi"):
        handle = _load(linux)
    assert handle["window_sized"] is False
    (info,) = _records(caplog, "partly on the GPU", logging.INFO)
    assert "measured" in info.getMessage()
    # The window memory keeps its cap and faulty window: the brake re-arms
    # at the next load of the alias.
    assert CUDA_Engine._window_memory["erudi-7"]["cap"] == 30_720
    assert CUDA_Engine._window_memory["erudi-7"]["faulty"] == W
    # memory_bound is stamped from the budget alone.
    assert CUDA_Engine.memory_bound_tokens() is None


def test_a_windows_spill_at_load_does_not_read_as_partial(windows):
    # 2 GiB of the child already demoted to shared memory at load: its
    # dedicated VRAM alone would read as partial.
    windows.spill = 2 * GIB + cuda_memory.CUDA_SPILL_BASELINE_BYTES
    dedicated = WEIGHTS - TOK_EMBD + W * KV + windows.extras - windows.spill
    predicted = WEIGHTS - TOK_EMBD + W * KV
    assert cuda_memory.placement_partial(
        predicted_bytes=predicted, block_count=28, measured_bytes=dedicated
    )
    assert _load(windows)["window_sized"] is True


def test_no_reader_answering_assumes_full(windows, caplog):
    windows.vram_unavailable = "PDH denied"
    with caplog.at_level(logging.DEBUG, logger="erudi"):
        handle = _load(windows)
    assert handle["window_sized"] is True
    assert _records(caplog, "placement unknown", logging.DEBUG)


# ===================== the spill brake (Windows) =====================


def test_a_spill_warning_shrinks_and_respawns_once_away_pinning_every_layer(windows, caplog):
    handle = _load(windows)
    windows.spill = 300 * MIB
    with caplog.at_level(logging.INFO, logger="erudi"):
        outcome = CUDA_Engine.brake_point("turn_end")
        assert outcome == BrakeOutcome(acted=True, action="shrink")
        assert CUDA_Engine._window_memory["erudi-7"]["cap"] == int(W * mb.SHRINK_FACTOR)
        # A reader between messages pays no respawn.
        assert len(windows.spawns) == 1
        sized, nvml = windows.sized, windows.nvml_reads
        _away(150)
        outcome = CUDA_Engine.brake_point("idle")
    assert outcome == BrakeOutcome(acted=True, action="respawn")
    new = CUDA_Engine._model
    assert new is not handle
    assert handle["proc"].stopped_by == "terminate"
    spawn = windows.spawns[-1]
    assert spawn["ctx_size"] == int(W * mb.SHRINK_FACTOR)
    assert spawn["gpu_layers"] == -2
    # A Windows respawn re-reads nothing and waits for nothing.
    assert windows.sized == sized
    assert windows.nvml_reads == nvml
    assert new["window_sized"] is True
    assert new["brake"].notice == "acted"
    assert CUDA_Engine.brake_reload_pending(7)
    _assert_no_key(caplog)


def test_a_critical_spill_at_turn_start_respawns_in_the_same_turn(windows):
    _load(windows)
    windows.spill = 2 * GIB
    outcome = CUDA_Engine.brake_point("turn_start", llm_id=7)
    assert outcome == BrakeOutcome(acted=True, action="shrink+respawn")
    assert CUDA_Engine._model["context_tokens"] == int(W * mb.SHRINK_FACTOR)
    assert CUDA_Engine.brake_reload_pending(7)


def test_a_spill_seen_by_a_respawn_seed_is_an_info(windows, caplog):
    _load(windows)
    windows.spill = 2 * GIB
    with caplog.at_level(logging.INFO, logger="erudi"):
        CUDA_Engine.brake_point("turn_start", llm_id=7)
    records = _records(caplog, "Shared GPU memory")
    assert records and all(r.levelno == logging.INFO for r in records)


def test_two_idle_criticals_unload(windows, caplog):
    _load(windows)
    windows.spill = 2 * GIB
    CUDA_Engine._last_used = datetime.now() - timedelta(seconds=600)
    with caplog.at_level(logging.INFO, logger="erudi"):
        first = CUDA_Engine.brake_point("idle")
        assert first.action == "shrink+respawn"
        CUDA_Engine._model["brake"].last_sample_at -= mb.BRAKE_IDLE_INTERVAL_S
        second = CUDA_Engine.brake_point("idle")
    assert second == BrakeOutcome(acted=True, action="unload")
    assert CUDA_Engine._model is None
    assert CUDA_Engine.brake_reload_pending(7)
    assert len(_records(caplog, "action=unload", logging.WARNING)) == 1


def test_respawns_reuse_the_live_window_whatever_the_free_vram_says(windows):
    """M-A: a series of respawns derives -c from the live handle's static
    window and the cap; the free VRAM of the moment is never read."""
    _load(windows)
    windows.spill = 2 * GIB
    sized = windows.sized
    windows.free = GIB  # a hog took the card: a fresh sizing would collapse
    CUDA_Engine.brake_point("turn_start", llm_id=7)
    CUDA_Engine.begin_memory_window()
    CUDA_Engine.brake_point("turn_end")
    CUDA_Engine.brake_point("turn_start", llm_id=7)
    windows_seen = [s["ctx_size"] for s in windows.spawns]
    first = int(W * mb.SHRINK_FACTOR)
    assert windows_seen == [W, first, int(first * mb.SHRINK_FACTOR) // 256 * 256]
    assert windows.sized == sized
    for spawn in windows.spawns[1:]:
        assert spawn["static_window_tokens"] == W
        assert spawn["gpu_layers"] == -2


def test_damage_last_tracks_only_the_ram_counter(windows):
    handle = _load(windows)
    windows.damage = 10 * MIB
    windows.spill = 300 * MIB
    CUDA_Engine.brake_point("turn_start", llm_id=7)
    assert handle["damage_last"] == 10 * MIB
    windows.spill = 900 * MIB
    windows.damage = 12 * MIB
    handle["brake"].turn_happened = True
    CUDA_Engine.brake_point("turn_end")
    assert CUDA_Engine._model["damage_last"] == 12 * MIB


def test_ram_pressure_on_a_full_offload_child_is_external_and_never_resizes(windows):
    _load(windows)
    windows.level = mb.LEVEL_WARNING
    windows.damage = 64 * MIB
    assert CUDA_Engine.brake_point("turn_end") == BrakeOutcome()
    assert CUDA_Engine._model["brake"].notice == "external"
    assert len(windows.spawns) == 1


def test_the_forced_pressure_seam_drives_nothing_on_a_fallback_child(windows, monkeypatch):
    windows.inputs = None  # shape unreadable: the fallback shape, brake off
    monkeypatch.setenv("ERUDI_FORCE_MEMORY_PRESSURE", "critical")
    handle = _load(windows)
    assert handle["window_sized"] is False
    reads = windows.reads
    CUDA_Engine._last_used = datetime.now() - timedelta(seconds=600)
    for point in ("turn_start", "turn_end", "idle", "idle"):
        assert CUDA_Engine.brake_point(point, llm_id=7) == BrakeOutcome()
    assert windows.reads == reads
    assert CUDA_Engine._model is handle


# ===================== the Linux respawn =====================


def test_a_linux_respawn_waits_for_the_pid_to_leave_and_keeps_auto_layers(linux, caplog):
    handle = _load(linux)
    linux.linger_reads = 0
    linux.children[handle["pid"]]["linger"] = 2
    # Linux has no spill counter by design; the engine reacts to whatever
    # read_vram answers, so the test feeds one.
    linux.spill = 2 * GIB
    with caplog.at_level(logging.DEBUG, logger="erudi"):
        outcome = CUDA_Engine.brake_point("turn_start", llm_id=7)
    assert outcome.action == "shrink+respawn"
    assert handle["proc"].stopped_by == "send_signal"
    spawn = linux.spawns[-1]
    assert spawn["gpu_layers"] == -1
    assert spawn["ctx_size"] == int(W * mb.SHRINK_FACTOR)
    assert _records(caplog, "VRAM wait after the stop: released")
    assert linux.now == pytest.approx(2 * cuda_memory.CUDA_RESPAWN_VRAM_POLL_S)


def test_a_linux_respawn_wait_ends_on_the_free_vram(linux, caplog):
    handle = _load(linux)
    linux.children[handle["pid"]]["linger"] = 100
    linux.spill = 2 * GIB
    linux.free = int(13.8 * GIB)
    # The free VRAM recovers the killed child's bytes at once.
    original = CUDA_Engine._read_vram_info.__func__

    def _freed(cls):
        info = original(cls)
        if not handle["proc"].alive:
            return cuda_memory.DeviceMemory(
                index=0,
                uuid=UUID,
                total_bytes=info.total_bytes,
                free_bytes=info.free_bytes + linux.device_bytes(handle["pid"]),
                processes=info.processes,
            )
        return info

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(CUDA_Engine, "_read_vram_info", classmethod(_freed))
        with caplog.at_level(logging.DEBUG, logger="erudi"):
            CUDA_Engine.brake_point("turn_start", llm_id=7)
    assert _records(caplog, "VRAM wait after the stop: freed")
    assert linux.now == 0


def test_a_linux_respawn_wait_times_out_and_still_passes_a_window(linux, caplog):
    handle = _load(linux)
    linux.children[handle["pid"]]["linger"] = 1000
    linux.spill = 2 * GIB
    with caplog.at_level(logging.DEBUG, logger="erudi"):
        CUDA_Engine.brake_point("turn_start", llm_id=7)
    assert _records(caplog, "VRAM wait after the stop: timeout")
    assert linux.now == pytest.approx(cuda_memory.CUDA_RESPAWN_VRAM_WAIT_S)
    assert linux.spawns[-1]["ctx_size"] == int(W * mb.SHRINK_FACTOR)


def test_a_linux_respawn_that_goes_partial_still_delivers_its_notice(linux, caplog):
    """M-4: fit re-measured a crowded GPU and dropped layers; the brake goes
    off, the window memory keeps the cap, the acted notice and the owed
    reload state are still delivered."""
    _load(linux)
    linux.spill = 2 * GIB
    linux.device_override = 3 * GIB
    with caplog.at_level(logging.INFO, logger="erudi"):
        outcome = CUDA_Engine.brake_point("turn_start", llm_id=7)
    assert outcome.action == "shrink+respawn"
    new = CUDA_Engine._model
    assert new["window_sized"] is False
    assert _records(caplog, "partly on the GPU", logging.INFO)
    assert CUDA_Engine._window_memory["erudi-7"]["cap"] == int(W * mb.SHRINK_FACTOR)
    assert CUDA_Engine.brake_reload_pending(7)
    assert CUDA_Engine.consume_brake_notice() == "acted"
    reads = linux.reads
    assert CUDA_Engine.brake_point("turn_end") == BrakeOutcome()
    assert linux.reads == reads


def test_a_respawn_never_falls_to_the_fallback_shape(linux):
    _load(linux)
    linux.spill = 2 * GIB
    linux.free = 0  # the card looks full: a fresh sizing would not fit the floor
    CUDA_Engine.brake_point("turn_start", llm_id=7)
    assert linux.spawns[-1]["ctx_size"] is not None
    assert linux.spawns[-1]["ctx_size"] >= MIN_KV_BOUND_TOKENS


def test_a_respawn_context_for_another_alias_is_a_fresh_load(windows):
    handle = _load(windows)
    sized = windows.sized
    ctx = CUDA_Engine._prepare_spawn_context(
        Path("/models/q.gguf"), "erudi-8", brake_cap=20_480, respawn_from=handle
    )
    assert windows.sized == sized + 1
    assert ctx["gpu_layers"] == -1


# ===================== the OOM retry at load =====================


def _oom():
    return EngineException(
        message="llama-server child exited before becoming ready",
        trace="cudaMalloc failed: out of memory",
        engine_code="CUDA_OUT_OF_MEMORY",
    )


def test_an_oom_at_load_retries_once_at_the_floor(windows, caplog):
    windows.probe_fail = [_oom()]
    with caplog.at_level(logging.WARNING, logger="erudi"):
        handle = _load(windows)
    assert [s["ctx_size"] for s in windows.spawns] == [W, MIN_KV_BOUND_TOKENS]
    assert handle["context_tokens"] == MIN_KV_BOUND_TOKENS
    (warning,) = _records(caplog, "out of memory", logging.WARNING)
    assert "host" in warning.getMessage()


def test_a_second_oom_is_the_load_error(windows):
    windows.probe_fail = [_oom(), _oom()]
    with pytest.raises(EngineException) as raised:
        _load(windows)
    assert raised.value.engine_code == "CUDA_OUT_OF_MEMORY"
    assert len(windows.spawns) == 2


def test_a_fallback_shaped_child_is_not_retried(windows):
    windows.inputs = None
    windows.probe_fail = [_oom()]
    with pytest.raises(EngineException):
        _load(windows)
    assert len(windows.spawns) == 1


def test_the_oom_retry_never_reads_the_stale_handle_of_the_previous_model(windows):
    """v4 note 1: during a load the class still holds the previous model's
    handle; the retry is a fresh load, never a respawn of that handle."""
    stale = {
        "alias": "erudi-3",
        "pid": 1,
        "proc": _Popen(1),
        "static_window_tokens": 8192,
        "window_sized": True,
        "gpu_uuid": "GPU-stale",
    }
    CUDA_Engine._model = stale
    windows.probe_fail = [_oom()]
    sized = windows.sized
    handle = CUDA_Engine._start_server(
        model_path=Path("/models/q.gguf"), alias="erudi-7", port=27200
    )
    retry = windows.spawns[-1]
    assert retry["gpu_layers"] == -1  # a respawn would pin -ngl -2 on Windows
    assert retry["gpu_uuid"] == UUID
    assert retry["static_window_tokens"] == W
    assert retry["ctx_size"] == MIN_KV_BOUND_TOKENS
    assert windows.sized == sized + 2
    assert handle["alias"] == "erudi-7"


def test_a_failed_respawn_degrades_into_an_unload_without_a_retry(windows, caplog):
    _load(windows)
    windows.spill = 2 * GIB
    windows.probe_fail = [_oom()]
    with caplog.at_level(logging.WARNING, logger="erudi"):
        outcome = CUDA_Engine.brake_point("turn_start", llm_id=7)
    assert outcome == BrakeOutcome(acted=True, action="respawn-failed")
    assert len(windows.spawns) == 2
    assert CUDA_Engine._model is None
    assert CUDA_Engine.brake_reload_pending(7)
