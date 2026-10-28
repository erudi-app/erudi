"""The CUDA window and its memory brake, executed by ``CUDA_Engine`` (child,
HTTP, NVML, PDH and RAM sensors all faked).

``CUDA_Engine`` sizes ``-c`` for a full offload against the VRAM free at load
(``cuda_memory``) and hands the layers to llama.cpp's fit (``-ngl -1``). After
load it classifies the placement fit chose -- ``offloaded N/M`` when the line
reaches the drainer, the per-pid device bytes otherwise -- and a partial child
runs brake-off. On Windows ``brake_point`` reads the WDDM spill gauge alone --
the Shared GPU memory above the child's baseline, its reading at load bounded
by the healthy level of its window -- into ``memory_brake.llama_step`` (unchanged) and executes it: a respawn
reuses the live handle's window facts folded with the cap, re-reads no VRAM
and pins every layer (``-ngl -2``). RAM pressure never drives it, and no RAM
sensor is read on the CUDA path. On Linux, which has no spill, the brake reads
nothing and never acts. The real NVML and the host's GPU are never touched:
``CUDA_Engine._read_vram_info`` and ``memory_pressure.read_vram`` are faked in
every test, and the fake child implements both ``terminate()`` and
``send_signal()`` (the stop path branches on the OS).
"""

from __future__ import annotations

import logging
import platform
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.core.exceptions import EngineException
from src.engines import base_llama_cpp_engine as base_mod
from src.engines import cuda_memory, gguf_memory, memory_pressure
from src.engines import memory_brake as mb
from src.engines.base_engine import BaseEngine
from src.engines.cuda_compatibility import classify_cuda_failure
from src.engines.cuda_engine import CUDA_Engine
from src.engines.gguf_memory import TensorBytes, kv_bytes_per_token, shape_from_fields
from src.engines.memory_brake import BrakeOutcome, BrakeState, llama_padded
from src.engines.memory_pressure import PressureReading, VramReading
from src.engines.working_window import MIN_KV_BOUND_TOKENS

REAL_READ_VRAM = memory_pressure.read_vram
REAL_READ_PRESSURE = memory_pressure.read_pressure

pytestmark = pytest.mark.unit

GIB = 1024**3
MIB = 1024**2
KEY = "cuda-key-never-logged-0123456789"
QWEN3_FIELDS = {
    "architecture": "qwen3",
    "block_count": 28,
    "context_length": 40_960,
    "embedding_length": 1024,
    "head_count": 16,
    "head_count_kv": 8,
    "key_length": 128,
    "value_length": 128,
}
QWEN3 = shape_from_fields(QWEN3_FIELDS)
KV = 114_688
W = 40_960
# A long-context model whose whole trained window fits the card: 8 KiB of KV
# per token, 2 GiB at 262144 tokens (a KV share of the VRAM the brake acts on).
LONG_W = 262_144
LONG_KV = 8192
LONG = shape_from_fields(
    {
        "architecture": "qwen3",
        "block_count": 8,
        "context_length": LONG_W,
        "embedding_length": 512,
        "head_count": 8,
        "head_count_kv": 4,
        "key_length": 64,
        "value_length": 64,
    }
)


def _trained_at(window):
    return shape_from_fields({**QWEN3_FIELDS, "context_length": window})


def _healthy(window, fa_on=True):
    return cuda_memory.healthy_shared_bytes(window, fa_on=fa_on)


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
        self.tail = ""
        self.inputs = (QWEN3, WEIGHTS)
        self.tensors = TensorBytes(tok_embd_bytes=TOK_EMBD, exps_bytes=0)
        self.spawns = []
        self.children = {}
        self.probe_fail = []
        self.props_fail = False
        self.reads = 0
        self.anonymous_reads = 0
        self.total_ram_reads = 0
        self.kv = KV
        self.vram_reads = []
        self.healthy_seen = []
        self.nvml_reads = 0
        self.sized = 0
        self._ports = iter(range(27201, 27299))
        self._pids = iter(range(5000, 5100))
        world = self

        monkeypatch.setattr(platform, "system", lambda: os_name)

        def _read_pressure():
            world.reads += 1
            return PressureReading(world.level, world.damage)

        monkeypatch.setattr(memory_pressure, "read_pressure", _read_pressure)

        def _child_anonymous_bytes(pid):
            world.anonymous_reads += 1
            return world.host_anonymous

        def _effective_total_ram_bytes():
            world.total_ram_reads += 1
            return world.total_ram

        monkeypatch.setattr(memory_pressure, "child_anonymous_bytes", _child_anonymous_bytes)
        monkeypatch.setattr(
            memory_pressure, "effective_total_ram_bytes", _effective_total_ram_bytes
        )

        def _read_vram(pid, *, healthy_bytes):
            # WDDM: the child's allocation is split between dedicated VRAM
            # and the shared memory it spilled to.
            world.vram_reads.append(pid)
            world.healthy_seen.append(healthy_bytes)
            if world.vram_unavailable:
                return VramReading(None, None, unavailable=world.vram_unavailable)
            total = world.device_bytes(pid)
            return VramReading(world.spill, max(0, total - world.spill))

        monkeypatch.setattr(memory_pressure, "read_vram", _read_vram)

        def _read_vram_info(cls):
            world.nvml_reads += 1
            return cuda_memory.DeviceMemory(
                index=0,
                uuid=UUID,
                total_bytes=world.vram_total,
                free_bytes=world.free,
                processes=tuple(
                    (pid, world.device_bytes(pid))
                    for pid, child in world.children.items()
                    if child["proc"].alive
                ),
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
            world.children[pid] = {"proc": proc, "ctx": ctx}
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
            if world.props_fail:
                raise base_mod.requests.ConnectionError("props down")
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

    def device_bytes(self, pid):
        """What child ``pid`` really holds on the device under a full
        offload: its weights without tok_embd, its KV, and some extras."""
        if self.device_override is not None:
            return self.device_override
        child = self.children.get(pid)
        window = (child["ctx"].get("ctx_size") if child else None) or W
        return WEIGHTS - TOK_EMBD + window * self.kv + self.extras

    def ram_sensor_reads(self):
        """Every read of a RAM sensor: the pressure, the child's anonymous
        memory, the machine's RAM."""
        return self.reads + self.anonymous_reads + self.total_ram_reads


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
    # The CUDA brake reads no RAM paging counter, even at its seed.
    assert "damage_last" not in handle
    # The spill baseline: the load reading, bounded by the healthy level.
    assert handle["fa_on"] is True
    assert handle["spill_baseline"] == 40 * MIB
    assert CUDA_Engine.memory_bound_tokens() is None
    assert _records(caplog, "full offload", logging.INFO)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    _assert_no_key(caplog)


def test_a_load_spill_above_the_healthy_level_warns_on_a_first_load(windows, caplog):
    windows.spill = 400 * MIB
    with caplog.at_level(logging.INFO, logger="erudi"):
        handle = _load(windows)
    assert handle["spill_baseline"] == _healthy(W)
    (warning,) = _records(caplog, "Shared GPU memory", logging.WARNING)
    assert "shared_mib=400" in warning.getMessage()
    assert f"healthy_mib={_healthy(W) // MIB}" in warning.getMessage()


@pytest.mark.parametrize(
    "window, shared_mib",
    # #661's healthy Shared Usage, measured with typeperf on raw llama-server
    # children (Flash Attention on): 0.5B at 8192, 27B at 4096, 4B at 40960.
    [(8192, 90), (4096, 106), (40_960, 130)],
)
def test_the_661_healthy_readings_stay_under_the_healthy_level(windows, caplog, window, shared_mib):
    windows.inputs = (_trained_at(window), WEIGHTS)
    windows.spill = shared_mib * MIB
    with caplog.at_level(logging.INFO, logger="erudi"):
        handle = _load(windows)
    assert handle["context_tokens"] == window
    assert handle["spill_baseline"] == shared_mib * MIB
    assert not _records(caplog, "Shared GPU memory")
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_flash_attention_fallback_takes_the_larger_host_slope(windows, caplog):
    """llama.cpp disabled FA at context creation: the pinned host compute
    buffer roughly doubles (#661: 20.01 MiB at 8192 with ``-fa off``, 11.51
    with FA), so the healthy level takes the non-FA slope."""
    shared = cuda_memory.CUDA_SHARED_FIXED_BYTES + 60 * MIB
    assert _healthy(W, fa_on=True) < shared < _healthy(W, fa_on=False)
    windows.tail = "llama_context: Flash Attention not supported, set to disabled\n"
    windows.spill = shared
    with caplog.at_level(logging.INFO, logger="erudi"):
        handle = _load(windows)
    assert handle["fa_on"] is False
    assert handle["spill_baseline"] == shared
    assert not _records(caplog, "Shared GPU memory")
    assert windows.healthy_seen == [_healthy(W, fa_on=False)]


def test_the_same_reading_with_flash_attention_on_is_above_the_healthy_level(windows, caplog):
    windows.spill = cuda_memory.CUDA_SHARED_FIXED_BYTES + 60 * MIB
    with caplog.at_level(logging.INFO, logger="erudi"):
        handle = _load(windows)
    assert handle["fa_on"] is True
    assert handle["spill_baseline"] == _healthy(W)
    assert len(_records(caplog, "Shared GPU memory", logging.WARNING)) == 1
    assert windows.healthy_seen == [_healthy(W)]


# A model the CUDA Flash-Attention kernels are not proven for (head size 192).
NOT_PROVEN = shape_from_fields(
    {**QWEN3_FIELDS, "context_length": 8192, "key_length": 192, "value_length": 192}
)
FA_ENABLED_LINE = "llama_context: Flash Attention enabled\n"
FA_DISABLED_LINE = "llama_context: Flash Attention not supported, set to disabled\n"


@pytest.mark.parametrize(
    "proven, tail, fa_on",
    [
        (False, FA_ENABLED_LINE, True),
        (True, FA_DISABLED_LINE, False),
        (True, "", True),
        (False, "", False),
    ],
)
def test_fa_on_is_what_llama_cpp_logged_else_the_proof(windows, proven, tail, fa_on):
    """The healthy slope follows what llama.cpp did at context creation (its
    ``Flash Attention enabled`` INFO or its ``not supported`` WARN); with
    neither line in the tail, the static proof decides."""
    shape = _trained_at(8192) if proven else NOT_PROVEN
    windows.inputs = (shape, WEIGHTS)
    windows.kv = kv_bytes_per_token(shape)
    windows.tail = tail
    handle = _load(windows)
    assert handle["fa_proven"] is proven
    assert handle["fa_on"] is fa_on
    assert windows.healthy_seen == [_healthy(8192, fa_on=fa_on)]


@pytest.mark.parametrize("world_name", ["windows", "linux"])
def test_an_unknown_window_loads_the_model_brake_off(request, monkeypatch, caplog, world_name):
    """``ERUDI_CTX=0`` passes ``-c 0`` (llama-server picks the window) and
    ``/props`` fails: the window is unknown. The model loads -- never a
    raise -- and runs brake-off."""
    world = request.getfixturevalue(world_name)
    monkeypatch.setenv("ERUDI_CTX", "0")
    world.props_fail = True
    world.level = mb.LEVEL_CRITICAL
    with caplog.at_level(logging.DEBUG, logger="erudi"):
        handle = _load(world)
        assert CUDA_Engine.brake_point("turn_start", llm_id=7) == BrakeOutcome()
    assert handle["context_tokens"] is None
    assert handle["window_sized"] is False
    assert "spill_baseline" not in handle
    assert len(_records(caplog, "window is unknown", logging.INFO)) == 1
    assert not _records(caplog, "Placement check failed")
    assert world.ram_sensor_reads() == 0


def test_the_linux_load_stamps_no_spill_state(linux):
    handle = _load(linux)
    assert handle["window_sized"] is True
    assert "fa_on" not in handle
    assert "spill_baseline" not in handle


def test_a_load_without_a_spill_reading_stamps_the_healthy_level(windows, caplog):
    """The load reading is missing (PDH answered nothing for the new child):
    the baseline is the healthy level of the window, never 0 -- a healthy
    large-window child read later stays calm."""
    windows.inputs = (LONG, WEIGHTS)
    windows.kv = LONG_KV
    windows.vram_unavailable = "no GPU Process Memory instance for pid"
    handle = _load(windows)
    assert handle["context_tokens"] == LONG_W
    assert handle["spill_baseline"] == _healthy(LONG_W)
    windows.vram_unavailable = None
    windows.spill = _healthy(LONG_W) - 4 * MIB
    with caplog.at_level(logging.INFO, logger="erudi"):
        assert CUDA_Engine.brake_point("turn_start", llm_id=7) == BrakeOutcome()
        handle["brake"].turn_happened = True
        assert CUDA_Engine.brake_point("turn_end") == BrakeOutcome()
    assert CUDA_Engine._model is handle
    assert handle["brake"].in_episode is False
    assert handle["brake"].cap is None
    assert not _records(caplog, "Memory brake")


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
    windows.spill = 2 * GIB + _healthy(W)
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


def test_a_respawned_child_restamps_its_baseline_on_its_new_window(windows):
    """A ``-ngl -2`` respawn is a new child on a smaller window: its healthy
    level -- and so its baseline and the seam's reading -- is the new
    window's, not the old one's."""
    handle = _load(windows)
    assert handle["spill_baseline"] == 0
    windows.spill = 2 * GIB
    assert CUDA_Engine.brake_point("turn_start", llm_id=7).action == "shrink+respawn"
    new = CUDA_Engine._model
    new_window = int(W * mb.SHRINK_FACTOR)
    assert new is not handle
    assert new["context_tokens"] == new_window
    assert _healthy(new_window) < _healthy(W)
    assert new["spill_baseline"] == min(2 * GIB, _healthy(new_window)) == _healthy(new_window)
    assert windows.healthy_seen[-1] == _healthy(new_window)


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


# ===================== RAM pressure never drives the CUDA brake =====================


def test_the_661_idle_unload_of_a_healthy_child_does_not_happen(windows, caplog):
    """#661 FAIL 1, replayed: Windows, the commit charge critical, the
    child's private commit 9053 MiB (WDDM counts its GPU allocations in it),
    a healthy 130 MiB of Shared GPU memory on a 40960 window, the user away
    for 10 minutes. Two idle samples: no unload, no notice, no shrink -- and
    no RAM sensor read at the load, the seed or any sample."""
    windows.level = mb.LEVEL_CRITICAL
    windows.damage = 64 * MIB
    windows.host_anonymous = 9053 * MIB
    windows.total_ram = int(31.1 * GIB)
    windows.spill = 130 * MIB
    with caplog.at_level(logging.INFO, logger="erudi"):
        handle = _load(windows)
        assert handle["context_tokens"] == W
        CUDA_Engine._last_used = datetime.now() - timedelta(seconds=600)
        first = CUDA_Engine.brake_point("idle")
        assert CUDA_Engine._model is handle
        CUDA_Engine._model["brake"].last_sample_at -= mb.BRAKE_IDLE_INTERVAL_S
        second = CUDA_Engine.brake_point("idle")
    assert first == second == BrakeOutcome()
    assert CUDA_Engine._model is handle
    assert len(windows.spawns) == 1
    assert handle["brake"].notice is None
    assert handle["brake"].cap is None
    assert handle["brake"].in_episode is False
    assert not CUDA_Engine.brake_reload_pending(7)
    assert CUDA_Engine.consume_brake_notice() is None
    assert windows.reads == 0
    assert windows.anonymous_reads == 0
    assert windows.total_ram_reads == 0
    # The load and both samples read the spill gauge: sampled, not skipped.
    assert len(windows.vram_reads) == 3
    assert not _records(caplog, "Memory brake sample failed")
    assert not _records(caplog, "Memory brake")
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.parametrize("seam", ["warning", "critical"])
def test_the_ram_seams_do_nothing_on_cuda(windows, monkeypatch, seam):
    """``ERUDI_FORCE_MEMORY_PRESSURE=warning|critical`` acts inside the RAM
    reading, which the CUDA path never calls: no notice, no shrink, no
    unload."""
    calls = []

    def _read_pressure():
        calls.append(1)
        return REAL_READ_PRESSURE()

    monkeypatch.setattr(memory_pressure, "read_pressure", _read_pressure)
    monkeypatch.setenv("ERUDI_FORCE_MEMORY_PRESSURE", seam)
    handle = _load(windows)
    assert CUDA_Engine.brake_point("turn_start", llm_id=7) == BrakeOutcome()
    handle["brake"].turn_happened = True
    assert CUDA_Engine.brake_point("turn_end") == BrakeOutcome()
    CUDA_Engine._last_used = datetime.now() - timedelta(seconds=600)
    for _ in range(2):
        CUDA_Engine._model["brake"].last_sample_at -= mb.BRAKE_IDLE_INTERVAL_S
        assert CUDA_Engine.brake_point("idle") == BrakeOutcome()
    assert CUDA_Engine._model is handle
    assert handle["brake"].notice is None
    assert len(windows.spawns) == 1
    assert calls == []
    assert windows.ram_sensor_reads() == 0


class _DeniedPdh(memory_pressure._WindowsProvider):
    """The Windows provider with ``\\GPU Process Memory`` refused by PDH."""

    def _pdh_gpu_process_memory(self):
        raise OSError("PdhAddEnglishCounterW status 0xc0000bb8")


def test_each_unreadable_episode_of_the_spill_gauge_is_one_warning(windows, caplog):
    """A transient failure does not spend the process's only WARNING: a
    successful reading re-arms it, so each failure episode says so once."""
    _load(windows)
    with caplog.at_level(logging.WARNING, logger="erudi"):
        windows.vram_unavailable = "no GPU Process Memory instance for pid"
        for _ in range(3):
            assert CUDA_Engine.brake_point("turn_start", llm_id=7) == BrakeOutcome()
        assert len(_records(caplog, "GPU spill gauge unreadable", logging.WARNING)) == 1
        windows.vram_unavailable = None
        assert CUDA_Engine.brake_point("turn_start", llm_id=7) == BrakeOutcome()
        windows.vram_unavailable = "PDH GPU Process Memory unreadable (OSError)"
        for _ in range(2):
            assert CUDA_Engine.brake_point("turn_start", llm_id=7) == BrakeOutcome()
    warnings = _records(caplog, "GPU spill gauge unreadable", logging.WARNING)
    assert len(warnings) == 2
    assert all(
        "the memory brake is off while it stays unreadable" in r.getMessage() for r in warnings
    )


def test_an_unreadable_spill_gauge_is_one_warning_and_the_brake_off(windows, monkeypatch, caplog):
    """PDH refuses the GPU counters: the brake has no signal. ONE WARNING in
    all -- worded for the GPU, never claiming a RAM brake stands -- and no
    sample acts."""
    monkeypatch.setattr(memory_pressure, "read_vram", REAL_READ_VRAM)
    monkeypatch.setattr(memory_pressure, "_PROVIDER", _DeniedPdh())
    windows.level = mb.LEVEL_CRITICAL
    with caplog.at_level(logging.INFO, logger="erudi"):
        handle = _load(windows)
        assert handle["spill_baseline"] == _healthy(W)
        CUDA_Engine._last_used = datetime.now() - timedelta(seconds=600)
        for point in ("turn_start", "turn_end", "idle", "idle"):
            assert CUDA_Engine.brake_point(point, llm_id=7) == BrakeOutcome()
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "GPU spill gauge unreadable" in message
    assert "OSError" in message
    assert "the memory brake is off" in message
    assert "RAM" not in message
    assert CUDA_Engine._model is handle
    assert windows.ram_sensor_reads() == 0


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


# ===================== the QA seam on the VRAM (Windows) =====================


class _CalmGpu:
    """The sensor provider of a calm Windows machine: no spill."""

    name = "calm-gpu"

    def __init__(self, world):
        self.world = world

    def vram(self, pid):
        return VramReading(0, self.world.device_bytes(pid))


@pytest.fixture
def vram_seam(windows, monkeypatch):
    """The real ``read_vram`` over a calm provider: only the seam spills."""
    monkeypatch.setattr(memory_pressure, "read_vram", REAL_READ_VRAM)
    monkeypatch.setattr(memory_pressure, "_PROVIDER", _CalmGpu(windows))
    return windows


def test_the_vram_warning_seam_shrinks_then_respawns_once_away(vram_seam, monkeypatch, caplog):
    monkeypatch.setenv("ERUDI_FORCE_MEMORY_PRESSURE", "vram-warning")
    with caplog.at_level(logging.INFO, logger="erudi"):
        handle = _load(vram_seam)
    # The forced reading sits above the healthy level: the at-load WARNING
    # stays visible under the seam.
    assert len(_records(caplog, "Shared GPU memory", logging.WARNING)) == 1
    assert handle["spill_baseline"] == _healthy(W)
    with caplog.at_level(logging.INFO, logger="erudi"):
        assert CUDA_Engine.brake_point("turn_end") == BrakeOutcome(acted=True, action="shrink")
        _away(150)
        assert CUDA_Engine.brake_point("idle") == BrakeOutcome(acted=True, action="respawn")
    spawn = vram_seam.spawns[-1]
    assert spawn["ctx_size"] == int(W * mb.SHRINK_FACTOR)
    assert spawn["gpu_layers"] == -2
    assert CUDA_Engine._model["brake"].notice == "acted"
    assert CUDA_Engine.brake_reload_pending(7)


def test_the_vram_critical_seam_respawns_at_the_turn_boundary(vram_seam, monkeypatch):
    monkeypatch.setenv("ERUDI_FORCE_MEMORY_PRESSURE", "vram-critical")
    _load(vram_seam)
    outcome = CUDA_Engine.brake_point("turn_start", llm_id=7)
    assert outcome == BrakeOutcome(acted=True, action="shrink+respawn")
    assert vram_seam.spawns[-1]["gpu_layers"] == -2
    assert CUDA_Engine.brake_reload_pending(7)


@pytest.mark.parametrize(
    "seam, level, tier",
    [
        ("vram-warning", mb.LEVEL_WARNING, cuda_memory.CUDA_SPILL_WARNING_BYTES),
        ("vram-critical", mb.LEVEL_CRITICAL, cuda_memory.CUDA_SPILL_CRITICAL_BYTES),
    ],
)
@pytest.mark.parametrize("trained", [MIN_KV_BOUND_TOKENS, LONG_W])
def test_the_vram_seams_put_the_spill_exactly_at_the_tier_at_any_window(
    vram_seam, monkeypatch, seam, level, tier, trained
):
    """Whatever the window -- the 4096 floor or a 262144-token model -- the
    forced reading is the child's healthy level plus the tier's threshold,
    so the spill above the baseline is exactly the tier."""
    if trained == LONG_W:
        vram_seam.inputs = (LONG, WEIGHTS)
        vram_seam.kv = LONG_KV
    else:
        vram_seam.inputs = (_trained_at(trained), WEIGHTS)
    monkeypatch.setenv("ERUDI_FORCE_MEMORY_PRESSURE", seam)
    handle = _load(vram_seam)
    assert handle["context_tokens"] == trained
    assert handle["spill_baseline"] == _healthy(trained)
    signals = CUDA_Engine._llama_read_signals(handle)
    assert signals.level == level
    assert signals.damage_bytes == tier
    assert signals.damage_available is True
    assert signals.ram_damage_total is None


def test_the_vram_critical_seam_unloads_at_the_floor_after_two_idle_criticals(
    vram_seam, monkeypatch, caplog
):
    """At the 4096 floor no shrink is possible: two idle criticals unload."""
    vram_seam.inputs = (_trained_at(MIN_KV_BOUND_TOKENS), WEIGHTS)
    monkeypatch.setenv("ERUDI_FORCE_MEMORY_PRESSURE", "vram-critical")
    _load(vram_seam)
    CUDA_Engine._last_used = datetime.now() - timedelta(seconds=600)
    with caplog.at_level(logging.INFO, logger="erudi"):
        assert CUDA_Engine.brake_point("idle") == BrakeOutcome()
        CUDA_Engine._model["brake"].last_sample_at -= mb.BRAKE_IDLE_INTERVAL_S
        assert CUDA_Engine.brake_point("idle") == BrakeOutcome(acted=True, action="unload")
    assert CUDA_Engine._model is None
    assert len(vram_seam.spawns) == 1
    assert CUDA_Engine.brake_reload_pending(7)
    assert len(_records(caplog, "action=unload", logging.WARNING)) == 1
    assert vram_seam.ram_sensor_reads() == 0


def test_the_vram_warning_seam_shrinks_a_262144_window(vram_seam, monkeypatch):
    vram_seam.inputs = (LONG, WEIGHTS)
    vram_seam.kv = LONG_KV
    monkeypatch.setenv("ERUDI_FORCE_MEMORY_PRESSURE", "vram-warning")
    _load(vram_seam)
    assert CUDA_Engine.brake_point("turn_end") == BrakeOutcome(acted=True, action="shrink")
    assert CUDA_Engine._window_memory["erudi-7"]["cap"] == int(LONG_W * mb.SHRINK_FACTOR)


# ===================== Linux: sized at spawn, no brake signal =====================


def test_linux_never_reads_the_spill_and_never_respawns(linux):
    """Linux has no spill: a VRAM reading never reaches the brake, so
    nothing shrinks the window -- not even a reader that reports one."""
    _load(linux)
    linux.spill = 2 * GIB
    reads = len(linux.vram_reads)
    assert CUDA_Engine.brake_point("turn_start", llm_id=7) == BrakeOutcome()
    CUDA_Engine.begin_memory_window()
    assert CUDA_Engine.brake_point("turn_end") == BrakeOutcome()
    assert len(linux.vram_reads) == reads
    assert len(linux.spawns) == 1


def test_linux_reads_no_sensor_and_never_acts_under_ram_pressure(linux, caplog):
    """Linux has no spill and RAM pressure does not drive the CUDA brake:
    ``brake_point`` returns before any read -- no notice, no unload, the
    window sized at spawn never moves."""
    linux.level = mb.LEVEL_CRITICAL
    linux.damage = 64 * MIB
    linux.host_anonymous = 9 * GIB
    linux.spill = 2 * GIB
    with caplog.at_level(logging.INFO, logger="erudi"):
        handle = _load(linux)
        CUDA_Engine._last_used = datetime.now() - timedelta(seconds=600)
        for point in ("turn_start", "turn_end", "idle", "idle", "idle"):
            assert CUDA_Engine.brake_point(point, llm_id=7) == BrakeOutcome()
    assert CUDA_Engine._model is handle
    assert handle["brake"].notice is None
    assert len(linux.spawns) == 1
    assert not CUDA_Engine.brake_reload_pending(7)
    assert CUDA_Engine.consume_brake_notice() is None
    assert linux.ram_sensor_reads() == 0
    assert linux.vram_reads == []
    assert not _records(caplog, "Memory brake")
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_respawn_never_falls_to_the_fallback_shape(windows):
    _load(windows)
    windows.spill = 2 * GIB
    windows.free = 0  # the card looks full: a fresh sizing would not fit the floor
    CUDA_Engine.brake_point("turn_start", llm_id=7)
    assert windows.spawns[-1]["ctx_size"] is not None
    assert windows.spawns[-1]["ctx_size"] >= MIN_KV_BOUND_TOKENS


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
