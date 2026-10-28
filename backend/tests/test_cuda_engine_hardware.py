"""Tests for `CUDA_Engine` hardware detection and scoring.

Complements `test_cuda_engine_server.py` (spawn hooks / config attrs) by
pinning the NVML-driven hardware paths with a fake `pynvml` module injected
into `sys.modules`, the spawn's VRAM read (`_read_vram_info`, the one NVML
seam of the window sizing) and the performance scoring pipeline.

No real GPU, subprocess or model is required anywhere in this file.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.exceptions import HardwareException
from src.engines.cuda_engine import CUDA_Engine


# =====================================================================
# Fake pynvml factory
# =====================================================================


class _FakeMem:
    total = 24 * 1024**3  # 24 GB
    free = 20 * 1024**3  # 20 GB


def _installed_toolkit_bin_dir() -> Path | None:
    """The CUDA bin dir the platform default search would find, if any.

    Mirrors steps 2 and 3 of CUDA_Engine._resolve_cuda_bin_dir (CUDA_PATH
    excluded). The guard used to test only the Linux default, so both
    no-toolkit tests failed on any Windows dev box with the toolkit installed
    at its default location -- the platform-agnostic gap #358 set out to close.
    """
    if os.name == "nt":
        base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
        if base.is_dir():
            for ver in sorted(base.iterdir(), reverse=True):
                if (ver / "bin").is_dir():
                    return ver / "bin"
        return None
    linux_default = Path("/usr/local/cuda/bin")
    return linux_default if linux_default.is_dir() else None


def make_fake_pynvml(
    *,
    device_count: int = 1,
    compute_capability: tuple[int, int] = (8, 9),
    mem_clock_mhz: int = 10500,
    sm_clock_mhz: int = 2520,
    idle_mem_clock_mhz: int = 405,
    idle_sm_clock_mhz: int = 300,
    bus_bits: int = 384,
    pcie_gen: int = 4,
    pcie_width: int = 16,
) -> types.ModuleType:
    """Build a plausible fake pynvml module (single RTX-4090-ish GPU).

    `mem_clock_mhz`/`sm_clock_mhz` are the card's RATED clocks, served by
    nvmlDeviceGetMaxClockInfo. nvmlDeviceGetClockInfo serves the far lower
    idle clocks a real GPU reports when the profile is built at startup —
    reading those instead is what made a 448 GB/s card look like a 13 GB/s one.
    """
    mod = types.ModuleType("pynvml")

    class NVMLError(Exception):
        pass

    mod.NVMLError = NVMLError
    mod.NVML_CLOCK_SM = 1
    mod.NVML_CLOCK_MEM = 2
    mod.nvmlInit = lambda: None
    mod.nvmlDeviceGetCount = lambda: device_count
    mod.nvmlDeviceGetHandleByIndex = lambda idx: f"handle-{idx}"
    mod.nvmlDeviceGetName = lambda h: "GeForce RTX 4090"
    mod.nvmlDeviceGetMemoryInfo = lambda h: _FakeMem
    mod.nvmlDeviceGetCudaComputeCapability = lambda h: compute_capability
    mod.nvmlSystemGetCudaDriverVersion = lambda: 12010
    mod.nvmlDeviceGetClockInfo = (
        lambda h, clock: idle_mem_clock_mhz if clock == mod.NVML_CLOCK_MEM else idle_sm_clock_mhz
    )
    mod.nvmlDeviceGetMaxClockInfo = (
        lambda h, clock: mem_clock_mhz if clock == mod.NVML_CLOCK_MEM else sm_clock_mhz
    )
    mod.nvmlDeviceGetMemoryBusWidth = lambda h: bus_bits
    mod.nvmlDeviceGetMaxPcieLinkGeneration = lambda h: pcie_gen
    mod.nvmlDeviceGetMaxPcieLinkWidth = lambda h: pcie_width
    return mod


@pytest.fixture
def fake_nvml(monkeypatch):
    """Inject a healthy single-GPU fake pynvml for the duration of a test."""
    mod = make_fake_pynvml()
    monkeypatch.setitem(sys.modules, "pynvml", mod)
    return mod


@pytest.fixture
def no_nvml(monkeypatch):
    """Make `import pynvml` fail (simulates a machine without the package)."""
    monkeypatch.setitem(sys.modules, "pynvml", None)


# =====================================================================
# UNIT - NVML helpers
# =====================================================================


@pytest.mark.unit
class TestNvmlHelpers:
    def test_init_nvml_success(self, fake_nvml):
        assert CUDA_Engine._init_nvml() is True

    def test_init_nvml_already_initialized_is_ok(self, monkeypatch):
        mod = make_fake_pynvml()

        def raise_already():
            raise mod.NVMLError("Already initialized")

        mod.nvmlInit = raise_already
        monkeypatch.setitem(sys.modules, "pynvml", mod)
        assert CUDA_Engine._init_nvml() is True

    def test_init_nvml_other_error_is_false(self, monkeypatch):
        mod = make_fake_pynvml()

        def raise_driver():
            raise mod.NVMLError("Driver Not Loaded")

        mod.nvmlInit = raise_driver
        monkeypatch.setitem(sys.modules, "pynvml", mod)
        assert CUDA_Engine._init_nvml() is False

    def test_init_nvml_import_error_is_false(self, no_nvml):
        assert CUDA_Engine._init_nvml() is False

    def test_cuda_available_true_with_gpu(self, fake_nvml):
        assert CUDA_Engine._cuda_available() is True

    def test_cuda_available_false_with_zero_gpus(self, monkeypatch):
        mod = make_fake_pynvml(device_count=0)
        monkeypatch.setitem(sys.modules, "pynvml", mod)
        assert CUDA_Engine._cuda_available() is False

    def test_cuda_available_false_without_nvml(self, no_nvml):
        assert CUDA_Engine._cuda_available() is False

    def test_get_compute_capability(self, fake_nvml):
        assert CUDA_Engine._get_compute_capability("handle-0") == (8, 9)

    def test_get_compute_capability_fallback(self, no_nvml):
        assert CUDA_Engine._get_compute_capability("handle-0") == (0, 0)

    def test_get_cuda_driver_version(self, fake_nvml):
        # 12010 -> major 12, minor 10 // 10 = 1
        assert CUDA_Engine._get_cuda_driver_version() == "12.1"

    def test_get_cuda_driver_version_fallback(self, no_nvml):
        assert CUDA_Engine._get_cuda_driver_version() == "Unknown"

    def test_get_nvml_gpus_lists_devices(self, monkeypatch):
        mod = make_fake_pynvml(device_count=2)
        monkeypatch.setitem(sys.modules, "pynvml", mod)
        gpus = CUDA_Engine._get_nvml_gpus()
        assert len(gpus) == 2
        assert gpus[0]["name"] == "GeForce RTX 4090"
        assert gpus[0]["vram_total_mb"] == pytest.approx(24 * 1024)
        assert gpus[1]["id"] == 1

    def test_get_nvml_gpus_skips_broken_device(self, monkeypatch):
        mod = make_fake_pynvml(device_count=2)

        def name_or_raise(handle):
            if handle == "handle-1":
                raise mod.NVMLError("GPU lost")
            return "OK GPU"

        mod.nvmlDeviceGetName = name_or_raise
        monkeypatch.setitem(sys.modules, "pynvml", mod)
        gpus = CUDA_Engine._get_nvml_gpus()
        assert [g["id"] for g in gpus] == [0]

    def test_get_nvml_gpus_outer_failure_returns_empty(self, monkeypatch):
        mod = make_fake_pynvml()

        def boom():
            raise RuntimeError("NVML meltdown")

        mod.nvmlDeviceGetCount = boom
        monkeypatch.setitem(sys.modules, "pynvml", mod)
        assert CUDA_Engine._get_nvml_gpus() == []

    def test_get_nvml_gpus_without_nvml_returns_empty(self, no_nvml):
        assert CUDA_Engine._get_nvml_gpus() == []

    def test_select_best_gpu_empty_is_none(self):
        assert CUDA_Engine._select_best_gpu([]) is None

    def test_select_best_gpu_picks_highest_vram(self):
        gpus = [
            {"id": 0, "vram_total_mb": 8192.0},
            {"id": 1, "vram_total_mb": 24576.0},
            {"id": 2, "vram_total_mb": 12288.0},
        ]
        assert CUDA_Engine._select_best_gpu(gpus)["id"] == 1

    def test_get_sm_count_reads_driver_attribute(self, monkeypatch):
        import ctypes

        class FakeNvcuda:
            def cuInit(self, flags):
                return 0

            def cuDeviceGet(self, device_ref, device_id):
                return 0

            def cuDeviceGetAttribute(self, out_ref, attr, device):
                assert attr == 16  # CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
                out_ref._obj.value = 128
                return 0

        monkeypatch.setattr(ctypes, "CDLL", lambda name: FakeNvcuda())
        assert CUDA_Engine._get_sm_count(0) == 128

    def test_get_sm_count_fallback_without_driver(self, monkeypatch):
        import ctypes

        def raise_oserror(name):
            raise OSError("nvcuda.dll not found")

        monkeypatch.setattr(ctypes, "CDLL", raise_oserror)
        assert CUDA_Engine._get_sm_count(0) == 0

    def test_get_cpu_performance_units_positive(self):
        assert CUDA_Engine._get_cpu_performance_units() > 0

    def test_get_cpu_performance_units_fallback(self, monkeypatch):
        import psutil

        def boom():
            raise RuntimeError("no freq")

        monkeypatch.setattr(psutil, "cpu_freq", boom)
        assert CUDA_Engine._get_cpu_performance_units() == 10.0

    def test_get_pcie_capacity(self, fake_nvml):
        assert CUDA_Engine._get_pcie_capacity("handle-0") == 64.0  # gen4 x16

    def test_get_pcie_capacity_fallback(self, no_nvml):
        assert CUDA_Engine._get_pcie_capacity("handle-0") == 16.0


# =====================================================================
# UNIT - _compute_gpu_layers (VRAM heuristic)
# =====================================================================


def _add_spawn_reads(mod, *, totals=(24 * 1024**3,), v2=True, processes=None):
    """Give the fake pynvml what the spawn's VRAM read needs: per-device
    memory (v2 when ``v2``), a UUID and the compute-process list."""

    def _memory(handle, version=None):
        index = int(handle.split("-")[1])
        if version is not None and not v2:
            raise mod.NVMLError("Function Not Found")
        return types.SimpleNamespace(total=totals[index], free=totals[index] - 1024**3)

    mod.nvmlMemory_v2 = 0x02000028
    mod.nvmlDeviceGetCount = lambda: len(totals)
    mod.nvmlDeviceGetMemoryInfo = _memory
    mod.nvmlDeviceGetUUID = lambda h: f"GPU-uuid-{h.split('-')[1]}"

    def _processes(handle):
        if processes is None:
            raise mod.NVMLError("Not Supported")
        return [types.SimpleNamespace(pid=pid, usedGpuMemory=used) for pid, used in processes]

    mod.nvmlDeviceGetComputeRunningProcesses = _processes
    return mod


@pytest.mark.unit
class TestReadVramInfo:
    """`_read_vram_info` is the ONE place the spawn path reads NVML: every
    other CUDA test monkeypatches it. Here the fake pynvml module stands in."""

    def test_the_largest_card_is_selected_with_its_uuid_and_processes(self, monkeypatch):
        mod = _add_spawn_reads(
            make_fake_pynvml(),
            totals=(8 * 1024**3, 24 * 1024**3),
            processes=[(4242, 3 * 1024**3)],
        )
        monkeypatch.setitem(sys.modules, "pynvml", mod)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        info = CUDA_Engine._read_vram_info()
        assert info.present
        assert info.index == 1
        assert info.uuid == "GPU-uuid-1"
        assert info.total_bytes == 24 * 1024**3
        assert info.free_bytes == 23 * 1024**3
        assert info.processes == ((4242, 3 * 1024**3),)

    def test_memory_info_falls_back_to_v1_and_processes_to_unknown(self, monkeypatch):
        mod = _add_spawn_reads(make_fake_pynvml(), v2=False, processes=None)
        monkeypatch.setitem(sys.modules, "pynvml", mod)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        info = CUDA_Engine._read_vram_info()
        assert info.total_bytes == 24 * 1024**3
        assert info.processes is None

    def test_a_user_cuda_visible_devices_restricts_the_selection(self, monkeypatch):
        mod = _add_spawn_reads(make_fake_pynvml(), totals=(8 * 1024**3, 24 * 1024**3))
        monkeypatch.setitem(sys.modules, "pynvml", mod)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-uuid-0")
        info = CUDA_Engine._read_vram_info()
        assert info.index == 0
        assert info.visible_devices == 1
        # Two named devices: llama.cpp would split the model across both.
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-uuid-0,1")
        assert CUDA_Engine._read_vram_info().visible_devices == 2

    def test_an_empty_cuda_visible_devices_is_not_set(self, monkeypatch):
        """backend/.env.example lists the variable empty: a developer who
        copies it to .env must keep every GPU."""
        mod = _add_spawn_reads(make_fake_pynvml(), totals=(8 * 1024**3, 24 * 1024**3))
        monkeypatch.setitem(sys.modules, "pynvml", mod)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        info = CUDA_Engine._read_vram_info()
        assert info.present
        assert info.index == 1
        assert info.visible_devices == 1

    def test_no_device_is_no_gpu(self, monkeypatch):
        mod = _add_spawn_reads(make_fake_pynvml(), totals=())
        monkeypatch.setitem(sys.modules, "pynvml", mod)
        info = CUDA_Engine._read_vram_info()
        assert not info.present
        assert info.unavailable is None

    def test_without_pynvml_nvml_is_unreadable_and_silent(self, no_nvml, caplog):
        import logging

        with caplog.at_level(logging.DEBUG, logger="erudi"):
            info = CUDA_Engine._read_vram_info()
        assert not info.present
        assert info.unavailable
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_without_pynvml_the_spawn_falls_back_with_one_warning(
        self, no_nvml, monkeypatch, caplog
    ):
        """Poisoned pynvml: the spawn takes the fallback shape (fit decides),
        the VRAM readers answer unavailable, and nothing raises."""
        import logging

        from src.engines import memory_pressure

        monkeypatch.delenv("ERUDI_CTX", raising=False)
        monkeypatch.delenv("ERUDI_VRAM_OVERRIDE_GB", raising=False)
        with caplog.at_level(logging.WARNING, logger="erudi"):
            ctx = CUDA_Engine._prepare_spawn_context(Path("/m.gguf"), "erudi-1")
        assert (ctx["ctx_size"], ctx["gpu_layers"], ctx["window_sized"]) == (None, -1, False)
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1
        assert memory_pressure.child_vram_bytes(1, CUDA_Engine._read_vram_info().processes) is None
        # The host's own GPU counters are never read here: the null provider.
        monkeypatch.setattr(memory_pressure, "_PROVIDER", memory_pressure._NullProvider())
        assert memory_pressure.read_vram(1, healthy_bytes=0).spill_bytes is None


# =====================================================================
# UNIT - _resolve_cuda_bin_dir
# =====================================================================


@pytest.mark.unit
class TestResolveCudaBinDir:
    def test_cuda_path_env_with_bin(self, monkeypatch, tmp_path):
        (tmp_path / "bin").mkdir()
        monkeypatch.setenv("CUDA_PATH", str(tmp_path))
        assert CUDA_Engine._resolve_cuda_bin_dir() == tmp_path / "bin"

    def test_cuda_path_env_without_bin_is_ignored(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CUDA_PATH", str(tmp_path))  # no bin/ inside
        # CUDA_PATH is ignored, so the answer is whatever the platform default
        # search finds -- None on a host with no toolkit installed.
        result = CUDA_Engine._resolve_cuda_bin_dir()
        assert result is None or result == _installed_toolkit_bin_dir()

    def test_no_env_no_default_returns_none(self, monkeypatch):
        monkeypatch.delenv("CUDA_PATH", raising=False)
        if _installed_toolkit_bin_dir() is not None:
            pytest.skip("host actually has a CUDA toolkit installed")
        assert CUDA_Engine._resolve_cuda_bin_dir() is None


# =====================================================================
# UNIT - get_hardware_info
# =====================================================================


@pytest.mark.unit
class TestGetHardwareInfo:
    def test_full_cuda_path(self, fake_nvml):
        with patch.object(CUDA_Engine, "_get_sm_count", return_value=128):
            info = CUDA_Engine.get_hardware_info()
        gpu = info["gpu"]
        assert info["backend_type"] == "cuda"
        assert gpu["gpu_name"] == "GeForce RTX 4090"
        assert gpu["cuda_available"] is True
        assert gpu["compute_capability"] == "8.9"
        assert gpu["cuda_cores"] == 128 * 128  # 128 SM x 128 cores/SM (Ampere+)
        assert gpu["vram_total_gb"] == pytest.approx(24.0)
        assert gpu["vram_available_gb"] == pytest.approx(20.0)
        # bandwidth = 2 * 10500/1000 * 384/8 = 1008 GB/s
        assert gpu["memory_bandwidth_gbs"] == pytest.approx(1008.0)
        assert info["memory"]["total_memory_gb"] > 0
        assert info["storage"]["total_gb"] > 0

    def test_bandwidth_failure_keeps_zero(self, monkeypatch):
        mod = make_fake_pynvml()

        def clock_boom(handle, clock):
            raise RuntimeError("clock query unsupported")

        mod.nvmlDeviceGetClockInfo = clock_boom
        mod.nvmlDeviceGetMaxClockInfo = clock_boom
        monkeypatch.setitem(sys.modules, "pynvml", mod)
        with patch.object(CUDA_Engine, "_get_sm_count", return_value=64):
            info = CUDA_Engine.get_hardware_info()
        assert info["gpu"]["memory_bandwidth_gbs"] == 0.0

    def test_bandwidth_uses_rated_clock_not_idle_clock(self, monkeypatch):
        """An idle GPU must still be rated at its real bandwidth.

        Regression for the RTX 5060 Ti profile: NVML reported 405 MHz of memory
        clock on an idle card whose rated clock is 14001, and the profile is
        built at startup, when the card is idle by definition. Reading the
        instantaneous clock turned 448 GB/s into 13 GB/s.
        """
        mod = make_fake_pynvml(mem_clock_mhz=14001, idle_mem_clock_mhz=405, bus_bits=128)
        monkeypatch.setitem(sys.modules, "pynvml", mod)
        with patch.object(CUDA_Engine, "_get_sm_count", return_value=36):
            info = CUDA_Engine.get_hardware_info()
        assert info["gpu"]["memory_bandwidth_gbs"] == pytest.approx(448.0)

    def test_bandwidth_falls_back_to_current_clock_when_max_unavailable(self, monkeypatch):
        """A driver without nvmlDeviceGetMaxClockInfo keeps a usable field."""
        mod = make_fake_pynvml(idle_mem_clock_mhz=10500)

        def no_max(handle, clock):
            raise RuntimeError("nvmlDeviceGetMaxClockInfo unsupported")

        mod.nvmlDeviceGetMaxClockInfo = no_max
        monkeypatch.setitem(sys.modules, "pynvml", mod)
        with patch.object(CUDA_Engine, "_get_sm_count", return_value=64):
            info = CUDA_Engine.get_hardware_info()
        assert info["gpu"]["memory_bandwidth_gbs"] == pytest.approx(1008.0)

    def test_no_gpu_fallback_fields(self, no_nvml):
        info = CUDA_Engine.get_hardware_info()
        gpu = info["gpu"]
        assert gpu["gpu_name"] == "No NVIDIA GPU"
        assert gpu["cuda_available"] is False
        assert gpu["vram_total_gb"] == 0.0
        assert info["backend_type"] == "cuda"

    def test_missing_psutil_raises_hardware_exception(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "psutil", None)
        with pytest.raises(HardwareException):
            CUDA_Engine.get_hardware_info()


# =====================================================================
# UNIT - warm_up_accelerator
# =====================================================================


@pytest.mark.unit
class TestWarmUp:
    def test_returns_false_without_cuda(self, no_nvml):
        assert CUDA_Engine.warm_up_accelerator(0.1) is False

    def test_skipped_even_with_cuda(self, fake_nvml):
        # torch matmul warm-up is intentionally disabled (CPU-only torch)
        assert CUDA_Engine.warm_up_accelerator(0.1) is False


# =====================================================================
# UNIT - get_performance_evaluation / get_flat_hardware_data
# =====================================================================


@pytest.mark.unit
class TestPerformanceEvaluation:
    def _eval_with(self, monkeypatch, mod, sm_count=128):
        monkeypatch.setitem(sys.modules, "pynvml", mod)
        with patch.object(CUDA_Engine, "_get_sm_count", return_value=sm_count):
            return CUDA_Engine.get_performance_evaluation()

    def test_full_path_ampere_class_gpu(self, monkeypatch):
        result = self._eval_with(monkeypatch, make_fake_pynvml())
        assert result["backend_type"] == "cuda"
        assert result["accelerator_available"] is True
        assert result["architecture"] == "Ampere"  # CC major 8
        assert result["cuda_version"] == "12.1"
        assert result["estimated_tflops"] > 0
        assert result["tensor_tflops"]["bf16"] > 0
        assert 0 < result["global_inference_score"] <= 100
        assert result["global_inference_label"] in {
            "Amazing",
            "Excellent",
            "Very High",
            "High",
            "Good",
            "Medium",
            "Bad",
            "Very Bad",
            "Poor",
            "Terrible",
        }
        pb = result["performance_breakdown"]
        # These are the names PerformanceBreakdown reads (hardware/endpoints.py
        # _build_performance_breakdown). The engine used to publish
        # gpu_compute_score/vram_capacity_score, which that .get() silently
        # defaulted to 0.0, so /hardware/detailed reported a 24 GB card with
        # 69 TFLOPs as compute_score=0, memory_capacity_score=0.
        assert set(pb) >= {"compute_score", "memory_bandwidth_score", "memory_capacity_score"}
        assert pb["compute_score"] > 0
        assert pb["memory_capacity_score"] > 0

    def test_turing_architecture_label(self, monkeypatch):
        result = self._eval_with(
            monkeypatch, make_fake_pynvml(compute_capability=(7, 5)), sm_count=48
        )
        assert result["architecture"] == "Turing"
        # bf16 requires CC >= 8 -> zero, fp16 available on Turing
        assert result["tensor_tflops"]["bf16"] == 0.0
        assert result["tensor_tflops"]["fp16"] > 0

    def test_hopper_ada_architecture_label(self, monkeypatch):
        result = self._eval_with(monkeypatch, make_fake_pynvml(compute_capability=(9, 0)))
        assert result["architecture"] == "Ada Lovelace / Hopper"

    def test_unknown_architecture_label(self, monkeypatch):
        result = self._eval_with(monkeypatch, make_fake_pynvml(compute_capability=(12, 0)))
        assert result["architecture"] == "Compute 12.0"

    def test_clock_query_failure_uses_fallback_clock(self, monkeypatch):
        mod = make_fake_pynvml()
        calls = {"n": 0}

        def clock_boom(handle, clock):
            raise RuntimeError("no clocks")

        mod.nvmlDeviceGetClockInfo = clock_boom
        mod.nvmlDeviceGetMaxClockInfo = clock_boom
        monkeypatch.setitem(sys.modules, "pynvml", mod)
        with patch.object(CUDA_Engine, "_get_sm_count", return_value=128):
            result = CUDA_Engine.get_performance_evaluation()
        assert result["sm_clock_ghz"] == 1.5
        assert calls["n"] == 0  # sanity: fake used, not real NVML

    def test_no_cuda_fallback_scores(self, no_nvml):
        result = CUDA_Engine.get_performance_evaluation()
        assert result["accelerator_available"] is False
        assert result["gpu_score"] == 0.0
        assert result["estimated_tflops"] == 0.0
        assert result["cuda_version"] == "N/A"
        assert result["architecture"] == "N/A"
        # PCIe fallback of 50 still contributes to the weighted score
        assert result["performance_breakdown"]["pcie_score"] == 50.0

    def test_detection_failure_raises_hardware_exception(self):
        with patch.object(CUDA_Engine, "get_hardware_info", side_effect=RuntimeError("boom")):
            with pytest.raises(HardwareException):
                CUDA_Engine.get_performance_evaluation()

    def test_flat_hardware_data_strips_non_entity_keys(self):
        fake_eval = {
            "backend_type": "cuda",
            "gpu_name": "X",
            "tensor_tflops": {"fp16": 1.0},
            "sm_clock_ghz": 2.5,
            "compute_units": 1000,
            "accelerator_available": True,
            "global_inference_score": 42.0,
        }
        with patch.object(CUDA_Engine, "get_performance_evaluation", return_value=dict(fake_eval)):
            flat = CUDA_Engine.get_flat_hardware_data()
        assert "tensor_tflops" not in flat
        assert "sm_clock_ghz" not in flat
        assert "compute_units" not in flat
        assert "accelerator_available" not in flat
        assert flat["global_inference_score"] == 42.0
