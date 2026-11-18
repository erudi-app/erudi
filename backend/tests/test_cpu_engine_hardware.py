"""Tests for `CPU_Engine` hardware detection and scoring.

Complements `test_cpu_engine_server.py` (spawn hooks / GGUF picker / kwarg
translation) by pinning the hardware-info structure and the performance scoring
pipeline with its fallbacks. No subprocess or model download happens anywhere
in this file.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from src.engines import cpu_engine as cpu_mod
from src.engines.cpu_engine import CPU_Engine


# =====================================================================
# UNIT - get_hardware_info
# =====================================================================


@pytest.mark.unit
class TestGetHardwareInfo:
    def test_structure_and_cpu_only_gpu_block(self):
        info = CPU_Engine.get_hardware_info()
        assert info["backend_type"] == "cpu"
        assert info["gpu"]["gpu_name"] == "CPU Only"
        assert info["gpu"]["mps_supported"] is False
        assert info["gpu"]["unified_memory"] is False
        assert info["cpu"]["total_cores"] >= 1
        assert info["memory"]["total_memory_gb"] > 0
        assert info["storage"]["total_gb"] > 0
        assert info["memory"]["memory_type"] == "system"

    def test_without_psutil_falls_back_to_os_cpu_count(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "psutil", None)
        info = CPU_Engine.get_hardware_info()
        assert info["cpu"]["total_cores"] >= 1
        # Memory/disk metrics require psutil -> None without it
        assert info["memory"]["total_memory_gb"] is None
        assert info["storage"]["total_gb"] is None

    def test_detection_failure_returns_minimal_fallback(self, monkeypatch):
        def boom():
            raise RuntimeError("sysctl exploded")

        monkeypatch.setattr(cpu_mod, "get_cpu_brand", boom)
        info = CPU_Engine.get_hardware_info()
        assert info["backend_type"] == "cpu"
        assert info["cpu"]["model"] == "Unknown CPU"
        assert info["gpu"]["gpu_name"] == "CPU Only"
        assert info["memory"]["total_memory_gb"] is None

    def test_cpu_brand_overrides_generic_processor(self, monkeypatch):
        monkeypatch.setattr(cpu_mod, "get_cpu_brand", lambda: "Erudium X1")
        info = CPU_Engine.get_hardware_info()
        assert info["cpu"]["model"] == "Erudium X1"


# =====================================================================
# UNIT - get_performance_evaluation / get_flat_hardware_data
# =====================================================================


@pytest.mark.unit
class TestPerformanceEvaluation:
    def test_scores_derived_from_hardware_info(self):
        fake_info = {
            "cpu": {"model": "Erudium X1", "total_cores": 16, "architecture": "arm64"},
            "memory": {"total_memory_gb": 64.0, "available_memory_gb": 32.0},
            "storage": {"total_gb": 1000.0, "available_gb": 500.0},
            "system": {"platform": "Darwin"},
        }
        with patch.object(CPU_Engine, "get_hardware_info", return_value=fake_info):
            result = CPU_Engine.get_performance_evaluation()

        assert result["backend_type"] == "cpu"
        assert result["gpu_name"] == "CPU Only"
        assert result["cpu_model"] == "Erudium X1"
        assert result["cpu_performance_units"] == 16
        # cpu_score = 16/64*100 = 25 ; memory = 64/128*100 = 50
        assert result["cpu_score"] == 25.0
        assert result["memory_score"] == 50.0
        # bandwidth est. = 16 * 1.5 = 24 GB/s ; disk = 500/500*100 = 100
        assert result["memory_bandwidth_gbs"] == 24.0
        assert result["performance_breakdown"]["disk_score"] == 100.0
        # inference = 25*.4 + 50*.3 + 24*.2 + 100*.1 = 39.8 -> "Medium" (>= 25)
        assert result["global_inference_score"] == pytest.approx(39.8)
        assert result["global_inference_label"] == "Medium"
        assert result["gpu_score"] == 0.0
        assert result["unified_memory"] is False

    @pytest.mark.parametrize(
        "cores,expected_label",
        [
            (64, "Excellent"),  # 100*.4 + 50*.3 + 96*.2 + 100*.1 = 84.2 -> >= 70
            (1, "Medium"),  # 1.56*.4 + 50*.3 + 1.5*.2 + 100*.1 = 25.9 -> >= 25
        ],
    )
    def test_labels_track_core_count(self, cores, expected_label):
        fake_info = {
            "cpu": {"model": "X", "total_cores": cores, "architecture": "x86_64"},
            "memory": {"total_memory_gb": 64.0, "available_memory_gb": 32.0},
            "storage": {"total_gb": 1000.0, "available_gb": 500.0},
            "system": {"platform": "Linux"},
        }
        with patch.object(CPU_Engine, "get_hardware_info", return_value=fake_info):
            result = CPU_Engine.get_performance_evaluation()
        assert result["global_inference_label"] == expected_label

    def test_none_metrics_are_coalesced_to_zero(self):
        fake_info = {
            "cpu": {"model": "X", "total_cores": None, "architecture": "x86_64"},
            "memory": {"total_memory_gb": None, "available_memory_gb": None},
            "storage": {"total_gb": None, "available_gb": None},
            "system": {"platform": "Linux"},
        }
        with patch.object(CPU_Engine, "get_hardware_info", return_value=fake_info):
            result = CPU_Engine.get_performance_evaluation()
        assert result["cpu_performance_units"] == 1
        assert result["total_memory_gb"] == 0
        assert result["performance_breakdown"]["disk_score"] == 0.0

    def test_failure_returns_core_based_fallback(self):
        with patch.object(CPU_Engine, "get_hardware_info", side_effect=RuntimeError("boom")):
            result = CPU_Engine.get_performance_evaluation()
        assert result["backend_type"] == "cpu"
        assert result["global_inference_label"] == "Poor"
        assert result["cpu_performance_units"] >= 1
        assert result["total_memory_gb"] is None
        assert result["memory_score"] == 0.0

    def test_flat_hardware_data_delegates_to_evaluation(self):
        sentinel = {"backend_type": "cpu", "cpu_score": 12.0}
        with patch.object(CPU_Engine, "get_performance_evaluation", return_value=sentinel):
            assert CPU_Engine.get_flat_hardware_data() is sentinel
