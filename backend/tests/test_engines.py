"""Tests for engine hardware detection methods.

Tests get_flat_hardware_data() implementation across all engine backends
to ensure compatibility with HardwareProfile entity structure.
"""

import pytest
from src.engines.base_engine import BaseEngine
from src.engines.mlx_engine import MLX_Engine
from src.engines.cuda_engine import CUDA_Engine
from src.engines.cpu_engine import CPU_Engine
from src.entities.HardwareProfile import HardwareProfile
from tests._helpers import is_cuda_platform, is_mlx_platform


class TestEngineFlatHardwareData:
    """Test get_flat_hardware_data() returns compatible structure."""

    def test_get_engine_returns_valid_class(self):
        """Test that get_engine() returns a valid engine class."""
        engine = BaseEngine.get_engine()
        assert engine is not None
        assert engine in [MLX_Engine, CUDA_Engine, CPU_Engine]

    def test_flat_hardware_data_has_required_keys(self):
        """Test that get_flat_hardware_data() returns all required keys."""
        engine = BaseEngine.get_engine()
        data = engine.get_flat_hardware_data()

        # Required common fields
        required_keys = [
            "backend_type",
            "cpu_model",
            "total_memory_gb",
            "available_memory_gb",
            "disk_total_gb",
            "disk_available_gb",
            "global_inference_score",
            "global_inference_label",
            "cpu_score",
            "memory_score",
        ]

        for key in required_keys:
            assert key in data, f"Missing required key: {key}"

    def test_flat_hardware_data_backend_type_valid(self):
        """Test that backend_type is one of the valid values."""
        engine = BaseEngine.get_engine()
        data = engine.get_flat_hardware_data()

        assert data["backend_type"] in ["mlx", "cuda", "cpu"]

    def test_flat_hardware_data_scores_in_range(self):
        """Test that performance scores are in valid range."""
        engine = BaseEngine.get_engine()
        data = engine.get_flat_hardware_data()

        # Check score ranges (0-100)
        score_keys = [
            "global_inference_score",
            "cpu_score",
            "memory_score",
        ]

        for key in score_keys:
            score = data.get(key)
            if score is not None:
                assert 0 <= score <= 100, f"{key} out of range: {score}"

    @pytest.mark.skipif(
        not is_mlx_platform(), reason="MLX-specific test, only runs on Apple Silicon"
    )
    def test_mlx_specific_fields(self):
        """Test MLX backend returns MLX-specific fields."""
        data = MLX_Engine.get_flat_hardware_data()

        assert data["backend_type"] == "mlx"
        assert "mlx_chip_model" in data
        assert "mlx_gpu_cores" in data
        assert "mps_available" in data
        assert "neural_engine_tops" in data
        assert data.get("unified_memory") is True

    @pytest.mark.skipif(
        not is_cuda_platform(), reason="CUDA-specific test, only runs on NVIDIA systems"
    )
    def test_cuda_specific_fields(self):
        """Test CUDA backend returns CUDA-specific fields."""
        data = CUDA_Engine.get_flat_hardware_data()

        assert data["backend_type"] == "cuda"
        assert "cuda_cores" in data
        assert "cuda_version" in data
        assert "compute_capability" in data
        assert "vram_total_gb" in data
        assert "vram_available_gb" in data

    def test_hardware_profile_creation_from_flat_data(self):
        """Test that HardwareProfile can be instantiated from flat data."""
        engine = BaseEngine.get_engine()
        data = engine.get_flat_hardware_data()

        try:
            profile = HardwareProfile(**data)
            assert profile.backend_type == data["backend_type"]
            assert profile.cpu_model == data["cpu_model"]
            assert profile.global_inference_score == data["global_inference_score"]
        except TypeError as e:
            pytest.fail(f"HardwareProfile instantiation failed: {e}")

    def test_flat_data_types_correct(self):
        """Test that flat data returns correct types for each field."""
        engine = BaseEngine.get_engine()
        data = engine.get_flat_hardware_data()

        # Type checks
        assert isinstance(data["backend_type"], str)
        assert isinstance(data["cpu_model"], str)
        assert isinstance(data["total_memory_gb"], (int, float))
        assert isinstance(data["available_memory_gb"], (int, float))
        assert isinstance(data["global_inference_score"], (int, float))
        assert isinstance(data["global_inference_label"], str)


@pytest.mark.unit
class TestMlxMetalProbe:
    """``mps_available`` in the MLX hardware profile is MLX's own Metal probe
    (``mlx.core.metal.is_available``): no torch, and a failing probe is one
    WARNING and False, never an exception out of hardware detection."""

    @staticmethod
    def _fake_mlx(monkeypatch, is_available):
        import sys
        import types

        core = types.ModuleType("mlx.core")
        core.metal = types.SimpleNamespace(is_available=is_available)
        package = types.ModuleType("mlx")
        package.core = core
        monkeypatch.setitem(sys.modules, "mlx", package)
        monkeypatch.setitem(sys.modules, "mlx.core", core)

    def test_reports_what_mlx_says(self, monkeypatch):
        self._fake_mlx(monkeypatch, lambda: True)
        assert MLX_Engine._metal_available() is True
        self._fake_mlx(monkeypatch, lambda: False)
        assert MLX_Engine._metal_available() is False

    def test_a_failing_probe_is_a_warning_and_false(self, monkeypatch, caplog):
        def boom():
            raise RuntimeError("no Metal device")

        self._fake_mlx(monkeypatch, boom)
        with caplog.at_level("WARNING", logger="erudi"):
            assert MLX_Engine._metal_available() is False
        assert any("Metal availability probe failed" in r.message for r in caplog.records)


class TestServesArtifactUnder:
    """``serves_artifact_under`` names the weights folder a live child runs on (#762)."""

    @staticmethod
    def _engine(model_path):
        from src.engines.base_chat_server_engine import BaseChatServerEngine

        class _Engine(BaseChatServerEngine):
            _model = None if model_path is None else {"model_path": str(model_path)}

        return _Engine

    def test_gguf_file_inside_the_folder(self, tmp_path):
        link = tmp_path / "m" / "384"
        assert self._engine(link / "model.gguf").serves_artifact_under(link)

    def test_mlx_folder_itself(self, tmp_path):
        link = tmp_path / "m" / "384"
        assert self._engine(link).serves_artifact_under(link)

    def test_sibling_folder_with_a_common_prefix(self, tmp_path):
        loaded = self._engine(tmp_path / "m" / "384" / "x.gguf")
        assert not loaded.serves_artifact_under(tmp_path / "m" / "38")
        loaded = self._engine(tmp_path / "m" / "38" / "x.gguf")
        assert not loaded.serves_artifact_under(tmp_path / "m" / "384")

    def test_nothing_loaded(self, tmp_path):
        assert not self._engine(None).serves_artifact_under(tmp_path)

    def test_base_engine_serves_nothing(self, tmp_path):
        assert BaseEngine.serves_artifact_under(tmp_path) is False
