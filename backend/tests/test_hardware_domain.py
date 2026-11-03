"""Tests for the hardware domain HTTP layer and data access.

Complements `test_hardware.py` (pure service logic with mocked repository) by
exercising `Hardware_Repository` against the real test database and the three
REST endpoints through the FastAPI test client, including their error paths.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.core import config
from src.core.exceptions import DatabaseException, HardwareException
from src.domains.hardware.repository import Hardware_Repository
from src.domains.hardware.services import Hardware_Service
from src.engines.base_engine import BaseEngine
from src.entities.HardwareProfile import HardwareProfile


def _profile_data(**overrides) -> dict:
    data = dict(
        backend_type="cpu",
        cpu_model="Test CPU",
        total_memory_gb=16.0,
        available_memory_gb=8.0,
        disk_total_gb=512.0,
        disk_available_gb=256.0,
        global_inference_score=40.0,
        global_inference_label="Medium",
        cpu_score=50.0,
        memory_score=60.0,
        gpu_score=0.0,
        cpu_performance_units=8.0,
        architecture="x86_64",
        system_platform="Linux",
        performance_breakdown={
            "compute_score": 50.0,
            "memory_bandwidth_score": 30.0,
            "memory_capacity_score": 60.0,
            "cpu_performance_score": 50.0,
            "disk_score": 51.2,
        },
    )
    data.update(overrides)
    return data


class _FakeCpuEngineType:
    """Stands in for config.LLM_Engine: class-like object with __name__."""

    __name__ = "CPU_Engine"
    _flat_data = _profile_data()

    @classmethod
    def get_flat_hardware_data(cls):
        return dict(cls._flat_data)

    @classmethod
    def warm_up_accelerator(cls, duration):
        return True


# =====================================================================
# INTEGRATION - Hardware_Repository against the real database
# =====================================================================


@pytest.mark.integration
class TestHardwareRepository:
    def test_get_profile_returns_none_when_empty(self, test_db_session):
        repo = Hardware_Repository(test_db_session)
        assert repo.get_profile() is None

    def test_create_then_get_roundtrip(self, test_db_session):
        repo = Hardware_Repository(test_db_session)
        created = repo.create_profile(_profile_data())
        assert created.id is not None
        fetched = repo.get_profile()
        assert fetched.id == created.id
        assert fetched.backend_type == "cpu"
        assert fetched.cpu_model == "Test CPU"

    def test_get_profile_prunes_duplicates_keeping_most_recent(self, test_db_session):
        repo = Hardware_Repository(test_db_session)
        repo.create_profile(_profile_data(cpu_model="Old CPU"))
        newest = repo.create_profile(_profile_data(cpu_model="New CPU"))

        survivor = repo.get_profile()

        assert survivor.id == newest.id
        remaining = test_db_session.query(HardwareProfile).all()
        assert len(remaining) == 1

    def test_update_profile_applies_known_fields(self, test_db_session):
        repo = Hardware_Repository(test_db_session)
        profile = repo.create_profile(_profile_data())
        updated = repo.update_profile(
            profile, {"available_memory_gb": 4.5, "cpu_model": "Upgraded CPU"}
        )
        assert updated.available_memory_gb == 4.5
        assert updated.cpu_model == "Upgraded CPU"

    def test_update_profile_ignores_unknown_fields(self, test_db_session):
        repo = Hardware_Repository(test_db_session)
        profile = repo.create_profile(_profile_data())
        updated = repo.update_profile(profile, {"warp_drive": True})
        assert not hasattr(updated, "warp_drive")

    def test_delete_profile_removes_row(self, test_db_session):
        repo = Hardware_Repository(test_db_session)
        profile = repo.create_profile(_profile_data())
        repo.delete_profile(profile)
        assert repo.get_profile() is None

    def test_create_profile_invalid_column_raises_database_exception(self, test_db_session):
        repo = Hardware_Repository(test_db_session)
        with pytest.raises(DatabaseException):
            repo.create_profile({"no_such_column": 1})

    def test_get_profile_wraps_query_failure(self, test_db_session):
        repo = Hardware_Repository(test_db_session)

        def boom(*args, **kwargs):
            raise RuntimeError("connection lost")

        with patch.object(test_db_session, "query", side_effect=boom):
            with pytest.raises(DatabaseException):
                repo.get_profile()

    def test_update_profile_wraps_flush_failure(self, test_db_session):
        repo = Hardware_Repository(test_db_session)
        profile = repo.create_profile(_profile_data())
        with patch.object(test_db_session, "flush", side_effect=RuntimeError("disk full")):
            with pytest.raises(DatabaseException):
                repo.update_profile(profile, {"cpu_model": "X"})

    def test_delete_profile_wraps_failure(self, test_db_session):
        repo = Hardware_Repository(test_db_session)
        profile = repo.create_profile(_profile_data())
        with patch.object(test_db_session, "delete", side_effect=RuntimeError("locked")):
            with pytest.raises(DatabaseException):
                repo.delete_profile(profile)


# =====================================================================
# INTEGRATION - Hardware_Service paths not reachable with a mocked repo
# =====================================================================


@pytest.mark.integration
class TestHardwareServicePaths:
    def test_detect_hardware_without_engine_raises(self):
        service = Hardware_Service(repository=None)
        with patch.object(config, "LLM_Engine", None):
            with pytest.raises(HardwareException):
                service._detect_hardware()

    def test_warm_up_without_engine_returns_false(self):
        service = Hardware_Service(repository=None)
        with patch.object(config, "LLM_Engine", None):
            assert service.warm_up(1) is False

    def test_warm_up_delegates_to_engine(self):
        service = Hardware_Service(repository=None)
        with patch.object(config, "LLM_Engine", _FakeCpuEngineType):
            assert service.warm_up(1) is True

    def test_warm_up_engine_failure_raises_hardware_exception(self):
        class BrokenEngine:
            __name__ = "CPU_Engine"

            @classmethod
            def warm_up_accelerator(cls, duration):
                raise RuntimeError("thermal runaway")

        service = Hardware_Service(repository=None)
        with patch.object(config, "LLM_Engine", BrokenEngine):
            with pytest.raises(HardwareException):
                service.warm_up(1)

    @pytest.mark.parametrize(
        "score,label",
        [
            (85.0, "Excellent"),
            (65.0, "Good"),
            (45.0, "Fair"),
            (25.0, "Poor"),
            (5.0, "Weak"),
        ],
    )
    def test_label_thresholds(self, score, label):
        service = Hardware_Service(repository=None)
        assert service._get_label(score) == label

    def test_refresh_profile_creates_when_none(self, test_db_session):
        service = Hardware_Service(Hardware_Repository(test_db_session))
        with patch.object(config, "LLM_Engine", _FakeCpuEngineType):
            profile = service.refresh_profile()
        assert profile.backend_type == "cpu"

    def test_refresh_profile_updates_existing(self, test_db_session):
        repo = Hardware_Repository(test_db_session)
        existing = repo.create_profile(_profile_data(cpu_model="Stale CPU"))
        service = Hardware_Service(repo)
        with patch.object(config, "LLM_Engine", _FakeCpuEngineType):
            profile = service.refresh_profile()
        assert profile.id == existing.id
        assert profile.cpu_model == "Test CPU"

    def test_refresh_profile_failure_raises_hardware_exception(self, test_db_session):
        service = Hardware_Service(Hardware_Repository(test_db_session))
        with patch.object(config, "LLM_Engine", None):
            with pytest.raises(HardwareException):
                service.refresh_profile()

    def test_get_or_create_backend_mismatch_recreates(self, test_db_session):
        repo = Hardware_Repository(test_db_session)
        repo.create_profile(_profile_data(backend_type="mlx"))
        service = Hardware_Service(repo)
        with patch.object(config, "LLM_Engine", _FakeCpuEngineType):
            profile = service.get_or_create_profile()
        assert profile.backend_type == "cpu"
        assert len(test_db_session.query(HardwareProfile).all()) == 1


# =====================================================================
# INTEGRATION - REST endpoints through the test client
# =====================================================================


@pytest.mark.integration
class TestHardwareEndpoints:
    @pytest.fixture(autouse=True)
    def _fake_engine(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_Engine", _FakeCpuEngineType)

    def test_app_startup_returns_boosted_scores(self, client):
        resp = client.get("/erudi/hardware/app_startup")
        assert resp.status_code == 200
        body = resp.json()
        assert body["backend_type"] == "cpu"
        assert body["raw_inference_score"] == 40.0
        assert body["global_inference_score"] == 60.0  # raw + 20 boost
        assert body["global_inference_label"] == "Good"
        # Window derives from real usable memory (#199 Part 2), not the score tier:
        # CPU 16GB -> 50% usable = 8GB, less the 1.5GB reserve, / 0.6 GB-per-B = 10.8B
        assert body["recommended_param_min"] == 5.4
        assert body["recommended_param_max"] == 10.8

    def test_app_startup_hardware_error_maps_to_500(self, client):
        with patch.object(
            Hardware_Service,
            "get_or_create_profile",
            side_effect=HardwareException("no sensors", trace="t"),
        ):
            resp = client.get("/erudi/hardware/app_startup")
        assert resp.status_code == 500

    def test_app_startup_unexpected_error_maps_to_500(self, client):
        with patch.object(
            Hardware_Service,
            "get_or_create_profile",
            side_effect=RuntimeError("surprise"),
        ):
            resp = client.get("/erudi/hardware/app_startup")
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Internal server error"

    def test_detailed_cpu_backend(self, client):
        resp = client.get("/erudi/hardware/detailed")
        assert resp.status_code == 200
        body = resp.json()
        hw = body["hardware"]
        assert hw["backend_type"] == "cpu"
        assert hw["compute_units"] == 8
        assert hw["gpu_score"] == 0.0
        assert body["boosted_inference_score"] == 60.0
        assert body["performance_breakdown"]["disk_score"] == 51.2

    def test_detailed_cuda_backend(self, client, test_db_session):
        cuda_data = _profile_data(
            backend_type="cuda",
            gpu_name="RTX 4090",
            cuda_cores=16384,
            cuda_version="12.1",
            compute_capability="8.9",
            vram_total_gb=24.0,
            vram_available_gb=20.0,
            estimated_tflops=82.6,
            memory_bandwidth_gbs=1008.0,
            gpu_score=90.0,
        )

        class FakeCuda(_FakeCpuEngineType):
            __name__ = "CUDA_Engine"
            _flat_data = cuda_data

        with patch.object(config, "LLM_Engine", FakeCuda):
            resp = client.get("/erudi/hardware/detailed")
        assert resp.status_code == 200
        hw = resp.json()["hardware"]
        assert hw["backend_type"] == "cuda"
        assert hw["gpu_name"] == "RTX 4090"
        assert hw["cuda_cores"] == 16384
        assert hw["unified_memory"] is False

    def test_detailed_mlx_backend(self, client, test_db_session):
        mlx_data = _profile_data(
            backend_type="mlx",
            mlx_chip_model="Apple M3 Max",
            mlx_gpu_cores=40,
            mps_available=True,
            neural_engine_tops=35.0,
            estimated_tflops=28.0,
            memory_bandwidth_gbs=400.0,
            gpu_score=70.0,
        )

        class FakeMlx(_FakeCpuEngineType):
            __name__ = "MLX_Engine"
            _flat_data = mlx_data

        with patch.object(config, "LLM_Engine", FakeMlx):
            resp = client.get("/erudi/hardware/detailed")
        assert resp.status_code == 200
        hw = resp.json()["hardware"]
        assert hw["backend_type"] == "mlx"
        assert hw["mlx_chip_model"] == "Apple M3 Max"
        assert hw["unified_memory"] is True

    def test_detailed_error_maps_to_500(self, client):
        with patch.object(
            Hardware_Service,
            "get_or_create_profile",
            side_effect=DatabaseException("db down", trace="t"),
        ):
            resp = client.get("/erudi/hardware/detailed")
        assert resp.status_code == 500

    def test_detailed_unexpected_error_maps_to_500(self, client):
        with patch.object(
            Hardware_Service,
            "get_or_create_profile",
            side_effect=RuntimeError("surprise"),
        ):
            resp = client.get("/erudi/hardware/detailed")
        assert resp.status_code == 500

    def test_refresh_reports_backend(self, client):
        resp = client.post("/erudi/hardware/refresh")
        assert resp.status_code == 200
        body = resp.json()
        assert body["backend_type"] == "cpu"
        assert "refreshed" in body["message"]

    def test_refresh_error_maps_to_500(self, client):
        with patch.object(
            Hardware_Service,
            "refresh_profile",
            side_effect=HardwareException("no sensors", trace="t"),
        ):
            resp = client.post("/erudi/hardware/refresh")
        assert resp.status_code == 500

    def test_refresh_unexpected_error_maps_to_500(self, client):
        with patch.object(
            Hardware_Service, "refresh_profile", side_effect=RuntimeError("surprise")
        ):
            resp = client.post("/erudi/hardware/refresh")
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Internal server error"


# =====================================================================
# UNIT - the GPU contention status the conversation page polls
# =====================================================================

STATUS_PATH = "/erudi/hardware/gpu_contention/status"


def _status_app():
    from fastapi import FastAPI

    from src.domains.hardware.endpoints import router

    app = FastAPI()
    app.include_router(router, prefix="/erudi")
    return app


async def _get_status(app):
    import httpx

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        return await http.get(STATUS_PATH)


class _ContendedEngine(BaseEngine):
    """An engine whose loaded child another app squeezes off the card."""

    state = {
        "active": True,
        "evicted_bytes": 4672 * 1024**2,
        "device_free_bytes": 400 * 1024**2,
        "read_at": 1_700_000_000.5,
    }

    @classmethod
    def gpu_contention_state(cls):
        return cls.state


@pytest.mark.unit
class TestGpuContentionStatus:
    async def test_reports_the_cached_contention_of_the_loaded_child(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_Engine", _ContendedEngine)
        resp = await _get_status(_status_app())
        assert resp.status_code == 200
        assert resp.json() == _ContendedEngine.state

    async def test_an_engine_without_a_contention_state_is_unknown(self, monkeypatch):
        """CPU, MLX, Linux CUDA, no child loaded, a respawned child not read
        yet: no reading says neither contended nor over (``active: null``),
        so the conversation page keeps a notice it shows."""
        monkeypatch.setattr(config, "LLM_Engine", BaseEngine)
        resp = await _get_status(_status_app())
        assert resp.json() == {
            "active": None,
            "evicted_bytes": None,
            "device_free_bytes": None,
            "read_at": None,
        }

    async def test_no_engine_is_unknown(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_Engine", None)
        resp = await _get_status(_status_app())
        assert resp.json()["active"] is None

    async def test_the_real_cuda_engine_without_a_child_is_unknown(self, monkeypatch):
        from src.engines.cuda_engine import CUDA_Engine

        monkeypatch.setattr(config, "LLM_Engine", CUDA_Engine)
        monkeypatch.setattr(CUDA_Engine, "_model", None)
        resp = await _get_status(_status_app())
        assert resp.json()["active"] is None

    async def test_a_reading_that_contention_ended_is_false(self, monkeypatch):
        """Only a reading says the contention is over."""
        ended = {**_ContendedEngine.state, "active": False}
        monkeypatch.setattr(_ContendedEngine, "state", ended)
        monkeypatch.setattr(config, "LLM_Engine", _ContendedEngine)
        resp = await _get_status(_status_app())
        assert resp.json()["active"] is False

    async def test_answers_while_a_generation_holds_the_lock(self, monkeypatch):
        """The status only reads the cached state: it never waits for the
        generation lock (an answer may stream for minutes under a hog)."""
        import asyncio

        monkeypatch.setattr(config, "LLM_Engine", _ContendedEngine)
        app = _status_app()
        async with _ContendedEngine._acquire_generation_lock():
            resp = await asyncio.wait_for(_get_status(app), timeout=2.0)
        assert resp.json()["active"] is True

    def test_its_access_log_is_a_polling_line(self):
        from src.core.api import _is_polling_path

        assert _is_polling_path(STATUS_PATH)
