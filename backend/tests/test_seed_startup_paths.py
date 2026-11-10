"""Tests for the boot-time populate/cleanup/backfill paths of `src.database.seed`.

The catalog discovery pipeline is covered by `test_model_catalog.py` and the
snapshot reconcile by `test_catalog_snapshot_reconcile.py`. This file pins the
remaining startup spans: connectivity probe, offline fallback loading and
seeding, base-catalog assembly with metadata fallbacks, interrupted-job
cleanup (download / KB / orphaned dirs), hardware and startup-variable
initializers, the populate facade, the wire-tools backfill, the destructive
dev reset, and the snapshot helpers in `src.database.catalog_snapshot`.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.core import config
from src.core.exceptions import FileSystemException, HuggingFaceAPIException
from src.database import catalog_snapshot as snapshot_mod
from src.database import core as db_core
from src.database import seed as seed_mod
from src.database.seed import (
    Database_Seeder,
    Hardware_Initializer,
    Job_Cleanup_Service,
    Model_Config,
    Model_Seeder,
    Search_Config,
    Startup_Initializer,
    _safetensors_total,
    load_base_models_fallback,
)
from src.entities.DownloadJob import DownloadJobModel
from src.entities.HardwareProfile import HardwareProfile
from src.domains.hardware.services import PROFILING_LOGIC_VERSION
from src.entities.KBJob import KBJobModel
from src.entities.KnowledgeBase import KnowledgeBase
from src.entities.Llm import Llm
from src.entities.StartupVariables import StartupVariables


class _AcceptingEngine:
    """Engine whose integrity gate (#88) accepts the artifact on disk."""

    __name__ = "CPU_Engine"
    FORMAT_TAG = "gguf"

    @classmethod
    def validate_local_artifact(cls, path):
        return None


class _RejectingEngine:
    """Engine whose integrity gate rejects the artifact on disk."""

    __name__ = "CPU_Engine"
    FORMAT_TAG = "gguf"

    @classmethod
    def validate_local_artifact(cls, path):
        raise RuntimeError("incomplete artifact")


class _FakeEngineType:
    FORMAT_TAG = "gguf"

    @classmethod
    def get_flat_hardware_data(cls):
        return {
            "backend_type": "cpu",
            "cpu_model": "Seed CPU",
            "total_memory_gb": 16.0,
            "available_memory_gb": 8.0,
            "disk_total_gb": 512.0,
            "disk_available_gb": 256.0,
            "global_inference_score": 40.0,
            "global_inference_label": "Medium",
            "cpu_score": 50.0,
            "memory_score": 60.0,
            "gpu_score": 0.0,
            "system_platform": "Linux",
            "performance_breakdown": {},
        }


# Hardware_Service derives the current backend from the engine's class name
# ("CPU_Engine" -> "cpu") and compares it to the profile's backend_type, so the
# fake has to answer "CPU_Engine" or it looks like a different machine on every
# call. Assigning __name__ inside the class body does NOT do that: type.__name__
# is a data descriptor on the metaclass, so it wins over the class __dict__ and
# the attribute still reads "_FakeEngineType". Assigning after the class runs
# goes through that descriptor's setter, which is what actually renames it.
_FakeEngineType.__name__ = "CPU_Engine"


# =====================================================================
# UNIT - module helpers
# =====================================================================


@pytest.mark.unit
class TestModuleHelpers:
    def test_load_base_models_fallback_reads_bundled_json(self):
        models = load_base_models_fallback()
        assert models, "bundled fallback must not be empty"
        assert {"name", "link", "type"} <= set(models[0])

    def test_load_base_models_fallback_missing_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "ROOT_DIR", tmp_path)
        with pytest.raises(FileSystemException, match="not found"):
            load_base_models_fallback()

    def test_load_base_models_fallback_corrupt_json(self, monkeypatch, tmp_path):
        target = tmp_path / "src" / "database"
        target.mkdir(parents=True)
        (target / "base_models_fallback.json").write_text("{not json")
        monkeypatch.setattr(config, "ROOT_DIR", tmp_path)
        with pytest.raises(FileSystemException, match="parse"):
            load_base_models_fallback()

    def test_safetensors_total_object_form(self):
        info = SimpleNamespace(safetensors=SimpleNamespace(total=1_500_000_000))
        assert _safetensors_total(info) == 1_500_000_000

    def test_safetensors_total_dict_form(self):
        info = SimpleNamespace(safetensors={"total": 7})
        assert _safetensors_total(info) == 7

    def test_safetensors_total_absent(self):
        assert _safetensors_total(SimpleNamespace(safetensors=None)) is None
        assert _safetensors_total(SimpleNamespace(safetensors={})) is None

    def test_model_config_validation(self):
        with pytest.raises(ValueError, match="Invalid model config"):
            Model_Config(name="", link="a/b", model_type="x")

    def test_search_config_validation(self):
        with pytest.raises(ValueError, match="Invalid search config"):
            Search_Config("term", "", 7.0)
        with pytest.raises(ValueError, match="Invalid param size"):
            Search_Config("term", "x", 0.0)


# =====================================================================
# UNIT - instruct sibling preference (#122)
# =====================================================================


@pytest.mark.unit
class TestPreferInstructSiblings:
    def _mc(self, link):
        return Model_Config(link.split("/")[-1], link, "gemma")

    def test_pretrain_dropped_when_instruct_sibling_exists(self):
        seeder = Model_Seeder(db=None)
        out = seeder._prefer_instruct_siblings(
            [self._mc("google/gemma-2-9b"), self._mc("google/gemma-2-9b-it")]
        )
        assert [m.link for m in out] == ["google/gemma-2-9b-it"]

    def test_lone_release_without_suffix_is_kept(self):
        seeder = Model_Seeder(db=None)
        out = seeder._prefer_instruct_siblings([self._mc("deepseek-ai/DeepSeek-V3")])
        assert [m.link for m in out] == ["deepseek-ai/DeepSeek-V3"]

    def test_families_are_independent(self):
        seeder = Model_Seeder(db=None)
        out = seeder._prefer_instruct_siblings(
            [
                self._mc("google/gemma-2-9b"),
                self._mc("google/gemma-2-9b-it"),
                self._mc("mistralai/Mistral-7B-v0.1"),
            ]
        )
        assert {m.link for m in out} == {
            "google/gemma-2-9b-it",
            "mistralai/Mistral-7B-v0.1",
        }


# =====================================================================
# UNIT - build_base_models assembly and fallbacks
# =====================================================================


@pytest.mark.unit
class TestBuildBaseModels:
    ORGS = [("google", "gemma", "Gemma")]

    def _seeder_with_candidates(self, monkeypatch, candidates):
        seeder = Model_Seeder(db=None, hf_api=MagicMock())
        monkeypatch.setattr(seeder, "discover_instruct_models", lambda org, mtype: candidates)
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngineType)
        return seeder

    def _mc(self, link):
        return Model_Config(link.split("/")[-1], link, "gemma")

    def test_resolved_quants_are_built_and_deduped(self, monkeypatch):
        candidates = [self._mc("google/gemma-2-9b-it"), self._mc("google/gemma-2-2b-it")]
        seeder = self._seeder_with_candidates(monkeypatch, candidates)
        # Both bases resolve to the SAME quant repo -> only one row (#122)
        monkeypatch.setattr(seed_mod, "resolve_quant", lambda link, tag, api: "quanter/gemma-gguf")
        built = []

        def fake_create(mc, quant):
            built.append((mc.link, quant))
            return Llm(name=mc.name, local=0, link=quant, type=mc.model_type)

        monkeypatch.setattr(seeder, "_create_base_llm", fake_create)
        out = seeder.build_base_models(self.ORGS)
        assert len(out) == 1
        assert built == [("google/gemma-2-9b-it", "quanter/gemma-gguf")]

    def test_unresolvable_base_is_skipped(self, monkeypatch):
        seeder = self._seeder_with_candidates(monkeypatch, [self._mc("google/gemma-2-9b-it")])
        monkeypatch.setattr(seed_mod, "resolve_quant", lambda link, tag, api: None)
        assert seeder.build_base_models(self.ORGS) == []

    def test_resolver_crash_is_contained(self, monkeypatch):
        seeder = self._seeder_with_candidates(monkeypatch, [self._mc("google/gemma-2-9b-it")])

        def broken_resolver(link, tag, api):
            raise RuntimeError("HF 500")

        monkeypatch.setattr(seed_mod, "resolve_quant", broken_resolver)
        assert seeder.build_base_models(self.ORGS) == []

    def test_hf_metadata_failure_falls_back(self, monkeypatch):
        seeder = self._seeder_with_candidates(monkeypatch, [self._mc("google/gemma-2-9b-it")])
        monkeypatch.setattr(seed_mod, "resolve_quant", lambda link, tag, api: "quanter/gemma-gguf")

        def broken_create(mc, quant):
            raise HuggingFaceAPIException("metadata gone", trace="t")

        fallback_row = Llm(name="Gemma", local=0, link="quanter/gemma-gguf", type="gemma")
        monkeypatch.setattr(seeder, "_create_base_llm", broken_create)
        monkeypatch.setattr(seeder, "_create_base_llm_fallback", lambda mc, quant: fallback_row)
        out = seeder.build_base_models(self.ORGS)
        assert out == [fallback_row]

    def test_fallback_failure_drops_only_that_model(self, monkeypatch):
        seeder = self._seeder_with_candidates(monkeypatch, [self._mc("google/gemma-2-9b-it")])
        monkeypatch.setattr(seed_mod, "resolve_quant", lambda link, tag, api: "quanter/gemma-gguf")

        def broken_create(mc, quant):
            raise HuggingFaceAPIException("metadata gone", trace="t")

        def broken_fallback(mc, quant):
            raise RuntimeError("size probe failed")

        monkeypatch.setattr(seeder, "_create_base_llm", broken_create)
        monkeypatch.setattr(seeder, "_create_base_llm_fallback", broken_fallback)
        assert seeder.build_base_models(self.ORGS) == []

    def test_generic_build_failure_is_contained(self, monkeypatch):
        seeder = self._seeder_with_candidates(monkeypatch, [self._mc("google/gemma-2-9b-it")])
        monkeypatch.setattr(seed_mod, "resolve_quant", lambda link, tag, api: "quanter/gemma-gguf")

        def broken_create(mc, quant):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(seeder, "_create_base_llm", broken_create)
        assert seeder.build_base_models(self.ORGS) == []

    def test_create_base_llm_and_fallback_build_rows(self, monkeypatch):
        class _Size:
            size_bytes = 5_100_000_000

            def to_string(self):
                return "5.1 GB"

        monkeypatch.setattr(
            seed_mod, "get_disk_size_after_quant", lambda link, hf_api=None: _Size()
        )
        api = MagicMock()
        api.model_info.return_value = SimpleNamespace(
            id="google/gemma-2-9b-it",
            author="google",
            downloads=1000,
            likes=100,
            tags=[],
            library_name="transformers",
        )
        monkeypatch.setattr(seed_mod, "format_model_info_metadata", lambda info, size, q: "meta")
        seeder = Model_Seeder(db=None, hf_api=api)
        mc = Model_Config(
            "gemma-2-9b-it",
            "google/gemma-2-9b-it",
            "gemma",
            safetensors_total=9_000_000_000,
            category="general",
        )

        row = seeder._create_base_llm(mc, "quanter/gemma-gguf")
        assert row.link == "quanter/gemma-gguf"
        assert row.is_base is True
        assert row.quantized is True
        assert row.param_size == 9.0
        assert row.supports_tools is None

        fb = seeder._create_base_llm_fallback(mc, "quanter/gemma-gguf")
        assert fb.link == "quanter/gemma-gguf"
        assert "5.1 GB" in fb.model_metadata
        assert fb.is_base is True


# =====================================================================
# INTEGRATION - offline seeding
# =====================================================================


@pytest.mark.integration
class TestOfflineSeeding:
    FALLBACK = [
        {
            "name": "Test Gemma",
            "link": "quanter/test-gemma-gguf",
            "type": "gemma",
            "param_size": 1.0,
            "model_metadata": "Size: 1 GB",
            "quantized": True,
        }
    ]

    def test_seed_base_models_offline_adds_rows(self, test_db_session, monkeypatch):
        monkeypatch.setattr(seed_mod, "load_base_models_fallback", lambda: list(self.FALLBACK))
        seeder = Model_Seeder(test_db_session, offline_mode=True)
        assert seeder.seed_base_models_offline() == 1
        row = test_db_session.query(Llm).filter(Llm.link == "quanter/test-gemma-gguf").one()
        assert row.is_base is True
        assert row.local == 0

    def test_seed_base_models_offline_skips_existing(self, test_db_session, monkeypatch):
        monkeypatch.setattr(seed_mod, "load_base_models_fallback", lambda: list(self.FALLBACK))
        seeder = Model_Seeder(test_db_session, offline_mode=True)
        assert seeder.seed_base_models_offline() == 1
        assert seeder.seed_base_models_offline() == 0  # idempotent

    def test_seed_base_models_offline_survives_bad_row(self, test_db_session, monkeypatch):
        # Missing 'type' fails inside the guarded per-row block, not the loop.
        bad = {"name": "Broken", "link": "x/broken"}
        monkeypatch.setattr(
            seed_mod,
            "load_base_models_fallback",
            lambda: [dict(self.FALLBACK[0]), bad],
        )
        seeder = Model_Seeder(test_db_session, offline_mode=True)
        assert seeder.seed_base_models_offline() == 1

    def test_seed_initial_catalog_prefers_snapshot(self, test_db_session, monkeypatch):
        seeder = Model_Seeder(test_db_session, offline_mode=True)
        monkeypatch.setattr(seeder, "seed_from_snapshot", lambda: 42)
        assert seeder.seed_initial_catalog() == 42

    def test_seed_initial_catalog_falls_back_to_offline_json(self, test_db_session, monkeypatch):
        seeder = Model_Seeder(test_db_session, offline_mode=True)
        monkeypatch.setattr(seeder, "seed_from_snapshot", lambda: 0)
        monkeypatch.setattr(seeder, "seed_base_models_offline", lambda: 3)
        assert seeder.seed_initial_catalog() == 3

    def test_seed_initial_catalog_never_raises(self, test_db_session, monkeypatch):
        seeder = Model_Seeder(test_db_session, offline_mode=True)

        def boom():
            raise RuntimeError("snapshot corrupted")

        monkeypatch.setattr(seeder, "seed_from_snapshot", boom)
        monkeypatch.setattr(seeder, "seed_base_models_offline", boom)
        assert seeder.seed_initial_catalog() == 0

    def test_seed_from_snapshot_without_format_tag(self, test_db_session, monkeypatch):
        monkeypatch.setattr(config, "LLM_Engine", None)
        seeder = Model_Seeder(test_db_session)
        assert seeder.seed_from_snapshot() == 0

    def test_seed_from_snapshot_with_entries(self, test_db_session, monkeypatch):
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngineType)
        entries = [{"name": "Snap Model", "link": "quanter/snap-gguf", "type": "gemma"}]
        monkeypatch.setattr(snapshot_mod, "load_catalog_snapshot", lambda tag: entries)
        seeder = Model_Seeder(test_db_session)
        assert seeder.seed_from_snapshot() == 1
        assert test_db_session.query(Llm).filter(Llm.link == "quanter/snap-gguf").count() == 1

    def test_seed_from_snapshot_empty_snapshot(self, test_db_session, monkeypatch):
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngineType)
        monkeypatch.setattr(snapshot_mod, "load_catalog_snapshot", lambda tag: [])
        assert Model_Seeder(test_db_session).seed_from_snapshot() == 0


# =====================================================================
# INTEGRATION - interrupted job cleanup
# =====================================================================


@pytest.mark.integration
class TestJobCleanup:
    def _download_job(self, db, llm_id=None, temp_dir="", status="running"):
        job = DownloadJobModel(
            remote_model_id="9",
            local_model_id=llm_id,
            remote_model_link="org/model",
            temp_local_model_link=temp_dir,
            status=status,
        )
        db.add(job)
        db.flush()
        return job

    def test_download_cleanup_marks_failed_and_removes_files(
        self, test_db_session, tmp_path, monkeypatch
    ):
        # Incomplete artifact: the engine's integrity gate rejects it, so the
        # historical delete-and-mark-failed behavior still applies (#314).
        monkeypatch.setattr(config, "LLM_Engine", _RejectingEngine)
        model_dir = tmp_path / "model-42"
        model_dir.mkdir()
        temp_dir = tmp_path / "temp_42"
        temp_dir.mkdir()
        llm = Llm(name="Partial", local=2, link=str(model_dir), type="x")
        test_db_session.add(llm)
        test_db_session.flush()
        job = self._download_job(test_db_session, llm_id=llm.id, temp_dir=str(temp_dir))

        count = Job_Cleanup_Service(test_db_session)._cleanup_download_jobs()

        assert count == 1
        assert job.status == "failed"
        assert "interrupted" in job.error_message
        assert job.temp_local_model_link == ""
        assert not model_dir.exists()
        assert not temp_dir.exists()

    def test_download_cleanup_ignores_finished_jobs(self, test_db_session):
        self._download_job(test_db_session, status="completed")
        assert Job_Cleanup_Service(test_db_session)._cleanup_download_jobs() == 0

    # ---- #314: a complete artifact must survive an unfinished job row ----
    # #291 strands the job at 'running' AFTER the transfer fully succeeded, so
    # deleting on the strength of the status alone destroys a valid multi-GB
    # download. Cleanup must validate the artifact before removing anything.

    def test_download_cleanup_preserves_complete_artifact(
        self, test_db_session, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(config, "LLM_Engine", _AcceptingEngine)
        model_dir = tmp_path / "model-748"
        model_dir.mkdir()
        (model_dir / "model.Q4_K_M.gguf").write_bytes(b"GGUF" + b"\x00" * 64)
        temp_dir = tmp_path / "temp_748"
        temp_dir.mkdir()
        llm = Llm(name="Qwen3 14B", local=2, link=str(model_dir), type="x")
        test_db_session.add(llm)
        test_db_session.flush()
        job = self._download_job(test_db_session, llm_id=llm.id, temp_dir=str(temp_dir))

        count = Job_Cleanup_Service(test_db_session)._cleanup_download_jobs()

        assert count == 1
        # The artifact and its DB row survive, finalized rather than destroyed.
        assert model_dir.exists()
        assert (model_dir / "model.Q4_K_M.gguf").exists()
        assert test_db_session.query(Llm).filter(Llm.id == llm.id).first() is not None
        assert llm.local == 1
        assert job.status == "completed"
        assert job.progress == 100.0
        assert job.error_message is None
        # Staging space is still reclaimed: its contents were already moved.
        assert not temp_dir.exists()
        assert job.temp_local_model_link == ""

    def test_download_cleanup_preserves_when_size_covers_total_bytes(
        self, test_db_session, tmp_path, monkeypatch
    ):
        # No engine bound: fall back to the recorded total_bytes rather than
        # deleting a full artifact just because the engine was unavailable.
        monkeypatch.setattr(config, "LLM_Engine", None)
        model_dir = tmp_path / "model-900"
        model_dir.mkdir()
        (model_dir / "weights.gguf").write_bytes(b"x" * 4096)
        llm = Llm(name="Sized", local=2, link=str(model_dir), type="x")
        test_db_session.add(llm)
        test_db_session.flush()
        job = self._download_job(test_db_session, llm_id=llm.id)
        job.total_bytes = 4096

        assert Job_Cleanup_Service(test_db_session)._cleanup_download_jobs() == 1

        assert model_dir.exists()
        assert llm.local == 1
        assert job.status == "completed"

    def test_download_cleanup_removes_short_artifact_against_total_bytes(
        self, test_db_session, tmp_path, monkeypatch
    ):
        # Same path, but the transfer really was truncated: still deleted.
        monkeypatch.setattr(config, "LLM_Engine", None)
        model_dir = tmp_path / "model-901"
        model_dir.mkdir()
        (model_dir / "weights.gguf").write_bytes(b"x" * 10)
        llm = Llm(name="Truncated", local=2, link=str(model_dir), type="x")
        test_db_session.add(llm)
        test_db_session.flush()
        job = self._download_job(test_db_session, llm_id=llm.id)
        job.total_bytes = 4096

        assert Job_Cleanup_Service(test_db_session)._cleanup_download_jobs() == 1

        assert not model_dir.exists()
        assert job.status == "failed"

    def test_download_cleanup_removes_nearly_complete_artifact(
        self, test_db_session, tmp_path, monkeypatch
    ):
        # The blind spot of the two pins above (#316): they truncate to a
        # fraction of a percent, so ANY threshold drift still deletes them. A
        # transfer cut at ~95% is the case that separates a byte-exact
        # comparison from one that has drifted upward, and it is also the
        # realistic one -- interrupted downloads die near the end, not at byte 10.
        monkeypatch.setattr(config, "LLM_Engine", None)
        model_dir = tmp_path / "model-902"
        model_dir.mkdir()
        (model_dir / "weights.gguf").write_bytes(b"x" * 3900)  # 95.2% of 4096
        llm = Llm(name="NearlyThere", local=2, link=str(model_dir), type="x")
        test_db_session.add(llm)
        test_db_session.flush()
        job = self._download_job(test_db_session, llm_id=llm.id)
        job.total_bytes = 4096

        assert Job_Cleanup_Service(test_db_session)._cleanup_download_jobs() == 1

        assert not model_dir.exists()
        assert job.status == "failed"

    def test_download_cleanup_threshold_ignores_the_display_gb_divisor(
        self, test_db_session, tmp_path, monkeypatch
    ):
        # The completeness threshold must be a property of the BYTES, not of
        # whatever "a GB" currently means on screen (#316). Pinned by making the
        # display helper answer in decimal GB: a footprint read through it and
        # scaled back by 1024**3 inflates 7.4%, which is enough to mark this
        # 95%-truncated artifact "complete", keep it, and flip it to local=1 --
        # a broken model presented as installed, never cleaned up again.
        from src.utils.hf_model_metadata import measure_dir_size_bytes

        monkeypatch.setattr(config, "LLM_Engine", None)
        monkeypatch.setattr(
            seed_mod,
            "measure_dir_size_gb",
            lambda path: measure_dir_size_bytes(path) / 1_000_000_000,
        )
        model_dir = tmp_path / "model-903"
        model_dir.mkdir()
        (model_dir / "weights.gguf").write_bytes(b"x" * 3900)
        llm = Llm(name="DivisorProof", local=2, link=str(model_dir), type="x")
        test_db_session.add(llm)
        test_db_session.flush()
        job = self._download_job(test_db_session, llm_id=llm.id)
        job.total_bytes = 4096

        assert Job_Cleanup_Service(test_db_session)._cleanup_download_jobs() == 1

        assert not model_dir.exists()
        assert llm.local != 1
        assert job.status == "failed"

    # ---- #314 end to end against the REAL engine validator, not a double ----
    # The pins above use engine doubles, which prove the branching but not that
    # a real GGUF artifact is actually judged complete. This replays the exact
    # shape from the 2.0.0 RC post-mortem (job 40 / llm 748: status 'running',
    # progress 100, complete artifact on disk) through CPU_Engine's real
    # validate_local_artifact, which is what runs in production.

    def test_real_engine_keeps_a_complete_gguf_left_running(
        self, test_db_session, tmp_path, monkeypatch
    ):
        from src.engines.cpu_engine import CPU_Engine

        monkeypatch.setattr(config, "LLM_Engine", CPU_Engine)
        model_dir = tmp_path / "748"
        model_dir.mkdir()
        # A real GGUF container: the magic the integrity gate actually checks.
        (model_dir / "Qwen3-14B-Q4_K_M.gguf").write_bytes(b"GGUF" + b"\x00" * 4096)
        (model_dir / "config.json").write_text("{}")
        llm = Llm(name="Qwen3 14B", local=2, link=str(model_dir), type="qwen")
        test_db_session.add(llm)
        test_db_session.flush()
        job = self._download_job(test_db_session, llm_id=llm.id)
        job.status = "running"
        job.progress = 100.0

        Job_Cleanup_Service(test_db_session)._cleanup_download_jobs()

        assert (model_dir / "Qwen3-14B-Q4_K_M.gguf").exists(), "the 9 GB artifact was destroyed"
        assert test_db_session.query(Llm).filter(Llm.id == llm.id).first() is not None
        assert llm.local == 1
        assert job.status == "completed"

    def test_real_engine_still_deletes_a_corrupt_gguf_left_running(
        self, test_db_session, tmp_path, monkeypatch
    ):
        # The guard must not become "never delete anything": a truncated file
        # with no GGUF magic is exactly what the cleanup exists to reclaim.
        from src.engines.cpu_engine import CPU_Engine

        monkeypatch.setattr(config, "LLM_Engine", CPU_Engine)
        model_dir = tmp_path / "749"
        model_dir.mkdir()
        (model_dir / "partial.gguf").write_bytes(b"NOTG" + b"\x00" * 16)
        llm = Llm(name="Broken", local=2, link=str(model_dir), type="qwen")
        test_db_session.add(llm)
        test_db_session.flush()
        job = self._download_job(test_db_session, llm_id=llm.id)

        Job_Cleanup_Service(test_db_session)._cleanup_download_jobs()

        assert not model_dir.exists()
        assert job.status == "failed"

    def test_download_cleanup_logs_the_reclaimed_size_on_delete(
        self, test_db_session, tmp_path, monkeypatch, caplog
    ):
        # A multi-GB delete must never be silent again (#314).
        monkeypatch.setattr(config, "LLM_Engine", _RejectingEngine)
        model_dir = tmp_path / "model-902"
        model_dir.mkdir()
        (model_dir / "partial.bin").write_bytes(b"x" * 2048)
        llm = Llm(name="Debris", local=2, link=str(model_dir), type="x")
        test_db_session.add(llm)
        test_db_session.flush()
        self._download_job(test_db_session, llm_id=llm.id)

        with caplog.at_level("INFO"):
            Job_Cleanup_Service(test_db_session)._cleanup_download_jobs()

        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "removing incomplete model" in messages
        assert str(model_dir) in messages
        assert "reclaiming" in messages

    def test_download_cleanup_reports_the_staging_directory_too(
        self, test_db_session, tmp_path, monkeypatch, caplog
    ):
        """On a truncated download almost all the bytes are in staging.

        Killing the app at 26% of a 4.7 GB model left an EMPTY final directory
        and 1.28 GB of staging. The delete line measured only the final one, so
        the log announced "reclaiming ~0.00 GB" and the real 1.28 GB went
        without a word -- the opposite of what the comment above it asks for.
        """
        monkeypatch.setattr(config, "LLM_Engine", _RejectingEngine)
        final_dir = tmp_path / "model-903"
        final_dir.mkdir()  # left empty, exactly like the observed case
        staging = tmp_path / "temp_903"
        staging.mkdir()
        (staging / "shard.gguf").write_bytes(b"x" * 4096)

        llm = Llm(name="Truncated", local=2, link=str(final_dir), type="x")
        test_db_session.add(llm)
        test_db_session.flush()
        self._download_job(test_db_session, llm_id=llm.id, temp_dir=str(staging))

        with caplog.at_level("INFO"):
            Job_Cleanup_Service(test_db_session)._cleanup_download_jobs()

        messages = " ".join(r.getMessage() for r in caplog.records)
        assert str(staging) in messages, "the staging path must be named"
        assert "4096 bytes" in messages, "and the bytes it actually freed"
        assert not staging.exists()

    def _kb_job(self, db, base_id, new_id, kb_id, status="running"):
        job = KBJobModel(base_model_id=base_id, new_model_id=new_id, kb_id=kb_id, status=status)
        db.add(job)
        db.flush()
        return job

    def test_kb_creation_cleanup_rolls_back_assistant_and_kb(self, test_db_session, mock_llm):
        kb = KnowledgeBase()
        test_db_session.add(kb)
        test_db_session.flush()
        assistant = Llm(name="Assistant", local=1, link="/a", type="x", kb_id=kb.id)
        test_db_session.add(assistant)
        test_db_session.flush()
        job = self._kb_job(test_db_session, mock_llm.id, assistant.id, kb.id)

        count = Job_Cleanup_Service(test_db_session)._cleanup_kb_jobs()

        assert count == 1
        assert job.status == "failed"
        assert "creation interrupted" in job.error_message
        assert test_db_session.query(Llm).filter(Llm.id == assistant.id).count() == 0
        assert test_db_session.query(KnowledgeBase).filter(KnowledgeBase.id == kb.id).count() == 0

    def test_kb_update_cleanup_keeps_existing_kb(self, test_db_session, mock_llm_with_kb):
        llm, kb = mock_llm_with_kb
        job = self._kb_job(test_db_session, llm.id, llm.id, kb.id)  # update: new == base

        count = Job_Cleanup_Service(test_db_session)._cleanup_kb_jobs()

        assert count == 1
        assert job.status == "failed"
        assert "update interrupted" in job.error_message
        assert test_db_session.query(Llm).filter(Llm.id == llm.id).count() == 1
        assert test_db_session.query(KnowledgeBase).filter(KnowledgeBase.id == kb.id).count() == 1

    def test_kb_cleanup_turns_documents_left_indexing_into_failed(
        self, test_db_session, mock_llm_with_kb
    ):
        """A quit mid-file leaves a row `indexing` with no chunks. `failed` is
        what lets the file be added again (dedup skips every other status)."""
        from src.entities.KnowledgeDocument import KnowledgeDocument

        llm, kb = mock_llm_with_kb
        self._kb_job(test_db_session, llm.id, llm.id, kb.id)
        for name, status in (("cut.docx", "indexing"), ("done.txt", "active")):
            test_db_session.add(
                KnowledgeDocument(
                    kb_id=kb.id,
                    name=name,
                    content_hash_sha256=name.ljust(64, "0"),
                    size_bytes=1,
                    status=status,
                )
            )
        test_db_session.flush()

        Job_Cleanup_Service(test_db_session)._cleanup_kb_jobs()

        statuses = {
            d.name: d.status
            for d in test_db_session.query(KnowledgeDocument).filter_by(kb_id=kb.id)
        }
        assert statuses["cut.docx"] == "failed"
        assert statuses["done.txt"] == "active"

    def test_orphaned_models_cleanup(self, test_db_session, monkeypatch, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        monkeypatch.setattr(config, "LLM_DIR", models_dir)

        valid = Llm(name="Valid", local=1, link=str(models_dir / "1"), type="x")
        test_db_session.add(valid)
        test_db_session.flush()

        (models_dir / str(valid.id)).mkdir()  # belongs to a local row
        (models_dir / "999999").mkdir()  # orphan
        (models_dir / "temp_123").mkdir()  # interrupted download temp
        (models_dir / "stray.txt").write_text("x")  # non-dir: ignored

        count = Job_Cleanup_Service(test_db_session)._cleanup_orphaned_models()

        assert count == 2
        assert (models_dir / str(valid.id)).exists()
        assert not (models_dir / "999999").exists()
        assert not (models_dir / "temp_123").exists()

    def test_orphaned_models_cleanup_missing_dir_is_noop(
        self, test_db_session, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(config, "LLM_DIR", tmp_path / "nope")
        assert Job_Cleanup_Service(test_db_session)._cleanup_orphaned_models() == 0

    def test_cleanup_all_aggregates_counts(self, test_db_session, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "LLM_DIR", tmp_path / "models-none")
        service = Job_Cleanup_Service(test_db_session)
        counts = service.cleanup_all_unfinished_jobs()
        assert counts == {"download": 0, "kb": 0, "orphaned": 0}


# =====================================================================
# INTEGRATION - hardware / startup initializers
# =====================================================================


@pytest.mark.integration
class TestInitializers:
    def test_hardware_initializes_once(self, test_db_session, monkeypatch):
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngineType)
        init = Hardware_Initializer(test_db_session)
        assert init.initialize_if_needed() is True
        assert init.initialize_if_needed() is False  # second boot: cached

    def test_hardware_reprofiles_when_the_profiling_logic_moved_on(
        self, test_db_session, monkeypatch
    ):
        """An upgrade that fixes profiling must reach machines already profiled.

        Startup used to return the moment a row existed, so #365's corrected
        bandwidth never reached anyone who had already launched the app -- the
        only way to get it was Clear All Data, which also wipes their models,
        conversations and knowledge bases.
        """
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngineType)
        init = Hardware_Initializer(test_db_session)
        assert init.initialize_if_needed() is True

        # Simulate a row written by an older build: stale numbers, stale stamp.
        stored = test_db_session.query(HardwareProfile).first()
        stored.profiling_version = None
        stored.cpu_model = "Stale CPU"
        test_db_session.commit()
        stale_id = stored.id

        assert init.initialize_if_needed() is True  # re-profiled, not skipped
        refreshed = test_db_session.query(HardwareProfile).first()
        assert refreshed.id != stale_id
        assert refreshed.cpu_model == "Seed CPU"
        assert refreshed.profiling_version == PROFILING_LOGIC_VERSION

        # And it settles: the next boot is a plain cache hit again.
        assert init.initialize_if_needed() is False

    def test_hardware_failure_creates_fallback_profile(self, test_db_session, monkeypatch):
        monkeypatch.setattr(config, "LLM_Engine", None)  # detection will fail
        init = Hardware_Initializer(test_db_session)
        assert init.initialize_if_needed() is True
        profile = test_db_session.query(HardwareProfile).first()
        assert profile is not None
        assert profile.cpu_model == "Unknown CPU"
        assert profile.global_inference_label == "Poor"

    def test_startup_variables_initialize_once(self, test_db_session):
        init = Startup_Initializer(test_db_session)
        assert init.initialize_if_needed() is True
        row = test_db_session.query(StartupVariables).first()
        assert row.welcome_popup_has_already_displayed is False
        assert init.initialize_if_needed() is False


# =====================================================================
# INTEGRATION - Database_Seeder facade
# =====================================================================


@pytest.mark.integration
class TestDatabaseSeederFacade:
    async def test_create_tables_requires_initialized_database(self, monkeypatch):
        monkeypatch.setattr(db_core, "db_engine", None)
        with pytest.raises(RuntimeError, match="Database not initialized"):
            await Database_Seeder().create_tables()

    async def test_create_tables_is_idempotent(self, monkeypatch, test_db_engine):
        monkeypatch.setattr(db_core, "db_engine", test_db_engine)
        await Database_Seeder().create_tables()  # tables already exist: no error

    async def test_create_tables_wraps_failure(self, monkeypatch):
        broken = MagicMock()
        monkeypatch.setattr(db_core, "db_engine", broken)
        with patch.object(
            seed_mod.Base.metadata, "create_all", side_effect=RuntimeError("no dbspace")
        ):
            with pytest.raises(RuntimeError, match="no dbspace"):
                await Database_Seeder().create_tables()

    async def test_populate_startup_data_with_snapshot_resync(
        self, test_db_session, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngineType)
        monkeypatch.setattr(config, "LLM_DIR", tmp_path / "models")
        seeder = Database_Seeder()
        monkeypatch.setattr(
            seeder,
            "reconcile_catalog_from_snapshot",
            lambda db: {
                "resynced": True,
                "base_models_added": 5,
                "derived_models_added": 7,
            },
        )
        results = await seeder.populate_startup_data(db=test_db_session)
        assert results["models_seeded"] is True
        assert results["base_models_added"] == 5
        assert results["derived_models_added"] == 7
        assert results["startup_vars_initialized"] is True
        assert results["hardware_initialized"] is True
        assert results["jobs_cleaned"] == {"download": 0, "kb": 0, "orphaned": 0}

    async def test_populate_startup_data_offline_fallback_on_empty_catalog(
        self, test_db_session, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngineType)
        monkeypatch.setattr(config, "LLM_DIR", tmp_path / "models")
        seeder = Database_Seeder()
        monkeypatch.setattr(
            seeder, "reconcile_catalog_from_snapshot", lambda db: {"resynced": False}
        )

        class FakeModelSeeder:
            def __init__(self, db, offline_mode=False):
                assert offline_mode is True

            def seed_initial_catalog(self):
                return 4

        monkeypatch.setattr(seed_mod, "Model_Seeder", FakeModelSeeder)
        results = await seeder.populate_startup_data(db=test_db_session)
        assert results["base_models_added"] == 4
        assert results["models_seeded"] is False

    async def test_populate_startup_data_rolls_back_and_reraises(
        self, test_db_session, monkeypatch
    ):
        seeder = Database_Seeder()

        def boom(db):
            raise RuntimeError("snapshot io error")

        monkeypatch.setattr(seeder, "reconcile_catalog_from_snapshot", boom)
        with pytest.raises(RuntimeError, match="snapshot io error"):
            await seeder.populate_startup_data(db=test_db_session)

    async def test_legacy_startup_populate_delegates(self, monkeypatch):
        called = {}

        async def fake_populate(self, db=None):
            called["db"] = db
            return {"ok": True}

        monkeypatch.setattr(Database_Seeder, "populate_startup_data", fake_populate)
        assert await seed_mod.startup_populate_database() == {"ok": True}

    async def test_legacy_create_tables_delegates(self, monkeypatch):
        called = {}

        async def fake_create(self):
            called["yes"] = True

        monkeypatch.setattr(Database_Seeder, "create_tables", fake_create)
        await seed_mod.create_tables()
        assert called == {"yes": True}


# =====================================================================
# INTEGRATION - wire-tools backfill (#298)
# =====================================================================


@pytest.mark.integration
class TestWireToolsBackfill:
    def _local_llm(self, db, link, wire=None):
        llm = Llm(name="L", local=1, link=link, type="x", supports_tools_wire=wire)
        db.add(llm)
        db.flush()
        return llm

    def test_backfill_persists_verdict_and_caches_shared_links(
        self, test_db_session, monkeypatch, tmp_path
    ):
        model_dir = tmp_path / "m1"
        model_dir.mkdir()
        a = self._local_llm(test_db_session, str(model_dir))
        b = self._local_llm(test_db_session, str(model_dir))  # KB assistant shares link
        calls = []

        def fake_detect(link):
            calls.append(link)
            return True

        monkeypatch.setattr("src.domains.llms.repository.detect_wire_tools", fake_detect)
        updated = Database_Seeder().backfill_wire_tools(test_db_session)
        assert updated == 2
        assert a.supports_tools_wire is True
        assert b.supports_tools_wire is True
        assert calls == [str(model_dir)]  # computed once per shared link

    def test_backfill_skips_missing_artifacts_and_none_verdicts(
        self, test_db_session, monkeypatch, tmp_path
    ):
        gone = self._local_llm(test_db_session, str(tmp_path / "gone"))
        present_dir = tmp_path / "present"
        present_dir.mkdir()
        undecided = self._local_llm(test_db_session, str(present_dir))
        monkeypatch.setattr("src.domains.llms.repository.detect_wire_tools", lambda link: None)
        assert Database_Seeder().backfill_wire_tools(test_db_session) == 0
        assert gone.supports_tools_wire is None
        assert undecided.supports_tools_wire is None  # retried next boot

    def test_backfill_startup_entrypoint_uses_own_session(self, monkeypatch):
        session = MagicMock()
        monkeypatch.setattr(seed_mod, "SessionLocal", lambda: session)
        with patch.object(Database_Seeder, "backfill_wire_tools", return_value=3):
            assert seed_mod.backfill_wire_tools_startup() == 3
        session.close.assert_called_once()

    def test_backfill_startup_entrypoint_swallows_errors(self, monkeypatch):
        def boom():
            raise RuntimeError("no database yet")

        monkeypatch.setattr(seed_mod, "SessionLocal", boom)
        assert seed_mod.backfill_wire_tools_startup() == 0


# =====================================================================
# INTEGRATION - destructive dev reset
# =====================================================================


@pytest.mark.integration
class TestDeleteAllData:
    async def test_cancelled_without_confirmation(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda prompt: "no")
        factory = MagicMock()
        monkeypatch.setattr(seed_mod, "SessionLocal", factory)
        await Database_Seeder().delete_all_data()
        factory.assert_not_called()  # cancelled before any session was opened

    async def test_confirmed_deletes_rows_and_storage(
        self, test_db_session, monkeypatch, tmp_path, mock_llm
    ):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "leftover").mkdir()
        monkeypatch.setattr(config, "LLM_DIR", models_dir)
        monkeypatch.setattr("builtins.input", lambda prompt: "yes")
        monkeypatch.setattr(seed_mod, "SessionLocal", lambda: test_db_session)
        # Neutralize close(): the fixture still owns this session's lifecycle.
        monkeypatch.setattr(test_db_session, "close", lambda: None)

        await seed_mod.delete_all_data()  # legacy wrapper delegates to the seeder

        assert test_db_session.query(Llm).count() == 0
        assert models_dir.exists()
        assert list(models_dir.iterdir()) == []  # recreated empty


# =====================================================================
# UNIT - catalog snapshot helpers (#112)
# =====================================================================


@pytest.mark.unit
class TestCatalogSnapshotHelpers:
    def test_load_snapshot_broken_json_returns_empty(self, monkeypatch, tmp_path):
        target = tmp_path / "src" / "database"
        target.mkdir(parents=True)
        (target / "catalog_snapshot_gguf.json").write_text("[{broken")
        monkeypatch.setattr(config, "ROOT_DIR", tmp_path)
        assert snapshot_mod.load_catalog_snapshot("gguf") == []

    def test_generate_snapshot_requires_format_tag(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_Engine", None)
        with pytest.raises(RuntimeError, match="FORMAT_TAG"):
            snapshot_mod.generate_snapshot()

    def test_generate_snapshot_writes_catalog_json(self, monkeypatch, tmp_path):
        target = tmp_path / "src" / "database"
        target.mkdir(parents=True)
        monkeypatch.setattr(config, "ROOT_DIR", tmp_path)
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngineType)
        monkeypatch.setattr(config, "get_hf_api", lambda: MagicMock(), raising=False)
        base = [Llm(name="Base", local=0, link="a/base-gguf", type="x", is_base=True)]
        derived = [Llm(name="Derived", local=0, link="b/derived-gguf", type="x")]
        with patch.object(Database_Seeder, "build_fresh_catalog", return_value=(base, derived)):
            path = snapshot_mod.generate_snapshot()
        data = json.loads(Path(path).read_text())
        assert [e["link"] for e in data] == ["a/base-gguf", "b/derived-gguf"]
        assert data[0]["is_base"] is True

    def test_main_selects_engine_when_missing(self, monkeypatch, capsys):
        monkeypatch.setattr(config, "LLM_Engine", None)
        from src.engines.base_engine import BaseEngine

        monkeypatch.setattr(BaseEngine, "get_engine", classmethod(lambda cls: _FakeEngineType))
        monkeypatch.setattr(snapshot_mod, "generate_snapshot", lambda: Path("/tmp/snap.json"))
        snapshot_mod.main()
        assert config.LLM_Engine is _FakeEngineType
        assert "snap.json" in capsys.readouterr().out
