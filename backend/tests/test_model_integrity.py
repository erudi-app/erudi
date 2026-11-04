"""Integrity validation of downloaded artifacts + precise load errors (#88).

Covers three seams, all CPU-only so CI (which runs the CPU engine) exercises
them end to end:

1. The pure validators in ``src.engines.integrity`` (GGUF magic/size, HF/MLX
   snapshot essential-file set) with ``tmp_path`` artifacts.
2. ``CPU_Engine.validate_local_artifact`` -- the GGUF gate wired into both the
   download-completion path and the pre-spawn load path.
3. ``endpoints._assert_downloaded_artifact_ok`` -- the failure path cleans up
   the artifacts and raises (so the job finalizes failed, never local=1).
4. ``runner._construction_error_message`` -- an EngineException raised in the
   load path reaches the chat user as its specific message (no traceback).

No network, no subprocess spawn: every artifact is a handful of bytes on disk.
"""

from __future__ import annotations

import pytest

from src.core import config
from src.core.exceptions import EngineException
from src.engines import integrity
from src.utils.hf_model_metadata import measure_dir_size_gb
from tests._helpers import FakeDownloadQuery

pytestmark = pytest.mark.unit


# A minimal-but-valid GGUF payload: magic + some padding so size > 0.
_VALID_GGUF = integrity.GGUF_MAGIC + b"\x00" * 64


def _write_gguf_dir(tmp_path, name="model-q4_k_m.gguf", data=_VALID_GGUF):
    """Create a model directory holding a single .gguf and return the dir."""
    d = tmp_path / "mdl"
    d.mkdir(exist_ok=True)
    (d / name).write_bytes(data)
    return d


def _write_snapshot(tmp_path, *, config_json=True, tokenizer=True, weights=True):
    """Create an HF/MLX snapshot directory with a selectable essential set."""
    d = tmp_path / "snap"
    d.mkdir(exist_ok=True)
    if config_json:
        (d / "config.json").write_text('{"model_type": "qwen2"}', encoding="utf-8")
    if tokenizer:
        (d / "tokenizer.json").write_text("{}", encoding="utf-8")
    if weights:
        (d / "model.safetensors").write_bytes(b"\x00" * 128)
    return d


# =====================================================================
# UNIT -- pure GGUF file validator
# =====================================================================


class TestValidateGgufFile:
    def test_valid_magic_passes(self, tmp_path):
        f = tmp_path / "ok.gguf"
        f.write_bytes(_VALID_GGUF)
        # No exception == pass.
        integrity.validate_gguf_file(f)

    def test_missing_file_raises_missing(self, tmp_path):
        with pytest.raises(EngineException, match="missing"):
            integrity.validate_gguf_file(tmp_path / "nope.gguf")

    def test_empty_file_raises_empty(self, tmp_path):
        f = tmp_path / "empty.gguf"
        f.write_bytes(b"")
        with pytest.raises(EngineException, match="empty"):
            integrity.validate_gguf_file(f)

    def test_truncated_wrong_magic_raises_corrupted(self, tmp_path):
        f = tmp_path / "bad.gguf"
        f.write_bytes(b"ZZZZ" + b"\x00" * 32)  # right size, wrong magic
        with pytest.raises(EngineException, match="corrupted"):
            integrity.validate_gguf_file(f)

    def test_shorter_than_magic_raises_corrupted(self, tmp_path):
        f = tmp_path / "short.gguf"
        f.write_bytes(b"GG")  # non-empty but < 4 magic bytes
        with pytest.raises(EngineException, match="corrupted"):
            integrity.validate_gguf_file(f)


# =====================================================================
# UNIT -- pure HF/MLX snapshot validator
# =====================================================================


class TestValidateHfSnapshot:
    def test_complete_snapshot_passes(self, tmp_path):
        integrity.validate_hf_snapshot(_write_snapshot(tmp_path))

    def test_missing_dir_raises_folder_missing(self, tmp_path):
        with pytest.raises(EngineException, match="folder is missing"):
            integrity.validate_hf_snapshot(tmp_path / "ghost")

    def test_missing_config_raises(self, tmp_path):
        with pytest.raises(EngineException, match="config.json"):
            integrity.validate_hf_snapshot(_write_snapshot(tmp_path, config_json=False))

    def test_missing_tokenizer_raises(self, tmp_path):
        with pytest.raises(EngineException, match="tokenizer"):
            integrity.validate_hf_snapshot(_write_snapshot(tmp_path, tokenizer=False))

    def test_missing_weights_raises(self, tmp_path):
        with pytest.raises(EngineException, match="weights"):
            integrity.validate_hf_snapshot(_write_snapshot(tmp_path, weights=False))

    def test_empty_config_counts_as_missing(self, tmp_path):
        d = _write_snapshot(tmp_path)
        (d / "config.json").write_text("", encoding="utf-8")  # present but empty
        with pytest.raises(EngineException, match="config.json"):
            integrity.validate_hf_snapshot(d)

    def test_sentencepiece_tokenizer_accepted(self, tmp_path):
        d = _write_snapshot(tmp_path, tokenizer=False)
        (d / "tokenizer.model").write_bytes(b"\x00" * 16)  # spm instead of tokenizer.json
        integrity.validate_hf_snapshot(d)


# =====================================================================
# UNIT -- incomplete-artifact message policy
# =====================================================================


class TestIncompleteMessage:
    def test_message_is_ascii_and_actionable(self):
        msg = integrity.incomplete_message("missing tokenizer")
        assert msg.isascii()
        assert "missing tokenizer" in msg
        # Blames the model, offers the two real remedies, and explicitly does
        # NOT pin it on the network.
        assert "not your connection" in msg
        assert "retry the download" in msg
        assert "choose another model" in msg


# =====================================================================
# UNIT -- CPU_Engine.validate_local_artifact (GGUF gate, CI CPU path)
# =====================================================================


class TestCpuEngineValidateArtifact:
    def _engine(self):
        from src.engines.cpu_engine import CPU_Engine

        return CPU_Engine

    def test_valid_dir_passes(self, tmp_path):
        self._engine().validate_local_artifact(_write_gguf_dir(tmp_path))

    def test_valid_single_file_passes(self, tmp_path):
        f = tmp_path / "model-q4_k_m.gguf"
        f.write_bytes(_VALID_GGUF)
        self._engine().validate_local_artifact(f)

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(EngineException, match="folder is missing"):
            self._engine().validate_local_artifact(tmp_path / "absent")

    def test_dir_without_gguf_raises(self, tmp_path):
        d = tmp_path / "mdl"
        d.mkdir()
        (d / "config.json").write_text("{}", encoding="utf-8")
        with pytest.raises(EngineException, match="no GGUF weights file"):
            self._engine().validate_local_artifact(d)

    def test_only_mmproj_gguf_is_not_a_model(self, tmp_path):
        d = tmp_path / "mdl"
        d.mkdir()
        (d / "mmproj-model-f16.gguf").write_bytes(_VALID_GGUF)
        with pytest.raises(EngineException, match="no GGUF weights file"):
            self._engine().validate_local_artifact(d)

    def test_corrupt_gguf_raises(self, tmp_path):
        with pytest.raises(EngineException, match="corrupted"):
            self._engine().validate_local_artifact(
                _write_gguf_dir(tmp_path, data=b"ZZZZ" + b"\x00" * 16)
            )

    def test_empty_gguf_raises(self, tmp_path):
        with pytest.raises(EngineException, match="empty"):
            self._engine().validate_local_artifact(_write_gguf_dir(tmp_path, data=b""))


# =====================================================================
# UNIT -- download-completion gate cleans up and never lets local=1
# =====================================================================


class TestDownloadCompletionGate:
    def _use_cpu_engine(self, monkeypatch):
        from src.engines.cpu_engine import CPU_Engine

        monkeypatch.setattr(config, "LLM_Engine", CPU_Engine)

    def test_valid_artifact_passes_and_keeps_files(self, tmp_path, monkeypatch):
        from src.domains.llms.endpoints import _assert_downloaded_artifact_ok

        self._use_cpu_engine(monkeypatch)
        final_dir = _write_gguf_dir(tmp_path)
        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()

        _assert_downloaded_artifact_ok(final_dir, temp_dir)  # no raise

        assert final_dir.exists()  # valid artifact is left in place

    def test_invalid_artifact_raises_and_cleans_up(self, tmp_path, monkeypatch):
        from src.domains.llms.endpoints import _assert_downloaded_artifact_ok

        self._use_cpu_engine(monkeypatch)
        # A "completed" download whose final dir has no usable GGUF.
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        (final_dir / "config.json").write_text("{}", encoding="utf-8")
        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()

        with pytest.raises(EngineException, match="incomplete or invalid"):
            _assert_downloaded_artifact_ok(final_dir, temp_dir)

        # Artifacts removed (mirrors delete_llm): both dirs are gone.
        assert not final_dir.exists()
        assert not temp_dir.exists()

    def test_no_engine_validator_is_a_noop(self, tmp_path, monkeypatch):
        from src.domains.llms.endpoints import _assert_downloaded_artifact_ok

        class _NoValidator:
            pass

        monkeypatch.setattr(config, "LLM_Engine", _NoValidator)
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        # Must not raise and must not delete anything when the engine has no gate.
        _assert_downloaded_artifact_ok(final_dir, tmp_path / "temp")
        assert final_dir.exists()


# =====================================================================
# UNIT -- download completion rewrites the displayed size from disk (#220)
# =====================================================================


class TestDownloadCompletionSize:
    """After a (fake) download passes the #88 gate, the model's displayed size is
    the REAL on-disk footprint, not the catalog-time guess. Drives the actual
    ``_run_download_task`` with in-memory rows (SessionLocal is not DB-bound in
    tests) so the exact finalization path is exercised."""

    def test_run_download_task_rewrites_size_from_disk(self, tmp_path, monkeypatch):
        from src.domains.llms import endpoints
        from src.domains.llms import repository
        from src.engines.cpu_engine import CPU_Engine
        from src.entities.Llm import Llm
        from src.entities.DownloadJob import DownloadJobModel

        monkeypatch.setattr(config, "LLM_Engine", CPU_Engine)
        # Keep capability detection hermetic and fast (no real model load).
        monkeypatch.setattr(repository, "detect_supports_tools", lambda link: None)
        monkeypatch.setattr(repository, "detect_supports_vision", lambda link: None)

        # A "completed" download: a valid GGUF on disk so the #88 gate passes.
        final_dir = _write_gguf_dir(tmp_path)
        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()

        llm = Llm(
            name="Model",
            local=2,
            type="qwen",
            param_size=7.0,
            link=str(final_dir),
            model_metadata="Model ID: org/model\nSize: ~40.2 GB\nParameters: 7B",
        )
        job = DownloadJobModel(
            remote_model_id="org/model",
            local_model_id=1,
            remote_model_link="org/model",
            temp_local_model_link=str(temp_dir),
            final_local_model_link=str(final_dir),
            status="running",
        )

        class _FakeSession:
            def query(self, model):
                return FakeDownloadQuery(model, job, llm)

            def commit(self):
                pass

            def rollback(self):
                pass

            def close(self):
                pass

        monkeypatch.setattr(endpoints, "SessionLocal", lambda: _FakeSession())

        async def _fake_download(*args, **kwargs):
            return None

        monkeypatch.setattr(endpoints, "download_llm", _fake_download)

        endpoints._run_download_task("org/model", 1, temp_dir, final_dir, job_id=1)

        # Finalized ready, with the catalog guess replaced by the measured size.
        assert llm.local == 1
        assert job.status == "completed"
        assert "40.2" not in llm.model_metadata  # catalog guess gone
        assert "Disk Size GB:" in llm.model_metadata  # numeric field added
        assert "Parameters: 7B" in llm.model_metadata  # other lines preserved
        measured = measure_dir_size_gb(final_dir)
        assert f"Disk Size GB: {measured:.2f}" in llm.model_metadata

    def test_size_is_measured_before_the_job_goes_terminal(self, tmp_path, monkeypatch):
        """The size rewrite must not sit behind the capability probes (#349).

        The UI refreshes the instant it polls "completed". While the rewrite
        lived after the probes -- which take ~10s on a 7B GGUF, and minutes in
        the #291 case -- the freshly installed card rendered the catalog's
        pre-download guess and only corrected itself on a manual reload.

        Probing is best-effort, so a probe blowing up must not take the measured
        size with it either: that is the same ordering, stated as a failure.
        """
        from src.domains.llms import endpoints
        from src.domains.llms import repository
        from src.engines.cpu_engine import CPU_Engine
        from src.entities.Llm import Llm
        from src.entities.DownloadJob import DownloadJobModel

        monkeypatch.setattr(config, "LLM_Engine", CPU_Engine)

        def _probe_boom(link):
            raise RuntimeError("probe hung and was torn down")

        monkeypatch.setattr(repository, "detect_supports_tools", _probe_boom)
        monkeypatch.setattr(repository, "detect_supports_vision", _probe_boom)

        final_dir = _write_gguf_dir(tmp_path)
        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()

        llm = Llm(
            name="Model",
            local=2,
            type="qwen",
            param_size=7.0,
            link=str(final_dir),
            model_metadata="Model ID: org/model\nSize: ~40.2 GB\nParameters: 7B",
        )
        job = DownloadJobModel(
            remote_model_id="org/model",
            local_model_id=1,
            remote_model_link="org/model",
            temp_local_model_link=str(temp_dir),
            final_local_model_link=str(final_dir),
            status="running",
        )

        # Record the metadata as it stood the moment the job became terminal --
        # that is exactly what the UI's next poll would render.
        seen = {}

        class _FakeSession:
            def query(self, model):
                return FakeDownloadQuery(model, job, llm)

            def commit(self):
                if job.status == "completed" and "at_completion" not in seen:
                    seen["at_completion"] = llm.model_metadata

            def rollback(self):
                pass

            def close(self):
                pass

        monkeypatch.setattr(endpoints, "SessionLocal", lambda: _FakeSession())

        async def _fake_download(*args, **kwargs):
            return None

        monkeypatch.setattr(endpoints, "download_llm", _fake_download)

        endpoints._run_download_task("org/model", 1, temp_dir, final_dir, job_id=1)

        assert job.status == "completed"
        measured = measure_dir_size_gb(final_dir)
        # The size was already correct at the instant the job went terminal.
        assert f"Disk Size GB: {measured:.2f}" in seen["at_completion"]
        assert "40.2" not in seen["at_completion"]
        # And a failed probe left it correct rather than rolling it back.
        assert f"Disk Size GB: {measured:.2f}" in llm.model_metadata


# =====================================================================
# UNIT -- runner surfaces the specific engine load error to the user
# =====================================================================


class TestConstructionErrorMessage:
    def test_engine_exception_carries_specific_message(self):
        from src.agents.runner import ERROR_SENTINEL, _construction_error_message

        exc = EngineException(
            message=integrity.incomplete_message("the GGUF weights file is corrupted")
        )
        out = _construction_error_message(exc)

        assert out.startswith(ERROR_SENTINEL)
        assert "corrupted" in out
        assert "choose another model" in out
        assert "Traceback" not in out  # curated message only, never a stack trace

    def test_generic_exception_uses_generic_message(self):
        from src.agents.runner import ERROR_MESSAGE, _construction_error_message

        assert _construction_error_message(RuntimeError("boom")) == ERROR_MESSAGE
