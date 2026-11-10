"""The real artifact size travels from the catalog to the installed row.

The frontend used to show a per-parameter estimate ("~2.3 GB" for
Qwen2.5-VL-3B-Instruct) that misses a VLM's vision tower by 25-35 %; on a
nearly full disk that is misleading. The backend now records the bytes the
downloader would actually fetch (``llms.artifact_size_bytes``, captured at
snapshot time -- see test_catalog_snapshot), copies it onto the download
placeholder, and replaces it with the measured on-disk footprint the moment a
download completes. NULL = unknown: the frontend keeps its estimate.

- ``unit``: the API field, the entity guard, the completion-time measurement.
- ``integration``: the column round-trips, the by-id download copies the
  catalog value, a KB assistant inherits its base's size.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import inspect

from src.core import config
from src.domains.knowledge_base.repository import COPIED_FIELDS
from src.domains.llms.schemas import LLMResponse
from src.entities.Llm import Llm
from tests._helpers import FakeDownloadQuery


# ---------------------------------------------------------------- unit: API + entity


@pytest.mark.unit
def test_llmresponse_exposes_artifact_size_bytes_default_none():
    assert "artifact_size_bytes" in LLMResponse.model_fields
    assert LLMResponse.model_fields["artifact_size_bytes"].default is None


@pytest.mark.unit
def test_llmresponse_reads_the_column_from_the_orm_row():
    row = Llm(
        id=1,
        name="Qwen2.5 VL 3B",
        local=0,
        type="qwen",
        is_base=True,
        link="mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
        artifact_size_bytes=3_090_000_000,
    )
    assert LLMResponse.model_validate(row).artifact_size_bytes == 3_090_000_000
    assert "artifact_size_bytes" in LLMResponse.model_validate(row).model_dump()


@pytest.mark.unit
def test_entity_rejects_a_non_positive_size_but_allows_unknown():
    """A zero or negative byte count is never a measurement -- the only honest
    unknown is NULL, which is what an estimate-backed catalog row carries."""
    assert Llm(name="x", local=0, type="t", artifact_size_bytes=None).artifact_size_bytes is None
    with pytest.raises(ValueError):
        Llm(name="x", local=0, type="t", artifact_size_bytes=0)
    with pytest.raises(ValueError):
        Llm(name="x", local=0, type="t", artifact_size_bytes=-1)


@pytest.mark.unit
def test_kb_assistant_inherits_its_base_artifact_size():
    # An assistant shares its base's weights (COPIED link); its card must show
    # the same real size rather than falling back to the estimate.
    assert "artifact_size_bytes" in COPIED_FIELDS


# ---------------------------------------------------------------- unit: completion


@pytest.mark.unit
class TestDownloadCompletionRecordsTheRealBytes:
    """``_run_download_task`` measures the installed directory once (bytes) and
    stores it alongside the GB rewrite of the metadata string (#220/#349), so
    an installed row is exact even when the catalog carried an estimate."""

    def _arm(self, monkeypatch, final_dir, temp_dir, *, catalog_bytes):
        from src.domains.llms import endpoints, repository
        from src.entities.DownloadJob import DownloadJobModel

        monkeypatch.setattr(config, "LLM_Engine", SimpleNamespace())  # no validator
        for name in ("detect_supports_tools", "detect_wire_tools", "detect_supports_vision"):
            monkeypatch.setattr(repository, name, lambda link: None)
        monkeypatch.setattr(endpoints, "_capture_hints_for_download", lambda *a: None)

        llm = Llm(
            name="Model",
            local=2,
            type="qwen",
            link=str(final_dir),
            model_metadata="Model ID: org/model\nSize: ~2.3 GB",
            artifact_size_bytes=catalog_bytes,
        )
        job = DownloadJobModel(
            remote_model_id="org/model",
            local_model_id=1,
            remote_model_link="org/model",
            temp_local_model_link=str(temp_dir),
            final_local_model_link=str(final_dir),
            status="pending",  # as _start_download creates it
        )

        class _Session:
            def query(self, model):
                return FakeDownloadQuery(model, job, llm)

            def commit(self):
                pass

            def rollback(self):
                pass

            def close(self):
                pass

        monkeypatch.setattr(endpoints, "SessionLocal", lambda: _Session())

        async def _done(**kwargs):
            return None

        monkeypatch.setattr(endpoints, "download_llm", _done)
        return endpoints, llm, job

    def test_installed_row_carries_the_measured_bytes(self, tmp_path, monkeypatch):
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        (final_dir / "model.safetensors").write_bytes(b"w" * 3_000)
        (final_dir / "config.json").write_bytes(b"{}" + b" " * 90)
        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()
        endpoints, llm, job = self._arm(monkeypatch, final_dir, temp_dir, catalog_bytes=2_300)

        endpoints._run_download_task("org/model", 1, temp_dir, final_dir, job_id=1)

        assert (llm.local, job.status) == (1, "completed")
        assert llm.artifact_size_bytes == 3_092  # the real footprint
        assert "Disk Size GB: 0.00" in llm.model_metadata  # same walk feeds the GB line

    def test_unmeasurable_directory_keeps_the_catalog_value(self, tmp_path, monkeypatch):
        # The directory vanished between the transfer and the measurement (a
        # cancel race): the walk returns 0, which must not be written as a size.
        final_dir = tmp_path / "gone"
        temp_dir = tmp_path / "temp"
        endpoints, llm, job = self._arm(monkeypatch, final_dir, temp_dir, catalog_bytes=2_300)

        endpoints._run_download_task("org/model", 1, temp_dir, final_dir, job_id=1)

        assert job.status == "completed"
        assert llm.artifact_size_bytes == 2_300


# ---------------------------------------------------------------- integration


@pytest.mark.integration
def test_artifact_size_bytes_column_exists(test_db_engine):
    cols = {c["name"] for c in inspect(test_db_engine).get_columns("llms")}
    assert "artifact_size_bytes" in cols


@pytest.mark.integration
def test_artifact_size_bytes_roundtrips_beyond_32_bits(test_db_session):
    # A 70B quant is ~40 GB: the column must hold more than a 4-byte integer.
    llm = Llm(name="Big", local=0, type="llama", link="org/big", artifact_size_bytes=40_000_000_000)
    test_db_session.add(llm)
    test_db_session.commit()
    test_db_session.refresh(llm)
    assert llm.artifact_size_bytes == 40_000_000_000


@pytest.mark.integration
def test_catalog_download_copies_the_size_to_the_local_row(client, test_db_session):
    """POST /llms/{id}/download: the placeholder row starts with the catalog's
    real size, so the card never regresses to an estimate mid-download."""
    remote = Llm(
        name="Remote",
        local=0,
        type="qwen",
        link="mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
        param_size=3.0,
        artifact_size_bytes=3_090_000_000,
    )
    test_db_session.add(remote)
    test_db_session.commit()

    with (
        patch("src.domains.llms.endpoints.download_llm") as mock_download,
        patch("pathlib.Path.exists", return_value=False),
    ):
        mock_download.return_value = AsyncMock()
        resp = client.post(f"/erudi/llms/{remote.id}/download")
    assert resp.status_code == 200
    local_id = resp.json()["local_model_id"]
    assert test_db_session.query(Llm).get(local_id).artifact_size_bytes == 3_090_000_000


@pytest.mark.integration
def test_list_endpoint_serializes_the_size(client, test_db_session):
    remote = Llm(
        name="Remote",
        local=0,
        type="qwen",
        link="org/sized",
        param_size=3.0,
        artifact_size_bytes=3_090_000_000,
    )
    unknown = Llm(name="Unknown", local=0, type="qwen", link="org/unsized", param_size=3.0)
    test_db_session.add_all([remote, unknown])
    test_db_session.commit()

    by_link = {m["link"]: m for m in client.get("/erudi/llms/").json()}
    assert by_link["org/sized"]["artifact_size_bytes"] == 3_090_000_000
    assert by_link["org/unsized"]["artifact_size_bytes"] is None
