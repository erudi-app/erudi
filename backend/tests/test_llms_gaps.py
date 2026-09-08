"""Gap coverage for the llms domain on top of `test_llms.py`.

Pins the paths the main suite leaves naked: the download integrity gate
(#88), the background download task's failure finalization, rebind guard
rails (#209/#225), delete guards for dependents and inconsistent KB
assistants, terminal-status polling cleanup, the DB progress updater thread
body, `cleanup_job_files`, the offline-error classifier (#109), and the
fully mocked `download_llm` pipeline including its auth/offline error
mapping.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.core import config
from src.core.exceptions import (
    HuggingFaceAPIException,
    UnsupportedPlatformException,
)
from src.domains.llms import endpoints as llm_endpoints
from src.domains.llms import repository as llm_repo_mod
from src.domains.llms import services as llm_services
from src.domains.llms.repository import Download_Job_Repository, Llm_Repository
from src.entities.DownloadJob import DownloadJobModel
from src.entities.KnowledgeBase import KnowledgeBase
from src.entities.Llm import Llm


def _add_llm(db, **overrides):
    data = dict(name="Model", local=1, link="/models/x", type="qwen", param_size=4.0)
    data.update(overrides)
    llm = Llm(**data)
    db.add(llm)
    db.flush()
    return llm


def _add_job(db, llm_id, status="running", **overrides):
    data = dict(
        remote_model_id="42",
        local_model_id=llm_id,
        remote_model_link="org/model",
        temp_local_model_link="",
        status=status,
    )
    data.update(overrides)
    job = DownloadJobModel(**data)
    db.add(job)
    db.commit()
    return job


# =====================================================================
# UNIT - integrity gate (#88)
# =====================================================================


@pytest.mark.unit
class TestArtifactIntegrityGate:
    def test_engine_without_validator_is_noop(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_Engine", SimpleNamespace())
        llm_endpoints._assert_downloaded_artifact_ok("/final", "/temp")

    def test_valid_artifact_passes(self, monkeypatch):
        validator = MagicMock()
        monkeypatch.setattr(
            config, "LLM_Engine", SimpleNamespace(validate_local_artifact=validator)
        )
        llm_endpoints._assert_downloaded_artifact_ok("/final", "/temp")
        validator.assert_called_once_with("/final")

    def test_invalid_artifact_cleans_dirs_and_reraises(self, monkeypatch, tmp_path):
        final_dir = tmp_path / "final"
        final_dir.mkdir()
        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()

        def validator(path):
            raise ValueError("missing tokenizer.json")

        monkeypatch.setattr(
            config, "LLM_Engine", SimpleNamespace(validate_local_artifact=validator)
        )
        with pytest.raises(ValueError, match="missing tokenizer.json"):
            llm_endpoints._assert_downloaded_artifact_ok(final_dir, temp_dir)
        assert not final_dir.exists()
        assert not temp_dir.exists()

    def test_cleanup_failure_still_reraises_original_error(self, monkeypatch, tmp_path):
        def validator(path):
            raise ValueError("bad artifact")

        monkeypatch.setattr(
            config, "LLM_Engine", SimpleNamespace(validate_local_artifact=validator)
        )

        def exists_boom(path):
            raise OSError("filesystem gone")

        monkeypatch.setattr(llm_endpoints.os.path, "exists", exists_boom)
        with pytest.raises(ValueError, match="bad artifact"):
            llm_endpoints._assert_downloaded_artifact_ok(tmp_path / "final", tmp_path / "temp")


# =====================================================================
# UNIT - background download task failure finalization
# =====================================================================


@pytest.mark.unit
class TestRunDownloadTaskFailure:
    def test_failed_download_marks_job_failed(self, monkeypatch):
        job = SimpleNamespace(status="pending", error_message=None, updated_at=None)
        session = MagicMock()
        session.query.return_value.get.return_value = job
        monkeypatch.setattr(llm_endpoints, "SessionLocal", lambda: session)

        async def broken_download(**kwargs):
            raise RuntimeError("network died mid-transfer")

        monkeypatch.setattr(llm_endpoints, "download_llm", broken_download)

        llm_endpoints._run_download_task("org/model", 1, "/tmp/t", "/tmp/f", job_id=9)

        assert job.status == "failed"
        assert "network died" in job.error_message
        session.close.assert_called_once()


# =====================================================================
# INTEGRATION - endpoints: search proxy, dependents, update, delete
# =====================================================================


@pytest.mark.integration
class TestEndpointGaps:
    def test_hf_search_route_proxies_results(self, client, monkeypatch):
        monkeypatch.setattr(llm_endpoints, "search_huggingface", lambda q, limit=30: [])
        resp = client.get("/erudi/llms/search/huggingface", params={"q": "gemma"})
        assert resp.status_code == 200
        assert resp.json() == []

    def test_dependents_unknown_model_404(self, client):
        assert client.get("/erudi/llms/987654/dependents").status_code == 404

    def test_dependents_payload_lists_assistants(self, client, test_db_session):
        base = _add_llm(test_db_session, name="Base", link="/models/base")
        kb = KnowledgeBase()
        test_db_session.add(kb)
        test_db_session.flush()
        assistant = _add_llm(
            test_db_session,
            name="Helper",
            link="/models/base",  # shares COPIED link (#209)
            is_attached_to_kb=True,
            kb_id=kb.id,
        )
        test_db_session.commit()
        resp = client.get(f"/erudi/llms/{base.id}/dependents")
        assert resp.status_code == 200
        body = resp.json()
        assert [a["id"] for a in body["assistants"]] == [assistant.id]

    def test_update_unknown_model_404(self, client):
        resp = client.put("/erudi/llms/987654", json={"name": "X", "local": 0, "link": "y"})
        assert resp.status_code == 404

    def test_update_failure_maps_to_database_exception(self, client, test_db_session):
        llm = _add_llm(test_db_session, local=0)
        test_db_session.commit()
        with patch.object(Llm_Repository, "update", side_effect=RuntimeError("constraint")):
            resp = client.put(f"/erudi/llms/{llm.id}", json={"name": "X", "local": 0, "link": "y"})
        assert resp.status_code == 500

    def test_delete_base_with_dependents_conflicts_without_optin(self, client, test_db_session):
        base = _add_llm(test_db_session, name="Base", link="/models/base")
        _add_llm(test_db_session, name="Helper", link="/models/base", is_attached_to_kb=True)
        test_db_session.commit()
        resp = client.delete(f"/erudi/llms/{base.id}")
        assert resp.status_code == 409
        # The base survives the guarded refusal
        assert client.get(f"/erudi/llms/{base.id}").status_code == 200

    def test_delete_base_with_dependents_proceeds_with_optin(self, client, test_db_session):
        base = _add_llm(test_db_session, name="Base", link="/models/base-nonexistent")
        helper = _add_llm(
            test_db_session,
            name="Helper",
            link="/models/base-nonexistent",
            is_attached_to_kb=True,
        )
        test_db_session.commit()
        resp = client.delete(f"/erudi/llms/{base.id}", params={"orphan_dependents": "true"})
        assert resp.status_code == 200
        # The assistant remains (now orphaned, rebindable)
        assert client.get(f"/erudi/llms/{helper.id}").status_code == 200

    def test_delete_inconsistent_assistant_without_kb_drops_row(self, client, test_db_session):
        ghost = _add_llm(test_db_session, name="Ghost", is_attached_to_kb=True, kb_id=None)
        test_db_session.commit()
        resp = client.delete(f"/erudi/llms/{ghost.id}")
        assert resp.status_code == 200
        assert client.get(f"/erudi/llms/{ghost.id}").status_code == 404

    def test_delete_failure_maps_to_database_exception(self, client, test_db_session):
        llm = _add_llm(test_db_session, link="/models/does-not-exist")
        test_db_session.commit()
        with patch.object(Llm_Repository, "delete", side_effect=RuntimeError("locked")):
            resp = client.delete(f"/erudi/llms/{llm.id}")
        assert resp.status_code == 500


# =====================================================================
# INTEGRATION - rebind guard rails (#209/#225)
# =====================================================================


@pytest.mark.integration
class TestRebind:
    def _assistant_and_base(self, db, tmp_path):
        weights = tmp_path / "weights"
        weights.mkdir()
        kb = KnowledgeBase()
        db.add(kb)
        db.flush()
        assistant = _add_llm(
            db,
            name="Assistant",
            link="/models/gone",
            is_attached_to_kb=True,
            kb_id=kb.id,
        )
        base = _add_llm(db, name="NewBase", link=str(weights))
        db.commit()
        return assistant, base

    def test_rebind_success_copies_link(self, client, test_db_session, tmp_path):
        assistant, base = self._assistant_and_base(test_db_session, tmp_path)
        resp = client.post(
            f"/erudi/llms/{assistant.id}/rebind",
            json={"new_base_llm_id": base.id},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["link"] == base.link
        assert body["name"] == "Assistant"  # own identity preserved

    def test_rebind_unknown_assistant_404(self, client):
        resp = client.post("/erudi/llms/987654/rebind", json={"new_base_llm_id": 1})
        assert resp.status_code == 404

    def test_rebind_non_assistant_conflicts(self, client, test_db_session):
        plain = _add_llm(test_db_session, name="Plain")
        test_db_session.commit()
        resp = client.post(f"/erudi/llms/{plain.id}/rebind", json={"new_base_llm_id": 1})
        assert resp.status_code == 409

    def test_rebind_unknown_target_404(self, client, test_db_session, tmp_path):
        assistant, _ = self._assistant_and_base(test_db_session, tmp_path)
        resp = client.post(f"/erudi/llms/{assistant.id}/rebind", json={"new_base_llm_id": 987654})
        assert resp.status_code == 404

    def test_rebind_onto_assistant_conflicts(self, client, test_db_session, tmp_path):
        assistant, _ = self._assistant_and_base(test_db_session, tmp_path)
        other = _add_llm(test_db_session, name="OtherAssistant", is_attached_to_kb=True)
        test_db_session.commit()
        resp = client.post(f"/erudi/llms/{assistant.id}/rebind", json={"new_base_llm_id": other.id})
        assert resp.status_code == 409

    def test_rebind_onto_undownloaded_conflicts(self, client, test_db_session, tmp_path):
        assistant, _ = self._assistant_and_base(test_db_session, tmp_path)
        remote = _add_llm(test_db_session, name="Remote", local=0)
        test_db_session.commit()
        resp = client.post(
            f"/erudi/llms/{assistant.id}/rebind", json={"new_base_llm_id": remote.id}
        )
        assert resp.status_code == 409

    def test_rebind_onto_missing_weights_conflicts(self, client, test_db_session, tmp_path):
        assistant, _ = self._assistant_and_base(test_db_session, tmp_path)
        no_weights = _add_llm(test_db_session, name="NoWeights", link=str(tmp_path / "missing"))
        test_db_session.commit()
        resp = client.post(
            f"/erudi/llms/{assistant.id}/rebind",
            json={"new_base_llm_id": no_weights.id},
        )
        assert resp.status_code == 409

    def test_rebind_failure_maps_to_database_exception(self, client, test_db_session, tmp_path):
        assistant, base = self._assistant_and_base(test_db_session, tmp_path)
        with patch.object(Llm_Repository, "update", side_effect=RuntimeError("constraint")):
            resp = client.post(
                f"/erudi/llms/{assistant.id}/rebind",
                json={"new_base_llm_id": base.id},
            )
        assert resp.status_code == 500


# =====================================================================
# INTEGRATION - download routes error mapping
# =====================================================================


@pytest.mark.integration
class TestDownloadRouteErrors:
    def test_download_generic_failure_maps_to_500(self, client, test_db_session):
        remote = _add_llm(test_db_session, local=0, link="org/model")
        test_db_session.commit()
        with patch.object(llm_endpoints, "_start_download", side_effect=RuntimeError("disk full")):
            resp = client.post(f"/erudi/llms/{remote.id}/download")
        assert resp.status_code == 500

    def test_hf_download_generic_failure_maps_to_500(self, client):
        with patch.object(llm_endpoints, "_start_download", side_effect=RuntimeError("disk full")):
            resp = client.post("/erudi/llms/download/huggingface", json={"link": "org/model"})
        assert resp.status_code == 500

    def test_cancel_generic_failure_maps_to_500(self, client):
        with patch.object(llm_endpoints, "cancel_download_job", side_effect=RuntimeError("boom")):
            resp = client.post("/erudi/llms/downloads/1/cancel")
        assert resp.status_code == 500


# =====================================================================
# INTEGRATION - download status polling cleanup
# =====================================================================


@pytest.mark.integration
class TestDownloadStatusCleanup:
    def test_failed_job_cleans_temp_llm_and_files(self, client, test_db_session):
        llm = _add_llm(test_db_session, local=2)
        job = _add_job(test_db_session, llm.id, status="failed")
        with patch.object(Download_Job_Repository, "cleanup_job_files") as cleanup:
            resp = client.get(f"/erudi/llms/downloads/{job.id}/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "failed"
        cleanup.assert_called_once()
        assert test_db_session.query(Llm).filter(Llm.id == llm.id).count() == 0

    def test_failed_job_with_missing_llm_is_already_cleaned_up(self, client, test_db_session):
        """#509: cleanup already ran once (e.g. by cancel_download_job), so a
        missing LLM row on a terminal job is normal, not an error. Polling it
        again must keep returning the job, not a permanent 404.
        """
        job = _add_job(test_db_session, None, status="failed")
        with patch.object(Download_Job_Repository, "cleanup_job_files") as cleanup:
            resp = client.get(f"/erudi/llms/downloads/{job.id}/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "failed"
        cleanup.assert_not_called()

    def test_cancelled_job_with_missing_llm_is_already_cleaned_up(self, client, test_db_session):
        """#509: cancel_download_job already deletes the temp LLM and cleans up
        files before the job ever reaches `cancelled`; Postgres then nulls
        `local_model_id` through the FK's ON DELETE SET NULL. The status
        endpoint must not treat that as `ModelNotFoundException`.
        """
        job = _add_job(test_db_session, None, status="cancelled")
        with patch.object(Download_Job_Repository, "cleanup_job_files") as cleanup:
            resp = client.get(f"/erudi/llms/downloads/{job.id}/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"
        cleanup.assert_not_called()

    def test_cancel_then_poll_twice_both_succeed(self, client, test_db_session):
        """#509 reproduction: cancel a download through the real API, then poll
        its status endpoint twice. Both polls must succeed with `cancelled`,
        not 404 `MODEL_NOT_FOUND` / "LLM None" on the second poll.
        """
        llm = _add_llm(test_db_session, local=2)
        job = _add_job(test_db_session, llm.id, status="running")

        with patch.object(Download_Job_Repository, "cleanup_job_files"):
            cancel_resp = client.post(f"/erudi/llms/downloads/{job.id}/cancel")
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["status"] == "cancelled"

        first = client.get(f"/erudi/llms/downloads/{job.id}/status")
        second = client.get(f"/erudi/llms/downloads/{job.id}/status")

        assert first.status_code == 200
        assert first.json()["status"] == "cancelled"
        assert second.status_code == 200
        assert second.json()["status"] == "cancelled"

    def test_completed_job_marks_llm_ready(self, client, test_db_session):
        llm = _add_llm(test_db_session, local=2)
        job = _add_job(test_db_session, llm.id, status="completed")
        resp = client.get(f"/erudi/llms/downloads/{job.id}/status")
        assert resp.status_code == 200
        test_db_session.refresh(llm)
        assert llm.local == 1

    def test_completed_job_writes_once_not_on_every_poll(self, client, test_db_session):
        # The frontend polls this endpoint every 2s and keeps polling a completed
        # job until the user navigates away. Marking the LLM ready is a write plus
        # a commit from a GET handler, so it has to happen once, not once per
        # tick, for a row that already says local = 1.
        llm = _add_llm(test_db_session, local=2)
        job = _add_job(test_db_session, llm.id, status="completed")

        with patch.object(
            Llm_Repository, "update", autospec=True, side_effect=Llm_Repository.update
        ) as update:
            first = client.get(f"/erudi/llms/downloads/{job.id}/status")
            for _ in range(4):  # the poll keeps going after the job is terminal
                client.get(f"/erudi/llms/downloads/{job.id}/status")

        assert first.status_code == 200
        test_db_session.refresh(llm)
        assert llm.local == 1
        assert update.call_count == 1, (
            f"local=1 written {update.call_count} times across 5 polls; "
            "the completed branch must be guarded on the current value"
        )

    def test_status_generic_failure_maps_to_500(self, client):
        with patch.object(Download_Job_Repository, "get_by_id", side_effect=RuntimeError("boom")):
            resp = client.get("/erudi/llms/downloads/1/status")
        assert resp.status_code == 500

    # ---- polling without a job id ----

    def test_recent_active_running_job_is_returned(self, client, test_db_session):
        llm = _add_llm(test_db_session, local=2)
        _add_job(test_db_session, llm.id, status="running")
        resp = client.get("/erudi/llms/downloads/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"

    def test_no_recent_active_job_404(self, client):
        assert client.get("/erudi/llms/downloads/status").status_code == 404

    def test_recent_job_that_turned_failed_is_cleaned(self, client, test_db_session):
        """Race guard: the job may finalize between poll query and status check."""
        llm = _add_llm(test_db_session, local=2)
        job = _add_job(test_db_session, llm.id, status="failed")
        with patch.object(Download_Job_Repository, "get_most_recent_active", return_value=job):
            with patch.object(Download_Job_Repository, "cleanup_job_files") as cleanup:
                resp = client.get("/erudi/llms/downloads/status")
        assert resp.status_code == 200
        cleanup.assert_called_once()
        assert test_db_session.query(Llm).filter(Llm.id == llm.id).count() == 0

    def test_recent_job_completed_writes_once_not_on_every_poll(self, client, test_db_session):
        """#509: get_download_status_without_jobId's `completed` branch used to
        write `local=1` on every poll (no guard on the current value), unlike
        the by-id endpoint. Same idempotency guard, same expectation here.
        """
        llm = _add_llm(test_db_session, local=2)
        job = _add_job(test_db_session, llm.id, status="completed")

        with patch.object(Download_Job_Repository, "get_most_recent_active", return_value=job):
            with patch.object(
                Llm_Repository, "update", autospec=True, side_effect=Llm_Repository.update
            ) as update:
                first = client.get("/erudi/llms/downloads/status")
                for _ in range(4):
                    client.get("/erudi/llms/downloads/status")

        assert first.status_code == 200
        test_db_session.refresh(llm)
        assert llm.local == 1
        assert update.call_count == 1, (
            f"local=1 written {update.call_count} times across 5 polls; "
            "the completed branch must be guarded on the current value"
        )

    def test_recent_job_that_turned_completed_marks_ready(self, client, test_db_session):
        llm = _add_llm(test_db_session, local=2)
        job = _add_job(test_db_session, llm.id, status="completed")
        with patch.object(Download_Job_Repository, "get_most_recent_active", return_value=job):
            resp = client.get("/erudi/llms/downloads/status")
        assert resp.status_code == 200
        test_db_session.refresh(llm)
        assert llm.local == 1

    def test_recent_status_generic_failure_maps_to_500(self, client):
        with patch.object(
            Download_Job_Repository,
            "get_most_recent_active",
            side_effect=RuntimeError("boom"),
        ):
            resp = client.get("/erudi/llms/downloads/status")
        assert resp.status_code == 500


# =====================================================================
# UNIT - repository: cleanup_job_files + progress updater thread
# =====================================================================


@pytest.mark.unit
class TestCleanupJobFiles:
    # NOTE: the test name must not contain the substring 'temp' - pytest embeds
    # it in tmp_path, and the fallback sweep triggers only when the job's link
    # does NOT contain 'temp'.
    def test_removes_fallback_and_final_dirs(self, monkeypatch, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        monkeypatch.setattr(config, "LLM_DIR", models_dir)
        # temp link WITHOUT 'temp' in its path triggers the fallback sweep
        odd_temp = tmp_path / "partial"
        odd_temp.mkdir()
        fallback = models_dir / "temp_7"
        fallback.mkdir()
        final = tmp_path / "final"
        final.mkdir()
        job = SimpleNamespace(
            id=1,
            local_model_id=7,
            temp_local_model_link=str(odd_temp),
            final_local_model_link=str(final),
        )

        Download_Job_Repository(db=MagicMock()).cleanup_job_files(job)

        assert not odd_temp.exists()
        assert not fallback.exists()
        assert not final.exists()

    def test_empty_links_are_noop(self):
        job = SimpleNamespace(
            id=1, local_model_id=None, temp_local_model_link="", final_local_model_link=""
        )
        Download_Job_Repository(db=MagicMock()).cleanup_job_files(job)

    def test_undeletable_temp_dir_is_reported_not_swallowed(self, monkeypatch, tmp_path):
        """A cancel that cannot delete the partial files must say so.

        Cancellation is signalled, not awaited, so on Windows the downloader may
        still hold the .gguf open -- rmtree fails, ignore_errors discards the
        failure, and the log claims success. Observed on a cancelled 1B
        download: 230 MB of temp_651 survived while the log read "Cancelled
        download job 2 and deleted temp LLM 651".
        """
        stuck = tmp_path / "partial"
        stuck.mkdir()
        (stuck / "model.gguf").write_bytes(b"\x00" * 2048)

        # Simulate the Windows behaviour: rmtree cannot remove the open file.
        monkeypatch.setattr(llm_repo_mod.shutil, "rmtree", lambda *a, **kw: None)
        monkeypatch.setattr(llm_repo_mod.time, "sleep", lambda _s: None)

        job = SimpleNamespace(
            id=9,
            local_model_id=9,
            temp_local_model_link=str(stuck),
            final_local_model_link="",
        )

        with patch.object(llm_repo_mod.logger, "warning") as warn:
            Download_Job_Repository(db=MagicMock()).cleanup_job_files(job)

        assert warn.called, "surviving bytes must be reported, not swallowed"
        message = warn.call_args[0][0]
        assert str(stuck) in message  # names the path
        assert "2048 bytes" in message  # and how much is still there

    def test_remove_tree_reporting_returns_zero_when_the_tree_goes(self, tmp_path):
        doomed = tmp_path / "gone"
        doomed.mkdir()
        (doomed / "f.bin").write_bytes(b"\x00" * 16)

        assert llm_repo_mod.remove_tree_reporting(doomed) == 0
        assert not doomed.exists()

    def test_remove_tree_reporting_is_a_noop_on_a_missing_path(self, tmp_path):
        assert llm_repo_mod.remove_tree_reporting(tmp_path / "never-existed") == 0

    def test_dir_size_bytes_sums_the_tree(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "a.bin").write_bytes(b"\x00" * 100)
        (tmp_path / "sub" / "b.bin").write_bytes(b"\x00" * 23)

        assert llm_repo_mod.dir_size_bytes(tmp_path) == 123
        assert llm_repo_mod.dir_size_bytes(tmp_path / "nope") == 0


@pytest.mark.unit
class TestUpdateDbWithProgress:
    def _session_with(self, monkeypatch, job_row, llm_row):
        session = MagicMock()

        def query(model):
            q = MagicMock()
            q.get.return_value = job_row if model is DownloadJobModel else llm_row
            return q

        session.query.side_effect = query
        monkeypatch.setattr(llm_repo_mod, "SessionLocal", lambda: session)
        return session

    def test_missing_rows_exits_early(self, monkeypatch):
        session = self._session_with(monkeypatch, None, None)
        llm_repo_mod.update_db_with_progress(MagicMock(), 1, 2)
        session.close.assert_called_once()

    def test_finished_tracker_exits_loop_immediately(self, monkeypatch):
        job_row = SimpleNamespace()
        session = self._session_with(monkeypatch, job_row, SimpleNamespace())
        tracker = SimpleNamespace(percent=100.0, should_continue=lambda: True)
        llm_repo_mod.update_db_with_progress(tracker, 1, 2)
        session.close.assert_called_once()

    def test_cancelled_tracker_stops_polling(self, monkeypatch):
        job_row = SimpleNamespace()
        session = self._session_with(monkeypatch, job_row, SimpleNamespace())
        monkeypatch.setattr(llm_repo_mod.time, "sleep", lambda s: None)
        tracker = SimpleNamespace(percent=10.0, should_continue=lambda: False)
        llm_repo_mod.update_db_with_progress(tracker, 1, 2)
        session.commit.assert_not_called()
        session.close.assert_called_once()

    def test_failure_marks_job_failed_and_cleans_files(self, monkeypatch, tmp_path):
        temp_dir = tmp_path / "temp_dl"
        temp_dir.mkdir()
        final_dir = tmp_path / "final_dl"
        final_dir.mkdir()
        job_row = SimpleNamespace(
            status="running",
            error_message=None,
            temp_local_model_link=str(temp_dir),
            final_local_model_link=str(final_dir),
        )
        llm_row = SimpleNamespace()
        session = self._session_with(monkeypatch, job_row, llm_row)
        monkeypatch.setattr(llm_repo_mod.time, "sleep", lambda s: None)
        monkeypatch.setattr(config, "LLM_DIR", tmp_path)

        class ExplodingTracker:
            @property
            def percent(self):
                raise RuntimeError("tracker corrupted")

        llm_repo_mod.update_db_with_progress(ExplodingTracker(), 1, 2)

        assert job_row.status == "failed"
        assert "tracker corrupted" in job_row.error_message
        assert not temp_dir.exists()
        assert not final_dir.exists()
        session.delete.assert_called_once_with(llm_row)
        session.close.assert_called_once()


# =====================================================================
# UNIT - services: offline classifier, runnability gate, download pipeline
# =====================================================================


@pytest.mark.unit
class TestOfflineErrorClassifier:
    def test_requests_connection_error_is_offline(self):
        from requests.exceptions import ConnectionError as RCE

        assert llm_services._is_offline_download_error(RCE("boom")) is True

    def test_marker_in_message_is_offline(self):
        assert (
            llm_services._is_offline_download_error(RuntimeError("Max retries exceeded with url"))
            is True
        )

    def test_marker_in_cause_chain_is_offline(self):
        inner = OSError("network is unreachable")
        outer = RuntimeError("download failed")
        outer.__cause__ = inner
        assert llm_services._is_offline_download_error(outer) is True

    def test_unrelated_error_is_not_offline(self):
        assert llm_services._is_offline_download_error(ValueError("bad json")) is False


@pytest.mark.unit
class TestAssertRunnable:
    def test_known_broken_quant_is_rejected(self, monkeypatch):
        monkeypatch.setattr(
            config,
            "LLM_Engine",
            SimpleNamespace(is_runnable=lambda link: False, __name__="CPU_Engine"),
        )
        with pytest.raises(UnsupportedPlatformException):
            llm_services._assert_runnable("org/broken-quant")

    def test_runnable_quant_passes(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_Engine", SimpleNamespace(is_runnable=lambda link: True))
        llm_services._assert_runnable("org/fine-quant")

    def test_nemotron_v1_quant_is_rejected_on_mlx(self, monkeypatch):
        """#506: the download gate refuses a Nemotron-NAS v1 quant up front
        because MLX_Engine.KNOWN_BROKEN lists it -- its tokenizer_config.json
        declares the abstract PreTrainedTokenizer, which transformers cannot
        instantiate."""
        from src.engines.mlx_engine import MLX_Engine

        monkeypatch.setattr(config, "LLM_Engine", MLX_Engine)
        with pytest.raises(UnsupportedPlatformException):
            llm_services._assert_runnable("mlx-community/Llama-3_3-Nemotron-Super-49B-v1-mlx-4bit")


class _FakeFs:
    """HfFileSystem stand-in: writes a small file for every requested path."""

    def get_file(self, remote, dest, callback):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"\x00" * 16)
        callback.relative_update(16)


def _fake_api(files):
    api = MagicMock()
    api.repo_info.return_value = SimpleNamespace(
        siblings=[SimpleNamespace(rfilename=f, size=16) for f in files],
        tags=["mlx"],
        library_name="mlx",
    )
    api.list_repo_files.return_value = list(files)
    return api


@pytest.mark.unit
class TestDownloadLlmPipeline:
    async def test_non_gguf_download_moves_to_final_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            config,
            "LLM_Engine",
            SimpleNamespace(is_runnable=lambda link: True, USES_GGUF=False, FORMAT_TAG="mlx"),
        )
        files = ["config.json", "model-00001.safetensors"]
        monkeypatch.setattr(llm_services, "HfApi", lambda token=None: _fake_api(files))
        monkeypatch.setattr(llm_services, "HfFileSystem", lambda token=None: _FakeFs())

        # ETA monitoring is covered by test_llms.py; skip its 5s minimum sleep here.
        async def instant_eta(self, interval=20.0):
            return None

        monkeypatch.setattr(llm_services.DownloadTracker, "monitor_eta", instant_eta)
        temp_dir = tmp_path / "temp_1"
        final_dir = tmp_path / "1"

        await llm_services.download_llm(
            model_link="org/model",
            model_id=1,
            temp_save_dir=str(temp_dir),
            final_save_dir=str(final_dir),
            job_id=None,
        )

        assert (final_dir / "config.json").exists()
        assert (final_dir / "model-00001.safetensors").exists()
        assert not temp_dir.exists()  # moved, not copied

    async def test_gated_repo_maps_to_unsupported_platform(self, monkeypatch, tmp_path):
        from huggingface_hub.errors import GatedRepoError

        monkeypatch.setattr(
            config,
            "LLM_Engine",
            SimpleNamespace(is_runnable=lambda link: True, USES_GGUF=False, FORMAT_TAG="mlx"),
        )
        api = MagicMock()
        api.repo_info.side_effect = GatedRepoError("401 gated", response=MagicMock(status_code=401))
        monkeypatch.setattr(llm_services, "HfApi", lambda token=None: api)
        monkeypatch.setattr(llm_services, "HfFileSystem", lambda token=None: _FakeFs())

        with pytest.raises(UnsupportedPlatformException):
            await llm_services.download_llm(
                model_link="org/gated",
                model_id=1,
                temp_save_dir=str(tmp_path / "t"),
                final_save_dir=str(tmp_path / "f"),
            )

    async def test_offline_failure_maps_to_hf_api_exception(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            config,
            "LLM_Engine",
            SimpleNamespace(is_runnable=lambda link: True, USES_GGUF=False, FORMAT_TAG="mlx"),
        )
        api = MagicMock()
        api.repo_info.side_effect = OSError("Network is unreachable")
        monkeypatch.setattr(llm_services, "HfApi", lambda token=None: api)
        monkeypatch.setattr(llm_services, "HfFileSystem", lambda token=None: _FakeFs())

        with pytest.raises(HuggingFaceAPIException):
            await llm_services.download_llm(
                model_link="org/model",
                model_id=1,
                temp_save_dir=str(tmp_path / "t"),
                final_save_dir=str(tmp_path / "f"),
            )

    async def test_unrelated_failure_propagates(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            config,
            "LLM_Engine",
            SimpleNamespace(is_runnable=lambda link: True, USES_GGUF=False, FORMAT_TAG="mlx"),
        )
        api = MagicMock()
        api.repo_info.side_effect = ValueError("bad metadata")
        monkeypatch.setattr(llm_services, "HfApi", lambda token=None: api)
        monkeypatch.setattr(llm_services, "HfFileSystem", lambda token=None: _FakeFs())

        with pytest.raises(ValueError, match="bad metadata"):
            await llm_services.download_llm(
                model_link="org/model",
                model_id=1,
                temp_save_dir=str(tmp_path / "t"),
                final_save_dir=str(tmp_path / "f"),
            )

    def test_tracker_registry_roundtrip(self):
        tracker = llm_services.DownloadTracker()
        llm_services._register_tracker(777, tracker)
        assert llm_services.get_active_download_tracker(777) is tracker
        llm_services._unregister_tracker(777)
        assert llm_services.get_active_download_tracker(777) is None

    def test_tracker_cancel_stops_continuation(self):
        tracker = llm_services.DownloadTracker()
        assert tracker.should_continue() is True
        tracker.cancel()
        assert tracker.should_continue() is False
