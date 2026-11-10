"""Cancelling a download must stop the transfer, not merely flag it (#377).

Cancellation is signal-and-check by design (see #372): `cancel_download_job`
flips `DownloadTracker._cancelled` and returns. Before this fix nothing on the
download side ever looked at that flag until every shard had been transferred
to the end, so a 3 GB model cancelled at 25 % kept downloading for minutes,
and the endpoint then logged "completed successfully" for a cancelled job.

Four guarantees are pinned here:
  1. the shard loop does not start a shard once the tracker is cancelled;
  2. the per-chunk fsspec callback aborts the in-flight transfer;
  3. the endpoint's completion log reflects the real outcome;
  4. a cancel racing the download task leaves the job cancelled, with one
     INFO and no ERROR, on a real database (#661).

The cancel endpoint deletes the files only once the transfer stopped (#690):
see test_download_stop.py.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.core import config
from src.domains.llms import endpoints as llm_endpoints
from src.domains.llms import repository as llm_repo_mod
from src.domains.llms import services as llm_services
from src.domains.llms.services import DownloadCancelled, DownloadTracker, make_callback
from src.entities.DownloadJob import DownloadJobModel
from tests._helpers import FakeDownloadQuery

SHARDS = [f"model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)]


def _tracker(total: int) -> DownloadTracker:
    tracker = DownloadTracker()
    tracker.total_bytes = total
    return tracker


# =====================================================================
# UNIT - the shard loop stops scheduling once cancelled
# =====================================================================


@pytest.mark.unit
class TestShardLoopHonoursCancel:
    async def test_remaining_shards_are_never_requested_after_cancel(self, tmp_path):
        tracker = _tracker(3 * 16)
        callback = make_callback(tracker)
        requested: list[str] = []

        class Fs:
            def get_file(self, remote, dest, cb):
                # The cancel lands while this (first) shard is in flight; it
                # finishes on its own, the others must never be started.
                requested.append(remote)
                tracker.cancel()
                Path(dest).write_bytes(b"\x00" * 16)

        # One worker so the shards run strictly one after the other; the
        # assertion is about scheduling, not about how many threads there are.
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))

        with pytest.raises(DownloadCancelled):
            await llm_services.download_files_concurrent(
                Fs(), callback, [("org/m", s) for s in SHARDS], str(tmp_path), tracker=tracker
            )

        assert requested == ["org/m/" + SHARDS[0]]
        assert not (tmp_path / SHARDS[1]).exists()
        assert not (tmp_path / SHARDS[2]).exists()

    async def test_already_cancelled_tracker_requests_nothing(self, tmp_path):
        tracker = _tracker(16)
        tracker.cancel()
        fs = MagicMock()

        with pytest.raises(DownloadCancelled):
            await llm_services.download_files_concurrent(
                fs, make_callback(tracker), [("org/m", SHARDS[0])], str(tmp_path), tracker=tracker
            )

        fs.get_file.assert_not_called()


# =====================================================================
# UNIT - the per-chunk callback aborts the in-flight transfer
# =====================================================================


@pytest.mark.unit
class TestChunkCallbackHonoursCancel:
    def test_callback_counts_until_cancel_then_raises(self):
        tracker = _tracker(100)
        callback = make_callback(tracker)

        callback.relative_update(10)
        assert tracker.downloaded_bytes == 10

        tracker.cancel()
        with pytest.raises(DownloadCancelled):
            callback.relative_update(10)

    async def test_in_flight_transfer_exits_promptly(self, tmp_path):
        tracker = _tracker(100)
        served: list[int] = []

        class Fs:
            def get_file(self, remote, dest, cb):
                for i in range(100):
                    served.append(i)
                    if i == 2:
                        tracker.cancel()
                    cb.relative_update(1)

        with pytest.raises(DownloadCancelled):
            await llm_services.download_files_concurrent(
                Fs(), make_callback(tracker), [("org/m", SHARDS[0])], str(tmp_path), tracker=tracker
            )

        # Three chunks were served (the third is the one that noticed), not 100.
        assert served == [0, 1, 2]


# =====================================================================
# UNIT - download_llm hands the abort to the download task
# =====================================================================


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
class TestDownloadLlmCancelledMidTransfer:
    async def test_cancel_mid_shard_stops_transfer_and_skips_finalization(
        self, monkeypatch, tmp_path, caplog
    ):
        monkeypatch.setattr(
            config,
            "LLM_Engine",
            SimpleNamespace(is_runnable=lambda link: True, USES_GGUF=False, FORMAT_TAG="mlx"),
        )
        files = ["config.json", *SHARDS]
        monkeypatch.setattr(llm_services, "HfApi", lambda token=None: _fake_api(files))

        async def instant_eta(self, interval=20.0):
            return None

        monkeypatch.setattr(DownloadTracker, "monitor_eta", instant_eta)
        # The DB progress thread is covered in test_llms_gaps; keep it out of here.
        monkeypatch.setattr(llm_services, "update_db_with_progress", lambda *a, **k: None)

        requested: list[str] = []

        class Fs:
            def get_file(self, remote, dest, cb):
                requested.append(remote)
                Path(dest).parent.mkdir(parents=True, exist_ok=True)
                Path(dest).write_bytes(b"\x00" * 16)
                if remote.endswith(SHARDS[0]):
                    # User hits cancel while the first shard is streaming.
                    llm_services.get_active_download_tracker(42).cancel()
                cb.relative_update(16)

        monkeypatch.setattr(llm_services, "HfFileSystem", lambda token=None: Fs())
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))

        temp_dir = tmp_path / "temp_1"
        final_dir = tmp_path / "1"
        with caplog.at_level(logging.INFO, logger="erudi"):
            # The download task takes the cancel from this, never from the job
            # row, which the cancel endpoint commits only after its cleanup.
            with pytest.raises(DownloadCancelled):
                await llm_services.download_llm(
                    model_link="org/model",
                    model_id=1,
                    temp_save_dir=str(temp_dir),
                    final_save_dir=str(final_dir),
                    job_id=42,
                )

        # Only config.json and the in-flight shard were ever requested.
        assert requested == ["org/model/config.json", "org/model/" + SHARDS[0]]
        # Nothing was promoted to the final directory.
        assert not any(final_dir.iterdir())
        messages = [r.message for r in caplog.records]
        assert not any("All shards downloaded" in m for m in messages)
        # Said once, by the download task.
        assert not any("cancelled" in m for m in messages)
        # Tracker is unregistered on the way out, like every other exit path.
        assert llm_services.get_active_download_tracker(42) is None


# =====================================================================
# UNIT - the endpoint logs the real outcome
# =====================================================================


def _session_with(monkeypatch, job_row, llm_row):
    session = MagicMock()

    session.query.side_effect = lambda model: FakeDownloadQuery(model, job_row, llm_row)
    monkeypatch.setattr(llm_endpoints, "SessionLocal", lambda: session)
    return session


@pytest.mark.unit
class TestRunDownloadTaskOutcomeLog:
    def _arm(self, monkeypatch, *, ends_as: str):
        job = SimpleNamespace(status="pending", error_message=None, updated_at=None, progress=0.0)
        llm = SimpleNamespace(link="/models/1", model_metadata=None, local=2)
        session = _session_with(monkeypatch, job, llm)

        async def download(**kwargs):
            # The cancel endpoint flips the row while the transfer runs.
            job.status = ends_as

        monkeypatch.setattr(llm_endpoints, "download_llm", download)
        monkeypatch.setattr(config, "LLM_Engine", SimpleNamespace())  # no validator
        monkeypatch.setattr(llm_endpoints, "measure_dir_size_bytes", lambda p: 0)
        for name in ("detect_supports_tools", "detect_wire_tools", "detect_supports_vision"):
            monkeypatch.setattr(llm_repo_mod, name, lambda p: None)
        return job, session

    def test_cancelled_job_is_not_logged_as_completed(self, monkeypatch, caplog):
        job, _ = self._arm(monkeypatch, ends_as="cancelled")

        with caplog.at_level(logging.INFO, logger="erudi"):
            llm_endpoints._run_download_task("org/model", 1, "/tmp/t", "/tmp/f", job_id=7)

        messages = [r.message for r in caplog.records]
        assert not any("completed successfully" in m for m in messages)
        assert any("job 7" in m and "cancelled" in m for m in messages)
        assert job.status == "cancelled"

    def test_completed_job_still_logs_success(self, monkeypatch, caplog):
        job, _ = self._arm(monkeypatch, ends_as="running")

        with caplog.at_level(logging.INFO, logger="erudi"):
            llm_endpoints._run_download_task("org/model", 1, "/tmp/t", "/tmp/f", job_id=8)

        messages = [r.message for r in caplog.records]
        assert any("Download job 8 completed successfully" in m for m in messages)
        assert job.status == "completed"


# =====================================================================
# INTEGRATION - a cancel racing the download task, on a real database
# =====================================================================


class _RealDownload:
    """The download task, the cancel endpoint's body and Postgres, real; the
    Hub faked. #661: the cancel endpoint flushes ``cancelled`` and commits
    only after its ~400 ms cleanup, so until then the task reads ``running``
    (READ COMMITTED), and its write of ``failed`` waits on the row lock and
    lands over the cancel once that commits."""

    def __init__(self, monkeypatch, engine, tmp_path):
        from sqlalchemy.orm import sessionmaker

        from src.entities.Llm import Llm

        self.monkeypatch = monkeypatch
        # The production session factory's settings, on the test cluster.
        self.Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
        monkeypatch.setattr(llm_endpoints, "SessionLocal", self.Session)
        self.temp_dir = tmp_path / "temp_1"
        self.final_dir = tmp_path / "1"
        with self.Session() as s:
            llm = Llm(name="Qwen2.5 7B", local=2, link="org/model", type="text")
            s.add(llm)
            s.flush()
            job = DownloadJobModel(
                remote_model_id="org/model",
                remote_model_link="org/model",
                local_model_id=llm.id,
                temp_local_model_link=str(self.temp_dir),
                final_local_model_link=str(self.final_dir),
                status="pending",
            )
            s.add(job)
            s.commit()
            self.llm_id, self.job_id = llm.id, job.id
        self.gate_calls: list[str] = []
        # Runs in the transfer, right after the cancel: the transfer has not
        # seen it yet (that happens at its next chunk).
        self.on_cancelled = None
        self.transfer_errors: list[BaseException] = []
        self.flushed = threading.Event()
        self.canceller = None
        self.cancel_errors: list[Exception] = []
        monkeypatch.setattr(
            config,
            "LLM_Engine",
            SimpleNamespace(
                is_runnable=lambda link: True,
                USES_GGUF=False,
                FORMAT_TAG="mlx",
                validate_local_artifact=self.gate,
            ),
        )
        monkeypatch.setattr(llm_services, "update_db_with_progress", lambda *a, **k: None)

        async def instant_eta(self, interval=20.0):
            return None

        monkeypatch.setattr(DownloadTracker, "monitor_eta", instant_eta)
        repo = llm_repo_mod.Download_Job_Repository
        real_cleanup = repo.cleanup_job_files
        real_mark_cancelled = getattr(repo, "mark_cancelled_if_active", None)
        flushed = self.flushed

        def mark_cancelled_if_active(self, job_id):
            rows = real_mark_cancelled(self, job_id)
            flushed.set()
            return rows

        def slow_cleanup(self, job):
            # The tester's cancel committed ~400 ms after its flush.
            time.sleep(0.4)
            real_cleanup(self, job)

        monkeypatch.setattr(repo, "mark_cancelled_if_active", mark_cancelled_if_active)
        monkeypatch.setattr(repo, "cleanup_job_files", slow_cleanup)

    def gate(self, path):
        from src.core.exceptions import InvalidInputException

        self.gate_calls.append(path)
        raise InvalidInputException(
            "The downloaded files for this model are incomplete or invalid "
            "(no weights file was found)"
        )

    def cancel_now(self):
        """The user's cancel, on its own session and thread; returns once
        the endpoint has flushed ``cancelled`` (not committed yet). The
        endpoint writes at once, before it waits for the transfer to stop."""
        from src.domains.llms.repository import Download_Job_Repository, Llm_Repository

        def run():
            with self.Session() as db:
                try:
                    llm_services.cancel_download_job(
                        self.job_id, Download_Job_Repository(db), Llm_Repository(db), db
                    )
                except Exception as exc:
                    # The cancel endpoint rolls back and answers an error.
                    db.rollback()
                    self.cancel_errors.append(exc)

        self.canceller = threading.Thread(target=run)
        self.canceller.start()
        assert self.flushed.wait(5)

    def hub_cancelled_mid_shard(self):
        files = ["config.json", *SHARDS]
        self.monkeypatch.setattr(llm_services, "HfApi", lambda token=None: _fake_api(files))
        harness = self

        class Fs:
            def get_file(self, remote, dest, cb):
                Path(dest).parent.mkdir(parents=True, exist_ok=True)
                Path(dest).write_bytes(b"\x00" * 16)
                if remote.endswith(SHARDS[0]):
                    try:
                        harness.cancel_now()
                        if harness.on_cancelled is not None:
                            harness.on_cancelled()
                    except BaseException as exc:
                        # A cancelled job ends as cancelled whatever its
                        # threads raised: keep the harness's own failures.
                        harness.transfer_errors.append(exc)
                        raise
                cb.relative_update(16)

        self.monkeypatch.setattr(llm_services, "HfFileSystem", lambda token=None: Fs())

    def hub_recording(self, requested):
        """A Hub whose transfers only record what was asked of them."""
        files = ["config.json", *SHARDS]
        self.monkeypatch.setattr(llm_services, "HfApi", lambda token=None: _fake_api(files))

        class Fs:
            def get_file(self, remote, dest, cb):
                requested.append(remote)
                Path(dest).parent.mkdir(parents=True, exist_ok=True)
                Path(dest).write_bytes(b"\x00" * 16)
                cb.relative_update(16)

        self.monkeypatch.setattr(llm_services, "HfFileSystem", lambda token=None: Fs())

    def run(self):
        llm_endpoints._run_download_task(
            "org/model", self.llm_id, str(self.temp_dir), str(self.final_dir), self.job_id
        )
        if self.canceller is not None:
            self.canceller.join(15)
            assert not self.canceller.is_alive()
        assert self.transfer_errors == []

    def job(self):
        with self.Session() as s:
            return s.get(DownloadJobModel, self.job_id)

    def drop(self):
        from src.entities.Llm import Llm

        with self.Session() as s:
            for model, pk in ((DownloadJobModel, self.job_id), (Llm, self.llm_id)):
                row = s.get(model, pk)
                if row is not None:
                    s.delete(row)
            s.commit()


@pytest.fixture
def real_download(monkeypatch, test_db_engine, tmp_path):
    harness = _RealDownload(monkeypatch, test_db_engine, tmp_path)
    yield harness
    harness.drop()


def _errors(caplog):
    return [r for r in caplog.records if r.levelno >= logging.ERROR]


async def _transferred(**kwargs):
    """A transfer that completed (the files are the gate's business)."""
    return kwargs["temp_save_dir"]


@pytest.mark.integration
class TestCancelRacingTheDownloadTask:
    def test_a_cancel_during_the_transfer_stays_cancelled_and_is_no_error(
        self, real_download, caplog
    ):
        real_download.hub_cancelled_mid_shard()

        with caplog.at_level(logging.INFO, logger="erudi"):
            real_download.run()

        job = real_download.job()
        assert job.status == "cancelled"
        assert job.error_message is None
        assert real_download.gate_calls == []
        assert _errors(caplog) == []
        said = [
            r.levelno
            for r in caplog.records
            if f"Download job {real_download.job_id} was cancelled" in r.getMessage()
        ]
        assert said == [logging.INFO]

    def test_a_cancel_before_the_task_starts_downloads_nothing(self, real_download, caplog):
        """The cancel lands while the job is still pending (no transfer, no
        tracker): the task must not put the row back to running and fetch the
        model anyway, leaving files nobody owns."""
        from src.domains.llms.repository import Download_Job_Repository, Llm_Repository

        requested: list[str] = []
        real_download.hub_recording(requested)
        with real_download.Session() as db:
            llm_services.cancel_download_job(
                real_download.job_id, Download_Job_Repository(db), Llm_Repository(db), db
            )

        with caplog.at_level(logging.INFO, logger="erudi"):
            real_download.run()

        assert requested == []
        assert real_download.job().status == "cancelled"
        assert not real_download.temp_dir.exists()
        assert _errors(caplog) == []

    def test_a_cancel_racing_the_start_stops_the_task_at_once(
        self, real_download, monkeypatch, caplog
    ):
        """The cancel lands as the task claims its row: the task's tracker is
        already registered, so the cancel signals it, and the task stops
        without transferring; the cancel does not wait out its 10 s."""
        requested: list[str] = []
        real_download.hub_recording(requested)
        real_claim = llm_endpoints._claim_job

        def claim_racing_a_cancel(session, job_id):
            assert llm_services.get_active_download_tracker(job_id) is not None
            real_download.cancel_now()
            return real_claim(session, job_id)

        monkeypatch.setattr(llm_endpoints, "_claim_job", claim_racing_a_cancel)

        started = time.monotonic()
        with caplog.at_level(logging.INFO, logger="erudi"):
            real_download.run()

        assert time.monotonic() - started < 5
        assert requested == []
        assert real_download.job().status == "cancelled"
        assert _errors(caplog) == []

    def test_a_cancel_of_a_job_that_just_ended_is_refused_and_touches_nothing(
        self, real_download, monkeypatch, tmp_path
    ):
        """The job completes between the cancel's status check and its write:
        the write finds no active row, the cancel is refused, the model and
        its files stay."""
        from src.core.exceptions import StateConflictException
        from src.domains.llms.repository import Download_Job_Repository, Llm_Repository

        real_download.final_dir.mkdir(parents=True)
        (real_download.final_dir / "model.gguf").write_bytes(b"GGUF")
        real_get = Llm_Repository.get_by_id

        def get_while_the_job_completes(self, llm_id):
            llm = real_get(self, llm_id)
            with real_download.Session() as other:
                other.get(DownloadJobModel, real_download.job_id).status = "completed"
                other.commit()
            return llm

        monkeypatch.setattr(Llm_Repository, "get_by_id", get_while_the_job_completes)

        with real_download.Session() as db:
            with pytest.raises(StateConflictException):
                llm_services.cancel_download_job(
                    real_download.job_id, Download_Job_Repository(db), Llm_Repository(db), db
                )

        assert real_download.job().status == "completed"
        assert (real_download.final_dir / "model.gguf").exists()

    def test_a_status_poll_during_the_cancel_finds_nothing_to_clean(
        self, real_download, monkeypatch, caplog
    ):
        """The frontend keeps polling while the cancel request waits for the
        transfer to stop: the poll must not run a second cleanup (two rmtree,
        two DELETEs of the model row in opposite lock orders)."""
        from src.domains.llms.repository import Download_Job_Repository, Llm_Repository

        poll_cleanups: list[int] = []
        polled: list[bool] = []
        real_cleanup = llm_repo_mod.Download_Job_Repository.cleanup_job_files

        def poll():
            # Wait for the cancel's commit, then poll as the frontend would.
            deadline = time.monotonic() + 5
            while real_download.job().status != "cancelled":
                assert time.monotonic() < deadline
                time.sleep(0.02)
            with real_download.Session() as db:
                calls_before = len(poll_cleanups)

                def counting(self, job):
                    poll_cleanups.append(job.id)
                    real_cleanup(self, job)

                monkeypatch.setattr(Download_Job_Repository, "cleanup_job_files", counting)
                llm_endpoints.get_download_status_by_jobId(
                    real_download.job_id, Llm_Repository(db), Download_Job_Repository(db), db
                )
                monkeypatch.setattr(Download_Job_Repository, "cleanup_job_files", real_cleanup)
                assert len(poll_cleanups) == calls_before
            polled.append(True)

        real_download.on_cancelled = poll
        real_download.hub_cancelled_mid_shard()

        with caplog.at_level(logging.INFO, logger="erudi"):
            real_download.run()

        assert polled == [True]
        assert poll_cleanups == []
        assert real_download.job().status == "cancelled"
        assert _errors(caplog) == []

    def test_a_failure_while_a_cancel_is_in_flight_never_overwrites_it(
        self, real_download, monkeypatch, caplog
    ):
        """The transfer finished; the user cancels while the integrity gate
        runs: the gate's failure is the cancel's doing, not a failed job."""
        monkeypatch.setattr(llm_endpoints, "download_llm", _transferred)

        def gate_racing_a_cancel(path):
            real_download.cancel_now()
            real_download.gate(path)

        monkeypatch.setattr(config.LLM_Engine, "validate_local_artifact", gate_racing_a_cancel)

        with caplog.at_level(logging.INFO, logger="erudi"):
            real_download.run()

        job = real_download.job()
        assert job.status == "cancelled"
        assert job.error_message is None
        assert _errors(caplog) == []
        said = [
            r.levelno
            for r in caplog.records
            if f"Download job {real_download.job_id} was cancelled" in r.getMessage()
        ]
        assert said == [logging.INFO]

    def test_a_cancel_that_failed_after_stopping_the_transfer_still_ends_the_job(
        self, real_download, monkeypatch, caplog
    ):
        """The cancel endpoint stopped the transfer (tracker flagged), then
        failed and rolled back: the job must not stay ``running`` until the
        next boot. The task records the cancel itself."""

        def delete_fails(self, llm):
            raise RuntimeError("database went away")

        monkeypatch.setattr(llm_repo_mod.Llm_Repository, "delete", delete_fails)
        real_download.hub_cancelled_mid_shard()

        with caplog.at_level(logging.INFO, logger="erudi"):
            real_download.run()

        assert len(real_download.cancel_errors) == 1
        assert real_download.job().status == "cancelled"
        assert real_download.gate_calls == []
        assert _errors(caplog) == []

    def test_a_failure_on_a_vanished_job_row_is_still_an_error(
        self, real_download, monkeypatch, caplog
    ):
        """No row to mark failed is not a cancel: the failure is recorded."""
        monkeypatch.setattr(llm_endpoints, "download_llm", _transferred)

        def gate_on_a_vanished_row(path):
            with real_download.Session() as s:
                s.delete(s.get(DownloadJobModel, real_download.job_id))
                s.commit()
            real_download.gate(path)

        monkeypatch.setattr(config.LLM_Engine, "validate_local_artifact", gate_on_a_vanished_row)

        with caplog.at_level(logging.INFO, logger="erudi"):
            real_download.run()

        (error,) = _errors(caplog)
        assert f"Download job {real_download.job_id} failed" in error.getMessage()
        assert error.exc_info is not None

    def test_a_cancel_committed_during_the_finish_is_never_overwritten_by_completed(
        self, real_download, monkeypatch, caplog
    ):
        """The integrity gate passed while the user's cancel ran to its
        commit: the job stays cancelled, and nothing says it completed."""
        monkeypatch.setattr(llm_endpoints, "download_llm", _transferred)

        def gate_passes_while_a_cancel_commits(path):
            real_download.cancel_now()
            real_download.canceller.join(5)

        monkeypatch.setattr(
            config.LLM_Engine, "validate_local_artifact", gate_passes_while_a_cancel_commits
        )

        with caplog.at_level(logging.INFO, logger="erudi"):
            real_download.run()

        assert real_download.cancel_errors == []
        assert real_download.job().status == "cancelled"
        assert _errors(caplog) == []
        messages = [r.getMessage() for r in caplog.records]
        assert not any("completed successfully" in m for m in messages)
        said = [m for m in messages if f"Download job {real_download.job_id} was cancelled" in m]
        assert len(said) == 1

    def test_a_genuine_integrity_failure_is_one_error_and_fails_the_job(
        self, real_download, monkeypatch, caplog
    ):
        monkeypatch.setattr(llm_endpoints, "download_llm", _transferred)

        with caplog.at_level(logging.INFO, logger="erudi"):
            real_download.run()

        job = real_download.job()
        assert job.status == "failed"
        assert "incomplete or invalid" in job.error_message
        assert len(real_download.gate_calls) == 1
        (error,) = _errors(caplog)
        assert f"Download job {real_download.job_id} failed" in error.getMessage()
        assert error.exc_info is not None
