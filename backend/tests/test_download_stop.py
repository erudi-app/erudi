"""A download stops with its app, and a cancelled download leaves nothing (#706, #690).

Two defects share one cause: nothing waited for the transfer threads to stop.

* #706: the backend outlived its app for minutes. The lifespan shutdown never
  stopped the downloads, stopped Postgres under the progress writer, and the
  interpreter's exit then joined the executor threads still streaming the
  model to the end.
* #690 item 8: the cancel endpoint deleted the files right after flagging the
  tracker, while the transfer still held the ``.gguf`` open for one more
  chunk; Windows refused, 1.37 GB stayed on disk and a WARNING reached
  Diagnostics. ``asyncio.gather`` also returned at the first cancelled shard
  while its siblings still wrote.

Pinned here:
  1. a batch of shards ends only once every shard's thread has ended;
  2. the cancel endpoint deletes the files only once the transfer stopped, and
     hands the deletion to the download when the transfer outlives its wait;
     a user cancel records nothing at WARNING or above;
  3. a shutdown interrupts every download (``DownloadInterrupted``), waits for
     the download tasks for a bounded time, leaves the job ``running`` for the
     startup cleanup, and the lifespan does it before ``stop_postgres``.
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
from fastapi import FastAPI

from src.core import api, config
from src.domains.llms import endpoints as llm_endpoints
from src.domains.llms import repository as llm_repo_mod
from src.domains.llms import services as llm_services
from src.domains.llms.services import (
    DownloadCancelled,
    DownloadInterrupted,
    DownloadTracker,
    make_callback,
)
from tests._helpers import FakeDownloadQuery

SHARDS = [f"model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)]


@pytest.fixture(autouse=True)
def _fresh_download_registry(monkeypatch):
    """Every test starts with no download registered and the shutdown not
    begun; monkeypatch restores the module state afterwards."""
    monkeypatch.setattr(llm_services, "_active_trackers", {})
    monkeypatch.setattr(llm_services, "_running_downloads", set())
    monkeypatch.setattr(llm_services, "_downloads_closed", False)


def _tracker(total: int) -> DownloadTracker:
    tracker = DownloadTracker()
    tracker.total_bytes = total
    return tracker


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


def _fake_api(files, size=16):
    api_ = MagicMock()
    api_.repo_info.return_value = SimpleNamespace(
        siblings=[SimpleNamespace(rfilename=f, size=size) for f in files],
        tags=["mlx"],
        library_name="mlx",
    )
    api_.list_repo_files.return_value = list(files)
    return api_


def _mlx_engine(monkeypatch):
    monkeypatch.setattr(
        config,
        "LLM_Engine",
        SimpleNamespace(is_runnable=lambda link: True, USES_GGUF=False, FORMAT_TAG="mlx"),
    )

    async def instant_eta(self, interval=20.0):
        return None

    monkeypatch.setattr(DownloadTracker, "monitor_eta", instant_eta)
    monkeypatch.setattr(llm_services, "update_db_with_progress", lambda *a, **k: None)


class _ChunkedFs:
    """A Hub file system whose one file arrives in slow chunks, like a large
    ``.gguf`` on a slow line: the cancel is only seen at the next chunk."""

    def __init__(self, chunk_seconds: float, chunks: int = 200, on_chunk=None):
        self.chunk_seconds = chunk_seconds
        self.chunks = chunks
        self.on_chunk = on_chunk
        self.closed_at: float | None = None

    def get_file(self, remote, dest, cb):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(dest, "wb") as handle:
                for i in range(self.chunks):
                    time.sleep(self.chunk_seconds)
                    handle.write(b"\x00" * 16)
                    cb.relative_update(16)
                    # After the chunk's own check: a cancel landing here is
                    # only seen at the next chunk, one chunk_seconds later.
                    if self.on_chunk is not None:
                        self.on_chunk(i)
        finally:
            self.closed_at = time.time()


# =====================================================================
# 1. A batch of shards ends once every shard has ended
# =====================================================================


@pytest.mark.unit
class TestShardBatchWaitsForEveryShard:
    async def test_a_cancel_returns_only_once_every_shard_thread_has_ended(self, tmp_path):
        tracker = _tracker(3 * 16)
        ended: list[str] = []
        # The three shards are all in flight when the cancel lands.
        in_flight = threading.Barrier(3, timeout=5)

        class Fs:
            def get_file(self, remote, dest, cb):
                in_flight.wait()
                try:
                    with open(dest, "wb") as handle:
                        if remote.endswith(SHARDS[0]):
                            tracker.cancel()  # the user's cancel, seen at once here
                        else:
                            time.sleep(0.5)  # the chunk in flight on a sibling
                            handle.write(b"\x00" * 16)
                        cb.relative_update(16)
                finally:
                    ended.append(remote)

        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=3))

        with pytest.raises(DownloadCancelled):
            await llm_services.download_files_concurrent(
                Fs(), make_callback(tracker), [("org/m", s) for s in SHARDS], str(tmp_path), tracker
            )

        assert sorted(ended) == sorted(f"org/m/{s}" for s in SHARDS)

    async def test_a_failed_shard_stops_its_siblings_and_its_error_surfaces(self, tmp_path):
        tracker = _tracker(3 * 1000)
        ended: list[str] = []
        sibling_chunks: list[int] = []

        class Fs:
            def get_file(self, remote, dest, cb):
                try:
                    if remote.endswith(SHARDS[0]):
                        time.sleep(0.05)
                        raise ConnectionResetError("connection reset by peer")
                    for _ in range(1000):
                        time.sleep(0.01)
                        sibling_chunks.append(1)
                        cb.relative_update(1)
                finally:
                    ended.append(remote)

        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=3))

        with pytest.raises(ConnectionResetError):
            await llm_services.download_files_concurrent(
                Fs(), make_callback(tracker), [("org/m", s) for s in SHARDS], str(tmp_path), tracker
            )

        assert sorted(ended) == sorted(f"org/m/{s}" for s in SHARDS)
        # The siblings stopped at their next chunk, not at the end of their file.
        assert len(sibling_chunks) < 500
        # A failed shard is not the user's cancel.
        assert not tracker.cancelled


# =====================================================================
# 2. The cancel endpoint deletes the files once the transfer has stopped
# =====================================================================


class _CancelHarness:
    """``cancel_download_job`` over in-memory rows and a real temp tree."""

    def __init__(self, tmp_path, job_id: int):
        self.temp_dir = tmp_path / "temp_7"
        self.final_dir = tmp_path / "7"
        self.job = SimpleNamespace(
            id=job_id,
            status="running",
            local_model_id=7,
            temp_local_model_link=str(self.temp_dir),
            final_local_model_link=str(self.final_dir),
        )
        self.events: list[str] = []
        self.job_repo = MagicMock()
        self.job_repo.get_by_id.return_value = self.job
        self.job_repo.cleanup_job_files.side_effect = self._cleanup
        self.llm_repo = MagicMock()
        self.llm_repo.get_by_id.return_value = SimpleNamespace(id=7, local=2)
        self.db = MagicMock()
        self.result: dict | None = None
        self.thread: threading.Thread | None = None

    def _cleanup(self, job):
        self.events.append("cleanup")
        llm_repo_mod.Download_Job_Repository.cleanup_job_files(MagicMock(), job)

    def cancel_in_background(self):
        def run():
            self.result = llm_services.cancel_download_job(
                self.job.id, self.job_repo, self.llm_repo, self.db
            )

        self.thread = threading.Thread(target=run)
        self.thread.start()

    def join(self):
        assert self.thread is not None
        self.thread.join(15)
        assert not self.thread.is_alive()


@pytest.mark.unit
class TestCancelDeletesOnceTheTransferStopped:
    async def test_the_files_go_only_after_the_transfer_released_them(
        self, monkeypatch, tmp_path, caplog
    ):
        _mlx_engine(monkeypatch)
        harness = _CancelHarness(tmp_path, job_id=51)
        monkeypatch.setattr(llm_services, "HfApi", lambda token=None: _fake_api(["model.bin"]))

        def on_chunk(i):
            if i == 2:
                harness.cancel_in_background()

        fs = _ChunkedFs(chunk_seconds=0.3, on_chunk=on_chunk)
        monkeypatch.setattr(llm_services, "HfFileSystem", lambda token=None: fs)

        with caplog.at_level(logging.INFO, logger="erudi"):
            with pytest.raises(DownloadCancelled):
                await llm_services.download_llm(
                    model_link="org/model",
                    model_id=7,
                    temp_save_dir=str(harness.temp_dir),
                    final_save_dir=str(harness.final_dir),
                    job_id=51,
                )
            await asyncio.to_thread(harness.join)

        assert harness.events == ["cleanup"]
        cleanup_record = next(r for r in caplog.records if "Cleaning up files" in r.getMessage())
        # The file was closed before the cleanup started (Windows refuses otherwise).
        assert fs.closed_at is not None and fs.closed_at <= cleanup_record.created + 0.001
        assert not harness.temp_dir.exists()
        assert not harness.final_dir.exists()
        assert harness.result["status"] == "cancelled"
        assert _warnings(caplog) == []

    async def test_a_transfer_that_outlives_the_wait_removes_its_own_files(
        self, monkeypatch, tmp_path, caplog
    ):
        _mlx_engine(monkeypatch)
        monkeypatch.setattr(llm_services, "CANCEL_STOP_WAIT_SECONDS", 0.2)
        harness = _CancelHarness(tmp_path, job_id=52)
        monkeypatch.setattr(llm_services, "HfApi", lambda token=None: _fake_api(["model.bin"]))

        def on_chunk(i):
            if i == 1:
                harness.cancel_in_background()

        # One chunk outlasts the endpoint's wait.
        fs = _ChunkedFs(chunk_seconds=0.8, on_chunk=on_chunk)
        monkeypatch.setattr(llm_services, "HfFileSystem", lambda token=None: fs)

        with caplog.at_level(logging.INFO, logger="erudi"):
            with pytest.raises(DownloadCancelled):
                await llm_services.download_llm(
                    model_link="org/model",
                    model_id=7,
                    temp_save_dir=str(harness.temp_dir),
                    final_save_dir=str(harness.final_dir),
                    job_id=52,
                )
            await asyncio.to_thread(harness.join)

        # The endpoint answered without touching the files; the download
        # removed them once its transfer had stopped.
        assert harness.events == []
        assert harness.result["status"] == "cancelled"
        assert not harness.temp_dir.exists()
        assert not harness.final_dir.exists()
        assert _warnings(caplog) == []

    def test_a_job_with_no_transfer_running_is_cancelled_without_a_warning(self, tmp_path, caplog):
        harness = _CancelHarness(tmp_path, job_id=53)
        harness.temp_dir.mkdir()

        with caplog.at_level(logging.INFO, logger="erudi"):
            llm_services.cancel_download_job(53, harness.job_repo, harness.llm_repo, harness.db)

        assert harness.events == ["cleanup"]
        assert not harness.temp_dir.exists()
        assert _warnings(caplog) == []


# =====================================================================
# 3. A shutdown interrupts the downloads, before Postgres stops
# =====================================================================


@pytest.mark.unit
class TestShutdownInterruptsDownloads:
    async def test_an_interrupted_download_raises_download_interrupted(self, monkeypatch, tmp_path):
        _mlx_engine(monkeypatch)
        monkeypatch.setattr(llm_services, "HfApi", lambda token=None: _fake_api(["model.bin"]))

        def on_chunk(i):
            if i == 2:
                llm_services.interrupt_active_downloads()

        fs = _ChunkedFs(chunk_seconds=0.05, on_chunk=on_chunk)
        monkeypatch.setattr(llm_services, "HfFileSystem", lambda token=None: fs)

        with pytest.raises(DownloadInterrupted):
            await llm_services.download_llm(
                model_link="org/model",
                model_id=7,
                temp_save_dir=str(tmp_path / "temp_7"),
                final_save_dir=str(tmp_path / "7"),
                job_id=61,
            )
        # The partial files stay for the startup cleanup.
        assert (tmp_path / "temp_7" / "model.bin").exists()
        assert llm_services.get_active_download_tracker(61) is None

    async def test_the_progress_writer_has_stopped_when_the_download_returns(
        self, monkeypatch, tmp_path
    ):
        """The writer commits progress every second: it must be gone before
        the lifespan stops Postgres under it (the two AdminShutdown ERRORs)."""
        _mlx_engine(monkeypatch)
        monkeypatch.setattr(llm_services, "HfApi", lambda token=None: _fake_api(["model.bin"]))
        writer_ended = threading.Event()

        def progress_writer(tracker, job_id, model_id):
            while tracker.should_continue():
                time.sleep(0.05)
            time.sleep(0.3)  # a commit in flight when the stop landed
            writer_ended.set()

        monkeypatch.setattr(llm_services, "update_db_with_progress", progress_writer)

        def on_chunk(i):
            if i == 2:
                llm_services.interrupt_active_downloads()

        fs = _ChunkedFs(chunk_seconds=0.05, on_chunk=on_chunk)
        monkeypatch.setattr(llm_services, "HfFileSystem", lambda token=None: fs)

        with pytest.raises(DownloadInterrupted):
            await llm_services.download_llm(
                model_link="org/model",
                model_id=7,
                temp_save_dir=str(tmp_path / "temp_7"),
                final_save_dir=str(tmp_path / "7"),
                job_id=62,
            )
        assert writer_ended.is_set()

    async def test_a_download_registered_after_the_shutdown_began_never_transfers(
        self, monkeypatch, tmp_path
    ):
        _mlx_engine(monkeypatch)
        monkeypatch.setattr(llm_services, "HfApi", lambda token=None: _fake_api(["model.bin"]))
        fs = MagicMock()
        monkeypatch.setattr(llm_services, "HfFileSystem", lambda token=None: fs)

        llm_services.interrupt_active_downloads()
        with pytest.raises(DownloadInterrupted):
            await llm_services.download_llm(
                model_link="org/model",
                model_id=7,
                temp_save_dir=str(tmp_path / "temp_7"),
                final_save_dir=str(tmp_path / "7"),
                job_id=63,
            )
        fs.get_file.assert_not_called()

    async def test_stop_active_downloads_waits_for_the_download_tasks(self):
        tracker = _tracker(100)
        llm_services._register_tracker(71, tracker)
        task_ended = threading.Event()

        def download_task():
            with llm_services.download_task_running():
                while tracker.should_continue():
                    time.sleep(0.02)
                time.sleep(0.2)  # the task's own wrap-up after the transfer stopped
            task_ended.set()

        worker = threading.Thread(target=download_task)
        worker.start()
        time.sleep(0.05)

        await llm_services.stop_active_downloads(timeout=5)

        assert tracker.interrupted
        assert task_ended.is_set()
        worker.join(1)

    async def test_stop_active_downloads_is_bounded(self, caplog):
        release = threading.Event()

        def stuck_task():
            with llm_services.download_task_running():
                release.wait(10)  # a blocking network read that never returns

        worker = threading.Thread(target=stuck_task)
        worker.start()
        try:
            started = time.monotonic()
            with caplog.at_level(logging.INFO, logger="erudi"):
                await llm_services.stop_active_downloads(timeout=0.3)
            assert time.monotonic() - started < 2
            (warning,) = _warnings(caplog)
            assert "still running" in warning.getMessage()
        finally:
            release.set()
            worker.join(5)


@pytest.mark.unit
class TestRunDownloadTaskOnShutdown:
    def test_an_interrupted_download_leaves_the_job_running_for_the_startup_cleanup(
        self, monkeypatch, caplog
    ):
        job = SimpleNamespace(status="pending", error_message=None, progress=0.0)
        llm = SimpleNamespace(link="/models/1", model_metadata=None, local=2)
        session = MagicMock()
        session.query.side_effect = lambda model: FakeDownloadQuery(model, job, llm)
        monkeypatch.setattr(llm_endpoints, "SessionLocal", lambda: session)
        commits_after_start: list[str] = []

        async def download(**kwargs):
            session.commit.side_effect = lambda: commits_after_start.append(job.status)
            raise DownloadInterrupted("the backend is stopping")

        monkeypatch.setattr(llm_endpoints, "download_llm", download)

        with caplog.at_level(logging.INFO, logger="erudi"):
            llm_endpoints._run_download_task("org/model", 1, "/tmp/t", "/tmp/f", job_id=9)

        # Postgres is stopping: nothing is written, the row stays "running",
        # which the next start resolves ("Download interrupted due to
        # application shutdown"), never the user's "cancelled".
        assert job.status == "running"
        assert commits_after_start == []
        assert _warnings(caplog) == []
        assert any(
            "Download job 9" in r.getMessage() and "interrupted" in r.getMessage()
            for r in caplog.records
        )


class _FakeEngine:
    def start_cleanup_task(self):
        pass

    def stop_cleanup_task(self):
        pass

    def start_pressure_task(self):
        pass

    async def stop_pressure_task(self):
        pass

    def set_idle_unload_minutes(self, minutes):
        pass

    def cleanup(self):
        pass


class _FakeCheckpointerCM:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, *exc):
        return False


@pytest.mark.unit
class TestLifespanStopsDownloadsFirst:
    def _arm(self, monkeypatch, order):
        handle = SimpleNamespace(
            sqlalchemy_url="postgresql+psycopg://x/erudi", psycopg_url="postgresql://x/erudi"
        )
        monkeypatch.setattr(api, "start_postgres", lambda *_: handle)
        monkeypatch.setattr(api, "stop_postgres", lambda *_: order.append("stop_postgres"))
        monkeypatch.setattr(api, "init_database", lambda *_: None)
        monkeypatch.setattr(api, "run_migrations", lambda *_: None)
        monkeypatch.setattr(api, "init_kb_store", lambda *_: None)
        monkeypatch.setattr(api, "close_kb_store", lambda *_a, **_k: None)
        monkeypatch.setattr(api, "open_checkpointer", lambda *_: _FakeCheckpointerCM())

        async def _fake_populate():
            return {}

        monkeypatch.setattr(api, "startup_populate_database", _fake_populate)
        monkeypatch.setattr(api, "BaseEngine", SimpleNamespace(get_engine=lambda: _FakeEngine()))
        monkeypatch.setattr(config, "LLM_Engine", None, raising=False)

        async def stop_downloads(timeout):
            order.append("stop_active_downloads")

        monkeypatch.setattr(api, "stop_active_downloads", stop_downloads)

    async def test_downloads_are_stopped_before_the_database(self, monkeypatch):
        order: list[str] = []
        self._arm(monkeypatch, order)

        async with api.lifespan(FastAPI()):
            pass

        assert order == ["stop_active_downloads", "stop_postgres"]

    async def test_the_launcher_is_told_once_every_service_has_stopped(self, monkeypatch):
        order: list[str] = []
        self._arm(monkeypatch, order)
        app = FastAPI()
        app.state.services_stopped = threading.Event()

        async with api.lifespan(app):
            assert not app.state.services_stopped.is_set()

        assert app.state.services_stopped.is_set()
        assert order[-1] == "stop_postgres"
