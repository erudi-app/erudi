"""What the domains, the launcher and the DB watchdog log when work fails.

Background tasks are the cases that matter most: nobody awaits a
``BackgroundTask`` or the launcher's polling loop, so the record written at
the boundary is the only trace of the failure. See docs/logging.md.
"""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


# ============ Knowledge-base ingestion task ============


def test_kb_task_death_is_logged_once_with_job_id_and_traceback(monkeypatch, caplog):
    from src.domains.knowledge_base import endpoints as kb_endpoints

    class _Session:
        def close(self):
            pass

    class _Service:
        def process_and_index_documents(self, **kwargs):
            raise RuntimeError("embedding model exploded")

    monkeypatch.setattr(kb_endpoints, "SessionLocal", lambda: _Session())
    monkeypatch.setattr(kb_endpoints, "KB_Service", lambda: _Service())

    with caplog.at_level(logging.INFO, logger="erudi"):
        kb_endpoints._run_kb_creation_task(77, ["/tmp/a.pdf"])

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert "job 77" in errors[0].getMessage()
    assert "embedding model exploded" in errors[0].getMessage()
    assert errors[0].exc_info


# ============ Launcher: the reason reaches backend.log ============


def test_log_startup_failure_writes_the_traceback_to_the_backend_log(caplog):
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    import run as launcher

    try:
        raise ValueError("alembic exploded")
    except ValueError as exc:
        with caplog.at_level(logging.INFO, logger="erudi"):
            launcher.log_startup_failure("UNEXPECTED_ERROR", "Server thread crashed", exc)

    (record,) = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert "startup_error UNEXPECTED_ERROR" in record.getMessage()
    assert "Server thread crashed" in record.getMessage()
    assert record.exc_info and str(record.exc_info[1]) == "alembic exploded"


def test_log_startup_failure_falls_back_to_stderr_before_the_logger_exists(monkeypatch, capsys):
    import run as launcher

    monkeypatch.setitem(sys.modules, "src.core.logging", None)  # import raises
    try:
        raise OSError("read-only data dir")
    except OSError as exc:
        launcher.log_startup_failure("DATA_PREP_ERROR", "Failed to prepare data directories", exc)
    err = capsys.readouterr().err
    assert "[ERROR] startup_error DATA_PREP_ERROR" in err
    assert "read-only data dir" in err
    assert "Traceback" in err


# ============ DB watchdog: the final ERROR carries the last failure ============


async def test_recovery_failure_record_carries_the_last_exception(monkeypatch, caplog):
    from src.launcher import db_watchdog as wd

    def _no_cluster(*_a, **_k):
        raise RuntimeError("pg_ctl refused")

    monkeypatch.setattr(wd, "_evict_pgserver_cache", lambda *_: None)
    monkeypatch.setattr(wd, "start_postgres", _no_cluster)
    monkeypatch.setattr(wd, "_BACKOFF_LADDER", [0.0])
    monkeypatch.setattr(wd.config, "POSTGRES_DATA_DIR", "/nonexistent", raising=False)
    monkeypatch.setattr(wd, "db_state", wd.DB_RECOVERING)

    with caplog.at_level(logging.INFO, logger="erudi"):
        await wd._run_recovery_episode()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("attempt 1 failed" in r.getMessage() and r.exc_info for r in warnings)
    (error,) = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert "recovery failed after 1 attempts" in error.getMessage()
    assert "pg_ctl refused" in error.getMessage()
    assert error.exc_info and error.exc_info[1] is not None
    assert wd.db_state == wd.DB_FAILED


# ============ Download task: the record names the model ============


def test_download_task_failure_names_job_and_model(monkeypatch, caplog):
    from src.domains.llms import endpoints as llm_endpoints

    job_row = SimpleNamespace(status="pending", error_message=None, updated_at=None)

    class _Query:
        def get(self, _id):
            return job_row

    class _Session:
        def query(self, *_):
            return _Query()

        def commit(self):
            pass

        def close(self):
            pass

    async def _explode(*_a, **_k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(llm_endpoints, "SessionLocal", lambda: _Session())
    monkeypatch.setattr(llm_endpoints, "download_llm", _explode)

    with caplog.at_level(logging.INFO, logger="erudi"):
        llm_endpoints._run_download_task(
            model_link="org/repo",
            model_id=3,
            temp_save_dir="/tmp/erudi-test-temp",
            final_save_dir="/tmp/erudi-test-final",
            job_id=9,
        )

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    message = errors[0].getMessage()
    assert "job 9" in message and "LLM 3" in message and "org/repo" in message
    assert errors[0].exc_info
    assert job_row.status == "failed"
