"""The FastAPI lifespan emits startup-progress phases in order.

Fully mocked: every heavy startup dependency is stubbed, so this never touches
a real Postgres cluster. It verifies only that the lifespan surfaces the
`preparing_database → running_migrations → loading_catalog` phases (via the
`app.state.emit_phase` hook run.py injects) in the right order.
"""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI

from src.core import api
from src.core import config


class _FakeEngine:
    def start_cleanup_task(self):
        pass

    def stop_cleanup_task(self):
        pass

    def cleanup(self):
        pass


class _FakeCheckpointerCM:
    async def __aenter__(self):
        return object()

    async def __aexit__(self, *exc):
        return False


@pytest.mark.unit
async def test_lifespan_emits_phases_in_order(monkeypatch):
    handle = SimpleNamespace(
        sqlalchemy_url="postgresql+psycopg://x/erudi",
        psycopg_url="postgresql://x/erudi",
    )

    monkeypatch.setattr(api, "start_postgres", lambda *_: handle)
    monkeypatch.setattr(api, "stop_postgres", lambda *_: None)
    monkeypatch.setattr(api, "init_database", lambda *_: None)
    monkeypatch.setattr(api, "run_migrations", lambda *_: None)
    monkeypatch.setattr(api, "init_kb_store", lambda *_: None)
    monkeypatch.setattr(api, "close_kb_store", lambda *_a, **_k: None)
    monkeypatch.setattr(api, "open_checkpointer", lambda *_: _FakeCheckpointerCM())

    async def _fake_populate():
        return {}  # catalog reconcile happens inside; nothing else is scheduled

    monkeypatch.setattr(api, "startup_populate_database", _fake_populate)

    fake_base = SimpleNamespace(get_engine=lambda: _FakeEngine())
    monkeypatch.setattr(api, "BaseEngine", fake_base)
    monkeypatch.setattr(config, "LLM_Engine", None, raising=False)

    phases: list[str] = []
    app = FastAPI()
    app.state.emit_phase = phases.append

    async with api.lifespan(app):
        pass

    assert phases == ["preparing_database", "running_migrations", "loading_catalog"]


@pytest.mark.unit
async def test_lifespan_without_emitter_does_not_crash(monkeypatch):
    """No emit_phase on app.state (e.g. plain uvicorn in dev) -> phases skipped."""
    handle = SimpleNamespace(
        sqlalchemy_url="postgresql+psycopg://x/erudi",
        psycopg_url="postgresql://x/erudi",
    )
    monkeypatch.setattr(api, "start_postgres", lambda *_: handle)
    monkeypatch.setattr(api, "stop_postgres", lambda *_: None)
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

    app = FastAPI()  # no app.state.emit_phase
    async with api.lifespan(app):
        pass  # must not raise


@pytest.mark.unit
async def test_lifespan_logs_a_startup_failure_with_its_traceback(monkeypatch, caplog):
    """uvicorn reports a failed lifespan on stderr only; backend.log must carry
    the failure itself, at ERROR, with the traceback, before it propagates."""
    import logging

    handle = SimpleNamespace(
        sqlalchemy_url="postgresql+psycopg://x/erudi",
        psycopg_url="postgresql://x/erudi",
    )
    monkeypatch.setattr(api, "start_postgres", lambda *_: handle)
    monkeypatch.setattr(api, "stop_postgres", lambda *_: None)
    monkeypatch.setattr(api, "init_database", lambda *_: None)
    monkeypatch.setattr(api, "BaseEngine", SimpleNamespace(get_engine=lambda: _FakeEngine()))
    monkeypatch.setattr(config, "LLM_Engine", None, raising=False)

    def _broken_migrations(*_):
        raise RuntimeError("alembic exploded")

    monkeypatch.setattr(api, "run_migrations", _broken_migrations)

    app = FastAPI()
    with caplog.at_level(logging.INFO, logger="erudi"):
        with pytest.raises(RuntimeError, match="alembic exploded"):
            async with api.lifespan(app):
                pass  # never reached

    errors = [rec for rec in caplog.records if rec.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert "Startup failed: alembic exploded" in errors[0].getMessage()
    assert errors[0].exc_info and errors[0].exc_info[1] is not None


@pytest.mark.unit
async def test_background_task_death_is_logged_at_error():
    """The done callback attached to fire-and-forget tasks logs their death."""
    import asyncio
    import logging

    async def _dies():
        raise ValueError("backfill blew up")

    task = asyncio.create_task(_dies(), name="wire-backfill")
    with pytest.raises(ValueError):
        await task

    with _capture(logging.ERROR) as records:
        api._log_background_task_failure(task)

    assert len(records) == 1
    assert "wire-backfill" in records[0].getMessage()
    assert "backfill blew up" in records[0].getMessage()
    assert records[0].exc_info


@pytest.mark.unit
async def test_background_task_cancellation_is_not_logged():
    import asyncio
    import logging

    async def _forever():
        await asyncio.sleep(3600)

    task = asyncio.create_task(_forever())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    with _capture(logging.ERROR) as records:
        api._log_background_task_failure(task)
    assert records == []


class _capture:
    """Collect the ``erudi`` records at or above ``level`` (caplog is a
    function-scoped fixture; these helpers keep the two tests above compact)."""

    def __init__(self, level):
        import logging

        self.level = level
        self.records = []
        self.handler = logging.Handler(level)
        self.handler.emit = self.records.append

    def __enter__(self):
        import logging

        logging.getLogger("erudi").addHandler(self.handler)
        return self.records

    def __exit__(self, *exc):
        import logging

        logging.getLogger("erudi").removeHandler(self.handler)
        return False
