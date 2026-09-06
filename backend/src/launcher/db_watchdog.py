"""Embedded-Postgres watchdog: detect a dead cluster, resurrect it, expose state.

The embedded PostgreSQL cluster (``src.launcher.postgres_runtime``) can die
mid-session -- the whole lineage killed externally (a corporate EDR/console
cleaner reaping the shared conhost was reproduced on the packaged app, #162).
Before this module the app ran as a zombie: ``/erudi/health/`` kept answering
``{"status": "ok"}`` (a static reply that never touches the DB) while every
DB-backed endpoint 500ed -- 17 observed zombie minutes in the July incident.

Design (validated on #162, PR B):

- **State machine**: ``db_state`` in {``ok``, ``recovering``, ``failed``},
  a module global read by ``/health`` (see ``get_db_state``).
- **Reactive detection**: a SQLAlchemy ``handle_error`` listener on the live
  engine flips the state to ``recovering`` the instant a request hits a
  connection-level disconnect. The hook does NOTHING else (no I/O): it flags
  state and wakes the recovery loop thread-safely.
- **Proactive detection**: while healthy the loop probes ``SELECT 1`` every
  ~30s (off the event loop) so the zero-traffic case (the July incident: no DB
  request to trip on for 17 minutes) is still caught.
- **Resurrection**: on ``recovering`` the loop restarts the cluster through the
  field-proven ``start_postgres`` machinery (WAL recovery, the #215 cache
  eviction, retries) and re-binds the three DB tenants -- SQLAlchemy engine
  (``init_database``), the LangGraph checkpointer, and the KB vector store
  (``init_kb_store``) -- as a FULL re-init, because a resurrected postmaster
  may come back on a different port/URI (pool.dispose would keep the stale
  URI). Backoff ladder 5s/15s/60s; after the third failed attempt the state
  goes ``failed`` and the loop parks until the next error/probe wake.

All log literals are ASCII (repo rule).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

import pgserver.postgres_server as _pg_server_mod
import psycopg
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import event, text
from sqlalchemy.engine import Engine

from src.agents.checkpoint import open_checkpointer
from src.core import config
from src.core.logging import logger
from src.database import core as db_core
from src.database.core import init_database
from src.ingestion.vector_store import close_kb_store, init_kb_store
from src.launcher.postgres_runtime import start_postgres

# ---- state machine ---------------------------------------------------------

DB_OK = "ok"
DB_RECOVERING = "recovering"
DB_FAILED = "failed"

# Module global read by the health endpoint. Default "ok" so a partial boot or
# a test that never starts the watchdog still reports a sane value.
db_state: str = DB_OK

# Wait after a failed resurrection attempt, before the next one. Three entries
# => three attempts per episode; after the third failure the state goes
# "failed" and the loop parks (design ref #162).
_BACKOFF_LADDER: tuple[int, ...] = (5, 15, 60)

# Cadence of the proactive SELECT 1 probe while healthy, and of the re-arm
# while parked in "failed" (so a zero-traffic dead cluster can still self-heal).
_PROBE_INTERVAL_SECONDS: float = 30.0

# ---- runtime handles (set by start_watchdog) -------------------------------

_app: Optional[FastAPI] = None
_loop_task: Optional[asyncio.Task] = None
_wake: Optional[asyncio.Event] = None
_event_loop: Optional[asyncio.AbstractEventLoop] = None
# The exact Engine instance the error listener is attached to. init_database
# rebinds core.db_engine to a NEW engine on every (re-)init, so the listener
# must be moved onto whichever engine is live -- never an imported-by-value copy.
_listener_engine: Optional[Engine] = None


def get_db_state() -> str:
    """Current DB health as seen by the watchdog: ok | recovering | failed."""
    return db_state


# ---- detection: the SQLAlchemy disconnect hook -----------------------------


def _is_disconnect(exc_ctx) -> bool:
    """True for a connection-level disconnect (dead/closed cluster).

    Primary signal is SQLAlchemy's own ``is_disconnect`` (set by the psycopg
    dialect for "server closed the connection unexpectedly" and friends). We
    also treat any raw ``psycopg.OperationalError`` as a disconnect so a
    connect-time failure (cluster fully down, nothing to close) still trips.
    Non-connection errors (ProgrammingError, IntegrityError, ...) return False.
    """
    if getattr(exc_ctx, "is_disconnect", False):
        return True
    original = getattr(exc_ctx, "original_exception", None)
    return isinstance(original, psycopg.OperationalError)


def _on_db_error(exc_ctx) -> None:
    """SQLAlchemy ``handle_error`` listener. Runs in whatever thread executed
    the failing statement (often a threadpool worker), so it touches NO asyncio
    primitive directly -- it flags state and schedules the wake thread-safely.
    """
    if not _is_disconnect(exc_ctx):
        return
    _flag_down("disconnect detected by the SQLAlchemy error hook")


def _flag_down(reason: str) -> None:
    """Flip to ``recovering`` (unless already there) and wake the loop."""
    global db_state
    if db_state != DB_RECOVERING:
        logger.warning(f"DB watchdog: {reason}; state {db_state}->{DB_RECOVERING}")
        db_state = DB_RECOVERING
    _signal_wake()


def _signal_wake() -> None:
    """Wake the recovery loop from any thread (asyncio.Event is not
    thread-safe, so hop onto the loop thread via call_soon_threadsafe)."""
    loop, wake = _event_loop, _wake
    if loop is None or wake is None:
        return
    try:
        loop.call_soon_threadsafe(wake.set)
    except RuntimeError:
        # Loop already closed during shutdown -- nothing to wake.
        pass


def _register_error_listener() -> None:
    """Attach the disconnect hook to the LIVE engine (core.db_engine).

    init_database replaces core.db_engine on every (re-)init, so this moves the
    listener onto the current engine and detaches it from the previous one.
    """
    global _listener_engine
    engine = db_core.db_engine
    if engine is None:
        logger.warning("DB watchdog: no live engine to attach the error hook to")
        return
    if engine is _listener_engine:
        return
    _remove_error_listener()
    event.listen(engine, "handle_error", _on_db_error)
    _listener_engine = engine


def _remove_error_listener() -> None:
    """Detach the disconnect hook from the engine it is currently on (if any)."""
    global _listener_engine
    if _listener_engine is not None:
        try:
            event.remove(_listener_engine, "handle_error", _on_db_error)
        except Exception:
            # Listener already gone (engine disposed) -- best effort.
            pass
    _listener_engine = None


# ---- probe -----------------------------------------------------------------


def _probe_sync() -> None:
    """One lightweight ``SELECT 1`` on the live engine. Raises on failure."""
    engine = db_core.db_engine
    if engine is None:
        raise RuntimeError("database engine not initialized")
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))


# The exception of the last failed resurrection attempt, attached to the
# final ERROR of a recovery episode so the record says why recovery failed.
_last_failure: Optional[BaseException] = None


async def _probe_ok() -> None:
    """Proactive health probe (healthy path). On failure flip to recovering."""
    try:
        await run_in_threadpool(_probe_sync)
    except Exception as exc:
        _flag_down(f"proactive probe failed ({type(exc).__name__}: {exc})")


# ---- resurrection ----------------------------------------------------------


def _evict_pgserver_cache(data_dir) -> None:
    """Drop pgserver's cached PostgresServer for this pgdata.

    ``pgserver.get_server`` returns a cached handle for a known pgdata WITHOUT
    re-checking that the postmaster is alive (postgres_server.py: the
    ``pgdata in _instances`` short-circuit). After the cluster is killed
    externally the cache still reports "running", so ``start_postgres`` would
    hand back the dead handle and fail to connect. Popping the entry forces a
    fresh ``PostgresServer()``, whose ``ensure_postgres_running`` sees the stale
    postmaster.pid and restarts the postmaster -- on a fresh port, hence the
    full tenant re-bind. Same eviction the #215 recovery path performs.
    """
    key = Path(data_dir).expanduser().resolve()
    _pg_server_mod.PostgresServer._instances.pop(key, None)


async def _rebind_checkpointer(handle) -> None:
    """Re-open the LangGraph checkpointer against the resurrected cluster.

    Mirrors the lifespan wiring (``open_checkpointer`` held open on
    ``app.state.checkpointer_cm``). Opens the new saver BEFORE closing the old
    one so ``app.state.checkpointer`` is never momentarily unset.
    """
    if _app is None:
        return
    old_cm = getattr(_app.state, "checkpointer_cm", None)
    new_cm = open_checkpointer(handle.psycopg_url)
    saver = await new_cm.__aenter__()
    _app.state.checkpointer = saver
    _app.state.checkpointer_cm = new_cm
    if old_cm is not None:
        try:
            await old_cm.__aexit__(None, None, None)
        except Exception as exc:
            # The old connection points at the dead cluster; closing it may
            # fail -- harmless, the new saver is already live.
            logger.warning(f"DB watchdog: closing the stale checkpointer failed: {exc}")


def _rebind_kb_store(handle) -> None:
    """Re-bind the KB vector store (sync; runs off the event loop)."""
    close_kb_store()
    store = init_kb_store(handle)
    if _app is not None:
        _app.state.kb_store = store


async def _resurrect_once(attempt: int) -> bool:
    """One full resurrection attempt. Returns True iff the DB answers again.

    Restarts the cluster (evict stale cache -> start_postgres) and re-binds all
    three tenants, then verifies with a ``SELECT 1``. Any failure is swallowed
    and reported as False so the caller can walk the backoff ladder.
    """
    try:
        logger.info(f"DB watchdog: resurrection attempt {attempt} - restarting embedded Postgres")
        _evict_pgserver_cache(config.POSTGRES_DATA_DIR)
        handle = await run_in_threadpool(start_postgres, config.POSTGRES_DATA_DIR)
        # Tenant 1: SQLAlchemy engine + session factory (rebinds core.db_engine).
        await run_in_threadpool(init_database, handle.sqlalchemy_url)
        # The engine object changed -> move the disconnect hook onto it.
        _register_error_listener()
        # Tenant 2: LangGraph checkpointer.
        await _rebind_checkpointer(handle)
        # Tenant 3: KB vector store.
        await run_in_threadpool(_rebind_kb_store, handle)
        # Confirm the resurrected cluster actually answers.
        await run_in_threadpool(_probe_sync)
        if _app is not None:
            _app.state.postgres = handle
        return True
    except Exception as exc:
        global _last_failure
        _last_failure = exc
        logger.warning(f"DB watchdog: resurrection attempt {attempt} failed: {exc}", exc_info=True)
        return False


async def _run_recovery_episode() -> None:
    """Walk the backoff ladder once. Sets db_state to ok or failed."""
    global db_state
    start = time.monotonic()
    for attempt, delay in enumerate(_BACKOFF_LADDER, start=1):
        if await _resurrect_once(attempt):
            db_state = DB_OK
            logger.info(
                f"DB watchdog: recovered in {time.monotonic() - start:.1f}s " f"(attempt {attempt})"
            )
            return
        await asyncio.sleep(delay)
    db_state = DB_FAILED
    logger.error(
        f"DB watchdog: recovery failed after {len(_BACKOFF_LADDER)} attempts; "
        f"state -> {DB_FAILED}; last failure: {_last_failure}",
        exc_info=_last_failure,
    )


# ---- the loop --------------------------------------------------------------


async def _wait_wake(timeout: float) -> bool:
    """Wait up to ``timeout`` for a wake signal. True if woken, False on timeout."""
    wake = _wake
    if wake is None:
        await asyncio.sleep(timeout)
        return False
    try:
        await asyncio.wait_for(wake.wait(), timeout=timeout)
        wake.clear()
        return True
    except asyncio.TimeoutError:
        return False


async def _recovery_loop() -> None:
    """The single asyncio recovery loop owned by the lifespan."""
    global db_state
    logger.info("DB watchdog started")
    while True:
        try:
            if db_state == DB_OK:
                woken = await _wait_wake(_PROBE_INTERVAL_SECONDS)
                if not woken and db_state == DB_OK:
                    await _probe_ok()
                continue
            if db_state == DB_FAILED:
                # Parked: re-arm on the next error/probe wake so a dead cluster
                # that comes back on its own is still picked up.
                await _wait_wake(_PROBE_INTERVAL_SECONDS)
                logger.info("DB watchdog: re-arming recovery after park")
                db_state = DB_RECOVERING
                continue
            # db_state == DB_RECOVERING
            await _run_recovery_episode()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # The loop must never die: log and take a breath.
            logger.error(f"DB watchdog loop error: {exc}", exc_info=True)
            await asyncio.sleep(1.0)


# ---- lifecycle (called by the FastAPI lifespan) ----------------------------


def start_watchdog(app: FastAPI) -> None:
    """Start the watchdog: attach the disconnect hook + spawn the recovery loop.

    Call AFTER ``init_database`` has bound the live engine (the hook attaches to
    ``core.db_engine``) and the checkpointer is open on ``app.state``.
    Idempotent: a second call is a no-op while the loop is running.
    """
    global _app, _loop_task, _wake, _event_loop, db_state
    if _loop_task is not None and not _loop_task.done():
        return
    _app = app
    _event_loop = asyncio.get_running_loop()
    _wake = asyncio.Event()
    db_state = DB_OK
    _register_error_listener()
    _loop_task = asyncio.create_task(_recovery_loop())


async def stop_watchdog() -> None:
    """Stop the watchdog cleanly: cancel + await the loop, detach the hook."""
    global _loop_task, _app, _wake, _event_loop
    task = _loop_task
    _loop_task = None
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    _remove_error_listener()
    _app = None
    _wake = None
    _event_loop = None
    logger.info("DB watchdog stopped")
