"""Tests for the embedded-Postgres watchdog (#162).

Unit tests exercise the state machine in isolation (disconnect hook, proactive
probe, backoff ladder, health field) with everything mocked. One integration
test is the incident reproduction: on a REAL throwaway cluster (NOT the session
fixture), kill the postmaster and drive one recovery episode, asserting the DB
answers again and the state returns to "ok".

The full three-tenant re-bind is heavy (the KB tenant loads the e5 embedding
model; the checkpointer opens an async psycopg pool), so the integration test
covers the SQLAlchemy re-bind end to end and STUBS the checkpointer + KB tenants
-- see the monkeypatches in ``test_watchdog_resurrects_a_killed_postmaster``.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import psycopg
import pytest

from src.launcher import db_watchdog


@pytest.fixture(autouse=True)
def _reset_watchdog_state():
    """Snapshot and restore the watchdog module globals around every test."""
    saved = (
        db_watchdog.db_state,
        db_watchdog._app,
        db_watchdog._loop_task,
        db_watchdog._wake,
        db_watchdog._event_loop,
        db_watchdog._listener_engine,
        db_watchdog._BACKOFF_LADDER,
    )
    db_watchdog.db_state = db_watchdog.DB_OK
    yield
    # Best-effort detach of any listener a test attached to a live engine.
    db_watchdog._remove_error_listener()
    (
        db_watchdog.db_state,
        db_watchdog._app,
        db_watchdog._loop_task,
        db_watchdog._wake,
        db_watchdog._event_loop,
        db_watchdog._listener_engine,
        db_watchdog._BACKOFF_LADDER,
    ) = saved


def _ctx(*, is_disconnect=False, original=None):
    """A stand-in for SQLAlchemy's ExceptionContext (only the fields we read)."""
    return SimpleNamespace(is_disconnect=is_disconnect, original_exception=original)


# ---- disconnect detection --------------------------------------------------


class TestDisconnectDetection:
    @pytest.mark.unit
    def test_is_disconnect_true_when_sqlalchemy_flags_it(self):
        assert db_watchdog._is_disconnect(_ctx(is_disconnect=True)) is True

    @pytest.mark.unit
    def test_is_disconnect_true_for_operational_error(self):
        ctx = _ctx(is_disconnect=False, original=psycopg.OperationalError("closed"))
        assert db_watchdog._is_disconnect(ctx) is True

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "error",
        [
            psycopg.errors.TooManyConnections("sorry, too many clients already"),
            psycopg.errors.QueryCanceled("canceling statement due to user request"),
            psycopg.errors.DeadlockDetected("deadlock detected"),
            psycopg.errors.SerializationFailure("could not serialize access"),
            psycopg.errors.DiskFull("could not extend file"),
        ],
        ids=["53300", "57014", "40P01", "40001", "53100"],
    )
    def test_is_disconnect_false_for_a_live_server_refusing_one_statement(self, error):
        # The server is up and answered: restarting it would cure nothing,
        # and on a full connection table it would loop.
        ctx = _ctx(is_disconnect=False, original=error)
        assert db_watchdog._is_disconnect(ctx) is False

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "message",
        [
            'connection failed: connection to server on socket "/tmp/x/.s.PGSQL.5432" '
            "failed: FATAL:  sorry, too many clients already",
            "connection failed: FATAL:  remaining connection slots are reserved for "
            "roles with the SUPERUSER attribute",
        ],
    )
    def test_is_disconnect_false_for_a_full_connection_table_at_connect(self, message):
        # psycopg reports a refused connect with no sqlstate (measured on the
        # bundled server): only the server's message says the table is full.
        ctx = _ctx(is_disconnect=False, original=psycopg.OperationalError(message))
        assert db_watchdog._is_disconnect(ctx) is False

    @pytest.mark.unit
    def test_hook_does_not_restart_on_too_many_clients(self):
        db_watchdog._on_db_error(
            _ctx(original=psycopg.errors.TooManyConnections("sorry, too many clients already"))
        )
        assert db_watchdog.db_state == db_watchdog.DB_OK

    @pytest.mark.unit
    def test_is_disconnect_false_for_non_connection_error(self):
        ctx = _ctx(is_disconnect=False, original=psycopg.ProgrammingError("bad sql"))
        assert db_watchdog._is_disconnect(ctx) is False


# ---- the error hook --------------------------------------------------------


class TestErrorHook:
    @pytest.mark.unit
    def test_hook_flips_ok_to_recovering_on_disconnect(self):
        db_watchdog.db_state = db_watchdog.DB_OK
        db_watchdog._on_db_error(_ctx(is_disconnect=True))
        assert db_watchdog.db_state == db_watchdog.DB_RECOVERING

    @pytest.mark.unit
    def test_hook_ignores_non_disconnect_errors(self):
        db_watchdog.db_state = db_watchdog.DB_OK
        db_watchdog._on_db_error(_ctx(is_disconnect=False, original=psycopg.ProgrammingError("x")))
        assert db_watchdog.db_state == db_watchdog.DB_OK

    @pytest.mark.unit
    def test_hook_leaves_failed_untouched_without_a_new_disconnect(self):
        # A benign (non-disconnect) error while parked in "failed" must not move
        # the state; only a real disconnect re-arms recovery.
        db_watchdog.db_state = db_watchdog.DB_FAILED
        db_watchdog._on_db_error(_ctx(is_disconnect=False))
        assert db_watchdog.db_state == db_watchdog.DB_FAILED


# ---- proactive probe -------------------------------------------------------


class TestProactiveProbe:
    @pytest.mark.unit
    async def test_probe_failure_flips_to_recovering(self, monkeypatch):
        def boom():
            raise psycopg.OperationalError("server closed the connection")

        monkeypatch.setattr(db_watchdog, "_probe_sync", boom)
        db_watchdog.db_state = db_watchdog.DB_OK
        await db_watchdog._probe_ok()
        assert db_watchdog.db_state == db_watchdog.DB_RECOVERING

    @pytest.mark.unit
    async def test_probe_success_leaves_state_ok(self, monkeypatch):
        monkeypatch.setattr(db_watchdog, "_probe_sync", lambda: None)
        db_watchdog.db_state = db_watchdog.DB_OK
        await db_watchdog._probe_ok()
        assert db_watchdog.db_state == db_watchdog.DB_OK


# ---- recovery episode / backoff ladder -------------------------------------


class TestRecoveryEpisode:
    @pytest.mark.unit
    async def test_success_on_first_attempt_resets_to_ok(self, monkeypatch):
        calls = []

        async def fake_resurrect(attempt):
            calls.append(attempt)
            return True

        monkeypatch.setattr(db_watchdog, "_resurrect_once", fake_resurrect)
        monkeypatch.setattr(db_watchdog, "_BACKOFF_LADDER", (0, 0, 0))
        db_watchdog.db_state = db_watchdog.DB_RECOVERING

        await db_watchdog._run_recovery_episode()

        assert calls == [1]  # stopped after the first success
        assert db_watchdog.db_state == db_watchdog.DB_OK

    @pytest.mark.unit
    async def test_success_on_second_attempt_counts_the_backoff(self, monkeypatch):
        calls = []

        async def fake_resurrect(attempt):
            calls.append(attempt)
            return attempt == 2  # first fails, second succeeds

        monkeypatch.setattr(db_watchdog, "_resurrect_once", fake_resurrect)
        monkeypatch.setattr(db_watchdog, "_BACKOFF_LADDER", (0, 0, 0))
        db_watchdog.db_state = db_watchdog.DB_RECOVERING

        await db_watchdog._run_recovery_episode()

        assert calls == [1, 2]
        assert db_watchdog.db_state == db_watchdog.DB_OK

    @pytest.mark.unit
    async def test_ladder_exhausted_parks_in_failed(self, monkeypatch):
        calls = []

        async def fake_resurrect(attempt):
            calls.append(attempt)
            return False  # never recovers

        monkeypatch.setattr(db_watchdog, "_resurrect_once", fake_resurrect)
        monkeypatch.setattr(db_watchdog, "_BACKOFF_LADDER", (0, 0, 0))
        db_watchdog.db_state = db_watchdog.DB_RECOVERING

        await db_watchdog._run_recovery_episode()

        assert calls == [1, 2, 3]  # exactly len(ladder) attempts
        assert db_watchdog.db_state == db_watchdog.DB_FAILED


# ---- health field ----------------------------------------------------------


class TestHealthField:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "state",
        [db_watchdog.DB_OK, db_watchdog.DB_RECOVERING, db_watchdog.DB_FAILED],
    )
    async def test_health_reflects_db_state(self, state):
        import src.core.health as health_mod

        db_watchdog.db_state = state
        body = await health_mod.health()
        # HTTP stays 200 always (health() never raises -> FastAPI serializes 200).
        assert body["status"] == "ok"
        assert body["db"] == state

    @pytest.mark.unit
    def test_db_state_defaults_ok_when_getter_raises(self, monkeypatch):
        import src.core.health as health_mod

        def boom():
            raise RuntimeError("watchdog exploded")

        # _db_state does a lazy `from ... import get_db_state`; patch it there.
        monkeypatch.setattr(db_watchdog, "get_db_state", boom)
        assert health_mod._db_state() == "ok"


# ---- integration: the incident reproduction --------------------------------


def _kill_cluster(pid: int) -> None:
    """Kill the postmaster and its whole process tree (the #162 reproduction)."""
    proc = psutil.Process(pid)
    victims = proc.children(recursive=True) + [proc]
    for p in victims:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(victims, timeout=10)


def _force_stop_cluster(data_dir) -> None:
    """Teardown: kill any postmaster for this throwaway pgdata, whatever
    refcount state the test left behind. The pgserver cache is left as-is; its
    atexit cleanup tolerates a dead process (a single benign atexit KeyError may
    surface at interpreter exit -- the documented cost of the #215 eviction the
    resurrection performs, not a test failure)."""
    pid_file = Path(data_dir) / "postmaster.pid"
    if pid_file.exists():
        try:
            _kill_cluster(int(pid_file.read_text().splitlines()[0]))
        except Exception:
            pass


class TestWatchdogResurrection:
    @pytest.mark.integration
    async def test_watchdog_resurrects_a_killed_postmaster(self, tmp_path_factory, monkeypatch):
        from src.database import core as db_core
        from src.launcher.postgres_runtime import start_postgres

        data_dir = tmp_path_factory.mktemp("wd-pg")
        # The watchdog restarts config.POSTGRES_DATA_DIR; point it at our
        # throwaway cluster, and keep the retry ladder snappy for the test.
        monkeypatch.setattr(db_watchdog.config, "POSTGRES_DATA_DIR", data_dir)
        monkeypatch.setattr(db_watchdog, "_BACKOFF_LADDER", (1, 2, 3))

        handle = start_postgres(data_dir)
        try:
            db_core.init_database(handle.sqlalchemy_url)
            assert db_watchdog._probe_sync() is None  # cluster answers

            # Cover the SQLAlchemy re-bind end to end; stub the two heavy tenants.
            async def _noop_checkpointer(h):
                return None

            monkeypatch.setattr(db_watchdog, "_rebind_checkpointer", _noop_checkpointer)
            monkeypatch.setattr(db_watchdog, "_rebind_kb_store", lambda h: None)
            db_watchdog._app = None  # no app.state to update

            # --- reproduce the incident: kill the whole cluster ---
            _kill_cluster(handle.server.get_pid())

            # The zombie window: the cluster is down but /health is still "ok"
            # until something touches the DB. Confirm the DB is unreachable.
            deadline = time.time() + 15
            while time.time() < deadline:
                try:
                    db_watchdog._probe_sync()
                except Exception:
                    break
                time.sleep(0.3)
            else:
                pytest.fail("cluster still answering after kill; cannot proceed")

            # --- drive one recovery episode (real start_postgres + init_database) ---
            db_watchdog.db_state = db_watchdog.DB_RECOVERING
            await db_watchdog._run_recovery_episode()

            assert db_watchdog.db_state == db_watchdog.DB_OK
            # The resurrected cluster answers again through the re-bound engine.
            assert db_watchdog._probe_sync() is None
        finally:
            db_watchdog._remove_error_listener()
            db_core.SessionLocal.configure(bind=None)
            db_core.db_engine = None
            _force_stop_cluster(data_dir)
