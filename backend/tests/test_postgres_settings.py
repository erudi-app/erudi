"""Server settings of the embedded PostgreSQL, sized for one user (#620).

Covers:
- the connection budget: ``max_connections`` is the sum of every client the
  backend opens, and the pools are created with the sizes it counts;
- the ``pg_ctl`` wrapper: the settings travel as ``-o "-c name=value"`` on a
  ``start`` only, and the console-isolation wrapper underneath stays in place;
- a real throwaway cluster runs with them (``SHOW``), a cluster joined while
  already running reports the settings it keeps, and ``pg_dump`` still works
  under ``wal_level=minimal``.
"""

import logging
import subprocess
from pathlib import Path

import pgserver
import psycopg
import pytest

from src.launcher import postgres_runtime, postgres_settings
from src.launcher.postgres_runtime import start_postgres, stop_postgres
from src.launcher.postgres_settings import (
    BACKUP_CONNECTIONS,
    BUSINESS_POOL_MAX_OVERFLOW,
    BUSINESS_POOL_SIZE,
    CHECKPOINTER_CONNECTIONS,
    CONNECTION_MARGIN,
    DIRECT_CONNECTIONS,
    KB_STORE_POOL_MAX_OVERFLOW,
    KB_STORE_POOL_SIZE,
    MIGRATION_CONNECTIONS,
    max_connections,
    postmaster_options,
    server_settings,
)


class TestConnectionBudget:
    @pytest.mark.unit
    def test_max_connections_is_the_sum_of_every_client(self):
        assert max_connections() == (
            BUSINESS_POOL_SIZE
            + BUSINESS_POOL_MAX_OVERFLOW
            + KB_STORE_POOL_SIZE
            + KB_STORE_POOL_MAX_OVERFLOW
            + CHECKPOINTER_CONNECTIONS
            + MIGRATION_CONNECTIONS
            + BACKUP_CONNECTIONS
            + DIRECT_CONNECTIONS
            + CONNECTION_MARGIN
        )

    @pytest.mark.unit
    def test_max_connections_leaves_room_above_the_superuser_reserve(self):
        # PostgreSQL refuses to start unless max_connections exceeds
        # superuser_reserved_connections (3) + reserved_connections (0).
        assert max_connections() > 3

    @pytest.mark.unit
    def test_pools_are_sized_for_one_user(self):
        # A session (KB assistant, title, two compactions, the renderer's
        # polls) never had more than 4 queries in flight at once and held
        # 6 connections: the pools keep few idle connections and still open
        # twice the measured peak.
        assert BUSINESS_POOL_SIZE + KB_STORE_POOL_SIZE <= 3
        assert BUSINESS_POOL_SIZE + BUSINESS_POOL_MAX_OVERFLOW >= 2 * 4
        assert KB_STORE_POOL_SIZE + KB_STORE_POOL_MAX_OVERFLOW >= 2
        assert max_connections() <= 25

    @pytest.mark.unit
    def test_settings_carry_the_computed_max_connections(self):
        assert server_settings()["max_connections"] == str(max_connections())

    @pytest.mark.unit
    def test_business_engine_uses_the_budgeted_pool(self, monkeypatch):
        from src.database import core

        # init_database rebinds module state other tests rely on: put it back.
        monkeypatch.setattr(core, "db_engine", core.db_engine)
        monkeypatch.setitem(core.SessionLocal.kw, "bind", core.SessionLocal.kw.get("bind"))
        engine = core.init_database("postgresql+psycopg://postgres:x@127.0.0.1:1/erudi")
        try:
            assert engine.pool.size() == BUSINESS_POOL_SIZE
            assert engine.pool._max_overflow == BUSINESS_POOL_MAX_OVERFLOW
        finally:
            engine.dispose()

    @pytest.mark.unit
    def test_kb_store_engine_uses_the_budgeted_pool(self, monkeypatch):
        from src.ingestion import vector_store

        recorded = {}

        def fake_from_connection_string(url, **kwargs):
            recorded.update(kwargs)
            raise RuntimeError("stop here")

        monkeypatch.setattr(
            vector_store.PGEngine, "from_connection_string", fake_from_connection_string
        )

        class _Conn:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, *args, **kwargs):
                class _Cursor:
                    def fetchone(self):
                        return (1,)

                return _Cursor()

        monkeypatch.setattr(vector_store.psycopg, "connect", lambda *a, **k: _Conn())
        handle = type("H", (), {"psycopg_url": "x", "sqlalchemy_url": "y"})()
        with pytest.raises(RuntimeError, match="stop here"):
            vector_store.init_kb_store(handle)
        assert recorded == {
            "pool_size": KB_STORE_POOL_SIZE,
            "max_overflow": KB_STORE_POOL_MAX_OVERFLOW,
        }


class TestSettingsValues:
    @pytest.mark.unit
    def test_wal_level_minimal_comes_with_no_wal_senders(self):
        # The postmaster refuses wal_level=minimal while max_wal_senders > 0.
        settings = server_settings()
        assert settings["wal_level"] == "minimal"
        assert settings["max_wal_senders"] == "0"

    @pytest.mark.unit
    def test_no_value_contains_whitespace(self):
        # pg_ctl hands the -o string to postgres, which splits it on
        # whitespace: a value with a space would break the start.
        for name, value in server_settings().items():
            assert value and not any(ch.isspace() for ch in value), name
            assert not any(ch.isspace() for ch in name), name

    @pytest.mark.unit
    def test_postmaster_options_are_one_dash_o_per_setting(self):
        options = postmaster_options({"shared_buffers": "32MB", "wal_level": "minimal"})
        assert options == ["-o", "-c shared_buffers=32MB", "-o", "-c wal_level=minimal"]

    @pytest.mark.unit
    def test_postmaster_options_default_to_the_server_settings(self):
        options = postmaster_options()
        assert len(options) == 2 * len(server_settings())
        assert options[1] == f"-c max_connections={max_connections()}"

    @pytest.mark.unit
    def test_postmaster_options_reject_whitespace(self):
        with pytest.raises(ValueError):
            postmaster_options({"shared_buffers": "32 MB"})

    @pytest.mark.unit
    @pytest.mark.parametrize("char", list('&|<>^%"'))
    def test_postmaster_options_reject_shell_metacharacters(self, char):
        # On Windows pg_ctl starts the postmaster through cmd.exe, which
        # would interpret these in the -o string.
        with pytest.raises(ValueError):
            postmaster_options({"shared_buffers": f"32MB{char}"})
        with pytest.raises(ValueError):
            postmaster_options({f"shared{char}buffers": "32MB"})


class TestPgCtlArgument:
    def _record(self):
        recorded = {}

        def fake_pg_ctl(args, pgdata=None, **kwargs):
            recorded["args"] = list(args)
            recorded["kwargs"] = kwargs
            return "ok"

        return recorded, fake_pg_ctl

    @pytest.mark.unit
    def test_start_carries_the_settings_before_the_action(self):
        recorded, fake = self._record()
        wrapped = postgres_runtime._with_server_settings(fake)
        # pgserver's POSIX start arguments (postgres_server.py).
        args = ["-w", "-o", '-h ""', "-o", "-k /tmp/s", "-l", "/pg/log", "start"]

        assert wrapped(args, pgdata="/pg", timeout=10) == "ok"

        assert recorded["args"] == [*args[:-1], *postmaster_options(), "start"]
        # pgserver's own arguments are untouched, kwargs forwarded.
        assert recorded["kwargs"] == {"timeout": 10}

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "args",
        [["-w", "stop", "-m", "fast"], ["reload"], ["-w", "stop"], []],
    )
    def test_other_actions_are_untouched(self, args):
        recorded, fake = self._record()
        postgres_runtime._with_server_settings(fake)(args, pgdata="/pg")
        assert recorded["args"] == args

    @pytest.mark.unit
    def test_start_logs_the_settings(self, caplog):
        _, fake = self._record()
        with caplog.at_level(logging.INFO, logger="erudi"):
            postgres_runtime._with_server_settings(fake)(["-w", "start"], pgdata="/pg")
        messages = [r.getMessage() for r in caplog.records]
        line = next(m for m in messages if "server settings" in m)
        assert f"max_connections={max_connections()}" in line
        assert f"shared_buffers={server_settings()['shared_buffers']}" in line
        assert line.isascii()

    @pytest.mark.unit
    def test_pg_ctl_is_wrapped_over_the_console_isolation(self):
        pg_ctl = postgres_runtime._pg_server_mod.pg_ctl
        assert getattr(pg_ctl, "_erudi_server_settings", False)
        assert getattr(pg_ctl, "_erudi_console_isolated", False)

    @pytest.mark.unit
    def test_wrapping_twice_is_a_no_op(self):
        _, fake = self._record()
        once = postgres_runtime._with_server_settings(fake)
        assert postgres_runtime._with_server_settings(once) is once


def _show(conn: psycopg.Connection, name: str) -> str:
    return conn.execute("SELECT current_setting(%s)", (name,)).fetchone()[0]


@pytest.fixture(scope="module")
def tuned(tmp_path_factory):
    base = tmp_path_factory.mktemp("pg-settings")
    handle = start_postgres(base / "data")
    try:
        yield handle
    finally:
        stop_postgres(handle)


class TestClusterRunsWithTheSettings:
    @pytest.mark.integration
    def test_every_setting_is_in_effect(self, tuned):
        with psycopg.connect(tuned.psycopg_url) as conn:
            running = {name: _show(conn, name) for name in server_settings()}
        assert running == server_settings()

    @pytest.mark.integration
    def test_max_connections_is_the_computed_budget(self, tuned):
        with psycopg.connect(tuned.psycopg_url) as conn:
            assert int(_show(conn, "max_connections")) == max_connections()

    @pytest.mark.integration
    def test_nothing_is_written_to_the_configuration_files(self, tuned):
        auto_conf = tuned.data_dir / "postgresql.auto.conf"
        text = auto_conf.read_text() if auto_conf.exists() else ""
        assert "shared_buffers" not in text and "max_connections" not in text
        conf = (tuned.data_dir / "postgresql.conf").read_text()
        assert "\nshared_buffers = 128MB" in conf  # initdb's line, untouched

    @pytest.mark.integration
    def test_pg_dump_backup_still_works(self, tuned):
        from src.database.backup import _pg_dump_bin, backup_database

        with psycopg.connect(tuned.psycopg_url, autocommit=True) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS settings_probe (id int)")
            conn.execute("INSERT INTO settings_probe VALUES (1)")
        dump = backup_database(tuned.psycopg_url, tuned.data_dir, label="settings")
        pg_dump = Path(_pg_dump_bin())
        pg_restore = pg_dump.with_name(pg_dump.name.replace("pg_dump", "pg_restore"))
        listing = subprocess.run(
            [str(pg_restore), "--list", str(dump)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        assert "settings_probe" in listing


class TestClusterFromAnOlderVersion:
    @pytest.mark.integration
    def test_a_cluster_run_with_the_defaults_restarts_with_these_settings(
        self, tmp_path, monkeypatch
    ):
        data_dir = tmp_path / "data"
        # An older version's cluster: started with initdb's defaults, data
        # written under wal_level=replica and 100 connections, stopped.
        monkeypatch.setattr(postgres_settings, "server_settings", lambda: {})
        old = start_postgres(data_dir)
        with psycopg.connect(old.psycopg_url, autocommit=True) as conn:
            assert _show(conn, "wal_level") == "replica"
            assert _show(conn, "max_connections") == "100"
            conn.execute("CREATE TABLE kept (id int)")
            conn.execute("INSERT INTO kept VALUES (42)")
        stop_postgres(old)
        monkeypatch.undo()

        new = start_postgres(data_dir)
        try:
            with psycopg.connect(new.psycopg_url) as conn:
                running = {name: _show(conn, name) for name in server_settings()}
                assert conn.execute("SELECT id FROM kept").fetchone()[0] == 42
            assert running == server_settings()
        finally:
            stop_postgres(new)


class TestJoinedPostmasterKeepsItsSettings:
    @pytest.mark.integration
    def test_a_stock_postmaster_is_reported_and_kept(self, tmp_path, monkeypatch, caplog):
        data_dir = tmp_path / "data"
        # A postmaster started before this version: no settings passed.
        monkeypatch.setattr(postgres_settings, "server_settings", lambda: {})
        first = start_postgres(data_dir)
        monkeypatch.undo()
        try:
            # A second start in this process joins it, as a backend restarting
            # over a surviving postmaster does.
            pgserver.postgres_server.PostgresServer._instances.pop(first.server.pgdata, None)
            with caplog.at_level(logging.INFO, logger="erudi"):
                joined = start_postgres(data_dir)
            with psycopg.connect(joined.psycopg_url) as conn:
                assert _show(conn, "max_connections") == "100"
            line = next(
                r.getMessage()
                for r in caplog.records
                if "keeps the server settings it was started with" in r.getMessage()
            )
            assert f"max_connections=100 (next start: {max_connections()})" in line
            assert line.isascii()
        finally:
            stop_postgres(first)
            if postgres_runtime._live_postmaster(data_dir) is not None:
                # Two handles of one process on one cluster: the second stop
                # sees this process still listed and leaves it running.
                postgres_runtime._pg_server_mod.pg_ctl(
                    ["-w", "stop", "-m", "fast"], pgdata=data_dir
                )
