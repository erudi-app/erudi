"""P1 tests — embedded PostgreSQL runtime (pgserver) + explicit database init.

Covers:
- start_postgres(): cluster boot, `erudi` database creation, pgvector extension,
  the two URL forms (SQLAlchemy + raw psycopg), idempotent restart.
- stop_postgres(): explicit shutdown.
- init_database(): explicit engine creation + SessionLocal binding, and the
  anti-B1 rule (create_tables must not rely on an imported-by-value engine).
"""

import logging
import platform
import re
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import psycopg
import pytest
from sqlalchemy import create_engine, inspect as sa_inspect, text

from src.core.subprocess_flags import hidden_console_creationflags
from src.launcher import postgres_runtime
from src.launcher.postgres_runtime import (
    PASSWORD_FILE_NAME,
    _console_isolated,
    _enforce_password_auth,
    _ensure_cluster_password,
    _recover_corrupt_pgdata,
    _relax_pg_hba,
    _uri_with_password,
    harden_pg_hba,
    start_postgres,
    stop_postgres,
)


def _without_password(uri: str) -> str:
    """The pgserver-shaped URI (empty password) for a handle's URI."""
    return re.sub(r"://postgres:[^@]*@", "://postgres:@", uri, count=1)


@pytest.fixture(scope="module")
def pg(tmp_path_factory):
    """One throwaway embedded cluster for the whole module (boot is seconds)."""
    handle = start_postgres(tmp_path_factory.mktemp("pgdata-p1"))
    yield handle
    stop_postgres(handle)


class TestPostgresRuntime:
    @pytest.mark.integration
    def test_url_shapes(self, pg):
        assert pg.sqlalchemy_url.startswith("postgresql+psycopg://")
        assert pg.psycopg_url.startswith("postgresql://")
        assert "+psycopg" not in pg.psycopg_url

    @pytest.mark.integration
    def test_sqlalchemy_connects_to_erudi_database(self, pg):
        eng = create_engine(pg.sqlalchemy_url)
        try:
            with eng.connect() as conn:
                assert conn.execute(text("SELECT current_database()")).scalar() == "erudi"
        finally:
            eng.dispose()

    @pytest.mark.integration
    def test_vector_extension_installed_in_erudi_database(self, pg):
        eng = create_engine(pg.sqlalchemy_url)
        try:
            with eng.connect() as conn:
                names = {row[0] for row in conn.execute(text("SELECT extname FROM pg_extension"))}
        finally:
            eng.dispose()
        assert "vector" in names

    @pytest.mark.integration
    def test_psycopg_url_accepted_by_raw_psycopg(self, pg):
        with psycopg.connect(pg.psycopg_url, autocommit=True) as conn:
            assert conn.execute("SELECT 1").fetchone()[0] == 1

    @pytest.mark.integration
    def test_start_postgres_is_idempotent(self, pg):
        password_before = (pg.data_dir / PASSWORD_FILE_NAME).read_text()
        hba_before = (pg.data_dir / "pg_hba.conf").read_text()

        again = start_postgres(pg.data_dir)

        assert again.sqlalchemy_url == pg.sqlalchemy_url
        assert again.psycopg_url == pg.psycopg_url
        # Joining a running cluster neither rotates the password nor rewrites
        # an already-hardened pg_hba.conf.
        assert (pg.data_dir / PASSWORD_FILE_NAME).read_text() == password_before
        assert (pg.data_dir / "pg_hba.conf").read_text() == hba_before
        # Same cluster, still answering.
        with psycopg.connect(again.psycopg_url, autocommit=True) as conn:
            assert conn.execute("SELECT 1").fetchone()[0] == 1


class TestClusterPasswordEnforced:
    """#462 - the cluster requires a per-cluster password on every host connection.

    Runs against the module cluster on every platform: the password file, the
    SCRAM verifier and the hardened pg_hba.conf are the same code path on
    macOS, Linux and Windows. Only the TCP refusal itself needs Windows, the
    one platform where pgserver listens on a loopback port instead of a
    trust-authenticated Unix socket.
    """

    @pytest.mark.integration
    def test_password_file_exists_and_is_in_the_urls(self, pg):
        secret_file = pg.data_dir / PASSWORD_FILE_NAME
        assert secret_file.exists()
        password = secret_file.read_text().strip()
        assert len(password) >= 32
        assert f"postgres:{password}@" in pg.psycopg_url
        assert f"postgres:{password}@" in pg.sqlalchemy_url

    @pytest.mark.integration
    def test_pg_hba_has_no_trusted_host_line_on_disk(self, pg):
        lines = (pg.data_dir / "pg_hba.conf").read_text().splitlines()
        rules = [ln.split() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
        host_rules = [r for r in rules if r[0].startswith("host")]
        assert host_rules, "pgserver's initdb always writes host rules"
        assert all(r[-1] == "scram-sha-256" for r in host_rules), host_rules
        # The Unix socket stays trust: filesystem permissions guard it on
        # POSIX and Windows never opens one.
        assert all(r[-1] == "trust" for r in rules if r[0] == "local")

    @pytest.mark.integration
    def test_role_has_a_scram_verifier(self, pg):
        with psycopg.connect(pg.psycopg_url) as conn:
            stored = conn.execute(
                "SELECT rolpassword FROM pg_authid WHERE rolname = 'postgres'"
            ).fetchone()[0]
        assert stored is not None and stored.startswith("SCRAM-SHA-256$")

    @pytest.mark.integration
    def test_hardened_hba_parses_and_was_reloaded(self, pg):
        with psycopg.connect(pg.psycopg_url) as conn:
            # The rewritten file must be one the server can load: no rule in
            # error, every host rule on SCRAM (pg_hba_file_rules parses the
            # file on disk, independently of what the postmaster has loaded).
            rows = conn.execute("SELECT type, auth_method, error FROM pg_hba_file_rules").fetchall()
            assert rows
            assert all(error is None for _, _, error in rows), rows
            assert all(
                method == "scram-sha-256" for kind, method, _ in rows if kind.startswith("host")
            )
            # And pg_reload_conf() was issued after the postmaster booted on
            # the trust rules: the config load time moved past the start time.
            reloaded = conn.execute(
                "SELECT pg_conf_load_time() > pg_postmaster_start_time()"
            ).fetchone()[0]
        assert reloaded is True

    @pytest.mark.integration
    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="TCP refusal needs Windows: on POSIX pgserver listens on a trust Unix socket only",
    )
    def test_tcp_connection_without_the_password_is_refused(self, pg):
        assert "127.0.0.1" in pg.psycopg_url  # the Windows loopback-port form
        with pytest.raises(psycopg.OperationalError, match="password"):
            psycopg.connect(_without_password(pg.psycopg_url))
        wrong = _uri_with_password(_without_password(pg.psycopg_url), "not-the-password")
        with pytest.raises(psycopg.OperationalError, match="password authentication failed"):
            psycopg.connect(wrong)
        with psycopg.connect(pg.psycopg_url) as conn:
            assert conn.execute("SELECT 1").fetchone()[0] == 1


class TestPgHbaHardening:
    """Pure text -> text rewrite of pg_hba.conf, no cluster needed."""

    INITDB_TRUST = (
        "# PostgreSQL Client Authentication Configuration File\n"
        "# TYPE  DATABASE        USER            ADDRESS                 METHOD\n"
        "\n"
        '# "local" is for Unix domain socket connections only\n'
        "local   all             all                                     trust\n"
        "# IPv4 local connections:\n"
        "host    all             all             127.0.0.1/32            trust\n"
        "# IPv6 local connections:\n"
        "host    all             all             ::1/128                 trust\n"
        "# Allow replication connections from localhost, by a user with the\n"
        "# replication privilege.\n"
        "local   replication     all                                     trust\n"
        "host    replication     all             127.0.0.1/32            trust\n"
        "host    replication     all             ::1/128                 trust\n"
    )

    @pytest.mark.unit
    def test_every_host_rule_switches_to_scram(self):
        out = harden_pg_hba(self.INITDB_TRUST)
        rules = [ln.split() for ln in out.splitlines() if ln and not ln.startswith("#")]
        host_rules = [r for r in rules if r[0] == "host"]
        assert len(host_rules) == 4  # all + replication, IPv4 + IPv6
        assert all(r[-1] == "scram-sha-256" for r in host_rules)

    @pytest.mark.unit
    def test_local_rules_stay_trust(self):
        out = harden_pg_hba(self.INITDB_TRUST)
        assert "local   all             all                                     trust\n" in out
        assert "local   replication     all                                     trust\n" in out

    @pytest.mark.unit
    def test_comments_and_blank_lines_are_preserved(self):
        out = harden_pg_hba(self.INITDB_TRUST)
        comments_in = [ln for ln in self.INITDB_TRUST.splitlines() if ln.startswith("#")]
        comments_out = [ln for ln in out.splitlines() if ln.startswith("#")]
        assert comments_out == comments_in
        assert out.count("\n\n") == self.INITDB_TRUST.count("\n\n")
        assert out.endswith("\n")

    @pytest.mark.unit
    def test_idempotent_on_already_hardened_text(self):
        once = harden_pg_hba(self.INITDB_TRUST)
        assert harden_pg_hba(once) == once

    @pytest.mark.unit
    def test_ssl_and_gssapi_host_variants_are_covered(self):
        text_in = (
            "hostssl all all 127.0.0.1/32 trust\n"
            "hostnossl all all ::1/128 trust\n"
            "hostgssenc all all 0.0.0.0/0 trust\n"
        )
        out = harden_pg_hba(text_in)
        assert "trust" not in out
        assert out.count("scram-sha-256") == 3

    @pytest.mark.unit
    def test_trailing_comment_on_a_rule_survives(self):
        out = harden_pg_hba("host all all 127.0.0.1/32 trust # loopback\n")
        assert out == "host all all 127.0.0.1/32 scram-sha-256 # loopback\n"

    @pytest.mark.unit
    def test_other_methods_are_left_alone(self):
        text_in = "host all all 127.0.0.1/32 md5\nhost all all ::1/128 reject\n"
        assert harden_pg_hba(text_in) == text_in


class TestPgHbaRelax:
    """Pure inverse of harden_pg_hba, used to re-key a cluster whose secret is lost."""

    @pytest.mark.unit
    def test_relax_is_the_inverse_of_harden(self):
        hardened = harden_pg_hba(TestPgHbaHardening.INITDB_TRUST)
        assert _relax_pg_hba(hardened) == TestPgHbaHardening.INITDB_TRUST

    @pytest.mark.unit
    def test_relax_leaves_local_rules_and_other_methods_alone(self):
        text_in = "local all all scram-sha-256\nhost all all 127.0.0.1/32 md5 # keep\n"
        assert _relax_pg_hba(text_in) == text_in

    @pytest.mark.unit
    def test_relax_is_idempotent(self):
        assert _relax_pg_hba(TestPgHbaHardening.INITDB_TRUST) == TestPgHbaHardening.INITDB_TRUST


class _FakeConn:
    """Records executed statements; usable as a context manager like psycopg."""

    def __init__(self, log):
        self._log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, *args):
        self._log.append(("sql", str(query)))
        return self


class TestLostPasswordRecovery:
    """A deleted erudi_db_password on a SCRAM-hardened cluster must self-heal.

    Pure-unit with mocks: psycopg.connect raises the auth failure once, then the
    normal path runs. The expected sequence is relax pg_hba -> pg_ctl reload
    (no connection possible) -> reconnect -> ALTER ROLE -> re-harden -> reload.
    """

    AUTH_ERROR = 'connection failed: FATAL:  password authentication failed for user "postgres"'

    @pytest.fixture
    def hardened_dir(self, tmp_path):
        (tmp_path / "pg_hba.conf").write_text(harden_pg_hba(TestPgHbaHardening.INITDB_TRUST))
        return tmp_path

    @pytest.mark.unit
    def test_auth_failure_relaxes_reloads_and_rekeys(self, hardened_dir, monkeypatch):
        events = []
        attempts = {"n": 0}

        def fake_connect(uri, **kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise psycopg.OperationalError(self.AUTH_ERROR)
            # By the time the retry connects, the file must already be trust.
            hba = (hardened_dir / "pg_hba.conf").read_text()
            events.append(("connect", "trust" in hba and "scram-sha-256" not in hba))
            return _FakeConn(events)

        def fake_pg_ctl(args, pgdata=None, **kwargs):
            events.append(("pg_ctl", tuple(args), pgdata))
            return ""

        monkeypatch.setattr(postgres_runtime.psycopg, "connect", fake_connect)
        monkeypatch.setattr(postgres_runtime._pg_server_mod, "pg_ctl", fake_pg_ctl)

        _enforce_password_auth(
            "postgresql://postgres:new@127.0.0.1:1/postgres", hardened_dir, "new"
        )

        assert attempts["n"] == 2
        # relax -> reload without a connection -> reconnect on trust
        assert events[0] == ("pg_ctl", ("reload",), hardened_dir)
        assert events[1] == ("connect", True)
        sql_text = " ".join(event[1] for event in events if event[0] == "sql")
        assert "ALTER ROLE" in sql_text and "pg_reload_conf" in sql_text
        # ...and the normal path re-hardened the file afterwards.
        final = (hardened_dir / "pg_hba.conf").read_text()
        assert final == harden_pg_hba(TestPgHbaHardening.INITDB_TRUST)

    @pytest.mark.unit
    def test_the_password_statement_is_kept_out_of_the_postmaster_log(
        self, hardened_dir, monkeypatch
    ):
        """`ALTER ROLE ... PASSWORD '<clear>'` is the one statement in this app
        that carries a secret, and PostgreSQL writes the statement that failed
        to its own log (`log_min_error_statement` defaults to ERROR). Silence
        that setting for the session around it, so a refusal cannot put the
        cluster password in a file we quote into bug reports."""
        events = []
        monkeypatch.setattr(
            postgres_runtime.psycopg, "connect", lambda uri, **kw: _FakeConn(events)
        )
        monkeypatch.setattr(postgres_runtime._pg_server_mod, "pg_ctl", lambda *a, **k: "")

        _enforce_password_auth(
            "postgresql://postgres:new@127.0.0.1:1/postgres", hardened_dir, "new"
        )

        statements = [sql for kind, sql in events if kind == "sql"]
        silenced = next(i for i, s in enumerate(statements) if "log_min_error_statement" in s)
        altered = next(i for i, s in enumerate(statements) if "ALTER ROLE" in s)
        restored = next(
            i
            for i, s in enumerate(statements)
            if "RESET" in s.upper() and "log_min_error_statement" in s
        )
        assert silenced < altered < restored
        assert "panic" in statements[silenced].lower()

    @pytest.mark.unit
    def test_the_rekey_record_carries_the_refusal(self, hardened_dir, monkeypatch, caplog):
        """A recovery nobody asked for gets one record, and the postmaster's
        refusal is what names the role and the rule that rejected us."""
        attempts = {"n": 0}

        def fake_connect(uri, **kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise psycopg.OperationalError(self.AUTH_ERROR)
            return _FakeConn([])

        monkeypatch.setattr(postgres_runtime.psycopg, "connect", fake_connect)
        monkeypatch.setattr(postgres_runtime._pg_server_mod, "pg_ctl", lambda *a, **k: "")

        with caplog.at_level(logging.WARNING, logger="erudi"):
            _enforce_password_auth(
                "postgresql://postgres:new@127.0.0.1:1/postgres", hardened_dir, "new"
            )

        rekey = [r for r in caplog.records if "re-keying the cluster" in r.getMessage()]
        assert rekey, "the re-key was not recorded"
        assert rekey[0].exc_info is not None, "the record does not carry the refusal"

    @pytest.mark.unit
    def test_other_operational_errors_are_not_retried(self, hardened_dir, monkeypatch):
        attempts = {"n": 0}

        def fake_connect(uri, **kwargs):
            attempts["n"] += 1
            raise psycopg.OperationalError("connection refused")

        reloads = []
        monkeypatch.setattr(postgres_runtime.psycopg, "connect", fake_connect)
        monkeypatch.setattr(
            postgres_runtime._pg_server_mod, "pg_ctl", lambda *a, **k: reloads.append(a)
        )

        with pytest.raises(psycopg.OperationalError, match="connection refused"):
            _enforce_password_auth(
                "postgresql://postgres:x@127.0.0.1:1/postgres", hardened_dir, "x"
            )

        assert attempts["n"] == 1
        assert reloads == []
        # The file was not touched either.
        assert (hardened_dir / "pg_hba.conf").read_text() == harden_pg_hba(
            TestPgHbaHardening.INITDB_TRUST
        )

    @pytest.mark.unit
    def test_auth_failure_that_persists_after_rekey_propagates(self, hardened_dir, monkeypatch):
        # One recovery attempt only: if trust + reload still does not let us
        # in, the error is real and must surface.
        monkeypatch.setattr(
            postgres_runtime.psycopg,
            "connect",
            lambda uri, **kw: (_ for _ in ()).throw(psycopg.OperationalError(self.AUTH_ERROR)),
        )
        monkeypatch.setattr(postgres_runtime._pg_server_mod, "pg_ctl", lambda *a, **k: "")

        with pytest.raises(psycopg.OperationalError, match="password authentication failed"):
            _enforce_password_auth(
                "postgresql://postgres:x@127.0.0.1:1/postgres", hardened_dir, "x"
            )


class TestClusterPasswordFile:
    """The secret lives inside PGDATA and is generated once per cluster."""

    @pytest.mark.unit
    def test_created_on_first_call_with_a_strong_url_safe_value(self, tmp_path):
        password = _ensure_cluster_password(tmp_path)
        secret_file = tmp_path / PASSWORD_FILE_NAME
        assert secret_file.exists()
        assert secret_file.read_text() == password
        assert len(password) >= 32
        # URL-safe alphabet: the value can sit in a URI verbatim.
        assert re.fullmatch(r"[A-Za-z0-9_-]+", password)

    @pytest.mark.unit
    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits; NTFS ACLs differ")
    def test_file_is_private_to_the_user(self, tmp_path):
        _ensure_cluster_password(tmp_path)
        mode = stat.S_IMODE((tmp_path / PASSWORD_FILE_NAME).stat().st_mode)
        assert mode == 0o600

    @pytest.mark.unit
    def test_second_call_reuses_the_same_value(self, tmp_path):
        first = _ensure_cluster_password(tmp_path)
        second = _ensure_cluster_password(tmp_path)
        assert first == second

    @pytest.mark.unit
    def test_surrounding_whitespace_in_the_file_is_ignored(self, tmp_path):
        (tmp_path / PASSWORD_FILE_NAME).write_text("  abc-DEF_123  \n")
        assert _ensure_cluster_password(tmp_path) == "abc-DEF_123"

    @pytest.mark.unit
    def test_an_empty_file_is_regenerated(self, tmp_path):
        # A truncated file would set an empty password, which PostgreSQL
        # stores as NULL: every SCRAM host connection would then be refused.
        (tmp_path / PASSWORD_FILE_NAME).write_text("\n")
        password = _ensure_cluster_password(tmp_path)
        assert re.fullmatch(r"[A-Za-z0-9_-]{32,}", password)
        assert (tmp_path / PASSWORD_FILE_NAME).read_text() == password

    @pytest.mark.unit
    def test_regenerated_after_the_data_dir_is_wiped(self, tmp_path):
        first = _ensure_cluster_password(tmp_path)
        # A half-initialised PGDATA (no PG_VERSION) is wiped by the recovery
        # path; the secret goes with it and a fresh cluster gets a fresh one.
        _recover_corrupt_pgdata(tmp_path)
        assert not (tmp_path / PASSWORD_FILE_NAME).exists()
        assert _ensure_cluster_password(tmp_path) != first


class TestUriWithPassword:
    @pytest.mark.unit
    def test_socket_form(self):
        uri = "postgresql://postgres:@/postgres?host=/tmp/erudi-pg-ab12"
        assert (
            _uri_with_password(uri, "s3cret")
            == "postgresql://postgres:s3cret@/postgres?host=/tmp/erudi-pg-ab12"
        )

    @pytest.mark.unit
    def test_tcp_form(self):
        uri = "postgresql://postgres:@127.0.0.1:54329/postgres"
        assert (
            _uri_with_password(uri, "s3cret")
            == "postgresql://postgres:s3cret@127.0.0.1:54329/postgres"
        )

    @pytest.mark.unit
    def test_password_is_percent_quoted(self):
        uri = "postgresql://postgres:@127.0.0.1:54329/postgres"
        out = _uri_with_password(uri, "a/b@c?d")
        assert out == "postgresql://postgres:a%2Fb%40c%3Fd@127.0.0.1:54329/postgres"
        # And it round-trips through psycopg's parser.
        assert psycopg.conninfo.conninfo_to_dict(out)["password"] == "a/b@c?d"

    @pytest.mark.unit
    def test_replaces_an_existing_password(self):
        uri = "postgresql://postgres:old@127.0.0.1:1/postgres"
        assert _uri_with_password(uri, "new") == "postgresql://postgres:new@127.0.0.1:1/postgres"

    @pytest.mark.unit
    def test_rejects_a_uri_without_a_user(self):
        with pytest.raises(ValueError):
            _uri_with_password("postgresql://127.0.0.1:1/postgres", "x")

    @pytest.mark.integration
    def test_stale_handle_pids_are_pruned(self, tmp_path_factory):
        """pgserver refcounts cluster users in <pgdata>/.handle_pids.json but
        never prunes dead pids: a crashed/SIGKILLed backend leaves a ghost
        entry that makes every later cleanup() skip the server stop forever.
        start_postgres must prune dead pids so the last live handle really
        stops the cluster."""
        import json
        import os
        import signal
        import subprocess

        data_dir = tmp_path_factory.mktemp("pgdata-prune")
        try:
            first = start_postgres(data_dir)
            stop_postgres(first)

            # Simulate a previous owner that died brutally (pid is real but dead).
            ghost = subprocess.Popen(["sleep", "0"])
            ghost.wait()
            (data_dir / ".handle_pids.json").write_text(json.dumps([ghost.pid]))

            handle = start_postgres(data_dir)
            stop_postgres(handle)

            # Without pruning, the ghost pid would block the stop and the
            # postmaster would survive (postmaster.pid still present).
            assert not (data_dir / "postmaster.pid").exists()
        finally:
            # The red scenario is precisely "the stop is skipped": never leak
            # a live postmaster on the dev machine when this test regresses.
            pid_file = data_dir / "postmaster.pid"
            if pid_file.exists():
                os.kill(int(pid_file.read_text().splitlines()[0]), signal.SIGTERM)


class TestCorruptPgdataRecovery:
    """#145 — a half-initialized pgdata (no PG_VERSION) must not brick boot."""

    @pytest.mark.unit
    def test_recover_leaves_initialized_cluster_untouched(self, tmp_path):
        (tmp_path / "PG_VERSION").write_text("16\n")
        (tmp_path / "base").mkdir()
        _recover_corrupt_pgdata(tmp_path)
        assert (tmp_path / "PG_VERSION").exists()
        assert (tmp_path / "base").exists()

    @pytest.mark.unit
    def test_recover_wipes_partial_dir_without_pg_version(self, tmp_path):
        (tmp_path / "junk.tmp").write_text("x")
        (tmp_path / "global").mkdir()
        (tmp_path / "global" / "leftover").write_text("y")
        assert list(tmp_path.iterdir())  # non-empty, no PG_VERSION
        _recover_corrupt_pgdata(tmp_path)
        assert list(tmp_path.iterdir()) == []  # wiped clean

    @pytest.mark.unit
    def test_recover_noop_on_empty_dir(self, tmp_path):
        _recover_corrupt_pgdata(tmp_path)  # must not raise
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.unit
    def test_an_unreadable_data_dir_is_recorded(self, tmp_path, monkeypatch, caplog):
        """initdb is about to fail on the same directory; this warning is the
        only hint of why (permissions, a dismounted volume, a synced folder
        gone read-only)."""

        def _refuse(self):
            raise PermissionError("[Errno 13] Permission denied")

        monkeypatch.setattr(Path, "iterdir", _refuse)
        with caplog.at_level(logging.WARNING, logger="erudi"):
            _recover_corrupt_pgdata(tmp_path)
        assert any(
            "Could not inspect the Postgres data dir" in r.getMessage() for r in caplog.records
        )


class TestPostmasterLog:
    """The postmaster's own words, which nothing used to read.

    pgserver starts the server with ``pg_ctl -l <pgdata>/log`` and leaves
    `logging_collector` off, so that ONE file holds every line Postgres wrote:
    WAL replay, a corrupt page, a full disk, the FATAL that stopped the boot.
    Our records say what Erudi asked for; only this says what Postgres
    answered.
    """

    @pytest.mark.unit
    def test_returns_the_last_lines(self, tmp_path):
        (tmp_path / "log").write_text(
            "\n".join(f"LOG:  line {i}" for i in range(200)), encoding="utf-8"
        )

        tail = postgres_runtime.postmaster_log_tail(tmp_path, max_lines=40)

        assert tail.count("\n") == 39
        assert "line 199" in tail
        assert "line 159" not in tail

    @pytest.mark.unit
    def test_is_empty_when_there_is_no_log(self, tmp_path):
        assert postgres_runtime.postmaster_log_tail(tmp_path) == ""

    @pytest.mark.unit
    def test_decodes_defensively(self, tmp_path):
        """Postgres writes in the server encoding and the OS language: a byte
        that is not UTF-8 costs a character, never the diagnostic."""
        (tmp_path / "log").write_bytes(b"FATAL:  base de donn\xe9es corrompue\n")

        tail = postgres_runtime.postmaster_log_tail(tmp_path)

        assert tail.startswith("FATAL:")
        assert "corrompue" in tail

    @pytest.mark.unit
    def test_is_bounded(self, tmp_path):
        (tmp_path / "log").write_text("x" * 50_000, encoding="utf-8")
        assert len(postgres_runtime.postmaster_log_tail(tmp_path)) <= 4000

    @pytest.mark.unit
    def test_reads_only_a_window_from_the_end(self, tmp_path, monkeypatch):
        """The postmaster log grows for the life of the cluster and nothing
        rotates it. Reading it whole to quote 40 lines would load megabytes
        into memory at the exact moment the app is already failing."""
        log = tmp_path / "log"
        log.write_text("\n".join(f"LOG:  line {i}" for i in range(150_000)), encoding="utf-8")
        assert log.stat().st_size > 2_000_000

        read_bytes = []
        real_open = Path.open

        class _Counting:
            def __init__(self, inner):
                self._inner = inner

            def __enter__(self):
                self._inner.__enter__()
                return self

            def __exit__(self, *exc):
                return self._inner.__exit__(*exc)

            def seek(self, *args):
                return self._inner.seek(*args)

            def read(self, *args):
                data = self._inner.read(*args)
                read_bytes.append(len(data))
                return data

        def counting_open(self, *args, **kwargs):
            return _Counting(real_open(self, *args, **kwargs))

        monkeypatch.setattr(Path, "open", counting_open)
        tail = postgres_runtime.postmaster_log_tail(tmp_path)

        assert sum(read_bytes) <= postgres_runtime.POSTMASTER_LOG_WINDOW_BYTES
        assert "line 149999" in tail  # still the END of the file

    @pytest.mark.unit
    def test_a_password_in_a_logged_statement_is_redacted(self, tmp_path):
        """PostgreSQL logs the statement that failed (`log_min_error_statement`
        defaults to ERROR), so a refused `ALTER ROLE ... PASSWORD '<clear>'`
        puts the cluster password in this file -- which is quoted into
        backend.log, the Diagnostics page and whatever the user pastes into a
        public issue."""
        (tmp_path / "log").write_text(
            'ERROR:  syntax error at or near "PASSWORD"\n'
            "STATEMENT:  ALTER ROLE \"postgres\" PASSWORD 'S3cretToken123'\n"
            'STATEMENT:  CREATE USER other WITH ENCRYPTED PASSWORD "AlsoS3cret"\n',
            encoding="utf-8",
        )

        tail = postgres_runtime.postmaster_log_tail(tmp_path)

        assert "S3cretToken123" not in tail
        assert "AlsoS3cret" not in tail
        # ...and the statement is still recognisable, which is the point of
        # quoting it at all.
        assert "ALTER ROLE" in tail and "CREATE USER" in tail
        assert "[redacted]" in tail

    @pytest.mark.unit
    def test_a_failed_start_note_carries_no_password(self, tmp_path, monkeypatch):
        (tmp_path / "PG_VERSION").write_text("16\n")
        (tmp_path / "log").write_text(
            "STATEMENT:  ALTER ROLE \"postgres\" PASSWORD 'S3cretToken123'\n", encoding="utf-8"
        )
        monkeypatch.setattr(
            postgres_runtime,
            "_get_server_with_recovery",
            lambda data_dir: (_ for _ in ()).throw(AssertionError()),
        )

        with pytest.raises(AssertionError) as excinfo:
            start_postgres(tmp_path)

        assert "S3cretToken123" not in "\n".join(getattr(excinfo.value, "__notes__", []))

    @pytest.mark.unit
    def test_a_failed_start_carries_the_postmaster_log(self, tmp_path, monkeypatch):
        """The lifespan writes one ERROR with the traceback; the note travels
        inside it, so the cause reaches backend.log and the Diagnostics page
        with the failure it explains."""
        (tmp_path / "PG_VERSION").write_text("16\n")  # a real cluster, not a half-init
        (tmp_path / "log").write_text(
            "LOG:  database system was not properly shut down\n"
            "FATAL:  could not write to file: No space left on device\n",
            encoding="utf-8",
        )
        boom = subprocess.TimeoutExpired(cmd="pg_ctl", timeout=10)
        monkeypatch.setattr(
            postgres_runtime,
            "_get_server_with_recovery",
            lambda data_dir: (_ for _ in ()).throw(boom),
        )

        with pytest.raises(subprocess.TimeoutExpired) as excinfo:
            start_postgres(tmp_path)

        notes = "\n".join(getattr(excinfo.value, "__notes__", []))
        assert "No space left on device" in notes
        assert str(tmp_path / "log") in notes

    @pytest.mark.unit
    def test_a_failed_start_says_so_even_with_no_postmaster_log(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            postgres_runtime,
            "_get_server_with_recovery",
            lambda data_dir: (_ for _ in ()).throw(AssertionError()),
        )

        with pytest.raises(AssertionError) as excinfo:
            start_postgres(tmp_path)

        notes = "\n".join(getattr(excinfo.value, "__notes__", []))
        assert "empty or unreadable" in notes

    @pytest.mark.unit
    def test_stopping_the_cluster_is_announced_before_it_can_hang(self, caplog):
        handle = SimpleNamespace(server=SimpleNamespace(cleanup=lambda: None), data_dir="/tmp/pg")

        with caplog.at_level(logging.INFO, logger="erudi"):
            stop_postgres(handle)

        messages = [r.getMessage() for r in caplog.records]
        assert any("Stopping the embedded PostgreSQL cluster" in m for m in messages)
        assert any("Embedded PostgreSQL stopped" in m for m in messages)

    @pytest.mark.integration
    def test_start_postgres_recovers_from_missing_pg_version(self, tmp_path_factory):
        import os
        import signal

        data_dir = tmp_path_factory.mktemp("pgdata-recover")
        try:
            first = start_postgres(data_dir)
            stop_postgres(first)

            # Simulate an initdb interrupted before PG_VERSION was written:
            # the dir is populated but has no PG_VERSION. Without recovery,
            # pgserver's initdb would refuse the non-empty directory.
            (data_dir / "PG_VERSION").unlink()
            assert list(data_dir.iterdir())  # still populated

            handle = start_postgres(data_dir)
            try:
                assert (data_dir / "PG_VERSION").exists()  # re-initialized
                with psycopg.connect(handle.psycopg_url, autocommit=True) as conn:
                    assert conn.execute("SELECT 1").fetchone()[0] == 1
            finally:
                stop_postgres(handle)
        finally:
            pid_file = data_dir / "postmaster.pid"
            if pid_file.exists():
                os.kill(int(pid_file.read_text().splitlines()[0]), signal.SIGTERM)


class TestInitDatabase:
    @pytest.mark.integration
    async def test_init_database_binds_engine_and_session_factory(self, pg):
        from src.database import core

        try:
            engine = core.init_database(pg.sqlalchemy_url)
            assert engine is core.db_engine
            session = core.SessionLocal()
            try:
                assert session.execute(text("SELECT 1")).scalar() == 1
            finally:
                session.close()
        finally:
            core.SessionLocal.configure(bind=None)
            core.db_engine = None

    @pytest.mark.integration
    async def test_create_tables_works_after_init_database(self, pg):
        """Anti-B1: Database_Seeder.create_tables must read the LIVE engine,
        not a stale imported-by-value copy frozen at import time."""
        from src.database import core
        from src.database.seed import Database_Seeder

        try:
            engine = core.init_database(pg.sqlalchemy_url)
            await Database_Seeder().create_tables()
            tables = set(sa_inspect(engine).get_table_names())
            assert {"llms", "conversations", "messages"} <= tables
        finally:
            core.SessionLocal.configure(bind=None)
            core.db_engine = None

    @pytest.mark.integration
    async def test_create_tables_without_init_raises_explicit_error(self, pg):
        from src.database import core
        from src.database.seed import Database_Seeder

        assert core.db_engine is None  # process-fresh or reset by previous tests
        with pytest.raises(RuntimeError, match="init_database"):
            await Database_Seeder().create_tables()


class FakePostmaster:
    """Minimal stand-in for pgserver's PostmasterInfo readiness view."""

    def __init__(self, running, status):
        self._running = running
        self.status = status

    def is_running(self):
        return self._running


class TestRecoverySecondChance:
    """#161 — survive a slow WAL crash-recovery past pgserver's 10s pg_ctl timeout.

    Pure-unit: no real cluster. The postmaster/pgserver/time surfaces are
    monkeypatched on the postgres_runtime module so the second-chance logic can
    be exercised deterministically.
    """

    @pytest.mark.unit
    def test_wait_for_postmaster_ready_polls_until_ready(self, tmp_path, monkeypatch):
        # None (no pidfile yet) -> running but still recovering -> ready.
        sequence = [
            None,
            FakePostmaster(running=True, status="starting"),
            FakePostmaster(running=True, status="ready"),
        ]
        monkeypatch.setattr(
            postgres_runtime.PostmasterInfo,
            "read_from_pgdata",
            lambda data_dir: sequence.pop(0),
        )
        monkeypatch.setattr(postgres_runtime.time, "sleep", lambda _s: None)
        phases = []
        monkeypatch.setattr(postgres_runtime, "emit_phase", phases.append)

        assert postgres_runtime._wait_for_postmaster_ready(tmp_path, 90) is True
        # The wait announced itself so the renderer can label the pause.
        assert phases == ["recovering_database"]
        assert sequence == []  # consumed exactly through the ready reading

    @pytest.mark.unit
    def test_wait_for_postmaster_ready_times_out(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            postgres_runtime.PostmasterInfo,
            "read_from_pgdata",
            lambda data_dir: None,  # never comes up
        )
        monkeypatch.setattr(postgres_runtime.time, "sleep", lambda _s: None)
        monkeypatch.setattr(postgres_runtime, "emit_phase", lambda _p: None)

        # Tiny deadline + no-op sleep -> the loop bails almost immediately.
        assert postgres_runtime._wait_for_postmaster_ready(tmp_path, 0.01) is False

    @pytest.mark.unit
    def test_get_server_with_recovery_retries_after_timeout(self, tmp_path, monkeypatch):
        sentinel = object()
        calls = {"n": 0}

        def fake_get_server(path):
            calls["n"] += 1
            if calls["n"] == 1:
                raise subprocess.TimeoutExpired(cmd="pg_ctl", timeout=10)
            return sentinel  # second call reuses the recovered postmaster

        monkeypatch.setattr(postgres_runtime.pgserver, "get_server", fake_get_server)
        monkeypatch.setattr(postgres_runtime, "_wait_for_postmaster_ready", lambda d, s: True)

        assert postgres_runtime._get_server_with_recovery(tmp_path) is sentinel
        assert calls["n"] == 2  # first timed out, second reused the live postmaster

    @pytest.mark.unit
    def test_get_server_with_recovery_retries_after_bare_assertion_error(
        self, tmp_path, monkeypatch
    ):
        """ensure_postgres_running's already-running fast path never waits: it
        asserts status == 'ready' with no check first, so a postmaster that is
        running-but-still-WAL-recovering raises a bare AssertionError instead
        of TimeoutExpired. That must get the same second chance (reproduced on
        a real packaged build: a hard-killed prior session left Postgres
        recovering for ~26s, and the app showed a permanent crash screen
        because only TimeoutExpired used to trigger the retry)."""
        sentinel = object()
        calls = {"n": 0}

        def fake_get_server(path):
            calls["n"] += 1
            if calls["n"] == 1:
                raise AssertionError
            return sentinel

        monkeypatch.setattr(postgres_runtime.pgserver, "get_server", fake_get_server)
        monkeypatch.setattr(postgres_runtime, "_wait_for_postmaster_ready", lambda d, s: True)

        assert postgres_runtime._get_server_with_recovery(tmp_path) is sentinel
        assert calls["n"] == 2

    @pytest.mark.unit
    def test_get_server_with_recovery_reraises_assertion_error_when_wait_fails(
        self, tmp_path, monkeypatch
    ):
        calls = {"n": 0}

        def fake_get_server(path):
            calls["n"] += 1
            raise AssertionError

        monkeypatch.setattr(postgres_runtime.pgserver, "get_server", fake_get_server)
        monkeypatch.setattr(postgres_runtime, "_wait_for_postmaster_ready", lambda d, s: False)

        with pytest.raises(AssertionError):
            postgres_runtime._get_server_with_recovery(tmp_path)
        assert calls["n"] == 1

    @pytest.mark.unit
    def test_get_server_with_recovery_reraises_when_wait_fails(self, tmp_path, monkeypatch):
        calls = {"n": 0}

        def fake_get_server(path):
            calls["n"] += 1
            raise subprocess.TimeoutExpired(cmd="pg_ctl", timeout=10)

        monkeypatch.setattr(postgres_runtime.pgserver, "get_server", fake_get_server)
        monkeypatch.setattr(postgres_runtime, "_wait_for_postmaster_ready", lambda d, s: False)

        with pytest.raises(subprocess.TimeoutExpired):
            postgres_runtime._get_server_with_recovery(tmp_path)
        assert calls["n"] == 1  # never retried get_server after the wait failed

    @pytest.mark.unit
    def test_get_server_with_recovery_evicts_the_stale_cached_instance(self, tmp_path, monkeypatch):
        # pgserver registers the instance in _instances BEFORE starting the
        # server, so the first TimeoutExpired leaves a half-built object
        # (_postmaster_info=None) in the cache and get_server would hand it
        # back to the retry, crashing on get_uri()'s assert despite a
        # successful recovery (#215). The second chance must evict it first.
        broken = object()
        fresh = object()
        monkeypatch.setattr(
            postgres_runtime._pg_server_mod.PostgresServer,
            "_instances",
            {tmp_path: broken},
        )
        calls = {"n": 0}

        def fake_get_server(path):
            calls["n"] += 1
            if calls["n"] == 1:
                raise subprocess.TimeoutExpired(cmd="pg_ctl", timeout=10)
            # The retry must run only AFTER the stale entry was evicted.
            instances = postgres_runtime._pg_server_mod.PostgresServer._instances
            assert tmp_path not in instances
            return fresh

        monkeypatch.setattr(postgres_runtime.pgserver, "get_server", fake_get_server)
        monkeypatch.setattr(postgres_runtime, "_wait_for_postmaster_ready", lambda d, s: True)

        assert postgres_runtime._get_server_with_recovery(tmp_path) is fresh
        assert calls["n"] == 2


class TestConsoleIsolation:
    """#162 - the postgres lineage must NOT share the backend's hidden console.

    On the packaged Windows build every postgres process (pg_ctl/initdb, the
    postmaster + its helpers) inherits the backend's single hidden conhost, so
    that conhost's death kills the whole cluster while the backend survives - the
    app then runs green over a dead database. The module rebinds pgserver's
    pg_ctl/initdb with wrappers that inject creationflags=CREATE_NO_WINDOW so each
    child gets its own console; off Windows the injected value is 0 (a no-op).

    Pure-unit: the wrapper is exercised with a fake underlying command, never a
    real cluster (the throwaway-cluster integration tests above already drive the
    patched pg_ctl/initdb path in CI).
    """

    @pytest.mark.unit
    def test_pg_ctl_and_initdb_are_wrapped(self):
        # Module import rebinds both names on pgserver's postgres_server module.
        assert getattr(postgres_runtime._pg_server_mod.pg_ctl, "_erudi_console_isolated", False)
        assert getattr(postgres_runtime._pg_server_mod.initdb, "_erudi_console_isolated", False)

    @pytest.mark.unit
    def test_wrapper_injects_hidden_console_creationflags(self):
        recorded = {}

        def fake_command(args, pgdata=None, **kwargs):
            recorded["args"] = args
            recorded["pgdata"] = pgdata
            recorded["kwargs"] = kwargs
            return "ok"

        wrapped = _console_isolated(fake_command)
        assert wrapped(["-w", "start"], pgdata="/pg") == "ok"
        # On Windows this is CREATE_NO_WINDOW; on the Linux CI runner it is 0.
        assert recorded["kwargs"]["creationflags"] == hidden_console_creationflags()
        # The wrapper is transparent to the real call arguments.
        assert recorded["args"] == ["-w", "start"]
        assert recorded["pgdata"] == "/pg"

    @pytest.mark.unit
    def test_wrapper_does_not_override_explicit_creationflags(self):
        recorded = {}

        def fake_command(args, pgdata=None, **kwargs):
            recorded["kwargs"] = kwargs
            return "ok"

        wrapped = _console_isolated(fake_command)
        wrapped([], pgdata="/pg", creationflags=0x1)  # caller is explicit
        # setdefault semantics: a caller-provided value is never clobbered.
        assert recorded["kwargs"]["creationflags"] == 0x1

    @pytest.mark.unit
    def test_wrapper_forwards_pgserver_kwargs(self):
        # pgserver calls pg_ctl(..., user=..., timeout=10); those must survive.
        recorded = {}

        def fake_command(args, pgdata=None, **kwargs):
            recorded["kwargs"] = kwargs
            return "ok"

        wrapped = _console_isolated(fake_command)
        wrapped([], pgdata="/pg", user="postgres", timeout=10)
        assert recorded["kwargs"]["user"] == "postgres"
        assert recorded["kwargs"]["timeout"] == 10

    @pytest.mark.unit
    def test_injected_value_matches_platform(self):
        # CREATE_NO_WINDOW on Windows, 0 (harmless no-op) everywhere else.
        expected = 0x08000000 if platform.system() == "Windows" else 0
        assert hidden_console_creationflags() == expected

    @pytest.mark.unit
    def test_console_isolated_is_idempotent(self):
        # Re-wrapping an already-wrapped command returns it unchanged (guard attr),
        # so a module re-run never stacks wrappers.
        already = postgres_runtime._pg_server_mod.pg_ctl
        assert _console_isolated(already) is already
