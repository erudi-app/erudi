"""Embedded PostgreSQL runtime (pgserver) lifecycle for Erudi.

Starts/stops the local cluster bundled by the `pgserver` wheel (no Docker, no
system install), ensures the `erudi` database and the pgvector extension
exist, and derives the two URL forms consumed by the stack:

- ``sqlalchemy_url`` — ``postgresql+psycopg://…`` for the sync SQLAlchemy
  engine (business layer) and langchain-postgres ``PGEngine``.
- ``psycopg_url`` — ``postgresql://…`` for ``AsyncPostgresSaver`` and raw
  psycopg connections.

pgserver defaults to a Unix-domain-socket URI (``…?host=<socket dir>``) on
POSIX; ``get_server`` is idempotent (initdb on first run, refcounted across
processes) and registers an atexit cleanup. We still stop the cluster
explicitly from the FastAPI lifespan shutdown for a deterministic order
(checkpointer first, cluster last).

Authentication (#462). pgserver runs ``initdb --auth=trust --auth-local=trust``
and, on Windows, where there are no Unix sockets, starts the postmaster on a
loopback TCP port -- so any process under the user's account could open the
database with no credential. Every ``start_postgres`` therefore generates a
random password once per cluster (``<data_dir>/erudi_db_password``), sets it
on the ``postgres`` role, rewrites ``pg_hba.conf`` so every ``host`` rule
requires ``scram-sha-256`` (``local`` socket rules stay ``trust``: filesystem
permissions guard the socket), reloads the postmaster, and carries the
password in both derived URLs. The same code runs on every platform so the
POSIX test suites exercise it too; only the TCP refusal itself is Windows-only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import pgserver
import pgserver.postgres_server as _pg_server_mod
import psutil
import psycopg
from pgserver.utils import PostmasterInfo
from pgserver.utils import find_suitable_socket_dir as _orig_find_socket_dir
from pgserver.utils import socket_name_length_ok
from psycopg import sql

from src.core.logging import logger
from src.core.subprocess_flags import hidden_console_creationflags
from src.launcher.events import emit_phase

DB_NAME = "erudi"

# The superuser pgserver creates at initdb (``-U postgres``) and the only role
# the app connects as.
ROLE_NAME = "postgres"

# Per-cluster secret, inside PGDATA: initdb creates PGDATA with mode 0700, and
# the half-initialised-dir recovery wipes it together with the cluster it
# belongs to. Never logged.
PASSWORD_FILE_NAME = "erudi_db_password"

# How long a WAL crash-recovery may take before we give up on the boot. Must
# stay under run.py's STARTUP_TIMEOUT_SECONDS (120s non-first-run) minus the
# ~25s of cold imports — a recovery never happens on first run (no WAL yet).
RECOVERY_WAIT_SECONDS = 90


def _space_safe_socket_dir(pgdata, runtime_path):
    """Choose a postgres unix-socket dir that contains no spaces.

    pgserver hands the socket dir to postgres via ``pg_ctl -o "-k <dir>"`` — a
    string postgres re-parses by splitting on whitespace, so a dir with a space
    (e.g. the macOS-idiomatic ``~/Library/Application Support/…``) breaks
    startup with ``postgres: invalid argument``. The DATA dir stays in the
    platform's idiomatic per-OS location (it is passed to ``-D`` as a list
    argument, which preserves spaces); only the ephemeral socket is relocated to
    a short, space-free temp dir — where unix sockets conventionally live. When
    pgdata has no space, pgserver's own logic is used unchanged.
    """
    pgdata = Path(pgdata)
    if " " not in str(pgdata):
        return _orig_find_socket_dir(pgdata, runtime_path)
    base = Path(tempfile.gettempdir())  # space-free on macOS/Windows/Linux
    digest = hashlib.sha256(f"{pgdata}-{pgdata.stat().st_ino}".encode()).hexdigest()[:10]
    socket_dir = base / f"erudi-pg-{digest}"
    socket_dir.mkdir(parents=True, exist_ok=True)
    if not socket_name_length_ok(socket_dir / ".s.PGSQL.5432"):
        return _orig_find_socket_dir(pgdata, runtime_path)
    logger.info(f"Using space-free postgres socket dir: {socket_dir}")
    return socket_dir


# pgserver looks this name up on its own module at call time, so patch there.
_pg_server_mod.find_suitable_socket_dir = _space_safe_socket_dir


def _console_isolated(orig_command):
    """Wrap a pgserver command fn so its child gets its OWN hidden console (#162).

    Constraint: the postgres lineage must NOT share the backend's console. On the
    packaged Windows build the backend is a console app whose window is hidden by
    the Electron launcher; pgserver spawns pg_ctl/initdb (and through them the
    postmaster + its ~10 helpers) via plain ``subprocess.run`` with no
    ``creationflags``, so every postgres process inherits that single shared
    conhost. Killing that one conhost then tears down the whole cluster while the
    backend survives - the app runs green over a dead database (the #162
    packaged-app reproduction of the 0xC000013A incident). ``CREATE_NO_WINDOW``
    gives each child a console of its own, breaking the shared-conhost coupling.

    ``setdefault`` so an explicit caller-provided ``creationflags`` always wins;
    off Windows ``hidden_console_creationflags()`` is 0, a harmless no-op.
    Idempotent via a guard attribute so a re-run never double-wraps.
    """
    if getattr(orig_command, "_erudi_console_isolated", False):
        return orig_command

    def command(args, pgdata=None, **kwargs):
        kwargs.setdefault("creationflags", hidden_console_creationflags())
        return orig_command(args, pgdata=pgdata, **kwargs)

    command._erudi_console_isolated = True
    return command


# Same call-time lookup as the socket-dir patch: postgres_server resolves both
# names on its own module, so rebind them there.
_pg_server_mod.pg_ctl = _console_isolated(_pg_server_mod.pg_ctl)
_pg_server_mod.initdb = _console_isolated(_pg_server_mod.initdb)


def _prune_stale_handle_pids(data_dir: Path) -> None:
    """Drop dead pids from pgserver's per-cluster refcount registry.

    pgserver tracks cluster users in ``<pgdata>/.handle_pids.json`` but never
    prunes dead entries: a crashed/SIGKILLed backend leaves a ghost pid that
    makes every later ``cleanup()`` skip the server stop — forever. Pruning
    before joining the cluster guarantees the LAST live handle really stops
    the postmaster on graceful shutdown.
    """
    handle_file = data_dir / ".handle_pids.json"
    if not handle_file.exists():
        return
    try:
        pids = json.loads(handle_file.read_text() or "[]")
        alive = [pid for pid in pids if psutil.pid_exists(pid)]
        if alive != pids:
            handle_file.write_text(json.dumps(alive))
            logger.info(f"Pruned stale pgserver handle pids: {sorted(set(pids) - set(alive))}")
    except (OSError, ValueError) as exc:
        # Unreadable/corrupt registry — pgserver will rebuild it on boot.
        logger.warning(f"Could not prune pgserver handle pids: {exc}")


def _recover_corrupt_pgdata(data_dir: Path) -> None:
    """Wipe a half-initialized data dir so pgserver can run a clean initdb.

    An interrupted first-run initdb (e.g. the launcher was killed mid-init by an
    over-eager watchdog) can leave the data dir populated but WITHOUT a
    ``PG_VERSION`` file. On the next boot ``initdb`` refuses a non-empty target
    directory, so every later launch fails permanently until the user manually
    deletes the folder. Detect that state and clear the directory before handing
    it to pgserver, which then initializes a fresh cluster. A dir that already
    has ``PG_VERSION`` is a real cluster and is left untouched; an empty dir is
    fine as-is.
    """
    if (data_dir / "PG_VERSION").exists():
        return  # a real, initialized cluster — never touch it
    try:
        entries = list(data_dir.iterdir())
    except OSError:
        return
    if not entries:
        return  # empty — pgserver will initdb into it cleanly
    logger.warning(f"Recovering half-initialized Postgres data dir (no PG_VERSION): {data_dir}")
    for entry in entries:
        try:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(f"Could not remove {entry} during pgdata recovery: {exc}")


@dataclass(frozen=True)
class PostgresHandle:
    """Live embedded-cluster handle with the two derived connection URLs."""

    server: "pgserver.PostgresServer"
    data_dir: Path
    psycopg_url: str
    sqlalchemy_url: str


def _uri_for_db(base_uri: str, dbname: str) -> str:
    """Swap the database name in a pgserver URI (``…/postgres?host=…``)."""
    head, _, query = base_uri.partition("?")
    head = head.rsplit("/", 1)[0] + f"/{dbname}"
    return f"{head}?{query}" if query else head


# ``postgresql://<user>:<password>@`` -- pgserver always emits the user with an
# empty password (``postgres:@``); the authority that follows is either empty
# (socket form, host in the query) or ``host:port`` (TCP form).
_URI_CREDENTIALS = re.compile(
    r"^(?P<scheme>postgresql://)(?P<user>[^:@/?]+):(?P<password>[^@/?]*)@"
)


def _uri_with_password(uri: str, password: str) -> str:
    """Inject (or replace) the password in a pgserver URI, both socket and TCP forms.

    The password is percent-quoted so a value containing ``/``, ``@`` or ``?``
    cannot be mistaken for a URI delimiter. The generated secret is URL-safe
    already, so in practice quoting is the identity.
    """
    match = _URI_CREDENTIALS.match(uri)
    if match is None:
        raise ValueError("expected a postgresql://<user>:<password>@ URI")
    return f"{match['scheme']}{match['user']}:{quote(password, safe='')}@{uri[match.end() :]}"


def _ensure_cluster_password(data_dir: Path) -> str:
    """Read the per-cluster password, generating it on the cluster's first boot.

    Must run AFTER initdb: initdb refuses a non-empty target directory, so the
    file can only be added once ``PG_VERSION`` exists. The file is created
    ``0600`` (owner-only) through ``O_EXCL`` so two processes joining a brand
    new cluster at once cannot both win. On Windows the mode is best effort:
    ``%LOCALAPPDATA%`` is already private to the user's account, and NTFS ACL
    hardening is out of scope.
    """
    secret_file = data_dir / PASSWORD_FILE_NAME
    try:
        fd = os.open(secret_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return secret_file.read_text(encoding="ascii").strip()
    with os.fdopen(fd, "w", encoding="ascii") as fh:
        password = secrets.token_urlsafe(32)
        fh.write(password)
    try:
        secret_file.chmod(0o600)  # O_CREAT mode is subject to the umask
    except OSError:
        pass
    logger.info(f"Generated the embedded PostgreSQL password file: {secret_file}")
    return password


# A ``host`` / ``hostssl`` / ``hostnossl`` / ``hostgssenc`` / ``hostnogssenc``
# rule whose method is ``trust``: TYPE DATABASE USER ADDRESS METHOD, then an
# optional trailing comment. ``local`` rules are deliberately not matched.
_TRUSTED_HOST_RULE = re.compile(
    r"^(?P<rule>[ \t]*host\w*[ \t]+\S+[ \t]+\S+[ \t]+\S+[ \t]+)trust(?P<rest>[ \t]*(?:#.*)?)$",
    re.MULTILINE,
)


def harden_pg_hba(text: str) -> str:
    """Rewrite ``pg_hba.conf`` so every ``host`` rule authenticates with SCRAM.

    Pure text -> text: ``trust`` on a ``host*`` line becomes ``scram-sha-256``;
    ``local`` (Unix socket) rules, comments, blank lines, spacing and every
    other method are left byte-for-byte as they were. Idempotent.
    """
    return _TRUSTED_HOST_RULE.sub(r"\g<rule>scram-sha-256\g<rest>", text)


def _enforce_password_auth(admin_uri: str, data_dir: Path, password: str) -> None:
    """Set the role password and require it on every host connection.

    Runs on every boot, right after the cluster is up and before anything else
    connects. ``admin_uri`` already carries the password: under ``trust`` (a
    fresh or pre-#462 cluster) it is ignored, under ``scram-sha-256`` it is
    required, so one URI form works before and after the switch and existing
    installs migrate on their next start with no special path. ``ALTER ROLE``
    is idempotent; the ``pg_hba.conf`` rewrite only touches the file (and
    reloads the postmaster) when a ``trust`` host rule is still present, which
    also covers a cluster left running by a previous process (pgserver
    refcounts it): ``pg_reload_conf()`` is enough, no restart.
    """
    with psycopg.connect(admin_uri, autocommit=True) as conn:
        # scram-sha-256 is the default since PostgreSQL 14 (bundled: 16); pin
        # it for the session anyway so the stored verifier can never be md5.
        conn.execute("SET password_encryption = 'scram-sha-256'")
        conn.execute(
            sql.SQL("ALTER ROLE {} PASSWORD {}").format(
                sql.Identifier(ROLE_NAME), sql.Literal(password)
            )
        )
        hba_file = data_dir / "pg_hba.conf"
        before = hba_file.read_text(encoding="utf-8")
        after = harden_pg_hba(before)
        if after != before:
            tmp_file = hba_file.with_name(hba_file.name + ".erudi-tmp")
            tmp_file.write_text(after, encoding="utf-8")
            os.replace(tmp_file, hba_file)
            conn.execute("SELECT pg_reload_conf()")
            logger.info("Embedded PostgreSQL host connections now require the cluster password")


def _wait_for_postmaster_ready(data_dir: Path, deadline_seconds: float) -> bool:
    """Wait for a background postmaster to finish WAL crash-recovery (#161).

    pgserver starts the cluster with ``pg_ctl -w start`` under a HARDCODED
    ``timeout=10`` and re-raises ``subprocess.TimeoutExpired`` when that wait
    elapses. But the timeout only kills the *waiter* (``pg_ctl``), not the
    postmaster it already spawned: after an unclean shutdown the postmaster keeps
    replaying the WAL in the background and typically comes up fine a little
    later. A slow recovery is not a failure, so we give the live postmaster a
    patient (bounded) second chance instead of crashing the boot.

    Readiness reuses pgserver's OWN predicate (its post-start wait loop):
    ``PostmasterInfo.read_from_pgdata`` reports a running server whose ``status``
    is ``ready``. Returns True once ready, False if the deadline expires.
    """
    emit_phase("recovering_database")
    logger.info(
        f"Waiting up to {deadline_seconds:.0f}s for Postgres crash recovery to "
        f"finish (data_dir={data_dir})"
    )
    start = time.monotonic()
    deadline = start + deadline_seconds
    while time.monotonic() < deadline:
        pinfo = PostmasterInfo.read_from_pgdata(data_dir)
        if pinfo is not None and pinfo.is_running() and pinfo.status == "ready":
            logger.info(
                f"Postgres crash recovery finished after "
                f"{time.monotonic() - start:.1f}s; reusing the recovered postmaster"
            )
            return True
        time.sleep(1.0)
    logger.warning(f"Postgres crash recovery did not report ready within {deadline_seconds:.0f}s")
    return False


def _get_server_with_recovery(data_dir: Path):
    """Boot (or join) the cluster, tolerating a slow WAL crash-recovery (#161).

    ``pgserver.get_server`` starts the postmaster with ``pg_ctl -w`` under a
    hardcoded 10s timeout and re-raises ``subprocess.TimeoutExpired`` when the
    wait elapses. After an unclean shutdown the first boot must crash-recover,
    which routinely takes longer than 10s — yet the timeout only kills the
    ``pg_ctl`` waiter, not the postmaster, which keeps recovering in the
    background.

    ``ensure_postgres_running`` has a SECOND way to hit this same "still
    recovering" state: when a ``postmaster.pid`` already shows a running
    process (exactly what a retry after a timeout finds, or what a second
    launch attempt finds), it skips ``pg_ctl`` entirely and jumps straight to
    ``assert self._postmaster_info.status == 'ready'`` with no wait at all --
    it never checks status before asserting on it. A recovering-but-not-yet-
    ready postmaster then raises a bare ``AssertionError`` instead of
    ``TimeoutExpired``, which used to skip the patience logic below entirely
    and crash the boot on a build that would have come up fine given a few
    more seconds (reproduced live during QA: a hard-killed prior session left
    Postgres WAL-recovering for ~26s, longer than the fast path's zero-wait
    tolerance, and the app showed a permanent "Backend crashed on startup").

    Second-chance semantics: catch either exception, wait (bounded) for the
    live postmaster to report ready, then call ``get_server`` again — it
    reuses the already-running postmaster without touching ``pg_ctl`` (a
    manual Retry minutes after such a crash booted in 1.3s through exactly
    that path). If the wait expires we re-raise the ORIGINAL error, preserving
    today's failure path after real patience and clear logs.
    """
    try:
        return pgserver.get_server(str(data_dir))
    except (subprocess.TimeoutExpired, AssertionError):
        logger.warning(
            "pgserver reported the postmaster not ready yet (pg_ctl's "
            "hardcoded 10s timeout, or its no-wait already-running fast "
            "path); it is likely still WAL crash-recovering in the "
            "background - waiting for it to finish before retrying"
        )
        if _wait_for_postmaster_ready(data_dir, RECOVERY_WAIT_SECONDS):
            # pgserver caches the instance in _instances BEFORE starting the
            # server (postgres_server.py:62 precedes ensure_postgres_running at
            # :64), so after the TimeoutExpired a half-built object with
            # _postmaster_info=None is still cached — and get_server would hand
            # that corpse back, crashing on get_uri()'s assert despite the
            # successful recovery (#215). Evict it so the retry runs the full
            # constructor and rejoins the live postmaster. (The stale object's
            # atexit cleanup is harmless: with _postmaster_info=None it never
            # stops the server.)
            _pg_server_mod.PostgresServer._instances.pop(data_dir, None)
            return pgserver.get_server(str(data_dir))
        raise


def start_postgres(data_dir: Path | str) -> PostgresHandle:
    """Boot (or join) the embedded cluster and make the `erudi` DB ready.

    Idempotent: safe to call on an already-initialized data dir or while the
    cluster is already running (pgserver refcounts users of the data dir).
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    _prune_stale_handle_pids(data_dir)
    _recover_corrupt_pgdata(data_dir)

    server = _get_server_with_recovery(data_dir)
    # After initdb (the secret file must not pre-exist in PGDATA) and before
    # any other connection: the URLs handed out below all carry the password.
    password = _ensure_cluster_password(data_dir)
    admin_uri = _uri_with_password(server.get_uri(), password)
    _enforce_password_auth(admin_uri, data_dir, password)

    # CREATE DATABASE has no IF NOT EXISTS → guard on pg_database.
    # DB_NAME is an internal constant, never user input.
    with psycopg.connect(admin_uri, autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,)).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{DB_NAME}"')

    psycopg_url = _uri_for_db(admin_uri, DB_NAME)

    # pgvector extensions are per-database → create inside `erudi`, not the
    # admin DB probed above.
    with psycopg.connect(psycopg_url, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    sqlalchemy_url = psycopg_url.replace("postgresql://", "postgresql+psycopg://", 1)
    logger.info(f"Embedded PostgreSQL ready (data_dir={data_dir})")
    return PostgresHandle(
        server=server,
        data_dir=data_dir,
        psycopg_url=psycopg_url,
        sqlalchemy_url=sqlalchemy_url,
    )


def stop_postgres(handle: PostgresHandle) -> None:
    """Stop the embedded cluster explicitly (deterministic shutdown order)."""
    handle.server.cleanup()
    logger.info("Embedded PostgreSQL stopped")
