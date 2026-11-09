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

Handle list (#695). pgserver's refcount is ``<pgdata>/.handle_pids.json``, and
its ``cleanup()`` stops the postmaster only when that list is exactly the
calling process. A backend that dies without cleaning up (a crash, a forced
kill, Windows shut down with the app open) leaves its pid there for good, and
on Windows that pid soon names an unrelated process. So ``start_postgres``
first drops every entry that is not a live backend (same program, started
before the list was last written), and ``stop_postgres`` stops a postmaster
that ``cleanup()`` left running when no live backend still uses it. Both run
under pgserver's own inter-process lock.

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
import logging
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

# pgserver's list of the processes holding a handle on the cluster, inside
# PGDATA (postgres_server.py: ``DiskList(pgdata / '.handle_pids.json')``). Its
# ``cleanup()`` stops the postmaster only when this list is exactly the
# calling process.
HANDLE_PIDS_FILE_NAME = ".handle_pids.json"

# A process that registered in the handle list existed before the list was
# last written. Wall-clock create times and file mtimes are compared, so allow
# for coarse clocks: Linux derives a create time from the boot time, which it
# reports in whole seconds.
HANDLE_REGISTRATION_SLACK_SECONDS = 2.0

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


# pgserver starts the postmaster with ``pg_ctl -l <pgdata>/log``
# (postgres_server.py:41,159), and `logging_collector` stays off, so the
# server's ENTIRE stderr -- WAL replay progress, a corrupt page, "could not
# write: No space left on device", the FATAL that stopped the boot -- is that
# one file. It is the only account of why the database did not come up: our
# own records say what WE asked for, never what Postgres answered.
POSTMASTER_LOG_NAME = "log"

# Lines quoted from it, and a hard cap on the text, so one runaway line cannot
# dominate the record the Diagnostics page shows.
POSTMASTER_LOG_TAIL_LINES = 40
POSTMASTER_LOG_TAIL_CHARS = 4000

# Nothing rotates that file: it grows for the life of the cluster. Read a
# window from its end rather than the whole thing -- quoting 40 lines must not
# load megabytes at the moment the app is already failing to start.
POSTMASTER_LOG_WINDOW_BYTES = 64 * 1024

# ``PASSWORD '<literal>'`` in any statement PostgreSQL echoed into its log.
# `log_min_error_statement` defaults to ERROR, so a REFUSED
# ``ALTER ROLE ... PASSWORD '<clear>'`` (see `_enforce_password_auth`) is
# written there verbatim -- and this file is quoted into backend.log, the
# Diagnostics page, and whatever a user pastes into a public issue. That
# statement is silenced at the source; this is the second layer, because a
# password must not depend on one setting being right.
_LOGGED_PASSWORD = re.compile(
    r"(?i)\bPASSWORD\s+('(?:[^']|'')*'|\"(?:[^\"]|\"\")*\")",
)


def _redact_passwords(text: str) -> str:
    """Replace every ``PASSWORD '<literal>'`` value with a marker."""
    return _LOGGED_PASSWORD.sub("PASSWORD '[redacted]'", text)


def postmaster_log_tail(data_dir: Path | str, max_lines: int = POSTMASTER_LOG_TAIL_LINES) -> str:
    """The embedded postmaster's own last lines, or ``""`` when it wrote none.

    Reads a bounded window from the end of the file (the log is unbounded),
    drops the partial line that window starts on, and redacts any password
    literal PostgreSQL echoed from a failing statement.

    Decoded defensively: Postgres writes its messages in the server encoding
    and in the operating system's language, so a byte that is not UTF-8 must
    cost a character, never the whole diagnostic. Never raises -- an absent or
    unreadable file simply has nothing to say.
    """
    log_file = Path(data_dir) / POSTMASTER_LOG_NAME
    try:
        size = log_file.stat().st_size
        start = max(0, size - POSTMASTER_LOG_WINDOW_BYTES)
        with log_file.open("rb") as handle:
            if start:
                handle.seek(start)
            raw = handle.read(POSTMASTER_LOG_WINDOW_BYTES)
    except OSError:
        return ""
    text = raw.decode("utf-8", errors="replace")
    if start:
        # The window landed mid-line; that fragment belongs to a line whose
        # beginning was not read.
        _, _, text = text.partition("\n")
    lines = [line for line in text.splitlines() if line.strip()]
    tail = _redact_passwords("\n".join(lines[-max_lines:]))
    return tail[-POSTMASTER_LOG_TAIL_CHARS:] if len(tail) > POSTMASTER_LOG_TAIL_CHARS else tail


def _attach_postmaster_log(exc: BaseException, data_dir: Path) -> None:
    """Carry the postmaster's last words along with the exception.

    A note (PEP 678) rather than a new exception: the failure paths propagate
    ``subprocess.TimeoutExpired`` and ``AssertionError`` on purpose (the
    watchdog and the launcher recognise them), and a note travels inside the
    traceback the lifespan already writes to ``backend.log`` at ERROR. The
    cause therefore reaches the Diagnostics page with the failure it explains,
    the same way a dead inference child's output does.
    """
    tail = postmaster_log_tail(data_dir)
    log_file = Path(data_dir) / POSTMASTER_LOG_NAME
    note = (
        f"Embedded PostgreSQL log ({log_file}):\n{tail}"
        if tail
        else f"The embedded PostgreSQL log ({log_file}) is empty or unreadable."
    )
    try:
        exc.add_note(note)
    except Exception:
        # A diagnostic must never be what turns a failure into a crash.
        pass


def _own_identity() -> tuple[str | None, str | None]:
    """This process's executable path and name: what a backend looks like."""
    me = psutil.Process()
    try:
        exe = me.exe()
    except psutil.Error:
        exe = None
    return exe, me.name()


def _is_live_backend(
    pid: int, registered_by: float, identity: tuple[str | None, str | None]
) -> bool:
    """Whether ``pid`` can be a live backend that registered a handle by ``registered_by``.

    Three checks, because a pid alone proves nothing once its process is gone:
    Windows hands a freed pid to the next process within seconds, and after a
    reboot a stale entry names whatever system service now holds that number.

    - the process is alive (not a zombie);
    - it runs the backend's program: the same executable as this process, or
      the same name (a Windows venv starts Python through a launcher whose
      path differs from the interpreter's, under the same name);
    - it existed when the list was last written (``registered_by``, the
      file's mtime): a handle is registered by a running process, so one
      started later is a stranger that inherited the number.

    Anything unreadable (another user's process) is not a backend: the
    backend always runs under the account that starts this one.
    """
    exe, name = identity
    try:
        proc = psutil.Process(pid)
        if proc.status() == psutil.STATUS_ZOMBIE:
            return False
        if proc.create_time() > registered_by + HANDLE_REGISTRATION_SLACK_SECONDS:
            return False
        if proc.name() == name:
            return True
        try:
            return exe is not None and proc.exe() == exe
        except psutil.Error:
            return False
    except psutil.Error:
        return False


def _read_handle_pids(handle_file: Path) -> tuple[list[int], float] | None:
    """The handle list and the time it was last written; ``None`` without a file.

    Raises ``ValueError`` for a file that is not a JSON list of pids.
    """
    try:
        registered_by = handle_file.stat().st_mtime
    except FileNotFoundError:
        return None
    pids = json.loads(handle_file.read_text() or "[]")
    if not isinstance(pids, list) or not all(
        isinstance(pid, int) and not isinstance(pid, bool) for pid in pids
    ):
        raise ValueError(f"expected a JSON list of pids, got {pids!r}")
    return pids, registered_by


def _prune_stale_handle_pids(data_dir: Path) -> None:
    """Keep only live backends in pgserver's handle list before joining the cluster.

    pgserver records every process holding a handle in
    ``<pgdata>/.handle_pids.json`` and never removes an entry whose process
    died without calling ``cleanup()`` (a crash, a forced kill, Windows shut
    down with the app open). Its ``cleanup()`` stops the postmaster only when
    the list is exactly the calling process, so one such entry used to keep
    every later quit from stopping the cluster: the OS then killed the
    postmaster with the backend, and every start ran a crash recovery.

    An entry survives when it is this process (a restart of the cluster from
    inside the same backend, where the cached pgserver handle is reused
    without registering again) or a live backend (:func:`_is_live_backend`):
    nothing keeps a second app instance from starting, and its backend shares
    this data dir. Everything else is dropped.

    Runs under pgserver's own inter-process lock (the one its constructor and
    ``cleanup()`` hold while they read and write the list), so it never
    interleaves with another process joining or leaving the cluster. A file
    that is not a list of pids is reset: pgserver's reader would raise on it
    and fail the start.
    """
    handle_file = data_dir / HANDLE_PIDS_FILE_NAME
    with _pg_server_mod.PostgresServer._lock:
        try:
            listed = _read_handle_pids(handle_file)
        except ValueError as exc:
            handle_file.write_text("[]")
            logger.warning(f"Reset pgserver's unreadable handle list {handle_file}: {exc}")
            return
        except OSError:
            # Not logged here: pgserver reads the same file next and fails the
            # start with this error, which the start's caller records once.
            return
        if listed is None:
            return
        pids, registered_by = listed
        identity = _own_identity()
        kept = [
            pid
            for pid in pids
            if pid == os.getpid() or _is_live_backend(pid, registered_by, identity)
        ]
        if kept == pids:
            return
        handle_file.write_text(json.dumps(kept))
    pruned = [pid for pid in pids if pid not in kept]
    logger.info(
        f"Pruned pgserver handle pids that are not live Erudi backends: {pruned} (kept: {kept})"
    )


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
    except OSError as exc:
        # Unreadable data dir: initdb is about to fail on the same directory,
        # and this is the first and only hint of why (a permissions change, a
        # dismounted volume, a synced folder gone read-only).
        logger.warning(f"Could not inspect the Postgres data dir {data_dir}: {exc}")
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
        existing = secret_file.read_text(encoding="ascii").strip()
        if existing:
            return existing
        # A truncated file would set an empty password, which PostgreSQL
        # stores as NULL and can never match on a SCRAM host connection.
        logger.warning("Embedded PostgreSQL password file is empty; generating a new one")
        fd = os.open(secret_file, os.O_WRONLY | os.O_TRUNC, 0o600)
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


# The same rule shape with the hardened method: the inverse of the above.
_SCRAM_HOST_RULE = re.compile(
    r"^(?P<rule>[ \t]*host\w*[ \t]+\S+[ \t]+\S+[ \t]+\S+[ \t]+)scram-sha-256(?P<rest>[ \t]*(?:#.*)?)$",
    re.MULTILINE,
)


def _relax_pg_hba(text: str) -> str:
    """Inverse of ``harden_pg_hba``: ``scram-sha-256`` host rules back to ``trust``.

    Only used to re-key a cluster whose password file is gone (see
    ``_connect_admin_rekeying``); same byte-for-byte guarantees, idempotent.
    """
    return _SCRAM_HOST_RULE.sub(r"\g<rule>trust\g<rest>", text)


def _write_pg_hba(hba_file: Path, text: str) -> None:
    """Atomic replace so the postmaster never reads a half-written file."""
    tmp_file = hba_file.with_name(hba_file.name + ".erudi-tmp")
    tmp_file.write_text(text, encoding="utf-8")
    os.replace(tmp_file, hba_file)


def _is_password_failure(exc: psycopg.OperationalError) -> bool:
    return "password authentication failed" in str(exc)


def _connect_admin_rekeying(admin_uri: str, data_dir: Path) -> psycopg.Connection:
    """Open the admin connection, re-keying the cluster if its secret was lost.

    On Windows (host connections only) a ``pg_hba.conf`` already on SCRAM plus
    a deleted ``erudi_db_password`` means the freshly generated secret is one
    the role does not have: every connection is refused and, without this,
    every later boot too. When the FIRST connection fails on password
    authentication, the host rules are relaxed back to ``trust``, the
    postmaster is reloaded through ``pg_ctl reload`` (no connection is
    possible yet), and the connection is retried once -- the caller then sets
    the new password and re-hardens the file as on any boot. Any other
    ``OperationalError``, and a second password failure, propagate.
    """
    try:
        return psycopg.connect(admin_uri, autocommit=True)
    except psycopg.OperationalError as exc:
        if not _is_password_failure(exc):
            raise
        refusal = exc
    hba_file = data_dir / "pg_hba.conf"
    _write_pg_hba(hba_file, _relax_pg_hba(hba_file.read_text(encoding="utf-8")))
    _pg_server_mod.pg_ctl(["reload"], pgdata=data_dir)
    # `exc_info` explicitly (the except block is over, so `True` would find
    # nothing): the postmaster's own refusal is what says WHICH role and rule
    # rejected us, and this is the only record of a recovery nobody asked for.
    logger.warning(
        "Embedded PostgreSQL refused the cluster password (password file regenerated?); "
        "re-keying the cluster with the new secret",
        exc_info=refusal,
    )
    return psycopg.connect(admin_uri, autocommit=True)


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
    refcounts it): ``pg_reload_conf()`` is enough, no restart. A lost password
    file is recovered by ``_connect_admin_rekeying`` before any of this runs.
    """
    with _connect_admin_rekeying(admin_uri, data_dir) as conn:
        # scram-sha-256 is the default since PostgreSQL 14 (bundled: 16); pin
        # it for the session anyway so the stored verifier can never be md5.
        conn.execute("SET password_encryption = 'scram-sha-256'")
        # `log_min_error_statement` defaults to ERROR, which makes PostgreSQL
        # echo the statement that failed into its own log -- and the next
        # statement is the only one in this app that carries a secret in
        # clear. That log is read back on a failed start and quoted into
        # backend.log and the Diagnostics page, so a refused ALTER ROLE would
        # publish the cluster password. Silenced for this session only (the
        # connection is closed a few lines below) and restored right after, so
        # nothing else stops being logged. `SET`, not `SET LOCAL`: the
        # connection is autocommit, where a LOCAL setting dies with the
        # implicit transaction of the very statement that set it.
        conn.execute("SET log_min_error_statement = 'panic'")
        try:
            conn.execute(
                sql.SQL("ALTER ROLE {} PASSWORD {}").format(
                    sql.Identifier(ROLE_NAME), sql.Literal(password)
                )
            )
        finally:
            conn.execute("RESET log_min_error_statement")
        hba_file = data_dir / "pg_hba.conf"
        before = hba_file.read_text(encoding="utf-8")
        after = harden_pg_hba(before)
        if after != before:
            _write_pg_hba(hba_file, after)
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


class _HeldPgserverErrors(logging.Filter):
    """Hold back pgserver's ``ERROR`` and above until a start's outcome is known.

    Attached to the ``pgserver`` logger by :func:`_get_server_with_recovery`,
    which decides afterwards at which level each held record is written.
    Records below ``ERROR`` pass through untouched.
    """

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            self.records.append(record)
            return False
        return True


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

    pgserver logs its own ``ERROR`` before raising -- ``Timeout starting
    server.`` followed by the postmaster's whole log and its absolute path --
    and at that moment nothing knows yet whether the start will happen. So
    pgserver's ``ERROR`` and ``CRITICAL`` records are held back for the whole
    call (:class:`_HeldPgserverErrors`; its lower levels pass through) and
    written once the outcome is known:

    - the retry after a completed recovery wait returns a server: the start
      happened, so each held record is written at ``INFO`` through the app
      logger, which keeps it in ``backend.log`` and off the Diagnostics
      page's error list (a library's ``INFO`` would not reach the file);
    - any other outcome -- the first call succeeds, raises something not
      caught here, the wait expires, or the retry raises: each held record is
      replayed unchanged, at its own level, on the ``pgserver`` logger, before
      the exception propagates.
    """
    pg_logger = logging.getLogger("pgserver")
    held = _HeldPgserverErrors()
    recovered = False
    pg_logger.addFilter(held)
    try:
        try:
            return pgserver.get_server(str(data_dir))
        except (subprocess.TimeoutExpired, AssertionError) as exc:
            # `TimeoutExpired` and `AssertionError` are two different stories
            # about the same symptom: the warning names which one this was,
            # and the traceback says where. The traceback is written at INFO,
            # not on the warning, because `TimeoutExpired`'s message is
            # pg_ctl's whole command line -- the absolute data and log paths
            # -- and the warning is listed on the Diagnostics page.
            logger.warning(
                "pgserver reported the postmaster not ready yet (pg_ctl's "
                "hardcoded 10s timeout, or its no-wait already-running fast "
                "path); it is likely still WAL crash-recovering in the "
                "background - waiting for it to finish before retrying "
                f"({type(exc).__name__})"
            )
            logger.info("Traceback of the postmaster not-ready report above", exc_info=exc)
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
                server = pgserver.get_server(str(data_dir))
                recovered = True
                return server
            raise
    finally:
        # Removed before anything is written: `Logger.handle` applies the
        # logger's filters, so a replay with the filter attached is held again.
        pg_logger.removeFilter(held)
        for record in held.records:
            if recovered:
                logger.info(
                    "pgserver reported this during a start that crash recovery "
                    f"then completed: {record.getMessage()}"
                )
            else:
                pg_logger.handle(record)


def start_postgres(data_dir: Path | str) -> PostgresHandle:
    """Boot (or join) the embedded cluster and make the `erudi` DB ready.

    Idempotent: safe to call on an already-initialized data dir or while the
    cluster is already running (pgserver refcounts users of the data dir).

    Every failure below -- the postmaster that never came up, an admin
    connection refused, a database or an extension that could not be created
    -- leaves with the postmaster's own last lines attached
    (:func:`_attach_postmaster_log`), because the reason is in the server's
    log and nowhere else.
    """
    data_dir = Path(data_dir)
    try:
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
            exists = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,)
            ).fetchone()
            if not exists:
                conn.execute(f'CREATE DATABASE "{DB_NAME}"')

        psycopg_url = _uri_for_db(admin_uri, DB_NAME)

        # pgvector extensions are per-database → create inside `erudi`, not the
        # admin DB probed above.
        with psycopg.connect(psycopg_url, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except BaseException as exc:
        _attach_postmaster_log(exc, data_dir)
        raise

    sqlalchemy_url = psycopg_url.replace("postgresql://", "postgresql+psycopg://", 1)
    logger.info(f"Embedded PostgreSQL ready (data_dir={data_dir})")
    return PostgresHandle(
        server=server,
        data_dir=data_dir,
        psycopg_url=psycopg_url,
        sqlalchemy_url=sqlalchemy_url,
    )


def _live_postmaster(data_dir: Path) -> psutil.Process | None:
    """The postmaster ``postmaster.pid`` names, if that process is a live postgres.

    The name check matters: a postmaster killed with the OS leaves its pid
    file behind, and the pid it names may since belong to another program,
    which ``pg_ctl stop`` would signal.
    """
    try:
        info = PostmasterInfo.read_from_pgdata(data_dir)
    except (OSError, ValueError, AssertionError):
        # Absent, unreadable, or half-written: no postmaster to stop here.
        return None
    if info is None or info.process is None:
        return None
    try:
        proc = info.process
        if (
            proc.is_running()
            and proc.status() != psutil.STATUS_ZOMBIE
            and proc.name().lower().startswith("postgres")
        ):
            return proc
    except psutil.Error:
        # Exited (or became unreadable) between the pid file and the check.
        pass
    return None


def _stop_postmaster_pgserver_kept(handle: PostgresHandle) -> bool:
    """Stop the postmaster ``cleanup()`` left running, unless a live backend uses it.

    pgserver's ``cleanup()`` returns without stopping anything when its handle
    list holds anything besides this process. The start prunes the list
    (:func:`_prune_stale_handle_pids`), but an entry can still arrive later,
    and a postmaster left running at quit is killed with the backend, which
    costs a crash recovery on the next start. Under pgserver's lock: when no
    remaining entry is a live backend, stop the postmaster with
    ``pg_ctl -w stop -m fast`` (the console-isolated wrapper) and empty the
    list; when one is, leave the cluster to it.

    Either way pgserver's cached handle is evicted when it is still ours: it
    no longer counts this process, so a later start in this process must
    register again rather than reuse it.

    Returns whether the cluster is stopped.
    """
    cache = _pg_server_mod.PostgresServer._instances
    for key, server in list(cache.items()):
        if server is handle.server:
            del cache[key]
    data_dir = Path(handle.data_dir)
    handle_file = data_dir / HANDLE_PIDS_FILE_NAME
    with _pg_server_mod.PostgresServer._lock:
        if _live_postmaster(data_dir) is None:
            return True
        try:
            listed = _read_handle_pids(handle_file)
        except (OSError, ValueError):
            # Nothing readable says another backend is there.
            listed = None
        pids, registered_by = listed if listed is not None else ([], 0.0)
        identity = _own_identity()
        users = [pid for pid in pids if _is_live_backend(pid, registered_by, identity)]
        if users:
            logger.info(
                f"Embedded PostgreSQL left running: another live backend uses it (pids {users})"
            )
            return False
        logger.warning(
            f"pgserver's handle count did not allow a clean stop (handles left: {pids}); "
            "stopping the embedded PostgreSQL cluster with pg_ctl"
        )
        try:
            _pg_server_mod.pg_ctl(["-w", "stop", "-m", "fast"], pgdata=data_dir)
        except (subprocess.CalledProcessError, OSError) as exc:
            logger.error(
                "Could not stop the embedded PostgreSQL cluster; the next start "
                "will run a crash recovery",
                exc_info=exc,
            )
            return False
        handle_file.write_text("[]")
    return True


def stop_postgres(handle: PostgresHandle) -> None:
    """Stop the embedded cluster explicitly (deterministic shutdown order).

    pgserver's ``cleanup()`` stops the postmaster when this process holds the
    only handle; :func:`_stop_postmaster_pgserver_kept` stops it when pgserver
    declined and no other live backend uses it.
    """
    # Announced before the call: pgserver's cleanup() stops the postmaster
    # (pg_ctl stop, then a kill), and a shutdown that hangs or throws there
    # would otherwise leave the log ending on the last thing the app did.
    logger.info(f"Stopping the embedded PostgreSQL cluster (data_dir={handle.data_dir})")
    handle.server.cleanup()
    if _stop_postmaster_pgserver_kept(handle):
        logger.info("Embedded PostgreSQL stopped")
