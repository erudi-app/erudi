"""Pre-migration logical backup of the embedded database.

Why a dump and not a filesystem copy: migrations run while the embedded cluster
is LIVE (lifespan starts Postgres, then migrates), and copying a running cluster's
data dir yields a torn/inconsistent snapshot. ``pg_dump`` is the safe online backup.

Why we keep it even though Postgres DDL is transactional: a failed migration rolls
back to the last good revision (no corruption) and the new app fails fast rather
than running on a mismatched schema — recovery is to reinstall the previous app
version. This dump is the extra safety net for DESTRUCTIVE migrations (column
drops, data rewrites) where the pre-migration DATA would otherwise be lost: the
snapshot is taken BEFORE a migration applies, so the old data can be restored
(``pg_restore``) and used with the matching previous app version.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pgserver
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from src.core.logging import logger
from src.core.subprocess_flags import hidden_console_creationflags

# Number of most-recent snapshots to retain; older ones are pruned.
KEEP_BACKUPS = 3


def _dump_target(psycopg_url: str) -> tuple[str, dict[str, str]]:
    """Split the cluster URL into a password-free conninfo and the child env.

    ``pg_dump`` receives ``--dbname`` on its command line, which the process
    list shows to every process of the user and which ``CalledProcessError``
    repeats in the log on failure. The per-cluster password (#462) therefore
    travels through ``PGPASSWORD`` in the child's environment instead; both
    the Unix-socket and the loopback-TCP URI forms are handled by psycopg's
    parser.
    """
    params = conninfo_to_dict(psycopg_url)
    password = params.pop("password", None)
    env = dict(os.environ)
    if password:
        env["PGPASSWORD"] = password
    return make_conninfo(**params), env


def _pg_dump_bin() -> Path:
    """The pg_dump binary from pgserver's bundled Postgres (dev and frozen).

    pgserver extracts pginstall/ to the filesystem to spawn Postgres, so this
    path resolves both in dev and in the PyInstaller bundle (collect_all ships it).
    """
    exe = "pg_dump.exe" if os.name == "nt" else "pg_dump"
    return Path(pgserver.__file__).resolve().parent / "pginstall" / "bin" / exe


def backups_dir_for(data_dir: Path | str) -> Path:
    """Snapshots live beside the cluster dir, in the same persistent data area."""
    return Path(data_dir).resolve().parent / "db-backups"


def backup_database(psycopg_url: str, data_dir: Path | str, label: str) -> Path:
    """pg_dump the database (custom format) before a migration. Returns the path.

    Raises ``subprocess.CalledProcessError`` if pg_dump fails (the caller decides
    whether to proceed — a backup failure must not silently skip the snapshot).
    """
    out_dir = backups_dir_for(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_path = out_dir / f"erudi-{label}.dump"
    conninfo, env = _dump_target(psycopg_url)

    subprocess.run(
        [
            str(_pg_dump_bin()),
            "--format=custom",
            "--dbname",
            conninfo,
            "--file",
            str(dump_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        # pg_dump is a console exe; keep it from flashing a terminal window
        # at boot on Windows (#175). No-op (0) on POSIX.
        creationflags=hidden_console_creationflags(),
    )

    _prune(out_dir)
    logger.info("Pre-migration backup written: %s", dump_path)
    return dump_path


def _prune(out_dir: Path) -> None:
    dumps = sorted(
        out_dir.glob("erudi-*.dump"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for stale in dumps[KEEP_BACKUPS:]:
        try:
            stale.unlink()
        except OSError:
            pass
