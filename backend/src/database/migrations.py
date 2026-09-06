"""Programmatic Alembic runner for the embedded PostgreSQL cluster.

The embedded cluster's data dir is persistent and survives app updates, so the
on-disk schema must be reconciled to the models on every startup. This applies
forward-only Alembic revisions (lifespan, after ``init_database``), replacing the
old ``Base.metadata.create_all`` path which could never add a column/table to an
already-existing database.

Transition handling (adopting Alembic on a live install base): a database created
by the pre-Alembic ``create_all`` path holds the business tables but has no
``alembic_version`` table. Replaying the baseline there would fail (its
``CREATE TABLE`` collides with the existing tables), so such a database is
**stamped** to the baseline first; ``upgrade head`` then applies anything newer.
A fresh (empty) database simply runs ``upgrade head`` from zero.
"""

from __future__ import annotations

import subprocess

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect

from src.core.config import ROOT_DIR
from src.core.logging import logger
from src.database.backup import backup_database, describe_dump_failure
from src.launcher.postgres_runtime import PostgresHandle

# The root revision (see alembic/versions/…_baseline_schema.py). A pre-Alembic
# database already matches this schema, so it is stamped to this revision.
BASELINE_REVISION = "5ac171e299c6"

# A business table present in ANY pre-Alembic database (created by create_all).
# Its presence without an alembic_version table signals "adopt at baseline".
_SENTINEL_TABLE = "llms"


def _alembic_config(sqlalchemy_url: str) -> Config:
    """Build a Config bound to the live cluster, resolved against ROOT_DIR.

    ROOT_DIR is the backend root in dev and the bundle root when frozen; both
    ship ``alembic.ini`` + ``alembic/`` there (see the PyInstaller specs). Paths
    are set explicitly so resolution never depends on the process cwd.
    """
    cfg = Config(str(ROOT_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT_DIR / "alembic"))
    # ConfigParser interpolates '%' in option values; the URL carries the
    # per-cluster password (#462), percent-quoted, so escape it the way the
    # Alembic documentation prescribes.
    cfg.set_main_option("sqlalchemy.url", sqlalchemy_url.replace("%", "%%"))
    # Do NOT let env.py's fileConfig reconfigure (and disable) the app's loggers.
    cfg.attributes["configure_logger"] = False
    return cfg


def _head_revision(cfg: Config) -> str:
    """The latest revision in the bundled migration scripts."""
    return ScriptDirectory.from_config(cfg).get_current_head()


def run_migrations(handle: PostgresHandle) -> None:
    """Bring the database schema to head, snapshotting first when one will apply.

    Synchronous (Alembic is sync) — call via ``run_in_threadpool`` from the async
    lifespan so it does not block the event loop. On Postgres the upgrade runs in a
    transaction (env.py wraps it), so a failure rolls back to the last good
    revision rather than leaving a half-migrated schema.
    """
    cfg = _alembic_config(handle.sqlalchemy_url)
    engine = create_engine(handle.sqlalchemy_url)
    try:
        with engine.connect() as conn:
            current = MigrationContext.configure(conn).get_current_revision()
            has_business_tables = inspect(conn).has_table(_SENTINEL_TABLE)

        # Adopt a pre-Alembic database (full schema, no alembic_version) by stamping
        # the baseline — replaying the baseline's CREATE TABLEs would collide.
        if current is None and has_business_tables:
            logger.info(
                "Existing pre-Alembic schema detected - stamping baseline %s",
                BASELINE_REVISION,
            )
            command.stamp(cfg, BASELINE_REVISION)
            current = BASELINE_REVISION

        head = _head_revision(cfg)
        # Snapshot only when a migration will ACTUALLY apply to existing data
        # (a fresh DB has nothing to lose; an at-head DB has nothing to do).
        if has_business_tables and current != head:
            try:
                backup_database(handle.psycopg_url, handle.data_dir, label=current or "unknown")
            except subprocess.CalledProcessError as exc:
                # A failed snapshot must NOT silently proceed into a (possibly
                # destructive) migration with no safety net. pg_dump's own
                # stderr is the reason; its argv is not, and repeating it (as
                # `exc_info` would, through CalledProcessError.__str__) puts a
                # connection string in a file meant to be pasted into an issue.
                logger.error(
                    f"Pre-migration backup failed - aborting migration. "
                    f"{describe_dump_failure(exc)}"
                )
                raise
            except Exception:
                logger.error("Pre-migration backup failed - aborting migration", exc_info=True)
                raise

        try:
            command.upgrade(cfg, "head")
        except Exception:
            logger.error(
                "Migration to head failed; schema rolled back to the last good "
                "revision. Restore the pre-migration snapshot with the previous app "
                "version if data is affected (see db-backups/).",
                exc_info=True,
            )
            raise
        logger.info("Database schema is at head")
    finally:
        engine.dispose()
