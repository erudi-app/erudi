"""Server settings of the embedded PostgreSQL, sized for one user (#620).

``initdb`` writes a configuration for a shared server: 100 connections,
128 MB of shared buffers, a replication-ready WAL. Erudi runs one cluster for
one person, so the backend passes its own values at every postmaster start,
as ``pg_ctl start -o "-c name=value"`` (see
``postgres_runtime._with_server_settings``). Nothing is written to
``postgresql.conf`` or ``postgresql.auto.conf``: a postmaster started by this
backend -- a first boot, a normal start, the start that a crash recovery
follows, a watchdog restart -- runs with these values, and a postmaster this
backend joins while it is already running keeps the ones it was started with
until it next restarts (``start_postgres`` logs the difference).

``max_connections`` is not chosen: it is the sum of every connection the
backend can hold at once (the constants below, which the pools are created
with) plus a margin.
"""

from __future__ import annotations

# --- connection budget --------------------------------------------------------
# Each constant is a connection the backend can hold on the cluster at the
# same time as all the others.

# The business SQLAlchemy engine (`src/database/core.py`, `init_database`):
# a QueuePool keeps up to `pool_size` connections open between requests and
# opens up to `max_overflow` more while that many are checked out at once.
# Each connection kept open is a server process of its own (5 to 9 MB), so
# two are kept. A streamed answer (chat, arena, title) holds its request's
# session -- and its connection, `idle in transaction` -- for the whole
# generation, waiting turns included. The worst one-user case measured (a
# Knowledge Base ingestion, two arena answers, two titles and a chat answer
# at once, the renderer polling on three threads) had 10 connections in a
# transaction at once, and exhausted a pool of 8; up to 12 can be open.
BUSINESS_POOL_SIZE = 2
BUSINESS_POOL_MAX_OVERFLOW = 10

# The langchain-postgres `PGEngine` behind the Knowledge Base store
# (`src/ingestion/vector_store.py`, `init_kb_store`): an async pool of its own.
# A search and an ingestion each use one connection at a time.
KB_STORE_POOL_SIZE = 1
KB_STORE_POOL_MAX_OVERFLOW = 2

# How long a request waits for a pooled connection before it fails with
# DATABASE_BUSY (503, `src/core/exceptions.py`). SQLAlchemy's own default,
# stated so the error can say it.
POOL_TIMEOUT_SECONDS = 30

# The LangGraph `AsyncPostgresSaver` (`src/agents/checkpoint.py`): one
# connection held for the app's lifetime.
CHECKPOINTER_CONNECTIONS = 1

# `run_migrations` (`src/database/migrations.py`) opens an engine of its own
# at boot, beside the business engine. It has no pool (NullPool): one
# connection at a time while a step runs, none once the schema is at head.
MIGRATION_CONNECTIONS = 1

# The pre-migration `pg_dump` (`src/database/backup.py`): one connection, no
# parallel jobs.
BACKUP_CONNECTIONS = 1

# Short `psycopg.connect` calls outside the pools: the admin connection of
# `start_postgres`, `init_kb_store`'s schema checks, and the dense-similarity
# read of every Knowledge Base search (`_dense_similarities`). Two cover one
# search at a time beside a boot step; searches running at once (two arena
# answers on Knowledge Base assistants) open one each and use the margin.
DIRECT_CONNECTIONS = 2

# Room for what the sum does not hold: a `psql` opened during a bug hunt, the
# overlap of a DB watchdog restart, where the old checkpointer is closed only
# after the new one is open, and the direct connections of Knowledge Base
# searches running at once beyond `DIRECT_CONNECTIONS`.
CONNECTION_MARGIN = 5


def max_connections() -> int:
    """Every connection the backend can hold at once, plus the margin."""
    return (
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


def server_settings() -> dict[str, str]:
    """The settings passed to the postmaster at start, in ``SHOW``'s own format.

    Each value is written the way ``current_setting()`` reports it, so a
    running cluster can be compared with this dict as it is.
    """
    return {
        "max_connections": str(max_connections()),
        # A desktop database of tens of MB; the KB search latency measured at
        # 128, 32 and 16 MB is in docs/architecture.md (Persistence).
        "shared_buffers": "32MB",
        # No replica and no point-in-time recovery ever read this cluster's
        # WAL; `pg_dump` (the only backup) is a logical dump and needs no WAL.
        # `minimal` requires `max_wal_senders = 0`.
        "wal_level": "minimal",
        "max_wal_senders": "0",
        # No subscription ever runs here: without workers the postmaster does
        # not start the logical replication launcher, one process fewer.
        "max_logical_replication_workers": "0",
        # One writer, a handful of small tables: one autovacuum worker at a
        # time instead of three after an ingestion.
        "autovacuum_max_workers": "1",
        # WAL kept on disk between checkpoints: two 16 MB segments are
        # recycled instead of five, and a large ingestion checkpoints at
        # 256 MB of WAL instead of 1 GB.
        "min_wal_size": "32MB",
        "max_wal_size": "256MB",
    }


# Characters cmd.exe interprets in the command line pg_ctl builds on Windows.
_CMD_METACHARACTERS = frozenset('&|<>^%"')


def postmaster_options(settings: dict[str, str] | None = None) -> list[str]:
    """``pg_ctl`` arguments that pass ``settings`` to the postmaster.

    One ``-o "-c name=value"`` per setting. ``pg_ctl`` joins the ``-o``
    strings and postgres splits them on whitespace, and on Windows ``pg_ctl``
    starts the postmaster through ``cmd.exe``, which interprets
    ``& | < > ^ % "`` in that string: a name or value carrying either would
    break the start or run something else, so it is refused here.
    """
    if settings is None:
        settings = server_settings()
    options: list[str] = []
    for name, value in settings.items():
        text = name + value
        if (
            not name
            or not value
            or any(ch.isspace() for ch in text)
            or any(ch in _CMD_METACHARACTERS for ch in text)
        ):
            raise ValueError(
                f"postmaster setting {name!r}={value!r} must be non-empty, "
                "without spaces or cmd metacharacters"
            )
        options += ["-o", f"-c {name}={value}"]
    return options
