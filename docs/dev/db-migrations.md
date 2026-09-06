# Database migrations (Alembic)

Erudi's embedded PostgreSQL data dir is **persistent and survives app updates**, so
the on-disk schema must be reconciled to the models on every launch. We use
**Alembic**, applied automatically at startup.

## How it works

- **Auto-upgrade at startup.** The lifespan (`src/core/api.py`, step 4) calls
  `src.database.migrations.run_migrations(handle)` right after the cluster is up:
  it brings the schema to `head` (`alembic upgrade head`). This replaces the old
  `create_all`, which could never alter an already-existing database.
- **Forward-only.** We do not run `downgrade()` in production. Roll problems
  forward with a new revision.
- **Transactional.** On PostgreSQL the upgrade runs inside a transaction
  (`env.py` wraps it), so a failed migration **rolls back to the last good
  revision** — never a half-migrated schema.
- **Scope.** Alembic owns only the SQLAlchemy business tables (`Base.metadata`).
  The LangGraph checkpointer tables and the `rag` vector-store schema are managed
  by their own libraries and are filtered out of autogenerate (`env.py`
  `include_name` + `include_schemas=False`).
- **Adopting existing installs.** A database created by the old `create_all` path
  has the tables but no `alembic_version`. `run_migrations` detects this and
  **stamps** the baseline (instead of replaying its `CREATE TABLE`s, which would
  collide), then upgrades.
- **Packaging.** The `alembic/` tree + `alembic.ini` are bundled into the frozen
  backend (the PyInstaller specs); at runtime `ROOT_DIR` resolves to the bundle
  root, where the runner finds them.

## Adding a migration

1. Change the SQLAlchemy models under `src/entities/`.
2. Generate a revision against a database **already at head** (autogenerate diffs
   the models against the live schema):

   ```bash
   cd backend && source venv/bin/activate
   # point Alembic at a running cluster that is already at head, e.g. the dev one.
   # The cluster requires its per-cluster password on every host connection:
   # read it from <data_dir>/erudi_db_password (backend/data/postgres/ in dev).
   export ERUDI_ALEMBIC_URL='postgresql+psycopg://postgres:<password>@.../erudi'
   alembic upgrade head
   alembic revision --autogenerate -m "describe the change"
   ```

3. **Review the generated script** — autogenerate is a draft. Check it matches the
   intent, and that it touched no checkpointer/`rag` tables.
4. The migration ships automatically (auto-upgrade at next startup). The test suite
   runs the whole chain (`conftest` applies `upgrade head`) and a guard test
   asserts the models and the migration chain stay in sync (`alembic check`).

## Recovery if an update's migration fails

Because app version N expects schema N, rolling the **schema** back alone does not
let app N run — recovery is at the **app-version** level:

1. The failed migration already rolled back transactionally → the schema is intact
   at the previous revision.
2. The new app **fails fast** (it will not serve on a mismatched schema): the
   launcher emits a `startup_error`.
3. **Reinstall the previous app version** (kept as a signed release). It matches the
   rolled-back schema and runs again.
4. For a **destructive** migration (column drop / data rewrite) that lost data, a
   `pg_dump` snapshot is taken **before** each applied migration, in
   `…/db-backups/erudi-<from_rev>.dump`. The three most recent snapshots are
   kept (`KEEP_BACKUPS = 3` in `src/database/backup.py`); older ones are pruned.
   Restore one with the bundled `pg_restore` while on the previous app version.
   That binary ships inside the pgserver install — `pginstall/bin/pg_restore`
   next to `pg_dump`, under `pgserver.__file__`'s directory — which resolves the
   same way in dev and in the PyInstaller bundle:

   ```bash
   PGPASSWORD="$(cat '<data_dir>/erudi_db_password')" \
     pg_restore --clean --dbname '<psycopg_url>' '…/db-backups/erudi-<rev>.dump'
   ```

   `pg_restore` authenticates like every other client: the cluster password
   lives in `erudi_db_password` inside the data dir (`backup.py` passes it to
   `pg_dump` the same way, through `PGPASSWORD`, so it never sits on a command
   line).
