"""Alembic migration guards (#96).

These spin their OWN throwaway pgserver cluster because they mutate the schema
(the session-scoped cluster is shared and migrated to head once). Two guarantees:

1. ``run_migrations`` on a fresh DB brings it to head AND the migration chain
   stays in sync with the SQLAlchemy models (``alembic check`` finds no diff).
2. ``run_migrations`` on a pre-Alembic DB (created by ``create_all``, no
   ``alembic_version``) ADOPTS it by stamping the baseline — it must not replay
   the baseline's CREATE TABLEs (which would collide) and must keep the data.
"""

import pytest
from alembic import command
from sqlalchemy import create_engine, inspect, text

from src.database.backup import _dump_target, backup_database, backups_dir_for
from src.database.core import Base
from src.database.migrations import (
    BASELINE_REVISION,
    ROOT_DIR,
    _alembic_config,
    _head_revision,
    run_migrations,
)
from src.launcher.postgres_runtime import start_postgres, stop_postgres


def _alembic_version(url: str) -> str | None:
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            if not inspect(conn).has_table("alembic_version"):
                return None
            return conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
    finally:
        engine.dispose()


@pytest.fixture
def fresh_cluster(tmp_path_factory):
    # Nest the data dir one level down so backups_dir_for (data_dir.parent/db-backups)
    # is unique per test — mktemp dirs otherwise share a parent and leak snapshots.
    base = tmp_path_factory.mktemp("pg-migrations")
    handle = start_postgres(base / "data")
    try:
        yield handle
    finally:
        stop_postgres(handle)


@pytest.mark.integration
def test_fresh_db_upgrades_to_head_and_matches_models(fresh_cluster):
    url = fresh_cluster.sqlalchemy_url

    run_migrations(fresh_cluster)

    engine = create_engine(url)
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
    assert "llms" in tables and "conversations" in tables
    # training_jobs was dropped by revision 7bc061d58b4e (dead fine-tuning code).
    assert "training_jobs" not in tables
    assert _alembic_version(url) == _head_revision(_alembic_config(url))

    # The migration chain must equal the models: autogenerate detects no diff.
    # command.check raises CommandError if the schema drifts from Base.metadata.
    command.check(_alembic_config(url))


@pytest.mark.integration
def test_pre_alembic_db_is_stamped_not_replayed(fresh_cluster):
    url = fresh_cluster.sqlalchemy_url

    # Simulate a database created by the old create_all path: the FULL historical
    # schema (including the since-dropped training_jobs) but no alembic_version
    # table. Build it from the baseline revision, then strip alembic_version so it
    # looks pre-Alembic — Base.metadata.create_all no longer carries training_jobs
    # and so cannot stand in for the schema the drop migration expects.
    cfg = _alembic_config(url)
    command.upgrade(cfg, BASELINE_REVISION)
    engine = create_engine(url)
    try:
        with engine.begin() as conn:
            conn.execute(text("DROP TABLE alembic_version"))
        assert inspect(engine).has_table("training_jobs")
    finally:
        engine.dispose()
    assert _alembic_version(url) is None

    # Must STAMP the baseline (no CREATE TABLE collision), then apply newer
    # revisions to head — here, dropping training_jobs.
    run_migrations(fresh_cluster)

    assert _alembic_version(url) == _head_revision(cfg)
    engine = create_engine(url)
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
    assert "llms" in tables
    assert "training_jobs" not in tables


GENERATION_HINTS_REVISION = "f3b7a9c2d5e1"


@pytest.mark.integration
def test_generation_hints_column_is_nullable_and_backfills_null(fresh_cluster):
    """#388: llms.generation_hints lands as a nullable JSON column; rows that
    predate the revision read NULL (= "no hints", the fallback sampling)."""
    url = fresh_cluster.sqlalchemy_url
    cfg = _alembic_config(url)
    command.upgrade(cfg, f"{GENERATION_HINTS_REVISION}-1")
    engine = create_engine(url)
    try:
        assert "generation_hints" not in {c["name"] for c in inspect(engine).get_columns("llms")}
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO llms (name, local, link, type, quantized, is_base, category, "
                    "is_attached_to_kb) VALUES ('Old', 0, 'org/old', 'qwen', true, false, "
                    "'general', false)"
                )
            )
    finally:
        engine.dispose()

    run_migrations(fresh_cluster)

    engine = create_engine(url)
    try:
        columns = {c["name"]: c for c in inspect(engine).get_columns("llms")}
        assert columns["generation_hints"]["nullable"] is True
        with engine.connect() as conn:
            stored = conn.execute(
                text("SELECT generation_hints FROM llms WHERE link='org/old'")
            ).scalar()
        assert stored is None
    finally:
        engine.dispose()
    assert _alembic_version(url) == _head_revision(cfg)


ARTIFACT_SIZE_REVISION = "b6e1a4d8c2f7"


@pytest.mark.integration
def test_artifact_size_bytes_column_is_nullable_bigint_and_backfills_null(fresh_cluster):
    """llms.artifact_size_bytes lands as a nullable BIGINT (model artifacts exceed
    2^31 bytes); rows that predate the revision read NULL (= size unknown, the
    frontend keeps its estimate)."""
    url = fresh_cluster.sqlalchemy_url
    cfg = _alembic_config(url)
    command.upgrade(cfg, f"{ARTIFACT_SIZE_REVISION}-1")
    engine = create_engine(url)
    try:
        assert "artifact_size_bytes" not in {c["name"] for c in inspect(engine).get_columns("llms")}
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO llms (name, local, link, type, quantized, is_base, category, "
                    "is_attached_to_kb) VALUES ('Old', 0, 'org/old', 'qwen', true, false, "
                    "'general', false)"
                )
            )
    finally:
        engine.dispose()

    run_migrations(fresh_cluster)

    engine = create_engine(url)
    try:
        columns = {c["name"]: c for c in inspect(engine).get_columns("llms")}
        assert columns["artifact_size_bytes"]["nullable"] is True
        assert "BIGINT" in str(columns["artifact_size_bytes"]["type"]).upper()
        with engine.begin() as conn:
            stored = conn.execute(
                text("SELECT artifact_size_bytes FROM llms WHERE link='org/old'")
            ).scalar()
            assert stored is None
            conn.execute(
                text("UPDATE llms SET artifact_size_bytes = 3090000000 WHERE link='org/old'")
            )
            assert (
                conn.execute(
                    text("SELECT artifact_size_bytes FROM llms WHERE link='org/old'")
                ).scalar()
                == 3_090_000_000
            )
    finally:
        engine.dispose()
    assert _alembic_version(url) == _head_revision(cfg)


@pytest.mark.integration
def test_language_column_backfills_existing_settings_row(fresh_cluster):
    # #385: a pre-existing user_settings row (created before the language
    # setting existed) must come out of the migration with language='en'
    # and the column NOT NULL, so the API never serves a null language.
    url = fresh_cluster.sqlalchemy_url
    cfg = _alembic_config(url)
    command.upgrade(cfg, "d1a4f7c39b52")
    engine = create_engine(url)
    try:
        with engine.begin() as conn:
            conn.execute(text("INSERT INTO user_settings (web_search_enabled) VALUES (true)"))
    finally:
        engine.dispose()

    run_migrations(fresh_cluster)

    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            row = conn.execute(text("SELECT web_search_enabled, language FROM user_settings")).one()
            columns = {c["name"]: c for c in inspect(conn).get_columns("user_settings")}
    finally:
        engine.dispose()
    assert row == (True, "en")
    assert columns["language"]["nullable"] is False


@pytest.mark.integration
def test_backup_database_writes_a_dump(fresh_cluster):
    # pg_dump (custom format) of the LIVE cluster produces a non-empty snapshot.
    engine = create_engine(fresh_cluster.sqlalchemy_url)
    try:
        Base.metadata.create_all(bind=engine)
    finally:
        engine.dispose()

    dump = backup_database(fresh_cluster.psycopg_url, fresh_cluster.data_dir, label="baseline")

    assert dump.exists() and dump.stat().st_size > 0
    assert dump.parent == backups_dir_for(fresh_cluster.data_dir)


@pytest.mark.unit
def test_dump_target_keeps_the_password_out_of_the_command_line():
    # pg_dump's argv is visible in the process list and echoed by
    # CalledProcessError on failure: the cluster password goes through the
    # environment (PGPASSWORD), never through --dbname.
    conninfo, env = _dump_target("postgresql://postgres:s3cr-_et@127.0.0.1:5433/erudi")
    assert "s3cr-_et" not in conninfo
    assert "dbname=erudi" in conninfo and "host=127.0.0.1" in conninfo and "port=5433" in conninfo
    assert env["PGPASSWORD"] == "s3cr-_et"


@pytest.mark.unit
def test_dump_target_socket_form_without_password():
    conninfo, env = _dump_target("postgresql://postgres:@/erudi?host=/tmp/erudi-pg-ab12")
    assert conninfo == "user=postgres dbname=erudi host=/tmp/erudi-pg-ab12"
    assert "PGPASSWORD" not in env


@pytest.mark.unit
def test_alembic_config_survives_a_percent_sign_in_the_url():
    # ConfigParser interpolates '%' in set_main_option values; a percent-encoded
    # password must round-trip unchanged (Alembic documents the %% escape).
    url = "postgresql+psycopg://postgres:a%2Fb@127.0.0.1:1/erudi"
    assert _alembic_config(url).get_main_option("sqlalchemy.url") == url


@pytest.mark.integration
def test_fresh_db_migration_takes_no_backup(fresh_cluster):
    # Nothing to lose on a fresh install -> run_migrations must not snapshot.
    run_migrations(fresh_cluster)

    backups = backups_dir_for(fresh_cluster.data_dir)
    assert not backups.exists() or not list(backups.glob("*.dump"))


@pytest.mark.unit
def test_alembic_ini_is_pure_ascii():
    # A packaged app can launch without a UTF-8 locale (macOS Finder sets no LANG),
    # so configparser reads alembic.ini with the ASCII codec. Any non-ASCII byte
    # then crashes the boot-time migration with UnicodeDecodeError (#149). Keep the
    # file pure ASCII (comments included) so it loads under any locale.
    raw = (ROOT_DIR / "alembic.ini").read_bytes()
    try:
        raw.decode("ascii")
    except UnicodeDecodeError as exc:
        pytest.fail(
            f"alembic.ini has a non-ASCII byte at offset {exc.start} "
            f"(0x{raw[exc.start]:02x}); it must stay pure ASCII so it loads under an "
            f"ASCII locale (macOS Finder launch). See #149."
        )
