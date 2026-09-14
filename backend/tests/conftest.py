"""Pytest configuration and shared fixtures for all test modules.

Provides database session fixtures, test client, and mock data factories
following the repository pattern used in the codebase.

Architecture:
    Tests run against a REAL embedded PostgreSQL cluster (pgserver) — the
    exact mechanism production uses (vector extension included). The cluster
    and the schema are session-scoped (boot + create_all once); per-test
    isolation comes from the outer-transaction rollback in test_db_session,
    so nothing a test commits ever reaches the shared tables.

    Note: PostgreSQL sequences are non-transactional — ids keep growing
    across tests. Never assert on absolute primary-key values.
"""

# Force the multiprocessing start method to "spawn" BEFORE any heavy import
# below loads modules that would interact with multiprocessing internals.
# This mirrors `backend/run.py:force_mp_spawn()` and is required because:
#   1. The MLX engine refactor will use `multiprocessing.Process` to spawn
#      `mlx_lm.server` as a child process. Spawn is the only start method
#      that works inside PyInstaller frozen builds (`mp.freeze_support()`).
#   2. On Linux (CI), the default is `fork`, which after importing torch /
#      sentence_transformers has undefined behaviour.
#   3. Setting `force=True` here makes the test environment match production.
import multiprocessing as _mp

try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    # Already configured (e.g. pytest re-execution) — safe to ignore.
    pass

# Match production's asyncio event loop policy BEFORE any loop can exist.
# `run.py:set_event_loop_policy` is the single source of truth (it installs
# WindowsSelectorEventLoopPolicy on Windows, no-op elsewhere); calling it here
# rather than duplicating the choice keeps the two from diverging. Without it,
# pytest-asyncio builds its loops under Windows' default Proactor policy and
# psycopg refuses to run in async mode on them ("Psycopg cannot use the
# 'ProactorEventLoop'"), so on Windows the suite exercised a configuration the
# shipped app never runs in (#335). This must stay at import time: a
# session-scoped autouse fixture runs after pytest-asyncio has already built a
# loop for the first test. `run` is importable because pytest.ini sets
# `pythonpath = .` (rootdir = backend/), the same import tests/test_launcher.py
# already relies on.
import run as _run

_run.set_event_loop_policy()

import os
import sys
import pytest
from typing import Generator
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from fastapi.testclient import TestClient

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.database.core import Base

# Side-effect import: pulls the whole application tree so every entity is
# registered on Base.metadata BEFORE the session-scoped create_all runs.
from src.main import app  # noqa: F401
from src.database.core import get_db

from tests._helpers import is_mlx_platform


# ============ Database Fixtures ============


@pytest.fixture(scope="session")
def pg_test_cluster(tmp_path_factory):
    """One embedded PostgreSQL cluster for the whole test session.

    Same mechanism as production (src.launcher.postgres_runtime): pgserver
    boot on a throwaway data dir, `erudi` database, pgvector extension.
    """
    from src.launcher.postgres_runtime import start_postgres, stop_postgres

    handle = start_postgres(tmp_path_factory.mktemp("pg-test-cluster"))
    yield handle
    stop_postgres(handle)


@pytest.fixture(scope="session")
def _session_db_engine(pg_test_cluster):
    """Session-scoped engine with the full schema applied via Alembic.

    Uses the REAL startup path (``run_migrations`` → ``alembic upgrade head``)
    rather than ``create_all``, so the whole suite exercises the migration chain
    that ships to users — a broken/incomplete migration fails fast here.
    """
    from src.database.migrations import run_migrations

    run_migrations(pg_test_cluster)
    engine = create_engine(pg_test_cluster.sqlalchemy_url)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="function")
def test_db_engine(_session_db_engine):
    """Engine handle for tests (kept function-scoped for fixture compat).

    Tables are session-scoped; per-test isolation is provided by
    test_db_session's outer-transaction rollback, NOT by recreating tables.
    """
    yield _session_db_engine


@pytest.fixture(scope="function")
def test_db_session(test_db_engine) -> Generator[Session, None, None]:
    """Create test database session with savepoint-based nested transactions.

    Uses SAVEPOINT to allow code under test to call commit() without affecting
    the outer test transaction. All changes are rolled back at test completion.

    Args:
        test_db_engine: SQLAlchemy engine fixture.

    Yields:
        Database session that supports nested transactions via savepoints.
    """
    connection = test_db_engine.connect()
    transaction = connection.begin()  # Outer transaction
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=connection)
    session = TestingSessionLocal(bind=connection)

    # Begin nested transaction (savepoint)
    nested = connection.begin_nested()

    @event.listens_for(session, "after_transaction_end")
    def restart_savepoint(session, transaction):
        """Automatically restart savepoint after each commit."""
        if transaction.nested and not transaction._parent.nested:
            # Re-establish savepoint after inner transaction ends
            session.begin_nested()

    try:
        yield session
    finally:
        session.close()
        transaction.rollback()  # Roll back everything
        connection.close()


@pytest.fixture(scope="function")
def client(test_db_session):
    """Create FastAPI test client with dependency injection.

    Overrides get_db dependency to use test database session.
    Creates app without lifespan to avoid production DB interactions.

    Args:
        test_db_session: Test database session fixture.

    Yields:
        FastAPI TestClient instance.
    """
    from fastapi import FastAPI
    from src.core.api import register_routers, add_exception_handlers, add_middleware
    from src.engines.base_engine import BaseEngine
    from src.core import config

    # Create app WITHOUT lifespan
    test_app = FastAPI(title="Erudi Test", version="0.1.0")
    add_middleware(app=test_app)
    add_exception_handlers(app=test_app)
    register_routers(app=test_app)

    # Override get_db to use test session
    def override_get_db():
        try:
            yield test_db_session
        finally:
            pass

    test_app.dependency_overrides[get_db] = override_get_db

    # Initialize engine for tests
    if not config.LLM_Engine:
        config.LLM_Engine = BaseEngine.get_engine()

    # Provide an in-memory checkpointer (production lifespan is bypassed here).
    from langgraph.checkpoint.memory import InMemorySaver

    test_app.state.checkpointer = InMemorySaver()

    # base_url pins the Host header to a local name: TrustedHostMiddleware
    # (#89) rejects TestClient's default "testserver" host.
    with TestClient(
        test_app, raise_server_exceptions=False, base_url="http://127.0.0.1"
    ) as test_client:
        yield test_client

    test_app.dependency_overrides.clear()


# ============ Mock Data Fixtures ============


@pytest.fixture
def mock_llm(test_db_session):
    """Create mock base LLM for testing KB attachment.

    Args:
        test_db_session: Test database session.

    Returns:
        Llm entity with local=1, ready for KB attachment.
    """
    from src.entities.Llm import Llm

    llm = Llm(
        name="Test Base Model",
        description="Base model for testing",
        local=1,
        link="test/model/path",
        type="test",
        is_attached_to_kb=False,  # Boolean
        param_size=7.0,
        quantized=True,  # Boolean
    )
    test_db_session.add(llm)
    test_db_session.commit()
    test_db_session.refresh(llm)
    return llm


@pytest.fixture
def mock_llm_with_kb(test_db_session):
    """Create mock specialized LLM with an attached KnowledgeBase.

    Args:
        test_db_session: Test database session.

    Returns:
        Tuple of (Llm, KnowledgeBase).
    """
    from src.entities.Llm import Llm
    from src.entities.KnowledgeBase import KnowledgeBase
    from src.entities.KnowledgeDocument import KnowledgeDocument

    kb = KnowledgeBase()
    test_db_session.add(kb)
    test_db_session.flush()

    test_db_session.add(
        KnowledgeDocument(
            kb_id=kb.id,
            name="doc1.pdf",
            content_hash_sha256="0" * 64,
            size_bytes=1024,
        )
    )

    # Create LLM with KB attachment
    llm = Llm(
        name="Test Model with KB",
        description="Model with existing KB",
        local=1,
        link="test/model/path",
        type="test",
        is_attached_to_kb=True,  # Boolean
        kb_id=kb.id,
        param_size=7.0,
        quantized=True,  # Boolean
    )
    test_db_session.add(llm)
    test_db_session.commit()

    test_db_session.refresh(llm)
    test_db_session.refresh(kb)

    return llm, kb


# ============ MLX-specific fixtures (Phase 0 — refactor/mlx-server-subprocess) ============
#
# Shared infrastructure for the MLX server-mode refactor test suite. These
# fixtures are session-scoped to amortize the cost of downloading the test
# models across the whole suite, and skip cleanly on non-Apple-Silicon hosts
# so that Linux CI stays green.
#
# Two models are provided:
#
#   `mlx_test_model_path` — default, always available
#       repo: mlx-community/Qwen2.5-0.5B-Instruct-4bit
#       - Apache 2.0 (no HF license accept required, works in unattended CI)
#       - 4-bit quantized, ~280 MB on disk
#       - No <think> tokens (keeps text-only assertions simple)
#       - Standard ChatML template (validates apply_chat_template path)
#
#   `mlx_thinking_model_path` — opt-in via ERUDI_TEST_THINKING=1
#       repo: mlx-community/Qwen3-0.6B-4bit (override: ERUDI_MLX_THINKING_MODEL_REPO)
#       - Required to validate the #554 reasoning chain on a real server:
#         mlx_vlm.server's native split routes chain-of-thought to the
#         dedicated `delta.reasoning` field (which `Erudi_Chat_OpenAI`
#         carries to the runner's `thinking` events) and no
#         `<think>...</think>` marker leaks into `delta.content`.
#       - Without this fixture, a regression that dropped the reasoning on
#         the floor or leaked markers into the visible answer would be
#         invisible.
#
# Gemma-specific EOS regression coverage is opt-in via ERUDI_TEST_GEMMA=1
# (added in Phase 1 once we have the corresponding test).

MLX_TEST_MODEL_REPO = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
MLX_TEST_THINKING_MODEL_REPO_DEFAULT = "mlx-community/Qwen3-0.6B-4bit"


def _download_mlx_model(repo_id: str, local_dir_env_var: str | None = None) -> Path:
    """Download (or reuse cache for) an MLX-quantized model from the HF Hub.

    Returns the absolute local path as a `Path`. Skips the calling test on:
      - non-MLX platforms (Linux CI, Mac Intel, etc.)
      - missing huggingface_hub
      - network failure with no local cache (offline dev)

    Args:
        repo_id: HF repo id, e.g. "mlx-community/Qwen2.5-0.5B-Instruct-4bit".
        local_dir_env_var: optional env var name that, if set and non-empty,
            overrides the HF cache directory for offline / pre-seeded setups.
    """
    if not is_mlx_platform():
        pytest.skip("MLX_Engine not selected on this platform — MLX fixture skipped")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        pytest.skip(f"huggingface_hub not available: {exc}")

    # Treat "" as None so an unset env var doesn't get passed as an empty path.
    override = os.environ.get(local_dir_env_var) if local_dir_env_var else None
    local_dir = override or None

    try:
        local_path = snapshot_download(repo_id=repo_id, local_dir=local_dir)
    except Exception as exc:
        # Covers LocalEntryNotFoundError, HfHubHTTPError, ConnectionError, etc.
        # On offline runs with cold cache we skip rather than ERROR so the
        # rest of the suite is not blocked.
        pytest.skip(f"Cannot fetch MLX test model {repo_id!r} (offline?): {exc}")

    return Path(local_path)


@pytest.fixture(scope="session")
def mlx_test_model_path() -> Path:
    """Local path to the default MLX test model (no thinking tokens).

    See module-level comment for repo details and rationale.
    """
    return _download_mlx_model(
        MLX_TEST_MODEL_REPO,
        local_dir_env_var="ERUDI_MLX_TEST_MODEL_DIR",
    )


@pytest.fixture(scope="session")
def mlx_thinking_model_path() -> Path:
    """Local path to an MLX test model that emits `<think>` tokens.

    Opt-in via ERUDI_TEST_THINKING=1 to keep the default suite fast. The repo
    can be overridden via ERUDI_MLX_THINKING_MODEL_REPO for forward
    compatibility (e.g. when newer thinking-capable MLX repos appear).
    """
    if os.environ.get("ERUDI_TEST_THINKING") != "1":
        pytest.skip(
            "Thinking-model integration tests are opt-in. "
            "Run with ERUDI_TEST_THINKING=1 to enable."
        )

    repo_id = os.environ.get(
        "ERUDI_MLX_THINKING_MODEL_REPO",
        MLX_TEST_THINKING_MODEL_REPO_DEFAULT,
    )
    return _download_mlx_model(
        repo_id,
        local_dir_env_var="ERUDI_MLX_THINKING_MODEL_DIR",
    )
