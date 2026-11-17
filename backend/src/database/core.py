"""SQLAlchemy database configuration and session management.

This module provides the core database infrastructure:
- Explicit engine initialization against the embedded PostgreSQL cluster
- Session factory for transaction management
- Declarative base for ORM models
- Dependency injection helper for FastAPI endpoints

Initialization contract (PostgreSQL migration):
    The engine URL is only known at runtime, once the embedded cluster
    (``src.launcher.postgres_runtime``) is up. Nothing is bound at import
    time::

        from src.database import core

        engine = core.init_database(handle.sqlalchemy_url)  # lifespan step 1

    - ``SessionLocal`` is a module-level factory, UNBOUND until
      ``init_database()`` configures it in place. Importing it by value is
      safe (the factory object is stable).
    - ``db_engine`` MUST NOT be imported by value: it is rebound by
      ``init_database()``, so an imported copy stays frozen at ``None``.
      Access it as ``core.db_engine``, or use the engine returned by
      ``init_database()``.

Example:
    Use in FastAPI endpoint::

        from fastapi import Depends
        from sqlalchemy.orm import Session
        from src.database.core import get_db
        from src.entities.Llm import Llm

        @router.get("/models")
        async def list_models(db: Session = Depends(get_db)):
            models = db.query(Llm).all()
            return {"models": models}

Note:
    - Always use get_db() for endpoint dependency injection
    - Manual sessions (``SessionLocal()``) must be explicitly closed in
      finally blocks; they require init_database() to have run first
    - Session is NOT thread-safe; create one per request
"""

from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from src.core.logging import logger
from src.launcher.postgres_settings import (
    BUSINESS_POOL_MAX_OVERFLOW,
    BUSINESS_POOL_SIZE,
    POOL_TIMEOUT_SECONDS,
)

# Session factory — configured in place by init_database(). Safe to import.
SessionLocal = sessionmaker(autocommit=False, autoflush=False)

# Live engine — rebound by init_database(). Access as core.db_engine only.
db_engine: Optional[Engine] = None

# Base class for ORM models
Base = declarative_base()


def _sanitize_url_for_log(sqlalchemy_url: str) -> str:
    """Credential-free rendering of a DB URL for log lines.

    The embedded cluster URL carries the per-cluster password (#462): it is
    masked here and never printed. Never raises — an unparseable URL degrades
    to a placeholder.
    """
    try:
        return make_url(sqlalchemy_url).render_as_string(hide_password=True)
    except Exception:
        return "<unparseable database url>"


def init_database(sqlalchemy_url: str) -> Engine:
    """Create the SQLAlchemy engine and bind the session factory to it.

    Called once the embedded PostgreSQL cluster is up (FastAPI lifespan,
    step 1) and by the test harness against throwaway clusters.

    Args:
        sqlalchemy_url: ``postgresql+psycopg://…`` URL from
            ``postgres_runtime.PostgresHandle.sqlalchemy_url``.

    Returns:
        The live engine (also published as ``core.db_engine``).
    """
    global db_engine
    # A rebind (the DB watchdog restarting the cluster) replaces the engine:
    # close the idle connections of the one it replaces, which would
    # otherwise stay open until garbage collection, counted against the
    # cluster's max_connections. A connection still checked out goes back to
    # the old pool and closes with it once nothing references it.
    if db_engine is not None:
        db_engine.dispose()
    # Sized by the connection budget the cluster's max_connections is
    # computed from (src/launcher/postgres_settings.py).
    db_engine = create_engine(
        sqlalchemy_url,
        pool_size=BUSINESS_POOL_SIZE,
        max_overflow=BUSINESS_POOL_MAX_OVERFLOW,
        pool_timeout=POOL_TIMEOUT_SECONDS,
    )
    SessionLocal.configure(bind=db_engine)
    logger.info(f"Database bound: {_sanitize_url_for_log(sqlalchemy_url)}")
    return db_engine


def get_db():
    """Provide database session for FastAPI dependency injection.

    Yields a SQLAlchemy session that is automatically closed after the
    request completes. Ensures proper cleanup even if exceptions occur.

    Yields:
        Session: Active database session bound to the embedded PostgreSQL
        cluster (after init_database()).
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
