"""FastAPI application factory and lifecycle management.

This module provides the core FastAPI application configuration including:
- Router registration for all domain endpoints
- Exception handler setup for application-level errors
- CORS middleware configuration for cross-origin requests
- Application lifespan management (startup/shutdown hooks)

The lifespan context manager orchestrates:
1. Engine initialization and selection (MLX/CUDA/CPU)
2. Database table creation and seeding
3. Background cleanup tasks for model memory management

Architecture:
    Application Bootstrap Flow:
    ┌─────────────────────────────────────────────────────────────┐
    │ lifespan() startup:                                         │
    │  1. Select engine (MLX_Engine/CUDA_Engine/CPU_Engine)       │
    │  2. Create SQLAlchemy tables                                │
    │  3. Seed database with default models                       │
    │  4. Start cleanup task (30s interval)                       │
    └─────────────────────────────────────────────────────────────┘
                              ↓
    ┌─────────────────────────────────────────────────────────────┐
    │ register_routers():                                         │
    │  - /erudi/llms          → Model management                  │
    │  - /erudi/conversations → Chat/streaming                    │
    │  - /erudi/knowledge_base → RAG/vectorization                │
    │  - /erudi/arena         → Model comparison                  │
    │  - /erudi/hardware      → System monitoring                 │
    │  - /erudi/health        → Health checks                     │
    │  - /erudi/diagnostics   → Bug-report environment summary    │
    │  - /erudi/startup       → Initialization state              │
    └─────────────────────────────────────────────────────────────┘

Example:
    Create and configure the FastAPI application::

        from fastapi import FastAPI
        from src.core.api import (
            register_routers,
            add_exception_handlers,
            add_middleware,
            lifespan
        )

        app = FastAPI(lifespan=lifespan)
        register_routers(app)
        add_exception_handlers(app)
        add_middleware(app)

        # Run with: uvicorn src.main:app --reload

Note:
    The lifespan pattern ensures proper resource cleanup even if the
    application crashes or is forcefully terminated.
"""

import asyncio
import time

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.datastructures import MutableHeaders
from contextlib import asynccontextmanager
from src.database.core import init_database
from src.database.migrations import run_migrations
from src.database.seed import backfill_wire_tools_startup, startup_populate_database
from src.launcher.postgres_runtime import start_postgres, stop_postgres
from src.launcher.db_watchdog import start_watchdog, stop_watchdog
from src.ingestion.vector_store import close_kb_store, init_kb_store

from src.core.exceptions import (
    AppBaseException,
    app_base_exception_handler,
    unhandled_exception_handler,
)
from src.core import config
from src.core.request_context import new_request_id, request_id_var
from src.engines.base_engine import BaseEngine
from src.core.logging import logger
from src.agents.checkpoint import open_checkpointer

from src.domains.llms.endpoints import router as llms_router
from src.domains.arena.endpoints import router as arena_router
from src.domains.conversations.endpoints import router as conversations_router
from src.domains.hardware.endpoints import router as hardware_router
from src.domains.knowledge_base.endpoints import router as knowledge_base_router
from src.domains.startup.endpoints import router as startup_router
from src.domains.user_settings.endpoints import router as user_settings_router
from src.domains.diagnostics.endpoints import router as diagnostics_router
from src.core.health import router as health_router


def _is_polling_path(path: str) -> bool:
    """True for endpoints the frontend polls — their access log goes to DEBUG."""
    return "/health" in path or path.endswith("/status")


class RequestLoggingMiddleware:
    """Pure ASGI middleware: request-id propagation + one access-log line.

    Deliberately NOT BaseHTTPMiddleware: that wrapper re-buffers streaming
    responses through an internal queue (breaking SSE chunk pacing) and adds
    an extra task per request. Here the per-chunk cost is a single message
    type comparison — response bodies are never inspected or buffered.

    Per request:
    - Reads ``X-Request-ID`` (case-insensitive) or generates ``be-<8 hex>``.
    - Stores the id in ``request_id_var`` so every log line emitted while
      handling the request is tagged with it (see src.core.logging).
    - Echoes the id back as the ``X-Request-ID`` response header.
    - When the response body completes (final ``http.response.body`` chunk),
      logs exactly one line: ``HTTP <METHOD> <path> -> <status> in <ms>ms`` at
      INFO — DEBUG for polling endpoints (paths containing ``/health`` or
      ending in ``/status``) to keep the log readable. Logging at body-complete
      rather than after the ASGI app returns keeps starlette's post-response
      BackgroundTasks out of the reported duration (#204); a slow background
      tail (>1s) is noted separately at DEBUG.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = self._incoming_request_id(scope) or new_request_id()
        # Each request runs in its own task (own context copy), so no reset
        # is needed and the id stays visible to the outer exception handlers.
        request_id_var.set(request_id)

        method = scope.get("method", "-")
        path = scope.get("path", "-")
        start = time.perf_counter()
        status_code = 500  # overwritten on http.response.start
        # CORS preflights carry no QA signal — keep them out of the INFO stream.
        quiet = _is_polling_path(path) or method == "OPTIONS"
        logged = False
        body_complete_ms = 0.0

        def _log_access() -> None:
            # The access line must land when the response body is done, NOT when
            # the ASGI app returns: starlette runs BackgroundTasks after the body
            # is sent but before returning, so logging after the app call folded
            # minutes of background work (e.g. a model download) into the request
            # duration and delayed the line just as long (#204).
            nonlocal logged, body_complete_ms
            body_complete_ms = (time.perf_counter() - start) * 1000
            log = logger.debug if quiet else logger.info
            log(f"HTTP {method} {path} -> {status_code} in {body_complete_ms:.1f}ms")
            logged = True

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                message["headers"] = list(message.get("headers") or [])
                headers = MutableHeaders(scope=message)
                headers.append("X-Request-ID", request_id)
            await send(message)
            # Log once the final body chunk is out. Streaming/SSE responses end
            # with a chunk whose more_body is falsy, so the line lands at
            # end-of-stream -- the real response duration, unchanged behavior.
            if (
                not logged
                and message["type"] == "http.response.body"
                and not message.get("more_body")
            ):
                _log_access()

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            # Access line for the crashed request; the traceback itself is
            # logged by unhandled_exception_handler (ServerErrorMiddleware).
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error(
                f"HTTP {method} {path} -> 500 in {duration_ms:.1f}ms (unhandled exception)"
            )
            raise
        if not logged:
            # Defensive: an app that returned without ever sending a body still
            # gets exactly one access line.
            _log_access()
            return
        # Body was already logged. If the app then spent real time on
        # BackgroundTasks (>1s past body-complete), note the long tail at DEBUG
        # so the total is still discoverable without skewing the access line.
        total_ms = (time.perf_counter() - start) * 1000
        if total_ms - body_complete_ms > 1000:
            logger.debug(f"HTTP {method} {path} background work finished in {total_ms:.1f}ms")

    @staticmethod
    def _incoming_request_id(scope) -> str | None:
        """Extract a non-empty X-Request-ID header value, case-insensitively."""
        for name, value in scope.get("headers") or ():
            if name.lower() == b"x-request-id":
                request_id = value.decode("latin-1").strip()
                if request_id:
                    return request_id
        return None


def register_routers(app: FastAPI) -> None:
    """Register all domain routers to the FastAPI application.

    Attaches endpoint routers from each domain module with the /erudi prefix.
    Order of registration does not affect routing behavior due to FastAPI's
    path-matching algorithm.

    Args:
        app: The FastAPI application instance to register routers to.

    Returns:
        None. Modifies the app instance in-place.

    Example:
        ::

            from fastapi import FastAPI
            from src.core.api import register_routers

            app = FastAPI()
            register_routers(app)
            # All domain endpoints now accessible under /erudi/*
    """

    app.include_router(llms_router, prefix="/erudi")
    app.include_router(hardware_router, prefix="/erudi")
    app.include_router(arena_router, prefix="/erudi")
    app.include_router(knowledge_base_router, prefix="/erudi")
    app.include_router(conversations_router, prefix="/erudi")
    app.include_router(health_router, prefix="/erudi")
    app.include_router(startup_router, prefix="/erudi")
    app.include_router(user_settings_router, prefix="/erudi")
    app.include_router(diagnostics_router, prefix="/erudi")


def add_exception_handlers(app: FastAPI) -> None:
    """Attach application-level exception handlers to FastAPI.

    Registers custom exception handlers for application-level errors:
    - AppBaseException and its subclasses (ModelNotFoundException,
      InvalidInputException, EngineException, ...)
    - plain Exception as a last-resort fallback, returning the same
      structured 500 JSON shape and logging the traceback with the
      request id.

    Args:
        app: The FastAPI application instance to attach handlers to.

    Returns:
        None. Modifies the app instance in-place.

    Example:
        ::

            from fastapi import FastAPI
            from src.core.api import add_exception_handlers

            app = FastAPI()
            add_exception_handlers(app)
            # Exceptions now return structured JSON responses with proper HTTP codes

    Note:
        See src.core.exceptions.app_base_exception_handler for response format.
    """
    app.add_exception_handler(AppBaseException, app_base_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)


# The only two origins the renderer ever runs under (#89): the webpack dev
# server, and the packaged app's ``file://`` page — which browsers serialize
# to the literal origin string "null" on cross-origin requests. Anything else
# (i.e. any website the user visits while Erudi runs) gets no CORS grant, so
# the browser blocks it from reading the unauthenticated localhost API.
# Residual: sandboxed iframes also send "null" (mitigated by Chromium's
# private-network-access; per-session token tracked in #89).
RENDERER_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "null",
]

# DNS-rebinding guard (#89): an attacker domain resolving to 127.0.0.1 makes
# the browser send a same-origin request that bypasses CORS entirely — but its
# Host header stays the attacker's domain, so pinning Host to local names
# closes the vector.
LOCAL_HOSTS = ["127.0.0.1", "localhost"]


def add_middleware(app: FastAPI) -> None:
    """Configure the middleware stack: host pinning + CORS + request logging.

    The API binds 127.0.0.1 without auth (single-user desktop), so the
    browser's same-origin machinery is the wall between a malicious website
    and the user's data (#89): CORS grants are restricted to the two real
    renderer origins with credentials off, and TrustedHostMiddleware rejects
    non-local Host headers (DNS rebinding). Request logging is added last so
    it sits outermost and also covers CORS-short-circuited requests.

    Args:
        app: The FastAPI application instance to configure.

    Returns:
        None. Modifies the app instance in-place.
    """
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=LOCAL_HOSTS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=RENDERER_ORIGINS,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
    # Added last -> outermost: every request gets an id and one access line.
    app.add_middleware(RequestLoggingMiddleware)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan with startup and shutdown hooks.

    This async context manager orchestrates the full application lifecycle:
    - Startup: Engine selection, database initialization, cleanup task scheduling
    - Shutdown: Task cancellation, engine cleanup, resource deallocation

    Args:
        app: The FastAPI application instance (passed by FastAPI framework).

    Yields:
        None. Control is yielded to the FastAPI application for request handling.

    Raises:
        Exception: Propagates any exception during startup (database errors,
            engine initialization failures). Application will not start if
            startup fails.

    Example:
        ::

            from fastapi import FastAPI
            from src.core.api import lifespan

            app = FastAPI(lifespan=lifespan)
            # Engine initialized automatically on startup
            # Cleanup runs automatically on shutdown

    Note:
        The cleanup task runs every 300 seconds to free inactive model memory.
        See BaseEngine.start_cleanup_task() for details.

    Lifecycle Flow:
        1. Start the embedded PostgreSQL cluster (postgres_runtime)
        2. Bind the SQLAlchemy engine/session factory (init_database)
        3. Select engine via platform detection (BaseEngine.get_engine)
        4. Migrate the schema to head (Alembic, forward-only)
        5. Seed database with default models (startup_populate_database)
        6. Open the LangGraph checkpointer (app.state.checkpointer)
        7. Start cleanup background task (300s interval)
        8. **[YIELD]** → Application handles requests
        9. Shutdown (reverse order): cleanup task → engine → checkpointer
           → embedded PostgreSQL cluster last
    """
    # Before yield comes the startup code
    logger.info("==== Starting up... ====")
    # Startup-progress phases for the Electron loader. run.py injects the emitter
    # on app.state (same process/stdout); absent (plain uvicorn in dev/tests) this
    # is a no-op. Phases are informational only — readiness is still the `ready`
    # event / a confirming health check, never a phase.
    _emit_phase = getattr(app.state, "emit_phase", None)

    def _phase(name: str) -> None:
        if _emit_phase is not None:
            _emit_phase(name)

    # Step 0: embedded PostgreSQL cluster — must precede any DB usage. On first
    # run this pays a one-time initdb (the long pole), so surface it explicitly.
    _phase("preparing_database")
    app.state.postgres = start_postgres(config.POSTGRES_DATA_DIR)
    # Step 1: bind the SQLAlchemy engine/session factory to the live cluster.
    init_database(app.state.postgres.sqlalchemy_url)
    config.LLM_Engine = BaseEngine.get_engine()
    # Step 4: migrate the schema to head (forward-only). Alembic is sync, so run
    # it off the event loop. Replaces create_all — which never altered an existing
    # (persisted) database — and auto-adopts pre-Alembic schemas (stamp baseline).
    _phase("running_migrations")
    await run_in_threadpool(run_migrations, app.state.postgres)
    # await delete_all_data()
    # Startup data (vars, cleanup, hardware) + the catalog reconciled from the
    # bundled snapshot — zero network, all inside startup_populate_database
    # (#131, #163). The catalog follows app releases; no live HF resync exists.
    _phase("loading_catalog")
    await startup_populate_database()
    # Hybrid KB vector store (rag.kb_chunks) — AFTER the schema migration: its
    # cross-schema FKs reference the business tables.
    app.state.kb_store = init_kb_store(app.state.postgres)
    # LangGraph conversation-state checkpointer (AsyncPostgresSaver on the
    # same `erudi` database as the business schema), held open for the whole
    # app lifetime and exposed on app.state.checkpointer.
    checkpointer_cm = open_checkpointer(app.state.postgres.psycopg_url)
    app.state.checkpointer = await checkpointer_cm.__aenter__()
    # The DB watchdog may re-open the checkpointer on a resurrected cluster and
    # swap this CM (#162), so publish it on app.state and let shutdown close the
    # LIVE one rather than this now-possibly-stale local.
    app.state.checkpointer_cm = checkpointer_cm
    config.LLM_Engine.start_cleanup_task()
    # DB watchdog: detect a dead embedded Postgres, resurrect it, expose db
    # state on /health (#162). Started AFTER init_database bound the live engine
    # (the disconnect hook attaches to it) and the checkpointer is on app.state.
    start_watchdog(app)
    # Post-ready backfill (#298): verify the tool-call wire capability of
    # models downloaded before the supports_tools_wire column existed. Each
    # verification loads a tokenizer (seconds per model), so it runs AFTER the
    # ready handshake in a threadpool — never inside the awaited boot sequence
    # (same non-blocking rationale as the post-ready resync of #109). Until a
    # row is verified it stays NULL and its KB turns route systematic.
    app.state.wire_backfill_task = asyncio.create_task(
        run_in_threadpool(backfill_wire_tools_startup)
    )
    yield
    logger.info("==== Shutting down... ====")
    wire_backfill_task = getattr(app.state, "wire_backfill_task", None)
    if wire_backfill_task is not None and not wire_backfill_task.done():
        wire_backfill_task.cancel()
    await stop_watchdog()
    config.LLM_Engine.stop_cleanup_task()
    config.LLM_Engine.cleanup()
    await app.state.checkpointer_cm.__aexit__(None, None, None)
    close_kb_store()
    # Cluster stops LAST: every DB consumer above must be closed first.
    stop_postgres(app.state.postgres)
