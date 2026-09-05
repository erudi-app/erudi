# Architecture

Technical architecture of Erudi: layering, multi-engine inference, startup sequence,
storage, and the main request flows.

## Overview

The backend is a FastAPI application organised by business domain. Each domain owns its
HTTP surface, its business logic, and its data access; infrastructure concerns (engines,
database, ingestion, launcher) live outside the domains.

```text
┌─────────────────────────────────────────────────────────────────┐
│ Frontend (Electron + React + Tailwind)                          │
│  - Contexts: KnowledgeBase, DownloadModal                       │
│  - services/api/client.js (retry, timeout, X-Request-ID)         │
└────────────────┬────────────────────────────────────────────────┘
                 │ HTTP on 127.0.0.1:27182, all routes under /erudi
┌────────────────▼────────────────────────────────────────────────┐
│ Backend FastAPI                                                 │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ endpoints.py — routes, Pydantic schemas, dependencies       │ │
│ ├─────────────────────────────────────────────────────────────┤ │
│ │ services.py — business logic, orchestration                 │ │
│ ├─────────────────────────────────────────────────────────────┤ │
│ │ repository.py — data access                                 │ │
│ ├─────────────────────────────────────────────────────────────┤ │
│ │ entities/*.py — SQLAlchemy models                           │ │
│ └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ Infrastructure                                                  │
│  - engines/    MLX / CUDA / CPU behind BaseEngine               │
│  - agents/     LangChain agent runner + LangGraph checkpointer  │
│  - ingestion/  document reader, chunking, embeddings, vectors   │
│  - database/   embedded PostgreSQL + pgvector, seeding          │
│  - launcher/   runtime paths, Postgres runtime, DB watchdog     │
└─────────────────────────────────────────────────────────────────┘
```

## Domains

Routers are mounted under the `/erudi` prefix in `register_routers`
(`backend/src/core/api.py`): `llms`, `hardware`, `arena`, `knowledge_base`,
`conversations`, `health`, `startup`, `user_settings`.

### 1. Conversations

- **Responsibility**: chat sessions, streaming generation, message persistence
- **Endpoints**: `POST /erudi/conversations/`, `GET /erudi/conversations/`,
  `POST /erudi/conversations/{id}/query`, `PATCH /erudi/conversations/{id}`,
  `GET /erudi/conversations/{id}/fetch_messages`,
  `POST /erudi/conversations/{id}/generate_title`,
  `POST /erudi/conversations/star_message` and `/unstar_message`
- **Entities**: `Conversation`, `Message`
- **Conversation state**: held by the LangGraph checkpointer, not rebuilt from the
  message table; older turns are summarized by a middleware as the thread grows
- [Reference](reference/conversations.md)

### 2. LLMs

- **Responsibility**: catalog, downloads, deletion, KB assistant binding
- **Endpoints**: `GET /erudi/llms/`, `GET /erudi/llms/search?name=`,
  `GET /erudi/llms/search/huggingface`, `POST /erudi/llms/{llm_id}/download`,
  `POST /erudi/llms/download/huggingface`, the `downloads/*` job routes,
  `DELETE /erudi/llms/{id}`, `GET /erudi/llms/{llm_id}/dependents`,
  `POST /erudi/llms/{assistant_id}/rebind`
- **Entities**: `Llm`, `DownloadJob`
- **Features**: Hugging Face integration, download of pre-built quants only, KB attachment
- [Reference](reference/llms.md)

### 3. Knowledge Base

- **Responsibility**: document ingestion and hybrid retrieval
- **Endpoints**: `POST /erudi/knowledge_base/create`,
  `GET /erudi/knowledge_base/{llm_id}/status`,
  `GET|POST /erudi/knowledge_base/embedding-model/status|download`
- **Entities**: `KnowledgeBase`, `KnowledgeDocument`, `KBJob`; chunks live in the
  `rag.kb_chunks` table
- **Formats**: `.pdf`, `.docx`, `.xlsx`, `.csv`, `.txt`, `.md`. Images are accepted and
  recorded as `pending_vision` — no OCR tier is bundled
- **Deletion**: there is no KB delete route; a KB assistant is removed through
  `DELETE /erudi/llms/{llm_id}`
- [Reference](reference/knowledge_base.md)

### 4. Arena

- **Responsibility**: comparing models on the same prompt
- **Endpoint**: `POST /erudi/arena/{llm_id}/query` (one call per model, plain-text stream)
- [Reference](reference/arena.md)

### 5. Hardware

- **Responsibility**: hardware profiling and performance scoring
- **Endpoints**: `GET /erudi/hardware/app_startup`, `GET /erudi/hardware/detailed`,
  `POST /erudi/hardware/refresh`
- **Entities**: `HardwareProfile`
- [Reference](reference/hardware.md) · [Hardware guide](guides/hardware.md)

### 6. Startup and user settings

- `GET /erudi/startup/welcome-popup` drives first-run UI state (`StartupVariables`).
- `GET|PUT /erudi/user_settings/` reads and writes the UI language, the global
  web-search default and the automatic-update preference (`UserSettings`).
  See [Internationalization](i18n.md).
- Connectivity for the status pill is read in the renderer from `navigator.onLine`
  and the `online` / `offline` events, corrected by requests that fail on the wire
  (`frontend/src/utils/networkStatus.js`). No request is made to answer it.

## Multi-engine architecture

### The uniform "subprocess + OpenAI-compatible HTTP" pattern

The three inference engines follow the **same pattern**: they spawn an
OpenAI-compatible HTTP server in a child process and talk to it over SSE at
`http://127.0.0.1:<port>/v1/chat/completions`. That uniformity is what lets the agent
layer address all three through a single `ChatOpenAI(base_url=...)` client, with no
per-backend code.

```text
            ┌───────────────────────────────────┐
            │ FastAPI backend (parent process)  │
            │   src/engines/<engine>.py         │
            │      └─ requests.post(stream=True)│
            └────────────────┬──────────────────┘
                             │  HTTP SSE
                             ▼
            ┌───────────────────────────────────┐
            │ OpenAI-compatible HTTP server     │
            │   /v1/chat/completions  /health   │
            │   (child process)                 │
            └───────────────────────────────────┘
```

How the child is launched differs:

- **CPU/CUDA**: `subprocess.Popen([llama-server, ...])` — a native binary at
  `backend/artifacts/llama-cpp/<cpu|cuda>/bin/llama-server`.
- **MLX**: `multiprocessing.Process(target=run_mlx_vlm_server, args=(argv,))`. There is
  no native binary on the MLX side, and a PyInstaller frozen build has no Python
  interpreter at `sys.executable` to pass `-m` to, so the process is spawned with
  `mp.spawn` (configured in `backend/run.py` via `mp.freeze_support()` and
  `set_start_method("spawn", force=True)`), which re-executes the launcher in child mode.

Shared lifecycle (port pick, two-stage `/health` + chat-ping probe, SSE byte-buffer
parser, atexit storage, idle-cleanup active marker, kwarg translation) lives in
`BaseChatServerEngine`; `BaseLlamaCppEngine` factors what is specific to `llama-server`.
See [Engines Architecture](dev/architecture/engines.md).

### Available engines

| Engine | Platform | Inference backend | Child launch | Port pool |
|---|---|---|---|---|
| **MLX_Engine** | macOS 14+ Apple Silicon | `mlx_vlm.server` | `mp.Process` | 27300-27399 |
| **CUDA_Engine** | Windows / Linux + NVIDIA | `llama-server` (CUDA build) | `subprocess.Popen` | 27200-27299 |
| **CPU_Engine** | Windows / Linux | `llama-server` (CPU build) | `subprocess.Popen` | 27200-27299 |

The FastAPI backend itself binds 27182 and scans 27182-27199, stopping short of 27200 so
the launcher and the inference pools can never fight over a port. Erudi's whole port
footprint is 271xx-273xx. See [Backend Launcher](guides/backend-run.md).

### Automatic selection

At startup the engine class is selected by `BaseEngine.get_engine()`
(`backend/src/engines/base_engine.py`):

1. **macOS ARM** (`platform.system() == "Darwin"` and `"arm" in platform.machine()`) → `MLX_Engine`
2. **Linux/Windows with an NVIDIA GPU** (`pynvml.nvmlDeviceGetCount() > 0`) → `CUDA_Engine`
3. **Otherwise** → `CPU_Engine`

`ERUDI_FORCE_CPU=1` short-circuits detection entirely and returns `CPU_Engine`.

Engines are never instantiated: they expose classmethods only, and the selected class is
stored on `src.core.config.LLM_Engine`.

## Agent layer (`src/agents`)

Conversations and Arena share one streaming primitive, `AgentRunner`
(`backend/src/agents/runner.py`), built on LangChain's `create_agent`:

- **Conversations** run with a `thread_id`, summarization enabled, and the LangGraph
  checkpointer, so history is restored from the checkpointer (only the new message is
  sent) and older turns are summarized in the agent state.
- **Arena** runs stateless: no `thread_id`, no summarization, no checkpointer.

A turn is captured as structured events — `answer`, `thinking`, `tool_call`,
`tool_result` — which the conversation service frames as NDJSON. Arena projects the same
stream down to answer text only, preserving its plain-text contract.

Other modules in the layer:

| Module | Role |
|---|---|
| `checkpoint.py` | Opens the `AsyncPostgresSaver` checkpointer held on `app.state.checkpointer` |
| `kb_mode.py` | Chooses agentic (tool-driven) vs systematic KB retrieval; `ERUDI_KB_AGENTIC` overrides |
| `tools.py` | Tool definitions, including `search_knowledge_base` |
| `prompts.py` | System-prompt construction |
| `middleware.py` | Summarization and related agent middleware |
| `model_factory.py` | Builds the `ChatOpenAI` client pointed at the engine's child server |
| `think_splitter.py` | Splits inline reasoning tokens out of the answer stream |

## Startup sequence (lifespan)

`backend/src/core/api.py:lifespan` runs, in order:

1. **Embedded PostgreSQL** — `start_postgres(config.POSTGRES_DATA_DIR)`. On a first run
   this pays a one-time `initdb`; the phase event `preparing_database` is emitted so the
   Electron loader can say so.
2. **Bind SQLAlchemy** — `init_database(app.state.postgres.sqlalchemy_url)`.
3. **Engine selection** — `config.LLM_Engine = BaseEngine.get_engine()`.
4. **Schema migration** — Alembic, forward-only, run off the event loop
   (`run_in_threadpool(run_migrations, ...)`). Phase event: `running_migrations`. See
   [Database migrations](dev/db-migrations.md).
5. **Seed and catalog** — `startup_populate_database()` sets startup variables, the
   hardware profile, and reconciles the model catalog from the bundled snapshots. Phase
   event: `loading_catalog`.
6. **KB vector store** — `init_kb_store(...)`, after the migration because its
   cross-schema foreign keys reference the business tables.
7. **Checkpointer** — `open_checkpointer(...)` on the same database, kept open for the
   app lifetime and published on `app.state.checkpointer`.
8. **Idle cleanup task** — `config.LLM_Engine.start_cleanup_task()`; the monitor ticks
   every 300 s and unloads the model once it has been idle longer than
   `_max_idle_time` (300 s).
9. **Database watchdog** — `start_watchdog(app)` detects a dead embedded cluster,
   resurrects it, and exposes the state on `/erudi/health/` as `db`.
10. **Post-ready backfill** — a background task verifies the tool-call wire capability of
    models downloaded before that column existed. It runs after readiness, in a
    threadpool, never inside the awaited boot sequence.

Shutdown reverses the order: cancel the backfill task, stop the watchdog, stop the
cleanup task, clean up the engine, close the checkpointer, close the KB store, and stop
the PostgreSQL cluster last.

Startup phase events are informational. Readiness is the launcher's `ready` event or a
confirming health check — never a phase.

### Catalog snapshots

The remote catalog (which quant exists for each foundation model in a given engine
format) is identical for every user, so it is resolved once at build time and shipped as
JSON: `backend/src/database/catalog_snapshot_mlx.json` and
`catalog_snapshot_gguf.json`. First boot loads the snapshot for the active engine's
`FORMAT_TAG` with zero Hugging Face calls. Regeneration is a build-time CLI:

```bash
cd backend
python -m src.database.catalog_snapshot                     # active engine (mlx on Mac)
ERUDI_FORCE_CPU=1 python -m src.database.catalog_snapshot   # gguf
```

## Code conventions

### Naming

- **snake_case** for variables, functions, files, directories
- **Capitalized_Snake_Case** for classes (`MLX_Engine`, `CUDA_Engine`)
- **UPPER_SNAKE_CASE** for constants
- Absolute imports: always `from src.core.config import ...`

### Domain structure

```text
backend/src/domains/<domain>/
├── __init__.py
├── endpoints.py      # FastAPI routes
├── schemas.py        # Pydantic models
├── services.py       # business logic
└── repository.py     # data access (optional)
```

Request flow:

```text
HTTP request
   ↓
endpoints.py (Pydantic validation)
   ↓
services.py (business logic)
   ↓
repository.py (DB queries)
   ↓
entities/*.py (SQLAlchemy models)
```

### Error handling

All business exceptions inherit from `AppBaseException`
(`backend/src/core/exceptions.py`), which carries a message, an HTTP status code, an
`erudi_code`, an optional `trace` (logged, never returned), and an optional `detail`
payload returned verbatim to the client.

```python
from src.core.exceptions import ModelNotFoundException, StateConflictException

raise ModelNotFoundException("qwen2.5-7b")            # 404 MODEL_NOT_FOUND
raise StateConflictException(                          # 409 STATE_CONFLICT
    "Base model has dependent assistants",
    status_code=409,
    detail={"dependents": [...]},
)
```

Two handlers are registered in `add_exception_handlers` (`core/api.py`):
`app_base_exception_handler` for `AppBaseException`, and `unhandled_exception_handler`
for bare `Exception`. Both return the same shape:

```json
{
  "success": false,
  "error": {
    "type": "MODEL_NOT_FOUND",
    "message": "Model 'qwen2.5-7b' not found",
    "detail": { "…": "present only when the exception carries one" }
  }
}
```

See [Exception Handling](dev/exceptions.md).

### Structured logging

```python
from src.core.logging import logger

logger.info("Model loaded")
```

Log-message literals must stay ASCII-only. Logs are written to
`backend/logs/backend.log` in development (rotating file, 10 MB, 10 backups). Packaged
builds write to the OS log directory instead — macOS: `~/Library/Logs/erudi/backend.log`,
Windows: `%LOCALAPPDATA%\erudi\logs\backend.log`, Linux:
`${XDG_STATE_HOME:-~/.local/state}/erudi/logs/backend.log`. See
[Logging & Traceability](logging.md).

## Persistence

Storage is an **embedded PostgreSQL cluster with pgvector**, provided by `pgserver` as
pip wheels — no Docker, no system install. The lifespan boots the cluster
(`src/launcher/postgres_runtime.py`; data directory `backend/data/postgres/` in
development, a user-writable directory in packaged builds), creates the `erudi` database
and the `vector` extension, then binds SQLAlchemy through `init_database(url)` over
psycopg3 (`postgresql+psycopg://`).

Never import `db_engine` by value — read it through `src.database.core` attributes after
initialization.

One database, three tenants:

| Tenant | Location | Owner |
|---|---|---|
| Business tables | `public` | SQLAlchemy ORM |
| Conversation state | `public` (LangGraph tables) | `AsyncPostgresSaver` |
| KB chunks | `rag.kb_chunks` | langchain-postgres `PGVectorStore` |

### Main entities

- **Llm** — catalog and local models (name, link, `local`, `quantized`, `param_size`,
  `is_base`, `supports_tools`, KB binding)
- **Conversation** — chat sessions (llm_id, temperature, top_p, max_tokens, custom_prompt)
- **Message** — individual messages (conversation_id, sender, content, starred)
- **KnowledgeBase** — KB metadata and status
- **KnowledgeDocument** — one row per ingested file, with its SHA-256 for dedup and a
  status (`ingested`, `empty`, `pending_vision`, …)
- **KBJob** — ingestion jobs
- **DownloadJob** — download jobs (status, progress, total bytes)
- **HardwareProfile** — detected hardware and its performance scores
- **StartupVariables**, **UserSettings** — first-run flags and UI language

See [Entities Reference](reference/entities.md).

PostgreSQL sequences are non-transactional, so tests never assert absolute primary-key
values.

## Retrieval-augmented generation

### Ingestion pipeline (`src/ingestion`)

```text
Source files (.pdf .docx .xlsx .csv .txt .md; images → pending_vision)
   ↓  reader.py — DocumentReader routes to a per-format extractor
Markdown pivot
   ↓  cleaning.py — non-destructive cleaning (accents and casing preserved)
   ↓  chunking.py — 3-pass, token-accurate on the e5 tokenizer
Chunks (~180 tokens, ~15 % overlap)
   ↓  embeddings.py — intfloat/multilingual-e5-small, 384 dimensions
   ↓  vector_store.py — add_kb_chunks into rag.kb_chunks
KnowledgeDocument rows (SHA-256 dedup, per-file status)
```

Chunk defaults live in `backend/src/ingestion/chunking.py`
(`DEFAULT_TARGET_TOKENS = 180`, `DEFAULT_OVERLAP_TOKENS = 27`).

The `query:` and `passage:` prefixes required by the e5 model family are mandatory and
applied by `E5Embeddings`; skipping them silently degrades retrieval.

### Retrieval

`rag.kb_chunks` carries both a dense HNSW index (cosine over 384-dim embeddings) and a
sparse `tsvector` column (`pg_catalog.simple`). A query runs both and fuses the two
rankings with Reciprocal Rank Fusion (k = 60).

Always search through `search_kb_chunks`: langchain-postgres 0.0.17 freezes the first
query's `fts_query` on the shared hybrid config, so each call needs a fresh config.

The conversation-side entry point is `retrieve_kb_excerpts`
(`backend/src/utils/kb_utils.py`).

Whether the model retrieves by calling the `search_knowledge_base` tool (agentic) or the
service retrieves on every turn (systematic) depends on the model's tool-calling support;
`ERUDI_KB_AGENTIC` forces either mode.

See the [Knowledge Base guide](guides/knowledge_base.md).

## Context budget by model size

`get_prompting_strategy(param_size)` (`backend/src/utils/prompt_utils.py`) returns the
system-prompt tier and the **KB token budget** — a ceiling in e5 tokens (roughly 180 per
chunk), not a chunk count. The adaptive cut in `kb_utils` decides per query how much of
the budget to consume.

| Parameter size | Tier | KB token budget |
|---|---|---|
| ≤ 2B (or unknown) | `tiny` | 400 |
| ≤ 4B | `small` | 700 |
| < 8B | `medium` | 1000 |
| ≤ 16B | `large` | 1400 |
| > 16B | `xlarge` | 2000 |

## Performance notes

### Idle model cleanup

`BaseEngine` keeps a single model in memory as class state. The cleanup monitor started
in the lifespan ticks every 300 seconds and unloads the model once it has been idle
longer than `_max_idle_time` (300 seconds). A generation in flight sets the active marker
`_last_used = None`, which blocks the monitor from reaping the model mid-stream.

### Streaming

Conversation turns stream as **NDJSON** (`application/x-ndjson`): one JSON event per
line, typed `answer`, `thinking`, `tool_call`, `tool_result`, `error`, or `done`. Title
generation and Arena keep a plain-text stream.

## Main flows

### Conversation turn

```text
POST /erudi/conversations/{id}/query  {"question": "..."}
  ↓
endpoints.query_and_respond → StreamingResponse(application/x-ndjson)
  ↓
ConversationService.query_and_respond_stream
  ↓
AgentRunner: create_agent(thread_id, checkpointer, summarization)
  ↓  history restored from the checkpointer; KB retrieval agentic or systematic
engine child server /v1/chat/completions (SSE)
  ↓
structured events → one JSON line each → renderer
  ↓
user + assistant messages persisted after the stream completes
```

### Model download

```text
POST /erudi/llms/{id}/download          (catalog)
  or POST /erudi/llms/download/huggingface {"link": ...}   (search / pasted link)
  ↓
Create DownloadJob (status=pending)
  ↓
Background task starts
  ↓
List repo files → gate: the repo MUST ship an artefact in the engine's format
(FORMAT_TAG: an MLX repo tag on Apple Silicon, a .gguf file on CPU/CUDA)
  → otherwise InvalidInputException BEFORE a single byte is downloaded
  ↓
Download the pre-built quant from Hugging Face → update progress
  ↓
Move into place (no local conversion or quantization)
  ↓
Llm row local=1, DownloadJob status=completed
```

See the [LLMs guide](guides/llms.md).

## Quick reference

- [Core Reference](reference/core.md) — config, logging, exceptions
- [Engines Reference](reference/engines.md) — MLX, CUDA, CPU
- [Conversations Reference](reference/conversations.md) — chat endpoints
- [LLMs Reference](reference/llms.md) — model management
- [KB Reference](reference/knowledge_base.md) — ingestion and retrieval
- [Entities Reference](reference/entities.md) — SQLAlchemy models
