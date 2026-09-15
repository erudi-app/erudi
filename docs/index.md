# Erudi Documentation

Technical documentation for Erudi — a desktop application that runs open-source
LLMs locally, with retrieval over your own documents.

## Overview

Erudi lets you:

- **Run LLMs locally** on Apple Silicon (MLX), NVIDIA GPUs (CUDA), or CPU (llama.cpp)
- **Ground a model on your documents** by attaching a Knowledge Base (hybrid RAG)
- **Compare models** side by side in the Arena
- **Manage the model lifecycle**: catalog, download of pre-built quants, idle unloading

Erudi downloads pre-built MLX and GGUF artifacts. It never converts or quantizes
weights locally.

## Sections

### Getting started

- [Getting Started](usage.md) — prerequisites, backend/frontend setup, first API calls
- [Backend Launcher](guides/backend-run.md) — `run.py`, ports, lifecycle events

### Concepts

- [Architecture](architecture.md) — DDD layering, multi-engine, startup sequence
- [Logging & Traceability](logging.md) — log files, request-id correlation, tracing a bug
- [Internationalization](i18n.md) — locale files, the no-literal-string rule

### Guides

- [Conversations](guides/conversations.md) — sessions, streaming, parameters
- [LLMs](guides/llms.md) — catalog, downloads, model management
- [Knowledge Base](guides/knowledge_base.md) — ingestion, retrieval, attaching a KB
- [Hardware Detection](guides/hardware.md) — engine selection and performance scores

### Development

- [Exception Handling](dev/exceptions.md) — exception hierarchy and error wire format
- [Engines Architecture](dev/architecture/engines.md) — engine hierarchy, llama.cpp builds
- [Database Migrations](dev/db-migrations.md) — Alembic workflow

### API reference

- [Core](reference/core.md) — configuration, logging, health
- [Engines](reference/engines.md) — MLX, CUDA, CPU
- [Agents](reference/agents.md) — LangChain runner, prompts, middlewares, checkpointer
- [Ingestion](reference/ingestion.md) — document reader, chunking, embeddings, vector store
- [Launcher](reference/launcher.md) — runtime paths, embedded PostgreSQL
- [Conversations](reference/conversations.md) — chat endpoints
- [LLMs](reference/llms.md) — model management
- [Knowledge Base](reference/knowledge_base.md) — ingestion and retrieval
- [Arena](reference/arena.md) — model comparison
- [Hardware](reference/hardware.md) — system monitoring
- [User settings](reference/user_settings.md) — global preferences (web search, language, reasoning effort)
- [Startup](reference/startup.md) — first-boot payload for the frontend
- [Entities](reference/entities.md) — SQLAlchemy models
- [Database](reference/database.md) — seeding and DB access

## Architecture at a glance

```text
┌─────────────────────────────────────────────────────────────┐
│ Frontend (Electron + React)                                 │
└────────────────┬────────────────────────────────────────────┘
                 │ HTTP on 127.0.0.1:27182, routes under /erudi
┌────────────────▼────────────────────────────────────────────┐
│ Backend (FastAPI)                                           │
│  ├─ Domains: conversations, llms, knowledge_base, arena,    │
│  │            hardware, startup, user_settings              │
│  ├─ Agents:  LangChain/LangGraph runner + checkpointer      │
│  ├─ Engines: MLX_Engine / CUDA_Engine / CPU_Engine          │
│  ├─ Database: embedded PostgreSQL + pgvector (SQLAlchemy)   │
│  └─ Ingestion: document reader, chunking, e5 embeddings,    │
│                hybrid dense + keyword retrieval             │
└─────────────────────────────────────────────────────────────┘
```

## Quick start

```bash
# Backend (from the repo root)
cd backend
source venv/bin/activate         # macOS/Linux; Windows: .\venv\Scripts\Activate
python run.py --port 27182

# Frontend (second terminal)
cd frontend
npm start
```

See [Getting Started](usage.md) for the full setup, including the llama.cpp build step.

## Links

- [GitHub repository](https://github.com/erudi-app/erudi)
- [Architecture](architecture.md)
- [API reference](reference/core.md)
