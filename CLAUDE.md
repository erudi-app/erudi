# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Erudi is

Desktop app that runs open-source LLMs locally. Two processes:

- **Backend** — Python **3.12** FastAPI server (3.12 is required: `pgserver` only ships wheels up to cp312), launched by `backend/run.py`, listens on `127.0.0.1:27182` by default. Routes are mounted under the `/erudi` prefix.
- **Frontend** — Electron + React + Tailwind, packaged with electron-builder. In production the main process spawns the bundled PyInstaller backend executable; in dev it expects the backend to be running already.

Hardware backend is selected at startup by `BaseEngine.get_engine()` (`backend/src/engines/base_engine.py:528`): `MLX_Engine` on macOS ARM, `CUDA_Engine` on Linux/Windows with NVIDIA, `CPU_Engine` otherwise. Set `ERUDI_FORCE_CPU=1` to bypass GPU detection. Every environment variable the backend reads is listed in `backend/.env.example` (the only secret is `HF_TOKEN`, read from the environment at runtime and never embedded in a build); `tests/test_env_example.py` fails on any new `os.getenv` that is not listed there.

**All three engines follow the same pattern**: they spawn an OpenAI-compatible HTTP server in a child process and talk to it over `http://127.0.0.1:<port>/v1/chat/completions` (SSE). The shared lifecycle (port pick, two-stage `/health` + chat-ping probe, SSE byte-buffer parser, atexit storage with proper unregister-on-swap, idle-cleanup active marker, kwarg translation) lives in `BaseChatServerEngine`. `BaseLlamaCppEngine` sits between it and the CPU/CUDA concretes to factor what's specific to the `llama-server` binary (Popen lifecycle, GGUF picker, install-dir resolution, `repetition_penalty → repeat_penalty` kwarg rename). Concrete subclasses implement four small hooks: `_spawn_child` (CPU/CUDA via `subprocess.Popen`, MLX via `multiprocessing.Process(target=run_mlx_vlm_server, ...)` because PyInstaller frozen builds have no Python interpreter at `sys.executable` to pass `-m` to), `_terminate_process`, `_proc_is_alive`, and `_resolve_model_artifact`.

## Common commands

### Backend

```bash
# First-time setup (pick your platform)
bash scripts/dev/backend/setup-mac-silicon.sh
bash scripts/dev/backend/setup-linux-cuda.sh
.\scripts\dev\backend\setup-win-cuda.ps1

# Build llama.cpp (required before running on macOS/CPU)
bash scripts/dev/backend/build-llamacpp-cpu-macos-silicon.sh

# Run dev server (from repo root). run.py supervises uvicorn and emits
# newline-delimited JSON lifecycle events on stdout — do not replace it
# with a raw `uvicorn` call when testing the Electron integration.
cd backend && source venv/bin/activate && python run.py --port 27182

# Alternative: uvicorn with reload (skips JSON events, fine for API-only work)
cd backend && source venv/bin/activate && PYTHONPATH=. uvicorn src.main:app --reload --port 27182

# Tests
cd backend && pytest tests/                              # full suite (local Mac)
cd backend && pytest tests/test_engines.py -x            # single file
cd backend && pytest tests/ -k "test_name"               # by name
cd backend && pytest tests/ --ignore=tests/e2e -m "not mlx_only"  # CI mode (Linux)
cd backend && pytest tests/ -m mlx_only                  # MLX integration only (Mac)
ERUDI_TEST_THINKING=1 pytest tests/ -k thinking          # opt-in regression: <think> tokens
ERUDI_TEST_GEMMA=1 pytest tests/ -k gemma                # opt-in regression: Gemma <end_of_turn> EOS

# Lint and format
# (from the repo root; `backend scripts` is every .py in the repo)
ruff check backend scripts
ruff format backend scripts              # apply
ruff format --check backend scripts      # what CI checks
```

`pytest.ini` sets `asyncio_mode = auto`, `pythonpath = .`, and `addopts = --strict-markers` — any `@pytest.mark.<name>` not declared in `pytest.ini:markers` is a hard error. Declared markers: `unit`, `integration`, `mlx_only`, `e2e`, `network`. Imports use `from src.*` and `from tests._helpers import is_mlx_platform` (etc.) for the platform-skip helpers.

MLX integration tests rely on a session-scoped fixture (`mlx_test_model_path` in `tests/conftest.py`) that downloads `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (~280 MB, Apache 2.0, no HF license accept) on first run. The fixture `pytest.skip`s cleanly on non-MLX hosts.

### Frontend

```bash
cd frontend
npm install              # first time
npm start                # dev mode (expects backend already running)
npm run lint             # ESLint with autofix
npm run lint:check       # ESLint without autofix (matches CI)
npm run format:check     # Prettier check (matches CI)
npm run dist:mac         # macOS arm64 via electron-builder
npm run dist:win         # Windows x64 via electron-builder
npm run dist:linux       # Linux x64 AppImage via electron-builder
```

### Combined dev workflow

`bash scripts/dev/dev-start.sh` opens two Terminal windows (backend + frontend, macOS only) and kills anything on `BACKEND_PORT` first. Set `BACKEND_PORT` to override 27182.

### Production build

Backend → PyInstaller bundle → copied into `frontend/backend/` → packaged by electron-builder. See `BUILD.md` for the orchestrated scripts; the spec files are `backend/backend.spec` (Windows CUDA), `backend/backend-mac-silicon.spec` (macOS), and `backend/backend-cpu.spec` (Windows CPU and Linux CPU).

## Architecture essentials

### Backend layering (DDD)

```
backend/src/
├── main.py              FastAPI app = api.lifespan + register_routers
├── core/                api.py (lifespan, CORS, exception handler), config, logging, exceptions, health
├── engines/             BaseEngine + MLX/CUDA/CPU — single singleton model in memory
├── agents/              LangChain layer: runner (create_agent/turn), prompts, checkpoint (AsyncPostgresSaver)
├── domains/<name>/      endpoints.py → services.py → repository.py → entities/ (Pydantic in schemas.py)
├── entities/            SQLAlchemy ORM models (Conversation, Message, Llm, KnowledgeBase, KnowledgeDocument, …)
├── database/            core.py (init_database/session), seed.py (create_tables, startup_populate_database)
├── ingestion/           KB pipeline: DocumentReader façade + *Extractor backends, cleaning,
│                        3-pass chunking (e5 tokenizer), E5Embeddings, vector_store (rag.kb_chunks)
├── launcher/            runtime_paths.py (packaged vs. dev paths), postgres_runtime.py (embedded cluster)
└── utils/               kb_utils (hybrid retrieval façade), prompt_utils, hf_model_metadata
```

Routers mounted under `/erudi` (in `core/api.py:register_routers`): `llms`, `hardware`, `arena`, `knowledge_base`, `conversations`, `user_settings`, `startup` (from `domains/`) plus `health` (from `core/health.py`, not a domain). Fine-tuning was removed as dead code (#99) — there is no `training` router and no `file_processor`. The frontend hits `http://127.0.0.1:27182/erudi/...` (see `frontend/src/config/api.js`).

**Engine singleton.** `BaseEngine` keeps `_model`, `_tokenizer`, `_model_id`, `_last_used` as class attributes shared across requests, guarded by `_lock`. A 300s idle cleanup task (`start_cleanup_task`) is registered in `lifespan`. Don't instantiate engines — call class methods on the result of `BaseEngine.get_engine()`. Selected engine lives in `src.core.config.LLM_Engine`.

**Adding an engine.** Subclass `BaseEngine`, implement every `@abstractmethod` (`get_model_and_tokenizer`, `generate_stream`, `get_hardware_info`, `warm_up_accelerator`, `get_performance_evaluation`, `get_flat_hardware_data`), set `FORMAT_TAG` to the HF library tag of the pre-built artefacts it loads (`mlx` / `gguf` — the catalog, the HF search and the by-link download gate are all filtered on it), then wire it into `BaseEngine.get_engine()`. There is no local HF→engine-format conversion: the app only downloads pre-built quants (#408). Keep OS/hardware branching out of services — it belongs in engines.

**Exceptions.** Raise `AppBaseException` subclasses (`EngineException`, `ModelNotFoundException`, `InvalidInputException` in `src/core/exceptions.py`); the global handler in `core/api.py` returns structured JSON. Don't raise bare `Exception` in domain code.

### Launcher contract

`backend/run.py` is **not** a thin wrapper — it's the production entrypoint expected by the Electron main process and emits newline-delimited JSON events on stdout: `starting`, `ready`, `shutdown`, `startup_error` (codes: `PORT_IN_USE`, `CRASH_BEFORE_READY`, `PORT_TIMEOUT`, `IMPORT_ERROR`, `DATA_PREP_ERROR`, `NO_PORT_AVAILABLE`, `UNEXPECTED_ERROR`, `POLLING_ERROR`). It scans ports `27182-27199` (canonical port 27182 — the digits of e; the scan stops short of the inference pools at 27200+) and falls back to killing the PID on the middle of the window if all are busy. Preserve this protocol if you touch the file.

### Frontend layering

```
frontend/src/
├── main.js              Electron main: spawns/kills backend (process group on POSIX,
│                        taskkill /F /T on Windows), parses backend JSON events,
│                        owns auto-update via electron-updater
├── preload.js           contextBridge surface for the renderer
├── renderer.js          React entry
├── App.jsx              react-router-dom routes
├── pages/               top-level screens (ChatPage, ConversationPage, ArenaPage, …)
├── components/          shared UI (Tailwind + lucide-react + framer-motion)
├── contexts/            React contexts (KnowledgeBase, DownloadModal)
├── services/api/client.js  fetch wrapper with retry + timeout + error normalization
├── config/api.js        API_BASE_URL = http://127.0.0.1:27182/erudi
└── utils/               logger, hardwareTransform
```

`nodeIntegration` is off — anything renderer-needs-from-Node goes through `preload.js` via `contextBridge.exposeInMainWorld` and `ipcMain.handle`.

## Conventions

- **Python**: `snake_case` files/functions, `Capitalized_Snake_Case` classes (yes, with underscores — see `MLX_Engine`, `CUDA_Engine`), absolute imports from `src.*`. Use `pathlib.Path`, never string paths. Logging via `from src.core.logging import logger` — no `print()` in production paths.
- **ASCII-only where it executes or gets parsed**: log-message literals (`logger.*(...)` strings) and machine-parsed bundled data files (e.g. `alembic.ini`) must stay ASCII-only — no `→`, `—`, accents (see #168/#149). Non-ASCII is fine in comments and docstrings (Python reads source as UTF-8 regardless of locale).
- **Async-first.** Don't block the event loop with synchronous I/O in endpoints/services.
- **Ruff config** (`backend/ruff.toml`) only enforces `F` + `E7`. `E501`/`E402`/`F841`/`E701` are intentionally ignored — don't reintroduce them as blockers. `ruff format` is the project's only formatter, and `backend/ruff.toml` sets `line-length = 100` at the top level so the formatter and the linter agree without CLI flags. The root `ruff.toml` carries the same width for the one Python file outside `backend/` (`scripts/ci/smoke_boot.py`); ruff picks the nearest config per file and never merges the two. `ruff format` runs from `.pre-commit-config.yaml` and is checked in CI — never hand-format around it.
- **Frontend**: ESLint + Prettier are enforced by CI (`lint:check`, `format:check`).
- **Frontend copy is translated**: no hardcoded user-facing string in JSX or in the Electron menus/dialogs — every string is a `t('ns:key')` over `frontend/src/locales/<lang>/*.json` (`en` source of truth; `fr`, `es`, `zh`), guarded by `locales.test.js` and the `i18next/no-literal-string` ESLint rule. See `docs/i18n.md`.
- **Documentation moves with the code, in the same change.** A change that alters behaviour and leaves a page describing the old behaviour is incomplete, not a follow-up. Grep the repository for every mention of what you touched — the function name, the endpoint, the path, the behaviour — and fix each one: `README.md`, `docs/` (`docs/privacy.md` above all), the comments and docstrings beside the code, this file, `CONTRIBUTING.md`, and `QA-SCENARIOS.md` when an expected result changes. Nothing in CI enforces this. It matters most where the documentation *is* the product: a privacy page listing a request the app no longer makes is a false statement about the reason people install Erudi, not a stale detail.
- **Documentation is present-tense.** It describes what is true now — never "this used to", never "we fixed", never how the code got here. That narrative belongs in the commit message and the pull request.
- **Commits**: `type(scope): description` (`feat`, `fix`, `docs`, `chore`, `ci`). Don't mention Claude/AI or add `Co-Authored-By: Claude`.
- **Branches and pushes**: `main` is protected, PRs are **squash-merged**, and the
  repo requires branches to be **up to date** before merging. Two consequences:
  - **Never rebase or amend a branch that has been pushed.** Rewriting history
    forces a `--force-push`, which can dismiss reviews and strands inline review
    comments on SHAs that no longer exist. When GitHub says your branch is behind,
    use **"Update branch"** (or `git merge origin/main`) and push normally.
  - **Don't groom the branch history.** The squash discards it anyway, so
    incremental commits are fine and cost nothing. Tidying costs a force-push.
- **Requirements**: never edit a single platform file blindly. Common deps live in `backend/requirements/meta/base.txt`; platform/hardware specifics in `meta/*-specs.txt`; entrypoints (`entrypoints/dev/*.txt`, `entrypoints/prod/*.txt`) compose them. Read `backend/requirements/README.md` before adding a dep.

## Data and storage

- **Embedded PostgreSQL + pgvector** via `pgserver` (pip wheels, no Docker, no system install). The FastAPI lifespan boots the cluster (`src/launcher/postgres_runtime.py`, data dir `backend/data/postgres/` in dev; user-writable dir via `runtime_paths.py` in packaged builds), creates the `erudi` database + `vector` extension, then binds SQLAlchemy through `init_database(url)` (psycopg3, `postgresql+psycopg://`). Never import `db_engine` by value — read it via `database.core` attributes after init.
- One database, three tenants: business tables in `public` (SQLAlchemy), LangGraph checkpointer tables in `public` (`AsyncPostgresSaver`, conversation state), KB chunks in `rag.kb_chunks` (langchain-postgres `PGVectorStore`).
- Knowledge Base = hybrid retrieval over `rag.kb_chunks`: dense HNSW (cosine) on `multilingual-e5-small` embeddings (384-dim, `query:`/`passage:` prefixes mandatory) + sparse tsvector (`pg_catalog.simple`) fused by RRF (k=60). Ingestion pipeline lives in `src/ingestion/` (DocumentReader → non-destructive cleaning → 3-pass token-accurate chunking ~180 tokens/15 % overlap → `add_kb_chunks`); per-file dedup via `KnowledgeDocument` SHA-256. Images/scanned PDFs are accepted as `pending_vision` (no OCR tier bundled yet). ⚠ langchain-postgres 0.0.17 freezes the first query's `fts_query` on the shared hybrid config — always search through `search_kb_chunks` (fresh config per call).
- Tests run against a REAL throwaway pgserver cluster (session-scoped fixture in `tests/conftest.py`); per-test isolation via outer-transaction rollback. PG sequences are non-transactional — never assert absolute pk values.

## CI gates (must pass before merge)

- **Backend** (`.github/workflows/backend-ci.yml`, Python 3.12, 3-leg matrix): `compileall`, `ruff check backend scripts`, `ruff format --check backend scripts`, `from src.main import app`, `pytest tests/ -q --ignore=tests/e2e -m "not mlx_only"`. All three legs (`ubuntu-latest`, `windows-latest`, `macos-14`) gate merges and run with `-x` — see `docs/dev/backend-ci-multi-os.md` for the triage that got them there. `pytest.ini` sets `timeout = 600` so a hung test fails instead of consuming the job. The Ubuntu leg runs against `CPU_Engine` only — keep CPU paths working.
- **Frontend** (`.github/workflows/frontend-ci.yml`, Node 20): `npm ci`, `npm run lint:check`, `npm run format:check`, `npm run test:run` (Vitest).
- **Full-app smoke** (`.github/workflows/app-build-smoke.yml`): builds the complete shippable app on macOS, Windows and Linux, then boots the BUNDLED backend and asserts it reaches `ready`. It is the heaviest gate and the one most likely to fail on a packaging change — it runs on `pull_request` (ready for review) and in the merge queue. **The Linux leg is advisory**, not required: Linux has never had a manual pass on real hardware, so its signal is not yet trusted enough to block a merge. Read it, don't ignore it.

These four workflows run on `pull_request` and again in the merge queue, on the
candidate merged with `main` — so every change is validated twice, the second
time against the code it will actually land beside. None of them filters on
paths: a required check that a path filter skips never reports at all, which
blocks the pull request forever.

Not a gate, but worth knowing: **`llamacpp-build.yml`** compiles the inference
binary, and is the only thing in CI that does. It is path-filtered to pull
requests touching `scripts/dev/backend/build-llamacpp-*` or the llama.cpp
submodule. Change a compile flag anywhere else and nothing verifies it before a
release tag. It also starts what it compiles: the CPU binary is run with
`--version`, which prints llama.cpp's build header and exits 0 from the argument
parser without loading a model. The CUDA binary cannot be started on a runner —
ggml links the NVIDIA driver library (`libcuda.so.1` / `nvcuda.dll`, shipped with
the driver, not the toolkit), so the loader fails before `main()` — so the Linux
CUDA leg checks instead that every dependency but that one resolves, and the
Windows CUDA leg only checks the file is there. `release.yml` starts the copy
PyInstaller froze into `backend/dist`, which is a separate claim: the freeze
copies the file and can pick up the wrong flavour or lose a sibling library.
Neither runs inference; only the release QA pass on real hardware does.

## Logs

- Backend: `os.tmpdir()/erudi-backend.log`, written by `frontend/src/main.js` — that is `/tmp` on Linux, `%TEMP%` on Windows, and `$TMPDIR` (a per-user folder under `/var/folders/…`, NOT `/tmp`) on macOS. Backend's own logger writes to `backend/logs/backend.log`.
- Frontend (production): electron-log default location.

## Conflict with the global CLAUDE.md

Nothing in this repository takes a stance on the language of chat replies, so a global rule mandating one wins by default. The repo-level rules on naming, exceptions, async, and engine encapsulation stack on top — no conflicts to flag today.
