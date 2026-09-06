# Getting Started

How to set up and run Erudi from source. All commands are run **from the repository
root** unless stated otherwise.

## Prerequisites

### Platform support

| Platform | Backend | Status |
|---|---|---|
| Windows (NVIDIA GPU) | CUDA via `llama-server` | Supported |
| Windows (no GPU) | CPU via `llama-server` | Supported |
| macOS Apple Silicon (macOS 14+) | MLX | Supported |
| Linux (NVIDIA GPU) | CUDA via `llama-server` | Builds and launches in CI, not yet tested on real hardware |
| Linux (CPU) | CPU via `llama-server` | Builds and launches in CI, not yet tested on real hardware |

Intel Macs are not a target.

### Software

- **Python 3.12 exactly** — `pgserver`, which ships the embedded PostgreSQL cluster,
  publishes wheels up to cp312 only.
- **Node.js >= 20.9** — the floor `webpack-cli` declares; the CI legs run Node 20, which satisfies it.
- **Git**, with submodule support (llama.cpp is a submodule).
- Platform extras: a CUDA toolkit on Windows/Linux to compile `llama-server` for an
  NVIDIA GPU — any 12.x builds, and releases use 12.8, the first that emits native
  code for RTX 50 cards; Xcode Command Line Tools on macOS.

!!! note "This page is for building Erudi, not for running it"
    The published installers carry the inference engine and its CUDA runtime. A user
    with an NVIDIA GPU needs only a driver; everyone else needs nothing installed at
    all. Nothing on this page applies to them.

### Recommended hardware

- **Apple Silicon**: M1 or later, 16 GB or more of unified memory
- **NVIDIA GPU**: 8 GB or more of VRAM
- **CPU only**: 16 GB or more of RAM (slower fallback)

The app computes a recommended model-size range for your machine and exposes it on
`GET /erudi/hardware/app_startup`. See the [Hardware guide](guides/hardware.md).

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/erudi-app/erudi.git
cd erudi
git submodule update --init --recursive
```

The submodule at `backend/forks/llama-cpp` is required for the CPU and CUDA builds.

### 2. Set up the backend

Run the script for your platform from the repository root. Each one creates
`backend/venv` and installs the matching entrypoint from `backend/requirements/`.

| Platform | Script |
|---|---|
| macOS Apple Silicon | `bash scripts/dev/backend/setup-mac-silicon.sh` |
| Windows CUDA | `.\scripts\dev\backend\setup-win-cuda.ps1` |
| Windows CPU | `.\scripts\dev\backend\setup-win-cpu.ps1` |
| Linux CUDA | `bash scripts/dev/backend/setup-linux-cuda.sh` |
| Linux CPU | `bash scripts/dev/backend/setup-linux-cpu.sh` |

### 3. Build llama.cpp

The `llama-server` binary is not committed; build it for your platform before running
the CPU or CUDA engine.

```bash
# macOS Apple Silicon (local CPU engine, for development)
bash scripts/dev/backend/build-llamacpp-cpu-macos-silicon.sh

# Linux CPU
bash scripts/dev/backend/build-llamacpp-cpu-linux.sh

# Linux CUDA
bash scripts/dev/backend/build-llamacpp-cuda-linux.sh
```

On Windows, in PowerShell:

```powershell
.\scripts\dev\backend\build-llamacpp-cpu-win.ps1
.\scripts\dev\backend\build-llamacpp-cuda-win.ps1
```

Artifacts land in `backend/artifacts/llama-cpp/<cpu|cuda>/`. See
[Engines Architecture](dev/architecture/engines.md) for the build flags and what the
scripts do.

### 4. Set up the frontend

```bash
cd frontend
npm install
```

## Running

### Backend

```bash
cd backend
source venv/bin/activate        # macOS/Linux
# Windows: .\venv\Scripts\Activate

python run.py --port 27182
```

`run.py` is the production entrypoint: it prepares data and log directories, starts the
embedded PostgreSQL cluster through the app lifespan, and emits newline-delimited JSON
lifecycle events on stdout (`starting`, `ready`, `shutdown`, `startup_error`). Keep using
it when testing the Electron integration.

For API-only iteration you can skip the JSON events:

```bash
cd backend && source venv/bin/activate
PYTHONPATH=. uvicorn src.main:app --reload --port 27182
```

The backend prefers port 27182 and scans 27182-27199 if it is taken, announcing the
resolved port in its `ready` event. See [Backend Launcher](guides/backend-run.md).

### Frontend

```bash
cd frontend
npm start
```

The Electron window opens on its own. In development the frontend expects the backend to
be running already.

### Both at once (macOS)

```bash
bash scripts/dev/dev-start.sh
```

This opens two Terminal windows (backend and frontend) and first kills whatever is
listening on the port. Set `BACKEND_PORT` to override 27182.

### Environment variables

Every variable the backend reads is documented in
[`backend/.env.example`](https://github.com/erudi-app/erudi/blob/main/backend/.env.example).
Copy it to the git-ignored `backend/.env`, or export the variables in your shell — the
backend runs fine with no `.env` at all. The only secret is `HF_TOKEN`, needed for gated
Hugging Face repositories.

Two useful ones during development:

```bash
ERUDI_FORCE_CPU=1   # force CPU_Engine even when a GPU is present
ERUDI_LOG_LEVEL=DEBUG
```

## First checks

### Health

```bash
curl http://127.0.0.1:27182/erudi/health/
```

The trailing slash matters. Expected response:

```json
{"status": "ok", "message": "Backend is running", "db": "ok"}
```

The HTTP status is always 200 so that the Electron boot sequence never mistakes a dead
database for a dead backend; `db` carries the truth (`ok`, `recovering`, or `failed`).

### List models

```bash
curl http://127.0.0.1:27182/erudi/llms/
```

On a first boot the catalog is reconciled from the snapshot bundled with the app, with no
network calls. Rows with `local = 0` are catalog suggestions; `local = 1` means the model
is downloaded.

### Create a conversation

```bash
curl -X POST http://127.0.0.1:27182/erudi/conversations/ \
  -H "Content-Type: application/json" \
  -d '{"llm_id": 1, "temperature": 0.2, "top_p": 0.5, "max_tokens": 1024}'
```

`llm_id` is the only required field; there is no `name` field — a title is generated
later through `POST /erudi/conversations/{id}/generate_title`.

### Ask a question

```bash
curl -N -X POST http://127.0.0.1:27182/erudi/conversations/1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'
```

The response is `application/x-ndjson`: one JSON event per line, typed `answer`,
`thinking`, `tool_call`, `tool_result`, `error`, or `done`. See the
[Conversations guide](guides/conversations.md).

## Repository layout

```text
erudi/
├── backend/
│   ├── src/
│   │   ├── core/         config, logging, exceptions, API wiring, health
│   │   ├── domains/      conversations, llms, knowledge_base, arena, hardware,
│   │   │                 startup, user_settings
│   │   ├── engines/      MLX / CUDA / CPU behind BaseEngine
│   │   ├── agents/       LangChain runner, checkpointer, prompts, tools
│   │   ├── ingestion/    document reader, cleaning, chunking, embeddings, vectors
│   │   ├── entities/     SQLAlchemy models
│   │   ├── database/     init, migrations, seeding, catalog snapshots
│   │   ├── launcher/     runtime paths, embedded Postgres, DB watchdog
│   │   └── utils/        hf_model_metadata, kb_utils, prompt_utils
│   ├── alembic/          migration scripts
│   ├── artifacts/        built llama.cpp binaries (git-ignored)
│   ├── forks/llama-cpp/  llama.cpp submodule
│   ├── data/             downloaded models + embedded Postgres cluster
│   ├── logs/             backend.log
│   ├── requirements/     composed dependency files
│   └── run.py            launcher / production entrypoint
├── frontend/
│   └── src/              Electron main, preload, React app
├── scripts/
│   ├── dev/              setup, llama.cpp builds, dev-start.sh
│   └── build/            distribution builds
└── docs/                 this documentation
```

## Tests and lint

```bash
cd backend && pytest tests/                                       # full suite
cd backend && pytest tests/ --ignore=tests/e2e -m "not mlx_only"  # what CI runs
ruff check backend scripts                                        # lint
ruff format --check backend scripts                               # ruff is also the formatter

cd frontend && npm run lint:check && npm run format:check
```

## Next steps

- [Architecture](architecture.md) — how the code is organised
- [Conversations guide](guides/conversations.md) — the chat API
- [LLMs guide](guides/llms.md) — catalog and downloads
- [API reference](reference/conversations.md)

## Troubleshooting

### The backend does not start

1. Check the virtualenv is active: `which python` must point at `backend/venv/bin/python`.
2. Reinstall dependencies by re-running your platform's setup script.
3. Read `backend/logs/backend.log`, and the launcher's `startup_error` JSON line on
   stdout — its `code` names the failure (`PORT_IN_USE`, `NO_PORT_AVAILABLE`,
   `IMPORT_ERROR`, `DATA_PREP_ERROR`, …). See [Backend Launcher](guides/backend-run.md).

### `No module named 'src'`

Run the backend from the `backend/` directory with the virtualenv active, or set
`PYTHONPATH=.` when invoking uvicorn directly.

### The wrong engine was selected

`BaseEngine.get_engine()` dispatches on `platform.system()` and `platform.machine()`, and
detects NVIDIA GPUs through `pynvml` — not through PyTorch.

```bash
# What the selector sees
python -c "import platform; print(platform.system(), platform.machine())"

# NVIDIA detection, the same way the backend does it
python -c "import pynvml; pynvml.nvmlInit(); print(pynvml.nvmlDeviceGetCount())"
```

- **MLX** requires macOS with an `arm64` machine string. Under Rosetta the machine string
  is `x86_64`, so the selector will not pick MLX.
- **CUDA** requires `pynvml` to report at least one device. Check the driver with
  `nvidia-smi`.
- Set `ERUDI_FORCE_CPU=1` to bypass detection and force `CPU_Engine`.
- Check `user_settings.inference_backend` (`GET /erudi/user_settings/`): `cpu` pins
  `CPU_Engine` on a machine with a usable GPU. The lifespan applies it after the
  migrations, and `ERUDI_FORCE_CPU` wins over it.

The selected engine is logged at startup (`Engine chosen: ...`).

### The GPU is detected but Erudi says it cannot use it

The CUDA pre-flight compares the card's compute capability and the driver's CUDA version
to the floors of the bundled `llama-server` build and emits an `engine_notice` event when
either falls short. The verdict and both readings are in `backend/logs/backend.log`
(`CUDA pre-flight: ...`), and the compatibility matrix is in
[Hardware Detection](guides/hardware.md). Nothing is switched automatically — the app
proposes processor mode and waits.

### `llama-server` not found

Build it (step 3 above) and confirm the binary exists at
`backend/artifacts/llama-cpp/<cpu|cuda>/bin/llama-server`. If the submodule directory is
empty, run `git submodule update --init --recursive`.
