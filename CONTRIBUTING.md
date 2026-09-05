# Contributing to Erudi

Thank you for your interest in contributing. This document explains how the project is structured, how to set up a dev environment, and what we expect from pull requests.

---

## Table of Contents

- [How to contribute](#how-to-contribute)
- [Dev environment setup](#dev-environment-setup)
- [Project conventions](#project-conventions)
- [Backend guide](#backend-guide)
- [Frontend guide](#frontend-guide)
- [Submitting a pull request](#submitting-a-pull-request)
- [Good first issues](#good-first-issues)

---

## How to contribute

1. **Open an issue first** for anything non-trivial (new features, engine changes, architecture decisions). It saves everyone time if we align before you write code.
2. **Fork and branch** — branch off an up-to-date `main`, name your branch descriptively (`fix/cpu-engine-port-reuse`, `feat/ollama-backend`, etc.).
3. **Keep PRs focused** — one concern per PR. A bug fix doesn't need a refactor attached.
4. **Test on your platform** — at minimum run the dev stack end-to-end before opening a PR.

---

## Dev environment setup

See [README.md](README.md) for the full setup walkthrough. The short version:

```bash
# 1. Clone
git clone https://github.com/erudi-app/erudi.git && cd erudi

# 2. Backend (pick your platform script)
bash scripts/dev/backend/setup-mac-silicon.sh

# 3. Build llama.cpp (pick your platform script)
bash scripts/dev/backend/build-llamacpp-cpu-macos-silicon.sh

# 4. Run
cd backend && source venv/bin/activate && python run.py
cd frontend && npm install && npm start
```

---

## Project conventions

### Python

- Python 3.12 exactly — `pgserver` only publishes cp312 wheels, and CI pins `3.12` on all three OS legs
- No type annotations required on code you didn't write, but add them to new public methods
- Logging via the shared app logger: `from src.core.logging import logger` — no bare `print()` in production paths
- Exceptions: use `EngineException` (from `src.core.exceptions`) for engine errors, `DatabaseException` for DB errors — don't raise generic `Exception` in domain code
- Keep engine-specific code inside the engine classes (`CUDA_Engine`, `CPU_Engine`, `MLX_Engine`) — `base_engine.py` and `services.py` should stay platform-agnostic

### Formatting

Python is formatted by **`ruff format`**, and by nothing else. Ruff is also the
linter, so one tool covers both and there is no second formatter to disagree
with it. The width is 100 columns everywhere: `backend/ruff.toml` sets it for
`backend/`, and the root `ruff.toml` sets it for the Python outside it. Ruff
finds the right one per file, so no command here needs a flag. Frontend code is
formatted by Prettier, unchanged.

Let the hooks do it for you:

```bash
pip install pre-commit
pre-commit install
```

That runs `ruff --fix` and `ruff format` on the Python files in each commit,
pinned to the same version `backend/requirements/meta/dev.txt` installs and CI
uses. Or run it by hand, from the repository root:

```bash
ruff format backend scripts            # apply
ruff format --check backend scripts    # what CI asks
```

The tree was reformatted in one pass, so `git blame` on a Python file can land
on that commit instead of the change you were looking for. This makes git skip
it:

```bash
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

GitHub's blame view already skips it without any setup.

### JavaScript / React

- ESLint config is in `.eslintrc.json` — run `npm run lint` before committing
- Components in `frontend/src/components/`, pages in `frontend/src/pages/`
- IPC between main and renderer goes through `preload.js` — don't add `nodeIntegration: true`

### Git

- Conventional commits: `type(scope): description` — `feat`, `fix`, `docs`, `chore`, `ci`. One logical change per commit (`fix(cpu-engine): stop the frozen build from reusing a stale port`)
- Branch from an **up to date** `main`; the repo requires branches to be current before they can merge
- **Never rebase or amend a branch you have already pushed.** Rewriting history forces a force-push, which dismisses reviews and strands inline comments on SHAs that no longer exist. When GitHub says your branch is behind, use **"Update branch"** (or `git merge origin/main`) and push normally
- Don't groom the branch history either — PRs are squash-merged through a merge queue, so the individual commits are discarded anyway
- Don't commit `.env` files, model files, `venv/`, `node_modules/`, or PyInstaller `dist/`/`build/` directories

### Documentation

**Documentation moves with the code, in the same change.** A pull request that
alters behaviour and leaves a page describing the old behaviour is incomplete —
not "to be finished later". Before you open it, search the repository for every
mention of what you touched (the function name, the endpoint, the file path, the
behaviour itself) and correct each one:

- `README.md` — what the app is and what it does
- `docs/` — the published site, `docs/privacy.md` above all
- Comments and docstrings next to the code you changed
- `CLAUDE.md` and this file, when a convention or a gate changes
- `QA-SCENARIOS.md`, when a scenario's expected result changes

This matters most where the documentation *is* the product. Erudi promises that
nothing leaves your machine, and `docs/privacy.md` is where that promise is
spelled out request by request. A page that still lists a request the app no
longer makes is not a stale detail; it is a false statement about the thing
people install this app for.

**Write in the present tense, describing what is true now.** Documentation is
not a changelog: no "this used to…", no "we fixed…", no history of how the code
got here. That story belongs in the commit message and the pull request, where
reviewers look for it. The page describes the software as it stands today.

---

## Backend guide

### Engine system

The backend selects an inference engine at startup based on hardware:

```
src/engines/
├── base_engine.py            ← abstract base + engine selection logic
├── cuda_engine.py            ← NVIDIA GPU via llama-server subprocess (Windows/Linux)
├── cpu_engine.py             ← CPU via llama-server subprocess (all platforms, fallback)
├── mlx_engine.py             ← Apple Silicon via mlx_vlm.server child process (macOS ARM)
└── _mlx_vlm_server_runner.py ← picklable target for the MLX server child process
```

The MLX engine runs `mlx_vlm.server` (pinned `mlx-vlm==0.6.17`), which is a
superset of `mlx_lm.server`: same wire protocol, plus vision and native tool
calling. There is no `embedder_engine.py` — embeddings are handled outside the
engine layer by `E5Embeddings` in `src/ingestion/embeddings.py`
(`intfloat/multilingual-e5-small`, 384 dimensions).

All three inference engines follow the same pattern: they spawn an
OpenAI-compatible HTTP server in a child process and talk to it over
`http://127.0.0.1:<port>/v1/chat/completions` (streaming SSE). The shared
lifecycle (port pick, two-stage `GET /health` + chat-ping probe, SSE
byte-buffer parser, atexit storage, idle-cleanup active marker, kwarg
translation) lives in **`BaseChatServerEngine`**. **`BaseLlamaCppEngine`**
sits between it and the CPU/CUDA concretes to factor the bits specific
to the `llama-server` binary (Popen lifecycle, GGUF picker, install-dir
resolution, `repetition_penalty → repeat_penalty` wire-name rename).

Inheritance:
```
BaseEngine
└── BaseChatServerEngine
    ├── MLX_Engine        (mp.Process + mlx_vlm.server)
    └── BaseLlamaCppEngine
        ├── CPU_Engine    (Popen + llama-server, -ngl 0)
        └── CUDA_Engine   (Popen + llama-server cuda, -ngl <computed>)
```

Adding a new chat-server engine: subclass `BaseChatServerEngine` (or
`BaseLlamaCppEngine` if it wraps `llama-server`) and implement the four
hooks: `_spawn_child`, `_terminate_process`, `_proc_is_alive`,
`_resolve_model_artifact`. Register the class in `base_engine.get_engine()`.

### API domains

```
src/domains/
├── conversations/      ← chat, message history, context/memory
├── llms/               ← model catalog, download, management
├── knowledge_base/     ← multi-format ingestion, pgvector hybrid RAG
├── arena/              ← side-by-side model comparison
├── hardware/           ← hardware detection and scoring
├── user_settings/      ← persisted user preferences
└── startup/            ← first-run seeding, job cleanup
```

Each domain has `endpoints.py` (FastAPI routes), `repository.py` (DB queries) and `schemas.py` (Pydantic models); most also have `services.py` (business logic).

### Running backend tests

```bash
cd backend
source venv/bin/activate
pytest tests/                                            # full suite (local Mac)
pytest tests/ -q --ignore=tests/e2e -m "not mlx_only"    # exactly what CI runs
pytest tests/ -m "mlx_only"                              # only MLX integration (local Mac)
pytest tests/ -m "e2e"                                   # only full-stack e2e tests
```

Pytest markers (declared in `backend/pytest.ini`):

- `unit` — fully mocked, no external dep, runs everywhere
- `integration` — cross-component, may hit DB/filesystem
- `mlx_only` — requires Apple Silicon + `mlx-vlm` + a downloaded MLX model;
  skipped automatically off Apple Silicon by `is_mlx_platform()`
  (`backend/tests/_helpers.py`), which mirrors `BaseEngine.get_engine()`
- `e2e` — full-stack via FastAPI TestClient + real model
- `network` — hits the live Hugging Face API; opt-in via `ERUDI_TEST_NETWORK=1`
  and skipped in CI

`pytest.ini` sets `addopts = --strict-markers`, so any `@pytest.mark.<name>` not
declared there is a hard error.

MLX integration tests use a shared session-scoped fixture
(`mlx_test_model_path`) that downloads `mlx-community/Qwen2.5-0.5B-Instruct-4bit`
(~280 MB, Apache 2.0, no HF license accept) on first run via
`huggingface_hub.snapshot_download` — cached locally afterwards.

Test-mode environment variables:

- `ERUDI_TEST_THINKING=1` — enable the `<think>` token regression suite
  against `mlx-community/Qwen3-0.6B-4bit` (~600 MB)
- `ERUDI_TEST_GEMMA=1` — enable the Gemma `<end_of_turn>` EOS regression
  test against `mlx-community/gemma-3-270m-it-4bit`
- `ERUDI_MLX_TEST_MODEL_DIR=/path` — override the default HF cache for
  the standard MLX test model (offline / pre-seeded environments)
- `ERUDI_MLX_THINKING_MODEL_REPO=mlx-community/...` — override the
  thinking-test model repo
- `ERUDI_FORCE_CPU=1` — short-circuit GPU detection in
  `BaseEngine.get_engine()` to force `CPU_Engine` for testing fallback paths

---

## Frontend guide

The frontend is Electron + React, bundled with webpack and packaged with
electron-builder (`frontend/electron-builder.yml`).

```
frontend/src/
├── main.js             ← Electron main process (backend lifecycle, IPC)
├── preload.js          ← IPC bridge (contextBridge)
├── renderer.js         ← React entry point
├── pages/              ← top-level route components
├── components/         ← shared UI components
└── services/api/       ← API client wrappers
```

### Useful commands

```bash
cd frontend

npm start            # dev mode (hot reload)
npm run lint         # ESLint with autofix
npm run lint:check   # ESLint without autofix (what CI runs)
npm run format:check # Prettier check (what CI runs)
npx vitest run       # unit tests (renderer + utils)
npm run dist:mac     # macOS DMG + zip
npm run dist:win     # Windows NSIS installer
npm run dist:linux   # Linux AppImage
```

Every user-facing string goes through `t('ns:key')` over
`frontend/src/locales/<lang>/*.json` — hardcoded copy in JSX or in the Electron
menus/dialogs is caught by the `i18next/no-literal-string` ESLint rule and by
`locales.test.js`. See `docs/i18n.md`.

### IPC pattern

Main process exposes handlers in `main.js` via `ipcMain.handle(...)`. The renderer calls them via the preload bridge:

```javascript
// preload.js — exposed to renderer
contextBridge.exposeInMainWorld('electronAPI', {
  openDirectory: () => ipcRenderer.invoke('dialog:openDirectory'),
  // ...
})

// In renderer
const path = await window.electronAPI.openDirectory()
```

Don't call Node APIs directly from renderer code.

---

## Submitting a pull request

Run the same gates CI runs, locally, before you push.

**Backend** (`.github/workflows/backend-ci.yml`, Python 3.12, blocking on
`ubuntu-latest`, `windows-latest` and `macos-14`):

```bash
ruff check backend scripts
ruff format --check backend scripts

cd backend
pytest tests/ -q --ignore=tests/e2e -m "not mlx_only"
```

**Frontend** (`.github/workflows/frontend-ci.yml`, Node 20):

```bash
cd frontend
npm run lint:check
npm run format:check
npm run test:run
```

**Full-app smoke** (`.github/workflows/app-build-smoke.yml`) also blocks the merge.
It builds the complete shippable app on all three platforms, then boots the bundled
backend and asserts it reaches `ready`. You cannot reproduce it in one command, but
it is the gate most likely to catch a packaging or dependency change, so expect it
to run for a while after you mark a pull request ready for review. Its **Linux leg
is advisory** — Linux has never had a manual pass on real hardware, so a red Linux
leg is a warning worth reading rather than a merge blocker.

Every gate runs twice: once on your pull request, then again in the merge queue on
your branch merged with `main`. A green pull request can still fail in the queue if
`main` moved underneath it — that is the point of the second pass, not a flake.

If your change touches `scripts/dev/backend/build-llamacpp-*` or the llama.cpp
submodule, **`llamacpp-build.yml`** will compile the inference binary for you and
then start it. It is the only place in CI that does either, so treat a failure
there as real: nothing else will catch a bad compile flag before a release tag.

Starting it means running `llama-server --version`, which prints llama.cpp's build
header and exits 0 from inside the argument parser, before any model is loaded.
That is enough to prove the file is a loadable executable for the target
architecture and that every library it links resolves — none of which a check on
the filename can tell you. The CUDA binary is the exception: ggml links the NVIDIA
driver library (`libcuda.so.1` on Linux, `nvcuda.dll` on Windows), which ships with
the driver rather than the toolkit, so a driverless runner cannot load it at all
and no argument changes that. The Linux CUDA leg checks that every *other*
dependency resolves; the Windows CUDA leg checks only that the binary exists. On a
release tag, `release.yml` repeats the start against the copy PyInstaller froze
into `backend/dist` — a separate claim, because the freeze copies the file into a
new tree and can pick up the wrong flavour or lose a sibling runtime library.
Neither workflow runs inference; the release QA pass on real hardware is the first
thing that does.

Then:

1. Test the full dev stack end-to-end on your platform
2. Update every page your change makes inaccurate — see [Documentation](#documentation). No gate enforces this; a reviewer will ask
3. Write a clear PR description: what changed, why, and how to test it
4. Reference any related issues (`Closes #123`)

PRs are squash-merged through a merge queue, so keep the PR title in the
`type(scope): description` form — it becomes the commit on `main`.

PRs that touch engine code (`cuda_engine.py`, `cpu_engine.py`, `mlx_engine.py`, `base_engine.py`) should be tested on the relevant platform before merging.

---

## Good first issues

Look for issues tagged `good first issue` on GitHub. Some areas that are always welcome:

- **Documentation** — improve setup guides, add docstrings to undocumented methods
- **Tests** — the test suite has gaps, especially around engine selection and model download flows
- **Linux support** — the CPU and CUDA builds are produced by CI, but they get far less real-hardware testing than macOS and Windows
- **Error messages** — many engine errors surface as generic 500s; better user-facing messages are always useful
