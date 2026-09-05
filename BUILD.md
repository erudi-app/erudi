# Building Erudi for Distribution

Erudi ships as an Electron app with a PyInstaller-frozen Python backend bundled
inside it. Every distributable is produced the same way:

1. Freeze the backend with PyInstaller → `backend/dist/backend/`
2. Stage that bundle where electron-builder expects it → `frontend/backend/`
3. Package the Electron app with **electron-builder** → `frontend/dist/`

There is no electron-forge in this repository: `frontend/electron-builder.yml`
is the single packaging configuration, and it picks the frozen backend up from
`../backend/dist/backend` via `extraResources`.

PyInstaller cannot cross-compile, so **each platform is built on that platform**
(natively or on a matching CI runner).

---

## Prerequisites

| Tool | Version | Why |
|---|---|---|
| Python | **exactly 3.12** | `pgserver` only ships cp312 wheels; CI pins `3.12` |
| Node.js | **20** | matches the CI runners |
| npm | bundled with Node 20 | |
| PyInstaller | latest | installed into the backend venv |
| CUDA toolkit | any **12.x**; releases use **12.8** | only for the Windows/Linux CUDA legs, to compile `llama-server`. 12.8 is the first that emits native code for Blackwell (RTX 50); on an older 12.x the build scripts drop `120-real` and those cards JIT from PTX instead. Not needed for a CPU or macOS build — and never needed to *run* Erudi. |

Backend dependencies are not a single `requirements.txt`. Composed entrypoints
live under `backend/requirements/entrypoints/`:

```
backend/requirements/entrypoints/dev/{mac-silicon,win-cpu,win-cuda,linux-cpu,linux-cuda}.txt
backend/requirements/entrypoints/prod/{mac-silicon,win-cpu,win-cuda,linux-cpu,linux-cuda}-prod.txt
```

Read `backend/requirements/README.md` before touching any of them — common deps
live in `meta/base.txt` and platform/hardware specifics in `meta/*-specs.txt`;
the entrypoints only compose those.

On Windows and Linux the inference binary (`llama-server`) is compiled from the
`backend/forks/llama-cpp` submodule before PyInstaller runs, so clone with
`--recurse-submodules` (or run `git submodule update --init --recursive`).
macOS does not need it: inference there goes through MLX, which is a pip
dependency.

### PyInstaller specs

| Spec | Target |
|---|---|
| `backend/backend-mac-silicon.spec` | macOS Apple Silicon |
| `backend/backend.spec` | Windows CUDA |
| `backend/backend-cpu.spec` | Windows CPU **and** Linux CPU |

The Linux CUDA leg reuses `backend/backend.spec`.

---

## Local builds

### macOS (Apple Silicon)

```bash
bash scripts/build/build-mac-silicon.sh
```

The script verifies the backend venv and npm, installs PyInstaller if missing,
runs `backend-mac-silicon.spec` into `backend/dist/backend/`, copies the bundle
to `frontend/backend/`, then runs `npm run dist:mac`. The DMG lands in
`frontend/dist/`.

To sign and notarize, export the credentials before running it:

```bash
# .env.notarize
export APPLE_ID=...
export APPLE_ID_PASSWORD=...     # app-specific password
export APPLE_TEAM_ID=...
export APPLE_SIGNING_IDENTITY=...

source .env.notarize
bash scripts/build/build-mac-silicon.sh
```

Without `APPLE_SIGNING_IDENTITY` the script still builds, but warns that the app
is unsigned and macOS may block it on first launch. See `NOTARIZATION.md` for
how to obtain the certificate and the app-specific password.

### Windows (CUDA)

```powershell
.\scripts\build\build-win-cuda.ps1
```

Same shape: prerequisites, PyInstaller with `backend.spec`, copy to
`frontend\backend\`, then `npm run dist:win`. The NSIS installer lands in
`frontend\dist\`.

### Linux, and Windows CPU

There is no local build script for these two. Build them through CI (a `v*` tag,
see below) or run the manual sequence.

### Manual sequence (any platform)

```bash
# 1. Backend
cd backend
source venv/bin/activate                       # venv\Scripts\activate on Windows
pip install -r requirements/entrypoints/prod/<platform>-prod.txt
pip install pyinstaller
pyinstaller <spec> --distpath dist             # -> backend/dist/backend/

# 2. Frontend
cd ../frontend
npm ci
npm run dist:mac     # or dist:win / dist:linux
```

`electron-builder.yml` pulls the backend straight from `../backend/dist/backend`,
so the copy into `frontend/backend/` that the platform scripts do is a
convenience, not a requirement of the manual path.

---

## Build outputs

Everything electron-builder produces goes to `frontend/dist/` (`directories.output`):

| Platform | Artifacts |
|---|---|
| macOS | `.dmg` (first install) + `.zip` (electron-updater deltas), arm64 |
| Windows | NSIS `.exe`, x64 |
| Linux | `AppImage`, x64 |

macOS builds set `hardenedRuntime`, apply `assets/entitlements.mac.plist`,
`notarize: true`, and explicitly sign the bundled backend through
`mac.binaries` (`Contents/Resources/backend/backend`) — electron-builder does
not discover arbitrary Mach-O binaries under `Resources/` on its own.

---

## Development mode

Development does **not** need a frozen backend. Run the two processes side by
side; the Electron main process expects the backend to be up already and simply
health-checks it.

```bash
# Terminal 1 — backend
cd backend && source venv/bin/activate && python run.py --port 27182

# Terminal 2 — frontend
cd frontend && npm start
```

`bash scripts/dev/dev-start.sh` does both in two Terminal windows (macOS only)
after killing whatever holds `BACKEND_PORT`.

Keep `run.py` as the entrypoint rather than a bare `uvicorn` call: it supervises
uvicorn, scans ports `27182-27199`, and emits the newline-delimited JSON
lifecycle events (`starting`, `ready`, `shutdown`, `startup_error`) that the
Electron main process parses.

In production the main process spawns the packaged backend with only two extra
environment variables — `PYTHONUTF8=1` and `ERUDI_WATCH_STDIN=1`
(`frontend/src/main.js`). Everything else (data directories, database URL) is
resolved by the backend itself in `src/launcher/runtime_paths.py`.

---

## CI release pipeline

`.github/workflows/release.yml` runs on any `v*` tag with a five-leg matrix:

| Leg | Runner | Requirements | Spec | llama-server |
|---|---|---|---|---|
| `mac-arm` | `macos-14` | `mac-silicon-prod.txt` | `backend-mac-silicon.spec` | — (MLX) |
| `win-cpu` | `windows-2022` | `win-cpu-prod.txt` | `backend-cpu.spec` | cpu |
| `win-cuda` | `windows-2022` | `win-cuda-prod.txt` | `backend.spec` | cuda |
| `linux-cpu` | `ubuntu-22.04` | `linux-cpu-prod.txt` | `backend-cpu.spec` | cpu |
| `linux-cuda` | `ubuntu-22.04` | `linux-cuda-prod.txt` | `backend.spec` | cuda |

Each leg: checks out with submodules, sets up Python 3.12 and Node 20, compiles
`llama-server` from the submodule (Windows/Linux only), runs
`pyinstaller backend/<spec> --distpath backend/dist`, verifies the frozen tree,
then `npm ci`, `npm version <tag>` and `npm run release:<platform>`.

The verification step locates `llama-server` inside `backend/dist` and, on the CPU
legs, starts it with `--version` — which prints llama.cpp's build header and exits
0 from the argument parser, without loading a model. A filename alone would match
a truncated file or a wrong-architecture build; starting it proves the copy
PyInstaller made is a loadable executable whose libraries resolve. The CUDA legs
are not started: ggml links the NVIDIA driver library (`libcuda.so.1` /
`nvcuda.dll`, shipped with the driver rather than the toolkit), so the loader
fails before `main()` on a driverless runner. They are checked for the bundled
CUDA runtime DLLs instead, and real hardware in the QA pass is the first thing
that runs them.

The CUDA legs inject `-c.publish.channel=cuda` (plus a distinct artifact name)
so the GPU builds feed a separate `cuda` auto-update channel while the CPU
builds feed the default `latest` channel. macOS signing and notarization happen
on the `mac-arm` leg only.

### Draft, QA, promote

A tag publishes a **draft** GitHub Release (`releaseType: draft` in
`electron-builder.yml`). electron-updater's feed 404s while a release is
drafted, so no installed client is touched.

Promotion is a separate manual step: `.github/workflows/release-promote.yml`
(`workflow_dispatch`, takes the tag as input). It refuses to publish unless all
five platform legs of the tag's `release.yml` run concluded green, which
prevents a partial release with a missing `latest-*.yml` feed.

Between the two, run the manual QA pass on the real signed artifact:
`docs/dev/release-qa-checklist.md`.
