---
title: Backend Launcher
description: How the unified Erudi backend launcher works in development and production, and how it picks its port.
---

# Backend Launcher

`backend/run.py` is a single entrypoint for launching the Erudi FastAPI backend on every
build target: macOS (Apple Silicon), Windows (CUDA/CPU), and Linux (CUDA/CPU). It is also
what you run locally in development. It emits newline-delimited JSON events on stdout so
the Electron shell can track backend state.

It is not a thin wrapper around uvicorn. If you touch the file, preserve the stdout
protocol below.

## Responsibilities

- Configure third-party libraries for deterministic, low-noise startup (Torch,
  tokenizers, MKL).
- Force the Windows selector event-loop policy for library compatibility.
- Normalize stdout/stderr to line-buffered mode so lifecycle events flush immediately.
- Detect PyInstaller bundles and relocate runtime data and logs to OS-appropriate,
  user-writable folders.
- Preserve the macOS `~/Library/Application Support/erudi/backend/prod` symlink behaviour.
- Initialize multiprocessing with the `spawn` strategy before importing heavy modules
  (Torch, FastAPI) — the MLX engine depends on it.
- Launch uvicorn on `127.0.0.1:27182` (scanning 27182-27199 if busy) in a background
  thread, monitor readiness, and emit structured error events for crashes, timeouts, or
  port conflicts.
- Shut down cleanly when the parent process dies, bounded so a quit is a quit.

## Ports

Erudi's canonical port is **27182** — the leading digits of Euler's number *e* (2.7182…),
a wink for an app built for erudites. It is also a safe default on every OS: unassigned
by IANA, below every ephemeral range (Linux 32768-60999, Windows/macOS 49152-65535, plus
the Windows Hyper-V/WSL exclusions inside that range), and clear of the crowded local LLM
defaults (Ollama 11434, LM Studio 1234, vLLM 8000, llama.cpp and Tomcat 8080).

The whole Erudi footprint sits in the 271xx-273xx block:

| Range | Owner |
|---|---|
| 27182-27199 | FastAPI backend (`CANONICAL_PORT` + `PORT_SCAN_COUNT = 18` in `backend/run.py`) |
| 27200-27299 | `llama-server` child process (CPU and CUDA engines) |
| 27300-27399 | `mlx_vlm.server` child process (MLX engine) |

The backend prefers 27182 and scans forward to 27199 on collision, stopping short of
27200 so it can never take a port from an inference pool. The resolved port is announced
in the `starting` and `ready` events, and the renderer uses the announced value — it
never assumes 27182.

Override the port on the command line:

```bash
cd backend && source venv/bin/activate
python run.py --port 8000
```

### `BACKEND_PORT` is development-only

The backend itself never reads `BACKEND_PORT`; it takes `--port`. The variable is read in
two places only:

- `scripts/dev/dev-start.sh`, which uses it to kill whatever is on the port and to launch
  both processes;
- `frontend/src/main.js`, **in development mode only**, where the Electron main process
  assumes the backend is already running and health-checks
  `http://127.0.0.1:${BACKEND_PORT}/erudi/health/` (default 27182).

In a packaged build Electron spawns the backend itself and captures the real port from
the JSON events, so `BACKEND_PORT` has no effect there.

```bash
# Run both on a non-default port in development
BACKEND_PORT=8000 bash scripts/dev/dev-start.sh
```

`BACKEND_PORT` is documented as dev-only in
[`backend/.env.example`](https://github.com/erudi-app/erudi/blob/main/backend/.env.example).

### Checking which port answers

```bash
curl http://127.0.0.1:27182/erudi/health/
# {"status":"ok","message":"Backend is running","db":"ok"}
```

The trailing slash is required. If the renderer reports
`ERR_CONNECTION_REFUSED` on `/erudi/health/`, read the backend's `ready` event to see the
port it actually bound, and align `BACKEND_PORT` with it.

## Data and log locations

| Mode | Data directory | Log directory |
|------|----------------|---------------|
| Development | `backend/data` | `backend/logs` |
| macOS bundle | `~/Library/Application Support/erudi/backend/prod/data` | `~/Library/Logs/erudi` |
| Windows bundle | `%LOCALAPPDATA%\erudi\backend\prod\data` | `%LOCALAPPDATA%\erudi\logs` |
| Linux bundle | `${XDG_DATA_HOME:-~/.local/share}/erudi/backend/prod/data` | `${XDG_STATE_HOME:-~/.local/state}/erudi/logs` |

`backend/run.py` initializes `src.launcher.runtime_paths` with these directories before
importing the app, so `src/core/config.py` and `src/core/logging.py` adopt them on
import. `ERUDI_DATA_ROOT` overrides the root.

The backend writes `backend.log` inside the log directory (rotating, 10 MB, 10 backups).
See [Logging & Traceability](../logging.md) for the Electron-side log and request-id
correlation.

## Lifecycle events

Every event is one JSON object per line on stdout, with a `ts` field (UTC ISO-8601 with
milliseconds and a `Z` suffix) so it can be correlated with the Electron and backend
logs. The reader must tolerate interleaved non-JSON log lines.

```json
{"event":"starting","arch":"...","mode":"...","data_path":"...","ts":"..."}
{"event":"phase","phase":"preparing_database","ts":"..."}
{"event":"ready","port":27182,"ts":"..."}
{"event":"shutdown","ts":"..."}
{"event":"startup_error","code":"PORT_IN_USE","message":"...","ts":"..."}
{"event":"engine_notice","code":"CUDA_DRIVER_TOO_OLD","gpu_name":"...","ts":"..."}
```

### Startup phases

`phase` events report startup progress so the loader can say what is happening. They are
emitted from inside the FastAPI lifespan through `emit_phase`
(`backend/src/launcher/events.py`), which `run.py` injects on `app.state`; under plain
uvicorn they are a no-op.

| Phase | Emitted before |
|---|---|
| `preparing_database` | starting the embedded PostgreSQL cluster (first run pays a one-time `initdb`) |
| `recovering_database` | waiting for a PostgreSQL cluster to finish WAL crash recovery after an unclean shutdown |
| `running_migrations` | the Alembic upgrade to head |
| `loading_catalog` | seeding and reconciling the model catalog from the bundled snapshot |

Phases are informational. Readiness is the `ready` event or a confirming health check —
never a phase.

### Engine notice

`engine_notice` is emitted from the lifespan, after the migrations and the catalog, when
the selected engine is `CUDA_Engine` and the pre-flight finds a GPU or a driver below what
the bundled CUDA build needs. It is **not** a startup failure — the backend is healthy and
about to serve — so the frontend holds it until the app is past `ready` and shows it as a
decision, not as an error screen.

```json
{
  "event": "engine_notice",
  "code": "CUDA_DRIVER_TOO_OLD",
  "gpu_name": "NVIDIA GeForce GTX 1080",
  "compute_capability": "6.1",
  "driver_cuda_version": "12.1",
  "required_cuda_version": "12.8",
  "raw": "...",
  "ts": "..."
}
```

| Code | Meaning |
|---|---|
| `CUDA_COMPUTE_CAPABILITY_TOO_LOW` | the card is below compute capability 5.0; no driver fixes it |
| `CUDA_DRIVER_TOO_OLD` | the driver's CUDA version is below what this card needs to run the bundled build |

At most one notice per boot, and none at all when a reading is unavailable. The floors and
where each comes from are in `backend/src/engines/cuda_compatibility.py`; the
per-generation table is in [Hardware Detection](hardware.md).

### Error codes

`startup_error` carries one of:

| Code | Meaning |
|---|---|
| `PORT_IN_USE` | the requested port is taken |
| `NO_PORT_AVAILABLE` | every port in the 27182-27199 scan is busy (transient; the frontend retries) |
| `CRASH_BEFORE_READY` | the server thread exited before binding a port |
| `PORT_TIMEOUT` | the server did not bind within the startup window |
| `IMPORT_ERROR` | the FastAPI application failed to import |
| `DATA_PREP_ERROR` | data directories could not be prepared |
| `UNEXPECTED_ERROR` | unhandled exception in the server thread |
| `POLLING_ERROR` | unhandled exception in the startup polling loop |

A non-zero process exit code also signals a startup failure.

## Shutdown and orphan prevention

A backend that outlives its parent is a stranded process holding a port, a model, and a
database cluster. Two environment variables control the guards, both documented in
`backend/.env.example`:

| Variable | Effect |
|---|---|
| `ERUDI_WATCH_STDIN` | Set to `1` by the Electron main process: the launcher exits on stdin EOF, which is what a dead parent produces. |
| `ERUDI_NO_PARENT_WATCHDOG` | Set to `1` to disable the parent-process watchdog for a deliberately detached run. |

The parent watchdog otherwise polls every couple of seconds. Whichever path triggers the
stop (watchdog, SIGTERM relay, stdin EOF), uvicorn's wait for in-flight requests is
bounded, then the lifespan shutdown runs: the inference child is terminated and the
embedded PostgreSQL cluster is stopped.

## Usage

- **Development**: `cd backend && source venv/bin/activate && python run.py --port 27182`
- **Bundled build**: PyInstaller executes the same module; no platform-specific launcher
  is required.
- The frontend reads stdout line by line and acts on the JSON events.
