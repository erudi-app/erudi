# 🪵 Logging & Traceability

Erudi writes two log files. Together they let you follow a single user action
from the click in the UI down to the backend work it triggered.

## Log files

| File | Written by | Contents |
|------|------------|----------|
| `backend.log` | Backend (FastAPI process) | Every HTTP request (method, path, status, duration), model generation lifecycle, knowledge-base ingestion phases, RAG searches |
| `erudi-backend.log` | Electron main process | Backend stdout/stderr (launcher lifecycle events), every UI interaction from the renderer — clicks, drops, pastes, committed input values — and every uncaught renderer error, persisted via IPC |

### Where to find them

| File | Development | Packaged app |
|------|-------------|--------------|
| `backend.log` | `backend/logs/backend.log` | macOS: `~/Library/Logs/erudi/backend.log` · Windows: `%LOCALAPPDATA%\erudi\logs\backend.log` · Linux: `${XDG_STATE_HOME:-~/.local/state}/erudi/logs/backend.log` |
| `erudi-backend.log` | OS temp directory (same as packaged) | macOS: `$TMPDIR/erudi-backend.log` (run `echo $TMPDIR` in a terminal to resolve the folder) · Windows: `%TEMP%\erudi-backend.log` · Linux: `/tmp/erudi-backend.log` |

### Rotation

- `backend.log` rotates at 10 MB and keeps up to 10 previous files (`backend.log.1` … `backend.log.10`).
- `erudi-backend.log` rotates at 10 MB and keeps one previous file, `erudi-backend.old.log`.

All timestamps in both files are UTC, ISO-8601, with milliseconds — so lines
from the two files can be correlated reliably.

## Request-id correlation

Every user interaction in the renderer gets a request id of the form `fe-…`.
The frontend sends it as the `X-Request-ID` header on the resulting API call,
the backend injects the same id into every log line produced while handling
that request, and echoes it back in the response.

One user click therefore leaves this trail:

1. `erudi-backend.log` — the UI event (click, drop, paste, input) with its `fe-…` id.
2. `backend.log` — the HTTP request line and all backend work it triggered (generation, ingestion, RAG search), each line tagged with the same `fe-…` id.

## Log level

The default level is `INFO` everywhere. Set the `ERUDI_LOG_LEVEL` environment
variable to change it — for example `DEBUG` when investigating an issue:

```bash
# Development (backend)
cd backend && ERUDI_LOG_LEVEL=DEBUG python run.py

# Packaged app: set the variable in the environment before launching, e.g. on macOS
ERUDI_LOG_LEVEL=DEBUG open -a Erudi
```

## The Diagnostics page

Neither log file has to be found by hand. The **Diagnostics** page — where the
bug icon at the bottom of the left rail leads — shows the same information the
two files carry, without leaving the app:

- **Your setup**: the Erudi version, the operating system and architecture, the
  selected inference engine, the CPU and GPU (with VRAM and compute capability
  on NVIDIA), the model held in memory, the backend's Python version, and the
  state of the embedded database. The absolute path of both log files is not
  listed on screen — **Open log folder** is how you get to them — but it still
  travels in the copied report below, for whoever triages the issue.
- **Recent errors**: the last 200 records at `WARNING` or above, merged from
  three sources and sorted on one timeline — `backend.log`,
  `erudi-backend.log`, and the errors this window caught itself. Each row shows
  its timestamp, level, source, request id when it has one, and how many times
  an identical error repeated. When nothing was recorded, the area says so in
  one line and asks nothing.
- **Open log folder**, which reveals `backend.log` in the file manager.
- **Report a problem**: a button that opens this repository's bug report form
  with the version, operating system, hardware and model already filled in,
  and the contact page for reporters without a GitHub account — both there in
  either state. When there is at least one recent error, a *Copy the full
  report* button also appears; it copies the setup summary above and the error
  list, including the log paths, as plain text to paste into the form's
  **Logs** field. When nothing was recorded there is nothing to copy, so the
  button is not offered.

Two properties of that list are deliberate.

**`INFO` records never appear.** They are the ones that carry conversation
content (see [privacy](privacy.md)), and this page exists to be pasted into a
public issue. The filter is default-exclude in both readers: a record that does
not state a level of `WARNING` or above is dropped, along with its continuation
lines. In `erudi-backend.log` the levelled lines are the renderer's
(`[renderer:<ns>] WARN|ERROR …`) and the main process's own failures
(`[main] WARN|ERROR …`: a backend that exited, a spawn that failed, a renderer
that crashed); main's lifecycle chatter carries no level and stays out.

**Each error appears once.** The same failure is legitimately written by more
than one writer, and the merge removes the overlap rather than the writers: an
uncaught renderer error is shown from the app log (the record that survives a
reload), with the session buffer contributing only its repeat count; the app
log's echo of the backend's stdout is dropped when the backend answered, since
`backend.log` holds the same record with its traceback, and kept when it did
not, since it is then the backend's last words.

**The page sends nothing.** It reads `GET /erudi/diagnostics/` over loopback,
reads the app log through the Electron preload bridge, and reads this window's
error buffer from memory. What reaches a bug tracker is what you copy and paste
there yourself. Warnings and errors can still quote a filename or a query, so
the report block says so next to the text.

The page also survives the case it is most needed in. If the backend does not
answer, the backend half of the report is reported as missing rather than
silently left empty, and the app's own version, platform, log path and errors
are still shown — which is usually enough to describe a backend that will not
start.

## The inference child's own log

Inference runs in a child process, and what that child prints is the only
account of a model that would not load or a server that died mid-answer.

`llama-server` (CPU and NVIDIA) is a subprocess whose merged output the backend
drains as it comes: nothing of it is written to a file of its own, and its last
lines travel inside the backend's records.

`mlx_vlm.server` (Apple Silicon) is a *process*, not a subprocess — the backend
has no pipe to it — so the child redirects its own standard output and error,
including everything MLX and Metal write from native code, into a file beside
`backend.log`:

| File | Where |
|------|-------|
| `mlx-child-<port>.log` | Development: `backend/logs/` · Packaged app: next to `backend.log` (the table at the top of this page) |

- One file per spawn, named after the port that child serves. The previous
  spawn's file is kept as `mlx-child-<port>.log.1`; older ones are removed, so
  a port never holds more than two.
- The live file rolls the same way once it passes 2 MB, so a talkative server
  cannot fill the disk.
- Stopping a model — switching to another one, the idle reap, quitting — deletes
  both files. A child that died on its own keeps them: that output is the whole
  account of the death.

The backend quotes the tail of that file in the record it writes when the child
crashes, fails its readiness probe, or is found dead by a later request — so
the child's last words appear on the **Diagnostics** page and in a copied
report, without anyone having to find the file.

## Uncaught errors in the app window

An exception that escapes a React render, an uncaught `window.onerror`, and an
unhandled promise rejection are all recorded: they go to `erudi-backend.log`
under the namespace `renderer:uncaught` and into the session buffer the
Diagnostics page reads (which shows the file record once, with the buffer's
repeat count). Identical errors are counted rather than logged again, so a
render loop costs one line and a repeat count instead of filling the file.
A render that throws replaces the screen with a recoverable page carrying the
error, a *Reload* button and the same report block.

## Logging rules

The Diagnostics page is only as truthful as the records underneath it. Every
place where something can go wrong — an `except` block, a `.catch(...)`, a
child-process exit, a timeout, a retry, a fallback, a global handler — follows
these rules, in both processes.

### Level

- **`ERROR`** — the operation the user or the app needed did not happen: a
  request ended in 5xx, an inference child died or never became ready, a
  download or an ingestion failed, the database or a migration failed, a
  background task died, the backend or the renderer crashed. The record
  always carries the exception: `logger.error(..., exc_info=True)` or
  `logger.exception(...)` in Python, the error object (its stack) in JS.
- **`WARNING`** — the app recovered or degraded on its own, and a maintainer
  would want to know: a retry that succeeded after failures, a fallback that
  was taken (the other `llama-server` flavour, the processor because NVML sees
  no GPU, a default title, an answer without its knowledge-base excerpts), a
  file skipped, an invalid value replaced by a default, a child that printed
  something alarming but kept running, a probe that timed out once.
- **`INFO`** — the expected outcomes of user actions and the normal
  lifecycle. A 4xx that the client asked for — a model deleted a moment ago
  (404), a guarded delete (409), a rejected input (422) — is one of them: it is
  not a defect, and it must not fill a page meant for bug reports. The backend
  logs it at `INFO` in the request's record, and the API client logs the same
  status at `info` on its side.
- Nothing that indicates a defect is logged at `INFO`/`DEBUG` or not at all.

### One record per failure

A failed request has exactly one record, written where the exception is
handled, never where it is raised: the handler for `AppBaseException`
(`backend/src/core/exceptions.py`) writes the method, path, status, Erudi code,
message and the trace the raiser attached, at the level of the status; the
fallback handler for anything else writes the traceback at `ERROR`, for a crash
inside a streaming body too. Constructing an exception logs nothing, so an
exception that is caught and recovered from leaves no record, and a repository
or an endpoint does not log before raising. An `except` that recovers logs its
own outcome, at the level of what it did.

A background task nobody awaits (`BackgroundTasks`, `asyncio.create_task`, a
thread) writes its own record at its boundary, with the traceback: that
record is the only trace of its death. A task created with `create_task` has a
done callback for the same reason.

### Content

A record says what failed, on what, and why: the model id, path, port, request
id, exit code or HTTP status, and the exception with its traceback when it was
unexpected. Huge payloads are truncated. Secrets never reach a line: the
`HF_TOKEN`, the `--api-key` minted for `llama-server`, database URLs with a
password, `Authorization` headers; argv, headers and environment blocks are
not logged.

### Silence

No `except Exception: pass`, `except: return None`, `.catch(() => {})` or empty
`catch {}` without a comment saying why the failure is genuinely irrelevant. A
silent branch that would hide a real failure gets a record at the level above.

### Process boundaries

A parent logs the failures of its children. The backend logs an inference
child that exits with its pid, port, exit code and the tail of its output
(`llama-server` is drained by `ChildOutputDrainer`; `mlx_vlm.server` captures
itself — see [The inference child's own log](#the-inference-childs-own-log)).
The Electron main process logs the
backend's exit (`ERROR` unless main asked it to stop), a spawn that failed,
every `startup_error` it receives, and what no `catch` sees: an uncaught
exception or unhandled rejection in the main process, a renderer or Chromium
child process that is gone, a window that stops responding. The backend
launcher (`backend/run.py`) writes every `startup_error` it emits to
`backend.log` with its traceback, and the lifespan logs a failed startup
before it propagates, because uvicorn reports it on stderr only.

## Tracing a bug (QA recipe)

1. Reproduce the problem and note the time (remember: logs are in UTC).
2. Grab both files from the locations above.
3. In `erudi-backend.log`, find the UI event at that time and copy its `fe-…` request id. *Diagnostics → Open log folder* takes you there.
4. Grep `backend.log` for that id — every backend line for that action carries it:

```bash
grep "fe-abc123" backend.log
```

> ⚠️ **Privacy** — logs include conversation and message content as well as
> document names. Review and redact them before sharing publicly, for example
> when attaching them to a GitHub issue.
