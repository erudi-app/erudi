# 🪵 Logging & Traceability

Erudi writes two log files. Together they let you follow a single user action
from the click in the UI down to the backend work it triggered.

## Log files

| File | Written by | Contents |
|------|------------|----------|
| `backend.log` | Backend (FastAPI process) | Every HTTP request (method, path, status, duration), model generation lifecycle, knowledge-base ingestion phases, RAG searches |
| `erudi-backend.log` | Electron main process | Backend stdout/stderr (launcher lifecycle events) and every UI interaction from the renderer — clicks, drops, pastes, committed input values — persisted via IPC |

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

## The Diagnostics panel

Neither log file has to be found by hand. **Settings → Diagnostics** — also
where the bug icon at the bottom of the left rail leads — shows the same
information the two files carry, without leaving the app:

- **Your setup**: the Erudi version, the operating system and architecture, the
  selected inference engine, the CPU and GPU (with VRAM and compute capability
  on NVIDIA), the model held in memory, the backend's Python version, the state
  of the embedded database, and the absolute path of both log files on this
  machine.
- **Recent errors**: the last 200 records at `WARNING` or above, merged from
  three sources and sorted on one timeline — `backend.log`,
  `erudi-backend.log`, and the errors this window caught itself. Each row shows
  its timestamp, level, source, request id when it has one, and how many times
  an identical error repeated.
- **Open log folder**, which reveals `backend.log` in the file manager.
- **Report this**: the whole summary as plain text, a *Copy* button, a button
  that opens this repository's bug report form with the version, operating
  system, hardware and model already filled in, and the contact page for
  reporters without a GitHub account.

Two properties of that list are deliberate.

**`INFO` records never appear.** They are the ones that carry conversation
content (see [privacy](privacy.md)), and this panel exists to be pasted into a
public issue. The filter is default-exclude in both readers: a record that does
not state a level of `WARNING` or above is dropped, along with its continuation
lines. In `erudi-backend.log` that also excludes the Electron main process's own
unlevelled lines.

**The panel sends nothing.** It reads `GET /erudi/diagnostics/` over loopback,
reads the app log through the Electron preload bridge, and reads this window's
error buffer from memory. What reaches a bug tracker is what you copy and paste
there yourself. Warnings and errors can still quote a filename or a query, so
read the text before you post it.

The panel also survives the case it is most needed in. If the backend does not
answer, the backend half of the report is reported as missing rather than
silently left empty, and the app's own version, platform, log path and errors
are still shown — which is usually enough to describe a backend that will not
start.

## Uncaught errors in the app window

An exception that escapes a React render, an uncaught `window.onerror`, and an
unhandled promise rejection are all recorded: they go to `erudi-backend.log`
under the namespace `renderer:uncaught` and into the session buffer the
Diagnostics panel reads. Identical errors are counted rather than logged again,
so a render loop costs one line and a repeat count instead of filling the file.
A render that throws replaces the screen with a recoverable page carrying the
error, a *Reload* button and the same report block.

## Tracing a bug (QA recipe)

1. Reproduce the problem and note the time (remember: logs are in UTC).
2. Grab both files from the locations above.
3. In `erudi-backend.log`, find the UI event at that time and copy its `fe-…` request id. *Settings → Diagnostics → Open log folder* takes you there.
4. Grep `backend.log` for that id — every backend line for that action carries it:

```bash
grep "fe-abc123" backend.log
```

> ⚠️ **Privacy** — logs include conversation and message content as well as
> document names. Review and redact them before sharing publicly, for example
> when attaching them to a GitHub issue.
