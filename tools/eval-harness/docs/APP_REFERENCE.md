# Erudi packaged app — facts the harness relies on

Verified against the source at upstream `main` c8ca417 (2026-09-15), release line 1.1.x. File references are relative to the repository root. If a fact here disagrees with the build under test, the build wins: record the discrepancy in the run report.

## Install, data and log locations

| | macOS (Apple Silicon) | Windows | Linux |
|---|---|---|---|
| Installed app | `/Applications/Erudi.app` (main binary `Contents/MacOS/Erudi`) | `%LOCALAPPDATA%\Programs\Erudi\Erudi.exe` (per-user NSIS default; user may change it) | wherever the AppImage is |
| Resources dir (`process.resourcesPath`) | `Erudi.app/Contents/Resources/` | `<install>\resources\` | `<mount>/resources/` |
| Backend executable | `Resources/backend/backend` | `resources\backend\backend.exe` | `resources/backend/backend` |
| Bundled llama-server | n/a (MLX) | `resources\backend\_internal\artifacts\llama-cpp\<cpu\|cuda>\bin\llama-server.exe` (verify on the build) | same layout |
| Backend data root | `~/Library/Application Support/erudi/backend/prod/` | `%LOCALAPPDATA%\erudi\backend\prod\` | `$XDG_DATA_HOME/erudi/backend/prod/` |
| ├ `data/models/` | downloaded models (MLX dirs and GGUF files) | same | same |
| ├ `data/models_cache/` | KB embedding model (`intfloat/multilingual-e5-small`, HF cache layout) | same | same |
| ├ `data/postgres/` | embedded Postgres cluster (`log` file inside = postmaster log) | same | same |
| └ `db-backups/` | pre-migration `pg_dump` files (`erudi-<label>.dump`, 3 kept) | same | same |
| Backend log dir | `~/Library/Logs/erudi/` (`backend.log`, `mlx-child-<port>.log`) | `%LOCALAPPDATA%\erudi\logs\` | `$XDG_STATE_HOME/erudi/logs/` |
| Electron stdout capture (lifecycle events) | `$TMPDIR/erudi-backend.log` | `%TEMP%\erudi-backend.log` | `/tmp/erudi-backend.log` |
| Electron logs / profile | `~/Library/Logs/Erudi/`, `~/Library/Application Support/Erudi/` (same dir as data on case-insensitive APFS) | `%APPDATA%\Erudi\logs\`, `%APPDATA%\Erudi\` | `~/.config/Erudi/logs/`, `~/.config/Erudi/` |

Logs survive reinstalls: always filter by the run's start timestamp.

## Launch, readiness, quit

- Launch with CDP: macOS `open -a Erudi --args --remote-debugging-port=9222` (or exec the main binary directly to get its PID); Windows `Start-Process "$env:LOCALAPPDATA\Programs\Erudi\Erudi.exe" -ArgumentList "--remote-debugging-port=9222"`; Linux `Erudi-<v>.AppImage --remote-debugging-port=9222`. No fuse blocks the Chromium debugging port (`frontend/electron-builder.yml` `electronFuses`).
- Ready when `http://127.0.0.1:9222/json/version` answers and `GET http://127.0.0.1:27182/erudi/health/` returns 200 (the bare `/erudi/health` path redirects 307).
- Graceful quit: macOS `osascript -e 'quit app "Erudi"'`; Windows `taskkill /IM Erudi.exe` (without `/F`); Linux `pkill -TERM -f Erudi`. On macOS closing the window does not quit.

## Process tree

| Role | How to recognise it from outside |
|---|---|
| Electron main | `Erudi.app/Contents/MacOS/Erudi` / `Erudi.exe` / AppImage binary |
| Electron helpers | macOS `Erudi Helper (Renderer)`, `Erudi Helper (GPU)`, `Erudi Helper` (utility, `--type=utility`); Windows/Linux: same executable with `--type=renderer\|gpu-process\|utility` in the cmdline. Children of the main process. |
| Backend | executable named `backend` / `backend.exe` under the resources dir, cmdline `--port 27182`; child of Electron main (POSIX: own process group) |
| Embedded Postgres | executable path containing `pginstall/bin/postgres` (`pginstall\bin\postgres.exe`); one postmaster + several backend/background processes. **On POSIX the postmaster is re-parented away from the backend** (started via `pg_ctl`), so find it by path, not by parentage. On Windows it is a child of the backend (Job Object). |
| Inference child (Windows/Linux) | `llama-server[.exe]`, child of the backend, `--port 272xx`, `--alias erudi-<llm_id>` identifies the model |
| Inference child (macOS) | a `multiprocessing` spawn of the frozen `backend` executable: same executable as the backend, **child of the backend PID**, listening on a port in 27300–27399; its OS cmdline shows multiprocessing bootstrap args, not `mlx_vlm.server` |
| Transient | `pg_ctl`, `initdb`, `pg_dump` (next to `postgres`) |
| KB embedder | **in-process inside the backend** (sentence-transformers + torch), no separate process; never unloaded |

The inference child requires a per-spawn API key held only by the backend: always drive inference through the Erudi API.

## HTTP API (127.0.0.1:27182, prefix `/erudi`, no auth, Host must be 127.0.0.1 or localhost)

Every response carries `X-Request-ID`; send your own `X-Request-ID: eval-<run>-<n>` to find the harness's calls in `backend.log`.

- `GET /health/` → `{"status":"ok","message":...,"db":"ok"|"recovering"|"failed"}`
- `GET /hardware/app_startup` → `{backend_type, global_inference_score, global_inference_label, raw_inference_score, recommended_param_min, recommended_param_max}`
- `GET /hardware/detailed` → `{hardware:{backend_type, cpu_model, total_memory_gb, available_memory_gb, disk_*, ... MLX: mlx_chip_model, mlx_gpu_cores, unified_memory ... CUDA: gpu_name, vram_total_gb, vram_available_gb, compute_capability, cuda_version ...}, performance_breakdown, boosted_inference_score}`
- `GET /user_settings/`, `PUT /user_settings/` body any of `{web_search_enabled, language, auto_update_enabled, inference_backend}` (at least one field). Set `auto_update_enabled=false` at the start of a run and restore it at the end.
- `GET /diagnostics/?limit=N` → `{environment:{platform, engine, loaded_model_id, loaded_model, gpu_name, vram_total_gb, db, backend_log_path, frozen, ...}, recent_errors:[{timestamp, level, request_id, message}]}` — `loaded_model_id` tells whether a model is resident.
- `GET /llms/local`, `GET /llms/remote`, `GET /llms/search?name=<q>`, `GET /llms/{id}`. `LLMResponse`: `id, name, local (0 catalog, 1 downloaded, 2 downloading), link, param_size, supports_tools, is_attached_to_kb, kb_id, artifact_size_bytes, context_window, allocated_context_window, weights_available, supports_vision, ...`
- `POST /llms/{id}/download` → `DownloadJobResponse {id (job id), remote_model_id, local_model_id, status: pending|running|completed|failed|cancelled, total_bytes, progress 0-100, error_message, ...}`; poll `GET /llms/downloads/{job_id}/status`; `POST /llms/downloads/{job_id}/cancel`.
- `DELETE /llms/{id}` (409 with dependents unless `?orphan_dependents=true`). Deleting the resident model unloads it first. **No endpoint forces an unload otherwise.**
- `POST /conversations/` body `{llm_id (required), temperature?, top_p?, custom_prompt?, web_search_enabled?}` → 201 `{id, llm_id, ...}`. A KB is used by passing the KB assistant's `llm_id`.
- `POST /conversations/{id}/query` body `{question, images?: [data-URL], attachments?: [local paths], temperature?, top_p?, custom_prompt?}` → NDJSON, one object per line with `t` ∈ `answer{text}`, `thinking{text}`, `tool_call{name,args}`, `tool_result{name,text}`, `memory_warning{used_fraction, conversation_bytes?, footprint_bytes?}`, `error{text, code?, raw?}`, `done`.
- `POST /conversations/{id}/generate_title` (same body) → text/plain stream.
- `GET /conversations/{id}`, `GET /conversations/{id}/fetch_messages`, `DELETE /conversations/{id}`.
- `POST /arena/{llm_id}/query` body `{question, images?, attachments?, ...}` → text/plain stream (stateless).
- `GET /knowledge_base/embedding-model/status`, `POST /knowledge_base/embedding-model/download` → `{available, downloading, error}`.
- `POST /knowledge_base/create` JSON `{paths: [absolute local paths], selectedModel: <base llm id>, modelName: <assistant name>, description?}` → `{msg, model_id}` (the new assistant's llm id); poll `GET /knowledge_base/{model_id}/status` → `{status: pending|running|completed|failed, status_updated_at, error_message}`. Same `selectedModel` with an existing KB = update. Duplicate `modelName` → 409.

## Lifecycle events (boot timeline)

Electron appends every backend stdout line to the stdout capture log prefixed `Backend stdout: `. JSON events carry `ts` (UTC ISO-8601 ms, `Z`):
`starting{arch, mode, data_path, port, first_run}` → `phase{preparing_database}` → `phase{recovering_database}` (only after an unclean shutdown) → `phase{running_migrations}` → `phase{loading_catalog}` → `ready{port}`; also `shutdown`, `startup_error{code, message}`, `engine_notice{code, ...}`.

## Timing constants

- Idle unload: threshold 300 s, check tick 300 s → the inference child exits 300–600 s after the last use (`backend/src/engines/base_engine.py`).
- Backend startup budgets: 120 s (300 s on first run); Electron outer cap 330 s.
