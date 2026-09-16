# erudi-eval — specification

## Purpose

Measure, on the **installed packaged app**, how much memory and storage each layer of the Erudi stack uses, at every step a user goes through, reproducibly, on macOS, Windows (CPU and CUDA) and Linux. The numbers are the baseline for the rewrite (#553): every future change is compared against them.

The harness **observes and drives**; it never modifies the app, and it never deletes user data it did not create.

## Non-goals

- Functional QA (that is the repository's automated-qa skill). The harness checks only what it needs to trust its numbers (right model bound, turn actually ran, KB actually searched).
- Answer quality evaluation.

## Runtime and packaging

- Python ≥ 3.11, run with `uv run erudi_eval.py ...` (PEP 723 inline dependencies) so a fresh machine needs only `uv`. Dependencies kept minimal: `psutil`, `websocket-client` (CDP). Optional: `nvidia-ml-py` (NVML) when an NVIDIA GPU is present, else fall back to `nvidia-smi`.
- Pure stdlib HTTP (`urllib`/`http.client`) for the Erudi API, including NDJSON streaming.
- Layout: `erudi_eval.py` (CLI entry) + package `erudi_eval/` + `tests/` + `corpus/` + `docs/` + `README.md`.
- Tests: `uv run --with pytest pytest tests/` must pass on macOS without the app installed; platform-specific tests skip cleanly elsewhere.

## Layers ("categories") every sample is attributed to

| Category | Processes |
|---|---|
| `electron_main` | Electron main process |
| `electron_renderer` | renderer helper(s) |
| `electron_gpu` | GPU helper |
| `electron_utility` | utility/network/other helpers |
| `backend` | Python backend (includes the in-process KB embedder today) |
| `database` | embedded Postgres: postmaster + all its processes |
| `inference` | llama-server or MLX child |
| `embedding` | reserved for a future dedicated embedding server (#594); empty today |
| `transient` | pg_ctl, initdb, pg_dump, anything else spawned by the tree |

Derived totals per sample: `app_total` (all categories), `app_overhead` (= `app_total` − `inference` − `embedding`), `inference_total`.

Discovery must be re-run every sample (processes come and go), from: the Electron main PID and its descendants, plus path-based matching for re-parented Postgres (`pginstall/bin/postgres` under the Erudi data dir) and a safety scan for any process whose executable lives inside the install/resources dir. Classification rules live in one pure function with unit tests over recorded cmdline fixtures for all three OSes (macOS MLX child = child of backend PID running the backend executable; Windows/Linux = `llama-server`).

## Metrics

### Per process, per sample (interval default 1 s)

| Metric | macOS | Windows | Linux |
|---|---|---|---|
| **primary** (what the OS memory UI shows) | `phys_footprint` via `proc_pid_rusage(RUSAGE_INFO_V4)` (ctypes) | private working set (`psutil` `uss`) | PSS (`psutil` `pss`) |
| peak | `ri_lifetime_max_phys_footprint` | `peak_wset` | track max of samples |
| also | RSS, USS | RSS, private bytes (commit, `private`), pagefile | RSS, USS, swap |
| CPU % and threads | psutil | psutil | psutil |
| GPU memory | included in `phys_footprint` (unified memory); system wired/compressed come from the machine context below | NVML/`nvidia-smi` per-process used GPU memory | same as Windows |

Record which metric names were available; never silently substitute one metric for another. If a process cannot be read (permissions, exited mid-read), record the error, do not drop the sample.

### Per sample, machine context

The app's numbers are read in the context of the machine, so every sample also records the machine. Reading it stays cheap: no subprocess per sample on macOS.

- **macOS**: `host_statistics64(HOST_VM_INFO64)` through ctypes (page size from `hw.pagesize`; `vm_stat` parsing only as a fallback): free, active, inactive, speculative, wired, purgeable, `compressor_page_count`, `total_uncompressed_pages_in_compressor`, internal and external pages. Derived as Activity Monitor does: app memory (internal − purgeable), wired, compressed, cached files (external + purgeable), memory used (app + wired + compressed). Cumulative pageins, pageouts, swapins, swapouts, compressions, decompressions and faults become per-second rates between consecutive samples. Swap total/used/free from `vm.swapusage`; `kern.memorystatus_vm_pressure_level`. `iogpu.wired_limit_mb` (0 = default) once per run in `system.json`.
- **All OSes**: total/available memory, system CPU % (non-blocking), load average where available, process count, disk read/write bytes per second, free space on the data-root volume, GPU total/used VRAM and utilisation (NVIDIA).
- **Windows**: psutil virtual/swap memory; `GetPerformanceInfo` commit charge and limit, SystemCache, KernelPaged, KernelNonpaged, handle/process/thread counts. Hard page faults/s are recorded as unavailable (they need a PDH counter).
- **Linux**: `/proc/meminfo` (MemAvailable, Cached, SwapCached, AnonPages, Shmem, SwapTotal/SwapFree), `/proc/vmstat` rates (pswpin, pswpout, pgmajfault), PSI memory `some` and `full`.
- **Every 10 s** (subprocess allowed): macOS `pmset -g therm` (thermal/performance warning level), `pmset -g batt` (power source), `pmset -g` (low power mode). Windows/Linux: power source from psutil's battery sensor; thermal state unavailable.
- **Every 5 s**: the 10 largest processes that are neither Erudi nor the harness, by RSS (the metric is named), and the total RSS of all such processes.
- **Harness self-cost**: the harness process's primary metric and CPU %, in a separate `harness` block, never in app totals; `sample_cost_ms` (discovery + metric reads + machine context) and the target and effective sampling interval.

A counter that goes backwards (reset, wrap) gives no rate, never a negative one. Unreadable fields are recorded as errors, never substituted.

### Renderer internals (CDP, per phase and every 5 s while a phase runs)

`Performance.getMetrics` (JSHeapUsedSize, JSHeapTotalSize, Nodes, Documents, JSEventListeners, LayoutCount, RecalcStyleCount) and `Runtime.getHeapUsage` on the app page. DOM node count is the proxy for non-virtualized rendering.

### Timings

- Boot: harness launch t0 → CDP answering → `starting` → each `phase` → `ready` → health 200 → renderer `load` event (from `Performance.timing`/`performance.timeOrigin` via CDP). Events are parsed from the stdout capture log, only lines after t0; each harness-observed mark is the first instant its probe succeeded. During `cold_boot` the sampler runs at `--boot-sample-interval` (default 0.25 s), switched on and woken **before** the binary is exec'd; the effective interval is recorded, not assumed. The boot timeline shows, at each mark, the app memory (`app_total`, `app_overhead`, per category) of the nearest sample, plus the peak during boot.
- Model download: duration, bytes, throughput.
- Chat turn: request sent → first `thinking` or `answer` event (TTFT) → `done`. Report `wall_s` (what the user waits), `tool_time_s` (sum of the gaps from each `tool_call` to the next streamed token), `generation_s` (sum of the intervals where tokens actually streamed, excluding the pre-first-token wait and the tool gaps), and characters per second of answer and of answer+thinking **over `generation_s`** — unavailable, never a huge number, when `generation_s` < 0.05 s. Token counts are not exposed; say so in the report. Keep the raw per-event offsets in `events.jsonl` so the formula can be redone afterwards. Record whether the model was already resident (from `/diagnostics/` before the turn) so cold-load turns are labelled.
- Embedding model download and KB ingestion duration; KB query turn TTFT.
- Idle unload: last use → inference process gone.
- Quit: quit command → every Erudi process gone; list survivors (orphans) at +5 s and +30 s.

### Storage (snapshot at start, after each phase that writes, and at the end)

- Install footprint: size of the install/app bundle, broken down to the top-level dirs of the resources dir and the 25 largest entries of the backend's bundled library dir (`_internal` or equivalent).
- Data root breakdown: `data/models` (per model dir/file), `data/models_cache`, `data/postgres` (total, `base/`, `pg_wal/`), `db-backups`, anything else; log dirs and the stdout capture log.
- Deltas between snapshots attributed to the phase.
- Use allocated size where the OS exposes it (`st_blocks * 512` on POSIX), logical size otherwise, and report which. Do not follow symlinks (the macOS bundle's `data` symlink points into the data root).

The harness must not read `data/erudi_db_password` or connect to Postgres directly.

## Flavours (platform + inference backend)

A run measures one flavour: `mac-mlx`, `win-cuda`, `win-cpu`, `linux-cuda`, `linux-cpu`. Each is a file in `flavours/<name>.json` carrying the default model link (MLX vs GGUF), the expected inference process shape (`llama-server[.exe]` vs the MLX multiprocessing child), whether per-process GPU memory exists (CUDA: NVML/`nvidia-smi`, and the report says when the driver returns `[N/A]`; MLX: unified memory, no separate figure; CPU: none), the extra facts to record (CUDA: `-ngl` from the child's command line and the GPU/driver from `/hardware/detailed`; CPU: `--threads` and `-ngl 0`) and the default workload file.

The flavour is guessed from the installed artifact, can be forced with `--flavour`, and is confirmed against the running app (`/hardware/app_startup` `backend_type`, `/diagnostics/` `environment.engine`). A detected flavour that contradicts the requested one fails the run: a CUDA run on a CPU build is not the same measurement. The flavour is part of the run id and the report header, and `compare` warns loudly when two runs do not share it.

## Profiles and declared workload

A run carries a named **profile** (`--profile`, default `clean`) that says what else was running on the machine:

- `clean`: reference for the app's own overhead. After a reboot, only the terminal running the harness.
- `nominal`: realistic pressure (does the model fit, swap, boot and TTFT under load). Browser, chat apps, a terminal running several sessions of a coding-agent CLI.

The profile is part of the run id and appears in `config.json`, `system.json` and the report header. Two runs with different profiles stay comparable because their conditions are printed side by side.

The background is **declared** in a workload file. The files in `workloads/` are examples whose rules are `<placeholders>`; a machine's real rules live in `workloads/local-<profile>-<os>.json`, which is gitignored and picked up by default when present (`--workload` overrides both). A group whose placeholders are unfilled is reported as "not configured" and matches nothing. A workload file holds: a list of groups, each with match rules (executable path prefixes or substrings, executable or argv[0] basenames or prefixes, command-line substrings, `orphaned` for processes re-parented to init), and:

- `sessions`: a session is a matching process whose parent does not match (a coding-agent CLI session, not its subprocesses);
- `include_descendants` / `descendants_of`: attribute a group's descendants (MCP servers, shells) to it, stated in the report;
- `expected_count`: verified at preflight; `expect_absent`: the group must not run (clean profile);
- `declared`: facts that cannot be verified from outside (browser tabs), reported as declared;
- `optional`: absence is not a warning.

A process belongs to at most one group (first matching group in file order). Erudi's processes, the harness and everything the harness started are never in a group.

Checks and measurements:

- At preflight each group is verified (present as declared, `expected_count`, `expect_absent`); a mismatch is a warning, an abort with `--strict-workload`.
- Every sample records per group: process count, sum of the primary metric where readable, RSS sum under its own name (never mixed with the primary), CPU %, unreadable counts. Measured cost: about 11 ms for ~60 processes on an M4, so it runs on every sample.
- Drift is reported per group and phase: process count changed, memory mean away from the baseline phase by more than `--workload-drift-pct` (default 25 %), or mean CPU above `--workload-busy-cpu` (default 20 %).

## Conditions (`system.json`)

Uptime and last boot time, swap used/total at start, memory pressure at start, power source, battery %, low power mode, OS build, display count where cheap, and the workload summary (groups found, counts, memory at start).

## Phases (scenario)

Each phase: mark start/end in `events.jsonl`, settle, sample continuously. Every phase can be selected or skipped from the CLI; phases that need a model or the network say so and are skipped with a reason when unavailable.

| # | Phase | What it does | Settle |
|---|---|---|---|
| 0 | `preflight` | OS, CPU, RAM, GPU/VRAM/driver, app version (Info.plist / exe version / AppImage name), free disk, harness version; refuse to start if any Erudi process is already running (unless `--attach`); refuse downloads when free disk < model size + `--disk-headroom-gb` (default 5) | — |
| 0b | `baseline` | app not running (skipped with `--attach`): sample the idle machine for `--baseline-seconds` (default 60) at the normal interval; the reference every later phase is compared to (memory breakdown, swap, pressure, CPU, top consumers) | — |
| 1 | `cold_boot` | switch the sampler to `--boot-sample-interval` (default 0.25 s), launch app with CDP port, wait CDP + health + `ready`, restore the normal interval | until ready |
| 2 | `idle_after_boot` | nothing | 60 s (`--idle-seconds`) |
| 3 | `ui_tour` | via CDP hash navigation: models catalog (`#/erudi/models`), chat, knowledge base, settings, back to chat; renderer metrics per page | 5 s per page |
| 4 | `model_ready` | find the installed model: resolve the catalog row by `link` to get its `name`, then accept an installed row (`local == 1`, not a KB assistant, weights present) matching on `link` **or** `name` (Erudi rewrites the link to the local path after a download), preferring a link inside the data root; report which rule matched. If missing and allowed, download from `/llms/remote`, poll status; record storage delta. Mark `downloaded_by_harness` | — |
| 5 | `chat_cold` | create conversation on the model; one short prompt; this turn pays model load | 30 s |
| 6 | `chat_warm` | N more turns (default 5) on the same conversation, fixed prompts asking for short answers | 30 s |
| 7 | `ui_stream` | open the conversation in the UI (`#/erudi/conversations/<id>`), type one prompt in the input and submit through the UI, wait for `done` via the API message count; renderer metrics during streaming | 30 s |
| 8 | `long_conversation_render` | continue up to `--long-turns` total turns (default 20) via API, then reload the conversation page in the UI and record renderer metrics | 30 s |
| 9 | `embedding_model` | if unavailable and allowed, download the embedding model; record storage delta | — |
| 10 | `kb_ingest` | `POST /knowledge_base/create` with `corpus/standard/*` (absolute paths), unique assistant name; poll to `completed`; backend memory delta is the embedder + ingestion cost | 30 s |
| 11 | `kb_query` | conversation on the KB assistant; 2 questions whose answers are in the corpus; verify a `tool_call` or KB mode in the stream/log | 30 s |
| 12 | `kb_stress` (opt-in) | ingest `corpus/stress/*` into a second assistant | 30 s |
| 13 | `web_search` (opt-in, network) | enable web search on a conversation, one question needing fresh facts | 30 s |
| 14 | `arena` (opt-in) | one arena query | 30 s |
| 15 | `idle_unload` | no activity until the inference process exits (timeout 660 s); memory after unload | 60 s after exit |
| 16 | `quit` | graceful quit; orphan check | — |
| 17 | `cleanup` | only with `--cleanup`: delete conversations and KB assistants the harness created, and models it downloaded (never anything else); restore settings changed by the harness | — |

Default models (catalog `link`, override with `--model-link`):
- macOS: `lmstudio-community/Qwen3-4B-MLX-4bit` (Qwen3 4B, thinking + tools, ~2.3 GB)
- Windows/Linux: `Qwen/Qwen3-4B-GGUF` (~2.5 GB)

## Outputs

`results/<run-id>/` with `run-id = <UTC timestamp>-<os>-<arch>-<build flavour>`:
- `system.json` (preflight), `config.json` (effective CLI/config)
- `samples.jsonl` (one line per sample: ts, phase, target/effective interval, `sample_cost_ms`, per-process records with category, machine context block, `harness` block, `top_other` every 5 s)
- `renderer.jsonl`, `events.jsonl` (phase marks, lifecycle events, turn metrics, downloads, errors)
- `storage/*.json` snapshots
- `summary.json` and `report.md`:
  - per phase × category: mean and peak of the primary metric (MB), plus `app_overhead`, `inference`, `app_total`
  - boot timeline table with app memory at each mark, the peak during boot and the boot sampling quality
  - machine context: for the baseline and each phase, min available memory, app memory / wired / compressed / cached files (mean, max), swap used (start, max, end), swap-in/out totals and peak rates, pageout and compression peak rates, worst pressure, CPU % mean/max, disk read/write peaks, thermal/power changes, harness self-cost; a baseline-vs-phase delta table (available, swap used, compressed); top other processes at baseline and at the phase with the lowest available memory
  - turn table (TTFT, duration, chars/s, cold/warm)
  - storage tables (install breakdown, data root breakdown, deltas per phase)
  - renderer table per page/phase
  - workload: declared facts, verification result, per group and phase mean/peak memory, process count, CPU %, drift warnings
  - anomalies: workload warnings, survivors after quit, unreadable processes, phases skipped and why, metric substitutions
- `logs/`: copies of the log lines written during the run (stdout capture, backend.log, mlx-child logs), filtered by time — never the whole historic files.

A `compare` subcommand takes two run dirs and prints, first, a **Conditions** section (profile, uptime, swap at start, power, workload groups and their memory at the start of each run), then machine-context deltas per phase (min available, max swap used, compressed) and the per phase × category deltas (absolute and %).

## Safety rules

- Never delete the data root, never touch models/conversations/KBs not created by the harness, and delete created ones only with `--cleanup`.
- Check free disk before every download.
- Leave the machine as found: quit the app at the end (unless `--leave-running`), restore changed settings.
- Never print or store secrets; never read the DB password file; never talk to the inference child directly.
- All waits are condition-based with explicit timeouts; a timeout ends the phase with a recorded failure, and the run continues with phases that do not depend on it.
