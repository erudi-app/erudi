# erudi-eval

Measures how much memory and storage each layer of the **installed, packaged** Erudi app uses at every step a user goes through, on macOS, Windows and Linux. The numbers are the baseline for the rewrite (#553). The harness observes and drives the app; it never modifies it and never deletes user data it did not create.

The specification is [`docs/SPEC.md`](docs/SPEC.md); the app facts it relies on are in [`docs/APP_REFERENCE.md`](docs/APP_REFERENCE.md).

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) (nothing else: `uv run` installs `psutil`, `websocket-client` and, off macOS, `nvidia-ml-py` into a cached environment).
- Erudi installed (macOS `/Applications/Erudi.app`, Windows per-user install, Linux AppImage), **not running**, unless you use `--attach`.
- Free disk: the default model is ~2.3 GB (MLX) / ~2.5 GB (GGUF), the KB embedding model (checked against a conservative 1.2 GB estimate), plus `--disk-headroom-gb` (default 5). Downloads are refused below that.
- A quiet machine: other heavy apps change the system-wide numbers (per-process numbers are unaffected).

First check what this machine can measure (no app needed, ~2 s):

```bash
uv run erudi_eval.py selftest
```

## Running

macOS:

```bash
cd rewrite/eval-harness
uv run erudi_eval.py selftest
uv run erudi_eval.py run                       # full scenario, launches and quits the app
uv run erudi_eval.py run --cleanup             # same, then deletes what the harness created
```

Windows (PowerShell):

```powershell
cd rewrite\eval-harness
uv run erudi_eval.py selftest
uv run erudi_eval.py run
# custom install location:
uv run erudi_eval.py run --app-path "D:\Apps\Erudi\Erudi.exe"
```

Linux:

```bash
cd rewrite/eval-harness
uv run erudi_eval.py selftest
uv run erudi_eval.py run --app-path ~/Applications/Erudi-1.1.2.AppImage
```

Useful options (all in `uv run erudi_eval.py run --help`):

| Option | Effect |
|---|---|
| `--profile clean\|nominal` | measurement profile (default `clean`); goes into the run id, `system.json` and the report |
| `--workload <file>` | workload file (default `workloads/<profile>-<os>.json` when it exists) |
| `--strict-workload` | abort when the declared workload does not match what is running (default: warning) |
| `--workload-drift-pct`, `--workload-busy-cpu` | thresholds for the drift warnings (defaults 25 %, 20 %) |
| `--flavour <name>` | `mac-mlx`, `win-cuda`, `win-cpu`, `linux-cuda`, `linux-cpu` (default: guessed from the install, then verified against the running app) |
| `--attach` | measure an app that is already running (boot is not measured; UI phases need it to have been started with `--remote-debugging-port=9222`) |
| `--phases a,b` / `--skip a,b` | select or skip phases by name |
| `--opt-in kb_stress,web_search,arena` | enable the opt-in phases |
| `--model-link <link>` | chat model (catalog `link`); default comes from the flavour (MLX on `mac-mlx`, GGUF elsewhere) |
| `--no-download` | never download; phases needing a missing model are skipped with the reason |
| `--idle-seconds`, `--warm-turns`, `--long-turns` | scenario sizes (defaults 60, 5, 20) |
| `--baseline-seconds` | how long the idle machine is sampled before the app is launched (default 60) |
| `--boot-sample-interval` | sampling interval during `cold_boot` (default 0.25 s; the effective interval is recorded) |
| `--disk-headroom-gb` | free space required beyond a download's size (default 5) |
| `--cleanup` | delete the conversations, KB assistants and models this run created |
| `--leave-running` | do not quit the app at the end |
| `--results-dir`, `--sample-interval` | output location, sampling period (default 1 s) |

Advanced options exist for tests and unusual installs (`--api-port`, `--cdp-port`, `--main-pid`, `--data-root`, `--backend-log-dir`, `--capture-log`, `--settle-seconds`, timeouts). `--settle-seconds` replaces every settle duration and makes the run incomparable with a normal one; it exists for the test suite.

## Flavours (platform + inference backend)

A run measures one **flavour**: the pair (platform, inference backend). They behave differently enough that mixing them is meaningless, so the flavour is part of the run id, checked against the running app, and `compare` warns loudly when two runs do not share it.

| Flavour | Model downloaded | Inference process | Per-process GPU memory | Extra facts recorded |
|---|---|---|---|---|
| `mac-mlx` | `lmstudio-community/Qwen3-4B-MLX-4bit` (MLX) | multiprocessing child of the backend (`mlx_child`) | unified memory: already inside `phys_footprint`, no separate VRAM figure | — |
| `win-cuda` | `Qwen/Qwen3-4B-GGUF` | `llama-server.exe` (CUDA build) | NVML / `nvidia-smi`; reported as unavailable when the driver hides it (common under WDDM) | `-ngl`, `--threads`, GPU name/VRAM/CUDA version from `/hardware/detailed` |
| `win-cpu` | `Qwen/Qwen3-4B-GGUF` | `llama-server.exe` (CPU build) | none expected | `--threads`, `-ngl` (must be 0) |
| `linux-cuda` | `Qwen/Qwen3-4B-GGUF` | `llama-server` (CUDA build) | NVML / `nvidia-smi` | `-ngl`, `--threads`, GPU detail |
| `linux-cpu` | `Qwen/Qwen3-4B-GGUF` | `llama-server` (CPU build) | none expected | `--threads`, `-ngl` (must be 0) |

The flavour is guessed from the installed artifact (bundled `llama-cpp/<cpu|cuda>` directory, DMG/AppImage name) and can be forced with `--flavour`. As soon as the app answers, the harness asks it what it really is (`GET /hardware/app_startup` → `backend_type`, `/diagnostics/` → `environment.engine`): **a contradiction fails the run** (in `--attach` at preflight, otherwise at `cold_boot`), because a CUDA run against a CPU build is not the same measurement. An app that reports neither is taken as requested, and the report says so.

The definitions live in `flavours/<name>.json` (model link, expected inference process, GPU expectation, extra checks, default workload file). The Windows and Linux files are marked `UNVERIFIED` in their `notes`.

## Profiles and the declared workload

A clean machine is not reality, so a run says what else was running:

- **`clean`** — reference for the app's own overhead, reproducible: after a reboot, only the terminal running the harness.
- **`nominal`** — realistic pressure: a browser, chat apps, a terminal running several sessions of a coding-agent CLI. It answers "does the model still fit", "does the machine swap", "what do boot and TTFT cost under load".

### Recommended protocol (macOS)

Both runs: on AC power, Erudi quit before starting, same model and same app version.

1. **`clean`**: reboot, open one terminal for the harness (at most one idle coding-agent CLI session, declared in your local clean file), then
   `uv run erudi_eval.py run --profile clean --cleanup`.
2. **`nominal`**, right after, without rebooting: open the browser with the 4 declared tabs, the chat apps idle, and the terminal with 5 coding-agent CLI sessions open and idle for the whole run, then
   `uv run erudi_eval.py run --profile nominal --cleanup`.
3. `uv run erudi_eval.py compare results/<clean-run> results/<nominal-run>`.

The swap state after a reboot will not match a machine that has been up for days. That is stated in the report's conditions (uptime, swap at start), not simulated.

### Local workload files

The files in `workloads/` are **examples**: their rules are `<placeholders>`, because install paths and the tools someone runs are specific to a machine. To measure a real machine:

```bash
cp workloads/nominal-mac.json workloads/local-nominal-mac.json   # or clean-mac / *-windows
# fill in every <placeholder> with the paths on this machine, delete the rules you do not need
uv run erudi_eval.py run --profile nominal --workload workloads/local-nominal-mac.json
```

`workloads/local-*.json` is gitignored, and a run with no `--workload` picks `workloads/local-<profile>-<os>.json` when it exists, the example otherwise. A group whose placeholders are still there is reported as **not configured**: it matches nothing, and the run continues. Filling it in changes no behaviour, only what the rules match.

The example groups are deliberately generic: `browser`, `chat_app`, `terminal`, `coding_agent_cli` (a terminal coding-agent CLI), `coding_agent_children` (the shells and MCP servers such a session spawns) and `coding_agent_orphans` (servers a past session left behind). Rename or add groups freely in your local copy.

### Workload file format

`workloads/<profile>-<os>.json` (example) or `workloads/local-<profile>-<os>.json` (yours): `profile`, `notes`, and `groups`. Each group has a `name`, match rules, and declarations:

| Key | Meaning |
|---|---|
| `exe_prefixes`, `exe_contains` | executable path prefix / substring (`%LOCALAPPDATA%`, `$HOME` and `~` are expanded) |
| `exe_basenames`, `argv0_basenames`, `argv0_prefixes` | basename of the executable or of argv[0], or argv[0] as a path prefix |
| `cmdline_contains` | substring of the whole command line |
| `orphaned` | only processes re-parented to init/launchd (e.g. MCP servers left behind) |
| `sessions` | a session is a match whose parent does not match; `expected_count` then counts sessions |
| `include_descendants` / `descendants_of` | attribute descendants (helpers, MCP servers, shells) to this group |
| `expected_count` | verified at preflight |
| `expect_absent` | the group must not be running (used by the `clean` profile) |
| `optional` | absence is not a warning |
| `declared` | facts nobody can verify from outside (browser tabs, "idle"), reported as declared |
| `notes` | free text, printed in the report |

A process lands in at most one group (first match in file order). Erudi's processes, the harness and everything the harness starts are never counted in a group. Rules are matched against the executable path **and** argv[0], because an app is not always where its name says: on this Mac the browser's main process runs from a code-signature clone under `/private/var/folders/...`, and the coding-agent CLI's executable basename is its version number (`2.1.272`) while argv[0] is the CLI name.

Every sample measures each group: process count, sum of the primary metric where readable, RSS sum reported separately (never mixed), CPU %, unreadable counts (~11 ms for ~60 processes on an M4). The report flags **workload drift**: a group that changed size, whose memory moved more than `--workload-drift-pct` from the baseline phase, or that used more than `--workload-busy-cpu` on average (an agent session that started working).

Shipped examples: `workloads/clean-mac.json`, `workloads/nominal-mac.json`, `workloads/clean-windows.json`, `workloads/nominal-windows.json`. The macOS rule shapes (code-signature clone, version-named executable) come from a read-only look at a real Mac; the Windows ones are written from the usual install locations and have **never been checked on Windows** — see the `notes` in those files.

## What a run does

Every phase is marked in `events.jsonl`, and the sampler tags each 1 s sample with the current phase.

| # | Phase | What happens | Typical duration |
|---|---|---|---|
| 0 | `preflight` | system info, app version, free disk, `iogpu.wired_limit_mb`; **refuses to start if Erudi is running** (unless `--attach`); first storage snapshot | 2–5 s |
| 0b | `baseline` | the app is not running yet: samples the idle machine (skipped with `--attach`) | 60 s |
| 1 | `cold_boot` | switches the sampler to 0.25 s and wakes it, then launches the app binary with `--remote-debugging-port`; records when CDP, the lifecycle events, health 200 and the renderer `load` event first appear; builds the boot timeline with app memory at each mark; restores the normal interval | 5–60 s (first run is longer) |
| 2 | `idle_after_boot` | nothing | 60 s |
| 3 | `ui_tour` | hash navigation models → chat → knowledge base → settings → chat; renderer metrics per page | ~30 s |
| 4 | `model_ready` | finds the installed model (see below), else downloads it from `/llms/remote` after the disk check | 0 s, or the download (5–30 min) |
| 5 | `chat_cold` | new conversation (binding verified through the API), one prompt; this turn pays the model load | 30–60 s |
| 6 | `chat_warm` | 5 more short turns | 1–4 min |
| 7 | `ui_stream` | types a prompt into the conversation page and submits it through the UI; renderer metrics while streaming | ~1 min |
| 8 | `long_conversation_render` | API turns up to 20 user turns, then reloads the conversation page and records renderer metrics | 3–10 min |
| 9 | `embedding_model` | downloads the KB embedding model if missing | 0 s or 1–3 min |
| 10 | `kb_ingest` | creates a KB assistant from `corpus/standard/*`, polls to `completed` | 1–5 min |
| 11 | `kb_query` | 2 questions answered by the corpus; fails unless the stream has a `search_knowledge_base` call or `backend.log` states a KB turn mode | ~2 min |
| 12 | `kb_stress` (opt-in) | ingests `corpus/stress/*` into a second assistant | 2–10 min |
| 13 | `web_search` (opt-in, network) | conversation with web search on; fails without a `web_search` tool call | ~1 min |
| 14 | `arena` (opt-in) | one arena query | ~1 min |
| 15 | `idle_unload` | waits until the inference process exits on its own (the backend unloads 300–600 s after last use; timeout 660 s), then 60 s | 5–11 min |
| 16 | `cleanup` | only with `--cleanup` (see deviations) | seconds |
| 17 | `quit` | graceful quit; lists processes still alive at +5 s and +30 s; final storage snapshot | 5–35 s |

A full default run takes **about 26–46 minutes** when the models are already on disk, plus download time otherwise. A phase whose precondition is missing is skipped with the reason; a phase that fails records why and the run continues with the phases that do not depend on it.

### Process categories

Every process of the installation is attributed to one category, re-discovered at every sample (`erudi_eval/discovery.py`, one pure `classify()` with fixtures in `tests/fixtures/`):

`electron_main`, `electron_renderer`, `electron_gpu`, `electron_utility` (by Chromium `--type=`), `backend` (the bundled `backend` executable; its multiprocessing resource tracker is attributed to it too), `database` (every process of the bundled `pginstall/bin/postgres`, found by path because the postmaster is re-parented on POSIX), `inference` (`llama-server`, or on macOS the multiprocessing child of the backend running the backend executable), `embedding` (empty today: the embedder runs inside the backend), `transient` (`pg_ctl`, `initdb`, `pg_dump`, anything else in the tree). Derived per sample: `app_total`, `app_overhead` (= total − inference − embedding), `inference_total`.

A process is part of the installation when it descends from the Electron main process, or executes a file inside the install directory, or is a `pginstall` postgres of the bundle / serving the Erudi data root. A developer's `pgserver` cluster or a system PostgreSQL never matches.

### Finding an already-installed model

After a download Erudi rewrites the row's `link` to the local model directory, so an installed model no longer carries the catalog link (observed: catalog row `lmstudio-community/Qwen3-4B-MLX-4bit`, installed row link `.../data/models/358`). The harness therefore resolves the catalog row by link to get its **name**, then accepts an installed row (`local == 1`, not a KB assistant, weights present) whose link **or** name matches. When several match it prefers a row whose link points inside the data root, then the oldest id, and the report says **matched by name** with the caveat, so a name collision cannot pass silently. Before this, every run downloaded a second copy of a model it already had.

### Turn timings and throughput

| Column | Meaning |
|---|---|
| `TTFT s` | request sent → first `thinking` or `answer` event (includes the model load on a cold turn) |
| `wall s` | request sent → `done`: what the user waits for |
| `tool s` | sum of the gaps from each `tool_call` to the next token that streams (tool execution) |
| `generation s` | sum of the intervals during which tokens actually streamed — the pre-first-token wait and the tool gaps are excluded |
| `answer chars/s`, `total chars/s` | characters (answer, and answer+thinking) over `generation s` |

Token counts are not exposed by the API, so throughput is in characters. When `generation s` is under 0.05 s the figure is reported as `n/a` instead of a huge number: a turn whose events all arrive in one burst after a tool call (seen at 39 000 chars/s in the first real run) says something about buffering, not about the model. Every turn keeps its raw `event_timeline` (offset in seconds, event kind) in `events.jsonl`, so another formula can be applied after the run without measuring again.

### Metrics

| | macOS | Windows | Linux |
|---|---|---|---|
| primary | `phys_footprint` (`proc_pid_rusage` v4) | `private_working_set` (psutil USS) | `pss` |
| OS peak | `lifetime_max_phys_footprint` | `peak_wset` | max over samples |
| also | `rss`, `uss` (needs root for other processes: recorded as unreadable) | `rss`, `private` (commit), `pagefile` | `rss`, `uss`, `swap` |
| GPU | unified memory, inside `phys_footprint` | NVML or `nvidia-smi` per process + per GPU | same as Windows |

A metric that cannot be read is recorded as an error under its own name and listed in the report; the harness never substitutes one metric for another.

### Machine context (every sample)

So the app's numbers can be read against the state of the machine (`erudi_eval/machine.py`):

| | macOS | Windows | Linux |
|---|---|---|---|
| memory breakdown | `host_statistics64` via ctypes (no subprocess): free/active/inactive/speculative/wired/purgeable/compressor pages, internal/external pages; Activity Monitor's app memory, wired, compressed, cached files, memory used | psutil + `GetPerformanceInfo`: commit charge/limit, system cache, kernel paged/nonpaged, handles, processes, threads | `/proc/meminfo`: MemAvailable, Cached, SwapCached, AnonPages, Shmem, SwapTotal/SwapFree |
| swap | `vm.swapusage` total/used/free | psutil | psutil |
| paging rates (/s) | pageins, pageouts, swapins, swapouts, compressions, decompressions, faults | hard page faults: unavailable (needs PDH) | pswpin, pswpout, pgmajfault |
| pressure | `kern.memorystatus_vm_pressure_level` | commit charge vs limit | PSI memory some/full |
| every 10 s | `pmset -g therm` / `batt` / low power mode | power source (psutil battery); thermal unavailable | same as Windows |

On every OS: system CPU %, load average, process count, disk read/write bytes/s, free space on the data-root volume. Every 5 s: the 10 largest non-Erudi processes by RSS and their total. Each sample also carries `sample_cost_ms`, the target and effective interval, and a `harness` block (the harness's own memory and CPU, never counted in app totals). `uv run erudi_eval.py selftest` prints this block and the measured cost of one sample.

Renderer internals (`Performance.getMetrics`, `Runtime.getHeapUsage`) are recorded at each UI stop and every 5 s while CDP is connected.

## Outputs

`results/<UTC timestamp>-<os>-<arch>-<flavour>/`:

| File | Content |
|---|---|
| `system.json`, `config.json` | preflight facts (including conditions: uptime, swap, power, OS build, workload summary), effective options |
| `workload.json` | the workload file used, copied into the run |
| `samples.jsonl` | one line per sample: phase, target/effective interval, `sample_cost_ms`, every process (category, role, metrics, errors), per-category totals, machine context (`system`), `harness` block, `top_other` every 5 s |
| `renderer.jsonl` | renderer metrics with phase, page and trigger |
| `events.jsonl` | phase marks, boot timeline, turns, downloads, KB evidence, settings changes, quit result, errors |
| `phases.json` | status, reason, duration and data of each phase |
| `storage/NN-<label>.json` | storage snapshots (allocated bytes on POSIX, logical on Windows; symlinks never followed) with the delta from the previous one |
| `summary.json`, `report.md` | header with profile and conditions, workload section (declared facts, verification, per group × phase, drift), tables: memory per phase × category (mean / peak MB), largest process peaks, boot timeline with app memory at each mark and the boot peak, machine context per phase (memory breakdown, swap and paging, CPU/disk/power, harness cost), baseline-vs-phase deltas, top other processes at baseline and at the lowest-memory phase, turns (TTFT, duration, chars/s, cold/warm), install and data-root breakdown, storage deltas, renderer, anomalies |
| `logs/` | this run's lines of the stdout capture log, `backend.log` and `mlx-child-*.log`, filtered by time |

## Comparing two runs

```bash
uv run erudi_eval.py compare results/<run-a> results/<run-b> --out compare.md
```

Prints a **Conditions** section first (profile, uptime, swap at start, memory pressure, power, OS build, and each workload group with its process count and memory at the start of both runs), then machine-context deltas per phase (min available, max swap used, compressed), then per phase × category mean and peak with the absolute and relative delta. It warns when the two runs use different profiles or different primary metrics (different OS).

## Safety rules the harness enforces

- Refuses to run on top of a running Erudi (unless `--attach`); never touches another app's processes. Quit targets only the Electron main process: AppleScript `quit app "Erudi"` when the main binary is the bundle's, `taskkill /PID <pid>` (no `/F`) on Windows, `SIGTERM` to the main PID otherwise. An explicit `--main-pid` is honoured only when that process is this installation's Erudi binary; otherwise the harness discovers the main process itself.
- Checks free disk before every download (size + headroom); unknown sizes are refused.
- Deletes only what it created, and only with `--cleanup`; never uses `orphan_dependents=true`. The embedding model cannot be deleted through the API and stays in `data/models_cache` if the harness downloaded it.
- Sets `auto_update_enabled=false` for the run and restores it at the end.
- Never reads `data/erudi_db_password` (the storage walker only `stat`s files), never connects to Postgres or to the inference child, redacts `--api-key` and similar values from recorded command lines.
- Every wait has a timeout; a timeout fails the phase, not the run.

## UI driving: selector strategy

Verified against the frontend source (`frontend/src/App.jsx`, `components/QuestionInput.jsx`, `pages/ConversationPage.jsx`), not yet against a running build:

- Navigation sets `location.hash` to the HashRouter routes `#/erudi/models`, `#/erudi/chat`, `#/erudi/attach_knowledge_base`, `#/erudi/settings`, `#/erudi/conversations/<id>` and reads `location.hash` back.
- The composer is the visible `<textarea>` whose container (up to 4 ancestors) holds the send button's `svg.lucide-arrow-right`. The harness focuses it, inserts the text with `Input.insertText`, **reads the value back**, then sends an Enter `rawKeyDown`/`keyUp` (QuestionInput submits on Enter without Shift). If the text is still in the box 1.5 s later it clicks the send button instead and records `submit_method`.
- The turn counts only when `GET /conversations/<id>/fetch_messages` shows two more messages.

## Tests

```bash
uv run --with pytest pytest tests/
```

Unit tests cover classification over process fixtures (one recorded from a real Erudi 1.1.2 on macOS, modelled ones for macOS, Windows and Linux), the NDJSON parser and turn metrics, lifecycle log parsing, storage walking (symlinks, hard links), the report and compare. `tests/test_fake_app_run.py` runs a full `run --attach --cleanup` against `tests/fake_app.py`, a stdlib stand-in that serves the endpoints the phases use, writes lifecycle and backend logs, and spawns processes shaped like the real tree (helpers with `--type=`, a backend with a multiprocessing inference child, a re-parented `pginstall/bin/postgres` binary). A second fake-app test runs `preflight,baseline,cold_boot,idle_after_boot,quit` without `--attach`: the harness samples the machine, then launches the fake main as the app binary with 0.25 s boot sampling. Tests kill only the PIDs the fake app wrote to its own registry. Set `ERUDI_EVAL_KEEP_RESULTS=<dir>` to keep those runs' output. macOS-only tests skip elsewhere; the fake-app test skips on Windows.

## Known limitations

- **No token counts**: the API does not expose them; throughput is characters per second of answer text.
- **Idle unload is slow by design**: the backend checks every 300 s, so the inference process exits 300–600 s after the last use.
- **MLX child identification** relies on "child of the backend running the backend executable, not the multiprocessing resource tracker". A future helper process spawned the same way would be counted as inference.
- **KB embedder** runs inside the backend and is never unloaded: its cost shows up as a `backend` delta across `kb_ingest`, not as `embedding`.
- On macOS `uss` of other processes needs root and is recorded as unreadable; `phys_footprint` is readable for the app's processes (checked on a running 1.1.2).
- Per-process GPU memory under Windows WDDM is often `[N/A]` in NVML/`nvidia-smi`; it is recorded as unavailable.
- Windows and Linux paths, the Linux AppImage executable name (`erudi` inside the mount) and the Windows exe version query were written from the source and `docs/APP_REFERENCE.md`, not run on those systems.
- **A flavour cannot be verified before the app runs**: the artifact guess (bundled `llama-cpp` directory) is confirmed only once the app answers. On a launch run the mismatch therefore fails `cold_boot`, not `preflight`.
- **The workload is declared, not controlled**: the harness verifies what it can see (processes, counts) and reports the rest as declared (browser tabs, "idle"). A tab that loads a heavy page during a run shows up as drift, not as a corrected number.
- **MCP servers spawned by a coding-agent CLI survive their session**, re-parented to launchd (21 of them, ~437 MB, on the Mac used for development). They belong to their own group (`coding_agent_orphans`) so they are not mistaken for a live session's cost.
- Workload groups are matched by executable path and argv[0]; an app installed elsewhere needs its rule adjusted. The Windows rules have never been run.
- `--attach` cannot measure boot or take a baseline, and UI phases need the running app to expose CDP.
- **Sampling cost bounds the boot resolution**: one sample (process discovery + reads + machine context) measured ~55–95 ms on an M4 with ~1,000 processes. 0.25 s holds there; on a slower machine the effective interval is longer and is recorded in every sample.
- **The harness itself shows up in machine-wide figures** (CPU %, a few tens of MB): its own cost is recorded in the `harness` block so it can be subtracted by the reader, but machine totals are not corrected.
- **Top other processes** use RSS (the only cheap per-process figure readable for other users' processes), and processes the harness cannot read are counted as unreadable, not estimated. On macOS about a quarter of processes (system daemons) are unreadable without root.
- **Swap-ins/outs on macOS** are counted in compressor pages (`vm_statistics64.swapins/swapouts`), not bytes. Thermal state on Windows/Linux and hard page faults on Windows are recorded as unavailable.

## Deviations from `docs/SPEC.md`

- `cleanup` runs **before** `quit` (spec order is quit then cleanup): deleting through the API needs the app running.
- Linux graceful quit sends `SIGTERM` to the main PID instead of `pkill -TERM -f Erudi` (which also matches unrelated command lines, including a harness invoked with an AppImage path); Windows uses `taskkill /PID` instead of `/IM Erudi.exe`.
- The multiprocessing resource tracker child of the backend is categorised `backend` (role `mp_helper`), not `inference`.
- Re-parented Postgres is matched when the `pginstall/bin/postgres` binary lives in the install directory **or** its command line names the Erudi data root (the binary is bundled under `resources/backend/_internal/pgserver`, not under the data dir).
- Extra options not in the spec: `--no-download`, `--warm-turns`, and the advanced/testing group.
- `cold_boot` waits for one sample at the fast rate before exec'ing the binary, so the first sample after launch is within one boot interval.
