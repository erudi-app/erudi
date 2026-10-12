# Engines Architecture

`backend/src/engines/` holds every inference engine. All three run an
OpenAI-compatible HTTP server in a child process and stream from it over SSE.

## Hierarchy

```text
BaseEngine
└── BaseChatServerEngine        ← shared: port pick, /health + chat-ping probe,
    │                             SSE byte-buffer parser, atexit storage,
    │                             last-used stamp, kwarg translation
    ├── MLX_Engine               (mp.Process + mlx_vlm.server, ports 27300-27399)
    └── BaseLlamaCppEngine      ← shared CPU/CUDA: Popen, llama-server resolution,
        │                         GGUF picker, `repetition_penalty → repeat_penalty`
        ├── CPU_Engine           (Popen + llama-server CPU, -ngl 0, ports 27200-27299;
        │                         sizes -c itself, binds the llama.cpp memory brake)
        └── CUDA_Engine          (Popen + llama-server CUDA, -ngl -1 with fit, 27200-27299;
                                  sizes -c against the VRAM, binds the brake on a full offload)
```

Concrete engines implement only the four hooks that are genuinely backend-specific:

| Hook | CPU / CUDA | MLX |
|---|---|---|
| `_spawn_child` | `subprocess.Popen` | `multiprocessing.Process(target=run_mlx_vlm_server, ...)` |
| `_terminate_process` | process-group / job-object teardown | child process terminate |
| `_proc_is_alive` | `Popen.poll()` | `Process.is_alive()` |
| `_resolve_model_artifact` | pick a `.gguf` in the model directory | resolve the MLX repo directory |

The llama.cpp subclasses additionally implement `_build_spawn_argv` and
`_build_spawn_env` (CUDA prepends the CUDA toolkit `bin/` to `PATH` so the runtime DLLs
resolve, and pins the GPU it sized the window on with `CUDA_VISIBLE_DEVICES`).

`multiprocessing.Process` is required for MLX because a PyInstaller frozen build has no
Python interpreter at `sys.executable` to pass `-m` to; `mp.spawn` (configured in
`backend/run.py`) re-executes the binary in child mode. CPU and CUDA can use `Popen`
because the `llama-server` binary is bundled at
`backend/artifacts/llama-cpp/<cpu|cuda>/bin/`.

## Engine selection

```python
from src.engines.base_engine import BaseEngine
from src.core import config

# Auto-select the engine CLASS (not an instance — engines expose only
# classmethods; instantiation is intentionally blocked).
config.LLM_Engine = BaseEngine.get_engine()
```

`BaseEngine.get_engine()` (`base_engine.py`) dispatches at startup:

- macOS ARM (`platform.system() == "Darwin"` and `"arm" in platform.machine()`) → `MLX_Engine`
- Windows/Linux with `pynvml.nvmlDeviceGetCount() > 0` → `CUDA_Engine`
- otherwise → `CPU_Engine`

`ERUDI_FORCE_CPU=1` bypasses GPU detection entirely.

Detection runs before the migrations, so it cannot read the database. The lifespan
finishes the job later, once the schema is at head: `apply_inference_backend_preference()`
replaces `CUDA_Engine` with `CPU_Engine` when `user_settings.inference_backend` is `cpu`
(safe only because both share `FORMAT_TAG = "gguf"` — the catalog reconciled under one is
identical under the other), and `emit_engine_notice(app)` then runs the CUDA pre-flight on
whichever engine won. `ERUDI_FORCE_CPU` wins over the preference. See
`src/engines/cuda_compatibility.py` for the floors and
[Hardware Detection](../../guides/hardware.md) for the compatibility matrix.

A crash that gets past the pre-flight is classified where it happens:
`BaseChatServerEngine._probe_ready` matches the dead child's captured tail against the
strings ggml prints and raises `EngineException(engine_code=...)`, which the runner and
the conversation service carry to the chat stream's error event as `code` + `raw`.

## Model specifications

| Engine | Hardware | Model format | Server | Child launch | Ports |
|---|---|---|---|---|---|
| MLX | Apple Silicon | MLX repos (`mlx-community/*`) | `mlx_vlm.server` | `mp.Process` | 27300-27399 |
| CUDA | NVIDIA GPU | GGUF | `llama-server` (CUDA build) | `subprocess.Popen` | 27200-27299 |
| CPU | Windows / Linux CPU | GGUF | `llama-server` (CPU build) | `subprocess.Popen` | 27200-27299 |

The backend's own HTTP server binds 27182-27199, below both pools, so the three local
servers never contend for a port.

## llama-server slots and prompt cache

`BaseLlamaCppEngine._spawn_child` appends the same flags to every `llama-server` it spawns
(CPU and CUDA alike), after the engine's own argv: `--api-key` (minted per spawn),
`--no-slots`, `--no-webui`, `-np 1`, `--cache-ram 0` and `--no-cache-idle-slots`.

`-np 1 --cache-ram 0 --no-cache-idle-slots` makes the child hold **one conversation's KV at a time**, like the MLX
engine's single-conversation prefix cache (next sections). The upstream defaults are four
slots and `--cache-ram 8192`, a host-RAM cache of past prompts' full KV state of up to 8 GiB
that nothing in Erudi counts: each distinct long conversation added roughly its KV size to
the child's memory (about 1.3 GiB for a 12.5k-token conversation on a small model) and could
push the machine into swap. With one slot and no prompt cache:

- the follow-up of the **current** conversation still reuses its prefix inside the slot;
- going back to an **older** conversation re-reads its history once (a longer first answer);
- the child's memory no longer grows with each distinct conversation. `-np 1` turns
  unified KV off; with one sequence the slot's window is the whole `n_ctx`
  (`n_ctx_seq = n_ctx / 1`), so the window and `/props` `n_ctx` are unchanged; the
  sliding-window cache of models such as Gemma 2/3 shrinks, since it no longer reserves room
  for four sequences. `--no-cache-idle-slots` makes explicit that idle slots are not cached
  (llama-server otherwise warns at every spawn that this needs the RAM cache and disables it).

The single slot is why the first message of a conversation requests its title *before* the
chat turn (`ConversationPage.jsx`): a title run after the turn would overwrite the
conversation's KV in the slot and the second message would re-read the whole conversation.

## Context windows

Two numbers describe "the window", and they must never be confused:

- the **declared ceiling** — `BaseEngine.max_context_tokens()`: what configuration
  imposes on every spawn (`ERUDI_CTX` when the user pinned one, `None` otherwise). It
  exists before any model is loaded and never depends on which model is.
- the **allocated window** — `BaseEngine.effective_context_tokens()`: what the currently
  loaded child actually runs with, read from the live engine handle
  (`context_tokens`). It exists only while a child is up, differs per model *and* per
  machine or memory state, and is therefore never stored in the database. Everything
  that budgets or warns against "the window" (compaction thresholds, the first-token
  watchdog ceiling, the model-info display) reads this one.

The two engine families resolve the allocated window through **genuinely different
mechanisms**, each kept inside its own engine:

|  | llama-server (CPU / CUDA) | mlx_vlm.server (MLX) |
|---|---|---|
| KV allocation | **Fixed upfront**: the whole window's KV cache is allocated (and touched) at load — a ggml architectural fact (static graphs, fixed-shape tensors), linear in the window size | **Native lazy growth**: nothing allocated upfront, the cache grows with the conversation |
| Window resolution | **CPU: sized by Erudi at spawn** — `-c = min(trained window, what 0.75 × RAM affords)`, folded with the memory brake's cap ([below](#the-window-on-the-cpu-engine)). **CUDA: sized by Erudi at spawn for a full offload** — `-c = min(trained window, what min(0.9 × VRAM, VRAM free at load) − 1 GiB affords)`, with `-ngl -1` so llama.cpp's fit places the layers ([below](#the-window-on-the-cuda-engine)). **Fallback (a GGUF header the engine cannot read or model, a CUDA floor that does not fit): fit at load** — with no `-c`, the server itself resolves the window to the model's trained window and *continuously reduces* it against measured free memory, down to a 4096 floor, logging `context size reduced from X to Y` (`--fit`, ON by default in the pinned b10883) | **Preflight bound at spawn, lowered by memory after load**: the engine reads the trained window from the local artifact's `config.json` (`read_local_generation_hints`) and passes it as `--max-kv-size` — a validator, not a rotating cache. Once the child is up and its base footprint measured, a memory ceiling below the trained window lowers that bound live (see below), down to a 4096 floor; under memory pressure the memory brake moves it again |
| `ERUDI_CTX` set | Passed verbatim as `-c`; llama-server honours an explicit `-c` unchanged ("no change"), and the memory brake of either llama.cpp engine never resizes it | Not read — the bound always comes from the artifact |
| Overflow behaviour | Honest 400 carrying `n_prompt_tokens` and `n_ctx` | Honest 400 naming the exact budget (`prompt + max generation` vs `MAX_KV_SIZE`); *without* the bound, an oversized prompt would return 200 with silently degenerate output |
| How the allocated window is known | **Read back after boot**: `_read_server_properties` (called by `_start_server` right after the probe) does one bounded `GET /props` and stamps `default_generation_settings.n_ctx`, `chat_template_caps` and `chat_template` on the handle | **Known at spawn, rewritten when memory binds**: the bound just passed *is* the window; `_read_server_properties` measures the child's base footprint and, when memory binds, stamps the memory bound as the window |
| No derivable window | Cannot happen — the server always resolves one | No `--max-kv-size` (today's unbounded behaviour, one WARNING); a number is never invented |

Either way, a `/props` failure or a missing window degrades (one WARNING) and the model
still loads: a llama-server spawned with `-c` keeps the window it was given as
`context_tokens`, any other child gets `context_tokens = None`, which consumers treat as
"window unknown". A `/props` window other than the `-c` passed is one WARNING. A
stopped child clears `context_tokens` on its handle: a dead child holds no KV memory,
so its allocated window dies with it.

### The memory bound (MLX)

The window a conversation plans against is the **working window**,
`min(allocated window, memory ceiling)` (`src/engines/working_window.py`); the memory
ceiling is the conversation token count at which the child's predicted footprint
reaches the memory budget (`MemoryBudget.tokens_at_ceiling`, Apple Silicon only). When
the model's window is the smaller of the two, nothing below runs. When the memory
ceiling is smaller, the MLX child itself must refuse what memory cannot hold, and two
facts make that cheap: the ceiling is constant for the life of a child (its inputs — the
measured base, the working-set total, the model's shape and the prior — do not move while
it runs), and mlx_vlm re-reads `max_kv_size` on every request (a live setting, outside
the fingerprint that triggers a model reload). So `_read_server_properties`, right after
measuring the base footprint and before anything but the readiness ping reached the
child, computes `kv_bound(trained window, min(ceiling, brake cap))` — the ceiling (or the
smaller cap a [memory brake](#the-memory-brake-mlx) reload carried), never below 4096 (the
floor `--fit` applies on llama.cpp) and never above the trained window — and sends it
through `PATCH /v1/settings` (`{"max_kv_size": bound}`, the per-spawn API key as a
Bearer token, 2 s timeout). The same applier (`_apply_memory_bound`) runs again whenever
the memory brake moves its cap, so the bound is not constant for the life of a child: it
follows `min(ceiling, cap)`, and once memory no longer binds it sends the trained window
back and no memory bound is recorded (a child spawned without a trained window cannot be
given its unbounded window back: it keeps its last bound, no longer reported as a memory
bound). When the answer does not confirm it — an error status, a
timeout (which can fire after the child applied the change), an unreadable or unexpected
body — one `GET /v1/settings` with the same key and timeout reads the child's live
`max_kv_size` back, and the handle follows what the child really enforces. Once the
child enforces the bound, it becomes the handle's `context_tokens`,
so every consumer of the allocated window (the model-info card, the first-token
watchdog, the preflight retry, compaction) follows it unchanged, exactly as with a
window `--fit` reduced, and `memory_bound_tokens()` reports it; `backend.log` carries
`[MLX_Engine] Window bounded by memory: trained=..., ceiling=..., bound=...` at load (with
`, brake_cap=...` when a brake reload carried a cap). A trained window
already at or below the floor needs no request: the child enforces it since its spawn,
and it is recorded as the memory bound all the same. The prefix-cache pool stays sized on
the trained window (it fills lazily), and a child spawned without a trained window waits
3660 s per token (the first-token watchdog's absolute maximum plus 60 s), so it outlives
any ceiling a memory bound can give the parent.

The bound is the ceiling, which already reserves the output budget's 512-token floor, and
the child counts `prompt + max_tokens` against it: those 512 tokens are counted twice,
which keeps the child on the safe side and its bound equal to the window the output
budget plans against. Only the memory brake re-applies it during the child's life. Where
it cannot be applied, the child keeps the window it runs with and no new bound is recorded
— at load, no amber memory notice can then fire for that child: a working-set total that
only resolves after the load, a `PATCH` the child did not apply (or whose read-back
failed), an unreadable base footprint, unreadable static facts. Each failure is one record — a WARNING naming the
HTTP status or the exception's class name, never the key or a response body; an
unreadable base is the base measurement's own WARNING when it raised, otherwise one INFO
line. A lost answer the read-back settles in the child's favour is one WARNING too, and
the bound is recorded. A failure there never fails the load. A stopped MLX child clears
the trained window and the memory bound with `context_tokens` (`MLX_Engine`'s own
`_stop_server_if_running`; on the CPU engine a stopped child has no allocated window, and
`memory_bound_tokens()` answers `None` with it).

### The memory brake (MLX)

The MLX child's memory grows with the conversation, so when macOS comes under memory
pressure the engine gives some of it back. The control law is a pure module
(`src/engines/memory_brake.py`, table-tested); `MLX_Engine.brake_point(point)` reads
the sensors, applies it and executes the decision. It never raises: a failure is one
WARNING with the exception's class name, and the sample is lost.

**Safe points only, never during a generation.** `turn_start` runs before the turn
resolves its model, and only when the loaded child is the one that turn will use
(`loaded_child_for(llm_id)`, a check with no side effect); a shrink there applies to
that very turn. `turn_end` runs once per turn right after the memory window closes —
conversation and Arena turns alike, and the turns that recover from an error too. `idle`
runs from `_pressure_monitor`, a background task started next to the idle-cleanup
monitor: every 60 s (`BRAKE_IDLE_INTERVAL_S`), under the generation lock, only with a
loaded live child. Every sample runs through `run_reset_shielded`, like a prefix-cache
reset, so the next lock holder drains it; at shutdown the lifespan stops the monitor,
awaits it, and drains any pending reset before it stops the child. An idle sample is
skipped within 55 s of the previous sample or of the end of the last turn (the monitor's
period is the 60 s interval itself, so a tick a few seconds early is not skipped).

**Sensors.** The system pressure level (`kern.memorystatus_vm_pressure_level`: 1 calm,
2 warning, 4 critical — never `kern.memorystatus_level`), the system's swap-outs during
the turn (`turn_end` only; 64 MiB of them count as a warning), the child's footprint
above its base, the size of the last turn's largest request (`last_request_tokens`,
written when the memory window closes: the measured size, else the estimate of the call
in flight for a turn that was abandoned or recovered from an error before any usage
arrived; a turn with neither keeps the previous value; a missing footprint SPI changes
nothing) with the compaction threshold of the conversation that sent it
(`last_request_threshold`, written with it: none for an Arena turn or a child no turn has
measured yet, which means the default 80 %; an idle sample judges the last measured pair),
and the child's share of the memory total. An unreadable pressure level turns
the brake off, with one WARNING per backend process.

**Pressure episodes.** Every action is bounded per episode, not per sample: an episode
opens at the first non-calm sample and ends after 2 calm samples in a row.

| Action | When | What it does |
|---|---|---|
| Shrink | One per episode under warning, one more once critical; never two without a turn between them; only when the last turn's largest request reaches the compaction trigger of the shrunk window, at the threshold of the conversation that sent it (80 % for an Arena turn; the next turn compacts and frees its KV) | The brake's cap becomes `max(4096, 0.75 × working window)`, and the child enforces `min(ceiling, cap)` (`PATCH /v1/settings`) |
| Release | Idle under critical pressure, idle under warning once the user has been away 180 s (`WARNING_RELEASE_IDLE_S`: a reader keeps the prefix cache), `turn_end` under critical pressure; never `turn_start`; only when it would give back 256 MiB or more beyond what the last release left | `POST /v1/cache/reset`, then a non-streamed 1-token request; both barriers it may send are bounded by 30 s. The prefix owner is kept |
| Unload | Idle only: critical at two consecutive idle samples with no turn between them (a calm or warning idle sample in between starts over), at most once per episode, never within 600 s of a brake reload, only for a child holding 5 % or more of the memory total | The idle reap's path, under the lock the monitor holds |
| Grow | Outside an episode, every 3 calm samples | One step back up (`cap / 0.75`), never back to 90 % of the window that hurt while it is remembered (forgotten after 10 calm samples); the cap is dropped once growth reaches the window without brake |

**Why a release is a reset and a request.** Measured on the streamed path every
conversation turn takes (Qwen2.5-0.5B, a 22.9k-token streamed turn, prefix cache on): the
child went from a 0.727 GiB base to 1.833 GiB; the prefix-cache reset alone left it at
1.832 GiB (mlx_vlm's `APCManager.clear` drops the blocks without clearing MLX's buffer
cache, and its streamed path never clears it); a non-streamed 1-token request after it
brought it to about 1.05 GiB, because mlx_vlm's non-streamed chat handler calls
`mx.clear_cache()` after every generation. That clear runs before the answer is sent,
but the footprint the kernel reports follows a moment later, so the engine reads what
the release left, at idle, as the lowest footprint over a bounded 2 s window
(`RELEASE_SETTLE_S`), and at `turn_end` as one read (the turn's last events must not
wait); about 0.33 GiB stays, and it never triggers another release. A release whose reset
or barrier failed records nothing: the next eligible sample tries again, and no
`external` notice is drawn from it.

**Notices.** A shrink sets the notice `acted` (the conversation is now summarized
earlier); pressure the conversation does not cause — what a release would give back is
under 256 MiB and no useful shrink is left — sets `external`, never over `acted`, and
never on a child a brake reload seeded before its episode ends. A release alone sets
none: the user loses nothing visible. A stateful turn end consumes the notice
(`consume_brake_notice`); the runner turns it into a `memory_warning` event (see
[Conversations](../../guides/conversations.md)).

**The carry.** An unload records `_brake_reload` on the engine — the model's alias
(`erudi-<llm id>`), the cap, the window that hurt, the episode (level, shrinks, the
unload itself) and the time. The next spawn of the same alias within 30 minutes
(`BRAKE_RELOAD_TTL_S`) is seeded with it before anything else in
`_read_server_properties`, so it loads with the reduced window, takes no second unload
in the same episode and no extra shrink. Any successful spawn, an idle-timeout reap and
the shutdown forget the record; `brake_reload_pending(llm_id)` lets the runner announce
the reload (`memory_reload`) before it resolves the model.

**Logs.** A shrink, a grow and a release are one INFO line each,
`[MLX_Engine] Memory brake: point=..., level=..., action=..., window=A->B, excess_mib=...`
(a release quotes the excess before and after); an unload is one WARNING, the app having
degraded on its own (see [Logging](../../logging.md)). A reset or a barrier that fails is
its own WARNING (a release barrier's says the memory release is incomplete) and no brake
line follows. A shrink or a grow the child did not apply is the applier's WARNING alone,
and the brake forgets the decision (cap, episode count, notice) as if it never happened.

### The window on the CPU engine

`llama-server` allocates (and touches) the KV cache of its whole window at load, so the
window it is given is what the child costs for its whole life. The CPU engine chooses it
itself (`CPU_Engine._prepare_spawn_context`, `src/engines/gguf_memory.py`) rather than
leaving it to `--fit`, which sizes against total RAM minus 1 GiB on macOS and Linux and
against the free RAM of the moment on Windows:

```
budget = 0.75 × physical RAM − weights − 256 MiB
window = min(trained window, budget / (KV bytes per token + 1 KiB))
```

rounded down to 256 tokens, never under the 4096 floor (a model whose floor does not fit
still loads at the floor: the catalog is never gated on RAM), and folded with the memory
brake's cap for the model (`min(window, cap)`, [below](#the-memory-brake-cpu)). It is
passed as `-c`; `/props` reads it back.

- **Physical RAM** is `psutil.virtual_memory().total` (`memory_pressure.effective_total_ram_bytes`),
  bounded on Linux by our own cgroup's `memory.max` when it is finite (a container; a
  desktop's `max` is ignored, and so are ancestor cgroups). The 0.75 does not subtract
  the app's other processes (backend, embeddings, Postgres, Electron).
- **Weights** are the GGUF on disk (every part of a split family) plus the `mmproj` file
  when the engine loads one. One weights copy is the right term whether the build repacks
  or not: the CPU build repacks the weights of Erudi's preferred quants into anonymous
  memory (measured: 603.90 MiB for a 601 MiB Q8_0, 762.55 MiB for a 763 MiB Q4_K_M, with
  `CPU_REPACK`), after which the mapped file pages go cold; an unrepacked model keeps its
  mapped pages resident instead.
- **KV bytes per token** come from the GGUF's KV header (the `gguf` package, imported
  lazily: the macOS build does not ship it): `Σ over layers (key_length + value_length) ×
  kv_heads[layer] × 2 B` (f16 cache). Measured on the pinned b10883 with one slot, the
  `KV buffer size` llama-server prints is exactly `n_ctx ×` that number (Qwen3-0.6B
  114,688 B/token: 896.00 MiB at 8192 tokens, 3584.00 MiB at 32768; Llama-3.2-1B
  32,768 B/token: 256.00 / 2048.00 MiB at 8192 / 65536). A hybrid architecture's
  recurrent layers declare 0 KV heads and count 0.
- **1 KiB per token** is the Flash-Attention compute buffer: one f16 buffer of
  `n_ctx × n_ubatch` (the default `n_ubatch` is 512), measured exactly on both bench
  models (34.01 → 58.01 MiB over 24,576 tokens, 72.01 → 128.01 MiB over 57,344), and
  independent of the thread count. Its intercept (26 MiB at embedding 1024, 64 MiB at
  2048) is budgeted at 256 MiB for wider models. The argv keeps `flash_attn = auto`,
  which resolves to the FA path on the pinned build; without FA the buffer costs about
  34 KiB per token (290 → 1106 MiB over the same 24,576 tokens).
- **Shapes the formula does not model.** MLA (DeepSeek lineage, `kv_lora_rank`) is
  refused: no `-c`, llama-server's fit sizes it. A sliding window (Gemma lineage) keeps
  the full-attention formula, an over-estimate — such a model gets a smaller window than
  would fit (Gemma-3-4B on 16 GiB: about 66-71k tokens of its 131k) and its window is
  never reported as memory-bound, even when memory does bind it. A GGUF whose header
  cannot be read (or a missing `gguf` package) is one WARNING and no `-c`.

When the budget, not the trained window, binds (and the shape is not a sliding window),
the window is the child's memory bound: `memory_bound_tokens()` reports it, and the amber
`bound` notice quotes the child's allocation at load — KV and compute for the window
plus the weights (`memory_bound_footprint_bytes`). These constants are tied to the pinned
llama.cpp (its `n_ubatch` default, the f16 KV, the FA graph, the repack set): a
submodule bump re-measures them (`llama-server -lv 5 -np 1 --cache-ram 0` at two `-c`
values: the `KV buffer size`, `compute buffer size`, `flash_attn` and `CPU_REPACK`
lines), and the shipped Windows (OpenBLAS) and Linux builds confirm them in QA.

**The post-load check.** Right after `/props`, the child's anonymous memory —
`phys_footprint` on macOS, `Anonymous:` + `Swap:` of `/proc/<pid>/smaps_rollup` on Linux,
the private commit on Windows; all exclude clean file-backed mappings — is compared with
the prediction: `window × (KV bytes per token + 1 KiB) + 64 MiB` (the measured intercept)
`+ one weights copy`, times 1.1 plus 256 MiB. Above it is one WARNING naming both numbers
(`llama-server allocated more than the window budget predicted`), never an action. It
catches a gross mis-prediction, and a fallback to non-FA attention whenever its cost
(`window × ~33 KiB`) clears the headroom — reliably for a big window (a 131k window adds
about 4.3 GiB) or small weights, not for a short window on a heavy model (the 25 % of RAM
the budget leaves covers that case). It is skipped, with one DEBUG line, when a vision
projector is loaded: the encoder's compute buffer is not modelled.

### The memory brake (CPU)

`llama-server` cannot give memory back while it runs: nothing frees without a restart
with a smaller `-c`, and a smaller `-c` always frees. So the CPU engine's brake keeps the
MLX brake's safe points, pacing, episodes, unload and notices, and replaces the release
and the useful-shrink test with a **respawn in place**. The law is
`memory_brake.llama_step`; `BaseLlamaCppEngine._llama_brake_point` executes it and
`CPU_Engine` binds it (`brake_point`, `begin_memory_window`, `memory_bound_tokens`, the
respawn-aware `brake_reload_pending`). `CUDA_Engine` binds the same law over its own
sensors ([below](#the-memory-brake-cuda)).

**Sensors** (`src/engines/memory_pressure.py`, one provider per OS; every Linux and
Windows threshold is a starting value to calibrate on real machines):

| | Pressure level (1/2/4) | Paging counter (the damage) | Anonymous memory (unload share, post-load check) |
|---|---|---|---|
| macOS | `kern.memorystatus_vm_pressure_level` | swap-outs × 16 KiB | `phys_footprint` |
| Linux | PSI: the system's `/proc/pressure/memory` — our cgroup's `memory.pressure` only inside a real container (finite `memory.max`: a desktop's systemd scope counts only our own stalls, and an idle `llama-server` never stalls). Warning at 10 % of `some` stall, critical at 5 % of `full`, on the fraction since the previous sample when 5 s or more passed, else on `avg10`. Without PSI: swap-out ≥ 8 MiB/s or major faults ≥ 200/s warn, swap-out ≥ 64 MiB/s is critical | swap-out bytes (`psutil`); without swap, major faults × the page size (file pages read back from disk) | `Anonymous:` + `Swap:` of `/proc/<pid>/smaps_rollup` |
| Windows | Commit charge `1 − ullAvailPageFile / ullTotalPageFile` (`GlobalMemoryStatusEx`): warning at 85 %, critical at 93 % | `\Memory\Pages Output/sec` read raw through PDH by its English name (× 4 KiB) | private commit |

An unreadable level turns the brake off with one WARNING naming why; a missing paging
counter (PDH denied) is one WARNING too, and the level then decides alone.
`ERUDI_FORCE_MEMORY_PRESSURE=warning|critical` (QA only) forces the level and paging
evidence on every RAM sample of both llama.cpp engines (on CUDA a full-offload child
reads RAM pressure as `external`; the seam's `vram-warning|vram-critical` values force
the spill instead: see [below](#the-memory-brake-cuda)).

**The rules**, per pressure episode:

| Action | When | What it does |
|---|---|---|
| Shrink | One per episode under warning — with at least 8 MiB of paging since the previous sample, when the machine counts paging — one more once critical; never two without a turn between them; only when the window's KV is 10 % of RAM or more | The cap becomes `0.75 × min(window, cap)`, rounded down to a multiple of 256, never under 4096; state only until a respawn applies it |
| Respawn | The cap is below the allocated window, and: critical, at any safe point; or warning, at an idle sample once the user has been away 120 s (two ticks: a reader between messages pays no re-prefill) and less than 180 s — the idle reap (300 s) minus `LLAMA_REAP_MARGIN_S` (120 s): a child about to be reaped is not worth a reload, the window memory applies the cap at its next spawn. At most one per turn: a `turn_start` respawn is not repeated at the same turn's end; the next turn or an idle sample re-arms it | The same model restarts in place with `-c = min(budget window, cap)` |
| Unload | Idle only: critical at two consecutive idle samples with no turn between them, at most once per episode, never within 600 s of a brake reload (a respawn sets no such grace), only for a child holding 5 % or more of RAM | The idle reap's path; the episode goes to `_brake_reload` |
| Grow | Outside an episode, every 3 calm samples (the MLX rule) | State only: never a respawn. The larger window applies at the next spawn of the model |

A window KV under 10 % of RAM means the pressure is not ours: the notice is `external`
(never over `acted`, never on a child a brake reload seeded), and nothing shrinks. A
respawn that applies a shrink sets `acted` (a shrink deferred to an idle respawn is said
only once applied); a warning shrink held back for want of paging sets no notice at all.
A cap the episode never applied (pressure ended before a respawn) is dropped when the
episode ends, with nothing said; the window that hurt is kept.

**Caps are multiples of 256.** llama.cpp pads the window it allocates up to a multiple of
256 (`GGML_PAD(n_ctx, 256)` in `llama-context.cpp`; `/props` reports the padded
value), so every cap the brake sets — a shrink, a grow, the cap folded into `-c` — is
rounded down to a multiple of 256 (never under 4096), and the law compares the allocated
window with the padded cap: an unpadded cap would stay below the window its own respawn
allocates and respawn again at every turn. A `-c` the user pins off that grid
(`ERUDI_CTX=8000`) is compared padded too (8192), so it draws no mismatch WARNING.

**The emergency unload is intended.** Under lasting critical pressure while the user is
away, a respawn is followed by an unload: a critical `turn_end` respawns the child
smaller, the first idle tick arms the unload (no second shrink without a turn), and the
second unloads it; an idle respawn is followed by the unload at the very next tick. The
next message reloads the model with the reduced window. With `ERUDI_CTX` pinned nothing is resized and
no `acted` notice is drawn; the unload still applies.

**The respawn.** Under the generation lock (the sample runs shielded, like every brake
sample), the engine stops the child, picks a port and starts the same GGUF with the
brake's cap handed to the spawn context (`_start_server(..., brake_cap=)`, never left to
the window memory), so the new `-c` is the reduced window.
`_model_id`, the tokenizer and the idle clock are kept (the 300 s reap still sees an
abandoned child); the brake state is carried whole — episode, cap, notice, the
critical-unload clock — with no reload grace. One INFO line,
`[CPU_Engine] Memory brake: point=..., level=..., action=shrink+respawn, window=A->B, footprint_mib=...`.
A warm reload answers in 1-2 s, during which the transient peak is about two weights
copies (the old file pages, reclaimable, and the new repacked copy); at `turn_end` the
turn's last events wait for it. The respawn's readiness probe is bounded by 30 s
(`LLAMA_RESPAWN_PROBE_TIMEOUT_S`, passed to `_start_server` → `_probe_ready` for this
spawn only). **Degraded chain under critical pressure:** critical pressure usually evicts
the page cache, a big model may reload from disk and miss the 30 s; the respawn then
fails like an unload — the child is stopped, the engine state cleared, the episode
recorded in `_brake_reload` — with one WARNING (`action=respawn-failed (...)`; the model
reloads at the next message), which under critical pressure frees the most; the next
message shows the reload state and spawns with the reduced window, and a second failure
surfaces as the load error it is.

**The reload state.** A respawn marks the new child (`brake_respawned`): a respawn at
`turn_start` is said in that same turn (the runner yields `memory_reload` before it
resolves the model, unless the turn already announced a reload); one at `turn_end` or
idle is said by the next turn, through `brake_reload_pending`. The turn's
`begin_memory_window` consumes the mark.

**Three carries.** The live state crosses a respawn whole (`BrakeState.respawned`). An
unload carries the episode to the next spawn of the same model (`_brake_reload`, 30
minutes, forgotten by any successful spawn, a reap and the shutdown). And every change
of the cap or of the window that hurt is written through to a per-model **window
memory** (`_window_memory`, keyed by alias, 30 minutes from its last write, checked at
every read; an entry whose cap and faulty window are both gone is deleted). Every spawn
of the model — a brake reload, a swap back, a reap's reload, a crash respawn — reads it:
the cap folds into `-c` and seeds the brake (an unload's record seeds first). A reap
forgets the unload's record, never the window memory, so growth never returns to 90 % of
the window that hurt while it is remembered, whatever reloads happen.

### The window on the CUDA engine

`CUDA_Engine` sizes `-c` itself, for a **full offload**, against the memory of the GPU it
selects (`_prepare_spawn_context`, `src/engines/cuda_memory.py`):

```
ceiling = min(0.9 × VRAM total, VRAM free at load) − 1 GiB
budget  = ceiling − weights − 256 MiB (compute intercept) − 512 MiB (CUDA context)
window  = min(trained window, budget / (KV bytes per token + compute slope))
```

rounded down to 256 tokens and folded with the brake's cap, like the CPU window. The KV
bytes per token, the weights (GGUF and mmproj on disk) and the shapes the formula refuses
(MLA) or over-estimates (a sliding window, never reported memory-bound) are the CPU
engine's (`gguf_memory`). The window is passed with `-ngl -1 -fitt 1024`.

- **The free VRAM of the load** is read through NVML (`nvmlDeviceGetMemoryInfo`, v2 when the
  binding and the driver have it) on the selected GPU — the largest card the child may
  see. Sizing on 0.9 × total alone would leave llama.cpp's fit dropping layers on any
  desktop that holds 1-3 GiB of VRAM; sizing on the free of the moment means the window
  may differ from one load to the next. A brake respawn never re-reads it (below).
- **The 1 GiB margin** (`CUDA_FREE_MARGIN_BYTES`) is room for the CUDA pool, which grows
  during a generation; an allocation that fails there aborts the child mid-turn
  (`CUDA_CHECK` → `ggml_cuda_error`), which surfaces as the curated child-death error,
  and the next load sizes against the new free memory. The brake never acts
  mid-generation; the margin is what keeps that abort rare.
- **The compute slope** is the Flash-Attention KQ mask, 1 KiB per token, when FA is
  proven for the model — the static mirror of the b10883 CUDA kernel gate
  (`fattn.cu`): equal K and V head sizes in {40, 64, 72, 80, 96, 112, 128, 256}, no
  ALiBi. Otherwise the non-FA attention scores, `n_ubatch (512) × n_head_max × 4 B` per
  token. When llama.cpp disables FA anyway (its `Flash Attention not supported, set to
  disabled` WARN in the child's output, scanned after the probe) one WARNING says the
  window was sized on the wrong slope.
- **The RAM ceiling does not bind** a full offload: the KV and the compute buffer live on
  the device, the host keeps clean mapped file pages and pinned buffers. RAM pressure
  still reaches the child through the brake's sensors.
- **`ERUDI_VRAM_OVERRIDE_GB`** (QA only) stands in for the free VRAM of the sizing; the
  total is then the larger of the override and the real total. It never invents a GPU (NVML
  failed or saw none: it is ignored), and an empty value is unset.
- Every CUDA constant that is not measured on the pinned build (the CUDA context, the
  compute terms, the margins, the spill tiers and baseline, the classification
  tolerance) is a starting value calibrated in QA on a Windows CUDA
  machine and a Linux+NVIDIA one.

**Division of labour with fit.** Erudi fixes the window: fit never changes an explicit
`-c`, and the brake's cap, the window memory, `/context` and the compaction threshold all
read it. Fit fixes the layers: `-ngl -1` is "auto" only while fit is on (at the model
level a negative value means ALL layers, `llama.h:320`), and fit measures the real free
memory (`cudaMemGetInfo`) with the margin Erudi hands it (`-fitt 1024`), so an
optimistic estimate yields a partial offload instead of a dead child.

**The spawn matrix.**

| Case | Arguments | Brake |
|---|---|---|
| Sized load | `-c W -ngl -1 -fitt 1024` | On (`window_sized`), pending the placement below |
| Brake respawn (Windows only: a spill is what drives it) | `-c min(static window, max(4096, cap)) -ngl -2 -fitt 1024`: nothing re-read; fit cannot change an explicit `-ngl` and keeps every layer on the device, where WDDM spills instead of failing | On, its state carried |
| `ERUDI_CTX` pinned | `-c ERUDI_CTX -ngl -1 -fitt 1024` | Sampled, never resized |
| Floor unfit, MLA, GGUF or NVML unreadable | No `-c`, `-ngl -1 -fitt 1024`: fit sizes the window and places the layers (one WARNING) | Off (`window_sized=False`): never sampled |
| The user's `CUDA_VISIBLE_DEVICES` names several GPUs | No `-c`, `-ngl -1 -fitt 1024`: llama.cpp splits the layers across the cards, which a one-card sizing and measurement cannot model (one INFO) | Off |
| NVML sees no GPU | No `-c`, `-ngl 0` (one WARNING) | Off |

A brake respawn reuses the live handle's static window folded with the cap, so a series
of respawns is deterministic and never reads the free VRAM a game left; it never falls
to the fallback shape. A fresh sizing — and a fresh free reading — happens only at a
load: a reap's reload, a swap back, the reload after an unload. On Linux the window is
sized at spawn and never resized while the child runs: there is no spill to react to,
and RAM pressure on a full-offload child is not the window's (below), so the brake there
informs or unloads.

**One GPU.** The child sees one GPU, pinned by its NVML UUID
(`CUDA_VISIBLE_DEVICES=GPU-<uuid>`), immune to ordering differences between NVML and the
CUDA runtime. A `CUDA_VISIBLE_DEVICES` the user set is never overridden: the selection is
restricted to the devices it names (UUIDs by prefix, integers as NVML indexes); an
empty value counts as unset (`backend/.env.example` lists the variable empty). When it
names several devices the window is not sized and the brake is off (the matrix above).

**Out of memory at load.** When a sized load dies with `CUDA_OUT_OF_MEMORY`, the engine
retries once at the 4096 floor (one WARNING, which says the failed allocation may be the
host's pinned memory: `cudaHostAlloc` prints the same); a second failure is the curated
load error. A fit-sized, pinned or respawned child is not retried.

**The placement.** Right after `/props`, before the brake is seeded, the engine
classifies what fit did. "Partial" means some KV is not on the device — layers lost:
that is what breaks the VRAM attribution. llama.cpp's own `offloaded N/M layers to GPU`
line decides when it reaches the child's output (it is an INFO line, hidden at the
default verbosity). Otherwise the child's device bytes — NVML's per-process memory on
Linux, `Dedicated + (Shared − 128 MiB)` on Windows, so a spill at load does not read as
partial — are compared with a floor of what a full offload certainly holds: the weights
without `token_embd.weight` (the input embedding stays on the CPU) and without the MoE
expert tensors, plus the KV the child really allocates (for a sliding-window model the
exact mix of full and sliding layers when the GGUF carries the layer pattern, nothing
when it does not). Partial means more than one layer's worth (net of 64 MiB) is missing;
compute, the CUDA context and the pool are not counted, so a thin missing layer can hide
behind them. An MoE whose experts fit moved to system memory keeps every layer and its
KV on the device: it is a full offload, the brake stays on and its weights are partly in
RAM. Models whose later layers share the KV of earlier ones (Gemma 3n,
`attention.shared_kv_layers`) are not modelled: the KV is over-counted, in the safe
direction — such a child may be classified partial and run brake-off. A partial child runs brake-off (one INFO naming the signal); the window memory keeps
its cap, so the brake re-arms at the next load. No signal at all (Windows without PDH)
is assumed full.

**The post-load device check.** On a full child, its device bytes above
`1.1 × (KV + compute + CUDA context + weights) + 256 MiB` are one WARNING — a
mis-calibrated constant or a Flash-Attention mis-prediction. Skipped with a vision
projector. The CPU engine's host-memory check is skipped for a child with GPU layers
(one DEBUG line).

### The memory brake (CUDA)

`CUDA_Engine` binds the CPU engine's hooks and runs the same law,
`memory_brake.llama_step`, unchanged; `_llama_read_signals` fuses two sensors into its
inputs (`cuda_memory.fuse_signals`):

- **The RAM half** is the CPU engine's (`memory_pressure.read_pressure`, the paging
  counter, the child's anonymous memory).
- **The VRAM half** is the spill. On Windows, WDDM satisfies an allocation that does not
  fit from system RAM (the driver's "CUDA Sysmem Fallback Policy", on by default since
  driver 536.40): the child keeps running, very slowly. NVML has no per-process memory
  under WDDM, so `memory_pressure.read_vram` reads PDH's
  `\GPU Process Memory(pid_<pid>_luid_..._phys_<n>)\Shared Usage` and `Dedicated Usage`
  (English names, gauges) on the adapter holding most of the child's dedicated memory —
  the discrete GPU on an Optimus laptop. The spill is the Shared Usage above the child's
  baseline (its reading right after load, bounded by 128 MiB; a reading above 128 MiB is
  one WARNING at a load and an INFO after a brake respawn). Warning at 256 MiB of spill,
  critical at 1 GiB. PDH denied is one WARNING: spill detection is off, the RAM brake and
  the window sized at spawn stand. On Linux there is no spill: an allocation that does
  not fit fails, so the window is sized to fit, the engine never reads a spill there, and
  the brake sees the RAM half alone.

The level is the higher of the two; the resource at that level decides the attribution
(VRAM on a tie: its lever, the window, is ours):

| Dominant | KV share | Footprint share | Damage |
|---|---|---|---|
| VRAM (spill at warning or worse) | window × KV per token / VRAM | the child's dedicated + shared / VRAM | the spill itself: our bytes in system RAM pay the PCIe bus on every token |
| RAM | 0: the KV is on the device, a smaller `-c` frees no RAM, so the notice is `external` and the window never moves | the child's anonymous memory / RAM | the paging since the previous sample |

A calm spill is never damage (a stable small spill would otherwise count as a paging
burst at every turn end). An unreadable RAM level does not turn the brake off while the
spill reads. Under RAM pressure a full-offload child can still be unloaded after two idle
criticals when its host share passes the 5 % gate: the unload frees its host commit (the
CUDA runtime, cuBLAS workspaces, pinned buffers).

**The QA seam** (`ERUDI_FORCE_MEMORY_PRESSURE`, never set in normal use): `warning` and
`critical` force the RAM half, so on a full-offload child they draw the `external` notice
and, under `critical` while the user is away, can drive the unload — never a shrink.
`vram-warning` and `vram-critical` force the spill reading of `read_vram` to the healthy
baseline plus the tier's threshold (256 MiB or 1 GiB above it; the dedicated VRAM stays
the real one), so on Windows the spill path runs deterministically on a calm machine:
a shrink at the turn end and a `-ngl -2` respawn once the user is away (warning), or a
shrink and a respawn at the next turn boundary (critical), with the `acted` notice and
the reload state. The forced spill is also seen at load (the spill-at-load WARNING, and
possibly the device check's). On Linux the engine never reads the spill, so the VRAM
values do nothing there.

A child whose window Erudi did not size (the fallback shape) or that loaded partly on the
GPU is never sampled: `brake_point` returns before any sensor is read, so nothing — not
even the unload — acts on it.

**Driver policy.** With the NVIDIA Control Panel's "Prefer No Sysmem Fallback", Windows
fails an allocation like Linux: a `-ngl -2` respawn under a VRAM hog fails instead of spilling, the
respawn degrades into the unload-shaped record (one WARNING), the next message reloads
with `-ngl -1`, fit likely places the model partly on the GPU, and the brake stays off
until a later load.

**fit's probe pass** (a no-alloc model and context build) runs on every spawn, brake
respawns included; it counts against the respawn's 30 s probe bound.

`BaseLlamaCppEngine._find_llama_server` tries the configured flavour first and falls back
to the other one: a CUDA-built artifact runs CPU inference fine, whereas the CPU artifact
simply will not use the GPU.

When the model path is a directory, `BaseLlamaCppEngine` picks the GGUF file by quant
preference — `q4_k_m` > `q4_0` > `q5_k_m` > `q8_0` > `f16`, then the smallest remaining
file — skipping `mmproj` sidecars.

## Generation

Streaming does not go through the engine directly. The agent layer
(`backend/src/agents/runner.py`) talks to the child server through
`ChatOpenAI(base_url=...)` and wraps model resolution plus the whole token stream in
`BaseEngine.generation_guard()`:

```python
from src.core import config

async with config.LLM_Engine.generation_guard():
    ...  # resolve the model and stream the turn
```

The guard serializes concurrent requests on one asyncio lock so they cannot thrash the
single-model subprocess, and the idle-cleanup tick shares that same lock, so a model can
never be reaped mid-stream. On exit the idle clock restarts from the end of the
generation. Both take the lock through one helper, `_acquire_generation_lock`, which
first drains a prefix-cache reset an earlier holder left running (next section).

## Prefix cache (MLX)

The `mlx_vlm.server` child keeps an Automatic Prefix Cache pool, sized from the window at
spawn. It holds **one conversation at a time**; the ownership lives on the engine handle
(`prefix_owner`) and is driven by three `BaseEngine` hooks that are no-ops everywhere but
`MLX_Engine` (the llama.cpp engines write nothing):

| Hook | Called by | MLX behaviour |
|---|---|---|
| `claim_prefix(owner)` | the runner, every conversation (`conv:<id>`) or Arena (`arena`) turn, after the model is resolved | Same owner: nothing. A freshly spawned child (`<fresh>`, empty pool): records the owner without a reset. Anything else: reset, then record the owner. |
| `on_history_rewritten()` | the compaction middleware, when it rewrote the history | Reset; the owner is kept. |
| `note_stream_abandoned(handle)` | the chat client (`abandon_hook`, bound at build time) and the runner, when a stream did not end normally | Sets `abandoned` on the handle the stream was built against. |

A reset is `POST /v1/cache/reset` with the child's Bearer key (5 s). A child without a
prefix cache answers `{"enabled": false}`, which counts as success. When the handle is
flagged `abandoned`, a **barrier** goes first: a one-token chat request, the readiness
ping's shape, bounded by the first-chunk ceiling. mlx-vlm's `clear()` pushes every block
back on the free list without guarding against a double push, and a request cancelled
during its last prefill step still releases its blocks after the cancellation. The
barrier's completion proves the cancelled request is settled **only because the child runs
one sequence at a time**: the runner sets `MLX_VLM_MAX_NUM_SEQS=1` in the child's
environment (`_mlx_vlm_server_runner._apply_child_runtime_env`). mlx-vlm's GPU loop admits
new requests before it drains cancellations, and its batch is unbounded by default, so a
wider batch would let the barrier share a step with the cancelled request and finish
first; with one sequence the barrier is admitted only once that request has left the
batch. The cap costs no concurrency — every request to the child is already serialized
behind the generation guard — but it does cost a wait right after an abandon: ANY request
sent then (not only the barrier: the next turn, a title, a summary call) waits for the
cancelled request to leave the batch, typically one prefill step. A failed barrier or reset is one
WARNING and leaves the owner `<dirty>`, so the next claim retries; it never fails a turn.
Each performed reset logs `Prefix cache reset: reason=<conversation_change|arena|compaction>`.

The reset is blocking HTTP, so it runs in a worker thread through `run_reset_shielded`
(`base_engine.py`), which waits for the thread whatever cancellation arrives: an
`asyncio.shield` loop absorbs a native `task.cancel()`, and an
`anyio.CancelScope(shield=True)` around it stops anyio from re-delivering a client
disconnect at every iteration. A compaction reset runs inside a LangGraph node task; the
runner consumes LangGraph from a child task (`src/agents/isolated_stream.py`) that a client
disconnect reaches as ONE native cancellation and that the runner awaits, shielded, before
leaving the guard — so the node, and the reset it waits on, finish inside the guard. As
defense in depth, the reset's task is also registered as `BaseEngine._pending_reset`, and
the next lock acquirer drains it before doing anything should a holder ever leave the
guard while a reset still runs. A reset that raised is logged once by the task's done
callback and never reaches the next holder.

## Memory lifecycle

`BaseEngine` keeps `_model`, `_tokenizer`, `_model_id` and `_last_used` as class
attributes shared across requests. A cleanup monitor started in the FastAPI lifespan
(`core/api.py`, `start_cleanup_task()`) ticks every 30 seconds and unloads the model once
it has been idle for longer than `_max_idle_time` (300 seconds, `base_engine.py`), so about 300 to
330 seconds after its last use. A tick that finds nothing to reap returns before the lock, so it
never queues behind a generation; the idle check is repeated under the lock before any reap.
A generation in flight cannot be reaped because the idle tick (`_cleanup_tick`) takes the
same lock as `generation_guard`, through the same `_acquire_generation_lock`: it waits until
the generation releases the lock, by which point `_last_used` has been refreshed to the end
of that generation. On every engine a second monitor, `start_pressure_task()`, takes
the memory brake's idle samples every 60 seconds under the same lock (see
[The memory brake (MLX)](#the-memory-brake-mlx), [(CPU)](#the-memory-brake-cpu) and
[(CUDA)](#the-memory-brake-cuda)); it starts for an engine that overrides `brake_point`, and
on CUDA a child whose window Erudi did not size, or that loaded partly on the GPU, is never
sampled.

## Embeddings

Embeddings are **not** an engine. The Knowledge Base uses `E5Embeddings`
(`backend/src/ingestion/embeddings.py`), a LangChain `Embeddings` implementation over a
resident `intfloat/multilingual-e5-small` singleton (384 dimensions), cached inside the
app data directory. The `passage: ` and `query: ` prefixes required by the e5 family are
applied by `embed_documents` and `embed_query` respectively and are mandatory. Vectors go
to `rag.kb_chunks` through langchain-postgres' `PGVectorStore`. See the
[Knowledge Base guide](../../guides/knowledge_base.md).

## Error handling

Engines raise the structured exceptions from `src.core.exceptions`:

```python
from src.core.exceptions import EngineException, ModelLoadingException, GenerationException

raise ModelLoadingException(f"Failed to load {model_path}", trace=str(e))
```

`EngineException` (`LLM_ENGINE_FAILURE`) covers subprocess and server lifecycle failures,
`ModelLoadingException` covers load failures, and `GenerationException` covers failures
during a stream. See [Exception Handling](../exceptions.md).

## Building llama.cpp

The `llama-server` binary is never committed. `backend/forks/llama-cpp` is a git
submodule tracking `ggml-org/llama.cpp` directly, pinned to a release tag, so a
fresh clone needs:

```bash
git submodule update --init --recursive
```

A clone made before the submodule moved to upstream still has the old remote
recorded, and needs one extra command first:

```bash
git submodule sync --recursive
```

Bumping the pin is `git -C backend/forks/llama-cpp fetch --tags`, checking out
the tag, and committing the new pointer. Nothing is patched on top of upstream:
the build scripts pass every option they need on the cmake command line.

Then build for your platform, from the repository root:

```bash
# macOS Apple Silicon (local CPU engine, development convenience)
bash scripts/dev/backend/build-llamacpp-cpu-macos-silicon.sh

# Linux
bash scripts/dev/backend/build-llamacpp-cpu-linux.sh
bash scripts/dev/backend/build-llamacpp-cuda-linux.sh
```

```powershell
# Windows
.\scripts\dev\backend\build-llamacpp-cpu-win.ps1
.\scripts\dev\backend\build-llamacpp-cuda-win.ps1
```

The shipped CPU and CUDA engines target **Windows and Linux**. On macOS the shipped
engine is MLX; the macOS CPU build exists so the llama.cpp path can be exercised locally
on a Mac, not as a distribution target.

### What the macOS Apple Silicon script does

`scripts/dev/backend/build-llamacpp-cpu-macos-silicon.sh`:

1. Checks the toolchain (`clang` from the Xcode Command Line Tools, `curl`, `tar`).
2. Installs `cmake` into `backend/venv` — no global or Homebrew install.
3. Offers to delete a previous `backend/forks/llama-cpp/build-cpu` and
   `backend/artifacts/llama-cpp/cpu`.
4. Configures CMake for an arm64 CPU-only build. It deliberately does **not** set
   `CMAKE_OSX_ARCHITECTURES`, letting it default to the host arm64:

   | Flag | Why |
   |---|---|
   | `GGML_CPU=ON`, `GGML_NATIVE=ON` | let ggml pick the Apple M-series optimized kernels |
   | `GGML_ACCELERATE=ON` | link Apple's Accelerate framework for BLAS |
   | `GGML_BLAS=OFF` | no third-party BLAS |
   | `GGML_OPENMP=OFF` | AppleClang ships no libomp; skips a slow probe |
   | `GGML_METAL=OFF` | the CPU engine must not call Metal |
   | `GGML_CUDA/HIP/VULKAN/SYCL/RPC/WEBGPU=OFF` | CPU backend only |
   | `BUILD_SHARED_LIBS=ON` with `CMAKE_INSTALL_RPATH=@executable_path/../lib` | keeps the install tree self-contained |

5. Builds into `backend/forks/llama-cpp/build-cpu/` and installs into
   `backend/artifacts/llama-cpp/cpu/`.

Verify the result:

```bash
backend/artifacts/llama-cpp/cpu/bin/llama-cli -h
```

The other scripts follow the same shape with their platform's toolchain and flags.

### Build outputs are never committed

`backend/forks/llama-cpp/build-*/` and `backend/artifacts/llama-cpp/` are git-ignored.
Build outputs are environment-specific; CI rebuilds them per OS and bundles only the
artifact matching the target.

## Testing

Engine tests mock the child process rather than loading real models:

```bash
cd backend && pytest tests/test_engines.py -x
cd backend && pytest tests/ -m mlx_only      # MLX integration, Apple Silicon only
```

## See also

- [Architecture](../../architecture.md) — how engines fit into the backend
- [Engines Reference](../../reference/engines.md) — API documentation
- [Hardware guide](../../guides/hardware.md) — detection and performance scores
- [Backend Launcher](../../guides/backend-run.md) — ports and lifecycle events
