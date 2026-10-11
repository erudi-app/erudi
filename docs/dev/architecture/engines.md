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
        ├── CPU_Engine           (Popen + llama-server CPU, -ngl 0, ports 27200-27299)
        └── CUDA_Engine          (Popen + llama-server CUDA, -ngl <computed>, 27200-27299)
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
resolve).

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
| Window resolution | **Fit at load**: with no `-c`, the server itself resolves the window to the model's trained window and *continuously reduces* it against measured free memory, down to a 4096 floor, logging `context size reduced from X to Y` (`--fit`, ON by default in the pinned b10883) | **Preflight bound at spawn, lowered by memory after load**: the engine reads the trained window from the local artifact's `config.json` (`read_local_generation_hints`) and passes it as `--max-kv-size` — a validator, not a rotating cache. Once the child is up and its base footprint measured, a memory ceiling below the trained window lowers that bound live (see below), down to a 4096 floor; under memory pressure the memory brake moves it again |
| `ERUDI_CTX` set | Passed verbatim as `-c`; llama-server honours an explicit `-c` unchanged ("no change") | Not read — the bound always comes from the artifact |
| Overflow behaviour | Honest 400 carrying `n_prompt_tokens` and `n_ctx` | Honest 400 naming the exact budget (`prompt + max generation` vs `MAX_KV_SIZE`); *without* the bound, an oversized prompt would return 200 with silently degenerate output |
| How the allocated window is known | **Read back after boot**: `_read_server_properties` (called by `_start_server` right after the probe) does one bounded `GET /props` and stamps `default_generation_settings.n_ctx`, `chat_template_caps` and `chat_template` on the handle | **Known at spawn, rewritten when memory binds**: the bound just passed *is* the window; `_read_server_properties` measures the child's base footprint and, when memory binds, stamps the memory bound as the window |
| No derivable window | Cannot happen — the server always resolves one | No `--max-kv-size` (today's unbounded behaviour, one WARNING); a number is never invented |

Either way, a `/props` failure or a missing window degrades to `context_tokens = None`
(one WARNING) and the model still loads; consumers treat `None` as "window unknown". A
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
`_stop_server_if_running`; the llama.cpp handles never carry those keys).

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
nothing), and the child's share of the memory total. An unreadable pressure level turns
the brake off, with one WARNING per backend process.

**Pressure episodes.** Every action is bounded per episode, not per sample: an episode
opens at the first non-calm sample and ends after 2 calm samples in a row.

| Action | When | What it does |
|---|---|---|
| Shrink | One per episode under warning, one more once critical; never two without a turn between them; only when the last turn's largest request reaches 80 % of the shrunk window (the next turn compacts and frees its KV) | The brake's cap becomes `max(4096, 0.75 × working window)`, and the child enforces `min(ceiling, cap)` (`PATCH /v1/settings`) |
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
(`core/api.py`, `start_cleanup_task()`) ticks every 300 seconds and unloads the model once
it has been idle for longer than `_max_idle_time` (300 seconds, `base_engine.py`).
A generation in flight cannot be reaped because the idle tick (`_cleanup_tick`) takes the
same lock as `generation_guard`, through the same `_acquire_generation_lock`: it waits until
the generation releases the lock, by which point `_last_used` has been refreshed to the end
of that generation. On Apple Silicon a second monitor, `start_pressure_task()`, takes the
memory brake's idle samples every 60 seconds under the same lock (see
[The memory brake](#the-memory-brake-mlx)); on the llama.cpp engines it is a no-op.

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
