# Engines Architecture

`backend/src/engines/` holds every inference engine. All three run an
OpenAI-compatible HTTP server in a child process and stream from it over SSE.

## Hierarchy

```text
BaseEngine
└── BaseChatServerEngine        ← shared: port pick, /health + chat-ping probe,
    │                             SSE byte-buffer parser, atexit storage,
    │                             idle-cleanup active marker, kwarg translation
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
| Window resolution | **Fit at load**: with no `-c`, the server itself resolves the window to the model's trained window and *continuously reduces* it against measured free memory, down to a 4096 floor, logging `context size reduced from X to Y` (`--fit`, ON by default in the pinned b10883) | **Preflight bound at spawn**: the engine reads the trained window from the local artifact's `config.json` (`read_local_generation_hints`) and passes it as `--max-kv-size` — a validator, not a rotating cache |
| `ERUDI_CTX` set | Passed verbatim as `-c`; llama-server honours an explicit `-c` unchanged ("no change") | Not read — the bound always comes from the artifact |
| Overflow behaviour | Honest 400 carrying `n_prompt_tokens` and `n_ctx` | Honest 400 naming the exact budget (`prompt + max generation` vs `MAX_KV_SIZE`); *without* the bound, an oversized prompt would return 200 with silently degenerate output |
| How the allocated window is known | **Read back after boot**: `_read_server_properties` (called by `_start_server` right after the probe) does one bounded `GET /props` and stamps `default_generation_settings.n_ctx`, `chat_template_caps` and `chat_template` on the handle | **Known at spawn**: the bound just passed *is* the window; `_read_server_properties` stays the base no-op |
| No derivable window | Cannot happen — the server always resolves one | No `--max-kv-size` (today's unbounded behaviour, one WARNING); a number is never invented |

Either way, a `/props` failure or a missing window degrades to `context_tokens = None`
(one WARNING) and the model still loads; consumers treat `None` as "window unknown". A
stopped child clears `context_tokens` on its handle: a dead child holds no KV memory,
so its allocated window dies with it.

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
generation.

## Memory lifecycle

`BaseEngine` keeps `_model`, `_tokenizer`, `_model_id` and `_last_used` as class
attributes shared across requests. A cleanup monitor started in the FastAPI lifespan
(`core/api.py`, `start_cleanup_task()`) ticks every 300 seconds and unloads the model once
it has been idle for longer than `_max_idle_time` (300 seconds, `base_engine.py`).
While a generation is in flight the active marker `_last_used = None` makes
`_should_cleanup()` return `False`.

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
