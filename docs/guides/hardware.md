# Hardware Detection & Performance

How Erudi detects your hardware, which inference backend it picks, and what the
performance scores mean.

## Overview

Erudi detects the machine at startup and routes inference to the best available backend:

- **Apple Silicon (M1/M2/M3/M4)**: the **MLX** backend, running `mlx_vlm.server` on
  unified memory
- **NVIDIA GPUs**: the **CUDA** backend, running the CUDA build of `llama-server`
- **CPU fallback**: the CPU build of `llama-server`, on Windows and Linux

## How detection works

Three things decide the engine, in this order: the developer override, the hardware, and
the user's own preference.

**1. The override.** `ERUDI_FORCE_CPU=1` returns `CPU_Engine` and skips everything below.
It is the supported way to exercise the CPU path on a GPU machine, and it wins over the
persisted preference; see
[`backend/.env.example`](https://github.com/erudi-app/erudi/blob/main/backend/.env.example).

**2. The hardware.** `BaseEngine.get_engine()` (`backend/src/engines/base_engine.py`)
dispatches on `platform.system()` and `platform.machine()`, and detects NVIDIA GPUs
through `pynvml` (not PyTorch):

1. macOS with an `arm` machine string → `MLX_Engine`
2. Windows or Linux where `pynvml.nvmlDeviceGetCount() > 0` → `CUDA_Engine`
3. Otherwise → `CPU_Engine`

```text
Priority: MLX > CUDA > CPU
```

**3. The preference.** `user_settings.inference_backend` holds `auto` (the default) or
`cpu`. The lifespan reads it after the migrations and the catalog population — the
earliest point where the column is guaranteed to exist — and replaces `CUDA_Engine` with
`CPU_Engine` when it says `cpu`. Both engines carry `FORMAT_TAG = "gguf"`, so the catalog
reconciled a moment earlier under CUDA is identical under CPU; the swap changes what runs
the model, not what the app offers. The preference is inert on Apple Silicon. It is read
once per boot, so changing it in Settings restarts the backend.

The chosen engine is logged at startup (`Engine chosen: ...`).

### The CUDA pre-flight

`nvmlDeviceGetCount() > 0` says an NVIDIA card is present, not that it can run the CUDA
`llama-server` Erudi ships. When the selected engine is `CUDA_Engine`, the lifespan reads
two more NVML values — the compute capability of the first device, and the CUDA version
the driver provides — and compares them to the floors in
`backend/src/engines/cuda_compatibility.py`:

| Verdict | Condition | Event code |
|---|---|---|
| Card below CUDA 12's floor | compute capability < 5.0 | `CUDA_COMPUTE_CAPABILITY_TOO_LOW` |
| Driver too old to JIT our PTX | driver CUDA < 12.8, card has no native code | `CUDA_DRIVER_TOO_OLD` |
| Driver below the runtime baseline | driver CUDA < 12.0, card has native code | `CUDA_DRIVER_TOO_OLD` |

A verdict is emitted as an `engine_notice` lifecycle event on the same newline-JSON stdout
channel as `starting` / `ready`, carrying the code, the GPU name, the readings and a
one-line summary. The frontend shows it as a decision once the app is past `ready`.
**Nothing is applied automatically**: the app proposes processor mode, and only the
persisted preference (or reinstalling with the CPU build) changes the engine.

An unreadable NVML value produces no verdict at all and a `WARNING` in the log. Guessing
from a failed read is how a healthy card gets told its driver is too old, so silence is
the safe answer; the runtime classification below still catches the failure if it happens.

A failure that slips past the pre-flight is caught when the child dies: the captured tail
is matched against the strings ggml prints and the chat stream's error event carries the
matching code — `CUDA_COMPUTE_CAPABILITY_TOO_LOW`, `CUDA_DRIVER_TOO_OLD`,
`CUDA_OUT_OF_MEMORY`, or `CUDA_ERROR` for anything else CUDA-shaped. The same dialog
opens, with the trace in a copyable block.

### GPU compatibility matrix

**To run Erudi** you need an NVIDIA driver, nothing else: the installer carries the CUDA
runtime. **To build it** you need a CUDA 12.x toolkit — 12.8 specifically if you want
native code for RTX 50 cards, since it is the first that can emit `sm_120`.

The CUDA build compiles `llama-server` for this architecture list (see
`scripts/dev/backend/build-llamacpp-cuda-linux.sh` and `build-llamacpp-cuda-win.ps1`):

```text
50-virtual;61-virtual;70-virtual;75-virtual;80-virtual;86-real;89-real  (+120-real on CUDA 12.8)
```

`-real` is native SASS the driver runs as-is. `-virtual` is PTX, which the driver
JIT-compiles on first run — and NVIDIA's minor-version compatibility explicitly excludes
PTX, so a card that depends on it needs a driver at least as new as the toolkit that
produced the PTX.

| Generation | Compute capability | In the binary | Driver needed |
|---|---|---|---|
| Kepler and older | ≤ 3.7 | nothing | Not supported at all — CUDA 12 dropped it |
| Maxwell (GTX 900) | 5.0, 5.2 | PTX | 570+ (CUDA 12.8) |
| Pascal (GTX 10) | 6.0, 6.1 | PTX | 570+ (CUDA 12.8) |
| Volta (Titan V) | 7.0 | PTX | 570+ (CUDA 12.8) |
| Turing (RTX 20, GTX 16) | 7.5 | PTX | 570+ (CUDA 12.8) |
| Ampere A100 | 8.0 | PTX | 570+ (CUDA 12.8) |
| Ampere (RTX 30) | 8.6 | native SASS | 525+ (CUDA 12.0) |
| Ada (RTX 40) | 8.9 | native SASS | 525+ (CUDA 12.0) |
| Hopper (H100) | 9.0 | PTX | 570+ (CUDA 12.8) |
| Blackwell (RTX 50) | 12.0 | native SASS | 525+ (CUDA 12.0) |

Anything newer than Blackwell runs from the `80-virtual` PTX, which is what upstream
llama.cpp intends — and therefore needs the 570+ driver like every other PTX case.

### What gets detected

**All backends**

- CPU model and core count
- Total and available RAM
- Disk space, total and available
- Operating system and architecture

**MLX (Apple Silicon)**

- Chip model (for example "M3 Max")
- GPU core count
- Neural Engine TOPS
- Memory bandwidth
- Unified memory capacity

**CUDA (NVIDIA)**

- GPU name
- CUDA core count and compute capability
- CUDA runtime version
- VRAM, total and available
- Memory bandwidth

**CPU**

- Logical core count
- Estimated memory bandwidth
- No GPU acceleration

## Performance scores

Each backend computes a 0-100 inference score from a weighted mix of compute, memory
bandwidth, memory capacity, and storage. There is a single score: inference. Erudi does
not train or fine-tune models.

### Score labels differ by backend

The numeric score is comparable across backends, but the **label thresholds are not
shared** — each engine has its own scale. Do not compare labels across machines with
different backends.

| Source | Scale |
|---|---|
| `MLX_Engine` (`backend/src/engines/mlx_engine.py`) | 80 Excellent · 60 Good · 40 Fair · 20 Poor · below Weak |
| `CPU_Engine` (`backend/src/engines/cpu_engine.py`) | 85 Amazing · 70 Excellent · 55 Very Good · 40 Good · 25 Medium · 10 Poor · below Terrible |
| `CUDA_Engine` (`backend/src/engines/cuda_engine.py`) | 90 Amazing · 80 Excellent · 70 Very High · 60 High · 50 Good · 40 Medium · 30 Bad · 20 Very Bad · 10 Poor · below Terrible |
| Hardware service (`backend/src/domains/hardware/services.py`) | 80 Excellent · 60 Good · 40 Fair · 20 Poor · below Weak |

The label the UI displays comes from the hardware service, applied to the boosted score.

### Boosted scores in the UI

The UI shows a **boosted score**: the raw score plus 20 points, capped at 100
(`Hardware_Service.calculate_boosted_scores`).

```text
Raw 65/100 → UI 85/100
Raw 82/100 → UI 100/100
```

Both values are returned by the API, so the boost is always visible for debugging.

### Recommended model size

Alongside the scores, the backend computes the model-size window the machine runs
comfortably at 4-bit, in billions of parameters: `recommended_param_min` and
`recommended_param_max` on `GET /erudi/hardware/app_startup`
(`recommended_param_range` in `backend/src/domains/hardware/services.py`).

The window is the smaller of two limits — what fits in usable memory after an overhead
reserve, and what is fast enough given memory bandwidth — clamped to a floor and a
ceiling. The model library uses it to tell you which catalog entries fit your machine.

## Refreshing hardware data

Hardware is re-detected on every app launch. Force a re-detection without restarting:

```bash
curl -X POST http://127.0.0.1:27182/erudi/hardware/refresh
```

Refresh after a RAM upgrade, a GPU change, or if the reported specs look wrong.

## API endpoints

### Startup summary

```bash
curl http://127.0.0.1:27182/erudi/hardware/app_startup
```

Returns the backend type, the boosted score and its label, the raw score, and the
recommended parameter range.

### Detailed diagnostics

```bash
curl http://127.0.0.1:27182/erudi/hardware/detailed
```

Returns the full profile with the score breakdown and both raw and boosted values.

### Refresh

```bash
curl -X POST http://127.0.0.1:27182/erudi/hardware/refresh
```

Forces re-detection and updates the stored profile.

## Backend differences

Expected relative throughput for a similarly sized model:

| Backend | Relative speed | Best for |
|---|---|---|
| MLX (Apple Silicon) | baseline | Macs; good balance of speed and memory |
| CUDA (recent NVIDIA GPU) | faster | Windows/Linux with a discrete GPU |
| CPU | much slower | machines with no supported GPU |

### Memory model

- **Apple Silicon** uses unified memory shared between CPU and GPU, which is why a Mac
  can hold a larger model than its GPU-only equivalent.
- **NVIDIA** uses dedicated VRAM; the CUDA engine offloads as many layers as fit and
  leaves the rest on the CPU.
- **CPU** uses system RAM only.

## Troubleshooting

### Hardware shows an error in the UI

1. Check the backend is running: `curl http://127.0.0.1:27182/erudi/health/`
2. Look for a `HARDWARE_ERROR` entry in `backend/logs/backend.log`
3. Force a refresh with `POST /erudi/hardware/refresh`

### The wrong backend was selected

```bash
# What the selector sees
python -c "import platform; print(platform.system(), platform.machine())"

# NVIDIA detection, the same way the backend does it
python -c "import pynvml; pynvml.nvmlInit(); print(pynvml.nvmlDeviceGetCount())"
```

- On macOS, MLX requires an `arm64` machine string. Running under Rosetta reports
  `x86_64` and the selector falls through.
- On Windows and Linux, CUDA requires `pynvml` to report at least one device; check the
  driver with `nvidia-smi`.
- Confirm `ERUDI_FORCE_CPU` is not set in your environment.
- Check the inference engine in Settings: `Processor only` pins `CPU_Engine` even on a
  machine with a working GPU. `GET /erudi/user_settings/` reports the stored value.

### A low score

The score reflects sustained inference throughput, so it is affected by available RAM,
background load, and thermal throttling. Closing other applications and improving cooling
both help. A mid-range score still runs the models inside your recommended parameter
range.

## FAQ

**Can I use MLX and CUDA at the same time?**
No. One backend is selected per session by the detection cascade.

**Can I choose the backend manually?**
Partly. Settings → Inference engine offers `Automatic` (the hardware decides) and
`Processor only`, which pins the CPU engine on a machine that would otherwise use its
NVIDIA GPU. Erudi restarts its engine to apply the change. There is no way to force the
GPU on hardware where it was not detected, and no way to pick MLX or CUDA specifically.
`ERUDI_FORCE_CPU=1` remains the developer override and wins over the setting.

**Erudi says my graphics card or driver is too old. What now?**
Update the NVIDIA driver if the message names a driver version — that is the real fix.
If the card itself is below compute capability 5.0, no driver helps: switch to processor
mode, or reinstall with the processor build. The dialog offers both, and nothing changes
until you choose.

**Does a higher score mean better answers?**
No. It means faster generation. Answer quality depends on the model.

**What if I have several GPUs?**
The CUDA backend uses the primary GPU.

## Technical details

- [Hardware Domain Reference](../reference/hardware.md) — schemas and endpoints
- [Engines Architecture](../dev/architecture/engines.md) — engine hierarchy and lifecycle
