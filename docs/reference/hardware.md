# Hardware Domain

Hardware detection and inference scoring for the three engine backends: MLX (Apple Silicon), CUDA (NVIDIA) and CPU fallback.

## Overview

Detection is delegated to the engine selected at startup by `BaseEngine.get_engine()`; the domain only persists the result and serves it. The engine that ends up selected can differ from what detection chose: `user_settings.inference_backend` pins the CPU engine when the user asked for it, and the CUDA pre-flight (`src/engines/cuda_compatibility.py`) reports — without changing anything — a GPU the bundled CUDA build cannot drive. See [Hardware Detection](../guides/hardware.md).

```
Endpoints -> Service -> Repository -> HardwareProfile entity -> PostgreSQL
                 |
             Engine (MLX / CUDA / CPU)
```

The profile is a singleton row. Only an **inference** score is computed and stored — there is no fine-tuning or training score anywhere in the codebase.

## API endpoints

The router is mounted with the `/hardware` prefix under the app-wide `/erudi` prefix, so the full paths are `/erudi/hardware/...`.

### GET /erudi/hardware/app_startup

Minimal payload consumed by the renderer on boot (`HardwareAppStartupInfo` in `schemas.py`). All six fields are required.

```json
{
  "backend_type": "mlx",
  "global_inference_score": 92.0,
  "global_inference_label": "Excellent",
  "raw_inference_score": 72.0,
  "recommended_param_min": 4.0,
  "recommended_param_max": 14.0
}
```

- `global_inference_score` is the boosted score, `raw_inference_score` the engine output.
- `recommended_param_min` / `recommended_param_max` are the model-size window (in billions of parameters) that drives the "Models For You" list on the model page.

### GET /erudi/hardware/detailed

Full diagnostics (`DetailedHardwareInfo`): the backend-specific hardware block, the typed performance breakdown, and the boosted score.

```json
{
  "hardware": { "backend_type": "mlx", "...": "backend-specific fields" },
  "performance_breakdown": {
    "compute_score": 95.0,
    "memory_bandwidth_score": 92.0,
    "memory_capacity_score": 88.0,
    "cpu_performance_score": 75.0,
    "disk_score": 75.0
  },
  "boosted_inference_score": 92.0
}
```

The raw score stays available inside `hardware.raw_inference_score`.

### POST /erudi/hardware/refresh

Re-runs detection through the engine and updates the stored profile. No request body.

```json
{
  "message": "Hardware profile refreshed successfully",
  "backend_type": "mlx"
}
```

Useful after a hardware change, or to refresh the dynamic fields (`available_memory_gb`, `disk_available_gb`) without restarting the app.

## Schemas

`hardware/schemas.py` builds the response types from a discriminated union keyed on `backend_type`:

- `BaseHardwareInfo` — fields every backend provides (CPU model, memory, disk, `raw_inference_score`, `global_inference_label`, `cpu_score`, `memory_score`).
- `MLXHardwareInfo` — adds `mlx_chip_model`, `mlx_gpu_cores`, `mps_available`, `neural_engine_tops`, `estimated_tflops`, `memory_bandwidth_gbs`, `gpu_score`, `unified_memory`.
- `CUDAHardwareInfo` — adds `gpu_name`, `cuda_cores`, `cuda_version`, `compute_capability`, `vram_total_gb`, `vram_available_gb`, `estimated_tflops`, `memory_bandwidth_gbs`, `gpu_score`.
- `CPUHardwareInfo` — adds `compute_units`, `cpu_performance_units`, `accelerator_available`; `gpu_score` is always `0.0`.
- `PerformanceBreakdown` — typed component scores (`compute_score`, `memory_bandwidth_score`, `memory_capacity_score`, `cpu_performance_score`, optional `disk_score`).

## Scoring

Each engine implements `get_performance_evaluation()` and returns a single 0-100 inference score built from normalized component scores.

### MLX (`engines/mlx_engine.py`)

| Component | Weight | Normalization (= 100 points) |
|---|---|---|
| GPU compute | 35 % | 20 TFLOPS |
| Memory bandwidth | 30 % | 400 GB/s |
| Memory capacity | 20 % | 64 GB |
| Neural Engine | 10 % | 20 TOPS |
| CPU | 5 % | 20 CPU units (performance cores x 2.5) |

### CUDA (`engines/cuda_engine.py`)

Weights live in `_WEIGHTS_INFERENCE`, normalization factors in `_NORM_INFERENCE`.

| Component | Weight | Normalization (= 100 points) |
|---|---|---|
| GPU compute (`gpu_compute`) | 40 % | 80 TFLOPS (BF16/FP16) |
| Memory bandwidth (`gpu_bw`) | 30 % | 500 GB/s |
| VRAM (`gpu_vram`) | 15 % | 12 GB |
| CPU single-core (`cpu_single`) | 5 % | 3.6 GHz |
| System RAM (`sys_ram`) | 5 % | 24 GB |
| PCIe (`pcie`) | 5 % | 32 (Gen3 x16 / Gen4 x8) |

### CPU (`engines/cpu_engine.py`)

Weights live in `INF_WEIGHTS`. Memory bandwidth is estimated, not measured: `1.5 GB/s per core`.

| Component | Weight | Normalization (= 100 points) |
|---|---|---|
| CPU cores | 40 % | 64 cores |
| Memory capacity | 30 % | 128 GB |
| Memory bandwidth | 20 % | 100 GB/s (estimated) |
| Disk | 10 % | 500 GB available |

### Display boost and labels

`Hardware_Service.calculate_boosted_scores()` returns both the raw score and a boosted one for the UI:

```python
boosted_inf = min(100.0, raw_inf + 20.0)
```

The label is derived from the **boosted** score by `Hardware_Service._get_label()`:

| Boosted score | Label |
|---|---|
| >= 80 | Excellent |
| >= 60 | Good |
| >= 40 | Fair |
| >= 20 | Poor |
| < 20 | Weak |

## Service layer

```python
from src.domains.hardware.services import Hardware_Service

service = Hardware_Service(repository)

profile = service.get_or_create_profile()   # cached singleton, detects on first call
scores = service.calculate_boosted_scores(profile)
service.warm_up(duration_seconds=5)         # warm the accelerator before benchmarking
profile = service.refresh_profile()         # force re-detection
db.commit()
```

## Engine contract

Every engine implements the same three hardware class methods:

```python
@classmethod
def get_hardware_info(cls) -> Dict[str, Any]: ...

@classmethod
def warm_up_accelerator(cls, duration_seconds: float) -> bool: ...

@classmethod
def get_performance_evaluation(cls) -> Dict[str, Any]: ...
```

`get_flat_hardware_data()` flattens the nested detection dict into the columns the entity stores.

## Database entity

`HardwareProfile` (`entities/HardwareProfile.py`) is a singleton row with backend-specific columns left nullable. It stores `global_inference_score` and `global_inference_label` (no fine-tuning columns) plus the JSON `performance_breakdown`.

It has **no** `compute_units` column: the CPU count is stored in `cpu_performance_units` and cast to an int when the endpoint builds `CPUHardwareInfo`.

## Frontend

`frontend/src/utils/hardwareTransform.js` exports a single helper, `transformAppStartupInfo`, used to normalize the `/erudi/hardware/app_startup` payload for the renderer.

## Troubleshooting

**Stale hardware data** — force a re-detection:

```bash
curl -X POST http://127.0.0.1:27182/erudi/hardware/refresh
```

**Missing backend-specific fields** — branch on the `backend_type` discriminator before reading `mlx_*`, `cuda_*` or CPU-only fields; each backend only populates its own.

## Code reference

::: src.domains.hardware.endpoints

::: src.domains.hardware.schemas

::: src.domains.hardware.services
