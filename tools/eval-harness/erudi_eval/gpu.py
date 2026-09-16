"""NVIDIA GPU memory: NVML (nvidia-ml-py) when importable, else `nvidia-smi`, else unavailable.

On Apple Silicon GPU memory is unified and already inside `phys_footprint`;
this module reports `source: none` there.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Any

MB = 1024 * 1024


def parse_smi_gpus(text: str) -> list[dict[str, Any]]:
    """`nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version --format=csv,noheader,nounits`."""
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "total_mb": _num(parts[2]),
                "used_mb": _num(parts[3]),
                "util_pct": _num(parts[4]),
                "driver": parts[5],
            }
        )
    return gpus


def parse_smi_apps(text: str) -> dict[int, float | None]:
    """`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits`."""
    out: dict[int, float | None] = {}
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        pid, used = int(parts[0]), _num(parts[1])
        out[pid] = None if used is None else (out.get(pid) or 0) + used
    return out


def _num(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None  # "[N/A]" (e.g. per-process memory under WDDM)


class GpuReader:
    def __init__(self, min_interval: float = 2.0):
        self.source = "none"
        self.error: str | None = None
        self._nvml = None
        self._handles: list = []
        self._smi = None
        self._min_interval = min_interval
        self._cache: dict[str, Any] = {}
        self._cache_at = 0.0
        try:
            import pynvml  # provided by nvidia-ml-py

            pynvml.nvmlInit()
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
            self._nvml = pynvml
            self.source = "nvml"
        except Exception as e:  # noqa: BLE001 - no library, no driver, no GPU: all mean "try nvidia-smi"
            self.error = f"nvml: {type(e).__name__}"
            self._smi = shutil.which("nvidia-smi")
            if self._smi:
                self.source = "nvidia-smi"

    def read(self) -> dict[str, Any]:
        """{'source', 'gpus': [...], 'per_pid_mb': {pid: mb|None}, 'errors'}."""
        if self.source == "none":
            return {"source": "none", "detail": self.error}
        if time.monotonic() - self._cache_at < self._min_interval and self._cache:
            return self._cache
        try:
            data = self._read_nvml() if self._nvml else self._read_smi()
        except Exception as e:  # noqa: BLE001 - recorded, sampling continues
            data = {"source": self.source, "error": f"{type(e).__name__}: {e}"}
        self._cache, self._cache_at = data, time.monotonic()
        return data

    def _read_nvml(self) -> dict[str, Any]:
        nv = self._nvml
        gpus, per_pid = [], {}
        for i, h in enumerate(self._handles):
            mem = nv.nvmlDeviceGetMemoryInfo(h)
            util = nv.nvmlDeviceGetUtilizationRates(h)
            name = nv.nvmlDeviceGetName(h)
            gpus.append(
                {
                    "index": i,
                    "name": name.decode() if isinstance(name, bytes) else name,
                    "total_mb": mem.total / MB,
                    "used_mb": mem.used / MB,
                    "util_pct": util.gpu,
                }
            )
            for getter in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
                try:
                    for p in getattr(nv, getter)(h):
                        used = p.usedGpuMemory
                        per_pid[p.pid] = None if used is None else (per_pid.get(p.pid) or 0) + used / MB
                except Exception:  # noqa: BLE001 - not supported on some drivers/WDDM; per-pid stays absent
                    pass
        return {"source": "nvml", "gpus": gpus, "per_pid_mb": per_pid}

    def _read_smi(self) -> dict[str, Any]:
        gpu_q = [self._smi, "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version", "--format=csv,noheader,nounits"]
        app_q = [self._smi, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]
        gpus = parse_smi_gpus(subprocess.run(gpu_q, capture_output=True, text=True, timeout=10).stdout)
        apps = parse_smi_apps(subprocess.run(app_q, capture_output=True, text=True, timeout=10).stdout)
        return {"source": "nvidia-smi", "gpus": gpus, "per_pid_mb": apps}
