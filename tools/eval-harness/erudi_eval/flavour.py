"""Platform + inference backend as a first-class dimension of a measurement.

A flavour (`mac-mlx`, `win-cuda`, `win-cpu`, `linux-cuda`, `linux-cpu`) decides which model is
downloaded, what the inference process looks like, whether per-process GPU memory exists at all, and
which extra facts to record (`-ngl` on CUDA, `--threads` on CPU). A CUDA run against a CPU build is
not the same measurement, so a detected flavour that contradicts the requested one fails the run.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BACKEND_TO_FLAVOUR = {
    ("darwin", "mlx"): "mac-mlx",
    ("darwin", "cpu"): "mac-mlx",  # the macOS build only ships MLX
    ("windows", "cuda"): "win-cuda",
    ("windows", "cpu"): "win-cpu",
    ("linux", "cuda"): "linux-cuda",
    ("linux", "cpu"): "linux-cpu",
}
ENGINE_TO_BACKEND = {"MLX_Engine": "mlx", "CUDA_Engine": "cuda", "CPU_Engine": "cpu"}
DEFAULT_BY_OS = {"darwin": "mac-mlx", "windows": "win-cpu", "linux": "linux-cpu"}


@dataclass(frozen=True)
class Flavour:
    name: str
    os: str
    backend_type: str
    engine: str
    default_model_link: str
    inference_process: dict[str, Any]
    per_process_gpu_memory: str
    gpu_note: str
    extra_checks: list[str]
    workload_file: str | None
    model_note: str | None = None
    notes: str | None = None
    path: str | None = None

    @property
    def inference_role(self) -> str:
        return self.inference_process.get("role", "")


def flavours_dir(harness_dir: Path) -> Path:
    return Path(harness_dir) / "flavours"


def available(harness_dir: Path) -> list[str]:
    return sorted(p.stem for p in flavours_dir(harness_dir).glob("*.json"))


def load(harness_dir: Path, name: str) -> Flavour:
    path = flavours_dir(harness_dir) / f"{name}.json"
    if not path.exists():
        raise ValueError(f"unknown flavour {name!r}; available: {', '.join(available(harness_dir))}")
    return Flavour(**json.loads(path.read_text(encoding="utf-8")), path=str(path))


def default_name(os_name: str, artifact_flavour: str | None) -> str:
    """Flavour guessed before the app runs, from the installed artifact (bundled llama-cpp dir or DMG/AppImage name)."""
    return BACKEND_TO_FLAVOUR.get((os_name, (artifact_flavour or "").lower()), DEFAULT_BY_OS.get(os_name, "linux-cpu"))


def detect(os_name: str, startup: dict[str, Any] | None, environment: dict[str, Any] | None) -> dict[str, Any]:
    """What the running app says: `/hardware/app_startup` backend_type and `/diagnostics/` engine."""
    startup, environment = startup or {}, environment or {}
    backend_type = startup.get("backend_type")
    engine = environment.get("engine")
    effective = backend_type or ENGINE_TO_BACKEND.get(engine or "")
    return {
        "backend_type": backend_type,
        "engine": engine,
        "gpu_name": environment.get("gpu_name") or startup.get("gpu_name"),
        "flavour": BACKEND_TO_FLAVOUR.get((os_name, (effective or "").lower())) if effective else None,
    }


def check(requested: Flavour, detected: dict[str, Any]) -> tuple[bool, str]:
    """(ok, message). An app that says nothing about its backend cannot contradict the request."""
    seen_engine, seen_backend, seen_flavour = detected.get("engine"), detected.get("backend_type"), detected.get("flavour")
    if not seen_flavour and not seen_engine:
        return True, "the running app could not be detected (no backend_type, no engine): flavour taken as requested"
    engine_ok = seen_engine is None or seen_engine == requested.engine
    backend_ok = seen_backend is None or seen_backend == requested.backend_type
    if engine_ok and backend_ok and (seen_flavour in (None, requested.name)):
        return True, f"running app confirms {requested.name} (backend_type={seen_backend}, engine={seen_engine})"
    return False, (f"requested flavour {requested.name} (backend_type={requested.backend_type}, engine={requested.engine}) "
                   f"but the running app reports backend_type={seen_backend}, engine={seen_engine}"
                   + (f" ({seen_flavour})" if seen_flavour else ""))


def inference_matches(f: Flavour, processes: list[dict[str, Any]]) -> tuple[bool | None, str | None]:
    """(ok, role seen) for the resident inference process; (None, None) when none is resident."""
    inference = [p for p in processes if p.get("category") == "inference"]
    if not inference:
        return None, None
    seen = inference[0].get("role")
    return seen == f.inference_role, seen


def gpu_expectation(f: Flavour) -> str:
    return f.per_process_gpu_memory


def extra_check_values(f: Flavour, cmdline: str) -> dict[str, str]:
    """Facts pulled from the inference child's command line, per flavour (`-ngl`, `--threads`)."""
    out: dict[str, str] = {}
    if not f.extra_checks:  # MLX spawns a multiprocessing child: no llama-server flags to read
        return out
    for key, pattern in (("threads", r"--threads[= ]+(\d+)"), ("ngl", r"(?:-ngl|--n-gpu-layers)[= ]+(\d+)")):
        m = re.search(pattern, cmdline)
        if m:
            out[key] = m.group(1)
    return out
