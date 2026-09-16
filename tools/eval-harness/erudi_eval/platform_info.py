"""Preflight: OS, CPU, RAM, GPU/VRAM/driver, app version, free disk."""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import psutil

from . import __version__
from .gpu import parse_smi_gpus
from .layout import AppLayout, app_version


def _run(cmd: list[str], timeout: float = 20) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def cpu_model() -> str:
    if sys.platform == "darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip() or platform.processor()
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def gpu_info() -> list[dict[str, Any]]:
    if sys.platform == "darwin":
        try:
            data = json.loads(_run(["system_profiler", "SPDisplaysDataType", "-json"], timeout=30) or "{}")
        except json.JSONDecodeError:
            return []
        return [
            {"name": d.get("sppci_model"), "cores": d.get("sppci_cores"), "vram": "unified", "displays": len(d.get("spdisplays_ndrvs") or [])}
            for d in data.get("SPDisplaysDataType", [])
        ]
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    return parse_smi_gpus(
        _run([smi, "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version", "--format=csv,noheader,nounits"])
    )


def free_disk_bytes(path: Path) -> int | None:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return shutil.disk_usage(p).free
    except OSError:
        return None


def system_info(layout: AppLayout) -> dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "harness_version": __version__,
        "python": sys.version.split()[0],
        "os": {"system": platform.system(), "release": platform.release(), "version": platform.version(), "machine": platform.machine(),
               "mac_ver": platform.mac_ver()[0] or None},
        "cpu": {"model": cpu_model(), "physical_cores": psutil.cpu_count(logical=False), "logical_cores": psutil.cpu_count()},
        "memory_total_bytes": vm.total,
        "gpus": gpu_info(),
        "app": {
            "path": str(layout.app_path) if layout.app_path else None,
            "installed": bool(layout.app_path and layout.app_path.exists()),
            **app_version(layout),
        },
        "paths": {
            "data_root": str(layout.data_root),
            "backend_log_dir": str(layout.backend_log_dir),
            "capture_logs": [str(p) for p in layout.capture_logs],
        },
        "free_disk_bytes": free_disk_bytes(layout.data_root),
    }


def os_build() -> str | None:
    if sys.platform == "darwin":
        return _run(["sw_vers", "-buildVersion"]).strip() or None
    if sys.platform.startswith("linux"):
        for line in Path("/etc/os-release").read_text().splitlines() if Path("/etc/os-release").exists() else []:
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip('"')
        return platform.release()
    return platform.version()


def conditions(machine_block: dict[str, Any], gpus: list[dict[str, Any]]) -> dict[str, Any]:
    """How the machine stood when the run started: uptime, swap, pressure, power, OS build, displays."""
    boot = psutil.boot_time()
    power = machine_block.get("power_thermal") or {}
    displays = sum(g.get("displays") or 0 for g in gpus) or None
    return {
        "boot_time": datetime.fromtimestamp(boot, tz=timezone.utc).isoformat(timespec="seconds"),
        "uptime_s": round(time.time() - boot),
        "uptime_human": f"{(time.time() - boot) / 3600:.1f} h",
        "swap_used_bytes": machine_block.get("swap_used_bytes"),
        "swap_total_bytes": machine_block.get("swap_total_bytes"),
        "available_bytes": machine_block.get("available_bytes"),
        "memory_pressure_level": machine_block.get("memory_pressure_level"),
        "power_source": power.get("power_source"),
        "battery_percent": power.get("battery_percent"),
        "low_power_mode": power.get("low_power_mode"),
        "os_build": os_build(),
        "display_count": displays if displays else "not collected on this OS",
    }


def disk_allows_download(free_bytes: int | None, size_bytes: int | None, headroom_gb: float) -> tuple[bool, str]:
    """Refuse when free disk < model size + headroom. Unknown sizes are refused too (never guess)."""
    if free_bytes is None:
        return False, "free disk space could not be read"
    if size_bytes is None:
        return False, "download size unknown"
    need = size_bytes + int(headroom_gb * 1024**3)
    if free_bytes < need:
        return False, f"free disk {free_bytes / 1024**3:.1f} GB < size {size_bytes / 1024**3:.1f} GB + headroom {headroom_gb} GB"
    return True, "ok"
