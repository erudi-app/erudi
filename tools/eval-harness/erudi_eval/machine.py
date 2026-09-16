"""Machine context recorded in every sample: memory breakdown, swap, paging rates, CPU, disk, power.

Cheap sources are read every sample (ctypes/psutil/procfs, no subprocess on macOS); the
slow block (pmset) is refreshed every SLOW_EVERY_S. Cumulative counters are turned into
per-second rates between consecutive reads; a counter that goes backwards (reset, wrap)
yields no rate rather than a negative one. Unreadable fields are listed in `errors`.
"""

from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil

IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform.startswith("win")
IS_LINUX = sys.platform.startswith("linux")
SLOW_EVERY_S = 10.0
PRESSURE_ORDER = {"normal": 0, "warn": 1, "critical": 2}


# --- pure helpers ----------------------------------------------------------------------------


def counter_rates(prev: dict[str, int] | None, cur: dict[str, int], dt: float) -> dict[str, float | None]:
    """Per-second rates of cumulative counters. No previous read, dt <= 0, or a decrease -> None."""
    out: dict[str, float | None] = {}
    for key, value in cur.items():
        before = (prev or {}).get(key)
        if before is None or dt <= 0 or value is None or value < before:
            out[key] = None
        else:
            out[key] = round((value - before) / dt, 3)
    return out


# --- macOS ------------------------------------------------------------------------------------


class VmStatistics64(ctypes.Structure):
    """struct vm_statistics64 from <mach/vm_statistics.h> (152 bytes, HOST_VM_INFO64_COUNT = 38)."""

    _fields_ = [
        ("free_count", ctypes.c_uint32),
        ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32),
        ("wire_count", ctypes.c_uint32),
        ("zero_fill_count", ctypes.c_uint64),
        ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64),
        ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64),
        ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64),
        ("hits", ctypes.c_uint64),
        ("purges", ctypes.c_uint64),
        ("purgeable_count", ctypes.c_uint32),
        ("speculative_count", ctypes.c_uint32),
        ("decompressions", ctypes.c_uint64),
        ("compressions", ctypes.c_uint64),
        ("swapins", ctypes.c_uint64),
        ("swapouts", ctypes.c_uint64),
        ("compressor_page_count", ctypes.c_uint32),
        ("throttled_count", ctypes.c_uint32),
        ("external_page_count", ctypes.c_uint32),
        ("internal_page_count", ctypes.c_uint32),
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
    ]


HOST_VM_INFO64 = 4
HOST_VM_INFO64_COUNT = ctypes.sizeof(VmStatistics64) // 4
MAC_PAGE_FIELDS = (
    "free_count", "active_count", "inactive_count", "speculative_count", "wire_count", "purgeable_count",
    "compressor_page_count", "total_uncompressed_pages_in_compressor", "internal_page_count", "external_page_count",
)
MAC_COUNTERS = ("pageins", "pageouts", "swapins", "swapouts", "compressions", "decompressions", "faults")


def derive_mac_memory(vm: dict[str, int], page_size: int) -> dict[str, int]:
    """Activity Monitor's figures from vm_statistics64 page counts."""
    app = (vm["internal_page_count"] - vm["purgeable_count"]) * page_size
    wired = vm["wire_count"] * page_size
    compressed = vm["compressor_page_count"] * page_size
    return {
        "app_memory_bytes": app,
        "wired_bytes": wired,
        "compressed_bytes": compressed,
        "cached_files_bytes": (vm["external_page_count"] + vm["purgeable_count"]) * page_size,
        "memory_used_bytes": app + wired + compressed,
        "free_bytes": vm["free_count"] * page_size,
    }


def vm_struct_to_dict(s: VmStatistics64) -> dict[str, int]:
    return {name: int(getattr(s, name)) for name, _ in VmStatistics64._fields_}


def parse_vm_stat(text: str) -> dict[str, int]:
    """Fallback: wired and compressed bytes from `vm_stat` output."""
    page = 4096
    m = re.search(r"page size of (\d+) bytes", text)
    if m:
        page = int(m.group(1))
    out = {}
    for key, label in (("wired_bytes", "Pages wired down"), ("compressed_bytes", "Pages occupied by compressor")):
        m = re.search(re.escape(label) + r":\s+(\d+)", text)
        if m:
            out[key] = int(m.group(1)) * page
    return out


class _XswUsage(ctypes.Structure):
    _fields_ = [("xsu_total", ctypes.c_uint64), ("xsu_avail", ctypes.c_uint64), ("xsu_used", ctypes.c_uint64),
                ("xsu_pagesize", ctypes.c_uint32), ("xsu_encrypted", ctypes.c_int32)]


_libc = None


def _mac_libc():
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        _libc.mach_host_self.restype = ctypes.c_uint32
        _libc.host_statistics64.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        _libc.sysctlbyname.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t]
    return _libc


def sysctl_raw(name: str, size: int) -> bytes | None:
    buf = ctypes.create_string_buffer(size)
    length = ctypes.c_size_t(size)
    if _mac_libc().sysctlbyname(name.encode(), buf, ctypes.byref(length), None, 0) != 0:
        return None
    return buf.raw[: length.value]


def sysctl_int(name: str) -> int | None:
    raw = sysctl_raw(name, 8)
    if raw is None or len(raw) not in (4, 8):
        return None
    return int.from_bytes(raw, sys.byteorder, signed=True)


_host_port = None


def mac_vm_statistics() -> dict[str, int]:
    """host_statistics64(HOST_VM_INFO64). Raises OSError when the call fails."""
    global _host_port
    libc = _mac_libc()
    if _host_port is None:
        _host_port = libc.mach_host_self()  # one send right kept for the process lifetime
    info = VmStatistics64()
    count = ctypes.c_uint32(HOST_VM_INFO64_COUNT)
    kr = libc.host_statistics64(_host_port, HOST_VM_INFO64, ctypes.byref(info), ctypes.byref(count))
    if kr != 0:
        raise OSError(f"host_statistics64 kern_return {kr}")
    return vm_struct_to_dict(info)


def mac_swap() -> dict[str, int]:
    raw = sysctl_raw("vm.swapusage", ctypes.sizeof(_XswUsage))
    if raw is None or len(raw) < ctypes.sizeof(_XswUsage):
        raise OSError("sysctl vm.swapusage failed")
    x = _XswUsage.from_buffer_copy(raw)
    return {"swap_total_bytes": x.xsu_total, "swap_used_bytes": x.xsu_used, "swap_free_bytes": x.xsu_avail}


def parse_pmset_therm(text: str) -> dict[str, Any]:
    """`pmset -g therm`: explicit levels when recorded, else 'none recorded'."""
    out: dict[str, Any] = {}
    for kind in ("thermal", "performance"):
        if re.search(rf"No {kind} warning level has been recorded", text):
            out[f"{kind}_warning_level"] = "none recorded"
        m = re.search(rf"{kind} warning level set to (\w+)", text, re.IGNORECASE)
        if m:
            out[f"{kind}_warning_level"] = m.group(1)
    for key, value in re.findall(r"(CPU_\w+)\s*=\s*(\d+)", text):
        out[key.lower()] = int(value)
    return out


def parse_pmset_batt(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    m = re.search(r"drawing from '([^']+)'", text)
    if m:
        out["power_source"] = m.group(1)
    m = re.search(r"(\d+)%;\s*([\w ]+?);", text)
    if m:
        out["battery_percent"] = int(m.group(1))
        out["battery_state"] = m.group(2)
    return out


def parse_pmset_settings(text: str) -> dict[str, Any]:
    m = re.search(r"^\s*lowpowermode\s+(\d)", text, re.MULTILINE)
    return {"low_power_mode": bool(int(m.group(1)))} if m else {}


# --- Linux / Windows parsers ----------------------------------------------------------------


MEMINFO_KEYS = ("MemTotal", "MemAvailable", "Cached", "SwapCached", "AnonPages", "Shmem", "SwapTotal", "SwapFree")


def parse_meminfo(text: str) -> dict[str, int]:
    out = {}
    for line in text.splitlines():
        m = re.match(r"(\w+):\s+(\d+)\s*kB", line)
        if m and m.group(1) in MEMINFO_KEYS:
            out[m.group(1)] = int(m.group(2)) * 1024
    return out


def parse_vmstat_counters(text: str, keys=("pswpin", "pswpout", "pgmajfault")) -> dict[str, int]:
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in keys:
            out[parts[0]] = int(parts[1])
    return out


def parse_psi(text: str) -> dict[str, dict[str, float]]:
    """/proc/pressure/memory -> {'some': {avg10, avg60, avg300}, 'full': {...}}."""
    out = {}
    for line in text.splitlines():
        kind = line.split(" ", 1)[0]
        if kind in ("some", "full"):
            out[kind] = {k: float(v) for k, v in re.findall(r"(avg10|avg60|avg300)=([\d.]+)", line)}
    return out


class _WinPerfInfo(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
        ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t), ("PhysicalAvailable", ctypes.c_size_t),
        ("SystemCache", ctypes.c_size_t), ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
        ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t), ("HandleCount", ctypes.c_uint32),
        ("ProcessCount", ctypes.c_uint32), ("ThreadCount", ctypes.c_uint32),
    ]


# --- reader --------------------------------------------------------------------------------


class MachineReader:
    def __init__(self, disk_path: Path | None = None, slow_every: float = SLOW_EVERY_S):
        self.disk_path = Path(disk_path) if disk_path else Path.home()
        self.slow_every = slow_every
        self._prev_counters: dict[str, int] | None = None
        self._prev_t: float | None = None
        self._slow: dict[str, Any] = {}
        self._slow_at = -1e9
        self.page_size = sysctl_int("hw.pagesize") if IS_MAC else None
        psutil.cpu_percent(None)  # prime the non-blocking system CPU reading

    def read(self) -> dict[str, Any]:
        now = time.monotonic()
        out: dict[str, Any] = {}
        errors: dict[str, str] = {}
        counters: dict[str, int] = {}

        def attempt(name: str, fn) -> None:
            try:
                fn()
            except Exception as e:  # noqa: BLE001 - every unreadable field is recorded, never substituted
                errors[name] = f"{type(e).__name__}: {e}"[:200]

        vm = psutil.virtual_memory()
        out["total_bytes"], out["available_bytes"] = vm.total, vm.available

        if IS_MAC:
            attempt("vm_statistics64", lambda: self._mac_memory(out, counters))
            attempt("swap", lambda: out.update(mac_swap()))
            level = sysctl_int("kern.memorystatus_vm_pressure_level")
            if level is None:
                errors["memory_pressure_level"] = "sysctl failed"
            else:
                out["memory_pressure_level"] = {1: "normal", 2: "warn", 4: "critical"}.get(level, str(level))
        else:
            def swap():
                sw = psutil.swap_memory()
                out.update(swap_total_bytes=sw.total, swap_used_bytes=sw.used, swap_free_bytes=sw.free)
            attempt("swap", swap)
        if IS_LINUX:
            attempt("meminfo", lambda: out.update({f"meminfo_{k}": v for k, v in parse_meminfo(Path("/proc/meminfo").read_text()).items()}))
            attempt("vmstat", lambda: counters.update(parse_vmstat_counters(Path("/proc/vmstat").read_text())))
            attempt("memory_psi", lambda: out.__setitem__("memory_psi", parse_psi(Path("/proc/pressure/memory").read_text())))
        if IS_WIN:
            attempt("performance_info", lambda: self._win_perf(out))
            errors["hard_page_faults_per_s"] = "unavailable: needs a PDH counter query"

        out["cpu_percent"] = psutil.cpu_percent(None)
        attempt("load_average", lambda: out.__setitem__("load_average", [round(x, 2) for x in os.getloadavg()]))
        attempt("process_count", lambda: out.__setitem__("process_count", len(psutil.pids())))

        def disk():
            io = psutil.disk_io_counters()
            if io is None:
                raise OSError("no disk counters")
            counters.update(disk_read_bytes=io.read_bytes, disk_write_bytes=io.write_bytes)
        attempt("disk_io", disk)
        attempt("data_volume_free_bytes", lambda: out.__setitem__("data_volume_free_bytes", shutil.disk_usage(_existing(self.disk_path)).free))

        dt = (now - self._prev_t) if self._prev_t is not None else 0.0
        out["counters"] = counters
        out["rates_per_s"] = counter_rates(self._prev_counters, counters, dt)
        self._prev_counters, self._prev_t = counters, now

        if now - self._slow_at >= self.slow_every:
            self._slow = self._read_slow()
            self._slow_at = now
        out["power_thermal"] = self._slow
        if errors:
            out["errors"] = errors
        return out

    def _mac_memory(self, out: dict[str, Any], counters: dict[str, int]) -> None:
        try:
            vm = mac_vm_statistics()
        except OSError:
            text = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
            out.update(parse_vm_stat(text))
            out["vm_source"] = "vm_stat (fallback)"
            raise
        page = self.page_size or 4096
        out["page_size"] = page
        out["vm_pages"] = {k: vm[k] for k in MAC_PAGE_FIELDS}
        out.update(derive_mac_memory(vm, page))
        counters.update({k: vm[k] for k in MAC_COUNTERS})

    @staticmethod
    def _win_perf(out: dict[str, Any]) -> None:
        info = _WinPerfInfo()
        info.cb = ctypes.sizeof(info)
        if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
            raise OSError("GetPerformanceInfo failed")
        page = info.PageSize
        out.update(
            commit_charge_bytes=info.CommitTotal * page, commit_limit_bytes=info.CommitLimit * page,
            system_cache_bytes=info.SystemCache * page, kernel_paged_bytes=info.KernelPaged * page,
            kernel_nonpaged_bytes=info.KernelNonpaged * page, handle_count=info.HandleCount,
            process_count_kernel=info.ProcessCount, thread_count=info.ThreadCount,
        )

    @staticmethod
    def _read_slow() -> dict[str, Any]:
        out: dict[str, Any] = {"read_at": time.time()}
        errors: dict[str, str] = {}
        if IS_MAC:
            for key, cmd, parser in (
                ("therm", ["pmset", "-g", "therm"], parse_pmset_therm),
                ("batt", ["pmset", "-g", "batt"], parse_pmset_batt),
                ("settings", ["pmset", "-g"], parse_pmset_settings),
            ):
                try:
                    out.update(parser(subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout))
                except (OSError, subprocess.SubprocessError) as e:
                    errors[key] = type(e).__name__
        else:
            try:
                batt = psutil.sensors_battery()
                if batt is None:
                    out["power_source"] = "no battery reported"
                else:
                    out.update(power_source="AC Power" if batt.power_plugged else "Battery Power", battery_percent=batt.percent)
            except (AttributeError, OSError, RuntimeError) as e:
                errors["power_source"] = type(e).__name__
            errors["thermal"] = "unavailable on this OS (no cheap, reliable source)"
        if errors:
            out["errors"] = errors
        return out


def _existing(path: Path) -> Path:
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def static_info() -> dict[str, Any]:
    """Once per run (system.json)."""
    out: dict[str, Any] = {}
    if IS_MAC:
        value = sysctl_int("iogpu.wired_limit_mb")
        out["iogpu_wired_limit_mb"] = value if value is not None else "unreadable"
        out["iogpu_wired_limit_note"] = "0 = macOS default GPU wired limit"
        out["page_size"] = sysctl_int("hw.pagesize")
    return out


def top_other_processes(exclude: set[int], limit: int = 10) -> dict[str, Any]:
    """Largest processes by RSS that are not Erudi and not the harness."""
    rows, total, unreadable = [], 0, 0
    for proc in psutil.process_iter(["pid", "name", "memory_info"]):
        info = proc.info
        if info["pid"] in exclude:
            continue
        mi = info.get("memory_info")
        if mi is None:
            unreadable += 1
            continue
        total += mi.rss
        rows.append((mi.rss, info["pid"], info.get("name") or ""))
    rows.sort(reverse=True)
    mb = 1024 * 1024
    return {
        "metric": "rss",
        "total_rss_mb": round(total / mb, 1),
        "unreadable": unreadable,
        "top": [{"pid": pid, "name": name, "rss_mb": round(rss / mb, 1)} for rss, pid, name in rows[:limit]],
    }
