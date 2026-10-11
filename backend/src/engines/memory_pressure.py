"""Memory-pressure sensors for the llama.cpp memory brake, one provider per OS.

The brake (``src.engines.memory_brake.llama_step``) needs the same answers on
every OS; a provider chosen once per process on ``sys.platform`` gives them:

* ``read_pressure()``: the pressure LEVEL on the 1/2/4 scale (calm, warning,
  critical) and a CUMULATIVE paging counter in bytes, ``damage_total_bytes``
  -- the harm the brake exists for; the caller diffs it between samples.
  ``None`` means no counter on this machine, which is not a zero delta.
* ``child_footprint_bytes(pid)``: what the child holds, for the unload's share.
* ``child_anonymous_bytes(pid)``: the child's ANONYMOUS memory (clean
  file-backed mappings excluded: the mapped weights), for the post-load check.
* ``effective_total_ram_bytes()``: the physical RAM the window budget and the
  shares are computed on.

Providers:

* **macOS**: ``kern.memorystatus_vm_pressure_level`` and the swap-outs of
  ``host_statistics64`` (16 KiB pages), through ``process_footprint``; the
  physical footprint is both the footprint and the anonymous counter.
* **Linux**: PSI, the SYSTEM ``/proc/pressure/memory`` -- our cgroup v2
  ``memory.pressure`` only when the same cgroup's ``memory.max`` is finite (a
  real container: a desktop's systemd scope counts only our own stalls, and an
  idle llama-server never stalls while the machine thrashes). Tiered on the
  stall fraction since the previous sample when at least
  ``PSI_MIN_SAMPLE_INTERVAL_S`` passed, else on the kernel's ``avg10``.
  Without PSI (kernel < 4.20, ``CONFIG_PSI=n``, ``psi=0``): the swap-out and
  major-fault rates (``/proc/vmstat``). Damage: ``psutil``'s swap-out bytes.
  Footprint: USS + swap; anonymous: ``Anonymous:`` + ``Swap:`` of
  ``/proc/<pid>/smaps_rollup``. RAM: bounded by a finite ``memory.max`` of our
  own cgroup (ancestors are not read).
* **Windows**: the commit charge from ``GlobalMemoryStatusEx``
  (``1 - ullAvailPageFile / ullTotalPageFile``); the paging counter is PDH's
  ``\\Memory\\Pages Output/sec`` added by its English name (locale-proof),
  read RAW (cumulative pages, 4 KiB); without PDH the tiers stand and the
  brake decides on the level alone. Footprint and anonymous: the private
  commit (``memory_info().private``).

Every threshold for Linux and Windows is a starting value, to be calibrated
on real machines. Nothing native is loaded at import: the Windows and macOS
libraries are opened on first use, so every CI leg imports this module.

``ERUDI_FORCE_MEMORY_PRESSURE=warning|critical`` (a QA seam, never set in
normal use) forces the level AND the damage evidence of every reading -- each
read advances a synthetic counter by ``LLAMA_WARNING_DAMAGE_MIN_BYTES`` -- so a
shrink and an idle respawn can be exercised on a calm machine. It acts here,
so on the llama.cpp engines only: the MLX brake reads ``process_footprint``.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import psutil

from src.core.logging import logger
from src.engines.memory_brake import (
    LEVEL_CALM,
    LEVEL_CRITICAL,
    LEVEL_WARNING,
    LLAMA_WARNING_DAMAGE_MIN_BYTES,
)

_MIB = 1024**2

MAC_PAGE_BYTES = 16384
WIN_PAGE_BYTES = 4096
# Linux PSI (INFERRED starting values, calibrated in QA).
PSI_MIN_SAMPLE_INTERVAL_S = 5
PSI_SOME_WARN_STALL_FRACTION = 0.10
PSI_FULL_CRITICAL_STALL_FRACTION = 0.05
# Linux without PSI (INFERRED).
LINUX_SWAPOUT_WARN_BYTES_PER_S = 8 * _MIB
LINUX_SWAPOUT_CRITICAL_BYTES_PER_S = 64 * _MIB
LINUX_MAJFAULT_WARN_PER_S = 200
# Windows commit charge (INFERRED).
WIN_COMMIT_WARN_FRACTION = 0.85
WIN_COMMIT_CRITICAL_FRACTION = 0.93

_FORCED_LEVELS = {"warning": LEVEL_WARNING, "critical": LEVEL_CRITICAL}
_FORCED_DAMAGE_BYTES = 0
_FORCE_WARNED = False
_PROVIDER: Any = None


@dataclass(frozen=True)
class PressureReading:
    """One sensor reading. ``unavailable`` says why ``level`` is None."""

    level: Optional[int]
    damage_total_bytes: Optional[int]
    unavailable: Optional[str] = None


def _psutil_total() -> Optional[int]:
    total = int(psutil.virtual_memory().total)
    return total if total > 0 else None


class _NullProvider:
    """A platform without a sensor: the brake stays off."""

    name = "none"

    def read(self) -> PressureReading:
        return PressureReading(None, None, unavailable=f"no sensor on {sys.platform}")

    def footprint(self, pid: int) -> Optional[int]:
        return None

    def anonymous(self, pid: int) -> Optional[int]:
        return None

    def total_ram_bytes(self) -> Optional[int]:
        return _psutil_total()


class _MacProvider(_NullProvider):
    name = "macos"

    def read(self) -> PressureReading:
        from src.engines import process_footprint

        level = process_footprint.pressure_level()
        swapouts = process_footprint.swapouts()
        return PressureReading(
            level,
            swapouts * MAC_PAGE_BYTES if swapouts is not None else None,
            unavailable=None if level is not None else "kern.memorystatus_vm_pressure_level",
        )

    def footprint(self, pid: int) -> Optional[int]:
        from src.engines import process_footprint

        return process_footprint.footprint(pid)

    # phys_footprint already excludes clean file-backed mappings.
    anonymous = footprint


def _tier(some: float, full: float) -> int:
    if full >= PSI_FULL_CRITICAL_STALL_FRACTION:
        return LEVEL_CRITICAL
    if some >= PSI_SOME_WARN_STALL_FRACTION:
        return LEVEL_WARNING
    return LEVEL_CALM


def _parse_psi(text: str) -> Dict[str, Tuple[float, int]]:
    """``{"some": (avg10 fraction, total µs), "full": (...)}``."""
    out: Dict[str, Tuple[float, int]] = {}
    for line in text.splitlines():
        kind, _, rest = line.partition(" ")
        fields = dict(item.split("=", 1) for item in rest.split() if "=" in item)
        if kind in ("some", "full") and "avg10" in fields and "total" in fields:
            out[kind] = (float(fields["avg10"]) / 100.0, int(fields["total"]))
    return out


def _read_int(path: Path) -> Optional[int]:
    try:
        raw = path.read_text(encoding="ascii").strip()
    except OSError:
        return None
    return int(raw) if raw.isdigit() else None


class _LinuxProvider(_NullProvider):
    name = "linux"

    def __init__(
        self,
        *,
        proc_root: Path = Path("/proc"),
        cgroup_root: Path = Path("/sys/fs/cgroup"),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._proc = proc_root
        self._cgroup_root = cgroup_root
        self._clock = clock
        self._psi_previous: Optional[Tuple[float, int, int]] = None
        self._rates_previous: Optional[Tuple[float, Optional[int], Optional[int]]] = None
        self._rates_level = LEVEL_CALM

    def _own_cgroup(self) -> Optional[Path]:
        """Our cgroup v2 directory, or None (cgroup v1, unreadable)."""
        try:
            lines = (self._proc / "self" / "cgroup").read_text(encoding="utf-8").splitlines()
        except OSError:
            return None
        for line in lines:
            if line.startswith("0::"):
                return self._cgroup_root / line[3:].strip().lstrip("/")
        return None

    def _container_limit(self) -> Optional[int]:
        """Our cgroup's ``memory.max`` when finite (``max``: no container)."""
        cgroup = self._own_cgroup()
        return _read_int(cgroup / "memory.max") if cgroup is not None else None

    def _psi_file(self) -> Path:
        if self._container_limit() is not None:
            cgroup_psi = self._own_cgroup() / "memory.pressure"
            if cgroup_psi.is_file():
                return cgroup_psi
        return self._proc / "pressure" / "memory"

    def _psi_level(self) -> Optional[int]:
        try:
            psi = _parse_psi(self._psi_file().read_text(encoding="ascii"))
        except OSError:
            return None
        if "some" not in psi:
            return None
        some_avg, some_total = psi["some"]
        full_avg, full_total = psi.get("full", (0.0, 0))
        now = self._clock()
        previous = self._psi_previous
        if previous is not None and now - previous[0] >= PSI_MIN_SAMPLE_INTERVAL_S:
            elapsed_us = (now - previous[0]) * 1e6
            some = max(0, some_total - previous[1]) / elapsed_us
            full = max(0, full_total - previous[2]) / elapsed_us
        else:
            some, full = some_avg, full_avg
        if previous is None or now - previous[0] >= PSI_MIN_SAMPLE_INTERVAL_S:
            self._psi_previous = (now, some_total, full_total)
        return _tier(some, full)

    def _major_faults(self) -> Optional[int]:
        try:
            for line in (self._proc / "vmstat").read_text(encoding="ascii").splitlines():
                name, _, value = line.partition(" ")
                if name == "pgmajfault":
                    return int(value)
        except (OSError, ValueError):
            return None
        return None

    def _rates_level_of(self, swapout: Optional[int]) -> Optional[int]:
        """The tier from the swap-out and major-fault rates (no PSI). The first
        read primes the rates: calm, no evidence of paging yet; a read sooner
        than ``PSI_MIN_SAMPLE_INTERVAL_S`` keeps the previous tier."""
        faults = self._major_faults()
        if swapout is None and faults is None:
            return None
        now = self._clock()
        previous = self._rates_previous
        if previous is None:
            self._rates_previous = (now, swapout, faults)
            return self._rates_level
        elapsed = now - previous[0]
        if elapsed < PSI_MIN_SAMPLE_INTERVAL_S:
            return self._rates_level
        swap_rate = (
            max(0, swapout - previous[1]) / elapsed
            if swapout is not None and previous[1] is not None
            else 0.0
        )
        fault_rate = (
            max(0, faults - previous[2]) / elapsed
            if faults is not None and previous[2] is not None
            else 0.0
        )
        if swap_rate >= LINUX_SWAPOUT_CRITICAL_BYTES_PER_S:
            level = LEVEL_CRITICAL
        elif swap_rate >= LINUX_SWAPOUT_WARN_BYTES_PER_S or fault_rate >= LINUX_MAJFAULT_WARN_PER_S:
            level = LEVEL_WARNING
        else:
            level = LEVEL_CALM
        self._rates_previous = (now, swapout, faults)
        self._rates_level = level
        return level

    def read(self) -> PressureReading:
        try:
            swapout: Optional[int] = int(psutil.swap_memory().sout)
        except Exception:
            # No swap accounting readable: the damage counter is unavailable.
            swapout = None
        level = self._psi_level()
        if level is None:
            level = self._rates_level_of(swapout)
        return PressureReading(
            level,
            swapout,
            unavailable=None if level is not None else "no PSI, no /proc/vmstat, no swap counter",
        )

    def footprint(self, pid: int) -> Optional[int]:
        info = psutil.Process(pid).memory_full_info()
        return int(info.uss) + int(getattr(info, "swap", 0) or 0)

    def anonymous(self, pid: int) -> Optional[int]:
        try:
            text = (self._proc / str(pid) / "smaps_rollup").read_text(encoding="ascii")
        except OSError:
            return None
        found = {}
        for line in text.splitlines():
            name, _, rest = line.partition(":")
            if name in ("Anonymous", "Swap"):
                found[name] = int(rest.split()[0]) * 1024
        return sum(found.values()) if "Anonymous" in found else None

    def total_ram_bytes(self) -> Optional[int]:
        total = _psutil_total()
        limit = self._container_limit()
        if total is not None and limit is not None:
            return min(total, limit)
        return total


class _WindowsProvider(_NullProvider):
    name = "windows"

    _PAGES_OUTPUT = "\\Memory\\Pages Output/sec"

    def __init__(self) -> None:
        self._pdh: Any = None
        self._pdh_warned = False

    def _page_file(self) -> Tuple[int, int]:
        """``(ullAvailPageFile, ullTotalPageFile)`` of ``GlobalMemoryStatusEx``."""
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError(ctypes.get_last_error())
        return int(status.ullAvailPageFile), int(status.ullTotalPageFile)

    def _pdh_pages_output(self) -> int:
        """The raw (cumulative) value of ``\\Memory\\Pages Output/sec``."""
        import ctypes
        from ctypes import wintypes

        class _RawCounter(ctypes.Structure):
            _fields_ = [
                ("CStatus", wintypes.DWORD),
                ("TimeStamp", wintypes.FILETIME),
                ("FirstValue", ctypes.c_longlong),
                ("SecondValue", ctypes.c_longlong),
                ("MultiCount", wintypes.DWORD),
            ]

        pdh = ctypes.windll.pdh
        if self._pdh is None:
            query, counter = ctypes.c_void_p(), ctypes.c_void_p()
            status = pdh.PdhOpenQueryW(None, None, ctypes.byref(query))
            if status != 0:
                raise OSError(f"PdhOpenQueryW status {status & 0xFFFFFFFF:#x}")
            status = pdh.PdhAddEnglishCounterW(
                query, self._PAGES_OUTPUT, None, ctypes.byref(counter)
            )
            if status != 0:
                raise OSError(f"PdhAddEnglishCounterW status {status & 0xFFFFFFFF:#x}")
            self._pdh = (query, counter)
        query, counter = self._pdh
        status = pdh.PdhCollectQueryData(query)
        if status != 0:
            raise OSError(f"PdhCollectQueryData status {status & 0xFFFFFFFF:#x}")
        kind, raw = wintypes.DWORD(), _RawCounter()
        status = pdh.PdhGetRawCounterValue(counter, ctypes.byref(kind), ctypes.byref(raw))
        if status != 0:
            raise OSError(f"PdhGetRawCounterValue status {status & 0xFFFFFFFF:#x}")
        return int(raw.FirstValue)

    def read(self) -> PressureReading:
        try:
            available, total = self._page_file()
        except Exception as exc:
            return PressureReading(
                None, None, unavailable=f"GlobalMemoryStatusEx unreadable ({type(exc).__name__})"
            )
        commit = 1.0 - available / total if total else 0.0
        if commit >= WIN_COMMIT_CRITICAL_FRACTION:
            level = LEVEL_CRITICAL
        elif commit >= WIN_COMMIT_WARN_FRACTION:
            level = LEVEL_WARNING
        else:
            level = LEVEL_CALM
        try:
            damage: Optional[int] = self._pdh_pages_output() * WIN_PAGE_BYTES
        except Exception as exc:
            damage = None
            if not self._pdh_warned:
                # Degraded once per process: the tiers stand.
                self._pdh_warned = True
                logger.warning(
                    f"Paging counter unavailable (PDH, {type(exc).__name__}); "
                    f"the memory brake decides on the pressure level alone"
                )
        return PressureReading(level, damage)

    def footprint(self, pid: int) -> Optional[int]:
        return int(psutil.Process(pid).memory_info().private)

    # The private commit counts paged-out KV and a repacked weights copy, never
    # the mapped weights.
    anonymous = footprint


def _provider() -> Any:
    """The provider of this process, chosen once on ``sys.platform``."""
    global _PROVIDER
    if _PROVIDER is None:
        if sys.platform == "darwin":
            _PROVIDER = _MacProvider()
        elif sys.platform.startswith("linux"):
            _PROVIDER = _LinuxProvider()
        elif sys.platform == "win32":
            _PROVIDER = _WindowsProvider()
        else:
            _PROVIDER = _NullProvider()
    return _PROVIDER


def sensor_name() -> str:
    return _provider().name


def _forced_level() -> Optional[int]:
    global _FORCE_WARNED
    raw = os.environ.get("ERUDI_FORCE_MEMORY_PRESSURE", "").strip().lower()
    if not raw:
        return None
    level = _FORCED_LEVELS.get(raw)
    if level is None and not _FORCE_WARNED:
        _FORCE_WARNED = True
        logger.warning("ERUDI_FORCE_MEMORY_PRESSURE is not 'warning' or 'critical'; ignoring it")
    return level


def read_pressure() -> PressureReading:
    """The pressure level and the cumulative paging counter now. Never raises:
    a sensor that fails reads as no level, with the reason."""
    global _FORCED_DAMAGE_BYTES
    forced = _forced_level()
    if forced is not None:
        _FORCED_DAMAGE_BYTES += LLAMA_WARNING_DAMAGE_MIN_BYTES
        return PressureReading(forced, _FORCED_DAMAGE_BYTES)
    try:
        return _provider().read()
    except Exception as exc:
        return PressureReading(None, None, unavailable=f"{type(exc).__name__}")


def _guarded(read: Callable[[], Optional[int]]) -> Optional[int]:
    try:
        value = read()
    except Exception:
        # A process gone or a counter denied: the caller treats None as
        # unknown (no share, no check), which is what this sample can say.
        return None
    return int(value) if isinstance(value, int) and value > 0 else None


def child_footprint_bytes(pid: Optional[int]) -> Optional[int]:
    """What the child holds (the unload's share), or None."""
    return _guarded(lambda: _provider().footprint(pid)) if pid else None


def child_anonymous_bytes(pid: Optional[int]) -> Optional[int]:
    """The child's anonymous memory (the post-load check), or None."""
    return _guarded(lambda: _provider().anonymous(pid)) if pid else None


def effective_total_ram_bytes() -> Optional[int]:
    """Physical RAM (bounded by a container's finite limit on Linux), or None."""
    return _guarded(lambda: _provider().total_ram_bytes())
