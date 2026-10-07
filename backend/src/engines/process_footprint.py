"""Measure the MLX inference child's memory from the parent (macOS only).

The memory prior (``src.engines.memory_budget``) is checked against what the
child really uses. The MLX GPU loop is a thread of the ``mlx_vlm.server``
process, so the child's pid carries the whole footprint: weights, KV cache,
the prefix-cache pool, MLX's buffer cache and the prefill transients.

* ``footprint(pid)``: the physical footprint now (``ri_phys_footprint`` of
  ``proc_pid_rusage(RUSAGE_INFO_V4)``) -- what Activity Monitor shows as
  Memory.
* ``begin_peak_window(pid)``: resets the footprint INTERVAL through
  ``proc_reset_footprint_interval``, a private libproc SPI resolved
  defensively: a missing symbol or a non-zero return answers ``False`` and
  writes ONE WARNING per backend process.
* ``peak_since(pid, started)``: the interval maximum
  (``ri_interval_max_phys_footprint``) when the window started, else the
  lifetime maximum (``ri_lifetime_max_phys_footprint``, public) -- which can
  only over-state.
* ``pressure_level()`` / ``swapouts()``: system-wide context recorded with an
  observation (``kern.memorystatus_vm_pressure_level``; ``swapouts`` of
  ``host_statistics64``).

Everywhere but macOS every function answers ``None`` / ``False`` and never
raises. Nothing native is loaded at import time: the libraries are opened on
first use, so the Ubuntu and Windows CI legs import this module cleanly.
"""

from __future__ import annotations

import sys
from typing import Any, Optional

from src.core.logging import logger

# ``flavor`` of proc_pid_rusage: struct rusage_info_v4.
_RUSAGE_INFO_V4 = 4
# host_statistics64 flavor: vm_statistics64.
_HOST_VM_INFO64 = 4

# Opened on first use (see ``_libproc``); ``None`` until then.
_LIBPROC: Any = None
_SPI_WARNED = False
# ``mach_host_self()`` hands out a send right on every call: taken once and
# reused for the life of the process (``_host_port``).
_HOST_PORT: Optional[int] = None

_V4_FIELDS = (
    "ri_user_time",
    "ri_system_time",
    "ri_pkg_idle_wkups",
    "ri_interrupt_wkups",
    "ri_pageins",
    "ri_wired_size",
    "ri_resident_size",
    "ri_phys_footprint",
    "ri_proc_start_abstime",
    "ri_proc_exit_abstime",
    "ri_child_user_time",
    "ri_child_system_time",
    "ri_child_pkg_idle_wkups",
    "ri_child_interrupt_wkups",
    "ri_child_pageins",
    "ri_child_elapsed_abstime",
    "ri_diskio_bytesread",
    "ri_diskio_byteswritten",
    "ri_cpu_time_qos_default",
    "ri_cpu_time_qos_maintenance",
    "ri_cpu_time_qos_background",
    "ri_cpu_time_qos_utility",
    "ri_cpu_time_qos_legacy",
    "ri_cpu_time_qos_user_initiated",
    "ri_cpu_time_qos_user_interactive",
    "ri_billed_system_time",
    "ri_serviced_system_time",
    "ri_logical_writes",
    "ri_lifetime_max_phys_footprint",
    "ri_instructions",
    "ri_cycles",
    "ri_billed_energy",
    "ri_serviced_energy",
    "ri_interval_max_phys_footprint",
    "ri_runnable_time",
)


def _on_macos() -> bool:
    return sys.platform == "darwin"


def _libproc():
    """libproc, opened once; ``None`` when it cannot be."""
    global _LIBPROC
    if _LIBPROC is None:
        import ctypes

        try:
            _LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        except OSError:
            _LIBPROC = False
    return _LIBPROC or None


def _rusage(pid: int):
    """``rusage_info_v4`` of ``pid``, or ``None``."""
    if not _on_macos():
        return None
    import ctypes

    lib = _libproc()
    if lib is None:
        return None

    class _RUsageInfoV4(ctypes.Structure):
        _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
            (name, ctypes.c_uint64) for name in _V4_FIELDS
        ]

    info = _RUsageInfoV4()
    try:
        rc = lib.proc_pid_rusage(int(pid), _RUSAGE_INFO_V4, ctypes.byref(info))
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    return info if rc == 0 else None


def footprint(pid: int) -> Optional[int]:
    """Physical footprint of ``pid`` in bytes, or ``None``."""
    info = _rusage(pid)
    return int(info.ri_phys_footprint) if info is not None else None


def _reset_interval_function():
    """``proc_reset_footprint_interval`` (private SPI), or ``None``."""
    import ctypes

    try:
        system = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        function = system.proc_reset_footprint_interval
    except (OSError, AttributeError):
        return None
    function.argtypes = [ctypes.c_int]
    function.restype = ctypes.c_int
    return function


def begin_peak_window(pid: int) -> bool:
    """Reset ``pid``'s footprint interval; ``True`` when the window started."""
    global _SPI_WARNED
    if not _on_macos():
        return False
    function = _reset_interval_function()
    try:
        rc = function(int(pid)) if function is not None else None
    except (OSError, ValueError, TypeError):
        rc = None
    if rc == 0:
        return True
    if not _SPI_WARNED:
        # Degraded, not failed: the prior alone applies and no observation is
        # recorded on this machine. Once per backend process.
        _SPI_WARNED = True
        logger.warning(
            f"Footprint interval reset unavailable (proc_reset_footprint_interval "
            f"{'missing' if function is None else f'returned {rc}'}); "
            f"memory observations are not recorded on this machine"
        )
    return False


def peak_since(pid: int, started: bool) -> Optional[int]:
    """The interval peak when the window ``started``, else the lifetime peak
    (an over-statement); ``None`` when unreadable."""
    info = _rusage(pid)
    if info is None:
        return None
    if started:
        return int(info.ri_interval_max_phys_footprint)
    return int(info.ri_lifetime_max_phys_footprint)


def pressure_level() -> Optional[int]:
    """macOS memory pressure level (1 normal, 2 warning, 4 critical), or ``None``."""
    if not _on_macos():
        return None
    import ctypes
    import ctypes.util

    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        value = ctypes.c_int(0)
        size = ctypes.c_size_t(ctypes.sizeof(value))
        rc = libc.sysctlbyname(
            b"kern.memorystatus_vm_pressure_level",
            ctypes.byref(value),
            ctypes.byref(size),
            None,
            ctypes.c_size_t(0),
        )
    except (OSError, AttributeError, TypeError):
        return None
    return int(value.value) if rc == 0 else None


def _host_port() -> Optional[int]:
    """This host's Mach port, taken once (macOS), or ``None``."""
    global _HOST_PORT
    if _HOST_PORT is None and _on_macos():
        import ctypes
        import ctypes.util

        try:
            libc = ctypes.CDLL(ctypes.util.find_library("c"))
            libc.mach_host_self.restype = ctypes.c_uint32
            _HOST_PORT = int(libc.mach_host_self())
        except (OSError, AttributeError, TypeError):
            return None
    return _HOST_PORT


def swapouts() -> Optional[int]:
    """System-wide pages swapped out since boot, or ``None``."""
    if not _on_macos():
        return None
    import ctypes
    import ctypes.util

    class _VmStatistics64(ctypes.Structure):
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

    host = _host_port()
    if host is None:
        return None
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        stats = _VmStatistics64()
        count = ctypes.c_uint32(ctypes.sizeof(stats) // 4)
        rc = libc.host_statistics64(
            ctypes.c_uint32(host), _HOST_VM_INFO64, ctypes.byref(stats), ctypes.byref(count)
        )
    except (OSError, AttributeError, TypeError):
        return None
    return int(stats.swapouts) if rc == 0 else None
