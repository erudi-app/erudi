"""Per-process and system-wide memory readers, one code path per OS.

Metric names are recorded verbatim; a metric that cannot be read is reported in
`errors` under its own name and never replaced by another metric.
"""

from __future__ import annotations

import ctypes
import os
import sys
from typing import Any

import psutil

IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform.startswith("win")
IS_LINUX = sys.platform.startswith("linux")

PRIMARY_METRIC = "phys_footprint" if IS_MAC else "private_working_set" if IS_WIN else "pss"
PEAK_METRIC = "lifetime_max_phys_footprint" if IS_MAC else "peak_wset" if IS_WIN else None


# --- macOS: proc_pid_rusage(RUSAGE_INFO_V4) -----------------------------------------------

RUSAGE_INFO_V4 = 4


class RusageInfoV4(ctypes.Structure):
    """struct rusage_info_v4 from <sys/resource.h>."""

    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups",
            "ri_pageins", "ri_wired_size", "ri_resident_size", "ri_phys_footprint",
            "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
            "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
            "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
            "ri_diskio_byteswritten", "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance",
            "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
            "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive",
            "ri_billed_system_time", "ri_serviced_system_time", "ri_logical_writes",
            "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles",
            "ri_billed_energy", "ri_serviced_energy", "ri_interval_max_phys_footprint",
            "ri_runnable_time",
        )
    ]


_libproc = None


def mac_rusage(pid: int) -> dict[str, int]:
    """phys_footprint and lifetime max for `pid`. Raises OSError when unreadable."""
    global _libproc
    if _libproc is None:
        _libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        _libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(RusageInfoV4)]
        _libproc.proc_pid_rusage.restype = ctypes.c_int
    info = RusageInfoV4()
    if _libproc.proc_pid_rusage(pid, RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return {
        "phys_footprint": info.ri_phys_footprint,
        "lifetime_max_phys_footprint": info.ri_lifetime_max_phys_footprint,
    }


# --- per process ---------------------------------------------------------------------------


def read_process(proc: psutil.Process) -> dict[str, Any]:
    """Memory, CPU and thread metrics for one process. Never raises."""
    metrics: dict[str, Any] = {}
    errors: dict[str, str] = {}

    def attempt(name: str, fn):
        try:
            fn()
        except (psutil.Error, OSError, AttributeError) as e:
            errors[name] = type(e).__name__ + (f": {e}" if isinstance(e, OSError) else "")

    if IS_MAC:
        attempt("phys_footprint", lambda: metrics.update(mac_rusage(proc.pid)))
        attempt("rss", lambda: metrics.__setitem__("rss", proc.memory_info().rss))
        attempt("uss", lambda: metrics.__setitem__("uss", proc.memory_full_info().uss))
    elif IS_WIN:
        def win_info():
            mi = proc.memory_info()
            metrics["rss"] = mi.rss
            for field in ("peak_wset", "private", "pagefile"):
                if hasattr(mi, field):
                    metrics[field] = getattr(mi, field)
                else:
                    errors[field] = "not provided by psutil"
        attempt("rss", win_info)
        # psutil's USS on Windows is the private working set.
        attempt("private_working_set", lambda: metrics.__setitem__("private_working_set", proc.memory_full_info().uss))
    else:
        def linux_full():
            mf = proc.memory_full_info()
            metrics.update(rss=mf.rss, uss=mf.uss, pss=mf.pss, swap=mf.swap)
        attempt("pss", linux_full)

    attempt("cpu_percent", lambda: metrics.__setitem__("cpu_percent", proc.cpu_percent(None)))
    attempt("threads", lambda: metrics.__setitem__("threads", proc.num_threads()))
    return {"metrics": metrics, "errors": errors}


def read_light(proc: psutil.Process) -> dict[str, Any]:
    """Primary metric, RSS and CPU % only (for workload groups): each is None when unreadable, never substituted."""
    out: dict[str, Any] = {"primary": None, "rss": None, "cpu": None}
    try:
        if IS_MAC:
            out["primary"] = mac_rusage(proc.pid)["phys_footprint"]
        elif IS_WIN:
            out["primary"] = proc.memory_full_info().uss
        else:
            out["primary"] = proc.memory_full_info().pss
    except (psutil.Error, OSError, AttributeError):
        pass  # counted as unreadable by the caller
    try:
        out["rss"] = proc.memory_info().rss
    except (psutil.Error, OSError):
        pass  # counted as unreadable by the caller
    try:
        out["cpu"] = proc.cpu_percent(None)
    except (psutil.Error, OSError):
        pass  # a process that exited mid-read has no CPU figure
    return out
