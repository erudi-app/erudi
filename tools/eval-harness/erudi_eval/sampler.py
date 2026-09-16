"""Background samplers: process/system memory -> samples.jsonl, renderer metrics -> renderer.jsonl."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import psutil

from . import machine, memory, workload as workload_mod
from .discovery import CATEGORIES, DiscoveryContext, classify_snapshot, take_snapshot
from .gpu import GpuReader
from .layout import AppLayout
from .util import MB, JsonlWriter, redact_cmdline, utc_iso


def category_totals(processes: list[dict[str, Any]], primary: str) -> dict[str, float]:
    """Sum of the primary metric per category (MB) plus app_total, app_overhead, inference_total."""
    totals = {c: 0.0 for c in CATEGORIES}
    for p in processes:
        value = p["metrics"].get(primary)
        if value is not None:
            totals[p["category"]] += value / MB
    totals["app_total"] = sum(totals[c] for c in CATEGORIES)
    totals["inference_total"] = totals["inference"]
    totals["app_overhead"] = totals["app_total"] - totals["inference"] - totals["embedding"]
    return {k: round(v, 3) for k, v in totals.items()}


def incomplete_categories(processes: list[dict[str, Any]], primary: str) -> list[str]:
    """Categories whose total under-reports because a member's primary metric
    could not be read. The per-metric rule (an unreadable value is an error,
    never a substitute) cannot hold for a sum -- so the sum says when it is
    short instead of pretending completeness."""
    return sorted({p["category"] for p in processes if p["metrics"].get(primary) is None})


@dataclass
class DiscoveryState:
    layout: AppLayout
    api_port: int
    main_pid: int | None = None
    main_exe: str = ""
    exclude_pids: set[int] = field(default_factory=lambda: {os.getpid()})

    def context(self) -> DiscoveryContext:
        return DiscoveryContext(
            main_pid=self.main_pid,
            install_dirs=self.layout.install_dirs,
            data_root=str(self.layout.data_root),
            api_port=self.api_port,
            exclude_pids=frozenset(self.exclude_pids),
        )


TOP_OTHER_EVERY_S = 5.0


class Sampler(threading.Thread):
    def __init__(self, writer: JsonlWriter, state: DiscoveryState, interval: float = 1.0, gpu: GpuReader | None = None,
                 machine_reader: machine.MachineReader | None = None, workload: workload_mod.Workload | None = None,
                 workload_every_s: float = 0.0):
        super().__init__(name="erudi-eval-sampler", daemon=True)
        self.writer, self.state, self.interval = writer, state, interval
        self.gpu = gpu or GpuReader()
        self.machine = machine_reader or machine.MachineReader(disk_path=state.layout.data_root)
        self.phase = "none"
        self.latest: dict[str, Any] | None = None
        self._halt = threading.Event()
        self._wake = threading.Event()
        self._procs: dict[tuple[int, float], psutil.Process] = {}
        self._self_proc = psutil.Process()
        self._lock = threading.Lock()
        self._prev_start: float | None = None
        self._top_other_at = -1e9
        self.workload = workload
        self.workload_every_s = workload_every_s  # 0 = every sample
        self._workload_at = -1e9
        self._wl_procs: dict[tuple[int, float], psutil.Process] = {}
        self.samples_written = 0

    def stop(self) -> None:
        self._halt.set()
        self._wake.set()

    def refresh_top_other(self) -> None:
        """Take the top-other-processes list at the next sample instead of waiting for the 5 s cadence."""
        self._top_other_at = -1e9

    def set_interval(self, seconds: float) -> None:
        """Change the sampling period and cut the current wait short, so the new rate applies at once."""
        self.interval = seconds
        self._wake.set()

    def run(self) -> None:
        while not self._halt.is_set():
            started = time.monotonic()
            # Cleared before sampling: a set_interval() that lands during the sample still wakes the next wait.
            self._wake.clear()
            try:
                self.sample_once()
            except Exception as e:  # noqa: BLE001 - a sampler crash must not end the run; it is recorded
                self.writer.write({"ts": utc_iso(), "phase": self.phase, "sampler_error": f"{type(e).__name__}: {e}"})
            self._wake.wait(max(0.0, self.interval - (time.monotonic() - started)))

    def discover(self, with_procs: bool = False):
        procs, snap_errors = take_snapshot()
        main_pid, classified = classify_snapshot(procs, self.state.context())
        if main_pid is not None and main_pid != self.state.main_pid:
            self.state.main_pid = main_pid
            self.state.main_exe = classified[main_pid].proc.exe
            self.state.layout.learn_main_exe(self.state.main_exe)
        return (classified, snap_errors, procs) if with_procs else (classified, snap_errors)

    def measure_workload(self, procs, erudi_pids) -> tuple[dict, dict]:
        """Assign workload groups on this snapshot and read each member. Returns (measured, assignment)."""
        by_pid = {p.pid: p for p in procs}
        assignment = workload_mod.assign_groups(procs, self.workload, erudi_pids=set(erudi_pids), harness_pid=self._self_proc.pid)
        alive = set()

        def read(pid: int) -> dict:
            key = (pid, by_pid[pid].create_time)
            alive.add(key)
            proc = self._wl_procs.get(key)
            if proc is None:
                try:
                    proc = psutil.Process(pid)
                except psutil.Error:
                    return {}
                self._wl_procs[key] = proc
            return memory.read_light(proc)

        measured = workload_mod.measure_groups(assignment, read)
        for key in list(self._wl_procs):
            if key not in alive:
                del self._wl_procs[key]
        return measured, assignment

    def sample_once(self) -> dict[str, Any]:
        t = time.time()
        started = time.monotonic()
        effective = round(started - self._prev_start, 3) if self._prev_start is not None else None
        self._prev_start = started
        phase = self.phase
        classified, snap_errors, procs = self.discover(with_procs=True)
        gpu = self.gpu.read()
        per_pid_gpu = gpu.get("per_pid_mb") or {}
        records = []
        alive_keys = set()
        for pid, c in classified.items():
            key = (pid, c.proc.create_time)
            alive_keys.add(key)
            proc = self._procs.get(key)
            if proc is None:
                try:
                    proc = psutil.Process(pid)
                except psutil.Error as e:
                    records.append(self._record(c, {}, {"process": type(e).__name__}, None))
                    continue
                self._procs[key] = proc
            reading = memory.read_process(proc)
            errors = dict(reading["errors"])
            if pid in snap_errors:
                errors["discovery"] = snap_errors[pid]
            records.append(self._record(c, reading["metrics"], errors, per_pid_gpu.get(pid)))
        for key in list(self._procs):
            if key not in alive_keys:
                del self._procs[key]
        sample = {
            "ts": utc_iso(t),
            "t": round(t, 3),
            "phase": phase,
            "interval_target_s": self.interval,
            "interval_effective_s": effective,
            "primary_metric": memory.PRIMARY_METRIC,
            "main_pid": self.state.main_pid,
            "processes": records,
            "totals_mb": category_totals(records, memory.PRIMARY_METRIC),
            "totals_incomplete": incomplete_categories(records, memory.PRIMARY_METRIC),
            "system": {**self.machine.read(), "gpu": {k: v for k, v in gpu.items() if k != "per_pid_mb"}},
        }
        if time.monotonic() - self._top_other_at >= TOP_OTHER_EVERY_S:
            self._top_other_at = time.monotonic()
            sample["top_other"] = machine.top_other_processes(set(classified) | {self._self_proc.pid})
        if self.workload is not None and time.monotonic() - self._workload_at >= self.workload_every_s:
            self._workload_at = time.monotonic()
            wl_started = time.monotonic()
            sample["workload"], _ = self.measure_workload(procs, classified)
            sample["workload_cost_ms"] = round(1000 * (time.monotonic() - wl_started), 1)
        # Cost of discovery + metric reads + machine context: what one sample takes from the machine.
        sample["sample_cost_ms"] = round(1000 * (time.monotonic() - started), 1)
        own = memory.read_process(self._self_proc)
        sample["harness"] = {
            "pid": self._self_proc.pid,
            memory.PRIMARY_METRIC: own["metrics"].get(memory.PRIMARY_METRIC),
            "cpu_percent": own["metrics"].get("cpu_percent"),
            **({"errors": own["errors"]} if own["errors"] else {}),
        }
        self.writer.write(sample)
        with self._lock:
            self.latest = sample
            self.samples_written += 1
        return sample

    @staticmethod
    def _record(c, metrics: dict, errors: dict, gpu_mb) -> dict[str, Any]:
        cmd = " ".join(redact_cmdline(list(c.proc.cmdline)))
        rec = {
            "pid": c.pid,
            "ppid": c.proc.ppid,
            "name": c.proc.name,
            "category": c.category,
            "role": c.role,
            "cmd": cmd[:400],
            "metrics": metrics,
        }
        if gpu_mb is not None:
            rec["gpu_mem_mb"] = gpu_mb
        if errors:
            rec["errors"] = errors
        return rec

    def categories_present(self) -> set[str]:
        """Categories seen in a fresh discovery (not the last sample, which can be a second old)."""
        classified, _ = self.discover()
        return {c.category for c in classified.values()}

    def member_pids(self) -> dict[int, str]:
        classified, _ = self.discover()
        return {pid: f"{c.category}:{c.role}:{c.proc.name}" for pid, c in classified.items()}


class RendererSampler(threading.Thread):
    """Every `interval` seconds while connected, record renderer metrics tagged with the current phase."""

    def __init__(self, writer: JsonlWriter, cdp, sampler: Sampler, interval: float = 5.0):
        super().__init__(name="erudi-eval-renderer", daemon=True)
        self.writer, self.cdp, self.sampler, self.interval = writer, cdp, sampler, interval
        self._halt = threading.Event()

    def stop(self) -> None:
        self._halt.set()

    def record(self, trigger: str, page: str | None = None) -> dict[str, Any] | None:
        if not self.cdp or not self.cdp.connected:
            return None
        try:
            metrics = self.cdp.renderer_metrics()
            row = {"ts": utc_iso(), "phase": self.sampler.phase, "trigger": trigger, "page": page or metrics.get("location_hash"), "metrics": metrics}
        except Exception as e:  # noqa: BLE001 - renderer busy or reloading; recorded, not fatal
            row = {"ts": utc_iso(), "phase": self.sampler.phase, "trigger": trigger, "page": page, "error": f"{type(e).__name__}: {e}"}
        self.writer.write(row)
        return row

    def run(self) -> None:
        while not self._halt.wait(self.interval):
            self.record("periodic")
