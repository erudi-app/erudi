"""Runs the scenario: phase selection, samplers, storage snapshots, results files.

A phase (see phases.py) raises `Skip` (precondition missing, with the reason) or
`PhaseFailed` (the thing it measures did not happen); any other exception is also
a failure. A failed phase never stops the run: later phases check their own
preconditions. Only a preflight refusal (`Abort`) ends the run early.
"""

from __future__ import annotations

import platform
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import flavour as flavour_mod
from . import lifecycle, storage
from .api import ErudiApi, parse_ndjson, summarize_turn
from .cdp import CdpClient
from .layout import AppLayout, app_version
from .sampler import DiscoveryState, RendererSampler, Sampler
from .workload import Workload, default_workload_path, load_workload, workload_drift
from .util import JsonlWriter, utc_iso, wait_until, write_json

HARNESS_DIR = Path(__file__).resolve().parent.parent
OPT_IN = ("kb_stress", "web_search", "arena")
# Phases after which storage is snapshotted (they write to the data root or the database).
WRITING_PHASES = {"model_ready", "chat_cold", "chat_warm", "ui_stream", "long_conversation_render", "embedding_model", "kb_ingest", "kb_query", "kb_stress", "web_search", "cleanup"}


class Skip(Exception):
    pass


class PhaseFailed(Exception):
    pass


class Abort(Exception):
    """Stops the whole run (preflight refusal)."""


@dataclass
class Config:
    results_dir: Path
    corpus_dir: Path
    app_path: str | None = None
    attach: bool = False
    phases: list[str] | None = None
    skip: list[str] = field(default_factory=list)
    opt_in: list[str] = field(default_factory=list)
    model_link: str | None = None
    no_download: bool = False
    profile: str = "clean"
    flavour: str | None = None
    workload_path: str | None = None
    strict_workload: bool = False
    workload_drift_pct: float = 25.0
    workload_busy_cpu: float = 20.0
    idle_seconds: float = 60
    baseline_seconds: float = 60
    boot_sample_interval: float = 0.25
    warm_turns: int = 5
    long_turns: int = 20
    disk_headroom_gb: float = 5.0
    cleanup: bool = False
    leave_running: bool = False
    sample_interval: float = 1.0
    renderer_interval: float = 5.0
    settle_seconds: float | None = None  # overrides every settle (testing only)
    unload_timeout: float = 660
    download_timeout: float = 7200
    turn_timeout: float = 900
    api_port: int = 27182
    cdp_port: int = 9222
    main_pid: int | None = None
    data_root: str | None = None
    backend_log_dir: str | None = None
    capture_log: str | None = None


@dataclass
class PhaseResult:
    name: str
    status: str
    reason: str | None = None
    started: str | None = None
    ended: str | None = None
    duration_s: float | None = None
    data: dict[str, Any] = field(default_factory=dict)


class Run:
    def __init__(self, cfg: Config, layout: AppLayout, run_id: str | None = None):
        self.cfg, self.layout = cfg, layout
        if cfg.data_root:
            layout.data_root = Path(cfg.data_root)
        if cfg.backend_log_dir:
            layout.backend_log_dir = Path(cfg.backend_log_dir)
        if cfg.capture_log:
            layout.capture_logs = [Path(cfg.capture_log)]
        self.flavour = flavour_mod.load(HARNESS_DIR, cfg.flavour or flavour_mod.default_name(layout.os_name, app_version(layout)["flavour"]))
        self.flavour_detected: dict[str, Any] = {}
        self.workload: Workload | None = self._load_workload()
        self.run_id = run_id or make_run_id(layout, cfg.profile, self.flavour.name)
        self.dir = Path(cfg.results_dir) / self.run_id
        (self.dir / "storage").mkdir(parents=True, exist_ok=True)
        self.events = JsonlWriter(self.dir / "events.jsonl")
        self.api = ErudiApi(port=cfg.api_port, run_tag=self.run_id.split("-")[0])
        self.cdp = CdpClient(port=cfg.cdp_port)
        self.state = DiscoveryState(layout, cfg.api_port, main_pid=cfg.main_pid)
        # The workload costs ~11 ms per sample for ~60 processes (measured on an M4), against ~70 ms for the
        # rest of a sample: cheap enough to record on every sample rather than on a slower cadence.
        self.sampler = Sampler(JsonlWriter(self.dir / "samples.jsonl"), self.state, interval=cfg.sample_interval,
                               workload=self.workload, workload_every_s=0.0)
        self.renderer = RendererSampler(JsonlWriter(self.dir / "renderer.jsonl"), self.cdp, self.sampler, interval=cfg.renderer_interval)
        self.t0 = time.time()
        self.results: list[PhaseResult] = []
        self.created: dict[str, list[int]] = {"conversations": [], "assistants": [], "models": []}
        self.settings_to_restore: dict[str, Any] = {}
        self.launched_proc = None
        self.model: dict[str, Any] | None = None
        self.conversation_id: int | None = None
        self.kb_assistant_id: int | None = None
        self.embedding_ready = False
        self.last_use_t: float | None = None
        self.last_snapshot: dict[str, Any] | None = None
        self.snapshot_count = 0
        self.app_prepared = False

    def _load_workload(self) -> Workload | None:
        path = Path(self.cfg.workload_path) if self.cfg.workload_path else default_workload_path(HARNESS_DIR, self.cfg.profile, self.layout.os_name)
        if path is None:
            return None
        wl = load_workload(path)
        if wl.profile != self.cfg.profile:
            raise ValueError(f"workload file {path} declares profile {wl.profile!r}, run asks for {self.cfg.profile!r}")
        return wl

    def model_link(self) -> str:
        """The model this flavour measures: MLX on macOS, GGUF on the llama.cpp builds."""
        return self.cfg.model_link or self.flavour.default_model_link

    def check_flavour(self) -> tuple[bool, str]:
        """Compare the requested flavour with what the running app reports. Caches the detection."""
        startup = self.api.get("/hardware/app_startup")
        diagnostics = self.api.get("/diagnostics/?limit=1")
        environment = (diagnostics.body or {}).get("environment", {}) if diagnostics.ok else {}
        detected = flavour_mod.detect(self.layout.os_name, startup.body if startup.ok else {}, environment)
        ok, message = flavour_mod.check(self.flavour, detected)
        self.flavour_detected = {**detected, "ok": ok, "message": message}
        self.event("flavour_check", requested=self.flavour.name, ok=ok, message=message, **detected)
        return ok, message

    def record_flavour_facts(self) -> dict[str, Any]:
        """Once a model is loaded: inference process shape, per-process VRAM, and the flavour's extra facts."""
        sample = self.sampler.latest or {}
        processes = sample.get("processes", [])
        shape_ok, role_seen = flavour_mod.inference_matches(self.flavour, processes)
        inference = [p for p in processes if p.get("category") == "inference"]
        cmdline = inference[0].get("cmd", "") if inference else ""
        facts: dict[str, Any] = {
            "flavour": self.flavour.name,
            "inference_role_expected": self.flavour.inference_role,
            "inference_role_seen": role_seen,
            "inference_shape_ok": shape_ok,
            "gpu_memory_expectation": flavour_mod.gpu_expectation(self.flavour),
            **flavour_mod.extra_check_values(self.flavour, cmdline),
        }
        if self.flavour.per_process_gpu_memory == "nvml":
            reported = [p.get("gpu_mem_mb") for p in inference]
            facts["inference_gpu_mem_mb"] = reported[0] if reported else None
            if not reported or reported[0] is None:
                facts["gpu_memory_note"] = "per-process VRAM not reported by NVML/nvidia-smi (common under WDDM): recorded as unavailable"
        if "gpu_detail" in self.flavour.extra_checks:
            detailed = self.api.get("/hardware/detailed")
            hardware = (detailed.body or {}).get("hardware", {}) if detailed.ok else {}
            facts["gpu"] = {k: hardware.get(k) for k in ("gpu_name", "vram_total_gb", "vram_available_gb", "cuda_version", "compute_capability") if hardware.get(k) is not None}
        self.event("flavour_facts", **facts)
        return facts

    # --- helpers ------------------------------------------------------------------------

    def event(self, kind: str, **data: Any) -> None:
        self.events.write({"ts": utc_iso(), "type": kind, "phase": self.sampler.phase, **data})

    def settle(self, seconds: float) -> None:
        time.sleep(self.cfg.settle_seconds if self.cfg.settle_seconds is not None else seconds)

    def require_app(self) -> None:
        if not self.api.health_ok():
            raise Skip(f"app not answering on 127.0.0.1:{self.cfg.api_port}")
        self.prepare_app()

    def require_cdp(self) -> None:
        if self.cdp.connected:
            return
        if not self.cdp.version():
            raise Skip(f"no CDP endpoint on 127.0.0.1:{self.cfg.cdp_port} (launch the app with --remote-debugging-port)")
        try:
            self.cdp.connect()
        except Exception as e:  # noqa: BLE001 - reported as the skip reason
            raise Skip(f"CDP connect failed: {e}") from e

    def require_model(self) -> dict[str, Any]:
        if not self.model:
            raise Skip("no model bound (model_ready did not succeed)")
        return self.model

    def snapshot_storage(self, label: str) -> dict[str, Any]:
        install = {"app": self.layout.app_path, "resources": self.layout.resources_dir, "backend_lib": self.layout.backend_lib}
        logs = {"backend_log_dir": self.layout.backend_log_dir, **{p.name: p for p in self.layout.capture_logs}}
        if self.layout.electron_log_dir != self.layout.backend_log_dir and not self.cfg.backend_log_dir:
            logs["electron_log_dir"] = self.layout.electron_log_dir  # same directory on case-insensitive APFS
        snap = storage.snapshot(label, install, self.layout.data_root, logs)
        snap["ts"] = utc_iso()
        snap["phase"] = self.sampler.phase
        if self.last_snapshot is not None:
            snap["delta_from"] = self.last_snapshot["label"]
            snap["delta"] = storage.delta(self.last_snapshot, snap, min_bytes=4096)
        self.snapshot_count += 1
        write_json(self.dir / "storage" / f"{self.snapshot_count:02d}-{label}.json", snap)
        self.last_snapshot = snap
        return snap

    def message_count(self, conversation_id: int) -> tuple[int, int]:
        r = self.api.get(f"/conversations/{conversation_id}/fetch_messages")
        if not r.ok or not isinstance(r.body, list):
            raise PhaseFailed(f"fetch_messages -> {r.status}")
        return len(r.body), sum(1 for m in r.body if m.get("sender") == "user")

    def resident_model_id(self) -> Any:
        r = self.api.get("/diagnostics/?limit=1")
        return (r.body or {}).get("environment", {}).get("loaded_model_id") if r.ok else "unavailable"

    def create_conversation(self, llm_id: int, web_search: bool = False) -> int:
        r = self.api.post("/conversations/", {"llm_id": llm_id, "web_search_enabled": web_search})
        if r.status != 201 or not isinstance(r.body, dict):
            raise PhaseFailed(f"create conversation -> {r.status}: {r.body}")
        conv_id = int(r.body["id"])
        self.created["conversations"].append(conv_id)
        bound = self.api.get(f"/conversations/{conv_id}").body or {}
        if bound.get("llm_id") != llm_id:  # verify the state, never assume it
            raise PhaseFailed(f"conversation {conv_id} bound to llm {bound.get('llm_id')}, expected {llm_id}")
        self.event("conversation_created", conversation_id=conv_id, llm_id=llm_id, web_search_enabled=web_search)
        return conv_id

    def run_turn(self, conversation_id: int, question: str, label: str, model_ids: set[int]) -> dict[str, Any]:
        before, _ = self.message_count(conversation_id)
        resident = self.resident_model_id()
        turn_start_wall = time.time()
        sent = time.monotonic()
        stream = self.api.stream_lines(f"/conversations/{conversation_id}/query", {"question": question}, read_timeout=self.cfg.turn_timeout)
        first = next(stream)
        request_id = first[1].split(":", 1)[1]
        metrics = summarize_turn(sent, parse_ndjson(stream))
        stream.close()
        self.last_use_t = time.time()
        # Messages are stored after the stream ends: the turn counts only once both are persisted.
        after = wait_until(lambda: self.message_count(conversation_id)[0] >= before + 2, timeout=30, interval=1)
        row = {
            "conversation_id": conversation_id,
            "label": label,
            "question": question,
            "request_id": request_id,
            "resident_before": resident,
            "cold": resident not in model_ids,
            "persisted": after is not None,
            "wall_start": turn_start_wall,
            **metrics.as_dict(),
        }
        self.event("turn", **row)
        if not metrics.done or metrics.errors or after is None:
            raise PhaseFailed(f"turn did not complete: done={metrics.done} errors={metrics.errors} persisted={after is not None}")
        return row

    def prepare_app(self) -> None:
        """Once the app answers: disable auto-update for the run (restored in finish)."""
        if self.app_prepared:
            return
        self.app_prepared = True
        r = self.api.get("/user_settings/")
        if r.ok and isinstance(r.body, dict) and r.body.get("auto_update_enabled"):
            put = self.api.put("/user_settings/", {"auto_update_enabled": False})
            if put.ok:
                self.settings_to_restore["auto_update_enabled"] = True
                self.event("settings_change", field="auto_update_enabled", old=True, new=False)

    def restore_settings(self) -> None:
        if not self.settings_to_restore:
            return
        try:
            r = self.api.put("/user_settings/", dict(self.settings_to_restore))
            self.event("settings_restore", fields=self.settings_to_restore, status=r.status)
            if r.ok:
                self.settings_to_restore = {}
        except OSError as e:
            self.event("error", where="settings_restore", error=str(e))

    def copy_logs(self) -> None:
        try:
            counts = lifecycle.copy_logs(self.dir / "logs", self.t0, self.layout.capture_logs, self.layout.backend_log_dir)
            self.event("logs_copied", files=counts)
        except OSError as e:
            self.event("error", where="copy_logs", error=str(e))

    # --- driver ------------------------------------------------------------------------

    def selected(self, name: str) -> tuple[bool, str | None]:
        if name in self.cfg.skip:
            return False, "skipped by --skip"
        if self.cfg.phases and name not in self.cfg.phases:
            return False, "not selected by --phases"
        if name in OPT_IN and name not in self.cfg.opt_in:
            return False, "opt-in phase (enable with --opt-in)"
        if name == "cleanup" and not self.cfg.cleanup:
            return False, "cleanup runs only with --cleanup"
        return True, None

    def execute(self) -> Path:
        write_json(self.dir / "config.json", {k: str(v) if isinstance(v, Path) else v for k, v in self.cfg.__dict__.items()} | {"run_id": self.run_id})
        self.sampler.phase = "preflight"
        self.sampler.start()
        self.renderer.start()
        aborted = None
        try:
            from .phases import PHASES

            for name, fn in PHASES:
                ok, reason = self.selected(name)
                if name == "preflight":
                    ok, reason = True, None  # never skippable: it is the safety gate
                if not ok:
                    self.results.append(PhaseResult(name, "skipped", reason))
                    self.event("phase_skipped", name=name, reason=reason)
                    continue
                self.run_phase(name, fn)
                if name == "preflight" and self.results[-1].status != "ok":
                    aborted = self.results[-1].reason
                    break
        except KeyboardInterrupt:
            aborted = "interrupted by user"
            self.event("aborted", reason=aborted)
            if self.launched_proc is not None and not self.cfg.leave_running and self.state.main_pid:
                from .phases import phase_quit

                self.run_phase("quit", phase_quit)
        finally:
            self.finish(aborted)
        return self.dir

    def run_phase(self, name: str, fn: Callable[["Run"], dict | None]) -> None:
        self.sampler.phase = name
        started = time.time()
        self.event("phase_start", name=name)
        result = PhaseResult(name, "ok", started=utc_iso(started))
        try:
            result.data = fn(self) or {}
        except Skip as s:
            result.status, result.reason = "skipped", str(s)
        except (PhaseFailed, Abort) as e:
            result.status, result.reason = "failed", str(e)
        except Exception as e:  # noqa: BLE001 - a crash in one phase is a recorded failure
            result.status, result.reason = "failed", f"{type(e).__name__}: {e}"
            self.event("error", where=name, traceback=traceback.format_exc())
        result.ended = utc_iso()
        result.duration_s = round(time.time() - started, 3)
        self.results.append(result)
        self.event("phase_end", name=name, status=result.status, reason=result.reason, duration_s=result.duration_s)
        if name in WRITING_PHASES and result.status != "skipped":
            try:
                self.snapshot_storage(f"after_{name}")
            except OSError as e:
                self.event("error", where="storage_snapshot", error=str(e))

    def record_workload_drift(self) -> None:
        """A workload group that changed size, moved in memory or got busy makes the numbers less comparable."""
        if self.workload is None:
            return
        from .util import read_jsonl

        drift = workload_drift(read_jsonl(self.dir / "samples.jsonl"), self.cfg.workload_drift_pct, self.cfg.workload_busy_cpu)
        self.event("workload_drift", reference_phase=drift["reference_phase"], warnings=drift["warnings"])

    def finish(self, aborted: str | None) -> None:
        from . import report

        self.sampler.phase = "finish"
        if self.api.health_ok():
            self.restore_settings()
        self.copy_logs()
        self.record_workload_drift()
        if not any(r.name == "quit" and r.status == "ok" for r in self.results):
            try:
                self.snapshot_storage("end")
            except OSError:
                pass
        self.renderer.stop()
        self.sampler.stop()
        self.sampler.join(timeout=10)
        self.cdp.close()
        write_json(self.dir / "phases.json", [r.__dict__ for r in self.results])
        if aborted:
            write_json(self.dir / "aborted.json", {"reason": aborted})
        report.write_report(self.dir)
        self.events.close()


def make_run_id(layout: AppLayout, profile: str = "clean", flavour_name: str = "unknown") -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    machine = platform.machine().lower().replace("x86_64", "x64").replace("amd64", "x64").replace("aarch64", "arm64")
    return f"{stamp}-{machine}-{flavour_name}-{profile}"


