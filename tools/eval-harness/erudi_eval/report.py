"""summary.json and report.md from a run directory (works on partial runs too)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import context_report, workload_report
from .discovery import CATEGORIES
from .util import MB, read_jsonl, write_json

DERIVED = ("app_overhead", "inference_total", "app_total")
PEAK_METRICS = ("lifetime_max_phys_footprint", "peak_wset")


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def memory_by_phase(samples: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    """{phase: {category|derived: {mean_mb, peak_mb, samples[, incomplete_samples]}}}
    over per-sample category totals. ``incomplete_samples`` counts the samples
    whose total for that category under-reports because a member process's
    primary metric could not be read (``totals_incomplete`` on the sample) --
    without it the headline table would silently substitute zero for the
    unreadable share. The derived sums inherit the mark from any category."""
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    short: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    order: list[str] = []
    for s in samples:
        totals = s.get("totals_mb")
        if not totals:
            continue
        phase = s.get("phase", "none")
        if phase not in order:
            order.append(phase)
        missing = set(s.get("totals_incomplete") or ())
        for key in (*CATEGORIES, *DERIVED):
            acc[phase][key].append(float(totals.get(key, 0.0)))
            # inference_total derives from inference alone; the other two sums
            # are touched by any short category.
            derived_short = bool(missing) if key in ("app_total", "app_overhead") else "inference" in missing
            if key in missing or (key in DERIVED and derived_short):
                short[phase][key] += 1
    out = {}
    for phase in order:
        out[phase] = {
            key: {
                "mean_mb": round(sum(v) / len(v), 1),
                "peak_mb": round(max(v), 1),
                "samples": len(v),
                **({"incomplete_samples": short[phase][key]} if short[phase].get(key) else {}),
            }
            for key, v in acc[phase].items()
        }
    return out


def process_peaks(samples: list[dict[str, Any]], primary: str) -> list[dict[str, Any]]:
    """Per process: the OS-reported peak (macOS lifetime max / Windows peak working set) or the max primary sample."""
    peaks: dict[int, dict[str, Any]] = {}
    for s in samples:
        for p in s.get("processes", []):
            m = p.get("metrics", {})
            os_peak = next((m[k] for k in PEAK_METRICS if m.get(k) is not None), None)
            value, source = (os_peak, "os_peak") if os_peak is not None else (m.get(primary), f"max_{primary}")
            if value is None:
                continue
            row = peaks.setdefault(p["pid"], {"pid": p["pid"], "name": p.get("name"), "category": p["category"], "role": p.get("role"), "peak_mb": 0.0, "source": source})
            row["peak_mb"] = max(row["peak_mb"], round(value / MB, 1))
    return sorted(peaks.values(), key=lambda r: r["peak_mb"], reverse=True)


def read_anomalies(samples, phases, events) -> dict[str, Any]:
    unreadable: dict[str, dict[str, Any]] = {}
    missing_primary: dict[str, int] = defaultdict(int)
    metrics_seen: set[str] = set()
    sampler_errors = 0
    for s in samples:
        if "sampler_error" in s:
            sampler_errors += 1
            continue
        primary = s.get("primary_metric")
        for p in s.get("processes", []):
            metrics_seen.update(p.get("metrics", {}).keys())
            for metric, err in (p.get("errors") or {}).items():
                row = unreadable.setdefault(metric, {"reads": 0, "processes": set(), "example": err})
                row["reads"] += 1
                row["processes"].add(f"{p['pid']} {p.get('name')} ({p['category']})")
            if primary not in p.get("metrics", {}):
                missing_primary[p["category"]] += 1
    quit_rows = [e for e in events if e.get("type") == "quit_result"]
    return {
        "survivors_after_quit": quit_rows[-1]["survivors"] if quit_rows else None,
        "unreadable_metrics": {k: {"reads": v["reads"], "example_error": v["example"], "processes": sorted(v["processes"])[:10]} for k, v in unreadable.items()},
        "primary_metric_missing_reads": dict(missing_primary),
        "metrics_available": sorted(metrics_seen),
        "metric_substitutions": "none: missing metrics are recorded as errors, never replaced",
        "phases_not_ok": [{"phase": r["name"], "status": r["status"], "reason": r.get("reason")} for r in phases if r["status"] != "ok"],
        "sampler_errors": sampler_errors,
        "harness_errors": [e.get("where") for e in events if e.get("type") == "error"],
    }


def system_by_phase(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per phase: min available memory, max swap, pressure levels seen, max GPU memory used (MB)."""
    out: dict[str, dict[str, Any]] = {}
    for s in samples:
        block = s.get("system")
        if not block or "phase" not in s:
            continue
        row = out.setdefault(s["phase"], {"min_available_mb": None, "max_swap_used_mb": None, "pressure": set(), "max_gpu_used_mb": None, "max_commit_mb": None})

        def keep(key: str, mb: float | None, pick) -> None:
            if mb is not None:
                row[key] = round(mb, 1) if row[key] is None else pick(row[key], round(mb, 1))

        to_mb = lambda v: None if v is None else v / MB  # noqa: E731
        keep("min_available_mb", to_mb(block.get("available_bytes")), min)
        keep("max_swap_used_mb", to_mb(block.get("swap_used_bytes")), max)
        keep("max_commit_mb", to_mb(block.get("commit_charge_bytes")), max)
        if block.get("memory_pressure_level"):
            row["pressure"].add(block["memory_pressure_level"])
        for gpu in (block.get("gpu") or {}).get("gpus") or []:
            keep("max_gpu_used_mb", gpu.get("used_mb"), max)  # NVML/nvidia-smi values are already MB
    for row in out.values():
        row["pressure"] = sorted(row["pressure"])
    return out


def renderer_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Explicit records (page stops, streaming, reload) plus, per phase, the max of periodic records."""
    out, periodic = [], {}
    for r in rows:
        if "metrics" not in r:
            continue
        m = r["metrics"]
        row = {
            "phase": r["phase"],
            "page": r.get("page"),
            "trigger": r.get("trigger"),
            # An unreadable heap stays blank: 0.0 would be the substitute-zero
            # this report forbids everywhere else.
            "js_heap_used_mb": (
                round(m["JSHeapUsedSize"] / MB, 1) if m.get("JSHeapUsedSize") is not None else None
            ),
            "nodes": m.get("Nodes"),
            "documents": m.get("Documents"),
            "listeners": m.get("JSEventListeners"),
            "layouts": m.get("LayoutCount"),
            "recalc_styles": m.get("RecalcStyleCount"),
        }
        if r.get("trigger") == "periodic":
            cur = periodic.get(r["phase"])
            if cur is None or (row["nodes"] or 0) > (cur["nodes"] or 0):
                periodic[r["phase"]] = {**row, "trigger": "periodic (max nodes)"}
        elif r.get("trigger") != "streaming":
            out.append(row)
        else:
            key = (r["phase"], "streaming")
            cur = periodic.get(key)
            if cur is None or (row["nodes"] or 0) > (cur["nodes"] or 0):
                periodic[key] = {**row, "trigger": "streaming (max nodes)"}
    return out + list(periodic.values())


def build_summary(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    samples = read_jsonl(run_dir / "samples.jsonl")
    events = read_jsonl(run_dir / "events.jsonl")
    phases = _load_json(run_dir / "phases.json", [])
    system = _load_json(run_dir / "system.json", {})
    snapshots = [_load_json(p, {}) for p in sorted((run_dir / "storage").glob("*.json"))]
    primary = next((s["primary_metric"] for s in samples if s.get("primary_metric")), None)
    boot = next((e["rows"] for e in events if e.get("type") == "boot_timeline"), [])
    turns = [
        {k: e.get(k) for k in ("phase", "label", "conversation_id", "cold", "resident_before", "ttft_s", "first_event_type", "wall_s", "generation_s", "tool_time_s", "answer_chars", "thinking_chars", "answer_chars_per_s", "total_chars_per_s", "chars_per_s_note", "tool_calls", "done", "persisted")}
        for e in events
        if e.get("type") == "turn"
    ]
    turns += [{"phase": e["phase"], "label": "ui", "wall_s": e.get("duration_s"), "submit_method": e.get("submit_method")} for e in events if e.get("type") == "ui_turn"]
    turns += [{"phase": e["phase"], "label": "arena", "ttft_s": e.get("ttft_s"), "wall_s": e.get("duration_s"), "answer_chars": e.get("chars")} for e in events if e.get("type") == "arena_turn"]
    return {
        "run_id": _load_json(run_dir / "config.json", {}).get("run_id", run_dir.name),
        "aborted": _load_json(run_dir / "aborted.json", {}).get("reason"),
        "primary_metric": primary,
        "system": {k: system.get(k) for k in ("os", "cpu", "memory_total_bytes", "gpus", "app", "harness_version", "free_disk_bytes")},
        "phases": phases,
        "memory": memory_by_phase(samples),
        "system_by_phase": system_by_phase(samples),
        "process_peaks": process_peaks(samples, primary)[:25] if primary else [],
        "boot_timeline": boot,
        "boot_memory": context_report.boot_memory(boot, samples),
        "machine_context": context_report.machine_context(samples),
        "machine_static": system.get("machine"),
        "profile": system.get("profile"),
        "flavour": {**(system.get("flavour") or {}),
                    "facts": next((e for e in reversed(events) if e.get("type") == "flavour_facts"), {}),
                    "check": next((e for e in reversed(events) if e.get("type") == "flavour_check"), {})},
        "model": next((e for e in reversed(events) if e.get("type") == "model_bound"), {}),
        "download": next((e for e in reversed(events) if e.get("type") == "download"), {}),
        "conditions": system.get("conditions"),
        "workload": workload_report.build(run_dir, samples, events, system),
        "turns": turns,
        "storage": {
            "size_kind": snapshots[0].get("size_kind") if snapshots else None,
            "install": snapshots[0].get("install") if snapshots else {},
            "data": snapshots[-1].get("data") if snapshots else {},
            "logs": snapshots[-1].get("logs") if snapshots else {},
            "deltas": [{"snapshot": s.get("label"), "phase": s.get("phase"), "rows": s.get("delta", [])} for s in snapshots[1:]],
        },
        "renderer": renderer_rows(read_jsonl(run_dir / "renderer.jsonl")),
        "anomalies": read_anomalies(samples, phases, events),
    }


# --- markdown ---------------------------------------------------------------------------


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_none_\n"
    def fmt(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.3f}" if 0 < abs(v) < 0.1 else f"{v:.2f}"
        return str(v).replace("|", "\\|")

    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _flavour_section(s: dict[str, Any]) -> str:
    fl = s.get("flavour") or {}
    facts, check = fl.get("facts") or {}, fl.get("check") or {}
    model, download = s.get("model") or {}, s.get("download") or {}
    rows = [
        ["requested flavour", fl.get("requested")],
        ["detected on the running app", check.get("message") or "not checked (app never answered)"],
        ["inference process", f"expected `{facts.get('inference_role_expected') or (fl.get('backend_type') == 'mlx' and 'mlx_child' or 'llama_server')}`, seen `{facts.get('inference_role_seen')}`" if facts else "not observed (no turn ran)"],
        ["per-process GPU memory", fl.get("gpu") or facts.get("gpu_memory_expectation")],
    ]
    for key, label in (("threads", "llama-server --threads"), ("ngl", "llama-server -ngl"), ("inference_gpu_mem_mb", "inference VRAM MB"), ("gpu", "GPU reported by the app")):
        if facts.get(key) is not None:
            rows.append([label, facts[key]])
    if facts.get("gpu_memory_note"):
        rows.append(["GPU memory note", facts["gpu_memory_note"]])
    if fl.get("notes"):
        rows.append(["flavour file notes", fl["notes"]])
    rows.append(["model requested", fl.get("model_link")])
    if model:
        matched = model.get("matched_by")
        rows.append(["model bound", f"id {model.get('llm_id')} ({model.get('name')}), matched by {matched}"])
        if matched == "name":
            rows.append(["model match caveat", model.get("note")])
        if model.get("installed_link") and model.get("installed_link") != fl.get("model_link"):
            rows.append(["installed row link", model["installed_link"]])
    if download:
        rows.append(["downloaded by the harness", f"{_mb(download.get('bytes'))} MB in {download.get('duration_s')} s ({download.get('throughput_mb_s')} MB/s)"])
    return _table(["", ""], rows)


def _conditions_line(c: dict[str, Any]) -> str:
    if not c:
        return ""
    return (f"- Conditions at start: uptime {c.get('uptime_human')}, swap {_mb(c.get('swap_used_bytes'))} / {_mb(c.get('swap_total_bytes'))} MB used, "
            f"available {_mb(c.get('available_bytes'))} MB, pressure {c.get('memory_pressure_level')}, {c.get('power_source')} "
            f"{c.get('battery_percent')}%, low power mode {c.get('low_power_mode')}, OS build {c.get('os_build')}, displays {c.get('display_count')}\n")


def _mb(b: Any) -> Any:
    return round(b / MB, 1) if isinstance(b, (int, float)) else None


def render_markdown(s: dict[str, Any]) -> str:
    sysinfo = s["system"]
    app = sysinfo.get("app") or {}
    out = [f"# erudi-eval report — {s['run_id']}\n"]
    if s.get("aborted"):
        out.append(f"**Run aborted:** {s['aborted']}\n")
    out.append(
        f"- OS: {(sysinfo.get('os') or {}).get('system')} {(sysinfo.get('os') or {}).get('release')} "
        f"({(sysinfo.get('os') or {}).get('machine')}); CPU: {(sysinfo.get('cpu') or {}).get('model')}; "
        f"RAM: {_mb(sysinfo.get('memory_total_bytes'))} MB\n"
        f"- App: {app.get('path')} version {app.get('version')} flavour {app.get('flavour')}\n"
        f"- Primary memory metric: `{s['primary_metric']}` (MB below are sums of it per category)\n"
        "- Throughput is in characters per second: the API does not expose token counts.\n"
    )
    fl = s.get("flavour") or {}
    out.append(f"- Flavour: `{fl.get('requested')}` (backend {fl.get('backend_type')}, engine {fl.get('engine')}, model `{fl.get('model_link')}`)\n")
    wl_file = (s.get("workload") or {}).get("workload_file")
    out.append(f"- Profile: `{s.get('profile')}`" + (f" (workload `{wl_file}`)\n" if wl_file else " (no workload file)\n"))
    out.append(_conditions_line(s.get("conditions") or {}))

    out.append("\n## Phases\n")
    out.append(_table(["phase", "status", "duration s", "reason"], [[p["name"], p["status"], p.get("duration_s"), p.get("reason")] for p in s["phases"]]))

    out.append("\n## Memory per phase (mean / peak MB)\n")
    cols = ["electron_main", "electron_renderer", "electron_gpu", "electron_utility", "backend", "database", "inference", "embedding", "transient", "app_overhead", "inference_total", "app_total"]
    rows = []
    any_short = False
    for phase, cats in s["memory"].items():
        cells = []
        for c in cols:
            mark = "*" if cats[c].get("incomplete_samples") else ""
            any_short = any_short or bool(mark)
            cells.append(f"{cats[c]['mean_mb']:.0f} / {cats[c]['peak_mb']:.0f}{mark}")
        rows.append([phase, next(iter(cats.values()))["samples"]] + cells)
    out.append(_table(["phase", "samples", *cols], rows))
    if any_short:
        out.append("\n`*` under-reported: at least one member process's primary metric could not be read in some samples of that phase (counts in `summary.json` -> `memory.<phase>.<category>.incomplete_samples`).\n")

    out.append("\n### System-wide\n")
    out.append(_table(["phase", "min available MB", "max swap used MB", "pressure", "max commit MB", "max GPU used MB"],
                      [[ph, r["min_available_mb"], r["max_swap_used_mb"], ",".join(r["pressure"]), r["max_commit_mb"], r["max_gpu_used_mb"]] for ph, r in s.get("system_by_phase", {}).items()]))

    out.append("\n### Largest process peaks\n")
    out.append(_table(["pid", "name", "category", "role", "peak MB", "source"], [[p["pid"], p["name"], p["category"], p["role"], p["peak_mb"], p["source"]] for p in s["process_peaks"][:15]]))

    out.append("\n## Boot timeline\n")
    out.append(context_report.render_boot(s.get("boot_memory") or {"rows": s["boot_timeline"]}, _table))

    out.append("\n## Flavour and model\n")
    out.append(_flavour_section(s))

    out.append("\n## Workload\n")
    out.append(workload_report.render(s.get("workload") or {}, _table))

    out.append("\n## Machine context\n")
    static = s.get("machine_static") or {}
    if static:
        out.append("Static: " + ", ".join(f"{k}={v}" for k, v in static.items()) + "\n\n")
    out.append(context_report.render_machine_context(s.get("machine_context") or {}, _table))

    out.append("\n## Turns\n")
    out.append(
        "`wall s` is what the user waits for. `generation s` counts only the intervals where tokens streamed:"
        " the wait before the first token and the tool gaps (`tool s`) are excluded, and characters per second"
        " are computed over it. Token counts are not exposed by the API, so throughput is in characters.\n\n"
    )
    out.append(_table(
        ["phase", "label", "cold", "TTFT s", "first event", "wall s", "tool s", "generation s", "answer chars", "thinking chars", "answer chars/s", "total chars/s", "tools", "done"],
        [[t.get("phase"), t.get("label"), t.get("cold"), t.get("ttft_s"), t.get("first_event_type"), t.get("wall_s"), t.get("tool_time_s"), t.get("generation_s"),
          t.get("answer_chars"), t.get("thinking_chars"), t.get("answer_chars_per_s") if t.get("answer_chars_per_s") is not None else (t.get("chars_per_s_note") and "n/a"),
          t.get("total_chars_per_s"), ",".join(t.get("tool_calls") or []), t.get("done")] for t in s["turns"]],
    ))
    notes = {t.get("chars_per_s_note") for t in s["turns"] if t.get("chars_per_s_note")}
    for note in sorted(notes):
        out.append(f"\n- `n/a`: {note}\n")

    st = s["storage"]
    out.append(f"\n## Storage (sizes are {st.get('size_kind')} bytes, MB)\n")
    inst = st.get("install") or {}
    out.append(f"\n### Install footprint: {_mb((inst.get('total') or {}).get('bytes'))} MB\n")
    out.append(_table(["resources entry", "MB"], [[r["name"], _mb(r["bytes"])] for r in inst.get("resources_top", [])]))
    out.append("\n#### Backend bundled libraries (25 largest)\n")
    out.append(_table(["entry", "MB"], [[r["name"], _mb(r["bytes"])] for r in inst.get("backend_lib_top25", [])]))
    data = st.get("data") or {}
    out.append(f"\n### Data root: {_mb((data.get('total') or {}).get('bytes'))} MB\n")
    drows = [["models/" + r["name"], _mb(r["bytes"])] for r in data.get("models", [])]
    if data.get("models_cache"):
        drows.append(["models_cache", _mb(data["models_cache"]["bytes"])])
    for k in ("total", "base", "pg_wal"):
        if (data.get("postgres") or {}).get(k):
            drows.append([f"postgres/{k}", _mb(data["postgres"][k]["bytes"])])
    if data.get("db_backups"):
        drows.append(["db-backups", _mb(data["db_backups"]["bytes"])])
    drows += [["other: " + r["name"], _mb(r["bytes"])] for r in data.get("other", [])]
    out.append(_table(["entry", "MB"], drows))
    out.append("\n### Logs\n")
    out.append(_table(["log", "MB"], [[k, _mb(v.get("bytes"))] for k, v in (st.get("logs") or {}).items()]))
    out.append("\n### Storage deltas per phase\n")
    out.append(_table(["snapshot", "entry", "delta MB"], [[d["snapshot"], r["entry"], round(r["delta"] / MB, 3)] for d in st.get("deltas", []) for r in d["rows"]]))

    out.append("\n## Renderer\n")
    out.append(_table(["phase", "page", "trigger", "JS heap MB", "nodes", "documents", "listeners", "layouts", "style recalcs"],
                      [[r["phase"], r["page"], r["trigger"], r["js_heap_used_mb"], r["nodes"], r["documents"], r["listeners"], r["layouts"], r["recalc_styles"]] for r in s["renderer"]]))

    an = s["anomalies"]
    out.append("\n## Anomalies\n")
    surv = an.get("survivors_after_quit")
    out.append(f"- Survivors after quit: {'quit not run' if surv is None else json.dumps(surv)}\n")
    out.append(f"- Phases not ok: {len(an['phases_not_ok'])}\n")
    for p in an["phases_not_ok"]:
        out.append(f"  - `{p['phase']}` {p['status']}: {p['reason']}\n")
    if an["unreadable_metrics"]:
        out.append("- Unreadable metrics (recorded, not substituted):\n")
        for metric, v in an["unreadable_metrics"].items():
            out.append(f"  - `{metric}`: {v['reads']} reads failed ({v['example_error']}), e.g. {', '.join(v['processes'][:3])}\n")
    else:
        out.append("- Unreadable metrics: none\n")
    out.append(f"- Primary metric missing (reads per category): {json.dumps(an['primary_metric_missing_reads']) if an['primary_metric_missing_reads'] else 'none'}\n")
    out.append(f"- Metric substitutions: {an['metric_substitutions']}\n")
    out.append(f"- Metrics available: {', '.join(an['metrics_available'])}\n")
    if an["sampler_errors"] or an["harness_errors"]:
        out.append(f"- Sampler errors: {an['sampler_errors']}; harness errors in: {an['harness_errors']}\n")
    return "".join(out)


def write_report(run_dir: Path) -> dict[str, Any]:
    summary = build_summary(run_dir)
    write_json(Path(run_dir) / "summary.json", summary)
    (Path(run_dir) / "report.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary
