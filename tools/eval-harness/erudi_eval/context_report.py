"""Report sections that read the app in the context of the machine: machine context per phase,
baseline deltas, top other consumers, and app memory at each boot timeline mark."""

from __future__ import annotations

from statistics import median
from typing import Any

from .util import MB

PRESSURE_ORDER = {"normal": 0, "warn": 1, "critical": 2}
# Phases sampled before the app is launched: never "the phase with the lowest available memory".
PRE_APP_PHASES = ("preflight", "baseline")
BOOT_CATEGORIES = ("electron_main", "electron_renderer", "electron_gpu", "electron_utility", "backend", "database", "inference", "transient")
# macOS counters are pages; Linux /proc/vmstat pswpin/pswpout are pages too, pgmajfault is a count.
RATE_KEYS = ("swapins", "swapouts", "pageouts", "compressions", "decompressions", "pageins", "faults", "pswpin", "pswpout", "pgmajfault")
TOTAL_KEYS = ("swapins", "swapouts", "pageouts", "pswpin", "pswpout")


def _mb(v: Any) -> float | None:
    return round(v / MB, 1) if isinstance(v, (int, float)) else None


def _stat(values: list[float], fn) -> float | None:
    values = [v for v in values if v is not None]
    return round(fn(values), 1) if values else None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _phases_in_order(samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for s in samples:
        if "system" in s and "phase" in s:
            out.setdefault(s["phase"], []).append(s)
    return out


def _power_state(block: dict[str, Any]) -> str | None:
    pt = block.get("power_thermal") or {}
    parts = [f"{k}={pt[k]}" for k in ("power_source", "low_power_mode", "thermal_warning_level", "performance_warning_level", "cpu_speed_limit") if k in pt]
    return ", ".join(parts) or None


def phase_context(samples: list[dict[str, Any]]) -> dict[str, Any]:
    systems = [s["system"] for s in samples]
    col = lambda key: [b.get(key) for b in systems]  # noqa: E731
    row: dict[str, Any] = {"samples": len(samples)}
    avail = [_mb(v) for v in col("available_bytes")]
    row["available_min_mb"], row["available_mean_mb"] = _stat(avail, min), _stat(avail, _mean)
    for key in ("app_memory", "wired", "compressed", "cached_files"):
        vals = [_mb(v) for v in col(f"{key}_bytes")]
        row[f"{key}_mean_mb"], row[f"{key}_max_mb"] = _stat(vals, _mean), _stat(vals, max)
    swap = [_mb(v) for v in col("swap_used_bytes") if v is not None]
    row["swap_used_start_mb"] = swap[0] if swap else None
    row["swap_used_max_mb"] = max(swap) if swap else None
    row["swap_used_end_mb"] = swap[-1] if swap else None
    row["swap_used_mean_mb"] = _stat(swap, _mean)
    for key in TOTAL_KEYS:
        seq = [b.get("counters", {}).get(key) for b in systems if b.get("counters", {}).get(key) is not None]
        if len(seq) >= 2:
            row[f"{key}_total"] = seq[-1] - seq[0] if seq[-1] >= seq[0] else None  # reset inside the phase: unknown
    for key in RATE_KEYS:
        rates = [b.get("rates_per_s", {}).get(key) for b in systems]
        peak = _stat(rates, max)
        if peak is not None:
            row[f"{key}_peak_per_s"] = peak
    levels = [b.get("memory_pressure_level") for b in systems if b.get("memory_pressure_level")]
    row["pressure_worst"] = max(levels, key=lambda lv: PRESSURE_ORDER.get(lv, -1)) if levels else None
    cpu = col("cpu_percent")
    row["cpu_mean_pct"], row["cpu_max_pct"] = _stat(cpu, _mean), _stat(cpu, max)
    load = [b["load_average"][0] for b in systems if b.get("load_average")]
    row["load1_max"] = _stat(load, max)
    for key, name in (("disk_read_bytes", "disk_read_peak_mb_s"), ("disk_write_bytes", "disk_write_peak_mb_s")):
        rates = [b.get("rates_per_s", {}).get(key) for b in systems]
        row[name] = _mb(max([r for r in rates if r is not None], default=None))
    states: list[str] = []
    for b in systems:
        st = _power_state(b)
        if st and (not states or states[-1] != st):
            states.append(st)
    row["power_thermal_states"] = states
    harness = [s.get("harness") or {} for s in samples]
    primary = next((s.get("primary_metric") for s in samples if s.get("primary_metric")), None)
    row["harness_mean_mb"] = _stat([_mb(h.get(primary)) for h in harness], _mean)
    row["harness_cpu_mean_pct"] = _stat([h.get("cpu_percent") for h in harness], _mean)
    costs = [s.get("sample_cost_ms") for s in samples if s.get("sample_cost_ms") is not None]
    row["sample_cost_mean_ms"], row["sample_cost_max_ms"] = _stat(costs, _mean), _stat(costs, max)
    gaps = [s.get("interval_effective_s") for s in samples if s.get("interval_effective_s") is not None]
    row["interval_target_s"] = samples[-1].get("interval_target_s")
    row["interval_effective_median_s"] = round(median(gaps), 3) if gaps else None
    row["interval_effective_max_s"] = round(max(gaps), 3) if gaps else None
    return row


def machine_context(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_phase = _phases_in_order(samples)
    per_phase = {phase: phase_context(rows) for phase, rows in by_phase.items()}
    base = per_phase.get("baseline")
    deltas = []
    if base:
        for phase, row in per_phase.items():
            if phase == "baseline":
                continue
            d = {"phase": phase}
            for key in ("available_mean_mb", "swap_used_mean_mb", "compressed_mean_mb"):
                d[key] = round(row[key] - base[key], 1) if row.get(key) is not None and base.get(key) is not None else None
            deltas.append(d)
    top = {}
    if "baseline" in by_phase:
        base_rows = by_phase["baseline"]
        top["baseline"] = _top_near(samples, base_rows[-1]["t"], within=base_rows)
    app_phases = [(row["available_min_mb"], phase) for phase, row in per_phase.items()
                  if phase not in PRE_APP_PHASES and row.get("available_min_mb") is not None]
    if app_phases:
        _, phase = min(app_phases)
        lowest = min(by_phase[phase], key=lambda r: r["system"].get("available_bytes") or float("inf"))
        near = _top_near(samples, lowest["t"])  # the list is taken every 5 s: nearest one in time, any phase
        if near:
            top["lowest_available"] = {"phase": phase, **near}
    return {"per_phase": per_phase, "baseline_deltas": deltas, "top_other": top}


def _top_near(samples: list[dict[str, Any]], t: float | None, within: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    pool = [r for r in (within if within is not None else samples) if r.get("top_other") and r.get("t") is not None]
    if not pool or t is None:
        return None
    best = min(pool, key=lambda r: abs(r["t"] - t))
    return {"ts": best.get("ts"), "offset_s": round(best["t"] - t, 1), **best["top_other"]}


def boot_memory(timeline: list[dict[str, Any]], samples: list[dict[str, Any]]) -> dict[str, Any]:
    """App memory at each boot mark (nearest sample), the peak during cold_boot, and the sampling quality."""
    with_totals = [s for s in samples if s.get("totals_mb") and s.get("t") is not None]
    rows = []
    for mark in timeline:
        if not with_totals or mark.get("t") is None:
            rows.append({**mark, "sample_offset_s": None, "totals_mb": None})
            continue
        nearest = min(with_totals, key=lambda s: abs(s["t"] - mark["t"]))
        rows.append({**mark, "sample_offset_s": round(nearest["t"] - mark["t"], 3), "totals_mb": nearest["totals_mb"]})
    boot = [s for s in with_totals if s.get("phase") == "cold_boot"]
    peak = None
    if boot:
        top = max(boot, key=lambda s: s["totals_mb"].get("app_total", 0))
        t0 = timeline[0]["t"] if timeline else boot[0]["t"]
        peak = {"since_t0_s": round(top["t"] - t0, 3), "totals_mb": top["totals_mb"]}
    # Sampling quality over the samples taken at the boot rate after launch (the phase's tail is back at the normal rate).
    launch_t = timeline[0]["t"] if timeline else None
    after_launch = [s for s in boot if launch_t is None or s["t"] >= launch_t - 0.001]
    targets = [s.get("interval_target_s") for s in after_launch if s.get("interval_target_s") is not None]
    boot_target = min(targets) if targets else None
    window = [s for s in after_launch if s.get("interval_target_s") == boot_target] or after_launch
    gaps = sorted(s["interval_effective_s"] for s in window if s.get("interval_effective_s") is not None)
    costs = sorted(s["sample_cost_ms"] for s in window if s.get("sample_cost_ms") is not None)
    quality = {
        "samples": len(window),
        "interval_target_s": boot_target,
        "interval_effective_median_s": round(median(gaps), 3) if gaps else None,
        "interval_effective_p95_s": round(gaps[int(0.95 * (len(gaps) - 1))], 3) if gaps else None,
        "interval_effective_max_s": round(gaps[-1], 3) if gaps else None,
        "sample_cost_median_ms": round(median(costs), 1) if costs else None,
        "sample_cost_max_ms": round(costs[-1], 1) if costs else None,
    }
    return {"rows": rows, "peak": peak, "sampling": quality}


# --- markdown -------------------------------------------------------------------------------


def render_boot(boot: dict[str, Any], table) -> str:
    out = []
    q = boot.get("sampling") or {}
    if q.get("samples"):
        out.append(
            f"Boot sampling: target {q['interval_target_s']} s, effective median {q['interval_effective_median_s']} s "
            f"(p95 {q['interval_effective_p95_s']} s, max {q['interval_effective_max_s']} s) over {q['samples']} samples; "
            f"one sample costs {q['sample_cost_median_ms']} ms median, {q['sample_cost_max_ms']} ms max.\n\n"
        )
    cols = ["app_total", "app_overhead", *BOOT_CATEGORIES]
    rows = []
    for r in boot.get("rows", []):
        t = r.get("totals_mb") or {}
        rows.append([r["label"], r["since_t0_s"], r.get("sample_offset_s")] + [round(t[c], 1) if c in t else None for c in cols])
    out.append(table(["event", "s since launch", "sample offset s", *[f"{c} MB" for c in cols]], rows))
    peak = boot.get("peak")
    if peak:
        t = peak["totals_mb"]
        parts = ", ".join(f"{c} {t[c]:.1f}" for c in BOOT_CATEGORIES if t.get(c))
        out.append(f"\nPeak during boot: app_total {t['app_total']:.1f} MB at +{peak['since_t0_s']} s ({parts}).\n")
    return "".join(out)


def render_machine_context(ctx: dict[str, Any], table) -> str:
    per = ctx.get("per_phase") or {}
    if not per:
        return "_no machine samples_\n"
    out = ["Machine-wide figures (not the app's). Pages are the OS page size; rates are per second between samples.\n"]
    out.append("\n### Memory\n")
    out.append(table(
        ["phase", "samples", "min available MB", "app memory mean/max MB", "wired mean/max MB", "compressed mean/max MB", "cached files mean/max MB", "pressure worst"],
        [[p, r["samples"], r["available_min_mb"], _pair(r, "app_memory"), _pair(r, "wired"), _pair(r, "compressed"), _pair(r, "cached_files"), r["pressure_worst"]] for p, r in per.items()],
    ))
    out.append("\n### Swap and paging\n")
    out.append(table(
        ["phase", "swap used start/max/end MB", "swap-ins total", "swap-outs total", "swap-in peak /s", "swap-out peak /s", "pageouts peak /s", "compressions peak /s", "decompressions peak /s"],
        [[p, f"{_s(r['swap_used_start_mb'])} / {_s(r['swap_used_max_mb'])} / {_s(r['swap_used_end_mb'])}",
          r.get("swapins_total", r.get("pswpin_total")), r.get("swapouts_total", r.get("pswpout_total")),
          r.get("swapins_peak_per_s", r.get("pswpin_peak_per_s")), r.get("swapouts_peak_per_s", r.get("pswpout_peak_per_s")),
          r.get("pageouts_peak_per_s"), r.get("compressions_peak_per_s"), r.get("decompressions_peak_per_s")] for p, r in per.items()],
    ))
    out.append("\n### CPU, disk, power and harness cost\n")
    out.append(table(
        ["phase", "CPU mean/max %", "load1 max", "disk read peak MB/s", "disk write peak MB/s", "power / thermal", "harness MB", "harness CPU %", "sample cost mean/max ms", "interval target / effective median s"],
        [[p, f"{_s(r['cpu_mean_pct'])} / {_s(r['cpu_max_pct'])}", r["load1_max"], r["disk_read_peak_mb_s"], r["disk_write_peak_mb_s"],
          " → ".join(r["power_thermal_states"]) or None, r["harness_mean_mb"], r["harness_cpu_mean_pct"],
          f"{_s(r['sample_cost_mean_ms'])} / {_s(r['sample_cost_max_ms'])}", f"{_s(r['interval_target_s'])} / {_s(r['interval_effective_median_s'])}"] for p, r in per.items()],
    ))
    if ctx.get("baseline_deltas"):
        out.append("\n### Baseline vs phase (mean, MB)\n")
        out.append(table(["phase", "Δ available", "Δ swap used", "Δ compressed"],
                         [[d["phase"], d["available_mean_mb"], d["swap_used_mean_mb"], d["compressed_mean_mb"]] for d in ctx["baseline_deltas"]]))
    for label, block in (ctx.get("top_other") or {}).items():
        if not block or not block.get("top"):
            continue
        title = "at baseline" if label == "baseline" else f"when available memory was lowest (phase `{block.get('phase')}`)"
        out.append(f"\n### Top other processes {title} (RSS, excludes Erudi and the harness)\n")
        offset = f", {block['offset_s']:+} s from the lowest-memory sample" if label != "baseline" and block.get("offset_s") is not None else ""
        out.append(f"All other processes: {block['total_rss_mb']} MB RSS ({block['unreadable']} unreadable), sampled at {block.get('ts')}{offset}.\n\n")
        out.append(table(["pid", "name", "RSS MB"], [[r["pid"], r["name"], r["rss_mb"]] for r in block["top"]]))
    return "".join(out)


def _pair(row: dict[str, Any], key: str) -> str | None:
    mean, mx = row.get(f"{key}_mean_mb"), row.get(f"{key}_max_mb")
    return None if mean is None and mx is None else f"{_s(mean)} / {_s(mx)}"


def _s(v: Any) -> str:
    return "" if v is None else (f"{v:.0f}" if isinstance(v, float) and abs(v) >= 10 else str(v))
