import json

from erudi_eval import compare, report
from erudi_eval.sampler import category_totals
from erudi_eval.util import MB, write_json

PRIMARY = "phys_footprint"


def proc(pid, category, mb, peak_mb=None, errors=None, name="p"):
    metrics = {PRIMARY: mb * MB} if mb is not None else {}
    if peak_mb is not None:
        metrics["lifetime_max_phys_footprint"] = peak_mb * MB
    rec = {"pid": pid, "name": name, "category": category, "role": "r", "metrics": metrics}
    if errors:
        rec["errors"] = errors
    return rec


def sample(phase, procs, t):
    return {"ts": "x", "t": t, "phase": phase, "primary_metric": PRIMARY, "processes": procs, "totals_mb": category_totals(procs, PRIMARY), "system": {}}


def test_category_totals_derived():
    t = category_totals([proc(1, "electron_main", 100), proc(2, "inference", 2000), proc(3, "backend", 500), proc(4, "database", None)], PRIMARY)
    assert t["app_total"] == 2600 and t["inference_total"] == 2000 and t["app_overhead"] == 600 and t["database"] == 0


def make_run(tmp_path, name, backend_mb, inference_mb):
    d = tmp_path / name
    (d / "storage").mkdir(parents=True)
    with open(d / "samples.jsonl", "w") as fh:
        for i, phase in enumerate(["idle_after_boot", "idle_after_boot", "chat_cold", "chat_cold"]):
            inf = inference_mb if phase == "chat_cold" else None
            procs = [proc(10, "electron_main", 150, peak_mb=160, name="Erudi"), proc(11, "backend", backend_mb + i, peak_mb=backend_mb + 50, name="backend")]
            if inf:
                procs.append(proc(12, "inference", inf + i, name="backend"))
            procs.append(proc(13, "database", None, errors={"phys_footprint": "OSError: [Errno 1]"}, name="postgres"))
            fh.write(json.dumps(sample(phase, procs, 1000 + i)) + "\n")
    events = [
        {"type": "boot_timeline", "phase": "cold_boot", "rows": [{"label": "harness_launch", "t": 0, "since_t0_s": 0.0}, {"label": "ready", "t": 4.2, "since_t0_s": 4.2}]},
        {"type": "turn", "phase": "chat_cold", "label": "cold", "cold": True, "ttft_s": 7.5, "first_event_type": "thinking", "duration_s": 12.0, "answer_chars": 40, "thinking_chars": 300, "answer_chars_per_s": 20.0, "tool_calls": [], "done": True},
        {"type": "quit_result", "phase": "quit", "survivors": {"plus_5s": {"13": "database:postmaster:postgres"}, "plus_30s": {}}},
    ]
    with open(d / "events.jsonl", "w") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")
    write_json(d / "phases.json", [{"name": "preflight", "status": "ok"}, {"name": "ui_tour", "status": "skipped", "reason": "no CDP endpoint"}])
    write_json(d / "config.json", {"run_id": name})
    write_json(d / "storage" / "01-start.json", {"label": "start", "size_kind": "allocated", "install": {"total": {"bytes": 1200 * MB, "files": 1}, "resources_top": [{"name": "backend", "bytes": 1000 * MB}]}, "data": {"models": []}})
    write_json(d / "storage" / "02-after_model_ready.json", {"label": "after_model_ready", "phase": "model_ready", "data": {"models": [{"name": "Qwen3-4B", "bytes": 2300 * MB}]}, "delta": [{"entry": "data/models/Qwen3-4B", "delta": 2300 * MB}]})
    with open(d / "renderer.jsonl", "w") as fh:
        fh.write(json.dumps({"phase": "ui_tour", "page": "models", "trigger": "page", "metrics": {"JSHeapUsedSize": 30 * MB, "Nodes": 1200}}) + "\n")
        fh.write(json.dumps({"phase": "ui_tour", "trigger": "periodic", "metrics": {"JSHeapUsedSize": 31 * MB, "Nodes": 1500}}) + "\n")
    return d


def test_report_from_synthetic_run(tmp_path):
    d = make_run(tmp_path, "runA", backend_mb=500, inference_mb=2500)
    s = report.write_report(d)
    mem = s["memory"]
    assert mem["idle_after_boot"]["backend"] == {"mean_mb": 500.5, "peak_mb": 501.0, "samples": 2}
    assert mem["chat_cold"]["inference"]["peak_mb"] == 2503.0
    assert mem["chat_cold"]["app_overhead"]["mean_mb"] == 150 + 502.5
    assert s["process_peaks"][0]["name"] == "backend" and s["process_peaks"][0]["source"] in ("os_peak", "max_phys_footprint")
    an = s["anomalies"]
    assert an["primary_metric_missing_reads"] == {"database": 4}
    assert an["unreadable_metrics"]["phys_footprint"]["processes"] == ["13 postgres (database)"]
    assert an["phases_not_ok"] == [{"phase": "ui_tour", "status": "skipped", "reason": "no CDP endpoint"}]
    md = (d / "report.md").read_text()
    for needle in ("## Boot timeline", "| ready | 4.2", "## Turns", "thinking", "Qwen3-4B", "no CDP endpoint", "token counts", "plus_5s", "periodic (max nodes)"):
        assert needle in md, needle


def test_compare_two_runs(tmp_path):
    a = report.write_report(make_run(tmp_path, "runA", backend_mb=500, inference_mb=2500))
    b = report.write_report(make_run(tmp_path, "runB", backend_mb=400, inference_mb=2500))
    rows = compare.compare(compare.load_summary(tmp_path / "runA"), compare.load_summary(tmp_path / "runB"))
    backend = next(r for r in rows if r["phase"] == "idle_after_boot" and r["category"] == "backend")
    assert backend["mean_mb"]["delta"] == -100.0 and backend["mean_mb"]["pct"] == -20.0
    assert not any(r["category"] == "embedding" for r in rows)  # empty on both sides
    text = compare.render(a, b, rows)
    assert "| idle_after_boot | backend | 500.5 | 400.5 | -100.0 | -20.0% |" in text


def machine_sample(phase, t, avail_mb, swap_mb, compressed_mb, swapouts, cpu, *, app_total=0.0, top=None, power="AC Power", cost=40.0, gap=1.0):
    s = {
        "t": t, "ts": f"t{t}", "phase": phase, "primary_metric": PRIMARY, "processes": [], "interval_target_s": 1.0, "interval_effective_s": gap,
        "sample_cost_ms": cost, "harness": {"pid": 1, PRIMARY: 30 * MB, "cpu_percent": 2.0},
        "totals_mb": {**category_totals([], PRIMARY), "app_total": app_total, "backend": app_total},
        "system": {
            "available_bytes": avail_mb * MB, "swap_used_bytes": swap_mb * MB, "compressed_bytes": compressed_mb * MB,
            "app_memory_bytes": 4000 * MB, "wired_bytes": 2000 * MB, "cached_files_bytes": 3000 * MB,
            "memory_pressure_level": "warn" if avail_mb < 1000 else "normal", "cpu_percent": cpu, "load_average": [2.0, 1.0, 1.0],
            "counters": {"swapouts": swapouts}, "rates_per_s": {"swapouts": None, "disk_write_bytes": 5 * MB},
            "power_thermal": {"power_source": power, "thermal_warning_level": "none recorded"},
        },
    }
    if top:
        s["top_other"] = {"metric": "rss", "total_rss_mb": 9000.0, "unreadable": 3, "top": top}
    return s


def test_machine_context_baseline_deltas_top_and_boot_memory(tmp_path):
    from erudi_eval import context_report

    samples = [
        machine_sample("baseline", 100, 6000, 1000, 800, 50, 10, top=[{"pid": 9, "name": "Safari", "rss_mb": 900.0}]),
        machine_sample("baseline", 101, 6100, 1000, 800, 60, 20),
        machine_sample("cold_boot", 102, 5000, 1100, 900, 70, 80, app_total=300, gap=0.3, cost=90),
        machine_sample("cold_boot", 102.3, 4000, 1300, 1000, 5, 90, app_total=700, gap=0.3, cost=120, power="Battery Power"),
        machine_sample("chat_cold", 103, 800, 1500, 1200, 20, 50, app_total=3000, top=[{"pid": 7, "name": "Chrome", "rss_mb": 1500.0}]),
        machine_sample("chat_cold", 104, 900, 1400, 1100, 40, 40, app_total=3100),
    ]
    samples[3]["system"]["rates_per_s"]["swapouts"] = None  # counter reset between samples: no rate
    samples[5]["system"]["rates_per_s"]["swapouts"] = 20.0
    ctx = context_report.machine_context(samples)
    base, boot, chat = ctx["per_phase"]["baseline"], ctx["per_phase"]["cold_boot"], ctx["per_phase"]["chat_cold"]
    assert base["available_min_mb"] == 6000 and base["swap_used_start_mb"] == 1000 and base["cpu_max_pct"] == 20
    assert base["swapouts_total"] == 10 and "swapouts_total" in chat and chat["swapouts_total"] == 20
    assert boot["swapouts_total"] is None  # the counter went backwards inside the phase
    assert chat["swapouts_peak_per_s"] == 20.0 and chat["pressure_worst"] == "warn"
    assert boot["power_thermal_states"] == ["power_source=AC Power, thermal_warning_level=none recorded", "power_source=Battery Power, thermal_warning_level=none recorded"]
    assert boot["interval_effective_median_s"] == 0.3 and boot["sample_cost_max_ms"] == 120
    delta = {d["phase"]: d for d in ctx["baseline_deltas"]}
    assert delta["chat_cold"]["available_mean_mb"] == 850 - 6050 and delta["chat_cold"]["compressed_mean_mb"] == 1150 - 800
    assert ctx["top_other"]["baseline"]["top"][0]["name"] == "Safari"
    assert ctx["top_other"]["lowest_available"]["phase"] == "chat_cold" and ctx["top_other"]["lowest_available"]["top"][0]["name"] == "Chrome"

    timeline = [{"label": "harness_launch", "t": 102.0, "since_t0_s": 0.0}, {"label": "phase:running_migrations", "t": 102.1, "since_t0_s": 0.1},
                {"label": "ready", "t": 102.35, "since_t0_s": 0.35}]
    bm = context_report.boot_memory(timeline, samples)
    assert [r["totals_mb"]["app_total"] for r in bm["rows"]] == [300, 300, 700]
    assert bm["rows"][2]["sample_offset_s"] == -0.05
    assert bm["peak"]["totals_mb"]["app_total"] == 700 and bm["peak"]["since_t0_s"] == 0.3
    assert bm["sampling"]["samples"] == 2 and bm["sampling"]["interval_effective_median_s"] == 0.3
    # A lowest-memory phase without its own top list borrows the nearest one in time from another phase.
    no_top = [dict(s, top_other=None) if s["phase"] == "chat_cold" else s for s in samples]
    near = context_report.machine_context(no_top)["top_other"]["lowest_available"]
    assert near["phase"] == "chat_cold" and near["top"][0]["name"] == "Safari" and near["offset_s"] == -3.0

    table = report._table
    md = context_report.render_machine_context(ctx, table) + context_report.render_boot(bm, table)
    for needle in ("### Memory", "### Swap and paging", "### CPU, disk, power and harness cost", "### Baseline vs phase", "at baseline", "Chrome", "Peak during boot: app_total 700.0 MB", "Boot sampling: target 1.0 s"):
        assert needle in md, needle
