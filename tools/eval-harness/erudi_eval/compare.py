"""Per phase x category memory deltas between two runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import workload_report
from .report import DERIVED, build_summary, _table
from .discovery import CATEGORIES


def load_summary(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / "summary.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return build_summary(Path(run_dir))


def compare(a: dict[str, Any], b: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for phase in [p for p in a["memory"] if p in b["memory"]]:
        for key in (*CATEGORIES, *DERIVED):
            ca, cb = a["memory"][phase].get(key), b["memory"][phase].get(key)
            if not ca or not cb:
                continue
            if ca["peak_mb"] == 0 and cb["peak_mb"] == 0:
                continue
            row = {"phase": phase, "category": key}
            for stat in ("mean_mb", "peak_mb"):
                d = cb[stat] - ca[stat]
                row[stat] = {"a": ca[stat], "b": cb[stat], "delta": round(d, 1), "pct": round(100 * d / ca[stat], 1) if ca[stat] else None}
            rows.append(row)
    return rows


def machine_deltas(a: dict[str, Any], b: dict[str, Any]) -> list[dict[str, Any]]:
    """Per phase: how the machine itself differed between the two runs."""
    pa = (a.get("machine_context") or {}).get("per_phase") or {}
    pb = (b.get("machine_context") or {}).get("per_phase") or {}
    rows = []
    for phase in [p for p in pa if p in pb]:
        row = {"phase": phase}
        for key in ("available_min_mb", "swap_used_max_mb", "compressed_mean_mb"):
            va, vb = pa[phase].get(key), pb[phase].get(key)
            row[key] = {"a": va, "b": vb, "delta": round(vb - va, 1) if va is not None and vb is not None else None}
        rows.append(row)
    return rows


def render(a: dict[str, Any], b: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    out = [f"# compare\n\nA: {a['run_id']} (`{a.get('primary_metric')}`)\nB: {b['run_id']} (`{b.get('primary_metric')}`)\n"]
    out.append("\n## Conditions\n")
    out.append(_table(["", "A", "B"], workload_report.conditions_rows(a, b)))
    if a.get("profile") != b.get("profile"):
        out.append(f"\nThe runs use different profiles (`{a.get('profile')}` vs `{b.get('profile')}`): the deltas below mix the app's own changes with the background workload.\n")
    md = machine_deltas(a, b)
    if md:
        out.append("\n## Machine context deltas (B - A)\n")
        out.append(_table(["phase", "min available A/B MB", "Δ", "max swap used A/B MB", "Δ", "compressed mean A/B MB", "Δ"],
                          [[r["phase"], f"{r['available_min_mb']['a']} / {r['available_min_mb']['b']}", r["available_min_mb"]["delta"],
                            f"{r['swap_used_max_mb']['a']} / {r['swap_used_max_mb']['b']}", r["swap_used_max_mb"]["delta"],
                            f"{r['compressed_mean_mb']['a']} / {r['compressed_mean_mb']['b']}", r["compressed_mean_mb"]["delta"]] for r in md]))
    out.append("\n## Memory per phase and category\n")
    if a.get("primary_metric") != b.get("primary_metric"):
        out.append("\n**Warning: the runs use different primary metrics (different OS); deltas are not like for like.**\n")
    only = sorted(set(a["memory"]) ^ set(b["memory"]))
    if only:
        out.append(f"\nPhases present in only one run: {', '.join(only)}\n")
    out.append("\n| phase | category | mean A | mean B | Δ mean | Δ% | peak A | peak B | Δ peak | Δ% |\n|---|---|---|---|---|---|---|---|---|---|\n")
    pct = lambda v: "" if v is None else f"{v:+.1f}%"  # noqa: E731
    for r in rows:
        m, p = r["mean_mb"], r["peak_mb"]
        out.append(f"| {r['phase']} | {r['category']} | {m['a']:.1f} | {m['b']:.1f} | {m['delta']:+.1f} | {pct(m['pct'])} | {p['a']:.1f} | {p['b']:.1f} | {p['delta']:+.1f} | {pct(p['pct'])} |\n")
    return "".join(out)
