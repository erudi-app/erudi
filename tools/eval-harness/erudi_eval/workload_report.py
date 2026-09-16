"""Report and compare sections for the declared background workload."""

from __future__ import annotations

from typing import Any

from .workload import workload_by_phase


def build(run_dir, samples: list[dict[str, Any]], events: list[dict[str, Any]], system: dict[str, Any]) -> dict[str, Any]:
    declared = system.get("workload") or {}
    drift_events = [e for e in events if e.get("type") == "workload_drift"]
    return {
        "profile": system.get("profile"),
        "workload_file": declared.get("workload_file"),
        "notes": declared.get("notes") or declared.get("note"),
        "checks": declared.get("checks") or [e for e in events if e.get("type") == "workload_check"],
        "groups_at_start": declared.get("groups_at_start") or {},
        "per_phase": workload_by_phase(samples),
        "declared_facts": {g["name"]: g.get("declared") for g in _groups(run_dir) if g.get("declared")},
        "drift": drift_events[-1] if drift_events else {"reference_phase": None, "warnings": []},
    }


def _groups(run_dir) -> list[dict[str, Any]]:
    import json
    from pathlib import Path

    path = Path(run_dir) / "workload.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("groups", [])
    except (OSError, ValueError):
        return []


def render(wl: dict[str, Any], table) -> str:
    if not wl.get("workload_file"):
        return f"No workload file for profile `{wl.get('profile')}`: the background was not declared, verified or measured.\n"
    out = [f"Profile `{wl['profile']}`, workload file `{wl['workload_file']}`.\n"]
    if wl.get("notes"):
        out.append(f"\n> {wl['notes']}\n")
    out.append("\n### Declared (not verifiable from outside)\n")
    facts = wl.get("declared_facts") or {}
    out.append(table(["group", "declared"], [[g, ", ".join(f"{k}: {v}" for k, v in (d or {}).items())] for g, d in facts.items()]))
    out.append("\n### Verification at preflight\n")
    out.append(table(["group", "status", "what was found", "expected"],
                     [[c["group"], c["status"], c["message"], c.get("expected_count") if not c.get("expect_absent") else "absent"] for c in wl.get("checks", [])]))
    out.append("\n### Per group and phase (primary metric sum; RSS is reported separately, never mixed)\n")
    rows = []
    for phase, groups in (wl.get("per_phase") or {}).items():
        for group, st in groups.items():
            if not st["count_max"]:
                continue
            rows.append([phase, group, st["count_min"] if st["count_min"] == st["count_max"] else f"{st['count_min']}-{st['count_max']}",
                         st["primary_mb_mean"], st["primary_mb_peak"], st["rss_mb_mean"], st["cpu_percent_mean"], st["cpu_percent_peak"],
                         st["unreadable_primary_max"] or None])
    out.append(table(["phase", "group", "processes", "primary mean MB", "primary peak MB", "RSS mean MB", "CPU mean %", "CPU peak %", "unreadable"], rows))
    out.append("\nDescendants of a coding-agent CLI session (shells, MCP servers it spawned) are attributed to the group's `..._children`; Erudi's own processes and the harness are never counted in any group.\n")
    drift = wl.get("drift") or {}
    out.append(f"\n### Workload drift (reference phase: `{drift.get('reference_phase')}`)\n")
    warnings = drift.get("warnings") or []
    if not warnings:
        out.append("None: every group kept its size, its memory and an idle CPU through the run.\n")
    for w in warnings:
        out.append(f"- {w['message']}\n")
    return "".join(out)


def conditions_rows(a: dict[str, Any], b: dict[str, Any]) -> list[list[Any]]:
    """Side-by-side conditions of two runs, for `compare`."""
    ca, cb = a.get("conditions") or {}, b.get("conditions") or {}
    mb = 1024 * 1024
    rows = [["profile", a.get("profile"), b.get("profile")],
            ["flavour", (a.get("flavour") or {}).get("requested"), (b.get("flavour") or {}).get("requested")],
            ["model", (a.get("flavour") or {}).get("model_link"), (b.get("flavour") or {}).get("model_link")],
            ["run id", a.get("run_id"), b.get("run_id")],
            ["app version", (a.get("system", {}).get("app") or {}).get("version"), (b.get("system", {}).get("app") or {}).get("version")]]
    for label, key, conv in (("uptime", "uptime_human", None), ("swap used MB", "swap_used_bytes", mb), ("swap total MB", "swap_total_bytes", mb),
                             ("available at start MB", "available_bytes", mb), ("memory pressure", "memory_pressure_level", None),
                             ("power source", "power_source", None), ("battery %", "battery_percent", None),
                             ("low power mode", "low_power_mode", None), ("OS build", "os_build", None), ("displays", "display_count", None)):
        va, vb = ca.get(key), cb.get(key)
        if conv:
            va = round(va / conv) if isinstance(va, (int, float)) else va
            vb = round(vb / conv) if isinstance(vb, (int, float)) else vb
        rows.append([label, va, vb])
    groups = sorted(set((a.get("workload") or {}).get("groups_at_start") or {}) | set((b.get("workload") or {}).get("groups_at_start") or {}))
    for g in groups:
        ga = ((a.get("workload") or {}).get("groups_at_start") or {}).get(g) or {}
        gb = ((b.get("workload") or {}).get("groups_at_start") or {}).get(g) or {}
        rows.append([f"workload {g} (procs, MB)", f"{ga.get('count', 0)}, {ga.get('primary_mb')}", f"{gb.get('count', 0)}, {gb.get('primary_mb')}"])
    return rows
