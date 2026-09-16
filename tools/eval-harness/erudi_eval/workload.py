"""Declared background workload: which other apps run during a measurement, verified and measured.

A workload file names groups of processes (a browser, a chat app, a terminal coding-agent CLI and
the servers it spawns...) with match rules and declared facts. The harness verifies what it can (presence, session counts), reports
the rest as declared, measures each group every sample, and flags drift (a group whose process
count changes, whose memory moves away from its reference, or that becomes busy).

Everything except `measure_groups` is pure and unit-tested.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .discovery import ProcInfo, children_map, descendants, identity_paths, norm_path, path_stem

GROUP_KEYS = {
    "name", "exe_prefixes", "exe_contains", "exe_basenames", "argv0_basenames", "argv0_prefixes", "cmdline_contains",
    "orphaned", "sessions", "include_descendants", "descendants_of", "expected_count", "expect_absent",
    "optional", "declared", "notes",
}
OS_SUFFIX = {"darwin": "mac", "windows": "windows", "linux": "linux"}


PLACEHOLDER = re.compile(r"<[^>]+>")  # "<path fragment of your agent CLI install>" in a shipped example


def is_placeholder(value: str) -> bool:
    """A shipped example carries placeholders instead of machine-specific paths."""
    return bool(PLACEHOLDER.search(value))


@dataclass(frozen=True)
class Group:
    name: str
    exe_prefixes: tuple[str, ...] = ()
    exe_contains: tuple[str, ...] = ()
    exe_basenames: tuple[str, ...] = ()
    argv0_basenames: tuple[str, ...] = ()
    argv0_prefixes: tuple[str, ...] = ()  # argv[0] as a path, for a binary that is not where its name suggests
    cmdline_contains: tuple[str, ...] = ()
    orphaned: bool = False  # only processes re-parented to init/launchd (ppid 1) or whose parent is gone
    sessions: bool = False  # sessions = matches whose parent does not match; expected_count counts sessions
    include_descendants: bool = False
    descendants_of: str | None = None
    expected_count: int | None = None
    expect_absent: bool = False
    optional: bool = False
    declared: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None
    unconfigured: bool = False  # every rule was still a placeholder: the group matches nothing
    raw_rules: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def has_rules(self) -> bool:
        return any((self.exe_prefixes, self.exe_contains, self.exe_basenames, self.argv0_basenames, self.argv0_prefixes, self.cmdline_contains))


@dataclass(frozen=True)
class Workload:
    profile: str
    groups: tuple[Group, ...]
    notes: str | None = None
    path: str | None = None


@dataclass
class Members:
    pids: set[int] = field(default_factory=set)
    session_pids: set[int] = field(default_factory=set)


def _expand(value: str) -> str:
    """%VAR% (Windows style, on any OS), $VAR / ${VAR} and ~. An unset variable is left as written."""
    value = re.sub(r"%([A-Za-z_][A-Za-z0-9_]*)%", lambda m: os.environ.get(m.group(1), m.group(0)), value)
    return os.path.expanduser(os.path.expandvars(value))


def parse_workload(data: dict[str, Any], path: str | None = None) -> Workload:
    groups = []
    names = set()
    for raw in data.get("groups", []):
        unknown = set(raw) - GROUP_KEYS
        if unknown:
            raise ValueError(f"workload group {raw.get('name')!r}: unknown keys {sorted(unknown)}")
        if not raw.get("name"):
            raise ValueError("workload group without a name")
        kwargs = dict(raw)
        raw_rules, dropped = {}, 0
        for key in ("exe_prefixes", "exe_contains", "exe_basenames", "argv0_basenames", "argv0_prefixes", "cmdline_contains"):
            values = tuple(raw.get(key, []))
            raw_rules[key] = values
            # A placeholder is dropped rather than matched: an unfilled example must never capture everything.
            kept = tuple(_expand(v) for v in values if not is_placeholder(v))
            dropped += len(values) - len(kept)
            kwargs[key] = kept
        kwargs["raw_rules"] = raw_rules
        group = Group(**kwargs)
        if not group.has_rules and not group.descendants_of:
            if not dropped:
                raise ValueError(f"workload group {group.name!r} has no match rule and no descendants_of")
            group = replace(group, unconfigured=True)
        groups.append(group)
        names.add(group.name)
    for g in groups:
        if g.descendants_of and g.descendants_of not in names:
            raise ValueError(f"workload group {g.name!r}: descendants_of {g.descendants_of!r} is not a group")
    return Workload(profile=data.get("profile", "unnamed"), groups=tuple(groups), notes=data.get("notes"), path=path)


def load_workload(path: Path) -> Workload:
    return parse_workload(json.loads(Path(path).read_text(encoding="utf-8")), path=str(path))


def default_workload_path(harness_dir: Path, profile: str, os_name: str) -> Path | None:
    """This machine's local file when it exists, else the shipped example (whose placeholders read as
    "not configured" until they are filled in)."""
    suffix = OS_SUFFIX.get(os_name, os_name)
    for name in (f"local-{profile}-{suffix}.json", f"{profile}-{suffix}.json"):
        p = Path(harness_dir) / "workloads" / name
        if p.exists():
            return p
    return None


def _matches(p: ProcInfo, g: Group, by_pid: dict[int, ProcInfo]) -> bool:
    if not g.has_rules:
        return False
    paths = [norm_path(x) for x in identity_paths(p)]
    stems = {path_stem(x) for x in identity_paths(p)}
    argv0_path = norm_path(p.cmdline[0]) if p.cmdline else ""
    argv0 = path_stem(p.cmdline[0]) if p.cmdline else ""
    cmd = norm_path(" ".join(p.cmdline))
    hit = (
        any(x.startswith(norm_path(pre)) for x in paths for pre in g.exe_prefixes)
        or any(norm_path(sub) in x for x in paths for sub in g.exe_contains)
        or bool(stems & {path_stem(b) for b in g.exe_basenames})
        or argv0 in {path_stem(b) for b in g.argv0_basenames}
        or (bool(argv0_path) and any(argv0_path.startswith(norm_path(pre)) for pre in g.argv0_prefixes))
        or any(norm_path(sub) in cmd for sub in g.cmdline_contains)
    )
    if hit and g.orphaned:
        hit = p.ppid <= 1 or p.ppid not in by_pid
    return hit


def assign_groups(procs: list[ProcInfo], wl: Workload, erudi_pids: set[int], harness_pid: int) -> dict[str, Members]:
    """Each process goes to at most one group, first matching group in file order wins.
    Erudi processes, the harness and everything the harness started are never captured."""
    by_pid = {p.pid: p for p in procs}
    kids = children_map(procs)
    excluded = set(erudi_pids) | {harness_pid} | descendants(harness_pid, kids)
    assigned: dict[int, str] = {}
    out = {g.name: Members() for g in wl.groups}
    for g in wl.groups:
        for p in procs:
            if p.pid in excluded or p.pid in assigned or not _matches(p, g, by_pid):
                continue
            assigned[p.pid] = g.name
            out[g.name].pids.add(p.pid)
        if g.sessions:
            out[g.name].session_pids = {pid for pid in out[g.name].pids if not (by_pid[pid].ppid in by_pid and _matches(by_pid[by_pid[pid].ppid], g, by_pid))}
    for g in wl.groups:
        roots = None
        if g.descendants_of:
            src = out[g.descendants_of]
            roots = src.session_pids or src.pids
        elif g.include_descendants:
            roots = set(out[g.name].pids)
        for root in roots or ():
            for pid in descendants(root, kids):
                if pid in excluded or pid in assigned:
                    continue
                assigned[pid] = g.name
                out[g.name].pids.add(pid)
    return out


def verify_presence(assignment: dict[str, Members], wl: Workload) -> list[dict[str, Any]]:
    checks = []
    for g in wl.groups:
        m = assignment.get(g.name, Members())
        count = len(m.session_pids) if g.sessions else len(m.pids)
        unit = "sessions" if g.sessions else "processes"
        status, message = "ok", f"{count} {unit} found"
        if g.unconfigured:
            status = "not configured"
            message = "rules are still placeholders: copy this example to workloads/local-<name>.json, fill in the paths for this machine and pass it with --workload"
        elif g.expect_absent:
            if m.pids:
                status, message = "warning", f"expected absent, found {len(m.pids)} processes"
        elif g.expected_count is not None:
            if count != g.expected_count:
                status, message = "warning", f"expected {g.expected_count} {unit}, found {count}"
        elif not m.pids and not g.optional and not g.descendants_of:
            status, message = "warning", "declared in the workload but not running"
        checks.append({"group": g.name, "status": status, "message": message, "count": count, "processes": len(m.pids),
                       "expected_count": g.expected_count, "expect_absent": g.expect_absent, "declared": g.declared})
    return checks


def measure_groups(assignment: dict[str, Members], read) -> dict[str, dict[str, Any]]:
    """Per group: count, sum of the primary metric where readable, RSS sum under its own name, CPU %.
    `read(pid)` returns {'primary': bytes|None, 'rss': bytes|None, 'cpu': pct|None}."""
    mb = 1024 * 1024
    out = {}
    for name, m in assignment.items():
        primary = rss = cpu = 0.0
        unreadable_primary = unreadable_rss = 0
        for pid in m.pids:
            r = read(pid)
            if r.get("primary") is None:
                unreadable_primary += 1
            else:
                primary += r["primary"]
            if r.get("rss") is None:
                unreadable_rss += 1
            else:
                rss += r["rss"]
            cpu += r.get("cpu") or 0.0
        n = len(m.pids)
        out[name] = {
            "count": n,
            "sessions": len(m.session_pids) if m.session_pids else None,
            "primary_mb": round(primary / mb, 1) if n and unreadable_primary < n else None,
            "unreadable_primary": unreadable_primary,
            "rss_mb": round(rss / mb, 1) if n and unreadable_rss < n else None,
            "unreadable_rss": unreadable_rss,
            "cpu_percent": round(cpu, 1),
        }
    return out


def workload_by_phase(samples: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    acc: dict[str, dict[str, dict[str, list]]] = {}
    for s in samples:
        for group, g in (s.get("workload") or {}).items():
            row = acc.setdefault(s["phase"], {}).setdefault(group, {"count": [], "primary_mb": [], "rss_mb": [], "cpu_percent": [], "unreadable_primary": []})
            for key in row:
                if g.get(key) is not None:
                    row[key].append(g[key])
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for phase, groups in acc.items():
        for group, v in groups.items():
            stat = {}
            for key in ("primary_mb", "rss_mb", "cpu_percent"):
                vals = v[key]
                stat[f"{key}_mean"] = round(sum(vals) / len(vals), 1) if vals else None
                stat[f"{key}_peak"] = round(max(vals), 1) if vals else None
            stat["count_min"] = min(v["count"]) if v["count"] else None
            stat["count_max"] = max(v["count"]) if v["count"] else None
            stat["unreadable_primary_max"] = max(v["unreadable_primary"]) if v["unreadable_primary"] else 0
            stat["samples"] = len(v["count"])
            out.setdefault(phase, {})[group] = stat
    return out


def workload_drift(samples: list[dict[str, Any]], drift_pct: float, busy_cpu: float) -> dict[str, Any]:
    """Warnings for count changes, memory drift from the reference phase, and busy groups."""
    per_phase = workload_by_phase(samples)
    if not per_phase:
        return {"reference_phase": None, "warnings": []}
    reference = "baseline" if "baseline" in per_phase else next(iter(per_phase))
    warnings = []
    last_count: dict[str, int] = {}
    reported: set[tuple[str, str]] = set()
    for s in samples:
        for group, g in (s.get("workload") or {}).items():
            before = last_count.get(group)
            if before is not None and g["count"] != before and (group, s["phase"]) not in reported:
                reported.add((group, s["phase"]))
                warnings.append({"kind": "count_changed", "group": group, "phase": s["phase"],
                                 "message": f"workload drift: {group} went from {before} to {g['count']} processes during {s['phase']}"})
            last_count[group] = g["count"]
    ref = per_phase[reference]
    for phase, groups in per_phase.items():
        for group, st in groups.items():
            base = ref.get(group) or {}
            metric = "primary_mb" if st.get("primary_mb_mean") is not None and base.get("primary_mb_mean") is not None else "rss_mb"
            cur, ref_mean = st.get(f"{metric}_mean"), base.get(f"{metric}_mean")
            if phase != reference and cur is not None and ref_mean:
                pct = 100 * (cur - ref_mean) / ref_mean
                if abs(pct) > drift_pct:
                    warnings.append({"kind": "memory_drift", "group": group, "phase": phase, "metric": metric, "pct": round(pct, 1),
                                     "message": f"workload drift: {group} {metric} mean {cur} MB in {phase} vs {ref_mean} MB in {reference} ({pct:+.0f} %)"})
            if (st.get("cpu_percent_mean") or 0) > busy_cpu:
                warnings.append({"kind": "busy", "group": group, "phase": phase, "cpu": st["cpu_percent_mean"],
                                 "message": f"workload drift: {group} used {st['cpu_percent_mean']} % CPU on average during {phase} (threshold {busy_cpu} %)"})
    return {"reference_phase": reference, "warnings": warnings}


def summarize_at(assignment: dict[str, Members], measured: dict[str, dict[str, Any]], checks: list[dict[str, Any]]) -> dict[str, Any]:
    """Compact workload summary for system.json."""
    status = {c["group"]: c for c in checks}
    return {name: {**measured.get(name, {}), "check": status.get(name, {}).get("status"), "check_message": status.get(name, {}).get("message")}
            for name in assignment}
