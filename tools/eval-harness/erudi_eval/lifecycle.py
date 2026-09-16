"""Backend lifecycle events from the Electron stdout capture log, and time-filtered log copies.

Capture log line shape (frontend/src/main.js `log()`):
    [2026-09-15T10:00:00.123Z] Backend stdout: {"event": "phase", "phase": "running_migrations", "ts": "..."}
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .util import parse_iso

CAPTURE_LINE = re.compile(r"^\[(?P<logged>[^\]]+)\] (?P<msg>.*)$")
STDOUT_PREFIX = "Backend stdout: "
# backend.log: "[INFO] 2026-09-15 10:00:00.123Z [req] - name ..." or "2026-09-15 10:00:00.123Z [req] ..."
BACKEND_LOG_TS = re.compile(r"^(?:\[[A-Z]+\]\s+)?(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?Z?)")


def parse_capture_log(text: str, t0: float) -> list[dict[str, Any]]:
    """Structured backend events logged at or after t0 (epoch seconds)."""
    events = []
    for line in text.splitlines():
        m = CAPTURE_LINE.match(line)
        if not m or not m.group("msg").startswith(STDOUT_PREFIX):
            continue
        logged = parse_iso(m.group("logged"))
        if logged is None or logged < t0:
            continue
        try:
            payload = json.loads(m.group("msg")[len(STDOUT_PREFIX):])
        except json.JSONDecodeError:
            continue  # ordinary non-JSON backend output
        if not isinstance(payload, dict) or "event" not in payload:
            continue
        payload = dict(payload)
        payload["logged_at"] = logged
        payload["ts_epoch"] = parse_iso(payload.get("ts", "")) or logged
        events.append(payload)
    return events


def read_capture_events(paths: list[Path], t0: float) -> list[dict[str, Any]]:
    """Events from the capture log and its rotated sibling (erudi-backend.old.log)."""
    events = []
    for p in paths:
        if p.exists():
            events += parse_capture_log(p.read_text(encoding="utf-8", errors="replace"), t0)
    events.sort(key=lambda e: e["ts_epoch"])
    return events


def boot_timeline(t0: float, events: list[dict[str, Any]], marks: dict[str, float | None]) -> list[dict[str, Any]]:
    """Rows (label, epoch, seconds since t0) for the boot sequence.

    marks: harness-observed instants, e.g. {"cdp_answering": t, "health_200": t, "renderer_load": t}.
    """
    rows = [{"label": "harness_launch", "t": t0, "since_t0_s": 0.0}]
    for e in events:
        name = e["event"]
        if name == "phase":
            label = f"phase:{e.get('phase')}"
        elif name in ("starting", "ready", "startup_error", "engine_notice", "shutdown"):
            label = name + (f":{e.get('code')}" if e.get("code") else "")
        else:
            continue
        rows.append({"label": label, "t": e["ts_epoch"], "since_t0_s": round(e["ts_epoch"] - t0, 3)})
    for label, t in marks.items():
        if t is not None:
            rows.append({"label": label, "t": t, "since_t0_s": round(t - t0, 3)})
    rows.sort(key=lambda r: r["t"])
    return rows


def filter_capture_lines(text: str, t0: float) -> list[str]:
    out, keep = [], False
    for line in text.splitlines():
        m = CAPTURE_LINE.match(line)
        if m:
            ts = parse_iso(m.group("logged"))
            keep = ts is not None and ts >= t0
        if keep:
            out.append(line)
    return out


def filter_backend_log_lines(text: str, t0: float) -> list[str]:
    """Lines with a leading UTC timestamp >= t0; continuation lines (tracebacks) follow their record."""
    out, keep = [], False
    for line in text.splitlines():
        m = BACKEND_LOG_TS.match(line)
        if m:
            raw = m.group(1).replace(",", ".").replace(" ", "T")
            ts = parse_iso(raw if raw.endswith("Z") else raw + "Z")
            keep = ts is not None and ts >= t0
        if keep:
            out.append(line)
    return out


def copy_logs(dest: Path, t0: float, capture_logs: list[Path], log_dir: Path | None) -> dict[str, int]:
    """Write time-filtered copies into dest/. Returns {file name: lines}. Never copies whole historic files."""
    dest.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for p in capture_logs:
        if p.exists():
            lines = filter_capture_lines(p.read_text(encoding="utf-8", errors="replace"), t0)
            (dest / p.name).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            counts[p.name] = len(lines)
    if log_dir and log_dir.is_dir():
        for p in sorted(log_dir.glob("backend.log*")) + sorted(log_dir.glob("mlx-child-*.log*")):
            if not p.is_file() or p.stat().st_mtime < t0:
                continue  # untouched during this run
            text = p.read_text(encoding="utf-8", errors="replace")
            lines = filter_backend_log_lines(text, t0)
            if not lines and p.name.startswith("mlx-child-") and p.stat().st_mtime >= t0:
                # Child logs have no guaranteed timestamp format: keep the tail of a file written during the run.
                lines = text.splitlines()[-2000:]
            if lines:
                (dest / p.name).write_text("\n".join(lines) + "\n", encoding="utf-8")
                counts[p.name] = len(lines)
    return counts


def find_log_lines(log_dir: Path | None, needle: str, t0: float) -> list[str]:
    """Lines of backend.log written after t0 that contain `needle` (e.g. 'Turn mode:')."""
    if not log_dir:
        return []
    p = Path(log_dir) / "backend.log"
    if not p.exists():
        return []
    return [line for line in filter_backend_log_lines(p.read_text(encoding="utf-8", errors="replace"), t0) if needle in line]
