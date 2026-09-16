"""Small shared helpers: time stamps, JSONL writing, condition waits."""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

MB = 1024 * 1024


def utc_iso(t: float | None = None) -> str:
    """UTC ISO-8601 with milliseconds and a Z suffix (same shape as the app's events)."""
    t = time.time() if t is None else t
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def parse_iso(value: str) -> float | None:
    """Parse an ISO-8601 timestamp (Z or offset; naive = UTC) to epoch seconds."""
    if not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


class JsonlWriter:
    """Thread-safe append-only JSON-lines file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=str, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            self._fh.close()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    p = Path(path)
    if not p.exists():
        return out
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn last line after a crash is not worth failing the report
    return out


def write_json(path: Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, default=str, ensure_ascii=False) + "\n", encoding="utf-8")


def wait_until(
    predicate: Callable[[], Any],
    timeout: float,
    interval: float = 1.0,
    stop: threading.Event | None = None,
) -> Any:
    """Poll `predicate` until it returns a truthy value or `timeout` elapses.

    Returns the truthy value, or None on timeout. Exceptions from the predicate
    count as "not yet" (the thing being waited for is usually not up yet).
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            value = predicate()
            if value:
                return value
        except Exception:  # noqa: BLE001 - "not ready" is expressed as an exception often
            pass
        if time.monotonic() >= deadline or (stop is not None and stop.is_set()):
            return None
        time.sleep(min(interval, max(0.0, deadline - time.monotonic())))


_SECRET_FLAGS = ("--api-key", "--api_key", "--password", "--token")
_SECRET_INLINE = re.compile(r"(?i)(password|passwd|api[-_]?key|token|secret)=([^\s&]+)")


def redact_cmdline(cmdline: list[str]) -> list[str]:
    """Hide secret values in a command line (llama-server carries `--api-key <key>`)."""
    out: list[str] = []
    hide_next = False
    for arg in cmdline:
        if hide_next:
            out.append("<redacted>")
            hide_next = False
            continue
        lowered = arg.lower()
        if lowered in _SECRET_FLAGS:
            out.append(arg)
            hide_next = True
            continue
        if any(lowered.startswith(flag + "=") for flag in _SECRET_FLAGS):
            out.append(arg.split("=", 1)[0] + "=<redacted>")
            continue
        out.append(_SECRET_INLINE.sub(lambda m: f"{m.group(1)}=<redacted>", arg))
    return out
