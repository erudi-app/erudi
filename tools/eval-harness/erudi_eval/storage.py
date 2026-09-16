"""Disk usage: allocated-size tree walker (no symlink following), snapshots and deltas."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HAS_BLOCKS = hasattr(os.stat_result, "st_blocks")
SIZE_KIND = "allocated" if HAS_BLOCKS else "logical"


@dataclass
class Usage:
    bytes: int = 0  # allocated when the OS exposes st_blocks, logical otherwise
    logical_bytes: int = 0
    files: int = 0
    dirs: int = 0
    symlinks: int = 0
    errors: int = 0

    def add(self, other: "Usage") -> None:
        for f in ("bytes", "logical_bytes", "files", "dirs", "symlinks", "errors"):
            setattr(self, f, getattr(self, f) + getattr(other, f))

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


def _entry_size(st: os.stat_result) -> tuple[int, int]:
    allocated = st.st_blocks * 512 if HAS_BLOCKS else st.st_size
    return allocated, st.st_size


def usage(path: Path, _seen: set | None = None) -> Usage:
    """Recursive size of `path`. Symlinks and junctions are counted as links, never followed.
    Hard-linked files are counted once."""
    seen = set() if _seen is None else _seen
    u = Usage()
    try:
        st = os.lstat(path)
    except OSError:
        u.errors += 1
        return u
    if stat.S_ISLNK(st.st_mode):
        u.symlinks += 1
        u.bytes, u.logical_bytes = _entry_size(st)
        return u
    if not stat.S_ISDIR(st.st_mode):
        key = (st.st_dev, st.st_ino)
        if st.st_nlink > 1 and key in seen:
            return u
        seen.add(key)
        u.files += 1
        u.bytes, u.logical_bytes = _entry_size(st)
        return u
    u.dirs += 1
    stack = [str(path)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                entries = list(it)
        except OSError:
            u.errors += 1
            continue
        for entry in entries:
            try:
                est = entry.stat(follow_symlinks=False)
            except OSError:
                u.errors += 1
                continue
            is_junction = getattr(entry, "is_junction", lambda: False)()
            if entry.is_symlink() or is_junction:
                u.symlinks += 1
                a, lg = _entry_size(est)
                u.bytes += a
                u.logical_bytes += lg
            elif stat.S_ISDIR(est.st_mode):
                u.dirs += 1
                stack.append(entry.path)
            else:
                key = (est.st_dev, est.st_ino)
                if est.st_nlink > 1:
                    if key in seen:
                        continue
                    seen.add(key)
                u.files += 1
                a, lg = _entry_size(est)
                u.bytes += a
                u.logical_bytes += lg
    return u


def children_usage(path: Path, top: int | None = None) -> list[dict[str, Any]]:
    """Per-entry usage of the direct children of `path`, largest first."""
    p = Path(path)
    rows = []
    try:
        entries = sorted(os.scandir(p), key=lambda e: e.name)
    except OSError:
        return rows
    for entry in entries:
        rows.append({"name": entry.name, **usage(Path(entry.path)).as_dict()})
    rows.sort(key=lambda r: r["bytes"], reverse=True)
    return rows[:top] if top else rows


def snapshot(label: str, install: dict[str, Path | None], data_root: Path | None, log_paths: dict[str, Path | None]) -> dict[str, Any]:
    """One storage snapshot.

    install: {"app": bundle/exe/AppImage path, "resources": resources dir, "backend_lib": _internal dir}
    """
    snap: dict[str, Any] = {"label": label, "size_kind": SIZE_KIND, "install": {}, "data": {}, "logs": {}}
    app, resources, backend_lib = install.get("app"), install.get("resources"), install.get("backend_lib")
    if app and Path(app).exists():
        snap["install"]["total"] = usage(Path(app)).as_dict()
    if resources and Path(resources).is_dir():
        snap["install"]["resources_top"] = children_usage(Path(resources))
    if backend_lib and Path(backend_lib).is_dir():
        snap["install"]["backend_lib_top25"] = children_usage(Path(backend_lib), top=25)

    if data_root and Path(data_root).is_dir():
        root = Path(data_root)
        snap["data"]["root"] = str(root)
        snap["data"]["total"] = usage(root).as_dict()
        data = root / "data"
        snap["data"]["models"] = children_usage(data / "models")
        for name in ("models_cache",):
            if (data / name).exists():
                snap["data"][name] = usage(data / name).as_dict()
        pg = data / "postgres"
        if pg.exists():
            snap["data"]["postgres"] = {
                "total": usage(pg).as_dict(),
                "base": usage(pg / "base").as_dict(),
                "pg_wal": usage(pg / "pg_wal").as_dict(),
            }
        if (root / "db-backups").exists():
            snap["data"]["db_backups"] = usage(root / "db-backups").as_dict()
        known_data = {"models", "models_cache", "postgres"}
        snap["data"]["other"] = [r for r in children_usage(data) if r["name"] not in known_data] + [
            r for r in children_usage(root) if r["name"] not in {"data", "db-backups"}
        ]
    for name, path in log_paths.items():
        if path and Path(path).exists():
            snap["logs"][name] = usage(Path(path)).as_dict()
    return snap


def flatten(snap: dict[str, Any]) -> dict[str, int]:
    """{label: bytes} over every sized entry of a snapshot, for deltas."""
    flat: dict[str, int] = {}

    def walk(prefix: str, node: Any) -> None:
        if isinstance(node, dict):
            if "bytes" in node and "files" in node:
                flat[prefix] = node["bytes"]
                return
            for k, v in node.items():
                if k in ("label", "size_kind", "root"):
                    continue
                walk(f"{prefix}/{k}" if prefix else k, v)
        elif isinstance(node, list):
            for row in node:
                if isinstance(row, dict) and "name" in row:
                    flat[f"{prefix}/{row['name']}"] = row["bytes"]

    walk("", {k: snap.get(k) for k in ("install", "data", "logs")})
    return flat


def delta(before: dict[str, Any], after: dict[str, Any], min_bytes: int = 1) -> list[dict[str, Any]]:
    a, b = flatten(before), flatten(after)
    rows = []
    for key in sorted(set(a) | set(b)):
        d = b.get(key, 0) - a.get(key, 0)
        if abs(d) >= min_bytes:
            rows.append({"entry": key, "before": a.get(key, 0), "after": b.get(key, 0), "delta": d})
    return rows
