"""The measured memory of the MLX child, turn by turn, kept on disk.

Every conversation or Arena turn on Apple Silicon records one observation:
the child's real peak above its base footprint next to what the memory prior
predicted (``src.engines.memory_budget``). Nothing here changes the prior: the
file is the material a later calibration will be designed on, collected by QA
across machines, and an observation above its prediction is already one
WARNING in ``backend.log``.

File: ``<data folder>/memory_calibration.json`` (local only, never sent
anywhere; deleting it loses only the recorded observations). One entry per
(model, machine, runtime) key, whose components are kept in clear -- none is a
path: the model, the artifact's size, the GPU working set, the macOS, mlx and
mlx_vlm versions, the child's runtime configuration. Not the app version. An
entry holds the last measured base footprint and a ring of the last
``RING_PER_KEY`` observations. Entries of an older runtime are kept, labelled
by their version components, within ``MAX_OBSERVATIONS`` overall (oldest
dropped first).

Several backends can run at once (ports 27182-27199): every write re-reads the
file under an exclusive ``flock`` on a sibling lock file, merges (rings
de-duplicated by the observation's id), and replaces the file atomically. An
unreadable or corrupt file is ignored with one WARNING and rewritten on the
next update.
"""

from __future__ import annotations

import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.logging import logger

FILE_NAME = "memory_calibration.json"
SCHEMA_VERSION = 1
RING_PER_KEY = 64
MAX_OBSERVATIONS = 512


def default_observations_path() -> Path:
    """``<data folder>/memory_calibration.json``."""
    from src.core import config

    return Path(config.DATA_ROOT) / FILE_NAME


# The path every read and write uses (tests redirect it).
observations_path = default_observations_path


def _package_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        # A missing distribution is a fact worth recording as unknown, not an
        # error: the observation stays usable.
        return None


def key_components(
    *,
    model: Optional[str],
    artifact_bytes: Optional[int],
    working_set_bytes: Optional[int],
    runtime: Dict[str, Any],
) -> Dict[str, Any]:
    """The key of an entry, in clear: what a measurement depends on."""
    return {
        "model": model,
        "artifact_bytes": artifact_bytes,
        "working_set_bytes": working_set_bytes,
        "macos": platform.mac_ver()[0] or None,
        "mlx": _package_version("mlx"),
        "mlx_vlm": _package_version("mlx-vlm"),
        "runtime": dict(runtime),
    }


def entry_key(components: Dict[str, Any]) -> str:
    return json.dumps(components, sort_keys=True, separators=(",", ":"))


def _read(path: Path) -> Dict[str, Any]:
    """The file's content, or an empty document when absent or corrupt (one
    WARNING; the update in progress rewrites the file)."""
    empty: Dict[str, Any] = {"version": SCHEMA_VERSION, "entries": {}}
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return empty
    except OSError as exc:
        logger.warning(
            f"Memory observations unreadable ({type(exc).__name__}); starting a new file"
        )
        return empty
    try:
        data = json.loads(raw)
        if not isinstance(data, dict) or not isinstance(data.get("entries"), dict):
            raise ValueError("unexpected shape")
    except ValueError:
        logger.warning("Memory observations file is corrupt; ignoring it and rewriting it")
        return empty
    return data


def load() -> Dict[str, Any]:
    """The recorded entries (read-only; no lock)."""
    return _read(observations_path())


def _bound(entries: Dict[str, Any]) -> None:
    """At most ``MAX_OBSERVATIONS`` overall: the oldest go first."""
    everything = [
        (observation.get("at", 0), key, observation.get("id"))
        for key, entry in entries.items()
        for observation in entry.get("observations", [])
    ]
    excess = len(everything) - MAX_OBSERVATIONS
    if excess <= 0:
        return
    doomed = {(key, obs_id) for _, key, obs_id in sorted(everything, key=lambda t: t[0])[:excess]}
    for key, entry in entries.items():
        entry["observations"] = [
            o for o in entry.get("observations", []) if (key, o.get("id")) not in doomed
        ]


def _merge(
    data: Dict[str, Any],
    components: Dict[str, Any],
    base_bytes: Optional[int],
    observation: Dict[str, Any],
) -> None:
    entries = data.setdefault("entries", {})
    key = entry_key(components)
    entry = entries.setdefault(key, {"components": components, "observations": []})
    if base_bytes is not None:
        entry["base_bytes"] = base_bytes
    ring = [o for o in entry.get("observations", []) if o.get("id") != observation.get("id")]
    ring.append(observation)
    entry["observations"] = ring[-RING_PER_KEY:]
    _bound(entries)
    data["version"] = SCHEMA_VERSION


def record(
    components: Dict[str, Any], base_bytes: Optional[int], observation: Dict[str, Any]
) -> None:
    """Merge one observation into the file, under an exclusive lock.

    Raises ``OSError`` when the data folder cannot be written; the caller
    decides what that costs (one observation, never a turn).
    """
    import fcntl  # POSIX only: this module runs on macOS (MLX).

    path = observations_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with open(lock_path, "a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            data = _read(path)
            _merge(data, components, base_bytes, observation)
            fd, temp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(data, handle, indent=1, sort_keys=True)
                os.replace(temp, path)
            except BaseException:
                Path(temp).unlink(missing_ok=True)
                raise
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
