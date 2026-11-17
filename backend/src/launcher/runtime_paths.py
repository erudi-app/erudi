"""Runtime path management for launcher and backend modules.

This module centralizes the logic that determines where the Erudi backend
should read and write its runtime artifacts (embedded PostgreSQL data dir,
log files, cached models). The launcher initializes this state before
FastAPI imports occur, but a development fallback keeps local `uvicorn`
workflows working without the launcher.

``ERUDI_DATA_ROOT`` (optional env var) points the whole instance -- data dir
AND log dir -- at an explicit root instead of the default dev/prod layout.
It exists for isolated side-by-side instances (integration tests booting a
throwaway backend, QA runs next to a live app) that must not share the
default embedded-Postgres cluster or append to the same log files.
"""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

APP_NAME = "erudi"

# The file that marks a data or log folder as Erudi's. Clear All Data (the
# Electron main process, frontend/src/utils/dataLocations.js) deletes a folder
# only when it carries this file, so a folder Erudi did not create is never
# deleted -- an ERUDI_DATA_ROOT pointed at an existing folder that already
# holds a `data/` of the user's own, for one. The name is pinned on both sides
# by backend/tests/fixtures/runtime_paths_layout.json.
OWNERSHIP_SENTINEL = ".erudi-owned"
_SENTINEL_TEXT = (
    "Erudi created this folder and keeps its data or logs here.\n"
    "Help > Clear All Data deletes it.\n"
)


@dataclass(frozen=True)
class RuntimePaths:
    """Resolved directories for the current Erudi process."""

    mode: str
    backend_root: Path
    data_dir: Path
    log_dir: Path


_RUNTIME_PATHS: Optional[RuntimePaths] = None


def initialize_runtime_paths(
    mode: str, backend_root: Path, packaged_data_dir: Optional[Path] = None
) -> RuntimePaths:
    """Initialize runtime paths for the current process.

    Args:
        mode: Runtime mode, typically ``"dev"`` or ``"prod"``.
        backend_root: Path to the backend root directory (contains ``src``).
        packaged_data_dir: Optional path to the read-only data payload bundled
            with the application. Only required for bundled builds.

    Returns:
        RuntimePaths: Resolved directory information for the process.

    Raises:
        ValueError: If runtime paths are initialized more than once or the mode
            is unsupported.
    """
    global _RUNTIME_PATHS
    if _RUNTIME_PATHS is not None:
        raise ValueError("Runtime paths already initialized")

    runtime_paths = _compute_runtime_paths(
        mode=mode, backend_root=backend_root, packaged_data_dir=packaged_data_dir
    )
    _RUNTIME_PATHS = runtime_paths
    return runtime_paths


def ensure_runtime_paths_initialized(backend_root: Optional[Path] = None) -> RuntimePaths:
    """Ensure runtime paths are initialized, defaulting to development mode.

    Args:
        backend_root: Explicit backend root directory. When omitted, defaults to
            the repository layout (``backend/``).

    Returns:
        RuntimePaths: Resolved runtime directories.
    """
    global _RUNTIME_PATHS
    if _RUNTIME_PATHS is None:
        root = backend_root or Path(__file__).resolve().parents[2]
        packaged_data = root / "data"
        _RUNTIME_PATHS = _compute_runtime_paths(
            mode="dev", backend_root=root, packaged_data_dir=packaged_data
        )
    return _RUNTIME_PATHS


def get_runtime_paths() -> RuntimePaths:
    """Return the current runtime paths.

    Returns:
        RuntimePaths: Current runtime path configuration.

    Raises:
        RuntimeError: If runtime paths were not initialized yet.
    """
    if _RUNTIME_PATHS is None:
        raise RuntimeError("Runtime paths not initialized")
    return _RUNTIME_PATHS


def _compute_runtime_paths(
    mode: str, backend_root: Path, packaged_data_dir: Optional[Path]
) -> RuntimePaths:
    """Resolve runtime paths based on mode and platform."""
    normalized_mode = mode.lower()
    if normalized_mode not in {"dev", "prod"}:
        raise ValueError(f"Unsupported runtime mode: {mode}")

    backend_root = backend_root.resolve()
    packaged_data_dir = (packaged_data_dir or backend_root / "data").resolve()

    override = _data_root_override()
    if override is not None:
        data_dir = override / "data"
        log_dir = override / "logs"
        # Under a root the user chose, a folder is Erudi's only if Erudi
        # creates it (or marked it on an earlier start).
        for folder in (data_dir, log_dir):
            created = not folder.exists()
            folder.mkdir(parents=True, exist_ok=True)
            _claim_folder(folder, owned=created)
        if normalized_mode == "prod":
            # A packaged build pointed at an explicit root still needs its
            # bundled payload (catalog snapshot, seed data) -- but never the
            # macOS symlink swap: the bundle must keep pointing at the default
            # location, not at a possibly-temporary override root.
            _copy_packaged_payload(packaged_data_dir, data_dir)
        return RuntimePaths(
            mode=normalized_mode, backend_root=backend_root, data_dir=data_dir, log_dir=log_dir
        )

    if normalized_mode == "dev":
        data_dir, log_dir = _setup_dev_paths(backend_root)
    else:
        data_dir, log_dir = _setup_prod_paths(packaged_data_dir)
    # The default folders are Erudi's by name (an `erudi` folder in the
    # platform's app-data and log locations, or the repository's backend/):
    # marked even when an install from before the sentinel created them.
    for folder in (data_dir, log_dir):
        _claim_folder(folder, owned=True)

    return RuntimePaths(
        mode=normalized_mode, backend_root=backend_root, data_dir=data_dir, log_dir=log_dir
    )


def _claim_folder(folder: Path, owned: bool) -> None:
    """Write the ownership sentinel in ``folder`` when Erudi owns it."""
    sentinel = folder / OWNERSHIP_SENTINEL
    if not owned or sentinel.exists():
        return
    try:
        sentinel.write_text(_SENTINEL_TEXT, encoding="utf-8")
    except OSError:
        # Logging is not configured yet (it needs these paths). Without the
        # file, Clear All Data skips this folder and names it: the safe side.
        pass


def _env_dir(name: str, default: Path) -> Path:
    """An env var naming a folder; unset and empty both mean ``default``."""
    value = os.getenv(name)
    return Path(value) if value else default


def _data_root_override() -> Optional[Path]:
    """Explicit instance root from ``ERUDI_DATA_ROOT`` (empty/unset -> None)."""
    raw = os.getenv("ERUDI_DATA_ROOT")
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def _setup_dev_paths(backend_root: Path) -> tuple[Path, Path]:
    """Prepare development data and log directories."""
    data_dir = backend_root / "data"
    if data_dir.exists() and data_dir.is_symlink():
        data_dir.unlink()
    data_dir.mkdir(parents=True, exist_ok=True)

    log_dir = backend_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, log_dir


def _setup_prod_paths(packaged_data_dir: Path) -> tuple[Path, Path]:
    """Prepare production data and log directories in user-writable locations."""
    data_dir, log_dir = _determine_prod_directories()
    _copy_packaged_payload(packaged_data_dir, data_dir)

    if platform.system() == "Darwin":
        _ensure_macos_symlink(packaged_data_dir, data_dir)

    log_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, log_dir


def _determine_prod_directories() -> tuple[Path, Path]:
    """Resolve OS-appropriate data and log directories for bundled builds."""
    system = platform.system()
    if system == "Darwin":
        data_root = Path.home() / "Library" / "Application Support" / APP_NAME / "backend" / "prod"
        log_dir = Path.home() / "Library" / "Logs" / APP_NAME
    elif system == "Windows":
        local_app_data = _env_dir("LOCALAPPDATA", Path.home() / "AppData" / "Local")
        data_root = local_app_data / APP_NAME / "backend" / "prod"
        log_dir = local_app_data / APP_NAME / "logs"
    else:
        data_home = _env_dir("XDG_DATA_HOME", Path.home() / ".local" / "share")
        state_home = _env_dir("XDG_STATE_HOME", Path.home() / ".local" / "state")
        data_root = data_home / APP_NAME / "backend" / "prod"
        log_dir = state_home / APP_NAME / "logs"

    data_dir = data_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, log_dir


def _copy_packaged_payload(source: Path, destination: Path) -> None:
    """Copy packaged data files into the destination directory."""
    if not source.exists():
        return

    try:
        if source.resolve(strict=False) == destination.resolve():
            return
    except Exception:
        pass

    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if target.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _ensure_macos_symlink(packaged_path: Path, target_path: Path) -> None:
    """Ensure packaged data directory is a symlink to the writable target."""
    if not packaged_path.exists():
        return

    target_resolved = target_path.resolve()

    try:
        resolved_target = packaged_path.resolve(strict=False)
    except Exception:
        resolved_target = None

    if resolved_target == target_resolved:
        return

    if packaged_path.is_symlink():
        packaged_path.unlink()
    elif packaged_path.is_dir():
        shutil.rmtree(packaged_path)
    else:
        packaged_path.unlink()

    os.symlink(target_resolved, packaged_path)
