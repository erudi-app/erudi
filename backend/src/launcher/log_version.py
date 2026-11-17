"""``backend.log`` belongs to the app version that is running (#576).

The log directory is per user and outlives updates and reinstalls, and the
Diagnostics page lists the WARNING-and-above records of ``backend.log``. Left
alone, that page shows an install's whole life: records an older version wrote
about defects since fixed, indistinguishable from today's problems.

So at a start whose app version differs from the last run's, ``run.py`` rolls
``backend.log`` over into its numbered history (``backend.log.1`` ...
``backend.log.10``, the scheme ``RotatingFileHandler`` already uses) before the
logger opens it. The older history stays in those files; the Diagnostics page
reads the live file only. The version of the last run is kept beside the log,
in ``backend.log.version``, so a restart or a reinstall of the same version
keeps its records.

The app version is the Electron app's: the backend has none of its own, so the
main process passes ``app.getVersion()`` as ``ERUDI_APP_VERSION``. A backend
started without it (by hand in development, the CI smoke boot) changes nothing.

Renaming the live file can fail -- on Windows, when another process holds it
open. The start goes on, and the switch is still recorded: the first record
written for the new version is a marker (``VERSION_SWITCH_MARKER``), and the
Diagnostics reader drops every record that precedes the last marker it sees.

This module imports nothing from ``src.core.logging``: it runs before the
logger is configured, which opens the file this module renames.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

LOG_FILE_NAME = "backend.log"
LOG_BACKUP_COUNT = 10  # backend.log.1 ... backend.log.10
VERSION_FILE_NAME = "backend.log.version"
APP_VERSION_ENV = "ERUDI_APP_VERSION"

# The first record of a version's log. The Diagnostics reader
# (src/domains/diagnostics/log_reader.py) keeps only what follows the last one.
VERSION_SWITCH_MARKER = "Diagnostics starts here: backend.log now belongs to app version"

# A plain version string: what electron-builder stamps (1.3.0, 1.3.0-beta.2,
# 1.3.0+build.7). Anything else is ignored rather than written beside a file
# name and into an ASCII log line.
_VERSION_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z.+-]{0,63}$")


@dataclass(frozen=True)
class LogVersionSwitch:
    """A start whose app version differs from the last run's.

    Attributes:
        version: The app version now running.
        previous: The last run's version; None when unknown (a fresh install,
            or a log written before versioned logs existed).
        rotated: True when the previous ``backend.log`` went into the history.
        error: Why the roll-over or the version file failed, else None.
    """

    version: str
    previous: Optional[str]
    rotated: bool
    error: Optional[str]


def app_version_from_env() -> Optional[str]:
    """The app version the Electron main process passed, or None."""
    raw = (os.environ.get("ERUDI_APP_VERSION") or "").strip()
    return raw if _VERSION_RE.match(raw) else None


def _read_previous_version(version_file: Path) -> Optional[str]:
    try:
        return version_file.read_text(encoding="utf-8").strip() or None
    except OSError:
        # Absent on a fresh install and before versioned logs existed; an
        # unreadable one is treated the same: the log is rolled over.
        return None


def _roll_over(log_file: Path) -> None:
    """``RotatingFileHandler.doRollover`` without a handler: shift the history, then the file."""
    for index in range(LOG_BACKUP_COUNT - 1, 0, -1):
        source = log_file.with_name(f"{log_file.name}.{index}")
        if source.exists():
            os.replace(source, log_file.with_name(f"{log_file.name}.{index + 1}"))
    os.replace(log_file, log_file.with_name(f"{log_file.name}.1"))


def switch_log_for_version(log_dir: Path, version: Optional[str]) -> Optional[LogVersionSwitch]:
    """Give ``backend.log`` to ``version`` when the last run was another version.

    Must run before ``src.core.logging`` is imported. Never raises.

    Returns:
        The switch, or None when there is nothing to switch: no version known,
        or the same version as the last run.
    """
    if not version:
        return None
    log_dir = Path(log_dir)
    version_file = log_dir / VERSION_FILE_NAME
    previous = _read_previous_version(version_file)
    if previous == version:
        return None

    log_file = log_dir / LOG_FILE_NAME
    rotated = False
    error: Optional[str] = None
    try:
        if log_file.exists() and log_file.stat().st_size > 0:
            _roll_over(log_file)
            rotated = True
    except OSError as exc:
        # Typically Windows, the file held by another process. The marker the
        # caller writes next hides the older records from Diagnostics instead.
        error = f"{type(exc).__name__}: {exc}"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        version_file.write_text(version, encoding="utf-8")
    except OSError as exc:
        # The next start switches again; the marker still holds for this run.
        error = error or f"{type(exc).__name__}: {exc}"
    return LogVersionSwitch(version=version, previous=previous, rotated=rotated, error=error)


def record_log_version(
    logger: logging.Logger, version: Optional[str], switch: Optional[LogVersionSwitch]
) -> None:
    """Write the version records of this start, once the logger exists.

    A switch writes the marker the Diagnostics reader cuts at, then, when the
    old file could not be rolled over, one WARNING after it (so the page shows
    it). Every start with a known version says which version it is, so a raw
    read of the log shows where each version begins.
    """
    if switch is not None:
        logger.info(f"{VERSION_SWITCH_MARKER} {switch.version} (previous: {switch.previous})")
        if switch.error:
            logger.warning(
                f"Could not start a fresh {LOG_FILE_NAME} for app version {switch.version}; "
                f"the records of earlier versions stay above the marker: {switch.error}"
            )
    if version:
        logger.info(f"Erudi backend starting, app version {version}")
