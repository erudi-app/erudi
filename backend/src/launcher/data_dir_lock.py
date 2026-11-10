"""One backend per data folder.

Two backends on one data folder would share the embedded PostgreSQL cluster,
the model files and the download staging directories, each believing it owns
them. The launcher (``backend/run.py``) therefore takes an exclusive OS lock
on ``<data dir>/backend.lock`` before anything touches the cluster, and holds
it for the life of the process. A second backend fails to take it and stops
with the ``DATA_DIR_IN_USE`` startup error.

The lock is the operating system's, not the file's existence: ``flock`` on
POSIX, ``msvcrt.locking`` (``LockFileEx``) on Windows. Both are released by
the OS when the process ends, however it ends (``kill -9``, a crash, a power
cut leaves no lock behind), so a stale file never blocks a start. The file
holds the holder's pid, for the error message only. Both locks belong to one
open file, so a second acquisition in the same process is refused too, and
the descriptor is not inherited by child processes (PEP 446), so neither
PostgreSQL nor an inference server keeps the folder locked after the backend
is gone.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

LOCK_FILE_NAME = "backend.lock"

# Windows locks a byte range, and a locked range cannot be read by another
# process: the lock sits far past the pid, which stays readable.
_WINDOWS_LOCK_OFFSET = 1 << 20


class DataDirInUse(Exception):
    """Another live process holds the data folder's lock."""

    def __init__(self, data_dir: Path, holder_pid: Optional[int]) -> None:
        self.data_dir = data_dir
        self.holder_pid = holder_pid
        holder = f" (pid {holder_pid})" if holder_pid else ""
        super().__init__(f"Another Erudi backend{holder} is already using {data_dir}")


@dataclass
class DataDirLock:
    """A held lock; it lasts until ``release`` or the end of the process."""

    path: Path
    fd: int

    def release(self) -> None:
        """Give the lock back (closing the descriptor releases it)."""
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


def _try_lock(fd: int) -> None:
    """Take the exclusive lock without waiting; ``OSError`` when it is held."""
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _read_pid(fd: int) -> Optional[int]:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        return int(os.read(fd, 32).split()[0])
    except (OSError, ValueError, IndexError):
        # The pid only decorates the message; an unreadable one is left out.
        return None


def acquire_data_dir_lock(data_dir: Path) -> DataDirLock:
    """Lock ``data_dir`` for this process, or raise ``DataDirInUse``.

    The caller keeps the returned lock for as long as the backend runs.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / LOCK_FILE_NAME
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        _try_lock(fd)
    except OSError:
        holder = _read_pid(fd)
        os.close(fd)
        raise DataDirInUse(data_dir, holder) from None
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode("ascii"))
    return DataDirLock(path=path, fd=fd)
