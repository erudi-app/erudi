"""Keep what the `mlx_vlm.server` child prints, so its death has last words.

`llama-server` is a `subprocess.Popen`: its merged stdout+stderr is a pipe the
parent drains into a bounded tail (`child_output.ChildOutputDrainer`), and that
tail is what a crash report quotes. The MLX child is a
`multiprocessing.Process`, which has no pipe -- its file descriptors are the
BACKEND's own, so before this module the child's output went to the backend's
stdout (the newline-delimited JSON channel `run.py` owns) and nothing of it
reached `backend.log` or the Diagnostics page. A probe timeout or a Metal
failure surfaced as an exit code and nothing else.

Since the parent cannot read the child, the child captures itself: this module
redirects its output into a per-spawn file in the backend's log directory, and
the parent reads the tail of that file exactly where the llama-cpp engines read
their drainer.

Why `os.dup2` and not `contextlib.redirect_stdout`
--------------------------------------------------
`redirect_stdout`/`redirect_stderr` rebind the `sys.stdout` / `sys.stderr`
OBJECTS, which covers Python-level writes and nothing else:

  - mlx and Metal are native code. A GPU command-buffer error, an allocator
    abort, a dyld failure -- the things that kill this child -- are written
    straight to file descriptor 1 or 2 by C, which never consults `sys.stdout`.
    Those are the last words worth having, and `dup2` is the only way to get
    them.
  - Handlers built before the redirect keep the stream they captured.
    `mlx_vlm.server.cli.main` calls `logging.basicConfig`, and uvicorn
    configures its own handlers, both AFTER this runs -- but any handler built
    earlier (in an import) would keep writing to the inherited descriptor.
  - `dup2` also stops the child from writing into the backend's stdout, which
    the Electron main process parses as launcher events.

`sys.stdout` / `sys.stderr` are additionally rebound to line-buffered wrappers
over the redirected descriptors: Python block-buffers 8 KiB when its output is
a file, and a child killed with a full buffer would take its last lines with
it.

Layout and lifetime
-------------------
One file per spawn, named after the port the child serves
(``mlx-child-<port>.log`` in the log directory `runtime_paths` resolves). Each
spawn rolls the previous file to ``.1`` and drops what falls off, so a port
keeps at most :data:`KEEP_PER_PORT` files; a size guard rolls the same way
while the child runs, so a chatty server cannot fill the disk. An orderly stop
deletes the files; a child that died on its own keeps them, because that is
the only account of the death there is.

Nothing secret reaches the file: the child receives its ``--api-key`` through
the pickled argv, and neither `mlx_vlm.server.cli` nor uvicorn ever prints it
(cli.py exports it to ``MLX_VLM_SERVER_API_KEY`` and moves on).
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Union

from src.launcher import ensure_runtime_paths_initialized

#: One live file plus one previous, per port.
KEEP_PER_PORT = 2

#: Roll the live file once it grows past this. mlx-vlm logs a line per request
#: and, at DEBUG, per generation step.
MAX_CHILD_LOG_BYTES = 2 * 1024 * 1024

#: How often the in-child guard looks at the file size.
SIZE_CHECK_INTERVAL_S = 5.0

#: How much of the tail a crash report quotes, matching the llama-cpp engines.
DEFAULT_TAIL_CHARS = 2000

CHILD_LOG_PREFIX = "mlx-child-"

_CHILD_LOGGER_NAME = "erudi.mlx-child"


def _log_dir() -> Path:
    """The backend's log directory, beside ``backend.log``."""
    return ensure_runtime_paths_initialized().log_dir


def child_log_path(port: int, log_dir: Optional[Path] = None) -> Path:
    """Where the child serving ``port`` writes what it prints."""
    return (log_dir if log_dir is not None else _log_dir()) / f"{CHILD_LOG_PREFIX}{port}.log"


def prepare_child_log(port: int, log_dir: Optional[Path] = None) -> Path:
    """Make room for a new spawn's file and return its path.

    Called in the PARENT, before the child starts: only the parent knows where
    the log directory is (a frozen child re-executes the binary with
    uninitialized runtime paths, and would resolve the development layout
    inside a read-only bundle), so the resolved path travels down as a spawn
    argument.

    Raises:
        OSError: If the directory cannot be created or the previous file
            cannot be rolled. The caller treats capture as best effort.
    """
    path = child_log_path(port, log_dir=log_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    roll_child_log(path)
    return path


def roll_child_log(path: Union[str, Path], keep: int = KEEP_PER_PORT) -> None:
    """Roll ``path`` to ``path.1`` (then ``.2`` ...), dropping the oldest.

    A no-op when nothing has been written yet. ``keep`` counts the live file,
    so the default keeps ``path`` and ``path.1``.
    """
    path = Path(path)
    if keep <= 1:
        path.unlink(missing_ok=True)
        return
    Path(f"{path}.{keep - 1}").unlink(missing_ok=True)
    for index in range(keep - 2, 0, -1):
        source = Path(f"{path}.{index}")
        if source.exists():
            os.replace(source, f"{path}.{index + 1}")
    if path.exists():
        os.replace(path, f"{path}.1")


def discard_child_log(path: Union[str, Path], keep: int = KEEP_PER_PORT) -> None:
    """Delete a port's files. Used after an orderly stop, never after a crash."""
    path = Path(path)
    path.unlink(missing_ok=True)
    for index in range(1, keep):
        Path(f"{path}.{index}").unlink(missing_ok=True)


def read_child_log_tail(path: Union[str, Path], max_chars: int = DEFAULT_TAIL_CHARS) -> str:
    """The child's last ``max_chars`` characters, oldest first, or ``""``.

    Reads the rolled ``.1`` file too: a roll that happened moments before the
    crash would otherwise leave a tail of two lines. Never raises -- an absent,
    unreadable or half-written file is answered with what could be read.
    """
    path = Path(path)
    tail = _read_tail(path, max_chars)
    if len(tail) < max_chars:
        previous = _read_tail(Path(f"{path}.1"), max_chars - len(tail))
        if previous:
            tail = f"{previous}{tail}"
    return tail


def _read_tail(path: Path, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            # 4 bytes is the widest a UTF-8 character gets, so this always
            # holds at least `max_chars` characters.
            if size > max_chars * 4:
                handle.seek(size - max_chars * 4)
            raw = handle.read()
    except OSError:
        return ""
    text = raw.decode("utf-8", errors="replace")
    return text[-max_chars:] if len(text) > max_chars else text


# ----------------------------------------------------------------------
# In-child: capture
# ----------------------------------------------------------------------


def redirect_stdio_to(
    path: Union[str, Path],
    *,
    max_bytes: int = MAX_CHILD_LOG_BYTES,
    check_interval: float = SIZE_CHECK_INTERVAL_S,
    keep: int = KEEP_PER_PORT,
) -> None:
    """Send this process's descriptors 1 and 2 to ``path`` (child side).

    After this call every write -- `print`, the stdlib logging handlers
    mlx-vlm and uvicorn build next, and anything native code writes straight to
    a descriptor -- lands in ``path``, line by line.

    A daemon thread watches the file and rolls it past ``max_bytes`` so an
    endless child cannot fill the disk; pass ``max_bytes=0`` to disable it.

    Raises:
        OSError: If the file cannot be opened. The caller keeps the server
            running without capture rather than failing the spawn.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _bind_stdio(path)
    if max_bytes > 0:
        threading.Thread(
            target=_size_guard,
            args=(path, max_bytes, check_interval, keep),
            name="mlx-child-log-guard",
            daemon=True,
        ).start()


def _bind_stdio(path: Path) -> None:
    """Point descriptors 1 and 2 at ``path`` and rebind the Python streams."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None:
                stream.flush()
        except Exception:
            # A stream that cannot be flushed is one we are replacing anyway.
            pass
    handle = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.dup2(handle, 1)
        os.dup2(handle, 2)
    finally:
        os.close(handle)
    # Line buffering: a child killed mid-load must not take its last lines
    # away in an 8 KiB block buffer. `closefd=False` keeps 1 and 2 alive when
    # a wrapper is replaced by the next roll.
    sys.stdout = open(1, "w", buffering=1, encoding="utf-8", errors="replace", closefd=False)
    sys.stderr = open(2, "w", buffering=1, encoding="utf-8", errors="replace", closefd=False)


def _size_guard(path: Path, max_bytes: int, check_interval: float, keep: int) -> None:
    """Roll ``path`` whenever it outgrows ``max_bytes`` (daemon thread).

    Rolling is a rename plus a fresh open: descriptors 1 and 2 keep writing
    into the renamed file until they are pointed at the new one, so no line is
    lost in between.
    """
    while True:
        time.sleep(check_interval)
        try:
            if path.stat().st_size <= max_bytes:
                continue
            roll_child_log(path, keep=keep)
            _bind_stdio(path)
        except OSError as exc:
            # Windows refuses to rename an open file; MLX only ever runs on
            # Apple Silicon, so this costs the cap and nothing else. Recorded
            # once and then given up on: retrying every interval would fill
            # the very file it failed to bound.
            child_warning(f"child log size cap disabled: {type(exc).__name__}: {exc}")
            return


def write_child_record(level: str, message: str) -> None:
    """Write one levelled line to the captured stderr, as the backend would.

    Deliberately not the stdlib `logging` module: this runs before mlx-vlm's
    own `logging.basicConfig`, `src.core.logging` must not be imported in the
    child (it would open a SECOND writer on `backend.log`, with two processes
    rotating one file), and `sys.stderr` has to be resolved at call time
    because a roll replaces it.
    """
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    try:
        stream = sys.stderr
        if stream is None:
            return
        stream.write(f"[{level}] {stamp}Z - {_CHILD_LOGGER_NAME} - {message}\n")
        stream.flush()
    except Exception:
        # The child's own diagnostics must never be what kills the child.
        pass


def child_warning(message: str) -> None:
    """Record a degraded-but-running condition from inside the child."""
    write_child_record("WARNING", message)
