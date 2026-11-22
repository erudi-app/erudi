"""Windows console-window suppression for backend child processes (#175).

The packaged Windows backend is a console-subsystem executable whose own window
is hidden by the Electron launcher. Its children are console programs too
(``llama-server.exe``, ``pg_dump.exe``): on Windows a
console child either inherits the parent's console or, when the parent has no
inheritable console (e.g. it was spawned detached), allocates its OWN visible
console window — the terminal flashes seen in QA on every model load and boot.

``CREATE_NO_WINDOW`` gives the child a console with no window at all,
independent of the parent's console state, while keeping pipes fully
functional. Every backend-owned spawn of a console executable must pass
``creationflags=hidden_console_creationflags()``.
"""

from __future__ import annotations

import platform
import subprocess
from typing import Optional

# CREATE_NO_WINDOW's documented value; the constant only exists in the
# subprocess module on Windows, so tests on POSIX fall back to the literal.
_CREATE_NO_WINDOW = 0x08000000


def signed_exit_code(code: Optional[int]) -> Optional[int]:
    """A child's exit status as the signed value people recognise.

    Windows hands back the process exit status as an unsigned 32-bit value, so
    ``-1`` reads ``4294967295`` and a crash status ``0xC0000005`` reads
    ``3221225477``. On Windows a status past 2**31 is folded back to its signed
    form (``-1``, ``-1073741819``); POSIX statuses are already signed and pass
    through, as does ``None`` (still running, or unknown).
    """
    if code is not None and platform.system() == "Windows" and code >= 0x80000000:
        return code - 0x100000000
    return code


def hidden_console_creationflags() -> int:
    """``creationflags`` so a spawned console child never opens a terminal window.

    Returns ``CREATE_NO_WINDOW`` on Windows and ``0`` elsewhere — ``Popen``
    rejects a non-zero ``creationflags`` on POSIX, so the return value is safe
    to pass unconditionally on every platform.
    """
    if platform.system() == "Windows":
        return getattr(subprocess, "CREATE_NO_WINDOW", _CREATE_NO_WINDOW)
    return 0
