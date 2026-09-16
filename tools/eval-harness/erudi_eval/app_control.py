"""Launch and quit the installed app. The harness only ever signals the app's main process."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

from .discovery import DiscoveryContext, ProcInfo, classify_snapshot
from .layout import AppLayout


def running_erudi(procs: list[ProcInfo], layout: AppLayout, api_port: int, exclude: set[int]) -> dict[int, str]:
    """{pid: category} of Erudi processes of this installation already running."""
    ctx = DiscoveryContext(install_dirs=layout.install_dirs, data_root=str(layout.data_root), api_port=api_port, exclude_pids=frozenset(exclude))
    _, classified = classify_snapshot(procs, ctx)
    return {pid: c.category for pid, c in classified.items()}


def clean_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """The harness runs inside a uv virtualenv: do not leak it into the app (its frozen backend, pg tools)."""
    env = dict(os.environ if env is None else env)
    venv = env.pop("VIRTUAL_ENV", None)
    for key in list(env):
        if key.startswith(("UV_", "PYTHON")) or key == "__PYVENV_LAUNCHER__":
            env.pop(key)
    if venv and "PATH" in env:
        env["PATH"] = os.pathsep.join(p for p in env["PATH"].split(os.pathsep) if not p.startswith(venv))
    return env


def launch(layout: AppLayout, cdp_port: int, console_log: Path) -> subprocess.Popen:
    """Exec the main binary directly (not `open -a`) so the harness knows the main PID."""
    if not layout.main_exe or not Path(layout.main_exe).exists():
        raise FileNotFoundError(f"app binary not found: {layout.main_exe}")
    console_log.parent.mkdir(parents=True, exist_ok=True)
    fh = open(console_log, "ab")
    kwargs: dict = {"stdout": fh, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL, "env": clean_env()}
    if sys.platform.startswith("win"):
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # Ctrl+C in the harness does not reach the app
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen([str(layout.main_exe), f"--remote-debugging-port={cdp_port}"], **kwargs)


def quit_command(layout: AppLayout, main_pid: int, main_exe: str) -> list[str] | None:
    """Graceful quit command, or None when a signal to main_pid is the graceful path.

    macOS: AppleScript quit, but only when the running main binary is the bundle's
    (otherwise `quit app` could target, or prompt for, a different app).
    Windows: taskkill without /F on the PID. Linux: SIGTERM to the PID (equivalent to
    `pkill -TERM -f Erudi` without matching unrelated command lines).
    """
    if layout.os_name == "darwin" and layout.app_path and layout.app_path.suffix == ".app":
        bundle_exe = str(layout.app_path / "Contents" / "MacOS") + "/"
        if main_exe.startswith(bundle_exe):
            return ["osascript", "-e", f'quit app "{layout.app_path.stem}"']
    if layout.os_name == "windows":
        return ["taskkill", "/PID", str(main_pid)]
    return None


def graceful_quit(layout: AppLayout, main_pid: int, main_exe: str) -> str:
    cmd = quit_command(layout, main_pid, main_exe)
    if cmd:
        subprocess.run(cmd, capture_output=True, timeout=30)
        return " ".join(cmd)
    os.kill(main_pid, signal.SIGTERM)
    return f"SIGTERM {main_pid}"
