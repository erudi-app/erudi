"""Cross-platform launcher for the Erudi FastAPI backend.

This module boots the FastAPI application in a background thread while
emitting newline-delimited JSON events for the Electron frontend. The
launcher is designed to run identically in development (editable source
tree) and in PyInstaller bundles targeting macOS, Windows, and Linux across
CUDA, MLX, and CPU builds.

**Lifecycle events (newline-delimited JSON to stdout):**
    - {"event": "starting", "arch": "...", "mode": "dev|prod", "data_path": "...", "port": N, "first_run": bool}
    - {"event": "phase", "phase": "preparing_database|recovering_database|running_migrations|loading_catalog"}
    - {"event": "ready", "port": N}
    - {"event": "shutdown"}
    - {"event": "startup_error", "code": "ERROR_CODE", "message": "..."}

    Every event also carries {"ts": "<UTC ISO-8601 ms, Z>"} (stamped by
    src.launcher.events.emit_event) so it can be correlated with the backend
    and Electron logs.

**Supported error codes:**
    - NO_PORT_AVAILABLE: Every candidate port in the scan range is busy
    - CRASH_BEFORE_READY: Backend thread exited before binding port
    - PORT_TIMEOUT: Server did not bind within the startup window
    - IMPORT_ERROR: Failed to import FastAPI application
    - DATA_PREP_ERROR: Failed to prepare data directories
    - UNEXPECTED_ERROR: Unhandled exception in server thread
    - POLLING_ERROR: Unhandled exception in startup polling loop

**Key responsibilities:**
    * Parse command-line arguments (--port) for flexible port binding.
    * Normalize environment variables for deterministic third-party libs (TOKENIZERS_PARALLELISM, etc.).
    * Configure asyncio selector policy on Windows for library compatibility.
    * Redirect data/log directories to user-writable locations on bundled builds.
    * Preserve macOS symlink behavior while adopting OS-appropriate folders on Windows/Linux.
    * Initialize multiprocessing spawn settings before importing heavy modules.
    * Guard startup with readiness polling (127.0.0.1:PORT), crash detection, and a
      first-run-aware timeout (longer on first boot, which pays a one-time initdb).
    * Support all build variants (CPU, CUDA, MLX) transparently via ERUDI_BUILD_VARIANT env var.
    * Watch stdin for EOF when ERUDI_WATCH_STDIN=1 (Electron's Windows graceful-quit
      signal, since there is no SIGTERM to relay) and turn it into a clean uvicorn shutdown.
    * Watch the parent process for death (SIGKILLed Electron main leaves no one to
      kill us, #224) and turn a reparenting into the same clean shutdown. Opt out
      with ERUDI_NO_PARENT_WATCHDOG=1 for deliberately detached runs (nohup/setsid).

**Usage:**
    Development:
        PYTHONPATH=backend python backend/run.py --port 8000

    Packaged (PyInstaller):
        ./backend --port 8000

    From Electron (frontend/src/main.js):
        spawn('./backend/backend', ['--port', '8000'])
"""

from __future__ import annotations

import asyncio
import os
import platform
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    import uvicorn
    from src.launcher import RuntimePaths


import argparse
import traceback

from src.launcher.events import emit_event, emit_phase

STARTUP_TIMEOUT_SECONDS = 120
# First boot also pays a one-time embedded-Postgres initdb (plus a cold disk
# cache / AV first-scan of the freshly extracted bundle), so allow much longer
# before declaring failure. The frontend mirrors this budget.
FIRST_RUN_TIMEOUT_SECONDS = 300
READINESS_POLL_SECONDS = 0.25

# Erudi's canonical port. 27182 = the leading digits of Euler's number e
# (2.7182…) — a wink for an app built for erudites. Practically, it's a good
# choice on every OS: IANA-unassigned, it sits below every OS's ephemeral range
# (Linux 32768–60999, Windows/macOS 49152–65535, plus Windows Hyper-V/WSL
# exclusions live inside that range), and it's clear of the crowded dev/LLM
# defaults Erudi users are likely to run alongside it (Ollama 11434, LM Studio
# 1234, vLLM 8000, llama.cpp/Tomcat 8080). The renderer never assumes this value:
# it learns the *actual* bound port from the `starting`/`ready` events. This is
# only the try-first default; find_available_port() scans forward on collision.
CANONICAL_PORT = 27182
# The backend scans 27182–27199 and stops short of 27200, which is where the
# inference pools live (llama.cpp 27200–27299, MLX 27300–27399) — so the three
# local servers can never fight over a port. Erudi's whole footprint is 271xx–273xx.
PORT_SCAN_COUNT = 18

# Parent-death watchdog cadence (#224). Cheap (one getppid()/psutil check per
# tick), so a few seconds keeps orphan lifetime short without busy-polling.
PARENT_POLL_SECONDS = 2.0

# Bound on uvicorn's wait for in-flight requests once should_exit is set.
# uvicorn's default (None) waits FOREVER, which is exactly the observed orphan:
# a SIGKILLed Electron mid-generation left the backend generating to completion
# (#224). Bounded, every should_exit path (parent watchdog, SIGTERM relay,
# stdin EOF) cancels the leftover tasks and still runs the lifespan shutdown
# (inference child terminated, embedded Postgres stopped) promptly. For a
# desktop app a quit is a quit: nobody is reading a 10s-stale stream anyway.
GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS = 10


def log_startup_failure(code: str, message: str, exc: "BaseException | None" = None) -> None:
    """Write a startup failure to ``backend.log``, with the traceback when there is one.

    The JSON ``startup_error`` event on stdout tells Electron WHAT failed; this
    record tells a maintainer WHY, in the file the Diagnostics panel reads.
    The logger is imported lazily because it needs the runtime paths, and a
    failure before they exist (``DATA_PREP_ERROR``) falls back to stderr, which
    Electron copies into its own log with the same ``[ERROR]`` marker.
    """
    try:
        from src.core.logging import logger
    except Exception:
        print(f"[ERROR] startup_error {code}: {message}", file=sys.stderr, flush=True)
        if exc is not None:
            traceback.print_exception(exc, file=sys.stderr)
        return
    logger.error(f"startup_error {code}: {message}", exc_info=exc)


def parse_args():
    """Parse command-line arguments for launcher."""
    parser = argparse.ArgumentParser(description="Erudi backend launcher")
    parser.add_argument(
        "--port",
        type=int,
        default=CANONICAL_PORT,
        help=f"Port to bind FastAPI server (default: {CANONICAL_PORT})",
    )
    return parser.parse_args()


def configure_library_env() -> None:
    """Tame noisy third-party libraries, and close every known egress switch.

    Two different rules apply here, and the difference matters.

    Performance knobs use ``setdefault``: an operator who pins a thread count
    means it, and overriding them would be rude.

    Anything that can send data off the machine is **hard-set**, because these
    variables are not ours to negotiate. The backend inherits the user's whole
    environment (the Electron main process passes ``process.env`` through), and
    ``load_dotenv()`` walks up from the working directory, so a value can arrive
    from a shell profile or a stray ``.env`` several levels up without anyone
    intending it. ``setdefault`` would honour that value; a local-first app
    must not.

    ``LANGSMITH_TRACING`` / ``LANGCHAIN_TRACING_V2`` are the serious one.
    ``langsmith`` is a transitive dependency of langchain and ships inside the
    frozen build, and either variable is enough to make LangChain POST the full
    system prompt, the knowledge-base excerpts, the user's question and the
    model's answer to ``api.smith.langchain.com``. Nothing in the UI would say
    so. The legacy name is set alongside the current one because old shell
    profiles outlive library renames.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("PYTHONNOUSERSITE", "1")

    # Egress switches: assigned, never defaulted. See the docstring.
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"  # local-first egress hygiene (#109)

    # One deliberate escape hatch, for a contributor debugging the agent layer.
    # It has to be typed out in full, which is the point: turning it on means
    # accepting that conversations leave the machine.
    if os.getenv("ERUDI_ALLOW_LANGSMITH_TRACING") != "1":
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"


def set_event_loop_policy() -> None:
    """Force selector event loop on Windows for broader library compatibility."""
    if platform.system() == "Windows":
        from asyncio import WindowsSelectorEventLoopPolicy

        asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())


def configure_stdio() -> None:
    """Force UTF-8 (never-raising) line-buffered stdout/stderr.

    The frozen interpreter ignores PYTHONUTF8 (PyInstaller pre-initializes
    CPython), so without this the streams inherit the locale code page
    (cp1252 on Windows) and any Unicode character in a log line blows up the
    console log handler, which writes to sys.stdout (#168). errors="replace"
    is the last-resort net: a log write can degrade a character to '?' but can
    never raise. Line buffering keeps JSON lifecycle events flushed promptly.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")
            except Exception:
                pass  # exotic stream: keep whatever it supports


def is_frozen() -> bool:
    """Return True when running from a PyInstaller bundle."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def backend_root_dir() -> Path:
    """Resolve the backend root directory for both dev and bundled modes."""
    if is_frozen():
        exe_dir = Path(sys.executable).resolve().parent
        bundle_dir = Path(getattr(sys, "_MEIPASS", exe_dir))

        candidates = [
            exe_dir / "backend",
            bundle_dir / "backend",
            bundle_dir,
        ]
        for candidate in candidates:
            # PyInstaller 6.x onedir: Python src is compiled into PYZ (no src/ dir),
            # but data files like artifacts/ are present in _internal/ (bundle_dir).
            if (candidate / "src").exists() or (candidate / "artifacts").exists():
                return candidate
        # Last resort: prefer bundle_dir (_internal/) over exe_dir so that
        # artifact paths resolve correctly in PyInstaller 6.x onedir builds.
        return bundle_dir if bundle_dir != exe_dir else exe_dir

    return Path(__file__).resolve().parent


def ensure_backend_on_path(backend_dir: Path) -> None:
    """Insert the backend directory on sys.path if needed."""
    backend_str = str(backend_dir)
    if backend_str not in sys.path:
        sys.path.insert(0, backend_str)


def ensure_working_directory(backend_dir: Path) -> None:
    """Switch the process working directory to the backend root.

    Best-effort: every path the backend opens is absolute (runtime_paths), so
    a chdir that fails changes nothing that matters.
    """
    try:
        os.chdir(backend_dir)
    except Exception:
        pass


def force_mp_spawn() -> None:
    """Configure multiprocessing to use spawn start method safely."""
    try:
        import multiprocessing as mp

        mp.freeze_support()
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        try:
            import torch.multiprocessing as tmp

            tmp.set_start_method("spawn", force=True)
        except Exception:
            pass  # torch absent (CPU build without it) or already configured
    except Exception:
        # Expected only on exotic interpreters; on the ones we ship the MLX
        # child would then fail to spawn, and that failure is logged by the
        # engine with its own record.
        pass


def compute_first_run(data_dir: Path | str) -> bool:
    """True when the embedded Postgres cluster has not been initialized yet.

    The canonical first-run signal for the whole app is the absence of
    ``<data_dir>/postgres/PG_VERSION`` — pgserver writes it once initdb
    completes. Used to widen the startup budget and to let the frontend show a
    "first launch may take longer" hint.
    """
    return not (Path(data_dir) / "postgres" / "PG_VERSION").exists()


def startup_timeout_seconds(first_run: bool) -> int:
    """Boot budget: longer on first run (one-time initdb), tighter afterwards."""
    return FIRST_RUN_TIMEOUT_SECONDS if first_run else STARTUP_TIMEOUT_SECONDS


def port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    """Return True when a TCP connection to the host:port succeeds."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def find_available_port(start_port: int, host: str, count: int = PORT_SCAN_COUNT) -> int | None:
    """Find a free port in [start_port, start_port + count). Returns None if all busy."""
    for port in range(start_port, start_port + count):
        if not port_open(host, port):
            return port
    return None


def kill_port_process(port: int) -> bool:
    """Attempt to kill process on given port. Returns True if successful."""
    import subprocess

    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"], capture_output=True, text=True, timeout=2
        )
        pid = result.stdout.strip()
        if pid:
            subprocess.run(["kill", "-9", pid], timeout=2)
            time.sleep(0.5)
            return True
    except Exception as exc:
        # lsof/kill absent (Windows) or timed out: the caller reports
        # NO_PORT_AVAILABLE, which names the port.
        print(f"[WARNING] could not free port {port}: {exc}", file=sys.stderr, flush=True)
    return False


def run_server(server: "uvicorn.Server") -> None:
    """Run the uvicorn server in the current thread; surface unexpected failures.

    uvicorn skips installing its own signal handlers when running outside the
    main thread — main() relays SIGTERM/SIGINT (the Electron quit path) via
    `server.should_exit` so the FastAPI lifespan shutdown actually runs
    (checkpointer close, embedded PostgreSQL stop) before the process exits.
    """
    try:
        server.run()
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # pragma: no cover - defensive
        log_startup_failure("UNEXPECTED_ERROR", f"Server thread crashed: {exc}", exc)
        emit_event(
            {
                "event": "startup_error",
                "code": "UNEXPECTED_ERROR",
                "message": f"Server thread crashed: {exc}",
            }
        )
        sys.exit(1)


# How often the Windows stdin watcher checks the pipe for EOF. Shutdown
# latency is bounded by this and Electron allows 8s of grace, so a quarter
# second is far inside the budget while staying invisible on the CPU.
_STDIN_POLL_SECONDS = 0.25


def stdin_watch_enabled() -> bool:
    """True when Electron asked the launcher to watch stdin for shutdown (opt-in).

    Electron sets ERUDI_WATCH_STDIN=1 when it spawns the backend. Leaving it
    unset (dev runs, the subprocess launcher tests) keeps stdin untouched, so
    the watcher is a no-op there.
    """
    return os.environ.get("ERUDI_WATCH_STDIN") == "1"


def _win_wait_for_stdin_eof(fd: int) -> None:
    """Windows: wait for the stdin pipe to close WITHOUT parking in the CRT.

    A blocking ``os.read`` here deadlocks the process loader (#321). ``os.read``
    enters the UCRT's ``_read()``, which takes the per-fd lock and then parks in
    ``ReadFile`` for the entire life of the process; while this daemon thread is
    parked there, ANY off-main-thread native import that pulls the OpenBLAS
    chain (scipy/sklearn via transformers or sentence_transformers) blocks
    forever inside ``LoadLibraryExW``. Bisected to this single env var: with
    ERUDI_WATCH_STDIN unset the same frozen bundle imports fine from a worker
    thread; with it set, the import never returns (identical frames, zero CPU).

    So on Windows the wait is a poll on the OS handle instead: PeekNamedPipe
    reports the write end closing as ERROR_BROKEN_PIPE, which is our EOF, and
    the CRT is never entered. Any bytes the parent sends are drained through
    ReadFile on the same handle for the same reason. Electron never writes to
    the backend's stdin, so the drain is defensive only.
    """
    import ctypes
    from ctypes import wintypes

    ERROR_BROKEN_PIPE = 109
    ERROR_PIPE_NOT_CONNECTED = 233

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    msvcrt = __import__("msvcrt")
    handle = msvcrt.get_osfhandle(fd)

    avail = wintypes.DWORD()
    read = wintypes.DWORD()
    buf = ctypes.create_string_buffer(4096)

    while True:
        ok = kernel32.PeekNamedPipe(
            wintypes.HANDLE(handle), None, 0, None, ctypes.byref(avail), None
        )
        if not ok:
            err = ctypes.get_last_error()
            if err in (ERROR_BROKEN_PIPE, ERROR_PIPE_NOT_CONNECTED):
                return  # parent closed the pipe: this is EOF
            # Not a pipe (console, file, redirected handle): there is no EOF
            # to watch for, so retire rather than fall back to the blocking read
            # this function exists to avoid. `_watch` catches this and returns.
            #
            # Retiring is safe because it is not the only shutdown path: the
            # parent-death watchdog (#224) pins the parent through psutil on its
            # own daemon thread, with no CRT involvement, and covers exactly the
            # case this one drops. The two are complementary; do not "fix" this
            # raise by reinstating os.read.
            raise OSError(err, "PeekNamedPipe failed on stdin")
        if avail.value:
            kernel32.ReadFile(
                wintypes.HANDLE(handle),
                buf,
                min(avail.value, len(buf)),
                ctypes.byref(read),
                None,
            )
            continue  # data, not EOF - keep watching
        time.sleep(_STDIN_POLL_SECONDS)


def start_stdin_eof_watcher(server: "uvicorn.Server") -> threading.Thread:
    """Request a graceful shutdown when the controlling stdin pipe closes.

    Windows has no SIGTERM to relay, so Electron signals a clean quit by
    closing the backend's stdin. EOF on that pipe flips uvicorn's exit flag,
    exactly like the POSIX signal relay, so the FastAPI lifespan shutdown
    (checkpointer close, embedded PostgreSQL stop) runs before the process
    exits.

    Two constraints shape how the wait is implemented, and they differ per
    platform:

    * It must never touch ``sys.stdin.buffer``. A blocking read on the shared
      BufferedReader holds that object's lock for as long as this daemon thread
      is parked; if the interpreter finalizes while it is still parked there
      (any exit path that is not the graceful stdin-EOF one -- e.g. a
      PORT_TIMEOUT sys.exit), CPython's buffered-IO finalization tries to
      acquire the same lock and aborts the process (_enter_buffered_busy ->
      exit 0xC0000005). Both platforms therefore work off the RAW fd (#283).
    * On Windows it must not park inside the CRT either. See
      ``_win_wait_for_stdin_eof`` -- a blocking ``os.read`` there deadlocks
      every off-main-thread native import in the frozen build (#321/#313).

    POSIX keeps the plain blocking ``os.read``: there is no loader lock to
    contend with and the CRT is not in the picture.

    Runs on a daemon thread and swallows any stdin error: a broken or absent
    stdin must never crash the launcher.
    """

    def _watch() -> None:
        stdin = sys.stdin
        if stdin is None:
            return  # no controlling stdin: nothing to watch
        try:
            fd = stdin.fileno()
        except Exception:
            return  # stdin closed/detached or has no real fd: leave it to signals
        try:
            if platform.system() == "Windows":
                _win_wait_for_stdin_eof(fd)
            else:
                while os.read(fd, 4096):  # blocks; b"" at EOF ends the loop
                    pass
        except Exception:
            return  # broken/absent stdin: nothing to watch, leave shutdown to signals
        try:
            from src.core.logging import logger

            logger.info("Parent closed stdin; requesting graceful shutdown")
        except Exception:
            pass  # logging must never gate the shutdown flag below
        server.should_exit = True

    thread = threading.Thread(target=_watch, name="stdin-eof-watcher", daemon=True)
    thread.start()
    return thread


def parent_watchdog_enabled() -> bool:
    """True unless ERUDI_NO_PARENT_WATCHDOG=1 opted the watchdog out.

    On by default everywhere, dev runs included: the predicate is a parent
    CHANGE (reparenting), and a dev's shell staying alive is the normal case,
    so plain ``python run.py`` from a terminal is unaffected. The opt-out is
    for deliberately detached runs (nohup/setsid, daemonized QA harnesses)
    where reparenting is expected and survival is intended (#224).
    """
    return os.environ.get("ERUDI_NO_PARENT_WATCHDOG") != "1"


def _parent_alive_probe(initial_ppid: int):
    """Build a zero-argument callable reporting whether the original parent lives.

    POSIX: the kernel rewrites ppid on parent death (reparenting to pid 1 or
    the nearest subreaper), so ``os.getppid() == initial_ppid`` is exact and
    dependency-free.

    Windows: ppid is never rewritten (a dead parent's pid may even be reused),
    so the parent is pinned through psutil (already a base dependency), whose
    ``is_running`` guards identity with the process create time. If the parent
    is already gone when the probe is built, it reports dead immediately; if
    psutil is unavailable for any reason, the probe reports alive forever and
    shutdown stays with the stdin-EOF watcher (#216) / taskkill.
    """
    if os.name == "posix":
        return lambda: os.getppid() == initial_ppid

    try:
        import psutil
    except Exception:
        return lambda: True  # no probe available: never force a shutdown

    try:
        parent = psutil.Process(initial_ppid)
    except psutil.NoSuchProcess:
        return lambda: False  # parent died before the watchdog started
    except Exception:
        return lambda: True
    return parent.is_running


def start_parent_watchdog(
    server: "uvicorn.Server",
    initial_ppid: int | None = None,
    poll_seconds: float = PARENT_POLL_SECONDS,
    parent_alive=None,
) -> threading.Thread:
    """Shut down cleanly when the parent process dies (#224).

    A SIGKILLed Electron main (crash, force-quit, power event) leaves no one
    to kill the backend: it survives orphaned with the embedded Postgres,
    holding the port, until an accidental stdout write to the dead parent's
    pipe SIGPIPEs it -- or forever. This watchdog detects parent death
    DIRECTLY: it records the parent at startup and polls; on a change
    (reparenting) it flips uvicorn's exit flag -- the exact same clean
    shutdown path as the SIGTERM relay and the stdin-EOF watcher -- so the
    FastAPI lifespan shutdown runs (inference child terminated, embedded
    Postgres stopped) and the port is released.

    The check runs BEFORE the first sleep so a parent that died during the
    long boot imports still triggers immediately. ``parent_alive`` is
    injectable for tests; production builds it from ``initial_ppid``.
    Runs on a daemon thread; a broken probe exits the thread without ever
    forcing a shutdown (signals/stdin-EOF remain in charge).
    """
    if initial_ppid is None:
        initial_ppid = os.getppid()
    if parent_alive is None:
        parent_alive = _parent_alive_probe(initial_ppid)

    def _watch() -> None:
        while True:
            try:
                alive = parent_alive()
            except Exception:
                return  # probe broke: never crash the launcher or force an exit
            if not alive:
                try:
                    from src.core.logging import logger

                    logger.info("Parent process died; requesting graceful shutdown")
                except Exception:
                    pass  # logging must never gate the shutdown flag below
                server.should_exit = True
                return
            time.sleep(poll_seconds)

    thread = threading.Thread(target=_watch, name="parent-death-watchdog", daemon=True)
    thread.start()
    return thread


def main() -> None:
    """Launch the backend, supervising readiness and emitting lifecycle events."""
    # Recorded FIRST: the reference parent for the death watchdog. Captured
    # before the heavy imports below so a parent lost during boot is still a
    # detectable ppid CHANGE rather than the recorded baseline (#224).
    initial_ppid = os.getppid()

    args = parse_args()
    requested_port = args.port
    host = "127.0.0.1"

    configure_library_env()
    set_event_loop_policy()
    configure_stdio()

    backend_dir = backend_root_dir()
    ensure_backend_on_path(backend_dir)
    ensure_working_directory(backend_dir)

    mode = "prod" if is_frozen() else "dev"
    try:
        from src.launcher import get_runtime_paths, initialize_runtime_paths

        runtime_paths: "RuntimePaths"
        try:
            runtime_paths = initialize_runtime_paths(
                mode=mode,
                backend_root=backend_dir,
                packaged_data_dir=backend_dir / "data",
            )
        except ValueError:
            runtime_paths = get_runtime_paths()
    except Exception as exc:
        log_startup_failure("DATA_PREP_ERROR", f"Failed to prepare data directories: {exc}", exc)
        emit_event(
            {
                "event": "startup_error",
                "code": "DATA_PREP_ERROR",
                "message": f"Failed to prepare data directories: {exc}",
            }
        )
        sys.exit(1)
    else:
        data_dir = runtime_paths.data_dir

    # Windows-only: bind Postgres/llama-server's lifetime to this process at
    # the kernel level, before either is ever spawned. Unlike the parent-death
    # watchdog below, this does not depend on this process staying alive to
    # react -- see windows_job.py for why that gap matters (#341).
    #
    # MUST run after initialize_runtime_paths() above, not before: importing
    # this module pulls in src.core.logging, whose module-level logger setup
    # calls ensure_runtime_paths_initialized() as a side effect. Doing that
    # BEFORE the real initialize_runtime_paths(mode="prod", ...) call above
    # got a chance to run silently locked in the wrong dev-mode data
    # directory (backend_root/"data", i.e. inside the frozen bundle itself)
    # for the rest of the process -- the later legitimate prod-mode call then
    # raises ValueError ("already initialized") and falls back to that wrong
    # path via get_runtime_paths(), so every model download and all app data
    # landed inside the installed app's own folder instead of the user data
    # dir, silently, with "mode": "prod" still printed in the ready event.
    # Reproduced on a real packaged build during QA; do not move this above
    # the runtime-paths block again.
    if platform.system() == "Windows":
        from src.launcher.windows_job import bind_children_to_this_process

        bind_children_to_this_process()

    force_mp_spawn()

    try:
        from src.main import app as fastapi_app
    except Exception as exc:  # pragma: no cover - defensive
        log_startup_failure("IMPORT_ERROR", f"Failed to import FastAPI application: {exc}", exc)
        emit_event(
            {
                "event": "startup_error",
                "code": "IMPORT_ERROR",
                "message": f"Failed to import FastAPI application: {exc}",
            }
        )
        sys.exit(1)

    # Find available port
    port = find_available_port(requested_port, host)

    if port is None:
        # Every candidate is busy — last resort, try to reclaim the middle of the
        # scan window. (POSIX-only via lsof/kill; a no-op on Windows, where this
        # branch is effectively unreachable since 18 consecutive busy ports is
        # absurd. NO_PORT_AVAILABLE is a transient code the frontend auto-retries.)
        fallback_port = requested_port + PORT_SCAN_COUNT // 2
        if kill_port_process(fallback_port):
            time.sleep(1)
            if not port_open(host, fallback_port):
                port = fallback_port

        if port is None:
            message = (
                f"Ports {requested_port}-{requested_port + PORT_SCAN_COUNT - 1} "
                f"all busy, failed to free {fallback_port}"
            )
            log_startup_failure("NO_PORT_AVAILABLE", message)
            emit_event({"event": "startup_error", "code": "NO_PORT_AVAILABLE", "message": message})
            sys.exit(1)

    first_run = compute_first_run(data_dir)
    emit_event(
        {
            "event": "starting",
            "arch": platform.machine(),
            "mode": mode,
            "data_path": str(data_dir),
            "port": port,
            "first_run": first_run,
        }
    )

    import uvicorn

    # Let the FastAPI lifespan emit startup-progress phases on the same stdout
    # stream (same process). Absent (e.g. plain uvicorn in dev), the lifespan
    # simply skips phase emission.
    fastapi_app.state.emit_phase = emit_phase

    server = uvicorn.Server(
        # access_log=False: uvicorn's unstructured per-request lines are
        # replaced by the request-logging middleware (method, path, status,
        # duration, request id — see src.core.api.RequestLoggingMiddleware).
        # timeout_graceful_shutdown: bound the wait on in-flight requests once
        # should_exit is set (uvicorn's None default waits forever behind a
        # long generation), so every exit path still reaches the lifespan
        # shutdown promptly (#224).
        uvicorn.Config(
            fastapi_app,
            host=host,
            port=port,
            log_level="info",
            workers=1,
            access_log=False,
            timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS,
        )
    )

    def _request_graceful_shutdown(signum: int, frame: object) -> None:
        # Relay to uvicorn's exit flag: the server thread notices it, drains
        # requests, runs the lifespan shutdown (checkpointer → embedded
        # PostgreSQL), then run_server returns and the join below unblocks.
        server.should_exit = True

    signal.signal(signal.SIGTERM, _request_graceful_shutdown)
    signal.signal(signal.SIGINT, _request_graceful_shutdown)

    # Windows has no SIGTERM: Electron signals a graceful quit by closing our
    # stdin instead (opt-in via ERUDI_WATCH_STDIN). Watch for that EOF so the
    # lifespan shutdown still runs and the embedded Postgres stops cleanly (#216).
    if stdin_watch_enabled():
        start_stdin_eof_watcher(server)

    # Parent-death watchdog (#224): a SIGKILLed Electron main never runs its
    # cleanup, so the backend must notice the reparenting itself and take the
    # same clean shutdown path as SIGTERM/stdin-EOF. On by default (the
    # predicate is a ppid CHANGE, so a dev's living shell never trips it);
    # ERUDI_NO_PARENT_WATCHDOG=1 opts deliberately detached runs out.
    if parent_watchdog_enabled():
        start_parent_watchdog(server, initial_ppid=initial_ppid)

    server_thread = threading.Thread(target=run_server, args=(server,), daemon=True)
    server_thread.start()

    deadline = time.time() + startup_timeout_seconds(first_run)
    try:
        while time.time() < deadline:
            if port_open(host, port):
                emit_event({"event": "ready", "port": port})
                server_thread.join(timeout=1.0)
                while server_thread.is_alive():
                    time.sleep(1.0)
                    server_thread.join(timeout=1.0)
                emit_event({"event": "shutdown"})
                break

            if not server_thread.is_alive():
                # Not logged here: the cause already is, with its traceback,
                # by the lifespan ("Startup failed") or by run_server
                # (UNEXPECTED_ERROR). This event only tells Electron.
                emit_event(
                    {
                        "event": "startup_error",
                        "code": "CRASH_BEFORE_READY",
                        "message": "Backend thread exited before binding the port",
                    }
                )
                sys.exit(1)

            time.sleep(READINESS_POLL_SECONDS)
        else:
            log_startup_failure(
                "PORT_TIMEOUT",
                f"Server did not bind port {port} within "
                f"{startup_timeout_seconds(first_run)}s (first_run={first_run})",
            )
            emit_event(
                {
                    "event": "startup_error",
                    "code": "PORT_TIMEOUT",
                    "message": "Server did not bind in time",
                }
            )
            sys.exit(1)
    except KeyboardInterrupt:
        emit_event({"event": "shutdown"})
        sys.exit(0)
    except Exception as exc:  # pragma: no cover - defensive
        log_startup_failure("POLLING_ERROR", f"Startup polling loop crashed: {exc}", exc)
        emit_event(
            {
                "event": "startup_error",
                "code": "POLLING_ERROR",
                "message": f"Startup polling loop crashed: {exc}",
            }
        )
        sys.exit(1)


if __name__ == "__main__":
    # MUST run before anything else (especially argparse): in PyInstaller
    # bundles the frozen exe is re-invoked as multiprocessing children and the
    # resource tracker. freeze_support() intercepts those relaunches and exits,
    # so our --port argparse never sees their internal args
    # ("-B -S -I -c from multiprocessing...") and the MLX inference subprocess
    # (multiprocessing.Process) can actually spawn.
    import multiprocessing

    multiprocessing.freeze_support()

    # NOTE: no logging.basicConfig here. A root handler would duplicate every
    # "erudi" log line on stdout (the app logger already owns a console
    # handler and propagates to root), polluting the JSON event stream the
    # Electron main process parses. Third-party WARNING+ records without
    # handlers still surface via logging.lastResort (stderr).
    main()
