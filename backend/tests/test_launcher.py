"""
Tests for Erudi backend launcher (run.py): argument parsing and JSON event emission.
"""

import sys
import json
import subprocess
import os
import platform
import pytest
from pathlib import Path

LAUNCHER_PATH = Path(__file__).parent.parent / "run.py"


@pytest.mark.parametrize("port", [8000, 9000, 12345])
def test_argparse_port(monkeypatch, port):
    import run

    monkeypatch.setattr(sys, "argv", ["run.py", "--port", str(port)])
    args = run.parse_args()
    assert args.port == port


@pytest.mark.unit
def test_default_port_is_canonical_27182(monkeypatch):
    # Erudi's canonical port: the leading digits of e (2.7182…).
    import run

    monkeypatch.setattr(sys, "argv", ["run.py"])
    assert run.parse_args().port == 27182
    assert run.CANONICAL_PORT == 27182


@pytest.mark.unit
def test_backend_scan_stays_below_inference_pools(monkeypatch):
    # The backend scans 27182–27199 and must stop short of 27200, where the
    # inference pools begin (llama.cpp 27200–27299, MLX 27300–27399), so the
    # three local servers never contend for a port.
    import run

    assert run.CANONICAL_PORT + run.PORT_SCAN_COUNT <= 27200

    # With every candidate free, it returns the canonical port; the highest port
    # it can ever return stays inside the backend's own window.
    monkeypatch.setattr(run, "port_open", lambda host, port, timeout=0.4: False)
    assert run.find_available_port(run.CANONICAL_PORT, "127.0.0.1") == run.CANONICAL_PORT

    # With everything busy, it gives up (None) rather than wandering into 27200+.
    monkeypatch.setattr(run, "port_open", lambda host, port, timeout=0.4: True)
    assert run.find_available_port(run.CANONICAL_PORT, "127.0.0.1") is None


@pytest.mark.unit
def test_compute_first_run(tmp_path):
    import run

    # No postgres/PG_VERSION yet -> first run.
    assert run.compute_first_run(tmp_path) is True

    pgdata = tmp_path / "postgres"
    pgdata.mkdir()
    (pgdata / "PG_VERSION").write_text("16\n")
    assert run.compute_first_run(tmp_path) is False


@pytest.mark.unit
def test_startup_timeout_is_first_run_aware():
    import run

    assert run.startup_timeout_seconds(True) == run.FIRST_RUN_TIMEOUT_SECONDS
    assert run.startup_timeout_seconds(False) == run.STARTUP_TIMEOUT_SECONDS
    assert run.FIRST_RUN_TIMEOUT_SECONDS > run.STARTUP_TIMEOUT_SECONDS


@pytest.mark.unit
def test_configure_stdio_forces_utf8_replace(monkeypatch):
    # The frozen interpreter ignores PYTHONUTF8, so configure_stdio must pin
    # both streams to UTF-8 with errors="replace" (never-raising) and keep them
    # line-buffered — otherwise a Unicode log line kills the handler (#168).
    import run

    class FakeStream:
        def __init__(self):
            self.kwargs = None

        def reconfigure(self, **kwargs):
            self.kwargs = kwargs

    fake_out = FakeStream()
    fake_err = FakeStream()
    monkeypatch.setattr(sys, "stdout", fake_out)
    monkeypatch.setattr(sys, "stderr", fake_err)

    run.configure_stdio()

    for stream in (fake_out, fake_err):
        assert stream.kwargs == {
            "line_buffering": True,
            "encoding": "utf-8",
            "errors": "replace",
        }


@pytest.mark.unit
def test_configure_library_env_sets_defaults(monkeypatch):
    # configure_library_env pins noisy-library defaults and closes every known
    # egress switch (#109).
    import run

    for key in (
        "TOKENIZERS_PARALLELISM",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "PYTHONNOUSERSITE",
        "HF_HUB_DISABLE_TELEMETRY",
        "LANGSMITH_TRACING",
        "LANGCHAIN_TRACING_V2",
    ):
        monkeypatch.delenv(key, raising=False)

    run.configure_library_env()

    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"
    assert os.environ["PYTHONNOUSERSITE"] == "1"


@pytest.mark.unit
def test_performance_knobs_stay_overridable(monkeypatch):
    # Thread counts are tuning, not egress: an operator who sets them means it.
    import run

    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    run.configure_library_env()
    assert os.environ["OMP_NUM_THREADS"] == "8"


@pytest.mark.unit
@pytest.mark.parametrize(
    "var",
    ["LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"],
)
def test_langchain_tracing_is_forced_off(monkeypatch, var):
    """LangChain ships prompts to a cloud service when an env var says so.

    `langsmith` rides in as a transitive dependency of langchain and is
    bundled in the frozen build, and enabling it takes nothing but an
    environment variable. A user with that variable exported in their shell
    profile would have every system prompt, knowledge-base excerpt, question
    and answer POSTed to api.smith.langchain.com without the app ever saying
    so. This is the one switch that must not be left to `setdefault`.
    """
    import run

    monkeypatch.setenv(var, "true")
    run.configure_library_env()
    assert os.environ[var] == "false"


@pytest.mark.unit
def test_hf_telemetry_cannot_be_re_enabled(monkeypatch):
    # Same reasoning as the tracing switch: an inherited HF_HUB_DISABLE_TELEMETRY=0
    # silently turned the hub client's telemetry back on for every call.
    import run

    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "0")
    run.configure_library_env()
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"


@pytest.mark.unit
def test_hf_xet_is_disabled(monkeypatch):
    # Model downloads never use Xet storage (HfFileSystem.get_file falls back
    # to plain fsspec streaming for the app's custom callback), and the
    # packaged app does not ship hf_xet, so a dev run must match it.
    import run

    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    run.configure_library_env()
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"


@pytest.mark.unit
def test_hf_xet_cannot_be_re_enabled(monkeypatch):
    # Same reasoning as the telemetry switch: an inherited HF_HUB_DISABLE_XET=0
    # would silently turn Xet back on for the small hf_hub_download fetches.
    import run

    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
    run.configure_library_env()
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"


@pytest.mark.unit
def test_configure_stdio_survives_streams_without_reconfigure(monkeypatch):
    # Some streams (bare objects, or ones that reject reconfigure kwargs) must
    # not break startup — configure_stdio guards each stream individually.
    import run

    class NoReconfigure:
        pass

    class RaisingStream:
        def reconfigure(self, **kwargs):
            raise TypeError("reconfigure not supported")

    # Bare stream with no reconfigure attribute at all.
    monkeypatch.setattr(sys, "stdout", NoReconfigure())
    monkeypatch.setattr(sys, "stderr", NoReconfigure())
    run.configure_stdio()  # must not raise

    # Stream whose reconfigure raises: swallowed by the per-stream guard.
    monkeypatch.setattr(sys, "stdout", RaisingStream())
    monkeypatch.setattr(sys, "stderr", RaisingStream())
    run.configure_stdio()  # must not raise


@pytest.mark.unit
def test_stdin_watch_enabled_reads_env(monkeypatch):
    # Opt-in: only ERUDI_WATCH_STDIN=1 turns the stdin-EOF watcher on. Anything
    # else (unset, "0", "true") leaves it off so dev runs and the subprocess
    # launcher test never grow a stdin reader.
    import run

    monkeypatch.delenv("ERUDI_WATCH_STDIN", raising=False)
    assert run.stdin_watch_enabled() is False

    monkeypatch.setenv("ERUDI_WATCH_STDIN", "0")
    assert run.stdin_watch_enabled() is False

    monkeypatch.setenv("ERUDI_WATCH_STDIN", "1")
    assert run.stdin_watch_enabled() is True


class _FakeStdin:
    """Stand-in for sys.stdin exposing a real fileno() for os.read (#283).

    The watcher now reads the RAW fd (os.read on fileno()), never a buffered
    object, so the fakes hand it a real OS file descriptor: the read end of an
    os.pipe(). Closing the write end delivers EOF (os.read -> b"").
    """

    def __init__(self, fd):
        self._fd = fd

    def fileno(self):
        return self._fd


@pytest.mark.unit
def test_stdin_eof_watcher_sets_should_exit(monkeypatch):
    # An EOF on stdin (parent closed the pipe) must flip uvicorn's exit flag so
    # the FastAPI lifespan shutdown runs — the Windows equivalent of the SIGTERM
    # relay, since Windows has no SIGTERM to catch.
    import run

    read_fd, write_fd = os.pipe()

    class FakeServer:
        should_exit = False

    monkeypatch.setattr(sys, "stdin", _FakeStdin(read_fd))
    server = FakeServer()

    try:
        thread = run.start_stdin_eof_watcher(server)
        # Closing the write end delivers EOF to os.read(read_fd, ...).
        os.close(write_fd)
        thread.join(timeout=2.0)
    finally:
        os.close(read_fd)

    assert not thread.is_alive()
    assert server.should_exit is True


@pytest.mark.unit
def test_stdin_eof_watcher_survives_broken_stdin(monkeypatch):
    # A stdin whose raw read raises must not crash the launcher (the thread just
    # exits) and must leave the exit flag untouched. A closed fd makes os.read
    # raise OSError(EBADF).
    import run

    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    os.close(write_fd)  # fd is now invalid -> os.read raises

    class FakeServer:
        should_exit = False

    monkeypatch.setattr(sys, "stdin", _FakeStdin(read_fd))
    server = FakeServer()

    thread = run.start_stdin_eof_watcher(server)
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert server.should_exit is False


@pytest.mark.unit
def test_stdin_eof_watcher_survives_none_stdin(monkeypatch):
    # No controlling stdin (sys.stdin is None, as under some frozen/detached
    # launches) must return immediately without touching the exit flag (#283).
    import run

    class FakeServer:
        should_exit = False

    monkeypatch.setattr(sys, "stdin", None)
    server = FakeServer()

    thread = run.start_stdin_eof_watcher(server)
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert server.should_exit is False


@pytest.mark.unit
@pytest.mark.skipif(platform.system() != "Windows", reason="Windows-only stdin path")
def test_stdin_eof_watcher_never_blocks_in_the_crt_on_windows(monkeypatch):
    # THE regression pin for #321/#313. A blocking os.read on the stdin fd parks
    # this daemon thread inside the UCRT for the life of the process, and while
    # it is parked there every off-main-thread native import in the frozen build
    # deadlocks in LoadLibraryExW. Bisected to exactly this: same bundle, same
    # location, same spawn shape, ERUDI_WATCH_STDIN=1 the only difference --
    # 4 runs without it imported fine, 3 runs with it hung with identical frames.
    #
    # So the Windows watcher must reach EOF without ever calling os.read. Making
    # os.read fatal is what makes a regression here fail loudly instead of
    # silently reintroducing a deadlock nothing in CI can see.
    import run

    def _forbidden(*args, **kwargs):
        raise AssertionError("os.read on the stdin fd reintroduces the #321 deadlock")

    read_fd, write_fd = os.pipe()

    class FakeServer:
        should_exit = False

    monkeypatch.setattr(sys, "stdin", _FakeStdin(read_fd))
    monkeypatch.setattr(run.os, "read", _forbidden)
    server = FakeServer()

    try:
        thread = run.start_stdin_eof_watcher(server)
        os.close(write_fd)
        thread.join(timeout=5.0)
    finally:
        os.close(read_fd)

    assert not thread.is_alive()
    assert server.should_exit is True


@pytest.mark.unit
@pytest.mark.skipif(platform.system() != "Windows", reason="Windows-only stdin path")
def test_stdin_eof_watcher_keeps_watching_through_written_bytes_on_windows(monkeypatch):
    # Only a CLOSED pipe means shutdown. Bytes on stdin must be drained and
    # ignored: treating a write as EOF would quit the backend the first time
    # anything spoke to it. Electron never writes, so this guards the drain
    # branch that would otherwise only ever run in production.
    import run

    read_fd, write_fd = os.pipe()

    class FakeServer:
        should_exit = False

    monkeypatch.setattr(sys, "stdin", _FakeStdin(read_fd))
    server = FakeServer()

    try:
        thread = run.start_stdin_eof_watcher(server)
        os.write(write_fd, b"noise")
        # Long enough to cover several poll ticks (0.25s each): the watcher
        # must still be parked on the pipe, not have read the write as EOF.
        thread.join(timeout=1.0)
        assert thread.is_alive()
        assert server.should_exit is False

        os.close(write_fd)
        thread.join(timeout=5.0)
    finally:
        os.close(read_fd)

    assert not thread.is_alive()
    assert server.should_exit is True


@pytest.mark.unit
def test_stdin_eof_watcher_dispatches_to_the_windows_path(monkeypatch):
    # Cross-platform companion to the two Windows-only tests above, so CI (which
    # runs Ubuntu) still guards the dispatch. The real Windows wait cannot run
    # here -- it is ctypes/kernel32 -- but which branch gets taken can, and that
    # is the half a refactor is most likely to get wrong.
    import run

    calls = []

    def _fake_win_wait(fd):
        calls.append(fd)

    read_fd, write_fd = os.pipe()

    class FakeServer:
        should_exit = False

    monkeypatch.setattr(sys, "stdin", _FakeStdin(read_fd))
    monkeypatch.setattr(run.platform, "system", lambda: "Windows")
    monkeypatch.setattr(run, "_win_wait_for_stdin_eof", _fake_win_wait)
    monkeypatch.setattr(
        run.os,
        "read",
        lambda *a, **k: pytest.fail("Windows must not take the os.read path (#321)"),
    )
    server = FakeServer()

    try:
        thread = run.start_stdin_eof_watcher(server)
        thread.join(timeout=5.0)
    finally:
        os.close(read_fd)
        os.close(write_fd)

    assert calls == [read_fd]
    assert server.should_exit is True


@pytest.mark.unit
def test_parent_watchdog_enabled_by_default(monkeypatch):
    # The watchdog is on by default everywhere, dev runs included: its
    # predicate is a PARENT CHANGE, and a dev's shell staying alive is the
    # normal case, so plain `python run.py` stays fully usable. The opt-out
    # exists only for deliberately detached runs (nohup/setsid), where
    # reparenting is expected and survival is intended (#224).
    import run

    monkeypatch.delenv("ERUDI_NO_PARENT_WATCHDOG", raising=False)
    assert run.parent_watchdog_enabled() is True

    monkeypatch.setenv("ERUDI_NO_PARENT_WATCHDOG", "0")
    assert run.parent_watchdog_enabled() is True

    monkeypatch.setenv("ERUDI_NO_PARENT_WATCHDOG", "1")
    assert run.parent_watchdog_enabled() is False


@pytest.mark.unit
def test_parent_alive_probe_posix_tracks_ppid():
    # POSIX probe: alive while os.getppid() still equals the recorded parent,
    # dead the moment the process is reparented (ppid changed) -- pid 1 or a
    # subreaper alike (#224).
    import run

    assert run._parent_alive_probe(os.getppid())() is True
    # Any initial ppid that is not the current one reads as "parent gone".
    bogus = os.getppid() + 12345
    assert run._parent_alive_probe(bogus)() is False


@pytest.mark.unit
def test_parent_alive_probe_windows_uses_psutil(monkeypatch):
    # Windows never rewrites ppid on parent death, so the probe pins the
    # parent through psutil (identity guarded by create_time against pid
    # reuse). psutil is cross-platform, so the branch is testable here.
    import subprocess
    import run

    monkeypatch.setattr(run.os, "name", "nt")
    try:
        # Our own live process stands in for a living parent.
        assert run._parent_alive_probe(os.getpid())() is True

        # A freshly exited (and reaped) child reads as a dead parent.
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait(timeout=10)
        assert run._parent_alive_probe(proc.pid)() is False
    finally:
        monkeypatch.undo()


class _WatchdogFakeServer:
    should_exit = False


@pytest.mark.unit
def test_parent_watchdog_fires_when_parent_dies():
    # The probe flips to dead after a few healthy polls: the watchdog must not
    # fire early, then must flip uvicorn's exit flag (the SAME clean shutdown
    # path the SIGTERM relay uses) and terminate its thread.
    import run

    calls = {"n": 0}

    def probe():
        calls["n"] += 1
        return calls["n"] < 3

    server = _WatchdogFakeServer()
    thread = run.start_parent_watchdog(server, poll_seconds=0.02, parent_alive=probe)
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert server.should_exit is True
    assert calls["n"] >= 3  # it polled while the parent lived, no early fire


@pytest.mark.unit
def test_parent_watchdog_fires_immediately_if_parent_already_gone():
    # Parent died during the (long) boot imports, before the watchdog started:
    # the first check fires without waiting a poll interval.
    import run

    server = _WatchdogFakeServer()
    thread = run.start_parent_watchdog(server, poll_seconds=30.0, parent_alive=lambda: False)
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert server.should_exit is True


@pytest.mark.unit
def test_parent_watchdog_survives_broken_probe():
    # A probe that raises must never crash the launcher or force a shutdown:
    # the thread exits and shutdown is left to signals/stdin-EOF.
    import run

    def probe():
        raise RuntimeError("probe broke")

    server = _WatchdogFakeServer()
    thread = run.start_parent_watchdog(server, poll_seconds=0.02, parent_alive=probe)
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert server.should_exit is False


@pytest.mark.unit
def test_graceful_shutdown_timeout_is_bounded():
    # should_exit alone waits FOREVER on in-flight tasks (uvicorn's
    # timeout_graceful_shutdown defaults to None), which is exactly the
    # observed orphan: a SIGKILLed Electron mid-generation left the backend
    # generating until completion (#224). The launcher must bound that wait so
    # every should_exit path (watchdog, SIGTERM, stdin EOF) reaches the
    # lifespan shutdown (inference child terminated, postgres stopped) promptly.
    import run

    assert 0 < run.GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS <= 30


def test_json_event_emission():
    import time

    launcher = Path(__file__).parent.parent / "run.py"
    proc = subprocess.Popen(
        [sys.executable, str(launcher), "--port", "12345"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": str(launcher.parent)},
    )

    starting_events = []
    deadline = time.time() + 15

    try:
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            decoded = line.decode().strip()
            if not decoded:
                continue
            try:
                event = json.loads(decoded)
                if event.get("event") == "starting":
                    starting_events.append(event)
                    break
            except json.JSONDecodeError:
                pass
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    assert starting_events, "No starting event emitted"
    event = starting_events[0]
    assert event["event"] == "starting"
    assert event["port"] == 12345
    assert "first_run" in event
    assert isinstance(event["first_run"], bool)
