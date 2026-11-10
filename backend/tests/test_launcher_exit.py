"""The backend process ends within seconds of a shutdown request (#706).

Seen: the app was killed at 18 % of a model download; the backend stopped
Postgres and its port, then lived 2 min 10 s more because the interpreter's
exit joins the executor threads still streaming the model. Every shutdown path
now goes through ``request_shutdown`` (downloads interrupted at once), the
launcher stops waiting for the server thread a short grace after the lifespan
reported its services stopped, and an exit deadline ends the process if a
thread still holds the interpreter's exit after that.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import run

BACKEND_DIR = Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_request_shutdown_interrupts_the_downloads_and_sets_the_exit_flag(monkeypatch):
    from src.domains.llms import services

    seen: list[bool] = []
    server = SimpleNamespace(should_exit=False)
    monkeypatch.setattr(
        services, "interrupt_active_downloads", lambda: seen.append(server.should_exit)
    )

    run.request_shutdown(server)

    # The transfers are told first, so they are already stopping while uvicorn
    # drains its requests (a download task is one of them).
    assert seen == [False]
    assert server.should_exit is True


@pytest.mark.unit
def test_request_shutdown_sets_the_flag_even_if_the_interrupt_fails(monkeypatch):
    from src.domains.llms import services

    def broken():
        raise RuntimeError("registry gone")

    monkeypatch.setattr(services, "interrupt_active_downloads", broken)
    server = SimpleNamespace(should_exit=False)

    run.request_shutdown(server)

    assert server.should_exit is True


@pytest.mark.unit
def test_the_parent_watchdog_goes_through_request_shutdown(monkeypatch):
    requested = threading.Event()
    monkeypatch.setattr(run, "request_shutdown", lambda server: requested.set())

    run.start_parent_watchdog(SimpleNamespace(should_exit=False), parent_alive=lambda: False)

    assert requested.wait(5)


@pytest.mark.unit
def test_the_stdin_watcher_goes_through_request_shutdown(monkeypatch):
    requested = threading.Event()
    monkeypatch.setattr(run, "request_shutdown", lambda server: requested.set())
    read_fd, write_fd = os.pipe()
    os.close(write_fd)  # EOF at once
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(fileno=lambda: read_fd))
    if sys.platform == "win32":
        monkeypatch.setattr(run, "_win_wait_for_stdin_eof", lambda fd: None)

    try:
        run.start_stdin_eof_watcher(SimpleNamespace(should_exit=False))
        assert requested.wait(5)
    finally:
        os.close(read_fd)


def _stuck_thread() -> tuple[threading.Thread, threading.Event]:
    release = threading.Event()
    thread = threading.Thread(target=release.wait, args=(30,), daemon=True)
    thread.start()
    return thread, release


@pytest.mark.unit
def test_the_wait_ends_a_grace_after_the_services_stopped():
    thread, release = _stuck_thread()
    services_stopped = threading.Event()
    services_stopped.set()
    try:
        started = time.monotonic()
        run.wait_for_server(
            SimpleNamespace(should_exit=True), thread, services_stopped, grace=0.3, hard_limit=30
        )
        elapsed = time.monotonic() - started
    finally:
        release.set()
    assert 0.3 <= elapsed < 3


@pytest.mark.unit
def test_the_wait_ends_at_the_hard_limit_when_the_services_never_report():
    thread, release = _stuck_thread()
    try:
        started = time.monotonic()
        run.wait_for_server(
            SimpleNamespace(should_exit=True), thread, threading.Event(), grace=30, hard_limit=0.3
        )
        elapsed = time.monotonic() - started
    finally:
        release.set()
    assert 0.3 <= elapsed < 3


@pytest.mark.unit
def test_the_wait_has_no_limit_while_no_shutdown_was_asked():
    """A running backend is never cut short: the bounds start with the shutdown."""
    thread, release = _stuck_thread()
    server = SimpleNamespace(should_exit=False)
    waiter = threading.Thread(
        target=run.wait_for_server,
        args=(server, thread, threading.Event()),
        kwargs={"grace": 0.1, "hard_limit": 0.1},
        daemon=True,
    )
    waiter.start()
    waiter.join(0.8)
    assert waiter.is_alive()
    release.set()
    waiter.join(5)
    assert not waiter.is_alive()


# A transfer thread parked in a network read that never returns: the default
# executor's worker, which the interpreter joins at exit.
EXITING_SOURCE = textwrap.dedent(
    """
    import sys
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import run

    blocked_read = threading.Event()
    ThreadPoolExecutor(max_workers=1).submit(blocked_read.wait)
    if sys.argv[1] == "armed":
        run.arm_exit_deadline(0.5)
    print("leaving main", flush=True)
    """
)


def _spawn_exiting(tmp_path: Path, mode: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", EXITING_SOURCE, mode],
        cwd=BACKEND_DIR,
        env={
            **os.environ,
            "PYTHONPATH": str(BACKEND_DIR),
            "ERUDI_DATA_ROOT": str(tmp_path / "instance"),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


@pytest.mark.integration
def test_a_thread_stuck_in_a_read_holds_the_interpreter_exit(tmp_path):
    """The control: without the deadline, the process does not end (#706)."""
    proc = _spawn_exiting(tmp_path, "unarmed")
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            proc.wait(3)
    finally:
        proc.kill()
        proc.communicate(timeout=10)


@pytest.mark.integration
def test_the_exit_deadline_ends_a_process_held_by_a_stuck_thread(tmp_path):
    proc = _spawn_exiting(tmp_path, "armed")
    try:
        started = time.monotonic()
        out, err = proc.communicate(timeout=15)
        elapsed = time.monotonic() - started
    finally:
        proc.kill()
    assert "leaving main" in out
    assert proc.returncode == 0, err
    assert elapsed < 10
    backend_log = (tmp_path / "instance" / "logs" / "backend.log").read_text(encoding="utf-8")
    assert "Exit still held" in backend_log


class _BrokenStdout:
    """stdout once the parent is gone: every write is a broken pipe."""

    def write(self, _text):
        raise BrokenPipeError(32, "Broken pipe")

    def flush(self):
        raise BrokenPipeError(32, "Broken pipe")


@pytest.mark.unit
def test_an_event_to_a_dead_parent_is_dropped_not_raised(monkeypatch):
    from src.launcher import events

    monkeypatch.setattr(sys, "stdout", _BrokenStdout())
    monkeypatch.setattr(events, "_stdout_lost", False)

    events.emit_event({"event": "shutdown"})  # must not raise
    events.emit_event({"event": "shutdown"})


@pytest.mark.unit
def test_after_ready_the_exit_deadline_is_armed_before_anything_is_written(monkeypatch):
    """#706 itself: the parent was SIGKILLed, so stdout is a broken pipe. The
    deadline must be armed whatever writing the `shutdown` event does."""
    from src.launcher import events

    order: list[str] = []
    monkeypatch.setattr(sys, "stdout", _BrokenStdout())
    monkeypatch.setattr(events, "_stdout_lost", False)
    monkeypatch.setattr(run, "arm_exit_deadline", lambda *a, **k: order.append("armed"))
    real_emit = run.emit_event

    def emit(payload):
        order.append(payload["event"])
        real_emit(payload)

    monkeypatch.setattr(run, "emit_event", emit)
    finished = threading.Thread(target=lambda: None)
    finished.start()
    finished.join()

    run.finish_after_ready(SimpleNamespace(should_exit=True), finished, threading.Event())

    assert order == ["armed", "shutdown"]


# The backend's parent closed its end of the pipe (it was killed): an event
# written afterwards must not crash the launcher.
ORPHAN_EMIT_SOURCE = textwrap.dedent(
    """
    import sys
    from src.launcher.events import emit_event

    sys.stdin.readline()  # the parent closes our stdout, then says go
    emit_event({"event": "shutdown"})
    emit_event({"event": "shutdown"})
    """
)


@pytest.mark.integration
def test_an_event_written_after_the_parent_closed_the_pipe_is_no_crash(tmp_path):
    err_path = tmp_path / "stderr.txt"
    with open(err_path, "w") as err:
        proc = subprocess.Popen(
            [sys.executable, "-c", ORPHAN_EMIT_SOURCE],
            cwd=BACKEND_DIR,
            env={
                **os.environ,
                "PYTHONPATH": str(BACKEND_DIR),
                "ERUDI_DATA_ROOT": str(tmp_path / "instance"),
            },
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=err,
            text=True,
        )
        proc.stdout.close()
        proc.stdin.write("go\n")
        proc.stdin.close()
        assert proc.wait(30) == 0
    assert "Traceback" not in err_path.read_text()


@pytest.mark.unit
def test_the_hard_limit_is_an_error(caplog):
    """Cutting a shutdown short can leave the database unstopped: the next
    start pays a crash recovery. The operation did not happen: ERROR."""
    import logging

    thread, release = _stuck_thread()
    try:
        with caplog.at_level(logging.INFO, logger="erudi"):
            run.wait_for_server(
                SimpleNamespace(should_exit=True),
                thread,
                threading.Event(),
                grace=30,
                hard_limit=0.3,
            )
    finally:
        release.set()
    (record,) = [r for r in caplog.records if "Shutdown not finished" in r.getMessage()]
    assert record.levelno == logging.ERROR
