"""Measuring a child process's memory from the parent (macOS only).

``footprint`` reads the physical footprint (``ri_phys_footprint``, rusage v4),
the number Activity Monitor shows as Memory. ``begin_peak_window`` resets the
footprint interval through a private libproc SPI that may be missing on a
given macOS; ``peak_since`` then reads the interval peak, or the lifetime peak
when no window started (it can only over-state). Everywhere else the module
answers ``None`` / ``False`` and never raises. These tests spawn a plain
Python child that allocates and frees memory -- no model.
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from src.engines import process_footprint

pytestmark = pytest.mark.unit

MIB = 1024 * 1024

_CHILD = """
import sys
held = None
for line in sys.stdin:
    cmd = line.strip()
    if cmd.startswith("alloc"):
        held = bytearray(int(cmd.split()[1]) * 1024 * 1024)
        for i in range(0, len(held), 4096):
            held[i] = 1
    elif cmd == "free":
        held = None
    print("ok", flush=True)
"""


@pytest.fixture
def child():
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", _CHILD],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )

    def command(text):
        proc.stdin.write(text + "\n")
        proc.stdin.flush()
        proc.stdout.readline()
        time.sleep(0.2)

    try:
        yield proc.pid, command
    finally:
        proc.stdin.close()
        proc.kill()
        proc.wait(timeout=10)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS process accounting")
def test_the_footprint_follows_an_allocation_and_the_peak_survives_the_free(child):
    pid, command = child
    idle = process_footprint.footprint(pid)
    assert idle is not None and idle > 0

    started = process_footprint.begin_peak_window(pid)
    command("alloc 200")
    held = process_footprint.footprint(pid)
    command("free")
    freed = process_footprint.footprint(pid)
    peak = process_footprint.peak_since(pid, started)

    # Wide tolerances: the allocator, the interpreter and the OS all move.
    assert held - idle > 150 * MIB
    assert freed < held - 100 * MIB
    assert peak is not None and peak >= held - 10 * MIB


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS process accounting")
def test_a_new_window_forgets_the_previous_peak_when_the_spi_is_there(child):
    pid, command = child
    command("alloc 200")
    command("free")
    started = process_footprint.begin_peak_window(pid)
    if not started:
        pytest.skip("proc_reset_footprint_interval is not available on this macOS")
    command("alloc 20")
    command("free")
    assert process_footprint.peak_since(pid, started) < 150 * MIB + process_footprint.footprint(pid)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS process accounting")
def test_a_dead_pid_reads_none():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=10)
    assert process_footprint.footprint(proc.pid) is None
    assert process_footprint.peak_since(proc.pid, False) is None


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS system counters")
def test_the_system_counters_read_on_macos():
    assert process_footprint.pressure_level() in (1, 2, 4)
    assert process_footprint.swapouts() >= 0


def test_elsewhere_everything_answers_none_without_raising(monkeypatch):
    monkeypatch.setattr(process_footprint.sys, "platform", "linux")
    assert process_footprint.footprint(1) is None
    assert process_footprint.begin_peak_window(1) is False
    assert process_footprint.peak_since(1, True) is None
    assert process_footprint.pressure_level() is None
    assert process_footprint.swapouts() is None


def test_a_missing_spi_is_false_with_one_warning_per_process(monkeypatch, caplog):
    import logging

    monkeypatch.setattr(process_footprint.sys, "platform", "darwin")
    monkeypatch.setattr(process_footprint, "_reset_interval_function", lambda: None)
    monkeypatch.setattr(process_footprint, "_SPI_WARNED", False)

    with caplog.at_level(logging.WARNING):
        assert process_footprint.begin_peak_window(1) is False
        assert process_footprint.begin_peak_window(1) is False

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].getMessage().isascii()


def test_importing_the_module_loads_no_native_library():
    import importlib
    import subprocess as sp

    probe = (
        "import sys, ctypes\n"
        "before = set(sys.modules)\n"
        "import src.engines.process_footprint as m\n"
        "assert m._LIBPROC is None\n"
        "print('ok')\n"
    )
    out = sp.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=str(importlib.import_module("pathlib").Path(__file__).resolve().parents[1]),
        timeout=60,
    )
    assert out.returncode == 0, out.stderr
