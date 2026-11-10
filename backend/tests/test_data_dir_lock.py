"""Never two backends on one data folder (#706).

A backend that outlived its app (a download it would not drop) kept writing
into the data folder while a relaunched app started a second backend on it.
The launcher now holds an OS lock on the data folder for its whole life; a
second backend is refused with ``DATA_DIR_IN_USE`` before it touches the
cluster, and the lock dies with the process that held it.

Every test locks its own temporary folder, so the suite stays parallel-safe
and never meets a running app or a developer's backend.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from src.launcher.data_dir_lock import DataDirInUse, acquire_data_dir_lock

BACKEND_DIR = Path(__file__).resolve().parents[1]
RUN_PY = BACKEND_DIR / "run.py"


@pytest.mark.unit
def test_a_second_acquisition_is_refused_until_the_first_is_released(tmp_path):
    first = acquire_data_dir_lock(tmp_path)
    try:
        with pytest.raises(DataDirInUse) as refused:
            acquire_data_dir_lock(tmp_path)
        assert refused.value.data_dir == tmp_path
        assert refused.value.holder_pid == os.getpid()
        assert str(tmp_path) in str(refused.value)
    finally:
        first.release()

    acquire_data_dir_lock(tmp_path).release()


@pytest.mark.unit
def test_two_data_folders_do_not_share_a_lock(tmp_path):
    a = acquire_data_dir_lock(tmp_path / "a")
    b = acquire_data_dir_lock(tmp_path / "b")
    a.release()
    b.release()


HOLDER_SOURCE = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from src.launcher.data_dir_lock import acquire_data_dir_lock

    lock = acquire_data_dir_lock(Path(sys.argv[1]))
    print("held", flush=True)
    time.sleep(120)
    """
)


@pytest.mark.integration
def test_the_lock_dies_with_the_process_that_held_it(tmp_path):
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLDER_SOURCE, str(tmp_path)],
        cwd=BACKEND_DIR,
        env={**os.environ, "PYTHONPATH": str(BACKEND_DIR)},
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(DataDirInUse) as refused:
            acquire_data_dir_lock(tmp_path)
        assert refused.value.holder_pid == holder.pid
    finally:
        # A hard kill: no cleanup runs in the holder, the OS releases the lock.
        holder.kill()
        holder.wait(30)
        holder.stdout.close()

    acquire_data_dir_lock(tmp_path).release()


@pytest.mark.integration
def test_the_launcher_refuses_a_data_folder_in_use(tmp_path):
    instance = tmp_path / "instance"
    data_dir = instance / "data"
    data_dir.mkdir(parents=True)
    held = acquire_data_dir_lock(data_dir)
    env = {
        **os.environ,
        "PYTHONPATH": str(BACKEND_DIR),
        "ERUDI_DATA_ROOT": str(instance),
        "ERUDI_NO_PARENT_WATCHDOG": "1",
    }
    env.pop("ERUDI_WATCH_STDIN", None)
    try:
        result = subprocess.run(
            [sys.executable, str(RUN_PY), "--port", "27198"],
            cwd=BACKEND_DIR,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        held.release()

    events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert result.returncode == 1, result.stderr
    assert events, result.stderr
    assert events[-1]["event"] == "startup_error"
    assert events[-1]["code"] == "DATA_DIR_IN_USE"
    assert str(data_dir) in events[-1]["message"]
    # Refused before anything else: no port announced, no cluster started.
    assert not any(e["event"] == "starting" for e in events)
    assert not (data_dir / "postgres").exists()
