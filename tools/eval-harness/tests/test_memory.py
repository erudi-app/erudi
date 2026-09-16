import os
import subprocess
import sys
import time

import psutil
import pytest

from erudi_eval import memory
from erudi_eval.gpu import GpuReader, parse_smi_apps, parse_smi_gpus

mac_only = pytest.mark.skipif(sys.platform != "darwin", reason="proc_pid_rusage is macOS-only")


@mac_only
def test_mac_rusage_current_process():
    r = memory.mac_rusage(os.getpid())
    assert r["phys_footprint"] > 1024 * 1024
    assert r["lifetime_max_phys_footprint"] >= r["phys_footprint"]


@mac_only
def test_mac_rusage_child_process_and_dead_pid():
    child = subprocess.Popen([sys.executable, "-c", "import time; b = bytearray(64*1024*1024); time.sleep(20)"])
    try:
        time.sleep(1.0)
        r = memory.mac_rusage(child.pid)
        assert r["lifetime_max_phys_footprint"] >= 60 * 1024 * 1024
    finally:
        child.kill()
        child.wait()
    with pytest.raises(OSError):
        memory.mac_rusage(child.pid)


def test_read_process_primary_metric_present_for_self():
    out = memory.read_process(psutil.Process())
    assert memory.PRIMARY_METRIC in out["metrics"], out["errors"]
    assert "threads" in out["metrics"]


def test_read_process_exited_records_error_not_raise():
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    proc = psutil.Process(child.pid)
    child.wait()
    out = memory.read_process(proc)
    assert memory.PRIMARY_METRIC not in out["metrics"]
    assert out["errors"]


# System-wide readers and their parsers are tested in test_machine.py.


def test_parse_nvidia_smi():
    gpus = parse_smi_gpus("0, NVIDIA GeForce RTX 4070, 12282, 3100, 17, 560.94\n")
    assert gpus == [{"index": 0, "name": "NVIDIA GeForce RTX 4070", "total_mb": 12282.0, "used_mb": 3100.0, "util_pct": 17.0, "driver": "560.94"}]
    apps = parse_smi_apps("4242, 2900\n5151, [N/A]\n")
    assert apps == {4242: 2900.0, 5151: None}


def test_gpu_reader_never_raises():
    data = GpuReader().read()
    assert "source" in data
