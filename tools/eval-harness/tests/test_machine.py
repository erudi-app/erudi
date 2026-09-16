import ctypes
import os
import re
import subprocess
import sys
import time

import pytest

from erudi_eval import machine

mac_only = pytest.mark.skipif(sys.platform != "darwin", reason="macOS machine readers")
PAGE = 16384


def test_vm_statistics64_layout_matches_the_kernel_header():
    assert ctypes.sizeof(machine.VmStatistics64) == 152
    assert machine.HOST_VM_INFO64_COUNT == 38
    assert machine.VmStatistics64.compressor_page_count.offset == 128
    assert machine.VmStatistics64.total_uncompressed_pages_in_compressor.offset == 144


def test_vm_statistics64_from_bytes_and_activity_monitor_derivation():
    s = machine.VmStatistics64(
        free_count=178307, active_count=235051, inactive_count=223608, wire_count=235015,
        pageins=64031988, pageouts=784198, faults=2506845818, purgeable_count=6566, speculative_count=11304,
        decompressions=233396114, compressions=274018393, swapins=19530223, swapouts=22773811,
        compressor_page_count=125587, external_page_count=182909, internal_page_count=287054,
        total_uncompressed_pages_in_compressor=1555165,
    )
    raw = bytes(s)  # what host_statistics64 writes into the buffer
    parsed = machine.vm_struct_to_dict(machine.VmStatistics64.from_buffer_copy(raw))
    assert parsed["compressor_page_count"] == 125587 and parsed["swapouts"] == 22773811
    d = machine.derive_mac_memory(parsed, PAGE)
    assert d["app_memory_bytes"] == (287054 - 6566) * PAGE
    assert d["wired_bytes"] == 235015 * PAGE
    assert d["compressed_bytes"] == 125587 * PAGE
    assert d["cached_files_bytes"] == (182909 + 6566) * PAGE
    assert d["memory_used_bytes"] == d["app_memory_bytes"] + d["wired_bytes"] + d["compressed_bytes"]


def test_counter_rates_and_reset():
    assert machine.counter_rates(None, {"swapouts": 10}, 1.0) == {"swapouts": None}
    rates = machine.counter_rates({"swapouts": 100, "pageins": 50}, {"swapouts": 160, "pageins": 40}, 2.0)
    assert rates == {"swapouts": 30.0, "pageins": None}  # a counter that went backwards gives no rate
    assert machine.counter_rates({"x": 1}, {"x": 5}, 0.0) == {"x": None}


def test_parsers_linux_and_pmset():
    meminfo = "MemTotal:       16000000 kB\nMemAvailable:    8000000 kB\nCached:  1000 kB\nSwapTotal: 2048 kB\nSwapFree: 1024 kB\nBuffers: 5 kB\n"
    assert machine.parse_meminfo(meminfo) == {"MemTotal": 16000000 * 1024, "MemAvailable": 8000000 * 1024, "Cached": 1024000, "SwapTotal": 2048 * 1024, "SwapFree": 1024 * 1024}
    assert machine.parse_vmstat_counters("nr_free_pages 12\npswpin 7\npswpout 9\npgmajfault 3\n") == {"pswpin": 7, "pswpout": 9, "pgmajfault": 3}
    psi = machine.parse_psi("some avg10=1.50 avg60=0.20 avg300=0.00 total=1\nfull avg10=0.10 avg60=0.00 avg300=0.00 total=0\n")
    assert psi == {"some": {"avg10": 1.5, "avg60": 0.2, "avg300": 0.0}, "full": {"avg10": 0.1, "avg60": 0.0, "avg300": 0.0}}
    therm = "Note: No thermal warning level has been recorded\nNote: No performance warning level has been recorded\nNote: No CPU power status has been recorded\n"
    assert machine.parse_pmset_therm(therm) == {"thermal_warning_level": "none recorded", "performance_warning_level": "none recorded"}
    assert machine.parse_pmset_therm("CPU_Scheduler_Limit = 100\nCPU_Speed_Limit = 80\n") == {"cpu_scheduler_limit": 100, "cpu_speed_limit": 80}
    batt = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=21889123)\t18%; discharging; 1:12 remaining present: true\n"
    assert machine.parse_pmset_batt(batt) == {"power_source": "Battery Power", "battery_percent": 18, "battery_state": "discharging"}
    assert machine.parse_pmset_settings(" displaysleep         10\n lowpowermode         1\n") == {"low_power_mode": True}


def test_vm_stat_fallback_parser():
    text = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages wired down:  100.\nPages occupied by compressor:  10.\n"
    assert machine.parse_vm_stat(text) == {"wired_bytes": 1638400, "compressed_bytes": 163840}


@mac_only
def test_host_statistics64_agrees_with_vm_stat():
    vm = machine.mac_vm_statistics()
    text = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    wired = int(re.search(r"Pages wired down:\s+(\d+)", text).group(1))
    swapouts = int(re.search(r"Swapouts:\s+(\d+)", text).group(1))
    assert abs(vm["wire_count"] - wired) <= max(2000, wired * 0.05)
    assert vm["swapouts"] <= swapouts + 1000 and vm["swapouts"] >= swapouts - 1000
    swap = machine.mac_swap()
    assert swap["swap_total_bytes"] >= swap["swap_used_bytes"]


def test_reader_block_and_rates():
    r = machine.MachineReader(disk_path=os.getcwd(), slow_every=3600)
    first = r.read()
    time.sleep(0.2)
    second = r.read()
    assert first["available_bytes"] > 0 and "cpu_percent" in second and second["process_count"] > 0
    assert all(v is None for v in first["rates_per_s"].values())
    assert second["rates_per_s"] and all(v is None or v >= 0 for v in second["rates_per_s"].values())
    assert "power_thermal" in second and second["data_volume_free_bytes"] > 0
    if sys.platform == "darwin":
        assert second["memory_pressure_level"] in {"normal", "warn", "critical"}
        assert second["wired_bytes"] > 0 and second["compressed_bytes"] >= 0 and second["swap_total_bytes"] >= 0
        assert "vm_source" not in second  # the ctypes path worked, no vm_stat subprocess
        assert "power_source" in second["power_thermal"]


def test_top_other_processes_excludes_given_pids():
    block = machine.top_other_processes({os.getpid()}, limit=5)
    assert block["metric"] == "rss" and len(block["top"]) <= 5
    assert os.getpid() not in {row["pid"] for row in block["top"]}
    assert block["total_rss_mb"] >= sum(row["rss_mb"] for row in block["top"])


class _CheapSampler:
    """Builds a Sampler whose sample is a timestamp: tests the loop, not the readers."""

    @staticmethod
    def make(interval):
        from erudi_eval.sampler import DiscoveryState, Sampler
        from erudi_eval.layout import resolve_layout

        class Writer:
            def write(self, record):
                pass

        class FakeGpu:
            def read(self):
                return {"source": "none"}

        class FakeMachine:
            def read(self):
                return {}

        s = Sampler(Writer(), DiscoveryState(resolve_layout(), 27182), interval=interval, gpu=FakeGpu(), machine_reader=FakeMachine())
        s.starts = []

        def sample_once():
            s.starts.append(time.monotonic())
            return {}

        s.sample_once = sample_once
        return s


def test_set_interval_wakes_the_sampler_immediately_and_holds_the_fast_rate():
    s = _CheapSampler.make(interval=30.0)
    s.start()
    try:
        deadline = time.monotonic() + 5
        while not s.starts and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(s.starts) == 1  # first sample, now sleeping for 30 s
        switched = time.monotonic()
        s.set_interval(0.05)
        time.sleep(0.6)
        after = [t for t in s.starts if t >= switched]
        assert after and after[0] - switched < 0.05  # woke at once, did not finish the 30 s sleep
        gaps = [b - a for a, b in zip(after, after[1:])]
        assert len(gaps) >= 8 and sorted(gaps)[len(gaps) // 2] == pytest.approx(0.05, abs=0.02)
        s.set_interval(30.0)
        time.sleep(0.2)
        count = len(s.starts)
        time.sleep(0.3)
        assert len(s.starts) == count  # back to the slow rate
    finally:
        s.stop()
        s.join(timeout=2)
    assert not s.is_alive()


def test_real_sample_records_cost_interval_harness_and_top_other(tmp_path):
    from erudi_eval.layout import resolve_layout
    from erudi_eval.sampler import DiscoveryState, Sampler
    from erudi_eval.util import JsonlWriter

    layout = resolve_layout(app_path=str(tmp_path / "NoApp.app"))
    layout.data_root = tmp_path
    s = Sampler(JsonlWriter(tmp_path / "s.jsonl"), DiscoveryState(layout, 1), interval=0.5)
    first = s.sample_once()
    second = s.sample_once()
    assert first["interval_effective_s"] is None and second["interval_effective_s"] > 0
    assert first["sample_cost_ms"] > 0 and "top_other" in first and "top_other" not in second
    assert first["harness"]["pid"] == os.getpid() and first["processes"] == []
    assert os.getpid() not in {r["pid"] for r in first["top_other"]["top"]}
    assert "available_bytes" in second["system"] and "rates_per_s" in second["system"]
