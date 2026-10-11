"""The memory-pressure sensors of the llama.cpp brake, one provider per OS.

``src.engines.memory_pressure`` answers the same questions everywhere -- a
pressure level on the 1/2/4 scale, a cumulative paging counter (the damage
the brake exists for), the child's anonymous memory (the unload's share and the
post-load check), the machine's physical RAM -- from a provider chosen on ``sys.platform``: macOS
(the kernel's pressure level and swap-outs), Linux (PSI, a ``vmstat``
fallback, cgroup v2 only inside a real container), Windows (the commit
charge from ``GlobalMemoryStatusEx``, the paging counter from PDH). The
parsing and the tiering run here on text fixtures and fake natives, on every
CI leg; the native calls themselves run on their own OS only.
"""

from __future__ import annotations

import logging
import mmap
import sys
from types import SimpleNamespace

import pytest

from src.engines import memory_pressure as mp
from src.engines import process_footprint
from src.engines.memory_brake import LLAMA_WARNING_DAMAGE_MIN_BYTES

pytestmark = pytest.mark.unit

GIB = 1024**3
MIB = 1024**2

PSI_CALM = (
    "some avg10=0.00 avg60=0.00 avg300=0.00 total=1000\n"
    "full avg10=0.00 avg60=0.00 avg300=0.00 total=500\n"
)


@pytest.fixture(autouse=True)
def _fresh_provider(monkeypatch):
    monkeypatch.delenv("ERUDI_FORCE_MEMORY_PRESSURE", raising=False)
    monkeypatch.setattr(mp, "_PROVIDER", None)
    monkeypatch.setattr(mp, "_FORCED_DAMAGE_BYTES", 0)
    monkeypatch.setattr(mp, "_FORCE_WARNED", False)
    yield


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _psi(some_avg10=0.0, some_total=0, full_avg10=0.0, full_total=0):
    return (
        f"some avg10={some_avg10:.2f} avg60=0.00 avg300=0.00 total={some_total}\n"
        f"full avg10={full_avg10:.2f} avg60=0.00 avg300=0.00 total={full_total}\n"
    )


def _linux_tree(tmp_path, *, psi=PSI_CALM, cgroup_max=None, cgroup_psi=None, vmstat=None):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self" / "cgroup").write_text("0::/user.slice/app.scope\n")
    if psi is not None:
        (proc / "pressure").mkdir()
        (proc / "pressure" / "memory").write_text(psi)
    if vmstat is not None:
        (proc / "vmstat").write_text(vmstat)
    cgroup = tmp_path / "cgroup"
    scope = cgroup / "user.slice" / "app.scope"
    scope.mkdir(parents=True)
    if cgroup_max is not None:
        (scope / "memory.max").write_text(f"{cgroup_max}\n")
    if cgroup_psi is not None:
        (scope / "memory.pressure").write_text(cgroup_psi)
    return proc, cgroup, scope


def _linux(tmp_path, monkeypatch, *, sout=0, swap_total=8 * GIB, **tree):
    proc, cgroup, scope = _linux_tree(tmp_path, **tree)
    swap = {"sout": sout, "total": swap_total}
    monkeypatch.setattr(
        mp.psutil,
        "swap_memory",
        lambda: SimpleNamespace(sout=swap["sout"], total=swap["total"]),
    )
    clock = _Clock()
    provider = mp._LinuxProvider(proc_root=proc, cgroup_root=cgroup, clock=clock)
    return provider, clock, swap, proc, scope


# ===================== provider selection =====================


@pytest.mark.parametrize(
    "platform, provider",
    [
        ("darwin", "_MacProvider"),
        ("linux", "_LinuxProvider"),
        ("win32", "_WindowsProvider"),
        ("freebsd14", "_NullProvider"),
    ],
)
def test_the_provider_is_chosen_once_on_the_platform(monkeypatch, platform, provider):
    monkeypatch.setattr(sys, "platform", platform)
    chosen = mp._provider()
    assert type(chosen).__name__ == provider
    assert mp._provider() is chosen


def test_a_provider_that_raises_reads_as_no_level_with_its_reason(monkeypatch):
    class _Broken:
        name = "broken"

        def read(self):
            raise OSError("boom")

    monkeypatch.setattr(mp, "_PROVIDER", _Broken())
    reading = mp.read_pressure()
    assert reading.level is None
    assert reading.damage_total_bytes is None
    assert "OSError" in reading.unavailable


# ===================== macOS =====================


def test_macos_wraps_the_kernel_level_and_counts_swapouts_in_bytes(monkeypatch):
    monkeypatch.setattr(process_footprint, "pressure_level", lambda: 2)
    monkeypatch.setattr(process_footprint, "swapouts", lambda: 10)
    monkeypatch.setattr(process_footprint, "footprint", lambda pid: 3 * GIB)
    provider = mp._MacProvider()
    reading = provider.read()
    assert reading.level == 2
    assert reading.damage_total_bytes == 10 * 16384
    # phys_footprint already excludes clean file-backed mappings.
    assert provider.anonymous(1) == 3 * GIB


def test_macos_unreadable_level_says_why(monkeypatch):
    monkeypatch.setattr(process_footprint, "pressure_level", lambda: None)
    monkeypatch.setattr(process_footprint, "swapouts", lambda: None)
    reading = mp._MacProvider().read()
    assert reading.level is None
    assert reading.damage_total_bytes is None
    assert "kern.memorystatus_vm_pressure_level" in reading.unavailable


# ===================== Linux: PSI =====================


def test_linux_first_read_tiers_on_avg10(tmp_path, monkeypatch):
    provider, *_ = _linux(tmp_path, monkeypatch, psi=_psi(some_avg10=12.0), sout=5 * MIB)
    reading = provider.read()
    assert reading.level == 2
    assert reading.damage_total_bytes == 5 * MIB


def test_linux_full_stalls_are_critical(tmp_path, monkeypatch):
    provider, *_ = _linux(tmp_path, monkeypatch, psi=_psi(some_avg10=30.0, full_avg10=6.0))
    assert provider.read().level == 4


def test_linux_tiers_on_the_stall_fraction_since_the_previous_sample(tmp_path, monkeypatch):
    provider, clock, _, proc, _ = _linux(tmp_path, monkeypatch, psi=_psi())
    assert provider.read().level == 1
    # 60 s later: 9 s of "some" stall (15 %), 1 s of "full" (1.7 %) -- warning,
    # whatever the avg10 fields say.
    clock.now += 60
    (proc / "pressure" / "memory").write_text(_psi(some_total=9_000_000, full_total=1_000_000))
    assert provider.read().level == 2
    # 60 s more: 4 s of "full" stall (6.7 %) -- critical.
    clock.now += 60
    (proc / "pressure" / "memory").write_text(_psi(some_total=20_000_000, full_total=5_000_000))
    assert provider.read().level == 4


def test_linux_a_sample_under_the_floor_uses_avg10(tmp_path, monkeypatch):
    provider, clock, _, proc, _ = _linux(tmp_path, monkeypatch, psi=_psi())
    provider.read()
    clock.now += mp.PSI_MIN_SAMPLE_INTERVAL_S - 1
    # A big total jump over 4 s would read as a stall storm; avg10 says calm.
    (proc / "pressure" / "memory").write_text(_psi(some_total=4_000_000))
    assert provider.read().level == 1


def test_linux_desktop_scope_reads_the_system_psi(tmp_path, monkeypatch):
    # A systemd scope (memory.max = max) counts only our own stalls: an idle
    # llama-server never stalls while the machine thrashes.
    provider, *_ = _linux(
        tmp_path,
        monkeypatch,
        psi=_psi(some_avg10=50.0),
        cgroup_max="max",
        cgroup_psi=_psi(),
    )
    assert provider.read().level == 2


def test_linux_a_real_container_reads_its_cgroup_psi(tmp_path, monkeypatch):
    provider, *_ = _linux(
        tmp_path,
        monkeypatch,
        psi=_psi(),
        cgroup_max=4 * GIB,
        cgroup_psi=_psi(full_avg10=10.0),
    )
    assert provider.read().level == 4


def test_linux_without_psi_tiers_on_swapout_and_major_fault_rates(tmp_path, monkeypatch):
    provider, clock, swap, proc, _ = _linux(
        tmp_path, monkeypatch, psi=None, vmstat="pgmajfault 100\npswpout 0\n"
    )
    # The first read primes the rates: no evidence of paging yet.
    assert provider.read().level == 1
    clock.now += 60
    swap["sout"] += 60 * 10 * MIB  # 10 MiB/s
    assert provider.read().level == 2
    clock.now += 60
    swap["sout"] += 60 * 100 * MIB  # 100 MiB/s
    assert provider.read().level == 4
    clock.now += 60
    (proc / "vmstat").write_text(f"pgmajfault {100 + 60 * 300}\n")  # 300/s
    assert provider.read().level == 2


def test_linux_without_swap_counts_major_faults_as_damage(tmp_path, monkeypatch):
    """No swap: swap-outs never move, so paging shows as major faults (the
    mapped weights read back from disk)."""
    provider, _, _, proc, _ = _linux(
        tmp_path, monkeypatch, swap_total=0, vmstat="pgmajfault 1000\n"
    )
    assert provider.read().damage_total_bytes == 1000 * mmap.PAGESIZE
    (proc / "vmstat").write_text("pgmajfault 1500\n")
    assert provider.read().damage_total_bytes == 1500 * mmap.PAGESIZE


def test_linux_with_swap_counts_swap_outs_as_damage(tmp_path, monkeypatch):
    provider, *_ = _linux(tmp_path, monkeypatch, sout=7 * MIB, vmstat="pgmajfault 1000\n")
    assert provider.read().damage_total_bytes == 7 * MIB


def test_linux_nothing_readable_has_no_level(tmp_path, monkeypatch):
    provider, *_ = _linux(tmp_path, monkeypatch, psi=None)

    def _no_swap():
        raise OSError("no /proc/swaps")

    monkeypatch.setattr(mp.psutil, "swap_memory", _no_swap)
    reading = provider.read()
    assert reading.level is None
    assert reading.damage_total_bytes is None
    assert reading.unavailable


def test_linux_anonymous_memory_is_read_from_smaps_rollup(tmp_path, monkeypatch):
    provider, _, _, proc, _ = _linux(tmp_path, monkeypatch)
    (proc / "42").mkdir()
    (proc / "42" / "smaps_rollup").write_text(
        "55d0-7ffd ---p 00000000 00:00 0 [rollup]\n"
        "Rss:             1000000 kB\n"
        "Anonymous:        600000 kB\n"
        "Swap:              10000 kB\n"
    )
    assert provider.anonymous(42) == 610_000 * 1024
    assert provider.anonymous(43) is None


@pytest.mark.parametrize(
    "limit, expected",
    [(4 * GIB, 4 * GIB), ("max", 16 * GIB), (None, 16 * GIB), (64 * GIB, 16 * GIB)],
)
def test_linux_total_ram_honours_only_a_finite_cgroup_limit(tmp_path, monkeypatch, limit, expected):
    provider, *_ = _linux(tmp_path, monkeypatch, cgroup_max=limit)
    monkeypatch.setattr(mp.psutil, "virtual_memory", lambda: SimpleNamespace(total=16 * GIB))
    assert provider.total_ram_bytes() == expected


# ===================== Windows =====================


class _FakeWindows(mp._WindowsProvider):
    def __init__(self, *, commit=0.5, pages=0):
        super().__init__()
        self.commit = commit
        self.pages = pages

    def _page_file(self):
        if self.commit is None:
            raise OSError("GlobalMemoryStatusEx failed")
        total = 32 * GIB
        return int(total * (1 - self.commit)), total

    def _pdh_pages_output(self):
        if self.pages is None:
            raise OSError("PdhAddEnglishCounterW failed")
        return self.pages


@pytest.mark.parametrize(
    "commit, level", [(0.5, 1), (0.84, 1), (0.85, 2), (0.92, 2), (0.93, 4), (0.99, 4)]
)
def test_windows_tiers_on_the_commit_charge(commit, level):
    assert _FakeWindows(commit=commit).read().level == level


def test_windows_pagefile_off_shape_still_tiers():
    # Without a page file the commit limit is the RAM: the same ratio.
    provider = _FakeWindows(commit=0.9)
    assert provider.read().level == 2


def test_windows_damage_is_the_raw_pages_output_counter_in_bytes():
    provider = _FakeWindows(pages=100)
    assert provider.read().damage_total_bytes == 100 * 4096
    provider.pages = 350
    assert provider.read().damage_total_bytes == 350 * 4096


def test_windows_without_pdh_keeps_its_tiers_and_warns_once(caplog):
    provider = _FakeWindows(commit=0.95, pages=None)
    with caplog.at_level(logging.WARNING, logger="erudi"):
        first = provider.read()
        second = provider.read()
    assert first.level == second.level == 4
    assert first.damage_total_bytes is None and second.damage_total_bytes is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "pressure level alone" in warnings[0].getMessage()


def test_windows_commit_unreadable_has_no_level():
    reading = _FakeWindows(commit=None).read()
    assert reading.level is None
    assert "GlobalMemoryStatusEx" in reading.unavailable


def test_windows_anonymous_memory_is_the_private_commit(monkeypatch):
    monkeypatch.setattr(
        mp.psutil,
        "Process",
        lambda pid: SimpleNamespace(memory_info=lambda: SimpleNamespace(private=2 * GIB)),
    )
    assert _FakeWindows().anonymous(9) == 2 * GIB


@pytest.mark.skipif(sys.platform != "win32", reason="the native Windows calls")
def test_windows_native_calls_answer():
    reading = mp._WindowsProvider().read()
    assert reading.level in (1, 2, 4)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the native Linux files")
def test_linux_native_read_never_raises():
    reading = mp._LinuxProvider().read()
    assert reading.level in (None, 1, 2, 4)


# ===================== the QA seam =====================


@pytest.mark.parametrize("forced, level", [("warning", 2), ("critical", 4)])
def test_the_seam_forces_the_tier_and_synthetic_damage(monkeypatch, forced, level):
    class _Calm:
        name = "calm"

        def read(self):
            return mp.PressureReading(level=1, damage_total_bytes=None)

    monkeypatch.setattr(mp, "_PROVIDER", _Calm())
    monkeypatch.setenv("ERUDI_FORCE_MEMORY_PRESSURE", forced)
    first = mp.read_pressure()
    second = mp.read_pressure()
    assert first.level == second.level == level
    assert second.damage_total_bytes - first.damage_total_bytes == LLAMA_WARNING_DAMAGE_MIN_BYTES


def test_an_invalid_seam_value_is_one_warning_and_ignored(monkeypatch, caplog):
    class _Calm:
        name = "calm"

        def read(self):
            return mp.PressureReading(level=1, damage_total_bytes=7)

    monkeypatch.setattr(mp, "_PROVIDER", _Calm())
    monkeypatch.setenv("ERUDI_FORCE_MEMORY_PRESSURE", "panic")
    with caplog.at_level(logging.WARNING, logger="erudi"):
        assert mp.read_pressure() == mp.PressureReading(level=1, damage_total_bytes=7)
        mp.read_pressure()
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_the_seam_does_not_reach_the_mlx_sensors(monkeypatch):
    # MLX reads process_footprint directly: the seam acts inside this module.
    monkeypatch.setenv("ERUDI_FORCE_MEMORY_PRESSURE", "critical")
    monkeypatch.setattr(process_footprint, "_on_macos", lambda: False)
    assert process_footprint.pressure_level() is None


# ===================== the module surface =====================


def test_the_module_surface_delegates_to_the_provider(monkeypatch):
    class _Fake:
        name = "fake"

        def anonymous(self, pid):
            return pid * 20

        def total_ram_bytes(self):
            return 8 * GIB

    monkeypatch.setattr(mp, "_PROVIDER", _Fake())
    assert mp.child_anonymous_bytes(3) == 60
    assert mp.effective_total_ram_bytes() == 8 * GIB
    assert mp.child_anonymous_bytes(None) is None
    assert not hasattr(mp, "child_footprint_bytes")


def test_a_failing_counter_reads_as_none(monkeypatch):
    class _Broken:
        name = "broken"

        def anonymous(self, pid):
            raise PermissionError("denied")

        def total_ram_bytes(self):
            raise OSError("no")

    monkeypatch.setattr(mp, "_PROVIDER", _Broken())
    assert mp.child_anonymous_bytes(3) is None
    assert mp.effective_total_ram_bytes() is None


def test_the_module_imports_no_native_library_at_import():
    source = mp.__file__
    with open(source, encoding="utf-8") as handle:
        top = handle.read().split("\nclass ", 1)[0]
    assert "windll" not in top
