import json
from pathlib import Path

import pytest

from erudi_eval.discovery import (
    DiscoveryContext,
    ProcInfo,
    classify_snapshot,
    find_main_pid,
)
from erudi_eval.util import redact_cmdline

FIXTURES = Path(__file__).parent / "fixtures"


def load(name):
    data = json.loads((FIXTURES / name).read_text())
    ctx = DiscoveryContext(
        install_dirs=tuple(data["context"]["install_dirs"]),
        data_root=data["context"]["data_root"],
        api_port=data["context"]["api_port"],
    )
    procs = [ProcInfo.from_dict(d) for d in data["processes"]]
    expect = {d["pid"]: d["expect"] for d in data["processes"]}
    return ctx, procs, expect


@pytest.mark.parametrize("fixture", ["processes_macos_recorded.json", "processes_macos.json", "processes_windows.json", "processes_linux.json"])
def test_fixture_classification(fixture):
    ctx, procs, expect = load(fixture)
    _, classified = classify_snapshot(procs, ctx)
    got = {pid: c.category for pid, c in classified.items()}
    for pid, category in expect.items():
        assert got.get(pid) == category, f"pid {pid}: expected {category}, got {got.get(pid)}"
    roles = {d["pid"]: d["expect_role"] for d in json.loads((FIXTURES / fixture).read_text())["processes"] if "expect_role" in d}
    for pid, role in roles.items():
        assert classified[pid].role == role, f"pid {pid}: expected role {role}, got {classified[pid].role}"


def test_macos_mlx_child_and_resource_tracker_roles():
    ctx, procs, _ = load("processes_macos.json")
    _, classified = classify_snapshot(procs, ctx)
    assert classified[512].role == "mlx_child"
    assert classified[511].role == "mp_helper"
    assert classified[520].role == "postmaster"  # re-parented to launchd, found by path
    assert classified[521].role == "pg_child"


def test_developer_pgserver_cluster_is_not_ours_even_without_install_dir():
    ctx, procs, _ = load("processes_macos.json")
    ctx = DiscoveryContext(data_root=ctx.data_root, main_pid=500)
    _, classified = classify_snapshot(procs, ctx)
    assert 600 not in classified and 601 not in classified
    assert classified[520].category == "database"  # still matched through the data root


def test_explicit_main_pid_wins_and_backend_by_port():
    procs = [
        ProcInfo(10, 1, "python", "/usr/bin/python3", ("/opt/fake/Erudi", "fake_main.py")),  # argv[0] names the Erudi binary
        ProcInfo(11, 10, "python", "/usr/bin/python3", ("/usr/bin/python3", "fake_backend.py", "--port", "28000")),
        ProcInfo(12, 11, "python", "/usr/bin/python3", ("/usr/bin/python3", "-c", "from multiprocessing.spawn import spawn_main", "--multiprocessing-fork")),
        ProcInfo(13, 10, "python", "/usr/bin/python3", ("/usr/bin/python3", "-c", "sleep", "--type=renderer")),
        ProcInfo(14, 1, "python", "/usr/bin/python3", ("/usr/bin/python3", "unrelated.py", "--port", "28000")),
    ]
    ctx = DiscoveryContext(main_pid=10, api_port=28000)
    main, classified = classify_snapshot(procs, ctx)
    assert main == 10
    assert {pid: c.category for pid, c in classified.items()} == {
        10: "electron_main",
        11: "backend",
        12: "inference",
        13: "electron_renderer",
    }


def test_no_main_found_without_erudi_process():
    procs = [ProcInfo(2, 1, "zsh", "/bin/zsh", ("/bin/zsh",))]
    assert find_main_pid(procs, DiscoveryContext(install_dirs=("/Applications/Erudi.app",))) is None


def test_pid_reuse_is_not_treated_as_child():
    procs = [
        ProcInfo(10, 1, "Erudi", "/Applications/Erudi.app/Contents/MacOS/Erudi", ("/Applications/Erudi.app/Contents/MacOS/Erudi",), 1000.0),
        ProcInfo(20, 10, "other", "/bin/other", ("/bin/other",), 10.0),
    ]
    _, classified = classify_snapshot(procs, DiscoveryContext(install_dirs=("/Applications/Erudi.app",)))
    assert 20 not in classified


def test_redact_cmdline_hides_api_key():
    cmd = ["llama-server", "--port", "27201", "--api-key", "SECRET", "--api-key=S2", "postgresql://u:x@h/db?password=abc"]
    out = redact_cmdline(cmd)
    assert "SECRET" not in " ".join(out) and "S2" not in " ".join(out) and "abc" not in " ".join(out)
    assert out[:4] == ["llama-server", "--port", "27201", "--api-key"]


def test_explicit_main_pid_must_be_an_erudi_main_binary():
    """--main-pid is what `quit` signals: a PID that is not this install's Erudi binary is ignored."""
    ctx, procs, _ = load("processes_macos_recorded.json")
    auto = find_main_pid(procs, ctx)
    assert auto is not None
    stranger = next(p for p in procs if p.pid != auto and "erudi" not in {s.lower() for s in (p.name,)})
    wrong = DiscoveryContext(install_dirs=ctx.install_dirs, data_root=ctx.data_root, api_port=ctx.api_port, main_pid=stranger.pid)
    assert find_main_pid(procs, wrong) == auto
    right = DiscoveryContext(install_dirs=ctx.install_dirs, data_root=ctx.data_root, api_port=ctx.api_port, main_pid=auto)
    assert find_main_pid(procs, right) == auto


def test_orphaned_backend_never_elects_launchd_as_main():
    """The main died first (every quit traverses that window): the backend is
    re-parented to launchd/init. The parent-of-backend fallback must not return
    pid 1 -- membership would adopt every user process and quit would signal
    outside the app tree."""
    procs = [
        ProcInfo(1, 0, "launchd", "/sbin/launchd", ("/sbin/launchd",)),
        ProcInfo(30, 1, "backend", "/Applications/Erudi.app/Contents/Resources/backend/backend", ("backend",)),
        ProcInfo(40, 1, "Safari", "/Applications/Safari.app/Contents/MacOS/Safari", ("Safari",)),
    ]
    ctx = DiscoveryContext(install_dirs=("/Applications/Erudi.app",))
    assert find_main_pid(procs, ctx) is None
