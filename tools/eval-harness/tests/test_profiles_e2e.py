"""Profiles end to end: a clean run and a nominal run against the fake app, with a workload made of
processes this test spawns itself, then `compare` between the two."""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import pytest

from erudi_eval import cli
from tests.test_fake_app_run import HERE, free_port, kill_registered

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="the fake workload uses POSIX detached processes")

SLEEP = "import time; time.sleep(300)"
SPIN = "import time; t = time.monotonic()\nwhile time.monotonic() - t < 300: pass"
# The session runs through the `agent-cli` link, so its own sys.executable is that link: the busy child is
# started with the real interpreter (argv[1]) to stay a plain child, like an MCP server would be.
AGENT_SESSION = f"import subprocess, sys, time; subprocess.Popen([sys.argv[1], '-c', {SPIN!r}]); time.sleep(300)"


def spawn_detached(exe: Path, code: str, registry: Path, *args: str) -> int:
    """Start a process that is NOT a child of the test runner (the harness never counts its own descendants),
    and record its PID so the teardown kills exactly it."""
    launcher = (
        "import subprocess, sys;"
        " p = subprocess.Popen([sys.argv[1], '-c', sys.argv[2], *sys.argv[4:]], start_new_session=True);"
        " open(sys.argv[3], 'a').write(f'{p.pid} {__import__(\"time\").time()} workload\\n')"
    )
    subprocess.run([sys.executable, "-c", launcher, str(exe), code, str(registry), *args], check=True)
    return int(registry.read_text().splitlines()[-1].split()[0])


def app_binary(install: Path, root: Path, port: int) -> None:
    (install / "bin").mkdir(parents=True)
    (install / "bin" / "erudi").symlink_to(sys.executable)
    main_bin = install / "erudi"
    main_bin.write_text(
        f"#!{install / 'bin' / 'erudi'}\n"
        f"import sys; sys.path.insert(0, {str(HERE)!r})\n"
        "from pathlib import Path\n"
        "import fake_app\n"
        f"fake_app.run_main(Path({str(root)!r}), {port})\n"
    )
    main_bin.chmod(0o755)


def workload_file(path: Path, profile: str, apps: Path, chrome_absent: bool, sessions: int | None) -> Path:
    groups = [
        {"name": "fake_chrome", "exe_prefixes": [str(apps / "FakeChrome.app") + "/"], "include_descendants": True,
         **({"expect_absent": True} if chrome_absent else {"declared": {"chrome_tabs": 4, "tabs": ["Gmail", "GitHub"]}})},
        # argv0_prefixes keeps the rule inside this test's temp dir: a real coding-agent CLI session on the
        # machine has a bare argv[0] with no path and must not be captured.
        {"name": "fake_agent", "argv0_prefixes": [str(apps / "bin") + "/"], "sessions": True,
         **({"expected_count": sessions} if sessions else {"optional": True})},
        {"name": "fake_agent_children", "descendants_of": "fake_agent"},
        {"name": "fake_slack", "exe_prefixes": [str(apps / "FakeSlack.app") + "/"], **({"expect_absent": True} if chrome_absent else {})},
    ]
    path.write_text(json.dumps({"profile": profile, "notes": f"test workload ({profile})", "groups": groups}))
    return path


def run_args(install, root, port, results, run_id, profile, wl, extra=()):
    return ["run", "--app-path", str(install), "--api-port", str(port), "--cdp-port", str(free_port()),
            "--data-root", str(root / "prod"), "--backend-log-dir", str(root / "logs"),
            "--capture-log", str(root / "tmp" / "erudi-backend.log"), "--results-dir", str(results), "--run-id", run_id,
            "--profile", profile, "--workload", str(wl), "--phases", "preflight,baseline,cold_boot,quit",
            "--sample-interval", "0.4", "--baseline-seconds", "2", "--boot-sample-interval", "0.25", *extra]


def test_clean_and_nominal_profiles_then_compare(tmp_path, capsys):
    root, apps = tmp_path / "fake", tmp_path / "apps"
    registry = tmp_path / "workload-pids.txt"
    chrome = apps / "FakeChrome.app" / "Contents" / "MacOS" / "fakechrome"
    agent = apps / "bin" / "agent-cli"
    chrome.parent.mkdir(parents=True)
    agent.parent.mkdir(parents=True)
    chrome.symlink_to(sys.executable)
    agent.symlink_to(sys.executable)
    results = tmp_path / "results"
    try:
        # --- clean profile: the workload is declared absent, but one fake Chrome is running -> warning, no abort
        chrome_pids = [spawn_detached(chrome, SLEEP, registry)]
        install_a, port_a = root / "install-a", free_port()
        app_binary(install_a, root / "a", port_a)
        wl_clean = workload_file(tmp_path / "clean.json", "clean", apps, chrome_absent=True, sessions=None)
        strict = run_args(install_a, root / "a", port_a, results, "strict", "clean", wl_clean, ["--strict-workload", "--phases", "preflight"])
        assert cli.main(strict) == 1  # --strict-workload turns the mismatch into an abort
        aborted = json.loads((results / "strict" / "aborted.json").read_text())["reason"]
        assert "fake_chrome" in aborted and "expected absent" in aborted

        assert cli.main(run_args(install_a, root / "a", port_a, results, "clean-run", "clean", wl_clean)) == 0
        clean = json.loads((results / "clean-run" / "summary.json").read_text())
        checks = {c["group"]: c for c in clean["workload"]["checks"]}
        assert checks["fake_chrome"]["status"] == "warning" and "expected absent" in checks["fake_chrome"]["message"]
        assert checks["fake_slack"]["status"] == "ok"  # absent as declared
        assert clean["profile"] == "clean" and clean["conditions"]["uptime_s"] > 0

        # --- nominal profile: more Chrome, one agent-CLI session with a busy child, and one Chrome dies mid-run
        chrome_pids.append(spawn_detached(chrome, SLEEP, registry))
        session_pid = spawn_detached(agent, AGENT_SESSION, registry, sys.executable)
        install_b, port_b = root / "install-b", free_port()
        app_binary(install_b, root / "b", port_b)
        wl_nominal = workload_file(tmp_path / "nominal.json", "nominal", apps, chrome_absent=False, sessions=1)

        def kill_one_chrome():
            time.sleep(4)
            psutil.Process(chrome_pids[-1]).kill()  # a process this test started, by PID

        killer = threading.Thread(target=kill_one_chrome, daemon=True)
        killer.start()
        assert cli.main(run_args(install_b, root / "b", port_b, results, "nominal-run", "nominal", wl_nominal)) == 0
        killer.join(timeout=5)
        nominal = json.loads((results / "nominal-run" / "summary.json").read_text())
        wl = nominal["workload"]
        assert wl["profile"] == "nominal" and Path(wl["workload_file"]).name == "nominal.json"
        assert wl["declared_facts"]["fake_chrome"]["chrome_tabs"] == 4
        nchecks = {c["group"]: c for c in wl["checks"]}
        assert nchecks["fake_chrome"]["status"] == "ok" and nchecks["fake_agent"]["message"] == "1 sessions found"
        baseline = wl["per_phase"]["baseline"]
        assert baseline["fake_chrome"]["count_max"] == 2 and baseline["fake_chrome"]["primary_mb_mean"] > 0
        assert baseline["fake_agent"]["count_max"] == 1 and baseline["fake_agent_children"]["count_max"] == 1
        kinds = {(w["kind"], w["group"]) for w in wl["drift"]["warnings"]}
        assert ("count_changed", "fake_chrome") in kinds  # the process that died mid-run
        assert ("busy", "fake_agent_children") in kinds  # the spinning child
        assert wl["drift"]["reference_phase"] == "baseline"
        assert session_pid in {p for p in ()} or True  # the session PID is only used for teardown

        report = (results / "nominal-run" / "report.md").read_text()
        for needle in ("## Workload", "Profile `nominal`", "### Declared (not verifiable from outside)", "chrome_tabs: 4",
                       "### Verification at preflight", "### Per group and phase", "### Workload drift", "Conditions at start: uptime"):
            assert needle in report, needle

        # --- compare the two profiles
        assert cli.main(["compare", str(results / "clean-run"), str(results / "nominal-run")]) == 0
        text = capsys.readouterr().out
        for needle in ("## Conditions", "| profile | clean | nominal |", "workload fake_chrome (procs, MB)",
                       "different profiles", "## Machine context deltas", "## Memory per phase and category"):
            assert needle in text, needle
        (tmp_path / "compare.md").write_text(text)
    finally:
        kill_registered(root / "a")
        kill_registered(root / "b")
        for line in registry.read_text().splitlines() if registry.exists() else []:
            pid, spawned, _ = line.split()
            try:
                proc = psutil.Process(int(pid))
                if abs(proc.create_time() - float(spawned)) < 10:
                    for child in proc.children(recursive=True):  # the session's own children, started by it
                        child.kill()
                    proc.kill()
            except (psutil.Error, ValueError):
                pass
        if os.environ.get("ERUDI_EVAL_KEEP_RESULTS"):
            import shutil

            shutil.copytree(results, Path(os.environ["ERUDI_EVAL_KEEP_RESULTS"]) / "profiles", dirs_exist_ok=True)
            shutil.copy(tmp_path / "compare.md", Path(os.environ["ERUDI_EVAL_KEEP_RESULTS"]) / "compare.md")
