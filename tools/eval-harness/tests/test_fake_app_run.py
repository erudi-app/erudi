"""End-to-end: `run --attach` against tests/fake_app.py (no CDP endpoint, so UI phases skip)."""

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from erudi_eval import cli
from erudi_eval.api import ErudiApi

HERE = Path(__file__).parent
HARNESS = HERE.parent

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="fake app uses POSIX process groups and FIFOs")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def kill_registered(root: Path) -> list[int]:
    """Kill exactly the PIDs the fake app registered in DIR/pids.txt, after checking each one is still that
    process (spawn time, or for the bash copies an executable inside DIR). Never scans the process list."""
    killed = []
    registry = root / "pids.txt"
    if not registry.exists():
        return killed
    for line in registry.read_text().splitlines():
        pid_s, spawned_s, _role = line.split(" ", 2)
        try:
            proc = psutil.Process(int(pid_s))
            spawned = float(spawned_s)
            same = abs(proc.create_time() - spawned) < 5 if spawned else proc.exe().startswith(str(root))
            if same and proc.status() != psutil.STATUS_ZOMBIE:
                proc.kill()
                killed.append(proc.pid)
        except (psutil.Error, OSError, ValueError):
            pass
    return killed


def make_fake_postgres(install: Path) -> bool:
    """A real executable living at .../pginstall/bin/postgres (a copy of bash), so discovery matches on the exe path."""
    target = install / "resources" / "backend" / "_internal" / "pgserver" / "pginstall" / "bin" / "postgres"
    target.parent.mkdir(parents=True)
    shutil.copyfile("/bin/bash", target)
    target.chmod(0o755)
    if sys.platform == "darwin":
        # A copied Apple binary is killed on exec until re-signed ad hoc (the copy only, in the test tmp dir).
        if subprocess.run(["codesign", "-f", "-s", "-", str(target)], capture_output=True).returncode != 0:
            target.unlink()
            return False
    return True


@pytest.fixture
def fake_app(tmp_path):
    root = tmp_path / "fake"
    install = root / "install"
    install.mkdir(parents=True)
    # The main stand-in runs through a link named after the app binary: argv[0] (what `ps` shows) is
    # the install's `erudi`, which the check guarding the PID `quit` signals requires; Python follows
    # the link to find its own stdlib.
    (install / "erudi").symlink_to(sys.executable)
    has_pg = make_fake_postgres(install)
    port = free_port()
    main = subprocess.Popen([str(install / "erudi"), str(HERE / "fake_app.py"), "main", "--root", str(root), "--api-port", str(port)])
    api = ErudiApi(port=port)
    deadline = time.monotonic() + 30
    while not api.health_ok():
        assert time.monotonic() < deadline, "fake app did not come up"
        time.sleep(0.2)
    yield {"root": root, "install": install, "port": port, "main": main, "has_pg": has_pg}
    # Teardown only kills PIDs this fixture's fake app registered (normally all gone after the harness quit).
    kill_registered(root)
    if main.poll() is None:
        main.wait(timeout=10)


def test_run_attach_against_fake_app(fake_app, tmp_path):
    root, port = fake_app["root"], fake_app["port"]
    results = tmp_path / "results"
    # The stress corpus is not committed (real runs point it at large local
    # documents), so the test builds its own: the committed standard corpus
    # plus two generated stress files, in a temp corpus dir.
    corpus = tmp_path / "corpus"
    shutil.copytree(HERE.parent / "corpus" / "standard", corpus / "standard")
    (corpus / "stress").mkdir()
    for name in ("stress_a.txt", "stress_b.txt"):
        (corpus / "stress" / name).write_text(f"Generated stress document {name}. " * 200)
    argv = [
        "run", "--attach",
        "--corpus-dir", str(corpus),
        "--app-path", str(fake_app["install"]),
        "--main-pid", str(fake_app["main"].pid),
        "--api-port", str(port),
        "--cdp-port", str(free_port()),  # nothing listens there: UI phases must skip with a reason
        "--data-root", str(root / "prod"),
        "--backend-log-dir", str(root / "logs"),
        "--capture-log", str(root / "tmp" / "erudi-backend.log"),
        "--results-dir", str(results),
        "--run-id", "fake-run",
        "--model-link", "fake/Qwen3-4B-FAKE",
        "--disk-headroom-gb", "0.01",
        "--idle-seconds", "1", "--settle-seconds", "0.3", "--sample-interval", "0.5",
        "--warm-turns", "2", "--long-turns", "4", "--unload-timeout", "30",
        "--opt-in", "kb_stress,web_search,arena",
        "--cleanup",
    ]
    assert cli.main(argv) == 0
    run_dir = results / "fake-run"
    if os.environ.get("ERUDI_EVAL_KEEP_RESULTS"):  # keep the run for inspection (pytest temp dirs are removed)
        shutil.copytree(run_dir, Path(os.environ["ERUDI_EVAL_KEEP_RESULTS"]) / "fake-run", dirs_exist_ok=True)

    for name in ("system.json", "config.json", "samples.jsonl", "events.jsonl", "renderer.jsonl", "phases.json", "summary.json", "report.md"):
        assert (run_dir / name).exists(), name
    assert len(list((run_dir / "storage").glob("*.json"))) >= 5
    assert (run_dir / "logs" / "erudi-backend.log").exists() and (run_dir / "logs" / "backend.log").exists()

    phases = {p["name"]: p for p in json.loads((run_dir / "phases.json").read_text())}
    expected_ok = ["preflight", "idle_after_boot", "model_ready", "chat_cold", "chat_warm", "long_conversation_render", "embedding_model",
                   "kb_ingest", "kb_query", "kb_stress", "web_search", "arena", "idle_unload", "cleanup", "quit"]
    for name in expected_ok:
        assert phases[name]["status"] == "ok", (name, phases[name])
    assert phases["cold_boot"]["status"] == "skipped" and "--attach" in phases["cold_boot"]["reason"]
    assert phases["baseline"]["status"] == "skipped" and "--attach" in phases["baseline"]["reason"]
    for name in ("ui_tour", "ui_stream"):
        assert phases[name]["status"] == "skipped" and "no CDP endpoint" in phases[name]["reason"]
    assert phases["model_ready"]["data"]["downloaded_by_harness"] is True
    assert phases["long_conversation_render"]["data"]["user_turns"] == 4
    assert phases["quit"]["data"]["survivors"]["plus_30s"] == {}

    summary = json.loads((run_dir / "summary.json").read_text())
    chat = summary["memory"]["chat_warm"]
    assert chat["electron_main"]["mean_mb"] > 0 and chat["electron_renderer"]["mean_mb"] > 0 and chat["electron_gpu"]["mean_mb"] > 0
    assert chat["backend"]["mean_mb"] > 0
    assert summary["memory"]["arena"]["inference"]["peak_mb"] > 30  # the 40 MB ballast child
    if fake_app["has_pg"]:
        assert chat["database"]["mean_mb"] > 0
    roles = {p["role"] for s in map(json.loads, (run_dir / "samples.jsonl").read_text().splitlines()) for p in s.get("processes", [])}
    assert {"main", "renderer", "gpu", "backend", "mlx_child", "mp_helper"} <= roles
    if fake_app["has_pg"]:
        assert {"postmaster", "pg_child"} <= roles
    turns = summary["turns"]
    assert sum(1 for t in turns if t["label"] == "warm") == 2 and turns[0]["cold"] is True
    assert all(t.get("done") for t in turns if t["label"] not in ("ui", "arena"))
    kb = next(e for e in map(json.loads, (run_dir / "events.jsonl").read_text().splitlines()) if e["type"] == "kb_evidence")
    assert all(r["kb_searched"] for r in kb["rows"])

    report = (run_dir / "report.md").read_text()
    for needle in ("## Memory per phase", "## Turns", "## Storage", "Storage deltas per phase", "models/Qwen3_4B_(fake)", "no CDP endpoint", "Survivors after quit",
                   "## Machine context", "### Swap and paging", "when available memory was lowest"):
        assert needle in report, needle
    assert fake_app["main"].wait(timeout=10) == 0  # the graceful quit reached the main process
    assert not (root / "prod" / "data" / "models" / "Qwen3_4B_(fake)").exists() or not any((root / "prod" / "data" / "models" / "Qwen3_4B_(fake)").iterdir())


def test_preflight_refuses_when_app_already_running(fake_app, tmp_path):
    """Without --attach, an answering API is a refusal: the harness never runs on top of a live app."""
    argv = ["run", "--app-path", str(fake_app["install"]), "--api-port", str(fake_app["port"]), "--cdp-port", str(free_port()),
            "--data-root", str(fake_app["root"] / "prod"), "--results-dir", str(tmp_path / "r"), "--run-id", "refused",
            "--capture-log", str(fake_app["root"] / "tmp" / "erudi-backend.log"), "--backend-log-dir", str(fake_app["root"] / "logs")]
    assert cli.main(argv) == 1
    phases = json.loads((tmp_path / "r" / "refused" / "phases.json").read_text())
    assert phases[0]["status"] == "failed" and "already running" in phases[0]["reason"]
    assert fake_app["main"].poll() is None  # not quit, not touched


def test_baseline_then_cold_boot_launch_and_quit(tmp_path):
    """Without --attach: sample the machine with no app, launch the (fake) app binary with fast boot sampling,
    build the boot timeline with app memory at each mark, quit it."""
    root = tmp_path / "fake"
    install = root / "install"
    install.mkdir(parents=True)
    port = free_port()
    # `install/erudi` is what the harness execs (with only --remote-debugging-port). It is a Python script whose
    # interpreter is a link named `erudi` inside the install, so the process's argv[0] is the install's `erudi`
    # binary (what find_main_pid requires before `quit` may signal it) and the PID is the fake main itself.
    (install / "bin").mkdir()
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
    results = tmp_path / "results"
    argv = ["run", "--app-path", str(install), "--api-port", str(port), "--cdp-port", str(free_port()),
            "--data-root", str(root / "prod"), "--backend-log-dir", str(root / "logs"), "--capture-log", str(root / "tmp" / "erudi-backend.log"),
            "--results-dir", str(results), "--run-id", "boot", "--phases", "preflight,baseline,cold_boot,idle_after_boot,quit",
            "--sample-interval", "0.5", "--baseline-seconds", "2", "--boot-sample-interval", "0.25", "--idle-seconds", "1"]
    try:
        assert not (root / "pids.txt").exists()  # nothing of the app exists before the run
        assert cli.main(argv) == 0
        run_dir = results / "boot"
        if os.environ.get("ERUDI_EVAL_KEEP_RESULTS"):
            shutil.copytree(run_dir, Path(os.environ["ERUDI_EVAL_KEEP_RESULTS"]) / "boot", dirs_exist_ok=True)
        phases = {p["name"]: p for p in json.loads((run_dir / "phases.json").read_text())}
        for name in ("baseline", "cold_boot", "idle_after_boot", "quit"):
            assert phases[name]["status"] == "ok", phases[name]
        labels = [r["label"] for r in phases["cold_boot"]["data"]["timeline"]]
        assert labels[0] == "harness_launch" and "starting" in labels and "phase:running_migrations" in labels
        assert labels.index("ready") < labels.index("health_200")
        assert "cdp_answering" not in labels  # no CDP on that port, recorded as absent rather than invented
        assert phases["quit"]["data"]["survivors"]["plus_30s"] == {}

        samples = [json.loads(line) for line in (run_dir / "samples.jsonl").read_text().splitlines()]
        baseline = [s for s in samples if s["phase"] == "baseline"]
        assert len(baseline) >= 3 and all(s["processes"] == [] for s in baseline)  # the app did not exist yet
        assert all(s["interval_target_s"] == 0.5 for s in baseline)
        events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
        from erudi_eval.util import parse_iso
        t_launch = parse_iso(next(e for e in events if e["type"] == "launched")["ts"])
        t_timeline = parse_iso(next(e for e in events if e["type"] == "boot_timeline")["ts"])
        boot = [s for s in samples if s["phase"] == "cold_boot"]
        during = [s for s in boot if t_launch <= s["t"] <= t_timeline]
        assert len(during) >= 4 and all(s["interval_target_s"] == 0.25 for s in during)
        after = [s["t"] for s in boot if s["t"] >= t_launch - 0.01]
        max_cost = max(s["sample_cost_ms"] for s in boot) / 1000
        assert after and after[0] - t_launch <= 0.25 + max_cost + 0.05  # first sample after exec within one boot interval
        assert all("sample_cost_ms" in s and "harness" in s and "available_bytes" in s["system"] for s in samples if "system" in s)
        idle = [s for s in samples if s["phase"] == "idle_after_boot"]
        assert idle and all(s["interval_target_s"] == 0.5 for s in idle)  # normal rate restored

        summary = json.loads((run_dir / "summary.json").read_text())
        rows = {r["label"]: r for r in summary["boot_memory"]["rows"]}
        assert rows["ready"]["totals_mb"]["app_total"] > 0 and rows["ready"]["totals_mb"]["electron_main"] > 0
        assert summary["boot_memory"]["peak"]["totals_mb"]["app_total"] > 0
        assert "baseline" in summary["machine_context"]["per_phase"]
        report = (run_dir / "report.md").read_text()
        for needle in ("## Boot timeline", "Boot sampling: target 0.25 s", "Peak during boot", "## Machine context", "### Swap and paging",
                       "### Baseline vs phase", "Top other processes at baseline", "| ready |"):
            assert needle in report, needle
    finally:
        kill_registered(root)


def test_model_ready_matches_an_installed_model_in_both_row_shapes(fake_app, tmp_path):
    """After a download Erudi rewrites the row's link to the local path; the harness must not download again."""
    root, port = fake_app["root"], fake_app["port"]
    results = tmp_path / "results"
    api = ErudiApi(port=port)

    def run(run_id):
        argv = ["run", "--attach", "--app-path", str(fake_app["install"]), "--main-pid", str(fake_app["main"].pid),
                "--api-port", str(port), "--cdp-port", str(free_port()), "--data-root", str(root / "prod"),
                "--backend-log-dir", str(root / "logs"), "--capture-log", str(root / "tmp" / "erudi-backend.log"),
                "--results-dir", str(results), "--run-id", run_id, "--model-link", "fake/Qwen3-4B-FAKE",
                "--disk-headroom-gb", "0.01", "--phases", "preflight,model_ready", "--sample-interval", "0.5", "--leave-running"]
        assert cli.main(argv) == 0
        return {p["name"]: p for p in json.loads((results / run_id / "phases.json").read_text())}["model_ready"]

    first = run("dl")
    assert first["status"] == "ok" and first["data"]["downloaded_by_harness"] is True
    models = root / "prod" / "data" / "models"
    assert len(list(models.iterdir())) == 1
    installed = [m for m in api.get("/llms/local").body if not m.get("is_attached_to_kb")]
    assert installed[0]["link"].startswith(str(models))  # the row no longer carries the catalog link

    second = run("by-name")
    assert second["data"]["downloaded_by_harness"] is False and second["data"]["matched_by"] == "name"
    assert "local path" in second["data"]["note"]
    assert len(list(models.iterdir())) == 1  # no duplicate download

    assert api.put("/_test/link_mode", {"mode": "catalog"}).ok  # the other shape: the row kept the catalog link
    third = run("by-link")
    assert third["data"]["downloaded_by_harness"] is False and third["data"]["matched_by"] == "link"
    assert len(list(models.iterdir())) == 1
    if os.environ.get("ERUDI_EVAL_KEEP_RESULTS"):
        shutil.copytree(results, Path(os.environ["ERUDI_EVAL_KEEP_RESULTS"]) / "model-shapes", dirs_exist_ok=True)


def test_preflight_refuses_a_flavour_the_running_app_contradicts(fake_app, tmp_path):
    argv = ["run", "--attach", "--app-path", str(fake_app["install"]), "--main-pid", str(fake_app["main"].pid),
            "--api-port", str(fake_app["port"]), "--cdp-port", str(free_port()), "--data-root", str(fake_app["root"] / "prod"),
            "--capture-log", str(fake_app["root"] / "tmp" / "erudi-backend.log"), "--backend-log-dir", str(fake_app["root"] / "logs"),
            "--results-dir", str(tmp_path / "r"), "--run-id", "wrong-flavour", "--flavour", "win-cuda", "--phases", "preflight", "--leave-running"]
    assert cli.main(argv) == 1
    reason = json.loads((tmp_path / "r" / "wrong-flavour" / "aborted.json").read_text())["reason"]
    assert "win-cuda" in reason and "engine=" in reason
    assert fake_app["main"].poll() is None  # the app was left alone
