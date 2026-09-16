"""Command line: run, compare, selftest."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent.parent


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    from .phases import OPT_IN, PHASE_NAMES

    p = argparse.ArgumentParser(prog="erudi_eval.py", description="Memory and storage evaluation of the installed Erudi app.")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the scenario against the installed app")
    run.add_argument("--flavour", help="platform + inference backend: mac-mlx, win-cuda, win-cpu, linux-cuda, linux-cpu (default: guessed from the install)")
    run.add_argument("--profile", default="clean", help="measurement profile: clean (reference) or nominal (realistic background workload)")
    run.add_argument("--workload", help="workload file (default: workloads/<profile>-<os>.json when it exists)")
    run.add_argument("--strict-workload", action="store_true", help="abort when the declared workload does not match what is running")
    run.add_argument("--workload-drift-pct", type=float, default=25.0, help="memory drift of a workload group that raises a warning (default 25 %%)")
    run.add_argument("--workload-busy-cpu", type=float, default=20.0, help="mean CPU %% of a workload group over a phase that raises a warning (default 20)")
    run.add_argument("--attach", action="store_true", help="observe an app that is already running instead of launching it")
    run.add_argument("--app-path", help="install location (Erudi.app, install dir, Erudi.exe or AppImage)")
    run.add_argument("--phases", type=_csv, help=f"comma list of phases to run (default all): {','.join(PHASE_NAMES)}")
    run.add_argument("--skip", type=_csv, default=[], help="comma list of phases to skip")
    run.add_argument("--opt-in", type=_csv, default=[], help=f"enable opt-in phases: {','.join(OPT_IN)}")
    run.add_argument("--model-link", help="catalog link of the chat model (default per OS)")
    run.add_argument("--no-download", action="store_true", help="never download a model; skip phases that would need one")
    run.add_argument("--idle-seconds", type=float, default=60, help="idle_after_boot duration (default 60)")
    run.add_argument("--baseline-seconds", type=float, default=60, help="machine sampled before the app is launched (default 60)")
    run.add_argument("--boot-sample-interval", type=float, default=0.25, help="sampling interval during cold_boot (default 0.25 s)")
    run.add_argument("--warm-turns", type=int, default=5, help="turns in chat_warm (default 5)")
    run.add_argument("--long-turns", type=int, default=20, help="total user turns for long_conversation_render (default 20)")
    run.add_argument("--disk-headroom-gb", type=float, default=5.0, help="free disk required beyond a download's size (default 5)")
    run.add_argument("--cleanup", action="store_true", help="delete what the harness created (conversations, KB assistants, downloaded models)")
    run.add_argument("--leave-running", action="store_true", help="do not quit the app at the end")
    run.add_argument("--results-dir", type=Path, default=HARNESS_DIR / "results")
    run.add_argument("--sample-interval", type=float, default=1.0, help="process sampling interval in seconds (default 1)")
    adv = run.add_argument_group("advanced / testing")
    adv.add_argument("--api-port", type=int, default=27182)
    adv.add_argument("--cdp-port", type=int, default=9222)
    adv.add_argument("--main-pid", type=int, help="PID of the Electron main process (attach mode, when discovery cannot find it)")
    adv.add_argument("--data-root", help="override the backend data root (.../erudi/backend/prod)")
    adv.add_argument("--backend-log-dir", help="override the backend log dir")
    adv.add_argument("--capture-log", help="override the Electron stdout capture log path")
    adv.add_argument("--settle-seconds", type=float, help="replace every settle duration (tests only; breaks comparability)")
    adv.add_argument("--unload-timeout", type=float, default=660)
    adv.add_argument("--download-timeout", type=float, default=7200)
    adv.add_argument("--turn-timeout", type=float, default=900)
    adv.add_argument("--renderer-interval", type=float, default=5.0)
    adv.add_argument("--corpus-dir", type=Path, default=HARNESS_DIR / "corpus")
    adv.add_argument("--run-id")

    cmp_ = sub.add_parser("compare", help="per phase x category deltas between two run dirs")
    cmp_.add_argument("run_a", type=Path)
    cmp_.add_argument("run_b", type=Path)
    cmp_.add_argument("--out", type=Path, help="also write the table to this file")

    st = sub.add_parser("selftest", help="check which metrics this machine can read (no app needed)")
    st.add_argument("--workload", help="workload file to check against what is running (default: workloads/nominal-<os>.json)")
    st.add_argument("--profile", default="nominal", help="profile whose default workload file the selftest checks (default nominal)")
    return p


def cmd_run(args: argparse.Namespace) -> int:
    from .layout import resolve_layout
    from .phases import OPT_IN, PHASE_NAMES, Config, Run

    unknown = [n for n in (args.phases or []) + args.skip if n not in PHASE_NAMES] + [n for n in args.opt_in if n not in OPT_IN]
    if unknown:
        print(f"unknown phase name(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    cfg = Config(
        results_dir=args.results_dir, corpus_dir=args.corpus_dir, app_path=args.app_path, attach=args.attach,
        phases=args.phases, skip=args.skip, opt_in=args.opt_in, model_link=args.model_link, no_download=args.no_download,
        profile=args.profile, flavour=args.flavour, workload_path=args.workload, strict_workload=args.strict_workload,
        workload_drift_pct=args.workload_drift_pct, workload_busy_cpu=args.workload_busy_cpu,
        idle_seconds=args.idle_seconds, baseline_seconds=args.baseline_seconds, boot_sample_interval=args.boot_sample_interval, warm_turns=args.warm_turns, long_turns=args.long_turns,
        disk_headroom_gb=args.disk_headroom_gb, cleanup=args.cleanup, leave_running=args.leave_running,
        sample_interval=args.sample_interval, renderer_interval=args.renderer_interval, settle_seconds=args.settle_seconds,
        unload_timeout=args.unload_timeout, download_timeout=args.download_timeout, turn_timeout=args.turn_timeout,
        api_port=args.api_port, cdp_port=args.cdp_port, main_pid=args.main_pid, data_root=args.data_root,
        backend_log_dir=args.backend_log_dir, capture_log=args.capture_log,
    )
    try:
        run = Run(cfg, resolve_layout(args.app_path), run_id=args.run_id)
    except (ValueError, OSError) as e:  # a broken or mismatched workload file must not start a run
        print(f"workload: {e}", file=sys.stderr)
        return 2
    print(f"erudi-eval run {run.run_id} -> {run.dir}", flush=True)
    out = run.execute()
    for r in run.results:
        print(f"  {r.name:<26} {r.status:<8} {r.reason or ''}")
    print(f"report: {out / 'report.md'}")
    return 1 if (out / "aborted.json").exists() else 0


def cmd_compare(args: argparse.Namespace) -> int:
    from .compare import compare, load_summary, render

    a, b = load_summary(args.run_a), load_summary(args.run_b)
    text = render(a, b, compare(a, b))
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    import psutil

    from . import __version__, machine, memory, storage
    from .app_control import running_erudi
    from .cdp import CdpClient
    from .api import ErudiApi
    from .discovery import take_snapshot
    from .gpu import GpuReader
    from .layout import resolve_layout
    from .util import MB

    def show(label: str, value) -> None:
        print(f"  {label:<34} {value}")

    print(f"erudi-eval {__version__} selftest — {sys.platform}, Python {sys.version.split()[0]}, psutil {psutil.__version__}")
    show("primary metric", memory.PRIMARY_METRIC)
    show("OS peak metric", memory.PEAK_METRIC or "none (max of samples)")

    def describe(proc: psutil.Process, label: str) -> None:
        proc.cpu_percent(None)
        time.sleep(0.2)
        r = memory.read_process(proc)
        mb = {k: round(v / MB, 1) for k, v in r["metrics"].items() if k not in ("cpu_percent", "threads")}
        show(f"{label} metrics (MB)", mb)
        show(f"{label} cpu%/threads", (r["metrics"].get("cpu_percent"), r["metrics"].get("threads")))
        show(f"{label} unreadable", r["errors"] or "none")

    describe(psutil.Process(), "this process")
    child = subprocess.Popen([sys.executable, "-c", "import time; b = bytearray(80*1024*1024); time.sleep(30)"])
    try:
        time.sleep(1.0)
        describe(psutil.Process(child.pid), "child process (80 MB)")
    finally:
        child.kill()
        child.wait()

    layout = resolve_layout()
    reader = machine.MachineReader(disk_path=layout.data_root)
    reader.read()
    time.sleep(1.0)
    t = time.monotonic()
    block = reader.read()
    cost_ms = 1000 * (time.monotonic() - t)
    print("  machine context (one sample; MB unless stated):")
    skip = {"counters", "rates_per_s", "power_thermal", "errors", "vm_pages", "page_size"}
    for key, value in block.items():
        if key not in skip:
            show(f"  {key.replace('_bytes', '')}", round(value / MB) if key.endswith("_bytes") and isinstance(value, int) else value)
    show("  rates per s (1 s apart)", {k: v for k, v in block["rates_per_s"].items()})
    show("  power / thermal (every 10 s)", block["power_thermal"])
    show("  unreadable", block.get("errors") or "none")
    show("  static (system.json)", machine.static_info())
    show("  machine block cost", f"{cost_ms:.1f} ms")
    gpu = GpuReader()
    show("GPU source", f"{gpu.source}" + (f" ({gpu.error})" if gpu.source == "none" and gpu.error else ""))

    t = time.monotonic()
    procs, errors = take_snapshot()
    show("process snapshot", f"{len(procs)} processes, {len(errors)} with unreadable exe/cmdline, {1000 * (time.monotonic() - t):.0f} ms")

    # The cost of one full sample (discovery + per-process reads + machine context), as `run` records it.
    from .sampler import DiscoveryState, Sampler

    class _Discard:
        def write(self, record) -> None:
            pass

    sampler = Sampler(_Discard(), DiscoveryState(layout, 27182), interval=1.0, machine_reader=reader)
    costs, top_costs = [], []
    for i in range(6):
        sample = sampler.sample_once()
        (top_costs if "top_other" in sample else costs).append(sample["sample_cost_ms"])
        time.sleep(0.25)
    costs.sort()
    show("sample cost (6 samples)", f"median {costs[len(costs) // 2]:.1f} ms, max {costs[-1]:.1f} ms; samples that also list the top other processes (every 5 s): {', '.join(f'{c:.1f}' for c in top_costs)} ms")
    show("fastest holdable interval", f"~{max(costs) / 1000:.2f} s (boot default 0.25 s; the effective interval is recorded per sample)")
    top = machine.top_other_processes({os.getpid()}, limit=3)
    show("top other processes (RSS)", f"{top['total_rss_mb']} MB total, {top['unreadable']} unreadable; top 3: " + ", ".join(f"{r['name']} {r['rss_mb']}" for r in top["top"]))
    show("harness self-cost", {k: (round(v / MB, 1) if k == memory.PRIMARY_METRIC and v else v) for k, v in sample["harness"].items()})

    from . import workload as workload_mod

    wl_path = Path(args.workload) if getattr(args, "workload", None) else workload_mod.default_workload_path(HARNESS_DIR, getattr(args, "profile", "nominal"), layout.os_name)
    if wl_path is None:
        show("workload", "no workload file for this profile/OS")
    else:
        wl = workload_mod.load_workload(wl_path)
        t = time.monotonic()
        procs, _ = take_snapshot()
        erudi = set(running_erudi(procs, layout, 27182, {os.getpid()}))
        sampler.workload = wl
        sampler.measure_workload(procs, erudi)  # prime per-process CPU counters
        time.sleep(0.5)
        measured, assignment = sampler.measure_workload(procs, erudi)
        cost = 1000 * (time.monotonic() - t) - 500
        checks = workload_mod.verify_presence(assignment, wl)
        print(f"  workload {wl_path.name} (profile {wl.profile}), measured in {cost:.0f} ms including the process snapshot:")
        for c in checks:
            g = measured.get(c["group"], {})
            declared = f" declared {c['declared']}" if c["declared"] else ""
            show(f"  {c['group']:<22} {c['status']}", f"{c['message']}; {g.get('count')} procs, primary {g.get('primary_mb')} MB, rss {g.get('rss_mb')} MB, cpu {g.get('cpu_percent')} %{declared}")

    t = time.monotonic()
    u = storage.usage(HARNESS_DIR)
    show("storage walker (harness dir)", f"{u.bytes / MB:.1f} MB {storage.SIZE_KIND}, {u.logical_bytes / MB:.1f} MB logical, {u.files} files, {u.symlinks} symlinks, {u.errors} errors, {1000 * (time.monotonic() - t):.0f} ms")

    show("app install", f"{layout.app_path} ({'present' if layout.app_path and layout.app_path.exists() else 'not found'})")
    show("data root", f"{layout.data_root} ({'present' if layout.data_root.exists() else 'not found'})")
    show("stdout capture log", ", ".join(f"{p} ({'present' if p.exists() else 'absent'})" for p in layout.capture_logs))
    running = running_erudi(procs, layout, 27182, {os.getpid()})
    show("Erudi processes running", running or "none")
    show("API 127.0.0.1:27182", "answering" if ErudiApi().health_ok() else "not answering")
    try:
        import websocket  # noqa: F401

        ws = "websocket-client importable"
    except ImportError:
        ws = "websocket-client MISSING"
    show("CDP 127.0.0.1:9222", ("answering" if CdpClient().version() else "not answering") + f"; {ws}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return {"run": cmd_run, "compare": cmd_compare, "selftest": cmd_selftest}[args.command](args)
