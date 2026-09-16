"""The scenario (docs/SPEC.md "Phases"): one function per phase, executed in order by `runner.Run`."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Callable

from . import app_control, lifecycle, machine, workload as workload_mod
from .discovery import take_snapshot
from .model_match import ModelMatchError, pick_installed
from .platform_info import conditions, disk_allows_download, free_disk_bytes, system_info
from .runner import OPT_IN, Abort, Config, PhaseFailed, Run, Skip
from .util import wait_until, write_json

__all__ = ["Config", "Run", "OPT_IN", "PHASES", "PHASE_NAMES", "default_model_link"]

DEFAULT_MODEL_LINK = {"darwin": "lmstudio-community/Qwen3-4B-MLX-4bit"}
DEFAULT_MODEL_LINK_OTHER = "Qwen/Qwen3-4B-GGUF"
EMBEDDING_MODEL_ESTIMATE_BYTES = int(1.2 * 1024**3)  # intfloat/multilingual-e5-small in HF cache layout; an estimate

COLD_PROMPT = "In one short sentence: what is the capital of France?"
WARM_PROMPTS = [
    "Name three primary colours. Answer in one short line.",
    "What is 12 times 7? Answer with the number only.",
    "Give one synonym for 'fast'. One word.",
    "In one sentence, what does a CPU do?",
    "Which planet is the largest in the solar system? One word.",
    "Translate 'good morning' into French. Answer with the translation only.",
    "How many days are in a leap year? Number only.",
    "Name one programming language created in the 1990s. One word.",
]
UI_PROMPT = "In one short sentence: why is the sky blue?"
KB_QUESTIONS = [
    "Selon la politique de sécurité des données, sous quel délai faut-il notifier l'autorité en cas d'incident ?",
    "Dans le tableau des responsabilités, qui est le RSSI et à quelle fréquence fait-il la revue ?",
]
WEB_QUESTION = "Search the web: what is the most recent stable version of the Python programming language? Answer in one sentence."
ARENA_QUESTION = "In one short sentence: what is photosynthesis?"

UI_PAGES = [
    ("models", "#/erudi/models"),
    ("chat", "#/erudi/chat"),
    ("knowledge_base", "#/erudi/attach_knowledge_base"),
    ("settings", "#/erudi/settings"),
    ("chat_again", "#/erudi/chat"),
]

def default_model_link(os_name: str) -> str:
    return DEFAULT_MODEL_LINK.get(os_name, DEFAULT_MODEL_LINK_OTHER)


# --- phases -------------------------------------------------------------------------------


def phase_preflight(run: Run) -> dict:
    info = system_info(run.layout)
    info["run_id"] = run.run_id
    info["profile"] = run.cfg.profile
    info["flavour"] = {"requested": run.flavour.name, "file": run.flavour.path, "backend_type": run.flavour.backend_type,
                       "engine": run.flavour.engine, "model_link": run.model_link(), "gpu": run.flavour.per_process_gpu_memory,
                       "notes": run.flavour.notes, "artifact_guess": info["app"].get("flavour")}
    info["machine"] = machine.static_info()
    info["conditions"] = conditions(run.sampler.machine.read(), info["gpus"])
    procs, _ = take_snapshot()
    running = app_control.running_erudi(procs, run.layout, run.cfg.api_port, run.state.exclude_pids)
    health = run.api.health_ok()
    run.event("preflight", running=running, health=health)
    info["workload"] = _check_workload(run, procs, set(running))
    write_json(run.dir / "system.json", info)
    if run.cfg.attach:
        if not health:
            raise Abort(f"--attach: no app answering on 127.0.0.1:{run.cfg.api_port}")
        ok, message = run.check_flavour()
        info["flavour"]["detected"] = run.flavour_detected
        write_json(run.dir / "system.json", info)
        if not ok:
            raise Abort(message)
    elif running or health:
        raise Abort(f"Erudi is already running ({len(running)} processes, API up={health}); quit it or use --attach")
    elif not info["app"]["installed"]:
        raise Abort(f"app not found at {run.layout.app_path} (use --app-path)")
    run.snapshot_storage("start")
    return {"free_disk_bytes": info["free_disk_bytes"], "app": info["app"]}


def _check_workload(run: Run, procs, erudi_pids: set[int]) -> dict:
    """Verify the declared workload before anything else runs, and record what it weighs at the start."""
    if run.workload is None:
        return {"profile": run.cfg.profile, "workload_file": None, "note": "no workload file for this profile"}
    measured, assignment = run.sampler.measure_workload(procs, erudi_pids)
    checks = workload_mod.verify_presence(assignment, run.workload)
    for c in checks:
        run.event("workload_check", **c)
    warnings = [c for c in checks if c["status"] == "warning"]
    if warnings and run.cfg.strict_workload:
        raise Abort("workload mismatch (--strict-workload): " + "; ".join(f"{c['group']}: {c['message']}" for c in warnings))
    shutil.copyfile(run.workload.path, run.dir / "workload.json")
    return {"profile": run.cfg.profile, "workload_file": run.workload.path, "notes": run.workload.notes,
            "checks": checks, "groups_at_start": workload_mod.summarize_at(assignment, measured, checks)}


def phase_baseline(run: Run) -> dict:
    """The idle machine before the app exists: the reference every later phase is read against."""
    if run.cfg.attach:
        raise Skip("--attach: the app is already running, no pre-launch baseline")
    started = run.sampler.samples_written
    run.sampler.refresh_top_other()  # the idle machine's top consumers are part of the reference
    time.sleep(run.cfg.baseline_seconds)
    samples = run.sampler.samples_written - started
    erudi_seen = sorted(run.sampler.member_pids().values())
    run.event("baseline", seconds=run.cfg.baseline_seconds, samples=samples, erudi_processes=erudi_seen)
    if erudi_seen:
        raise PhaseFailed(f"Erudi processes appeared during the baseline: {erudi_seen}")
    return {"seconds": run.cfg.baseline_seconds, "samples": samples}


def phase_cold_boot(run: Run) -> dict:
    if run.cfg.attach:
        raise Skip("--attach: the app was already running, boot not measured")
    try:
        return _cold_boot(run)
    finally:
        run.sampler.set_interval(run.cfg.sample_interval)


def _cold_boot(run: Run) -> dict:
    # Fast sampling is in force before the binary is exec'd: set_interval wakes the sampler at once,
    # and the harness waits for one fast-rate sample so the next one lands within a boot interval.
    before = run.sampler.samples_written
    run.sampler.set_interval(run.cfg.boot_sample_interval)
    wait_until(lambda: run.sampler.samples_written > before, timeout=max(5.0, 4 * run.cfg.boot_sample_interval), interval=0.01)
    t0 = time.time()
    run.t0 = min(run.t0, t0)
    proc = app_control.launch(run.layout, run.cfg.cdp_port, run.dir / "logs" / "electron-console.log")
    run.launched_proc = proc
    run.state.main_pid = proc.pid
    run.event("launched", pid=proc.pid, exe=str(run.layout.main_exe))
    marks: dict[str, float | None] = {"cdp_answering": None, "health_200": None}
    state: dict[str, Any] = {"terminal": None, "terminal_at": None}

    def boot_done():
        # Each probe records the first instant it succeeds, so a mark is never delayed by another wait.
        now = time.time()
        if proc.poll() is not None and state["terminal"] is None:
            state["terminal"] = {"event": "startup_error", "code": "APP_EXITED", "message": f"app process exited with {proc.returncode}"}
        if marks["cdp_answering"] is None and run.cdp.version():
            marks["cdp_answering"] = now
        if state["terminal"] is None:
            for e in lifecycle.read_capture_events(run.layout.capture_logs, t0):
                if e["event"] in ("ready", "startup_error"):
                    state["terminal"], state["terminal_at"] = e, now
                    break
        if marks["health_200"] is None and run.api.health_ok():
            marks["health_200"] = time.time()
        terminal = state["terminal"]
        if terminal is None:
            return False
        if terminal["event"] == "startup_error" or marks["health_200"] is not None:
            return True
        return now - state["terminal_at"] > 60  # ready but health never answered: give up after 60 s

    # CDP, the backend events and health come up independently: poll all of them together, bounded by
    # Electron's 330 s cap, then leave CDP a short grace period if it is still missing.
    wait_until(boot_done, timeout=330 + 60, interval=0.1)
    terminal = state["terminal"]
    if marks["cdp_answering"] is None:
        marks["cdp_answering"] = wait_until(lambda: run.cdp.version() and time.time(), timeout=10, interval=0.25)
    if marks["cdp_answering"]:
        try:
            run.require_cdp()
            load_ms = wait_until(run.cdp.load_event_epoch_ms, timeout=60, interval=0.5)
            marks["renderer_load"] = load_ms / 1000 if load_ms else None
        except Skip as s:
            run.event("error", where="cold_boot_cdp", error=str(s))
    events = lifecycle.read_capture_events(run.layout.capture_logs, t0)
    timeline = lifecycle.boot_timeline(t0, events, marks)
    run.event("boot_timeline", rows=timeline)
    if terminal is None:
        raise PhaseFailed("no ready/startup_error event in the stdout capture log within 330 s")
    if terminal["event"] == "startup_error":
        raise PhaseFailed(f"startup_error {terminal.get('code')}: {terminal.get('message')}")
    if not marks["health_200"]:
        raise PhaseFailed("health did not return 200")
    run.prepare_app()
    # First moment the app can be asked what it actually is: a CUDA run on a CPU build is another measurement.
    ok, message = run.check_flavour()
    if not ok:
        raise PhaseFailed(message)
    return {"timeline": timeline, "first_run": next((e.get("first_run") for e in events if e["event"] == "starting"), None),
            "boot_sample_interval_s": run.cfg.boot_sample_interval}


def phase_idle_after_boot(run: Run) -> dict:
    run.require_app()
    time.sleep(run.cfg.idle_seconds)
    return {"idle_seconds": run.cfg.idle_seconds}


def phase_ui_tour(run: Run) -> dict:
    run.require_app()
    run.require_cdp()
    visited = []
    for page, route in UI_PAGES:
        run.cdp.navigate_hash(route)
        landed = wait_until(lambda: run.cdp.evaluate("location.hash") == route, timeout=10, interval=0.25)
        run.settle(5)
        row = run.renderer.record("page", page)
        visited.append({"page": page, "route": route, "landed": bool(landed), "nodes": (row or {}).get("metrics", {}).get("Nodes")})
    if not all(v["landed"] for v in visited):
        raise PhaseFailed(f"navigation did not land: {[v['route'] for v in visited if not v['landed']]}")
    return {"pages": visited}


def phase_model_ready(run: Run) -> dict:
    run.require_app()
    link = run.model_link()
    local = run.api.get("/llms/local").body or []
    catalog = run.api.get("/llms/remote").body or []
    # After a download Erudi rewrites the row's link to the local path, so the catalog name is the bridge.
    try:
        match = pick_installed(local, catalog, link, str(run.layout.data_root))
    except ModelMatchError as e:
        raise PhaseFailed(str(e)) from e
    if match:
        run.model = match.model
        run.event("model_bound", llm_id=match.model["id"], link=link, installed_link=match.model.get("link"), name=match.model.get("name"),
                  matched_by=match.rule, candidates=match.candidates, note=match.note, downloaded_by_harness=False)
        return {"llm_id": match.model["id"], "link": link, "downloaded_by_harness": False, "matched_by": match.rule, "note": match.note}
    if run.cfg.no_download:
        raise Skip(f"model {link} not installed and --no-download given")
    remote = [m for m in catalog if m.get("link") == link]
    if not remote:
        raise PhaseFailed(f"model {link} not found in /llms/remote")
    size = remote[0].get("artifact_size_bytes")
    allowed, why = disk_allows_download(free_disk_bytes(run.layout.data_root), size, run.cfg.disk_headroom_gb)
    if not allowed:
        raise Skip(f"download refused: {why}")
    started = time.monotonic()
    job = run.api.post(f"/llms/{remote[0]['id']}/download")
    if not job.ok:
        raise PhaseFailed(f"download start -> {job.status}: {job.body}")
    job_id = job.body.get("id", job.body.get("job_id"))  # DownloadJobResponse.job_id is serialised under its alias "id"
    run.event("download_start", job_id=job_id, link=link, size_bytes=size)
    last_report = [0.0]

    def poll():
        st = run.api.get(f"/llms/downloads/{job_id}/status").body or {}
        if time.monotonic() - last_report[0] > 30:
            last_report[0] = time.monotonic()
            run.event("download_progress", job_id=job_id, progress=st.get("progress"), status=st.get("status"))
        return st if st.get("status") in ("completed", "failed", "cancelled") else None

    final = wait_until(poll, timeout=run.cfg.download_timeout, interval=2)
    duration = time.monotonic() - started
    if final is None:
        run.api.post(f"/llms/downloads/{job_id}/cancel")
        raise PhaseFailed(f"download did not finish within {run.cfg.download_timeout} s (cancelled)")
    if final["status"] != "completed":
        raise PhaseFailed(f"download {final['status']}: {final.get('error_message')}")
    llm_id = final.get("local_model_id")
    if llm_id is None:
        found = pick_installed(run.api.get("/llms/local").body or [], catalog, link, str(run.layout.data_root))
        llm_id = found.model["id"] if found else None
    if llm_id is None:
        raise PhaseFailed(f"download completed but no local model with link {link} was found")
    run.created["models"].append(llm_id)
    run.model = run.api.get(f"/llms/{llm_id}").body
    total = final.get("total_bytes") or size
    data = {
        "llm_id": llm_id,
        "link": link,
        "downloaded_by_harness": True,
        "duration_s": round(duration, 1),
        "bytes": total,
        "throughput_mb_s": round(total / 1024**2 / duration, 2) if total and duration else None,
    }
    run.event("download", **data)
    return data


def phase_chat_cold(run: Run) -> dict:
    run.require_app()
    model = run.require_model()
    run.conversation_id = run.create_conversation(model["id"])
    turn = run.run_turn(run.conversation_id, COLD_PROMPT, "cold", {model["id"]})
    facts = run.record_flavour_facts()  # the inference process exists now: check its shape and read its flags
    run.copy_logs()
    run.settle(30)
    return {"conversation_id": run.conversation_id, "flavour_facts": facts,
            "turn": {k: turn[k] for k in ("ttft_s", "wall_s", "generation_s", "answer_chars_per_s", "cold")}}


def phase_chat_warm(run: Run) -> dict:
    run.require_app()
    model = run.require_model()
    if not run.conversation_id:
        raise Skip("no conversation (chat_cold did not succeed)")
    for i in range(run.cfg.warm_turns):
        run.run_turn(run.conversation_id, WARM_PROMPTS[i % len(WARM_PROMPTS)], "warm", {model["id"]})
    run.copy_logs()
    run.settle(30)
    return {"turns": run.cfg.warm_turns}


def phase_ui_stream(run: Run) -> dict:
    run.require_app()
    if not run.conversation_id:
        raise Skip("no conversation (chat_cold did not succeed)")
    run.require_cdp()
    cid = run.conversation_id
    route = f"#/erudi/conversations/{cid}"
    run.cdp.navigate_hash(route)
    ready = wait_until(lambda: (lambda s: s and not s["disabled"])(run.cdp.composer_state()), timeout=30, interval=0.5)
    if not ready:
        raise PhaseFailed("composer textarea not found or disabled on the conversation page")
    before, _ = run.message_count(cid)
    run.renderer.record("before_submit", route)
    run.cdp.type_into_composer(UI_PROMPT)
    state = run.cdp.composer_state() or {}
    if state.get("value") != UI_PROMPT:  # read the UI state back before acting on it
        raise PhaseFailed(f"composer value did not take: {state.get('value')!r}")
    submitted = time.monotonic()
    run.cdp.press_enter()
    method = "enter"
    time.sleep(1.5)
    if (run.cdp.composer_state() or {}).get("value") == UI_PROMPT:
        method = "send_button" if run.cdp.click_send_button() else "none"
    if method == "none":
        raise PhaseFailed("neither Enter nor the send button submitted the prompt")

    def done():
        run.renderer.record("streaming", route)
        return run.message_count(cid)[0] >= before + 2

    finished = wait_until(done, timeout=run.cfg.turn_timeout, interval=2)
    run.last_use_t = time.time()
    duration = time.monotonic() - submitted
    run.event("ui_turn", conversation_id=cid, submit_method=method, completed=bool(finished), duration_s=round(duration, 2))
    if not finished:
        raise PhaseFailed("UI turn not persisted within the turn timeout")
    run.renderer.record("after_stream", route)
    run.settle(30)
    return {"submit_method": method, "duration_s": round(duration, 2)}


def phase_long_conversation_render(run: Run) -> dict:
    run.require_app()
    model = run.require_model()
    if not run.conversation_id:
        raise Skip("no conversation (chat_cold did not succeed)")
    cid = run.conversation_id
    _, user_turns = run.message_count(cid)
    i = 0
    while user_turns < run.cfg.long_turns:
        run.run_turn(cid, WARM_PROMPTS[i % len(WARM_PROMPTS)], "long", {model["id"]})
        i += 1
        _, user_turns = run.message_count(cid)
    data: dict[str, Any] = {"user_turns": user_turns, "turns_added": i}
    try:
        run.require_cdp()
    except Skip as s:
        data["renderer"] = f"not measured: {s}"
        return data
    route = f"#/erudi/conversations/{cid}"
    run.cdp.navigate_hash(route)
    time.sleep(1)
    run.cdp.reload()
    time.sleep(2)
    run.cdp.close()  # the page reload invalidates nothing on the socket, but a fresh session is simpler to trust
    wait_until(lambda: run.require_cdp() or True, timeout=30, interval=1)
    loaded = wait_until(lambda: run.cdp.composer_state() is not None, timeout=90, interval=1)
    run.settle(30)
    row = run.renderer.record("after_reload", route)
    data["renderer"] = (row or {}).get("metrics")
    if not loaded:
        raise PhaseFailed("conversation page did not render after reload")
    return data


def phase_embedding_model(run: Run) -> dict:
    run.require_app()
    st = run.api.get("/knowledge_base/embedding-model/status").body or {}
    if st.get("available"):
        run.embedding_ready = True
        return {"available": True, "downloaded_by_harness": False}
    if not st.get("downloading"):
        if run.cfg.no_download:
            raise Skip("embedding model not available and --no-download given")
        allowed, why = disk_allows_download(free_disk_bytes(run.layout.data_root), EMBEDDING_MODEL_ESTIMATE_BYTES, run.cfg.disk_headroom_gb)
        if not allowed:
            raise Skip(f"download refused: {why} (size is an estimate)")
        run.api.post("/knowledge_base/embedding-model/download")
    started = time.monotonic()
    final = wait_until(
        lambda: (lambda s: s if s.get("available") or s.get("error") else None)(run.api.get("/knowledge_base/embedding-model/status").body or {}),
        timeout=1800,
        interval=2,
    )
    if not final or not final.get("available"):
        raise PhaseFailed(f"embedding model download failed: {(final or {}).get('error') or 'timeout'}")
    run.embedding_ready = True
    data = {"available": True, "downloaded_by_harness": True, "duration_s": round(time.monotonic() - started, 1)}
    run.event("embedding_download", **data)
    return data


def _ingest(run: Run, corpus: Path, suffix: str) -> dict:
    run.require_app()
    model = run.require_model()
    if not run.embedding_ready:
        st = run.api.get("/knowledge_base/embedding-model/status").body or {}
        if not st.get("available"):
            raise Skip("embedding model not available")
        run.embedding_ready = True
    base = run.api.get(f"/llms/{model['id']}").body or {}
    if base.get("is_attached_to_kb"):
        raise Skip("the bound model already carries a KB: create would update the user's KB")
    if not corpus.is_dir():
        raise Skip(f"corpus directory {corpus} does not exist")
    paths = sorted(str(p.resolve()) for p in corpus.iterdir() if p.is_file())
    if not paths:
        raise Skip(f"no corpus files in {corpus}")
    name = f"erudi-eval {run.run_id[:16]} {suffix}"
    backend_before = (run.sampler.latest or {}).get("totals_mb", {}).get("backend")
    started = time.monotonic()
    r = run.api.post("/knowledge_base/create", {"paths": paths, "selectedModel": model["id"], "modelName": name, "description": f"erudi-eval run {run.run_id}"})
    if not r.ok:
        raise PhaseFailed(f"knowledge_base/create -> {r.status}: {r.body}")
    assistant_id = int(r.body["model_id"])
    run.created["assistants"].append(assistant_id)
    final = wait_until(
        lambda: (lambda s: s if s.get("status") in ("completed", "failed") else None)(run.api.get(f"/knowledge_base/{assistant_id}/status").body or {}),
        timeout=3600,
        interval=2,
    )
    duration = time.monotonic() - started
    backend_after = (run.sampler.latest or {}).get("totals_mb", {}).get("backend")
    data = {"assistant_id": assistant_id, "files": len(paths), "duration_s": round(duration, 1), "backend_mb_before": backend_before, "backend_mb_after": backend_after}
    run.event("kb_ingest", **data, status=(final or {}).get("status"))
    if not final:
        raise PhaseFailed("KB ingestion did not finish within 3600 s")
    if final["status"] != "completed":
        raise PhaseFailed(f"KB ingestion failed: {final.get('error_message')}")
    run.settle(30)
    return data


def phase_kb_ingest(run: Run) -> dict:
    data = _ingest(run, run.cfg.corpus_dir / "standard", "kb")
    run.kb_assistant_id = data["assistant_id"]
    return data


def phase_kb_query(run: Run) -> dict:
    run.require_app()
    model = run.require_model()
    if not run.kb_assistant_id:
        raise Skip("no KB assistant (kb_ingest did not succeed)")
    cid = run.create_conversation(run.kb_assistant_id)
    evidence = []
    for q in KB_QUESTIONS:
        start = time.time()
        turn = run.run_turn(cid, q, "kb", {run.kb_assistant_id, model["id"]})
        modes = [ln for ln in lifecycle.find_log_lines(run.layout.backend_log_dir, "Turn mode:", start - 1) if turn["request_id"] in ln] or \
            lifecycle.find_log_lines(run.layout.backend_log_dir, "Turn mode:", start - 1)
        kb_mode = any("KB" in ln for ln in modes)
        searched = "search_knowledge_base" in turn["tool_calls"]
        evidence.append({"request_id": turn["request_id"], "tool_calls": turn["tool_calls"], "turn_mode_lines": [ln[-200:] for ln in modes[:2]], "kb_searched": searched or kb_mode})
    run.event("kb_evidence", rows=evidence)
    run.copy_logs()
    if not all(e["kb_searched"] for e in evidence):
        raise PhaseFailed("no search_knowledge_base tool call and no KB turn mode in backend.log: KB numbers not trustworthy")
    run.settle(30)
    return {"conversation_id": cid, "evidence": evidence}


def phase_kb_stress(run: Run) -> dict:
    return _ingest(run, run.cfg.corpus_dir / "stress", "stress")


def phase_web_search(run: Run) -> dict:
    run.require_app()
    model = run.require_model()
    if not model.get("supports_tools"):
        raise Skip("bound model does not support tools")
    cid = run.create_conversation(model["id"], web_search=True)
    turn = run.run_turn(cid, WEB_QUESTION, "web_search", {model["id"]})
    if "web_search" not in turn["tool_calls"]:
        raise PhaseFailed(f"no web_search tool call (tool calls: {turn['tool_calls']})")
    run.settle(30)
    return {"conversation_id": cid, "tool_calls": turn["tool_calls"]}


def phase_arena(run: Run) -> dict:
    run.require_app()
    model = run.require_model()
    sent = time.monotonic()
    first = chars = None
    total_chars = 0
    for t, chunk in run.api.stream_chunks(f"/arena/{model['id']}/query", {"question": ARENA_QUESTION}, read_timeout=run.cfg.turn_timeout):
        if first is None and chunk.strip():
            first = t
        total_chars += len(chunk)
        chars = t
    run.last_use_t = time.time()
    if first is None:
        raise PhaseFailed("arena stream returned no text")
    data = {"ttft_s": round(first - sent, 3), "duration_s": round(chars - sent, 3), "chars": total_chars}
    run.event("arena_turn", **data)
    run.settle(30)
    return data


def phase_idle_unload(run: Run) -> dict:
    run.require_app()
    if "inference" not in run.sampler.categories_present():
        raise Skip("no inference process resident")
    run.copy_logs()  # mlx-child logs are deleted on an orderly stop
    started = time.monotonic()
    gone = wait_until(lambda: "inference" not in run.sampler.categories_present(), timeout=run.cfg.unload_timeout, interval=5)
    if not gone:
        raise PhaseFailed(f"inference process still resident after {run.cfg.unload_timeout} s")
    t_gone = time.time()
    data = {
        "waited_s": round(time.monotonic() - started, 1),
        "since_last_use_s": round(t_gone - run.last_use_t, 1) if run.last_use_t else None,
        "resident_after": run.resident_model_id(),
    }
    run.event("idle_unload", **data)
    run.settle(60)
    return data


def phase_cleanup(run: Run) -> dict:
    run.require_app()
    actions = []
    for cid in run.created["conversations"]:
        actions.append({"delete": f"conversation {cid}", "status": run.api.delete(f"/conversations/{cid}").status})
    for aid in run.created["assistants"]:
        actions.append({"delete": f"assistant {aid}", "status": run.api.delete(f"/llms/{aid}").status})
    for mid in run.created["models"]:
        # Never orphan_dependents=true: a 409 means something the harness did not create depends on it.
        actions.append({"delete": f"model {mid}", "status": run.api.delete(f"/llms/{mid}").status})
    run.restore_settings()
    run.event("cleanup", actions=actions)
    failed = [a for a in actions if not 200 <= a["status"] < 300]
    if failed:
        raise PhaseFailed(f"cleanup incomplete: {failed}")
    return {"actions": actions}


def phase_quit(run: Run) -> dict:
    if run.cfg.leave_running:
        raise Skip("--leave-running")
    if not run.state.main_pid:
        raise Skip("main process unknown")
    run.restore_settings()
    run.copy_logs()
    before = run.sampler.member_pids()
    import psutil

    tracked = {}
    for pid in before:
        try:
            tracked[pid] = psutil.Process(pid).create_time()
        except psutil.Error:
            pass

    def survivors() -> dict[int, str]:
        if run.launched_proc is not None:
            run.launched_proc.poll()  # reap our own child so it does not linger as a zombie
        alive = {}
        for pid, ctime in tracked.items():
            try:
                proc = psutil.Process(pid)
                if proc.create_time() == ctime and proc.status() != psutil.STATUS_ZOMBIE:
                    alive[pid] = before[pid]
            except psutil.Error:
                pass
        for pid, desc in run.sampler.member_pids().items():
            alive.setdefault(pid, desc)
        return alive

    run.cdp.close()
    command = app_control.graceful_quit(run.layout, run.state.main_pid, run.state.main_exe)
    t_quit = time.monotonic()
    run.event("quit_sent", command=command, processes_before=before)
    all_gone_s = None
    checkpoints: dict[str, dict] = {}
    while True:
        elapsed = time.monotonic() - t_quit
        alive = {str(k): v for k, v in survivors().items()}
        if not alive and all_gone_s is None:
            all_gone_s = round(elapsed, 2)
        if elapsed >= 5 and "plus_5s" not in checkpoints:
            checkpoints["plus_5s"] = alive
        if elapsed >= 30 or (all_gone_s is not None and "plus_5s" in checkpoints):
            # Once everything is gone (and stayed gone past +5 s) the +30 s list can only be empty.
            checkpoints["plus_30s"] = alive
            break
        time.sleep(0.5)
    run.state.main_pid = None
    if run.launched_proc is not None:
        try:
            run.launched_proc.wait(timeout=5)
        except Exception:  # noqa: BLE001 - exit status is informative only
            pass
    data = {"command": command, "all_gone_s": all_gone_s, "survivors": checkpoints}
    run.event("quit_result", **data)
    run.snapshot_storage("end")
    if checkpoints.get("plus_30s"):
        raise PhaseFailed(f"{len(checkpoints['plus_30s'])} process(es) still alive 30 s after quit: {checkpoints['plus_30s']}")
    return data


PHASES: list[tuple[str, Callable[[Run], dict | None]]] = [
    ("preflight", phase_preflight),
    ("baseline", phase_baseline),
    ("cold_boot", phase_cold_boot),
    ("idle_after_boot", phase_idle_after_boot),
    ("ui_tour", phase_ui_tour),
    ("model_ready", phase_model_ready),
    ("chat_cold", phase_chat_cold),
    ("chat_warm", phase_chat_warm),
    ("ui_stream", phase_ui_stream),
    ("long_conversation_render", phase_long_conversation_render),
    ("embedding_model", phase_embedding_model),
    ("kb_ingest", phase_kb_ingest),
    ("kb_query", phase_kb_query),
    ("kb_stress", phase_kb_stress),
    ("web_search", phase_web_search),
    ("arena", phase_arena),
    ("idle_unload", phase_idle_unload),
    # Deviation from the spec order (cleanup #17 after quit #16): deleting through the API needs the app up.
    ("cleanup", phase_cleanup),
    ("quit", phase_quit),
]
PHASE_NAMES = [n for n, _ in PHASES]

