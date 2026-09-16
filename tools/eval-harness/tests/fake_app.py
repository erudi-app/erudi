"""A stand-in for the packaged app, for the harness integration test. Stdlib only.

    python fake_app.py main --root DIR --api-port N

`main` plays the Electron main process: it spawns two helper processes carrying
`--type=renderer` / `--type=gpu-process`, spawns `backend` (this file) with
`--port N`, and appends every backend stdout line to DIR/tmp/erudi-backend.log
as `[ts] Backend stdout: ...`, like frontend/src/main.js.

`backend` serves the subset of /erudi endpoints the phases use, writes
DIR/logs/backend.log, starts a re-parented "postmaster" from
DIR/install/resources/backend/_internal/pgserver/pginstall/bin/postgres when
that binary exists, and runs "inference" as a multiprocessing spawn child
(the macOS MLX shape) that exits after IDLE_UNLOAD_S without a query.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import re
import signal
import subprocess
import platform
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

IDLE_UNLOAD_S = 3.0
# The stand-in claims the engine the real build would use on this host, so the flavour check is exercised.
FAKE_BACKEND = "mlx" if sys.platform == "darwin" else "cpu"
FAKE_ENGINE = {"mlx": "MLX_Engine", "cpu": "CPU_Engine"}[FAKE_BACKEND]
MB = 1024 * 1024


def iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def register(root: Path, pid: int, role: str) -> None:
    """Append `pid spawn-time role` to DIR/pids.txt: the only PIDs the tests may ever kill."""
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "pids.txt", "a", encoding="utf-8") as fh:
        fh.write(f"{pid} {time.time():.3f} {role}\n")


# --- Electron main stand-in --------------------------------------------------------------


def run_main(root: Path, api_port: int) -> None:
    capture = root / "tmp" / "erudi-backend.log"
    capture.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        with open(capture, "a", encoding="utf-8") as fh:
            fh.write(f"[{iso()}] {msg}\n")

    register(root, os.getpid(), "main")
    helpers = [
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)", "--type=renderer"]),
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3600)", "--type=gpu-process"]),
    ]
    for h in helpers:
        register(root, h.pid, "helper")
    backend = subprocess.Popen(
        [sys.executable, __file__, "backend", "--root", str(root), "--port", str(api_port)],
        stdout=subprocess.PIPE, text=True, start_new_session=True,
    )
    register(root, backend.pid, "backend")
    log(f"Backend process spawned with PID: {backend.pid}")

    def shutdown(*_):
        log("Quit requested; stopping backend")
        backend.terminate()
        try:
            backend.wait(15)
        except subprocess.TimeoutExpired:
            backend.kill()
        for h in helpers:
            h.kill()
        os._exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    for line in backend.stdout:
        if line.strip():
            log(f"Backend stdout: {line.strip()}")
    shutdown()


# --- backend stand-in --------------------------------------------------------------------


def inference_child(ready: "mp.synchronize.Event") -> None:
    ballast = bytearray(40 * MB)  # "model weights"
    ballast[::4096] = b"x" * len(ballast[::4096])
    ready.set()
    while True:
        time.sleep(1)


class Backend:
    def __init__(self, root: Path, port: int):
        self.root, self.port = root, port
        self.data = root / "prod" / "data"
        self.log_path = root / "logs" / "backend.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.ids = itertools.count(100)
        self.settings = {"web_search_enabled": False, "language": "en", "auto_update_enabled": True, "inference_backend": "auto"}
        self.catalog = [{"id": 1, "name": "Qwen3 4B (fake)", "link": "fake/Qwen3-4B-FAKE", "local": 0, "artifact_size_bytes": 3 * MB, "supports_tools": True, "is_attached_to_kb": 0}]
        self.local: dict[int, dict] = {}
        self.jobs: dict[int, dict] = {}
        self.kb_jobs: dict[int, dict] = {}
        self.conversations: dict[int, dict] = {}
        self.embedding = {"available": False, "downloading": False, "error": None}
        self.inference = None
        self.loaded_model_id = None
        self.last_used = time.monotonic()
        self.embedder_ballast = None
        self.pg_pid = None

    def log(self, level: str, rid: str, msg: str) -> None:
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(f"[{level}] {iso()} [{rid}] - fake.backend - fake_app.py:0 - {msg}\n")

    def emit(self, payload: dict) -> None:
        print(json.dumps({**payload, "ts": iso()}), flush=True)

    # -- child processes
    def start_postgres(self) -> None:
        pg = self.root / "install" / "resources" / "backend" / "_internal" / "pgserver" / "pginstall" / "bin" / "postgres"
        if not pg.exists():
            return
        fifo = self.root / "pg.fifo"
        if not fifo.exists():
            os.mkfifo(fifo)
        pgdata = self.data / "postgres"
        (pgdata / "base").mkdir(parents=True, exist_ok=True)
        (pgdata / "pg_wal").mkdir(parents=True, exist_ok=True)
        (pgdata / "base" / "1").write_bytes(b"\0" * 64000)
        # Both bash copies append their own PID ($$) to the registry (spawn time 0: bash 3.2 has no cheap clock;
        # the tests identify them by their executable path instead), so nothing is killed by group or pattern.
        pids = self.root / "pids.txt"
        script = (
            f'"$0" -c \'echo "$$ 0 pg_child" >> "{pids}"; read x <> "$1"\' "$0" "$1" & '
            f'echo "$$ 0 postmaster" >> "{pids}"; read x <> "$1"'
        )
        # A short-lived launcher starts the postmaster in its own session and exits, so the postmaster
        # is re-parented away from this backend, like pg_ctl does on POSIX.
        launcher = (
            "import subprocess, sys; p = subprocess.Popen(sys.argv[1:], stdin=subprocess.DEVNULL, start_new_session=True);"
            f" open({str(self.root / 'fake_pg.pid')!r}, 'w').write(str(p.pid))"
        )
        subprocess.run([sys.executable, "-c", launcher, str(pg), "-c", script, str(pg), str(fifo), "-D", str(pgdata)], check=True)
        self.pg_pid = int((self.root / "fake_pg.pid").read_text())

    def stop_postgres(self) -> None:
        """Stop the postmaster and its child: exactly the PIDs they registered (this backend started them)."""
        registry = self.root / "pids.txt"
        if not registry.exists():
            return
        for line in registry.read_text().splitlines():
            pid, _, role = line.split(" ", 2)
            if role in ("postmaster", "pg_child"):
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def ensure_inference(self, llm_id: int) -> None:
        with self.lock:
            if self.inference is None or not self.inference.is_alive():
                ctx = mp.get_context("spawn")
                ready = ctx.Event()
                self.inference = ctx.Process(target=inference_child, args=(ready,), daemon=True)
                self.inference.start()
                register(self.root, self.inference.pid, "inference")
                ready.wait(30)
            self.loaded_model_id = llm_id
            self.last_used = time.monotonic()

    def idle_loop(self) -> None:
        while True:
            time.sleep(0.5)
            with self.lock:
                if self.inference is not None and self.inference.is_alive() and time.monotonic() - self.last_used > IDLE_UNLOAD_S:
                    self.inference.terminate()
                    self.inference.join(5)
                    self.inference = None
                    self.loaded_model_id = None
                    self.log("INFO", "be-idle", "Unloaded idle model")

    # -- background jobs
    def run_download(self, job: dict, remote: dict) -> None:
        time.sleep(0.5)
        job["status"] = "running"
        target = self.data / "models" / remote["name"].replace(" ", "_")
        target.mkdir(parents=True, exist_ok=True)
        (target / "weights.bin").write_bytes(os.urandom(remote["artifact_size_bytes"]))
        time.sleep(0.5)
        # Like the real app: once downloaded, the row's link is the local model directory, not the catalog link.
        llm = {**remote, "id": job["local_model_id"], "local": 1, "weights_available": True, "path": str(target), "link": str(target)}
        self.local[llm["id"]] = llm
        job.update(status="completed", progress=100.0)

    def run_kb(self, job: dict) -> None:
        job["status"] = "running"
        self.embedder_ballast = bytearray(30 * MB)  # in-process embedder
        self.embedder_ballast[::4096] = b"x" * len(self.embedder_ballast[::4096])
        time.sleep(1.0)
        job["status"] = "completed"


def make_handler(b: Backend):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def rid(self) -> str:
            return self.headers.get("X-Request-ID") or "be-00000000"

        def body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n)) if n else {}

        def send_json(self, status: int, payload) -> None:
            raw = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("X-Request-ID", self.rid())
            self.end_headers()
            self.wfile.write(raw)

        def stream(self, content_type: str, chunks) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("X-Request-ID", self.rid())
            self.end_headers()
            for chunk in chunks:
                data = chunk.encode()
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")

        def route(self, method: str):
            path = self.path.split("?", 1)[0]
            if not path.startswith("/erudi/"):
                return self.send_json(404, {"detail": "not found"})
            p = path[len("/erudi"):]
            m = lambda pattern: re.fullmatch(pattern, p)  # noqa: E731
            with b.lock:
                if method == "GET" and p == "/health/":
                    return self.send_json(200, {"status": "ok", "message": "fake", "db": "ok"})
                if p == "/user_settings/":
                    if method == "PUT":
                        b.settings.update(self.body())
                    return self.send_json(200, b.settings)
                if method == "GET" and p == "/diagnostics/":
                    return self.send_json(200, {"environment": {"platform": sys.platform, "engine": FAKE_ENGINE, "loaded_model_id": b.loaded_model_id,
                                                                "gpu_name": None, "frozen": True}, "recent_errors": []})
                if method == "GET" and p == "/hardware/app_startup":
                    return self.send_json(200, {"backend_type": FAKE_BACKEND, "global_inference_score": 50, "global_inference_label": "fake"})
                if method == "GET" and p == "/hardware/detailed":
                    return self.send_json(200, {"hardware": {"backend_type": FAKE_BACKEND, "cpu_model": platform.processor() or "fake", "total_memory_gb": 16}})
                if method == "PUT" and p == "/_test/link_mode":
                    # Test hook: switch installed rows between the two shapes seen in the wild.
                    mode = self.body()["mode"]
                    for llm in b.local.values():
                        if not llm.get("is_attached_to_kb"):
                            llm["link"] = b.catalog[0]["link"] if mode == "catalog" else llm["path"]
                    return self.send_json(200, {"mode": mode})
                if method == "GET" and p == "/llms/local":
                    return self.send_json(200, list(b.local.values()))
                if method == "GET" and p == "/llms/remote":
                    return self.send_json(200, b.catalog)
                if (mm := m(r"/llms/(\d+)/download")) and method == "POST":
                    remote = next((c for c in b.catalog if c["id"] == int(mm.group(1))), None)
                    job = {"id": next(b.ids), "remote_model_id": remote["id"], "local_model_id": next(b.ids), "status": "pending", "progress": 0.0, "total_bytes": remote["artifact_size_bytes"], "error_message": None}
                    b.jobs[job["id"]] = job
                    threading.Thread(target=b.run_download, args=(job, remote), daemon=True).start()
                    return self.send_json(200, job)
                if (mm := m(r"/llms/downloads/(\d+)/status")) and method == "GET":
                    return self.send_json(200, b.jobs[int(mm.group(1))])
                if (mm := m(r"/llms/(\d+)")):
                    llm = b.local.get(int(mm.group(1)))
                    if llm is None:
                        return self.send_json(404, {"detail": "not found"})
                    if method == "GET":
                        return self.send_json(200, llm)
                    if method == "DELETE":
                        del b.local[llm["id"]]
                        if not llm.get("is_attached_to_kb") and llm.get("path"):
                            for f in Path(llm["path"]).iterdir():
                                f.unlink()
                            Path(llm["path"]).rmdir()
                        return self.send_json(200, {"message": "deleted"})
                if method == "POST" and p == "/conversations/":
                    body = self.body()
                    conv = {"id": next(b.ids), "llm_id": body["llm_id"], "web_search_enabled": bool(body.get("web_search_enabled")), "messages": []}
                    b.conversations[conv["id"]] = conv
                    return self.send_json(201, {k: v for k, v in conv.items() if k != "messages"})
                if (mm := m(r"/conversations/(\d+)(/fetch_messages)?")):
                    conv = b.conversations.get(int(mm.group(1)))
                    if conv is None:
                        return self.send_json(404, {"detail": "not found"})
                    if method == "DELETE":
                        del b.conversations[conv["id"]]
                        return self.send_json(200, {"message": "Conversation deleted successfully"})
                    if mm.group(2):
                        return self.send_json(200, conv["messages"])
                    return self.send_json(200, {k: v for k, v in conv.items() if k != "messages"})
                if method == "GET" and p == "/knowledge_base/embedding-model/status":
                    return self.send_json(200, b.embedding)
                if method == "POST" and p == "/knowledge_base/embedding-model/download":
                    b.embedding["downloading"] = True

                    def fetch():
                        cache = b.data / "models_cache" / "models--intfloat--multilingual-e5-small"
                        cache.mkdir(parents=True, exist_ok=True)
                        (cache / "model.safetensors").write_bytes(os.urandom(2 * MB))
                        time.sleep(0.5)
                        b.embedding.update(available=True, downloading=False)

                    threading.Thread(target=fetch, daemon=True).start()
                    return self.send_json(200, b.embedding)
                if method == "POST" and p == "/knowledge_base/create":
                    body = self.body()
                    if any(m["name"] == body["modelName"] for m in b.local.values()):
                        return self.send_json(409, {"detail": "duplicate name"})
                    base = b.local[int(body["selectedModel"])]
                    assistant = {**base, "id": next(b.ids), "name": body["modelName"], "is_attached_to_kb": 1, "kb_id": next(b.ids), "path": None}
                    b.local[assistant["id"]] = assistant
                    job = {"status": "pending", "error_message": None}
                    b.kb_jobs[assistant["id"]] = job
                    threading.Thread(target=b.run_kb, args=(job,), daemon=True).start()
                    return self.send_json(200, {"msg": "Knowledge Base Assistant is being created.", "model_id": assistant["id"]})
                if (mm := m(r"/knowledge_base/(\d+)/status")) and method == "GET":
                    job = b.kb_jobs.get(int(mm.group(1)))
                    return self.send_json(200, {"status": job["status"], "status_updated_at": iso(), "error_message": None}) if job else self.send_json(404, {})
            if (mm := m(r"/conversations/(\d+)/query")) and method == "POST":
                return self.query(int(mm.group(1)))
            if (mm := m(r"/arena/(\d+)/query")) and method == "POST":
                self.body()
                b.ensure_inference(int(mm.group(1)))
                self.stream("text/plain", self.slow(["Photosynthesis ", "turns light ", "into sugar."]))
                b.last_used = time.monotonic()
                return None
            return self.send_json(404, {"detail": f"no fake route for {method} {p}"})

        def slow(self, items):
            for item in items:
                time.sleep(0.05)
                yield item

        def query(self, cid: int) -> None:
            body = self.body()
            conv = b.conversations[cid]
            llm = b.local[conv["llm_id"]]
            b.ensure_inference(llm["id"])
            kb = bool(llm.get("is_attached_to_kb"))
            web = conv["web_search_enabled"]
            mode = f"agentic KB (kb_id={llm.get('kb_id')}" if kb else "plain (reason=no_kb"
            b.log("INFO", self.rid(), f"Turn mode: {mode}, web_search={'on' if web else 'off'})")
            events = [{"t": "thinking", "text": "Thinking about it."}]
            if kb:
                events.append({"t": "tool_call", "name": "search_knowledge_base", "args": {"query": body["question"][:40]}})
                events.append({"t": "tool_result", "name": "search_knowledge_base", "text": "72 heures"})
            if web:
                events.append({"t": "tool_call", "name": "web_search", "args": {"query": "python version"}})
            events += [{"t": "answer", "text": "Short "}, {"t": "answer", "text": "fake answer."}, {"t": "done"}]
            self.stream("application/x-ndjson", self.slow(json.dumps(e) + "\n" for e in events))
            with b.lock:
                conv["messages"].append({"id": next(b.ids), "sender": "user", "content": body["question"]})
                conv["messages"].append({"id": next(b.ids), "sender": "assistant", "content": "Short fake answer."})
                b.last_used = time.monotonic()

        def do_GET(self):
            self.route("GET")

        def do_POST(self):
            self.route("POST")

        def do_PUT(self):
            self.route("PUT")

        def do_DELETE(self):
            self.route("DELETE")

    return Handler


def run_backend(root: Path, port: int) -> None:
    b = Backend(root, port)
    b.emit({"event": "starting", "arch": "arm64", "mode": "prod", "data_path": str(b.data), "port": port, "first_run": False})
    b.emit({"event": "phase", "phase": "preparing_database"})
    b.start_postgres()
    b.emit({"event": "phase", "phase": "running_migrations"})
    b.emit({"event": "phase", "phase": "loading_catalog"})
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(b))
    server.daemon_threads = True
    threading.Thread(target=b.idle_loop, daemon=True).start()

    def shutdown(*_):
        b.emit({"event": "shutdown"})
        with b.lock:
            if b.inference is not None:
                b.inference.terminate()
        b.stop_postgres()
        os._exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    b.emit({"event": "ready", "port": port})
    server.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["main", "backend"])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--api-port", type=int)
    ap.add_argument("--port", type=int)
    a = ap.parse_args()
    if a.role == "main":
        run_main(a.root, a.api_port)
    else:
        run_backend(a.root, a.port)
