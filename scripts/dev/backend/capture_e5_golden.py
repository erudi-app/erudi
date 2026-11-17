"""Capture the golden vectors of the KB embedding child's CI double (#594).

backend-ci has no compiled ``llama-server``, so its suite runs the real
adapter against an HTTP double (``tests/fixtures/fake_llama_embedding_server.py``)
that answers with vectors captured HERE, once, from the REAL binary over the
REAL bundled GGUF -- the double then holds CI to what the real server
answers. Rerun this after any deliberate change of the GGUF (new snapshot
revision, new converter patch, new quant) and commit the fixture:

    python scripts/dev/backend/capture_e5_golden.py

It starts the server with EXACTLY the adapter's serving flags
(``src/ingestion/llama_embedder.py``), embeds a small multilingual corpus,
reads each text's ``/tokenize`` ids, and writes
``backend/tests/fixtures/e5_llama_golden.json``. QA can run it on an x86
machine and diff the fixture against the committed one to check
cross-architecture parity.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND = REPO_ROOT / "backend"
DEFAULT_SERVER = (
    BACKEND
    / "artifacts"
    / "llama-cpp"
    / "cpu"
    / "bin"
    / ("llama-server.exe" if os.name == "nt" else "llama-server")
)
DEFAULT_GGUF = BACKEND / "artifacts" / "kb-embedder" / "multilingual-e5-small-q8_0.gguf"
FIXTURE = BACKEND / "tests" / "fixtures" / "e5_llama_golden.json"

# A small multilingual slice of what the app embeds: the [N6] passage format,
# query prefixes, French accents, CJK, an emoji and a Markdown table row.
GOLDEN_TEXTS = [
    "passage: [document_name:guide.md]\n# Guide > ## Install\n\n"
    "Erudi runs open-source language models locally, with no account and no cloud.",
    "passage: [document_name:notes.md]\nLes modeles s'executent localement ; "
    "la sécurité des données est un préalable, pas une option.",
    "passage: [document_name:table.md]\n| Model | Size |\n|---|---|\n| e5-small | 384 |",
    "passage: [document_name:nihongo.md]\n日本語のテキストを埋め込みます。",
    "passage: OK",
    "query: how does the knowledge base work?",
    "query: sécurité des données locales ☕",
    "query: ping",
]


def main() -> int:
    server = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SERVER
    gguf = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_GGUF
    for path, hint in (
        (server, "scripts/dev/backend/build-llamacpp-*"),
        (gguf, "scripts/dev/backend/build_e5_gguf.py"),
    ):
        if not path.exists():
            raise SystemExit(f"[capture_e5_golden] missing {path} (build it: {hint})")

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    api_key = "golden-capture"
    env = os.environ.copy()
    env["LLAMA_API_KEY"] = api_key
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    # The adapter's exact serving flags (src/ingestion/llama_embedder.py).
    argv = [
        str(server),
        "-m",
        str(gguf),
        "--embedding",
        "--pooling",
        "mean",
        "-c",
        "512",
        "-b",
        "512",
        "-ub",
        "512",
        "-np",
        "1",
        "--poll",
        "0",
        "-ngl",
        "0",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--alias",
        "multilingual-e5-small",
        "--no-slots",
        "--no-webui",
    ]
    proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    base = f"http://127.0.0.1:{port}"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                if requests.get(f"{base}/health", timeout=2).status_code == 200:
                    break
            except requests.RequestException:
                pass
            if proc.poll() is not None or time.monotonic() > deadline:
                raise SystemExit("[capture_e5_golden] llama-server did not become ready")
            time.sleep(0.3)

        entries = []
        for text in GOLDEN_TEXTS:
            vector = requests.post(
                f"{base}/v1/embeddings",
                json={"model": "multilingual-e5-small", "input": [text]},
                headers=headers,
                timeout=60,
            ).json()["data"][0]["embedding"]
            tokens = requests.post(
                f"{base}/tokenize",
                json={"content": text, "add_special": True},
                headers=headers,
                timeout=10,
            ).json()["tokens"]
            entries.append({"text": text, "tokens": tokens, "embedding": vector})
            print(f"[capture_e5_golden] {len(tokens):4d} tokens  {text[:60]!r}")

        FIXTURE.write_text(
            json.dumps(
                {
                    "model": "multilingual-e5-small-q8_0.gguf",
                    "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
                    "texts": entries,
                },
                ensure_ascii=True,
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[capture_e5_golden] wrote {FIXTURE}")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
