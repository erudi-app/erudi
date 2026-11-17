"""HTTP double of ``llama-server --embedding`` for the CI suite (#594, F3).

backend-ci has no compiled llama-server, so the real adapter + manager are
exercised against THIS process instead: `tests/test_embedder_child_http.py`
substitutes ``[sys.executable, this_file]`` for the binary through the
adapter's ``_server_command`` seam, and everything else — the real
``subprocess.Popen``, the log redirect, the environment, the per-spawn key,
the two-stage probe, batching, the idle reap — is the production code path.

What it mimics, measured on the real binary (spike S4 + the golden capture):

- ``GET /health`` — 200, no authentication (llama-server exempts it);
- every other endpoint requires ``Authorization: Bearer <LLAMA_API_KEY>``
  (the key reaches the real server the same way, by environment) — 401
  otherwise;
- ``POST /tokenize`` — the e5 ids of ``content``, from the SAME frozen
  tokenizer the app chunks with (the spike proved the GGUF tokenization
  identical to it on 267/267 texts); ``add_special`` wraps with BOS 0/EOS 2;
- ``POST /v1/embeddings`` — golden vectors captured once from the real
  binary (``e5_llama_golden.json``) for known texts, a deterministic
  hash-seeded unit vector otherwise; any input past 512 tokens (specials
  included) answers HTTP 500 and fails the WHOLE batch, exactly like
  llama-server (it refuses, never truncates); the data rows come back in
  REVERSE index order, so a client that forgets to sort by ``index`` fails.

Run: ``python fake_llama_embedding_server.py <llama-server argv...>`` — the
argv is the real adapter's (``-m``, ``--port``, ...); only ``--port`` is
read, the rest is accepted the way the real argument parser would.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import struct
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_FIXTURES = Path(__file__).resolve().parent
_BACKEND = _FIXTURES.parents[1]
GOLDEN_FILE = _FIXTURES / "e5_llama_golden.json"
TOKENIZER_FILE = _BACKEND / "src" / "ingestion" / "e5_tokenizer.json.gz"

BOS_ID, EOS_ID = 0, 2
WINDOW = 512
DIMENSIONS = 384


def _load_tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_buffer(gzip.decompress(TOKENIZER_FILE.read_bytes()))


def _load_golden() -> dict:
    if not GOLDEN_FILE.exists():
        return {}
    payload = json.loads(GOLDEN_FILE.read_text(encoding="utf-8"))
    return {entry["text"]: entry["embedding"] for entry in payload["texts"]}


def _fallback_vector(text: str) -> list[float]:
    """A deterministic unit vector for a text outside the golden set."""
    raw = []
    seed = text.encode("utf-8")
    counter = 0
    while len(raw) < DIMENSIONS:
        digest = hashlib.sha256(seed + struct.pack("<I", counter)).digest()
        raw.extend(b / 255.0 - 0.5 for b in digest)
        counter += 1
    raw = raw[:DIMENSIONS]
    norm = math.sqrt(sum(v * v for v in raw)) or 1.0
    return [v / norm for v in raw]


def main() -> None:
    port = int(sys.argv[sys.argv.index("--port") + 1])
    api_key = os.environ.get("LLAMA_API_KEY", "")
    tokenizer = _load_tokenizer()
    golden = _load_golden()

    def ids_of(text: str) -> list[int]:
        return list(tokenizer.encode(text, add_special_tokens=False).ids)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            return self.headers.get("Authorization") == f"Bearer {api_key}"

        def _refuse(self) -> None:
            self._send(401, {"error": {"code": 401, "message": "Invalid API Key"}})

        def do_GET(self):  # noqa: N802 - http.server contract
            if self.path == "/health":
                self._send(200, {"status": "ok"})
                return
            if not self._authorized():
                self._refuse()
                return
            self._send(404, {"error": {"code": 404, "message": "not found"}})

        def do_POST(self):  # noqa: N802 - http.server contract
            if not self._authorized():
                self._refuse()
                return
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/tokenize":
                tokens = ids_of(request.get("content", ""))
                if request.get("add_special"):
                    tokens = [BOS_ID] + tokens + [EOS_ID]
                self._send(200, {"tokens": tokens})
                return
            if self.path == "/v1/embeddings":
                inputs = request.get("input", [])
                if isinstance(inputs, str):
                    inputs = [inputs]
                for text in inputs:
                    count = len(ids_of(text)) + 2
                    if count > WINDOW:
                        # The real server's refusal: the whole batch fails.
                        self._send(
                            500,
                            {
                                "error": {
                                    "code": 500,
                                    "message": (
                                        f"input ({count} tokens) is too large to "
                                        f"process. increase the physical batch "
                                        f"size (current batch size: {WINDOW})"
                                    ),
                                    "type": "server_error",
                                }
                            },
                        )
                        return
                data = [
                    {
                        "object": "embedding",
                        "index": i,
                        "embedding": golden.get(text) or _fallback_vector(text),
                    }
                    for i, text in enumerate(inputs)
                ]
                # Reverse order on purpose: the client must sort by index.
                self._send(
                    200,
                    {"object": "list", "data": list(reversed(data)), "model": "fake"},
                )
                return
            self._send(404, {"error": {"code": 404, "message": "not found"}})

        def log_message(self, fmt, *args):
            print(f"fake-llama-server: {fmt % args}", file=sys.stderr, flush=True)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"fake llama-server (embedding) listening on port {port}", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
