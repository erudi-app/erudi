"""Stdlib HTTP client for the Erudi API, NDJSON streaming and chat-turn metrics."""

from __future__ import annotations

import http.client
import itertools
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator


@dataclass
class ApiResponse:
    status: int
    body: Any
    request_id: str

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


class ApiError(RuntimeError):
    pass


class ErudiApi:
    """Talks to http://<host>:<port>/erudi. Every call carries X-Request-ID: eval-<run>-<n>."""

    def __init__(self, port: int = 27182, host: str = "127.0.0.1", run_tag: str = "run", prefix: str = "/erudi"):
        self.host, self.port, self.prefix = host, port, prefix
        self.run_tag = run_tag
        self._counter = itertools.count(1)
        self._lock = threading.Lock()

    def _request_id(self) -> str:
        with self._lock:
            return f"eval-{self.run_tag}-{next(self._counter)}"

    def _open(self, method: str, path: str, body: Any, timeout: float):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        rid = self._request_id()
        headers = {"X-Request-ID": rid, "Host": f"127.0.0.1:{self.port}", "Accept": "*/*"}
        payload = None
        if body is not None:
            payload = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        conn.request(method, self.prefix + path, body=payload, headers=headers)
        return conn, conn.getresponse(), rid

    def call(self, method: str, path: str, body: Any = None, timeout: float = 30.0) -> ApiResponse:
        conn, resp, rid = self._open(method, path, body, timeout)
        try:
            raw = resp.read()
        finally:
            conn.close()
        try:
            parsed = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            parsed = raw.decode(errors="replace")
        return ApiResponse(resp.status, parsed, resp.getheader("X-Request-ID") or rid)

    def get(self, path: str, timeout: float = 30.0) -> ApiResponse:
        return self.call("GET", path, timeout=timeout)

    def post(self, path: str, body: Any = None, timeout: float = 60.0) -> ApiResponse:
        return self.call("POST", path, body, timeout=timeout)

    def put(self, path: str, body: Any = None, timeout: float = 30.0) -> ApiResponse:
        return self.call("PUT", path, body, timeout=timeout)

    def delete(self, path: str, timeout: float = 120.0) -> ApiResponse:
        return self.call("DELETE", path, timeout=timeout)

    def health_ok(self) -> bool:
        try:
            return self.get("/health/", timeout=3).status == 200
        except OSError:
            return False

    def stream_lines(self, path: str, body: Any, read_timeout: float = 600.0) -> Iterator[tuple[float, str]]:
        """POST and yield (monotonic time, line) as each line arrives. First item is ('', request id)."""
        conn, resp, rid = self._open("POST", path, body, read_timeout)
        try:
            if resp.status != 200:
                raise ApiError(f"POST {path} -> {resp.status}: {resp.read()[:500]!r}")
            yield time.monotonic(), f"\x00request-id:{resp.getheader('X-Request-ID') or rid}"
            while True:
                line = resp.readline()
                if not line:
                    break
                yield time.monotonic(), line.decode("utf-8", errors="replace")
        finally:
            conn.close()

    def stream_chunks(self, path: str, body: Any, read_timeout: float = 600.0) -> Iterator[tuple[float, str]]:
        """POST a text/plain stream (arena, generate_title) and yield (monotonic time, chunk)."""
        conn, resp, rid = self._open("POST", path, body, read_timeout)
        try:
            if resp.status != 200:
                raise ApiError(f"POST {path} -> {resp.status}: {resp.read()[:500]!r}")
            while True:
                chunk = resp.read1(4096)
                if not chunk:
                    break
                yield time.monotonic(), chunk.decode("utf-8", errors="replace")
        finally:
            conn.close()


def parse_ndjson(lines: Iterable[tuple[float, str]]) -> Iterator[tuple[float, dict[str, Any]]]:
    """(t, raw line) -> (t, event). Tolerates split lines and non-JSON noise."""
    buffer = ""
    for t, raw in lines:
        buffer += raw
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                yield t, {"t": "_unparsed", "raw": line[:200]}
                continue
            if isinstance(event, dict):
                yield t, event
    if buffer.strip():
        try:
            yield time.monotonic(), json.loads(buffer)
        except json.JSONDecodeError:
            yield time.monotonic(), {"t": "_unparsed", "raw": buffer[:200]}


MIN_GENERATION_S = 0.05  # below this, a chars/s figure says more about buffering than about the model
TOKEN_EVENTS = ("answer", "thinking")


@dataclass
class TurnMetrics:
    """One chat turn.

    `wall_s` is everything the user waits for. `generation_s` counts only the intervals during which
    tokens actually streamed: the wait before the first token (model load, prompt processing) and the
    gaps while a tool runs are excluded, because dividing characters by them invents throughput.
    """

    sent_at: float
    ttft_s: float | None = None
    first_event_type: str | None = None
    first_answer_s: float | None = None
    wall_s: float | None = None
    generation_s: float = 0.0
    tool_time_s: float = 0.0
    done: bool = False
    answer_chunks: int = 0
    thinking_chunks: int = 0
    answer_chars: int = 0
    thinking_chars: int = 0
    answer_chars_per_s: float | None = None
    total_chars_per_s: float | None = None
    chars_per_s_note: str | None = None
    tool_calls: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    memory_warnings: int = 0
    event_times: list[tuple[float, str]] = field(default_factory=list)

    @property
    def duration_s(self) -> float | None:
        """Kept for compatibility with earlier runs: the wall clock of the turn."""
        return self.wall_s

    def as_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k not in ("event_times", "sent_at")}
        d["duration_s"] = self.wall_s
        d["events"] = len(self.event_times)
        # Raw per-event offsets (seconds since the request was sent): a formula can be redone after the run.
        d["event_timeline"] = [[round(t - self.sent_at, 4), kind] for t, kind in self.event_times]
        return d


def summarize_turn(sent_at: float, events: Iterable[tuple[float, dict[str, Any]]]) -> TurnMetrics:
    """TTFT = first `thinking` or `answer` event. Throughput is characters (the API exposes no token
    counts) over `generation_s`, and is reported as unavailable when that span is too short to mean
    anything (a whole turn delivered in one burst after a tool call)."""
    m = TurnMetrics(sent_at=sent_at)
    last_token_t: float | None = None  # last token event, when nothing interrupted the stream since
    pending_tool_t: float | None = None  # a tool started here and nothing has streamed since
    last_t = None
    for t, ev in events:
        kind = ev.get("t")
        m.event_times.append((t, str(kind)))
        last_t = t
        if kind in TOKEN_EVENTS:
            if m.ttft_s is None:
                m.ttft_s = round(t - sent_at, 4)
                m.first_event_type = kind
            if pending_tool_t is not None:
                m.tool_time_s += t - pending_tool_t
                pending_tool_t = None
            elif last_token_t is not None:
                m.generation_s += t - last_token_t
            last_token_t = t
            text = ev.get("text") or ""
            if kind == "answer":
                m.answer_chunks += 1
                m.answer_chars += len(text)
                if m.first_answer_s is None:
                    m.first_answer_s = round(t - sent_at, 4)
            else:
                m.thinking_chunks += 1
                m.thinking_chars += len(text)
            continue
        if kind in ("tool_call", "tool_result"):
            if kind == "tool_call":
                m.tool_calls.append(str(ev.get("name")))
            if pending_tool_t is None:
                pending_tool_t = t
            last_token_t = None  # the stream stops here; the next token starts a new generation span
        elif kind == "error":
            m.errors.append({"text": str(ev.get("text", ""))[:300], "code": ev.get("code")})
        elif kind == "memory_warning":
            m.memory_warnings += 1
        elif kind == "done":
            m.done = True
            break
    if last_t is not None:
        m.wall_s = round(last_t - sent_at, 4)
    m.generation_s = round(m.generation_s, 4)
    m.tool_time_s = round(m.tool_time_s, 4)
    if m.generation_s >= MIN_GENERATION_S:
        m.answer_chars_per_s = round(m.answer_chars / m.generation_s, 2)
        m.total_chars_per_s = round((m.answer_chars + m.thinking_chars) / m.generation_s, 2)
    else:
        m.chars_per_s_note = f"not reported: tokens streamed over {m.generation_s} s, under the {MIN_GENERATION_S} s floor"
    return m
