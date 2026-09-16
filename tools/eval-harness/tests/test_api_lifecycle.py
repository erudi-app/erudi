import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from erudi_eval import lifecycle
from erudi_eval.api import ErudiApi, parse_ndjson, summarize_turn
from erudi_eval.util import parse_iso


def lines(*events, t0=100.0, step=0.5):
    return [(t0 + i * step, json.dumps(e) + "\n") for i, e in enumerate(events)]


def test_ndjson_parser_handles_split_lines_and_noise():
    raw = [(1.0, '{"t": "thin'), (2.0, 'king", "text": "hm"}\n{"t":"answer","text":"Hi"}\n'), (3.0, "garbage\n"), (4.0, '{"t":"done"}')]
    events = list(parse_ndjson(raw))
    assert [e["t"] for _, e in events] == ["thinking", "answer", "_unparsed", "done"]
    assert events[0][0] == 2.0  # a split line is timestamped when it completes


def test_turn_metrics_thinking_tool_answer_done():
    evs = parse_ndjson(
        lines(
            {"t": "thinking", "text": "Let me check"},
            {"t": "tool_call", "name": "search_knowledge_base", "args": {"query": "x"}},
            {"t": "tool_result", "name": "search_knowledge_base", "text": "..."},
            {"t": "answer", "text": "Hello "},
            {"t": "answer", "text": "world"},
            {"t": "memory_warning", "used_fraction": 0.9},
            {"t": "done"},
            {"t": "answer", "text": "after done is ignored"},
        )
    )
    m = summarize_turn(99.0, evs)
    assert m.ttft_s == 1.0 and m.first_event_type == "thinking"
    assert m.first_answer_s == 2.5
    assert m.tool_calls == ["search_knowledge_base"]
    assert m.answer_chars == 11 and m.answer_chunks == 2 and m.thinking_chars == 12
    assert m.done and m.memory_warnings == 1
    assert m.wall_s == 4.0 and m.duration_s == 4.0
    assert m.tool_time_s == 1.0  # tool_call -> first token after it
    assert m.generation_s == 0.5 and m.answer_chars_per_s == 22.0  # only the gap between the two answer chunks


def test_turn_metrics_error_without_done():
    m = summarize_turn(0.0, parse_ndjson(lines({"t": "error", "text": "Model failed to load", "code": "ENGINE"}, t0=5.0)))
    assert not m.done and m.ttft_s is None and m.errors == [{"text": "Model failed to load", "code": "ENGINE"}]
    assert m.answer_chars_per_s is None


CAPTURE = """\
[2026-09-15T09:59:59.000Z] Backend stdout: {"event": "shutdown", "ts": "2026-09-15T09:59:59.000Z"}
[2026-09-15T10:00:01.000Z] Starting backend at /Applications/Erudi.app/Contents/Resources/backend/backend
[2026-09-15T10:00:01.500Z] Backend stdout: {"event": "starting", "arch": "arm64", "mode": "prod", "port": 27182, "first_run": false, "ts": "2026-09-15T10:00:01.400Z"}
[2026-09-15T10:00:02.000Z] Backend stdout: {"event": "phase", "phase": "preparing_database", "ts": "2026-09-15T10:00:01.900Z"}
[2026-09-15T10:00:02.100Z] Backend stdout: INFO:     Uvicorn running on http://127.0.0.1:27182
[2026-09-15T10:00:04.000Z] Backend stdout: {"event": "phase", "phase": "running_migrations", "ts": "2026-09-15T10:00:03.900Z"}
[2026-09-15T10:00:05.000Z] Backend stdout: {"event": "phase", "phase": "loading_catalog", "ts": "2026-09-15T10:00:04.900Z"}
[2026-09-15T10:00:06.000Z] Backend stdout: {"event": "ready", "port": 27182, "ts": "2026-09-15T10:00:05.950Z"}
"""


def test_capture_log_events_after_t0_only():
    t0 = parse_iso("2026-09-15T10:00:00.000Z")
    events = lifecycle.parse_capture_log(CAPTURE, t0)
    assert [e["event"] for e in events] == ["starting", "phase", "phase", "phase", "ready"]
    rows = lifecycle.boot_timeline(t0, events, {"cdp_answering": t0 + 1.0, "health_200": t0 + 6.2, "renderer_load": None})
    labels = [r["label"] for r in rows]
    assert labels == ["harness_launch", "cdp_answering", "starting", "phase:preparing_database", "phase:running_migrations", "phase:loading_catalog", "ready", "health_200"]
    assert rows[-2]["since_t0_s"] == 5.95


def test_log_filters_by_time():
    t0 = parse_iso("2026-09-15T10:00:00.000Z")
    kept = lifecycle.filter_capture_lines(CAPTURE, t0)
    assert len(kept) == 7 and "shutdown" not in kept[0]
    backend = (
        "[INFO] 2026-09-15T09:00:00.000Z [be-1] - old\n"
        "Traceback old\n"
        "[ERROR] 2026-09-15T10:00:03.000Z [eval-r-3] - new failure\n"
        "Traceback (most recent call last):\n"
        "  File x\n"
    )
    assert lifecycle.filter_backend_log_lines(backend, t0) == backend.splitlines()[2:]


class _Streamer(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("X-Request-ID", self.headers.get("X-Request-ID"))
        self.end_headers()
        for ev in ({"t": "answer", "text": "a"}, {"t": "answer", "text": "b"}, {"t": "done"}):
            self.wfile.write((json.dumps(ev) + "\n").encode())
            self.wfile.flush()
            time.sleep(0.2)


def test_stream_lines_timestamps_arrivals_over_http():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Streamer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        api = ErudiApi(port=server.server_address[1], run_tag="t")
        sent = time.monotonic()
        items = list(api.stream_lines("/conversations/1/query", {"question": "q"}))
        assert items[0][1].startswith("\x00request-id:eval-t-")
        m = summarize_turn(sent, parse_ndjson(items[1:]))
        assert m.done and m.answer_chars == 2
        assert m.duration_s >= 0.35  # events really arrived over time, not buffered
    finally:
        server.shutdown()


def stream(events):
    """(seconds since sent, event) -> the (monotonic, raw line) pairs the parser consumes."""
    return [(t, json.dumps(e) + "\n") for t, e in events]


def test_generation_time_excludes_the_model_load_wait_and_the_tool_gaps():
    """Recorded shape of a KB turn: nothing streams while a tool runs, then everything arrives at once."""
    evs = stream([
        (10.0, {"t": "thinking", "text": "12345"}),          # +8 s after sent: model load
        (10.5, {"t": "thinking", "text": "67890"}),          # 0.5 s of streaming
        (10.6, {"t": "tool_call", "name": "search_knowledge_base", "args": {}}),
        (13.6, {"t": "tool_result", "name": "search_knowledge_base", "text": "..."}),  # 3 s of tool time
        (14.0, {"t": "answer", "text": "abcd"}),             # first token after the tool
        (14.5, {"t": "answer", "text": "efgh"}),             # 0.5 s of streaming
        (14.6, {"t": "tool_call", "name": "search_knowledge_base", "args": {}}),
        (16.6, {"t": "tool_result", "name": "search_knowledge_base", "text": "..."}),  # 2 s of tool time
        (17.0, {"t": "answer", "text": "ij"}),
        (17.5, {"t": "answer", "text": "kl"}),               # 0.5 s of streaming
        (17.6, {"t": "done"}),
    ])
    m = summarize_turn(2.0, parse_ndjson(evs))
    assert m.ttft_s == 8.0 and m.first_answer_s == 12.0
    assert m.wall_s == 15.6                                  # sent -> done
    # Each tool gap runs from the tool_call to the next token that streams (3.4 s + 2.4 s here).
    assert m.tool_time_s == pytest.approx(5.8)
    assert m.generation_s == 1.5                             # 0.5 + 0.5 + 0.5, no load wait, no tool gaps
    assert m.answer_chars == 12 and m.thinking_chars == 10
    assert m.answer_chars_per_s == 8.0 and m.total_chars_per_s == round(22 / 1.5, 2)
    assert m.chars_per_s_note is None
    # The raw timeline is kept so a wrong formula can be recomputed after the run.
    assert m.as_dict()["event_timeline"][0] == [8.0, "thinking"]
    assert len(m.as_dict()["event_timeline"]) == 11


def test_throughput_is_unavailable_rather_than_absurd_when_everything_arrives_at_once():
    evs = stream([(30.0, {"t": "thinking", "text": "x" * 2277}), (30.605, {"t": "tool_call", "name": "search_knowledge_base", "args": {}}),
                  (30.61, {"t": "answer", "text": "y" * 300}), (30.614, {"t": "answer", "text": "z" * 232}), (30.62, {"t": "done"})])
    m = summarize_turn(0.0, parse_ndjson(evs))
    assert m.generation_s == pytest.approx(0.004, abs=1e-6)
    assert m.answer_chars_per_s is None and m.total_chars_per_s is None
    assert "0.05" in m.chars_per_s_note and m.wall_s == 30.62 and m.tool_time_s == pytest.approx(0.005, abs=1e-6)


def test_single_chunk_turn_has_no_generation_interval():
    m = summarize_turn(0.0, parse_ndjson(stream([(1.0, {"t": "answer", "text": "hi"}), (1.2, {"t": "done"})])))
    assert m.generation_s == 0.0 and m.answer_chars_per_s is None and m.chars_per_s_note
    assert m.wall_s == 1.2 and m.tool_time_s == 0.0
