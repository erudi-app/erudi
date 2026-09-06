"""Request logging middleware: id propagation, access log, 500 fallback, streaming.

Uses a minimal FastAPI app (no lifespan, no DB) so the middleware and the
exception handlers are exercised in isolation, mirroring the conftest
TestClient pattern (raise_server_exceptions=False).
"""

import logging
import re

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from src.core.api import RequestLoggingMiddleware, add_exception_handlers
from src.core.request_context import get_request_id

GENERATED_ID_PATTERN = r"be-[0-9a-f]{8}"


def _build_app() -> FastAPI:
    app = FastAPI()
    add_exception_handlers(app)
    app.add_middleware(RequestLoggingMiddleware)

    @app.get("/ping")
    async def ping():
        return {"request_id": get_request_id()}

    @app.get("/erudi/health")
    async def health():
        return {"status": "ok"}

    @app.get("/erudi/jobs/1/status")
    async def job_status():
        return {"state": "running"}

    @app.get("/boom")
    async def boom():
        raise RuntimeError("intentional test crash")

    @app.get("/missing")
    async def missing():
        from src.core.exceptions import ModelNotFoundException

        raise ModelNotFoundException("ghost")

    @app.get("/engine-down")
    async def engine_down():
        from src.core.exceptions import EngineException

        raise EngineException("engine died", trace="child said no")

    @app.get("/stream-crash")
    async def stream_crash():
        async def gen():
            yield "first chunk\n"
            raise RuntimeError("crash mid-stream")

        return StreamingResponse(gen(), media_type="text/plain")

    @app.get("/stream")
    async def stream():
        async def gen():
            for i in range(3):
                yield f"chunk-{i}\n"

        return StreamingResponse(gen(), media_type="text/plain")

    return app


@pytest.fixture
def logging_client():
    with TestClient(_build_app(), raise_server_exceptions=False) as test_client:
        yield test_client


def _http_records(caplog):
    return [rec for rec in caplog.records if rec.getMessage().startswith("HTTP ")]


_MS_RE = re.compile(r"in ([0-9.]+)ms")


def _parse_ms(message: str) -> float:
    """Pull the millisecond value out of an access / background log line."""
    match = _MS_RE.search(message)
    assert match, f"no duration found in: {message!r}"
    return float(match.group(1))


def _access_records(caplog):
    """The single 'HTTP <m> <p> -> <s> in <ms>ms' line(s), excluding the
    optional background-tail and the unhandled-exception variants."""
    return [
        rec
        for rec in caplog.records
        if rec.getMessage().startswith("HTTP ")
        and "background work finished" not in rec.getMessage()
        and "unhandled exception" not in rec.getMessage()
    ]


def _background_records(caplog):
    return [rec for rec in caplog.records if "background work finished" in rec.getMessage()]


class _ManualClock:
    """Deterministic ``time.perf_counter`` stand-in.

    The fake ASGI app advances it explicitly, so body-send time and the
    post-response background time are exact values instead of wall-clock
    measurements. This removes every source of timing jitter from the duration
    assertions below — the numbers are computed, not raced.
    """

    def __init__(self, start: float = 1000.0):
        self.now = start

    def perf_counter(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


async def _null_receive():
    return {"type": "http.request", "body": b"", "more_body": False}


async def _null_send(message):
    return None


# ---------------------------------------------------------------------------
# X-Request-ID propagation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_provided_request_id_is_echoed(logging_client):
    response = logging_client.get("/ping", headers={"X-Request-ID": "test-rid-42"})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-rid-42"
    # The contextvar is visible from inside the endpoint.
    assert response.json()["request_id"] == "test-rid-42"


@pytest.mark.unit
def test_request_id_header_lookup_is_case_insensitive(logging_client):
    response = logging_client.get("/ping", headers={"x-ReQuEsT-iD": "mixed-case-rid"})
    assert response.headers["X-Request-ID"] == "mixed-case-rid"


@pytest.mark.unit
def test_request_id_generated_when_absent(logging_client):
    response = logging_client.get("/ping")
    rid = response.headers.get("X-Request-ID")
    assert rid is not None
    assert re.fullmatch(GENERATED_ID_PATTERN, rid)
    assert response.json()["request_id"] == rid


# ---------------------------------------------------------------------------
# Access log line
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_access_log_has_method_path_status_duration_and_id(logging_client, caplog):
    with caplog.at_level(logging.INFO, logger="erudi"):
        logging_client.get("/ping", headers={"X-Request-ID": "rid-log-test"})
    records = _http_records(caplog)
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.INFO
    message = record.getMessage()
    assert "HTTP GET /ping -> 200 in " in message
    assert message.rstrip().endswith("ms")
    assert record.request_id == "rid-log-test"


@pytest.mark.unit
def test_polling_paths_log_at_debug(logging_client, caplog):
    with caplog.at_level(logging.DEBUG, logger="erudi"):
        logging_client.get("/erudi/health")
        logging_client.get("/erudi/jobs/1/status")
    records = _http_records(caplog)
    assert len(records) == 2
    assert all(rec.levelno == logging.DEBUG for rec in records)


@pytest.mark.unit
def test_polling_paths_are_silent_at_info(logging_client, caplog):
    with caplog.at_level(logging.INFO, logger="erudi"):
        logging_client.get("/erudi/health")
    assert _http_records(caplog) == []


# ---------------------------------------------------------------------------
# Unhandled exception fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unhandled_exception_returns_structured_500(logging_client, caplog):
    with caplog.at_level(logging.INFO, logger="erudi"):
        response = logging_client.get("/boom", headers={"X-Request-ID": "rid-boom"})
    assert response.status_code == 500
    body = response.json()
    assert body["success"] is False
    assert body["error"]["type"] == "INTERNAL_SERVER_ERROR"
    assert "message" in body["error"]
    # The fallback handler must echo the id itself (its response bypasses the
    # request-logging middleware's send wrapper).
    assert response.headers.get("X-Request-ID") == "rid-boom"

    # Exactly ONE defect record for one crash: the fallback handler's, with the
    # traceback and the request id. The middleware's line for the crashed
    # request is the access line, at the access line's level.
    error_records = [rec for rec in caplog.records if rec.levelno == logging.ERROR]
    assert len(error_records) == 1
    assert error_records[0].request_id == "rid-boom"
    assert error_records[0].exc_info  # traceback captured
    access = [rec for rec in caplog.records if rec.getMessage().startswith("HTTP ")]
    assert len(access) == 1
    assert access[0].levelno == logging.INFO
    assert "HTTP GET /boom -> 500 in " in access[0].getMessage()
    assert "(unhandled exception)" in access[0].getMessage()


@pytest.mark.unit
def test_crash_inside_a_streaming_body_is_logged_once_with_traceback(logging_client, caplog):
    """The response has started, so no error body can be sent -- but the
    crash must still leave one ERROR record with its traceback."""
    with caplog.at_level(logging.INFO, logger="erudi"):
        try:
            logging_client.get("/stream-crash", headers={"X-Request-ID": "rid-stream"})
        except Exception:
            # The test client re-raises what the server re-raised; the
            # assertion is about the log, not the transport.
            pass
    error_records = [rec for rec in caplog.records if rec.levelno == logging.ERROR]
    assert len(error_records) == 1
    assert error_records[0].request_id == "rid-stream"
    assert error_records[0].exc_info
    assert "crash mid-stream" in error_records[0].getMessage()


# ---------------------------------------------------------------------------
# Streaming: pass-through, unbuffered, single log line
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_streaming_chunks_pass_through_unbuffered():
    """Raw ASGI probe: each body message must be forwarded as-is, one send per
    chunk, never coalesced or buffered by the middleware."""
    chunks = [b"first", b"second", b"third"]

    async def raw_app(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        for i, chunk in enumerate(chunks):
            await send(
                {
                    "type": "http.response.body",
                    "body": chunk,
                    "more_body": i < len(chunks) - 1,
                }
            )

    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    wrapped = RequestLoggingMiddleware(raw_app)
    scope = {"type": "http", "method": "GET", "path": "/stream", "headers": []}
    await wrapped(scope, receive, send)

    body_messages = [m for m in sent if m["type"] == "http.response.body"]
    assert [m["body"] for m in body_messages] == chunks
    assert [m.get("more_body", False) for m in body_messages] == [True, True, False]

    start_messages = [m for m in sent if m["type"] == "http.response.start"]
    assert len(start_messages) == 1
    header_names = [name.lower() for name, _ in start_messages[0]["headers"]]
    assert b"x-request-id" in header_names


@pytest.mark.unit
def test_streaming_endpoint_still_streams_full_body(logging_client):
    with logging_client.stream("GET", "/stream") as response:
        content = b"".join(response.iter_raw())
    assert content == b"chunk-0\nchunk-1\nchunk-2\n"


@pytest.mark.unit
def test_streaming_request_logs_exactly_once(logging_client, caplog):
    with caplog.at_level(logging.INFO, logger="erudi"):
        logging_client.get("/stream")
    records = [rec for rec in caplog.records if rec.getMessage().startswith("HTTP GET /stream")]
    assert len(records) == 1
    assert "-> 200 in " in records[0].getMessage()


# ---------------------------------------------------------------------------
# Exception severity mapping (AppBaseException)
# ---------------------------------------------------------------------------


def _exception_records(caplog):
    """Every record that is not an access line: the handler's own record(s)."""
    return [rec for rec in caplog.records if not rec.getMessage().startswith("HTTP ")]


@pytest.mark.unit
def test_constructing_an_exception_logs_nothing(caplog):
    """The record belongs to whoever handles the exception, not to the raise
    site: an exception that is caught and recovered from must leave no
    ERROR behind, and a request must not get its record split in two."""
    from src.core.exceptions import EngineException, ModelNotFoundException

    with caplog.at_level(logging.DEBUG, logger="erudi"):
        ModelNotFoundException("ghost-model")
        EngineException("engine died", trace="child said no")
    assert caplog.records == []


@pytest.mark.unit
def test_client_error_logs_one_info_record_with_the_request(logging_client, caplog):
    """A 404 is the expected outcome of asking for something that is not
    there: one INFO record naming the request, and nothing at WARNING or
    above, so the Diagnostics panel stays free of normal traffic."""
    with caplog.at_level(logging.DEBUG, logger="erudi"):
        response = logging_client.get("/missing", headers={"X-Request-ID": "rid-404"})
    assert response.status_code == 404

    records = _exception_records(caplog)
    assert len(records) == 1
    (record,) = records
    assert record.levelno == logging.INFO
    assert record.getMessage() == "GET /missing -> 404 MODEL_NOT_FOUND: Model 'ghost' not found"
    assert record.request_id == "rid-404"
    assert all(rec.levelno < logging.WARNING for rec in caplog.records)


@pytest.mark.unit
def test_server_error_logs_one_error_record_with_trace_and_traceback(logging_client, caplog):
    """A 500-class business exception: exactly one ERROR record, carrying the
    method, path, status, code, message, the raiser's trace and the traceback."""
    with caplog.at_level(logging.DEBUG, logger="erudi"):
        response = logging_client.get("/engine-down", headers={"X-Request-ID": "rid-500"})
    assert response.status_code == 500

    errors = [rec for rec in caplog.records if rec.levelno >= logging.ERROR]
    assert len(errors) == 1
    (record,) = errors
    message = record.getMessage()
    assert message.startswith("GET /engine-down -> 500 LLM_ENGINE_FAILURE: engine died")
    assert "- Trace: child said no" in message
    assert record.request_id == "rid-500"
    assert record.exc_info and record.exc_info[1] is not None


@pytest.mark.unit
def test_query_string_stays_out_of_the_exception_record(logging_client, caplog):
    """The path identifies the route; a query can carry a search term."""
    with caplog.at_level(logging.DEBUG, logger="erudi"):
        logging_client.get("/missing?query=SECRET-SEARCH-TERM")
    (record,) = _exception_records(caplog)
    assert "SECRET-SEARCH-TERM" not in record.getMessage()


@pytest.mark.unit
def test_options_preflight_logs_at_debug(logging_client, caplog):
    with caplog.at_level(logging.DEBUG, logger="erudi"):
        logging_client.options("/ping")
    records = _http_records(caplog)
    assert len(records) == 1
    assert records[0].levelno == logging.DEBUG
    assert "HTTP OPTIONS /ping" in records[0].getMessage()


# ---------------------------------------------------------------------------
# Background tasks vs. request duration (#204)
#
# starlette runs a response's BackgroundTasks AFTER the body is sent but BEFORE
# the ASGI call returns. Logging the access line at body-complete (rather than
# after the app returns) keeps that background work out of the reported request
# duration. A ManualClock makes the timings exact, so these assertions never
# race the wall clock.
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_access_line_logs_at_body_complete_not_after_background(monkeypatch, caplog):
    """The endpoint answers instantly, then a 5 s BackgroundTask runs. The
    access line must report the 50 ms body-send time, and the 5 s tail must be
    a separate DEBUG line (the field bug logged the whole ~16 min as duration)."""
    clock = _ManualClock()
    monkeypatch.setattr("src.core.api.time", clock)

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        clock.advance(0.05)  # assembling / flushing the body: 50 ms
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})
        clock.advance(5.0)  # starlette runs the BackgroundTask here: 5 s

    wrapped = RequestLoggingMiddleware(app)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/erudi/llms/28/download",
        "headers": [],
    }

    with caplog.at_level(logging.DEBUG, logger="erudi"):
        await wrapped(scope, _null_receive, _null_send)

    access = _access_records(caplog)
    assert len(access) == 1
    assert access[0].levelno == logging.INFO
    assert "HTTP POST /erudi/llms/28/download -> 200 in " in access[0].getMessage()
    # Body-send time (50 ms), NOT body + 5 s of background work.
    assert _parse_ms(access[0].getMessage()) == pytest.approx(50.0, abs=1.0)

    background = _background_records(caplog)
    assert len(background) == 1
    assert background[0].levelno == logging.DEBUG
    assert "HTTP POST /erudi/llms/28/download background work finished in " in (
        background[0].getMessage()
    )
    # The tail line carries the full elapsed time (body + background = 5050 ms).
    assert _parse_ms(background[0].getMessage()) == pytest.approx(5050.0, abs=1.0)


@pytest.mark.unit
def test_normal_request_emits_no_background_line(logging_client, caplog):
    """A request with no background work: exactly one access line, no tail."""
    with caplog.at_level(logging.DEBUG, logger="erudi"):
        logging_client.get("/ping")
    assert len(_access_records(caplog)) == 1
    assert _background_records(caplog) == []


@pytest.mark.unit
async def test_streaming_logs_once_at_last_chunk(monkeypatch, caplog):
    """A three-chunk stream logs one line, at end-of-stream, with no tail: the
    reported duration is the time to the LAST chunk, not the first."""
    clock = _ManualClock()
    monkeypatch.setattr("src.core.api.time", clock)

    async def app(scope, receive, send):
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        for i in range(3):
            clock.advance(0.1)  # 100 ms per chunk
            await send(
                {
                    "type": "http.response.body",
                    "body": f"c{i}".encode(),
                    "more_body": i < 2,
                }
            )

    wrapped = RequestLoggingMiddleware(app)
    scope = {"type": "http", "method": "GET", "path": "/stream", "headers": []}

    with caplog.at_level(logging.DEBUG, logger="erudi"):
        await wrapped(scope, _null_receive, _null_send)

    access = _access_records(caplog)
    assert len(access) == 1
    # 3 chunks x 100 ms -> logged at 300 ms (end-of-stream), not 100 ms.
    assert _parse_ms(access[0].getMessage()) == pytest.approx(300.0, abs=1.0)
    assert _background_records(caplog) == []


@pytest.mark.unit
async def test_response_without_body_still_logs_once(monkeypatch, caplog):
    """Defensive fallback: an app that returns without ever sending a body chunk
    still yields exactly one access line (logged after the app returns)."""
    clock = _ManualClock()
    monkeypatch.setattr("src.core.api.time", clock)

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 204, "headers": []})
        clock.advance(0.2)
        # returns without any http.response.body message

    wrapped = RequestLoggingMiddleware(app)
    scope = {"type": "http", "method": "GET", "path": "/nobody", "headers": []}

    with caplog.at_level(logging.DEBUG, logger="erudi"):
        await wrapped(scope, _null_receive, _null_send)

    access = _access_records(caplog)
    assert len(access) == 1
    assert "HTTP GET /nobody -> 204 in " in access[0].getMessage()
    assert _parse_ms(access[0].getMessage()) == pytest.approx(200.0, abs=1.0)
    assert _background_records(caplog) == []
