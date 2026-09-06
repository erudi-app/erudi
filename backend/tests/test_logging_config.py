"""Logging configuration: env-driven level, UTC timestamps, request-id injection.

Covers src.core.logging (formatters, filter, level resolution, stable file
name) and src.core.request_context (id generation and defaults).
"""

import io
import logging
import os
import re
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core.logging import (
    FILE_LOG_FORMAT,
    LOG_FILE_NAME,
    CustomFormatter,
    RequestIdFilter,
    configure_logger,
    resolve_log_level,
    utc_formatter,
)
from src.core.request_context import get_request_id, new_request_id, request_id_var

UTC_TS_PATTERN = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z"


@pytest.fixture
def restore_logging():
    """Re-apply the default logging configuration after level-mutating tests."""
    yield
    os.environ.pop("ERUDI_LOG_LEVEL", None)
    configure_logger()


def _make_record(
    level: int = logging.INFO,
    msg: str = "hello",
    pathname: str = "/Users/dev/Work/erudi/backend/src/core/api.py",
) -> logging.LogRecord:
    record = logging.LogRecord(
        name="erudi",
        level=level,
        pathname=pathname,
        lineno=42,
        msg=msg,
        args=(),
        exc_info=None,
    )
    record.request_id = "-"
    return record


# ---------------------------------------------------------------------------
# Level resolution (ERUDI_LOG_LEVEL)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resolve_log_level_defaults_to_info():
    assert resolve_log_level(None) == logging.INFO
    assert resolve_log_level("") == logging.INFO


@pytest.mark.unit
def test_resolve_log_level_accepts_valid_names_case_insensitively():
    assert resolve_log_level("DEBUG") == logging.DEBUG
    assert resolve_log_level("warning") == logging.WARNING
    assert resolve_log_level(" error ") == logging.ERROR


@pytest.mark.unit
def test_resolve_log_level_invalid_value_falls_back_to_info():
    assert resolve_log_level("VERBOSE") == logging.INFO
    assert resolve_log_level("123abc") == logging.INFO


@pytest.mark.unit
def test_configure_logger_default_level_is_info(restore_logging):
    os.environ.pop("ERUDI_LOG_LEVEL", None)
    lg = configure_logger()
    assert lg.level == logging.INFO
    assert lg.handlers
    assert all(handler.level == logging.INFO for handler in lg.handlers)


@pytest.mark.unit
def test_configure_logger_respects_env_level(monkeypatch, restore_logging):
    monkeypatch.setenv("ERUDI_LOG_LEVEL", "DEBUG")
    lg = configure_logger()
    assert lg.level == logging.DEBUG
    assert all(handler.level == logging.DEBUG for handler in lg.handlers)


@pytest.mark.unit
def test_configure_logger_invalid_env_falls_back_to_info(monkeypatch, restore_logging):
    monkeypatch.setenv("ERUDI_LOG_LEVEL", "NOT_A_LEVEL")
    lg = configure_logger()
    assert lg.level == logging.INFO


@pytest.mark.unit
def test_configure_logger_is_idempotent(restore_logging):
    lg1 = configure_logger()
    handler_count = len(lg1.handlers)
    lg2 = configure_logger()
    assert lg2 is lg1
    assert len(lg2.handlers) == handler_count


# ---------------------------------------------------------------------------
# Formatters: UTC ISO-8601 with milliseconds, Z suffix
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_file_formatter_emits_utc_z_timestamp_with_ms():
    record = _make_record()
    record.created = 1751447732.0
    record.msecs = 123.0
    out = utc_formatter(FILE_LOG_FORMAT).format(record)
    expected = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(1751447732.0)) + ".123Z"
    assert expected in out


@pytest.mark.unit
def test_console_formatter_emits_utc_z_timestamp_with_ms():
    record = _make_record()
    out = CustomFormatter().format(record)
    assert re.search(UTC_TS_PATTERN, out)


@pytest.mark.unit
def test_console_formatter_does_not_mutate_record_pathname():
    record = _make_record(pathname="/Users/dev/Work/erudi/backend/src/core/api.py")
    original = record.pathname
    out = CustomFormatter().format(record)
    assert record.pathname == original
    assert "backend/src/core/api.py" in out


# ---------------------------------------------------------------------------
# Request-id filter and formatter tag
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_request_id_filter_injects_dash_outside_requests():
    record = logging.LogRecord("erudi", logging.INFO, __file__, 1, "msg", (), None)
    assert RequestIdFilter().filter(record) is True
    assert record.request_id == "-"


@pytest.mark.unit
def test_request_id_filter_injects_current_request_id():
    token = request_id_var.set("be-cafe1234")
    try:
        record = logging.LogRecord("erudi", logging.INFO, __file__, 1, "msg", (), None)
        RequestIdFilter().filter(record)
        assert record.request_id == "be-cafe1234"
    finally:
        request_id_var.reset(token)


@pytest.mark.unit
def test_both_formatters_include_request_id_tag():
    record = _make_record()
    record.request_id = "be-12345678"
    assert "[be-12345678]" in CustomFormatter().format(record)
    assert "[be-12345678]" in utc_formatter(FILE_LOG_FORMAT).format(record)


# ---------------------------------------------------------------------------
# File handler: stable name + rotation preserved
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_log_file_name_is_stable_with_rotation(restore_logging):
    lg = configure_logger()
    file_handlers = [h for h in lg.handlers if isinstance(h, RotatingFileHandler)]
    assert len(file_handlers) == 1
    handler = file_handlers[0]
    assert Path(handler.baseFilename).name == LOG_FILE_NAME == "backend.log"
    assert handler.maxBytes == 10 * 1024 * 1024
    assert handler.backupCount == 10


# ---------------------------------------------------------------------------
# Third-party silences
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_noisy_third_party_loggers_are_silenced():
    for name in ("httpx", "httpcore", "huggingface_hub", "uvicorn.access"):
        assert logging.getLogger(name).level >= logging.WARNING


# ---------------------------------------------------------------------------
# Another library's records reach backend.log (root bridge)
# ---------------------------------------------------------------------------


@pytest.fixture
def log_file(tmp_path, monkeypatch, restore_logging):
    """Reconfigure the app logger onto a throwaway backend.log and yield it.

    ``restore_logging`` puts the real configuration back afterwards, bridge
    included, so no other test inherits this file.
    """
    monkeypatch.setattr(
        "src.core.logging.ensure_runtime_paths_initialized",
        lambda: SimpleNamespace(log_dir=tmp_path),
    )
    configure_logger()
    yield tmp_path / LOG_FILE_NAME


def _flush_handlers():
    for handler in logging.getLogger("erudi").handlers + logging.getLogger().handlers:
        handler.flush()


@pytest.mark.unit
def test_a_library_error_reaches_the_file(log_file):
    """pgserver dumps the whole postgres log at ERROR when a start fails. It
    logs to its own name, which carries no handler: until the root bridge,
    that record -- the only account of why the database did not come up --
    reached stderr and nothing else."""
    logging.getLogger("pgserver").error("Failed to start server. postgres said: FATAL")
    _flush_handlers()

    written = log_file.read_text(encoding="utf-8")
    assert written.count("Failed to start server") == 1
    assert "[ERROR]" in written
    assert "pgserver" in written


@pytest.mark.unit
def test_an_app_record_still_lands_exactly_once(log_file):
    """The app logger propagates to root, where the bridge now sits: without
    the name check, every Erudi record would be written to the file twice."""
    logging.getLogger("erudi").error("a single record")
    logging.getLogger("erudi.domains.llms").error("a single child record")
    _flush_handlers()

    written = log_file.read_text(encoding="utf-8")
    assert written.count("a single record") == 1
    assert written.count("a single child record") == 1


@pytest.mark.unit
def test_a_library_info_line_is_not_kept(log_file):
    """A library's INFO says nothing about a defect, is never shown on the
    Diagnostics page, and would shorten the history rotation keeps."""
    logging.getLogger("alembic.runtime.migration").info("Running upgrade abc -> def")
    _flush_handlers()

    assert "Running upgrade" not in log_file.read_text(encoding="utf-8")


@pytest.mark.unit
def test_nothing_new_reaches_stdout(log_file):
    """Stdout is the launcher's JSON event channel that the Electron main
    process parses: only the file handler is bridged, never the console one.

    Asserted on the console handler's own stream rather than through pytest's
    capture, which owns stdout during a test and would answer for itself.
    """
    console = next(
        h
        for h in logging.getLogger("erudi").handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
    )
    console.stream = io.StringIO()

    logging.getLogger("erudi").warning("an erudi line")  # proves the stream IS live
    logging.getLogger("pgserver").error("a library line")
    _flush_handlers()

    on_stdout = console.stream.getvalue()
    assert "an erudi line" in on_stdout
    assert "a library line" not in on_stdout
    # ...and the library line did reach the file, so this is not a silence.
    assert "a library line" in log_file.read_text(encoding="utf-8")


@pytest.mark.unit
def test_the_bridge_writes_only_through_the_file_handler(log_file):
    """Nothing on the root logger may write to a console: a library line on
    stdout would land in the middle of the launcher's JSON events."""
    from src.core.logging import RootFileBridge

    for handler in logging.getLogger().handlers:
        assert not (
            isinstance(handler, logging.StreamHandler)
            and not isinstance(handler, RotatingFileHandler)
            and getattr(handler, "stream", None) in (sys.stdout, sys.stderr)
        ), handler
    bridge = next(h for h in logging.getLogger().handlers if isinstance(h, RootFileBridge))
    assert isinstance(bridge._target, RotatingFileHandler)
    assert bridge.level == logging.WARNING


@pytest.mark.unit
def test_the_bridge_is_attached_once(log_file):
    from src.core.logging import RootFileBridge

    configure_logger()
    configure_logger()

    bridges = [h for h in logging.getLogger().handlers if isinstance(h, RootFileBridge)]
    assert len(bridges) == 1


@pytest.mark.unit
def test_a_multi_line_library_record_parses_as_one_record(log_file):
    """pgserver's failure record embeds the postmaster's whole log. The
    Diagnostics reader must read that as ONE record with its continuation
    lines -- like a traceback -- and no inner line may pass for a header of
    its own."""
    from src.domains.diagnostics import log_reader

    logging.getLogger("pgserver").error(
        "Failed to start server. Showing contents of postgres server log:\n"
        "LOG:  database system was not properly shut down\n"
        "FATAL:  could not write to file: No space left on device\n"
        "[ERROR] not a header either"
    )
    _flush_handlers()

    records = log_reader.parse_records(log_reader.read_tail(log_file))

    matching = [r for r in records if "Failed to start server" in r["message"]]
    assert len(matching) == 1, [r["message"] for r in records]
    record = matching[0]
    assert record["level"] == "ERROR"
    # The header names the library, in the shape RECORD_RE expects.
    assert "- pgserver - " in log_file.read_text(encoding="utf-8")
    assert "No space left on device" in record["message"]
    assert "[ERROR] not a header either" in record["message"]
    # ...and the inner lines invented no records of their own.
    assert not [r for r in records if r["message"].startswith("not a header either")]


# ---------------------------------------------------------------------------
# Request-context helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_new_request_id_format_and_uniqueness():
    ids = {new_request_id() for _ in range(100)}
    assert len(ids) == 100
    assert all(re.fullmatch(r"be-[0-9a-f]{8}", rid) for rid in ids)


@pytest.mark.unit
def test_get_request_id_defaults_to_dash_outside_requests():
    assert get_request_id() == "-"
