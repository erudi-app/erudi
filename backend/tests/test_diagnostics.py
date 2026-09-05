"""Tests for the diagnostics domain: the log tail reader and the HTTP endpoint.

Three properties matter more than the rest and each has a test of its own:

1. The reader never loads a whole log file (10 MB is the rotation cap, so the
   normal case is a large file). It seeks near the end and reads a window.
2. The reader never returns an INFO record. Backend INFO lines deliberately
   carry conversation content (docs/privacy.md), so excluding them is a
   privacy property of this feature, not a display preference.
3. The endpoint answers even when the engine or the log file is unavailable,
   because a diagnostics panel is read exactly when something is broken.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.domains.diagnostics import log_reader
from src.domains.diagnostics import services as diag_services


def _record(level: str, message: str, ts: str = "2026-09-05T23:03:15.036Z", rid: str = "-") -> str:
    return f"[{level}] {ts} [{rid}] - erudi - module.py:42 - {message}"


class TestReadTail:
    """The bounded read at the bottom of the reader."""

    def test_returns_the_whole_file_when_it_is_small(self, tmp_path):
        path = tmp_path / "backend.log"
        path.write_text("a\nb\nc\n", encoding="utf-8")
        assert log_reader.read_tail(path, max_bytes=1024) == "a\nb\nc\n"

    def test_reads_only_the_last_max_bytes_of_a_large_file(self, tmp_path):
        path = tmp_path / "backend.log"
        path.write_text("X" * 5000 + "\nTAIL\n", encoding="utf-8")
        text = log_reader.read_tail(path, max_bytes=100)
        assert "TAIL" in text
        # A bounded read: the window is at most max_bytes, so the 5000 X's
        # cannot all be in it.
        assert len(text) <= 100
        assert "X" * 200 not in text

    def test_drops_the_first_partial_line_of_the_window(self, tmp_path):
        path = tmp_path / "backend.log"
        path.write_text("first line that is long\nsecond\n", encoding="utf-8")
        text = log_reader.read_tail(path, max_bytes=15)
        assert not text.startswith("first")
        assert "second" in text

    def test_missing_file_yields_empty_text(self, tmp_path):
        assert log_reader.read_tail(tmp_path / "nope.log", max_bytes=1024) == ""


class TestParseRecords:
    """Grouping physical lines into logical records and filtering by level."""

    def test_extracts_timestamp_level_and_message(self):
        text = _record("ERROR", "boom")
        (rec,) = log_reader.parse_records(text)
        assert rec["level"] == "ERROR"
        assert rec["timestamp"] == "2026-09-05T23:03:15.036Z"
        assert rec["message"] == "boom"
        assert rec["request_id"] is None

    def test_extracts_the_request_id_when_the_line_carries_one(self):
        text = _record("WARNING", "slow", rid="be-1f2e3d4c")
        (rec,) = log_reader.parse_records(text)
        assert rec["request_id"] == "be-1f2e3d4c"

    def test_keeps_only_warning_and_above(self):
        text = "\n".join(
            [
                _record("DEBUG", "d"),
                _record("INFO", "i"),
                _record("WARNING", "w"),
                _record("ERROR", "e"),
                _record("CRITICAL", "c"),
            ]
        )
        levels = [r["level"] for r in log_reader.parse_records(text)]
        assert levels == ["WARNING", "ERROR", "CRITICAL"]

    def test_joins_the_continuation_lines_of_a_kept_record(self):
        # AppBaseException logs a multi-line record; the detail lives on the
        # continuation lines, so dropping them would leave a useless header.
        text = "\n".join(
            [
                _record("ERROR", "- Status Code: 500"),
                "- Erudi Custom Code: MODEL_NOT_FOUND",
                "- Message: model 12 is missing",
                _record("INFO", "next"),
            ]
        )
        (rec,) = log_reader.parse_records(text)
        assert "MODEL_NOT_FOUND" in rec["message"]
        assert "model 12 is missing" in rec["message"]

    def test_drops_the_continuation_lines_of_a_filtered_record(self):
        # THE PRIVACY TEST. An INFO record carrying a user prompt must not
        # reach the panel through its own line or through a continuation.
        text = "\n".join(
            [
                _record("INFO", "Query received: SECRET-PROMPT-ALPHA"),
                "  continued: SECRET-PROMPT-BRAVO",
                _record("ERROR", "boom"),
            ]
        )
        records = log_reader.parse_records(text)
        blob = repr(records)
        assert "SECRET-PROMPT-ALPHA" not in blob
        assert "SECRET-PROMPT-BRAVO" not in blob
        assert [r["message"] for r in records] == ["boom"]

    def test_drops_a_leading_orphan_continuation(self):
        # The bounded read can start mid-record; those bytes belong to a
        # record whose level is unknown, so they are dropped.
        text = "\n".join(["- Message: SECRET-ORPHAN", _record("ERROR", "boom")])
        records = log_reader.parse_records(text)
        assert "SECRET-ORPHAN" not in repr(records)
        assert len(records) == 1

    def test_returns_the_newest_records_last_and_caps_the_count(self):
        text = "\n".join(_record("ERROR", f"e{i}") for i in range(10))
        records = log_reader.parse_records(text, limit=3)
        assert [r["message"] for r in records] == ["e7", "e8", "e9"]

    def test_truncates_an_enormous_message(self):
        text = _record("ERROR", "z" * 10000)
        (rec,) = log_reader.parse_records(text)
        assert len(rec["message"]) <= log_reader.MAX_MESSAGE_CHARS + 32


class TestRecentErrors:
    """The reader's public entry point."""

    def test_reads_warnings_and_errors_from_a_real_file(self, tmp_path):
        path = tmp_path / "backend.log"
        path.write_text(
            "\n".join([_record("INFO", "hello"), _record("ERROR", "boom")]) + "\n",
            encoding="utf-8",
        )
        records = log_reader.recent_errors(path, limit=200)
        assert [r["message"] for r in records] == ["boom"]

    def test_missing_file_yields_no_records_and_never_raises(self, tmp_path):
        assert log_reader.recent_errors(tmp_path / "nope.log") == []


class _FakeEngine:
    # `__name__` is a metaclass descriptor, so a class-body assignment would be
    # ignored; the rename below is what makes the stand-in answer "CUDA_Engine"
    # the way a real engine class does.
    _model_id = 7

    @classmethod
    def get_hardware_info(cls):
        return {
            "cpu": {"model": "Test CPU"},
            "gpu": {
                "gpu_name": "NVIDIA RTX 4070",
                "compute_capability": "8.9",
                "vram_total_gb": 12.0,
            },
        }


class _BrokenEngine:
    _model_id = None

    @classmethod
    def get_hardware_info(cls):
        raise RuntimeError("nvml exploded")


_FakeEngine.__name__ = "CUDA_Engine"
_BrokenEngine.__name__ = "MLX_Engine"


class TestBuildEnvironment:
    def test_reports_the_engine_the_model_and_the_gpu(self, monkeypatch):
        monkeypatch.setattr(diag_services.config, "LLM_Engine", _FakeEngine)
        env = diag_services.build_environment()
        assert env["engine"] == "CUDA_Engine"
        assert env["loaded_model_id"] == 7
        assert env["gpu_name"] == "NVIDIA RTX 4070"
        assert env["compute_capability"] == "8.9"
        assert env["vram_total_gb"] == 12.0
        assert env["cpu_model"] == "Test CPU"
        assert env["python_version"].startswith("3.")
        assert env["backend_log_path"].endswith("backend.log")

    def test_survives_an_engine_that_cannot_describe_its_hardware(self, monkeypatch):
        monkeypatch.setattr(diag_services.config, "LLM_Engine", _BrokenEngine)
        env = diag_services.build_environment()
        assert env["engine"] == "MLX_Engine"
        assert env["gpu_name"] is None
        assert env["loaded_model_id"] is None

    def test_survives_no_engine_at_all(self, monkeypatch):
        monkeypatch.setattr(diag_services.config, "LLM_Engine", None)
        env = diag_services.build_environment()
        assert env["engine"] is None

    def test_survives_a_database_that_cannot_be_queried(self, monkeypatch):
        # A dead database is one of the things this endpoint exists to report,
        # so it must not be the thing that stops it answering.
        class _DeadSession:
            def get(self, *args, **kwargs):
                raise RuntimeError("the cluster is gone")

        monkeypatch.setattr(diag_services.config, "LLM_Engine", _FakeEngine)
        env = diag_services.build_environment(_DeadSession())
        assert env["loaded_model"] is None
        assert env["loaded_model_id"] == 7


@pytest.mark.integration
class TestDiagnosticsEndpoint:
    def test_returns_environment_and_recent_errors(self, client, monkeypatch, tmp_path):
        path = tmp_path / "backend.log"
        path.write_text(_record("ERROR", "boom") + "\n", encoding="utf-8")
        monkeypatch.setattr(diag_services, "backend_log_path", lambda: path)
        monkeypatch.setattr(diag_services.config, "LLM_Engine", _FakeEngine)

        response = client.get("/erudi/diagnostics/")
        assert response.status_code == 200
        body = response.json()
        assert body["environment"]["engine"] == "CUDA_Engine"
        assert body["environment"]["backend_log_path"] == str(path)
        assert [r["message"] for r in body["recent_errors"]] == ["boom"]

    def test_resolves_the_loaded_model_name_from_the_database(
        self, client, test_db_session, monkeypatch, tmp_path
    ):
        from src.entities.Llm import Llm

        llm = Llm(name="mlx-community/Qwen3-4B-4bit", local=1, type="text")
        test_db_session.add(llm)
        test_db_session.flush()

        engine = SimpleNamespace(__name__="MLX_Engine", _model_id=llm.id, get_hardware_info=dict)
        monkeypatch.setattr(diag_services.config, "LLM_Engine", engine)
        monkeypatch.setattr(diag_services, "backend_log_path", lambda: tmp_path / "nope.log")

        body = client.get("/erudi/diagnostics/").json()
        assert body["environment"]["loaded_model"] == "mlx-community/Qwen3-4B-4bit"

    def test_answers_with_an_empty_error_list_when_the_log_is_gone(
        self, client, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(diag_services, "backend_log_path", lambda: tmp_path / "gone.log")
        monkeypatch.setattr(diag_services.config, "LLM_Engine", _BrokenEngine)
        response = client.get("/erudi/diagnostics/")
        assert response.status_code == 200
        assert response.json()["recent_errors"] == []
