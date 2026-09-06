"""Logging coverage assertions for the full-traceability chore.

Erudi is fully local, so log CONTENT (questions, queries, previews) is policy —
what these tests pin down is that the key silent steps now emit log lines, that
line sizes are bounded (``truncate_for_log``), and that credentials never reach
a log line (``init_database`` URL sanitization).

caplog note: the app logger is the ``erudi`` singleton from
``src.core.logging``; it propagates to the root logger (default), which is
where pytest's caplog handler lives.
"""

import logging
from types import SimpleNamespace

import pytest

from src.core.logutils import truncate_for_log


# ============ truncate_for_log (pure unit) ============


@pytest.mark.unit
class TestTruncateForLog:
    def test_short_string_unchanged(self):
        assert truncate_for_log("hello") == "hello"

    def test_strips_surrounding_whitespace(self):
        assert truncate_for_log("  hello \n") == "hello"

    def test_truncates_with_explicit_suffix(self):
        out = truncate_for_log("a" * 600, limit=500)
        assert out.startswith("a" * 500)
        assert out.endswith("… [+100 chars]")

    def test_exact_limit_is_not_truncated(self):
        assert truncate_for_log("a" * 500, limit=500) == "a" * 500

    def test_default_limit_is_500(self):
        out = truncate_for_log("b" * 501)
        assert out.endswith("… [+1 chars]")

    def test_non_string_input_is_converted(self):
        assert truncate_for_log(12345) == "12345"
        assert truncate_for_log(None) == "None"
        assert truncate_for_log(["a", "b"]) == "['a', 'b']"

    def test_custom_small_limit(self):
        out = truncate_for_log("abcdefghij", limit=4)
        assert out == "abcd… [+6 chars]"

    def test_invalid_limit_falls_back_to_default(self):
        # Never raises: a bogus limit degrades to the 500-char default.
        out = truncate_for_log("x" * 600, limit="not-a-number")
        assert out.endswith("… [+100 chars]")

    def test_non_positive_limit_falls_back_to_default(self):
        out = truncate_for_log("x" * 600, limit=0)
        assert out.endswith("… [+100 chars]")

    def test_never_raises_on_hostile_object(self):
        class Evil:
            def __str__(self):
                raise RuntimeError("boom")

            def __repr__(self):
                raise RuntimeError("boom")

        out = truncate_for_log(Evil())
        assert isinstance(out, str)
        assert out == "<unloggable>"


# ============ Conversation endpoints (mutations logged) ============


@pytest.mark.integration
class TestConversationEndpointLogging:
    def test_create_conversation_logs_info(self, client, mock_llm, caplog):
        with caplog.at_level(logging.INFO, logger="erudi"):
            response = client.post("/erudi/conversations/", json={"llm_id": mock_llm.id})
        assert response.status_code == 201
        conversation_id = response.json()["id"]
        messages = [r.message for r in caplog.records]
        assert any(
            "Conversation created" in m
            and f"id={conversation_id}" in m
            and f"llm_id={mock_llm.id}" in m
            for m in messages
        ), f"no create log found in: {messages}"

    def test_delete_conversation_logs_info(self, client, mock_llm, caplog):
        created = client.post("/erudi/conversations/", json={"llm_id": mock_llm.id})
        conversation_id = created.json()["id"]

        with caplog.at_level(logging.INFO, logger="erudi"):
            response = client.delete(f"/erudi/conversations/{conversation_id}")
        assert response.status_code == 200
        messages = [r.message for r in caplog.records]
        assert any(
            "Conversation deleted" in m and f"id={conversation_id}" in m for m in messages
        ), f"no delete log found in: {messages}"

    def test_fetch_endpoints_log_debug(self, client, mock_llm, caplog):
        with caplog.at_level(logging.DEBUG, logger="erudi"):
            client.get("/erudi/conversations/")
        assert any(
            r.levelno == logging.DEBUG and "Fetching all conversations" in r.message
            for r in caplog.records
        )


# ============ init_database URL sanitization ============


@pytest.mark.unit
class TestInitDatabaseSanitization:
    def test_password_never_reaches_the_log(self, caplog):
        from src.database import core

        previous_engine = core.db_engine
        url = "postgresql+psycopg://erudi:S3cretPassw0rd@127.0.0.1:5499/erudi"
        created_engine = None
        try:
            with caplog.at_level(logging.INFO, logger="erudi"):
                created_engine = core.init_database(url)  # lazy: no connection
            bound_lines = [r.message for r in caplog.records if "Database bound" in r.message]
            assert bound_lines, "init_database emitted no 'Database bound' log"
            assert "S3cretPassw0rd" not in bound_lines[0]
            assert "***" in bound_lines[0]  # SQLAlchemy masks the password
            # The rest of the URL stays useful for debugging.
            assert "127.0.0.1:5499/erudi" in bound_lines[0]
        finally:
            if created_engine is not None:
                created_engine.dispose()
            core.db_engine = previous_engine
            core.SessionLocal.configure(bind=previous_engine)

    def test_sanitizer_never_raises_on_garbage(self):
        from src.database.core import _sanitize_url_for_log

        assert _sanitize_url_for_log("!!not a url!!") == "<unparseable database url>"

    def test_sanitizer_keeps_passwordless_url_intact(self):
        from src.database.core import _sanitize_url_for_log

        url = "postgresql+psycopg://127.0.0.1:5433/erudi"
        assert _sanitize_url_for_log(url) == url


# ============ Ingestion: chunking stats (covers the read→chunk phase) ============


@pytest.mark.unit
class TestChunkingLogging:
    def test_chunk_document_logs_count_and_avg_tokens(self, caplog):
        from src.ingestion.chunking import chunk_document
        from src.ingestion.types import ExtractedDocument

        document = ExtractedDocument(
            markdown="# Title\n\n" + ("word " * 400),
            status="active",
            metadata={"extractor": "text"},
        )
        with caplog.at_level(logging.INFO, logger="erudi"):
            chunks = chunk_document(document)
        assert chunks
        messages = [r.message for r in caplog.records]
        assert any(
            "Document chunked" in m and f"{len(chunks)} chunks" in m and "tokens/chunk" in m
            for m in messages
        ), f"no chunking log found in: {messages}"

    def test_pending_vision_document_logs_nothing_at_info(self, caplog):
        from src.ingestion.chunking import chunk_document
        from src.ingestion.types import ExtractedDocument

        document = ExtractedDocument(markdown="", status="pending_vision")
        with caplog.at_level(logging.INFO, logger="erudi"):
            assert chunk_document(document) == []
        assert not [r for r in caplog.records if "Document chunked" in r.message]


# ============ Engine generation_guard bracket ============


@pytest.mark.unit
class TestGenerationGuardLogging:
    async def test_guard_logs_acquire_and_release_with_held_duration(self, caplog):
        from src.engines.base_engine import BaseEngine

        class _GuardProbeEngine(BaseEngine):
            """Guard is a classmethod: no instantiation, abstracts stay unimplemented."""

        with caplog.at_level(logging.INFO, logger="erudi"):
            async with _GuardProbeEngine.generation_guard():
                pass
        messages = [r.message for r in caplog.records]
        assert any("generation_guard acquired" in m for m in messages)
        assert any("generation_guard released" in m and "held" in m for m in messages)


# ============ KB mode decision (per-turn) ============


@pytest.mark.unit
class TestKbModeLogging:
    @staticmethod
    def _llm(**overrides):
        base = dict(
            name="M",
            param_size=7.0,
            is_attached_to_kb=False,
            kb_id=None,
            supports_tools=None,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_plain_mode_logged_with_reason(self, caplog):
        from src.agents.kb_mode import plan_turn

        with caplog.at_level(logging.INFO, logger="erudi"):
            plan_turn(self._llm(), question="hi", retrieve=lambda: [])
        messages = [r.message for r in caplog.records]
        assert any(
            "Turn mode: plain" in m and "no KB attached" in m for m in messages
        ), f"no plain-mode log found in: {messages}"

    def test_agentic_mode_logged(self, caplog, monkeypatch):
        from src.agents.kb_mode import plan_turn
        from src.core import config

        # Agentic KB is opt-in (#288); enable it to exercise the log line.
        monkeypatch.setattr(config, "KB_AGENTIC_MODE", True)
        llm = self._llm(is_attached_to_kb=True, kb_id=7, supports_tools=True)
        with caplog.at_level(logging.INFO, logger="erudi"):
            plan_turn(llm, question="hi", retrieve=lambda: [])
        assert any(
            "Turn mode: agentic KB" in r.message and "kb_id=7" in r.message for r in caplog.records
        )


# ============ Every record goes through the project logger ============


@pytest.mark.unit
class TestNoRootLoggerCalls:
    """``logging.info(...)`` on the ROOT logger is not how this app logs.

    The root logger carries a bridge into ``backend.log`` so that another
    LIBRARY's warnings are not lost (``RootFileBridge``), but it is a floor
    for code we do not own, not a second way to log:

    * it keeps ``WARNING`` and above, so a root ``INFO`` -- the lifecycle
      records the app writes on purpose -- disappears entirely, and
    * the root logger's own level is ``WARNING``, so those calls are dropped
      before any handler sees them.

    Records written there also lose the ``erudi`` name the readers key on.
    This scan is the enforcement: every log call in ``src/`` goes through a
    ``logger`` obtained from ``src.core.logging``.
    """

    LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical", "log"}

    def _root_log_calls(self) -> list[str]:
        import ast
        from pathlib import Path

        src = Path(__file__).resolve().parents[1] / "src"
        offenders: list[str] = []
        for path in sorted(src.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr in self.LOG_METHODS
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "logging"
                ):
                    where = f"{path.relative_to(src.parent)}:{node.lineno}"
                    offenders.append(f"{where} logging.{func.attr}")
        return offenders

    def test_no_module_logs_through_the_root_logger(self):
        offenders = self._root_log_calls()
        assert not offenders, (
            "these records never reach backend.log -- use "
            "`from src.core.logging import logger`:\n" + "\n".join(offenders)
        )
