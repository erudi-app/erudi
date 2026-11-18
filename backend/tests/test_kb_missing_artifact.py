""" "KB present, embedding model missing" is a STATE on every path (#594).

A Knowledge Base whose embedding artifact left the disk must degrade like an
empty search everywhere the KB is consulted — one INFO, never an ERROR per
call, never a 500 — because the interface (the gate, the conversation/Arena
banner) owns the state:

- the agentic tool answers "could not be searched right now" to the model;
- the conversation's systematic retrieval degrades to no excerpts;
- the Arena's retrieval degrades the same way instead of KB_CORRUPTED;
- an ingestion is refused UP FRONT with the 409 message, before any file is
  parsed and marked failed one by one.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from src.agents.tools import KB_SEARCH_FAILED_TEXT, TurnToolContext, run_kb_search
from src.core.exceptions import EmbeddingModelMissingException
from src.ingestion import embedder_child, embeddings

pytestmark = pytest.mark.integration


def _raise_missing(*args, **kwargs):
    raise EmbeddingModelMissingException()


class TestAgenticTool:
    async def test_missing_model_is_one_info_and_a_curated_tool_answer(self, monkeypatch, caplog):
        monkeypatch.setattr("src.agents.tools.retrieve_kb_excerpts", _raise_missing)
        with caplog.at_level(logging.INFO):
            excerpts = await run_kb_search("q", TurnToolContext(kb_id=3, kb_token_budget=100))
        assert excerpts is None  # the tool then answers KB_SEARCH_FAILED_TEXT
        assert KB_SEARCH_FAILED_TEXT  # the contract the caller serves
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors == []
        infos = [
            r
            for r in caplog.records
            if r.levelno == logging.INFO and "embedding model" in r.getMessage()
        ]
        assert len(infos) == 1


class TestConversationRetrieval:
    def test_missing_model_degrades_to_no_excerpts_with_one_info(self, monkeypatch, caplog):
        from src.domains.conversations.services import ConversationService

        service = ConversationService.__new__(ConversationService)
        monkeypatch.setattr(
            "src.domains.conversations.services.retrieve_kb_excerpts", _raise_missing
        )
        llm = SimpleNamespace(id=1, is_attached_to_kb=True, kb_id=3, param_size=4.0)
        with caplog.at_level(logging.INFO):
            excerpts = service._retrieve_kb_excerpts(llm, "q")
        assert excerpts == []
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
        assert any("embedding model" in r.getMessage() for r in caplog.records)


class TestArenaRetrieval:
    def test_missing_model_degrades_instead_of_kb_corrupted(self, monkeypatch, caplog):
        from src.domains.arena.services import ArenaService

        service = ArenaService.__new__(ArenaService)
        monkeypatch.setattr("src.domains.arena.services.retrieve_kb_excerpts", _raise_missing)
        llm = SimpleNamespace(id=1, is_attached_to_kb=True, kb_id=3, param_size=4.0)
        strategy = {"use_kb_context": True, "kb_token_budget": 100}
        with caplog.at_level(logging.INFO):
            excerpts = service._retrieve_kb_excerpts(llm, "q", strategy)
        assert excerpts == []
        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
        assert any("embedding model" in r.getMessage() for r in caplog.records)


class TestIngestionRefusal:
    def test_the_job_is_refused_up_front_with_the_409_message(
        self, test_db_session, mock_llm_with_kb, monkeypatch
    ):
        from src.domains.knowledge_base.repository import KB_Repository
        from src.domains.knowledge_base.services import KB_Service

        llm, kb = mock_llm_with_kb
        repo = KB_Repository()
        kb_job = repo.create_kb_job(test_db_session, llm.id, llm.id, kb.id)
        test_db_session.commit()

        monkeypatch.setattr(
            "src.domains.knowledge_base.services.embedding_artifact_missing", lambda: True
        )
        touched = []
        service = KB_Service()
        monkeypatch.setattr(
            service, "_ingest_one_file", lambda *a, **k: touched.append(a) or "failed"
        )
        with pytest.raises(EmbeddingModelMissingException):
            service.process_and_index_documents(
                test_db_session, kb_job.id, ["/tmp/nowhere.pdf"], is_update=True
            )
        assert touched == []  # refused before any file was parsed
        status = service.get_kb_job_status(test_db_session, llm.id)
        assert status["status"] == "failed"
        assert "not downloaded" in (status["error_message"] or "")


class TestArtifactMissingPredicate:
    def test_reports_the_adapter_absent_artifact(self, monkeypatch):
        adapter = SimpleNamespace(artifact_available=lambda: False)
        monkeypatch.setattr(embedder_child, "_adapter_override", adapter)
        assert embeddings.embedding_artifact_missing()
        adapter.artifact_available = lambda: True
        assert not embeddings.embedding_artifact_missing()

    def test_a_disarmed_manager_never_reports_missing(self, monkeypatch):
        monkeypatch.setattr(embedder_child, "_adapter_override", None)
        assert not embeddings.embedding_artifact_missing()
