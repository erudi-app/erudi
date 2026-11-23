"""P7 — KB services: assistant lifecycle + real ingestion pipeline.

Two harnesses on the same embedded cluster:
- Lifecycle/state-machine tests run inside the standard rollback harness
  (``test_db_session``).
- Pipeline tests use a REAL committed session (``real_db_session``): the
  vector store writes through its own connection, so harness-transaction
  rows would be invisible to the chunks' FKs — exactly like the production
  background task, which opens a fresh ``SessionLocal``.
"""

import logging
import threading

import pytest
import psycopg
from sqlalchemy.orm import sessionmaker

from src.core.exceptions import NothingIndexedException
from src.domains.knowledge_base.repository import COPIED_FIELDS
from src.domains.knowledge_base.services import KB_Service
from src.entities.KnowledgeBase import KnowledgeBase
from src.entities.KnowledgeDocument import KnowledgeDocument
from src.entities.Llm import Llm
from tests.test_ingestion_reader import INVALID_DOCUMENTS

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Lifecycle / state machine (rollback harness)
# ---------------------------------------------------------------------------


class TestAssistantLifecycle:
    def test_create_kb_assistant_contract(self, test_db_session, mock_llm):
        service = KB_Service()
        llm_id, job_id = service.create_kb_assistant(
            db=test_db_session,
            base_llm_id=mock_llm.id,
            model_name="Assistant Docs",
            description="Assistant spécialisé",
            file_paths=["/tmp/whatever.pdf"],
        )

        specialized = test_db_session.query(Llm).get(llm_id)
        assert specialized.is_attached_to_kb
        assert specialized.local == 1
        assert specialized.kb_id is not None
        assert specialized.name == "Assistant Docs"

        job = service.repo.get_kb_job_by_id(test_db_session, job_id)
        assert job.status == "pending"
        assert job.base_model_id == mock_llm.id
        assert job.new_model_id == llm_id
        assert job.kb_id == specialized.kb_id

    def test_specialized_llm_inherits_all_descriptive_metadata(self, test_db_session, mock_llm):
        # Regression for #209: create_specialized_llm used to copy only a
        # hand-picked subset of the base's columns, so the assistant's landing
        # card rendered generic/empty (category, model_metadata were dropped) and
        # downstream capability reads got wrong nulls. Every descriptive column
        # must be inherited; asserting the whole COPIED_FIELDS set keeps this
        # honest as the list grows.
        mock_llm.category = "code"
        mock_llm.model_metadata = '{"context_length": 32768}'
        mock_llm.supports_tools = True
        mock_llm.supports_tools_wire = True
        mock_llm.type = "qwen"
        mock_llm.param_size = 14.0
        mock_llm.quantized = False
        test_db_session.flush()

        service = KB_Service()
        llm_id, _ = service.create_kb_assistant(
            db=test_db_session,
            base_llm_id=mock_llm.id,
            model_name="Metadata-rich assistant",
            description="Assistant desc",
            file_paths=["/tmp/whatever.pdf"],
        )

        specialized = test_db_session.query(Llm).get(llm_id)
        # Every descriptive column mirrors the base, whatever the list grows to.
        for field in COPIED_FIELDS:
            assert getattr(specialized, field) == getattr(mock_llm, field), field

        # Identity/state stays assistant-specific, never copied from the base.
        assert specialized.id != mock_llm.id
        assert specialized.name == "Metadata-rich assistant"
        assert specialized.description == "Assistant desc"
        assert specialized.local == 1
        assert specialized.is_attached_to_kb
        assert specialized.is_base is False
        assert specialized.kb_id != mock_llm.kb_id

    def test_kb_assistant_inherits_supports_tools(self, test_db_session, mock_llm):
        # A KB assistant built from a tool-capable base model must stay
        # tool-capable: otherwise plan_turn routes it to the systematic path and
        # the agentic search_knowledge_base tool is never offered (#84).
        mock_llm.supports_tools = True
        test_db_session.flush()

        service = KB_Service()
        llm_id, _ = service.create_kb_assistant(
            db=test_db_session,
            base_llm_id=mock_llm.id,
            model_name="Tool-capable assistant",
            description="",
            file_paths=["/tmp/whatever.pdf"],
        )

        specialized = test_db_session.query(Llm).get(llm_id)
        assert specialized.supports_tools is True

    def test_kb_assistant_inherits_the_publisher_generation_hints(self, test_db_session, mock_llm):
        # An assistant shares its base's weights, so the publisher's sampling
        # hints (#388) are its own: without them a Qwen3 assistant chatted at
        # the neutral 0.2 / 0.95 without top_k and the UI claimed "no publisher
        # recommendation" for weights that do ship one.
        from src.database.generation_hints import resolve_sampling_defaults

        mock_llm.generation_hints = {
            "base_repo": "Qwen/Qwen3-4B",
            "generation_config": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
            "supports_thinking": True,
            "context_length": 40960,
            "captured_at": "2026-08-28",
            "source_stage": "base_generation_config",
            "evidence": None,
        }
        test_db_session.flush()

        service = KB_Service()
        llm_id, _ = service.create_kb_assistant(
            db=test_db_session,
            base_llm_id=mock_llm.id,
            model_name="Hinted assistant",
            description="",
            file_paths=["/tmp/whatever.pdf"],
        )

        specialized = test_db_session.query(Llm).get(llm_id)
        assert specialized.generation_hints == mock_llm.generation_hints
        resolved = resolve_sampling_defaults(specialized)
        assert resolved.to_dict() == resolve_sampling_defaults(mock_llm).to_dict()
        assert (resolved.temperature, resolved.top_k) == (0.6, 20)
        assert resolved.source == "base_generation_config"

    def test_kb_assistant_inherits_supports_tools_wire(self, test_db_session, mock_llm):
        # #298: the verified wire capability must follow the weights. An
        # assistant shares its base's artifact, so its wire verdict is the
        # base's; without inheritance every assistant would route systematic
        # until a redundant re-detection.
        mock_llm.supports_tools = True
        mock_llm.supports_tools_wire = True
        test_db_session.flush()

        service = KB_Service()
        llm_id, _ = service.create_kb_assistant(
            db=test_db_session,
            base_llm_id=mock_llm.id,
            model_name="Wire-verified assistant",
            description="",
            file_paths=["/tmp/whatever.pdf"],
        )

        specialized = test_db_session.query(Llm).get(llm_id)
        assert specialized.supports_tools_wire is True

    def test_create_requires_local_base_llm(self, test_db_session):
        service = KB_Service()
        with pytest.raises(ValueError, match="not found or not local"):
            service.create_kb_assistant(
                db=test_db_session,
                base_llm_id=99999999,
                model_name="X",
                description="",
                file_paths=[],
            )

    def test_update_requires_attached_kb(self, test_db_session, mock_llm):
        service = KB_Service()
        with pytest.raises(ValueError, match="not attached"):
            service.update_existing_kb(db=test_db_session, base_llm_id=mock_llm.id, file_paths=[])

    def test_update_creates_job_on_same_llm(self, test_db_session, mock_llm_with_kb):
        llm, kb = mock_llm_with_kb
        service = KB_Service()
        llm_id, job_id = service.update_existing_kb(
            db=test_db_session, base_llm_id=llm.id, file_paths=["/tmp/new.pdf"]
        )
        assert llm_id == llm.id
        job = service.repo.get_kb_job_by_id(test_db_session, job_id)
        assert job.new_model_id == llm.id == job.base_model_id  # update marker
        assert job.kb_id == kb.id

    def test_failed_job_status_triggers_cleanup(self, test_db_session, mock_llm):
        service = KB_Service()
        llm_id, job_id = service.create_kb_assistant(
            db=test_db_session,
            base_llm_id=mock_llm.id,
            model_name="Doomed",
            description="",
            file_paths=[],
        )
        kb_id = test_db_session.query(Llm).get(llm_id).kb_id
        job = service.repo.get_kb_job_by_id(test_db_session, job_id)
        service.repo.update_kb_job_status(test_db_session, job, "failed", "boom")

        status = service.get_kb_job_status(test_db_session, llm_id)

        assert status["status"] == "failed"
        assert status["error_message"] == "boom"
        # Cleanup ran: specialized LLM and its KB are gone, the job remains.
        assert test_db_session.query(Llm).get(llm_id) is None
        assert test_db_session.query(KnowledgeBase).get(kb_id) is None
        assert service.repo.get_kb_job_by_id(test_db_session, job_id) is not None

    def test_status_reflects_latest_job_not_the_first(self, test_db_session, mock_llm_with_kb):
        """An assistant accumulates one KBJob per creation/update, all
        sharing the same new_model_id. Without ordering by recency, the
        lookup used to return whichever row the query happened to hand back
        first -- the original creation job -- so a later update's real
        outcome (failure included) stayed permanently masked behind that
        first job's "completed" status. Reproduced live: an update that
        failed with "no searchable content" still reported success."""
        llm, kb = mock_llm_with_kb
        service = KB_Service()
        first_job = service.repo.create_kb_job(
            db=test_db_session,
            base_model_id=llm.id,
            new_model_id=llm.id,
            kb_id=kb.id,
            status="completed",
        )
        second_job = service.repo.create_kb_job(
            db=test_db_session,
            base_model_id=llm.id,
            new_model_id=llm.id,
            kb_id=kb.id,
        )
        service.repo.update_kb_job_status(test_db_session, second_job, "failed", "boom")
        assert first_job.id < second_job.id

        status = service.get_kb_job_status(test_db_session, llm.id)

        assert status["status"] == "failed"
        assert status["error_message"] == "boom"

    def _doc(self, service, db, kb, name, status, created_at=None):
        doc = service.repo.create_document(
            db, kb_id=kb.id, name=name, content_hash_sha256=name.ljust(64, "0"), size_bytes=1
        )
        service.repo.update_document_status(db, doc, status)
        if created_at is not None:
            doc.created_at = created_at
            db.flush()
        return doc

    def test_status_names_the_files_that_failed_in_this_job(
        self, test_db_session, mock_llm_with_kb
    ):
        from datetime import datetime, timedelta, timezone

        llm, kb = mock_llm_with_kb
        service = KB_Service()
        old = datetime.now(timezone.utc) - timedelta(days=2)
        # A failure left by an earlier job, and documents that did not fail.
        self._doc(service, test_db_session, kb, "earlier-job.docx", "failed", created_at=old)
        job = service.repo.create_kb_job(
            db=test_db_session, base_model_id=llm.id, new_model_id=llm.id, kb_id=kb.id
        )
        self._doc(service, test_db_session, kb, "maintenance.docx", "failed")
        self._doc(service, test_db_session, kb, "notes.txt", "active")
        self._doc(service, test_db_session, kb, "blank.txt", "empty")
        service.repo.update_kb_job_status(test_db_session, job, "completed")

        status = service.get_kb_job_status(test_db_session, llm.id)

        assert status["status"] == "completed"
        assert status["failed_files"] == ["maintenance.docx"]

    def test_status_has_no_failed_files_when_everything_was_indexed(
        self, test_db_session, mock_llm_with_kb
    ):
        llm, kb = mock_llm_with_kb
        service = KB_Service()
        job = service.repo.create_kb_job(
            db=test_db_session, base_model_id=llm.id, new_model_id=llm.id, kb_id=kb.id
        )
        self._doc(service, test_db_session, kb, "notes.txt", "active")
        service.repo.update_kb_job_status(test_db_session, job, "completed")

        assert service.get_kb_job_status(test_db_session, llm.id)["failed_files"] == []

    def test_a_failed_creation_still_names_its_files_before_the_cleanup(
        self, test_db_session, mock_llm
    ):
        service = KB_Service()
        llm_id, job_id = service.create_kb_assistant(
            db=test_db_session,
            base_llm_id=mock_llm.id,
            model_name="Doomed",
            description="",
            file_paths=[],
        )
        kb = service.repo.get_knowledge_base_by_id(
            test_db_session, test_db_session.query(Llm).get(llm_id).kb_id
        )
        self._doc(service, test_db_session, kb, "maintenance.docx", "failed")
        job = service.repo.get_kb_job_by_id(test_db_session, job_id)
        service.repo.update_kb_job_status(test_db_session, job, "failed", "boom")

        status = service.get_kb_job_status(test_db_session, llm_id)

        assert status["failed_files"] == ["maintenance.docx"]

    def test_status_carries_nothing_indexed_and_the_names_by_reason(
        self, test_db_session, mock_llm_with_kb
    ):
        """A job that indexed nothing says so with a code the interface
        translates, and names each file under the reason it was not indexed;
        a file that failed for an internal cause stays in failed_files."""
        from datetime import datetime, timedelta, timezone

        llm, kb = mock_llm_with_kb
        service = KB_Service()
        # The fixture's document belongs to an earlier job.
        for document in test_db_session.query(KnowledgeDocument).filter_by(kb_id=kb.id):
            document.created_at = datetime.now(timezone.utc) - timedelta(days=2)
        test_db_session.flush()
        job = service.repo.create_kb_job(
            db=test_db_session, base_model_id=llm.id, new_model_id=llm.id, kb_id=kb.id
        )
        self._doc(service, test_db_session, kb, "scan.pdf", "pending_vision")
        self._doc(service, test_db_session, kb, "blank.txt", "empty")
        self._doc(service, test_db_session, kb, "broken.docx", "unreadable")
        self._doc(service, test_db_session, kb, "maintenance.docx", "failed")
        service.repo.update_kb_job_status(
            test_db_session, job, "failed", "No searchable content could be indexed (...)."
        )

        status = service.get_kb_job_status(test_db_session, llm.id)

        assert status["status"] == "failed"
        assert status["error_code"] == "nothing_indexed"
        # The English message stays, for the log.
        assert status["error_message"].startswith("No searchable content")
        assert status["failed_files"] == ["maintenance.docx"]
        assert status["files_by_reason"] == {
            "pending_vision": ["scan.pdf"],
            "empty": ["blank.txt"],
            "unreadable": ["broken.docx"],
        }

    def test_a_failure_before_any_file_carries_no_error_code(self, test_db_session, mock_llm):
        service = KB_Service()
        llm_id, job_id = service.create_kb_assistant(
            db=test_db_session,
            base_llm_id=mock_llm.id,
            model_name="Doomed",
            description="",
            file_paths=[],
        )
        job = service.repo.get_kb_job_by_id(test_db_session, job_id)
        service.repo.update_kb_job_status(test_db_session, job, "failed", "boom")

        status = service.get_kb_job_status(test_db_session, llm_id)

        assert status["error_code"] is None
        assert status["files_by_reason"] == {"pending_vision": [], "empty": [], "unreadable": []}

    def test_a_completed_job_names_its_unreadable_files_without_an_error_code(
        self, test_db_session, mock_llm_with_kb
    ):
        llm, kb = mock_llm_with_kb
        service = KB_Service()
        job = service.repo.create_kb_job(
            db=test_db_session, base_model_id=llm.id, new_model_id=llm.id, kb_id=kb.id
        )
        self._doc(service, test_db_session, kb, "notes.txt", "active")
        self._doc(service, test_db_session, kb, "broken.docx", "unreadable")
        service.repo.update_kb_job_status(test_db_session, job, "completed")

        status = service.get_kb_job_status(test_db_session, llm.id)

        assert status["error_code"] is None
        assert status["failed_files"] == []
        assert status["files_by_reason"]["unreadable"] == ["broken.docx"]

    def test_update_is_refused_while_another_job_runs_on_the_same_kb(
        self, test_db_session, mock_llm_with_kb
    ):
        """Two jobs on one KB would both see the same failed row and race on the
        unique (kb_id, hash) constraint: the server refuses the second (409)."""
        from src.core.exceptions import StateConflictException

        llm, kb = mock_llm_with_kb
        service = KB_Service()
        service.update_existing_kb(db=test_db_session, base_llm_id=llm.id, file_paths=["/a.pdf"])

        with pytest.raises(StateConflictException) as exc:
            service.update_existing_kb(
                db=test_db_session, base_llm_id=llm.id, file_paths=["/b.pdf"]
            )
        assert exc.value.status_code == 409

    def test_update_is_accepted_once_the_previous_job_finished(
        self, test_db_session, mock_llm_with_kb
    ):
        llm, kb = mock_llm_with_kb
        service = KB_Service()
        _, first = service.update_existing_kb(
            db=test_db_session, base_llm_id=llm.id, file_paths=["/a.pdf"]
        )
        service.repo.update_kb_job_status(
            test_db_session, service.repo.get_kb_job_by_id(test_db_session, first), "failed", "x"
        )

        _, second = service.update_existing_kb(
            db=test_db_session, base_llm_id=llm.id, file_paths=["/b.pdf"]
        )
        assert second != first

    def test_failed_update_job_does_not_delete_the_existing_assistant(
        self, test_db_session, mock_llm_with_kb
    ):
        """A failed UPDATE (base_model_id == new_model_id, see
        update_existing_kb) must never trigger the create-failure cleanup:
        the assistant and its KB already existed and worked before this
        update was attempted -- only a failed CREATION has nothing to lose."""
        llm, kb = mock_llm_with_kb
        service = KB_Service()
        job = service.repo.create_kb_job(
            db=test_db_session,
            base_model_id=llm.id,
            new_model_id=llm.id,
            kb_id=kb.id,
        )
        service.repo.update_kb_job_status(test_db_session, job, "failed", "boom")

        status = service.get_kb_job_status(test_db_session, llm.id)

        assert status["status"] == "failed"
        assert test_db_session.query(Llm).get(llm.id) is not None
        assert test_db_session.query(KnowledgeBase).get(kb.id) is not None


# ---------------------------------------------------------------------------
# Real ingestion pipeline (committed session + live vector store)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kb_store(pg_test_cluster, _session_db_engine):
    from src.ingestion import vector_store

    store = vector_store.init_kb_store(pg_test_cluster)
    yield store
    vector_store.close_kb_store()
    with psycopg.connect(pg_test_cluster.psycopg_url, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS rag.kb_chunks CASCADE")


@pytest.fixture
def real_db_session(_session_db_engine):
    """A genuinely committing session (mirrors the background task's
    SessionLocal). Tracks created KBs and deletes them afterwards —
    cascades sweep documents and chunks."""
    factory = sessionmaker(autocommit=False, autoflush=False, bind=_session_db_engine)
    session = factory()
    created_kb_ids: list[int] = []
    yield session, created_kb_ids
    session.rollback()
    for kb_id in created_kb_ids:
        kb = session.query(KnowledgeBase).get(kb_id)
        if kb:
            session.delete(kb)
    session.commit()
    session.close()


def _make_kb_and_job(session, created_ids):
    service = KB_Service()
    kb = KnowledgeBase()
    session.add(kb)
    session.flush()
    created_ids.append(kb.id)
    job = service.repo.create_kb_job(db=session, base_model_id=None, new_model_id=None, kb_id=kb.id)
    session.commit()
    return service, kb, job


class TestIngestionPipeline:
    def test_full_pipeline_indexes_searches_and_flags_statuses(
        self, kb_store, real_db_session, tmp_path
    ):
        from src.ingestion.vector_store import search_kb_chunks_scored

        session, created = real_db_session
        service, kb, job = _make_kb_and_job(session, created)

        report = tmp_path / "procédure.md"
        report.write_text(
            "# Remboursement\n\nLa procédure de remboursement prend dix jours ouvrés.",
            encoding="utf-8",
        )
        photo = tmp_path / "photo.jpg"
        photo.write_bytes(b"\xff\xd8\xff\xe0 fake jpeg")

        service.process_and_index_documents(
            db=session, kb_job_id=job.id, file_paths=[str(report), str(photo)]
        )

        session.expire_all()
        assert service.repo.get_kb_job_by_id(session, job.id).status == "completed"

        documents = {d.name: d for d in session.query(KnowledgeDocument).filter_by(kb_id=kb.id)}
        assert documents["procédure.md"].status == "active"
        assert documents["photo.jpg"].status == "pending_vision"
        assert documents["procédure.md"].size_bytes > 0
        assert len(documents["procédure.md"].content_hash_sha256) == 64

        results = search_kb_chunks_scored("délai de remboursement", kb_id=kb.id)
        assert results and "dix jours" in results[0][0].page_content
        assert results[0][0].metadata["document_id"] == documents["procédure.md"].id

    def test_duplicate_file_is_skipped_not_reindexed(
        self, kb_store, real_db_session, tmp_path, pg_test_cluster
    ):
        session, created = real_db_session
        service, kb, job = _make_kb_and_job(session, created)

        doc = tmp_path / "notes.txt"
        doc.write_text("Contenu unique à ne pas dupliquer.", encoding="utf-8")

        service.process_and_index_documents(db=session, kb_job_id=job.id, file_paths=[str(doc)])
        # Same content again (update flow re-drops the same file).
        job2 = service.repo.create_kb_job(
            db=session, base_model_id=None, new_model_id=None, kb_id=kb.id
        )
        session.commit()
        service.process_and_index_documents(
            db=session, kb_job_id=job2.id, file_paths=[str(doc)], is_update=True
        )

        session.expire_all()
        assert service.repo.get_kb_job_by_id(session, job2.id).status == "completed"
        count = session.query(KnowledgeDocument).filter_by(kb_id=kb.id).count()
        assert count == 1
        with psycopg.connect(pg_test_cluster.psycopg_url) as conn:
            chunks = conn.execute(
                "SELECT COUNT(*) FROM rag.kb_chunks WHERE kb_id = %s", (kb.id,)
            ).fetchone()[0]
        assert chunks == 1  # not doubled

    def test_unsupported_file_marks_document_failed_but_job_completes(
        self, kb_store, real_db_session, tmp_path
    ):
        session, created = real_db_session
        service, kb, job = _make_kb_and_job(session, created)

        good = tmp_path / "ok.txt"
        good.write_text("Texte parfaitement lisible.", encoding="utf-8")
        bad = tmp_path / "mystère.xyz"
        bad.write_text("format inconnu")

        service.process_and_index_documents(
            db=session, kb_job_id=job.id, file_paths=[str(good), str(bad)]
        )

        session.expire_all()
        assert service.repo.get_kb_job_by_id(session, job.id).status == "completed"
        statuses = {
            d.name: d.status for d in session.query(KnowledgeDocument).filter_by(kb_id=kb.id)
        }
        assert statuses == {"ok.txt": "active", "mystère.xyz": "failed"}

    def test_job_fails_when_nothing_could_be_ingested(self, kb_store, real_db_session, tmp_path):
        session, created = real_db_session
        service, kb, job = _make_kb_and_job(session, created)

        bad = tmp_path / "broken.xyz"
        bad.write_text("nope")

        with pytest.raises(NothingIndexedException, match="No searchable content could be indexed"):
            service.process_and_index_documents(db=session, kb_job_id=job.id, file_paths=[str(bad)])

        session.expire_all()
        job_after = service.repo.get_kb_job_by_id(session, job.id)
        assert job_after.status == "failed"
        assert "No searchable content could be indexed" in job_after.error_message

    def test_empty_file_is_marked_empty_and_job_fails(self, kb_store, real_db_session, tmp_path):
        """A file that yields zero chunks must NOT be reported as indexed: the
        document is flagged 'empty' and a content-less job fails honestly (#133)."""
        session, created = real_db_session
        service, kb, job = _make_kb_and_job(session, created)

        blank = tmp_path / "blank.txt"
        blank.write_text("   \n\t\n   ", encoding="utf-8")  # whitespace -> 0 chunks

        with pytest.raises(NothingIndexedException, match="No searchable content could be indexed"):
            service.process_and_index_documents(
                db=session, kb_job_id=job.id, file_paths=[str(blank)]
            )

        session.expire_all()
        job_after = service.repo.get_kb_job_by_id(session, job.id)
        assert job_after.status == "failed"
        doc = session.query(KnowledgeDocument).filter_by(kb_id=kb.id, name="blank.txt").one()
        assert doc.status == "empty"

    def test_empty_file_alongside_content_still_completes(
        self, kb_store, real_db_session, tmp_path
    ):
        """An empty file does not sink a batch that indexed real content: the job
        completes, the good doc is 'active', the empty one is honestly 'empty'."""
        session, created = real_db_session
        service, kb, job = _make_kb_and_job(session, created)

        good = tmp_path / "good.md"
        good.write_text("# Notes\n\nLe délai est de dix jours ouvrés.", encoding="utf-8")
        blank = tmp_path / "blank.txt"
        blank.write_text("", encoding="utf-8")

        service.process_and_index_documents(
            db=session, kb_job_id=job.id, file_paths=[str(good), str(blank)]
        )

        session.expire_all()
        assert service.repo.get_kb_job_by_id(session, job.id).status == "completed"
        statuses = {
            d.name: d.status for d in session.query(KnowledgeDocument).filter_by(kb_id=kb.id)
        }
        assert statuses == {"good.md": "active", "blank.txt": "empty"}

    def test_a_failed_document_does_not_block_re_ingesting_the_same_file(
        self, kb_store, real_db_session, tmp_path, pg_test_cluster, monkeypatch
    ):
        """The dedup by SHA-256 must not keep a `failed` row: once the cause is
        fixed, dropping the same file again indexes it (#705). The failed row is
        replaced, so the file is neither duplicated nor left half-recorded."""
        session, created = real_db_session
        service, kb, job = _make_kb_and_job(session, created)

        doc = tmp_path / "maintenance.md"
        doc.write_text("# Maintenance\n\nLe groupe froid a été remplacé.", encoding="utf-8")

        real_read = service.reader.read

        def broken(path):
            raise FileNotFoundError("default-styles.xml")

        another = tmp_path / "other.md"
        another.write_text("# Autre\n\nUn second fichier lisible.", encoding="utf-8")
        monkeypatch.setattr(
            service.reader, "read", lambda p: real_read(p) if p.name == "other.md" else broken(p)
        )
        service.process_and_index_documents(
            db=session, kb_job_id=job.id, file_paths=[str(doc), str(another)]
        )
        session.expire_all()
        failed = session.query(KnowledgeDocument).filter_by(kb_id=kb.id, name="maintenance.md")
        assert [d.status for d in failed] == ["failed"]
        failed_id = failed.one().id

        # The cause is fixed: same bytes, same name, a new job.
        monkeypatch.undo()
        job2 = service.repo.create_kb_job(
            db=session, base_model_id=None, new_model_id=None, kb_id=kb.id
        )
        session.commit()
        service.process_and_index_documents(
            db=session, kb_job_id=job2.id, file_paths=[str(doc)], is_update=True
        )

        session.expire_all()
        assert service.repo.get_kb_job_by_id(session, job2.id).status == "completed"
        rows = session.query(KnowledgeDocument).filter_by(kb_id=kb.id, name="maintenance.md").all()
        assert [d.status for d in rows] == ["active"]
        assert rows[0].id != failed_id
        with psycopg.connect(pg_test_cluster.psycopg_url) as conn:
            chunks = conn.execute(
                "SELECT COUNT(*) FROM rag.kb_chunks WHERE document_id = %s", (rows[0].id,)
            ).fetchone()[0]
        assert chunks >= 1

    def test_a_file_whose_bytes_cannot_be_read_is_named_among_the_failed_files(
        self, kb_store, real_db_session, tmp_path, caplog
    ):
        """A file whose bytes cannot be read creates no document row; the job
        keeps its name in memory so the end of the job still names it. A moved
        file and a wrong path look the same here: an ERROR with the traceback,
        like every `failed` file."""
        session, created = real_db_session
        service, kb, job = _make_kb_and_job(session, created)
        good = tmp_path / "ok.txt"
        good.write_text("Texte parfaitement lisible.", encoding="utf-8")

        with caplog.at_level(logging.INFO, logger="erudi"):
            service.process_and_index_documents(
                db=session, kb_job_id=job.id, file_paths=[str(good), str(tmp_path / "ghost.txt")]
            )

        session.expire_all()
        failed, _, _ = service._job_files(session, service.repo.get_kb_job_by_id(session, job.id))
        assert failed == ["ghost.txt"]
        (error,) = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert "ghost.txt" in error.getMessage() and error.exc_info

    @pytest.mark.parametrize("name, build, library_error", INVALID_DOCUMENTS)
    def test_an_invalid_document_is_a_warning_without_traceback_and_unreadable(
        self, kb_store, real_db_session, tmp_path, caplog, name, build, library_error
    ):
        """A file that is not what its extension says is the user's file, not
        an app bug (#752): one WARNING line without a traceback, the document
        `unreadable`, the job named it under that reason, no ERROR anywhere."""
        session, created = real_db_session
        service, kb, job = _make_kb_and_job(session, created)
        good = tmp_path / "ok.txt"
        good.write_text("Texte parfaitement lisible.", encoding="utf-8")
        bad = tmp_path / name
        build(bad)

        with caplog.at_level(logging.INFO, logger="erudi"):
            service.process_and_index_documents(
                db=session, kb_job_id=job.id, file_paths=[str(good), str(bad)]
            )

        session.expire_all()
        job_row = service.repo.get_kb_job_by_id(session, job.id)
        assert job_row.status == "completed"
        statuses = {
            d.name: d.status for d in session.query(KnowledgeDocument).filter_by(kb_id=kb.id)
        }
        assert statuses == {"ok.txt": "active", name: "unreadable"}
        assert [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR] == []
        (warning,) = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and name in r.getMessage() and r.name == "erudi"
        ]
        assert library_error in warning.getMessage()
        assert not warning.exc_info
        failed, by_reason, _ = service._job_files(session, job_row)
        assert failed == []
        assert by_reason["unreadable"] == [name]

    @pytest.mark.parametrize("where", ["extractor", "embedding"])
    def test_an_internal_failure_stays_an_error_with_traceback(
        self, kb_store, real_db_session, tmp_path, caplog, monkeypatch, where
    ):
        from src.domains.knowledge_base import services

        session, created = real_db_session
        service, kb, job = _make_kb_and_job(session, created)
        doc = tmp_path / "notes.md"
        doc.write_text("# Notes\n\nContenu à indexer.", encoding="utf-8")

        def explode(*args, **kwargs):
            if where == "extractor":
                raise AttributeError("'NoneType' object has no attribute 'text'")
            raise RuntimeError("embedding child died")

        if where == "extractor":
            monkeypatch.setattr(service.reader, "read", explode)
        else:
            monkeypatch.setattr(services, "add_kb_chunks", explode)

        with caplog.at_level(logging.INFO, logger="erudi"):
            with pytest.raises(NothingIndexedException) as caught:
                service.process_and_index_documents(
                    db=session, kb_job_id=job.id, file_paths=[str(doc)]
                )

        assert caught.value.has_internal_failures
        session.expire_all()
        assert session.query(KnowledgeDocument).filter_by(kb_id=kb.id).one().status == "failed"
        (error,) = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert "notes.md" in error.getMessage()
        assert error.exc_info

    def test_a_job_that_indexed_only_user_unreadable_files_logs_no_error(
        self, kb_store, real_db_session, tmp_path, caplog, monkeypatch
    ):
        """A scan-only, an empty and an invalid file: the job fails (nothing
        searchable was added) but the user's files are the cause, so the task
        boundary writes one INFO line and nothing reaches ERROR (#764)."""
        from pypdf import PdfWriter

        from src.domains.knowledge_base import endpoints as kb_endpoints

        session, created = real_db_session
        service, kb, job = _make_kb_and_job(session, created)
        scan = tmp_path / "scan.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        with scan.open("wb") as handle:
            writer.write(handle)
        blank = tmp_path / "blank.txt"
        blank.write_text("", encoding="utf-8")
        broken = tmp_path / "broken.docx"
        broken.write_text("a text file renamed", encoding="utf-8")

        class _Session:
            """The task closes its session; the test keeps using this one."""

            def __getattr__(self, attribute):
                return getattr(session, attribute)

            def close(self):
                pass

        monkeypatch.setattr(kb_endpoints, "SessionLocal", lambda: _Session())
        monkeypatch.setattr(kb_endpoints, "KB_Service", lambda: service)

        with caplog.at_level(logging.INFO, logger="erudi"):
            kb_endpoints._run_kb_update_task(job.id, [str(scan), str(blank), str(broken)])

        assert [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR] == []
        (boundary,) = [r for r in caplog.records if f"update job {job.id}" in r.getMessage()]
        assert "indexed nothing" in boundary.getMessage()
        assert boundary.levelno == logging.INFO
        assert not boundary.exc_info
        session.expire_all()
        job_row = service.repo.get_kb_job_by_id(session, job.id)
        assert job_row.status == "failed"
        assert job_row.error_message.startswith("No searchable content could be indexed")
        failed, by_reason, _ = service._job_files(session, job_row)
        assert failed == []
        assert by_reason == {
            "pending_vision": ["scan.pdf"],
            "empty": ["blank.txt"],
            "unreadable": ["broken.docx"],
        }

    @pytest.mark.parametrize(
        "name, content, reason",
        [
            ("broken.docx", "a text file renamed", "unreadable"),
            ("blank.txt", "   ", "empty"),
            ("photo.jpg", "not a real jpeg", "pending_vision"),
        ],
    )
    def test_a_file_the_user_adds_again_is_named_again(
        self, kb_store, real_db_session, tmp_path, name, content, reason
    ):
        """A row an earlier job left unindexed (an invalid file, an empty one, a
        scan) does not hide the file behind the dedup: the next job reads it
        again and names it again, instead of passing it for a duplicate."""
        session, created = real_db_session
        service, kb, job = _make_kb_and_job(session, created)
        good = tmp_path / "ok.txt"
        good.write_text("Texte parfaitement lisible.", encoding="utf-8")
        again = tmp_path / name
        again.write_text(content, encoding="utf-8")
        service.process_and_index_documents(
            db=session, kb_job_id=job.id, file_paths=[str(good), str(again)]
        )

        job2 = service.repo.create_kb_job(
            db=session, base_model_id=None, new_model_id=None, kb_id=kb.id
        )
        session.commit()
        with pytest.raises(NothingIndexedException):
            service.process_and_index_documents(
                db=session, kb_job_id=job2.id, file_paths=[str(again)], is_update=True
            )

        session.expire_all()
        job2_row = service.repo.get_kb_job_by_id(session, job2.id)
        assert service._job_files(session, job2_row)[1][reason] == [name]
        assert session.query(KnowledgeDocument).filter_by(kb_id=kb.id).count() == 2

    def test_two_identical_files_in_one_job_are_both_named(
        self, kb_store, real_db_session, tmp_path
    ):
        """Same bytes under two names in one job: the second does not replace
        the first one's row (the hash is unique per KB), and both are named."""
        session, created = real_db_session
        service, kb, job = _make_kb_and_job(session, created)
        first = tmp_path / "broken.docx"
        first.write_text("a text file renamed", encoding="utf-8")
        second = tmp_path / "copy of broken.docx"
        second.write_text("a text file renamed", encoding="utf-8")

        with pytest.raises(NothingIndexedException) as caught:
            service.process_and_index_documents(
                db=session, kb_job_id=job.id, file_paths=[str(first), str(second)]
            )

        assert caught.value.counts["unreadable"] == 2
        session.expire_all()
        job_row = service.repo.get_kb_job_by_id(session, job.id)
        failed, by_reason, _ = service._job_files(session, job_row)
        assert failed == []
        assert by_reason["unreadable"] == ["broken.docx", "copy of broken.docx"]
        assert session.query(KnowledgeDocument).filter_by(kb_id=kb.id).count() == 1

    def test_a_row_is_indexing_until_its_chunks_are_stored(
        self, kb_store, real_db_session, tmp_path, monkeypatch
    ):
        """The row is committed before the chunks (FK), so it must not look
        `active` until they are in: a crash in between leaves `indexing`."""
        from src.domains.knowledge_base import services

        session, created = real_db_session
        service, kb, job = _make_kb_and_job(session, created)
        doc = tmp_path / "notes.md"
        doc.write_text("# Notes\n\nContenu à indexer.", encoding="utf-8")
        seen = []
        real_add = services.add_kb_chunks

        def spy(**kwargs):
            seen.append(
                session.query(KnowledgeDocument).filter_by(id=kwargs["document_id"]).one().status
            )
            return real_add(**kwargs)

        monkeypatch.setattr(services, "add_kb_chunks", spy)
        service.process_and_index_documents(db=session, kb_job_id=job.id, file_paths=[str(doc)])

        session.expire_all()
        assert seen == ["indexing"]
        assert session.query(KnowledgeDocument).filter_by(kb_id=kb.id).one().status == "active"

    def test_a_failure_registering_a_file_does_not_escape_or_strand_the_job(
        self, kb_store, real_db_session, tmp_path, monkeypatch
    ):
        """An IntegrityError while registering one file stays inside the
        'never raises' contract: rolled back, the file counts as failed, and the
        job ends instead of staying `running`."""
        from sqlalchemy.exc import IntegrityError

        session, created = real_db_session
        service, kb, job = _make_kb_and_job(session, created)
        doc = tmp_path / "maintenance.md"
        doc.write_text("# Maintenance\n\nTexte.", encoding="utf-8")

        def clash(*args, **kwargs):
            raise IntegrityError("insert", {}, Exception("uq_knowledge_documents_kb_hash"))

        monkeypatch.setattr(service.repo, "create_document", clash)
        with pytest.raises(NothingIndexedException, match="No searchable content"):
            service.process_and_index_documents(db=session, kb_job_id=job.id, file_paths=[str(doc)])

        session.expire_all()
        job_row = service.repo.get_kb_job_by_id(session, job.id)
        assert job_row.status == "failed"
        assert service._job_files(session, job_row)[0] == ["maintenance.md"]


class TestFirstIngestionTimings:
    """The embedding child starts beside the job's first file (its reading,
    its chunking and the e5 tokenizer's load) instead of after them, and the
    job ends with one ``KB job N timings:`` line saying where the time went."""

    @staticmethod
    def _timing_lines(caplog, job_id):
        return [
            r.getMessage()
            for r in caplog.records
            if r.getMessage().startswith(f"KB job {job_id} timings:")
        ]

    def test_the_job_logs_one_ascii_timing_line_with_every_phase(
        self, kb_store, real_db_session, tmp_path, caplog
    ):
        session, created = real_db_session
        service, kb, job = _make_kb_and_job(session, created)
        doc = tmp_path / "délais.md"
        doc.write_text("# Délais\n\nLa livraison prend trois jours.", encoding="utf-8")

        with caplog.at_level(logging.INFO, logger="erudi"):
            service.process_and_index_documents(db=session, kb_job_id=job.id, file_paths=[str(doc)])

        (line,) = self._timing_lines(caplog, job.id)
        assert line.isascii()
        fields = dict(part.split("=") for part in line.split(": ", 1)[1].split())
        assert set(fields) == {
            "total_ms",
            "spawn_ms",
            "read_ms",
            "chunk_ms",
            "tokenizer_ms",
            "embed_ms",
            "db_ms",
        }
        # The suite's manager is disarmed: no child was started.
        assert fields["spawn_ms"] == "none"
        for name in ("total_ms", "read_ms", "chunk_ms", "tokenizer_ms", "embed_ms", "db_ms"):
            assert int(fields[name]) >= 0
        assert int(fields["chunk_ms"]) >= int(fields["tokenizer_ms"])

    def test_the_start_and_the_chunking_run_at_the_same_time(
        self, kb_store, real_db_session, tmp_path, monkeypatch
    ):
        """Neither waits for the other: the fake start only finishes once the
        job has chunked, and the chunking only goes on once the start runs.
        Run one after the other, each would wait out its 5 s and fail."""
        from src.domains.knowledge_base import services

        session, created = real_db_session
        service, kb, job = _make_kb_and_job(session, created)
        first = tmp_path / "a.md"
        first.write_text("# A\n\nPremier document.", encoding="utf-8")
        second = tmp_path / "b.md"
        second.write_text("# B\n\nSecond document.", encoding="utf-8")
        order = []
        start_running = threading.Event()
        chunked = threading.Event()
        overlapped = []

        class FakeStart:
            def __init__(self):
                self.seconds = None
                self.error = None
                self._done = threading.Event()

            def wait(self, timeout=None):
                return self._done.wait(timeout)

            def run(self):
                start_running.set()
                overlapped.append(chunked.wait(5.0))
                self.seconds = 0.0
                self._done.set()

        def start_in_background():
            order.append("start")
            start = FakeStart()
            threading.Thread(target=start.run, daemon=True).start()
            return start

        real_read = service.reader.read
        real_chunk = services.chunk_document

        def read(path):
            order.append("read")
            return real_read(path)

        def chunk(document):
            overlapped.append(start_running.wait(5.0))
            chunks = real_chunk(document)
            chunked.set()
            return chunks

        monkeypatch.setattr(services.embedder_child, "start_in_background", start_in_background)
        monkeypatch.setattr(service.reader, "read", read)
        monkeypatch.setattr(services, "chunk_document", chunk)
        service.process_and_index_documents(
            db=session, kb_job_id=job.id, file_paths=[str(first), str(second)]
        )

        assert order == ["start", "read", "read"]  # once per job, before the first read
        assert overlapped[:2] == [True, True]
        session.expire_all()
        assert service.repo.get_kb_job_by_id(session, job.id).status == "completed"

    def test_a_job_with_nothing_to_embed_starts_no_child(
        self, kb_store, real_db_session, tmp_path, monkeypatch, caplog
    ):
        """Images wait for a later tier and duplicates are already indexed:
        neither needs the child, so it is not started for them."""
        from src.domains.knowledge_base import services

        session, created = real_db_session
        service, kb, job = _make_kb_and_job(session, created)
        doc = tmp_path / "notes.md"
        doc.write_text("# Notes\n\nDéjà indexé.", encoding="utf-8")
        service.process_and_index_documents(db=session, kb_job_id=job.id, file_paths=[str(doc)])

        starts = []
        monkeypatch.setattr(
            services.embedder_child, "start_in_background", lambda: starts.append(1)
        )
        photo = tmp_path / "photo.png"
        photo.write_bytes(b"\x89PNG fake")
        second = service.repo.create_kb_job(
            db=session, base_model_id=None, new_model_id=None, kb_id=kb.id
        )
        session.commit()
        with caplog.at_level(logging.INFO, logger="erudi"):
            service.process_and_index_documents(
                db=session, kb_job_id=second.id, file_paths=[str(doc), str(photo)]
            )

        assert starts == []
        (line,) = self._timing_lines(caplog, second.id)
        assert " spawn_ms=none " in line

    def test_a_failed_start_is_reported_once_by_the_file_that_needed_it(
        self, kb_store, real_db_session, tmp_path, monkeypatch, caplog
    ):
        """The child cannot come up: the file is `failed` with ONE error
        record carrying the child's answer, the background start adds none,
        and the timing line names the start's failure."""
        from src.ingestion import embedder_child, vector_store
        from src.ingestion.embeddings import ChildE5Embeddings
        from tests.test_embedder_child import FakeHttp, FakeResponse, make_adapter

        session, created = real_db_session
        service, kb, job = _make_kb_and_job(session, created)
        doc = tmp_path / "panne.md"
        doc.write_text("# Panne\n\nLe service est indisponible.", encoding="utf-8")
        adapter = make_adapter(tmp_path)
        http = FakeHttp()
        http.post_handler = lambda url, json: FakeResponse(status_code=500, text="no model")
        monkeypatch.setattr(embedder_child, "_adapter_override", adapter)
        monkeypatch.setattr(embedder_child, "requests", http)
        monkeypatch.setattr(vector_store, "get_embeddings", ChildE5Embeddings)
        embedder_child._reset_for_tests()
        try:
            with caplog.at_level(logging.INFO, logger="erudi"):
                with pytest.raises(NothingIndexedException):
                    service.process_and_index_documents(
                        db=session, kb_job_id=job.id, file_paths=[str(doc)]
                    )
        finally:
            embedder_child._reset_for_tests()

        session.expire_all()
        assert session.query(KnowledgeDocument).filter_by(kb_id=kb.id).one().status == "failed"
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(errors) == 1
        assert "panne.md" in errors[0].getMessage()
        assert "HTTP 500" in errors[0].getMessage()
        (line,) = self._timing_lines(caplog, job.id)
        assert "spawn_error=EngineException" in line
