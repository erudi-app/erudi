"""P7 — KB services: assistant lifecycle + real ingestion pipeline.

Two harnesses on the same embedded cluster:
- Lifecycle/state-machine tests run inside the standard rollback harness
  (``test_db_session``).
- Pipeline tests use a REAL committed session (``real_db_session``): the
  vector store writes through its own connection, so harness-transaction
  rows would be invisible to the chunks' FKs — exactly like the production
  background task, which opens a fresh ``SessionLocal``.
"""

import pytest
import psycopg
from sqlalchemy.orm import sessionmaker

from src.domains.knowledge_base.repository import COPIED_FIELDS
from src.domains.knowledge_base.services import KB_Service
from src.entities.KnowledgeBase import KnowledgeBase
from src.entities.KnowledgeDocument import KnowledgeDocument
from src.entities.Llm import Llm

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

        with pytest.raises(ValueError, match="No searchable content could be indexed"):
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

        with pytest.raises(ValueError, match="No searchable content could be indexed"):
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
