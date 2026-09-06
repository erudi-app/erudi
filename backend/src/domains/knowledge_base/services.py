"""Business logic layer for Knowledge Base domain.

Implements service pattern for Knowledge Base operations: KB/assistant
lifecycle, job state machine, and background ingestion orchestration.

Ingestion pipeline (per file, background task):
    SHA-256 → dedup against the KB's KnowledgeDocument rows → register the
    document → extract (DocumentReader, services stay format-agnostic) →
    ``pending_vision`` stops here (image/scanned PDF, indexed by the OCR/VLM
    tiers of a later release) → 3-pass chunking → hybrid vector store
    (``rag.kb_chunks``). Per-file failures mark THAT document ``failed`` and
    the run continues; the job only fails when nothing could be ingested.

Classes:
    KB_Service: Business logic for KB creation, updates, and job tracking.

Architecture:
    Endpoints → Services → Repository → Database
                     ↓
            src.ingestion (DocumentReader → chunking → vector store)
"""

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sqlalchemy.orm import Session

from src.domains.knowledge_base.repository import KB_Repository
from src.entities.KBJob import KBJobModel
from src.ingestion.chunking import chunk_document
from src.ingestion.reader import DocumentReader
from src.ingestion.vector_store import add_kb_chunks
from src.core.logging import logger


class KB_Service:
    """Business logic for Knowledge Base operations.

    Coordinates KB creation, updates, document processing, and background tasks.
    """

    def __init__(self):
        self.repo = KB_Repository()
        self.reader = DocumentReader()

    def get_kb_job_status(self, db: Session, llm_id: int) -> Dict[str, Any]:
        """Get KB job status with automatic cleanup for failed jobs.

        Args:
            db: Database session.
            llm_id: Specialized LLM ID (new_model_id in KBJob).

        Returns:
            Dict with status, updated_at, and error_message.

        Raises:
            ValueError: If KB job not found.
        """
        kb_job = self.repo.get_kb_job_by_model_id(db, llm_id)
        if not kb_job:
            raise ValueError(f"KB job not found for LLM ID {llm_id}")

        # Cleanup failed jobs -- but ONLY failed creations. An update job
        # shares its new_model_id with base_model_id (see update_existing_kb);
        # a creation job's new_model_id is a freshly-minted specialized LLM
        # distinct from its base. Deleting on any failure would wipe out an
        # already-working assistant just because a LATER update to it failed
        # (e.g. one bad file) -- the assistant and its existing KB content
        # must survive; only the half-built creation has nothing to lose.
        if kb_job.status == "failed" and kb_job.base_model_id != kb_job.new_model_id:
            self._cleanup_failed_job(db, llm_id, kb_job)

        return {
            "status": kb_job.status,
            "status_updated_at": kb_job.updated_at,
            "error_message": kb_job.error_message if kb_job.status == "failed" else None,
        }

    def _cleanup_failed_job(self, db: Session, llm_id: int, kb_job: KBJobModel) -> None:
        """Clean up database state after a failed KB job.

        Deletes the specialized LLM and its KnowledgeBase; KnowledgeDocument
        rows follow through ON DELETE CASCADE, and the failed job survives as
        an audit record with its refs nulled server-side (FK SET NULL).

        Args:
            db: Database session.
            llm_id: Specialized LLM ID to clean up.
            kb_job: Failed KBJob instance.
        """
        llm = self.repo.get_llm_by_id(db, llm_id)
        if not llm:
            return

        if llm.kb_id:
            kb = self.repo.get_knowledge_base_by_id(db, llm.kb_id)
            if kb:
                self.repo.delete_knowledge_base(db, kb)

        # Delete specialized LLM
        self.repo.delete_llm(db, llm)
        logger.info(f"Cleaned up failed KB job for LLM {llm_id}")

    def create_kb_assistant(
        self,
        db: Session,
        base_llm_id: int,
        model_name: str,
        description: str,
        file_paths: List[str],
    ) -> Tuple[int, int]:  # Returns (llm_id, kb_job_id)
        """Create new Knowledge Base assistant (database setup only).

        Creates specialized LLM, KnowledgeBase, and KBJob entities. Does NOT
        process documents — that's done in the background task.

        Args:
            db: Database session.
            base_llm_id: Base LLM ID to specialize from.
            model_name: Name for specialized assistant.
            description: Description for specialized assistant.
            file_paths: List of file paths to process.

        Returns:
            Tuple of (specialized_llm_id, kb_job_id) for background task.

        Raises:
            ValueError: If base LLM not found or not local.
        """
        # Validate base LLM
        base_llm = self.repo.get_local_llm_by_id(db, base_llm_id)
        if not base_llm:
            raise ValueError(f"Base LLM {base_llm_id} not found or not local")

        # Create Knowledge Base
        kb = self.repo.create_knowledge_base(db)

        # Create specialized LLM
        specialized_llm = self.repo.create_specialized_llm(
            db=db, name=model_name, description=description, base_llm=base_llm, kb_id=kb.id
        )

        # Create KBJob
        kb_job = self.repo.create_kb_job(
            db=db,
            base_model_id=base_llm.id,
            new_model_id=specialized_llm.id,
            kb_id=kb.id,
            status="pending",
        )

        db.commit()
        logger.info(
            f"Created KB assistant setup: LLM={specialized_llm.id}, " f"KB={kb.id}, Job={kb_job.id}"
        )

        return specialized_llm.id, kb_job.id

    def update_existing_kb(
        self, db: Session, base_llm_id: int, file_paths: List[str]
    ) -> Tuple[int, int]:  # Returns (llm_id, kb_job_id)
        """Queue update for existing KB with new documents (database setup only).

        Creates KBJob for update operation. Does NOT process documents - that's
        done in background task.

        Args:
            db: Database session.
            base_llm_id: Base LLM ID with existing KB attachment.
            file_paths: List of new file paths to add.

        Returns:
            Tuple of (base_llm_id, kb_job_id) for background task.

        Raises:
            ValueError: If LLM not found or KB not attached.
        """
        base_llm = self.repo.get_local_llm_by_id(db, base_llm_id)
        if not base_llm:
            raise ValueError(f"Base LLM {base_llm_id} not found or not local")

        if not base_llm.is_attached_to_kb or not base_llm.kb_id:
            raise ValueError(f"LLM {base_llm_id} is not attached to a KB")

        kb = self.repo.get_knowledge_base_by_id(db, base_llm.kb_id)
        if not kb:
            raise ValueError(f"KB {base_llm.kb_id} not found for LLM {base_llm_id}")

        # Create update job
        kb_job = self.repo.create_kb_job(
            db=db,
            base_model_id=base_llm.id,
            new_model_id=base_llm.id,  # Same LLM for updates
            kb_id=kb.id,
            status="pending",
        )

        db.commit()
        logger.info(f"Created KB update job: Job={kb_job.id}, KB={kb.id}")

        return base_llm.id, kb_job.id

    def process_and_index_documents(
        self, db: Session, kb_job_id: int, file_paths: List[str], is_update: bool = False
    ) -> None:
        """Ingest documents into the KB's vector store (background task).

        Per-file pipeline — see module docstring. Each file commits on its
        own: the document row must be VISIBLE to the vector store's separate
        connection (FK), and progress survives an interruption.

        Args:
            db: Database session (a fresh SessionLocal in background tasks).
            kb_job_id: KBJob ID to track progress.
            file_paths: List of file paths to process.
            is_update: True if updating existing KB, False if creating new.
        """
        start_s = time.perf_counter()
        try:
            kb_job = self.repo.get_kb_job_by_id(db, kb_job_id)
            if not kb_job:
                raise ValueError(f"KBJob {kb_job_id} not found")

            self.repo.update_kb_job_status(db, kb_job, "running")

            kb = self.repo.get_knowledge_base_by_id(db, kb_job.kb_id)
            if not kb:
                raise ValueError(f"KnowledgeBase {kb_job.kb_id} not found")

            logger.info(
                f"KB job {kb_job_id} started "
                f"({'update' if is_update else 'creation'}, kb_id={kb.id}, "
                f"{len(file_paths)} file(s))"
            )

            indexed, pending, skipped, empty, failed = 0, 0, 0, 0, 0
            for raw_path in file_paths:
                outcome = self._ingest_one_file(db, kb.id, raw_path)
                if outcome == "indexed":
                    indexed += 1
                elif outcome == "pending_vision":
                    pending += 1
                elif outcome == "skipped":
                    skipped += 1
                elif outcome == "empty":
                    empty += 1
                else:
                    failed += 1

            duration_ms = (time.perf_counter() - start_s) * 1000
            logger.info(
                f"KB job {kb_job_id} ({'update' if is_update else 'creation'}): "
                f"{indexed} indexed, {pending} pending_vision, "
                f"{skipped} duplicates, {empty} empty, {failed} failed, "
                f"duration_ms={duration_ms:.0f}"
            )

            # Don't report success when nothing queryable was added. A batch that
            # indexed zero chunks and had no duplicates (its files were empty,
            # pending_vision with no OCR tier, or failed) is a failure the user must
            # see — never a silent "completed". All-duplicates (skipped > 0) stays a
            # success: that content is already in the KB.
            if indexed == 0 and skipped == 0:
                raise ValueError(
                    f"No searchable content could be indexed "
                    f"({failed} failed, {empty} empty, {pending} pending vision)."
                )

            self.repo.update_kb_job_status(db, kb_job, "completed")

        except Exception as e:
            # Marked failed for the polling UI and re-raised: the background
            # task that owns this job logs the record, with the traceback.
            self._handle_indexing_error(db, kb_job_id, str(e))
            raise

    def _ingest_one_file(self, db: Session, kb_id: int, raw_path: str) -> str:
        """Ingest one file; returns "indexed" | "pending_vision" | "skipped"
        | "failed". Never raises — a broken file must not sink the batch."""
        path = Path(raw_path)
        try:
            content = path.read_bytes()
        except OSError as e:
            # One file skipped; the batch goes on (and fails as a whole only
            # when nothing at all could be indexed).
            logger.warning(f"KB {kb_id}: cannot read {path.name}, skipping it: {e}")
            return "failed"

        content_hash = hashlib.sha256(content).hexdigest()
        if self.repo.get_document_by_hash(db, kb_id, content_hash):
            logger.info(f"KB {kb_id}: {path.name} already ingested (dedup), skipping")
            return "skipped"

        document = self.repo.create_document(
            db,
            kb_id=kb_id,
            name=path.name,
            content_hash_sha256=content_hash,
            size_bytes=len(content),
        )
        # Commit BEFORE indexing: the vector store writes through its own
        # connection and the chunks' document_id FK must see this row.
        db.commit()

        try:
            extracted = self.reader.read(path)

            if extracted.status == "pending_vision":
                self.repo.update_document_status(db, document, "pending_vision")
                logger.info(f"KB {kb_id}: {path.name} accepted as pending_vision")
                return "pending_vision"

            chunks = chunk_document(extracted)
            if not chunks:
                # Extracted, but produced nothing indexable (blank/whitespace file
                # or a parser that yielded no text): honestly 'empty', never
                # reported as indexed.
                self.repo.update_document_status(db, document, "empty")
                logger.info(f"KB {kb_id}: {path.name} produced no indexable content")
                return "empty"
            add_kb_chunks(
                kb_id=kb_id,
                document_id=document.id,
                source_file=path.name,
                chunks=chunks,
            )
            logger.info(f"KB {kb_id}: {path.name} indexed ({len(chunks)} chunks)")
            return "indexed"

        except Exception as e:
            # The traceback matters here: this is where a broken embedding
            # model or a chunker that cannot load its tokenizer surfaces.
            logger.error(f"KB {kb_id}: ingestion failed for {path.name}: {e}", exc_info=True)
            self.repo.update_document_status(db, document, "failed")
            return "failed"

    def _handle_indexing_error(self, db: Session, kb_job_id: int, error_message: str) -> None:
        """Handle errors during indexing by marking the job as failed.

        Args:
            db: Database session.
            kb_job_id: KBJob ID that failed.
            error_message: Error description to store.
        """
        try:
            kb_job = self.repo.get_kb_job_by_id(db, kb_job_id)
            if not kb_job:
                return

            # Mark as failed
            self.repo.update_kb_job_status(db, kb_job, "failed", error_message=error_message)

            # Don't auto-cleanup here - let the status endpoint handle it
            # This way users can see the error message before cleanup

        except Exception as e:
            logger.error(f"Could not mark KB job {kb_job_id} as failed: {e}", exc_info=True)
