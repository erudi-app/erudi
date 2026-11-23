"""Business logic layer for Knowledge Base domain.

Implements service pattern for Knowledge Base operations: KB/assistant
lifecycle, job state machine, and background ingestion orchestration.

Ingestion pipeline (per file, background task):
    SHA-256 → dedup against the KB's KnowledgeDocument rows → register the
    document → extract (DocumentReader, services stay format-agnostic) →
    ``pending_vision`` stops here (image/scanned PDF, indexed by the OCR/VLM
    tiers of a later release) → 3-pass chunking → hybrid vector store
    (``rag.kb_chunks``). A file that is not a valid document of its type is
    marked ``unreadable`` (the user's file, a WARNING); any other per-file
    failure marks THAT document ``failed`` (an ERROR with the traceback). The
    run continues either way; the job only fails when nothing could be
    ingested (``NothingIndexedException``).

Classes:
    KB_Service: Business logic for KB creation, updates, and job tracking.

Architecture:
    Endpoints → Services → Repository → Database
                     ↓
            src.ingestion (DocumentReader → chunking → vector store)
"""

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sqlalchemy.orm import Session

from src.core.exceptions import (
    USER_FILE_REASONS,
    EmbeddingModelMissingException,
    NothingIndexedException,
    StateConflictException,
    UnreadableDocumentException,
)
from src.domains.knowledge_base.repository import KB_Repository
from src.entities.KBJob import KBJobModel
from src.ingestion.chunking import chunk_document
from src.ingestion.embedder_child import ingestion_lease
from src.ingestion.embeddings import embedding_artifact_missing
from src.ingestion.reader import DocumentReader
from src.ingestion.vector_store import add_kb_chunks
from src.core.logging import logger


@dataclass
class _JobFileLedger:
    """What a job knows about its files that its document rows cannot say.

    Attributes:
        rowless: Names of the job's files that have no document row, by
            outcome: a file whose bytes could not be read or whose row could
            not be registered (``failed``), and a file with the same bytes as
            one earlier in the job, which shares that file's row and outcome
            (the hash is unique per KB).
        outcome_by_hash: The outcome of each content this job ingested.
    """

    rowless: Dict[str, List[str]] = field(default_factory=dict)
    outcome_by_hash: Dict[str, str] = field(default_factory=dict)

    def add_rowless(self, outcome: str, name: str) -> None:
        self.rowless.setdefault(outcome, []).append(name)


# Kept in memory, by job id, for the status route. Only the latest jobs are kept.
_JOB_LEDGERS: Dict[int, _JobFileLedger] = {}
_JOB_LEDGERS_KEPT = 32


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
            Dict with:
              - status, status_updated_at;
              - error_message: the English failure text, for the log;
              - error_code: ``"nothing_indexed"`` when the job failed because
                none of its files added searchable content, else None. The
                interface builds its translated message from it;
              - failed_files: the files of this job whose ingestion failed for
                an internal cause (the reason is in the log and Diagnostics,
                never in the payload);
              - files_by_reason: the files the user's own files kept out, by
                reason (``pending_vision``, ``empty``, ``unreadable``), each
                key always present.

        Raises:
            ValueError: If KB job not found.
        """
        kb_job = self.repo.get_kb_job_by_model_id(db, llm_id)
        if not kb_job:
            raise ValueError(f"KB job not found for LLM ID {llm_id}")

        # Read before any cleanup: a failed creation deletes its KB, and the
        # documents with it.
        failed_files, files_by_reason, any_indexed = self._job_files(db, kb_job)
        nothing_indexed = (
            kb_job.status == "failed"
            and not any_indexed
            and bool(failed_files or any(files_by_reason.values()))
        )

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
            "error_code": "nothing_indexed" if nothing_indexed else None,
            "failed_files": failed_files,
            "files_by_reason": files_by_reason,
        }

    def _job_files(
        self, db: Session, kb_job: KBJobModel
    ) -> Tuple[List[str], Dict[str, List[str]], bool]:
        """The files of this job by outcome (no column: derived).

        Returns ``(failed, by_reason, any_indexed)``: the names left
        ``failed``, the names under each of ``USER_FILE_REASONS``, and whether
        one document of the job was indexed. The job's documents are the ones
        created from its start on: an earlier job's files are not this one's.
        Files without a row of their own (bytes that could not be read, a row
        that could not be registered, the same bytes as a file earlier in the
        job) come from the job's in-memory ledger. A failed job with none of
        these files failed before its first file: ``nothing_indexed`` does not
        apply to it.
        """
        failed: List[str] = []
        by_reason: Dict[str, List[str]] = {reason: [] for reason in USER_FILE_REASONS}
        any_indexed = False
        if kb_job.kb_id is not None and kb_job.created_at is not None:
            for document in self.repo.list_documents_since(db, kb_job.kb_id, kb_job.created_at):
                if document.status == "failed":
                    failed.append(document.name)
                elif document.status in by_reason:
                    by_reason[document.status].append(document.name)
                elif document.status == "active":
                    any_indexed = True
        ledger = _JOB_LEDGERS.get(kb_job.id)
        if ledger is not None:
            failed += ledger.rowless.get("failed", [])
            for reason, names in by_reason.items():
                names += ledger.rowless.get(reason, [])
        return failed, by_reason, any_indexed

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

        # One job at a time per KB: two would both see the same failed row and
        # race on the unique (kb_id, hash) constraint.
        if self.repo.get_unfinished_kb_job(db, kb.id):
            raise StateConflictException(
                "This assistant's knowledge base is still being updated. "
                "Wait for it to finish before adding more documents.",
                status_code=409,
            )

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

            # Refuse UP FRONT when the platform's embedder cannot run without
            # the e5 artifact (#594): one clear job failure with the state's
            # message, instead of parsing every file and failing them one by
            # one at the embed step.
            if embedding_artifact_missing():
                raise EmbeddingModelMissingException()

            logger.info(
                f"KB job {kb_job_id} started "
                f"({'update' if is_update else 'creation'}, kb_id={kb.id}, "
                f"{len(file_paths)} file(s))"
            )

            ledger = _JOB_LEDGERS.setdefault(kb_job_id, _JobFileLedger())
            while len(_JOB_LEDGERS) > _JOB_LEDGERS_KEPT:
                del _JOB_LEDGERS[next(iter(_JOB_LEDGERS))]

            counts = dict.fromkeys(("indexed", "skipped", "failed", *USER_FILE_REASONS), 0)
            # The lease keeps the dedicated embedding child alive for the
            # whole job (#594): its idle reaper never terminates it between
            # two files, however long a file takes to parse.
            with ingestion_lease():
                for raw_path in file_paths:
                    counts[self._ingest_one_file(db, kb.id, raw_path, ledger)] += 1

            duration_ms = (time.perf_counter() - start_s) * 1000
            logger.info(
                f"KB job {kb_job_id} ({'update' if is_update else 'creation'}): "
                f"{counts['indexed']} indexed, {counts['pending_vision']} pending_vision, "
                f"{counts['skipped']} duplicates, {counts['empty']} empty, "
                f"{counts['unreadable']} unreadable, {counts['failed']} failed, "
                f"duration_ms={duration_ms:.0f}"
            )

            # Don't report success when nothing queryable was added. A batch that
            # indexed zero chunks and had no duplicates (its files were empty,
            # pending_vision with no OCR tier, unreadable, or failed) is a failure
            # the user must see — never a silent "completed". All-duplicates
            # (skipped > 0) stays a success: that content is already in the KB.
            if counts["indexed"] == 0 and counts["skipped"] == 0:
                raise NothingIndexedException(counts)

            self.repo.update_kb_job_status(db, kb_job, "completed")

        except Exception as e:
            # Marked failed for the polling UI and re-raised: the background
            # task that owns this job writes its record.
            self._handle_indexing_error(db, kb_job_id, str(e))
            raise

    def _ingest_one_file(
        self, db: Session, kb_id: int, raw_path: str, ledger: _JobFileLedger | None = None
    ) -> str:
        """Ingest one file; returns "indexed" | "pending_vision" | "skipped"
        | "empty" | "unreadable" | "failed". Never raises — a broken file must
        not sink the batch.

        ``ledger`` records the job's files that have no document row of their
        own, so the job can still name them."""
        ledger = ledger if ledger is not None else _JobFileLedger()
        path = Path(raw_path)
        try:
            content = path.read_bytes()
        except OSError as e:
            # The file the user picked could not be read (moved, deleted,
            # denied) or the app handed a wrong path: the two cannot be told
            # apart here, so this is a `failed` file, and every `failed` file
            # has its ERROR with the traceback. The batch goes on (and fails
            # as a whole only when nothing at all could be indexed).
            logger.error(f"KB {kb_id}: cannot read {path.name}: {e}", exc_info=True)
            ledger.add_rowless("failed", path.name)
            return "failed"

        content_hash = hashlib.sha256(content).hexdigest()
        earlier = ledger.outcome_by_hash.get(content_hash)
        if earlier is not None:
            # The same bytes as a file earlier in this job: that file's row
            # (unique per hash) and outcome stand for this one too. Replacing
            # the row would erase the earlier file's name from the job.
            if earlier in ("indexed", "skipped"):
                logger.info(f"KB {kb_id}: {path.name} already ingested (dedup), skipping")
                return "skipped"
            logger.info(
                f"KB {kb_id}: {path.name} has the same content as a file earlier in this job "
                f"({earlier})"
            )
            ledger.add_rowless(earlier, path.name)
            return earlier
        outcome = self._ingest_new_content(db, kb_id, path, content, content_hash, ledger)
        ledger.outcome_by_hash[content_hash] = outcome
        return outcome

    def _ingest_new_content(
        self,
        db: Session,
        kb_id: int,
        path: Path,
        content: bytes,
        content_hash: str,
        ledger: _JobFileLedger,
    ) -> str:
        """The rest of ``_ingest_one_file`` for a content new to this job."""
        try:
            existing = self.repo.get_document_by_hash(db, kb_id, content_hash)
            if existing and existing.status == "active":
                logger.info(f"KB {kb_id}: {path.name} already ingested (dedup), skipping")
                return "skipped"
            if existing:
                # An earlier job left this row without indexing it: it failed
                # (maybe with a few chunks, which follow the row through ON
                # DELETE CASCADE), or the file was empty, a scan, or invalid.
                # It must not hide the same file for good: replace it, so the
                # retry starts clean and the job names the file again if it is
                # still not indexed, instead of passing it for a duplicate.
                logger.info(
                    f"KB {kb_id}: {path.name} was {existing.status} earlier, ingesting it again"
                )
                self.repo.delete_document(db, existing)

            # "indexing" until the chunks are stored: the row is committed first,
            # so a quit in between must not leave an "active" row without chunks.
            document = self.repo.create_document(
                db,
                kb_id=kb_id,
                name=path.name,
                content_hash_sha256=content_hash,
                size_bytes=len(content),
                status="indexing",
            )
            # Commit BEFORE indexing: the vector store writes through its own
            # connection and the chunks' document_id FK must see this row.
            db.commit()
        except Exception as e:
            # Replacing or registering the row failed (a concurrent insert on
            # the unique hash, a database error): this file fails, the batch goes on.
            db.rollback()
            logger.error(f"KB {kb_id}: cannot register {path.name}: {e}", exc_info=True)
            ledger.add_rowless("failed", path.name)
            return "failed"

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
            self.repo.update_document_status(db, document, "active")
            logger.info(f"KB {kb_id}: {path.name} indexed ({len(chunks)} chunks)")
            return "indexed"

        except UnreadableDocumentException as e:
            # The user's file is not what its extension says: the app skipped
            # it and names it at the end of the job. A WARNING without a
            # traceback (nothing in the app's code to look at).
            logger.warning(f"KB {kb_id}: {path.name} skipped: {e.reason}")
            self.repo.update_document_status(db, document, "unreadable")
            return "unreadable"

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
