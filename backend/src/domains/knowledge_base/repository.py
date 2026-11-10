"""Data access layer for Knowledge Base domain.

Implements repository pattern for database operations related to Knowledge
Bases, LLMs, and KBJobs. All SQL queries isolated here for testability and
separation of concerns.

Classes:
    KB_Repository: Database operations for Knowledge Base entities.

Architecture:
    Endpoints → Services → Repository → Database

    This layer handles:
    - CRUD operations on KnowledgeBase, Llm, KBJob
    - Complex queries (joins, filters)
    - Transaction management
    - Database session handling
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.entities.KnowledgeBase import KnowledgeBase
from src.entities.KnowledgeDocument import KnowledgeDocument
from src.entities.Llm import Llm
from src.entities.KBJob import KBJobModel
from src.core.logging import logger


# Descriptive Llm columns a KB assistant must inherit from its base model so its
# landing card renders identically (family/type, capability category, size,
# quantization, tool-calling, and the raw model_metadata / link the card and the
# derived supports_vision read from). Kept as one explicit allow-list, copied in
# a loop, INSTEAD of a hand-written kwarg list: the previous per-field copy
# rotted every time a descriptive column was added (category and model_metadata
# were both forgotten -> generic/empty cards and wrong supports_vision nulls,
# #209; generation_hints was forgotten next, so assistants lost the publisher's
# sampling recommendation and chatted at the neutral defaults). When you add a
# new *descriptive* column to Llm, add it HERE. The same list drives the
# rebind path (POST /llms/{id}/rebind), which re-copies it from the new base.
#
# Deliberately excluded because they are identity/state, not description, and
# must stay assistant-specific: id (fresh pk), name/description (the assistant's
# own), local (forced to 1 -- the assistant is ready to use), is_base (a catalog
# flag; an assistant is a derived row, never a curated base), is_attached_to_kb
# and kb_id (the KB wiring set below).
COPIED_FIELDS = (
    "link",
    "type",
    "category",
    "param_size",
    "quantized",
    "supports_tools",
    "supports_tools_wire",
    "model_metadata",
    "artifact_size_bytes",
    # The publisher's captured sampling facts (#388): resolved at read time into
    # sampling_defaults, so an assistant must carry its base's to chat (and
    # advertise) the same recommendation.
    "generation_hints",
)


class KB_Repository:
    """Repository for Knowledge Base domain database operations."""

    # ============ KnowledgeBase Operations ============

    def create_knowledge_base(self, db: Session) -> KnowledgeBase:
        """Create new KnowledgeBase entity.

        Source files become KnowledgeDocument rows during ingestion; the KB
        itself is just the registry row.

        Args:
            db: Database session.

        Returns:
            Created KnowledgeBase instance with ID assigned.
        """
        kb = KnowledgeBase()
        db.add(kb)
        db.flush()
        logger.info(f"Created KnowledgeBase with ID: {kb.id}")
        return kb

    def get_knowledge_base_by_id(self, db: Session, kb_id: int) -> Optional[KnowledgeBase]:
        """Fetch KnowledgeBase by ID.

        Args:
            db: Database session.
            kb_id: KnowledgeBase primary key.

        Returns:
            KnowledgeBase instance or None if not found.
        """
        return db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()

    def delete_knowledge_base(self, db: Session, kb: KnowledgeBase) -> None:
        """Delete KnowledgeBase (cascades to KnowledgeDocument rows and Llm).

        Args:
            db: Database session.
            kb: KnowledgeBase instance to delete.
        """
        db.delete(kb)
        db.commit()
        logger.info(f"Deleted KnowledgeBase ID: {kb.id}")

    # ============ KnowledgeDocument Operations ============

    def get_document_by_hash(
        self,
        db: Session,
        kb_id: int,
        content_hash_sha256: str,
    ) -> Optional[KnowledgeDocument]:
        """Fetch a document by its content hash inside one KB (dedup check)."""
        return (
            db.query(KnowledgeDocument)
            .filter(
                KnowledgeDocument.kb_id == kb_id,
                KnowledgeDocument.content_hash_sha256 == content_hash_sha256,
            )
            .first()
        )

    def delete_document(self, db: Session, document: KnowledgeDocument) -> None:
        """Delete one document row; its chunks follow through ON DELETE CASCADE."""
        db.delete(document)
        db.commit()
        logger.info(f"Deleted KnowledgeDocument {document.id} ({document.name})")

    def list_failed_documents_since(
        self, db: Session, kb_id: int, since: datetime
    ) -> list[KnowledgeDocument]:
        """Documents of one KB whose ingestion failed, created at or after ``since``.

        A KBJob has no column for its failures: the documents it created are
        the ones stamped from the job's start on, so ``since`` is the job's
        ``created_at``. Oldest first, the order the files were ingested.
        """
        return (
            db.query(KnowledgeDocument)
            .filter(
                KnowledgeDocument.kb_id == kb_id,
                KnowledgeDocument.status == "failed",
                KnowledgeDocument.created_at >= since,
            )
            .order_by(KnowledgeDocument.id)
            .all()
        )

    def create_document(
        self,
        db: Session,
        *,
        kb_id: int,
        name: str,
        content_hash_sha256: str,
        size_bytes: int,
        status: str = "active",
    ) -> KnowledgeDocument:
        """Create a KnowledgeDocument row (status defaults to "active").

        Flushes so the id is assigned; the CALLER commits — the chunk store
        writes through its own connection and must see the row (FK).
        """
        document = KnowledgeDocument(
            kb_id=kb_id,
            name=name,
            content_hash_sha256=content_hash_sha256,
            size_bytes=size_bytes,
            status=status,
        )
        db.add(document)
        db.flush()
        logger.info(f"Created KnowledgeDocument {document.id} ({name}) for KB {kb_id}")
        return document

    def update_document_status(
        self,
        db: Session,
        document: KnowledgeDocument,
        status: str,
    ) -> KnowledgeDocument:
        """Update a document's ingestion status (active/failed/pending_vision)."""
        document.status = status
        db.commit()
        db.refresh(document)
        return document

    # ============ Llm Operations ============

    def get_llm_by_id(self, db: Session, llm_id: int) -> Optional[Llm]:
        """Fetch Llm by ID.

        Args:
            db: Database session.
            llm_id: Llm primary key.

        Returns:
            Llm instance or None if not found.
        """
        return db.query(Llm).filter(Llm.id == llm_id).first()

    def get_local_llm_by_id(self, db: Session, llm_id: int) -> Optional[Llm]:
        """Fetch local Llm by ID (local=1 filter).

        Args:
            db: Database session.
            llm_id: Llm primary key.

        Returns:
            Llm instance or None if not found or not local.
        """
        return db.query(Llm).filter(Llm.id == llm_id, Llm.local == 1).first()

    def get_local_llm_by_name(self, db: Session, name: str) -> Optional[Llm]:
        """Fetch a local Llm whose name matches case-insensitively (trimmed).

        Used by the KB create guard (#317): a new assistant must not reuse the
        name of any installed model, or the two become indistinguishable in
        every picker (chat, arena, KB library, installed list).

        Args:
            db: Database session.
            name: Candidate assistant name (leading/trailing spaces ignored).

        Returns:
            Llm instance or None if no local model carries that name.
        """
        normalized = (name or "").strip().lower()
        if not normalized:
            return None
        return (
            db.query(Llm)
            .filter(
                Llm.local == 1,
                func.lower(func.trim(Llm.name)) == normalized,
            )
            .first()
        )

    def create_specialized_llm(
        self, db: Session, name: str, description: str, base_llm: Llm, kb_id: int
    ) -> Llm:
        """Create specialized LLM attached to Knowledge Base.

        Inherits every descriptive column of the base model (COPIED_FIELDS) so
        the assistant's landing card matches the base, then overrides the
        assistant-specific identity/state (name, description, local=1, KB wiring).
        Notably supports_tools is inherited so plan_turn routes the assistant to
        the agentic KB path just like its base (#84).

        Args:
            db: Database session.
            name: Name for specialized LLM.
            description: Description for specialized LLM.
            base_llm: Base LLM to copy descriptive attributes from.
            kb_id: KnowledgeBase foreign key.

        Returns:
            Created Llm instance with ID assigned.
        """
        inherited = {field: getattr(base_llm, field) for field in COPIED_FIELDS}
        specialized_llm = Llm(
            name=name,
            description=description,
            local=1,
            is_attached_to_kb=1,
            kb_id=kb_id,
            **inherited,
        )
        db.add(specialized_llm)
        db.flush()
        logger.info(f"Created specialized Llm with ID: {specialized_llm.id}")
        return specialized_llm

    def delete_llm(self, db: Session, llm: Llm) -> None:
        """Delete Llm.

        Args:
            db: Database session.
            llm: Llm instance to delete.
        """
        db.delete(llm)
        db.commit()
        logger.info(f"Deleted Llm ID: {llm.id}")

    # ============ KBJob Operations ============

    def create_kb_job(
        self,
        db: Session,
        base_model_id: int,
        new_model_id: int,
        kb_id: int,
        status: str = "pending",
    ) -> KBJobModel:
        """Create KBJob to track background task.

        Args:
            db: Database session.
            base_model_id: Base LLM ID.
            new_model_id: Specialized LLM ID (or same as base for updates).
            kb_id: KnowledgeBase ID.
            status: Initial status (default: "pending").

        Returns:
            Created KBJobModel instance with ID assigned.
        """
        kb_job = KBJobModel(
            base_model_id=base_model_id, new_model_id=new_model_id, kb_id=kb_id, status=status
        )
        db.add(kb_job)
        db.flush()
        logger.info(f"Created KBJob with ID: {kb_job.id}")
        return kb_job

    def get_kb_job_by_id(self, db: Session, job_id: int) -> Optional[KBJobModel]:
        """Fetch KBJob by ID.

        Args:
            db: Database session.
            job_id: KBJob primary key.

        Returns:
            KBJobModel instance or None if not found.
        """
        return db.query(KBJobModel).filter(KBJobModel.id == job_id).first()

    def get_kb_job_by_model_id(self, db: Session, model_id: int) -> Optional[KBJobModel]:
        """Fetch the MOST RECENT KBJob by new_model_id (specialized LLM).

        An assistant accumulates one KBJob per creation/update, all sharing
        the same new_model_id. Without ordering, `.first()` returns whichever
        row the database happens to hand back first -- typically the OLDEST
        (the original creation) -- so every later update's real outcome,
        failure included, was permanently masked behind the first job's
        status. Reproduced live: an update job that failed with "no
        searchable content" still reported the original creation's
        "completed" status, surfacing a false success to the user.

        Args:
            db: Database session.
            model_id: Specialized LLM ID (new_model_id).

        Returns:
            KBJobModel instance or None if not found.
        """
        return (
            db.query(KBJobModel)
            .filter(KBJobModel.new_model_id == model_id)
            .order_by(KBJobModel.id.desc())
            .first()
        )

    def get_unfinished_kb_job(self, db: Session, kb_id: int) -> Optional[KBJobModel]:
        """The pending or running job of one KB, if any (one at a time per KB)."""
        return (
            db.query(KBJobModel)
            .filter(KBJobModel.kb_id == kb_id, KBJobModel.status.in_(["pending", "running"]))
            .first()
        )

    def update_kb_job_status(
        self, db: Session, kb_job: KBJobModel, status: str, error_message: Optional[str] = None
    ) -> KBJobModel:
        """Update KBJob status and updated_at timestamp.

        Args:
            db: Database session.
            kb_job: KBJobModel instance to update.
            status: New status (pending/running/completed/failed).
            error_message: Optional error message for failed status.

        Returns:
            Updated KBJobModel instance.
        """
        # updated_at is stamped by onupdate=func.now().
        kb_job.status = status
        if error_message:
            kb_job.error_message = error_message
        db.commit()
        db.refresh(kb_job)
        logger.info(f"Updated KBJob {kb_job.id} to status: {status}")
        return kb_job
