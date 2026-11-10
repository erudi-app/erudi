"""SQLAlchemy entity for documents ingested into a Knowledge Base.

One row per source file dropped into a KB. Replaces the FAISS-era
``KnowledgeBase.file_names_list`` JSON blob: documents are first-class rows
with a content hash (dedup), a size, and a per-file ingestion status. The
actual text chunks + embeddings live in the pgvector-managed ``rag.kb_chunks``
table (langchain-postgres), keyed back here through a ``document_id`` metadata
column.

Statuses:
    - ``indexing``: registered, chunks not stored yet. The row is committed
      before indexing (the chunks' foreign key needs it), so this is what a
      quit mid-file leaves behind; startup turns it into ``failed``.
    - ``active``: extracted, chunked, and indexed — retrievable.
    - ``empty``: extracted but produced zero indexable chunks (blank/whitespace
      file or a parser that yielded no text); nothing searchable was added.
    - ``failed``: ingestion raised; error surfaced via the KB job.
    - ``pending_vision``: image or scanned PDF accepted but not yet readable
      (OCR/VLM tiers land in a later release); zero chunks indexed.
"""

from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

from src.database.core import Base

ALLOWED_STATUSES = ("indexing", "active", "empty", "failed", "pending_vision")


class KnowledgeDocument(Base):
    """A single source file tracked inside a Knowledge Base.

    Attributes:
        id: Primary key (auto-increment).
        kb_id: Owning KnowledgeBase (server-side ON DELETE CASCADE).
        name: Original file name (display + extension routing).
        content_hash_sha256: SHA-256 of the file bytes; UNIQUE per KB (dedup).
        size_bytes: File size at ingestion time.
        status: Per-file ingestion state (see module docstring).
        created_at: Server-stamped ingestion timestamp.
    """

    __tablename__ = "knowledge_documents"

    id = Column(Integer, primary_key=True, index=True)
    kb_id = Column(
        Integer,
        ForeignKey("knowledge_base.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name = Column(String, nullable=False)
    content_hash_sha256 = Column(String(64), nullable=False)
    size_bytes = Column(BigInteger, nullable=False)
    status = Column(String, default="active", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("kb_id", "content_hash_sha256", name="uq_knowledge_documents_kb_hash"),
    )

    kb = relationship("KnowledgeBase", back_populates="documents")

    @validates("status")
    def validate_status(self, key, value):
        """Ensure status is one of the allowed ingestion states."""
        if value not in ALLOWED_STATUSES:
            raise ValueError(f"Invalid status: {value}. Must be one of {list(ALLOWED_STATUSES)}")
        return value

    def __repr__(self) -> str:
        return (
            f"<KnowledgeDocument(id={self.id}, kb_id={self.kb_id}, "
            f"name='{self.name}', status='{self.status}')>"
        )
