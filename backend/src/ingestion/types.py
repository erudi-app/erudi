"""Shared datatypes for the document ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ExtractedPage:
    """One physical page of a paginated document (PDF)."""

    page_number: int  # 1-based
    text: str


@dataclass(slots=True)
class ExtractedDocument:
    """What `DocumentReader.read` returns — the only shape services see.

    Attributes:
        markdown: Full cleaned Markdown content ("" when pending_vision).
        status: "active" (readable now) or "pending_vision" (image / scanned
            PDF accepted but awaiting the OCR/VLM tiers of a later release).
        pages: Per-page texts for paginated formats (PDF), else None. Lets
            the chunker carry page numbers into chunk metadata.
        metadata: Extractor name, page counts, pending reasons, …
    """

    markdown: str
    status: str  # "active" | "pending_vision"
    pages: list[ExtractedPage] | None = None
    metadata: dict = field(default_factory=dict)
