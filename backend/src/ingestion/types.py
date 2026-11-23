"""Shared datatypes for the document ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


class UnreadableDocumentError(Exception):
    """The file is not a valid document of the type its extension names.

    Raised by an extractor in place of the parser's own exception when that
    exception means the user's file is the cause (a text file renamed
    ``.docx``, a damaged archive, a PDF locked by a password), never for a
    failure of the app's own code. Callers record it as a skipped file, at
    WARNING and without a traceback (docs/logging.md, "Logging rules").

    Attributes:
        reason: What is wrong with the file, ASCII, for the log line, e.g.
            ``not a valid Word file (PackageNotFoundError)``.
    """

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


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
