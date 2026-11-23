"""PDF extractor — deterministic fast-path over the native text layer.

Difficulty routing happens HERE (second dimension of the reader's 2-D
routing): a PDF with a usable text layer is extracted page by page with
pypdf; a scanned PDF (no/negligible text layer) is accepted as
``pending_vision`` — zero chunks now, indexed by the OCR/VLM tiers of a
later release.
"""

from __future__ import annotations

from pathlib import Path

from src.core.logging import logger
from src.ingestion.cleaning import clean_extracted_text
from src.ingestion.types import ExtractedDocument, ExtractedPage, UnreadableDocumentError


class PdfExtractor:
    # Below this average per page the "text layer" is noise (page numbers,
    # stray watermark glyphs), not content — treat as scanned.
    MIN_AVG_CHARS_PER_PAGE = 25

    def extract(self, path: Path) -> ExtractedDocument:
        # Function scope: pypdf stays out of the boot heap until the first PDF (#621).
        from pypdf import PdfReader
        from pypdf.errors import FileNotDecryptedError, PdfReadError

        try:
            reader = PdfReader(str(path))
            raw_pages = [page.extract_text() or "" for page in reader.pages]
        except FileNotDecryptedError as e:
            # Locked by a password the user did not give: nothing can be read.
            raise UnreadableDocumentError(
                f"PDF protected by a password ({type(e).__name__})"
            ) from e
        except PdfReadError as e:
            # pypdf's verdict on the file itself (PdfStreamError on a text
            # renamed .pdf, EmptyFileError, a damaged cross-reference table):
            # the user's file, not a defect. DependencyError (a package the
            # build lacks, e.g. for AES) is not a PdfReadError: it stays a
            # defect.
            raise UnreadableDocumentError(f"not a valid PDF ({type(e).__name__})") from e
        pages = [
            ExtractedPage(page_number=number, text=clean_extracted_text(text))
            for number, text in enumerate(raw_pages, start=1)
        ]

        total_chars = sum(len(p.text) for p in pages)
        logger.debug(f"PDF extracted: {path.name} ({len(pages)} pages, {total_chars} chars)")
        if not pages or total_chars / len(pages) < self.MIN_AVG_CHARS_PER_PAGE:
            logger.warning(
                f"PDF {path.name}: no usable text layer "
                f"({len(pages)} pages, {total_chars} chars) - "
                f"routed to pending_vision (scanned?)"
            )
            return ExtractedDocument(
                markdown="",
                status="pending_vision",
                metadata={
                    "extractor": "pdf",
                    "page_count": len(pages),
                    "reason": "no usable text layer (scanned?)",
                },
            )

        markdown = "\n\n".join(p.text for p in pages if p.text)
        return ExtractedDocument(
            markdown=markdown,
            status="active",
            pages=pages,
            metadata={"extractor": "pdf", "page_count": len(pages)},
        )
