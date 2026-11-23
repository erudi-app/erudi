"""DOCX extractor — python-docx, structure-preserving.

Heading styles map to Markdown headers (Heading 1 → ``#`` … Heading 4 →
``####``) so the 3-pass chunker can split on real document structure;
tables become Markdown tables. Paragraphs and tables are walked in document
order via ``Document.iter_inner_content``.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

from src.core.logging import logger
from src.ingestion.cleaning import clean_extracted_text
from src.ingestion.markdown import rows_to_markdown_table
from src.core.exceptions import UnreadableDocumentException
from src.ingestion.types import ExtractedDocument

_HEADING_STYLE_RE = re.compile(r"^Heading (\d)$")

# python-docx 1.2's ValueError for a package whose main part is not a Word
# document (docx/api.py: "file '%s' is not a Word file, content type is '%s'").
_NOT_A_WORD_FILE = "is not a Word file, content type is"


def _heading_prefix(style_name: str | None) -> str:
    match = _HEADING_STYLE_RE.match(style_name or "")
    if not match:
        return ""
    level = min(int(match.group(1)), 4)
    return "#" * level + " "


class DocxExtractor:
    def extract(self, path: Path) -> ExtractedDocument:
        # Function scope: python-docx (and lxml) stay out of the boot heap until
        # the first DOCX (#621).
        import docx
        from docx.opc.exceptions import PackageNotFoundError
        from docx.table import Table

        try:
            document = docx.Document(str(path))
        except (PackageNotFoundError, zipfile.BadZipFile, KeyError) as e:
            # The user's file is not a Word package: not a zip at all
            # (PackageNotFoundError, a text renamed .docx), a damaged zip, or
            # a zip without the package's parts (KeyError from the archive
            # lookup, a plain .zip renamed). Only the opening is guarded: a
            # failure past it is the extractor's own and stays a defect.
            # Compromise: a KeyError python-docx itself raised while opening a
            # valid file would be counted as the file's fault too.
            raise UnreadableDocumentException(f"not a valid Word file ({type(e).__name__})") from e
        except ValueError as e:
            # Another Office package renamed .docx (a workbook, a macro-enabled
            # .docm): python-docx opens it and refuses its main part. Matched
            # on python-docx's own wording so no other ValueError is hidden.
            if _NOT_A_WORD_FILE not in str(e):
                raise
            raise UnreadableDocumentException(
                f"not a Word document, another Office file ({type(e).__name__})"
            ) from e
        parts: list[str] = []

        for item in document.iter_inner_content():
            if isinstance(item, Table):
                rows = [[cell.text for cell in row.cells] for row in item.rows]
                table = rows_to_markdown_table(rows)
                if table:
                    parts.append(table)
                continue

            text = item.text.strip()
            if not text:
                continue
            parts.append(_heading_prefix(item.style.name) + text)

        markdown = clean_extracted_text("\n\n".join(parts))
        logger.debug(
            f"DOCX extracted: {path.name} ({len(parts)} blocks, " f"{len(markdown)} chars)"
        )
        return ExtractedDocument(
            markdown=markdown,
            status="active",
            metadata={"extractor": "docx"},
        )
