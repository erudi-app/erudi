"""XLSX extractor — openpyxl, every sheet, computed values.

Each worksheet becomes a ``## <sheet name>`` section holding one Markdown
table (first non-empty row = header). ``data_only=True`` reads cached
formula RESULTS, not formula strings.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from src.core.logging import logger
from src.ingestion.cleaning import clean_extracted_text
from src.ingestion.markdown import rows_to_markdown_table
from src.core.exceptions import UnreadableDocumentException
from src.ingestion.types import ExtractedDocument

# openpyxl 3.1's OSError for a zip package without a workbook part
# (openpyxl/reader/excel.py, ``_find_workbook_part``).
_NO_WORKBOOK_PART = "File contains no valid workbook part"


class XlsxExtractor:
    def extract(self, path: Path) -> ExtractedDocument:
        # Function scope: openpyxl stays out of the boot heap until the first XLSX (#621).
        import openpyxl
        from openpyxl.utils.exceptions import InvalidFileException

        try:
            workbook = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        except (InvalidFileException, zipfile.BadZipFile, KeyError) as e:
            # The user's file is not an Excel workbook: not a zip (BadZipFile,
            # a text renamed .xlsx), a zip without the workbook's parts
            # (KeyError from the archive lookup), or a format openpyxl refuses.
            # Only the opening is guarded, as in the DOCX extractor, with the
            # same compromise on a KeyError openpyxl itself would raise there.
            raise UnreadableDocumentException(
                f"not a valid Excel workbook ({type(e).__name__})"
            ) from e
        except OSError as e:
            # Another Office package renamed .xlsx (a Word document): openpyxl
            # finds no workbook part. Matched on openpyxl's own wording, so a
            # real I/O error (a file that vanished, a denied read) is not hidden.
            if _NO_WORKBOOK_PART not in str(e):
                raise
            raise UnreadableDocumentException(
                f"not an Excel workbook, another Office file ({type(e).__name__})"
            ) from e
        try:
            sections: list[str] = []
            for sheet in workbook.worksheets:
                rows = [
                    ["" if cell is None else str(cell) for cell in row]
                    for row in sheet.iter_rows(values_only=True)
                ]
                rows = [row for row in rows if any(cell.strip() for cell in row)]
                if not rows:
                    continue
                sections.append(f"## {sheet.title}\n\n{rows_to_markdown_table(rows)}")
        finally:
            workbook.close()

        markdown = clean_extracted_text("\n\n".join(sections))
        logger.debug(
            f"XLSX extracted: {path.name} ({len(sections)} non-empty sheets, "
            f"{len(markdown)} chars)"
        )
        return ExtractedDocument(
            markdown=markdown,
            status="active",
            metadata={"extractor": "xlsx", "sheet_count": len(sections)},
        )
