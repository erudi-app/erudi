"""CSV extractor — stdlib csv with dialect sniffing → one Markdown table."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from src.core.logging import logger
from src.ingestion.cleaning import clean_extracted_text
from src.ingestion.markdown import rows_to_markdown_table
from src.ingestion.types import ExtractedDocument

_SNIFF_BYTES = 8192


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning(
            f"Tabular file {path.name}: not valid UTF-8 - falling back to latin-1 decoding"
        )
        return raw.decode("latin-1")


class CsvExtractor:
    def extract(self, path: Path) -> ExtractedDocument:
        text = _read_text(path)

        try:
            dialect = csv.Sniffer().sniff(text[:_SNIFF_BYTES], delimiters=",;\t")
        except csv.Error:
            logger.warning(
                f"CSV {path.name}: dialect sniffing failed - "
                f"falling back to default comma dialect"
            )
            dialect = csv.excel  # default comma

        rows = [row for row in csv.reader(io.StringIO(text), dialect) if any(row)]
        logger.debug(f"CSV extracted: {path.name} ({len(rows)} non-empty rows)")
        markdown = clean_extracted_text(rows_to_markdown_table(rows))
        return ExtractedDocument(
            markdown=markdown,
            status="active",
            metadata={"extractor": "csv", "row_count": len(rows)},
        )
