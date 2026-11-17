"""3-pass Markdown chunking, token-accurate against the real e5 tokenizer.

The tokenizer is the e5 one frozen in ``e5_tokenizer.json.gz`` (see
``E5_TOKENIZER_FILE``), so chunking works offline and before the embedding
model is downloaded.

Ported from the R&D ingestion POC and re-targeted at multilingual-e5-small:

1. Split on Markdown headers (h1–h4) — chunks follow document structure.
2. Sub-split large sections with a token-accurate recursive splitter
   (separators favor paragraph > line > sentence > word boundaries). The
   FAISS-era chunker estimated ~3 chars/token against a model truncating at
   128 tokens — two thirds of every chunk was invisible to dense retrieval.
3. Prefix each chunk with its heading breadcrumb ("# A > ## B") and
   re-attach Markdown table headers lost mid-split.

Budget [N6]: the full embedded text is
``passage: [document_name:NAME]\\n<breadcrumb>\\n\\n<chunk>``; with the
default target (~180 tokens) plus breadcrumb and name the total stays far
inside e5's 512-token window. The R&D sweeps found 150–200-token chunks the
most consistent — big chunks measurably HURT dense retrieval.
"""

from __future__ import annotations

import gzip
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.core import config
from src.core.logging import logger
from src.ingestion.types import ExtractedDocument

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from tokenizers import Tokenizer

# NB: langchain_text_splitters is imported in FUNCTION scope only (issue #160).
# Its package __init__ eagerly imports sentence_transformers -> torch/sklearn
# (~3-4 s), and this module sits on the boot path (seed -> utils -> kb_utils),
# so a module-level import here taxes every backend start for a splitter that
# is only needed at KB ingestion time.

# The intfloat/multilingual-e5-small tokenizer (snapshot 614241f6), frozen once
# as transformers 5.14.1 builds it: ``AutoTokenizer.from_pretrained(...)
# .backend_tokenizer.to_str()``, gzipped (7.9 MB of JSON, 3.5 MB on disk). That
# build, not the snapshot's raw tokenizer.json, is the one chunking always used:
# the raw file normalizes differently and splits 7 of 43 measured documents
# into other chunks. Loaded with the pure ``tokenizers`` package, so counting
# needs neither transformers nor a download; tests/test_e5_tokenizer.py holds
# it to transformers' ids and chunks.
E5_TOKENIZER_FILE = config.ROOT_DIR / "src" / "ingestion" / "e5_tokenizer.json.gz"

DEFAULT_TARGET_TOKENS = 180
DEFAULT_OVERLAP_TOKENS = 27  # ~15 %

HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]

_tokenizer: Tokenizer | None = None
_tokenizer_lock = threading.Lock()


def _load_frozen_tokenizer() -> Tokenizer:
    from tokenizers import Tokenizer

    return Tokenizer.from_buffer(gzip.decompress(E5_TOKENIZER_FILE.read_bytes()))


def _get_tokenizer() -> Tokenizer:
    """Lazy resident e5 tokenizer, read from the frozen file (about 0.5 s and
    250 MB, kept for the life of the process)."""
    global _tokenizer
    tokenizer = _tokenizer
    if tokenizer is None:
        with _tokenizer_lock:
            if _tokenizer is None:
                _tokenizer = _load_frozen_tokenizer()
            tokenizer = _tokenizer
    return tokenizer


def count_tokens(text: str) -> int:
    """Real e5 subword count (no special tokens) — never a chars/N estimate."""
    return len(_get_tokenizer().encode(text, add_special_tokens=False).ids)


@dataclass(slots=True)
class Chunk:
    """One retrieval unit, ready for ``build_embedding_text`` + embedding."""

    chunk_index: int
    text: str
    token_count: int
    page_number: int | None = None


def _heading_breadcrumb(metadata: dict[str, str]) -> str:
    parts: list[str] = []
    for key in ("h1", "h2", "h3", "h4"):
        if key in metadata:
            level = int(key[1])
            parts.append(f"{'#' * level} {metadata[key]}")
    return " > ".join(parts)


def _reattach_table_headers(chunks: list[str]) -> list[str]:
    """Re-inject the Markdown table header into sub-chunks that lost it."""
    if not chunks:
        return chunks

    table_header: str | None = None
    result: list[str] = []

    for chunk in chunks:
        lines = chunk.split("\n")

        # Scan for a table header (header row + separator row) in this chunk.
        found_header = False
        for i in range(len(lines) - 1):
            line = lines[i].strip()
            next_line = lines[i + 1].strip()
            if (
                line.startswith("|")
                and line.endswith("|")
                and next_line.startswith("|")
                and "---" in next_line
            ):
                table_header = lines[i] + "\n" + lines[i + 1]
                found_header = True
                break

        if found_header:
            result.append(chunk)
            continue

        first_content = next((ln.strip() for ln in lines if ln.strip()), "")

        if first_content.startswith("|") and table_header is not None:
            # Headerless continuation of the running table.
            prefix_lines: list[str] = []
            rest_lines: list[str] = []
            in_prefix = True
            for line in lines:
                if in_prefix and not line.strip().startswith("|"):
                    prefix_lines.append(line)
                else:
                    in_prefix = False
                    rest_lines.append(line)
            if prefix_lines:
                result.append(
                    "\n".join(prefix_lines) + "\n" + table_header + "\n" + "\n".join(rest_lines)
                )
            else:
                result.append(table_header + "\n" + chunk)
        else:
            if not first_content.startswith("|"):
                table_header = None
            result.append(chunk)

    return result


def _make_sub_splitter(target_tokens: int, overlap_tokens: int) -> RecursiveCharacterTextSplitter:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Measured with count_tokens: from_huggingface_tokenizer only takes a
    # transformers tokenizer, and counted len(tokenize(text)), the same number.
    return RecursiveCharacterTextSplitter(
        chunk_size=target_tokens,
        chunk_overlap=overlap_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=count_tokens,
    )


def chunk_markdown(
    text: str,
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    page_number: int | None = None,
    start_index: int = 0,
) -> list[Chunk]:
    """Run the 3-pass chunker over one Markdown text. See module docstring."""
    if not text or not text.strip():
        return []

    # PostgreSQL text columns reject NUL bytes; some parsers emit them.
    text = text.replace("\x00", "")

    from langchain_text_splitters import MarkdownHeaderTextSplitter

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    sub_splitter = _make_sub_splitter(target_tokens, overlap_tokens)

    chunks: list[Chunk] = []
    for section in md_splitter.split_text(text):
        breadcrumb = _heading_breadcrumb(section.metadata)

        sub_texts = sub_splitter.split_text(section.page_content)
        sub_texts = _reattach_table_headers(sub_texts)

        for sub_text in sub_texts:
            chunk_text = f"{breadcrumb}\n\n{sub_text}" if breadcrumb else sub_text

            chunks.append(
                Chunk(
                    chunk_index=start_index + len(chunks),
                    text=chunk_text,
                    token_count=count_tokens(chunk_text),
                    page_number=page_number,
                )
            )

    return chunks


def chunk_document(
    document: ExtractedDocument,
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk an ``ExtractedDocument`` — page by page when paginated (PDF),
    so each chunk carries its ``page_number``; whole-Markdown otherwise.

    ``pending_vision`` documents yield no chunks (indexed by the OCR/VLM
    tiers of a later release).
    """
    if document.status != "active":
        return []

    if document.pages:
        chunks: list[Chunk] = []
        for page in document.pages:
            chunks.extend(
                chunk_markdown(
                    page.text,
                    target_tokens=target_tokens,
                    overlap_tokens=overlap_tokens,
                    page_number=page.page_number,
                    start_index=len(chunks),
                )
            )
    else:
        chunks = chunk_markdown(
            document.markdown,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
        )

    if chunks:
        avg_tokens = sum(chunk.token_count for chunk in chunks) / len(chunks)
        logger.info(
            f"Document chunked: {len(chunks)} chunks, "
            f"avg {avg_tokens:.0f} tokens/chunk "
            f"(extractor={document.metadata.get('extractor', '?')})"
        )
    else:
        logger.debug(
            f"Document chunked: 0 chunks " f"(extractor={document.metadata.get('extractor', '?')})"
        )
    return chunks
