"""P5 — 3-pass chunking, token-accurate against the REAL e5 tokenizer.

Pass 1 splits on Markdown headers (h1–h4) to follow document structure;
pass 2 sub-splits big sections with a token-accurate recursive splitter
(the FAISS-era chunker estimated ~3 chars/token against a model that
truncated at 128 — two thirds of every chunk was invisible to retrieval);
pass 3 prefixes each chunk with its heading breadcrumb ("# A > ## B") and
re-attaches Markdown table headers lost mid-split.
"""

import pytest

from src.ingestion.chunking import (
    DEFAULT_OVERLAP_TOKENS,
    DEFAULT_TARGET_TOKENS,
    chunk_document,
    chunk_markdown,
    count_tokens,
)
from src.ingestion.types import ExtractedDocument, ExtractedPage

pytestmark = pytest.mark.unit


def _long_paragraphs(n: int, stem: str = "Phrase utile numéro") -> str:
    return "\n\n".join(f"{stem} {i}, qui décrit un fait précis sur le produit." for i in range(n))


class TestTokenCounting:
    def test_count_tokens_uses_real_tokenizer(self):
        # Real subword counts, not a chars/3 estimate: an accented French
        # sentence must count to a plausible subword total.
        n = count_tokens("Le café des développeurs était fermé.")
        assert 5 <= n <= 20


class TestChunkMarkdown:
    def test_empty_input(self):
        assert chunk_markdown("") == []
        assert chunk_markdown("   \n  ") == []

    def test_small_text_is_one_chunk(self):
        chunks = chunk_markdown("Un paragraphe court et simple.")
        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0
        assert chunks[0].text == "Un paragraphe court et simple."

    def test_header_sections_carry_breadcrumbs(self):
        md = (
            "# Guide\n\nIntroduction générale du document.\n\n"
            "## Installation\n\nÉtapes détaillées pour installer le produit."
        )
        chunks = chunk_markdown(md)
        installation = [c for c in chunks if "Étapes détaillées" in c.text]
        assert installation
        assert "# Guide > ## Installation" in installation[0].text

    def test_long_section_is_split_within_token_budget(self):
        chunks = chunk_markdown(_long_paragraphs(60), target_tokens=64, overlap_tokens=8)
        assert len(chunks) > 2
        # No breadcrumb here (no headers): the splitter's own budget rules.
        assert all(c.token_count <= 64 + 8 for c in chunks)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_overlap_produces_more_chunks_than_no_overlap(self):
        text = _long_paragraphs(60)
        with_overlap = chunk_markdown(text, target_tokens=64, overlap_tokens=16)
        without = chunk_markdown(text, target_tokens=64, overlap_tokens=0)
        assert len(with_overlap) >= len(without)

    def test_table_header_reattached_to_continuation_chunks(self):
        rows = "\n".join(
            f"| Produit numéro {i} avec un libellé assez long | {i * 100} € |" for i in range(60)
        )
        md = f"# Catalogue\n\n| Article | Prix |\n| --- | --- |\n{rows}"
        chunks = chunk_markdown(md, target_tokens=96, overlap_tokens=0)
        table_chunks = [c for c in chunks if "| Produit numéro" in c.text]
        assert len(table_chunks) > 1
        for chunk in table_chunks:
            assert "| Article | Prix |" in chunk.text

    def test_nul_bytes_stripped(self):
        chunks = chunk_markdown("Texte\x00avec un NUL.")
        assert chunks and "\x00" not in chunks[0].text

    def test_default_budget_fits_e5_window(self):
        # [N6] target + breadcrumb + "[document_name:…]" + "passage: " must
        # stay well under e5's 512-token window.
        assert DEFAULT_TARGET_TOKENS + DEFAULT_OVERLAP_TOKENS < 512 // 2


class TestChunkDocument:
    def test_pending_vision_yields_no_chunks(self):
        doc = ExtractedDocument(markdown="", status="pending_vision")
        assert chunk_document(doc) == []

    def test_plain_document_chunks_have_no_page(self):
        doc = ExtractedDocument(markdown=_long_paragraphs(8), status="active")
        chunks = chunk_document(doc)
        assert chunks
        assert all(c.page_number is None for c in chunks)

    def test_paginated_document_carries_page_numbers(self):
        doc = ExtractedDocument(
            markdown="ignored when pages exist",
            status="active",
            pages=[
                ExtractedPage(page_number=1, text=_long_paragraphs(30)),
                ExtractedPage(page_number=2, text="Une page courte."),
            ],
        )
        chunks = chunk_document(doc, target_tokens=64, overlap_tokens=8)
        assert {c.page_number for c in chunks} == {1, 2}
        # chunk_index is global and contiguous across pages.
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        page2 = [c for c in chunks if c.page_number == 2]
        assert len(page2) == 1 and "Une page courte." in page2[0].text


class TestTruncateToTokens:
    """Erudi-side 512-token bound for the embedding child (#594): inputs are
    truncated with the frozen tokenizer at 510 content tokens (512 minus
    ``<s>``/``</s>``), so every input fits e5's window. The bound, not boundary identity, is the contract: the cut
    prefix can re-tokenize its last token differently (parity stays within
    cosine 0.9999, pinned by the mlx_only test)."""

    def test_short_text_is_returned_unchanged(self):
        from src.ingestion.chunking import truncate_to_tokens

        text = "passage: un texte court."
        assert truncate_to_tokens(text) == text

    def test_long_text_is_bounded_and_a_prefix_of_the_original(self):
        from src.ingestion.chunking import E5_MAX_CONTENT_TOKENS, truncate_to_tokens

        text = "query: " + _long_paragraphs(200)
        truncated = truncate_to_tokens(text)
        assert text.startswith(truncated)
        kept = count_tokens(truncated)
        # Bounded, and cut at the limit rather than far below it.
        assert kept <= E5_MAX_CONTENT_TOKENS
        assert kept >= E5_MAX_CONTENT_TOKENS - 1

    def test_default_bound_is_e5_window_minus_specials(self):
        from src.ingestion.chunking import E5_MAX_CONTENT_TOKENS

        assert E5_MAX_CONTENT_TOKENS == 510

    def test_custom_bound_is_honored(self):
        from src.ingestion.chunking import truncate_to_tokens

        text = "Un texte avec suffisamment de mots pour être coupé proprement."
        truncated = truncate_to_tokens(text, max_tokens=5)
        assert count_tokens(truncated) <= 5
        assert text.startswith(truncated)
        assert truncated

    def test_exact_bound_is_untouched(self):
        from src.ingestion.chunking import truncate_to_tokens

        text = "Bonjour le monde"
        exact = count_tokens(text)
        assert truncate_to_tokens(text, max_tokens=exact) == text
