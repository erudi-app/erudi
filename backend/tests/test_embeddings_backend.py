"""The KB embeddings backend, ``ChildE5Embeddings`` (#594).

- ``get_embeddings()`` serves ``ChildE5Embeddings`` on every platform.
- ``ChildE5Embeddings`` applies the asymmetric e5 prefixes, bounds every
  input at 512 tokens Erudi-side (prefix included, through the frozen
  tokenizer), and never instantiates anything at construction.
- ``build_embedding_text`` writes the frozen [N6] document-name prefix.
"""

from __future__ import annotations

import pytest

from src.ingestion import embedder_child, embeddings
from src.ingestion.chunking import E5_MAX_CONTENT_TOKENS, count_tokens

pytestmark = pytest.mark.unit


class TestGetEmbeddings:
    def test_the_child_backend_on_every_platform(self):
        assert isinstance(embeddings.get_embeddings(), embeddings.ChildE5Embeddings)


class TestChildE5Embeddings:
    @pytest.fixture
    def captured(self, monkeypatch):
        calls = []

        def fake_embed_texts(texts):
            calls.append(list(texts))
            return [[0.1, 0.2] for _ in texts]

        monkeypatch.setattr(embedder_child, "embed_texts", fake_embed_texts)
        return calls

    def test_construction_has_no_side_effect(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("constructing the backend must not touch the child")

        monkeypatch.setattr(embedder_child, "ensure_embedder_running", boom)
        monkeypatch.setattr(embedder_child, "embed_texts", boom)
        embeddings.ChildE5Embeddings()

    def test_documents_carry_the_passage_prefix(self, captured):
        vectors = embeddings.ChildE5Embeddings().embed_documents(["alpha", "beta"])
        assert captured == [["passage: alpha", "passage: beta"]]
        assert vectors == [[0.1, 0.2], [0.1, 0.2]]

    def test_query_carries_the_query_prefix(self, captured):
        vector = embeddings.ChildE5Embeddings().embed_query("question")
        assert captured == [["query: question"]]
        assert vector == [0.1, 0.2]

    def test_long_inputs_are_bounded_at_512_prefix_included(self, captured):
        long_text = "mot " * 3000
        embeddings.ChildE5Embeddings().embed_documents([long_text])
        sent = captured[0][0]
        assert sent.startswith("passage: ")
        assert count_tokens(sent) <= E5_MAX_CONTENT_TOKENS
        embeddings.ChildE5Embeddings().embed_query(long_text)
        sent = captured[1][0]
        assert sent.startswith("query: ")
        assert count_tokens(sent) <= E5_MAX_CONTENT_TOKENS

    def test_short_inputs_pass_through_untouched(self, captured):
        embeddings.ChildE5Embeddings().embed_documents(["bonjour le monde"])
        assert captured[0][0] == "passage: bonjour le monde"


class TestBuildEmbeddingText:
    def test_document_name_prefix_format(self):
        # [N6] frozen order: "passage: " (added by embed_documents) →
        # "[document_name:…]" → breadcrumb + chunk text.
        assert (
            embeddings.build_embedding_text(file_name="rapport.pdf", chunk_text="# A\n\nContenu.")
            == "[document_name:rapport.pdf]\n# A\n\nContenu."
        )
