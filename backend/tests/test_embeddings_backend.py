"""Platform selection of the embeddings backend, and the child-backed
``ChildE5Embeddings`` (#594).

- ``get_embeddings()`` serves ``ChildE5Embeddings`` where the dedicated
  embedding child exists (macOS ARM), the in-process ``E5Embeddings``
  elsewhere (Windows/Linux until F3), and honors the test seam the suite
  uses to stay off real children.
- ``ChildE5Embeddings`` applies the asymmetric e5 prefixes, bounds every
  input at 512 tokens Erudi-side (prefix included, through the frozen
  tokenizer), and never instantiates anything at construction.
"""

from __future__ import annotations

import pytest

from src.ingestion import embedder_child, embeddings
from src.ingestion.chunking import E5_MAX_CONTENT_TOKENS, count_tokens

pytestmark = pytest.mark.unit


class TestGetEmbeddings:
    def test_child_backend_where_the_child_is_supported(self, monkeypatch):
        monkeypatch.setattr(embeddings, "_forced_backend", None)
        monkeypatch.setattr(embedder_child, "_adapter_override", object())
        assert isinstance(embeddings.get_embeddings(), embeddings.ChildE5Embeddings)

    def test_in_process_backend_elsewhere(self, monkeypatch):
        monkeypatch.setattr(embeddings, "_forced_backend", None)
        monkeypatch.setattr(embedder_child, "_adapter_override", None)
        assert isinstance(embeddings.get_embeddings(), embeddings.E5Embeddings)

    def test_suite_seam_forces_in_process(self, monkeypatch):
        # The conftest autouse fixture sets this for every not-mlx_only test:
        # the unit suite must never reach a real child.
        monkeypatch.setattr(embeddings, "_forced_backend", "in-process")
        monkeypatch.setattr(embedder_child, "_adapter_override", object())
        assert isinstance(embeddings.get_embeddings(), embeddings.E5Embeddings)


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
