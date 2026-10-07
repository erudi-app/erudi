"""P5 — E5Embeddings: resident singleton, e5 prefixes, L2-normalized 384d.

multilingual-e5-small REQUIRES asymmetric prefixes ("query: " / "passage: ")
— without them retrieval quality collapses. The model stays resident (the
FAISS-era embedder reloaded ~470 MB on every operation).

The model comes from the local cache (``CACHE_DIR``). Without it, loading it
means downloading ~470 MB from the HF Hub, which can hold a run for many
minutes: the module is then a ``network`` test, opt-in through
``ERUDI_TEST_NETWORK=1`` like the other live-Hub tests.
"""

import math
import os

import pytest
from langchain_core.embeddings import Embeddings

from src.ingestion.embedding_model import embedding_model_available
from src.ingestion.embeddings import E5Embeddings, build_embedding_text

pytestmark = [
    pytest.mark.integration,
    pytest.mark.network,
    pytest.mark.skipif(
        not embedding_model_available() and not os.environ.get("ERUDI_TEST_NETWORK"),
        reason="the embedding model is not cached (a ~470 MB download); set ERUDI_TEST_NETWORK=1",
    ),
]


@pytest.fixture(scope="module")
def embedder():
    return E5Embeddings()


class TestE5Embeddings:
    def test_implements_langchain_embeddings_interface(self, embedder):
        assert isinstance(embedder, Embeddings)

    def test_documents_are_384d_and_l2_normalized(self, embedder):
        vectors = embedder.embed_documents(["Paris est la capitale de la France."])
        assert len(vectors) == 1
        assert len(vectors[0]) == 384
        norm = math.sqrt(sum(x * x for x in vectors[0]))
        assert abs(norm - 1.0) < 1e-3

    def test_query_is_384d_and_l2_normalized(self, embedder):
        vector = embedder.embed_query("Quelle est la capitale de la France ?")
        assert len(vector) == 384
        norm = math.sqrt(sum(x * x for x in vector))
        assert abs(norm - 1.0) < 1e-3

    def test_query_and_passage_prefixes_differ(self, embedder):
        # Same raw text through the two paths → different vectors, proving
        # the asymmetric prefixes are actually applied.
        text = "Le café des développeurs."
        as_query = embedder.embed_query(text)
        as_passage = embedder.embed_documents([text])[0]
        assert as_query != as_passage

    def test_semantic_smoke_french(self, embedder):
        # Minimal relevance sanity check ahead of the P6 golden-query gate.
        query = embedder.embed_query("Quelle est la capitale de la France ?")
        relevant, irrelevant = embedder.embed_documents(
            [
                "Paris est la capitale de la France.",
                "Le chat dort paisiblement sur le canapé du salon.",
            ]
        )
        sim_relevant = sum(q * d for q, d in zip(query, relevant))
        sim_irrelevant = sum(q * d for q, d in zip(query, irrelevant))
        assert sim_relevant > sim_irrelevant

    def test_model_is_a_resident_class_singleton(self, embedder):
        assert E5Embeddings._get_model() is E5Embeddings._get_model()
        assert E5Embeddings()._get_model() is embedder._get_model()


class TestBuildEmbeddingText:
    def test_document_name_prefix_format(self):
        # [N6] frozen order: "passage: " (added by embed_documents) →
        # "[document_name:…]" → breadcrumb + chunk text.
        assert (
            build_embedding_text(file_name="rapport.pdf", chunk_text="# A\n\nContenu.")
            == "[document_name:rapport.pdf]\n# A\n\nContenu."
        )
