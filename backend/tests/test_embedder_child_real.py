"""Real `mlx_vlm.server` embedding child on the cached e5 snapshot (#594).

Spawns the ACTUAL child through the platform adapter and holds it to the
suite's reference vectors (``tests/_e5_reference.py``, cosine >= 0.9999),
an input past the 512-token window included: the vectors every KB test
runs on are the ones the shipped child serves.

Skips cleanly when the snapshot is not already in ``data/models_cache``:
tests never download the ~470 MB model.
"""

from __future__ import annotations

import math
import time

import pytest
import requests

from src.ingestion import embedder_child
from src.ingestion.embedding_model import embedding_model_available
from src.ingestion.embeddings import ChildE5Embeddings
from tests._e5_reference import PARITY_DOCUMENTS, PARITY_QUERIES, ReferenceE5Embeddings
from tests._helpers import is_mlx_platform

pytestmark = pytest.mark.mlx_only

if not is_mlx_platform():  # pragma: no cover - platform guard
    pytest.skip("mlx_vlm embedding child requires Apple Silicon", allow_module_level=True)

if not embedding_model_available():  # pragma: no cover - cache guard
    pytest.skip(
        "e5 snapshot not cached in data/models_cache (tests never download it)",
        allow_module_level=True,
    )


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


@pytest.fixture(scope="module")
def real_child():
    # Opt back into the REAL platform adapter: the session-wide conftest
    # fixture disarms the manager for the whole suite, and this module is
    # exactly the place that must reach the real child.
    adapter_before = embedder_child._adapter_override
    embedder_child._adapter_override = embedder_child._PLATFORM
    embedder_child._reset_for_tests()
    yield
    embedder_child.shutdown_embedder()
    embedder_child._reset_for_tests()
    embedder_child._adapter_override = adapter_before


class TestRealEmbeddingChild:
    def test_vectors_match_the_reference(self, real_child, strict_e5_reference):
        spawn_start = time.monotonic()
        child_backend = ChildE5Embeddings()
        child_docs = child_backend.embed_documents(PARITY_DOCUMENTS)
        spawn_s = time.monotonic() - spawn_start
        child_queries = [child_backend.embed_query(q) for q in PARITY_QUERIES]

        reference = ReferenceE5Embeddings()
        reference_docs = reference.embed_documents(PARITY_DOCUMENTS)
        reference_queries = [reference.embed_query(q) for q in PARITY_QUERIES]

        for vector in child_docs + child_queries:
            assert len(vector) == 384
            # L2-normalized, as the store assumes (cosine == dot product).
            assert abs(math.sqrt(sum(v * v for v in vector)) - 1.0) < 1e-3
        cosines = [_cosine(c, r) for c, r in zip(child_docs, reference_docs)]
        cosines += [_cosine(c, r) for c, r in zip(child_queries, reference_queries)]
        assert min(cosines) >= 0.9999, f"cosines={cosines}"
        # Not an assertion — recorded for the PR's measurements.
        print(f"\n[measure] first embed_documents call (spawn+load+probe+embed): {spawn_s:.1f}s")
        print(f"[measure] cosines child vs reference: {[f'{c:.7f}' for c in cosines]}")

    def test_child_refuses_requests_without_the_key(self, real_child):
        handle = embedder_child.ensure_embedder_running()
        base_url = handle["base_url"]
        response = requests.post(
            f"{base_url}/v1/embeddings",
            json={"model": handle["model_value"], "input": ["x"]},
            timeout=10.0,
        )
        assert response.status_code == 401
        assert requests.get(f"{base_url}/health", timeout=10.0).status_code == 401

    def test_idle_reap_terminates_the_real_child(self, real_child):
        handle = embedder_child.ensure_embedder_running()
        proc = handle["proc"]
        assert proc.is_alive()
        assert embedder_child.reap_if_idle(
            now=embedder_child._last_used + embedder_child.EMBEDDER_IDLE_SECONDS + 1.0
        )
        deadline = time.monotonic() + 10.0
        while proc.is_alive() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not proc.is_alive()
        assert embedder_child._handle is None
