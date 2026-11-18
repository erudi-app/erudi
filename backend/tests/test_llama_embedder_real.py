"""The REAL ``llama-server --embedding`` child over the REAL bundled GGUF
(#594, F3) — vector parity with the suite's reference vectors, real authentication,
the real oversized-input refusal.

Needs two artifacts backend-ci does not build, so it SKIPS there and runs on
a dev machine (and in release QA) where both exist:

- the CPU ``llama-server`` (``scripts/dev/backend/build-llamacpp-*``);
- the e5 Q8_0 GGUF (``scripts/dev/backend/build_e5_gguf.py``).

The parity bar is the spike-S4 measurement: Q8_0's worst cosine against
the fp32 model was 0.99981, so 0.9998 with margin here. The reference is
the fp32 snapshot's (``tests/_e5_reference.py``, recorded on macOS). The CI-run
sibling of this file is ``test_embedder_child_http.py`` (the HTTP double
serving vectors captured from this very binary).
"""

from __future__ import annotations

import math
import subprocess
import sys
import time

import pytest
import requests

from src.core.exceptions import EngineException
from src.ingestion import embedder_child, llama_embedder
from src.ingestion.embeddings import ChildE5Embeddings
from tests._e5_reference import PARITY_DOCUMENTS, PARITY_QUERIES, ReferenceE5Embeddings

pytestmark = pytest.mark.integration


def _real_artifacts_present() -> bool:
    try:
        return llama_embedder._find_llama_server() is not None and (
            llama_embedder.artifact_available()
        )
    except EngineException:
        return False


requires_real = pytest.mark.skipif(
    not _real_artifacts_present(),
    reason="needs the compiled CPU llama-server and the e5 GGUF "
    "(build-llamacpp-* + build_e5_gguf.py)",
)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


@pytest.fixture
def real_child(monkeypatch, tmp_path):
    monkeypatch.setattr(
        llama_embedder, "_log_path", lambda port: tmp_path / f"embedder-child-{port}.log"
    )
    monkeypatch.setattr(embedder_child, "_adapter_override", llama_embedder)
    embedder_child._reset_for_tests()
    yield
    embedder_child._reset_for_tests()


@requires_real
class TestRealChildParity:
    def test_vectors_match_the_reference(self, real_child, strict_e5_reference):
        backend = ChildE5Embeddings()
        child_docs = backend.embed_documents(PARITY_DOCUMENTS)
        child_queries = [backend.embed_query(q) for q in PARITY_QUERIES]

        reference = ReferenceE5Embeddings()
        ref_docs = reference.embed_documents(PARITY_DOCUMENTS)
        ref_queries = [reference.embed_query(q) for q in PARITY_QUERIES]

        worst = min(
            min(_cosine(a, b) for a, b in zip(child_docs, ref_docs)),
            min(_cosine(a, b) for a, b in zip(child_queries, ref_queries)),
        )
        print(f"\n[real-parity] worst cosine vs the reference: {worst:.6f}")
        assert worst >= 0.9998

        # L2-normalized, as the store assumes (cosine == dot product).
        for vector in child_docs + child_queries:
            assert abs(math.sqrt(sum(v * v for v in vector)) - 1.0) < 1e-3

    def test_the_real_server_refuses_without_the_key(self, real_child):
        # Proves LLAMA_API_KEY reaches the real binary through the
        # environment: were it ignored, this request would answer 200 and the
        # loopback port would be open to any local process.
        handle = embedder_child.ensure_embedder_running()
        bare = requests.post(
            f"{handle['base_url']}/v1/embeddings",
            json={"model": "multilingual-e5-small", "input": ["x"]},
            timeout=10,
        )
        assert bare.status_code == 401

    def test_oversized_is_refused_raw_and_bounded_by_the_backend(self, real_child):
        oversized = "mot " * 2000
        with pytest.raises(EngineException) as excinfo:
            embedder_child.embed_texts([f"passage: {oversized}"])
        assert "too large" in str(excinfo.value.message)
        assert len(ChildE5Embeddings().embed_documents([oversized])) == 1

    def test_reap_frees_the_child_and_its_memory(self, real_child):
        handle = embedder_child.ensure_embedder_running()
        embedder_child.embed_texts(["passage: un texte pour charger le modele"])
        if sys.platform != "win32":
            rss_kib = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(handle["pid"])],
                capture_output=True,
                text=True,
            ).stdout.strip()
            print(f"\n[real-parity] child RSS after work: {int(rss_kib or 0) // 1024} MiB")
        assert embedder_child.reap_if_idle(
            now=time.monotonic() + embedder_child.EMBEDDER_IDLE_SECONDS + 1
        )
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and handle["proc"].poll() is None:
            time.sleep(0.05)
        assert handle["proc"].poll() is not None
