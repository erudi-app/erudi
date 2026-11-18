"""The Knowledge Base embeddings behind the LangChain ``Embeddings`` interface
(#594).

multilingual-e5-small (384d, MIT, 512-token window) REQUIRES asymmetric
prefixes: ``query: `` for searches, ``passage: `` for indexed chunks —
without them retrieval quality collapses. They are applied here, Erudi-side,
and the served vectors are mean-pooled and L2-normalized, so cosine
similarity is a plain dot product (pgvector COSINE / HNSW friendly).

``ChildE5Embeddings`` is the one backend, on every platform: the dedicated
embedding child (``src.ingestion.embedder_child``) — `mlx_vlm.server` over
the local e5 snapshot on macOS ARM (``mlx_embedder``), the bundled CPU
``llama-server --embedding`` over the e5 GGUF on Windows and Linux
(``llama_embedder``). Nothing resident in the backend; the child is released
after 5 minutes unused. Inputs are bounded at e5's 512-token window
Erudi-side (``chunking.truncate_to_tokens``, prefix included) before the
call — llama-server REFUSES a longer input (HTTP 500, the whole batch).

``get_embeddings()`` is what `vector_store` calls per operation, so the
backend stays out of the KB business layer.

Embedded-text format [N6], frozen:
    ``passage: [document_name:<file>]\\n<breadcrumb>\\n\\n<chunk>``
(``passage: `` added here; ``[document_name:…]`` by ``build_embedding_text``;
breadcrumb by the chunker.)
"""

from __future__ import annotations

import time

from langchain_core.embeddings import Embeddings

from src.core.logging import logger
from src.ingestion import embedder_child, embedding_model
from src.ingestion.chunking import truncate_to_tokens

EMBEDDING_DIMENSIONS = 384

#: The status ``error`` where the artifact ships with the installation and is
#: not on disk (a dev tree before ``build_e5_gguf.py``, a quarantined file):
#: the gate/banner shows it verbatim, so the user is told what to do instead
#: of being offered a download that cannot produce the file.
BUNDLED_ARTIFACT_MISSING_MESSAGE = (
    "the embedding model file that ships with Erudi is missing from this "
    "installation; reinstall Erudi to restore it (developers: run "
    "scripts/dev/backend/build_e5_gguf.py)"
)


def get_embeddings() -> Embeddings:
    """The KB embeddings backend (see module docstring)."""
    return ChildE5Embeddings()


def embedding_artifact_missing() -> bool:
    """Whether the embedding child's artifact is absent — the e5 snapshot on
    macOS, the bundled GGUF on Windows/Linux.

    The child serves a local artifact and cannot download it itself: this is
    the "KB present, embedding model missing" state the interface surfaces,
    and what lets an ingestion refuse its job up front instead of failing
    file by file.
    """
    return not embedder_child.artifact_available()


def embedding_artifact_status() -> dict:
    """What the interface's gate and banner poll: the embedding artifact's
    state, in the #146 shape (``available``/``downloading``/``error``).

    Where the artifact is downloadable (macOS: the child serves the snapshot
    the gate downloads) this is the #146 download state verbatim. Where it
    ships with the installation (Windows/Linux: the bundled GGUF) only its
    on-disk presence speaks — a packaged install always has it; an ABSENT
    file is an explicit ``error`` naming the remedy, because no download can
    produce it and a silent prompt would offer one.
    """
    if not embedder_child.artifact_downloadable():
        available = embedder_child.artifact_available()
        return {
            "available": available,
            "downloading": False,
            "error": None if available else BUNDLED_ARTIFACT_MISSING_MESSAGE,
        }
    return embedding_model.download_state()


def start_embedding_artifact_download() -> dict:
    """The gate's download action: starts the #146 download where it
    produces the child's artifact (macOS), and is a NO-OP — one INFO, the
    current state returned — where the artifact ships with the installation
    (downloading the snapshot would not produce the GGUF)."""
    if not embedder_child.artifact_downloadable():
        logger.info(
            "Embedding model download skipped: the artifact ships with the "
            "installation (dev: produce it with scripts/dev/backend/build_e5_gguf.py)"
        )
        return embedding_artifact_status()
    return embedding_model.start_download()


def build_embedding_text(*, file_name: str, chunk_text: str) -> str:
    """Prefix the chunk with its source document name (R&D POC format)."""
    return f"[document_name:{file_name}]\n{chunk_text}"


class ChildE5Embeddings(Embeddings):
    """e5 embeddings served by the dedicated embedding child (#594).

    Stateless by design: construction touches nothing (``init_kb_store``
    builds one in the lifespan), the child spawns lazily on the first call
    and `embedder_child` owns its whole lifecycle. The prefixes and the
    512-token bound are applied HERE so the transport (``_embed``) stays a
    dumb batch pipe.
    """

    @staticmethod
    def _embed(texts: list[str]) -> list[list[float]]:
        """The transport: prefixed, bounded texts in, vectors out."""
        return embedder_child.embed_texts(texts)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed indexed chunks (``passage: `` prefix, bounded, L2-normalized)."""
        start_s = time.perf_counter()
        vectors = self._embed([truncate_to_tokens(f"passage: {text}") for text in texts])
        duration_ms = (time.perf_counter() - start_s) * 1000
        # One line per BATCH — never per text.
        logger.info(f"Embedded batch via child: {len(texts)} texts in {duration_ms:.0f}ms")
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (``query: `` prefix, bounded, L2-normalized)."""
        return self._embed([truncate_to_tokens(f"query: {text}")])[0]
