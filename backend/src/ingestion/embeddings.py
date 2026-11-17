"""The Knowledge Base embeddings backends behind the LangChain ``Embeddings``
interface, and the platform factory that picks one (#594).

multilingual-e5-small (384d, MIT, 512-token window) REQUIRES asymmetric
prefixes: ``query: `` for searches, ``passage: `` for indexed chunks —
without them retrieval quality collapses. Both backends apply them here,
Erudi-side, and both produce mean-pooled, L2-normalized vectors, so cosine
similarity is a plain dot product (pgvector COSINE / HNSW friendly) and the
two are interchangeable vector-for-vector (measured cosine 0.9999999,
inputs past 512 tokens included) — existing macOS Knowledge Bases keep
their vectors with no re-indexing.

- ``ChildE5Embeddings`` (macOS ARM): the dedicated `mlx_vlm.server` child
  (``src.ingestion.embedder_child``). Nothing resident in the backend; the
  child is released after 5 minutes unused. Inputs are bounded at e5's
  512-token window Erudi-side (``chunking.truncate_to_tokens``, prefix
  included) before the call.
- ``E5Embeddings`` (Windows and Linux, until F3 of #594): the resident
  in-process sentence-transformers singleton (the FAISS-era embedder
  reloaded ~470 MB from disk on every single operation). F3 moves these
  platforms onto a ``llama-server --embedding`` child; F4 then deletes this
  class and the sentence-transformers/torch stack with it.

``get_embeddings()`` is the only selection point: `vector_store` calls it
per operation, so the choice stays out of the KB business layer.

Embedded-text format [N6], frozen:
    ``passage: [document_name:<file>]\\n<breadcrumb>\\n\\n<chunk>``
(``passage: `` added here; ``[document_name:…]`` by ``build_embedding_text``;
breadcrumb by the chunker.)
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, ClassVar, Optional

from langchain_core.embeddings import Embeddings

from src.core import config
from src.core.logging import logger
from src.ingestion import embedder_child
from src.ingestion.chunking import truncate_to_tokens
from src.ingestion.embedding_model import embedding_model_available, load_sentence_transformer

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sentence_transformers import SentenceTransformer

E5_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_DIMENSIONS = 384

#: Test seam (set by the suite's conftest): ``"in-process"`` pins
#: ``E5Embeddings`` so the not-mlx_only suite never reaches a real child,
#: ``"child"`` pins ``ChildE5Embeddings``; ``None`` selects by platform.
_forced_backend: Optional[str] = None


def get_embeddings() -> Embeddings:
    """The platform's KB embeddings backend (see module docstring)."""
    if _forced_backend == "in-process":
        return E5Embeddings()
    if _forced_backend == "child":
        return ChildE5Embeddings()
    if embedder_child.embedder_child_supported():
        return ChildE5Embeddings()
    return E5Embeddings()


def embedding_artifact_missing() -> bool:
    """Whether the selected backend NEEDS the e5 artifact and it is absent.

    True only on the child backend (the child serves the snapshot by path
    and cannot download): the "KB present, embedding model missing" state
    the interface surfaces, and what lets an ingestion refuse its job up
    front instead of failing file by file. The in-process backend still
    fetches lazily on first use (pre-#146 behaviour, gone with F4), so it
    never reports missing.
    """
    if _forced_backend == "in-process":
        return False
    if _forced_backend == "child" or embedder_child.embedder_child_supported():
        return not embedding_model_available()
    return False


def build_embedding_text(*, file_name: str, chunk_text: str) -> str:
    """Prefix the chunk with its source document name (R&D POC format)."""
    return f"[document_name:{file_name}]\n{chunk_text}"


class ChildE5Embeddings(Embeddings):
    """e5 embeddings served by the dedicated embedding child (#594, macOS).

    Stateless by design: construction touches nothing (``init_kb_store``
    builds one in the lifespan), the child spawns lazily on the first call
    and `embedder_child` owns its whole lifecycle. The prefixes and the
    512-token bound are applied HERE so the transport layer stays a dumb
    batch pipe F3 reuses unchanged.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed indexed chunks (``passage: `` prefix, bounded, L2-normalized)."""
        start_s = time.perf_counter()
        vectors = embedder_child.embed_texts(
            [truncate_to_tokens(f"passage: {text}") for text in texts]
        )
        duration_ms = (time.perf_counter() - start_s) * 1000
        # One line per BATCH — never per text.
        logger.info(f"Embedded batch via child: {len(texts)} texts in {duration_ms:.0f}ms")
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (``query: `` prefix, bounded, L2-normalized)."""
        return embedder_child.embed_texts([truncate_to_tokens(f"query: {text}")])[0]


class E5Embeddings(Embeddings):
    """LangChain ``Embeddings`` over a resident e5-small singleton."""

    _model: ClassVar["SentenceTransformer | None"] = None
    _model_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def _get_model(cls) -> "SentenceTransformer":
        if cls._model is None:
            with cls._model_lock:
                if cls._model is None:
                    logger.info(f"Loading embedding model {E5_MODEL_NAME} (resident)")
                    # cache_folder pins the model INSIDE the app data dir
                    # (CACHE_DIR = data/models_cache) instead of the global
                    # ~/.cache/huggingface, so it is self-contained and the #146
                    # gate can check/download it at a path the app owns.
                    # local_files_only once the cache is complete: without it,
                    # hub HEAD-revalidates every file on load and an offline
                    # machine fails on the DNS probe despite the pre-download
                    # (#164).
                    # load_sentence_transformer owns the device choice: MPS when
                    # Metal really works, CPU when it only claims to (#335).
                    cls._model = load_sentence_transformer(
                        E5_MODEL_NAME,
                        cache_folder=str(config.CACHE_DIR),
                        local_files_only=embedding_model_available(),
                    )
        return cls._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed indexed chunks (``passage: `` prefix, L2-normalized)."""
        start_s = time.perf_counter()
        vectors = self._get_model().encode(
            [f"passage: {text}" for text in texts],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        duration_ms = (time.perf_counter() - start_s) * 1000
        # One line per BATCH — never per text.
        logger.info(f"Embedded batch: {len(texts)} texts in {duration_ms:.0f}ms")
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (``query: `` prefix, L2-normalized)."""
        vector = self._get_model().encode(
            [f"query: {text}"],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return vector.tolist()
