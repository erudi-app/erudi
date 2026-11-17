"""Hybrid KB vector store — ``rag.kb_chunks`` via langchain-postgres.

One PGVectorStore-managed table for ALL knowledge bases, filtered by the
typed ``kb_id`` column at query time (D2). Retrieval is hybrid from day
one (D3): dense HNSW (cosine, e5 vectors) + sparse tsvector
(``pg_catalog.simple`` — language-neutral, no stemming, exact IDs intact;
the dense branch compensates) fused by Reciprocal Rank Fusion (k=60,
rank-only algebra — NOT a reranker model).

Lifecycle: ``init_kb_store(handle)`` runs in the FastAPI lifespan AFTER
``create_tables`` (the cross-schema FKs below reference the business
tables) and the resident store is reached through module-level accessors
(same pattern as ``config.LLM_Engine``). ``close_kb_store()`` runs on
shutdown BEFORE the cluster stops.

⚠ langchain-postgres 0.0.17 bug: ``asimilarity_search`` writes the first
query's ``fts_query`` onto the SHARED ``HybridSearchConfig`` — every later
sparse search reuses the first query's tsquery. ``search_kb_chunks_scored``
works around it by passing a FRESH config (with the right ``fts_query``)
on every call. Do not "simplify" this away.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Optional

import psycopg
from langchain_postgres import Column, PGEngine, PGVectorStore
from langchain_postgres.v2.hybrid_search_config import (
    HybridSearchConfig,
    reciprocal_rank_fusion,
)
from langchain_postgres.v2.indexes import DistanceStrategy, HNSWIndex

from src.core.logging import logger
from src.core.logutils import truncate_for_log
from src.launcher.postgres_settings import (
    KB_STORE_POOL_MAX_OVERFLOW,
    KB_STORE_POOL_SIZE,
    POOL_TIMEOUT_SECONDS,
)
from src.ingestion.embeddings import (
    EMBEDDING_DIMENSIONS,
    build_embedding_text,
    get_embeddings,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.documents import Document

    from src.ingestion.chunking import Chunk
    from src.launcher.postgres_runtime import PostgresHandle

SCHEMA_NAME = "rag"
TABLE_NAME = "kb_chunks"
TSV_COLUMN = "content_tsv"
TSV_LANG = "pg_catalog.simple"  # language-neutral: no stemming, IDs intact
RRF_K = 60
POOL_K = 20  # hybrid candidates considered per query (recall stage)
HNSW_INDEX_NAME = "idx_kb_chunks_embedding_hnsw"

METADATA_COLUMNS = ["kb_id", "document_id", "source_file", "page", "chunk_index"]

_pg_engine: Optional[PGEngine] = None
_kb_store: Optional[PGVectorStore] = None
_psycopg_url: Optional[str] = None  # retained for the direct similarity SQL


def _hybrid_config(fts_query: str | None = None, pool_k: int = POOL_K) -> HybridSearchConfig:
    """A FRESH hybrid config — required per search, see module docstring.

    ``primary_top_k``/``secondary_top_k`` are the per-branch SQL LIMITs and
    default to 4 in the lib — without overriding them the "wide pool"
    silently degrades to 4+4 candidates. ``fetch_top_k`` (post-fusion) is
    overwritten by the search call's ``k`` anyway; kept coherent here.
    """
    return HybridSearchConfig(
        tsv_column=TSV_COLUMN,
        tsv_lang=TSV_LANG,
        fusion_function=reciprocal_rank_fusion,
        fusion_function_parameters={"rrf_k": RRF_K, "fetch_top_k": pool_k},
        fts_query=fts_query,
        primary_top_k=pool_k,
        secondary_top_k=pool_k,
    )


def _ensure_fk(conn: psycopg.Connection, name: str, ddl: str) -> None:
    """Add a FK once (guarded on pg_constraint — init must be idempotent)."""
    exists = conn.execute("SELECT 1 FROM pg_constraint WHERE conname = %s", (name,)).fetchone()
    if not exists:
        conn.execute(ddl)


def init_kb_store(handle: "PostgresHandle") -> PGVectorStore:
    """Create/join ``rag.kb_chunks`` and build the resident hybrid store.

    Idempotent: safe on every boot (schema/table/FKs/HNSW all guarded).
    Must run AFTER the business tables exist (cross-schema FKs).
    """
    global _pg_engine, _kb_store, _psycopg_url

    with psycopg.connect(handle.psycopg_url, autocommit=True) as conn:
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}")
        table_exists = conn.execute(
            "SELECT 1 FROM information_schema.tables"
            " WHERE table_schema = %s AND table_name = %s",
            (SCHEMA_NAME, TABLE_NAME),
        ).fetchone()

    # Sized by the connection budget the cluster's max_connections is
    # computed from (src/launcher/postgres_settings.py).
    engine = PGEngine.from_connection_string(
        handle.sqlalchemy_url,
        pool_size=KB_STORE_POOL_SIZE,
        max_overflow=KB_STORE_POOL_MAX_OVERFLOW,
        pool_timeout=POOL_TIMEOUT_SECONDS,
    )

    if not table_exists:
        engine.init_vectorstore_table(
            table_name=TABLE_NAME,
            schema_name=SCHEMA_NAME,
            vector_size=EMBEDDING_DIMENSIONS,
            metadata_columns=[
                Column("kb_id", "INTEGER", nullable=False),
                Column("document_id", "INTEGER", nullable=False),
                Column("source_file", "TEXT", nullable=True),
                Column("page", "INTEGER", nullable=True),
                Column("chunk_index", "INTEGER", nullable=True),
            ],
            hybrid_search_config=_hybrid_config(),
        )

    with psycopg.connect(handle.psycopg_url, autocommit=True) as conn:
        _ensure_fk(
            conn,
            "fk_kb_chunks_kb_id",
            f"ALTER TABLE {SCHEMA_NAME}.{TABLE_NAME}"
            " ADD CONSTRAINT fk_kb_chunks_kb_id FOREIGN KEY (kb_id)"
            " REFERENCES public.knowledge_base(id) ON DELETE CASCADE",
        )
        _ensure_fk(
            conn,
            "fk_kb_chunks_document_id",
            f"ALTER TABLE {SCHEMA_NAME}.{TABLE_NAME}"
            " ADD CONSTRAINT fk_kb_chunks_document_id FOREIGN KEY (document_id)"
            " REFERENCES public.knowledge_documents(id) ON DELETE CASCADE",
        )

    store = PGVectorStore.create_sync(
        engine=engine,
        table_name=TABLE_NAME,
        schema_name=SCHEMA_NAME,
        embedding_service=get_embeddings(),
        metadata_columns=METADATA_COLUMNS,
        hybrid_search_config=_hybrid_config(),
    )

    if not store.is_valid_index(HNSW_INDEX_NAME):
        store.apply_vector_index(
            HNSWIndex(
                name=HNSW_INDEX_NAME,
                distance_strategy=DistanceStrategy.COSINE_DISTANCE,
            )
        )

    # Swap-in last: never leave a half-initialized store visible.
    if _pg_engine is not None:
        _close_engine(_pg_engine)
    _pg_engine, _kb_store, _psycopg_url = engine, store, handle.psycopg_url
    logger.info(f"KB vector store ready ({SCHEMA_NAME}.{TABLE_NAME}, hybrid RRF)")
    return store


def _close_engine(engine: PGEngine) -> None:
    # PGEngine.close() is async-only; _run_as_sync delegates to its internal
    # loop (the same bridge every sync method of the lib uses). Revisit if
    # the lib ever ships a public sync close.
    engine._run_as_sync(engine.close())


def get_kb_store() -> PGVectorStore:
    """The resident store (set by the lifespan via ``init_kb_store``)."""
    if _kb_store is None:
        raise RuntimeError(
            "KB vector store not initialized — call init_kb_store() first "
            "(FastAPI lifespan does this at startup)."
        )
    return _kb_store


def close_kb_store() -> None:
    """Release the store's PGEngine (lifespan shutdown, before cluster stop)."""
    global _pg_engine, _kb_store, _psycopg_url
    if _pg_engine is not None:
        _close_engine(_pg_engine)
    _pg_engine = None
    _kb_store = None
    _psycopg_url = None


def add_kb_chunks(
    *,
    kb_id: int,
    document_id: int,
    source_file: str,
    chunks: list["Chunk"],
) -> list[str]:
    """Embed and store one document's chunks (sync — delegates to the
    PGEngine's internal loop; the P0bis-validated ingestion bridge).

    The ``[document_name:…]`` prefix is EMBEDDING-time text only (it boosts
    retrieval, POC pattern): vectors are computed over the prefixed text,
    but the STORED content is the clean chunk — it goes verbatim into the
    LLM prompt at generation time, and small models loop on the bracketed
    prefix (live-E2E finding)."""
    if not chunks:
        return []
    start_s = time.perf_counter()
    embedding_texts = [
        build_embedding_text(file_name=source_file, chunk_text=chunk.text) for chunk in chunks
    ]
    vectors = get_embeddings().embed_documents(embedding_texts)
    texts = [chunk.text for chunk in chunks]
    metadatas = [
        {
            "kb_id": kb_id,
            "document_id": document_id,
            "source_file": source_file,
            "page": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            # Not a typed column: it lands in the langchain_metadata JSON and
            # comes back in Document.metadata, where retrieval's token budget
            # reads it instead of tokenizing every candidate on every turn.
            "token_count": chunk.token_count,
        }
        for chunk in chunks
    ]
    ids = get_kb_store().add_embeddings(texts=texts, embeddings=vectors, metadatas=metadatas)
    duration_ms = (time.perf_counter() - start_s) * 1000
    logger.info(
        f"KB chunks added: kb_id={kb_id}, document_id={document_id}, "
        f"source_file={source_file}, n_chunks={len(chunks)}, "
        f"duration_ms={duration_ms:.0f}"
    )
    return ids


def _dense_similarities(ids: list[str], query_vector: list[float]) -> dict[str, float]:
    """Cosine similarity of the STORED vectors against the query vector.

    One PK-indexed SQL read. Recomputing embeddings over the clean content
    would diverge: stored vectors embed the ``[document_name:…]`` prefixed
    text (see ``add_kb_chunks``), and the cut must use the same geometry
    the retrieval ranked with.
    """
    if _psycopg_url is None:
        raise RuntimeError("KB vector store not initialized — call init_kb_store() first.")
    with psycopg.connect(_psycopg_url) as conn:
        rows = conn.execute(
            f'SELECT langchain_id, 1 - ("embedding" <=> %s::vector)'
            f' FROM "{SCHEMA_NAME}"."{TABLE_NAME}" WHERE langchain_id = ANY(%s)',
            (str(query_vector), [uuid.UUID(i) for i in ids]),
        ).fetchall()
    return {str(row[0]): float(row[1]) for row in rows}


def search_kb_chunks_scored(
    query: str, *, kb_id: int, pool_k: int = POOL_K
) -> list[tuple["Document", float]]:
    """Hybrid candidate pool over one KB: documents in RRF order, each with
    its calibrated dense cosine similarity.

    The fusion overwrites per-row scores with RRF harmonics (rank algebra,
    no semantic scale), so the dense similarity is re-read from the stored
    vectors — that is what the adaptive cut upstream operates on. Fresh
    config per call (lib bug — see module docstring); the query is embedded
    once and shared by the search and the similarity read.
    """
    start_s = time.perf_counter()
    query_vector = get_embeddings().embed_query(query)
    documents = get_kb_store().similarity_search_by_vector(
        query_vector,
        k=pool_k,
        filter={"kb_id": kb_id},
        hybrid_search_config=_hybrid_config(fts_query=query, pool_k=pool_k),
    )
    if not documents:
        duration_ms = (time.perf_counter() - start_s) * 1000
        logger.info(
            f"KB search: kb_id={kb_id}, k={pool_k}, hits=0, top_score=n/a, "
            f"duration_ms={duration_ms:.0f}, "
            f"query={truncate_for_log(query, 2000)}"
        )
        return []
    similarities = _dense_similarities([doc.id for doc in documents], query_vector)
    duration_ms = (time.perf_counter() - start_s) * 1000
    top_score = max(similarities.values()) if similarities else float("nan")
    logger.info(
        f"KB search: kb_id={kb_id}, k={pool_k}, hits={len(documents)}, "
        f"top_score={top_score:.4f}, duration_ms={duration_ms:.0f}, "
        f"query={truncate_for_log(query, 2000)}"
    )
    return [(doc, similarities[doc.id]) for doc in documents]
