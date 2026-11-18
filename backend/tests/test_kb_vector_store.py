"""P6 — hybrid KB vector store on rag.kb_chunks (PGVectorStore).

Real embedded cluster and real e5 vectors (recorded from the embedding
child, strict: see tests/_e5_reference.py):
- schema/table/HNSW/FK wiring (cross-schema ON DELETE CASCADE [M3]),
- idempotent boot-time init,
- kb_id-filtered hybrid search (dense HNSW + sparse tsvector, RRF k=60),
- anti-regression for the langchain-postgres shared-config bug (the first
  query's fts_query froze onto the store config and contaminated every
  later sparse search — our layer passes a FRESH config per search),
- golden queries FR/EN — the 384d gate: if this fails, escalate the
  embedder to e5-base BEFORE merge.
"""

import pytest
import psycopg

from src.ingestion.chunking import Chunk

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("strict_e5_reference")]


@pytest.fixture(scope="module")
def kb_store(pg_test_cluster, _session_db_engine):
    """Boot-shaped store init: business tables first (FKs), then the store."""
    from src.ingestion import vector_store

    store = vector_store.init_kb_store(pg_test_cluster)
    yield store
    vector_store.close_kb_store()
    # Drop the rag table here (module teardown) so the session-scoped
    # Base.metadata.drop_all isn't blocked by the cross-schema FKs.
    with psycopg.connect(pg_test_cluster.psycopg_url, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS rag.kb_chunks CASCADE")


@pytest.fixture
def kb_rows(pg_test_cluster):
    """Two committed KBs + one document each (the store reads through its
    own connection, so harness-transaction rows would be invisible)."""
    created: list[int] = []
    with psycopg.connect(pg_test_cluster.psycopg_url, autocommit=True) as conn:
        ids = {}
        for label in ("a", "b"):
            kb_id = conn.execute(
                "INSERT INTO knowledge_base DEFAULT VALUES RETURNING id"
            ).fetchone()[0]
            doc_id = conn.execute(
                "INSERT INTO knowledge_documents"
                " (kb_id, name, content_hash_sha256, size_bytes, status)"
                " VALUES (%s, %s, %s, 1024, 'active') RETURNING id",
                (kb_id, f"doc_{label}.pdf", label * 64),
            ).fetchone()[0]
            ids[label] = (kb_id, doc_id)
            created.append(kb_id)
    yield ids
    with psycopg.connect(pg_test_cluster.psycopg_url, autocommit=True) as conn:
        for kb_id in created:
            conn.execute("DELETE FROM knowledge_base WHERE id = %s", (kb_id,))


def _chunks(texts: list[str]) -> list[Chunk]:
    from src.ingestion.chunking import count_tokens

    return [
        Chunk(chunk_index=i, text=t, token_count=count_tokens(t), page_number=None)
        for i, t in enumerate(texts)
    ]


def _add(kb_id: int, doc_id: int, texts: list[str], source_file: str = "doc.pdf"):
    from src.ingestion.vector_store import add_kb_chunks

    return add_kb_chunks(
        kb_id=kb_id,
        document_id=doc_id,
        source_file=source_file,
        chunks=_chunks(texts),
    )


class TestWiring:
    def test_table_columns_fks_and_hnsw_index(self, kb_store, pg_test_cluster):
        with psycopg.connect(pg_test_cluster.psycopg_url) as conn:
            columns = {
                row[0]
                for row in conn.execute(
                    "SELECT column_name FROM information_schema.columns"
                    " WHERE table_schema = 'rag' AND table_name = 'kb_chunks'"
                )
            }
            assert {
                "langchain_id",
                "content",
                "embedding",
                "content_tsv",
                "kb_id",
                "document_id",
                "source_file",
                "page",
                "chunk_index",
            } <= columns

            fks = {
                row[0]: row[1]
                for row in conn.execute(
                    "SELECT conname, confdeltype FROM pg_constraint"
                    " WHERE conrelid = 'rag.kb_chunks'::regclass AND contype = 'f'"
                )
            }
            assert fks.get("fk_kb_chunks_kb_id") == "c"  # ON DELETE CASCADE
            assert fks.get("fk_kb_chunks_document_id") == "c"

            indexes = {
                row[0]
                for row in conn.execute(
                    "SELECT indexname FROM pg_indexes"
                    " WHERE schemaname = 'rag' AND tablename = 'kb_chunks'"
                )
            }
            assert "idx_kb_chunks_embedding_hnsw" in indexes

    def test_init_is_idempotent_and_preserves_data(self, kb_store, kb_rows, pg_test_cluster):
        from src.ingestion import vector_store

        kb_id, doc_id = kb_rows["a"]
        _add(kb_id, doc_id, ["Donnée qui doit survivre à un second boot."])

        vector_store.init_kb_store(pg_test_cluster)  # second boot

        with psycopg.connect(pg_test_cluster.psycopg_url) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM rag.kb_chunks WHERE kb_id = %s", (kb_id,)
            ).fetchone()[0]
        assert count == 1


class TestSearch:
    def test_search_is_filtered_by_kb_and_carries_metadata(self, kb_store, kb_rows):
        from src.ingestion.vector_store import search_kb_chunks_scored

        kb_a, doc_a = kb_rows["a"]
        kb_b, doc_b = kb_rows["b"]
        _add(kb_a, doc_a, ["Paris est la capitale de la France."], "geo.pdf")
        _add(kb_b, doc_b, ["Paris est la capitale de la France."], "copie.pdf")

        results = search_kb_chunks_scored("capitale de la France", kb_id=kb_a)
        assert results
        assert all(doc.metadata["kb_id"] == kb_a for doc, _ in results)
        top = results[0][0].metadata
        assert top["document_id"] == doc_a
        assert top["source_file"] == "geo.pdf"
        assert top["chunk_index"] == 0

    def test_successive_searches_keep_correct_ranking(self, kb_store, kb_rows):
        """Anti-regression: langchain-postgres freezes the first query's
        fts_query onto the SHARED store config, poisoning every later
        sparse search. Our layer must pass a fresh config per search."""
        from src.ingestion.vector_store import search_kb_chunks_scored

        kb_id, doc_id = kb_rows["a"]
        _add(
            kb_id,
            doc_id,
            [
                "Paris est la capitale de la France.",
                "Le contrat REF-88412 expire en mars.",
                "Le chat dort paisiblement sur le canapé.",
            ],
        )

        first = search_kb_chunks_scored("REF-88412", kb_id=kb_id)
        assert "REF-88412" in first[0][0].page_content

        second = search_kb_chunks_scored("capitale de la France", kb_id=kb_id)
        assert "capitale" in second[0][0].page_content

        third = search_kb_chunks_scored("Qui dort sur le canapé ?", kb_id=kb_id)
        assert "chat" in third[0][0].page_content

    def test_pool_is_wider_than_the_lib_branch_default(self, kb_store, kb_rows):
        """Anti-regression (PR3 finding): HybridSearchConfig defaults its
        per-branch SQL LIMITs (primary_top_k/secondary_top_k) to 4 — without
        overriding them the "wide pool" silently degrades to 4+4 candidates.
        With 10 indexed chunks the dense branch alone must return all 10."""
        from src.ingestion.vector_store import search_kb_chunks_scored

        kb_id, doc_id = kb_rows["a"]
        _add(kb_id, doc_id, [f"Note interne numéro {i} sur des sujets divers." for i in range(10)])

        pool = search_kb_chunks_scored("notes internes", kb_id=kb_id)
        assert len(pool) == 10

    def test_similarities_are_calibrated_dense_cosines(self, kb_store, kb_rows):
        """The per-candidate score must be the dense cosine of the STORED
        vector (not the RRF fusion score, which is a rank harmonic): bounded,
        and higher for the on-topic chunk than for the off-topic one."""
        from src.ingestion.vector_store import search_kb_chunks_scored

        kb_id, doc_id = kb_rows["a"]
        _add(
            kb_id,
            doc_id,
            [
                "Le délai de préavis de résiliation est de quatre-vingt-dix jours.",
                "La recette de la tarte aux pommes demande trois pommes.",
            ],
        )

        pool = search_kb_chunks_scored("préavis de résiliation du contrat", kb_id=kb_id)
        sims = {doc.page_content: sim for doc, sim in pool}
        assert all(-1.0 <= sim <= 1.0 for sim in sims.values())
        on_topic = next(s for text, s in sims.items() if "préavis" in text)
        off_topic = next(s for text, s in sims.items() if "pommes" in text)
        assert on_topic > off_topic

    def test_stored_content_is_clean_for_generation(self, kb_store, kb_rows, pg_test_cluster):
        """The [document_name:…] prefix is EMBEDDING-time text only: stored
        content must be the clean chunk (it goes verbatim into the LLM's
        prompt — tiny models loop on the bracketed prefix), and the sparse
        tsv must still be populated by the embeddings-provided insert path."""
        kb_id, doc_id = kb_rows["a"]
        _add(kb_id, doc_id, ["Contenu du chunk."], "rapport.pdf")
        with psycopg.connect(pg_test_cluster.psycopg_url) as conn:
            content, tsv_filled = conn.execute(
                "SELECT content, content_tsv IS NOT NULL FROM rag.kb_chunks" " WHERE kb_id = %s",
                (kb_id,),
            ).fetchone()
        assert content == "Contenu du chunk."
        assert "[document_name:" not in content
        assert tsv_filled

    def test_document_name_contributes_to_the_embedding(self, kb_store, kb_rows, pg_test_cluster):
        """Same chunk text under two different file names → different vectors:
        proof the document name is part of the embedded text (retrieval
        benefit kept) even though the stored content is clean."""
        kb_id, doc_id = kb_rows["a"]
        _add(kb_id, doc_id, ["Texte identique."], "alpha.pdf")
        _add(kb_id, doc_id, ["Texte identique."], "beta.pdf")
        with psycopg.connect(pg_test_cluster.psycopg_url) as conn:
            distinct = conn.execute(
                "SELECT COUNT(DISTINCT embedding::text) FROM rag.kb_chunks" " WHERE kb_id = %s",
                (kb_id,),
            ).fetchone()[0]
        assert distinct == 2


class TestTokenCounts:
    """Each chunk stores its e5 token count in ``langchain_metadata`` (no
    migration: the JSON column takes every key that is not a typed column),
    so retrieval reads it instead of tokenizing the pool on every turn."""

    def test_stored_metadata_carries_the_token_count(self, kb_store, kb_rows, pg_test_cluster):
        from src.ingestion.chunking import count_tokens
        from src.ingestion.vector_store import search_kb_chunks_scored

        kb_id, doc_id = kb_rows["a"]
        text = "Le préavis de résiliation est de quatre-vingt-dix jours calendaires."
        _add(kb_id, doc_id, [text])
        with psycopg.connect(pg_test_cluster.psycopg_url) as conn:
            stored = conn.execute(
                "SELECT langchain_metadata->>'token_count' FROM rag.kb_chunks WHERE kb_id = %s",
                (kb_id,),
            ).fetchone()[0]
        assert int(stored) == count_tokens(text)

        (doc, _sim), *_ = search_kb_chunks_scored("préavis de résiliation", kb_id=kb_id)
        assert doc.metadata["token_count"] == count_tokens(text)


class TestCascade:
    def test_delete_kb_cascades_cross_schema_to_chunks(self, kb_store, kb_rows, pg_test_cluster):
        kb_id, doc_id = kb_rows["a"]
        _add(kb_id, doc_id, ["Chunk voué à disparaître avec sa KB."])

        with psycopg.connect(pg_test_cluster.psycopg_url, autocommit=True) as conn:
            conn.execute("DELETE FROM knowledge_base WHERE id = %s", (kb_id,))
            remaining = conn.execute(
                "SELECT COUNT(*) FROM rag.kb_chunks WHERE kb_id = %s", (kb_id,)
            ).fetchone()[0]
        assert remaining == 0


NIMBUS_LIKE_CORPUS = [
    "Le plan Starter coûte 89 euros par mois et inclut trois utilisateurs.",
    "Le plan Business coûte 290 euros par mois et inclut quinze utilisateurs.",
    "Le plan Enterprise est sur devis à partir de 1100 euros par mois.",
    "Le préavis de résiliation du contrat est de quatre-vingt-dix jours.",
    "La machine à café de l'étage trois est en panne depuis lundi.",
    "Le chat de la mascotte d'entreprise s'appelle Pixel.",
]


class TestKbUtilsAdaptiveSelection:
    """End-to-end selection on REAL e5 embeddings: the adaptive cut must
    scale the injected context to the question's shape (issue #81 problem
    #2 — the flat kb_top_k=1 starved panorama/cross-doc questions)."""

    def test_factoid_question_keeps_a_narrow_context(self, kb_store, kb_rows):
        from types import SimpleNamespace

        from src.utils.kb_utils import retrieve_kb_excerpts

        kb_id, doc_id = kb_rows["a"]
        _add(kb_id, doc_id, NIMBUS_LIKE_CORPUS)

        llm = SimpleNamespace(kb_id=kb_id)
        excerpts = retrieve_kb_excerpts(
            "Quel est le préavis de résiliation du contrat ?", llm.kb_id, token_budget=2000
        )
        texts = [e.text for e in excerpts]
        assert texts
        assert "préavis" in texts[0]
        assert excerpts[0].source_file == "doc.pdf"  # attribution carried through
        # The cut must at least shed the noise chunks (coffee machine, cat).
        assert len(texts) < len(NIMBUS_LIKE_CORPUS)
        assert not any("café" in t or "Pixel" in t for t in texts)

    def test_panorama_question_keeps_the_cluster(self, kb_store, kb_rows):
        from types import SimpleNamespace

        from src.utils.kb_utils import retrieve_kb_excerpts

        kb_id, doc_id = kb_rows["a"]
        _add(kb_id, doc_id, NIMBUS_LIKE_CORPUS)

        llm = SimpleNamespace(kb_id=kb_id)
        texts = [
            e.text
            for e in retrieve_kb_excerpts(
                "Quels sont les plans tarifaires disponibles et leurs prix ?",
                llm.kb_id,
                token_budget=2000,
            )
        ]
        # The three pricing chunks ride together: with kb_top_k=1 this
        # question answered 1 plan out of 3 (baseline T1 failure).
        plans_found = sum(
            1 for plan in ("Starter", "Business", "Enterprise") if any(plan in t for t in texts)
        )
        assert plans_found >= 2

    def test_token_budget_caps_the_context(self, kb_store, kb_rows):
        from types import SimpleNamespace

        from src.ingestion.chunking import count_tokens
        from src.utils.kb_utils import retrieve_kb_excerpts

        kb_id, doc_id = kb_rows["a"]
        _add(kb_id, doc_id, NIMBUS_LIKE_CORPUS)

        llm = SimpleNamespace(kb_id=kb_id)
        question = "Quels sont les plans tarifaires disponibles et leurs prix ?"
        unbounded = retrieve_kb_excerpts(question, llm.kb_id, token_budget=2000)
        assert len(unbounded) >= 2  # the cut alone keeps several candidates

        tight = retrieve_kb_excerpts(
            question, llm.kb_id, token_budget=count_tokens(unbounded[0].text)
        )
        assert tight == unbounded[:1]  # budget bites, best chunk survives


GOLDEN_CORPUS = [
    "La tour Eiffel mesure 330 mètres de haut et se trouve à Paris.",
    "Le télétravail est autorisé trois jours par semaine dans l'entreprise.",
    "La facture FAC-2024-0917 a été réglée le 15 mars.",
    "Pour réinitialiser votre mot de passe, cliquez sur « Mot de passe oublié ».",
    "Le chiffre d'affaires du troisième trimestre a progressé de 12 %.",
    "The quarterly security audit found no critical vulnerabilities.",
    "Employees must badge in before entering the server room.",
    "The new API gateway reduces median latency by 40 milliseconds.",
    "Annual leave requests must be submitted two weeks in advance.",
    "The Berlin office relocates to Alexanderplatz in January.",
]

GOLDEN_QUERIES = [
    ("Quelle est la hauteur de la tour Eiffel ?", 0),
    ("Combien de jours de télétravail par semaine ?", 1),
    ("FAC-2024-0917", 2),  # exact id — sparse branch must nail it
    ("Comment changer mon mot de passe ?", 3),  # paraphrase
    ("évolution du CA au troisième trimestre", 4),  # abbreviation paraphrase
    ("Were any critical vulnerabilities found in the audit?", 5),
    ("How early should I submit my vacation request?", 8),  # EN paraphrase
    ("Where is the Berlin office moving to?", 9),
]


class TestGoldenQueries:
    """The 384d gate: e5-small must resolve these on a realistic mixed
    corpus. Failure → escalate to e5-base (768d) BEFORE merge."""

    def test_golden_queries_fr_en(self, kb_store, kb_rows):
        from src.ingestion.vector_store import search_kb_chunks_scored

        kb_id, doc_id = kb_rows["a"]
        _add(kb_id, doc_id, GOLDEN_CORPUS, "golden.md")

        top1_hits = 0
        misses: list[str] = []
        for query, expected_idx in GOLDEN_QUERIES:
            expected = GOLDEN_CORPUS[expected_idx]
            results = search_kb_chunks_scored(query, kb_id=kb_id)[:2]
            contents = [doc.page_content for doc, _ in results]
            if contents and expected in contents[0]:
                top1_hits += 1
            elif not any(expected in c for c in contents):
                misses.append(f"{query!r} → got {contents[:1]}")

        # Hard gate: every query lands in top-2; soft gate: ≥ 6/8 are top-1.
        assert not misses, f"golden queries missing from top-2: {misses}"
        assert top1_hits >= 6, f"only {top1_hits}/8 golden queries hit top-1"
