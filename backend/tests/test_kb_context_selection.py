"""PR3 — adaptive KB context selection (issue #81, problem #2).

Unit tests for the two-stage selection that replaces the flat ``kb_top_k=1``:
1. adaptive cut — keep candidates above the largest dense-similarity
   drop-off (Adaptive-k transposed to our hybrid RRF pool: the cut runs on
   calibrated cosine similarities, never on RRF rank harmonics);
2. token budget — keep whole chunks best-first (RRF order) within the
   model-size budget from ``get_prompting_strategy``.

The selection math is pure (no cluster); ``retrieve_kb_excerpts`` is
exercised with a mocked pool. The real-embeddings behaviour (factoid vs
panorama questions) lives in test_kb_vector_store.py.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from src.core.exceptions import KnowledgeBaseNotFoundException
from src.ingestion.chunking import count_tokens
from src.utils.kb_utils import (
    KbExcerpt,
    _adaptive_threshold,
    _truncate_to_token_budget,
    retrieve_kb_excerpts,
)

pytestmark = pytest.mark.unit


class TestAdaptiveThreshold:
    def test_clear_gap_after_first_keeps_only_the_top(self):
        # One candidate crushes the rest → factoid signature, cut after it.
        sims = [0.92, 0.55, 0.54, 0.53]
        assert _adaptive_threshold(sims) == 0.92

    def test_gap_after_third_keeps_the_cluster(self):
        # Three close candidates then a drop → panorama signature.
        sims = [0.88, 0.86, 0.85, 0.45, 0.44]
        assert _adaptive_threshold(sims) == 0.85

    def test_flat_distribution_has_no_cut_signal(self):
        # No meaningful drop-off → None (the token budget decides alone).
        sims = [0.700, 0.696, 0.693, 0.690]
        assert _adaptive_threshold(sims) is None

    def test_single_candidate_has_no_cut(self):
        assert _adaptive_threshold([0.9]) is None

    def test_empty_pool_has_no_cut(self):
        assert _adaptive_threshold([]) is None

    def test_input_order_does_not_matter(self):
        # The pool arrives in RRF order, not sorted by similarity.
        rrf_order = [0.55, 0.92, 0.54, 0.53]
        assert _adaptive_threshold(rrf_order) == 0.92

    # --- Position-1 gap distrust + real field distributions (issue #221) ---

    def test_position_zero_outlier_gap_is_honored(self):
        # Legitimate case: one ultra-relevant doc, off-topic rest. The 0.16 top
        # gap is >> 3x the median head gap (0.01), so the keep-1 cut stands.
        sims = [0.86, 0.70, 0.69, 0.68]
        assert _adaptive_threshold(sims) == 0.86

    def test_position_zero_non_outlier_gap_is_distrusted(self):
        # Pathological field case (issue #221): the top gap 0.0168 sits at
        # index 0 but is NOT a true outlier (<= 3x median 0.0060), so it would
        # collapse the pool to the (wrong) top hit -> no cut, budget decides.
        sims = [0.8618, 0.8450, 0.8420, 0.8380, 0.8300]
        assert _adaptive_threshold(sims) is None

    def test_two_candidate_pool_never_top_cuts(self):
        # A single gap cannot be judged an outlier (issue #221): distrust it
        # instead of collapsing to the top hit; the recall floor keeps both.
        assert _adaptive_threshold([0.90, 0.40]) is None

    @pytest.mark.parametrize(
        "sims, expected_threshold, expected_kept",
        [
            # Captured on the live win-cpu test KB (issue #221, part 2). The
            # deciding gap sits mid-pool (not index 0), so the cut is honored
            # and keeps >= 2 excerpts in every case.
            ([0.8633, 0.8611, 0.8598, 0.8270, 0.8177], 0.8598, 3),
            ([0.8624, 0.8495, 0.8343, 0.8195, 0.7958], 0.8195, 4),
            ([0.8233, 0.8160, 0.7899, 0.7791, 0.7784], 0.8160, 2),
        ],
    )
    def test_real_field_distributions_keep_at_least_two(
        self, sims, expected_threshold, expected_kept
    ):
        threshold = _adaptive_threshold(sims)
        assert threshold == expected_threshold
        kept = [s for s in sims if s >= threshold]
        assert kept == sims[:expected_kept]
        assert len(kept) >= 2


class TestTokenBudgetTruncation:
    """``_truncate_to_token_budget`` takes ``(text, stored token_count)`` pairs
    and returns how many of them fit, best-first."""

    def test_keeps_whole_chunks_within_budget(self):
        texts = [
            "Le préavis de résiliation est de quatre-vingt-dix jours calendaires.",
            "La redevance mensuelle est facturée à terme à échoir.",
            "Le présent contrat est conclu pour une durée de trente-six mois.",
        ]
        budget = count_tokens(texts[0]) + count_tokens(texts[1])
        assert _truncate_to_token_budget([(t, None) for t in texts], budget) == 2

    def test_first_chunk_always_survives_a_tiny_budget(self):
        # Never return zero context because one chunk overflows the budget.
        chunks = [("Un chunk nettement plus long que le budget accordé ici.", None)]
        assert _truncate_to_token_budget(chunks, 1) == 1

    def test_large_budget_keeps_everything_in_order(self):
        chunks = [("alpha", None), ("bravo", None), ("charlie", None)]
        assert _truncate_to_token_budget(chunks, 10_000) == 3

    def test_empty_input(self):
        assert _truncate_to_token_budget([], 500) == 0

    def test_stored_counts_are_used_without_tokenizing(self, monkeypatch):
        def boom(text):
            raise AssertionError("a chunk with a stored count must not be tokenized")

        monkeypatch.setattr("src.utils.kb_utils.count_tokens", boom)
        chunks = [("alpha", 60), ("bravo", 50), ("charlie", 10)]
        assert _truncate_to_token_budget(chunks, 110) == 2

    def test_only_chunks_without_a_stored_count_are_counted(self, monkeypatch):
        counted = []

        def spy(text):
            counted.append(text)
            return 5

        monkeypatch.setattr("src.utils.kb_utils.count_tokens", spy)
        chunks = [("alpha", 10), ("bravo", None), ("charlie", 10), ("delta", None)]
        assert _truncate_to_token_budget(chunks, 1000) == 4
        assert counted == ["bravo", "delta"]

    @pytest.mark.parametrize("stored", [-1, "12", 1.5, True])
    def test_an_unusable_stored_count_is_recounted(self, stored, monkeypatch):
        monkeypatch.setattr("src.utils.kb_utils.count_tokens", lambda text: 7)
        assert _truncate_to_token_budget([("a", stored), ("b", stored)], 13) == 1


def _pool(entries, token_counts=None):
    token_counts = token_counts or {}
    return [
        (
            Document(
                page_content=text,
                id=str(i),
                metadata={"source_file": f"doc-{i}.md"}
                | ({"token_count": token_counts[i]} if i in token_counts else {}),
            ),
            sim,
        )
        for i, (text, sim) in enumerate(entries)
    ]


class TestRetrieveKbExcerpts:
    def test_cut_then_budget_in_rrf_order(self):
        # RRF order ≠ similarity order: the cut filters by similarity but the
        # surviving chunks keep their RRF (best-first) injection order.
        pool = _pool(
            [
                ("chunk pertinent A", 0.88),
                ("chunk pertinent B", 0.86),
                ("chunk hors-sujet", 0.45),
                ("chunk pertinent C", 0.85),
            ]
        )
        llm = SimpleNamespace(kb_id=7)
        with patch("src.utils.kb_utils.search_kb_chunks_scored", return_value=pool):
            excerpts = retrieve_kb_excerpts("question", llm.kb_id, token_budget=2000)
        assert [e.text for e in excerpts] == [
            "chunk pertinent A",
            "chunk pertinent B",
            "chunk pertinent C",
        ]

    def test_excerpts_carry_their_source_file(self):
        pool = _pool([("contenu", 0.9)])
        llm = SimpleNamespace(kb_id=7)
        with patch("src.utils.kb_utils.search_kb_chunks_scored", return_value=pool):
            excerpts = retrieve_kb_excerpts("question", llm.kb_id, token_budget=2000)
        assert excerpts == [KbExcerpt(source_file="doc-0.md", text="contenu")]

    def test_flat_pool_falls_back_to_budget_only(self):
        pool = _pool([("a", 0.701), ("b", 0.700), ("c", 0.699)])
        llm = SimpleNamespace(kb_id=7)
        with patch("src.utils.kb_utils.search_kb_chunks_scored", return_value=pool):
            excerpts = retrieve_kb_excerpts("question", llm.kb_id, token_budget=2000)
        assert [e.text for e in excerpts] == ["a", "b", "c"]

    def test_budget_caps_the_survivors(self):
        long_a = "mot " * 120  # well above a 100-token budget on its own
        pool = _pool([(long_a, 0.88), ("suite du contexte", 0.86)])
        llm = SimpleNamespace(kb_id=7)
        with patch("src.utils.kb_utils.search_kb_chunks_scored", return_value=pool):
            excerpts = retrieve_kb_excerpts("question", llm.kb_id, token_budget=100)
        assert [e.text for e in excerpts] == [long_a]

    def test_budget_reads_the_stored_token_counts(self, monkeypatch):
        # The stored count decides, not a recount: "court" claims 90 tokens,
        # so with a 100-token budget the second chunk no longer fits.
        def boom(text):
            raise AssertionError("stored counts must not be recounted")

        monkeypatch.setattr("src.utils.kb_utils.count_tokens", boom)
        pool = _pool([("court", 0.88), ("suite", 0.86)], token_counts={0: 90, 1: 20})
        llm = SimpleNamespace(kb_id=7)
        with patch("src.utils.kb_utils.search_kb_chunks_scored", return_value=pool):
            excerpts = retrieve_kb_excerpts("question", llm.kb_id, token_budget=100)
        assert [e.text for e in excerpts] == ["court"]

    def test_empty_pool_returns_empty(self):
        llm = SimpleNamespace(kb_id=7)
        with patch("src.utils.kb_utils.search_kb_chunks_scored", return_value=[]):
            assert retrieve_kb_excerpts("question", llm.kb_id, token_budget=500) == []

    def test_llm_without_kb_raises(self):
        llm = SimpleNamespace(kb_id=None)
        with pytest.raises(KnowledgeBaseNotFoundException):
            retrieve_kb_excerpts("question", llm.kb_id, token_budget=500)

    # --- Recall floor (issue #221) ---

    def test_recall_floor_reextends_below_k_min(self):
        # An outlier top gap keeps only hit 1, but hit 2 is within the 0.05
        # recall band, so the floor re-extends to K_MIN_EXCERPTS and stops (it
        # does not pull the whole tail back in).
        pool = _pool(
            [
                ("chunk A", 0.90),
                ("chunk B", 0.87),
                ("chunk C", 0.869),
                ("chunk D", 0.868),
            ]
        )
        llm = SimpleNamespace(kb_id=7)
        with patch("src.utils.kb_utils.search_kb_chunks_scored", return_value=pool):
            excerpts = retrieve_kb_excerpts("question", llm.kb_id, token_budget=2000)
        assert [e.text for e in excerpts] == ["chunk A", "chunk B"]

    def test_recall_floor_respects_the_band(self):
        # The floor never pulls in off-topic tail: hit 2 sits at top-0.10, past
        # the 0.05 band, so the pool legitimately stays at a single excerpt.
        pool = _pool(
            [
                ("chunk A", 0.90),
                ("chunk B", 0.80),
                ("chunk C", 0.79),
                ("chunk D", 0.78),
            ]
        )
        llm = SimpleNamespace(kb_id=7)
        with patch("src.utils.kb_utils.search_kb_chunks_scored", return_value=pool):
            excerpts = retrieve_kb_excerpts("question", llm.kb_id, token_budget=2000)
        assert [e.text for e in excerpts] == ["chunk A"]

    def test_pathological_field_case_no_longer_collapses_to_one(self):
        # End-to-end guard for the field false-negative (issue #221): the
        # non-outlier top gap yields no cut, so >= 2 excerpts reach the prompt
        # instead of the single wrong one.
        pool = _pool(
            [
                ("evaluation A", 0.8618),
                ("evaluation B", 0.8450),
                ("evaluation C", 0.8420),
                ("planning D", 0.8380),
                ("sujet E", 0.8300),
            ]
        )
        llm = SimpleNamespace(kb_id=7)
        with patch("src.utils.kb_utils.search_kb_chunks_scored", return_value=pool):
            excerpts = retrieve_kb_excerpts("Qui evalue ce PFE ?", llm.kb_id, token_budget=2000)
        assert len(excerpts) >= 2
        assert excerpts[0].text == "evaluation A"
