"""Knowledge Base retrieval — adaptive context selection over hybrid search.

Two-stage SOTA selection (issue #81), replacing the flat ``kb_top_k=1`` that
starved panorama and cross-document questions by construction:

1. ``search_kb_chunks_scored`` fetches a wide hybrid candidate pool (dense
   HNSW + sparse tsvector, RRF order) with each candidate's dense cosine
   similarity.
2. Adaptive cut (Adaptive-k transposed to our stack): keep the candidates
   above the largest similarity drop-off, so factoid questions inject a
   narrow context while panorama questions keep their whole cluster. The
   cut runs on calibrated dense cosines — NEVER on RRF scores, which are
   rank harmonics with no semantic scale. Two guards keep the purely
   relative cut from starving recall (issue #221): a widest gap right after
   the top hit is only honored when it is a true outlier, and a recall
   floor re-extends the pool when the cut collapses it below
   ``K_MIN_EXCERPTS``.
3. Token budget: keep whole chunks best-first (RRF order — primacy beats
   burying, per the lost-in-the-middle literature) within the model-size
   budget from ``get_prompting_strategy``, costing each chunk by the e5
   ``token_count`` stored with it (counted on the fly when missing).

Excerpts carry their ``source_file`` so the KB prompt can attribute each
one ("according to <document>" grounding) — see ``build_kb_system_prompt``.
"""

import statistics
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from src.core.exceptions import KnowledgeBaseNotFoundException
from src.ingestion.chunking import count_tokens
from src.ingestion.vector_store import search_kb_chunks_scored


@dataclass(frozen=True)
class KbExcerpt:
    """One selected KB chunk, ready for prompt injection."""

    source_file: str
    text: str


# Below this, a similarity drop-off is noise, not a cut signal: e5 cosines
# live in a compressed range, so a flat pool falls through to budget-only.
# Field data (issue #221): real pools sit in a ~0.78-0.86 band with head
# gaps of 0.001-0.03, so this 0.01 guard is crossed by ordinary head noise.
MIN_SIMILARITY_GAP = 0.01

# Recall floor (issue #221): the adaptive cut is purely relative with no
# lower bound, so a widest gap right after the top hit collapses the pool to
# a single excerpt even when hits 2-5 are strong in absolute terms. Field
# failure: "Qui evalue ce PFE ?" -> 12 hits, top 0.8618, ONE (wrong) excerpt
# injected -> false "not in the documents". Keep at least this many
# candidates, re-extending from the pool within RECALL_BAND of the top
# similarity; the token budget downstream still caps the total, so the floor
# is near-free (the measured ~0.08 top-to-tail band sizes the 0.05 window).
K_MIN_EXCERPTS = 2
RECALL_BAND = 0.05


def _adaptive_threshold(similarities: Sequence[float]) -> Optional[float]:
    """Similarity at the largest drop-off — candidates below it get cut.

    Returns ``None`` when there is no usable signal (fewer than two
    candidates, or a flat distribution): the caller keeps everything and
    lets the token budget decide alone.

    Position-1 gap distrust (issue #221): a widest gap sitting at index 0
    would cut everything after the top hit. Since e5 cosines live in a
    compressed band where head gaps of 0.001-0.03 are ordinary noise, only
    honor that top cut when the gap is a true outlier vs the rest of the
    distribution (``widest > 3 * median(gaps)``, needing >= 2 gaps to judge
    against); otherwise return ``None`` so the token budget decides. This
    preserves the legitimate one-ultra-relevant-doc-vs-off-topic-rest case
    while killing the keep-1 field failure.
    """
    if len(similarities) < 2:
        return None
    ordered = sorted(similarities, reverse=True)
    gaps = [ordered[i] - ordered[i + 1] for i in range(len(ordered) - 1)]
    widest = max(gaps)
    if widest < MIN_SIMILARITY_GAP:
        return None
    cut_index = gaps.index(widest)
    if cut_index == 0 and (len(gaps) < 2 or widest <= 3 * statistics.median(gaps)):
        return None
    return ordered[cut_index]


def _token_cost(text: str, stored: object) -> int:
    """The chunk's stored e5 count, or a count now when it has none.

    Chunks carry ``token_count`` in their metadata since ingestion stores it
    (and the post-ready backfill adds it to older ones); a chunk without a
    usable one is counted on the fly.
    """
    if isinstance(stored, int) and not isinstance(stored, bool) and stored >= 0:
        return stored
    return count_tokens(text)


def _truncate_to_token_budget(chunks: Sequence[Tuple[str, object]], token_budget: int) -> int:
    """How many ``(text, stored token_count)`` chunks fit, best-first, within
    ``token_budget`` e5 tokens; whole chunks only.

    The first chunk always survives — one oversized chunk must yield a big
    context, never an empty one. Only the chunks reached are costed.
    """
    kept = 0
    spent = 0
    for text, stored in chunks:
        cost = _token_cost(text, stored)
        if kept and spent + cost > token_budget:
            break
        kept += 1
        spent += cost
    return kept


def retrieve_kb_excerpts(
    query: str,
    kb_id: Optional[int],
    token_budget: int,
) -> List[KbExcerpt]:
    """Select the KB context for a query: pool → adaptive cut → token budget.

    Args:
        query: User query to search against the KB.
        kb_id: KnowledgeBase id selecting the corpus (the model's ``kb_id``;
            the agentic tool passes it from its runtime context).
        token_budget: Max context size in e5 tokens (from the model-size
            strategy, ``get_prompting_strategy``).

    Returns:
        Source-attributed excerpts in RRF order (best match first); empty
        when the KB holds no indexed chunks yet.

    Raises:
        KnowledgeBaseNotFoundException: If no ``kb_id`` is provided.
    """
    if not kb_id:
        raise KnowledgeBaseNotFoundException(kb_id)

    pool = search_kb_chunks_scored(query, kb_id=kb_id)
    if not pool:
        return []

    similarities = [sim for _, sim in pool]
    threshold = _adaptive_threshold(similarities)
    if threshold is not None:
        selected = [(doc, sim) for doc, sim in pool if sim >= threshold]
    else:
        selected = list(pool)

    # Recall floor (issue #221): the adaptive cut has no lower bound, so it can
    # collapse the pool to a single excerpt (the field false-negative). When it
    # left fewer than K_MIN_EXCERPTS, re-extend from the ORIGINAL pool in RRF
    # order with candidates still within RECALL_BAND of the top similarity,
    # until the floor is met or the band is exhausted. The band keeps off-topic
    # tail chunks out; the token budget below still caps the total.
    if len(selected) < K_MIN_EXCERPTS:
        floor = max(similarities) - RECALL_BAND
        kept_ids = {id(doc) for doc, _ in selected}
        for doc, sim in pool:
            if len(selected) >= K_MIN_EXCERPTS:
                break
            if id(doc) in kept_ids or sim < floor:
                continue
            selected.append((doc, sim))
            kept_ids.add(id(doc))

    # The budget keeps a prefix (best-first order).
    kept = _truncate_to_token_budget(
        [(doc.page_content, doc.metadata.get("token_count")) for doc, _ in selected],
        token_budget,
    )
    return [
        KbExcerpt(
            source_file=doc.metadata.get("source_file", ""),
            text=doc.page_content,
        )
        for doc, _ in selected[:kept]
    ]
