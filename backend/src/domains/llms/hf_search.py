"""Live HuggingFace search for the catalog search box (#122 follow-up).

Lets a user query HF directly from the app, beyond the curated catalog. Searches
only in the active engine's runnable format (``filter=FORMAT_TAG``) and keeps just
chat/vision LLMs (by ``pipeline_tag``), so the results are downloadable and not
polluted with token-classification / ASR / embedding repos that match the text
query. One ``list_models`` call (with ``expand``) — fast and rate-limit friendly.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.core import config
from src.core.logging import logger
from src.database.catalog_classify import (
    categorize,
    is_derivative,
    param_size_billions,
    VISION_PIPELINES,
)
from src.engines.model_resolver import base_key
from src.utils.hf_model_metadata import humanize_model_name

# Pipelines that correspond to a runnable chat/vision LLM (everything else a text
# query might match — token-classification, ASR, embeddings — is dropped).
_ALLOWED_PIPELINES = frozenset({"text-generation"}) | VISION_PIPELINES

# Floor to drop dead repos; deliberately low since the user asked for this exactly.
_MIN_DOWNLOADS = 10

# Interactive search is user-facing: a human is waiting behind a client-side timeout.
# Keep the 429 retry budget short so a rate-limited call fails fast (worst case
# ~1s + ~2s backoff + round-trips, well under the frontend's 30s abort) instead of
# running the bulk-resync ladder to ~32s and guaranteeing a client timeout (#210).
_SEARCH_MAX_RETRIES = 2
_SEARCH_MAX_BACKOFF = 4.0


def _safetensors_total(model_info) -> Any:
    st = getattr(model_info, "safetensors", None)
    if st is None:
        return None
    total = getattr(st, "total", None)
    if total is None and isinstance(st, dict):
        total = st.get("total")
    return int(total) if total else None


def search_huggingface(query: str, limit: int = 30) -> List[Dict[str, Any]]:
    """Search HF for runnable models matching ``query`` in the active engine format.

    Returns lightweight result dicts (link, name, param_size, category, downloads,
    likes, gated, pipeline_tag, quantized) — enough for the UI to render and to POST
    back to the by-link download endpoint. Returns [] on any failure (never raises
    into the request).
    """
    query = (query or "").strip()
    if not query:
        return []
    engine = getattr(config, "LLM_Engine", None)
    tag = getattr(engine, "FORMAT_TAG", None)
    if not tag:
        return []

    try:
        models = list(
            config.get_hf_api().list_models(
                filter=tag,
                search=query,
                sort="downloads",
                limit=max(limit * 3, 60),
                expand=["safetensors", "tags", "pipeline_tag", "gated", "downloads", "likes"],
                _max_retries=_SEARCH_MAX_RETRIES,
                _max_backoff=_SEARCH_MAX_BACKOFF,
            )
        )
    except Exception as e:
        logger.warning(f"HF search '{query}' failed: {e}")
        return []

    results: List[Dict[str, Any]] = []
    seen: set = set()
    for m in models:
        if (getattr(m, "downloads", 0) or 0) < _MIN_DOWNLOADS:
            continue
        if getattr(m, "pipeline_tag", None) not in _ALLOWED_PIPELINES:
            continue
        tags = list(getattr(m, "tags", None) or [])
        if is_derivative(tags):  # a quant-of-quant: prefer the real source if it surfaces
            pass  # keep it anyway — community quants are legitimately downloadable
        key = base_key(m.id)
        if key in seen:
            continue
        # Runnable by construction (came from filter=FORMAT_TAG); drop KNOWN_BROKEN.
        try:
            if not engine.is_runnable(m.id):
                continue
        except Exception:
            # The runnability check is a KNOWN_BROKEN lookup; if it cannot
            # decide, the hit stays listed and the download gate decides.
            logger.debug(f"is_runnable check failed for {m.id}; keeping the hit", exc_info=True)
        seen.add(key)
        slug = m.id.split("/")[-1]
        results.append(
            {
                "link": m.id,
                "name": humanize_model_name(m.id),
                "param_size": param_size_billions(_safetensors_total(m), slug),
                "category": categorize(slug, tags, getattr(m, "pipeline_tag", None)),
                "downloads": getattr(m, "downloads", 0) or 0,
                "likes": getattr(m, "likes", 0) or 0,
                "gated": bool(getattr(m, "gated", None)),
                "pipeline_tag": getattr(m, "pipeline_tag", None),
                "quantized": True,
            }
        )
        if len(results) >= limit:
            break
    return results
