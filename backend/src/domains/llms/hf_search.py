"""Live HuggingFace search for the catalog search box (#122 follow-up).

Lets a user query HF directly from the app, beyond the curated catalog. Searches
only in the active engine's runnable format (``filter=FORMAT_TAG``) and keeps just
chat/vision LLMs (by ``pipeline_tag``), so the results are downloadable and not
polluted with token-classification / ASR / embedding repos that match the text
query. One ``list_models`` call (with ``expand``) finds the hits; then each kept
hit gets its measured download size (#761), one bounded metadata request per hit.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

from src.core import config
from src.core.exceptions import AppBaseException
from src.core.logging import logger
from src.database.catalog_classify import (
    categorize,
    is_derivative,
    param_size_billions,
    VISION_PIPELINES,
)
from src.engines.model_resolver import base_key
from src.database.generation_hints import resolve_base_repo
from src.domains.llms.services import pick_best_gguf
from src.utils.hf_model_metadata import _chosen_artifact_bytes, humanize_model_name

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

# Measured sizes (#761). ``list_models`` cannot report a repo's size (``usedStorage``
# is refused alongside a filter and counts the LFS history anyway), so each kept hit
# costs one ``model_info(files_metadata=True)`` request, and the downloader's own
# file selection is applied to the listing: the size shown is the size installed.
# The requests are bounded so the search answers well inside the client's 30 s
# ceiling (#210): a small pool, no retry, one short budget for the whole batch.
# A hit whose size is not read in time keeps ``artifact_size_bytes = None`` and its
# card keeps the ``~`` estimate. The first 429 stops every request still to send,
# and no search sends a size request for a cooldown after it (the larger of 60 s
# and the reset Hugging Face announced), so a user re-running the search does not
# keep knocking on a Hub that already refuses.
_SIZE_WORKERS = 4
_SIZE_BUDGET_SECONDS = 4.0
_SIZE_CACHE_MAX = 256
_SIZE_CACHE_TTL_SECONDS = 3600.0
_SIZE_COOLDOWN_SECONDS = 60.0

# The clock of the cache expiry and the 429 cooldown (monotonic; tests inject one).
_clock: Callable[[], float] = time.monotonic

# repo id -> (expiry on ``_clock``, measured bytes). Only measured sizes are
# stored: a failure is asked again by the next search.
_size_cache: "OrderedDict[str, tuple[float, int]]" = OrderedDict()
_size_cache_lock = threading.Lock()

# ``_clock`` value before which no size request is sent (set by a 429).
_rate_limited_until = 0.0


def _cache_get(repo_id: str) -> Optional[int]:
    with _size_cache_lock:
        entry = _size_cache.get(repo_id)
        if entry is None:
            return None
        expires, size = entry
        if _clock() >= expires:
            del _size_cache[repo_id]
            return None
        _size_cache.move_to_end(repo_id)
        return size


def _cache_put(repo_id: str, size: int) -> None:
    with _size_cache_lock:
        _size_cache[repo_id] = (_clock() + _SIZE_CACHE_TTL_SECONDS, size)
        _size_cache.move_to_end(repo_id)
        while len(_size_cache) > _SIZE_CACHE_MAX:
            _size_cache.popitem(last=False)


def _cooldown_seconds(err: BaseException) -> float:
    """How long to send no size request after this 429: at least
    ``_SIZE_COOLDOWN_SECONDS``, longer when the response's ``Retry-After`` or
    ``RateLimit`` reset says so. An unreadable header leaves the floor."""
    headers = getattr(getattr(err, "response", None), "headers", None) or {}
    announced = 0.0
    try:
        retry_after = headers.get("Retry-After") or headers.get("retry-after")
        if retry_after:
            announced = float(retry_after)
    except (TypeError, ValueError, AttributeError):
        pass  # an HTTP-date or garbage Retry-After: the floor applies
    try:
        from huggingface_hub.utils import parse_ratelimit_headers

        info = parse_ratelimit_headers(headers)
        if info is not None:
            announced = max(announced, float(info.reset_in_seconds))
    except Exception:
        pass  # a malformed RateLimit header: the floor applies
    return max(_SIZE_COOLDOWN_SECONDS, announced)


def _run_bounded(fn: Callable[[str], Optional[int]], items: List[str], workers: int, budget: float):
    """Run ``fn`` over ``items`` on ``workers`` daemon threads; return what finished
    within ``budget`` seconds as ``{item: result}``.

    Daemon threads, not a ``ThreadPoolExecutor``: the executor's workers are joined
    at interpreter exit, so a request still on the wire when the user quits would
    hold the backend's exit up to its timeout (and trip the launcher's exit
    deadline). A daemon thread is dropped at exit; ``fn`` only reads the Hub and
    fills the in-memory cache, so nothing is lost.
    """
    todo: "queue.SimpleQueue[str]" = queue.SimpleQueue()
    for item in items:
        todo.put(item)
    results: Dict[str, Optional[int]] = {}
    lock = threading.Lock()
    all_done = threading.Event()

    def worker() -> None:
        while True:
            try:
                item = todo.get_nowait()
            except queue.Empty:
                return
            value = fn(item)
            with lock:
                results[item] = value
                if len(results) == len(items):
                    all_done.set()

    for _ in range(min(workers, len(items))):
        threading.Thread(target=worker, name="hf-size", daemon=True).start()
    all_done.wait(budget)
    with lock:
        return dict(results)


def _attach_artifact_sizes(results: List[Dict[str, Any]]) -> None:
    """Set ``artifact_size_bytes`` on every hit, measured or ``None``. Never raises.

    ``model_info`` is called rather than ``repo_info``: ``repo_info`` delegates to
    the client's retrying ``model_info`` override, whose default 429 ladder (five
    retries, up to 30 s each) would outlive the search, so the call passes
    ``_max_retries=0``. Never ``get_disk_size_after_quant``: it falls back to an
    estimate, and an estimate must not be shown as a measured size.

    For a GGUF engine, a repo from which the downloader selects no loadable GGUF
    (only raw byte pieces, only an ``mmproj``) gets ``None``: the download refuses
    it, and ``_chosen_artifact_bytes`` would report the whole repo's sum.

    Writes one INFO line per search (``HF search sizes: enriched=<n>/<hits>
    elapsed_ms=<ms> stopped_on_429=<bool> cooldown=<bool>``; ``cooldown`` is True
    when an earlier 429's cooldown kept every request from going out), a WARNING
    when Hugging Face rate-limited the batch, and one WARNING when requests failed
    otherwise. A gated repo this install has no token for (401/403) is expected
    and logged at DEBUG only.
    """
    started = time.perf_counter()
    for r in results:
        r["artifact_size_bytes"] = _cache_get(r["link"])
    pending = [r["link"] for r in results if r["artifact_size_bytes"] is None]
    in_cooldown = bool(pending) and _clock() < _rate_limited_until
    if in_cooldown:
        pending = []
    stop = threading.Event()
    rate_limited = threading.Event()
    failed: List[str] = []
    uses_gguf = bool(getattr(config.LLM_Engine, "USES_GGUF", False))

    def measure(repo_id: str) -> Optional[int]:
        global _rate_limited_until
        # A request not yet sent when the budget ran out or a 429 came back is
        # never sent.
        if stop.is_set():
            return None
        try:
            info = config.get_hf_api().model_info(
                repo_id,
                files_metadata=True,
                timeout=_SIZE_BUDGET_SECONDS,
                _max_retries=0,
            )
            names = [s.rfilename for s in info.siblings]
            if uses_gguf and pick_best_gguf(names, log=False) is None:
                logger.debug(f"HF search size of {repo_id}: no loadable GGUF selected")
                return None
            size = _chosen_artifact_bytes(info) or None
        except AppBaseException as e:
            # The downloader's selection refused the listing (a llama.cpp split
            # missing a part): a fact about the repo, which the download refuses
            # too, not a failed request.
            logger.debug(f"HF search size of {repo_id} not measurable: {e}")
            return None
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 429:
                rate_limited.set()
                stop.set()
                _rate_limited_until = max(_rate_limited_until, _clock() + _cooldown_seconds(e))
            elif status in (401, 403):
                logger.debug(f"HF search size of {repo_id} not readable ({status})")
            else:
                failed.append(f"{repo_id}: {type(e).__name__}: {e}")
            return None
        if size:
            _cache_put(repo_id, size)
        return size

    if pending:
        measured = _run_bounded(measure, pending, _SIZE_WORKERS, _SIZE_BUDGET_SECONDS)
        # A request already sent when the budget ran out ends on its own timeout
        # in the background; its size still lands in the cache.
        stop.set()
        for r in results:
            if r["link"] in measured:
                r["artifact_size_bytes"] = measured[r["link"]]

    if rate_limited.is_set():
        logger.warning("HF rate-limited (429) the search size requests; remaining sizes skipped")
    if failed:
        logger.warning(f"HF search size requests: {len(failed)} failed, first: {failed[0]}")
    enriched = sum(1 for r in results if r["artifact_size_bytes"] is not None)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        f"HF search sizes: enriched={enriched}/{len(results)} elapsed_ms={elapsed_ms} "
        f"stopped_on_429={rate_limited.is_set()} cooldown={in_cooldown}"
    )


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
    likes, gated, pipeline_tag, quantized, published_at, artifact_size_bytes) — enough for the UI to render and to POST
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
                expand=[
                    "safetensors",
                    "tags",
                    "pipeline_tag",
                    "gated",
                    "downloads",
                    "likes",
                    "createdAt",
                ],
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
        # Only a hit that is itself a base has a publisher date: a quant's upload
        # date would mislead, and resolving its base would cost a Hub call per hit.
        created = getattr(m, "created_at", None) if resolve_base_repo(m.id, tags) == m.id else None
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
                # ISO creation date of a hit that is itself a base, else None (#561).
                "published_at": created.isoformat() if created else None,
            }
        )
        if len(results) >= limit:
            break
    _attach_artifact_sizes(results)
    return results
