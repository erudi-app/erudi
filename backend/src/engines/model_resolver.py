"""Resolve a base model id to its canonical engine-format quant — no hand table.

Given a base model id (e.g. ``google/gemma-3-1b-it``) and an engine *format tag*
(``"mlx"`` or ``"gguf"``), :func:`resolve_quant` searches Hugging Face for repos
carrying that tag and returns the one whose name, once normalized, is an EXACT
match of the base slug. There is intentionally **no override table**: a base that
has no exact-format quant simply resolves to ``None`` (it won't appear on that
engine), rather than being patched by hand. Gated repos are never candidates
(:func:`is_gated`): the app downloads anonymously, so a gated quant is as good
as no quant.

``normalize`` is the shared heart of the system: it strips a leading
``<vendor>_`` prefix (how bartowski names repos, e.g. ``google_gemma-3-1b-it``)
and any trailing pure-format tokens (``4bit``, ``gguf``, ``mxfp4``, ``q8_0`` …),
so ``mlx-community/gemma-3-1b-it-4bit`` and ``bartowski/google_gemma-3-1b-it-GGUF``
both normalize to ``gemma-3-1b-it``. The same ``normalize`` is reused to dedupe
community finetunes against the base set (a base re-quant normalizes onto a base
slug; a genuine finetune does not).
"""

from __future__ import annotations

import re
from typing import Optional

from src.core.logging import logger

# Trailing tokens that are purely quantization / precision / format markers — they
# are stripped from the tail. Deliberately NOT here: tokens that are part of real
# model slugs (it, instruct, chat, mini, qat, preview, base, pt, vision, …).
FORMAT_TOKENS: frozenset = frozenset(
    {
        "4bit",
        "8bit",
        "6bit",
        "5bit",
        "3bit",
        "2bit",
        "4bits",
        "8bits",
        "16bit",
        "bf16",
        "fp16",
        "fp32",
        "f16",
        "f32",
        "fp8",
        "fp4",
        "mxfp4",
        "mxfp8",
        "nvfp4",
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q8",
        "q2_k",
        "q3_k_m",
        "q4_k_m",
        "q4_k",
        "q4_0",
        "q4_1",
        "q5_k_m",
        "q5_k",
        "q6_k",
        "q8_0",
        "int2",
        "int4",
        "int8",
        "mlx",
        "gguf",
        "ggml",
        "gptq",
        "awq",
        "safetensors",
        "dwq",
        "imatrix",
        "imat",
        "i1",
        "mx",
        "aq4_1",
    }
)

# Vendor prefixes some quanters bake into the repo *name* (owner_model). Matched
# against the base owner first, then this known set (covers cross-org quanters).
ALIAS_OWNERS: frozenset = frozenset(
    {
        "google",
        "qwen",
        "mistralai",
        "nvidia",
        "microsoft",
        "ibm-granite",
        "thudm",
        "cohereforai",
        "coherelabs",
        "meta-llama",
        "deepseek-ai",
        "openai",
        "allenai",
        "huggingfacetb",
        "openbmb",
        "lgai-exaone",
        "nousresearch",
        "tiiuae",
    }
)

_VENDOR_RE = re.compile(r"^([a-z0-9.\-]+)_(.+)$")


def normalize(name: str, owner: str = "") -> str:
    """Canonicalize a repo *name* (last path segment) to a comparable model slug.

    Steps, in order:
      1. lowercase;
      2. drop a leading ``<vendor>_`` if the vendor matches ``owner`` or a known
         quanter alias (so ``google_gemma-3-1b-it`` → ``gemma-3-1b-it``);
      3. unify ``_`` → ``-`` (so ``internlm2_5-20b-chat_8bit`` tokenizes cleanly);
      4. pop trailing pure-format tokens (``-4bit``, ``-gguf``, ``-mxfp4-q8`` …).
    """
    n = name.lower()
    m = _VENDOR_RE.match(n)
    if m and (m.group(1) == owner.lower() or m.group(1) in ALIAS_OWNERS):
        n = m.group(2)
    n = n.replace("_", "-")
    toks = n.split("-")
    while toks and toks[-1] in FORMAT_TOKENS:
        toks.pop()
    return "-".join(toks)


def base_key(base_id: str) -> str:
    """Normalized key of a base model id, used for resolution + dedup comparison."""
    owner, _, slug = base_id.partition("/")
    return normalize(slug or owner, owner)


# Quant preference for picking among several EXACT matches (MLX mostly; for GGUF
# the single best .gguf file is chosen later by pick_best_gguf, so repo choice
# falls back to download count, among the repos that ship a loadable file).
_QUANT_PREF = ("-4bit", "4bit", "-8bit", "8bit", "-6bit", "mxfp4", "bf16", "fp16")

# Quanters we trust to faithfully repackage a base model. Preferred over random
# uploaders when several exact-format quants of the same base exist, so a base's
# display name never binds to an unvetted reupload when a canonical one exists
# (#122). The base's own org is trusted implicitly (an official quant).
TRUSTED_QUANTERS: frozenset = frozenset({"mlx-community", "lmstudio-community"})


def _quant_rank(repo_id: str) -> int:
    low = repo_id.lower()
    for i, marker in enumerate(_QUANT_PREF):
        if marker in low:
            return i
    return len(_QUANT_PREF)


def _trust_rank(repo_id: str, base_owner: str = "") -> int:
    """0 = base's own org (official), 1 = a trusted quanter, 2 = anyone else."""
    owner = repo_id.split("/")[0].lower()
    if base_owner and owner == base_owner.lower():
        return 0
    if owner in TRUSTED_QUANTERS:
        return 1
    return 2


def is_gated(model_info) -> bool:
    """True when the Hub lists the repo as gated (``"auto"`` / ``"manual"`` / True).

    A gated repo needs an accepted license AND a token to download, and Erudi is
    account-less: the app downloads anonymously, so a gated repo lists fine but
    401s on download. It is therefore not runnable, whoever publishes it — the
    official org included. ``gated`` is only serialized when the ``list_models``
    call asks for it (``expand=["gated", ...]``); ``None`` (not requested) reads as
    public, so callers MUST request it to be protected.
    """
    return bool(getattr(model_info, "gated", None))


# ``list_models`` only serializes the fields it is asked for once ``expand`` is set,
# so the two the resolver reads travel together: ``gated`` for the runnability
# gate, ``downloads`` for the tie-breaker.
_RESOLVE_EXPAND = ("gated", "downloads")


def resolve_quant(
    base_id: str, format_tag: str, hf_api, *, limit: int = 40, trace: bool = False
) -> Optional[str]:
    """Return the canonical ``format_tag`` quant repo id for ``base_id``, or None.

    Searches ``list_models(filter=format_tag, search=<slug>)`` and keeps only
    candidates whose normalized name EQUALS the base key and that are NOT gated
    (see :func:`is_gated` — the app has no token, a gated quant is un-downloadable
    however official it is). Among those, prefers a trusted source (official org >
    mlx-community/lmstudio-community > anyone), then the canonical 4-bit quant,
    breaking ties by download count (#122). No public exact match → None: a base
    whose only quant is gated simply is not listed.

    For ``format_tag == "gguf"`` the ranked candidates are then walked in that
    order and the first whose file listing yields a loadable artefact wins (see
    :func:`_gguf_candidate_is_downloadable`); one listing call per candidate
    tried, usually just the first. MLX candidates are not listed.
    """
    owner, _, slug = base_id.partition("/")
    key = base_key(base_id)
    try:
        cands = list(
            hf_api.list_models(
                filter=format_tag,
                search=slug,
                sort="downloads",
                limit=limit,
                expand=list(_RESOLVE_EXPAND),
            )
        )
    except Exception as e:  # network/HF hiccup → treat as "not found", never crash seed
        logger.warning(f"resolve_quant({base_id}, {format_tag}) search failed: {e}")
        return None

    exact = [m for m in cands if normalize(m.id.split("/")[-1], owner) == key]
    gated = [m for m in exact if is_gated(m)]
    if gated:
        logger.info(
            f"[resolve {format_tag}] {base_id}: skipping gated "
            f"{', '.join(m.id for m in gated)} (no token in the app)"
        )
        exact = [m for m in exact if not is_gated(m)]
    if trace:
        logger.info(
            f"[resolve {format_tag}] {base_id} key='{key}' "
            f"{len(cands)} cands, {len(exact)} exact, {len(gated)} gated"
        )
    if not exact:
        return None
    # sorted() is stable, so its head is the element min() would pick.
    ranked = sorted(
        exact,
        key=lambda m: (
            _trust_rank(m.id, owner),
            _quant_rank(m.id),
            -(getattr(m, "downloads", 0) or 0),
        ),
    )
    if format_tag != "gguf":
        return ranked[0].id
    for m in ranked:
        if _gguf_candidate_is_downloadable(m.id, base_id, hf_api):
            return m.id
    logger.info(f"[resolve gguf] {base_id}: no exact candidate ships a loadable GGUF artefact")
    return None


def _gguf_candidate_is_downloadable(repo_id: str, base_id: str, hf_api) -> bool:
    """Whether the downloader would select a loadable artefact from GGUF repo ``repo_id``.

    The name and download count say nothing about the files: a repo can carry the
    ``gguf`` tag while its weights are raw byte chunks no loader opens, or a split
    with a missing part. The file listing is checked through the download path's
    own selection (``has_downloadable_gguf``), so the catalog never binds a base to
    a repo a download would refuse.

    A listing that fails skips the candidate: an unverified repo is never offered,
    the retrying catalog client has already absorbed rate limiting, and the next
    candidate is an exact, public match of the same base. When none remains the
    base is left out of this catalog build, as when the search itself fails.

    Imported lazily: the selection lives in the llms domain, above engines in the
    layering (``utils.hf_model_metadata`` sizes rows through it the same way).
    """
    from src.domains.llms.services import has_downloadable_gguf

    try:
        usable = has_downloadable_gguf(repo_id, hf_api)
    except Exception as e:
        logger.warning(
            f"[resolve gguf] {base_id}: could not list the files of {repo_id}, skipping it: {e}"
        )
        return False
    if not usable:
        logger.info(
            f"[resolve gguf] {base_id}: skipping {repo_id} (no loadable GGUF artefact, "
            f"e.g. weights published as raw byte chunks)"
        )
    return usable
