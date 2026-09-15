"""Per-model sampling defaults (#388, closes the #136-A item).

Two halves, deliberately separated:

* **Capture** (network, build/CI time and post-download): read the *facts* a
  model ships about how it likes to be sampled and store them verbatim in
  ``llms.generation_hints`` (JSON, nullable). The sampling values come from a
  cascade that stops at the first stage yielding at least one usable value:

  1. ``generation_config.json`` of the **base** repo;
  2. ``generation_config.json`` of the **quant** repo the catalog links to
     (MLX community quants sometimes ship their own; GGUF repos rarely do);
  3. a **conservative read of the base repo's model card** (README.md, public
     even when the weights are gated): prose lines carrying a recommendation
     cue *and* a ``temperature=0.15``-style pair, never a code block.

  Alongside: the context window, whether the chat template knows
  ``enable_thinking``, the stage that won (``source_stage``) and, for the card,
  the exact sentence matched (``evidence``). Never the resolved result.

  The **context window** has a cascade of its own, because the number that
  matters is the one the loaded artifact declares:

  1. for a GGUF entry (``quant_format="gguf"``), the ``context_length`` of the
     quant repo's GGUF metadata (``model_info(expand=["gguf"])``) -- that is
     ``n_ctx_train``, what llama-server reads out of the file it loads, and it
     disagrees with some base repos' ``config.json`` by a factor of 2 to 4;
  2. else ``config.json``, whose window is spelled ``max_position_embeddings``,
     ``n_positions`` or ``max_sequence_length``, at the top level or inside a
     ``text_config`` / ``language_config`` / ``llm_config`` sub-config.

  ``model_max_length`` (tokenizer_config.json) is **not** a source and must not
  become one: a field audit over the catalog found it garbage (sentinels such as
  ``1000000000000000019884624838656``) on 36 % of the entries carrying it, and
  disagreeing with the real window by factors of 2 to 10 in both directions.

* **Resolution** (pure, read time): ``resolve_sampling_defaults(llm)`` turns the
  stored facts into the defaults a new conversation / arena panel starts from
  and the extra sampling keys the model factory sends. A row without a usable
  value resolves to exactly today's constants with ``source == "none"``, so every
  request body validated by the #129 eval campaign stays byte-identical.
  ``top_k`` / ``min_p`` / ``presence_penalty`` exist only when the captured block
  defines them: they are never sent otherwise, so the llama-server implicit
  defaults (top_k 40 / min_p 0.05) that #129 validated on GGUF are untouched.

There is deliberately no hand-curated table: what the publisher says is what
we seed, and when the publisher says nothing the UI says so.
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from src.core.logging import logger

# ---------------------------------------------------------------------------
# Fallback constants -- the ONE place today's defaults live (#129-validated).
# ---------------------------------------------------------------------------
FALLBACK_TEMPERATURE = 0.2
FALLBACK_TOP_P = 0.95
FALLBACK_MAX_TOKENS = 1024
# Tuned through the #129 eval campaign (see the run journal in the issue): the
# legacy 5-token window was blind to sentence-level cycles (small models looped
# whole list items to the token cap), while 1.2 over a wide window over-penalized
# token reuse and mangled proper nouns from the question. 1.1 over 64 tokens
# kills the loops and leaves precision intact.
FALLBACK_REPETITION_PENALTY = 1.1
FALLBACK_REPETITION_CONTEXT_SIZE = 64
# "No cap": mirrors the Conversation.max_tokens validator upper bound.
UNBOUNDED_CONTEXT_TOKENS = 32768

# Below this, a vendor temperature means "greedy" and is normalised to an exact
# 0.0 on the wire (see guarded_generation_config).
GREEDY_TEMPERATURE_THRESHOLD = 0.05

# Capture stages, also the ``source`` the resolver reports (plus ``none``).
STAGE_BASE_GENERATION_CONFIG = "base_generation_config"
STAGE_QUANT_GENERATION_CONFIG = "quant_generation_config"
STAGE_MODEL_CARD = "model_card"
CAPTURE_STAGES: Tuple[str, ...] = (
    STAGE_BASE_GENERATION_CONFIG,
    STAGE_QUANT_GENERATION_CONFIG,
    STAGE_MODEL_CARD,
)
SOURCE_NONE = "none"

# Keys copied from generation_config.json. Everything else (token ids,
# transformers_version, ...) is noise for sampling.
GENERATION_CONFIG_KEYS: Tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "do_sample",
)
# Sampling keys that are user-facing / always on the wire.
_CORE_KEYS = ("temperature", "top_p", "repetition_penalty")
# Sampling keys sent ONLY when the captured block defines them.
_OPTIONAL_KEYS = ("top_k", "min_p", "presence_penalty")


# ---------------------------------------------------------------------------
# Resolved shape
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SamplingDefaults:
    temperature: float
    top_p: float
    max_tokens: int
    repetition_penalty: float
    repetition_context_size: int
    max_tokens_cap: int
    source: str
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    base_repo: Optional[str] = None
    # The model-card sentence the values were read from (stage 3 only).
    evidence: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def wire_kwargs(self) -> Dict[str, Any]:
        """Extra sampling params for the local server (HF vocabulary; engines
        translate). Optional keys are present ONLY when defined."""
        out: Dict[str, Any] = {
            "repetition_penalty": self.repetition_penalty,
            "repetition_context_size": self.repetition_context_size,
        }
        for key in _OPTIONAL_KEYS:
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return None if math.isnan(f) or math.isinf(f) else f


def _as_int(value: Any) -> Optional[int]:
    f = _as_float(value)
    return None if f is None else int(f)


def guarded_generation_config(block: Any) -> Optional[Dict[str, Any]]:
    """Sanitize a captured generation_config subset into usable sampling values.

    Rules (design #388 section 3.4):
    - ``do_sample: false`` -> the whole block is ignored (the vendor disables
      sampling; our sliders would contradict it).
    - ``top_k == 1`` or ``temperature < GREEDY_TEMPERATURE_THRESHOLD`` -> "vendor
      says greedy": the temperature is sent as EXACTLY ``0.0``. Qwen2.5-VL ships
      ``1e-06``; both inference servers only take the argmax path on ``temp == 0``
      and otherwise divide the logits by it, so a positive epsilon overflows into
      a stream of token 0 ("!!!!") — seen on the 2.0.0 QA pass.
    - temperature clamped to [0, 2]; top_p must be in (0, 1], top_k >= 0, min_p in
      [0, 1], presence_penalty in [-2, 2], repetition_penalty > 0 -- out-of-range
      or non-numeric keys are DROPPED (per-key fall-through), never coerced.
    """
    if not isinstance(block, dict):
        return None
    if block.get("do_sample") is False:
        return None
    out: Dict[str, Any] = {}
    temperature = _as_float(block.get("temperature"))
    if temperature is not None:
        out["temperature"] = min(2.0, max(0.0, temperature))
    top_p = _as_float(block.get("top_p"))
    if top_p is not None and 0.0 < top_p <= 1.0:
        out["top_p"] = top_p
    top_k = _as_int(block.get("top_k"))
    if top_k is not None and top_k >= 0:
        out["top_k"] = top_k
    if (temperature is not None and temperature < GREEDY_TEMPERATURE_THRESHOLD) or top_k == 1:
        out["temperature"] = 0.0
    min_p = _as_float(block.get("min_p"))
    if min_p is not None and 0.0 <= min_p <= 1.0:
        out["min_p"] = min_p
    presence = _as_float(block.get("presence_penalty"))
    if presence is not None and -2.0 <= presence <= 2.0:
        out["presence_penalty"] = presence
    rep = _as_float(block.get("repetition_penalty"))
    if rep is not None and rep > 0.0:
        out["repetition_penalty"] = rep
    return out or None


def _whitelist(block: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(block, dict):
        return None
    return {k: block[k] for k in GENERATION_CONFIG_KEYS if k in block}


def _hints_of(llm: Any) -> Dict[str, Any]:
    hints = getattr(llm, "generation_hints", None)
    return hints if isinstance(hints, dict) else {}


def max_tokens_cap(context_length: Any, engine: Any = None) -> int:
    """``min(model context window, engine context window)``, both optional.

    The engine window is the DECLARED ceiling (``max_context_tokens``): the
    ``ERUDI_CTX`` pin on llama.cpp engines when the user set one, ``None``
    everywhere else -- with no pin the engines resolve their ALLOCATED window
    at load and this cap simply falls back to the model's own window.
    Informational for the UI (the max-tokens field's ceiling, the #136 "2000"
    item) and a soft server-side clamp -- never a rejection.
    """
    if engine is None:
        from src.core import config

        engine = getattr(config, "LLM_Engine", None)
    engine_ctx = None
    probe = getattr(engine, "max_context_tokens", None) if engine is not None else None
    if callable(probe):
        try:
            engine_ctx = probe()
        except Exception:
            engine_ctx = None
    model_ctx = _as_int(context_length)
    cap = UNBOUNDED_CONTEXT_TOKENS
    if model_ctx is not None and model_ctx > 0:
        cap = min(cap, model_ctx)
    if isinstance(engine_ctx, int) and engine_ctx > 0:
        cap = min(cap, engine_ctx)
    return max(1, cap)


def resolve_sampling_defaults(llm: Any, *, engine: Any = None) -> SamplingDefaults:
    """Pure resolver: ``guarded captured generation_config > fallback``.

    Reads ``generation_hints`` off ``llm`` (ORM row, Pydantic model or any duck)
    and never touches the network or the DB. ``source`` is the capture stage
    the values came from, or ``"none"`` when the publisher said nothing usable
    (the neutral #129 constants then apply).
    """
    hints = _hints_of(llm)
    values: Dict[str, Any] = {
        "temperature": FALLBACK_TEMPERATURE,
        "top_p": FALLBACK_TOP_P,
        "max_tokens": FALLBACK_MAX_TOKENS,
        "repetition_penalty": FALLBACK_REPETITION_PENALTY,
        "repetition_context_size": FALLBACK_REPETITION_CONTEXT_SIZE,
    }
    optional: Dict[str, Any] = {}
    source = SOURCE_NONE
    evidence: Optional[str] = None

    captured = guarded_generation_config(hints.get("generation_config"))
    if captured:
        stage = hints.get("source_stage")
        # Rows captured before the cascade (#389 snapshots) carry no stage; the
        # only thing #389 ever read was the base repo's generation_config.json.
        source = stage if stage in CAPTURE_STAGES else STAGE_BASE_GENERATION_CONFIG
        raw_evidence = hints.get("evidence")
        evidence = raw_evidence if isinstance(raw_evidence, str) and raw_evidence else None
        for key in _CORE_KEYS:
            if key in captured:
                values[key] = captured[key]
        for key in _OPTIONAL_KEYS:
            if key in captured:
                optional[key] = captured[key]

    cap = max_tokens_cap(hints.get("context_length"), engine)
    values["max_tokens"] = min(int(values["max_tokens"]), cap)
    return SamplingDefaults(
        temperature=values["temperature"],
        top_p=values["top_p"],
        max_tokens=values["max_tokens"],
        repetition_penalty=values["repetition_penalty"],
        repetition_context_size=values["repetition_context_size"],
        max_tokens_cap=cap,
        source=source,
        top_k=optional.get("top_k"),
        min_p=optional.get("min_p"),
        presence_penalty=optional.get("presence_penalty"),
        base_repo=hints.get("base_repo") if isinstance(hints.get("base_repo"), str) else None,
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Model card extraction (stage 3)
# ---------------------------------------------------------------------------
# A recommendation cue. ``recomm\w*`` also catches "recommond" (Mistral's card
# ships that typo) and "recommended"/"recommendation".
_CUE_RE = re.compile(
    r"\b(?:recomm\w*|suggest\w*|best[\s-]+practices?|we\s+advise|advis(?:e|ed|able))\b",
    re.IGNORECASE,
)
_HEADING_CUE_RE = re.compile(r"best[\s-]+practices?|recommend|suggest", re.IGNORECASE)
# ``temperature=0.15`` / ``top_p: 0.9`` / ``TopP=0.95`` (Qwen3 writes CamelCase).
_PAIR_RE = re.compile(
    r"\b(temperature|top[_\- ]?p|top[_\- ]?k|min[_\- ]?p|presence[_\- ]?penalty"
    r"|repetition[_\- ]?penalty)\b\s*[=:]\s*(-?\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)
_KEY_BY_COMPACT = {
    "temperature": "temperature",
    "topp": "top_p",
    "topk": "top_k",
    "minp": "min_p",
    "presencepenalty": "presence_penalty",
    "repetitionpenalty": "repetition_penalty",
}
_THINKING_RE = re.compile(r"\bthinking\b", re.IGNORECASE)
_NON_THINKING_RE = re.compile(r"\bnon[\s-]?thinking\b|\binstruct\b", re.IGNORECASE)

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")
_BULLET_RE = re.compile(r"^(\s*)(?:[-*+]|\d+[.)])\s+(.*)$")
_BLOCKQUOTE_RE = re.compile(r"^\s*(?:>\s?)+")
_FENCE_RE = re.compile(r"^\s*(?:```|~~~)")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_FRONT_MATTER_RE = re.compile(r"\A---[ \t]*\n.*?\n---[ \t]*\n", re.DOTALL)
_EVIDENCE_MAX_CHARS = 400


def _prose_lines(markdown: str) -> List[str]:
    """The card without HTML comments, YAML front matter and fenced code blocks
    (an unclosed fence swallows the rest: never a number from code)."""
    text = _HTML_COMMENT_RE.sub("", markdown)
    text = _FRONT_MATTER_RE.sub("", text, count=1)
    out: List[str] = []
    in_code = False
    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_code = not in_code
            continue
        if not in_code:
            out.append(line)
    return out


def _pairs_in(text: str) -> Dict[str, Any]:
    plain = text.replace("`", "").replace("**", "")
    found: Dict[str, Any] = {}
    for key, number in _PAIR_RE.findall(plain):
        compact = re.sub(r"[_\- ]", "", key).lower()
        name = _KEY_BY_COMPACT.get(compact)
        if name and name not in found:
            found[name] = float(number)
    return found


def extract_card_recommendations(markdown: str) -> List[Dict[str, Any]]:
    """Candidate sampling recommendations in a model card, in document order.

    Conservative by construction: fenced code and HTML comments are dropped
    first; a candidate is a prose paragraph or list item (a bullet plus its
    indented continuation lines) that carries at least one
    ``<sampling key>=<number>`` pair AND a recommendation cue -- on the item
    itself, on the sentence introducing the list, or on the section heading
    ("Best Practices", "Recommended settings"). Values go through the same
    guards as a generation_config block. Each candidate is
    ``{"line": <exact text>, "values": {...}}``.
    """
    candidates: List[Dict[str, Any]] = []
    heading_cue = False
    intro_cue = False
    current: Optional[Dict[str, Any]] = None

    def flush() -> None:
        nonlocal current, intro_cue
        if current is None:
            return
        text = re.sub(r"\s+", " ", current["text"]).strip()
        has_cue = bool(_CUE_RE.search(text))
        values = guarded_generation_config(_pairs_in(text)) if text else None
        if values and (has_cue or current["cue"]):
            candidates.append({"line": text[:_EVIDENCE_MAX_CHARS], "values": values})
        if current["indent"] < 0:
            intro_cue = has_cue  # a paragraph resets the scope
        elif has_cue:
            intro_cue = True  # "- We suggest ...:" then nested items
        current = None

    for raw in _prose_lines(markdown):
        line = _BLOCKQUOTE_RE.sub("", raw)
        if not line.strip():
            flush()
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            flush()
            heading_cue = bool(_HEADING_CUE_RE.search(heading.group(1)))
            intro_cue = False
            continue
        bullet = _BULLET_RE.match(line)
        if bullet:
            flush()
            current = {
                "text": bullet.group(2).strip(),
                "indent": len(bullet.group(1)),
                "cue": heading_cue or intro_cue,
            }
            continue
        indent = len(line) - len(line.lstrip())
        if current is not None and indent > current["indent"]:
            current["text"] += " " + line.strip()
            continue
        flush()
        current = {"text": line.strip(), "indent": -1, "cue": heading_cue}
    flush()
    return candidates


def select_card_recommendation(
    candidates: List[Dict[str, Any]], supports_thinking: Optional[bool]
) -> Optional[Dict[str, Any]]:
    """The candidate matching the model's mode: a thinking model takes the line
    mentioning "thinking" (not "non-thinking"); otherwise the line mentioning
    "non-thinking" / "instruct"; else the first candidate."""
    if not candidates:
        return None
    if supports_thinking is True:

        def wanted(line: str) -> bool:
            return bool(_THINKING_RE.search(line)) and not _NON_THINKING_RE.search(line)
    else:

        def wanted(line: str) -> bool:
            return bool(_NON_THINKING_RE.search(line))

    for candidate in candidates:
        if wanted(candidate["line"]):
            return candidate
    return candidates[0]


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------
_GENERATION_CONFIG_FILE = "generation_config.json"
_CONFIG_FILE = "config.json"
_TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
_CHAT_TEMPLATE_FILE = "chat_template.jinja"
_README_FILE = "README.md"

# The engine format tag (``BaseEngine.FORMAT_TAG``) whose quant repos carry a
# .gguf file, and with it the Hub's gguf metadata block.
GGUF_FORMAT_TAG = "gguf"

# Context-window shapes read from a config.json, in precedence order: the
# preferred key first, then the aliases; the top level before the sub-configs.
# ``model_max_length`` (tokenizer_config.json) is deliberately NOT one of them
# -- see the module docstring.
_CONTEXT_KEYS: Tuple[str, ...] = (
    "max_position_embeddings",
    "n_positions",
    "max_sequence_length",
)
_CONTEXT_CONTAINERS: Tuple[str, ...] = ("text_config", "language_config", "llm_config")

# Memoized for the life of the process: the ~640 derived catalog rows collapse
# onto ~300 unique bases. Files are cached per (repo, filename) so a base
# shared by several quants is fetched once; the GGUF metadata per quant repo;
# the assembled hints per (base, quant, format) triple; failures are remembered
# too, so a gated family is not re-probed for every quant of it.
_CAPTURE_CACHE: Dict[Tuple[str, Optional[str], Optional[str]], Optional[Dict[str, Any]]] = {}
_FILE_CACHE: Dict[Tuple[str, str], Any] = {}
_GGUF_CONTEXT_CACHE: Dict[str, Optional[int]] = {}
_GATED_REPOS: Set[str] = set()
_MISSING_REPOS: Set[str] = set()


def reset_capture_cache() -> None:
    _CAPTURE_CACHE.clear()
    _FILE_CACHE.clear()
    _GGUF_CONTEXT_CACHE.clear()
    _GATED_REPOS.clear()
    _MISSING_REPOS.clear()


def _chat_template_text(chat_template: Any) -> Optional[str]:
    """``chat_template`` may be a string or a list of ``{name, template}``."""
    if isinstance(chat_template, str):
        return chat_template
    if isinstance(chat_template, list):
        parts = [t.get("template", "") for t in chat_template if isinstance(t, dict)]
        return "\n".join(p for p in parts if isinstance(p, str))
    return None


def _context_length_of(config: Optional[Dict[str, Any]]) -> Optional[int]:
    """The training context window declared by a ``config.json``.

    Shared by the online capture and ``read_local_generation_hints`` (which
    reads a downloaded artifact's own ``config.json``), so every shape added
    here widens both paths.

    Publishers name the window four ways and nest it three ways: the preferred
    ``max_position_embeddings``, else ``n_positions`` (phi-2 lineage) or
    ``max_sequence_length`` (Nemotron lineage); at the top level, else inside a
    ``text_config`` / ``language_config`` / ``llm_config`` sub-config (a VLM
    keeps its text model there -- deepseek-vl declares
    ``language_config.max_position_embeddings``). The preferred key wins over an
    alias, and within one key the top level wins over a sub-config; the first
    positive integer in that order is the answer.
    """
    if not isinstance(config, dict):
        return None
    scopes: List[Dict[str, Any]] = [config]
    for container in _CONTEXT_CONTAINERS:
        sub = config.get(container)
        if isinstance(sub, dict):
            scopes.append(sub)
    for key in _CONTEXT_KEYS:
        for scope in scopes:
            n = _as_int(scope.get(key))
            if n is not None and n > 0:
                return n
    return None


def _gguf_context_length(quant_repo: str, hf_api: Any) -> Optional[int]:
    """The window the GGUF file itself declares (``n_ctx_train``), as the Hub
    exposes it on the quant repo through ``model_info(expand=["gguf"])``.

    Authoritative for a GGUF entry: it is the value llama-server reads when it
    loads that file, and it disagrees with some base repos' ``config.json`` by a
    factor of 2 to 4. Memoized per repo (a catalog build asks once per repo) and
    best-effort: any failure answers ``None`` and the capture goes on with the
    ``config.json`` window.
    """
    if quant_repo in _GGUF_CONTEXT_CACHE:
        return _GGUF_CONTEXT_CACHE[quant_repo]
    window: Optional[int] = None
    try:
        info = hf_api.model_info(repo_id=quant_repo, expand=["gguf"])
        block = getattr(info, "gguf", None)
        raw = (
            block.get("context_length")
            if isinstance(block, dict)
            else getattr(block, "context_length", None)
        )
        n = _as_int(raw)
        if n is not None and n > 0:
            window = n
    except Exception as e:
        # Degraded, not failed: the config.json window still applies.
        logger.info(f"GGUF metadata unreadable for {quant_repo}: {e}")
    _GGUF_CONTEXT_CACHE[quant_repo] = window
    return window


def build_generation_hints(
    *,
    base_repo: Optional[str],
    generation_config: Optional[Dict[str, Any]],
    config: Optional[Dict[str, Any]],
    chat_template: Any,
    captured_at: Optional[str] = None,
    source_stage: Optional[str] = None,
    evidence: Optional[str] = None,
    context_length: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Assemble the stored facts from the (optional) source documents.

    ``generation_config`` is the block of the stage that won (whitelisted here);
    ``source_stage`` names it (defaults to the base repo's file when a block is
    given, ``None`` when none is). ``context_length``, when given, is the window
    a higher-priority source declared (the GGUF metadata) and replaces the one
    ``config`` carries; it is a capture on its own, so a repo whose only readable
    fact is that window still produces hints. Returns ``None`` when nothing at
    all was captured, so the column stays NULL ("no hints") rather than holding
    an empty envelope.
    """
    template_text = _chat_template_text(chat_template)
    if (
        generation_config is None
        and config is None
        and template_text is None
        and context_length is None
    ):
        return None
    hints: Dict[str, Any] = {"base_repo": base_repo}
    if isinstance(generation_config, dict):
        hints["generation_config"] = _whitelist(generation_config)
        if source_stage is None:
            source_stage = STAGE_BASE_GENERATION_CONFIG
    hints["supports_thinking"] = (
        ("enable_thinking" in template_text) if template_text is not None else None
    )
    hints["context_length"] = (
        context_length if context_length is not None else _context_length_of(config)
    )
    hints["captured_at"] = captured_at or date.today().isoformat()
    hints["source_stage"] = source_stage
    hints["evidence"] = evidence
    return hints


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _facts_from_readers(
    read_json: Callable[[str], Optional[Dict[str, Any]]], read_text: Callable[[str], Optional[str]]
) -> Tuple[Optional[Dict[str, Any]], Any]:
    """``(config, chat_template)`` of one repo. ``read_json(name)`` /
    ``read_text(name)`` return ``None`` for a missing file and raise on a
    broken one (the caller decides whether that aborts the capture)."""
    config = read_json(_CONFIG_FILE)
    tokenizer_config = read_json(_TOKENIZER_CONFIG_FILE)
    chat_template: Any = None
    if tokenizer_config is not None:
        chat_template = tokenizer_config.get("chat_template")
    if _chat_template_text(chat_template) is None:
        # transformers >= 4.5x moves the template to a standalone jinja file.
        chat_template = read_text(_CHAT_TEMPLATE_FILE)
    return config, chat_template


class _Hub_Reader:
    """Per-(repo, file) memoized reads through ``hf_api.hf_hub_download``.

    A gated answer marks the repo: its other files are not probed (they are
    gated too), only its README.md still is (public even when the weights are
    gated). A missing repo answers ``None`` for everything. Any other error
    propagates: the repo as a whole is unreadable right now.
    """

    def __init__(self, hf_api: Any) -> None:
        self._hf_api = hf_api

    def _fetch(self, repo: str, filename: str) -> Optional[Path]:
        from huggingface_hub.errors import (
            EntryNotFoundError,
            GatedRepoError,
            RepositoryNotFoundError,
        )

        if repo in _MISSING_REPOS or (repo in _GATED_REPOS and filename != _README_FILE):
            return None
        try:
            return Path(self._hf_api.hf_hub_download(repo_id=repo, filename=filename))
        except EntryNotFoundError:
            return None
        except GatedRepoError:
            _GATED_REPOS.add(repo)
            logger.info(f"Generation hints: {repo} is gated, only its model card is readable")
            return None
        except RepositoryNotFoundError:
            _MISSING_REPOS.add(repo)
            return None

    def _cached(self, repo: str, filename: str, load: Callable[[Path], Any]) -> Any:
        key = (repo, filename)
        if key not in _FILE_CACHE:
            path = self._fetch(repo, filename)
            _FILE_CACHE[key] = load(path) if path is not None else None
        return _FILE_CACHE[key]

    def json(self, repo: str, filename: str) -> Optional[Dict[str, Any]]:
        return self._cached(repo, filename, _load_json)

    def text(self, repo: str, filename: str) -> Optional[str]:
        return self._cached(repo, filename, lambda p: p.read_text(encoding="utf-8"))

    def facts(self, repo: str) -> Tuple[Optional[Dict[str, Any]], Any]:
        return _facts_from_readers(lambda f: self.json(repo, f), lambda f: self.text(repo, f))


def _usable(block: Any) -> Optional[Dict[str, Any]]:
    """The whitelisted block when it carries at least one usable value."""
    subset = _whitelist(block)
    return subset if subset and guarded_generation_config(subset) else None


def _capture_uncached(
    base_repo: str, quant_repo: Optional[str], hf_api: Any, quant_format: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    reader = _Hub_Reader(hf_api)
    stage: Optional[str] = None
    block: Optional[Dict[str, Any]] = None
    evidence: Optional[str] = None

    # Stage 1: the base repo's own generation_config.json.
    block = _usable(reader.json(base_repo, _GENERATION_CONFIG_FILE))
    if block:
        stage = STAGE_BASE_GENERATION_CONFIG

    # Facts (context window, thinking): the base repo, else the quant repo (an
    # MLX quant ships the same config.json / chat template; a gated base hides
    # its own).
    config, chat_template = reader.facts(base_repo)
    if config is None and _chat_template_text(chat_template) is None and quant_repo:
        config, chat_template = reader.facts(quant_repo)

    # Stage 2: the quant repo's generation_config.json.
    if stage is None and quant_repo and quant_repo != base_repo:
        block = _usable(reader.json(quant_repo, _GENERATION_CONFIG_FILE))
        if block:
            stage = STAGE_QUANT_GENERATION_CONFIG

    # Stage 3: the base repo's model card.
    if stage is None:
        card = reader.text(base_repo, _README_FILE)
        if card:
            template_text = _chat_template_text(chat_template)
            supports_thinking = ("enable_thinking" in template_text) if template_text else None
            chosen = select_card_recommendation(
                extract_card_recommendations(card), supports_thinking
            )
            if chosen:
                block, evidence, stage = chosen["values"], chosen["line"], STAGE_MODEL_CARD

    # Context window: for a GGUF entry the .gguf file's own declaration outranks
    # every config.json, because it is what llama-server loads. Only the quant
    # repo holds that file, and only the caller knows which engine side this
    # capture is for.
    gguf_context = None
    if quant_format == GGUF_FORMAT_TAG and quant_repo:
        gguf_context = _gguf_context_length(quant_repo, hf_api)

    return build_generation_hints(
        base_repo=base_repo,
        generation_config=block if stage else None,
        config=config,
        chat_template=chat_template,
        source_stage=stage,
        evidence=evidence,
        context_length=gguf_context,
    )


def capture_generation_hints(
    base_repo: Optional[str],
    hf_api: Any,
    *,
    quant_repo: Optional[str] = None,
    quant_format: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Read a model's sampling facts from HuggingFace through the cascade
    (base generation_config > quant generation_config > base model card), plus
    the context window. Best-effort: ``None`` on any failure (network, malformed
    file, nothing readable at all), never raises.

    ``hf_api`` is the retrying client from ``src.core.config.get_hf_api`` (its
    ``hf_hub_download`` and ``model_info`` pace and retry 429s like the other
    catalog calls).

    ``quant_format`` is the engine format tag of the entry being built
    (``BaseEngine.FORMAT_TAG``: ``gguf`` / ``mlx``). On ``gguf`` the window is
    read from the quant repo's GGUF metadata first; nothing else depends on it.
    """
    if not base_repo or hf_api is None:
        return None
    key = (base_repo, quant_repo or None, quant_format or None)
    if key in _CAPTURE_CACHE:
        return copy.deepcopy(_CAPTURE_CACHE[key])
    try:
        hints = _capture_uncached(base_repo, quant_repo or None, hf_api, quant_format or None)
    except Exception as e:
        logger.warning(f"Generation hints capture failed for {base_repo}: {e}")
        hints = None
    _CAPTURE_CACHE[key] = hints
    return copy.deepcopy(hints)


def read_local_generation_hints(
    model_dir: Path, base_repo: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Offline variant for a downloaded MLX directory (which ships the same
    files as its quant repo, hence ``quant_generation_config`` when its
    generation_config.json is usable). GGUF artifacts carry none of them ->
    ``None``. Never raises."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        return None

    def read_json(filename: str) -> Optional[Dict[str, Any]]:
        path = model_dir / filename
        return _load_json(path) if path.is_file() else None

    def read_text(filename: str) -> Optional[str]:
        path = model_dir / filename
        return path.read_text(encoding="utf-8") if path.is_file() else None

    try:
        block = _usable(read_json(_GENERATION_CONFIG_FILE))
        config, chat_template = _facts_from_readers(read_json, read_text)
        return build_generation_hints(
            base_repo=base_repo,
            generation_config=block,
            config=config,
            chat_template=chat_template,
            source_stage=STAGE_QUANT_GENERATION_CONFIG if block else None,
        )
    except Exception as e:
        logger.warning(f"Local generation hints unreadable in {model_dir}: {e}")
        return None


def resolve_base_repo(repo_id: str, tags: Optional[Iterable[str]]) -> str:
    """The repo whose generation facts apply to ``repo_id``: the first
    ``base_model:<relation>:<target>`` card tag (a quant/finetune inherits its
    base's sampling), else the repo itself."""
    from src.database.catalog_classify import _REL_RE

    for tag in tags or []:
        m = _REL_RE.match(str(tag))
        if m:
            return m.group(2)
    return repo_id
