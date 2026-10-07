"""Memory accounting behind the compaction ceiling and the amber warning.

The compaction ceiling and the amber memory warning both need to know how much
memory a conversation of N tokens costs on THIS machine. ``psutil``'s
``available`` is deliberately NOT used: on macOS, compression and swap make it
swing with whatever else the machine is doing. Instead the accounting is a
MEASURED, structural prior over facts that do not move during a chat:

* **base**: the inference child's physical footprint right after its readiness
  probe, measured at every spawn (``src.engines.process_footprint``; fallback:
  the artifact's on-disk size + ``BASE_FALLBACK_OVERHEAD_BYTES``);
* **kv**: KV-cache bytes per token from the model's own ``config.json``,
  ``2 (K and V) x layers x kv_heads x head_dim x 2 bytes (f16)``;
* **vocab** and the child's **prefill step**: the last prefill chunk computes
  full-sequence logits, ``step x vocab x 2 bytes`` at their largest;
* **total**: the GPU's usable working set on Apple Silicon --
  ``MEMORY_SIGNAL_SAFETY_FRACTION`` of the engine's
  ``max_recommended_working_set_bytes`` (Metal's recommended working set, well
  below total RAM). It is not net of the app's other processes (Electron, the
  backend and its embedding model, Postgres).

The prediction above base for a conversation of N tokens is

    predict(N) = m * (t0 + step * vocab * 2 B + c0 * kv * N)

with named priors measured on one M4 16 GB (cold prefills, three models up to
4B): ``c0`` (the marginal cost in KV units: the KV cache, its prefix-cache
copy, MLX's buffer cache), ``t0`` (the bounded prefill transient) and ``m``
(margin over the rep-to-rep noise). They are PROVISIONAL: a measurement
campaign confirms or replaces them before release. Nothing is learned at
runtime; every turn's real peak is recorded next to its prediction
(``src.engines.memory_observations``) and a peak above the prediction is one
WARNING. Requests that carry images are outside the predictor's scope (a
vision tower's activations do not scale with kv).

The ceiling the compaction and the output budget plan against is the token
count at which ``base + predict(N)`` reaches the total, minus the output
budget's 512-token floor (a turn can exceed the window by up to that much).
No margin is stacked on top: 15 % is the amber warning's threshold only.

The accounting is **MLX-only**. On both llama.cpp engines (CPU and CUDA) the KV
cache is allocated in full at load: memory use does not grow with the
conversation, and the engine's own fit already guaranteed the allocation fits,
so there is nothing per-token to measure -- every fact is ``None`` there, and
nothing is written on their handle (``total_memory_bytes`` also names the
partial-offload reason on discrete cards).

Any fact that cannot be read answers ``None`` and the ceiling and the warning
are simply OFF for that model -- notably a KV shape the formula cannot model
(sliding window, MLA: ``_formula_cannot_model``); a number is never guessed.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.logging import logger

_GIB = 1024**3

# --- The measured prior (PROVISIONAL) ---------------------------------------
#
# Fitted on ONE machine (M4 16 GB, mlx 0.32.2 / mlx-vlm 0.6.17, prefill steps
# 2048 and 512, default buffer cache): with these values all 107 points of the
# campaign stay under ``predict`` (minimum ratio 1.20, minimum slack 0.35 GiB;
# ``tests/fixtures/memory_prior_bench5.json``). The gate campaign of plan 3.2b
# Part II confirms or replaces them before release.
#
# t0 and c0 belong to a runtime configuration of the child (its prefill step
# and buffer-cache limit): ``MEMORY_PRIORS`` is keyed by it, and a child whose
# configuration has no measured prior gets no prediction at all.
PRIOR_FIXED_BYTES = int(1.0 * _GIB)  # t0: the bounded prefill transient
PRIOR_KV_MULTIPLIER = 3.5  # c0: marginal cost per token, in KV units
PRIOR_MARGIN = 1.10  # m: over the 8-11 % rep-to-rep noise
# Base when the child's footprint could not be measured at spawn: the weights
# on disk plus the runtime's own resident memory.
BASE_FALLBACK_OVERHEAD_BYTES = int(0.45 * _GIB)
# The last prefill chunk's full-sequence logits: step x vocab, f16.
LOGITS_BYTES_PER_VALUE = 2
# The child's prefill step: Erudi does not pass ``--prefill-step-size``, so
# mlx_vlm's own default applies (``DEFAULT_PREFILL_STEP_SIZE``, 0.6.17).
MLX_PREFILL_STEP_TOKENS = 2048
# The child's MLX buffer-cache limit: not set by Erudi (MLX's default).
MLX_BUFFER_CACHE_LIMIT = "default"
# The output budget's floor (``src.agents.output_budget``): a turn can run
# past the window by up to this much, so the ceiling reserves it.
_OUTPUT_FLOOR_TOKENS = 512


@dataclass(frozen=True)
class MemoryPrior:
    """t0 and c0 of one runtime configuration of the child."""

    fixed_bytes: int
    kv_multiplier: float


MEMORY_PRIORS: Dict[tuple, MemoryPrior] = {
    (MLX_PREFILL_STEP_TOKENS, MLX_BUFFER_CACHE_LIMIT): MemoryPrior(
        fixed_bytes=PRIOR_FIXED_BYTES, kv_multiplier=PRIOR_KV_MULTIPLIER
    ),
}

# QA/dev seam (``backend/.env.example``): multiplies the prediction the
# exceeded-prediction check compares against, so a live run can force the
# WARNING. Never set in normal use.
PRIOR_SCALE_ENV_VAR = "ERUDI_MEMORY_PRIOR_SCALE"

# f16 KV cache: 2 bytes per stored value, and each token stores a K and a V
# vector per layer. llama-server and mlx_vlm both keep the cache in f16 by
# default (KV quantization is opt-in and not exposed in this release).
_KV_BYTES_PER_VALUE = 2
_KV_TENSORS_PER_TOKEN = 2  # K and V

# The GPU working set is the hard ceiling on Apple Silicon; this reserves ~10%
# of it for the OS, WindowServer and other GPU consumers, so the signal counts
# against what the model can actually claim, not the whole working set.
MEMORY_SIGNAL_SAFETY_FRACTION = 0.9

# Where a VLM keeps its text model's shape facts. Mirrors the containers the
# context-window reader accepts (``generation_hints._CONTEXT_CONTAINERS``), so
# the two offline readers agree on where facts live.
_SUB_CONFIG_CONTAINERS = ("text_config", "language_config", "llm_config")

# Memoized per engine class: hardware totals do not change while the backend
# runs, and ``get_flat_hardware_data`` re-probes the platform on every call.
# Only resolved answers are memoized (a real total, or CUDA's policy None);
# an UNREADABLE total is retried on the next call ([L4]) and its record is
# written once per engine, not per turn.
_TOTALS_CACHE: Dict[type, Optional[int]] = {}
_TOTALS_WARNED: set = set()


def _positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def _scopes(config: Dict[str, Any]) -> list:
    """The dicts a shape fact may live in: top level, then the VLM containers."""
    scopes = [config]
    for container in _SUB_CONFIG_CONTAINERS:
        sub = config.get(container)
        if isinstance(sub, dict):
            scopes.append(sub)
    return scopes


def _scoped_fact(config: Dict[str, Any], key: str) -> Optional[int]:
    """First positive integer for ``key`` across ``_scopes``."""
    for scope in _scopes(config):
        n = _positive_int(scope.get(key))
        if n is not None:
            return n
    return None


def _formula_cannot_model(config: Dict[str, Any]) -> bool:
    """True when the full-attention KV formula would OVER-estimate this shape.

    Two families are detected ([M3]): a positive ``sliding_window`` smaller
    than the trained window (Gemma lineage -- sliding layers cap their cache
    at the window, not at the conversation), unless ``use_sliding_window`` is
    explicitly false (Qwen lineage ships the key disabled) or the window
    slides over nothing (as large as the trained window); and MLA
    (``kv_lora_rank``, DeepSeek lineage -- compressed latents, not per-head
    K/V). An over-estimate (4-7x measured on those shapes) would fire
    compaction far too early and silently amputate context, which is worse
    than no signal -- so the KV fact is refused and only the 80 % window
    signal protects those models.
    """
    if _scoped_fact(config, "kv_lora_rank") is not None:
        return True
    sliding = _scoped_fact(config, "sliding_window")
    if sliding is None:
        return False
    for scope in _scopes(config):
        if scope.get("use_sliding_window") is False:
            return False
    trained_window = _scoped_fact(config, "max_position_embeddings")
    return trained_window is None or sliding < trained_window


def kv_bytes_per_token(config: Any) -> Optional[int]:
    """KV-cache bytes one token costs, from a ``config.json`` dict.

    ``2 x num_hidden_layers x kv_heads x head_dim x 2 bytes``. Defined
    fallbacks only: ``num_key_value_heads`` absent falls back to
    ``num_attention_heads`` (transformers' own default — MHA, not a guess) and
    ``head_dim`` absent derives as ``hidden_size // num_attention_heads`` (the
    architectural definition). Any fact still missing answers ``None``, and so
    does a shape the formula would over-estimate — sliding-window or MLA
    caches, see ``_formula_cannot_model``.
    """
    if not isinstance(config, dict):
        return None
    if _formula_cannot_model(config):
        return None
    layers = _scoped_fact(config, "num_hidden_layers")
    attention_heads = _scoped_fact(config, "num_attention_heads")
    kv_heads = _scoped_fact(config, "num_key_value_heads") or attention_heads
    head_dim = _scoped_fact(config, "head_dim")
    if head_dim is None:
        hidden_size = _scoped_fact(config, "hidden_size")
        if hidden_size is not None and attention_heads is not None:
            head_dim = hidden_size // attention_heads
    if layers is None or kv_heads is None or not head_dim:
        return None
    return _KV_TENSORS_PER_TOKEN * layers * kv_heads * head_dim * _KV_BYTES_PER_VALUE


def model_type_of(config: Any) -> Optional[str]:
    """The architecture a ``config.json`` declares (``model_type``, top level
    first, then the VLM text container), or ``None``."""
    if not isinstance(config, dict):
        return None
    for scope in _scopes(config):
        value = scope.get("model_type")
        if isinstance(value, str) and value:
            return value
    return None


def vocab_size_of(config: Any) -> Optional[int]:
    """The vocabulary size a ``config.json`` declares (top level or the VLM
    text container), or ``None``."""
    if not isinstance(config, dict):
        return None
    return _scoped_fact(config, "vocab_size")


# A split GGUF part: "<stem>-00002-of-00003.gguf". The engine resolves the
# FIRST part; the server maps the whole family.
_GGUF_SPLIT_RE = re.compile(r"^(?P<stem>.+)-\d{5}-of-(?P<total>\d{5})\.gguf$", re.IGNORECASE)


def artifact_bytes(model_path: Path) -> Optional[int]:
    """On-disk size of the loaded artifact: the file for a GGUF (ALL sibling
    parts of a split family, [L3]), the recursive file sum for an MLX snapshot
    directory. ``None`` when unreadable."""
    try:
        path = Path(model_path)
        if path.is_file():
            split = _GGUF_SPLIT_RE.match(path.name)
            if split:
                family = re.compile(
                    re.escape(split.group("stem"))
                    + r"-\d{5}-of-"
                    + re.escape(split.group("total"))
                    + r"\.gguf$",
                    re.IGNORECASE,
                )
                return sum(
                    part.stat().st_size
                    for part in path.parent.iterdir()
                    if part.is_file() and family.match(part.name)
                )
            return path.stat().st_size
        if path.is_dir():
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except OSError as exc:
        logger.info(f"Artifact size unreadable at {model_path}: {type(exc).__name__}: {exc}")
    return None


def total_memory_bytes(engine: Any) -> Optional[int]:
    """The memory pool the signal accounts against, in bytes -- MLX ONLY:
    ``MEMORY_SIGNAL_SAFETY_FRACTION`` of the GPU's usable working set.

    The denominator is NOT total system RAM: on Apple Silicon Metal caps the
    GPU at a recommended working set well below RAM (~74 %), so counting
    against total RAM makes the signal fire far too late (#601). The engine's
    ``max_recommended_working_set_bytes`` reports that working set; the safety
    fraction reserves headroom for the OS and other GPU consumers.

    On both llama.cpp engines (CPU and CUDA) the answer is ``None`` by
    policy, which keeps the memory signal OFF there. The real reason: their
    KV cache is allocated IN FULL at load -- memory use does not grow with
    the conversation at all, and the engine's own fit already guaranteed the
    allocation fits at load time, so a per-conversation-token accounting
    would model a phenomenon that does not exist on those engines. (On a
    discrete card, partial layer offload would additionally split the weights
    between VRAM and system RAM in proportions this process cannot know,
    making any single-pool denominator dishonest.) Only MLX grows its cache
    lazily with usage, so only MLX has something to measure. The engine is
    identified by ``FORMAT_TAG`` -- the cache behaviour is a property of the
    child server family, not of the machine.

    Memoized per engine class -- totals are fixed for the life of the process.
    Only a resolved answer is memoized (a real total, or the policy ``None``):
    an unreadable total is retried on the next call and logged once ([L4])."""
    if engine is None:
        return None
    if engine in _TOTALS_CACHE:
        return _TOTALS_CACHE[engine]
    if getattr(engine, "FORMAT_TAG", None) != "mlx":
        # Policy, not a failure: memoized so nothing ever re-probes.
        _TOTALS_CACHE[engine] = None
        return None
    total: Optional[int] = None
    failure: Optional[str] = None
    try:
        working_set = engine.max_recommended_working_set_bytes()
        if isinstance(working_set, int) and not isinstance(working_set, bool) and working_set > 0:
            total = int(MEMORY_SIGNAL_SAFETY_FRACTION * working_set)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    if total is not None:
        _TOTALS_CACHE[engine] = total
    elif engine not in _TOTALS_WARNED:
        # Degraded, not failed -- and once per engine, since every later call
        # retries ([L4]) and would otherwise repeat this record each turn.
        _TOTALS_WARNED.add(engine)
        logger.warning(
            f"GPU working set unreadable ({failure or 'no readable working set'}); "
            f"memory signal off until it resolves"
        )
    return total


_PRIOR_SCALE_WARNED = False


def prior_scale() -> float:
    """``ERUDI_MEMORY_PRIOR_SCALE`` as a positive float, else 1.0 (an invalid
    value is one WARNING per process, not one per turn)."""
    global _PRIOR_SCALE_WARNED
    raw = os.getenv(PRIOR_SCALE_ENV_VAR)
    if raw is None or not raw.strip():
        return 1.0
    try:
        value = float(raw.strip())
    except ValueError:
        value = 0.0
    if value <= 0:
        if not _PRIOR_SCALE_WARNED:
            _PRIOR_SCALE_WARNED = True
            logger.warning(f"{PRIOR_SCALE_ENV_VAR}={raw!r} is not a positive number; ignoring it")
        return 1.0
    return value


def _static_facts(engine: Any, handle: Dict[str, Any]) -> Dict[str, Any]:
    """The facts of the loaded child that never change while it runs,
    computed once and cached on its handle: weights on disk, KV bytes per
    token, vocabulary, prefill step, raw GPU working set."""
    cached = handle.get("memory_facts")
    if isinstance(cached, dict):
        return cached
    weights: Optional[int] = None
    kv: Optional[int] = None
    vocab: Optional[int] = None
    model_type: Optional[str] = None
    raw_path = handle.get("model_path")
    if raw_path:
        path = Path(raw_path)
        weights = artifact_bytes(path)
        config_path = (path if path.is_dir() else path.parent) / "config.json"
        try:
            if config_path.is_file():
                config = json.loads(config_path.read_text(encoding="utf-8"))
                kv = kv_bytes_per_token(config)
                vocab = vocab_size_of(config)
                model_type = model_type_of(config)
        except (OSError, ValueError) as exc:
            # A corrupt config disables the accounting for this model; the
            # model itself keeps running, so one INFO record is enough.
            logger.info(
                f"config.json unreadable at {config_path}; memory accounting off: "
                f"{type(exc).__name__}: {exc}"
            )
    working_set: Optional[int] = None
    probe = getattr(engine, "max_recommended_working_set_bytes", None)
    if callable(probe):
        try:
            working_set = _positive_int(probe())
        except Exception:
            # Degraded: ``total_memory_bytes`` writes the one record about it.
            working_set = None
    facts = {
        "weights_bytes": weights,
        "kv_token_bytes": kv,
        "vocab_size": vocab,
        "prefill_step": MLX_PREFILL_STEP_TOKENS,
        "working_set_bytes": working_set,
        "model_type": model_type,
    }
    handle["memory_facts"] = facts
    return facts


_DEFAULT_PRIOR = MEMORY_PRIORS[(MLX_PREFILL_STEP_TOKENS, MLX_BUFFER_CACHE_LIMIT)]


@dataclass(frozen=True)
class MemoryBudget:
    """The accounting of one loaded child; any ``None`` disables what needs it.

    ``base_bytes``: the child's footprint after its readiness probe;
    ``total_bytes``: 0.9 x the GPU working set; ``kv_token_bytes``,
    ``vocab_size``, ``prefill_step``: the shape facts ``predict`` needs;
    ``weights_bytes``: the artifact on disk (the base fallback); ``prior``:
    t0 and c0 of the child's runtime configuration (``None``: no measured
    prior, no prediction).
    """

    base_bytes: Optional[int]
    total_bytes: Optional[int]
    kv_token_bytes: Optional[int]
    vocab_size: Optional[int]
    prefill_step: Optional[int]
    weights_bytes: Optional[int] = None
    prior: Optional[MemoryPrior] = _DEFAULT_PRIOR
    margin: float = PRIOR_MARGIN

    @classmethod
    def unaccounted(cls) -> "MemoryBudget":
        """Every fact ``None``: the accounting is off."""
        return cls(
            base_bytes=None,
            total_bytes=None,
            kv_token_bytes=None,
            vocab_size=None,
            prefill_step=None,
            weights_bytes=None,
            prior=None,
        )

    @classmethod
    def from_engine(cls, engine: Any) -> "MemoryBudget":
        """The accounting of the CURRENTLY LOADED child of ``engine``. Never
        raises: an unreadable fact is a ``None`` field."""
        if engine is None or getattr(engine, "FORMAT_TAG", None) != "mlx":
            # llama.cpp: KV allocated in full at load -- nothing to account,
            # and nothing written on the handle.
            return cls.unaccounted()
        return cls.from_handle(engine, getattr(engine, "_model", None))

    @classmethod
    def from_handle(cls, engine: Any, handle: Any) -> "MemoryBudget":
        """The accounting of the child ``handle`` describes (its static facts
        are cached on it at the first call)."""
        if not isinstance(handle, dict):
            return cls.unaccounted()
        facts = _static_facts(engine, handle)
        weights = facts.get("weights_bytes")
        base = _positive_int(handle.get("base_footprint_bytes"))
        if base is None and weights is not None:
            base = weights + BASE_FALLBACK_OVERHEAD_BYTES
        return cls(
            base_bytes=base,
            total_bytes=total_memory_bytes(engine),
            kv_token_bytes=facts.get("kv_token_bytes"),
            vocab_size=facts.get("vocab_size"),
            prefill_step=facts.get("prefill_step"),
            weights_bytes=weights,
            prior=MEMORY_PRIORS.get((facts.get("prefill_step"), MLX_BUFFER_CACHE_LIMIT)),
        )

    def _shape_known(self) -> bool:
        return bool(
            self.prior is not None and self.kv_token_bytes and self.vocab_size and self.prefill_step
        )

    def _fixed_bytes(self) -> float:
        return self.prior.fixed_bytes + self.prefill_step * self.vocab_size * LOGITS_BYTES_PER_VALUE

    def predict(self, conversation_tokens: int) -> Optional[int]:
        """Bytes above base a conversation of that many tokens is predicted to
        peak at: ``m * (t0 + step * vocab * 2 + c0 * kv * N)``."""
        if not self._shape_known():
            return None
        per_token = self.prior.kv_multiplier * self.kv_token_bytes
        return int(self.margin * (self._fixed_bytes() + per_token * max(0, conversation_tokens)))

    def conversation_bytes(self, conversation_tokens: int) -> Optional[int]:
        """What the conversation adds to the loaded child: ``predict(N)``."""
        return self.predict(conversation_tokens)

    def footprint_bytes(self, conversation_tokens: int) -> Optional[int]:
        """The child's predicted peak: ``base + predict(N)`` -- the model and
        the conversation together."""
        predicted = self.predict(conversation_tokens)
        if predicted is None or self.base_bytes is None:
            return None
        return self.base_bytes + predicted

    def used_fraction(self, conversation_tokens: int) -> Optional[float]:
        """``footprint / total``, or ``None`` if unaccountable."""
        footprint = self.footprint_bytes(conversation_tokens)
        if footprint is None or not self.total_bytes:
            return None
        return footprint / self.total_bytes

    def memory_margin_fraction(self, conversation_tokens: int) -> Optional[float]:
        """Fraction of the pool still free under this accounting (may go
        negative when the model plus the conversation exceed it)."""
        used = self.used_fraction(conversation_tokens)
        return None if used is None else 1.0 - used

    def tokens_at_ceiling(self) -> Optional[int]:
        """The conversation token count the ceiling allows:
        ``(H / m - t0 - logits) / (c0 * kv) - 512`` with ``H = total - base``.

        Zero or negative when the fixed part alone already fills the budget;
        callers clamp. ``None`` when the accounting is off.
        """
        if not self._shape_known() or self.base_bytes is None or not self.total_bytes:
            return None
        headroom = self.total_bytes - self.base_bytes
        per_token = self.prior.kv_multiplier * self.kv_token_bytes
        tokens = (headroom / self.margin - self._fixed_bytes()) / per_token
        return int(tokens) - _OUTPUT_FLOOR_TOKENS
