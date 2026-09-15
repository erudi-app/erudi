"""Deterministic memory accounting behind the compaction memory signal.

The compaction middleware and the amber memory warning both need to know how
close the machine is to memory saturation. ``psutil``'s ``available`` is
deliberately NOT used: on macOS, compression and swap make it swing with
whatever else the machine is doing, so a conversation would trigger (or miss)
compaction non-deterministically. Instead the accounting is derived from facts
that do not move during a chat:

* **weights**: the on-disk size of the loaded artifact (the whole MLX snapshot
  directory, or the selected ``.gguf`` file) — what the child has mapped;
* **KV cache**: per-token cost from the model's own ``config.json``,
  ``2 (K and V) x layers x kv_heads x head_dim x 2 bytes (f16)``, multiplied by
  the conversation's token count by the caller;
* **denominator**: the Apple Silicon unified-memory total
  (``get_flat_hardware_data``) — the signal is **MLX-only**. On both llama.cpp
  engines (CPU and CUDA) the KV cache is allocated in full at load: memory use
  does not grow with the conversation, and the engine's own fit already
  guaranteed the allocation fits, so there is nothing per-token to measure —
  the signal is OFF by policy there (see ``total_memory_bytes``, which also
  names the partial-offload reason on discrete cards).

The deliberate blind spot — the OS and other processes — is absorbed by the
margin floor the callers compare against (15 % in ``src.agents.runner``).

Any fact that cannot be read answers ``None`` and the signal is simply OFF for
that model (the app's downloader fetches a repo's small aux files next to the
``.gguf``, so most GGUF folders do carry a ``config.json`` — but one without it
runs with the signal off); a number is never guessed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.logging import logger

# f16 KV cache: 2 bytes per stored value, and each token stores a K and a V
# vector per layer. llama-server and mlx_vlm both keep the cache in f16 by
# default (KV quantization is opt-in and not exposed in this release).
_KV_BYTES_PER_VALUE = 2
_KV_TENSORS_PER_TOKEN = 2  # K and V

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
    """The memory pool the signal accounts against, in bytes -- MLX ONLY: the
    Apple Silicon unified-memory total.

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
        flat = engine.get_flat_hardware_data() or {}
        gb = flat.get("total_memory_gb")
        if isinstance(gb, (int, float)) and not isinstance(gb, bool) and gb > 0:
            total = int(gb * 1024**3)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    if total is not None:
        _TOTALS_CACHE[engine] = total
    elif engine not in _TOTALS_WARNED:
        # Degraded, not failed -- and once per engine, since every later call
        # retries ([L4]) and would otherwise repeat this record each turn.
        _TOTALS_WARNED.add(engine)
        logger.warning(
            f"Memory total unreadable ({failure or 'no readable total'}); "
            f"memory signal off until it resolves"
        )
    return total


@dataclass(frozen=True)
class MemoryBudget:
    """The three facts of the accounting; any ``None`` disables what needs it."""

    weights_bytes: Optional[int]
    kv_token_bytes: Optional[int]
    total_bytes: Optional[int]

    @classmethod
    def from_engine(cls, engine: Any) -> "MemoryBudget":
        """Derive the budget from the CURRENTLY LOADED child of ``engine``.

        Reads the live handle's ``model_path`` (stamped at spawn by every
        engine family): the artifact size on disk, the ``config.json`` beside
        it (the directory's own for MLX; the aux file the downloader saved
        next to a ``.gguf`` — absent, the KV fact is ``None`` and the signal
        off), and the family's memory total. Never raises: an unreadable fact
        is a ``None`` field.
        """
        handle = getattr(engine, "_model", None)
        raw_path = handle.get("model_path") if isinstance(handle, dict) else None
        weights: Optional[int] = None
        kv: Optional[int] = None
        if raw_path:
            path = Path(raw_path)
            weights = artifact_bytes(path)
            config_path = (path if path.is_dir() else path.parent) / "config.json"
            try:
                if config_path.is_file():
                    kv = kv_bytes_per_token(json.loads(config_path.read_text(encoding="utf-8")))
            except (OSError, ValueError) as exc:
                # A corrupt config disables the signal for this model; the
                # model itself keeps running, so one INFO record is enough.
                logger.info(
                    f"config.json unreadable at {config_path}; memory signal off: "
                    f"{type(exc).__name__}: {exc}"
                )
        return cls(
            weights_bytes=weights,
            kv_token_bytes=kv,
            total_bytes=total_memory_bytes(engine),
        )

    def conversation_bytes(self, conversation_tokens: int) -> Optional[int]:
        """KV bytes the conversation costs at its current token count."""
        if self.kv_token_bytes is None:
            return None
        return self.kv_token_bytes * max(0, conversation_tokens)

    def used_fraction(self, conversation_tokens: int) -> Optional[float]:
        """(weights + conversation KV) / total, or ``None`` if unaccountable."""
        kv = self.conversation_bytes(conversation_tokens)
        if self.weights_bytes is None or kv is None or not self.total_bytes:
            return None
        return (self.weights_bytes + kv) / self.total_bytes

    def memory_margin_fraction(self, conversation_tokens: int) -> Optional[float]:
        """Fraction of the pool still free under this accounting (may go
        negative when the model plus the conversation exceed it)."""
        used = self.used_fraction(conversation_tokens)
        return None if used is None else 1.0 - used

    def tokens_at_margin(self, margin: float) -> Optional[int]:
        """The conversation token count at which the margin reaches ``margin``.

        May be negative when the weights alone already blow the floor; callers
        clamp. ``None`` when the accounting is off.
        """
        if self.weights_bytes is None or not self.kv_token_bytes or not self.total_bytes:
            return None
        budget = (1.0 - margin) * self.total_bytes - self.weights_bytes
        return int(budget // self.kv_token_bytes)
