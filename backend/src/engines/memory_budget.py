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
* **denominator**: the engine family's memory pool, ONLY where a single pool
  makes the accounting honest — unified memory on Apple Silicon, system RAM on
  the CPU engine (``get_flat_hardware_data``). On CUDA the signal is OFF by
  policy: see ``total_memory_bytes``.

The deliberate blind spot — the OS and other processes — is absorbed by the
margin floor the callers compare against (15 % in ``src.agents.runner``).

Any fact that cannot be read answers ``None`` and the signal is simply OFF for
that model (the app's downloader fetches a repo's small aux files next to the
``.gguf``, so most GGUF folders do carry a ``config.json`` — but one without it
runs with the signal off); a number is never guessed.
"""

from __future__ import annotations

import json
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
_TOTALS_CACHE: Dict[type, Optional[int]] = {}


def _positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def _scoped_fact(config: Dict[str, Any], key: str) -> Optional[int]:
    """First positive integer for ``key``: top level, then the VLM containers."""
    scopes = [config]
    for container in _SUB_CONFIG_CONTAINERS:
        sub = config.get(container)
        if isinstance(sub, dict):
            scopes.append(sub)
    for scope in scopes:
        n = _positive_int(scope.get(key))
        if n is not None:
            return n
    return None


def kv_bytes_per_token(config: Any) -> Optional[int]:
    """KV-cache bytes one token costs, from a ``config.json`` dict.

    ``2 x num_hidden_layers x kv_heads x head_dim x 2 bytes``. Defined
    fallbacks only: ``num_key_value_heads`` absent falls back to
    ``num_attention_heads`` (transformers' own default — MHA, not a guess) and
    ``head_dim`` absent derives as ``hidden_size // num_attention_heads`` (the
    architectural definition). Any fact still missing answers ``None``.
    """
    if not isinstance(config, dict):
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


def artifact_bytes(model_path: Path) -> Optional[int]:
    """On-disk size of the loaded artifact: the file itself for a GGUF, the
    recursive file sum for an MLX snapshot directory. ``None`` when unreadable."""
    try:
        path = Path(model_path)
        if path.is_file():
            return path.stat().st_size
        if path.is_dir():
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except OSError as exc:
        logger.info(f"Artifact size unreadable at {model_path}: {type(exc).__name__}: {exc}")
    return None


def total_memory_bytes(engine: Any) -> Optional[int]:
    """The engine family's memory pool, in bytes -- ONLY where the accounting
    is honest: unified memory on Apple Silicon, system RAM on the CPU engine.

    On CUDA the answer is ``None`` by policy, which keeps the memory signal
    OFF there: llama-server can offload part of the layers to the card and
    keep the rest in system RAM, in proportions this process cannot know, so
    a VRAM-only denominator would read a partially offloaded model as
    saturating the card while it runs fine -- and compact every turn forever.
    The window signal still protects those machines, and llama's own fit
    already bounded the KV allocation against the card at load.

    Memoized per engine class -- totals are fixed for the life of the process."""
    if engine is None:
        return None
    if engine in _TOTALS_CACHE:
        return _TOTALS_CACHE[engine]
    total: Optional[int] = None
    try:
        flat = engine.get_flat_hardware_data() or {}
        if flat.get("backend_type") != "cuda":
            gb = flat.get("total_memory_gb")
            if isinstance(gb, (int, float)) and not isinstance(gb, bool) and gb > 0:
                total = int(gb * 1024**3)
    except Exception as exc:
        # Degraded, not failed: the memory signal stays off for this run.
        logger.warning(
            f"Hardware totals unreadable; memory signal disabled: " f"{type(exc).__name__}: {exc}"
        )
    _TOTALS_CACHE[engine] = total
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
