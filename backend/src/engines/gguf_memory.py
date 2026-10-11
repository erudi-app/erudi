"""The llama.cpp window, sized by Erudi at spawn from the GGUF header -- pure.

llama-server allocates its whole KV cache at load, for the window it is given
(``-c``). The CPU engine chooses that window itself instead of leaving it to
llama-server's own fit (which sizes against total RAM on macOS/Linux and
against the free RAM of the moment on Windows):

    budget = LLAMA_RAM_BUDGET_FRACTION x physical RAM - weights - LLAMA_COMPUTE_FIXED_BYTES
    window = min(trained window, budget / (KV bytes per token + LLAMA_COMPUTE_PER_TOKEN_BYTES))

rounded down to ``memory_brake.LLAMA_CTX_PAD_TOKENS`` (256: llama.cpp pads the
window it allocates up to a multiple of it), never under the 4096 floor
(``working_window.MIN_KV_BOUND_TOKENS``) -- a model whose floor does not fit
still loads at the floor: the catalog is never gated on RAM -- and folded with
the memory brake's cap when one is carried (``min(window, cap)``, the cap
rounded down the same way).

The KV bytes per token come from the GGUF's own KV header, read with the
``gguf`` package -- imported lazily, inside :func:`read_shape` only: the macOS
build does not ship it, and the engines must import without it.

    kv = sum over layers of (key_length + value_length) x kv_heads[layer] x 2 B

(f16 cache: Erudi passes no ``-ctk``/``-ctv``). Measured on the pinned build
(b10883, one slot): the ``KV buffer size`` llama-server prints is EXACTLY
``n_ctx x kv`` (Qwen3-0.6B 114,688 B/token, Llama-3.2-1B 32,768 B/token, at
8192 and 32768/65536 tokens). Two shapes the formula does not model:

* MLA (``kv_lora_rank``, DeepSeek lineage) is refused -- the formula would
  over-state it by an order of magnitude -- and llama-server's fit sizes it;
* a sliding window (``sliding_window`` below the trained window, Gemma
  lineage) keeps the full-attention formula, a conservative over-estimate
  (the window is smaller than what would fit), and is flagged
  (``swa_overestimated``): such a window is never reported as memory-bound,
  even when memory does bind it.

A hybrid recurrent architecture declares 0 KV heads on its recurrent layers,
which the per-layer sum counts as 0; their fixed per-sequence state rides in
``LLAMA_COMPUTE_FIXED_BYTES`` and the 25 % of RAM the budget leaves untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union

from src.core.logging import logger
from src.engines.memory_brake import LLAMA_CTX_PAD_TOKENS
from src.engines.memory_budget import _KV_BYTES_PER_VALUE, artifact_bytes
from src.engines.working_window import MIN_KV_BOUND_TOKENS

_MIB = 1024**2

# The share of physical RAM the child may take (weights, KV, compute). It is
# not net of the app's own other processes (backend, embeddings, Postgres,
# Electron).
LLAMA_RAM_BUDGET_FRACTION = 0.75
# The Flash-Attention compute graph reserves one f16 buffer of n_ctx x
# n_ubatch (the KQ mask): with the pinned build's default n_ubatch of 512,
# 512 x 2 B = 1 KiB per context token -- measured exactly on both bench models
# (34.01 -> 58.01 MiB over 24,576 tokens; 72.01 -> 128.01 MiB over 57,344).
# Valid on the FA path only (``flash_attn = auto`` resolves to it on the
# pinned build; the post-load check catches a fallback). Tied to b10883 and its
# n_ubatch default: a llama.cpp bump re-measures it.
LLAMA_COMPUTE_PER_TOKEN_BYTES = 1024
# The compute buffer's intercept, with room for wider models than the bench
# (measured 26 MiB at embedding 1024, 64 MiB at 2048).
LLAMA_COMPUTE_FIXED_BYTES = 256 * _MIB
# What the post-load check predicts for the intercept: the larger MEASURED
# one (the budget's 256 MiB over-provisions on purpose).
LLAMA_COMPUTE_INTERCEPT_MEASURED_BYTES = 64 * _MIB
# The post-load check: the child's anonymous memory above this margin over
# the prediction, plus this slack, is one WARNING.
LLAMA_PREDICTION_MARGIN = 1.1
LLAMA_PREDICTION_SLACK_BYTES = 256 * _MIB


@dataclass(frozen=True)
class GgufShape:
    """The KV-relevant facts of a GGUF's header."""

    architecture: str
    block_count: int
    context_length: int
    # Per layer (a hybrid architecture declares 0 on its recurrent layers).
    head_count_kv: Tuple[int, ...]
    key_length: Tuple[int, ...]
    value_length: Tuple[int, ...]
    sliding_window: Optional[int] = None
    kv_lora_rank: Optional[int] = None

    @property
    def swa_overestimated(self) -> bool:
        """The full-attention formula over-states this sliding-window cache."""
        return self.sliding_window is not None and self.sliding_window < self.context_length


@dataclass(frozen=True)
class SpawnWindow:
    """The window the CPU engine passes as ``-c``, and what it costs."""

    window: int
    trained: int
    # The window without the brake's cap: min(trained, what memory affords).
    static_window: int
    # Memory (not the trained window, not a sliding-window over-estimate) is
    # what limits the window. The brake's cap never sets it.
    memory_bound: bool
    kv_token_bytes: int
    # The child's anonymous memory at ``window`` without the weights: KV and
    # the compute buffer.
    static_anonymous_bytes: int
    # The GGUF (all parts of a split family) and its mmproj, on disk.
    weights_bytes: int


def _count(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _per_layer(value: Any, layers: int) -> Optional[Tuple[int, ...]]:
    """A per-layer list as is, a scalar repeated; None when unusable."""
    if isinstance(value, (list, tuple)):
        values = tuple(_count(item) for item in value)
        if len(values) != layers or any(item is None for item in values):
            return None
        return values
    scalar = _count(value)
    return (scalar,) * layers if scalar is not None else None


def shape_from_fields(fields: Mapping[str, Any]) -> Optional[GgufShape]:
    """The shape from the header's values, keyed without the architecture
    prefix (``block_count``, ``attention.head_count_kv`` as ``head_count_kv``,
    ...). ``None`` when an essential is missing.

    Defined fallbacks only, as llama.cpp applies them: the KV heads default to
    the attention heads (MHA), the key and value lengths to
    ``embedding_length // head_count``."""
    architecture = fields.get("architecture")
    layers = _count(fields.get("block_count"))
    trained = _count(fields.get("context_length"))
    if not isinstance(architecture, str) or not architecture or not layers or not trained:
        return None
    heads = _per_layer(fields.get("head_count"), layers)
    kv_heads = _per_layer(fields.get("head_count_kv"), layers)
    if kv_heads is None:
        kv_heads = heads
    if kv_heads is None:
        return None
    embedding = _count(fields.get("embedding_length"))
    derived = (
        tuple(embedding // h if h else 0 for h in heads)
        if embedding and heads is not None
        else None
    )
    keys = _per_layer(fields.get("key_length"), layers) or derived
    values = _per_layer(fields.get("value_length"), layers) or derived
    if keys is None or values is None:
        return None
    sliding = _count(fields.get("sliding_window"))
    lora = _count(fields.get("kv_lora_rank"))
    return GgufShape(
        architecture=architecture,
        block_count=layers,
        context_length=trained,
        head_count_kv=kv_heads,
        key_length=keys,
        value_length=values,
        sliding_window=sliding or None,
        kv_lora_rank=lora or None,
    )


def kv_bytes_per_token(shape: Optional[GgufShape]) -> Optional[int]:
    """KV-cache bytes one token costs (f16); ``None`` for MLA, which the
    formula cannot model, or a shape whose sum is zero."""
    if shape is None or shape.kv_lora_rank is not None:
        return None
    total = sum(
        (k + v) * heads * _KV_BYTES_PER_VALUE
        for k, v, heads in zip(shape.key_length, shape.value_length, shape.head_count_kv)
    )
    return total or None


def read_shape(gguf_path: Union[str, Path]) -> GgufShape:
    """The shape from the GGUF's KV header (one ``GGUFReader`` pass, the
    memmap closed after it: an open map locks the file on Windows). Raises
    on an unreadable file or a missing ``gguf`` package; ``spawn_window``
    turns that into its one WARNING."""
    from gguf import GGUFReader
    from gguf.constants import Keys

    reader = GGUFReader(str(gguf_path))
    try:

        def value(key: str) -> Any:
            field = reader.fields.get(key)
            return field.contents() if field is not None else None

        arch = value(Keys.General.ARCHITECTURE)
        if isinstance(arch, bytes):
            arch = arch.decode("utf-8", errors="replace")
        fields = {
            "architecture": arch,
            "block_count": value(Keys.LLM.BLOCK_COUNT.format(arch=arch)),
            "context_length": value(Keys.LLM.CONTEXT_LENGTH.format(arch=arch)),
            "embedding_length": value(Keys.LLM.EMBEDDING_LENGTH.format(arch=arch)),
            "head_count": value(Keys.Attention.HEAD_COUNT.format(arch=arch)),
            "head_count_kv": value(Keys.Attention.HEAD_COUNT_KV.format(arch=arch)),
            "key_length": value(Keys.Attention.KEY_LENGTH.format(arch=arch)),
            "value_length": value(Keys.Attention.VALUE_LENGTH.format(arch=arch)),
            "sliding_window": value(Keys.Attention.SLIDING_WINDOW.format(arch=arch)),
            "kv_lora_rank": value(Keys.Attention.KV_LORA_RANK.format(arch=arch)),
        }
    finally:
        mmap = getattr(getattr(reader, "data", None), "_mmap", None)
        if mmap is not None:
            try:
                mmap.close()
            except (BufferError, ValueError):
                # A view still exported: the map closes with the reader.
                pass
    shape = shape_from_fields(fields)
    if shape is None:
        raise ValueError("the KV header misses an essential field")
    return shape


def window_for(
    shape: GgufShape,
    *,
    weights_bytes: int,
    total_ram_bytes: int,
    brake_cap: Optional[int] = None,
) -> Optional[SpawnWindow]:
    """The spawn window of ``shape`` on a machine of ``total_ram_bytes``;
    ``None`` when the KV cannot be modelled (MLA)."""
    kv = kv_bytes_per_token(shape)
    if kv is None:
        return None
    trained = shape.context_length
    budget = (
        int(LLAMA_RAM_BUDGET_FRACTION * total_ram_bytes) - weights_bytes - LLAMA_COMPUTE_FIXED_BYTES
    )
    affordable = max(0, budget) // (kv + LLAMA_COMPUTE_PER_TOKEN_BYTES)
    affordable = affordable // LLAMA_CTX_PAD_TOKENS * LLAMA_CTX_PAD_TOKENS
    static = min(trained, max(MIN_KV_BOUND_TOKENS, affordable))
    window = static
    if brake_cap is not None and brake_cap > 0:
        cap = max(MIN_KV_BOUND_TOKENS, brake_cap // LLAMA_CTX_PAD_TOKENS * LLAMA_CTX_PAD_TOKENS)
        window = min(static, cap)
    return SpawnWindow(
        window=window,
        trained=trained,
        static_window=static,
        memory_bound=affordable < trained and not shape.swa_overestimated,
        kv_token_bytes=kv,
        static_anonymous_bytes=(
            window * kv + LLAMA_COMPUTE_FIXED_BYTES + window * LLAMA_COMPUTE_PER_TOKEN_BYTES
        ),
        weights_bytes=weights_bytes,
    )


def spawn_window(
    gguf_path: Union[str, Path],
    *,
    total_ram_bytes: Optional[int],
    brake_cap: Optional[int] = None,
    mmproj_path: Optional[Path] = None,
) -> Optional[SpawnWindow]:
    """The window for ``gguf_path`` (its mmproj, when the engine loads one,
    counts in the weights). ``None`` -- one WARNING -- when it cannot be
    computed: the engine then passes no ``-c`` and llama-server's own fit
    sizes the window. Never raises."""
    if not total_ram_bytes:
        logger.warning("Physical RAM unreadable; llama-server sizes the window itself")
        return None
    try:
        shape = read_shape(gguf_path)
    except Exception as exc:
        # Degraded to llama-server's own fit: the model still loads.
        logger.warning(
            f"GGUF shape unreadable at {gguf_path} ({type(exc).__name__}); "
            f"llama-server sizes the window itself"
        )
        return None
    weights = artifact_bytes(Path(gguf_path))
    if weights is None:
        logger.warning(f"GGUF size unreadable at {gguf_path}; llama-server sizes the window itself")
        return None
    if mmproj_path is not None:
        weights += artifact_bytes(mmproj_path) or 0
    window = window_for(
        shape, weights_bytes=weights, total_ram_bytes=total_ram_bytes, brake_cap=brake_cap
    )
    if window is None:
        reason = "MLA" if shape.kv_lora_rank is not None else "no attention KV cache"
        logger.warning(
            f"KV cache of {gguf_path} not modelled (architecture {shape.architecture}, "
            f"{reason}); llama-server sizes the window itself"
        )
    return window


def post_load_limit_bytes(*, window: int, kv_token_bytes: int, weights_bytes: int) -> int:
    """The anonymous memory a freshly loaded child may hold before the
    post-load check warns: KV, the compute buffer at its MEASURED intercept,
    and one copy of the weights -- the CPU build repacks the weights of
    Erudi's preferred quants into anonymous memory (measured: 603.90 MiB for a
    601 MiB Q8_0, 762.55 MiB for a 763 MiB Q4_K_M) -- times the margin, plus
    the slack."""
    prediction = (
        window * kv_token_bytes
        + window * LLAMA_COMPUTE_PER_TOKEN_BYTES
        + LLAMA_COMPUTE_INTERCEPT_MEASURED_BYTES
        + weights_bytes
    )
    return int(LLAMA_PREDICTION_MARGIN * prediction) + LLAMA_PREDICTION_SLACK_BYTES
