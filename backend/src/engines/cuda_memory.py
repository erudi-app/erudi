"""The CUDA window, its placement and the brake's spill signals -- pure.

The CUDA engine sizes the window it passes as ``-c`` for a FULL offload,
from the GGUF header (``gguf_memory``) and the selected GPU's memory, read
through NVML by the engine and passed in here (nothing here imports NVML,
PDH or ctypes, so every rule is table-tested on every CI leg):

    ceiling = min(CUDA_VRAM_BUDGET_FRACTION x VRAM total, VRAM free at load) - CUDA_FREE_MARGIN_BYTES
    budget  = ceiling - weights - CUDA_COMPUTE_FIXED_BYTES - CUDA_CONTEXT_OVERHEAD_BYTES
    window  = min(trained window, budget / (KV bytes per token + compute slope))

rounded down to ``LLAMA_CTX_PAD_TOKENS``, folded with the memory brake's cap.
The free VRAM of the moment keeps llama.cpp's fit from dropping layers on a
crowded card (Windows' desktop alone holds 1-3 GiB): the window may differ
between loads, never between the brake's respawns, which re-read nothing and
reuse the live window folded with the cap (:func:`respawn_window`). When the
floor (4096) does not fit, or the KV cannot be modelled (MLA), there is no
window: the engine passes no ``-c`` and fit decides window and layers.

Division of labour with fit (``common/fit.cpp``, ON by default, b10883):
Erudi fixes the window (an explicit ``-c`` is never changed by fit) and
hands fit its margin (``-fitt CUDA_FIT_TARGET_MIB``); fit fixes the layers
(``-ngl -1``, "auto" while fit is on), measuring the real free memory -- an
optimistic estimate yields a PARTIAL offload instead of a dead child. Under a
full offload the RAM neither binds the window nor drives the brake: the KV and
the compute buffer live on the device, the host keeps clean mapped file pages,
the fixed CUDA runtime and pinned buffers (the CUDA_Host compute buffer, about
1 KiB per token), so a smaller ``-c`` or an unload frees almost no RAM.

The compute slope is the Flash-Attention KQ mask (1 KiB per token at the
default ``n_ubatch`` 512) when FA is PROVEN for the model -- the static mirror
of the CUDA kernel gate of b10883 (``ggml/src/ggml-cuda/fattn.cu``) -- else
the non-FA attention scores, ``n_ubatch x n_head_max x 4 B`` per token.

After load, the engine classifies the placement fit chose. "Partial" means
some KV is not on the device (layers lost): a floor of what a full offload
certainly holds -- the weights without ``token_embd`` (kept on the CPU) and
without the MoE experts (which fit moves to RAM before it drops a layer), plus
the KV the child really allocates (exact for a sliding-window model when its
layer pattern is readable, excluded otherwise) -- against the measured device
bytes, with a one-layer threshold net of the uncounted extras. llama.cpp's own
``offloaded N/M layers to GPU`` line decides instead when it reaches the
drainer.

The brake's only signal is the Windows spill: the child's ``\\GPU Process
Memory`` Shared Usage above its baseline. A healthy child already holds
Shared GPU memory -- its pinned host buffers, a fixed part plus the CUDA_Host
compute buffer that grows with the window (:func:`healthy_shared_bytes`) --
so the baseline the spill is counted from is the child's reading at load
bounded by that healthy level (the level itself when the load reading is
missing), and :func:`spill_signals` turns what lies above the baseline into
the brake's level and damage. Linux has no spill (an allocation that does not fit fails) and so no
brake signal.

Every constant below that is not measured on the pinned build is INFERRED:
QA calibrates it on a Windows CUDA machine and a Linux+NVIDIA one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.engines.gguf_memory import GgufShape, kv_bytes_per_token
from src.engines.memory_brake import (
    LEVEL_CALM,
    LEVEL_CRITICAL,
    LEVEL_WARNING,
    LLAMA_CTX_PAD_TOKENS,
    llama_padded,
)
from src.engines.memory_budget import _KV_BYTES_PER_VALUE
from src.engines.working_window import MIN_KV_BOUND_TOKENS

_MIB = 1024**2

# The share of the GPU's total memory the child may take (decision).
CUDA_VRAM_BUDGET_FRACTION = 0.90
# Room left free on the device for the CUDA pool, which grows DURING
# generation (a failed growth aborts the child mid-turn): fit's own per-device
# margin as the start value (INFERRED; QA measures the growth).
CUDA_FREE_MARGIN_BYTES = 1024 * _MIB
# The per-device margin handed to fit (``-fitt``, MiB): pinned by Erudi rather
# than left to the upstream default.
CUDA_FIT_TARGET_MIB = 1024
# The CUDA primary context, cuBLAS workspaces and pool granularity, invisible
# to llama.cpp's buffer lines (INFERRED).
CUDA_CONTEXT_OVERHEAD_BYTES = 512 * _MIB
# The compute buffer's intercept, as on the CPU engine (INFERRED on CUDA).
CUDA_COMPUTE_FIXED_BYTES = 256 * _MIB
# The Flash-Attention compute slope: one f16 KQ mask of n_ctx x n_ubatch
# (measured exactly on the CPU build; INFERRED on CUDA).
CUDA_COMPUTE_PER_TOKEN_BYTES = 1024
# llama.cpp's default n_ubatch (b10883), which both slopes scale with.
LLAMA_N_UBATCH = 512
# The non-FA attention scores, f32: n_ubatch x n_head x 4 B per token.
_NON_FA_SCORE_BYTES = 4
# b10883 ggml/src/ggml-cuda/fattn.cu:535-542: the head dims the CUDA FA kernels
# dispatch on with equal K and V head sizes (192 and 320 need GQA conditions
# this predicate does not prove). A llama.cpp bump re-reads this set.
CUDA_FA_HEAD_DIMS = frozenset({40, 64, 72, 80, 96, 112, 128, 256})
# The Shared GPU memory of a healthy full-offload child (WDDM's view of its
# pinned host buffers): a fixed part plus the CUDA_Host compute buffer, which
# grows with the window. #661 measured 90 MiB at 8192 tokens (0.5B), 106 at
# 4096 (27B) and 130 at 40960 (4B) -- fixed parts of 78-99 MiB -- so the fixed
# part keeps a margin over the largest.
CUDA_SHARED_FIXED_BYTES = 128 * _MIB
# The CUDA_Host compute buffer per window token (#661, b10883): 11.51 MiB at
# 8192 and 35.51 at 32768 with Flash Attention (1 KiB per token), 20.01 MiB at
# 8192 with ``-fa off`` (about 2 KiB per token).
CUDA_SHARED_PER_TOKEN_FA_BYTES = 1024
CUDA_SHARED_PER_TOKEN_NON_FA_BYTES = 2048
# Spill tiers over the baseline (INFERRED).
CUDA_SPILL_WARNING_BYTES = 256 * _MIB
CUDA_SPILL_CRITICAL_BYTES = 1024 * _MIB
# The placement classification's tolerance.
CUDA_CLASSIFY_TOLERANCE_BYTES = 64 * _MIB
# The post-load device check: above this margin over the prediction, plus the
# slack, is one WARNING.
CUDA_PREDICTION_MARGIN = 1.1
CUDA_PREDICTION_SLACK_BYTES = 256 * _MIB

_OFFLOADED_RE = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")
# llama-context.cpp:546-551 (b10883): what the Flash Attention probe did at
# context creation, "%s: %s enabled" (INFO) or "... not supported, set to
# disabled" (WARN).
_FA_ENABLED = "Flash Attention enabled"
_FA_DISABLED = "Flash Attention not supported, set to disabled"


@dataclass(frozen=True)
class DeviceMemory:
    """One NVML reading of the selected GPU. ``index`` None with no
    ``unavailable`` reason: NVML answered and sees no (visible) GPU."""

    index: Optional[int]
    uuid: Optional[str]
    total_bytes: int
    free_bytes: int
    # ``(pid, usedGpuMemory)`` of the compute processes, None when unreadable
    # (WDDM does not report them).
    processes: Optional[Tuple[Tuple[int, Optional[int]], ...]] = None
    unavailable: Optional[str] = None
    # How many GPUs the child sees: more than one only when the user's own
    # CUDA_VISIBLE_DEVICES names several (llama.cpp then splits the layers).
    visible_devices: int = 1

    @property
    def present(self) -> bool:
        return self.unavailable is None and self.total_bytes > 0


@dataclass(frozen=True)
class VramWindow:
    """The window the CUDA engine passes as ``-c`` and what it costs."""

    window: int
    trained: int
    # The window without the brake's cap.
    static_window: int
    # The VRAM budget, not the trained window nor a sliding-window
    # over-estimate, limits the window. The brake's cap never sets it.
    memory_bound: bool
    kv_token_bytes: int
    weights_bytes: int
    vram_total_bytes: int
    device_budget_bytes: int
    # What the device holds at ``window`` besides the weights: KV, compute,
    # the CUDA context.
    static_device_bytes: int
    fa_proven: bool
    # The KV the child really allocates at ``window`` (the classification's
    # term); None when a sliding-window pattern is unreadable.
    kv_classify_bytes: Optional[int]


def fa_proven(shape: GgufShape) -> bool:
    """Whether the CUDA Flash-Attention kernels of b10883 serve every
    attention layer of ``shape``: equal K and V head sizes in
    ``CUDA_FA_HEAD_DIMS``, and no ALiBi (``max_bias != 0`` drops the fused
    path, fattn.cu:514-519). Recurrent layers (no KV heads) do not count."""
    if shape.kv_lora_rank is not None or shape.max_alibi_bias:
        return False
    layers = [
        (k, v)
        for k, v, heads in zip(shape.key_length, shape.value_length, shape.head_count_kv)
        if heads
    ]
    return bool(layers) and all(k == v and k in CUDA_FA_HEAD_DIMS for k, v in layers)


def compute_token_bytes(shape: GgufShape, *, fa: bool) -> int:
    """The compute buffer's cost per context token."""
    if fa:
        return CUDA_COMPUTE_PER_TOKEN_BYTES
    heads = max(shape.head_count or shape.head_count_kv)
    return LLAMA_N_UBATCH * heads * _NON_FA_SCORE_BYTES


def vram_ceiling_bytes(*, total_bytes: int, free_bytes: int) -> int:
    """What the child may hold on the device: ``min(0.9 x total, free at
    load) - margin`` -- one formula on every OS."""
    return min(int(CUDA_VRAM_BUDGET_FRACTION * total_bytes), free_bytes) - CUDA_FREE_MARGIN_BYTES


def _rounded(tokens: int) -> int:
    return tokens // LLAMA_CTX_PAD_TOKENS * LLAMA_CTX_PAD_TOKENS


def _capped(static: int, cap: Optional[int]) -> int:
    """``static`` folded with a brake cap rounded down to the padding, the
    cap never under the floor."""
    if cap is None or cap <= 0:
        return static
    return min(static, max(MIN_KV_BOUND_TOKENS, _rounded(cap)))


def _window(
    shape: GgufShape,
    *,
    window: int,
    static_window: int,
    memory_bound: bool,
    weights_bytes: int,
    vram_total_bytes: int,
    device_budget_bytes: int,
) -> Optional[VramWindow]:
    kv = kv_bytes_per_token(shape)
    if kv is None:
        return None
    fa = fa_proven(shape)
    slope = compute_token_bytes(shape, fa=fa)
    return VramWindow(
        window=window,
        trained=shape.context_length,
        static_window=static_window,
        memory_bound=memory_bound,
        kv_token_bytes=kv,
        weights_bytes=weights_bytes,
        vram_total_bytes=vram_total_bytes,
        device_budget_bytes=device_budget_bytes,
        static_device_bytes=(
            window * (kv + slope) + CUDA_COMPUTE_FIXED_BYTES + CUDA_CONTEXT_OVERHEAD_BYTES
        ),
        fa_proven=fa,
        kv_classify_bytes=kv_classify_bytes(shape, window),
    )


def window_for_vram(
    shape: GgufShape,
    *,
    weights_bytes: int,
    vram_total_bytes: int,
    vram_free_bytes: int,
    brake_cap: Optional[int] = None,
) -> Optional[VramWindow]:
    """The window of a load; None when the KV cannot be modelled (MLA) or
    the floor -- or the trained window when it is smaller -- does not fit."""
    kv = kv_bytes_per_token(shape)
    if kv is None:
        return None
    budget = (
        vram_ceiling_bytes(total_bytes=vram_total_bytes, free_bytes=vram_free_bytes)
        - weights_bytes
        - CUDA_COMPUTE_FIXED_BYTES
        - CUDA_CONTEXT_OVERHEAD_BYTES
    )
    slope = compute_token_bytes(shape, fa=fa_proven(shape))
    affordable = _rounded(max(0, budget) // (kv + slope))
    trained = shape.context_length
    if affordable < min(trained, MIN_KV_BOUND_TOKENS):
        return None
    static = min(trained, affordable)
    return _window(
        shape,
        window=_capped(static, brake_cap),
        static_window=static,
        memory_bound=affordable < trained and not shape.swa_overestimated,
        weights_bytes=weights_bytes,
        vram_total_bytes=vram_total_bytes,
        device_budget_bytes=budget,
    )


def respawn_window(
    shape: GgufShape,
    *,
    static_window: int,
    cap: Optional[int],
    weights_bytes: int,
    vram_total_bytes: int,
    device_budget_bytes: int,
    memory_bound: bool,
) -> Optional[VramWindow]:
    """The window of a brake respawn: the live child's static window folded
    with the cap, ``min(static, max(floor, cap))`` -- at worst the floor,
    never above the static window (a trained window under the floor stays
    whole) -- with no VRAM reading in, so a series of respawns is
    deterministic. The other facts carry over from the live child."""
    window = _capped(static_window, cap)
    return _window(
        shape,
        window=window,
        static_window=static_window,
        memory_bound=memory_bound,
        weights_bytes=weights_bytes,
        vram_total_bytes=vram_total_bytes,
        device_budget_bytes=device_budget_bytes,
    )


def pinned_window(
    shape: GgufShape,
    *,
    window: int,
    weights_bytes: int,
    vram_total_bytes: int,
    vram_free_bytes: int,
) -> Optional[VramWindow]:
    """The facts of a window the user pinned (``ERUDI_CTX``): llama.cpp
    allocates it padded; never memory-bound."""
    padded = llama_padded(window)
    return _window(
        shape,
        window=padded,
        static_window=padded,
        memory_bound=False,
        weights_bytes=weights_bytes,
        vram_total_bytes=vram_total_bytes,
        device_budget_bytes=(
            vram_ceiling_bytes(total_bytes=vram_total_bytes, free_bytes=vram_free_bytes)
            - weights_bytes
            - CUDA_COMPUTE_FIXED_BYTES
            - CUDA_CONTEXT_OVERHEAD_BYTES
        ),
    )


# ===================== the placement classification =====================


def _swa_candidates(shape: GgufShape) -> Optional[List[Tuple[bool, ...]]]:
    """The per-layer sliding-window flags the header allows: one list for a
    per-layer pattern, both of llama.cpp's conventions for a period
    (``set_swa_pattern``: dense-last, dense-first), None when unreadable."""
    pattern = shape.sliding_window_pattern
    layers = shape.block_count
    if isinstance(pattern, tuple):
        return [pattern]
    if isinstance(pattern, int):
        if pattern == 0:
            return [(True,) * layers]
        return [
            tuple(il % pattern < pattern - 1 for il in range(layers)),
            tuple(il % pattern != 0 for il in range(layers)),
        ]
    return None


def kv_classify_bytes(shape: GgufShape, window: int) -> Optional[int]:
    """The KV a child of ``shape`` allocates at ``window``. Without a sliding
    window: ``window x`` the KV bytes per token. With one and a readable
    layer pattern, the exact iSWA mix (``llama-kv-cache-iswa.cpp:73``, one
    sequence, ``swa_full`` off): a sliding-window layer holds
    ``GGML_PAD(min(window, n_swa + n_ubatch), 256)`` cells, a full one
    ``window`` -- for a period, the convention giving the smaller KV (a
    floor). With an unreadable pattern the layers cannot be told apart:
    None, the KV is left out of the floor."""
    if shape.kv_lora_rank is not None:
        return None
    if shape.sliding_window is None:
        kv = kv_bytes_per_token(shape)
        return window * kv if kv is not None else None
    candidates = _swa_candidates(shape)
    if candidates is None:
        return None
    swa_cells = llama_padded(min(window, shape.sliding_window + LLAMA_N_UBATCH))
    per_layer = [
        (k + v) * heads * _KV_BYTES_PER_VALUE
        for k, v, heads in zip(shape.key_length, shape.value_length, shape.head_count_kv)
    ]
    return min(
        sum(cost * (swa_cells if swa else window) for cost, swa in zip(per_layer, flags))
        for flags in candidates
    )


def predicted_device_floor_bytes(
    *,
    weights_bytes: int,
    tok_embd_bytes: int,
    exps_bytes: int,
    kv_classify_bytes: Optional[int],
) -> int:
    """What a full offload certainly holds on the device: the weights
    without ``token_embd`` and the MoE experts, plus the KV (0 when
    excluded). No compute, CUDA context or pool term: a floor."""
    return weights_bytes - tok_embd_bytes - exps_bytes + (kv_classify_bytes or 0)


def placement_partial(*, predicted_bytes: int, block_count: int, measured_bytes: int) -> bool:
    """At least one layer's worth is missing from the device, net of the
    tolerance: ``measured < predicted - (layer - tolerance)`` with ``layer =
    predicted / (block_count + 1)`` (the output layer counts as one). The
    uncounted extras (compute, CUDA context, pool) inflate ``measured``: a
    thin missing layer can hide behind them, which barely moves the VRAM
    attribution."""
    layer = predicted_bytes // (block_count + 1)
    return measured_bytes < predicted_bytes - (layer - CUDA_CLASSIFY_TOLERANCE_BYTES)


def placement_from_tail(tail: Optional[str]) -> Optional[Tuple[int, int]]:
    """llama.cpp's ``offloaded N/M layers to GPU`` (llama-model.cpp:1817,
    INFO) in the child's output, the last one; None when absent."""
    if not tail:
        return None
    found = _OFFLOADED_RE.findall(tail)
    if not found:
        return None
    offloaded, total = found[-1]
    return int(offloaded), int(total)


def healthy_shared_bytes(window: int, *, fa_on: bool) -> int:
    """The Shared GPU memory a healthy full-offload child of ``window``
    tokens holds on Windows: the fixed part plus the CUDA_Host compute
    buffer, whose slope doubles when Flash Attention is off."""
    slope = CUDA_SHARED_PER_TOKEN_FA_BYTES if fa_on else CUDA_SHARED_PER_TOKEN_NON_FA_BYTES
    return CUDA_SHARED_FIXED_BYTES + window * slope


def windows_device_bytes(*, dedicated_bytes: int, shared_bytes: int, healthy_bytes: int) -> int:
    """A WDDM child's device-side bytes: its dedicated VRAM plus what it
    spilled to shared memory above its healthy level
    (:func:`healthy_shared_bytes`), so a spill at load does not read as a
    partial offload."""
    return dedicated_bytes + max(0, shared_bytes - healthy_bytes)


def flash_attention_in(tail: Optional[str]) -> Optional[bool]:
    """What llama.cpp's Flash Attention probe did at context creation, from
    the child's output: True (its ``enabled`` INFO), False (its ``not
    supported`` WARN), None when neither line is there (an explicit ``-fa``
    prints none, a quiet verbosity may drop the INFO). fit's probe pass
    creates a context too: the last line is the child's."""
    if not tail:
        return None
    enabled, disabled = tail.rfind(_FA_ENABLED), tail.rfind(_FA_DISABLED)
    if enabled == disabled == -1:
        return None
    return enabled > disabled


def device_check_limit_bytes(*, static_device_bytes: int, weights_bytes: int) -> int:
    """The device bytes a freshly loaded full-offload child may hold before
    the post-load check warns."""
    return (
        int(CUDA_PREDICTION_MARGIN * (static_device_bytes + weights_bytes))
        + CUDA_PREDICTION_SLACK_BYTES
    )


def visible_devices(
    devices: Sequence[Dict[str, Any]], visible: Optional[str]
) -> Optional[List[Dict[str, Any]]]:
    """The NVML devices a ``CUDA_VISIBLE_DEVICES`` value lets the child see:
    every device when unset; UUID entries matched by prefix (as CUDA does),
    integer entries as NVML indexes (NVML and CUDA usually agree on the
    order). An empty value is unset (``backend/.env.example`` lists the
    variable empty). None when the value names nothing NVML can map (a MIG
    instance, an unknown id)."""
    entries = [entry.strip() for entry in (visible or "").split(",") if entry.strip()]
    if not entries:
        return list(devices)
    chosen = []
    for device in devices:
        uuid = device.get("uuid") or ""
        for entry in entries:
            if (entry.isdigit() and int(entry) == device.get("id")) or (
                entry.startswith("GPU-") and uuid.startswith(entry)
            ):
                chosen.append(device)
                break
    return chosen or None


# ===================== the brake's signals =====================


def spill_level(spill_bytes: int) -> int:
    """The VRAM level of a spill above the baseline."""
    if spill_bytes < CUDA_SPILL_WARNING_BYTES:
        return LEVEL_CALM
    if spill_bytes < CUDA_SPILL_CRITICAL_BYTES:
        return LEVEL_WARNING
    return LEVEL_CRITICAL


@dataclass(frozen=True)
class SpillSignals:
    """The level and the damage evidence ``memory_brake.llama_step`` sees of
    a full-offload CUDA child: the spill gauge alone."""

    level: int
    damage_bytes: Optional[int]
    damage_available: bool


def spill_signals(spill_above_baseline: int) -> SpillSignals:
    """The spill above the child's baseline as the brake's level and
    damage. A spill at warning or worse is its own damage -- our bytes in
    system RAM paying the PCIe bus on every token. A calm spill is never
    damage: a stable small spill would otherwise count as a paging burst
    at every turn end."""
    level = spill_level(spill_above_baseline)
    warned = level >= LEVEL_WARNING
    return SpillSignals(
        level=level,
        damage_bytes=spill_above_baseline if warned else None,
        damage_available=warned,
    )
