"""The CUDA window, its placement classification and the spill brake signals
(pure parts).

``src.engines.cuda_memory`` sizes the window the CUDA engine passes as ``-c``
for a FULL offload: ``min(0.9 x VRAM total, VRAM free at load) - margin``,
minus the weights, the compute intercept and the CUDA context, divided by the
per-token cost (KV + the compute slope). It classifies the placement llama.cpp's
fit chose after load (a floor of what is certainly on the device against the
measured bytes), derives a child's healthy Shared GPU memory from its window,
and turns the VRAM spill above it into the inputs of
``memory_brake.llama_step``. Nothing here touches NVML, PDH or a GPU: every
number is passed in, so these tables run on every CI leg.
"""

from __future__ import annotations

import pytest

from src.engines import cuda_memory as cm
from src.engines import memory_brake as mb
from src.engines.gguf_memory import kv_bytes_per_token, shape_from_fields
from src.engines.working_window import MIN_KV_BOUND_TOKENS

pytestmark = pytest.mark.unit

GIB = 1024**3
MIB = 1024**2

QWEN3 = {
    "architecture": "qwen3",
    "block_count": 28,
    "context_length": 40_960,
    "embedding_length": 1024,
    "head_count": 16,
    "head_count_kv": 8,
    "key_length": 128,
    "value_length": 128,
}
KV = 114_688


def _shape(**over):
    return shape_from_fields({**QWEN3, **over})


# ===================== the ceiling =====================


@pytest.mark.parametrize(
    "total, free, expected",
    [
        # Free binds: the desktop holds VRAM.
        (16 * GIB, int(13.8 * GIB), int(13.8 * GIB) - GIB),
        # 0.9 x total binds: an idle card.
        (16 * GIB, 16 * GIB, int(0.9 * 16 * GIB) - GIB),
    ],
)
def test_the_ceiling_is_one_formula_on_every_os(total, free, expected):
    assert cm.vram_ceiling_bytes(total_bytes=total, free_bytes=free) == expected
    assert cm.CUDA_VRAM_BUDGET_FRACTION == 0.90
    assert cm.CUDA_FREE_MARGIN_BYTES == GIB


# ===================== the FA predicate (pinned to b10883) =====================


@pytest.mark.parametrize("head_dim", [40, 64, 72, 80, 96, 112, 128, 256])
def test_fa_is_proven_for_the_b10883_kernel_head_dims(head_dim):
    # b10883 ggml/src/ggml-cuda/fattn.cu:535-542: the head dims the CUDA FA
    # kernels dispatch on with K == V. A llama.cpp bump re-reads this list.
    assert cm.CUDA_FA_HEAD_DIMS == frozenset({40, 64, 72, 80, 96, 112, 128, 256})
    assert cm.fa_proven(_shape(key_length=head_dim, value_length=head_dim)) is True


@pytest.mark.parametrize(
    "over",
    [
        {"key_length": 192, "value_length": 128},  # K != V (fattn.cu:544-551 needs GQA opts)
        {"key_length": 576, "value_length": 512},  # MLA-like dims
        {"key_length": 32, "value_length": 32},  # not a kernel head dim
        {"max_alibi_bias": 8.0},  # ALiBi drops the fused path (fattn.cu:514-519)
        {"key_length": [256, 512], "value_length": [256, 512], "block_count": 2},
    ],
)
def test_fa_is_not_proven_outside_the_kernel_gate(over):
    fields = {**QWEN3, **over}
    if "block_count" in over:
        fields["head_count_kv"] = [8, 8]
    assert cm.fa_proven(shape_from_fields(fields)) is False


def test_recurrent_layers_do_not_disprove_fa():
    shape = _shape(block_count=2, head_count_kv=[8, 0], key_length=[128, 0], value_length=[128, 0])
    assert cm.fa_proven(shape) is True


def test_the_compute_slope_is_the_kq_mask_with_fa_and_the_scores_without():
    shape = _shape()
    assert cm.compute_token_bytes(shape, fa=True) == cm.CUDA_COMPUTE_PER_TOKEN_BYTES == 1024
    # n_ubatch x n_head_max x 4 B: 512 x 16 x 4 = 32 KiB.
    assert cm.compute_token_bytes(shape, fa=False) == 512 * 16 * 4


# ===================== the sized window =====================


def test_worked_example_roomy_card_trained_window_binds():
    """RTX 4060 Ti 16 GB, 13.8 GiB free, a 2.5 GiB model: the budget affords
    88,576 tokens, the trained 40,960 binds."""
    window = cm.window_for_vram(
        _shape(),
        weights_bytes=int(2.5 * GIB),
        vram_total_bytes=16 * GIB,
        vram_free_bytes=int(13.8 * GIB),
    )
    assert window.device_budget_bytes == 10_254_234_419
    assert window.device_budget_bytes // (KV + 1024) // 256 * 256 == 88_576
    assert window.window == window.static_window == window.trained == 40_960
    assert window.memory_bound is False
    assert window.fa_proven is True
    assert window.kv_token_bytes == KV
    assert window.vram_total_bytes == 16 * GIB
    assert window.static_device_bytes == (
        40_960 * (KV + 1024) + cm.CUDA_COMPUTE_FIXED_BYTES + cm.CUDA_CONTEXT_OVERHEAD_BYTES
    )


def test_worked_example_8gb_card_memory_binds():
    window = cm.window_for_vram(
        _shape(),
        weights_bytes=int(2.5 * GIB),
        vram_total_bytes=8 * GIB,
        vram_free_bytes=int(6.5 * GIB),
    )
    assert window.device_budget_bytes == 2_415_919_104
    assert window.window == 20_736
    assert window.window % mb.LLAMA_CTX_PAD_TOKENS == 0
    assert window.memory_bound is True


def test_worked_example_13b_on_8gb_does_not_fit_the_floor():
    assert (
        cm.window_for_vram(
            _shape(),
            weights_bytes=int(7.5 * GIB),
            vram_total_bytes=8 * GIB,
            vram_free_bytes=int(6.5 * GIB),
        )
        is None
    )


def test_the_non_fa_slope_sizes_a_smaller_window():
    fa = cm.window_for_vram(
        _shape(), weights_bytes=GIB, vram_total_bytes=8 * GIB, vram_free_bytes=8 * GIB
    )
    non_fa = cm.window_for_vram(
        _shape(max_alibi_bias=8.0),
        weights_bytes=GIB,
        vram_total_bytes=8 * GIB,
        vram_free_bytes=8 * GIB,
    )
    assert non_fa.fa_proven is False
    assert non_fa.window < fa.window
    assert non_fa.static_device_bytes == (
        non_fa.window * (KV + 512 * 16 * 4)
        + cm.CUDA_COMPUTE_FIXED_BYTES
        + cm.CUDA_CONTEXT_OVERHEAD_BYTES
    )


def test_a_brake_cap_folds_in_rounded_without_making_the_window_memory_bound():
    window = cm.window_for_vram(
        _shape(),
        weights_bytes=GIB,
        vram_total_bytes=16 * GIB,
        vram_free_bytes=16 * GIB,
        brake_cap=20_500,
    )
    assert window.window == 20_480
    assert window.static_window == 40_960
    assert window.memory_bound is False


def test_a_cap_never_goes_under_the_floor():
    window = cm.window_for_vram(
        _shape(),
        weights_bytes=GIB,
        vram_total_bytes=16 * GIB,
        vram_free_bytes=16 * GIB,
        brake_cap=1000,
    )
    assert window.window == MIN_KV_BOUND_TOKENS


def test_mla_is_refused_and_swa_is_never_memory_bound():
    assert (
        cm.window_for_vram(
            _shape(kv_lora_rank=512),
            weights_bytes=GIB,
            vram_total_bytes=16 * GIB,
            vram_free_bytes=16 * GIB,
        )
        is None
    )
    swa = cm.window_for_vram(
        _shape(sliding_window=4096, context_length=131_072),
        weights_bytes=int(2.5 * GIB),
        vram_total_bytes=8 * GIB,
        vram_free_bytes=8 * GIB,
    )
    assert swa.window < 131_072
    assert swa.memory_bound is False


def test_a_trained_window_under_the_floor_is_sized():
    window = cm.window_for_vram(
        _shape(context_length=2048),
        weights_bytes=GIB,
        vram_total_bytes=8 * GIB,
        vram_free_bytes=8 * GIB,
    )
    assert window.window == 2048


# ===================== the brake respawn re-reads nothing =====================


@pytest.mark.parametrize(
    "static, cap, expected",
    [
        (40_960, 30_720, 30_720),
        (20_480, 30_720, 20_480),
        (40_960, 3_000, MIN_KV_BOUND_TOKENS),
        # A trained window under the floor is never exceeded.
        (2_048, MIN_KV_BOUND_TOKENS, 2_048),
    ],
)
def test_a_respawn_window_is_the_static_window_folded_with_the_cap(static, cap, expected):
    kwargs = dict(
        static_window=static,
        cap=cap,
        weights_bytes=GIB,
        vram_total_bytes=16 * GIB,
        device_budget_bytes=4 * GIB,
        memory_bound=True,
    )
    first = cm.respawn_window(_shape(), **kwargs)
    # Deterministic by construction: no free VRAM goes in.
    assert cm.respawn_window(_shape(), **kwargs) == first
    assert first.window == expected
    assert first.static_window == static
    assert first.device_budget_bytes == 4 * GIB
    assert first.memory_bound is True


# ===================== the placement classification =====================


def _floor(*, weights, tok_embd=0, exps=0, kv=0):
    return cm.predicted_device_floor_bytes(
        weights_bytes=weights, tok_embd_bytes=tok_embd, exps_bytes=exps, kv_classify_bytes=kv
    )


def test_the_floor_is_the_weights_without_tok_embd_and_experts_plus_the_kv():
    assert _floor(weights=4 * GIB, tok_embd=300 * MIB, exps=GIB, kv=2 * GIB) == (
        4 * GIB - 300 * MIB - GIB + 2 * GIB
    )
    # A KV excluded from the prediction (unreadable SWA pattern) counts 0.
    assert _floor(weights=4 * GIB, tok_embd=300 * MIB, kv=None) == 4 * GIB - 300 * MIB


def test_a_missing_dense_layer_is_partial_and_the_tolerance_reads_full():
    predicted = _floor(weights=4 * GIB, tok_embd=300 * MIB, kv=40_960 * KV)
    layer = predicted // (28 + 1)
    assert cm.placement_partial(
        predicted_bytes=predicted, block_count=28, measured_bytes=predicted - layer
    )
    # Exactly at the threshold is still full (strict).
    assert not cm.placement_partial(
        predicted_bytes=predicted,
        block_count=28,
        measured_bytes=predicted - (layer - cm.CUDA_CLASSIFY_TOLERANCE_BYTES),
    )
    assert not cm.placement_partial(
        predicted_bytes=predicted,
        block_count=28,
        measured_bytes=predicted - cm.CUDA_CLASSIFY_TOLERANCE_BYTES,
    )
    # The measured value over-reads by the uncounted extras: full.
    assert not cm.placement_partial(
        predicted_bytes=predicted, block_count=28, measured_bytes=predicted + GIB
    )


def _gemma_like(**over):
    return shape_from_fields(
        {
            "architecture": "gemma3",
            "block_count": 12,
            "context_length": 131_072,
            "embedding_length": 2560,
            "head_count": 8,
            "head_count_kv": 4,
            "key_length": 256,
            "value_length": 256,
            "sliding_window": 1024,
            **over,
        }
    )


def test_swa_full_offload_reads_full_with_the_iswa_kv():
    """M-B: a sliding-window child allocates far less KV than the sizing's
    full-attention formula; the classification uses the exact iSWA mix."""
    window = 32_768
    shape = _gemma_like(sliding_window_pattern=6)
    per_layer = 2 * 256 * 4 * 2
    # set_swa_pattern(6): layers 5 and 11 are full attention, the rest SWA;
    # an SWA layer holds GGML_PAD(min(W, n_swa + n_ubatch), 256) = 1536 cells.
    iswa = 2 * per_layer * window + 10 * per_layer * 1536
    assert cm.kv_classify_bytes(shape, window) == iswa
    weights, tok_embd = 3 * GIB, 400 * MIB
    # What the child really holds: weights without tok_embd, the iSWA KV, extras.
    measured = weights - tok_embd + iswa + 600 * MIB
    predicted = _floor(weights=weights, tok_embd=tok_embd, kv=iswa)
    assert not cm.placement_partial(
        predicted_bytes=predicted, block_count=12, measured_bytes=measured
    )
    # The sizing's full-attention KV would have misread it as partial.
    naive = _floor(weights=weights, tok_embd=tok_embd, kv=window * kv_bytes_per_token(shape))
    assert cm.placement_partial(predicted_bytes=naive, block_count=12, measured_bytes=measured)


def test_a_per_layer_swa_pattern_is_read_exactly():
    shape = _gemma_like(block_count=4, sliding_window_pattern=[True, False, True, True])
    per_layer = 2 * 256 * 4 * 2
    assert cm.kv_classify_bytes(shape, 32_768) == per_layer * (32_768 + 3 * 1536)


def test_a_scalar_pattern_takes_the_smaller_of_both_layer_conventions():
    # 12 layers, period 5: dense-last marks 2 full layers (4, 9), dense-first
    # 3 (0, 5, 10): the floor takes 2.
    shape = _gemma_like(sliding_window_pattern=5)
    per_layer = 2 * 256 * 4 * 2
    assert cm.kv_classify_bytes(shape, 32_768) == per_layer * (2 * 32_768 + 10 * 1536)


def test_an_unreadable_swa_pattern_excludes_the_kv_and_still_reads_full():
    shape = _gemma_like()
    assert cm.kv_classify_bytes(shape, 32_768) is None
    weights, tok_embd = 3 * GIB, 400 * MIB
    measured = weights - tok_embd + 300 * MIB
    predicted = _floor(weights=weights, tok_embd=tok_embd, kv=None)
    assert not cm.placement_partial(
        predicted_bytes=predicted, block_count=12, measured_bytes=measured
    )


def test_without_a_sliding_window_the_kv_is_the_full_formula():
    assert cm.kv_classify_bytes(_shape(), 8192) == 8192 * KV
    assert cm.kv_classify_bytes(_shape(kv_lora_rank=512), 8192) is None


def test_moe_with_every_expert_spilled_is_full():
    """M-C: fit moves *_exps tensors to RAM and keeps every layer and its KV
    on the device: not partial, the brake stays on."""
    weights, tok_embd, exps, kv = 18 * GIB, 500 * MIB, 15 * GIB, 2 * GIB
    measured = weights - tok_embd - exps + kv + 400 * MIB
    predicted = _floor(weights=weights, tok_embd=tok_embd, exps=exps, kv=kv)
    assert not cm.placement_partial(
        predicted_bytes=predicted, block_count=48, measured_bytes=measured
    )
    # Counting the experts on the device would have called it partial.
    naive = _floor(weights=weights, tok_embd=tok_embd, kv=kv)
    assert cm.placement_partial(predicted_bytes=naive, block_count=48, measured_bytes=measured)


@pytest.mark.parametrize(
    "tail, expected",
    [
        ("load_tensors: offloaded 29/29 layers to GPU\n", (29, 29)),
        ("x\nload_tensors: offloaded 20/29 layers to GPU\nmore\n", (20, 29)),
        ("no placement line here\n", None),
        ("", None),
        (None, None),
    ],
)
def test_the_offloaded_line_is_parsed_when_present(tail, expected):
    assert cm.placement_from_tail(tail) == expected


def test_the_fa_disabled_warn_is_found_and_the_fit_warn_is_not():
    disabled = "llama_context: Flash Attention not supported, set to disabled\n"
    fit = (
        "common_fit_params: failed to fit params to free device memory: "
        "n_gpu_layers already set by user\n"
    )
    assert cm.fa_disabled_in(disabled) is True
    assert cm.fa_disabled_in(fit) is False
    assert cm.fa_disabled_in("llama_context: Flash Attention enabled\n") is False
    assert cm.fa_disabled_in(None) is False


def test_the_device_check_limit_is_the_margin_over_the_prediction():
    assert cm.device_check_limit_bytes(static_device_bytes=3 * GIB, weights_bytes=4 * GIB) == (
        int(1.1 * 7 * GIB) + 256 * MIB
    )


# ===================== CUDA_VISIBLE_DEVICES =====================


DEVICES = [
    {"id": 0, "uuid": "GPU-aaaa-1111", "vram_total_mb": 8192},
    {"id": 1, "uuid": "GPU-bbbb-2222", "vram_total_mb": 24576},
]


@pytest.mark.parametrize(
    "visible, ids",
    [
        (None, [0, 1]),
        ("GPU-aaaa-1111", [0]),
        ("GPU-aaaa", [0]),  # CUDA accepts a UUID prefix
        ("1", [1]),
        ("1,GPU-aaaa-1111", [0, 1]),
        # Empty is "not set": a copied .env template must not hide the GPUs.
        ("", [0, 1]),
        ("  ", [0, 1]),
        ("MIG-unknown", None),
    ],
)
def test_a_user_cuda_visible_devices_restricts_the_selection(visible, ids):
    chosen = cm.visible_devices(DEVICES, visible)
    assert (None if chosen is None else [d["id"] for d in chosen]) == ids


# ===================== the healthy Shared level and the spill signals =====================


@pytest.mark.parametrize(
    "window, measured_mib",
    # #661: Shared Usage of healthy raw llama-server children, Flash Attention
    # on (typeperf): 0.5B at 8192, 27B at 4096, 4B at 40960.
    [(8192, 90), (4096, 106), (40_960, 130)],
)
def test_the_661_healthy_readings_stay_under_the_healthy_level(window, measured_mib):
    assert measured_mib * MIB < cm.healthy_shared_bytes(window, fa_on=True)


def test_the_healthy_level_is_the_fixed_part_plus_the_host_slope():
    fixed = cm.CUDA_SHARED_FIXED_BYTES
    assert fixed == 128 * MIB
    assert cm.healthy_shared_bytes(8192, fa_on=True) == fixed + 8192 * 1024
    assert cm.healthy_shared_bytes(8192, fa_on=False) == fixed + 8192 * 2048
    # #661: the CUDA_Host compute buffer grows by 24 MiB from 8192 to 32768
    # tokens with FA (11.51 -> 35.51 MiB), and -fa off adds 8.5 MiB at 8192
    # (20.01 MiB): the FA slope is 1 KiB per token, the non-FA one about 2.
    fa_growth = cm.healthy_shared_bytes(32_768, fa_on=True) - cm.healthy_shared_bytes(
        8192, fa_on=True
    )
    assert fa_growth == 24 * MIB


def test_windows_measures_dedicated_plus_the_shared_above_the_healthy_level():
    healthy = cm.healthy_shared_bytes(40_960, fa_on=True)
    assert cm.windows_device_bytes(
        dedicated_bytes=6 * GIB, shared_bytes=GIB, healthy_bytes=healthy
    ) == (6 * GIB + GIB - healthy)
    assert (
        cm.windows_device_bytes(
            dedicated_bytes=6 * GIB, shared_bytes=130 * MIB, healthy_bytes=healthy
        )
        == 6 * GIB
    )
    # A 262144-token window's healthy Shared memory is not device memory.
    long_healthy = cm.healthy_shared_bytes(262_144, fa_on=True)
    assert (
        cm.windows_device_bytes(
            dedicated_bytes=6 * GIB, shared_bytes=380 * MIB, healthy_bytes=long_healthy
        )
        == 6 * GIB
    )
    # A spill at load does not read as partial.
    predicted = _floor(weights=4 * GIB, kv=3 * GIB)
    measured = cm.windows_device_bytes(
        dedicated_bytes=5 * GIB, shared_bytes=2 * GIB + 200 * MIB, healthy_bytes=healthy
    )
    assert not cm.placement_partial(
        predicted_bytes=predicted, block_count=28, measured_bytes=measured
    )


@pytest.mark.parametrize(
    "spill, level",
    [
        (0, mb.LEVEL_CALM),
        (255 * MIB, mb.LEVEL_CALM),
        (256 * MIB, mb.LEVEL_WARNING),
        (GIB - 1, mb.LEVEL_WARNING),
        (GIB, mb.LEVEL_CRITICAL),
    ],
)
def test_the_spill_tiers(spill, level):
    assert cm.spill_level(spill) == level


def test_a_spill_at_warning_or_worse_is_the_level_and_the_damage():
    warning = cm.spill_signals(300 * MIB)
    assert warning == cm.SpillSignals(
        level=mb.LEVEL_WARNING, damage_bytes=300 * MIB, damage_available=True
    )
    critical = cm.spill_signals(2 * GIB)
    assert critical.level == mb.LEVEL_CRITICAL
    assert critical.damage_bytes == 2 * GIB


def test_a_calm_spill_contributes_no_damage_at_turn_end():
    """M5: a stable 100 MiB spill must not promote a warning through the
    swap-burst rule at every turn end."""
    signals = cm.spill_signals(100 * MIB)
    assert signals == cm.SpillSignals(
        level=mb.LEVEL_CALM, damage_bytes=None, damage_available=False
    )
    decision = mb.llama_step(
        mb.BrakeState(),
        mb.Sample(
            point=mb.POINT_TURN_END,
            now=1000.0,
            level=signals.level,
            damage_bytes=signals.damage_bytes,
            damage_available=signals.damage_available,
        ),
        mb.Context(
            window=40_960,
            ceiling=None,
            static_window=40_960,
            footprint_share=0.60,
            kv_share=0.30,
        ),
    )
    assert decision.level == mb.LEVEL_CALM
    assert decision.state.in_episode is False
