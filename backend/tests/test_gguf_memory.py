"""The llama.cpp window, sized by Erudi from the GGUF header (pure parts).

``src.engines.gguf_memory`` turns a GGUF's KV header into the KV bytes one
token costs, then the window the machine affords at spawn:
``0.75 x RAM - weights - compute``, divided by the per-token cost (KV + the
Flash-Attention compute buffer), clamped between the 4096 floor and the
trained window. The formula is pinned to the allocation lines llama-server
printed for the two bench models (Qwen3-0.6B: 896.00 MiB of KV at -c 8192;
Llama-3.2-1B: 256.00 MiB). Everything below runs on dict fixtures; only the
reader tests need the ``gguf`` package, which the macOS leg does not install.
"""

from __future__ import annotations

import logging
import sys

import pytest

from src.engines import gguf_memory
from src.engines.gguf_memory import (
    LLAMA_COMPUTE_FIXED_BYTES,
    LLAMA_COMPUTE_INTERCEPT_MEASURED_BYTES,
    LLAMA_COMPUTE_PER_TOKEN_BYTES,
    LLAMA_PREDICTION_MARGIN,
    LLAMA_PREDICTION_SLACK_BYTES,
    LLAMA_RAM_BUDGET_FRACTION,
    kv_bytes_per_token,
    post_load_limit_bytes,
    shape_from_fields,
    window_for,
)
from src.engines.memory_brake import LLAMA_CTX_PAD_TOKENS
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
LLAMA_1B = {
    "architecture": "llama",
    "block_count": 16,
    "context_length": 131_072,
    "embedding_length": 2048,
    "head_count": 32,
    "head_count_kv": 8,
    "key_length": 64,
    "value_length": 64,
}


def _shape(**over):
    return shape_from_fields({**QWEN3, **over})


# ===================== the KV formula =====================


def test_the_kv_formula_matches_the_measured_kv_buffers():
    # llama-server: "KV buffer size = 896.00 MiB" at -c 8192 (Qwen3-0.6B),
    # "256.00 MiB" at -c 8192 (Llama-3.2-1B). Zero error.
    assert kv_bytes_per_token(shape_from_fields(QWEN3)) == 114_688
    assert kv_bytes_per_token(shape_from_fields(LLAMA_1B)) == 32_768
    assert 8192 * 114_688 == 896 * MIB
    assert 65_536 * 32_768 == 2048 * MIB


def test_a_per_layer_kv_head_list_is_summed_and_zero_layers_cost_nothing():
    # A hybrid recurrent architecture: 0 KV heads on its recurrent layers.
    shape = _shape(block_count=4, head_count_kv=[8, 0, 8, 0])
    assert kv_bytes_per_token(shape) == 2 * (128 + 128) * 8 * 2


def test_missing_kv_heads_fall_back_to_the_attention_heads():
    shape = _shape(head_count_kv=None)
    assert kv_bytes_per_token(shape) == 28 * (128 + 128) * 16 * 2


def test_missing_key_and_value_lengths_derive_from_the_embedding():
    shape = _shape(key_length=None, value_length=None)
    # 1024 // 16 = 64 per K and per V.
    assert kv_bytes_per_token(shape) == 28 * (64 + 64) * 8 * 2


def test_mla_is_refused():
    assert kv_bytes_per_token(_shape(kv_lora_rank=512)) is None


def test_a_sliding_window_keeps_the_full_formula_and_is_flagged():
    shape = _shape(sliding_window=1024)
    assert shape.swa_overestimated is True
    assert kv_bytes_per_token(shape) == 114_688
    # A window that slides over the whole trained window slides over nothing.
    assert _shape(sliding_window=40_960).swa_overestimated is False
    assert _shape().swa_overestimated is False


@pytest.mark.parametrize("missing", ["architecture", "block_count", "context_length", "head_count"])
def test_missing_essentials_answer_none(missing):
    fields = dict(QWEN3)
    fields[missing] = None
    if missing == "head_count":
        fields["head_count_kv"] = None
    assert shape_from_fields(fields) is None


def test_lengths_without_an_embedding_to_derive_from_answer_none():
    assert _shape(key_length=None, value_length=None, embedding_length=None) is None


# ===================== the spawn window =====================


def test_the_trained_window_binds_on_a_roomy_machine():
    shape = shape_from_fields(QWEN3)
    window = window_for(shape, weights_bytes=601 * MIB, total_ram_bytes=16 * GIB)
    assert window.window == 40_960
    assert window.trained == 40_960
    assert window.static_window == 40_960
    assert window.memory_bound is False
    assert window.kv_token_bytes == 114_688


def test_the_budget_binds_and_is_rounded_down():
    shape = shape_from_fields(QWEN3)
    total = 4 * GIB
    weights = 601 * MIB
    window = window_for(shape, weights_bytes=weights, total_ram_bytes=total)
    budget = int(LLAMA_RAM_BUDGET_FRACTION * total) - weights - LLAMA_COMPUTE_FIXED_BYTES
    affordable = budget // (114_688 + LLAMA_COMPUTE_PER_TOKEN_BYTES)
    expected = affordable // LLAMA_CTX_PAD_TOKENS * LLAMA_CTX_PAD_TOKENS
    assert window.window == expected
    assert window.window % LLAMA_CTX_PAD_TOKENS == 0
    assert MIN_KV_BOUND_TOKENS < window.window < 40_960
    assert window.memory_bound is True
    assert window.static_window == expected


def test_the_floor_is_spawned_when_nothing_fits():
    shape = shape_from_fields(QWEN3)
    window = window_for(shape, weights_bytes=8 * GIB, total_ram_bytes=8 * GIB)
    assert window.window == MIN_KV_BOUND_TOKENS
    assert window.memory_bound is True


def test_a_trained_window_under_the_floor_is_kept():
    shape = _shape(context_length=2048)
    window = window_for(shape, weights_bytes=8 * GIB, total_ram_bytes=8 * GIB)
    assert window.window == 2048
    assert window.memory_bound is True


def test_a_brake_cap_folds_in_without_making_the_window_memory_bound():
    shape = shape_from_fields(QWEN3)
    window = window_for(shape, weights_bytes=601 * MIB, total_ram_bytes=16 * GIB, brake_cap=20_480)
    assert window.window == 20_480
    assert window.static_window == 40_960
    assert window.memory_bound is False
    # A cap above the window changes nothing.
    roomy = window_for(shape, weights_bytes=601 * MIB, total_ram_bytes=16 * GIB, brake_cap=90_000)
    assert roomy.window == 40_960


@pytest.mark.parametrize("cap, expected", [(20_000, 19_968), (7_872, 7_680), (4_100, 4_096)])
def test_a_brake_cap_is_rounded_down_to_the_padding(cap, expected):
    """llama.cpp pads -c up to a multiple of 256: the window passed is one."""
    shape = shape_from_fields(QWEN3)
    window = window_for(shape, weights_bytes=601 * MIB, total_ram_bytes=16 * GIB, brake_cap=cap)
    assert window.window == expected


def test_a_sliding_window_model_is_never_memory_bound():
    shape = _shape(sliding_window=1024)
    window = window_for(shape, weights_bytes=601 * MIB, total_ram_bytes=4 * GIB)
    assert window.window < 40_960
    assert window.memory_bound is False


def test_the_anonymous_prediction_excludes_the_weights():
    shape = shape_from_fields(QWEN3)
    window = window_for(shape, weights_bytes=601 * MIB, total_ram_bytes=16 * GIB)
    assert window.weights_bytes == 601 * MIB
    assert window.static_anonymous_bytes == (
        40_960 * 114_688 + LLAMA_COMPUTE_FIXED_BYTES + 40_960 * LLAMA_COMPUTE_PER_TOKEN_BYTES
    )


def test_an_mla_shape_has_no_window():
    assert window_for(_shape(kv_lora_rank=512), weights_bytes=GIB, total_ram_bytes=16 * GIB) is None


# ===================== the post-load check =====================


def test_the_post_load_limit_uses_the_measured_intercept_and_one_weights_copy():
    limit = post_load_limit_bytes(window=8192, kv_token_bytes=114_688, weights_bytes=601 * MIB)
    prediction = (
        8192 * 114_688
        + 8192 * LLAMA_COMPUTE_PER_TOKEN_BYTES
        + LLAMA_COMPUTE_INTERCEPT_MEASURED_BYTES
        + 601 * MIB
    )
    assert limit == int(LLAMA_PREDICTION_MARGIN * prediction) + LLAMA_PREDICTION_SLACK_BYTES
    # The measured repacked child (896 KV + 34 compute + 604 repack, MiB)
    # stays under it; a non-FA fallback at a big window does not.
    assert (896 + 34 + 604) * MIB < limit
    big = post_load_limit_bytes(window=131_072, kv_token_bytes=32_768, weights_bytes=763 * MIB)
    non_fa = 131_072 * 32_768 + 131_072 * 34 * 1024 + 763 * MIB
    assert non_fa > big


# ===================== the reader (needs gguf) =====================


def _write_gguf(path, *, arch="qwen3", kv_heads=(2, 0), sliding=None):
    gguf = pytest.importorskip("gguf")
    import numpy as np

    writer = gguf.GGUFWriter(str(path), arch)
    writer.add_block_count(len(kv_heads))
    writer.add_context_length(4096)
    writer.add_embedding_length(64)
    writer.add_head_count(4)
    writer.add_head_count_kv(list(kv_heads))
    writer.add_key_length(16)
    writer.add_value_length(16)
    if sliding is not None:
        writer.add_sliding_window(sliding)
    writer.add_tensor("x", np.zeros(4, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def test_read_shape_reads_the_header_and_releases_the_file(tmp_path):
    path = _write_gguf(tmp_path / "m.gguf", sliding=512)
    shape = gguf_memory.read_shape(path)
    assert shape.architecture == "qwen3"
    assert shape.block_count == 2
    assert shape.context_length == 4096
    assert shape.head_count_kv == (2, 0)
    assert shape.swa_overestimated is True
    assert kv_bytes_per_token(shape) == (16 + 16) * 2 * 2
    # The memmap is closed: the model file can be deleted (Windows locks it).
    path.unlink()
    assert not path.exists()


def test_spawn_window_adds_the_mmproj_to_the_weights(tmp_path):
    path = _write_gguf(tmp_path / "m.gguf")
    mmproj = tmp_path / "mmproj-m.gguf"
    mmproj.write_bytes(b"\0" * 4096)
    alone = gguf_memory.spawn_window(path, total_ram_bytes=16 * GIB)
    with_vision = gguf_memory.spawn_window(path, total_ram_bytes=16 * GIB, mmproj_path=mmproj)
    assert with_vision.weights_bytes == alone.weights_bytes + 4096
    assert alone.window == 4096


def test_a_purely_recurrent_model_is_not_called_mla(tmp_path, caplog):
    path = _write_gguf(tmp_path / "ssm.gguf", kv_heads=(0, 0))
    with caplog.at_level(logging.WARNING, logger="erudi"):
        assert gguf_memory.spawn_window(path, total_ram_bytes=16 * GIB) is None
    (warning,) = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert "MLA" not in warning
    assert "no attention KV cache" in warning


def test_an_unreadable_gguf_is_one_warning_and_no_window(tmp_path, caplog):
    pytest.importorskip("gguf")
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"not a gguf")
    with caplog.at_level(logging.WARNING, logger="erudi"):
        assert gguf_memory.spawn_window(bad, total_ram_bytes=16 * GIB) is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "llama-server sizes the window itself" in warnings[0].getMessage()


def test_without_the_gguf_package_the_spawn_falls_back(tmp_path, monkeypatch, caplog):
    # The macOS build does not ship gguf: nothing may raise ImportError.
    monkeypatch.setitem(sys.modules, "gguf", None)
    with caplog.at_level(logging.WARNING, logger="erudi"):
        assert gguf_memory.spawn_window(tmp_path / "m.gguf", total_ram_bytes=16 * GIB) is None
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_no_total_ram_means_no_window(tmp_path, caplog):
    path = _write_gguf(tmp_path / "m.gguf")
    with caplog.at_level(logging.WARNING, logger="erudi"):
        assert gguf_memory.spawn_window(path, total_ram_bytes=None) is None
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1
