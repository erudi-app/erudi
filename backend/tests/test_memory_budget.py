"""Deterministic memory accounting for the compaction memory signal (1.1.2).

Pure math over facts read from disk and the engine's hardware totals — never
``psutil available`` (macOS compression/swap makes it lie). Every test here is
hand-computed: the KV formula is 2 (K and V) x layers x kv_heads x head_dim x
2 bytes (f16) per token. Missing facts always answer ``None`` (signal off),
never a guessed number.
"""

import json

import pytest

from src.engines.base_engine import BaseEngine
from src.engines.memory_budget import (
    MemoryBudget,
    artifact_bytes,
    kv_bytes_per_token,
    total_memory_bytes,
)

pytestmark = pytest.mark.unit


# ===================== KV-per-token math =====================


def test_kv_bytes_per_token_matches_hand_computed_value():
    # 2 x 24 layers x 8 kv heads x 128 head_dim x 2 bytes = 98304 bytes/token.
    config = {"num_hidden_layers": 24, "num_key_value_heads": 8, "head_dim": 128}
    assert kv_bytes_per_token(config) == 98304


def test_kv_bytes_per_token_derives_head_dim_from_hidden_size():
    # head_dim absent: hidden_size / num_attention_heads = 4096/32 = 128.
    # 2 x 32 x 8 x 128 x 2 = 131072.
    config = {
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "num_attention_heads": 32,
        "hidden_size": 4096,
    }
    assert kv_bytes_per_token(config) == 131072


def test_kv_bytes_per_token_defaults_kv_heads_to_attention_heads():
    # num_key_value_heads absent: transformers defaults it to
    # num_attention_heads (MHA), so that is a definition, not a guess.
    # 2 x 16 x 16 x 64 x 2 = 65536.
    config = {"num_hidden_layers": 16, "num_attention_heads": 16, "head_dim": 64}
    assert kv_bytes_per_token(config) == 65536


def test_kv_bytes_per_token_reads_nested_text_config():
    # A VLM keeps its text model under text_config (same containers as the
    # context-window reader in generation_hints).
    config = {"text_config": {"num_hidden_layers": 24, "num_key_value_heads": 8, "head_dim": 128}}
    assert kv_bytes_per_token(config) == 98304


@pytest.mark.parametrize(
    "config",
    [
        None,
        {},
        {"num_key_value_heads": 8, "head_dim": 128},  # layers missing
        {"num_hidden_layers": 24},  # no heads at all
        {"num_hidden_layers": 24, "num_key_value_heads": 8},  # no head_dim derivable
        {"num_hidden_layers": 0, "num_key_value_heads": 8, "head_dim": 128},  # degenerate
    ],
)
def test_kv_bytes_per_token_missing_facts_answer_none(config):
    assert kv_bytes_per_token(config) is None


# [M3] Shapes the full-attention formula cannot model answer None: an
# over-estimate (4-7x on sliding-window or MLA caches) would fire compaction
# far too early and silently amputate context -- worse than no signal.


def test_kv_bytes_per_token_none_on_sliding_window_smaller_than_the_window():
    # Gemma lineage: per-layer sliding window far below the trained window.
    config = {
        "num_hidden_layers": 26,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "sliding_window": 4096,
        "max_position_embeddings": 32768,
    }
    assert kv_bytes_per_token(config) is None


def test_kv_bytes_per_token_keeps_signal_when_sliding_window_is_disabled():
    # Qwen2 lineage: sliding_window present but use_sliding_window is false --
    # the cache IS full attention, the formula holds.
    config = {
        "num_hidden_layers": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "sliding_window": 32768,
        "use_sliding_window": False,
        "max_position_embeddings": 32768,
    }
    assert kv_bytes_per_token(config) == 98304


def test_kv_bytes_per_token_keeps_signal_when_sliding_window_covers_the_window():
    # A sliding window as large as the trained window slides over nothing.
    config = {
        "num_hidden_layers": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "sliding_window": 32768,
        "max_position_embeddings": 32768,
    }
    assert kv_bytes_per_token(config) == 98304


def test_kv_bytes_per_token_none_on_mla():
    # DeepSeek lineage: MLA stores compressed latents, not per-head K/V.
    config = {
        "num_hidden_layers": 27,
        "num_key_value_heads": 16,
        "head_dim": 128,
        "kv_lora_rank": 512,
    }
    assert kv_bytes_per_token(config) is None


def test_kv_bytes_per_token_none_on_nested_sliding_window():
    config = {
        "text_config": {
            "num_hidden_layers": 26,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "sliding_window": 512,
            "max_position_embeddings": 131072,
        }
    }
    assert kv_bytes_per_token(config) is None


# ===================== Artifact size =====================


def test_artifact_bytes_of_a_file(tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"x" * 1000)
    assert artifact_bytes(gguf) == 1000


def test_artifact_bytes_of_a_directory_sums_files(tmp_path):
    (tmp_path / "weights.safetensors").write_bytes(b"x" * 700)
    (tmp_path / "config.json").write_bytes(b"y" * 300)
    assert artifact_bytes(tmp_path) == 1000


def test_artifact_bytes_of_a_missing_path_is_none(tmp_path):
    assert artifact_bytes(tmp_path / "absent") is None


# ===================== Denominator per engine family =====================


def _engine_with_flat_data(flat: dict):
    class _Engine(BaseEngine):
        @classmethod
        def get_flat_hardware_data(cls):
            return flat

    return _Engine


def test_total_memory_bytes_mlx_uses_unified_memory():
    engine = _engine_with_flat_data({"backend_type": "mlx", "total_memory_gb": 16.0})
    assert total_memory_bytes(engine) == 16 * 1024**3


def test_total_memory_bytes_cpu_uses_system_ram():
    engine = _engine_with_flat_data({"backend_type": "cpu", "total_memory_gb": 8.0})
    assert total_memory_bytes(engine) == 8 * 1024**3


def test_total_memory_bytes_cuda_signal_is_off():
    # Discrete GPUs run with the memory signal OFF: under partial offload the
    # weights split between VRAM and RAM in proportions we cannot know, so any
    # VRAM-only accounting is dishonest (a partially offloaded model would
    # read as saturating the card while running fine). llama's own fit already
    # bounded the allocation at load.
    engine = _engine_with_flat_data(
        {"backend_type": "cuda", "total_memory_gb": 64.0, "vram_total_gb": 12.0}
    )
    assert total_memory_bytes(engine) is None


def test_total_memory_bytes_missing_total_is_none():
    engine = _engine_with_flat_data({"backend_type": "cpu"})
    assert total_memory_bytes(engine) is None


def test_total_memory_bytes_is_memoized_per_engine_class():
    calls = []

    class _Engine(BaseEngine):
        @classmethod
        def get_flat_hardware_data(cls):
            calls.append(1)
            return {"backend_type": "cpu", "total_memory_gb": 4.0}

    assert total_memory_bytes(_Engine) == 4 * 1024**3
    assert total_memory_bytes(_Engine) == 4 * 1024**3
    assert len(calls) == 1


# ===================== The budget =====================

GIB = 1024**3


def test_margin_and_conversation_bytes_hand_computed():
    # 10 GiB weights + 1 MiB/token KV on a 16 GiB machine.
    budget = MemoryBudget(weights_bytes=10 * GIB, kv_token_bytes=1024**2, total_bytes=16 * GIB)
    # 1024 tokens -> 1 GiB of KV -> used 11 GiB of 16 -> margin 5/16 = 0.3125.
    assert budget.conversation_bytes(1024) == GIB
    assert budget.memory_margin_fraction(1024) == pytest.approx(5 / 16)
    assert budget.used_fraction(1024) == pytest.approx(11 / 16)


def test_margin_fifteen_percent_boundary():
    budget = MemoryBudget(weights_bytes=10 * GIB, kv_token_bytes=1024**2, total_bytes=16 * GIB)
    # margin hits exactly 0.15 when used = 0.85*16 = 13.6 GiB -> KV = 3.6 GiB
    # -> 3.6 * 1024 tokens.
    boundary_tokens = int(3.6 * 1024)
    assert budget.memory_margin_fraction(boundary_tokens) == pytest.approx(0.15, abs=1e-3)
    assert budget.tokens_at_margin(0.15) == boundary_tokens


def test_tokens_at_margin_negative_when_weights_alone_blow_the_floor():
    budget = MemoryBudget(weights_bytes=15 * GIB, kv_token_bytes=1024**2, total_bytes=16 * GIB)
    assert budget.tokens_at_margin(0.15) < 0


@pytest.mark.parametrize(
    "budget",
    [
        MemoryBudget(weights_bytes=None, kv_token_bytes=1024, total_bytes=GIB),
        MemoryBudget(weights_bytes=GIB, kv_token_bytes=None, total_bytes=GIB),
        MemoryBudget(weights_bytes=GIB, kv_token_bytes=1024, total_bytes=None),
    ],
)
def test_unaccountable_budget_disables_the_signal(budget):
    assert budget.memory_margin_fraction(100) is None
    assert budget.tokens_at_margin(0.15) is None
    assert budget.used_fraction(100) is None


def test_conversation_bytes_only_needs_the_kv_fact():
    budget = MemoryBudget(weights_bytes=None, kv_token_bytes=100, total_bytes=None)
    assert budget.conversation_bytes(10) == 1000
    budget = MemoryBudget(weights_bytes=GIB, kv_token_bytes=None, total_bytes=GIB)
    assert budget.conversation_bytes(10) is None


# ===================== from_engine =====================


def _loaded_engine(flat: dict, model_path):
    engine = _engine_with_flat_data(flat)
    engine._model = {"model_path": str(model_path)}
    return engine


def test_from_engine_mlx_directory(tmp_path, monkeypatch):
    (tmp_path / "weights.safetensors").write_bytes(b"w" * 5000)
    (tmp_path / "config.json").write_text(
        json.dumps({"num_hidden_layers": 24, "num_key_value_heads": 8, "head_dim": 128}),
        encoding="utf-8",
    )
    engine = _loaded_engine({"backend_type": "mlx", "total_memory_gb": 16.0}, tmp_path)
    try:
        budget = MemoryBudget.from_engine(engine)
    finally:
        engine._model = None
    config_bytes = (tmp_path / "config.json").stat().st_size
    assert budget.weights_bytes == 5000 + config_bytes
    assert budget.kv_token_bytes == 98304
    assert budget.total_bytes == 16 * 1024**3


def test_from_engine_gguf_with_config_on_cpu_is_fully_alive(tmp_path):
    # The app's downloader fetches a repo's small aux files alongside the
    # .gguf, so most GGUF folders DO carry a config.json: the memory signal
    # is alive on the CPU engine (one honest RAM pool).
    gguf = tmp_path / "model-q4.gguf"
    gguf.write_bytes(b"g" * 4000)
    (tmp_path / "config.json").write_text(
        json.dumps({"num_hidden_layers": 24, "num_key_value_heads": 8, "head_dim": 128}),
        encoding="utf-8",
    )
    engine = _loaded_engine({"backend_type": "cpu", "total_memory_gb": 8.0}, gguf)
    try:
        budget = MemoryBudget.from_engine(engine)
    finally:
        engine._model = None
    assert budget.kv_token_bytes == 98304
    assert budget.total_bytes == 8 * 1024**3
    assert budget.memory_margin_fraction(100) is not None


def test_from_engine_cuda_never_accounts(tmp_path):
    # [H1] Even with every fact readable, a CUDA engine's budget cannot warn:
    # its pool is None by policy (partial offload, see total_memory_bytes).
    gguf = tmp_path / "model-q4.gguf"
    gguf.write_bytes(b"g" * 4000)
    (tmp_path / "config.json").write_text(
        json.dumps({"num_hidden_layers": 24, "num_key_value_heads": 8, "head_dim": 128}),
        encoding="utf-8",
    )
    engine = _loaded_engine(
        {"backend_type": "cuda", "total_memory_gb": 64.0, "vram_total_gb": 12.0}, gguf
    )
    try:
        budget = MemoryBudget.from_engine(engine)
    finally:
        engine._model = None
    assert budget.total_bytes is None
    assert budget.memory_margin_fraction(100) is None
    assert budget.tokens_at_margin(0.15) is None


def test_from_engine_gguf_without_config_disables_the_kv_fact(tmp_path):
    # A GGUF artifact ships no config.json: the KV fact is underivable, the
    # memory signal is off (weights and total still resolve).
    gguf = tmp_path / "model-q4.gguf"
    gguf.write_bytes(b"g" * 4000)
    engine = _loaded_engine({"backend_type": "cpu", "total_memory_gb": 8.0}, gguf)
    try:
        budget = MemoryBudget.from_engine(engine)
    finally:
        engine._model = None
    assert budget.weights_bytes == 4000
    assert budget.kv_token_bytes is None
    assert budget.memory_margin_fraction(100) is None


def test_from_engine_without_a_loaded_child_is_all_none():
    engine = _engine_with_flat_data({"backend_type": "cpu", "total_memory_gb": 8.0})
    engine._model = None
    budget = MemoryBudget.from_engine(engine)
    assert budget.weights_bytes is None
    assert budget.kv_token_bytes is None
    assert budget.memory_margin_fraction(100) is None


def test_from_engine_none_engine_is_all_none():
    budget = MemoryBudget.from_engine(None)
    assert budget.memory_margin_fraction(100) is None
