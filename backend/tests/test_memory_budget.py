"""Deterministic memory accounting for the compaction memory signal (1.1.2).

Pure math over facts read from disk and the engine's hardware totals — never
``psutil available`` (macOS compression/swap makes it lie). Every test here is
hand-computed: the KV formula is 2 (K and V) x layers x kv_heads x head_dim x
2 bytes (f16) per token. Missing facts always answer ``None`` (signal off),
never a guessed number.
"""

import json
import logging

import pytest

from src.engines.base_engine import BaseEngine
from src.engines.memory_budget import (
    BASE_FALLBACK_OVERHEAD_BYTES,
    LOGITS_BYTES_PER_VALUE,
    MLX_PREFILL_STEP_TOKENS,
    PRIOR_FIXED_BYTES,
    PRIOR_KV_MULTIPLIER,
    PRIOR_MARGIN,
    MemoryBudget,
    artifact_bytes,
    kv_bytes_per_token,
    total_memory_bytes,
    vocab_size_of,
)

pytestmark = pytest.mark.unit


# ===================== KV-per-token math =====================


def test_the_vocabulary_is_read_from_the_top_level_or_the_text_config():
    assert vocab_size_of({"vocab_size": 32_000}) == 32_000
    assert vocab_size_of({"text_config": {"vocab_size": 262_144}}) == 262_144
    assert vocab_size_of({}) is None
    assert vocab_size_of({"vocab_size": True}) is None


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


def test_artifact_bytes_sums_split_gguf_parts(tmp_path):
    # [L3] A split GGUF maps ALL its parts: the resolved first part must
    # account for the whole family, not just itself -- and never for an
    # unrelated file in the same directory.
    (tmp_path / "model-Q4-00001-of-00003.gguf").write_bytes(b"a" * 100)
    (tmp_path / "model-Q4-00002-of-00003.gguf").write_bytes(b"b" * 200)
    (tmp_path / "model-Q4-00003-of-00003.gguf").write_bytes(b"c" * 300)
    (tmp_path / "other-model.gguf").write_bytes(b"z" * 1000)
    assert artifact_bytes(tmp_path / "model-Q4-00001-of-00003.gguf") == 600


def test_artifact_bytes_plain_gguf_is_just_the_file(tmp_path):
    gguf = tmp_path / "model-Q4.gguf"
    gguf.write_bytes(b"g" * 400)
    (tmp_path / "sibling-Q8.gguf").write_bytes(b"s" * 4000)
    assert artifact_bytes(gguf) == 400


# ===================== Denominator per engine family =====================


def _engine_with_flat_data(flat: dict, format_tag: str = "mlx", working_set_bytes=None):
    ws = working_set_bytes

    class _Engine(BaseEngine):
        FORMAT_TAG = format_tag

        @classmethod
        def get_flat_hardware_data(cls):
            return flat

        @classmethod
        def max_recommended_working_set_bytes(cls):
            return ws

    return _Engine


def test_total_memory_bytes_mlx_uses_working_set_not_total_ram():
    # The denominator is the GPU's usable working set (0.9 x the recommended
    # working set), NOT total_memory_gb. On Apple Silicon Metal caps the GPU at
    # ~74 % of RAM, so total RAM is a ~2.4x-too-large denominator (#601).
    working_set = 12_700_000_000  # ~12.7 GB on a 16 GB M4, not the 16 GiB total
    engine = _engine_with_flat_data(
        {"backend_type": "mlx", "total_memory_gb": 16.0},
        working_set_bytes=working_set,
    )
    assert total_memory_bytes(engine) == int(0.9 * working_set)


def test_total_memory_bytes_llama_cpp_engines_signal_is_off():
    # The memory signal is MLX-ONLY: on both llama.cpp engines (CPU and CUDA)
    # the KV cache is allocated IN FULL at load -- memory use does not grow
    # with the conversation, and the engine's own fit already guaranteed the
    # allocation fits. There is nothing per-token to measure; on a discrete
    # card, partial offload would additionally make any single-pool
    # accounting dishonest.
    for flat in (
        {"backend_type": "cpu", "total_memory_gb": 8.0},
        {"backend_type": "cuda", "total_memory_gb": 64.0, "vram_total_gb": 12.0},
    ):
        engine = _engine_with_flat_data(flat, format_tag="gguf")
        assert total_memory_bytes(engine) is None


def test_total_memory_bytes_missing_working_set_disables_signal_and_warns_once(caplog):
    # An unreadable working set -> None (signal off), and the degraded record
    # is written once even though every later call retries ([L4]).
    calls = []

    class _Engine(BaseEngine):
        FORMAT_TAG = "mlx"

        @classmethod
        def max_recommended_working_set_bytes(cls):
            calls.append(1)
            return None

    with caplog.at_level(logging.WARNING, logger="erudi"):
        assert total_memory_bytes(_Engine) is None
        assert total_memory_bytes(_Engine) is None
    assert len(calls) == 2  # retried, not memoized
    warned = [r for r in caplog.records if "memory signal off" in r.getMessage()]
    assert len(warned) == 1


def test_total_memory_bytes_none_working_set_is_retried_not_memoized():
    # [L4] A working set that could not be read is retried on the next call
    # (only a resolved answer is memoized).
    calls = []

    class _Engine(BaseEngine):
        FORMAT_TAG = "mlx"

        @classmethod
        def max_recommended_working_set_bytes(cls):
            calls.append(1)
            if len(calls) == 1:
                return None  # first probe: nothing readable
            return 10 * 1024**3

    assert total_memory_bytes(_Engine) is None
    assert total_memory_bytes(_Engine) == int(0.9 * 10 * 1024**3)
    assert len(calls) == 2


def test_total_memory_bytes_is_memoized_per_engine_class():
    calls = []

    class _Engine(BaseEngine):
        FORMAT_TAG = "mlx"

        @classmethod
        def max_recommended_working_set_bytes(cls):
            calls.append(1)
            return 8 * 1024**3

    assert total_memory_bytes(_Engine) == int(0.9 * 8 * 1024**3)
    assert total_memory_bytes(_Engine) == int(0.9 * 8 * 1024**3)
    assert len(calls) == 1


# ===================== The budget: a measured, structural prior =====================

GIB = 1024**3
KV = 114_688  # Qwen3-0.6B: 2 x 28 layers x 8 kv heads x 128 x 2 bytes
VOCAB = 151_936
STEP = 2048


def _budget(**overrides):
    facts = dict(
        base_bytes=2 * GIB,
        total_bytes=10 * GIB,
        kv_token_bytes=KV,
        vocab_size=VOCAB,
        prefill_step=STEP,
        weights_bytes=GIB,
    )
    facts.update(overrides)
    return MemoryBudget(**facts)


def _hand_predict(n):
    return PRIOR_MARGIN * (
        PRIOR_FIXED_BYTES + STEP * VOCAB * LOGITS_BYTES_PER_VALUE + PRIOR_KV_MULTIPLIER * KV * n
    )


def test_the_priors_are_the_named_provisional_values():
    assert PRIOR_FIXED_BYTES == 1 * GIB
    assert PRIOR_KV_MULTIPLIER == 3.5
    assert PRIOR_MARGIN == 1.10
    assert BASE_FALLBACK_OVERHEAD_BYTES == int(0.45 * GIB)
    assert MLX_PREFILL_STEP_TOKENS == 2048


def test_predict_is_the_structural_prior_hand_computed():
    budget = _budget()
    for n in (0, 256, 4096, 32_768):
        assert budget.predict(n) == int(_hand_predict(n))


def test_predict_is_linear_and_monotone():
    budget = _budget()
    values = [budget.predict(n) for n in range(0, 40_000, 2_000)]
    assert values == sorted(values)
    steps = {b - a for a, b in zip(values, values[1:])}
    assert max(steps) - min(steps) <= 1  # integer rounding only


def test_the_logits_term_comes_from_the_vocabulary_and_the_step():
    small = _budget(vocab_size=32_000)
    big = _budget(vocab_size=262_144)
    assert big.predict(0) - small.predict(0) == pytest.approx(
        PRIOR_MARGIN * STEP * (262_144 - 32_000) * 2, abs=2
    )
    step_512 = _budget(prefill_step=512)
    assert _budget().predict(0) - step_512.predict(0) == pytest.approx(
        PRIOR_MARGIN * (2048 - 512) * VOCAB * 2, abs=2
    )


@pytest.mark.parametrize("missing", ["kv_token_bytes", "vocab_size", "prefill_step"])
def test_an_unknown_shape_fact_turns_the_prediction_off(missing):
    budget = _budget(**{missing: None})
    assert budget.predict(100) is None
    assert budget.footprint_bytes(100) is None
    assert budget.conversation_bytes(100) is None
    assert budget.used_fraction(100) is None
    assert budget.memory_margin_fraction(100) is None
    assert budget.tokens_at_ceiling() is None


def test_footprint_conversation_and_fractions_hand_computed():
    budget = _budget()
    n = 8192
    predicted = int(_hand_predict(n))
    assert budget.conversation_bytes(n) == predicted
    assert budget.footprint_bytes(n) == 2 * GIB + predicted
    assert budget.used_fraction(n) == pytest.approx((2 * GIB + predicted) / (10 * GIB))
    assert budget.memory_margin_fraction(n) == pytest.approx(1 - (2 * GIB + predicted) / (10 * GIB))


def test_tokens_at_ceiling_closed_form_reserves_the_output_floor():
    budget = _budget()
    headroom = 10 * GIB - 2 * GIB
    expected = (
        headroom / PRIOR_MARGIN - PRIOR_FIXED_BYTES - STEP * VOCAB * LOGITS_BYTES_PER_VALUE
    ) / (PRIOR_KV_MULTIPLIER * KV) - 512
    assert budget.tokens_at_ceiling() == int(expected)
    # At the ceiling plus the 512-token floor the footprint reaches the budget.
    at = budget.tokens_at_ceiling() + 512
    assert budget.footprint_bytes(at) <= 10 * GIB < budget.footprint_bytes(at + 2)


def test_tokens_at_ceiling_is_non_positive_when_the_fixed_part_alone_blows_it():
    budget = _budget(base_bytes=9 * GIB)
    assert budget.tokens_at_ceiling() <= 0


@pytest.mark.parametrize("missing", ["base_bytes", "total_bytes"])
def test_without_base_or_total_there_is_no_ceiling(missing):
    budget = _budget(**{missing: None})
    assert budget.tokens_at_ceiling() is None
    assert budget.used_fraction(100) is None


def test_a_regression_pin_bench5_points_stay_under_the_prediction():
    """NOT a proof -- the priors were fitted on these points. The 107 points
    of the measurement campaign (``tests/fixtures/memory_prior_bench5.json``:
    three models, prefill steps 2048 and 512, dirty points included) stay
    under ``predict``; a change of the priors that puts one above fails
    here. The gate campaign's clean points join this fixture before merge."""
    from pathlib import Path

    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "memory_prior_bench5.json").read_text()
    )
    worst_slack = None
    for model, step, kv, vocab, n, measured_gib, _dirty in fixture["points"]:
        budget = _budget(kv_token_bytes=kv, vocab_size=vocab, prefill_step=step)
        slack = budget.predict(n) - measured_gib * GIB
        assert slack > 0, (model, step, n)
        worst_slack = slack if worst_slack is None else min(worst_slack, slack)
    assert len(fixture["points"]) == 107
    assert worst_slack > 0.35 * GIB


# ===================== from_engine =====================


def _loaded_engine(flat: dict, model_path, format_tag: str = "mlx", working_set=None):
    engine = _engine_with_flat_data(flat, format_tag, working_set_bytes=working_set)
    engine._model = {"model_path": str(model_path)}
    return engine


def _mlx_dir(tmp_path, **config):
    (tmp_path / "weights.safetensors").write_bytes(b"w" * 5000)
    shape = {"num_hidden_layers": 24, "num_key_value_heads": 8, "head_dim": 128}
    shape.update(config)
    (tmp_path / "config.json").write_text(json.dumps(shape), encoding="utf-8")
    return tmp_path


def test_from_engine_mlx_reads_the_static_facts_and_the_measured_base(tmp_path):
    model = _mlx_dir(tmp_path, vocab_size=VOCAB)
    working_set = 12 * GIB
    engine = _loaded_engine({"backend_type": "mlx"}, model, working_set=working_set)
    engine._model["base_footprint_bytes"] = 3 * GIB
    try:
        budget = MemoryBudget.from_engine(engine)
    finally:
        engine._model = None
    config_bytes = (tmp_path / "config.json").stat().st_size
    assert budget.weights_bytes == 5000 + config_bytes
    assert budget.kv_token_bytes == 98304
    assert budget.vocab_size == VOCAB
    assert budget.prefill_step == MLX_PREFILL_STEP_TOKENS
    assert budget.base_bytes == 3 * GIB
    assert budget.total_bytes == int(0.9 * working_set)


def test_from_engine_without_a_measured_base_falls_back_to_the_weights_plus_overhead(tmp_path):
    model = _mlx_dir(tmp_path, vocab_size=VOCAB)
    engine = _loaded_engine({"backend_type": "mlx"}, model, working_set=12 * GIB)
    try:
        budget = MemoryBudget.from_engine(engine)
    finally:
        engine._model = None
    assert budget.base_bytes == budget.weights_bytes + BASE_FALLBACK_OVERHEAD_BYTES


def test_the_static_facts_are_read_once_per_child_and_cached_on_the_handle(tmp_path, monkeypatch):
    import src.engines.memory_budget as mb

    model = _mlx_dir(tmp_path, vocab_size=VOCAB)
    engine = _loaded_engine({"backend_type": "mlx"}, model, working_set=12 * GIB)
    reads = []
    real = mb.artifact_bytes
    monkeypatch.setattr(mb, "artifact_bytes", lambda path: reads.append(path) or real(path))
    try:
        first = MemoryBudget.from_engine(engine)
        second = MemoryBudget.from_engine(engine)
        assert "memory_facts" in engine._model
    finally:
        engine._model = None
    assert len(reads) == 1
    assert first == second


def test_from_engine_llama_cpp_is_all_none_and_writes_nothing(tmp_path):
    gguf = tmp_path / "model-q4.gguf"
    gguf.write_bytes(b"g" * 4000)
    (tmp_path / "config.json").write_text(
        json.dumps({"num_hidden_layers": 24, "num_key_value_heads": 8, "head_dim": 128}),
        encoding="utf-8",
    )
    for flat in (
        {"backend_type": "cpu", "total_memory_gb": 8.0},
        {"backend_type": "cuda", "total_memory_gb": 64.0, "vram_total_gb": 12.0},
    ):
        engine = _loaded_engine(flat, gguf, format_tag="gguf")
        handle = engine._model
        try:
            budget = MemoryBudget.from_engine(engine)
        finally:
            engine._model = None
        assert budget == MemoryBudget.unaccounted()
        assert budget.predict(100) is None
        assert budget.tokens_at_ceiling() is None
        assert set(handle) == {"model_path"}, "no handle writes on llama.cpp"


def test_from_engine_without_a_loaded_child_is_all_none():
    engine = _engine_with_flat_data({"backend_type": "mlx"})
    engine._model = None
    budget = MemoryBudget.from_engine(engine)
    assert budget.predict(100) is None
    assert budget.memory_margin_fraction(100) is None


def test_from_engine_none_engine_is_all_none():
    budget = MemoryBudget.from_engine(None)
    assert budget.memory_margin_fraction(100) is None
    assert budget.tokens_at_ceiling() is None
