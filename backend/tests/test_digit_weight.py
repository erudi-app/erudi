"""How many tokens a digit costs comes from the loaded tokenizer.

Qwen, Gemma, Mistral and DeepSeek split numbers into single digits; Llama 3,
gpt-oss and Phi-4 group up to three (``\\p{N}{1,3}``). One digit weight for
every model either under-counts the first family or silently truncates the
second (a 60k-character log with 27k digits would hand Llama 3 the 512-token
floor). The budget therefore reads the artifact's ``tokenizer.json``
pre-tokenizer once per child: digits split one by one cost 1.0 token, grouped
or unknown 0.34. The compaction counter keeps 1.0 (counting high only compacts
earlier).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from src.agents.output_budget import compute_output_budget
from src.agents.token_accounting import (
    DIGIT_TOKENS,
    GROUPED_DIGIT_TOKENS,
    digit_tokens_of,
    kb_stamp,
    script_weight,
)
from src.engines.base_engine import BaseEngine
from src.engines.memory_budget import budget_digit_tokens

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parent / "fixtures" / "tokenizers"


def _tokenizer(name):
    return json.loads((FIXTURES / name).read_text())


# ===================== reading the pre-tokenizer =====================


def test_a_qwen_style_tokenizer_splits_every_digit():
    assert digit_tokens_of(_tokenizer("qwen2_tokenizer.json")) == DIGIT_TOKENS == 1.0


def test_a_llama3_style_tokenizer_groups_three_digits():
    assert digit_tokens_of(_tokenizer("llama3_tokenizer.json")) == GROUPED_DIGIT_TOKENS == 0.34


def test_a_digits_pre_tokenizer_says_it_directly():
    individual = {"pre_tokenizer": {"type": "Digits", "individual_digits": True}}
    grouped = {"pre_tokenizer": {"type": "Digits", "individual_digits": False}}
    assert digit_tokens_of(individual) == 1.0
    assert digit_tokens_of(grouped) == GROUPED_DIGIT_TOKENS


@pytest.mark.parametrize(
    "tokenizer",
    [
        {},
        {"pre_tokenizer": None},
        {"pre_tokenizer": {"type": "Metaspace"}},
        {"pre_tokenizer": {"type": "Split", "pattern": {"String": " "}}},
        "not a dict",
    ],
)
def test_an_unknown_pre_tokenizer_is_unknown(tokenizer):
    assert digit_tokens_of(tokenizer) is None


# ===================== the static fact on the MLX handle =====================


def _engine(format_tag="mlx"):
    class _Engine(BaseEngine):
        FORMAT_TAG = format_tag

        @classmethod
        def max_recommended_working_set_bytes(cls):
            return 12 * 1024**3

    return _Engine


def _model_dir(tmp_path, tokenizer_name=None):
    (tmp_path / "config.json").write_text(
        json.dumps({"num_hidden_layers": 2, "num_key_value_heads": 2, "head_dim": 8}),
        encoding="utf-8",
    )
    if tokenizer_name:
        shutil.copy(FIXTURES / tokenizer_name, tmp_path / "tokenizer.json")
    return tmp_path


@pytest.mark.parametrize(
    "tokenizer_name, expected",
    [
        ("qwen2_tokenizer.json", 1.0),
        ("llama3_tokenizer.json", GROUPED_DIGIT_TOKENS),
        (None, GROUPED_DIGIT_TOKENS),
    ],
)
def test_the_budget_digit_weight_is_read_once_from_the_loaded_artifact(
    tmp_path, tokenizer_name, expected
):
    engine = _engine()
    engine._model = {"model_path": str(_model_dir(tmp_path, tokenizer_name))}
    try:
        assert budget_digit_tokens(engine) == expected
        assert "digit_tokens" in engine._model["memory_facts"]
    finally:
        engine._model = None


def test_llama_cpp_engines_use_the_grouped_weight_and_write_nothing(tmp_path):
    engine = _engine("gguf")
    engine._model = {"model_path": str(_model_dir(tmp_path, "qwen2_tokenizer.json"))}
    try:
        assert budget_digit_tokens(engine) == GROUPED_DIGIT_TOKENS
        assert set(engine._model) == {"model_path"}
    finally:
        engine._model = None


def test_the_factory_hands_the_digit_weight_to_the_client(monkeypatch, tmp_path):
    from src.agents.model_factory import build_chat_model
    from src.core import config

    class _Engine:
        FORMAT_TAG = "mlx"
        _model = {"model_path": str(_model_dir(tmp_path, "qwen2_tokenizer.json"))}

        @classmethod
        def get_model_and_tokenizer(cls, llm_id, link):
            return ({"base_url": "http://127.0.0.1:1", "alias": "a", **cls._model}, {})

        @staticmethod
        def _payload_model_value(handle):
            return "m"

        @staticmethod
        def max_recommended_working_set_bytes():
            return 12 * 1024**3

    class _Llm:
        id = 1
        link = "/x"
        name = "x"

    monkeypatch.setattr(config, "LLM_Engine", _Engine)
    client = build_chat_model(_Llm(), temperature=0.1, top_p=0.9, max_tokens=8)
    assert client.digit_tokens == 1.0


# ===================== the budget on a digit-heavy log =====================


def _log(chars=60_000):
    line = "2026-10-07 21:13:37.762 worker 4821 queue 7731 batch 0093 lat 1288 ms\n"
    return (line * (chars // len(line) + 1))[:chars]


def test_a_digit_heavy_log_is_not_truncated_on_a_grouping_tokenizer():
    log = _log()
    digits = sum(c.isdigit() for c in log)
    assert digits > 27_000
    messages = [HumanMessage(log)]

    grouped = compute_output_budget(messages, 32_768, digit_tokens=GROUPED_DIGIT_TOKENS)
    split = compute_output_budget(messages, 32_768, digit_tokens=DIGIT_TOKENS)

    # At one token per digit the log alone fills the window: the floor.
    assert split == 512
    # Llama 3 groups three digits a token: the answer keeps a real budget.
    assert grouped > 5_000


def test_the_kb_stamp_uses_the_budgets_digit_weight():
    from src.agents.token_accounting import REQUEST_KB_REAL_KEY

    block = "Table: 1234 5678 9012 3456 " * 50
    assert (
        kb_stamp(block, digit_tokens=GROUPED_DIGIT_TOKENS)[REQUEST_KB_REAL_KEY]
        < (kb_stamp(block, digit_tokens=DIGIT_TOKENS)[REQUEST_KB_REAL_KEY])
    )


def test_the_counter_keeps_one_token_per_digit():
    from src.agents.runner import counter_weight

    message = HumanMessage("1234567890" * 10)
    assert counter_weight(message) == pytest.approx(
        script_weight(message.content, dense=1.0, digit_tokens=1.0)
    )
