"""Per-model sampling defaults (#388) and the capture cascade.

Capture stops at the first stage that yields a usable sampling value: the base
repo's generation_config.json, then the quant repo's, then a conservative read
of the base repo's model card. The pure resolver turns the stored facts into
sampling defaults; a row without a usable value MUST resolve to today's
constants (the #129-validated request bodies stay byte-identical) with
``source == "none"``, and optional keys (top_k / min_p / presence_penalty)
exist only when the captured block defines them.

The context window rides along with its own cascade: the GGUF metadata of the
quant repo for a GGUF entry (what llama-server reads out of the file), else the
config.json window under any of the keys and containers publishers use.
"""

import json
import types
from unittest.mock import MagicMock

import pytest

from src.core import config
from src.database import generation_hints as gh
from src.database.generation_hints import (
    FALLBACK_MAX_TOKENS,
    FALLBACK_REPETITION_CONTEXT_SIZE,
    FALLBACK_REPETITION_PENALTY,
    FALLBACK_TEMPERATURE,
    FALLBACK_TOP_P,
    UNBOUNDED_CONTEXT_TOKENS,
    build_generation_hints,
    capture_generation_hints,
    extract_card_recommendations,
    read_local_generation_hints,
    resolve_base_repo,
    resolve_sampling_defaults,
    select_card_recommendation,
)


class _Llm:
    def __init__(
        self, hints=None, type="qwen", link="mlx-community/Qwen3-0.6B-4bit", name="Qwen3 0.6B"
    ):
        self.generation_hints = hints
        self.type = type
        self.link = link
        self.name = name


class _MlxEngine:
    FORMAT_TAG = "mlx"

    @classmethod
    def max_context_tokens(cls):
        return None


class _LlamaEngine:
    FORMAT_TAG = "gguf"

    @classmethod
    def max_context_tokens(cls):
        return 4096


def _hints(**generation_config):
    return {
        "base_repo": "Qwen/Qwen3-0.6B",
        "generation_config": generation_config,
        "supports_thinking": True,
        "context_length": 40960,
        "captured_at": "2026-08-28",
        "source_stage": "base_generation_config" if generation_config else None,
        "evidence": None,
    }


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    gh.reset_capture_cache()
    monkeypatch.setattr(config, "LLM_Engine", _MlxEngine)
    yield
    gh.reset_capture_cache()


# ---------------------------------------------------------------- resolver


class TestResolveNone:
    def test_no_hints_resolves_to_todays_constants(self):
        d = resolve_sampling_defaults(_Llm(None))
        assert (d.temperature, d.top_p, d.max_tokens) == (
            FALLBACK_TEMPERATURE,
            FALLBACK_TOP_P,
            FALLBACK_MAX_TOKENS,
        )
        assert (d.repetition_penalty, d.repetition_context_size) == (
            FALLBACK_REPETITION_PENALTY,
            FALLBACK_REPETITION_CONTEXT_SIZE,
        )
        assert d.top_k is None and d.min_p is None and d.presence_penalty is None
        assert d.source == "none"
        assert d.evidence is None
        assert d.base_repo is None

    def test_fallback_constants_are_the_129_values(self):
        # The #129 campaign validated 0.2 / 0.95 / 1024 + 1.1 x 64.
        assert (FALLBACK_TEMPERATURE, FALLBACK_TOP_P, FALLBACK_MAX_TOKENS) == (0.2, 0.95, 1024)
        assert (FALLBACK_REPETITION_PENALTY, FALLBACK_REPETITION_CONTEXT_SIZE) == (1.1, 64)

    def test_object_without_generation_hints_attribute_is_none(self):
        d = resolve_sampling_defaults(types.SimpleNamespace(type="x", link="y", name="z"))
        assert d.source == "none"

    def test_non_dict_hints_are_ignored(self):
        d = resolve_sampling_defaults(_Llm("garbage"))
        assert d.source == "none"

    def test_optional_keys_absent_from_wire_dict(self):
        wire = resolve_sampling_defaults(_Llm(None)).wire_kwargs()
        assert wire == {
            "repetition_penalty": FALLBACK_REPETITION_PENALTY,
            "repetition_context_size": FALLBACK_REPETITION_CONTEXT_SIZE,
        }

    def test_facts_without_sampling_values_are_none_but_kept(self):
        # Gated base / vendor without numbers: the facts still ride along for
        # the UI and the max-tokens cap, the sampling stays neutral.
        d = resolve_sampling_defaults(_Llm(dict(_hints(), context_length=8192)))
        assert d.source == "none"
        assert d.base_repo == "Qwen/Qwen3-0.6B"
        assert d.max_tokens_cap == 8192
        assert (d.temperature, d.top_p) == (FALLBACK_TEMPERATURE, FALLBACK_TOP_P)


class TestResolveGenerationConfig:
    def test_qwen3_config_is_taken_as_shipped(self):
        d = resolve_sampling_defaults(
            _Llm(_hints(temperature=0.6, top_p=0.95, top_k=20, do_sample=True))
        )
        assert (d.temperature, d.top_p, d.top_k) == (0.6, 0.95, 20)
        assert d.min_p is None and d.presence_penalty is None
        assert d.source == "base_generation_config"
        assert d.evidence is None
        assert d.base_repo == "Qwen/Qwen3-0.6B"
        assert d.max_tokens == FALLBACK_MAX_TOKENS  # never taken from HF

    def test_wire_kwargs_carry_only_defined_optional_keys(self):
        wire = resolve_sampling_defaults(
            _Llm(
                _hints(
                    temperature=0.7,
                    top_k=20,
                    min_p=0.0,
                    presence_penalty=1.5,
                    repetition_penalty=1.05,
                )
            )
        ).wire_kwargs()
        assert wire == {
            "repetition_penalty": 1.05,
            "repetition_context_size": FALLBACK_REPETITION_CONTEXT_SIZE,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
        }

    def test_per_key_fall_through(self):
        # Only temperature shipped: top_p keeps the fallback.
        d = resolve_sampling_defaults(_Llm(_hints(temperature=0.7)))
        assert d.temperature == 0.7
        assert d.top_p == FALLBACK_TOP_P
        assert d.source == "base_generation_config"

    def test_do_sample_false_ignores_the_whole_block(self):
        d = resolve_sampling_defaults(
            _Llm(_hints(temperature=0.9, top_p=0.5, top_k=50, do_sample=False))
        )
        assert (d.temperature, d.top_p, d.top_k) == (FALLBACK_TEMPERATURE, FALLBACK_TOP_P, None)
        assert d.source == "none"

    def test_vendor_greedy_is_sent_as_exactly_zero(self):
        # top_k == 1 / temperature < 0.05 means "the vendor says greedy". The
        # servers only take the argmax path on temp == 0 and otherwise divide the
        # logits by it: Qwen2.5-VL's shipped 1e-06 overflowed into 2048 x "!"
        # on the 2.0.0 QA pass. Normalise to an exact 0.0, keep the other keys.
        d = resolve_sampling_defaults(
            _Llm(_hints(temperature=1e-06, top_p=0.95, repetition_penalty=1.05))
        )
        assert d.temperature == 0.0
        assert (d.top_p, d.repetition_penalty) == (0.95, 1.05)
        assert d.source == "base_generation_config"
        d = resolve_sampling_defaults(_Llm(_hints(temperature=0.01, top_p=0.001, top_k=1)))
        assert (d.temperature, d.top_p, d.top_k) == (0.0, 0.001, 1)
        d = resolve_sampling_defaults(_Llm(_hints(temperature=0.7, top_k=1)))
        assert d.temperature == 0.0  # top_k == 1 alone means greedy
        d = resolve_sampling_defaults(_Llm(_hints(temperature=0.0)))
        assert d.temperature == 0.0
        d = resolve_sampling_defaults(_Llm(_hints(temperature=0.05)))
        assert d.temperature == 0.05  # threshold is exclusive

    def test_clamps_and_drops_out_of_range_keys(self):
        d = resolve_sampling_defaults(
            _Llm(
                _hints(
                    temperature=5.0,
                    top_p=1.5,
                    top_k=-3,
                    min_p=2.0,
                    presence_penalty=9.0,
                    repetition_penalty=-1,
                )
            )
        )
        assert d.temperature == 2.0  # clamped
        assert d.top_p == FALLBACK_TOP_P  # (0, 1] violated -> dropped
        assert d.top_k is None and d.min_p is None and d.presence_penalty is None
        assert d.repetition_penalty == FALLBACK_REPETITION_PENALTY

    def test_non_numeric_values_are_dropped(self):
        d = resolve_sampling_defaults(_Llm(_hints(temperature="hot", top_k="many", top_p=None)))
        assert d.temperature == FALLBACK_TEMPERATURE
        assert d.top_k is None
        assert d.source == "none"

    def test_empty_block_is_none(self):
        d = resolve_sampling_defaults(_Llm(_hints()))
        assert d.source == "none"
        # Facts still ride along for the UI / cap even without sampling values.
        assert d.base_repo == "Qwen/Qwen3-0.6B"


class TestResolveSourceAndEvidence:
    def test_source_mirrors_the_capture_stage(self):
        for stage in ("base_generation_config", "quant_generation_config", "model_card"):
            hints = dict(_hints(temperature=0.7), source_stage=stage)
            assert resolve_sampling_defaults(_Llm(hints)).source == stage

    def test_legacy_hints_without_stage_are_the_base_generation_config(self):
        # Rows captured by #389 (bundled snapshots) predate ``source_stage``; the
        # only thing #389 ever read was the base repo's generation_config.json.
        hints = _hints(temperature=0.7)
        hints.pop("source_stage")
        hints.pop("evidence")
        assert resolve_sampling_defaults(_Llm(hints)).source == "base_generation_config"

    def test_unknown_stage_is_reported_as_the_base_generation_config(self):
        hints = dict(_hints(temperature=0.7), source_stage="garbage")
        assert resolve_sampling_defaults(_Llm(hints)).source == "base_generation_config"

    def test_model_card_evidence_rides_along(self):
        hints = dict(
            _hints(temperature=0.15),
            source_stage="model_card",
            evidence="We recommend `temperature=0.15`.",
        )
        d = resolve_sampling_defaults(_Llm(hints))
        assert d.source == "model_card"
        assert d.evidence == "We recommend `temperature=0.15`."
        assert d.to_dict()["evidence"] == "We recommend `temperature=0.15`."

    def test_unusable_block_is_none_whatever_the_stage_says(self):
        hints = dict(_hints(do_sample=False), source_stage="model_card", evidence="x")
        assert resolve_sampling_defaults(_Llm(hints)).source == "none"

    def test_no_curated_table_left(self):
        # Maintainer decision: no hand-curated sampling table in the repo.
        assert not hasattr(gh, "SAMPLING_PROFILES")
        assert not hasattr(gh, "CuratedProfile")


class TestMaxTokensCap:
    def test_unbounded_on_mlx_without_context_length(self):
        d = resolve_sampling_defaults(_Llm(None))
        assert d.max_tokens_cap == UNBOUNDED_CONTEXT_TOKENS

    def test_context_length_caps_on_mlx(self):
        d = resolve_sampling_defaults(_Llm(dict(_hints(), context_length=8192)))
        assert d.max_tokens_cap == 8192

    def test_engine_context_window_caps_on_llama_cpp(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_Engine", _LlamaEngine)
        assert resolve_sampling_defaults(_Llm(None)).max_tokens_cap == 4096
        assert (
            resolve_sampling_defaults(_Llm(dict(_hints(), context_length=2048))).max_tokens_cap
            == 2048
        )

    def test_max_tokens_never_exceeds_cap(self):
        d = resolve_sampling_defaults(_Llm(dict(_hints(), context_length=512)))
        assert d.max_tokens == 512

    def test_no_engine_selected_is_unbounded(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_Engine", None)
        assert resolve_sampling_defaults(_Llm(None)).max_tokens_cap == UNBOUNDED_CONTEXT_TOKENS

    def test_real_engines_expose_context_window(self, monkeypatch):
        # Deliberately inverted from the pre-dynamic-context behaviour: the
        # llama.cpp engines no longer declare a hardcoded 4096 ceiling. Unset
        # ERUDI_CTX means "no configured ceiling" (the engine's own fit
        # resolves the ALLOCATED window at load; read it via
        # ``effective_context_tokens``); ERUDI_CTX set is the explicit pin.
        from src.engines.base_engine import BaseEngine
        from src.engines.base_llama_cpp_engine import BaseLlamaCppEngine

        assert BaseEngine.max_context_tokens() is None
        monkeypatch.delenv("ERUDI_CTX", raising=False)
        assert BaseLlamaCppEngine.max_context_tokens() is None
        monkeypatch.setenv("ERUDI_CTX", "8192")
        assert BaseLlamaCppEngine.max_context_tokens() == 8192

    def test_invalid_erudi_ctx_degrades_to_no_ceiling(self, monkeypatch):
        from src.engines.base_llama_cpp_engine import BaseLlamaCppEngine

        monkeypatch.setenv("ERUDI_CTX", "not-a-number")
        assert BaseLlamaCppEngine.max_context_tokens() is None
        monkeypatch.setenv("ERUDI_CTX", "")
        assert BaseLlamaCppEngine.max_context_tokens() is None


class TestToDict:
    def test_to_dict_is_the_api_shape(self):
        d = resolve_sampling_defaults(_Llm(_hints(temperature=0.6, top_k=20))).to_dict()
        assert set(d) == {
            "temperature",
            "top_p",
            "max_tokens",
            "top_k",
            "min_p",
            "repetition_penalty",
            "repetition_context_size",
            "presence_penalty",
            "max_tokens_cap",
            "source",
            "base_repo",
            "evidence",
        }


# ---------------------------------------------------------------- model card

# Excerpts of the real cards (2026-08-28), kept verbatim so the extractor is
# tested against what vendors actually write, typos included.
MISTRAL_CARD = """\
### vLLM

We recommend using this model with the [vLLM library](https://github.com/vllm-project/vllm)
to implement production-ready inference pipelines.

**Note 1**: We recommond using a relatively low temperature, such as `temperature=0.15`.

**Note 2**: Make sure to add a system prompt to the model to best tailer it for your needs.

```py
# note that running this model on GPU requires over 60 GB of GPU RAM
llm = LLM(model=model_name, tokenizer_mode="mistral", tensor_parallel_size=8)

sampling_params = SamplingParams(max_tokens=512, temperature=0.15)
outputs = llm.chat(messages, sampling_params=sampling_params)
```
"""

QWEN35_CARD_TIP = """\
```shell
export OPENAI_BASE_URL="http://localhost:8000/v1"
```

> [!Tip]
> We recommend using the following set of sampling parameters for generation
> - Thinking mode for general tasks: `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`
> - Thinking mode for precise coding tasks (e.g. WebDev): `temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0`
> - Instruct (or non-thinking) mode for general tasks: `temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`
> - Instruct (or non-thinking) mode for reasoning tasks: `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`
>
> Please note that the support for sampling parameters varies according to inference frameworks.

#### Text-Only Input

```python
chat_response = client.chat.completions.create(
    model="Qwen/Qwen3.5-4B",
    max_tokens=81920,
    temperature=1.0,
    top_p=0.95,
    presence_penalty=1.5,
)
```
"""

QWEN35_CARD_BEST_PRACTICES = """\
> It is also recommended to modify the `factor` as needed.

## Best Practices

To achieve optimal performance, we recommend the following settings:

1. **Sampling Parameters**:
   - We suggest using the following sets of sampling parameters depending on the mode and task type:
     - **Thinking mode for general tasks**:
       `temperature=1.0`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`
     - **Thinking mode for precise coding tasks (e.g., WebDev)**:
       `temperature=0.6`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0`
     - **Instruct (or non-thinking) mode for general tasks**:
       `temperature=0.7`, `top_p=0.8`, `top_k=20`, `min_p=0.0`, `presence_penalty=1.5`, `repetition_penalty=1.0`
     - **Instruct (or non-thinking) mode for reasoning tasks**:
       `temperature=1.0`, `top_p=1.0`, `top_k=40`, `min_p=0.0`, `presence_penalty=2.0`, `repetition_penalty=1.0`
   - For supported frameworks, you can adjust the `presence_penalty` parameter between 0 and 2 to reduce endless repetitions.

2. **Adequate Output Length**: We recommend using an output length of 32,768 tokens for most queries.

## Citation
"""

AYA_CARD = """\
## Usage

Please install transformers from the source repository.

```python
gen_tokens = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3,
    )

gen_text = tokenizer.decode(gen_tokens[0])
```

### Model Details

**Input**: Models input text only.
"""

PHI_CARD = """\
## Usage

```python
sampling_params = SamplingParams(
  max_tokens=500,
  temperature=0.0,
)
```

```python
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}
```

Developers should apply responsible AI best practices, including mapping, measuring, and mitigating risks.
"""

LLAMA_CARD = """\
**Approach:** Llama is a foundational technology designed to be used in a variety of use cases.

**Technological Advancement:** Llama releases usually introduce new capabilities that require specific considerations in addition to the best practices that generally apply across all Generative AI use cases.
"""

GEMMA_CARD = """\
<!-- temperature=0.9 is not a recommendation, it lives in an HTML comment -->
# Gemma 3 model card

**Model Page**: [Gemma](https://ai.google.dev/gemma/docs/core)

Gemma is a family of lightweight, state-of-the-art open models from Google.
"""


class TestExtractCardRecommendations:
    def test_mistral_prose_recommendation(self):
        found = extract_card_recommendations(MISTRAL_CARD)
        assert len(found) == 1
        assert found[0]["values"] == {"temperature": 0.15}
        assert found[0]["line"] == (
            "**Note 1**: We recommond using a relatively low temperature, such as "
            "`temperature=0.15`."
        )

    def test_qwen35_tip_block_is_mode_aware(self):
        found = extract_card_recommendations(QWEN35_CARD_TIP)
        # The four bullets under the "We recommend" line; nothing from the
        # python snippet below them.
        assert [c["values"]["temperature"] for c in found] == [1.0, 0.6, 0.7, 1.0]
        thinking = select_card_recommendation(found, supports_thinking=True)
        assert thinking["values"] == {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
        }
        assert thinking["line"].startswith("Thinking mode for general tasks:")
        non_thinking = select_card_recommendation(found, supports_thinking=False)
        assert non_thinking["values"]["temperature"] == 0.7
        assert non_thinking["values"]["top_p"] == 0.8
        assert non_thinking["line"].startswith("Instruct (or non-thinking) mode for general tasks:")
        # Unknown mode: the non-thinking line as well (that is what a chat
        # without enable_thinking runs).
        assert select_card_recommendation(found, supports_thinking=None) is non_thinking

    def test_qwen35_best_practices_heading_with_values_on_the_next_line(self):
        found = extract_card_recommendations(QWEN35_CARD_BEST_PRACTICES)
        assert [c["values"]["temperature"] for c in found] == [1.0, 0.6, 0.7, 1.0]
        thinking = select_card_recommendation(found, supports_thinking=True)
        assert thinking["values"]["temperature"] == 1.0
        assert thinking["line"].startswith("**Thinking mode for general tasks**:")
        assert "`temperature=1.0`" in thinking["line"]
        non_thinking = select_card_recommendation(found, supports_thinking=False)
        assert (non_thinking["values"]["temperature"], non_thinking["values"]["top_p"]) == (
            0.7,
            0.8,
        )

    def test_aya_code_example_is_not_a_recommendation(self):
        assert extract_card_recommendations(AYA_CARD) == []

    def test_phi_snippets_are_not_a_recommendation(self):
        assert extract_card_recommendations(PHI_CARD) == []

    def test_llama_and_gemma_cards_have_nothing(self):
        assert extract_card_recommendations(LLAMA_CARD) == []
        assert extract_card_recommendations(GEMMA_CARD) == []

    def test_number_without_cue_is_ignored(self):
        assert extract_card_recommendations("The default is temperature=0.7 here.") == []

    def test_cue_without_number_is_ignored(self):
        assert extract_card_recommendations("We recommend a low temperature.") == []

    def test_unfenced_trailing_code_is_dropped(self):
        text = "We recommend `top_p=0.9`.\n```python\nsampling = dict(temperature=0.1)\n"
        found = extract_card_recommendations(text)
        assert [c["values"] for c in found] == [{"top_p": 0.9}]

    def test_camel_case_keys_and_colon_separator(self):
        # Qwen3 style: "Temperature=0.6, TopP=0.95, TopK=20, and MinP=0".
        text = "## Best Practices\n\nFor thinking mode, use Temperature=0.6, TopP=0.95, TopK=20, and MinP=0."
        found = extract_card_recommendations(text)
        assert found[0]["values"] == {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
        text = "We suggest temperature: 0.3 and top-p: 0.9 for chat."
        assert extract_card_recommendations(text)[0]["values"] == {"temperature": 0.3, "top_p": 0.9}

    def test_out_of_range_numbers_are_dropped_like_stage_one(self):
        assert extract_card_recommendations("We recommend `top_p=1.5`.") == []
        found = extract_card_recommendations("We recommend `temperature=3`, `top_k=20`.")
        assert found[0]["values"] == {"temperature": 2.0, "top_k": 20}

    def test_heading_scope_ends_at_the_next_heading(self):
        text = (
            "## Recommended settings\n\n- Chat: `temperature=0.5`\n\n## Other\n\n"
            "- Something with `temperature=0.9`\n"
        )
        assert [c["values"] for c in extract_card_recommendations(text)] == [{"temperature": 0.5}]

    def test_intro_cue_scope_ends_at_the_next_paragraph(self):
        text = (
            "We recommend the following:\n- `temperature=0.5`\n\n"
            "Another paragraph.\n- `temperature=0.9`\n"
        )
        assert [c["values"] for c in extract_card_recommendations(text)] == [{"temperature": 0.5}]

    def test_select_on_nothing(self):
        assert select_card_recommendation([], supports_thinking=True) is None

    def test_select_first_when_no_mode_line_matches(self):
        found = extract_card_recommendations(MISTRAL_CARD)
        assert select_card_recommendation(found, supports_thinking=True) is found[0]
        assert select_card_recommendation(found, supports_thinking=False) is found[0]


# ---------------------------------------------------------------- capture


class TestBuildGenerationHints:
    def test_whitelists_generation_config_keys(self):
        hints = build_generation_hints(
            base_repo="Qwen/Qwen3-0.6B",
            generation_config={
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "do_sample": True,
                "bos_token_id": 1,
                "eos_token_id": [2, 3],
                "transformers_version": "4.51",
            },
            config={"max_position_embeddings": 40960},
            chat_template="{% if enable_thinking %}...{% endif %}",
            captured_at="2026-08-28",
        )
        assert hints == {
            "base_repo": "Qwen/Qwen3-0.6B",
            "generation_config": {
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "do_sample": True,
            },
            "supports_thinking": True,
            "context_length": 40960,
            "captured_at": "2026-08-28",
            "source_stage": "base_generation_config",
            "evidence": None,
        }

    def test_stage_and_evidence_are_stored_verbatim(self):
        hints = build_generation_hints(
            base_repo="x/y",
            generation_config={"temperature": 0.15},
            config=None,
            chat_template=None,
            captured_at="d",
            source_stage="model_card",
            evidence="We recommend `temperature=0.15`.",
        )
        assert hints["source_stage"] == "model_card"
        assert hints["evidence"] == "We recommend `temperature=0.15`."

    def test_nothing_captured_is_none(self):
        assert (
            build_generation_hints(
                base_repo="x/y", generation_config=None, config=None, chat_template=None
            )
            is None
        )

    def test_partial_capture_keeps_unknowns_none(self):
        hints = build_generation_hints(
            base_repo="x/y",
            generation_config={"temperature": 1.0},
            config=None,
            chat_template=None,
            captured_at="d",
        )
        assert hints["context_length"] is None
        assert hints["supports_thinking"] is None
        assert hints["generation_config"] == {"temperature": 1.0}
        assert hints["source_stage"] == "base_generation_config"
        assert hints["evidence"] is None

    def test_context_length_from_nested_text_config(self):
        # VLM configs (Qwen2.5-VL) nest the text model's window.
        hints = build_generation_hints(
            base_repo="x/y",
            generation_config=None,
            config={"text_config": {"max_position_embeddings": 128000}},
            chat_template="plain",
            captured_at="d",
        )
        assert hints["context_length"] == 128000
        assert hints["supports_thinking"] is False
        assert "generation_config" not in hints
        assert hints["source_stage"] is None
        assert hints["evidence"] is None

    def test_chat_template_list_form(self):
        hints = build_generation_hints(
            base_repo="x/y",
            generation_config=None,
            config=None,
            chat_template=[
                {"name": "default", "template": "a"},
                {"name": "think", "template": "{{ enable_thinking }}"},
            ],
            captured_at="d",
        )
        assert hints["supports_thinking"] is True


class TestContextLengthSources:
    """The window is read from several config shapes (a field audit of the
    uncovered catalog entries): the key lives at the top level or inside a
    ``*_config`` container, and some lineages name it ``n_positions`` (phi-2) or
    ``max_sequence_length`` (Nemotron)."""

    @staticmethod
    def _window(config):
        hints = build_generation_hints(
            base_repo="x/y",
            generation_config=None,
            config=config,
            chat_template="plain",
            captured_at="d",
        )
        return hints["context_length"]

    def test_language_config_container(self):
        # deepseek-vl nests the text model under language_config.
        assert self._window({"language_config": {"max_position_embeddings": 16384}}) == 16384

    def test_llm_config_container(self):
        assert self._window({"llm_config": {"max_position_embeddings": 8192}}) == 8192

    def test_n_positions_alias(self):
        assert self._window({"n_positions": 2048}) == 2048

    def test_max_sequence_length_alias(self):
        assert self._window({"max_sequence_length": 131072}) == 131072

    def test_alias_inside_a_container(self):
        assert self._window({"text_config": {"n_positions": 4096}}) == 4096

    def test_max_position_embeddings_beats_an_alias_key(self):
        assert self._window({"n_positions": 2048, "max_position_embeddings": 32768}) == 32768
        assert self._window({"max_sequence_length": 512, "max_position_embeddings": 4096}) == 4096

    def test_the_preferred_key_wins_even_from_a_container(self):
        assert (
            self._window({"n_positions": 2048, "llm_config": {"max_position_embeddings": 16384}})
            == 16384
        )

    def test_top_level_beats_a_container_for_the_same_key(self):
        assert (
            self._window(
                {
                    "max_position_embeddings": 8192,
                    "language_config": {"max_position_embeddings": 16384},
                }
            )
            == 8192
        )

    def test_junk_values_are_ignored(self):
        assert self._window({"max_position_embeddings": 0, "n_positions": -1}) is None
        assert self._window({"max_sequence_length": "131072"}) is None
        assert self._window({"max_position_embeddings": None, "llm_config": None}) is None

    def test_a_junk_preferred_key_falls_through_to_the_next_source(self):
        assert (
            self._window({"max_position_embeddings": 0, "text_config": {"n_positions": 4096}})
            == 4096
        )


def _hub(tmp_path, repos, gated=()):
    """hf_api stub over ``repos`` (repo id -> {filename: obj/str}). A repo in
    ``gated`` serves README.md only: every other file raises GatedRepoError,
    like the Hub does anonymously."""
    api = MagicMock()

    def download(repo_id, filename, **_):
        from huggingface_hub.errors import EntryNotFoundError, GatedRepoError

        if repo_id in gated and filename != "README.md":
            raise GatedRepoError("401", response=MagicMock(status_code=401))
        files = repos.get(repo_id, {})
        if filename not in files:
            raise EntryNotFoundError(f"{repo_id}/{filename} missing")
        path = tmp_path / repo_id.replace("/", "__") / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = files[filename]
        path.write_text(
            payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8"
        )
        return str(path)

    api.hf_hub_download.side_effect = download
    return api


def _calls(api):
    return [
        (c.kwargs.get("repo_id") or c.args[0], c.kwargs.get("filename") or c.args[1])
        for c in api.hf_hub_download.call_args_list
    ]


def _fake_hf_api(tmp_path, files):
    """Single-repo stub: ``files`` serve for any repo id."""
    api = MagicMock()

    def download(repo_id, filename, **_):
        if filename not in files:
            from huggingface_hub.errors import EntryNotFoundError

            raise EntryNotFoundError(f"{filename} missing")
        path = tmp_path / repo_id.replace("/", "__") / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = files[filename]
        path.write_text(
            payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8"
        )
        return str(path)

    api.hf_hub_download.side_effect = download
    return api


def _set_gguf(api, by_repo):
    """Fake ``model_info(repo_id=..., expand=["gguf"])`` on a stub: maps a repo id
    to the ``gguf`` block the Hub answers for it (an ``Exception`` instance is
    raised instead, an id absent from the map answers no block at all)."""

    def model_info(repo_id=None, expand=None, **_):
        payload = by_repo.get(repo_id)
        if isinstance(payload, Exception):
            raise payload
        return types.SimpleNamespace(gguf=payload)

    api.model_info.side_effect = model_info
    return api


def _gguf_call(api):
    """``(repo_id, expand)`` of the single gguf-expand call made on ``api``."""
    assert api.model_info.call_count == 1
    call = api.model_info.call_args
    return (call.kwargs.get("repo_id") or (call.args[0] if call.args else None)), call.kwargs.get(
        "expand"
    )


_FACTS = {
    "config.json": {"max_position_embeddings": 40960},
    "tokenizer_config.json": {"chat_template": "{% if enable_thinking %}{% endif %}"},
}


class TestCaptureCascade:
    def test_stage_one_base_generation_config_wins(self, tmp_path):
        api = _hub(
            tmp_path,
            {
                "Qwen/Qwen3-0.6B": dict(
                    _FACTS,
                    **{
                        "generation_config.json": {
                            "temperature": 0.6,
                            "top_k": 20,
                            "eos_token_id": 1,
                        },
                        "README.md": "We recommend `temperature=0.9`.",
                    },
                ),
                "mlx-community/Qwen3-0.6B-4bit": {"generation_config.json": {"temperature": 0.1}},
            },
        )
        hints = capture_generation_hints(
            "Qwen/Qwen3-0.6B", api, quant_repo="mlx-community/Qwen3-0.6B-4bit"
        )
        assert hints["generation_config"] == {"temperature": 0.6, "top_k": 20}
        assert hints["source_stage"] == "base_generation_config"
        assert hints["evidence"] is None
        assert hints["context_length"] == 40960
        assert hints["supports_thinking"] is True
        assert hints["base_repo"] == "Qwen/Qwen3-0.6B"
        # Stopped at stage 1: neither the quant repo nor the card was fetched.
        repos = {r for r, _ in _calls(api)}
        assert repos == {"Qwen/Qwen3-0.6B"}
        assert ("Qwen/Qwen3-0.6B", "README.md") not in _calls(api)

    def test_stage_two_quant_generation_config(self, tmp_path):
        api = _hub(
            tmp_path,
            {
                "google/gemma-3-270m-it": {"README.md": "We recommend `temperature=0.9`."},
                "mlx-community/gemma-3-270m-it-4bit": dict(
                    _FACTS,
                    **{
                        "generation_config.json": {
                            "do_sample": True,
                            "top_k": 64,
                            "top_p": 0.95,
                            "cache_implementation": "hybrid",
                        }
                    },
                ),
            },
            gated=("google/gemma-3-270m-it",),
        )
        hints = capture_generation_hints(
            "google/gemma-3-270m-it", api, quant_repo="mlx-community/gemma-3-270m-it-4bit"
        )
        assert hints["generation_config"] == {"do_sample": True, "top_k": 64, "top_p": 0.95}
        assert hints["source_stage"] == "quant_generation_config"
        assert hints["evidence"] is None
        assert hints["base_repo"] == "google/gemma-3-270m-it"
        # The gated base hides its config: the facts come from the quant repo.
        assert hints["context_length"] == 40960
        assert hints["supports_thinking"] is True
        assert ("google/gemma-3-270m-it", "README.md") not in _calls(api)

    def test_stage_three_model_card(self, tmp_path):
        api = _hub(
            tmp_path,
            {
                "mistralai/Mistral-Small": dict(
                    _FACTS,
                    **{
                        "generation_config.json": {"do_sample": True, "bos_token_id": 1},
                        "README.md": MISTRAL_CARD,
                    },
                ),
                "mlx-community/Mistral-Small-4bit": {},
            },
        )
        hints = capture_generation_hints(
            "mistralai/Mistral-Small", api, quant_repo="mlx-community/Mistral-Small-4bit"
        )
        assert hints["generation_config"] == {"temperature": 0.15}
        assert hints["source_stage"] == "model_card"
        assert hints["evidence"] == (
            "**Note 1**: We recommond using a relatively low temperature, such as "
            "`temperature=0.15`."
        )
        assert hints["context_length"] == 40960
        assert ("mlx-community/Mistral-Small-4bit", "generation_config.json") in _calls(api)

    def test_stage_three_is_mode_aware(self, tmp_path):
        thinking = dict(_FACTS, **{"README.md": QWEN35_CARD_BEST_PRACTICES})
        api = _hub(tmp_path, {"Qwen/Qwen3.5-4B": thinking})
        hints = capture_generation_hints("Qwen/Qwen3.5-4B", api)
        assert hints["supports_thinking"] is True
        assert hints["generation_config"]["temperature"] == 1.0
        assert hints["generation_config"]["presence_penalty"] == 1.5
        assert hints["source_stage"] == "model_card"
        assert "Thinking mode for general tasks" in hints["evidence"]

        gh.reset_capture_cache()
        non_thinking = dict(thinking, **{"tokenizer_config.json": {"chat_template": "plain"}})
        api = _hub(tmp_path, {"Qwen/Qwen3.5-4B": non_thinking})
        hints = capture_generation_hints("Qwen/Qwen3.5-4B", api)
        assert hints["supports_thinking"] is False
        assert (hints["generation_config"]["temperature"], hints["generation_config"]["top_p"]) == (
            0.7,
            0.8,
        )
        assert "non-thinking" in hints["evidence"]

    def test_gated_base_readme_is_read_anonymously(self, tmp_path):
        api = _hub(
            tmp_path,
            {
                "meta-llama/Llama-3.2-1B-Instruct": {"README.md": "We suggest `temperature=0.6`."},
                "mlx-community/Llama-3.2-1B-Instruct-4bit": _FACTS,
            },
            gated=("meta-llama/Llama-3.2-1B-Instruct",),
        )
        hints = capture_generation_hints(
            "meta-llama/Llama-3.2-1B-Instruct",
            api,
            quant_repo="mlx-community/Llama-3.2-1B-Instruct-4bit",
        )
        assert hints["generation_config"] == {"temperature": 0.6}
        assert hints["source_stage"] == "model_card"
        assert hints["context_length"] == 40960
        # A gated repo is probed once for its files, then only its README.
        gated_calls = [f for r, f in _calls(api) if r == "meta-llama/Llama-3.2-1B-Instruct"]
        assert gated_calls == ["generation_config.json", "README.md"]

    def test_nothing_found_keeps_the_facts_without_sampling(self, tmp_path):
        api = _hub(
            tmp_path,
            {
                "meta-llama/Llama-3.2-1B-Instruct": {"README.md": LLAMA_CARD},
                "mlx-community/Llama-3.2-1B-Instruct-4bit": _FACTS,
            },
            gated=("meta-llama/Llama-3.2-1B-Instruct",),
        )
        hints = capture_generation_hints(
            "meta-llama/Llama-3.2-1B-Instruct",
            api,
            quant_repo="mlx-community/Llama-3.2-1B-Instruct-4bit",
        )
        assert "generation_config" not in hints
        assert hints["source_stage"] is None
        assert hints["evidence"] is None
        assert hints["base_repo"] == "meta-llama/Llama-3.2-1B-Instruct"
        assert hints["context_length"] == 40960
        assert hints["supports_thinking"] is True
        assert resolve_sampling_defaults(_Llm(hints)).source == "none"

    def test_unusable_base_config_falls_through_to_the_card(self, tmp_path):
        api = _hub(
            tmp_path,
            {
                "org/m": {
                    "generation_config.json": {"do_sample": False, "temperature": 0.9},
                    "README.md": "We recommend `temperature=0.3`.",
                }
            },
        )
        hints = capture_generation_hints("org/m", api)
        assert hints["generation_config"] == {"temperature": 0.3}
        assert hints["source_stage"] == "model_card"

    def test_memoized_per_base_and_quant_pair(self, tmp_path):
        api = _hub(
            tmp_path,
            {
                "org/base": {
                    "README.md": "nothing here",
                    "config.json": {"max_position_embeddings": 8},
                },
                "q/a": {"generation_config.json": {"temperature": 0.4}},
                "q/b": {},
            },
        )
        a = capture_generation_hints("org/base", api, quant_repo="q/a")
        assert capture_generation_hints("org/base", api, quant_repo="q/a") == a
        n = api.hf_hub_download.call_count
        capture_generation_hints("org/base", api, quant_repo="q/a")
        assert api.hf_hub_download.call_count == n  # pair memoized
        b = capture_generation_hints("org/base", api, quant_repo="q/b")
        assert b["source_stage"] is None
        # The base repo's files were fetched once for both pairs.
        base_calls = [f for r, f in _calls(api) if r == "org/base"]
        assert len(base_calls) == len(set(base_calls))
        a["generation_config"]["temperature"] = 9  # copies, not shared state
        assert (
            capture_generation_hints("org/base", api, quant_repo="q/a")["generation_config"][
                "temperature"
            ]
            == 0.4
        )


class TestGgufContextLength:
    """For a GGUF entry the authoritative window is the one the .gguf file
    declares (``n_ctx_train``), which the Hub exposes through
    ``model_info(expand=["gguf"])`` on the QUANT repo -- it is what llama-server
    reads, and it disagrees with some base configs by a factor of 2 to 4."""

    def _repos(self, extra_base=None):
        return {
            "Qwen/Qwen3-4B": dict(_FACTS, **(extra_base or {})),
            "bartowski/Qwen3-4B-GGUF": {},
        }

    def _capture(self, api, **kwargs):
        return capture_generation_hints(
            "Qwen/Qwen3-4B",
            api,
            quant_repo="bartowski/Qwen3-4B-GGUF",
            **kwargs,
        )

    def test_gguf_metadata_beats_the_base_config(self, tmp_path):
        api = _set_gguf(
            _hub(tmp_path, self._repos({"config.json": {"max_position_embeddings": 131072}})),
            {"bartowski/Qwen3-4B-GGUF": {"context_length": 40960}},
        )
        hints = self._capture(api, quant_format="gguf")
        assert hints["context_length"] == 40960
        # The rest of the capture is untouched.
        assert hints["supports_thinking"] is True
        assert _gguf_call(api) == ("bartowski/Qwen3-4B-GGUF", ["gguf"])

    def test_gguf_block_may_be_an_object(self, tmp_path):
        api = _set_gguf(
            _hub(tmp_path, self._repos()),
            {"bartowski/Qwen3-4B-GGUF": types.SimpleNamespace(context_length=262144)},
        )
        assert self._capture(api, quant_format="gguf")["context_length"] == 262144

    def test_a_failing_call_falls_back_to_config_json(self, tmp_path):
        api = _set_gguf(
            _hub(tmp_path, self._repos()),
            {"bartowski/Qwen3-4B-GGUF": RuntimeError("hub said no")},
        )
        assert self._capture(api, quant_format="gguf")["context_length"] == 40960

    def test_an_empty_block_falls_back_to_config_json(self, tmp_path):
        for payload in (None, {}, {"context_length": 0}, {"context_length": "40960"}):
            gh.reset_capture_cache()
            api = _set_gguf(_hub(tmp_path, self._repos()), {"bartowski/Qwen3-4B-GGUF": payload})
            assert self._capture(api, quant_format="gguf")["context_length"] == 40960

    def test_gguf_metadata_alone_captures_the_window(self, tmp_path):
        # Nothing readable on either repo: the window is still captured.
        api = _set_gguf(
            _hub(tmp_path, {"Qwen/Qwen3-4B": {}, "bartowski/Qwen3-4B-GGUF": {}}),
            {"bartowski/Qwen3-4B-GGUF": {"context_length": 32768}},
        )
        hints = self._capture(api, quant_format="gguf")
        assert hints["context_length"] == 32768
        assert hints["source_stage"] is None
        assert hints["supports_thinking"] is None

    def test_an_mlx_capture_never_asks_for_gguf_metadata(self, tmp_path):
        api = _set_gguf(
            _hub(tmp_path, self._repos()), {"bartowski/Qwen3-4B-GGUF": {"context_length": 1}}
        )
        assert self._capture(api, quant_format="mlx")["context_length"] == 40960
        api.model_info.assert_not_called()

    def test_an_unstated_format_never_asks(self, tmp_path):
        api = _set_gguf(
            _hub(tmp_path, self._repos()), {"bartowski/Qwen3-4B-GGUF": {"context_length": 1}}
        )
        assert self._capture(api)["context_length"] == 40960
        api.model_info.assert_not_called()

    def test_without_a_quant_repo_nothing_is_asked(self, tmp_path):
        api = _set_gguf(_hub(tmp_path, self._repos()), {})
        assert capture_generation_hints("Qwen/Qwen3-4B", api, quant_format="gguf")
        api.model_info.assert_not_called()

    def test_memoized_per_quant_repo(self, tmp_path):
        api = _set_gguf(
            _hub(
                tmp_path,
                {
                    "Qwen/Qwen3-4B": _FACTS,
                    "Qwen/Qwen3-8B": _FACTS,
                    "bartowski/Qwen3-4B-GGUF": {},
                },
            ),
            {"bartowski/Qwen3-4B-GGUF": {"context_length": 40960}},
        )
        for base in ("Qwen/Qwen3-4B", "Qwen/Qwen3-8B"):
            hints = capture_generation_hints(
                base, api, quant_repo="bartowski/Qwen3-4B-GGUF", quant_format="gguf"
            )
            assert hints["context_length"] == 40960
        assert api.model_info.call_count == 1


class TestCaptureGenerationHints:
    def test_captures_the_three_files(self, tmp_path):
        api = _fake_hf_api(
            tmp_path,
            {
                "generation_config.json": {"temperature": 0.6, "top_k": 20, "eos_token_id": 1},
                "config.json": {"max_position_embeddings": 40960},
                "tokenizer_config.json": {"chat_template": "{% if enable_thinking %}{% endif %}"},
            },
        )
        hints = capture_generation_hints("Qwen/Qwen3-0.6B", api)
        assert hints["generation_config"] == {"temperature": 0.6, "top_k": 20}
        assert hints["context_length"] == 40960
        assert hints["supports_thinking"] is True
        assert hints["base_repo"] == "Qwen/Qwen3-0.6B"
        assert hints["captured_at"]
        called = {c.kwargs.get("filename") or c.args[1] for c in api.hf_hub_download.call_args_list}
        assert called == {"generation_config.json", "config.json", "tokenizer_config.json"}

    def test_missing_generation_config_still_captures_facts(self, tmp_path):
        api = _fake_hf_api(
            tmp_path,
            {
                "config.json": {"max_position_embeddings": 8192},
                "tokenizer_config.json": {"chat_template": "x"},
            },
        )
        hints = capture_generation_hints("org/model", api)
        assert "generation_config" not in hints
        assert hints["context_length"] == 8192
        assert hints["supports_thinking"] is False
        assert hints["source_stage"] is None

    def test_chat_template_jinja_fallback(self, tmp_path):
        api = _fake_hf_api(
            tmp_path, {"tokenizer_config.json": {}, "chat_template.jinja": "{{ enable_thinking }}"}
        )
        assert capture_generation_hints("org/model", api)["supports_thinking"] is True

    def test_all_missing_is_none(self, tmp_path):
        assert capture_generation_hints("org/model", _fake_hf_api(tmp_path, {})) is None

    def test_any_failure_is_none_never_raises(self):
        api = MagicMock()
        api.hf_hub_download.side_effect = RuntimeError("network died")
        assert capture_generation_hints("org/model", api) is None

    def test_fully_gated_repo_is_none(self):
        from huggingface_hub.errors import GatedRepoError

        api = MagicMock()
        api.hf_hub_download.side_effect = GatedRepoError("403", response=MagicMock(status_code=403))
        assert capture_generation_hints("meta-llama/Llama-3.2-3B-Instruct", api) is None
        # Short-circuits: the first gated answer skips the other files; only
        # the README (public even when gated) is still tried.
        assert api.hf_hub_download.call_count == 2

    def test_no_api_or_repo_is_none(self):
        assert capture_generation_hints("org/model", None) is None
        assert capture_generation_hints("", MagicMock()) is None

    def test_memoized_per_base_repo(self, tmp_path):
        api = _fake_hf_api(tmp_path, {"config.json": {"max_position_embeddings": 1}})
        first = capture_generation_hints("org/model", api)
        second = capture_generation_hints("org/model", api)
        assert first == second
        # One pass (3 files + the chat_template.jinja fallback + the card), not two.
        assert api.hf_hub_download.call_count == 5
        second["context_length"] = 999  # copies, not shared state
        assert capture_generation_hints("org/model", api)["context_length"] == 1

    def test_failures_are_memoized_too(self):
        api = MagicMock()
        api.hf_hub_download.side_effect = RuntimeError("boom")
        capture_generation_hints("org/model", api)
        n = api.hf_hub_download.call_count
        capture_generation_hints("org/model", api)
        assert api.hf_hub_download.call_count == n

    def test_retrying_hf_api_exposes_hf_hub_download(self):
        # The capture rides the same retry wrapper as list_models/model_info.
        from src.core.config import _RetryingHfApi

        assert "hf_hub_download" in _RetryingHfApi.__dict__


class TestReadLocalGenerationHints:
    def test_reads_an_mlx_directory(self, tmp_path):
        (tmp_path / "generation_config.json").write_text(
            json.dumps({"temperature": 0.7, "top_p": 0.8, "pad_token_id": 0}), encoding="utf-8"
        )
        (tmp_path / "config.json").write_text(
            json.dumps({"max_position_embeddings": 32768}), encoding="utf-8"
        )
        (tmp_path / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": "no thinking here"}), encoding="utf-8"
        )
        hints = read_local_generation_hints(tmp_path, base_repo="mlx-community/x-4bit")
        assert hints["generation_config"] == {"temperature": 0.7, "top_p": 0.8}
        assert hints["context_length"] == 32768
        assert hints["supports_thinking"] is False
        assert hints["base_repo"] == "mlx-community/x-4bit"
        # The artifact IS the quant repo's copy of the file.
        assert hints["source_stage"] == "quant_generation_config"

    def test_unusable_local_config_leaves_no_stage(self, tmp_path):
        (tmp_path / "generation_config.json").write_text(
            json.dumps({"bos_token_id": 1}), encoding="utf-8"
        )
        (tmp_path / "config.json").write_text(
            json.dumps({"max_position_embeddings": 32768}), encoding="utf-8"
        )
        hints = read_local_generation_hints(tmp_path)
        assert "generation_config" not in hints
        assert hints["source_stage"] is None
        assert hints["context_length"] == 32768

    def test_context_length_aliases_apply_offline_too(self, tmp_path):
        # Same reader as the online capture: a downloaded artifact whose config
        # names the window ``n_positions`` still yields a window (what the MLX
        # spawn bound reads).
        (tmp_path / "config.json").write_text(json.dumps({"n_positions": 2048}), encoding="utf-8")
        assert read_local_generation_hints(tmp_path)["context_length"] == 2048

    def test_gguf_directory_has_nothing(self, tmp_path):
        (tmp_path / "model.gguf").write_bytes(b"GGUF")
        assert read_local_generation_hints(tmp_path) is None

    def test_broken_json_is_none(self, tmp_path):
        (tmp_path / "generation_config.json").write_text("{not json", encoding="utf-8")
        assert read_local_generation_hints(tmp_path) is None

    def test_missing_directory_is_none(self, tmp_path):
        assert read_local_generation_hints(tmp_path / "nope") is None


class TestResolveBaseRepo:
    def test_first_base_model_relation_wins(self):
        tags = ["mlx", "base_model:quantized:Qwen/Qwen3-0.6B", "base_model:finetune:x/y"]
        assert resolve_base_repo("mlx-community/Qwen3-0.6B-4bit", tags) == "Qwen/Qwen3-0.6B"

    def test_without_relation_the_repo_itself(self):
        assert resolve_base_repo("org/model", ["mlx", "conversational"]) == "org/model"
        assert resolve_base_repo("org/model", None) == "org/model"
