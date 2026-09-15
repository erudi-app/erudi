"""Per-artifact reasoning-lever verdict (1.1.2).

The verdict is read from the ARTIFACT, never from a family name or ``llm.type``
(a community fine-tune carries its parent's name and its own template), through
two sources:

1. llama.cpp engines -- ``chat_template_caps`` stamped on the engine handle by
   the one ``/props`` read at spawn. llama.cpp evaluates the template
   symbolically, so ``supports_reasoning_effort`` is exact and free. Its caps
   map carries NO ``enable_thinking`` entry (see ``common/jinja/caps.h``), so
   the toggle falls through to the probe below.
2. Every engine -- a DIFFERENTIAL template probe mirroring
   ``engines.system_role_capability``: render the generation prompt twice with
   ``reasoning_effort`` low vs high (differ -> the template reads the effort),
   then twice with ``enable_thinking`` true vs false (differ -> on/off only),
   else no lever. A template that cannot render at all is never blamed.

``is_thinker`` comes from the same probes: a template that differs under
``enable_thinking``, or whose generation prompt OPENS a thinking block, is a
reasoning model. ``generation_hints.supports_thinking`` stays a catalog display
hint and is never the runtime authority.

Fixtures are synthetic templates shaped like the real families (gpt-oss, Qwen3,
R1, Qwen2.5). No download, no network, no weights.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agents.reasoning_effort import ReasoningLever
from src.engines.reasoning_lever import (
    LeverVerdict,
    model_reasoning_lever,
    tokenizer_reasoning_lever,
)

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parent / "fixtures" / "chat_templates"


def _tokenizer_with_template(template: str):
    """A real ``PreTrainedTokenizerFast`` carrying ``template`` (the
    ``test_system_role_capability`` approach): real Jinja, no weights."""
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(
        models.WordLevel(vocab={"<s>": 0, "</s>": 1, "<unk>": 2}, unk_token="<unk>")
    )
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<s>", eos_token="</s>", unk_token="<unk>"
    )
    tok.chat_template = template
    return tok


def _fixture(name: str):
    return _tokenizer_with_template((FIXTURES / name).read_text())


# ===================== the differential probe =====================


def test_a_template_that_reads_the_effort_is_a_native_effort_lever():
    verdict = tokenizer_reasoning_lever(_fixture("reasoning-effort.jinja"))
    assert verdict == LeverVerdict(lever=ReasoningLever.NATIVE_EFFORT, is_thinker=True)


def test_a_template_that_only_reads_enable_thinking_is_a_toggle():
    verdict = tokenizer_reasoning_lever(_fixture("enable-thinking.jinja"))
    assert verdict == LeverVerdict(lever=ReasoningLever.NATIVE_TOGGLE, is_thinker=True)


def test_an_always_on_reasoner_has_no_lever_but_is_a_thinker():
    verdict = tokenizer_reasoning_lever(_fixture("always-thinking.jinja"))
    assert verdict == LeverVerdict(lever=ReasoningLever.NONE, is_thinker=True)


def test_a_plain_instruct_template_has_no_lever_and_does_not_think():
    verdict = tokenizer_reasoning_lever(_fixture("qwen2.5-instruct.jinja"))
    assert verdict == LeverVerdict(lever=ReasoningLever.NONE, is_thinker=False)


def test_a_closed_empty_thinking_block_is_not_an_open_one():
    # Guard for the marker heuristic: a template that emits "<think></think>"
    # has CLOSED the block, so it must not read as an always-on reasoner.
    tok = _tokenizer_with_template(
        "{% for m in messages %}{{ m['content'] }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<think>\\n\\n</think>\\n\\n' }}{% endif %}"
    )
    assert tokenizer_reasoning_lever(tok).is_thinker is False


class _Unrenderable:
    def apply_chat_template(self, conversation, **kwargs):
        raise ValueError("template unrenderable")


def test_an_unrenderable_template_is_never_blamed():
    # Graceful default: no lever, not a thinker -- the turn runs as it does
    # today rather than getting an instruction built on a failed probe.
    assert tokenizer_reasoning_lever(_Unrenderable()) == LeverVerdict(
        lever=ReasoningLever.NONE, is_thinker=False
    )


def test_a_missing_tokenizer_is_never_blamed():
    assert tokenizer_reasoning_lever(None) == LeverVerdict(
        lever=ReasoningLever.NONE, is_thinker=False
    )


# ===================== the GGUF template view forwards template kwargs =====================


def test_the_gguf_template_view_binds_extra_template_variables():
    # The probe is differential: without forwarding the bound variables the
    # two renders would be identical and every GGUF would read "no lever".
    from src.engines.gguf_chat_template import GgufChatTemplate

    view = GgufChatTemplate(
        chat_template=(FIXTURES / "reasoning-effort.jinja").read_text(),
        bos_token="<s>",
        eos_token="</s>",
    )
    low = view.apply_chat_template(
        [{"role": "user", "content": "hi"}], add_generation_prompt=True, reasoning_effort="low"
    )
    high = view.apply_chat_template(
        [{"role": "user", "content": "hi"}], add_generation_prompt=True, reasoning_effort="high"
    )
    assert "Reasoning: low" in low and "Reasoning: high" in high


def test_the_gguf_template_view_verdict_matches_the_tokenizer_one():
    from src.engines.gguf_chat_template import GgufChatTemplate

    view = GgufChatTemplate(chat_template=(FIXTURES / "enable-thinking.jinja").read_text())
    assert tokenizer_reasoning_lever(view) == LeverVerdict(
        lever=ReasoningLever.NATIVE_TOGGLE, is_thinker=True
    )


# ===================== the llama handle caps path =====================


class _CapsEngine:
    """Engine stub whose loaded child reported chat-template capabilities."""

    caps: dict = {}
    loaded_id = 7
    probed: list = []

    @classmethod
    def chat_template_caps(cls, llm_id=None):
        if llm_id is not None and llm_id != cls.loaded_id:
            return {}
        return cls.caps

    @classmethod
    def _load_capability_tokenizer(cls, local_path):
        cls.probed.append(local_path)
        return _fixture("qwen2.5-instruct.jinja")


@pytest.fixture
def caps_engine(monkeypatch):
    from src.core import config
    from src.engines import reasoning_lever

    reasoning_lever.reset_lever_cache()
    _CapsEngine.caps = {}
    _CapsEngine.probed = []
    monkeypatch.setattr(config, "LLM_Engine", _CapsEngine)
    yield _CapsEngine
    reasoning_lever.reset_lever_cache()


def test_the_handle_caps_decide_without_probing_anything(caps_engine):
    caps_engine.caps = {"supports_reasoning_effort": True, "supports_system_role": True}
    verdict = model_reasoning_lever("/models/gguf-effort", llm_id=7)
    assert verdict == LeverVerdict(lever=ReasoningLever.NATIVE_EFFORT, is_thinker=True)
    assert caps_engine.probed == []


def test_caps_from_another_loaded_model_are_ignored(caps_engine):
    # The child up right now may be serving a DIFFERENT model than the one the
    # turn is being planned for: its caps say nothing about this artifact.
    caps_engine.caps = {"supports_reasoning_effort": True}
    verdict = model_reasoning_lever("/models/other", llm_id=99)
    assert verdict.lever is ReasoningLever.NONE
    assert caps_engine.probed == ["/models/other"]


def test_caps_without_the_effort_capability_fall_through_to_the_probe(caps_engine):
    caps_engine.caps = {"supports_reasoning_effort": False, "supports_tools": True}
    verdict = model_reasoning_lever("/models/gguf-plain", llm_id=7)
    assert verdict == LeverVerdict(lever=ReasoningLever.NONE, is_thinker=False)
    assert caps_engine.probed == ["/models/gguf-plain"]


def test_the_probe_result_is_cached_per_artifact(caps_engine):
    model_reasoning_lever("/models/gguf-plain", llm_id=7)
    model_reasoning_lever("/models/gguf-plain", llm_id=7)
    assert caps_engine.probed == ["/models/gguf-plain"]


def test_a_tokenizer_that_cannot_be_loaded_is_never_blamed(monkeypatch, caps_engine):
    def _boom(cls, local_path):
        raise OSError("no such artifact")

    monkeypatch.setattr(_CapsEngine, "_load_capability_tokenizer", classmethod(_boom))
    assert model_reasoning_lever("/models/missing", llm_id=7) == LeverVerdict(
        lever=ReasoningLever.NONE, is_thinker=False
    )


def test_no_local_path_means_no_lever(caps_engine):
    assert model_reasoning_lever("", llm_id=7) == LeverVerdict(
        lever=ReasoningLever.NONE, is_thinker=False
    )
    assert caps_engine.probed == []
