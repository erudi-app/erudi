"""No gated repository ever enters the catalog.

Erudi is account-less: it downloads with no Hugging Face token, so a repo whose
``gated`` flag is set (``"auto"`` or ``"manual"``) lists fine but 401s on
download. The base->quant resolver has its own guard (test_model_resolver); this
covers the other door, the community search, which is filtered by format tag only
and would otherwise let a gated community quant through. The fake hits go through
the REAL ``ModelInfo`` so the ``gated`` attribute is read the way the Hub
serializes it (no network).
"""

from types import SimpleNamespace

import pytest
from huggingface_hub import ModelInfo

from src.database import seed as seed_mod
from src.database.seed import Model_Seeder
from src.engines.cpu_engine import CPU_Engine
from src.engines.mlx_engine import MLX_Engine

pytestmark = pytest.mark.unit


def _hit(model_id: str, gated) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        pipeline_tag="text-generation",
        tags=["gguf", "text-generation", "conversational"],
        downloads=123456,
        likes=789,
        gated=gated,
    )


def _api(hits):
    """Search returns ``hits``; every repo lists one loadable quant (see test_gguf_byte_splits)."""
    return SimpleNamespace(
        list_models=lambda **kw: list(hits),
        model_info=lambda repo_id, **kw: SimpleNamespace(
            siblings=[SimpleNamespace(rfilename="model-Q4_K_M.gguf")]
        ),
    )


@pytest.mark.parametrize("engine", [CPU_Engine, MLX_Engine])
def test_community_search_requests_the_gated_flag(engine):
    # Without it in ``expand`` the reduced list serialization leaves
    # ModelInfo.gated at None and the filter below is blind.
    assert "gated" in engine.community_search_kwargs("")["expand"]


@pytest.mark.parametrize("gated", ["manual", "auto", True])
def test_build_derived_models_skips_gated_hits(monkeypatch, gated):
    monkeypatch.setattr(seed_mod.config, "LLM_Engine", CPU_Engine)
    hits = [
        _hit("google/gemma-1.1-2b-it-GGUF", gated),  # official, gated
        _hit("bartowski/gemma-1.1-2b-it-GGUF", False),  # same key, public
        _hit("bartowski/Qwen3-8B-GGUF", False),
    ]
    seeder = Model_Seeder(db=None, hf_api=_api(hits))

    rows = seeder.build_derived_models(
        [seed_mod.Search_Config(search_term="", model_type="community", default_param_size=7.0)]
    )

    # The gated repo is dropped BEFORE the normalized-key dedup, so its public
    # twin still gets in rather than being shadowed by a row that never existed.
    assert [r.link for r in rows] == ["bartowski/gemma-1.1-2b-it-GGUF", "bartowski/Qwen3-8B-GGUF"]


def test_build_derived_models_keeps_public_hits(monkeypatch):
    monkeypatch.setattr(seed_mod.config, "LLM_Engine", CPU_Engine)
    hits = [_hit("bartowski/Qwen3-8B-GGUF", False)]
    seeder = Model_Seeder(db=None, hf_api=_api(hits))

    rows = seeder.build_derived_models(
        [seed_mod.Search_Config(search_term="", model_type="community", default_param_size=7.0)]
    )

    assert [r.link for r in rows] == ["bartowski/Qwen3-8B-GGUF"]
