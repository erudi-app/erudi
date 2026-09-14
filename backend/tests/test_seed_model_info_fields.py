"""The catalog build must read only DECLARED ``ModelInfo`` attributes (#242 fallout).

``huggingface_hub.ModelInfo.__init__`` ends with ``self.__dict__.update(**kwargs)``,
so raw JSON keys ride through as attributes. ``modelId`` was never a declared
attribute — it only existed because the plain ``list_models`` response happened to
carry it. Once ``community_search_kwargs`` started passing ``expand=[...]`` (to get
the ``pipeline_tag`` the non-chat filter needs), the API switched to a reduced
serialization that has ``id`` but NOT ``modelId``, and the weekly snapshot refresh
died on ``AttributeError: 'ModelInfo' object has no attribute 'modelId'``.

The fake here is built from the reduced payload through the REAL ``ModelInfo``
class, so a stub can't accidentally keep the dead field alive. No network.
"""

import ast
import pathlib
from types import SimpleNamespace

import pytest
from huggingface_hub import ModelInfo

from src.database import seed as seed_mod
from src.database.seed import Model_Seeder
from src.engines.cpu_engine import CPU_Engine

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def _expanded_hit(model_id: str, pipeline_tag: str, tags: list) -> ModelInfo:
    """A ``list_models(..., expand=[...])`` hit, as the Hub actually serializes it.

    Only the requested fields come back alongside ``id``; every other declared
    attribute lands as None and ``modelId`` is simply absent.
    """
    info = ModelInfo(
        id=model_id,
        pipeline_tag=pipeline_tag,
        tags=tags,
        downloads=123456,
        likes=789,
    )
    # Fidelity guard: if this ever holds, the fake stopped reproducing the bug.
    assert not hasattr(info, "modelId")
    return info


@pytest.mark.unit
def test_build_derived_models_handles_expanded_search_serialization(monkeypatch):
    monkeypatch.setattr(seed_mod.config, "LLM_Engine", CPU_Engine)

    seen_kwargs = {}

    hits = [
        _expanded_hit(
            "bartowski/Qwen3-8B-GGUF",
            "text-generation",
            ["gguf", "text-generation", "conversational"],
        ),
        _expanded_hit(
            "handy-computer/whisper-large-v3-gguf", "automatic-speech-recognition", ["gguf"]
        ),
    ]

    def fake_list_models(**kwargs):
        seen_kwargs.update(kwargs)
        return list(hits)

    api = SimpleNamespace(
        list_models=fake_list_models,
        # Every repo lists one loadable quant, so the GGUF file check passes.
        model_info=lambda repo_id, **kw: SimpleNamespace(
            siblings=[SimpleNamespace(rfilename="model-Q4_K_M.gguf")]
        ),
    )
    seeder = Model_Seeder(db=None, hf_api=api)
    rows = seeder.build_derived_models(
        [seed_mod.Search_Config(search_term="", model_type="community", default_param_size=7.0)]
    )

    # The expand is what triggers the reduced serialization; keep the test honest
    # about running against that exact request shape.
    assert "expand" in seen_kwargs and "pipeline_tag" in seen_kwargs["expand"]

    assert [r.link for r in rows] == ["bartowski/Qwen3-8B-GGUF"]  # ASR dropped via slug
    row = rows[0]
    assert row.param_size == 8.0
    assert row.is_base is False
    assert row.conversational is True
    # Proves the metadata formatter received a usable id, not an empty/error string.
    assert "Model ID: bartowski/Qwen3-8B-GGUF" in row.model_metadata


@pytest.mark.unit
def test_no_source_reads_the_undeclared_model_id_attribute():
    offenders = [
        f"{py.relative_to(_SRC)}:{node.lineno}"
        for py in sorted(_SRC.rglob("*.py"))
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8")))
        if isinstance(node, ast.Attribute) and node.attr == "modelId"
    ]
    assert not offenders, f"use the declared ModelInfo.id instead: {offenders}"
