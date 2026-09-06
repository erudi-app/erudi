"""A broken metadata formatter must not seed a fake catalog row (#354).

``format_model_info_metadata`` used to swallow every exception and RETURN the
string ``"Error formatting metadata: ..."``. Both catalog builders stored that
string as ``model_metadata``, so a broken row looked populated, the error was
frozen into the bundled snapshot, and nothing in the logs pointed at the model.

Now the formatter propagates, and the per-model failure paths that already
exist in ``build_base_models`` / ``build_derived_models`` log the repo id and
skip the model. No network anywhere in this file.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from huggingface_hub import ModelInfo

from src.database import seed as seed_mod
from src.database.seed import Model_Config, Model_Seeder, Search_Config
from src.engines.cpu_engine import CPU_Engine
from src.utils.hf_model_metadata import format_model_info_metadata

ERROR_PREFIX = "Error formatting metadata"


class _BrokenInfo(ModelInfo):
    """A Hub payload whose ``sha`` blows up when the formatter reads it."""

    @property
    def sha(self):
        raise RuntimeError("hub payload broken")

    @sha.setter
    def sha(self, value):
        pass


def _healthy_info(model_id: str) -> Mock:
    info = Mock()
    info.id = model_id
    info.author = model_id.split("/")[0]
    info.sha = "abc123"
    info.downloads = 1000
    info.likes = 50
    info.tags = ["gguf", "conversational"]
    info.created_at = info.last_modified = None
    info.library_name = info.pipeline_tag = None
    info.private = info.gated = False
    return info


class _Size:
    size_bytes = None

    def to_string(self):
        return "1 GB"


def _seeder(monkeypatch, hf_api) -> Model_Seeder:
    monkeypatch.setattr(seed_mod.config, "LLM_Engine", CPU_Engine)
    monkeypatch.setattr(seed_mod, "get_disk_size_after_quant", lambda *a, **k: _Size())
    monkeypatch.setattr(seed_mod, "capture_generation_hints", lambda *a, **k: None)
    monkeypatch.setattr(
        seed_mod,
        "resolve_quant",
        lambda link, tag, api: f"quanter/{link.split('/')[-1]}-GGUF",
    )
    seeder = Model_Seeder(db=None, hf_api=hf_api)
    monkeypatch.setattr(
        seeder,
        "discover_instruct_models",
        lambda org, model_type: [Model_Config("Test-7B", "org/test-7b", model_type)],
    )
    return seeder


# =====================================================================
# UNIT - the formatter no longer launders its own failure into a string
# =====================================================================


@pytest.mark.unit
def test_formatter_propagates_instead_of_returning_an_error_string():
    with pytest.raises(RuntimeError, match="hub payload broken"):
        format_model_info_metadata(_BrokenInfo(id="org/test-7b"))


@pytest.mark.unit
def test_formatter_healthy_payload_is_unchanged():
    out = format_model_info_metadata(_healthy_info("org/test-7b"), _Size(), True)
    assert out.startswith("Model ID: org/test-7b")
    assert "Size: 1 GB" in out
    assert ERROR_PREFIX not in out


# =====================================================================
# UNIT - base catalog: broken model skipped and logged, healthy one seeded
# =====================================================================


@pytest.mark.unit
class TestBaseCatalog:
    def test_broken_metadata_skips_the_model_and_logs_the_repo(self, monkeypatch, caplog):
        seeder = _seeder(
            monkeypatch,
            SimpleNamespace(model_info=lambda link: _BrokenInfo(id=link)),
        )
        with caplog.at_level(logging.WARNING, logger="erudi"):
            rows = seeder.build_base_models([("org", "test", "test")])

        assert rows == []
        # The catalog goes on without the model: a skipped row is a WARNING,
        # with the traceback attached so the broken payload can be traced.
        assert any(
            r.levelno == logging.WARNING
            and "org/test-7b" in r.message
            and "hub payload broken" in r.message
            and r.exc_info
            for r in caplog.records
        )

    def test_healthy_model_is_seeded_with_real_metadata(self, monkeypatch):
        seeder = _seeder(
            monkeypatch,
            SimpleNamespace(model_info=lambda link: _healthy_info(link)),
        )
        rows = seeder.build_base_models([("org", "test", "test")])

        assert [r.link for r in rows] == ["quanter/test-7b-GGUF"]
        assert rows[0].model_metadata.startswith("Model ID: org/test-7b")
        assert ERROR_PREFIX not in rows[0].model_metadata


# =====================================================================
# UNIT - derived catalog: same contract on the community path
# =====================================================================


def _hit(model_id: str, cls=ModelInfo) -> ModelInfo:
    return cls(
        id=model_id,
        pipeline_tag="text-generation",
        tags=["gguf", "text-generation", "conversational"],
        downloads=123456,
        likes=789,
    )


@pytest.mark.unit
class TestDerivedCatalog:
    def _seeder(self, monkeypatch, hits):
        monkeypatch.setattr(seed_mod.config, "LLM_Engine", CPU_Engine)
        monkeypatch.setattr(seed_mod, "get_disk_size_after_quant", lambda *a, **k: _Size())
        monkeypatch.setattr(seed_mod, "capture_generation_hints", lambda *a, **k: None)
        return Model_Seeder(db=None, hf_api=SimpleNamespace(list_models=lambda **kw: list(hits)))

    def test_broken_metadata_skips_the_model_and_logs_the_repo(self, monkeypatch, caplog):
        seeder = self._seeder(monkeypatch, [_hit("bartowski/Broken-8B-GGUF", _BrokenInfo)])
        with caplog.at_level(logging.WARNING, logger="erudi"):
            rows = seeder.build_derived_models(
                [Search_Config(search_term="", model_type="community", default_param_size=7.0)]
            )

        assert rows == []
        assert any(
            r.levelno >= logging.WARNING
            and "bartowski/Broken-8B-GGUF" in r.message
            and "hub payload broken" in r.message
            for r in caplog.records
        )

    def test_healthy_model_is_seeded_with_real_metadata(self, monkeypatch):
        seeder = self._seeder(monkeypatch, [_hit("bartowski/Qwen3-8B-GGUF")])
        rows = seeder.build_derived_models(
            [Search_Config(search_term="", model_type="community", default_param_size=7.0)]
        )

        assert [r.link for r in rows] == ["bartowski/Qwen3-8B-GGUF"]
        assert rows[0].model_metadata.startswith("Model ID: bartowski/Qwen3-8B-GGUF")
        assert ERROR_PREFIX not in rows[0].model_metadata
