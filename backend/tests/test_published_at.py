"""Publication date on model cards (#561).

The date a card shows is the ``Created:`` line of the row's stored metadata: for
a curated base row that is the BASE repo's creation date (the quant is resolved
afterwards and is always younger), for a derived/community row the repo's own.
It is read at response time from the stored text -- no new column, no network --
so an old snapshot or a fallback row without the line yields ``None``.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.domains.llms import hf_search
from src.domains.llms.schemas import HFSearchResult, LLMResponse
from src.engines.cpu_engine import CPU_Engine
from src.engines.mlx_engine import MLX_Engine
from src.utils.hf_model_metadata import published_at_from_metadata

_META = "Model ID: meta-llama/Llama-3.2-1B-Instruct\nAuthor: meta-llama\nCreated: {c}\nDownloads: 5 \nLikes: 1"


@pytest.mark.unit
class TestPublishedAtFromMetadata:
    def test_parses_the_created_line_to_iso(self):
        meta = _META.format(c="2024-09-18 15:12:47+00:00")
        assert published_at_from_metadata(meta) == "2024-09-18T15:12:47+00:00"

    @pytest.mark.parametrize(
        "meta",
        [None, "", "Size: ~1 GB", _META.format(c="Unknown"), _META.format(c="not a date")],
    )
    def test_unknown_is_none(self, meta):
        assert published_at_from_metadata(meta) is None


@pytest.mark.unit
class TestPayloads:
    def _row(self, metadata):
        return SimpleNamespace(
            id=1, name="x", local=0, link="a/b", model_metadata=metadata,
            type=None, description=None, quantized=True, supports_tools=None,
            param_size=1.0, is_base=True, conversational=True, category="general",
            generation_hints=None, artifact_size_bytes=None,
        )  # fmt: skip

    def test_llm_response_carries_iso_date(self):
        row = self._row(_META.format(c="2024-09-18 15:12:47+00:00"))
        dumped = LLMResponse.model_validate(row, from_attributes=True).model_dump()
        assert dumped["published_at"] == "2024-09-18T15:12:47+00:00"

    def test_llm_response_null_when_unknown(self):
        row = self._row(_META.format(c="Unknown"))
        assert LLMResponse.model_validate(row, from_attributes=True).published_at is None
        row = self._row(None)
        assert LLMResponse.model_validate(row, from_attributes=True).published_at is None

    def test_hf_search_result_field_defaults_to_none(self):
        assert HFSearchResult.model_fields["published_at"].default is None

    @pytest.mark.parametrize("created", [datetime(2026, 3, 4, 10, 0, tzinfo=timezone.utc), None])
    def test_hf_search_hit_carries_iso_or_null(self, monkeypatch, created):
        class _Engine:
            FORMAT_TAG = "mlx"

            @staticmethod
            def is_runnable(link):
                return True

        seen = {}

        def list_models(**kw):
            seen.update(kw)
            return [
                SimpleNamespace(
                    id="mlx-community/Qwen2.5-7B-Instruct-4bit", downloads=5000, likes=3,
                    gated=False, pipeline_tag="text-generation", tags=[], safetensors=None,
                    created_at=created,
                )
            ]  # fmt: skip

        monkeypatch.setattr(hf_search.config, "LLM_Engine", _Engine, raising=False)
        monkeypatch.setattr(
            hf_search.config, "get_hf_api", lambda: SimpleNamespace(list_models=list_models)
        )
        res = hf_search.search_huggingface("qwen")
        assert "createdAt" in seen["expand"]
        assert res[0]["published_at"] == (created.isoformat() if created else None)


@pytest.mark.parametrize("engine", [CPU_Engine, MLX_Engine])
def test_community_search_requests_the_creation_date(engine):
    # Without it the reduced list serialization leaves ModelInfo.created_at at
    # None and every derived row says "Created: Unknown".
    assert "createdAt" in engine.community_search_kwargs("")["expand"]
