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


# -- By-link download keeps the date the search hit carried --------------------


@pytest.mark.unit
class TestDownloadByLinkKeepsTheDate:
    def _stored_metadata(self, client, **extra):
        """The ``model_metadata`` the route hands the placeholder row."""
        from unittest.mock import patch

        from src.core.exceptions import InvalidInputException

        # The route is only asked what it hands the helper; the helper stops there.
        with patch(
            "src.domains.llms.endpoints._start_download", side_effect=InvalidInputException("stop")
        ) as start:
            client.post(
                "/erudi/llms/download/huggingface",
                json={"link": "mlx-community/Some-Model-4bit", "name": "Some Model", **extra},
            )
        return start.call_args.kwargs["model_metadata"]

    def test_published_at_lands_in_the_placeholder_metadata(self, client):
        meta = self._stored_metadata(client, published_at="2026-03-04T10:00:00+00:00")
        assert published_at_from_metadata(meta) == "2026-03-04T10:00:00+00:00"

    @pytest.mark.parametrize("extra", [{}, {"published_at": None}, {"published_at": "garbage"}])
    def test_no_or_invalid_date_stores_no_date(self, client, extra):
        assert published_at_from_metadata(self._stored_metadata(client, **extra)) is None


# -- The date is the publisher's: the base repo, never a quant's upload date ----

_BASE = "meta-llama/Llama-3.2-1B-Instruct"
_QUANT = "bartowski/Llama-3.2-1B-Instruct-GGUF"
_BASE_DATE = datetime(2024, 9, 18, 15, 12, 47, tzinfo=timezone.utc)
_QUANT_DATE = datetime(2025, 1, 5, 8, 0, 0, tzinfo=timezone.utc)


def _info(model_id, created, tags):
    from huggingface_hub import ModelInfo

    return ModelInfo(
        id=model_id,
        # The Hub serializes the date as a string; ModelInfo parses it.
        created_at=created.strftime("%Y-%m-%dT%H:%M:%S.000Z") if created else None,
        tags=tags,
        pipeline_tag="text-generation",
        downloads=1000,
        likes=10,
    )


@pytest.mark.unit
class TestDerivedRowsUseTheBaseDate:
    def _build(self, monkeypatch, hit, model_info):
        from src.database import seed as seed_mod
        from src.database.seed import Model_Seeder, Search_Config

        monkeypatch.setattr(seed_mod.config, "LLM_Engine", CPU_Engine)
        calls = []

        def fake_model_info(repo_id, **kw):
            calls.append(repo_id)
            return model_info(repo_id)

        api = SimpleNamespace(model_info=fake_model_info, list_models=lambda **kw: [])
        seeder = Model_Seeder(db=None, hf_api=api)
        row = seeder._create_derived_llm(hit, Search_Config("", "community", 7.0))
        return published_at_from_metadata(row.model_metadata), calls

    def test_quant_takes_its_base_repos_date(self, monkeypatch):
        hit = _info(_QUANT, _QUANT_DATE, ["gguf", f"base_model:quantized:{_BASE}"])
        date, calls = self._build(
            monkeypatch, hit, lambda repo: _info(_BASE, _BASE_DATE, ["text-generation"])
        )
        assert date == _BASE_DATE.isoformat()
        assert _BASE in calls

    def test_unresolvable_base_gives_no_date_not_the_quants(self, monkeypatch):
        hit = _info(_QUANT, _QUANT_DATE, ["gguf", f"base_model:quantized:{_BASE}"])

        def boom(repo):
            raise RuntimeError("404")

        date, _ = self._build(monkeypatch, hit, boom)
        assert date is None

    def test_a_repo_without_a_base_tag_is_its_own_publisher(self, monkeypatch):
        hit = _info("someone/Own-Finetune-GGUF", _QUANT_DATE, ["gguf"])
        date, calls = self._build(monkeypatch, hit, lambda repo: _info(repo, None, []))
        assert date == _QUANT_DATE.isoformat()
        assert _BASE not in calls


@pytest.mark.unit
class TestSearchHitsDateOnlyBases:
    def _search(self, monkeypatch, tags):
        class _Engine:
            FORMAT_TAG = "mlx"

            @staticmethod
            def is_runnable(link):
                return True

        hit = SimpleNamespace(
            id="org/Some-Model-4bit", downloads=5000, likes=3, gated=False,
            pipeline_tag="text-generation", tags=tags, safetensors=None,
            created_at=_QUANT_DATE,
        )  # fmt: skip
        monkeypatch.setattr(hf_search.config, "LLM_Engine", _Engine, raising=False)
        monkeypatch.setattr(
            hf_search.config, "get_hf_api", lambda: SimpleNamespace(list_models=lambda **k: [hit])
        )
        return hf_search.search_huggingface("some")[0]["published_at"]

    def test_a_quant_hit_shows_no_date(self, monkeypatch):
        assert self._search(monkeypatch, ["mlx", f"base_model:quantized:{_BASE}"]) is None

    def test_a_base_hit_shows_its_own_date(self, monkeypatch):
        assert self._search(monkeypatch, ["mlx"]) == _QUANT_DATE.isoformat()
