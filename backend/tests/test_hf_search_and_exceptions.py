"""Gap coverage: live HF search pipeline (#122 follow-up) and the structured
exception contract.

`search_huggingface` is exercised with a fake HF client through every filter
(downloads floor, pipeline allowlist, dedup, runnability, limit cap) and its
degrade-to-empty error paths. The exception matrix pins each AppBaseException
subclass's HTTP status and erudi code - the contract the global handler and
the frontend error normalizer rely on.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.core import config
from src.core import exceptions as exc
from src.domains.llms import hf_search


def _hit(
    model_id,
    downloads=10_000,
    pipeline="text-generation",
    tags=(),
    safetensors=None,
    gated=False,
    likes=50,
):
    return SimpleNamespace(
        id=model_id,
        downloads=downloads,
        likes=likes,
        pipeline_tag=pipeline,
        tags=list(tags),
        safetensors=safetensors,
        gated=gated,
    )


class _Engine:
    __name__ = "CPU_Engine"
    FORMAT_TAG = "gguf"

    @staticmethod
    def is_runnable(model_id):
        return "broken" not in model_id


@pytest.fixture
def fake_hf(monkeypatch):
    def install(models):
        api = MagicMock()
        api.list_models.return_value = list(models)
        monkeypatch.setattr(config, "LLM_Engine", _Engine)
        monkeypatch.setattr(config, "get_hf_api", lambda: api, raising=False)
        return api

    return install


# =====================================================================
# UNIT - search_huggingface
# =====================================================================


@pytest.mark.unit
class TestSearchHuggingface:
    def test_empty_query_returns_empty(self):
        assert hf_search.search_huggingface("   ") == []

    def test_no_engine_format_tag_returns_empty(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_Engine", None)
        assert hf_search.search_huggingface("gemma") == []

    def test_api_failure_degrades_to_empty(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_Engine", _Engine)
        api = MagicMock()
        api.list_models.side_effect = RuntimeError("HF 429")
        monkeypatch.setattr(config, "get_hf_api", lambda: api, raising=False)
        assert hf_search.search_huggingface("gemma") == []

    def test_results_are_filtered_and_shaped(self, fake_hf):
        fake_hf(
            [
                _hit("quanter/good-model-gguf", safetensors={"total": 7_000_000_000}),
                _hit("quanter/too-obscure-gguf", downloads=3),  # below floor
                _hit("quanter/whisper-gguf", pipeline="automatic-speech-recognition"),
                _hit("quanter/broken-model-gguf"),  # KNOWN_BROKEN
                _hit("other/good-model-gguf"),  # dedup by base key
            ]
        )
        results = hf_search.search_huggingface("model")
        assert [r["link"] for r in results] == ["quanter/good-model-gguf"]
        row = results[0]
        assert row["quantized"] is True
        assert row["param_size"] == 7.0
        assert row["downloads"] == 10_000
        assert row["gated"] is False

    def test_limit_caps_results(self, fake_hf):
        fake_hf([_hit(f"org/model-{i}-gguf") for i in range(10)])
        assert len(hf_search.search_huggingface("model", limit=4)) == 4

    def test_runnability_probe_failure_keeps_candidate(self, monkeypatch, fake_hf):
        fake_hf([_hit("org/model-gguf")])

        class FlakyEngine(_Engine):
            @staticmethod
            def is_runnable(model_id):
                raise RuntimeError("roster unavailable")

        monkeypatch.setattr(config, "LLM_Engine", FlakyEngine)
        results = hf_search.search_huggingface("model")
        assert [r["link"] for r in results] == ["org/model-gguf"]

    def test_safetensors_total_dict_and_absent(self):
        assert hf_search._safetensors_total(SimpleNamespace(safetensors={"total": 5})) == 5
        assert hf_search._safetensors_total(SimpleNamespace(safetensors=None)) is None
        assert hf_search._safetensors_total(SimpleNamespace(safetensors={})) is None


# =====================================================================
# UNIT - structured exception contract
# =====================================================================


@pytest.mark.unit
class TestExceptionContract:
    CASES = [
        (exc.AppBaseException(), 500, "INTERNAL_SERVER_ERROR"),
        (exc.ModelNotFoundException("m"), 404, "MODEL_NOT_FOUND"),
        (exc.InvalidInputException("field"), 422, "INVALID_INPUT"),
        (exc.StateConflictException("busy"), 400, None),
        (exc.StateConflictException("conflict", status_code=409), 409, None),
        (exc.DatabaseException("db", trace="t"), 500, None),
        (exc.FileSystemException("fs"), 500, None),
        (exc.HuggingFaceAPIException("hf"), 503, None),
        (exc.ModelLoadingException("load"), 500, None),
        (exc.GenerationException("gen"), 500, None),
        (exc.KnowledgeBaseNotFoundException(1), 404, None),
        (exc.KnowledgeBaseCorruptedException(1, "bad index"), 500, None),
        (exc.ConversationNotFoundException(1), 404, None),
        (exc.MessageNotFoundException(1), 404, None),
        (exc.InsufficientMemoryException("load 7B"), None, None),
        (exc.UnsupportedPlatformException("MLX", "needs Apple Silicon"), None, None),
        (exc.DownloadJobNotFoundException(1), 404, None),
        (exc.TokenizationException("tok"), None, None),
        (exc.ConfigurationException("cfg"), None, None),
        (exc.EngineException("engine"), None, None),
        (exc.HardwareException("hw"), None, None),
    ]

    @pytest.mark.parametrize(
        "instance,status_code,erudi_code",
        CASES,
        ids=[type(c[0]).__name__ + str(i) for i, c in enumerate(CASES)],
    )
    def test_every_exception_carries_the_contract(self, instance, status_code, erudi_code):
        assert isinstance(instance, exc.AppBaseException)
        assert isinstance(instance.status_code, int)
        if status_code is not None:
            assert instance.status_code == status_code
        if erudi_code is not None:
            assert instance.erudi_code == erudi_code
        assert instance.message
        assert isinstance(instance.erudi_code, str)

    def test_message_is_exactly_what_the_raiser_wrote(self):
        # The error message is the failure, nothing else. Where to report a
        # bug is the interface's job (the Diagnostics panel in Settings), so
        # no support address or reporting instruction is appended here: it
        # would reach the user twice, in one language, inside a payload that
        # domain code also logs and matches on.
        instance = exc.EngineException("model failed to load")
        assert instance.message == "model failed to load"
        assert str(instance) == "model failed to load"

    def test_message_carries_no_contact_details(self):
        instance = exc.AppBaseException("boom")
        assert "@" not in instance.message
        assert "report" not in instance.message.lower()
