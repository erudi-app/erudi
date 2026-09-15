"""Context-window exposure on the llms response.

Two first-class numbers, deliberately distinct:

- ``context_window`` -- the model's TRAINED window, a fact of the artifact
  (``generation_hints.context_length``): it survives restarts and machines.
- ``allocated_context_window`` -- what the engine's loaded child actually runs
  with (``effective_context_tokens``), resolved live and ONLY for the row that
  IS the currently loaded model. It depends on the machine and its memory
  state, so it is computed on the fly and never persisted.
"""

import pytest

from src.core import config
from src.domains.llms.schemas import LLMResponse

pytestmark = pytest.mark.unit


def _row(**kw):
    base = dict(id=7, name="Qwen3 0.6B", local=1, link="/models/qwen3")
    base.update(kw)
    return LLMResponse(**base)


class _LoadedEngine:
    """Engine stub whose loaded child (llm id 7) allocated an 8192 window."""

    _model_id = 7

    @classmethod
    def effective_context_tokens(cls):
        return 8192


@pytest.fixture(autouse=True)
def _engine(monkeypatch):
    monkeypatch.setattr(config, "LLM_Engine", _LoadedEngine)


class TestContextWindow:
    def test_reads_the_hints_context_length(self):
        assert _row(generation_hints={"context_length": 40960}).context_window == 40960

    def test_none_without_hints(self):
        assert _row().context_window is None
        assert _row(generation_hints={"base_repo": "x"}).context_window is None

    def test_garbage_hint_is_none(self):
        assert _row(generation_hints={"context_length": "big"}).context_window is None
        assert _row(generation_hints={"context_length": 0}).context_window is None


class TestAllocatedContextWindow:
    def test_the_loaded_row_reports_the_engines_window(self):
        assert _row(id=7).allocated_context_window == 8192

    def test_a_row_that_is_not_the_loaded_model_reports_none(self):
        assert _row(id=8).allocated_context_window is None

    def test_remote_and_downloading_rows_report_none(self):
        assert _row(id=7, local=0).allocated_context_window is None
        assert _row(id=7, local=2).allocated_context_window is None

    def test_no_engine_reports_none(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_Engine", None)
        assert _row(id=7).allocated_context_window is None

    def test_nothing_loaded_reports_none(self, monkeypatch):
        class _Idle(_LoadedEngine):
            _model_id = None

        monkeypatch.setattr(config, "LLM_Engine", _Idle)
        assert _row(id=7).allocated_context_window is None

    def test_a_string_loaded_id_still_matches(self, monkeypatch):
        # get_model_and_tokenizer's llm_id is typed str; the row id is int.
        class _StrId(_LoadedEngine):
            _model_id = "7"

        monkeypatch.setattr(config, "LLM_Engine", _StrId)
        assert _row(id=7).allocated_context_window == 8192

    def test_a_probe_that_raises_degrades_to_none(self, monkeypatch):
        class _Broken(_LoadedEngine):
            @classmethod
            def effective_context_tokens(cls):
                raise RuntimeError("boom")

        monkeypatch.setattr(config, "LLM_Engine", _Broken)
        assert _row(id=7).allocated_context_window is None

    def test_serialized_payload_carries_both_fields(self):
        data = _row(id=7, generation_hints={"context_length": 40960}).model_dump()
        assert data["context_window"] == 40960
        assert data["allocated_context_window"] == 8192
