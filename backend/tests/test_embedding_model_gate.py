"""KB embedding-model gate (#146, #164): availability is filesystem-driven
over the FULL required file set, post-availability loads are offline
(``local_files_only=True``), download is an idempotent background task, and
the routes don't collide with the parametric ``/{llm_id}/status`` KB route.
"""

import logging

import pytest

from src.ingestion import embedding_model as em


@pytest.fixture(autouse=True)
def _reset_state():
    em._reset_state_for_tests()
    yield
    em._reset_state_for_tests()


def _set_cached(monkeypatch, present: bool):
    monkeypatch.setattr(
        em,
        "try_to_load_from_cache",
        lambda repo_id, filename, **k: f"/cache/{filename}" if present else None,
    )


@pytest.mark.unit
def test_available_true_when_all_required_files_cached(monkeypatch):
    _set_cached(monkeypatch, True)
    assert em.embedding_model_available() is True


@pytest.mark.unit
def test_available_false_when_cache_empty(monkeypatch):
    _set_cached(monkeypatch, False)
    assert em.embedding_model_available() is False


@pytest.mark.unit
def test_required_files_cover_the_sentence_transformer_load():
    # A SentenceTransformer load needs more than the weights (#164): the ST
    # architecture, the transformer config, the pooling module and the
    # tokenizer files must ALL be present for an offline load to succeed.
    assert {
        "modules.json",
        "config.json",
        "model.safetensors",
        "sentence_bert_config.json",
        "1_Pooling/config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentencepiece.bpe.model",
    } <= set(em.REQUIRED_FILES)


@pytest.mark.unit
@pytest.mark.parametrize("missing", sorted(em.REQUIRED_FILES))
def test_available_false_when_any_required_file_missing(monkeypatch, missing):
    # A partial cache (e.g. interrupted download) must NOT pass for complete —
    # every required file is checked, not just the weights (#164).
    monkeypatch.setattr(
        em,
        "try_to_load_from_cache",
        lambda repo_id, filename, **k: None if filename == missing else f"/cache/{filename}",
    )
    assert em.embedding_model_available() is False


@pytest.mark.unit
def test_available_false_on_cached_no_exist_sentinel(monkeypatch):
    # huggingface_hub returns a non-str sentinel when it knows the file does not
    # exist; only a real str path counts as available.
    monkeypatch.setattr(em, "try_to_load_from_cache", lambda *a, **k: object())
    assert em.embedding_model_available() is False


@pytest.mark.unit
def test_download_state_shape(monkeypatch):
    _set_cached(monkeypatch, False)
    assert em.download_state() == {"available": False, "downloading": False, "error": None}


class _SpySentenceTransformer:
    """Records constructor kwargs; stands in for the real SentenceTransformer."""

    calls: list[dict] = []

    def __init__(self, model_id, **kwargs):
        type(self).calls.append({"model_id": model_id, **kwargs})


@pytest.fixture
def spy_sentence_transformer(monkeypatch):
    import sentence_transformers

    _SpySentenceTransformer.calls = []
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _SpySentenceTransformer)
    return _SpySentenceTransformer


@pytest.mark.unit
def test_load_model_is_offline_when_available(monkeypatch, spy_sentence_transformer):
    # Once the cache is complete, loads must NEVER touch the network (#164):
    # online, hub HEAD-revalidates every file; offline, the failed DNS probe
    # surfaces as a bogus load error despite a fully pre-downloaded model.
    _set_cached(monkeypatch, True)

    em._load_model()

    assert spy_sentence_transformer.calls[0]["local_files_only"] is True


@pytest.mark.unit
def test_load_model_allows_network_when_unavailable(monkeypatch, spy_sentence_transformer):
    _set_cached(monkeypatch, False)

    em._load_model()

    assert spy_sentence_transformer.calls[0]["local_files_only"] is False


@pytest.mark.unit
def test_start_download_runs_load_when_absent(monkeypatch):
    _set_cached(monkeypatch, False)
    monkeypatch.setattr(em, "_spawn", lambda target: target())  # run synchronously
    calls = []
    monkeypatch.setattr(em, "_load_model", lambda **kwargs: calls.append(kwargs))

    em.start_download()

    assert calls == [{"local_files_only": False}]  # the download path IS the fetch
    assert em._state["downloading"] is False  # finished
    assert em._state["error"] is None


@pytest.mark.unit
def test_download_success_logs_completion(monkeypatch, caplog):
    _set_cached(monkeypatch, False)
    monkeypatch.setattr(em, "_spawn", lambda target: target())
    monkeypatch.setattr(em, "_load_model", lambda **kwargs: None)

    with caplog.at_level(logging.INFO, logger="erudi"):
        em.start_download()

    assert any("Embedding model download complete" in record.message for record in caplog.records)


@pytest.mark.unit
def test_start_download_noop_when_already_available(monkeypatch):
    _set_cached(monkeypatch, True)
    calls = []
    monkeypatch.setattr(em, "_spawn", lambda target: calls.append("spawned"))
    monkeypatch.setattr(em, "_load_model", lambda: calls.append("loaded"))

    state = em.start_download()

    assert calls == []  # already present -> never spawned
    assert state["available"] is True
    assert state["downloading"] is False


@pytest.mark.unit
def test_start_download_idempotent_when_already_downloading(monkeypatch):
    _set_cached(monkeypatch, False)
    em._reset_state_for_tests(downloading=True)
    calls = []
    monkeypatch.setattr(em, "_spawn", lambda target: calls.append("spawned"))

    em.start_download()

    assert calls == []  # a download is already in flight -> no second task


@pytest.mark.unit
def test_download_error_is_surfaced(monkeypatch):
    _set_cached(monkeypatch, False)
    monkeypatch.setattr(em, "_spawn", lambda target: target())

    def boom(**kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(em, "_load_model", boom)

    em.start_download()

    assert em._state["downloading"] is False
    assert "disk full" in (em._state["error"] or "")


def _offline_exceptions():
    import requests
    from huggingface_hub.errors import OfflineModeIsEnabled

    yield OfflineModeIsEnabled("Cannot reach https://huggingface.co: offline mode is enabled.")
    yield requests.exceptions.ConnectionError("Max retries exceeded with url: /intfloat/...")
    yield OSError("[Errno 8] getaddrinfo failed")
    # transformers often re-wraps the network failure; the original cause
    # survives on __cause__/__context__ and must still be recognized.
    wrapped = RuntimeError("Cannot send a request, as the client has been closed.")
    wrapped.__cause__ = OSError("[Errno 8] nodename nor servname provided, or not known")
    yield wrapped


@pytest.mark.unit
@pytest.mark.parametrize("exc", _offline_exceptions(), ids=lambda e: type(e).__name__)
def test_offline_download_error_maps_to_clear_message(monkeypatch, exc):
    # Raw transformers/hub network errors are useless to the user (#164):
    # any offline-shaped failure must surface as a clear no-internet message.
    _set_cached(monkeypatch, False)
    monkeypatch.setattr(em, "_spawn", lambda target: target())

    def boom(**kwargs):
        raise exc

    monkeypatch.setattr(em, "_load_model", boom)

    em.start_download()

    assert em._state["error"] == em.OFFLINE_ERROR_MESSAGE
    assert "no internet connection" in em._state["error"]


@pytest.mark.unit
def test_embedding_model_routes_do_not_collide_with_llm_id_status(monkeypatch):
    # GET /knowledge_base/embedding-model/status must resolve to the gate route,
    # NOT be captured by the parametric GET /knowledge_base/{llm_id}/status
    # (which would 422 trying to parse "embedding-model" as an int).
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.domains.knowledge_base.endpoints import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    monkeypatch.setattr(em, "try_to_load_from_cache", lambda *a, **k: None)
    resp = client.get("/knowledge_base/embedding-model/status")

    assert resp.status_code == 200
    assert resp.json() == {"available": False, "downloading": False, "error": None}


@pytest.mark.unit
@pytest.mark.parametrize("available", [True, False])
def test_e5_embeddings_load_is_offline_when_available(
    monkeypatch, spy_sentence_transformer, available
):
    from src.ingestion import embeddings as embeddings_module
    from src.ingestion.embeddings import E5Embeddings

    monkeypatch.setattr(embeddings_module, "embedding_model_available", lambda: available)
    monkeypatch.setattr(E5Embeddings, "_model", None)  # reset the resident singleton

    E5Embeddings._get_model()

    assert spy_sentence_transformer.calls[0]["local_files_only"] is available
