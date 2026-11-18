"""KB embedding-model gate (#146, #164): availability is filesystem-driven
over the FULL required file set, the download fetches exactly that set into
the hub cache as an idempotent background task, and the routes don't collide
with the parametric ``/{llm_id}/status`` KB route.
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
def test_required_files_cover_the_child_load():
    # The mlx_vlm embedding child needs more than the weights (#164): the
    # model config, the pooling module (mean pooling is read from
    # 1_Pooling/config.json) and the tokenizer files must ALL be present.
    assert {
        "config.json",
        "model.safetensors",
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


@pytest.mark.unit
def test_fetch_downloads_exactly_the_required_files_into_the_cache(monkeypatch, tmp_path):
    # What the gate checks, what it fetches and what the child loads are one
    # list: the download is a snapshot_download restricted to it, into the
    # hub cache layout under CACHE_DIR (where try_to_load_from_cache looks).
    import huggingface_hub

    calls = []
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda repo_id, **kw: calls.append((repo_id, kw))
    )
    monkeypatch.setattr(em.config, "CACHE_DIR", tmp_path)

    em._fetch()

    assert calls == [
        (
            em.EMBEDDING_MODEL_ID,
            {"cache_dir": str(tmp_path), "allow_patterns": list(em.REQUIRED_FILES)},
        )
    ]


@pytest.mark.unit
def test_start_download_fetches_when_absent(monkeypatch):
    _set_cached(monkeypatch, False)
    monkeypatch.setattr(em, "_spawn", lambda target: target())  # run synchronously
    calls = []
    monkeypatch.setattr(em, "_fetch", lambda: calls.append("fetched"))

    em.start_download()

    assert calls == ["fetched"]
    assert em._state["downloading"] is False  # finished
    assert em._state["error"] is None


@pytest.mark.unit
def test_download_success_logs_completion(monkeypatch, caplog):
    _set_cached(monkeypatch, False)
    monkeypatch.setattr(em, "_spawn", lambda target: target())
    monkeypatch.setattr(em, "_fetch", lambda: None)

    with caplog.at_level(logging.INFO, logger="erudi"):
        em.start_download()

    assert any("Embedding model download complete" in record.message for record in caplog.records)


@pytest.mark.unit
def test_start_download_noop_when_already_available(monkeypatch):
    _set_cached(monkeypatch, True)
    calls = []
    monkeypatch.setattr(em, "_spawn", lambda target: calls.append("spawned"))
    monkeypatch.setattr(em, "_fetch", lambda: calls.append("fetched"))

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

    def boom():
        raise RuntimeError("disk full")

    monkeypatch.setattr(em, "_fetch", boom)

    em.start_download()

    assert em._state["downloading"] is False
    assert "disk full" in (em._state["error"] or "")


def _offline_exceptions():
    import requests
    from huggingface_hub.errors import OfflineModeIsEnabled

    yield OfflineModeIsEnabled("Cannot reach https://huggingface.co: offline mode is enabled.")
    yield requests.exceptions.ConnectionError("Max retries exceeded with url: /intfloat/...")
    yield OSError("[Errno 8] getaddrinfo failed")
    # huggingface_hub often re-wraps the network failure; the original cause
    # survives on __cause__/__context__ and must still be recognized.
    wrapped = RuntimeError("Cannot send a request, as the client has been closed.")
    wrapped.__cause__ = OSError("[Errno 8] nodename nor servname provided, or not known")
    yield wrapped


@pytest.mark.unit
@pytest.mark.parametrize("exc", _offline_exceptions(), ids=lambda e: type(e).__name__)
def test_offline_download_error_maps_to_clear_message(monkeypatch, exc):
    # Raw hub network errors are useless to the user (#164):
    # any offline-shaped failure must surface as a clear no-internet message.
    _set_cached(monkeypatch, False)
    monkeypatch.setattr(em, "_spawn", lambda target: target())

    def boom():
        raise exc

    monkeypatch.setattr(em, "_fetch", boom)

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
