"""The loaded model and its manual unload (#619).

``GET /erudi/llms/loaded`` names the model whose inference child is up (the
child's liveness is checked, not just the cached id) or answers null.
``POST /erudi/llms/loaded/unload`` unloads it under ``try_generation_guard``:
204 (also when nothing is loaded), 409 while the engine is busy. Both routes
are declared before every ``/{llm_id}`` route, which would otherwise read
``loaded`` as an id.
"""

import asyncio
from datetime import datetime
from typing import Any, ClassVar

import pytest
from fastapi.routing import APIRoute

from src.core import config
from src.engines import base_engine
from src.engines.base_engine import BaseEngine

pytestmark = pytest.mark.unit

LOADED = "/erudi/llms/loaded"
UNLOAD = "/erudi/llms/loaded/unload"


class _Engine(BaseEngine):
    """The engine singleton, its child faked by ``alive``."""

    alive: ClassVar[bool] = True
    cleanups: ClassVar[int] = 0

    @classmethod
    def loaded_child_for(cls, llm_id: Any) -> bool:
        return cls._model_id == llm_id and cls._model is not None and cls.alive

    @classmethod
    def cleanup(cls) -> None:
        cls.cleanups += 1
        super().cleanup()


@pytest.fixture(autouse=True)
def engine(monkeypatch):
    monkeypatch.setattr(config, "LLM_Engine", _Engine)
    _Engine.alive = True
    _Engine.cleanups = 0
    _Engine._model = None
    _Engine._tokenizer = None
    _Engine._model_id = None
    _Engine._last_used = None
    _Engine._idle_unload = None
    yield _Engine
    _Engine._model = None
    _Engine._tokenizer = None
    _Engine._model_id = None
    _Engine._last_used = None
    _Engine._idle_unload = None


def _load(llm_id):
    _Engine._model = {"proc": object()}
    _Engine._tokenizer = {"type": "remote"}
    _Engine._model_id = llm_id
    _Engine._last_used = datetime.now()


def test_the_routes_are_declared_before_every_id_route(client):
    paths = [r.path for r in client.app.routes if isinstance(r, APIRoute)]
    first_id_route = min(
        i
        for i, path in enumerate(paths)
        if path.startswith("/erudi/llms/{") and path != "/erudi/llms/{"
    )
    assert paths.index(LOADED) < first_id_route
    assert paths.index(UNLOAD) < first_id_route


def test_nothing_loaded_is_null(client):
    response = client.get(LOADED)
    assert response.status_code == 200, response.text
    assert response.json() is None


def test_the_loaded_model_is_named(client, mock_llm):
    _load(mock_llm.id)
    response = client.get(LOADED)
    assert response.status_code == 200, response.text
    assert response.json() == {"llm_id": mock_llm.id, "name": "Test Base Model"}


def test_a_dead_child_is_not_a_loaded_model(client, mock_llm):
    _load(mock_llm.id)
    _Engine.alive = False
    assert client.get(LOADED).json() is None


def test_a_loaded_model_without_its_row_is_not_named(client):
    # Deleting a model unloads it first; a child left for an id the database
    # no longer knows has nothing to name.
    _load(987_654)
    assert client.get(LOADED).json() is None


def test_unload_frees_the_model_and_records_a_manual_reload(client, mock_llm):
    _load(mock_llm.id)
    response = client.post(UNLOAD)
    assert response.status_code == 204, response.text
    assert response.content == b""
    assert _Engine._model is None
    assert _Engine.cleanups == 1
    assert _Engine._idle_unload == {"model_id": mock_llm.id, "reason": "manual"}
    assert client.get(LOADED).json() is None


def test_unload_with_nothing_loaded_is_a_204_no_op(client):
    response = client.post(UNLOAD)
    assert response.status_code == 204, response.text
    assert _Engine.cleanups == 0
    assert _Engine._idle_unload is None


def test_unload_while_the_engine_is_busy_is_a_409(client, mock_llm, monkeypatch):
    monkeypatch.setattr(base_engine, "GENERATION_BUSY_WAIT_S", 0.05)
    _load(mock_llm.id)

    # The TestClient runs the app on its own event loop: the lock the turn
    # holds is that loop's, taken through an endpoint-free coroutine that
    # parks inside generation_guard until the unload has been refused.
    release = None

    async def _hold():
        nonlocal release
        release = asyncio.Event()
        async with _Engine.generation_guard():
            await release.wait()

    portal = client.portal
    portal.start_task_soon(_hold)
    portal.call(asyncio.sleep, 0.05)

    response = client.post(UNLOAD)

    portal.call(release.set)
    portal.call(asyncio.sleep, 0.05)
    assert response.status_code == 409, response.text
    error = response.json()["error"]
    assert error["type"] == "ENGINE_BUSY"
    assert error["message"] == "The model is busy; try again in a moment."
    assert _Engine._model is not None
    assert _Engine.cleanups == 0
