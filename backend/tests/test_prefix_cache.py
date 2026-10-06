"""The single-conversation prefix cache: engine hooks and the reset shield.

The MLX child keeps an Automatic Prefix Cache (APC) pool that must hold ONLY
the current conversation: it is reset when the conversation changes and when
compaction rewrites the history. Three engine hooks carry that ownership
(``claim_prefix``, ``on_history_rewritten``, ``note_stream_abandoned``); only
``MLX_Engine`` implements them, the llama.cpp engines stay strict no-ops.

Every reset runs in a worker thread through ``run_reset_shielded`` and is
registered as ``BaseEngine._pending_reset``; whoever acquires the generation
lock next (a turn, a title, the idle tick) drains it before proceeding, so no
prefill can start while an orphan ``POST /v1/cache/reset`` or barrier still
runs. The HTTP side is faked: nothing here talks to a real child.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timedelta

import anyio
import pytest

from src.engines import base_engine as base_engine_module
from src.engines import mlx_engine as mlx_engine_module
from src.engines.base_engine import BaseEngine, run_reset_shielded
from src.engines.cpu_engine import CPU_Engine
from src.engines.cuda_engine import CUDA_Engine
from src.engines.mlx_engine import MLX_Engine

pytestmark = pytest.mark.unit


class _Resp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"enabled": True, "status": "cleared"}
        self.text = str(self._payload)

    def json(self):
        return self._payload


class _FakeHttp:
    """Records every POST the engine makes and answers from a script."""

    def __init__(self, responses=None):
        self.calls: list[dict] = []
        self._responses = list(responses or [])

    def post(self, url, json=None, headers=None, timeout=None, **kwargs):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if self._responses:
            outcome = self._responses.pop(0)
        else:
            outcome = _Resp()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def paths(self):
        return [call["url"].split("27300", 1)[-1] for call in self.calls]


def _handle(**extra):
    handle = {
        "base_url": "http://127.0.0.1:27300",
        "api_key": "secret-key-value",
        "model_path": "/models/fake",
        "context_tokens": 8192,
        "prefix_owner": "conv:1",
    }
    handle.update(extra)
    return handle


@pytest.fixture
def http(monkeypatch):
    fake = _FakeHttp()
    monkeypatch.setattr(mlx_engine_module.requests, "post", fake.post)
    return fake


@pytest.fixture
def mlx_handle(monkeypatch):
    handle = _handle()
    monkeypatch.setattr(MLX_Engine, "_model", handle)
    return handle


# ===================== MLX_Engine hooks =====================


def test_claim_with_the_same_owner_sends_nothing(http, mlx_handle):
    MLX_Engine.claim_prefix("conv:1")

    assert http.calls == []
    assert mlx_handle["prefix_owner"] == "conv:1"


def test_claim_with_a_different_owner_resets_then_records_the_owner(http, mlx_handle, caplog):
    with caplog.at_level(logging.INFO):
        MLX_Engine.claim_prefix("conv:2")

    assert http.paths() == ["/v1/cache/reset"]
    call = http.calls[0]
    assert call["headers"] == {"Authorization": "Bearer secret-key-value"}
    assert call["timeout"] == 5.0
    assert mlx_handle["prefix_owner"] == "conv:2"
    assert any(
        r.getMessage() == "Prefix cache reset: reason=conversation_change" for r in caplog.records
    )


def test_claim_for_the_arena_resets_with_the_arena_reason(http, mlx_handle, caplog):
    with caplog.at_level(logging.INFO):
        MLX_Engine.claim_prefix("arena")

    assert http.paths() == ["/v1/cache/reset"]
    assert mlx_handle["prefix_owner"] == "arena"
    assert any(r.getMessage() == "Prefix cache reset: reason=arena" for r in caplog.records)


def test_a_fresh_child_is_claimed_without_a_reset(http, monkeypatch):
    handle = _handle(prefix_owner="<fresh>")
    monkeypatch.setattr(MLX_Engine, "_model", handle)

    MLX_Engine.claim_prefix("conv:9")

    assert http.calls == []
    assert handle["prefix_owner"] == "conv:9"


def test_a_handle_without_an_owner_is_reset_on_claim(http, monkeypatch):
    handle = _handle()
    del handle["prefix_owner"]
    monkeypatch.setattr(MLX_Engine, "_model", handle)

    MLX_Engine.claim_prefix("conv:9")

    assert http.paths() == ["/v1/cache/reset"]
    assert handle["prefix_owner"] == "conv:9"


def test_a_failed_reset_logs_one_warning_marks_the_owner_dirty_and_the_next_claim_retries(
    monkeypatch, mlx_handle, caplog
):
    import requests

    fake = _FakeHttp([requests.ConnectionError("refused"), _Resp()])
    monkeypatch.setattr(mlx_engine_module.requests, "post", fake.post)

    with caplog.at_level(logging.INFO):
        MLX_Engine.claim_prefix("conv:2")

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "secret-key-value" not in message
    assert message.isascii()
    assert mlx_handle["prefix_owner"] == "<dirty>"
    assert not any("Prefix cache reset: reason" in r.getMessage() for r in caplog.records)

    # Same conversation again: the dirty owner forces the retry.
    MLX_Engine.claim_prefix("conv:2")
    assert len(fake.calls) == 2
    assert mlx_handle["prefix_owner"] == "conv:2"


def test_a_non_200_reset_is_a_failure(monkeypatch, mlx_handle, caplog):
    fake = _FakeHttp([_Resp(status_code=401, payload={"detail": "Invalid API key"})])
    monkeypatch.setattr(mlx_engine_module.requests, "post", fake.post)

    with caplog.at_level(logging.WARNING):
        MLX_Engine.claim_prefix("conv:2")

    assert mlx_handle["prefix_owner"] == "<dirty>"
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_a_child_without_apc_answers_enabled_false_and_that_is_a_success(monkeypatch, mlx_handle):
    fake = _FakeHttp([_Resp(payload={"enabled": False})])
    monkeypatch.setattr(mlx_engine_module.requests, "post", fake.post)

    MLX_Engine.claim_prefix("conv:2")

    assert mlx_handle["prefix_owner"] == "conv:2"


def test_history_rewritten_resets_and_keeps_the_owner(http, mlx_handle, caplog):
    with caplog.at_level(logging.INFO):
        MLX_Engine.on_history_rewritten()

    assert http.paths() == ["/v1/cache/reset"]
    assert mlx_handle["prefix_owner"] == "conv:1"
    assert any(r.getMessage() == "Prefix cache reset: reason=compaction" for r in caplog.records)


def test_history_rewritten_failure_marks_the_owner_dirty(monkeypatch, mlx_handle):
    fake = _FakeHttp([_Resp(status_code=500)])
    monkeypatch.setattr(mlx_engine_module.requests, "post", fake.post)

    MLX_Engine.on_history_rewritten()

    assert mlx_handle["prefix_owner"] == "<dirty>"


def test_an_abandoned_stream_sends_a_barrier_before_the_reset(http, monkeypatch):
    from src.agents.chat_model import first_chunk_ceiling_s

    handle = _handle(abandoned=True)
    monkeypatch.setattr(MLX_Engine, "_model", handle)

    MLX_Engine.claim_prefix("conv:2")

    assert http.paths() == ["/v1/chat/completions", "/v1/cache/reset"]
    barrier = http.calls[0]
    # Same shape as the readiness ping: one token, not streamed, the real model path.
    assert barrier["json"]["max_tokens"] == 1
    assert barrier["json"]["stream"] is False
    assert barrier["json"]["model"] == "/models/fake"
    assert barrier["headers"] == {"Authorization": "Bearer secret-key-value"}
    assert barrier["timeout"] == first_chunk_ceiling_s(8192)
    assert not handle.get("abandoned")
    assert handle["prefix_owner"] == "conv:2"


def test_a_failed_barrier_skips_the_reset_and_keeps_the_flag(monkeypatch, caplog):
    import requests

    fake = _FakeHttp([requests.ReadTimeout("slow")])
    monkeypatch.setattr(mlx_engine_module.requests, "post", fake.post)
    handle = _handle(abandoned=True)
    monkeypatch.setattr(MLX_Engine, "_model", handle)

    with caplog.at_level(logging.WARNING):
        MLX_Engine.claim_prefix("conv:2")

    assert [c["url"].rsplit("27300", 1)[-1] for c in fake.calls] == ["/v1/chat/completions"]
    assert handle["abandoned"] is True
    assert handle["prefix_owner"] == "<dirty>"
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_note_stream_abandoned_flags_the_given_handle_only(monkeypatch):
    built_for = _handle()
    current = _handle()
    monkeypatch.setattr(MLX_Engine, "_model", current)

    MLX_Engine.note_stream_abandoned(built_for)

    assert built_for["abandoned"] is True
    assert "abandoned" not in current


def test_hooks_act_on_the_handle_read_once_even_if_the_engine_swaps_mid_call(monkeypatch):
    """A swap between the read and the write must never make the hook write
    into another child's handle."""
    first = _handle(prefix_owner="conv:1")
    second = _handle(prefix_owner="<fresh>")
    monkeypatch.setattr(MLX_Engine, "_model", first)

    def _post_and_swap(url, **kwargs):
        MLX_Engine._model = second  # the engine swapped children meanwhile
        return _Resp()

    monkeypatch.setattr(mlx_engine_module.requests, "post", _post_and_swap)

    MLX_Engine.claim_prefix("conv:2")

    assert first["prefix_owner"] == "conv:2"
    assert second["prefix_owner"] == "<fresh>"


def test_hooks_on_a_missing_handle_are_no_ops(http, monkeypatch):
    monkeypatch.setattr(MLX_Engine, "_model", None)

    MLX_Engine.claim_prefix("conv:1")
    MLX_Engine.on_history_rewritten()
    MLX_Engine.note_stream_abandoned(None)

    assert http.calls == []


def test_a_spawned_child_starts_with_a_fresh_owner(monkeypatch, tmp_path):
    class _Proc:
        pid = 4242

        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(mlx_engine_module.mp, "Process", _Proc)
    monkeypatch.setattr(MLX_Engine, "_trained_window_of", classmethod(lambda cls, path: 4096))
    monkeypatch.setattr(
        mlx_engine_module.child_log, "prepare_child_log", lambda port: tmp_path / "c.log"
    )

    handle = MLX_Engine._spawn_child(model_path=tmp_path, alias="erudi-1", port=27301)

    assert handle["prefix_owner"] == "<fresh>"
    assert "abandoned" not in handle


# ===================== llama.cpp and base: strict no-ops =====================


@pytest.mark.parametrize("engine", [CPU_Engine, CUDA_Engine, BaseEngine])
def test_non_mlx_hooks_write_nothing(engine, monkeypatch):
    handle = {"base_url": "http://127.0.0.1:27200", "api_key": "k", "alias": "erudi-1"}
    snapshot = dict(handle)
    monkeypatch.setattr(engine, "_model", handle)

    def _no_http(*args, **kwargs):  # pragma: no cover - a call is the failure
        raise AssertionError("a llama.cpp engine must not talk to its child here")

    monkeypatch.setattr(mlx_engine_module.requests, "post", _no_http)

    engine.claim_prefix("conv:1")
    engine.on_history_rewritten()
    engine.note_stream_abandoned(handle)

    assert handle == snapshot


# ===================== the reset shield and the drain =====================


class _GuardEngine(BaseEngine):
    """Bare subclass: generation_guard and the idle tick without a child."""


@pytest.fixture(autouse=True)
def _clean_pending_reset():
    BaseEngine._pending_reset = None
    yield
    BaseEngine._pending_reset = None
    _GuardEngine._model = None
    _GuardEngine._model_id = None
    _GuardEngine._last_used = None


class _BlockingReset:
    """A reset function that blocks its worker thread until released."""

    def __init__(self, *, raises=None):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished_at = None
        self.raises = raises

    def __call__(self):
        self.entered.set()
        self.release.wait(10)
        self.finished_at = datetime.now()
        if self.raises is not None:
            raise self.raises
        return True


async def _wait_entered(reset: _BlockingReset):
    await asyncio.to_thread(reset.entered.wait, 5)
    assert reset.entered.is_set()


async def test_run_reset_shielded_returns_the_reset_result():
    assert await run_reset_shielded(lambda: True) is True
    assert BaseEngine._pending_reset is None


async def test_run_reset_shielded_waits_for_the_thread_even_when_cancelled():
    reset = _BlockingReset()
    waiter = asyncio.create_task(run_reset_shielded(reset))
    await _wait_entered(reset)

    waiter.cancel()
    await asyncio.sleep(0.05)
    assert not waiter.done(), "the cancelled caller must keep waiting for the thread"

    reset.release.set()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert reset.finished_at is not None


async def _orphan_reset(engine, reset):
    """A guard holder torn down while its reset thread still runs: the helper
    keeps waiting in its own task, the guard is released anyway."""
    holder_done = asyncio.Event()
    orphan = {}

    async def _holder():
        async with engine.generation_guard():
            orphan["task"] = asyncio.create_task(run_reset_shielded(reset))
            await _wait_entered(reset)
        holder_done.set()

    await _holder()
    await holder_done.wait()
    return orphan["task"]


@pytest.mark.parametrize("acquirer", ["generation_guard", "idle_tick"])
async def test_the_next_acquirer_drains_an_orphan_reset_before_proceeding(acquirer):
    reset = _BlockingReset()
    orphan = await _orphan_reset(_GuardEngine, reset)
    assert BaseEngine._pending_reset is not None

    proceeded_at = {}

    async def _next():
        if acquirer == "generation_guard":
            async with _GuardEngine.generation_guard():
                proceeded_at["t"] = datetime.now()
        else:
            # An idle model: the tick would reap it right away without the drain.
            _GuardEngine._model = object()
            _GuardEngine._model_id = "x"
            _GuardEngine._last_used = datetime.now() - timedelta(seconds=10_000)
            await _GuardEngine._cleanup_tick()
            proceeded_at["t"] = datetime.now()

    nxt = asyncio.create_task(_next())
    await asyncio.sleep(0.2)
    assert not nxt.done(), "the next holder proceeded while the orphan reset still ran"
    assert "t" not in proceeded_at

    reset.release.set()
    await asyncio.wait_for(nxt, timeout=5)
    await asyncio.wait_for(orphan, timeout=5)
    assert proceeded_at["t"] >= reset.finished_at
    assert BaseEngine._pending_reset is None


async def test_a_reset_that_raised_is_logged_once_and_never_fails_the_next_holder(caplog):
    reset = _BlockingReset(raises=RuntimeError("reset blew up"))
    orphan = await _orphan_reset(_GuardEngine, reset)

    async def _next():
        async with _GuardEngine.generation_guard():
            return "ran"

    with caplog.at_level(logging.INFO):
        nxt = asyncio.create_task(_next())
        await asyncio.sleep(0.05)
        reset.release.set()
        assert await asyncio.wait_for(nxt, timeout=5) == "ran"
        await asyncio.wait_for(orphan, timeout=5)
        await asyncio.sleep(0)

    failures = [r for r in caplog.records if "reset blew up" in str(r.exc_info) + r.getMessage()]
    assert len(failures) == 1
    assert failures[0].levelno == logging.ERROR


async def test_the_shield_wait_does_not_spin_under_an_anyio_cancel_scope(monkeypatch):
    """anyio re-delivers a cancelled scope's cancellation at every await; a
    bare ``asyncio.shield`` loop would wake on each one. The shielded anyio
    scope stops the re-delivery, so the loop waits once per real event."""
    calls = {"n": 0}
    real_shield = asyncio.shield

    def _counting_shield(fut):
        calls["n"] += 1
        return real_shield(fut)

    monkeypatch.setattr(base_engine_module.asyncio, "shield", _counting_shield)
    reset = _BlockingReset()

    async def _consumer(scope_holder):
        with anyio.CancelScope() as scope:
            scope_holder["scope"] = scope
            await run_reset_shielded(reset)

    holder: dict = {}
    task = asyncio.create_task(_consumer(holder))
    await _wait_entered(reset)
    holder["scope"].cancel()
    await asyncio.sleep(0.3)
    reset.release.set()
    await asyncio.wait_for(task, timeout=5)

    assert calls["n"] <= 3, f"the shield loop woke {calls['n']} times"
    assert reset.finished_at is not None
