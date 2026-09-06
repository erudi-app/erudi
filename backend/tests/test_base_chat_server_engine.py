"""Invariant + unit tests for `BaseChatServerEngine`.

Defines a minimal concrete `_TestEngine` subclass to exercise the shared
lifecycle without spawning real subprocesses. Integration coverage (real
`mlx_lm.server` / `llama-server` spawns) lives in
`test_mlx_engine_server.py`, `test_cpu_engine_server.py`, and
`test_cuda_engine_server.py`.
"""

from __future__ import annotations

import contextlib
import json
import socket as _stdlib_socket
import sys
from pathlib import Path
from typing import Iterator, List
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.core.exceptions import EngineException
from src.engines.base_chat_server_engine import BaseChatServerEngine


# =====================================================================
# Helpers
# =====================================================================


@contextlib.contextmanager
def _occupied_port(port: int):
    """Hold ``port`` so that another binder genuinely cannot take it (#331).

    ``SO_REUSEADDR`` does not mean the same thing on both platforms, and the
    difference inverts this helper's purpose:

    - POSIX: it mainly permits rebinding a port stuck in ``TIME_WAIT``. A live
      listening socket still excludes a second bind, so occupying works.
    - Windows: it permits binding a port that is **actively bound by another
      socket**. ``_pick_free_port`` sets ``SO_REUSEADDR`` on its probe, so it
      would bind straight over us, correctly conclude the port is free, and the
      test asserting "busy ports are skipped" would fail against working code.

    ``SO_EXCLUSIVEADDRUSE`` is the Windows counterpart that actually reserves
    the address against other binders, including ``SO_REUSEADDR`` ones. The
    ``listen()`` matters too: a bound-but-not-listening Windows socket is
    weaker at excluding a second bind.

    Yields the socket, and always closes it.
    """
    s = _stdlib_socket.socket(_stdlib_socket.AF_INET, _stdlib_socket.SOCK_STREAM)
    try:
        if sys.platform == "win32":
            s.setsockopt(
                _stdlib_socket.SOL_SOCKET,
                _stdlib_socket.SO_EXCLUSIVEADDRUSE,  # type: ignore[attr-defined]
                1,
            )
        else:
            s.setsockopt(_stdlib_socket.SOL_SOCKET, _stdlib_socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))
        s.listen(1)
        yield s
    finally:
        s.close()


def _sse_bytes(payloads: List[dict | str]) -> Iterator[bytes]:
    """Render payloads as raw SSE chunks. Strings are emitted verbatim (for
    `[DONE]` / corrupted / comment lines). Dicts are JSON-encoded with the
    `data: ` prefix."""
    for p in payloads:
        if isinstance(p, str):
            yield f"{p}\n\n".encode("utf-8")
        else:
            yield f"data: {json.dumps(p)}\n\n".encode("utf-8")


def _mock_streaming_post(sse_chunks: List[bytes]):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.iter_content.return_value = iter(sse_chunks)
    cm = MagicMock()
    cm.__enter__.return_value = response
    cm.__exit__.return_value = False
    return Mock(return_value=cm)


# =====================================================================
# Minimal concrete test subclass — overrides required attrs + abstract hooks.
# =====================================================================


class _TestEngine(BaseChatServerEngine):
    """Concrete subclass used only by these tests."""

    _port_range_start = 19000
    _port_range_count = 50
    _server_name = "test.server"
    _server_alias_prefix = "test-"
    _tokenizer_provider = "test-provider"
    _probe_timeout_s = 2.0
    _probe_poll_interval_s = 0.05

    # Hooks left abstract on the base — override with MagicMocks by default.
    _spawn_child = classmethod(
        lambda cls, **kw: {  # type: ignore[assignment]
            "pid": 1,
            "proc": MagicMock(),
            "port": kw["port"],
            "base_url": f"http://127.0.0.1:{kw['port']}",
            "alias": kw["alias"],
            "model_path": kw["model_path"],
        }
    )
    _terminate_process = classmethod(lambda cls, proc: None)  # type: ignore[assignment]
    _proc_is_alive = classmethod(lambda cls, proc: True)  # type: ignore[assignment]
    _resolve_model_artifact = classmethod(  # type: ignore[assignment]
        lambda cls, p: Path(p) if isinstance(p, (str, Path)) else p
    )


def _reset_test_engine_state() -> None:
    _TestEngine._model = None
    _TestEngine._tokenizer = None
    _TestEngine._model_id = None
    _TestEngine._last_used = None
    _TestEngine._atexit_handler = None


@pytest.fixture(autouse=True)
def _state_reset():
    _reset_test_engine_state()
    yield
    _reset_test_engine_state()


# =====================================================================
# UNIT — abstract enforcement
# =====================================================================


@pytest.mark.unit
class TestAbstractEnforcement:
    def test_base_is_abstract_cannot_instantiate(self):
        """Direct instantiation of BaseChatServerEngine must fail."""
        # BaseEngine's __init__ raises RuntimeError; the abstract methods are
        # the real guard. Verify abstract methods are declared.
        abstracts = BaseChatServerEngine.__abstractmethods__
        assert "_spawn_child" in abstracts
        assert "_terminate_process" in abstracts
        assert "_proc_is_alive" in abstracts
        assert "_resolve_model_artifact" in abstracts

    def test_subclass_missing_required_attrs_raises(self):
        """A subclass that forgets to set _port_range_start / _server_name /
        _tokenizer_provider must raise on _pick_free_port / _start_server."""

        class _Bad(BaseChatServerEngine):
            _spawn_child = classmethod(lambda cls, **kw: {})  # type: ignore[assignment]
            _terminate_process = classmethod(lambda cls, p: None)  # type: ignore[assignment]
            _proc_is_alive = classmethod(lambda cls, p: True)  # type: ignore[assignment]
            _resolve_model_artifact = classmethod(lambda cls, p: Path("/tmp"))  # type: ignore[assignment]

        with pytest.raises(EngineException, match="required class attrs"):
            _Bad._pick_free_port()


# =====================================================================
# UNIT — _pick_free_port
# =====================================================================


@pytest.mark.unit
class TestPickFreePort:
    def test_returns_port_in_configured_range(self):
        port = _TestEngine._pick_free_port()
        assert (
            _TestEngine._port_range_start
            <= port
            < (_TestEngine._port_range_start + _TestEngine._port_range_count)
        )

    def test_skips_busy_port_and_finds_next(self):
        busy = _TestEngine._port_range_start
        with _occupied_port(busy):
            port = _TestEngine._pick_free_port()
            assert port != busy

    def test_no_free_port_raises_with_server_name(self):
        """Exhausting the range raises EngineException mentioning the server."""

        class _Tight(_TestEngine):
            _port_range_start = 19500
            _port_range_count = 2

        with contextlib.ExitStack() as stack:
            for offset in range(_Tight._port_range_count):
                # A port already taken by something else on the machine is fine:
                # it is busy either way, which is what this test needs.
                with contextlib.suppress(OSError):
                    stack.enter_context(_occupied_port(_Tight._port_range_start + offset))
            with pytest.raises(EngineException, match=_Tight._server_name):
                _Tight._pick_free_port()


# =====================================================================
# UNIT — _probe_ready (two-stage)
# =====================================================================


@pytest.mark.unit
class TestProbeReady:
    def test_health_503_then_200_then_chat_200_succeeds(self):
        """Realistic warm-up: /health goes 503 → 503 → 200, then chat-ping OK."""
        responses = [
            MagicMock(status_code=503),
            MagicMock(status_code=503),
            MagicMock(status_code=200),
        ]
        chat_resp = MagicMock(status_code=200, text='{"choices":[]}')
        with (
            patch(
                "src.engines.base_chat_server_engine.requests.get",
                side_effect=responses,
            ),
            patch(
                "src.engines.base_chat_server_engine.requests.post",
                return_value=chat_resp,
            ),
        ):
            _TestEngine._probe_ready("http://127.0.0.1:19000")

    def test_early_proc_crash_raises_immediately(self):
        """If proc.is_alive returns False mid-poll, raise without waiting timeout."""

        class _Crashy(_TestEngine):
            _proc_is_alive = classmethod(lambda cls, p: False)  # type: ignore[assignment]

        with pytest.raises(EngineException, match="early crash"):
            _Crashy._probe_ready("http://127.0.0.1:19000", proc=MagicMock())

    def test_chat_ping_4xx_raises_with_body_preview(self):
        """Stage-2 ping returning 400 surfaces the body in the error message."""
        health_ok = MagicMock(status_code=200)
        chat_bad = MagicMock(status_code=400, text='{"error":"missing chat template"}')
        with (
            patch(
                "src.engines.base_chat_server_engine.requests.get",
                return_value=health_ok,
            ),
            patch(
                "src.engines.base_chat_server_engine.requests.post",
                return_value=chat_bad,
            ),
        ):
            with pytest.raises(EngineException, match="400"):
                _TestEngine._probe_ready("http://127.0.0.1:19000")

    def test_chat_ping_sends_provided_model_field(self):
        """The stage-2 payload's `model` is exactly the `model_field` passed in
        (computed by `_start_server` via `_payload_model_value(handle)`: the
        alias for llama-cpp, the preloaded model path for MLX/mlx-vlm)."""
        health_ok = MagicMock(status_code=200)
        chat_ok = MagicMock(status_code=200, text='{"choices":[]}')
        captured_payload: dict = {}

        def _capture_post(url, json=None, **kwargs):
            captured_payload.update(json)
            return chat_ok

        with (
            patch(
                "src.engines.base_chat_server_engine.requests.get",
                return_value=health_ok,
            ),
            patch(
                "src.engines.base_chat_server_engine.requests.post",
                side_effect=_capture_post,
            ),
        ):
            _TestEngine._probe_ready(
                "http://127.0.0.1:19000",
                model_field="/models/erudi-7",
            )
        assert captured_payload["model"] == "/models/erudi-7"

    def test_chat_ping_defaults_to_placeholder_model(self):
        """With no `model_field`, the probe sends a neutral placeholder. Real
        spawns always pass one; this only covers direct/defensive calls."""
        health_ok = MagicMock(status_code=200)
        chat_ok = MagicMock(status_code=200, text='{"choices":[]}')
        captured_payload: dict = {}

        def _capture_post(url, json=None, **kwargs):
            captured_payload.update(json)
            return chat_ok

        with (
            patch(
                "src.engines.base_chat_server_engine.requests.get",
                return_value=health_ok,
            ),
            patch(
                "src.engines.base_chat_server_engine.requests.post",
                side_effect=_capture_post,
            ),
        ):
            _TestEngine._probe_ready("http://127.0.0.1:19000")  # no model_field
        assert captured_payload["model"] == "default_model"

    def test_chat_ping_authenticates_when_the_child_has_a_key(self):
        """A key-protected child answers 401 to an unauthenticated probe.

        llama-server is spawned with `--api-key` so that no other local process
        (nor a web page POSTing to the loopback port) can use it. That key
        guards `/v1/chat/completions`, so the stage-2 ping must present it or
        every spawn would fail readiness on a perfectly healthy server.
        """
        health_ok = MagicMock(status_code=200)
        chat_ok = MagicMock(status_code=200, text='{"choices":[]}')
        captured_headers: dict = {}

        def _capture_post(url, json=None, headers=None, **kwargs):
            captured_headers["value"] = headers
            return chat_ok

        with (
            patch(
                "src.engines.base_chat_server_engine.requests.get",
                return_value=health_ok,
            ),
            patch(
                "src.engines.base_chat_server_engine.requests.post",
                side_effect=_capture_post,
            ),
        ):
            _TestEngine._probe_ready("http://127.0.0.1:19000", api_key="s3cret-token")
        assert captured_headers["value"]["Authorization"] == "Bearer s3cret-token"

    def test_chat_ping_sends_no_auth_header_without_a_key(self):
        """A handle without a key gets no `Authorization` header at all.

        Sending a bogus header would be at best noise and at worst rejected,
        so the header must appear only when a key exists.
        """
        health_ok = MagicMock(status_code=200)
        chat_ok = MagicMock(status_code=200, text='{"choices":[]}')
        captured_headers: dict = {"value": "unset"}

        def _capture_post(url, json=None, headers=None, **kwargs):
            captured_headers["value"] = headers
            return chat_ok

        with (
            patch(
                "src.engines.base_chat_server_engine.requests.get",
                return_value=health_ok,
            ),
            patch(
                "src.engines.base_chat_server_engine.requests.post",
                side_effect=_capture_post,
            ),
        ):
            _TestEngine._probe_ready("http://127.0.0.1:19000")
        assert not captured_headers["value"]

    def test_start_server_forwards_the_handles_key_to_the_probe(self):
        """The key is minted inside `_spawn_child`; only the handle carries it.

        Without this wiring the probe would authenticate with nothing and every
        llama.cpp model load would fail at readiness, so the link between the
        two is pinned explicitly.
        """
        captured: dict = {}

        def _capture_probe(base_url, proc=None, model_field="default_model", api_key=None):
            captured["api_key"] = api_key

        spawned = {
            "pid": 1,
            "proc": MagicMock(),
            "port": 19000,
            "base_url": "http://127.0.0.1:19000",
            "alias": "test-1",
            "model_path": "/m.gguf",
            "api_key": "minted-at-spawn",
        }
        with (
            patch.object(_TestEngine, "_spawn_child", return_value=spawned),
            patch.object(_TestEngine, "_probe_ready", side_effect=_capture_probe),
        ):
            _TestEngine._start_server(model_path=Path("/m.gguf"), alias="test-1", port=19000)
        assert captured["api_key"] == "minted-at-spawn"

    def test_timeout_message_mentions_port_for_toctou_hint(self):
        """When /health never reaches 200, error should mention the port so the
        user can run `lsof -i :PORT` to diagnose a stolen-port race."""

        class _Slow(_TestEngine):
            _probe_timeout_s = 0.2
            _probe_poll_interval_s = 0.05

        always_503 = MagicMock(status_code=503)
        with patch(
            "src.engines.base_chat_server_engine.requests.get",
            return_value=always_503,
        ):
            with pytest.raises(EngineException, match=r":19000"):
                _Slow._probe_ready("http://127.0.0.1:19000")


# =====================================================================
# UNIT — atexit storage + _stop_server_if_running
# =====================================================================


@pytest.mark.unit
class TestAtexitAndStop:
    def test_start_server_registers_and_stores_handler(self):
        """_start_server must store the registered handler on cls._atexit_handler."""
        with (
            patch.object(_TestEngine, "_probe_ready", return_value=None),
            patch("src.engines.base_chat_server_engine.atexit.register") as reg,
        ):
            handle = _TestEngine._start_server(
                model_path=Path("/tmp"),
                alias="test-x",
                port=19010,
            )
        assert _TestEngine._atexit_handler is not None
        reg.assert_called_once_with(_TestEngine._atexit_handler)
        assert handle["port"] == 19010

    def test_start_server_terminates_proc_on_probe_failure(self):
        """If `_probe_ready` raises after spawn, the child must be terminated
        and no atexit handler should be registered (otherwise we'd leak a
        handler holding a dead proc — exactly bug (b) from PR #76)."""
        spawned_proc = MagicMock()
        spawn_returns = {
            "pid": 1,
            "proc": spawned_proc,
            "port": 19015,
            "base_url": "http://127.0.0.1:19015",
            "alias": "test-x",
            "model_path": Path("/tmp"),
        }
        with (
            patch.object(_TestEngine, "_spawn_child", return_value=spawn_returns),
            patch.object(
                _TestEngine, "_probe_ready", side_effect=EngineException("probe rejected")
            ),
            patch.object(_TestEngine, "_terminate_process") as term,
            patch("src.engines.base_chat_server_engine.atexit.register") as reg,
        ):
            with pytest.raises(EngineException, match="probe rejected"):
                _TestEngine._start_server(
                    model_path=Path("/tmp"),
                    alias="test-x",
                    port=19015,
                )
        term.assert_called_once_with(spawned_proc)
        reg.assert_not_called()
        assert _TestEngine._atexit_handler is None

    def test_stop_server_unregisters_handler(self):
        """_stop_server_if_running must unregister the stored handler."""

        def sentinel() -> None: ...

        _TestEngine._atexit_handler = sentinel
        _TestEngine._model = {
            "pid": 1,
            "proc": MagicMock(),
            "port": 19011,
            "base_url": "http://127.0.0.1:19011",
            "alias": "test-x",
            "model_path": Path("/tmp"),
        }
        with patch("src.engines.base_chat_server_engine.atexit.unregister") as unreg:
            _TestEngine._stop_server_if_running()
        unreg.assert_called_once_with(sentinel)
        assert _TestEngine._atexit_handler is None

    def test_stop_idempotent_when_no_proc(self):
        """No handle, no handler — must not raise."""
        _TestEngine._model = None
        _TestEngine._atexit_handler = None
        _TestEngine._stop_server_if_running()  # must not raise

    def test_atexit_handler_isolated_per_subclass(self):
        """`cls._atexit_handler = ...` must shadow on the subclass dict, not
        on `BaseChatServerEngine`. Pinning this invariant prevents a future
        refactor from accidentally storing on the base and corrupting
        cross-engine state."""

        class _A(_TestEngine):
            pass

        class _B(_TestEngine):
            pass

        def handler_a() -> None: ...
        def handler_b() -> None: ...

        _A._atexit_handler = handler_a
        _B._atexit_handler = handler_b
        assert _A._atexit_handler is handler_a
        assert _B._atexit_handler is handler_b
        # Reset cleanup
        _A._atexit_handler = None
        _B._atexit_handler = None

    def test_model_swap_unregisters_previous_handler(self):
        """get_model_and_tokenizer with a new llm_id must stop the previous one
        (which unregisters its atexit handler) before spawning the new one."""
        prev_proc = MagicMock()

        def prev_handler() -> None: ...

        _TestEngine._model = {
            "pid": 1,
            "proc": prev_proc,
            "port": 19020,
            "base_url": "http://127.0.0.1:19020",
            "alias": "test-old",
            "model_path": Path("/old"),
        }
        _TestEngine._tokenizer = {"type": "remote", "provider": "test-provider"}
        _TestEngine._model_id = "old"
        _TestEngine._atexit_handler = prev_handler

        with (
            patch.object(_TestEngine, "_probe_ready", return_value=None),
            patch("src.engines.base_chat_server_engine.atexit.unregister") as unreg,
            patch("src.engines.base_chat_server_engine.atexit.register"),
        ):
            _TestEngine.get_model_and_tokenizer(llm_id="new", llm_local_path="/new")
        unreg.assert_called_once_with(prev_handler)


# =====================================================================
# UNIT — cached-handle liveness (a crashed child must not be reused forever)
# =====================================================================


@pytest.mark.unit
class TestCachedHandleLiveness:
    """A child that dies mid-generation must not silently poison every later
    request for the same llm_id (#365): the cache check must notice the proc
    is gone and force a fresh spawn instead of handing back a dead handle."""

    def test_dead_cached_proc_forces_respawn_not_reuse(self):
        dead_proc = MagicMock()
        _TestEngine._model = {
            "pid": 1,
            "proc": dead_proc,
            "port": 19030,
            "base_url": "http://127.0.0.1:19030",
            "alias": "test-662",
            "model_path": Path("/m"),
        }
        _TestEngine._tokenizer = {"type": "remote", "provider": "test-provider"}
        _TestEngine._model_id = "662"

        with (
            patch.object(_TestEngine, "_proc_is_alive", return_value=False),
            patch.object(_TestEngine, "_probe_ready", return_value=None),
            patch.object(_TestEngine, "_spawn_child", wraps=_TestEngine._spawn_child) as spawn,
        ):
            model, _ = _TestEngine.get_model_and_tokenizer(llm_id="662", llm_local_path="/m")

        spawn.assert_called_once()
        assert model["proc"] is not dead_proc

    def test_live_cached_proc_is_still_reused(self):
        """Sanity check: a genuinely alive cached proc is unaffected."""
        live_proc = MagicMock()
        _TestEngine._model = {
            "pid": 1,
            "proc": live_proc,
            "port": 19031,
            "base_url": "http://127.0.0.1:19031",
            "alias": "test-662",
            "model_path": Path("/m"),
        }
        _TestEngine._tokenizer = {"type": "remote", "provider": "test-provider"}
        _TestEngine._model_id = "662"

        with (
            patch.object(_TestEngine, "_proc_is_alive", return_value=True),
            patch.object(_TestEngine, "_spawn_child") as spawn,
        ):
            model, _ = _TestEngine.get_model_and_tokenizer(llm_id="662", llm_local_path="/m")

        spawn.assert_not_called()
        assert model["proc"] is live_proc


# =====================================================================
# UNIT — _payload_model_value default + override
# =====================================================================


@pytest.mark.unit
class TestPayloadModelValue:
    def test_default_returns_alias_from_handle(self):
        handle = {"alias": "test-erudi-7"}
        assert _TestEngine._payload_model_value(handle) == "test-erudi-7"

    def test_subclass_can_override_to_model_path(self):
        """MLX/mlx-vlm overrides this to send the real preloaded model path
        (mlx_vlm.server has no `default_model` sentinel)."""

        class _MlxLike(_TestEngine):
            @staticmethod
            def _payload_model_value(handle):
                return handle["model_path"]

        assert (
            _MlxLike._payload_model_value({"alias": "ignored", "model_path": "/models/x"})
            == "/models/x"
        )


# =====================================================================
# UNIT — cleanup contract
# =====================================================================


@pytest.mark.unit
class TestCleanup:
    def test_cleanup_calls_stop_then_super(self):
        with patch.object(_TestEngine, "_stop_server_if_running") as mock_stop:
            _TestEngine.cleanup()
        mock_stop.assert_called_once()
        assert _TestEngine._model is None
        assert _TestEngine._tokenizer is None
