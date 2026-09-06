"""Tests for `MLX_Engine` in server-mode (subprocess `mlx_lm.server`).

This file is written **before** the implementation (TDD-RED phase). All tests
target the post-refactor API described in `plan: refactor/mlx-server-subprocess`.
Until Phase 2 lands the new `MLX_Engine` implementation, these tests should
exercise the post-refactor server-mode API (spawn / probe / cleanup / swap)
and the regressions now run over the live ChatOpenAI path.

Test sections:
    - **Unit** (`@pytest.mark.unit`): fully mocked, no MLX dependency, no
      subprocess spawn. Runs on Linux CI.
    - **Integration engine** (`@pytest.mark.mlx_only`): spawns a real
      `mlx_lm.server` subprocess against a small downloaded model. Skipped
      on Linux CI via the `mlx_test_model_path` fixture.
    - **Thinking model regression** (`@pytest.mark.mlx_only`, opt-in via
      `ERUDI_TEST_THINKING=1`): two layers (#90) — the raw-SSE test pins the
      server contract (thinking ACTIVATES via `--enable-thinking` and arrives
      INLINE as `<think>...</think>` in `delta.content`, `delta.reasoning`
      silent), and the runner test pins that thinking flows as `thinking`
      events without leaking into the `answer` stream.
    - **Gemma EOS regression** (`@pytest.mark.mlx_only`, opt-in via
      `ERUDI_TEST_GEMMA=1`): validates the audit's GAP #15 (Gemma
      `<end_of_turn>` may not be in `eos_token_ids` natively). If it fails,
      Phase 2 must wire a per-family `stop` fallback.
    - **E2E full-stack** (`@pytest.mark.e2e @pytest.mark.mlx_only`): drives
      the new engine through the actual FastAPI endpoints
      (`POST /erudi/conversations/{id}/query`, ...) to confirm no
      observable contract regression downstream of the engine.

Patch targets (Phase 2 module shape assumed):
    - `src.engines.mlx_engine.mp` — `multiprocessing as mp`, used as `mp.Process(...)`.
    - `src.engines.base_chat_server_engine.requests` — http client (same pattern as
      `cpu_engine.py:25`).
    - `src.engines.base_chat_server_engine.socket` — module-level import for `_pick_free_port`.

Run examples:
    pytest backend/tests/test_mlx_engine_server.py -m unit          # CI-friendly
    pytest backend/tests/test_mlx_engine_server.py -m mlx_only      # local Mac
    ERUDI_TEST_THINKING=1 pytest ... -k thinking                    # opt-in
"""

from __future__ import annotations

import json
import logging
import socket as _stdlib_socket
import time
from pathlib import Path
from typing import Iterator, List
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.engines.mlx_engine import MLX_Engine


# =====================================================================
# Helpers shared by unit tests
# =====================================================================


def _sse_bytes(payloads: List[dict | str]) -> Iterator[bytes]:
    """Render a list of payloads as raw SSE bytes chunks.

    Mirrors what `requests.Response.iter_content(chunk_size=None)` yields when
    streaming from `mlx_lm.server`. Strings are emitted verbatim (used to
    inject `[DONE]` and corrupted lines). Dicts are JSON-encoded.
    """
    for p in payloads:
        if isinstance(p, str):
            yield f"data: {p}\n\n".encode("utf-8")
        else:
            yield f"data: {json.dumps(p)}\n\n".encode("utf-8")


def _mock_streaming_post(sse_chunks: List[bytes]):
    """Return a Mock suitable for patching `requests.post(..., stream=True)`.

    Context-manager (`with requests.post(...) as r:`) yields a Response-like
    Mock whose `iter_content` walks the supplied bytes chunks.
    """
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.iter_content.return_value = iter(sse_chunks)
    cm = MagicMock()
    cm.__enter__.return_value = response
    cm.__exit__.return_value = False
    return Mock(return_value=cm)


def _reset_mlx_engine_class_state() -> None:
    """Wipe the shared class-level state of MLX_Engine.

    Required between tests because `MLX_Engine` is a singleton-style class
    with mutable class attributes (`_model`, `_tokenizer`, `_model_id`,
    `_last_used`). Without this, a leaked Mock from one test would be
    interpreted as a cached real model by the next.
    """
    MLX_Engine._model = None
    MLX_Engine._tokenizer = None
    MLX_Engine._model_id = None
    MLX_Engine._last_used = None


@pytest.fixture(autouse=True)
def _mlx_engine_state_reset():
    """Reset MLX_Engine class state around every test in this file.

    Critically, the teardown also attempts `cleanup()` so that any real
    subprocess spawned by an integration test that raised mid-setup is
    terminated before the next test runs. Without this, an exception in
    `get_model_and_tokenizer` between spawn and the test's own `finally:
    cleanup()` would leak the child process.
    """
    _reset_mlx_engine_class_state()
    yield
    try:
        MLX_Engine.cleanup()
    except Exception:
        # cleanup() failures during teardown must not mask the real test
        # error — best-effort only.
        pass
    _reset_mlx_engine_class_state()


# =====================================================================
# UNIT — module-shape invariants
# =====================================================================
#
# These tests pin down the import structure expected by every other test in
# this file. Without them, a Phase 2 implementation that imports the same
# libraries under different aliases (e.g. `from multiprocessing import
# Process` instead of `import multiprocessing as mp`) would silently bypass
# the patch targets — the mocks would be no-ops and the unit suite would
# pass while testing nothing.
#
# When any of these fail, the fix is one of:
#   - Update the impl to match the expected alias, OR
#   - Update every `patch("src.engines.mlx_engine.<name>")` call site in
#     this file AND the matching invariant test below.


@pytest.mark.unit
class TestModuleImportInvariants:
    """Pin the module-level imports the mocks in this file rely on.

    Post-migration to BaseChatServerEngine: `requests`, `socket`, `atexit`,
    `time` are owned by the base module; only `mp` (multiprocessing) is
    still imported at the MLX module level because `_spawn_child` uses
    `mp.Process` directly.
    """

    def test_mlx_engine_exposes_mp_alias(self):
        """`import multiprocessing as mp` must be at module level."""
        from src.engines import mlx_engine as mod

        assert hasattr(mod, "mp"), (
            "src/engines/mlx_engine.py must `import multiprocessing as mp` at "
            "module level (patch target: src.engines.mlx_engine.mp). Without "
            "it, the mp.Process mocks in this file are no-ops."
        )

    def test_base_chat_server_engine_exposes_requests(self):
        """`import requests` is in the base module post-migration."""
        from src.engines import base_chat_server_engine as base

        assert hasattr(base, "requests")

    def test_base_chat_server_engine_exposes_socket(self):
        """`import socket` is in the base module post-migration."""
        from src.engines import base_chat_server_engine as base

        assert hasattr(base, "socket")

    def test_base_chat_server_engine_exposes_atexit(self):
        """`import atexit` is in the base module post-migration."""
        from src.engines import base_chat_server_engine as base

        assert hasattr(base, "atexit")


# =====================================================================
# NOTE: subprocess pattern unit tests (port pick / probe / start_server /
# atexit) are exercised against the shared base in
# `test_base_chat_server_engine.py`. The MLX-specific tests below cover
# only what MLX_Engine implements directly.


# =====================================================================
# UNIT — _terminate_process (mp.Process API)
# =====================================================================


@pytest.mark.unit
class TestTerminateProcess:
    """Termination must be idempotent, bounded in time, and safe on dead/None."""

    @staticmethod
    def _bounded_join_timeout(proc_mock: MagicMock) -> float:
        """Extract the `timeout=` kwarg passed to proc.join()."""
        assert proc_mock.join.called, "join() was not called"
        call = proc_mock.join.call_args
        timeout = call.kwargs.get("timeout")
        if timeout is None and call.args:
            # Some impls may pass positionally; tolerate either.
            timeout = call.args[0]
        assert timeout is not None, "join() was called without a timeout"
        return float(timeout)

    def test_terminate_then_join_with_bounded_timeout(self):
        """Must call terminate() and join(timeout≤10s) — never a blocking join()."""
        proc = MagicMock()
        proc.is_alive.return_value = True
        MLX_Engine._terminate_process(proc)
        proc.terminate.assert_called_once()
        timeout = self._bounded_join_timeout(proc)
        assert 0 < timeout <= 10, (
            f"join() timeout must be in (0, 10]s to avoid blocking shutdown, " f"got {timeout!r}"
        )

    def test_no_op_when_already_dead(self):
        proc = MagicMock()
        proc.is_alive.return_value = False
        MLX_Engine._terminate_process(proc)
        proc.terminate.assert_not_called()

    def test_force_kill_if_join_times_out(self):
        """If terminate+join didn't kill it, must escalate to .kill()."""
        proc = MagicMock()
        # First poll says alive, after .join() still alive → escalate.
        proc.is_alive.side_effect = [True, True, False]
        MLX_Engine._terminate_process(proc)
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()

    def test_none_proc_is_safe(self):
        """Passing None must not crash (mirrors cpu_engine.py:228-229)."""
        MLX_Engine._terminate_process(None)  # no exception


@pytest.mark.unit
class TestChildLifecycleLogging:
    """What the engine records about its child, and where those records land.

    Every line here goes through the ``erudi`` logger: the root logger has no
    handler in this app, so a record written there reaches neither
    ``backend.log`` nor the Diagnostics page (docs/logging.md).
    """

    @staticmethod
    def _erudi_records(caplog):
        return [r for r in caplog.records if r.name.startswith("erudi")]

    def test_orderly_stop_is_info(self, caplog):
        proc = MagicMock()
        proc.is_alive.side_effect = [True, False]
        proc.exitcode = -15
        with caplog.at_level(logging.INFO, logger="erudi"):
            MLX_Engine._terminate_process(proc)
        records = [r for r in self._erudi_records(caplog) if "Child terminated" in r.getMessage()]
        assert records, "the exit code was not recorded on the erudi logger"
        assert records[0].levelno == logging.INFO

    def test_sigkill_escalation_is_a_warning(self, caplog):
        proc = MagicMock()
        proc.is_alive.side_effect = [True, True, False]
        proc.exitcode = -9
        with caplog.at_level(logging.INFO, logger="erudi"):
            MLX_Engine._terminate_process(proc)
        records = [r for r in self._erudi_records(caplog) if "Child terminated" in r.getMessage()]
        assert records and records[0].levelno == logging.WARNING
        assert "SIGKILL" in records[0].getMessage()

    def test_a_child_that_died_on_its_own_is_a_warning(self, caplog):
        """A nonzero exit code from a child nobody asked to stop is the only
        trace of its death; INFO would keep it off the Diagnostics page."""
        proc = MagicMock()
        proc.is_alive.return_value = False
        proc.exitcode = 1
        with caplog.at_level(logging.INFO, logger="erudi"):
            MLX_Engine._terminate_process(proc)
        records = [r for r in self._erudi_records(caplog) if "Child terminated" in r.getMessage()]
        assert records and records[0].levelno == logging.WARNING
        assert "exitcode=1" in records[0].getMessage()

    def test_a_child_that_exited_cleanly_is_info(self, caplog):
        proc = MagicMock()
        proc.is_alive.return_value = False
        proc.exitcode = 0
        with caplog.at_level(logging.INFO, logger="erudi"):
            MLX_Engine._terminate_process(proc)
        records = [r for r in self._erudi_records(caplog) if "Child terminated" in r.getMessage()]
        assert records and records[0].levelno == logging.INFO

    def test_hardware_probe_failure_reaches_the_erudi_logger(self, caplog, monkeypatch):
        """`_detect_apple_silicon_chip` falls back to None; the reason must be
        readable in backend.log, not written to a handler-less root logger."""
        import subprocess as _subprocess

        def _boom(*args, **kwargs):
            raise _subprocess.SubprocessError("system_profiler is not here")

        monkeypatch.setattr("src.engines.mlx_engine.subprocess.run", _boom)
        with caplog.at_level(logging.WARNING, logger="erudi"):
            assert MLX_Engine._detect_apple_silicon_chip() is None
        assert any(
            "Failed to detect Apple Silicon chip" in r.getMessage()
            for r in self._erudi_records(caplog)
        )

    def test_vision_detection_failure_names_the_error(self, caplog, tmp_path):
        """The verdict is permissive (None) either way, so the record is what
        distinguishes an absent config from a corrupt one."""
        with caplog.at_level(logging.WARNING, logger="erudi"):
            assert MLX_Engine.model_supports_vision(tmp_path / "nope") is None
        messages = [r.getMessage() for r in self._erudi_records(caplog)]
        assert any("vision detection failed" in m for m in messages)
        assert any("Error" in m or "Exception" in m for m in messages)

    def test_spawn_failure_names_the_model_and_the_port(self, tmp_path):
        """`mp.Process.start()` can fail before a child exists (process or fd
        limit). A bare OSError there says nothing about what was spawned."""
        from src.core.exceptions import EngineException

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        proc = MagicMock()
        proc.start.side_effect = OSError("[Errno 35] Resource temporarily unavailable")
        with patch("src.engines.mlx_engine.mp.Process", return_value=proc):
            with pytest.raises(EngineException) as excinfo:
                MLX_Engine._spawn_child(model_path=model_dir, alias="erudi-x", port=9087)
        message = excinfo.value.message
        assert str(model_dir) in message
        assert "9087" in message
        assert "mlx-vlm" in message or "mlx_vlm" in message


@pytest.mark.unit
class TestCleanupAndCache:
    """Cleanup must terminate the subprocess; cache must avoid respawn."""

    def test_cleanup_terminates_subprocess(self):
        proc = MagicMock()
        proc.is_alive.return_value = True
        MLX_Engine._model = {
            "pid": 1,
            "proc": proc,
            "port": 9090,
            "base_url": "http://127.0.0.1:9090",
            "alias": "erudi-x",
            "model_path": "/x",
        }
        MLX_Engine._tokenizer = {"type": "remote", "provider": "mlx-vlm-server"}
        MLX_Engine._model_id = "x"

        MLX_Engine.cleanup()

        proc.terminate.assert_called_once()
        assert MLX_Engine._model is None
        assert MLX_Engine._tokenizer is None
        assert MLX_Engine._model_id is None

    def test_get_model_and_tokenizer_returns_cached_when_same_id(self):
        sentinel_model = {"pid": 7, "proc": MagicMock(), "cached": True}
        sentinel_tokenizer = {"type": "remote"}
        MLX_Engine._model = sentinel_model
        MLX_Engine._tokenizer = sentinel_tokenizer
        MLX_Engine._model_id = "abc"

        with (
            patch.object(MLX_Engine, "_start_server") as mock_start,
            patch.object(MLX_Engine, "_proc_is_alive", return_value=True),
        ):
            model, tokenizer = MLX_Engine.get_model_and_tokenizer(
                llm_id="abc",
                llm_local_path="/whatever",
            )

        mock_start.assert_not_called()
        assert model is sentinel_model
        assert tokenizer is sentinel_tokenizer

    def test_get_model_and_tokenizer_kills_old_when_switching(self, tmp_path):
        """Switching to a different llm_id must terminate the previous proc."""
        new_dir = tmp_path / "new"
        new_dir.mkdir()
        # Minimal valid snapshot: the pre-spawn integrity gate (#88) runs before
        # _start_server and rejects an empty model dir.
        (new_dir / "config.json").write_text('{"model_type": "test"}')
        (new_dir / "tokenizer.json").write_text("{}")
        (new_dir / "model.safetensors").write_bytes(b"x")
        resolved_new = new_dir.resolve()

        old_proc = MagicMock()
        old_proc.is_alive.return_value = True
        MLX_Engine._model = {
            "pid": 7,
            "proc": old_proc,
            "port": 9091,
            "base_url": "http://127.0.0.1:9091",
            "alias": "erudi-old",
            "model_path": "/old",
        }
        MLX_Engine._tokenizer = {"type": "remote", "provider": "mlx-vlm-server"}
        MLX_Engine._model_id = "old"

        new_handle = {
            "pid": 8,
            "proc": MagicMock(),
            "port": 9092,
            "base_url": "http://127.0.0.1:9092",
            "alias": "erudi-new",
            "model_path": str(resolved_new),
        }
        with (
            patch.object(MLX_Engine, "_start_server", return_value=new_handle) as mock_start,
            patch.object(MLX_Engine, "_pick_free_port", return_value=9092),
        ):
            model, tokenizer = MLX_Engine.get_model_and_tokenizer(
                llm_id="new",
                llm_local_path=str(new_dir),
            )

        old_proc.terminate.assert_called_once()
        mock_start.assert_called_once_with(
            model_path=resolved_new,
            alias="erudi-new",
            port=9092,
        )
        assert MLX_Engine._model_id == "new"
        assert model is new_handle


# =====================================================================
# UNIT — _mlx_vlm_server_runner helper module (picklable target)
# =====================================================================


@pytest.mark.unit
class TestMlxVlmServerRunnerHelper:
    """The runner is a module-level function so it can be pickled by spawn."""

    def test_module_function_is_importable(self):
        from src.engines import _mlx_vlm_server_runner

        assert hasattr(_mlx_vlm_server_runner, "run_mlx_vlm_server")
        assert callable(_mlx_vlm_server_runner.run_mlx_vlm_server)

    def test_runner_patches_sys_argv_and_calls_main(self, monkeypatch):
        import sys
        from src.engines import _mlx_vlm_server_runner as runner

        argv = ["mlx_vlm.server", "--model", "/x", "--host", "127.0.0.1", "--port", "9080"]
        captured: dict = {}
        fake_main = MagicMock(side_effect=lambda: captured.update(argv=list(sys.argv)))
        monkeypatch.setattr(runner, "_import_mlx_vlm_server_main", lambda: fake_main)
        monkeypatch.setattr(sys, "argv", ["pytest"])  # auto-restored by monkeypatch

        runner.run_mlx_vlm_server(argv)

        fake_main.assert_called_once()
        assert captured["argv"] == argv

    def test_runner_load_time_patch_roster_for_0613(self):
        """mlx-vlm 0.6.13 runs weight sanitize unconditionally in `load_model`
        (`utils.py:713` plus the per-component Vision/Language/Audio passes at
        `utils.py:715-727`), so the two 0.6.2-era load-time patches stay gone:

          - `_patch_text_only_tied_embeddings` — the `text_only` route is
            genuinely fixed upstream (`Model.sanitize` delegates to the inner
            mlx-lm model), BUT hardware validation showed Gemma3 text-only
            checkpoints no longer take that route at all: 0.6.13 ships a
            native `models/gemma3_text` module whose tied-head sanitize is
            quantization-unaware. That regression is covered by the adapted
            `_patch_gemma3_tied_lm_head_quant` below.
          - `_patch_gemma_shared_kv_sanitize` (#193) — `models/gemma4/language.py`
            `LanguageModel.sanitize` drops `_is_unused_shared_kv_weight` keys
            and is invoked on every load via the unconditional class-level
            pass (`utils.py:717-721`; `gemma4/__init__.py` exports
            `LanguageModel`, `ModelConfig.text_config` exists). gemma4-family
            heads are tied by construction (`embed_tokens.as_linear`, no
            `lm_head` parameter) so the gemma3_text blind spot has no analogue.
        """
        from src.engines import _mlx_vlm_server_runner as runner

        assert not hasattr(runner, "_patch_text_only_tied_embeddings")
        assert not hasattr(runner, "_patch_gemma_shared_kv_sanitize")
        assert callable(runner._patch_gemma3_tied_lm_head_quant)

    def test_runner_applies_inline_thinking_patch_before_main(self, monkeypatch):
        """The thinking-split neutralization must run before the server's main()
        so every `ThinkingStreamState` the server ever builds is already patched
        (#90 — reasoning must stay INLINE in delta.content).

        The sibling in-child patch is stubbed out so this test never imports
        the real mlx-vlm (absent on Linux CI, mutated-in-pytest-process on Mac).
        """
        import sys
        from src.engines import _mlx_vlm_server_runner as runner

        order: list[str] = []
        monkeypatch.setattr(runner, "_patch_gemma3_tied_lm_head_quant", lambda: True)
        monkeypatch.setattr(runner, "_patch_gemma_end_of_turn_stop", lambda: True)
        monkeypatch.setattr(
            runner,
            "_patch_inline_thinking",
            lambda: order.append("thinking-patch") or True,
        )
        fake_main = MagicMock(side_effect=lambda: order.append("main"))
        monkeypatch.setattr(runner, "_import_mlx_vlm_server_main", lambda: fake_main)
        monkeypatch.setattr(sys, "argv", ["pytest"])

        runner.run_mlx_vlm_server(["mlx_vlm.server", "--port", "9080"])

        assert order == ["thinking-patch", "main"]

    def test_runner_applies_tied_lm_head_patch_before_main(self, monkeypatch):
        """The tied-lm_head sanitize completion must run before the server's
        main() loads a model — it is a load-time patch: once `load_model` has
        rejected the checkpoint, there is nothing left to fix.

        Sibling in-child patches are stubbed out so this test never imports
        the real mlx-vlm (absent on Linux CI, mutated-in-pytest-process on Mac).
        """
        import sys
        from src.engines import _mlx_vlm_server_runner as runner

        order: list[str] = []
        monkeypatch.setattr(
            runner,
            "_patch_gemma3_tied_lm_head_quant",
            lambda: order.append("tied-lm-head") or True,
        )
        monkeypatch.setattr(runner, "_patch_gemma_end_of_turn_stop", lambda: True)
        monkeypatch.setattr(runner, "_patch_inline_thinking", lambda: True)
        fake_main = MagicMock(side_effect=lambda: order.append("main"))
        monkeypatch.setattr(runner, "_import_mlx_vlm_server_main", lambda: fake_main)
        monkeypatch.setattr(sys, "argv", ["pytest"])

        runner.run_mlx_vlm_server(["mlx_vlm.server", "--port", "9080"])

        assert order.index("tied-lm-head") < order.index("main")


@pytest.mark.unit
class TestGemmaEndOfTurnStopPatch:
    """`_patch_gemma_end_of_turn_stop` adds Gemma's `<end_of_turn>` to the server's
    stop-token set (#249).

    mlx_vlm (still on 0.6.13, `server/generation.py:_initialize_model`) builds
    `stop_tokens` from `config.eos_token_id` only. Gemma declares `eos_token` =
    `<eos>` (id 1) but its chat template ends turns with `<end_of_turn>` (id 106),
    so without this patch generation runs past the answer and streams the literal
    token + garbage. Verified live on `mlx-community/gemma-3-1b-it-4bit` (2048
    chunks of garbage → 7 chunks, clean). 0.6.13 additionally merges a
    checkpoint's `generation_config.json` eos ids into the config — the patch is
    kept as the checkpoint-independent guarantee and is a no-op when that merge
    already covers id 106.
    """

    def _install_fake_generation(self, monkeypatch, *, tokens, unk=3, base_stop=(1,)):
        """Inject a minimal fake `mlx_vlm.server.generation` with a ResponseGenerator."""
        import sys
        import types

        class _Tok:
            unk_token_id = unk

            def convert_tokens_to_ids(self, t):
                return tokens.get(t, unk)

        class ResponseGenerator:
            def _initialize_model(self):
                # Mirror the real method: eos-derived stop set + tokenizer attr.
                self.tokenizer = _Tok()
                self.stop_tokens = set(base_stop)

        mlx_vlm = types.ModuleType("mlx_vlm")
        server = types.ModuleType("mlx_vlm.server")
        generation = types.ModuleType("mlx_vlm.server.generation")
        generation.ResponseGenerator = ResponseGenerator
        server.generation = generation
        mlx_vlm.server = server
        monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
        monkeypatch.setitem(sys.modules, "mlx_vlm.server", server)
        monkeypatch.setitem(sys.modules, "mlx_vlm.server.generation", generation)
        return ResponseGenerator

    def test_returns_false_when_mlx_vlm_absent(self, monkeypatch):
        import sys
        import types
        from src.engines import _mlx_vlm_server_runner as runner

        bare_server = types.ModuleType("mlx_vlm.server")  # no `generation` attribute
        monkeypatch.setitem(sys.modules, "mlx_vlm.server", bare_server)
        monkeypatch.setitem(sys.modules, "mlx_vlm.server.generation", None)
        assert runner._patch_gemma_end_of_turn_stop() is False

    def test_adds_end_of_turn_id_for_gemma(self, monkeypatch):
        from src.engines import _mlx_vlm_server_runner as runner

        RG = self._install_fake_generation(monkeypatch, tokens={"<end_of_turn>": 106})
        assert runner._patch_gemma_end_of_turn_stop() is True

        rg = RG()
        rg._initialize_model()
        assert 106 in rg.stop_tokens  # the turn-ender is now a stop token
        assert 1 in rg.stop_tokens  # the original eos survives

    def test_no_op_for_non_gemma(self, monkeypatch):
        from src.engines import _mlx_vlm_server_runner as runner

        # Tokenizer doesn't know <end_of_turn> → convert returns unk (3), skipped.
        RG = self._install_fake_generation(monkeypatch, tokens={}, unk=3)
        assert runner._patch_gemma_end_of_turn_stop() is True

        rg = RG()
        rg._initialize_model()
        assert rg.stop_tokens == {1}  # unchanged; unk id never added

    def test_patch_is_idempotent(self, monkeypatch):
        from src.engines import _mlx_vlm_server_runner as runner

        RG = self._install_fake_generation(monkeypatch, tokens={"<end_of_turn>": 106})
        assert runner._patch_gemma_end_of_turn_stop() is True
        first = RG._initialize_model
        assert runner._patch_gemma_end_of_turn_stop() is True
        assert RG._initialize_model is first  # not double-wrapped

    def test_runner_applies_gemma_patch_before_main(self, monkeypatch):
        """The stop-token patch must run before the server's main() loads a model.

        The sibling in-child patch is stubbed out so this test never imports
        the real mlx-vlm (absent on Linux CI, mutated-in-pytest-process on Mac).
        """
        import sys
        from src.engines import _mlx_vlm_server_runner as runner

        order: list[str] = []
        monkeypatch.setattr(runner, "_patch_gemma3_tied_lm_head_quant", lambda: True)
        monkeypatch.setattr(
            runner,
            "_patch_gemma_end_of_turn_stop",
            lambda: order.append("gemma-stop") or True,
        )
        monkeypatch.setattr(runner, "_patch_inline_thinking", lambda: True)
        fake_main = MagicMock(side_effect=lambda: order.append("main"))
        monkeypatch.setattr(runner, "_import_mlx_vlm_server_main", lambda: fake_main)
        monkeypatch.setattr(sys, "argv", ["pytest"])

        runner.run_mlx_vlm_server(["mlx_vlm.server", "--port", "9080"])

        assert order.index("gemma-stop") < order.index("main")


@pytest.mark.unit
class TestGemma3TiedLmHeadQuantPatch:
    """`_patch_gemma3_tied_lm_head_quant` completes mlx-vlm 0.6.13's tied-head
    sanitize for quantized Gemma3 checkpoints (#273).

    Hardware-found regression: `mlx-community/gemma-3-270m-it-4bit`
    (`model_type == "gemma3_text"`, tied embeddings, no `lm_head.*` tensors)
    fails `mlx_vlm.utils.load` on real 0.6.13 with

        ValueError: Expected shape (262144, 640) but received shape
        (262144, 80) for parameter language_model.lm_head.weight

    because `gemma3.LanguageModel.sanitize` copies ONLY
    `model.embed_tokens.weight` (the 4-bit packed tensor) to `lm_head.weight`
    without the `.scales`/`.biases` sidecars, so `nn.quantize`'s
    `f"{p}.scales" in weights` predicate leaves `lm_head` an UNQUANTIZED
    `nn.Linear`. The patch copies the sidecars whenever the upstream tied
    fallback fired on a quantized embedding, and only then.

    The fake below is a behavioral double of
    `mlx_vlm/models/gemma3/language.py:LanguageModel.sanitize` on 0.6.13.
    """

    _PFX = "language_model."

    def _install_fake_gemma3_language(self, monkeypatch):
        """Inject a minimal fake `mlx_vlm.models.gemma3.language` module."""
        import sys
        import types

        class LanguageModel:
            def sanitize(self, weights):
                # Mirror upstream 0.6.13: the guard checks the UNPREFIXED key
                # (always absent after gemma3_text.Model.sanitize prefixed
                # everything), and copies only the packed weight.
                if "lm_head.weight" not in weights:
                    weights["language_model.lm_head.weight"] = weights[
                        "language_model.model.embed_tokens.weight"
                    ]
                return {
                    k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
                }

        mlx_vlm = types.ModuleType("mlx_vlm")
        models = types.ModuleType("mlx_vlm.models")
        gemma3 = types.ModuleType("mlx_vlm.models.gemma3")
        language = types.ModuleType("mlx_vlm.models.gemma3.language")
        language.LanguageModel = LanguageModel
        gemma3.language = language
        models.gemma3 = gemma3
        mlx_vlm.models = models
        monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
        monkeypatch.setitem(sys.modules, "mlx_vlm.models", models)
        monkeypatch.setitem(sys.modules, "mlx_vlm.models.gemma3", gemma3)
        monkeypatch.setitem(sys.modules, "mlx_vlm.models.gemma3.language", language)
        return LanguageModel

    def _tied_quantized_weights(self):
        """Checkpoint-shaped dict: tied 4-bit embeddings, no lm_head tensors."""
        p = self._PFX
        return {
            f"{p}model.embed_tokens.weight": object(),
            f"{p}model.embed_tokens.scales": object(),
            f"{p}model.embed_tokens.biases": object(),
            f"{p}model.layers.0.mlp.down_proj.weight": object(),
        }

    def test_returns_false_when_mlx_vlm_absent(self, monkeypatch):
        import sys
        import types
        from src.engines import _mlx_vlm_server_runner as runner

        bare = types.ModuleType("mlx_vlm.models.gemma3")  # no `language` attr
        monkeypatch.setitem(sys.modules, "mlx_vlm.models.gemma3", bare)
        monkeypatch.setitem(sys.modules, "mlx_vlm.models.gemma3.language", None)
        assert runner._patch_gemma3_tied_lm_head_quant() is False

    def test_copies_quant_sidecars_when_tied_fallback_fires(self, monkeypatch):
        from src.engines import _mlx_vlm_server_runner as runner

        LM = self._install_fake_gemma3_language(monkeypatch)
        assert runner._patch_gemma3_tied_lm_head_quant() is True

        p = self._PFX
        weights = self._tied_quantized_weights()
        out = LM().sanitize(dict(weights))

        # Upstream behavior preserved: packed weight aliased to the head.
        assert out[f"{p}lm_head.weight"] is weights[f"{p}model.embed_tokens.weight"]
        # Patch completion: the quant sidecars follow, so nn.quantize's
        # `f"{p}.scales" in weights` predicate converts lm_head too.
        assert out[f"{p}lm_head.scales"] is weights[f"{p}model.embed_tokens.scales"]
        assert out[f"{p}lm_head.biases"] is weights[f"{p}model.embed_tokens.biases"]

    def test_no_op_for_unquantized_tied_checkpoint(self, monkeypatch):
        from src.engines import _mlx_vlm_server_runner as runner

        LM = self._install_fake_gemma3_language(monkeypatch)
        assert runner._patch_gemma3_tied_lm_head_quant() is True

        p = self._PFX
        weights = {
            f"{p}model.embed_tokens.weight": object(),  # float, no sidecars
            f"{p}model.layers.0.mlp.down_proj.weight": object(),
        }
        out = LM().sanitize(dict(weights))

        assert out[f"{p}lm_head.weight"] is weights[f"{p}model.embed_tokens.weight"]
        assert f"{p}lm_head.scales" not in out
        assert f"{p}lm_head.biases" not in out

    def test_no_op_when_checkpoint_ships_its_own_lm_head(self, monkeypatch):
        from src.engines import _mlx_vlm_server_runner as runner

        LM = self._install_fake_gemma3_language(monkeypatch)
        assert runner._patch_gemma3_tied_lm_head_quant() is True

        p = self._PFX
        own_head_w, own_head_s = object(), object()
        weights = self._tied_quantized_weights()
        weights[f"{p}lm_head.weight"] = own_head_w
        weights[f"{p}lm_head.scales"] = own_head_s
        out = LM().sanitize(dict(weights))

        # A genuinely shipped (untied) head keeps its own sidecars untouched.
        assert out[f"{p}lm_head.scales"] is own_head_s
        assert f"{p}lm_head.biases" not in out

    def test_patch_is_idempotent(self, monkeypatch):
        from src.engines import _mlx_vlm_server_runner as runner

        LM = self._install_fake_gemma3_language(monkeypatch)
        assert runner._patch_gemma3_tied_lm_head_quant() is True
        first = LM.sanitize
        assert runner._patch_gemma3_tied_lm_head_quant() is True
        assert LM.sanitize is first  # not double-wrapped

        # And re-sanitizing already-completed weights changes nothing.
        p = self._PFX
        weights = self._tied_quantized_weights()
        once = LM().sanitize(dict(weights))
        twice = LM().sanitize(dict(once))
        assert twice == once


@pytest.mark.unit
class TestInlineThinkingPatch:
    """`_patch_inline_thinking` neutralizes mlx-vlm's server-side thinking split (#90).

    On the pinned mlx-vlm 0.6.13, `ThinkingStreamState` routes everything between
    `<think>` boundaries into `delta.reasoning` — which ChatOpenAI drops, so
    reasoning silently vanishes. The patch forces every state instance to start
    OUTSIDE thinking with unmatchable markers, so the raw model text (including
    inline `<think>...</think>`) flows through `delta.content` and the runner's
    single ThinkSplitter handles it — identical to llama-server with
    `--reasoning-format none`.

    0.6.13 adds a second splitting path the patch must also neutralize: the
    `make_response_stream_state` factory prefers a `ResponseTemplateStreamState`
    (a transformers response-template parser) whenever the tokenizer exposes a
    `response_template`, bypassing `ThinkingStreamState` entirely. The patch
    disables that bypass by neutralizing `_response_template_tokenizer` — a
    call-time global inside the factory, so it works even though the route
    modules from-import the factory at package import time.

    The fake below is a behavioral double: `__init__`/`feed`, the helpers, and
    the factory are copied from the real mlx-vlm 0.6.13
    `server/responses_state.py`, so the assertions exercise the exact upstream
    logic being neutralized while staying runnable on Linux CI (no mlx-vlm
    installed).
    """

    def _install_fake_responses_state(self, monkeypatch):
        """Inject `mlx_vlm.server.responses_state` with the real 0.6.13 splitter logic."""
        import sys
        import types
        from dataclasses import dataclass
        from typing import Optional, Tuple

        _CONTENT_MARKERS = ("<|START_TEXT|>", "<|END_TEXT|>")

        def _strip_content_markers(text):
            for marker in _CONTENT_MARKERS:
                text = text.replace(marker, "")
            return text

        @dataclass
        class ThinkingStreamDelta:
            reasoning: Optional[str] = None
            content: Optional[str] = None
            thinking_closed: bool = False

        class ThinkingStreamState:
            """Verbatim port of mlx-vlm 0.6.13 server/responses_state.py:40-171."""

            _DEFAULT_OPEN_CLOSE_MARKERS = (
                ("<|channel>thought", "<channel|>"),
                ("<think>", "</think>"),
                ("<|START_THINKING|>", "<|END_THINKING|>"),
            )

            def __init__(
                self,
                enable_thinking: bool = False,
                thinking_start_token: Optional[str] = None,
                thinking_end_token: Optional[str] = None,
            ):
                self.open_close_markers = self._build_open_close_markers(
                    thinking_start_token, thinking_end_token
                )
                self.open_markers = tuple(m for m, _ in self.open_close_markers)
                self.close_markers = tuple(m for _, m in self.open_close_markers)
                self.in_thinking = bool(enable_thinking)
                self.thinking_done = False
                self.buffer = ""

            def feed(self, text, last=False):
                self.buffer += text or ""
                reasoning = []
                content = []
                thinking_closed = False
                while self.buffer:
                    if self.in_thinking:
                        idx, marker = self._find_first(self.buffer, self.close_markers)
                        if idx < 0:
                            emit, self.buffer = self._split_partial(self.buffer, self.close_markers)
                            emit = self._strip_open_marker(emit)
                            if emit:
                                reasoning.append(emit)
                            break
                        before = self._strip_open_marker(self.buffer[:idx])
                        if before:
                            reasoning.append(before)
                        self.buffer = self.buffer[idx + len(marker) :].lstrip("\n")
                        self.in_thinking = False
                        self.thinking_done = True
                        thinking_closed = True
                        continue
                    if self.thinking_done:
                        emit, self.buffer = self._split_partial(self.buffer, _CONTENT_MARKERS)
                        emit = _strip_content_markers(emit)
                        if emit:
                            content.append(emit)
                        break
                    idx, marker = self._find_first(self.buffer, self.open_markers)
                    if idx < 0:
                        emit, self.buffer = self._split_partial(self.buffer, self.open_markers)
                        emit = _strip_content_markers(emit)
                        if emit:
                            content.append(emit)
                        break
                    if idx:
                        emit = _strip_content_markers(self.buffer[:idx])
                        if emit:
                            content.append(emit)
                    self.buffer = self.buffer[idx + len(marker) :].lstrip("\n")
                    self.in_thinking = True
                if last and self.buffer:
                    held, self.buffer = self.buffer, ""
                    if self.in_thinking:
                        reasoning.append(self._strip_open_marker(held))
                    else:
                        content.append(_strip_content_markers(held))
                return ThinkingStreamDelta(
                    reasoning="".join(reasoning) or None,
                    content="".join(content) or None,
                    thinking_closed=thinking_closed,
                )

            @classmethod
            def _build_open_close_markers(cls, thinking_start_token, thinking_end_token):
                markers = []
                if thinking_start_token and thinking_end_token:
                    markers.append((thinking_start_token, thinking_end_token))
                for marker_pair in cls._DEFAULT_OPEN_CLOSE_MARKERS:
                    if marker_pair not in markers:
                        markers.append(marker_pair)
                return tuple(markers)

            @staticmethod
            def _find_first(text, markers) -> Tuple[int, str]:
                found_idx = -1
                found_marker = ""
                for marker in markers:
                    idx = text.find(marker)
                    if idx >= 0 and (found_idx < 0 or idx < found_idx):
                        found_idx = idx
                        found_marker = marker
                return found_idx, found_marker

            @staticmethod
            def _split_partial(text, markers) -> Tuple[str, str]:
                hold = 0
                for marker in markers:
                    max_len = min(len(marker) - 1, len(text))
                    for length in range(max_len, 0, -1):
                        if text.endswith(marker[:length]):
                            hold = max(hold, length)
                            break
                if hold:
                    return text[:-hold], text[-hold:]
                return text, ""

            def _strip_open_marker(self, text):
                for marker in self.open_markers:
                    if marker in text:
                        before, after = text.split(marker, 1)
                        return before + after.lstrip("\n")
                return text

        class ResponseTemplateStreamState:
            """Stand-in for 0.6.13's template-parser splitter (the bypass)."""

            def __init__(self, parser):
                self.parser = parser

        mlx_vlm = types.ModuleType("mlx_vlm")
        server = types.ModuleType("mlx_vlm.server")
        responses_state = types.ModuleType("mlx_vlm.server.responses_state")
        responses_state.ThinkingStreamDelta = ThinkingStreamDelta
        responses_state.ThinkingStreamState = ThinkingStreamState
        responses_state.ResponseTemplateStreamState = ResponseTemplateStreamState

        def _response_template_tokenizer(processor):
            """Verbatim port of mlx-vlm 0.6.13 server/responses_state.py:214-220."""
            if processor is None:
                return None
            tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
            if getattr(tokenizer, "response_template", None) is None:
                return None
            return tokenizer

        def make_response_stream_state(
            processor,
            enable_thinking=False,
            thinking_start_token=None,
            thinking_end_token=None,
        ):
            """Verbatim port of mlx-vlm 0.6.13 server/responses_state.py:223-240,
            minus the logger fallback. Resolves `_response_template_tokenizer`
            through the module globals at call time — the seam the patch uses.
            """
            tokenizer = responses_state._response_template_tokenizer(processor)
            if tokenizer is not None and hasattr(tokenizer, "get_response_parser"):
                return ResponseTemplateStreamState(tokenizer.get_response_parser(prefix=""))
            return ThinkingStreamState(
                enable_thinking,
                thinking_start_token,
                thinking_end_token,
            )

        responses_state._response_template_tokenizer = _response_template_tokenizer
        responses_state.make_response_stream_state = make_response_stream_state
        server.responses_state = responses_state
        mlx_vlm.server = server
        monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
        monkeypatch.setitem(sys.modules, "mlx_vlm.server", server)
        monkeypatch.setitem(sys.modules, "mlx_vlm.server.responses_state", responses_state)
        return ThinkingStreamState

    @staticmethod
    def _feed_all(state, chunks):
        """Feed chunks and concatenate the reasoning/content channels."""
        reasoning, content = [], []
        for chunk in chunks:
            delta = state.feed(chunk)
            if delta.reasoning:
                reasoning.append(delta.reasoning)
            if delta.content:
                content.append(delta.content)
        return "".join(reasoning), "".join(content)

    def test_returns_false_when_mlx_vlm_absent(self, monkeypatch):
        import sys
        import types
        from src.engines import _mlx_vlm_server_runner as runner

        # Simulate a host without mlx-vlm (Linux CI): the parent package exists
        # but the `responses_state` submodule import raises (None sys.modules entry).
        bare_server = types.ModuleType("mlx_vlm.server")
        monkeypatch.setitem(sys.modules, "mlx_vlm.server", bare_server)
        monkeypatch.setitem(sys.modules, "mlx_vlm.server.responses_state", None)
        assert runner._patch_inline_thinking() is False

    def test_unpatched_state_splits_thinking(self, monkeypatch):
        """Baseline pin of the 0.6.13 behavior being fixed: with enable_thinking
        the state starts IN thinking, so everything before `</think>` lands in
        the reasoning channel and the tags never reach content.
        """
        state_cls = self._install_fake_responses_state(monkeypatch)

        state = state_cls(enable_thinking=True)
        reasoning, content = self._feed_all(
            state, ["<think>step ", "by step</think>", "The answer is 4."]
        )

        assert reasoning == "step by step"
        assert content == "The answer is 4."

    def test_patched_state_keeps_thinking_inline(self, monkeypatch):
        """After the patch, the same stream flows 100% through content —
        inline `<think>...</think>` included, reasoning channel silent.
        """
        from src.engines import _mlx_vlm_server_runner as runner

        state_cls = self._install_fake_responses_state(monkeypatch)
        assert runner._patch_inline_thinking() is True

        state = state_cls(enable_thinking=True)
        reasoning, content = self._feed_all(
            state, ["<think>step ", "by step</think>", "The answer is 4."]
        )

        assert reasoning == ""
        assert content == "<think>step by step</think>The answer is 4."

    def test_patched_factory_skips_template_parser_bypass(self, monkeypatch):
        """0.6.13's `make_response_stream_state` prefers a template-parser
        splitter when the tokenizer exposes a `response_template` — a path that
        routes reasoning to `delta.reasoning` while bypassing
        `ThinkingStreamState` entirely. After the patch, the factory must fall
        through to the (neutralized) `ThinkingStreamState` for every processor.
        """
        import sys
        from types import SimpleNamespace

        from src.engines import _mlx_vlm_server_runner as runner

        state_cls = self._install_fake_responses_state(monkeypatch)
        responses_state = sys.modules["mlx_vlm.server.responses_state"]

        tokenizer = SimpleNamespace(
            response_template="{% generation %}",
            get_response_parser=lambda prefix: object(),
        )
        processor = SimpleNamespace(tokenizer=tokenizer)

        # Baseline pin: unpatched, the factory takes the bypass.
        unpatched = responses_state.make_response_stream_state(processor)
        assert isinstance(unpatched, responses_state.ResponseTemplateStreamState)

        assert runner._patch_inline_thinking() is True

        patched = responses_state.make_response_stream_state(processor, enable_thinking=True)
        assert isinstance(patched, state_cls)
        assert patched.in_thinking is False  # and it is the neutralized state

    def test_patched_state_has_no_partial_marker_holdback(self, monkeypatch):
        """A chunk ending mid-`<think` must flush immediately once patched:
        the unmatchable sentinel markers share no prefix with model text, so
        `_split_partial` never holds back a suffix (no latency artifacts).
        """
        from src.engines import _mlx_vlm_server_runner as runner

        state_cls = self._install_fake_responses_state(monkeypatch)
        assert runner._patch_inline_thinking() is True

        state = state_cls(enable_thinking=False)
        delta = state.feed("text ending in <thin")
        assert delta.content == "text ending in <thin"
        assert delta.reasoning is None

    def test_patched_state_still_strips_content_markers(self, monkeypatch):
        """Upstream `<|START_TEXT|>`/`<|END_TEXT|>` stripping must survive the
        patch — only the thinking split is neutralized.
        """
        from src.engines import _mlx_vlm_server_runner as runner

        state_cls = self._install_fake_responses_state(monkeypatch)
        assert runner._patch_inline_thinking() is True

        state = state_cls(enable_thinking=False)
        delta = state.feed("<|START_TEXT|>hello<|END_TEXT|>")
        assert delta.content == "hello"
        assert delta.reasoning is None

    def test_patch_is_idempotent(self, monkeypatch):
        from src.engines import _mlx_vlm_server_runner as runner

        state_cls = self._install_fake_responses_state(monkeypatch)
        assert runner._patch_inline_thinking() is True
        first = state_cls.__init__
        assert runner._patch_inline_thinking() is True
        assert state_cls.__init__ is first  # not double-wrapped


# =====================================================================
# UNIT — MLX_Engine spawn argv + class attributes + payload model value
# =====================================================================


@pytest.mark.unit
class TestSpawnArgv:
    """`_spawn_child` must target the mlx-vlm runner with a 127.0.0.1 argv."""

    def test_spawn_argv_targets_mlx_vlm_runner(self, tmp_path):
        from src.engines._mlx_vlm_server_runner import run_mlx_vlm_server

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        captured: dict = {}

        def _fake_process(*, target, args, daemon):
            captured["target"] = target
            captured["argv"] = args[0]
            proc = MagicMock()
            proc.pid = 4321
            return proc

        with patch("src.engines.mlx_engine.mp.Process", side_effect=_fake_process):
            handle = MLX_Engine._spawn_child(
                model_path=model_dir,
                alias="erudi-x",
                port=9087,
            )

        assert captured["target"] is run_mlx_vlm_server
        argv = list(captured["argv"])
        # The per-spawn credential is asserted by `TestSpawnApiKey`; strip it
        # here so the rest of the argv is pinned verbatim.
        key_at = argv.index("--api-key")
        del argv[key_at : key_at + 2]
        assert argv == [
            "mlx_vlm.server",
            "--model",
            str(model_dir),
            "--host",
            "127.0.0.1",
            "--port",
            "9087",
            "--log-level",
            "INFO",
            "--enable-thinking",
        ]
        assert handle["port"] == 9087
        assert handle["alias"] == "erudi-x"
        assert handle["model_path"] == str(model_dir)
        assert handle["base_url"] == "http://127.0.0.1:9087"

    def test_spawn_does_not_export_dead_thinking_env_sentinel(self, tmp_path, monkeypatch):
        """MLX_VLM_THINKING_START_TOKEN exists on mlx-vlm 0.6.13 but cannot
        express "never split": `_build_open_close_markers` always APPENDS the
        built-in marker families after any custom pair, and it needs both a
        start AND an end token to register at all. Inline delivery is owned by
        the in-child `_patch_inline_thinking` monkeypatch instead, so
        `_spawn_child` must not touch the parent's environment.
        """
        import os

        monkeypatch.delenv("MLX_VLM_THINKING_START_TOKEN", raising=False)
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        with patch("src.engines.mlx_engine.mp.Process", return_value=MagicMock(pid=1)):
            MLX_Engine._spawn_child(model_path=model_dir, alias="erudi-x", port=9087)

        assert "MLX_VLM_THINKING_START_TOKEN" not in os.environ


def _spawn_mlx_child(tmp_path):
    """Run `_spawn_child` with `mp.Process` stubbed; return (handle, argv)."""
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    captured: dict = {}

    def _fake_process(*, target, args, daemon):
        captured["argv"] = list(args[0])
        return MagicMock(pid=4321)

    with patch("src.engines.mlx_engine.mp.Process", side_effect=_fake_process):
        handle = MLX_Engine._spawn_child(model_path=model_dir, alias="erudi-x", port=9087)
    return handle, captured["argv"]


@pytest.mark.unit
class TestSpawnApiKey:
    """mlx_vlm.server must not be left open to everything on the loopback.

    Spawned without `--api-key`, mlx_vlm.server authenticates NOTHING: any
    caller that can reach 127.0.0.1 -- another local process, or a web page
    the user has open, since a browser can POST across origins to a loopback
    port -- can run its own inference on the loaded model. mlx-vlm's own
    `--api-key` guard covers every route it registers, `/health` included
    (`TestMlxVlmApiKeyGuard` pins that upstream fact). These tests pin the
    same spawn contract `TestSpawnHardeningFlags` pins for llama-server.
    """

    def test_api_key_flag_carries_a_non_empty_secret(self, tmp_path):
        """`--api-key` is mlx-vlm's own flag; an empty value would leave the
        env var unset and the server unauthenticated, so the value itself is
        asserted, not just the flag."""
        _handle, argv = _spawn_mlx_child(tmp_path)
        assert "--api-key" in argv
        key = argv[argv.index("--api-key") + 1]
        assert isinstance(key, str) and len(key) >= 32

    def test_each_spawn_gets_a_different_key(self, tmp_path):
        """Per-spawn generation bounds a disclosure to the life of one child:
        swapping models (or a crash-respawn) invalidates a scraped key."""
        _h1, first = _spawn_mlx_child(tmp_path)
        _h2, second = _spawn_mlx_child(tmp_path)
        assert first[first.index("--api-key") + 1] != second[second.index("--api-key") + 1]

    def test_handle_exposes_the_key_to_the_callers_that_need_it(self, tmp_path):
        """The readiness probe and the ChatOpenAI client both reach the child
        only through the spawn handle; a key kept local to `_spawn_child`
        would lock Erudi out of its own server."""
        handle, argv = _spawn_mlx_child(tmp_path)
        assert handle["api_key"] == argv[argv.index("--api-key") + 1]

    def test_the_key_never_reaches_the_logs(self, tmp_path, caplog):
        """Backend logs are written to a world-readable temp file and shipped
        in bug reports; a key printed there outlives the process that used it."""
        import logging

        with caplog.at_level(logging.DEBUG):
            handle, _argv = _spawn_mlx_child(tmp_path)
        key = handle["api_key"]
        assert key
        for record in caplog.records:
            assert key not in record.getMessage()


@pytest.mark.unit
class TestMlxVlmApiKeyGuard:
    """The upstream fact the MLX key relies on, pinned against the installed mlx-vlm.

    `MLX_Engine` passes `--api-key` and nothing else: it is mlx-vlm's own guard
    (`_require_management_api_key`, a dependency of the router every inference
    route is registered on) that turns the key into 401s. If an mlx-vlm bump
    ever moved `/v1/chat/completions` or `/health` off that router, the key
    would guard nothing and this test is what says so. Skips where mlx-vlm is
    not installed (Linux CI).
    """

    ENV = "MLX_VLM_SERVER_API_KEY"

    @pytest.fixture
    def client(self, monkeypatch):
        pytest.importorskip("mlx_vlm.server.app")
        from fastapi.testclient import TestClient
        from mlx_vlm.server.app import app

        monkeypatch.setenv(self.ENV, "s3cret-token")
        # No lifespan (no model is loaded); with the key a request reaches the
        # route and fails there in whatever way mlx-vlm sees fit -- anything
        # but 401 is the proof that the guard, not the route, was the gate.
        return TestClient(app, raise_server_exceptions=False)

    def test_chat_completions_requires_the_key(self, client):
        body = {"model": "x", "messages": [{"role": "user", "content": "ping"}]}
        assert client.post("/v1/chat/completions", json=body).status_code == 401
        headers = {"Authorization": "Bearer s3cret-token"}
        assert client.post("/v1/chat/completions", json=body, headers=headers).status_code != 401

    def test_health_requires_the_key(self, client):
        assert client.get("/health").status_code == 401
        headers = {"Authorization": "Bearer s3cret-token"}
        assert client.get("/health", headers=headers).status_code != 401


@pytest.mark.unit
class TestClassAttributes:
    """Pin the BaseChatServerEngine config the swap retargets."""

    def test_server_name_is_mlx_vlm(self):
        assert MLX_Engine._server_name == "mlx_vlm.server"

    def test_tokenizer_provider_is_mlx_vlm(self):
        assert MLX_Engine._tokenizer_provider == "mlx-vlm-server"

    def test_port_range_start_in_canonical_block(self):
        # MLX owns the top slice of Erudi's 271xx–273xx block: 27300–27399,
        # clear of llama.cpp (27200–27299) and the backend (27182–27199).
        assert MLX_Engine._port_range_start == 27300


@pytest.mark.unit
class TestPayloadModelValue:
    """mlx-vlm requires the real preloaded model path, not a sentinel."""

    def test_returns_model_path(self):
        handle = {"alias": "erudi-x", "model_path": "/models/erudi-x"}
        assert MLX_Engine._payload_model_value(handle) == "/models/erudi-x"


@pytest.mark.unit
class TestTranslatePayloadKwargsSeed:
    """mlx_vlm.server seeds its sampler from ``DEFAULT_SEED`` whenever a request
    carries no ``seed`` (``generation.py``: ``self.seed = DEFAULT_SEED if seed is
    None``), so every generation replayed byte-for-byte whatever the
    temperature: four fresh conversations at 0.6 / 0.95 / top_k 20 gave the same
    answer and reasoning trace. The MLX translation must stamp a fresh random
    seed on every request; llama-server samples randomly by default and gets
    none (see test_cpu_engine_server)."""

    def test_adds_an_integer_seed(self):
        out = MLX_Engine._translate_payload_kwargs(
            {
                "repetition_penalty": 1.1,
                "repetition_context_size": 64,
            }
        )
        assert isinstance(out["seed"], int)
        # mlx_vlm masks the seed to 32 bits: keep it non-negative and in range.
        assert 0 <= out["seed"] < 2**32

    def test_seed_differs_between_two_requests(self):
        seeds = {MLX_Engine._translate_payload_kwargs({})["seed"] for _ in range(8)}
        assert len(seeds) > 1

    def test_keeps_the_hf_names_and_the_thinking_flag_untouched(self):
        kwargs = {
            "repetition_penalty": 1.1,
            "repetition_context_size": 64,
            "top_k": 20,
            "min_p": 0.0,
            "enable_thinking": False,
        }
        out = MLX_Engine._translate_payload_kwargs(kwargs)
        assert {k: v for k, v in out.items() if k != "seed"} == kwargs
        # The caller's dict is not mutated.
        assert "seed" not in kwargs

    def test_does_not_override_an_explicit_seed(self):
        out = MLX_Engine._translate_payload_kwargs({"seed": 7})
        assert out["seed"] == 7


# =====================================================================
# INTEGRATION — real mlx_lm.server subprocess + real model
# =====================================================================


@pytest.mark.mlx_only
class TestSubprocessReal:
    """Spawn a real `mlx_lm.server` against a small downloaded model.

    Uses the session-scoped `mlx_test_model_path` fixture, which skips on
    non-Apple-Silicon hosts. Covers the subprocess lifecycle (spawn / health /
    cleanup / swap); token streaming is exercised end-to-end through the live
    ChatOpenAI path in `TestE2EConversationsRealMLX` and the regression classes
    below.
    """

    def test_subprocess_starts_and_serves_health(self, mlx_test_model_path):
        import requests

        try:
            model, tokenizer = MLX_Engine.get_model_and_tokenizer(
                llm_id="qwen-test",
                llm_local_path=str(mlx_test_model_path),
            )
            assert model["proc"].is_alive(), "subprocess died right after spawn"
            r = requests.get(
                f"{model['base_url']}/health",
                timeout=5,
                headers={"Authorization": f"Bearer {model['api_key']}"},
            )
            assert r.status_code == 200
            assert tokenizer == {"type": "remote", "provider": "mlx-vlm-server"}
        finally:
            MLX_Engine.cleanup()

    def test_child_refuses_unauthenticated_requests_and_still_chats(self, mlx_test_model_path):
        """The real child, with the engine's own handle: mlx-vlm's `--api-key`
        guard rejects an unauthenticated (or wrongly keyed) chat request and
        `/health` with 401, while the backend's probe (already passed inside
        `get_model_and_tokenizer`) and the ChatOpenAI client, which read the
        key from the handle, still drive the model."""
        import requests

        try:
            model, _ = MLX_Engine.get_model_and_tokenizer(
                llm_id="qwen-test",
                llm_local_path=str(mlx_test_model_path),
            )
            base_url = model["base_url"]
            chat_body = {
                "model": model["model_path"],
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
                "stream": False,
            }
            r = requests.post(f"{base_url}/v1/chat/completions", json=chat_body, timeout=10)
            assert r.status_code == 401, r.text
            assert r.headers.get("WWW-Authenticate") == "Bearer"
            r = requests.get(f"{base_url}/health", timeout=5)
            assert r.status_code == 401
            r = requests.post(
                f"{base_url}/v1/chat/completions",
                json=chat_body,
                timeout=10,
                headers={"Authorization": "Bearer not-the-key"},
            )
            assert r.status_code == 401

            chat = _build_real_mlx_chat_model("qwen-test", mlx_test_model_path, max_tokens=8)
            reply = chat.invoke("Say hi.")
            assert isinstance(reply.content, str) and reply.content.strip()
        finally:
            MLX_Engine.cleanup()

    def test_real_child_writes_its_own_log_and_never_its_key(self, mlx_test_model_path):
        """The real mlx-vlm child, with the real argv: what it prints must land
        in its per-spawn file (that file is the whole crash report), and the
        `--api-key` it was spawned with must not -- the file is quoted in bug
        reports, and a leaked key outlives the process that used it."""
        from src.engines import mlx_child_log

        try:
            model, _ = MLX_Engine.get_model_and_tokenizer(
                llm_id="qwen-test",
                llm_local_path=str(mlx_test_model_path),
            )
            proc = model["proc"]
            log_path = MLX_Engine._child_log_path_of(proc)
            assert log_path, "the spawn captured no output file"
            assert Path(log_path).name == f"mlx-child-{model['port']}.log"

            captured = Path(log_path).read_text(encoding="utf-8", errors="replace")
            assert captured.strip(), "the child wrote nothing to its log"
            assert model["api_key"] not in captured
            # And the same file is what a crash report would quote.
            assert mlx_child_log.read_child_log_tail(log_path)
            assert model["api_key"] not in MLX_Engine._read_child_output(proc)
        finally:
            MLX_Engine.cleanup()

    def test_an_orderly_cleanup_leaves_no_child_log_behind(self, mlx_test_model_path):
        model, _ = MLX_Engine.get_model_and_tokenizer(
            llm_id="qwen-test",
            llm_local_path=str(mlx_test_model_path),
        )
        log_path = MLX_Engine._child_log_path_of(model["proc"])
        assert log_path and Path(log_path).exists()

        MLX_Engine.cleanup()

        assert not Path(log_path).exists()

    def test_cleanup_kills_subprocess_and_frees_port(self, mlx_test_model_path):
        model, _ = MLX_Engine.get_model_and_tokenizer(
            llm_id="qwen-test",
            llm_local_path=str(mlx_test_model_path),
        )
        port = model["port"]
        proc = model["proc"]
        assert proc.is_alive()

        MLX_Engine.cleanup()

        for _ in range(20):
            if not proc.is_alive():
                break
            time.sleep(0.1)
        assert not proc.is_alive(), "subprocess survived cleanup()"

        with _stdlib_socket.socket(_stdlib_socket.AF_INET, _stdlib_socket.SOCK_STREAM) as s:
            s.setsockopt(_stdlib_socket.SOL_SOCKET, _stdlib_socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))  # must not raise

    def test_switch_model_terminates_old_subprocess(self, mlx_test_model_path):
        """Calling get_model_and_tokenizer with a new llm_id must kill the old proc."""
        try:
            m1, _ = MLX_Engine.get_model_and_tokenizer(
                llm_id="qwen-test",
                llm_local_path=str(mlx_test_model_path),
            )
            old_proc = m1["proc"]
            assert old_proc.is_alive()

            m2, _ = MLX_Engine.get_model_and_tokenizer(
                llm_id="qwen-test-bis",
                llm_local_path=str(mlx_test_model_path),
            )
            for _ in range(20):
                if not old_proc.is_alive():
                    break
                time.sleep(0.1)
            assert not old_proc.is_alive(), "old subprocess was not terminated on switch"
            assert m2["proc"] is not old_proc
            assert m2["proc"].is_alive()
        finally:
            MLX_Engine.cleanup()

    async def test_idle_tick_reaps_real_subprocess_without_deadlock(self, mlx_test_model_path):
        """Wave C regression: the idle-cleanup tick reaps a REAL child subprocess
        without the old reentrant-lock deadlock (the monitor held ``cls._lock``
        then called a ``cleanup()`` that re-acquired the same non-reentrant lock).
        The tick must complete promptly and the child must actually die.
        """
        import asyncio
        from datetime import datetime, timedelta

        MLX_Engine.get_model_and_tokenizer(
            llm_id="qwen-test",
            llm_local_path=str(mlx_test_model_path),
        )
        proc = MLX_Engine._model["proc"]
        assert proc.is_alive()
        try:
            # Backdate the idle clock so the next tick treats the model as reapable.
            MLX_Engine._last_used = datetime.now() - timedelta(seconds=10_000)
            assert MLX_Engine._should_cleanup() is True
            # Must NOT hang — the old reentrant-lock path would deadlock here.
            await asyncio.wait_for(MLX_Engine._cleanup_tick(), timeout=15)
            assert MLX_Engine._model is None  # engine state reset
            for _ in range(30):
                if not proc.is_alive():
                    break
                time.sleep(0.1)
            assert not proc.is_alive(), "real subprocess was not terminated by the idle tick"
        finally:
            MLX_Engine.cleanup()


def _build_real_mlx_chat_model(llm_id, model_path, *, max_tokens):
    """Spawn the real mlx_lm.server for `model_path` and wrap it as the live
    ChatOpenAI model (`build_chat_model`). Caller must `config.LLM_Engine.cleanup()`.
    """
    from types import SimpleNamespace
    from src.core import config
    from src.engines.base_engine import BaseEngine
    from src.agents.model_factory import build_chat_model

    config.LLM_Engine = BaseEngine.get_engine()
    llm = SimpleNamespace(id=llm_id, link=str(model_path))
    return build_chat_model(llm, temperature=0.0, top_p=1.0, max_tokens=max_tokens)


# =====================================================================
# INTEGRATION — thinking-model regression (ChatOpenAI path, opt-in ERUDI_TEST_THINKING=1)
# =====================================================================


@pytest.mark.mlx_only
class TestThinkingServerSideActivation:
    """Both halves of the MLX thinking fix, proven at the raw SSE boundary (#90).

    Half 1 — activation: `_spawn_child` passes `--enable-thinking`, so a request
    that does not set `enable_thinking` (Erudi's runner never does) still gets
    thinking-on-by-default from mlx-vlm 0.6.13 — without it, a thinking model
    answers directly and no reasoning ever exists.

    Half 2 — inline delivery: the in-child `_patch_inline_thinking` monkeypatch
    neutralizes the server-side split, so the reasoning arrives as literal
    `<think>...</think>` INSIDE `delta.content` (the channel ChatOpenAI keeps)
    and the `delta.reasoning` channel (which ChatOpenAI drops) stays silent.

    Asserting on the raw stream (not the runner) pins the server contract the
    runner's ThinkSplitter depends on. Opt-in via `mlx_thinking_model_path`
    (ERUDI_TEST_THINKING=1).
    """

    def test_thinking_activates_and_streams_inline(self, mlx_thinking_model_path):
        import requests

        try:
            model, _ = MLX_Engine.get_model_and_tokenizer(
                llm_id="qwen3-thinking",
                llm_local_path=str(mlx_thinking_model_path),
            )
            payload = {
                # mlx_vlm.server resolves `model` through get_cached_model(),
                # so it must carry the real preloaded model path.
                "model": model["model_path"],
                "messages": [{"role": "user", "content": "What is 17*23? Think step by step."}],
                "max_tokens": 1500,
                "temperature": 0.0,
                "stream": True,
            }
            contents: List[str] = []
            reasonings: List[str] = []
            with requests.post(
                f"{model['base_url']}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=300,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith(b"data: "):
                        continue
                    data = line[len(b"data: ") :]
                    if data.strip() == b"[DONE]":
                        break
                    delta = json.loads(data)["choices"][0]["delta"]
                    if delta.get("content"):
                        contents.append(delta["content"])
                    if delta.get("reasoning"):
                        reasonings.append(delta["reasoning"])

            text = "".join(contents)
            assert reasonings == [], (
                f"server-side thinking split is still active: {len(reasonings)} "
                f"non-null delta.reasoning chunks (ChatOpenAI would drop them all)"
            )
            assert "<think>" in text, (
                f"thinking did not activate server-side (no inline <think> in "
                f"delta.content): {text[:200]!r}"
            )
            assert "</think>" in text, f"thinking block never closed: {text[:200]!r}"
            inner = text.split("<think>", 1)[1].split("</think>", 1)[0]
            assert inner.strip(), "thinking block is empty — activation failed"
        finally:
            MLX_Engine.cleanup()


@pytest.mark.mlx_only
class TestThinkingModelRegression:
    """Reasoning must actually FLOW as thinking events and never leak into the
    ANSWER text (#90). Since the design keeps ``<think>...</think>`` INLINE in
    the engine stream on purpose (the in-child `_patch_inline_thinking`
    monkeypatch neutralizes mlx-vlm's server-side split), the runner's streaming
    splitter is what separates thinking from answer -- so this regression
    asserts on the RUNNER's event stream, not the raw ChatOpenAI content (which
    now legitimately carries the inline tags). The non-empty `thinking`
    assertion is what makes this test meaningful: without server-side
    activation (`--enable-thinking`) the model never thinks and a no-leak-only
    assertion would pass vacuously. Opt-in via `mlx_thinking_model_path`
    (ERUDI_TEST_THINKING=1).
    """

    async def test_thinking_flows_and_does_not_leak_into_answer(
        self,
        mlx_thinking_model_path,
    ):
        from types import SimpleNamespace

        from langgraph.checkpoint.memory import InMemorySaver

        from src.core import config
        from src.engines.base_engine import BaseEngine
        from src.agents.runner import AgentRunner, GenParams

        config.LLM_Engine = BaseEngine.get_engine()
        llm = SimpleNamespace(
            id="qwen3-thinking", link=str(mlx_thinking_model_path), name="qwen3-thinking"
        )
        runner = AgentRunner(checkpointer=InMemorySaver())
        try:
            answer, thinking = "", ""
            async for event in runner.astream_text(
                llm=llm,
                user_message="What is 17*23? Think step by step.",
                system_prompt="You are a helpful assistant.",
                params=GenParams(temperature=0.0, top_p=1.0, max_tokens=1500),
                thread_id="think-regression",
                summarize=False,
                emit_events=True,
            ):
                if event["t"] == "answer":
                    answer += event["text"]
                elif event["t"] == "thinking":
                    thinking += event["text"]
            assert thinking.strip(), (
                "no thinking events reached the runner — server-side thinking "
                "is not activating (or the inline <think> contract broke)"
            )
            for needle in ["<think>", "</think>", "<|channel>", "<channel|>"]:
                assert (
                    needle not in answer
                ), f"reasoning marker {needle!r} leaked into the ANSWER text: {answer!r}"
        finally:
            config.LLM_Engine.cleanup()


# =====================================================================
# INTEGRATION — Gemma EOS regression (ChatOpenAI path, opt-in ERUDI_TEST_GEMMA=1)
# =====================================================================


@pytest.mark.mlx_only
class TestGemmaEOSRegression:
    """Gemma must stop on `<end_of_turn>` on the live ChatOpenAI path rather than
    running to the token cap. Opt-in via ERUDI_TEST_GEMMA=1.
    """

    @pytest.fixture(scope="class")
    def gemma_path(self):
        import os

        if os.environ.get("ERUDI_TEST_GEMMA") != "1":
            pytest.skip("Set ERUDI_TEST_GEMMA=1 to enable Gemma EOS regression test")
        from huggingface_hub import snapshot_download

        repo = os.environ.get("ERUDI_MLX_GEMMA_REPO", "mlx-community/gemma-3-270m-it-4bit")
        try:
            return Path(snapshot_download(repo_id=repo))
        except Exception as exc:
            pytest.skip(f"Cannot fetch Gemma model {repo!r}: {exc}")

    async def test_gemma_stops_within_reasonable_bound(self, gemma_path):
        from langchain_core.messages import HumanMessage
        from src.core import config

        model = _build_real_mlx_chat_model("gemma-test", gemma_path, max_tokens=200)
        try:
            visible_chunks = 0
            async for chunk in model.astream([HumanMessage("Say hello.")]):
                if isinstance(chunk.content, str) and chunk.content:
                    visible_chunks += 1
            assert visible_chunks < 150, (
                "Gemma did not stop on <end_of_turn> on the ChatOpenAI path "
                "(ran to the token cap)"
            )
        finally:
            config.LLM_Engine.cleanup()


# =====================================================================
# E2E — full FastAPI stack with real MLX engine
# =====================================================================


@pytest.mark.mlx_only
@pytest.mark.e2e
class TestE2EConversationsRealMLX:
    """Drive `POST /erudi/conversations/{id}/query` through the new engine.

    These tests confirm the contract is preserved at the HTTP boundary —
    services, repository, streaming response, message persistence all
    keep working when the engine is server-mode subprocess MLX.

    Notes for maintainers:
      - The `_force_mlx_engine_in_config` autouse fixture pins
        `src.core.config.LLM_Engine = MLX_Engine` for the duration of each
        e2e test. Without it, an unrelated earlier test (e.g.
        `test_engines.py::test_get_engine_returns_valid_class`) could have
        set `config.LLM_Engine = CPU_Engine`, in which case these e2e
        tests would silently exercise CPU_Engine and produce baffling
        failures.
      - The endpoint currently routes through `endpoints._stream_on_single_thread`
        (a 1-thread ThreadPoolExecutor). Phase 3 removes that wrapper;
        these tests must keep passing through both Phase 2 (wrapper still
        there) and Phase 3 (wrapper gone).
    """

    @pytest.fixture(autouse=True)
    def _force_mlx_engine_in_config(self):
        from src.core import config

        prev = getattr(config, "LLM_Engine", None)
        config.LLM_Engine = MLX_Engine
        yield
        config.LLM_Engine = prev

    def _make_llm_row(self, db_session, model_path: Path):
        """Insert an Llm pointing to the real MLX model path."""
        from src.entities.Llm import Llm

        llm = Llm(
            name="Qwen2.5-0.5B-Instruct-4bit",
            description="Integration test model",
            local=1,
            link=str(model_path),
            type="qwen",
            is_attached_to_kb=False,
            param_size=0.5,
            quantized=True,
        )
        db_session.add(llm)
        db_session.commit()
        db_session.refresh(llm)
        return llm

    def test_query_endpoint_streams_real_response(
        self,
        client,
        test_db_session,
        mlx_test_model_path,
    ):
        try:
            llm = self._make_llm_row(test_db_session, mlx_test_model_path)
            create_resp = client.post(
                "/erudi/conversations/",
                json={
                    "llm_id": llm.id,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 32,
                    "custom_prompt": "",
                },
            )
            assert create_resp.status_code == 201, create_resp.text
            conv_id = create_resp.json()["id"]

            resp = client.post(
                f"/erudi/conversations/{conv_id}/query",
                json={"question": "Say hi.", "max_new_tokens": 16},
            )
            assert resp.status_code == 200, resp.text
            assert len(resp.text.strip()) > 0, "empty response body"

            # User message + assistant message must both have been persisted.
            from src.entities.Message import Message

            msgs = test_db_session.query(Message).filter(Message.conversation_id == conv_id).all()
            senders = [m.sender for m in msgs]
            assert "user" in senders
            assert "llm" in senders
        finally:
            MLX_Engine.cleanup()

    def test_generate_title_endpoint_writes_name(
        self,
        client,
        test_db_session,
        mlx_test_model_path,
    ):
        try:
            llm = self._make_llm_row(test_db_session, mlx_test_model_path)
            create_resp = client.post(
                "/erudi/conversations/",
                json={
                    "llm_id": llm.id,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 32,
                    "custom_prompt": "",
                },
            )
            conv_id = create_resp.json()["id"]

            resp = client.post(
                f"/erudi/conversations/{conv_id}/generate_title",
                json={"question": "Explain Python decorators briefly"},
            )
            assert resp.status_code == 200, resp.text

            from src.entities.Conversation import Conversation

            test_db_session.expire_all()
            conv = test_db_session.query(Conversation).filter(Conversation.id == conv_id).first()
            # Either the model produced a title, or the empty-question fallback
            # kicked in; either way, name must not be the literal "New Conversation"
            # if the model emitted anything, AND it must not be empty.
            assert conv.name and len(conv.name.strip()) > 0
        finally:
            MLX_Engine.cleanup()

    def test_two_consecutive_queries_reuse_same_subprocess(
        self,
        client,
        test_db_session,
        mlx_test_model_path,
    ):
        """Cache contract: same llm.id ⇒ same subprocess across requests."""
        try:
            llm = self._make_llm_row(test_db_session, mlx_test_model_path)
            create_resp = client.post(
                "/erudi/conversations/",
                json={
                    "llm_id": llm.id,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 16,
                    "custom_prompt": "",
                },
            )
            conv_id = create_resp.json()["id"]

            for q in ("first", "second"):
                r = client.post(
                    f"/erudi/conversations/{conv_id}/query",
                    json={"question": q, "max_new_tokens": 4},
                )
                assert r.status_code == 200

            # After both queries, the singleton must still hold the same pid.
            assert MLX_Engine._model is not None
            # `_model_id` may be stored as int or str depending on the impl's
            # normalization. Accept either to avoid coupling the test to that
            # internal choice.
            assert MLX_Engine._model_id in (llm.id, str(llm.id)), (
                f"expected _model_id to be {llm.id!r} or {str(llm.id)!r}, "
                f"got {MLX_Engine._model_id!r}"
            )
            assert MLX_Engine._model["proc"].is_alive()
        finally:
            MLX_Engine.cleanup()
