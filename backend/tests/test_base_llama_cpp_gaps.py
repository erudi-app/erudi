"""Gap coverage for `BaseLlamaCppEngine` shared llama-server plumbing.

Complements `test_base_chat_server_engine.py` and the CPU/CUDA engine files:
binary resolution with the cross-flavour fallback, GGUF selection edge
cases (single file, unknown quants, mmproj exclusion), vision detection via
mmproj presence (#130), the Popen terminate/alive helpers, the spawn-child
handle assembly with the vision projector argv, and the default spawn env.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import EngineException
from src.engines import base_llama_cpp_engine as base_mod
from src.engines.cpu_engine import CPU_Engine
from src.engines.cuda_engine import CUDA_Engine


def _touch(path, size=16):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * size)
    return path


# =====================================================================
# UNIT - llama-server binary resolution
# =====================================================================


@pytest.mark.unit
class TestFindLlamaServer:
    def test_primary_flavour_wins(self, tmp_path):
        exe = "llama-server.exe" if os.name == "nt" else "llama-server"
        install = tmp_path / "cpu" / "bin"
        binary = _touch(install / exe)
        assert CPU_Engine._find_llama_server(install) == binary

    def test_falls_back_to_other_flavour(self, tmp_path, monkeypatch):
        exe = "llama-server.exe" if os.name == "nt" else "llama-server"
        monkeypatch.setattr(base_mod, "ROOT_DIR", tmp_path)
        cuda_binary = _touch(tmp_path / "artifacts" / "llama-cpp" / "cuda" / "bin" / exe)
        empty_cpu = tmp_path / "artifacts" / "llama-cpp" / "cpu" / "bin"
        empty_cpu.mkdir(parents=True)
        assert CPU_Engine._find_llama_server(empty_cpu) == cuda_binary

    def test_missing_everywhere_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(base_mod, "ROOT_DIR", tmp_path)
        install = tmp_path / "nothing" / "bin"
        install.mkdir(parents=True)
        with pytest.raises(EngineException, match="llama-server binary not found"):
            CPU_Engine._find_llama_server(install)


# =====================================================================
# UNIT - GGUF selection edge cases
# =====================================================================


@pytest.mark.unit
class TestSelectGgufEdgeCases:
    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(EngineException, match="not found"):
            CPU_Engine._select_gguf(tmp_path / "missing")

    def test_single_gguf_in_directory(self, tmp_path):
        only = _touch(tmp_path / "model-exotic.gguf")
        assert CPU_Engine._select_gguf(tmp_path) == only

    def test_mmproj_files_never_selected(self, tmp_path):
        model = _touch(tmp_path / "model-q4_k_m.gguf")
        _touch(tmp_path / "mmproj-model-f16.gguf")
        assert CPU_Engine._select_gguf(tmp_path) == model

    def test_unknown_quants_pick_smallest(self, tmp_path):
        _touch(tmp_path / "model-alpha.gguf", size=4096)
        small = _touch(tmp_path / "model-beta.gguf", size=64)
        assert CPU_Engine._select_gguf(tmp_path) == small


# =====================================================================
# UNIT - vision detection via mmproj (#130)
# =====================================================================


@pytest.mark.unit
class TestVisionDetection:
    def test_find_mmproj_none(self, tmp_path):
        model = _touch(tmp_path / "model-q4_k_m.gguf")
        assert CPU_Engine._find_mmproj(model) is None

    def test_find_mmproj_multiple_picks_first(self, tmp_path):
        model = _touch(tmp_path / "model-q4_k_m.gguf")
        _touch(tmp_path / "mmproj-a.gguf")
        _touch(tmp_path / "mmproj-b.gguf")
        found = CPU_Engine._find_mmproj(model)
        assert found is not None
        assert found.name.startswith("mmproj-")

    def test_model_supports_vision_true(self, tmp_path):
        _touch(tmp_path / "model-q4_k_m.gguf")
        _touch(tmp_path / "mmproj-model.gguf")
        assert CPU_Engine.model_supports_vision(tmp_path) is True

    def test_model_supports_vision_false(self, tmp_path):
        _touch(tmp_path / "model-q4_k_m.gguf")
        assert CPU_Engine.model_supports_vision(tmp_path) is False

    def test_model_supports_vision_unreadable_is_none(self, tmp_path):
        assert CPU_Engine.model_supports_vision(tmp_path / "missing") is None


# =====================================================================
# UNIT - Popen lifecycle helpers
# =====================================================================


@pytest.mark.unit
class TestProcessHelpers:
    def test_terminate_none_is_noop(self):
        CPU_Engine._terminate_process(None)

    def test_terminate_already_exited_is_noop(self):
        proc = MagicMock()
        proc.poll.return_value = 0
        CPU_Engine._terminate_process(proc)
        proc.send_signal.assert_not_called()
        proc.terminate.assert_not_called()

    def test_terminate_running_posix_sends_sigint(self):
        import signal as signal_mod

        proc = MagicMock()
        proc.poll.return_value = None
        with patch.object(base_mod.platform, "system", return_value="Linux"):
            CPU_Engine._terminate_process(proc)
        proc.send_signal.assert_called_once_with(signal_mod.SIGINT)
        proc.wait.assert_called_once()
        proc.kill.assert_not_called()

    def test_terminate_running_windows_uses_terminate(self):
        proc = MagicMock()
        proc.poll.return_value = None
        with patch.object(base_mod.platform, "system", return_value="Windows"):
            CPU_Engine._terminate_process(proc)
        proc.terminate.assert_called_once()

    def test_terminate_escalates_to_kill_on_stuck_wait(self):
        proc = MagicMock()
        proc.poll.return_value = None
        proc.wait.side_effect = TimeoutError("still alive")
        with patch.object(base_mod.platform, "system", return_value="Linux"):
            CPU_Engine._terminate_process(proc)
        proc.kill.assert_called_once()

    def test_terminate_swallows_poll_failure(self):
        proc = MagicMock()
        proc.poll.side_effect = OSError("gone")
        CPU_Engine._terminate_process(proc)  # best effort, no raise

    def test_proc_is_alive_matrix(self):
        assert CPU_Engine._proc_is_alive(None) is False
        running = MagicMock()
        running.poll.return_value = None
        assert CPU_Engine._proc_is_alive(running) is True
        exited = MagicMock()
        exited.poll.return_value = 1
        assert CPU_Engine._proc_is_alive(exited) is False
        broken = MagicMock()
        broken.poll.side_effect = OSError("gone")
        assert CPU_Engine._proc_is_alive(broken) is False


# =====================================================================
# UNIT - spawn-child handle assembly
# =====================================================================


def _spawn_cpu_child(tmp_path, monkeypatch, *, with_mmproj, engine=CPU_Engine, **ctx):
    """Run `_spawn_child` of a llama.cpp engine against a fake Popen, capturing the argv.

    `ctx` overrides the engine's own `_prepare_spawn_context()` (the CUDA
    context needs a GPU probe, so its tests pass the values explicitly).
    """
    model = _touch(tmp_path / "model-q4_k_m.gguf")
    if with_mmproj:
        _touch(tmp_path / "mmproj-model.gguf")
    server = _touch(tmp_path / "bin" / "llama-server")
    monkeypatch.setattr(engine, "_find_llama_server", classmethod(lambda cls, d=None: server))
    captured = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        # `stdout` is part of the Popen contract `_spawn_child` relies on:
        # it creates the child with stdout=PIPE and hands the stream to the
        # output drainer (#361). None here means "this test captures no
        # output", which the drainer accepts as a no-op.
        return SimpleNamespace(pid=4242, stdout=None)

    monkeypatch.setattr(base_mod.subprocess, "Popen", fake_popen)
    handle = engine._spawn_child(
        model_path=model,
        alias="erudi-1",
        port=27201,
        **(ctx or engine._prepare_spawn_context()),
    )
    return handle, captured


@pytest.mark.unit
class TestSpawnChild:
    def _spawn(self, tmp_path, monkeypatch, *, with_mmproj):
        return _spawn_cpu_child(tmp_path, monkeypatch, with_mmproj=with_mmproj)

    def test_handle_carries_context_and_base_url(self, tmp_path, monkeypatch):
        handle, captured = self._spawn(tmp_path, monkeypatch, with_mmproj=False)
        assert handle["pid"] == 4242
        assert handle["port"] == 27201
        assert handle["base_url"] == "http://127.0.0.1:27201"
        assert handle["alias"] == "erudi-1"
        assert handle["gpu_layers"] == 0  # CPU context preserved for observability
        assert handle["threads"] >= 1
        assert "--mmproj" not in captured["argv"]

    def test_mmproj_is_appended_to_argv(self, tmp_path, monkeypatch):
        _handle, captured = self._spawn(tmp_path, monkeypatch, with_mmproj=True)
        argv = captured["argv"]
        assert "--mmproj" in argv
        assert argv[argv.index("--mmproj") + 1].endswith("mmproj-model.gguf")

    def test_default_spawn_env_inherits_parent(self):
        env = CPU_Engine._build_spawn_env()
        assert env == os.environ.copy()


# =====================================================================
# UNIT - local-only hardening of the spawned llama-server
# =====================================================================


@pytest.mark.unit
class TestSpawnHardeningFlags:
    """llama-server must not be left open to everything on the loopback.

    Spawned without `--api-key`, llama-server authenticates NOTHING: every
    endpoint answers any caller that can reach 127.0.0.1. That includes
    `/slots`, enabled by default, which dumps the prompt of every in-flight
    request. So any other local process -- and any web page the user has open,
    since a browser can POST across origins to a loopback port -- can read what
    the user is currently asking the model and submit prompts of its own.
    These tests pin the three flags that close that surface.
    """

    def test_slots_and_webui_endpoints_are_disabled(self, tmp_path, monkeypatch):
        """`/slots` leaks in-flight prompts and the web UI is a second way in.

        Neither is used by Erudi (the backend only ever calls
        `/health` and `/v1/chat/completions`), so both are switched off rather
        than merely guarded by the key.
        """
        _handle, captured = _spawn_cpu_child(tmp_path, monkeypatch, with_mmproj=False)
        argv = captured["argv"]
        assert "--no-slots" in argv
        assert "--no-webui" in argv

    def test_api_key_flag_carries_a_non_empty_secret(self, tmp_path, monkeypatch):
        """`--api-key` is what turns on authentication at all in llama-server.

        An empty value would be accepted by the CLI and leave the server
        unauthenticated, so the value itself is asserted, not just the flag.
        """
        _handle, captured = _spawn_cpu_child(tmp_path, monkeypatch, with_mmproj=False)
        argv = captured["argv"]
        assert "--api-key" in argv
        key = argv[argv.index("--api-key") + 1]
        assert isinstance(key, str) and len(key) >= 32

    def test_each_spawn_gets_a_different_key(self, tmp_path, monkeypatch):
        """A key reused across spawns would stay valid after it leaked once.

        Per-spawn generation is what bounds the damage of a disclosure to the
        life of a single child process: swapping models (or a crash-respawn)
        invalidates whatever an attacker scraped.
        """
        _h1, first = _spawn_cpu_child(tmp_path, monkeypatch, with_mmproj=False)
        _h2, second = _spawn_cpu_child(tmp_path, monkeypatch, with_mmproj=False)
        first_argv, second_argv = first["argv"], second["argv"]
        first_key = first_argv[first_argv.index("--api-key") + 1]
        second_key = second_argv[second_argv.index("--api-key") + 1]
        assert first_key != second_key

    def test_handle_exposes_the_key_to_the_callers_that_need_it(self, tmp_path, monkeypatch):
        """The readiness probe and the inference client both authenticate now.

        Both reach the child only through the spawn handle, so a key kept
        local to `_spawn_child` would lock Erudi out of its own server.
        """
        handle, captured = _spawn_cpu_child(tmp_path, monkeypatch, with_mmproj=False)
        argv = captured["argv"]
        assert handle["api_key"] == argv[argv.index("--api-key") + 1]

    def test_the_key_never_reaches_the_logs(self, tmp_path, monkeypatch, caplog):
        """Backend logs are written to a world-readable temp file and shipped
        in bug reports; a key printed there outlives the process that used it
        and re-opens the very hole `--api-key` closes.
        """
        with caplog.at_level(logging.DEBUG):
            handle, _captured = _spawn_cpu_child(tmp_path, monkeypatch, with_mmproj=True)
        key = handle["api_key"]
        assert key
        for record in caplog.records:
            assert key not in record.getMessage()


@pytest.mark.unit
class TestSpawnSingleSlotNoPromptCache:
    """llama-server holds one conversation's KV at a time, with no prompt cache.

    Upstream defaults are four slots and an 8 GiB host-RAM cache of past
    prompts' full KV state (`--cache-ram 8192`), counted nowhere: each distinct
    long conversation added about 1.3 GiB to the child and pushed the machine
    into swap. `-np 1 --cache-ram 0` keeps the current conversation's prefix in
    its slot (the follow-up still reuses it) and nothing else.
    """

    @staticmethod
    def _value(argv, flag):
        assert flag in argv
        return argv[argv.index(flag) + 1]

    @pytest.mark.parametrize("erudi_ctx", [None, "8192"])
    def test_cpu_argv_pins_one_slot_and_no_prompt_cache(
        self, tmp_path, monkeypatch, erudi_ctx
    ):
        if erudi_ctx is None:
            monkeypatch.delenv("ERUDI_CTX", raising=False)
        else:
            monkeypatch.setenv("ERUDI_CTX", erudi_ctx)
        _handle, captured = _spawn_cpu_child(tmp_path, monkeypatch, with_mmproj=False)
        argv = captured["argv"]
        assert self._value(argv, "-np") == "1"
        assert self._value(argv, "--cache-ram") == "0"

    @pytest.mark.parametrize("ctx_size", [None, 8192])
    def test_cuda_argv_pins_one_slot_and_no_prompt_cache(self, tmp_path, monkeypatch, ctx_size):
        _handle, captured = _spawn_cpu_child(
            tmp_path,
            monkeypatch,
            with_mmproj=False,
            engine=CUDA_Engine,
            ctx_size=ctx_size,
            threads=4,
            gpu_layers=32,
        )
        argv = captured["argv"]
        assert self._value(argv, "-np") == "1"
        assert self._value(argv, "--cache-ram") == "0"
