"""What the engines and the agent layer log when something goes wrong.

The Diagnostics panel shows WARNING and above, so every record here is
checked for its level, its identifying details (exit code, pid, port, model,
llm id) and, for a death nobody awaits, for the fact that it is logged at all.
See docs/logging.md for the rules these tests pin down.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import EngineException
from src.engines import base_llama_cpp_engine as llama_mod
from src.engines.base_chat_server_engine import BaseChatServerEngine
from src.engines.base_engine import BaseEngine
from src.engines.child_output import ChildOutputDrainer
from src.engines.cpu_engine import CPU_Engine

pytestmark = pytest.mark.unit


class _Engine(BaseChatServerEngine):
    """Minimal concrete engine: only what the probe and the cache check touch."""

    _server_name = "llama-server"
    FORMAT_TAG = "gguf"
    _probe_timeout_s = 0.05
    _probe_poll_interval_s = 0.01

    @classmethod
    def _spawn_child(cls, **kwargs):  # pragma: no cover - never spawned here
        raise NotImplementedError

    @classmethod
    def _terminate_process(cls, proc):
        return None

    @classmethod
    def _proc_is_alive(cls, proc):
        return False

    @classmethod
    def _resolve_model_artifact(cls, llm_local_path):  # pragma: no cover
        raise NotImplementedError


class _DeadProc:
    """A Popen that exited: what the probe sees after an early crash."""

    pid = 4242
    returncode = 139


# ============ The child death record names the process ============


class TestChildDeathRecord:
    def test_early_crash_names_exit_code_pid_and_port(self):
        with patch.object(_Engine, "_read_child_output", classmethod(lambda cls, p: "CUDA boom")):
            with pytest.raises(EngineException) as excinfo:
                _Engine._probe_ready("http://127.0.0.1:19000", proc=_DeadProc(), model_field="m")
        message = excinfo.value.message
        assert "exit code 139" in message
        assert "pid 4242" in message
        assert "port 19000" in message
        assert "CUDA boom" in message

    def test_probe_timeout_carries_the_child_tail(self):
        class _Alive(_Engine):
            @classmethod
            def _proc_is_alive(cls, proc):
                return True

        health_503 = MagicMock(status_code=503)
        with (
            patch("src.engines.base_chat_server_engine.requests.get", return_value=health_503),
            patch.object(_Alive, "_read_child_output", classmethod(lambda cls, p: "loading...")),
        ):
            with pytest.raises(EngineException) as excinfo:
                _Alive._probe_ready("http://127.0.0.1:19000", proc=SimpleNamespace(pid=7))
        assert "did not become ready" in excinfo.value.message
        assert "loading..." in excinfo.value.trace

    def test_cached_dead_child_warning_names_the_exit_code(self, caplog):
        _Engine._model = {"proc": _DeadProc()}
        _Engine._tokenizer = object()
        _Engine._model_id = 12
        try:
            with patch.object(_Engine, "_read_child_output", classmethod(lambda cls, p: "tail")):
                with caplog.at_level(logging.WARNING, logger="erudi"):
                    assert _Engine._should_not_reload_model(12) is False
        finally:
            _Engine._model = None
            _Engine._tokenizer = None
            _Engine._model_id = None
        (record,) = [r for r in caplog.records if "no longer running" in r.getMessage()]
        assert record.levelno == logging.WARNING
        assert "exit code 139" in record.getMessage()
        assert "model 12" in record.getMessage()

    def test_crash_report_describes_a_dead_child_and_nothing_for_a_live_one(self):
        _Engine._model = {"proc": _DeadProc(), "port": 19000}
        try:
            with patch.object(_Engine, "_read_child_output", classmethod(lambda cls, p: "tail")):
                report = _Engine.child_crash_report()
            assert report is not None
            assert "exit code 139" in report
            assert "tail" in report
        finally:
            _Engine._model = None

        class _Live(_Engine):
            @classmethod
            def _proc_is_alive(cls, proc):
                return True

        _Live._model = {"proc": _DeadProc()}
        try:
            assert _Live.child_crash_report() is None
        finally:
            _Live._model = None
        assert _Engine.child_crash_report() is None  # no model loaded


# ============ The output drainer does not die quietly ============


class TestDrainerStop:
    def test_a_read_error_is_logged_at_warning_with_its_type(self, caplog):
        class _BrokenStream:
            def readline(self):
                raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

            def close(self):
                pass

        with caplog.at_level(logging.DEBUG, logger="erudi"):
            drainer = ChildOutputDrainer(_BrokenStream(), name="llama-server:1")
            drainer.join(timeout=5)
        (record,) = [r for r in caplog.records if "drainer stopped" in r.getMessage()]
        assert record.levelno == logging.WARNING
        assert "UnicodeDecodeError" in record.getMessage()

    def test_eof_is_not_a_warning(self, caplog):
        proc = subprocess.Popen(
            [sys.executable, "-c", "print('hello')"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        try:
            with caplog.at_level(logging.DEBUG, logger="erudi"):
                drainer = ChildOutputDrainer(proc.stdout, name="test")
                proc.wait(timeout=30)
                drainer.join(timeout=10)
        finally:
            if proc.poll() is None:
                proc.kill()
        assert all(r.levelno < logging.WARNING for r in caplog.records)


# ============ The idle-cleanup monitor survives a failing tick ============


class _MonitorEngine(BaseEngine):
    pass


async def test_cleanup_monitor_logs_a_failing_tick_and_keeps_running(caplog, monkeypatch):
    ticks = []
    resumed = asyncio.Event()

    async def _tick():
        ticks.append(1)
        if len(ticks) == 1:
            raise RuntimeError("cleanup exploded")
        resumed.set()

    monkeypatch.setattr(_MonitorEngine, "_cleanup_tick", classmethod(lambda cls: _tick()))
    monkeypatch.setattr(_MonitorEngine, "_cleanup_interval_s", 0.01)
    _MonitorEngine._cleanup_task = None
    with caplog.at_level(logging.INFO, logger="erudi"):
        _MonitorEngine.start_cleanup_task()
        try:
            await asyncio.wait_for(resumed.wait(), timeout=5)
        finally:
            _MonitorEngine.stop_cleanup_task()
            await asyncio.sleep(0)
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert "cleanup exploded" in errors[0].getMessage()
    assert errors[0].exc_info
    assert len(ticks) >= 2  # the loop did not die with the tick


# ============ Engine selection says why it fell back ============


class TestEngineSelection:
    def _select_on_linux(self, monkeypatch, fake_pynvml):
        monkeypatch.delenv("ERUDI_FORCE_CPU", raising=False)
        monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)
        with patch("src.engines.base_engine.platform.system", return_value="Linux"):
            with patch("src.engines.base_engine.platform.machine", return_value="x86_64"):
                return BaseEngine.get_engine()

    def test_a_broken_nvml_is_a_warning_with_the_cause(self, monkeypatch, caplog):
        def _nvml_init():
            raise RuntimeError("NVML library mismatch")

        fake = SimpleNamespace(nvmlInit=_nvml_init, nvmlDeviceGetCount=lambda: 0)
        with caplog.at_level(logging.INFO, logger="erudi"):
            engine = self._select_on_linux(monkeypatch, fake)
        assert engine is CPU_Engine
        (record,) = [r for r in caplog.records if "NVIDIA detection failed" in r.getMessage()]
        assert record.levelno == logging.WARNING
        assert "NVML library mismatch" in record.getMessage()
        assert record.exc_info

    def test_a_machine_without_a_driver_is_not_a_warning(self, monkeypatch, caplog):
        class NVMLError_LibraryNotFound(Exception):
            pass

        def _nvml_init():
            raise NVMLError_LibraryNotFound("NVML Shared Library Not Found")

        fake = SimpleNamespace(nvmlInit=_nvml_init, nvmlDeviceGetCount=lambda: 0)
        with caplog.at_level(logging.INFO, logger="erudi"):
            engine = self._select_on_linux(monkeypatch, fake)
        assert engine is CPU_Engine
        assert all(r.levelno < logging.WARNING for r in caplog.records)
        assert any("No NVIDIA driver" in r.getMessage() for r in caplog.records)


# ============ llama-server: flavour fallback and a spawn that fails ============


class TestLlamaServerSpawn:
    def test_falling_back_to_the_other_flavour_is_logged(self, tmp_path, monkeypatch, caplog):
        exe = "llama-server.exe" if os.name == "nt" else "llama-server"
        monkeypatch.setattr(llama_mod, "ROOT_DIR", tmp_path)
        cuda_binary = tmp_path / "artifacts" / "llama-cpp" / "cuda" / "bin" / exe
        cuda_binary.parent.mkdir(parents=True)
        cuda_binary.write_bytes(b"\x00")
        empty_cpu = tmp_path / "artifacts" / "llama-cpp" / "cpu" / "bin"
        empty_cpu.mkdir(parents=True)
        with caplog.at_level(logging.WARNING, logger="erudi"):
            assert CPU_Engine._find_llama_server(empty_cpu) == cuda_binary
        (record,) = caplog.records
        assert record.levelno == logging.WARNING
        assert str(cuda_binary) in record.getMessage()

    def test_a_popen_failure_becomes_an_engine_exception_naming_the_binary(
        self, tmp_path, monkeypatch
    ):
        exe = "llama-server.exe" if os.name == "nt" else "llama-server"
        binary = tmp_path / "bin" / exe
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"\x00")
        monkeypatch.setattr(
            CPU_Engine, "_default_install_dir", classmethod(lambda cls: binary.parent)
        )
        monkeypatch.setattr(CPU_Engine, "_find_mmproj", classmethod(lambda cls, p: None))

        def _refuse(*args, **kwargs):
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(llama_mod.subprocess, "Popen", _refuse)
        with pytest.raises(EngineException) as excinfo:
            CPU_Engine._spawn_child(model_path=tmp_path / "m.gguf", alias="m", port=19000)
        assert str(binary) in excinfo.value.message
        assert "Permission denied" in excinfo.value.message
        assert "PermissionError" in excinfo.value.trace
