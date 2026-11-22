"""The Windows/Linux adapter of the KB embedding child (#594, F3):
``llama-server --embedding`` around the bundled e5 GGUF.

Everything here is argv/env/path level — no real server is spawned
(``subprocess.Popen`` is recorded). The manager-with-real-process path runs
in ``tests/test_embedder_child_http.py`` against the HTTP double, and the
real binary in ``tests/test_llama_embedder_real.py`` (skipped where the
binary or the GGUF is absent).

Contract under test:
- the child runs the CPU ``llama-server`` (cuda/bin only as the dev fallback,
  with a WARNING), with ``CUDA_VISIBLE_DEVICES=-1`` and ``-ngl 0`` so it can
  never touch a graphics card;
- the spike-S4 serving flags: ``--embedding -c/-b/-ub 512 --pooling mean
  -np 1 --poll 0``, authentication by per-spawn key, no slots endpoint, no
  web UI;
- the API key travels by environment (``LLAMA_API_KEY``), never on the
  command line (invisible to ``ps``);
- the GGUF ships with the install: absent, the F2 missing-model STATE
  applies (``EmbeddingModelMissingException``), and nothing downloads it;
- the CUDA gauges (spill, eviction, ``gpu_contention``) sample the CHAT
  child's pid only — the embedder is never read by them.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core import config
from src.core.exceptions import EmbeddingModelMissingException, EngineException
from src.ingestion import embedder_child, embeddings, llama_embedder

pytestmark = pytest.mark.unit


@pytest.fixture
def install_root(monkeypatch, tmp_path):
    """A fake install tree ``config.ROOT_DIR`` points at."""
    monkeypatch.setattr(config, "ROOT_DIR", tmp_path)
    return tmp_path


_EXE = "llama-server.exe" if os.name == "nt" else "llama-server"


def _put_server(root: Path, flavour: str, exe: str = _EXE) -> Path:
    bindir = root / "artifacts" / "llama-cpp" / flavour / "bin"
    bindir.mkdir(parents=True)
    server = bindir / exe
    server.write_bytes(b"\x00")
    return server


def _put_gguf(root: Path) -> Path:
    path = root / "artifacts" / "kb-embedder" / llama_embedder.E5_GGUF_NAME
    path.parent.mkdir(parents=True)
    path.write_bytes(b"GGUF")
    return path


# ====================== Binary resolution ======================


class TestBinaryResolution:
    def test_the_cpu_flavour_is_the_binary(self, install_root):
        server = _put_server(install_root, "cpu")
        _put_server(install_root, "cuda")
        assert llama_embedder._find_llama_server() == server

    def test_falls_back_to_the_cuda_flavour_with_a_warning(self, install_root, caplog):
        server = _put_server(install_root, "cuda")
        with caplog.at_level(logging.WARNING):
            assert llama_embedder._find_llama_server() == server
        assert any("falling back" in r.getMessage() for r in caplog.records)

    def test_no_binary_raises_naming_the_build_scripts(self, install_root):
        with pytest.raises(EngineException) as excinfo:
            llama_embedder._find_llama_server()
        assert "build-llamacpp" in str(excinfo.value.message)

    def test_windows_looks_for_the_exe_name(self, install_root, monkeypatch):
        monkeypatch.setattr(llama_embedder.os, "name", "nt")
        server = _put_server(install_root, "cpu", exe="llama-server.exe")
        assert llama_embedder._find_llama_server() == server


# ====================== The artifact (bundled GGUF) ======================


class TestArtifact:
    def test_available_iff_the_gguf_is_on_disk(self, install_root):
        assert not llama_embedder.artifact_available()
        _put_gguf(install_root)
        assert llama_embedder.artifact_available()

    def test_resolve_model_missing_is_the_f2_state(self, install_root):
        with pytest.raises(EmbeddingModelMissingException):
            llama_embedder.resolve_model()

    def test_resolve_model_returns_the_bundled_gguf(self, install_root):
        gguf = _put_gguf(install_root)
        assert llama_embedder.resolve_model() == gguf

    def test_the_artifact_is_not_downloadable(self):
        # The GGUF ships INSIDE the installers; the #146 download gate must
        # never pretend a download would produce it.
        assert llama_embedder.artifact_downloadable is False

    def test_the_mlx_artifact_stays_downloadable(self):
        if sys.platform != "darwin":
            pytest.skip("mlx_embedder imports mlx-only modules")
        from src.ingestion import mlx_embedder

        assert mlx_embedder.artifact_downloadable is True


# ====================== Spawn: argv and environment ======================


class FakePopen:
    def __init__(self, argv, **kwargs):
        self.argv = argv
        self.kwargs = kwargs
        self.pid = 4321
        self.returncode = None

    def poll(self):
        return self.returncode


@pytest.fixture
def spawned(monkeypatch, install_root, tmp_path):
    """Record the Popen of one adapter spawn; return (proc, argv, kwargs)."""
    _put_server(install_root, "cpu")
    gguf = _put_gguf(install_root)
    captured = {}

    def fake_popen(argv, **kwargs):
        proc = FakePopen(argv, **kwargs)
        captured["proc"] = proc
        return proc

    monkeypatch.setattr(llama_embedder.subprocess, "Popen", fake_popen)
    log_path = str(tmp_path / "embedder-child-27400.log")
    proc = llama_embedder.spawn(port=27400, api_key="sekret", model_path=gguf, log_path=log_path)
    assert proc is captured["proc"]
    return proc


class TestSpawn:
    def test_the_spike_serving_flags(self, spawned):
        argv = spawned.argv
        joined = " ".join(argv)
        assert argv[0].endswith(os.path.join("cpu", "bin", _EXE))
        assert "--embedding" in argv
        for flag in ("-c", "-b", "-ub"):
            assert argv[argv.index(flag) + 1] == "512"
        # -np 1 is mandatory: the default 4 slots reserve ~560 MB of logits
        # instead of ~315 MB (spike S4).
        assert argv[argv.index("-np") + 1] == "1"
        # --poll 0: ggml's spin-wait threads collapse under CPU contention,
        # and this child shares the machine with the chat model.
        assert argv[argv.index("--poll") + 1] == "0"
        assert argv[argv.index("--pooling") + 1] == "mean"
        assert argv[argv.index("-ngl") + 1] == "0"
        assert argv[argv.index("--host") + 1] == "127.0.0.1"
        assert argv[argv.index("--port") + 1] == "27400"
        assert "--no-slots" in argv
        assert "--no-webui" in argv
        assert "sekret" not in joined

    def test_the_key_travels_by_environment_never_argv(self, spawned):
        env = spawned.kwargs["env"]
        assert env["LLAMA_API_KEY"] == "sekret"
        assert "--api-key" not in spawned.argv

    def test_the_child_cannot_see_a_gpu(self, spawned):
        # The exclusion from the CUDA gauges starts here: a child with every
        # device masked never appears as a CUDA compute process, and -ngl 0
        # keeps even a fallback CUDA binary off the card.
        assert spawned.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "-1"

    def test_the_backend_environment_is_left_untouched(self, spawned):
        assert os.environ.get("LLAMA_API_KEY") != "sekret"
        assert os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"

    def test_output_goes_to_the_log_file(self, spawned, tmp_path):
        stdout = spawned.kwargs["stdout"]
        assert stdout.name == str(tmp_path / "embedder-child-27400.log")
        # The parent's copy of the handle is closed right after the spawn.
        assert stdout.closed
        assert spawned.kwargs["stderr"] == subprocess.STDOUT

    def test_no_log_path_degrades_to_devnull(self, monkeypatch, install_root, tmp_path):
        _put_server(install_root, "cpu")
        gguf = _put_gguf(install_root)
        captured = {}
        monkeypatch.setattr(
            llama_embedder.subprocess,
            "Popen",
            lambda argv, **kwargs: captured.update(kwargs) or FakePopen(argv, **kwargs),
        )
        llama_embedder.spawn(port=27400, api_key="k", model_path=gguf, log_path=None)
        assert captured["stdout"] == subprocess.DEVNULL

    def test_a_popen_failure_is_an_engine_exception(self, monkeypatch, install_root, tmp_path):
        _put_server(install_root, "cpu")
        gguf = _put_gguf(install_root)

        def boom(argv, **kwargs):
            raise OSError(8, "Exec format error")

        monkeypatch.setattr(llama_embedder.subprocess, "Popen", boom)
        with pytest.raises(EngineException):
            llama_embedder.spawn(port=27400, api_key="k", model_path=gguf, log_path=None)


# ====================== Process lifecycle ======================


class TestLifecycle:
    def test_is_alive_follows_poll(self):
        proc = FakePopen([])
        assert llama_embedder.is_alive(proc)
        proc.returncode = 0
        assert not llama_embedder.is_alive(proc)
        assert not llama_embedder.is_alive(None)

    def test_exit_code_is_the_popen_returncode(self):
        proc = FakePopen([])
        proc.returncode = -15
        assert llama_embedder.exit_code(proc) == -15
        assert llama_embedder.exit_code(None) is None

    def test_windows_exit_code_is_signed(self, monkeypatch):
        monkeypatch.setattr("src.core.subprocess_flags.platform.system", lambda: "Windows")
        proc = FakePopen([])
        proc.returncode = 4294967295
        assert llama_embedder.exit_code(proc) == -1

    def test_mlx_exit_code_is_the_mp_exitcode(self):
        if sys.platform != "darwin":
            pytest.skip("mlx_embedder imports mlx-only modules")
        from src.ingestion import mlx_embedder

        assert mlx_embedder.exit_code(SimpleNamespace(exitcode=7)) == 7
        assert mlx_embedder.exit_code(None) is None

    def test_terminate_is_idempotent_and_bounded(self):
        calls = []
        proc = SimpleNamespace(
            returncode=None,
            poll=lambda: None,
            send_signal=lambda sig: calls.append(("signal", sig)),
            terminate=lambda: calls.append(("terminate",)),
            wait=lambda timeout=None: calls.append(("wait", timeout)),
            kill=lambda: calls.append(("kill",)),
        )
        llama_embedder.terminate(proc)
        assert ("wait", 5) in calls
        assert ("kill",) not in calls
        llama_embedder.terminate(None)  # no-op


# ====================== Platform dispatch ======================


class TestPlatformDispatch:
    @pytest.fixture
    def by_platform(self, monkeypatch):
        monkeypatch.setattr(embedder_child, "_adapter_override", embedder_child._PLATFORM)

        def set_platform(system: str, machine: str = "x86_64"):
            monkeypatch.setattr(embedder_child.platform, "system", lambda: system)
            monkeypatch.setattr(embedder_child.platform, "machine", lambda: machine)

        return set_platform

    def test_windows_gets_the_llama_adapter(self, by_platform):
        by_platform("Windows")
        assert embedder_child._get_adapter() is llama_embedder

    def test_linux_gets_the_llama_adapter(self, by_platform):
        by_platform("Linux")
        assert embedder_child._get_adapter() is llama_embedder

    def test_macos_arm_keeps_the_mlx_adapter(self, by_platform):
        if sys.platform != "darwin":
            pytest.skip("mlx_embedder imports mlx-only modules")
        from src.ingestion import mlx_embedder

        by_platform("Darwin", "arm64")
        assert embedder_child._get_adapter() is mlx_embedder


# ====================== The gate and the banner (artifact state) ======================


def _fake_adapter(available: bool, downloadable: bool):
    return SimpleNamespace(
        artifact_available=lambda: available,
        artifact_downloadable=downloadable,
    )


class TestArtifactState:
    def test_missing_follows_the_selected_adapter(self, monkeypatch):
        monkeypatch.setattr(embedder_child, "_adapter_override", _fake_adapter(False, False))
        assert embeddings.embedding_artifact_missing()
        monkeypatch.setattr(embedder_child, "_adapter_override", _fake_adapter(True, False))
        assert not embeddings.embedding_artifact_missing()

    def test_status_of_a_bundled_artifact_never_offers_a_download(self, monkeypatch):
        monkeypatch.setattr(embedder_child, "_adapter_override", _fake_adapter(True, False))
        assert embeddings.embedding_artifact_status() == {
            "available": True,
            "downloading": False,
            "error": None,
        }
        monkeypatch.setattr(embedder_child, "_adapter_override", _fake_adapter(False, False))
        status = embeddings.embedding_artifact_status()
        assert status["available"] is False
        assert status["downloading"] is False
        # A bundled artifact that is NOT on disk is an explicit error naming
        # the remedy (the banner/gate shows it verbatim): no download exists
        # that could produce it, so a silent prompt would mislead.
        assert status["error"] == embeddings.BUNDLED_ARTIFACT_MISSING_MESSAGE

    def test_status_of_a_downloadable_artifact_is_the_146_gate(self, monkeypatch):
        monkeypatch.setattr(embedder_child, "_adapter_override", _fake_adapter(False, True))
        monkeypatch.setattr(
            embeddings.embedding_model,
            "download_state",
            lambda: {"available": False, "downloading": True, "error": None},
        )
        assert embeddings.embedding_artifact_status()["downloading"] is True

    def test_download_is_a_no_op_on_a_bundled_artifact(self, monkeypatch, caplog):
        monkeypatch.setattr(embedder_child, "_adapter_override", _fake_adapter(False, False))

        def boom():
            raise AssertionError("must not start the e5 snapshot download")

        monkeypatch.setattr(embeddings.embedding_model, "start_download", boom)
        with caplog.at_level(logging.INFO):
            status = embeddings.start_embedding_artifact_download()
        assert status["available"] is False
        # The answer carries the explicit state, never a silent no-op.
        assert status["error"] == embeddings.BUNDLED_ARTIFACT_MISSING_MESSAGE

    def test_download_still_runs_the_gate_where_it_applies(self, monkeypatch):
        monkeypatch.setattr(embedder_child, "_adapter_override", _fake_adapter(False, True))
        started = []
        monkeypatch.setattr(
            embeddings.embedding_model,
            "start_download",
            lambda: started.append(True)
            or {"available": False, "downloading": True, "error": None},
        )
        assert embeddings.start_embedding_artifact_download()["downloading"] is True
        assert started == [True]


# ====================== CUDA gauges never read this child ======================


class TestCudaGaugeExclusion:
    """The spill, eviction and contention gauges sample the pid of the CHAT
    handle (``CUDA_Engine._model``) and nothing else; the embedding child
    lives in ``embedder_child`` module state, outside every engine class."""

    @pytest.fixture
    def cuda_engine(self):
        from src.engines.cuda_engine import CUDA_Engine

        return CUDA_Engine

    @pytest.fixture
    def embedder_running(self, monkeypatch, tmp_path):
        """A live (fake) embedding child, pid 9999."""
        from tests.test_embedder_child import FakeHttp, make_adapter

        adapter = make_adapter(tmp_path)
        monkeypatch.setattr(embedder_child, "_adapter_override", adapter)
        monkeypatch.setattr(embedder_child, "requests", FakeHttp())
        embedder_child._reset_for_tests()
        handle = embedder_child.ensure_embedder_running()
        yield handle
        embedder_child._reset_for_tests()

    def test_the_eviction_gauge_reads_the_chat_pid_only(
        self, monkeypatch, cuda_engine, embedder_running
    ):
        from src.engines import memory_pressure

        chat_handle = {
            "pid": 1111,
            "proc": SimpleNamespace(poll=lambda: None),
            "port": 27201,
            "gpu_uuid": "GPU-abc",
        }
        monkeypatch.setattr(cuda_engine, "_model", chat_handle)
        monkeypatch.setattr(cuda_engine, "_generation_ended_at", None)
        sampled = []

        def fake_read_vram(pid, healthy_bytes=0):
            sampled.append(pid)
            return SimpleNamespace(
                dedicated_bytes=10,
                shared_bytes=10,
                committed_bytes=20,
                committed_refused=False,
            )

        monkeypatch.setattr(memory_pressure, "read_vram", fake_read_vram)
        monkeypatch.setattr(cuda_engine, "_read_device_free", classmethod(lambda cls, uuid: 0))
        cuda_engine._sample_gpu_contention()

        assert sampled == [1111]
        assert embedder_running["pid"] not in sampled

    def test_the_spill_gauge_reads_the_chat_handle_only(
        self, monkeypatch, cuda_engine, embedder_running
    ):
        from src.engines import memory_pressure

        chat_handle = {
            "pid": 2222,
            "proc": SimpleNamespace(poll=lambda: None),
            "port": 27201,
            "spill_baseline": 0,
            "vram_total_bytes": 1 << 30,
            "context_tokens": 4096,
            "kv_token_bytes": 1024,
        }
        sampled = []

        def fake_read_vram(pid, healthy_bytes=0):
            sampled.append(pid)
            return SimpleNamespace(
                dedicated_bytes=10,
                shared_bytes=10,
                committed_bytes=None,
                committed_refused=False,
            )

        monkeypatch.setattr(memory_pressure, "read_vram", fake_read_vram)
        cuda_engine._llama_read_signals(chat_handle)
        assert sampled == [2222]

    def test_the_embedder_never_enters_the_engine_state(
        self, monkeypatch, cuda_engine, embedder_running
    ):
        # The structural exclusion itself: running the embedding child did not
        # write anything into the engine singleton the gauges sample.
        assert cuda_engine._model is None or embedder_running["pid"] != (
            cuda_engine._model.get("pid") if isinstance(cuda_engine._model, dict) else None
        )
