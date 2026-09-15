"""Tests for `CUDA_Engine` post-migration to `BaseLlamaCppEngine`.

Covers the CUDA-specific hooks (`_build_spawn_argv`, `_prepare_spawn_context`,
`_build_spawn_env` with CUDA PATH) and config attrs. The shared subprocess
+ SSE lifecycle is covered by `test_base_chat_server_engine.py`; the
GGUF picker / kwarg translation by `test_cpu_engine_server.py` (same
implementation, inherited from `BaseLlamaCppEngine`).
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.engines.cuda_engine import CUDA_Engine


# =====================================================================
# UNIT — hierarchy & config
# =====================================================================


@pytest.mark.unit
class TestCudaEngineHierarchy:
    def test_mro_includes_base_llama_cpp_and_chat_server(self):
        names = [c.__name__ for c in CUDA_Engine.__mro__]
        assert "BaseLlamaCppEngine" in names
        assert "BaseChatServerEngine" in names
        assert "BaseEngine" in names

    def test_uses_cuda_build_artifact_dir(self):
        assert CUDA_Engine._use_cuda_build is True
        # The default install dir should now point at the cuda flavour.
        assert "cuda" in str(CUDA_Engine._default_install_dir())

    def test_server_name_and_tokenizer_provider(self):
        assert CUDA_Engine._server_name == "llama-server"
        assert CUDA_Engine._tokenizer_provider == "llama-server-cuda"

    def test_payload_model_value_returns_handle_alias(self):
        assert CUDA_Engine._payload_model_value({"alias": "erudi-x"}) == "erudi-x"


# =====================================================================
# UNIT — spawn context / argv (NVML mocked)
# =====================================================================


@pytest.mark.unit
class TestSpawnContextAndArgv:
    def test_prepare_spawn_context_invokes_compute_gpu_layers(self, monkeypatch):
        monkeypatch.delenv("ERUDI_CTX", raising=False)
        with patch.object(CUDA_Engine, "_compute_gpu_layers", return_value=42):
            ctx = CUDA_Engine._prepare_spawn_context()
        assert ctx["gpu_layers"] == 42
        assert ctx["threads"] >= 1
        # Deliberate inversion of the old `>= 1` assertion: without ERUDI_CTX
        # the engine passes no window — llama-server's own fit resolves it at
        # load (trained window, reduced only when memory demands it).
        assert ctx["ctx_size"] is None

    def test_prepare_spawn_context_honours_erudi_ctx_env(self, monkeypatch):
        monkeypatch.setenv("ERUDI_CTX", "8192")
        with patch.object(CUDA_Engine, "_compute_gpu_layers", return_value=10):
            ctx = CUDA_Engine._prepare_spawn_context()
        assert ctx["ctx_size"] == 8192

    def test_build_spawn_argv_emits_required_flags(self):
        llama_server = Path("/bin/llama-server")
        model_gguf = Path("/m.gguf")
        argv = CUDA_Engine._build_spawn_argv(
            llama_server=llama_server,
            model_gguf=model_gguf,
            alias="erudi-9",
            port=8456,
            ctx_size=4096,
            threads=4,
            gpu_layers=32,
        )
        joined = " ".join(str(x) for x in argv)
        # str(Path(...)) uses the OS-native separator (backslash on Windows),
        # so the expected substrings must go through the same conversion
        # rather than hardcoding POSIX slashes (#357 CI finding).
        assert str(llama_server) in joined
        assert f"-m {model_gguf}" in joined
        assert "--port 8456" in joined
        assert "--alias erudi-9" in joined
        assert "-c 4096" in joined
        assert "--threads 4" in joined
        assert "-ngl 32" in joined  # CUDA uses computed layers

    def test_build_spawn_argv_omits_c_without_a_pinned_window(self):
        """No ERUDI_CTX -> no ``-c``: llama-server's fit (ON by default in the
        pinned b10883) resolves the window to the model's trained window and
        reduces it only against measured free memory."""
        argv = CUDA_Engine._build_spawn_argv(
            llama_server=Path("/bin/llama-server"),
            model_gguf=Path("/m.gguf"),
            alias="erudi-9",
            port=8456,
            ctx_size=None,
            threads=4,
            gpu_layers=32,
        )
        assert "-c" not in [str(x) for x in argv]

    def test_build_spawn_argv_keeps_native_reasoning_extraction_on(self):
        """#554: no ``--reasoning-format`` override. llama-server's default
        (``auto``) extracts each family's chain-of-thought into the dedicated
        ``delta.reasoning_content`` field, which ``Erudi_Chat_OpenAI`` carries
        to the runner's ``thinking`` events -- overriding it to ``none`` would
        put the raw tags back into the answer stream."""
        argv = [
            str(x)
            for x in CUDA_Engine._build_spawn_argv(
                llama_server=Path("/bin/llama-server"),
                model_gguf=Path("/m.gguf"),
                alias="erudi-9",
                port=8456,
            )
        ]
        assert "--reasoning-format" not in argv


# =====================================================================
# UNIT — _build_spawn_env (CUDA toolkit on PATH)
# =====================================================================


@pytest.mark.unit
class TestBuildSpawnEnv:
    def test_no_cuda_bin_leaves_path_unchanged(self):
        with patch.object(CUDA_Engine, "_resolve_cuda_bin_dir", return_value=None):
            env = CUDA_Engine._build_spawn_env()
        assert env.get("PATH") == os.environ.get("PATH")

    def test_cuda_bin_prepended_to_path(self):
        fake_bin = Path("/opt/cuda/12.4/bin")
        with patch.object(CUDA_Engine, "_resolve_cuda_bin_dir", return_value=fake_bin):
            env = CUDA_Engine._build_spawn_env()
        assert env["PATH"].startswith(str(fake_bin) + os.pathsep), env["PATH"]
        # Original PATH preserved after the new prefix.
        assert os.environ.get("PATH", "") in env["PATH"]
