"""Tests for `CUDA_Engine` post-migration to `BaseLlamaCppEngine`.

Covers the CUDA-specific hooks (`_build_spawn_argv`, `_prepare_spawn_context`,
`_build_spawn_env` with CUDA PATH) and config attrs. The shared subprocess
+ SSE lifecycle is covered by `test_base_chat_server_engine.py`; the
GGUF picker / kwarg translation by `test_cpu_engine_server.py` (same
implementation, inherited from `BaseLlamaCppEngine`).
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.engines import cuda_memory, gguf_memory, memory_pressure
from src.engines.cuda_engine import CUDA_Engine
from src.engines.gguf_memory import shape_from_fields
from src.engines.memory_brake import BrakeOutcome


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


GIB = 1024**3
MIB = 1024**2
QWEN3 = shape_from_fields(
    {
        "architecture": "qwen3",
        "block_count": 28,
        "context_length": 40_960,
        "embedding_length": 1024,
        "head_count": 16,
        "head_count_kv": 8,
        "key_length": 128,
        "value_length": 128,
    }
)
WEIGHTS = int(2.5 * GIB)
UUID = "GPU-8932f937-d72c-4106-c12f-20bd9faed9f6"


def _device(*, total=16 * GIB, free=int(13.8 * GIB), uuid=UUID):
    return cuda_memory.DeviceMemory(
        index=0, uuid=uuid, total_bytes=total, free_bytes=free, processes=()
    )


@pytest.fixture
def gpu(monkeypatch):
    """The NVML seam and the GGUF readers faked: a 16 GiB card with 13.8 GiB
    free and a Qwen3-shaped 2.5 GiB model. The real NVML is never touched."""
    state = SimpleNamespace(device=_device(), inputs=(QWEN3, WEIGHTS), reads=0)

    def _read_vram_info(cls):
        state.reads += 1
        return state.device

    monkeypatch.setattr(CUDA_Engine, "_read_vram_info", classmethod(_read_vram_info))
    monkeypatch.setattr(
        gguf_memory, "read_sizing_inputs", lambda path, mmproj_path=None: state.inputs
    )
    monkeypatch.setattr(
        gguf_memory,
        "read_tensor_bytes",
        lambda path: gguf_memory.TensorBytes(tok_embd_bytes=300 * MIB, exps_bytes=0),
    )
    monkeypatch.setattr(CUDA_Engine, "_find_mmproj", classmethod(lambda cls, p: None))
    monkeypatch.delenv("ERUDI_CTX", raising=False)
    monkeypatch.delenv("ERUDI_VRAM_OVERRIDE_GB", raising=False)
    monkeypatch.setattr(CUDA_Engine, "_window_memory", {})
    return state


def _argv(**ctx):
    return [
        str(x)
        for x in CUDA_Engine._build_spawn_argv(
            llama_server=Path("/bin/llama-server"),
            model_gguf=Path("/m.gguf"),
            alias="erudi-9",
            port=8456,
            **ctx,
        )
    ]


def _flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING]


@pytest.mark.unit
class TestSpawnContextAndArgv:
    def test_the_sized_path_passes_the_window_and_full_offload_to_fit(self, gpu, caplog):
        with caplog.at_level(logging.INFO, logger="erudi"):
            ctx = CUDA_Engine._prepare_spawn_context(Path("/m.gguf"), "erudi-9")
        assert ctx["ctx_size"] == 40_960
        assert ctx["gpu_layers"] == -1
        assert ctx["window_sized"] is True
        assert ctx["threads"] >= 1
        assert ctx["trained_context_tokens"] == ctx["static_window_tokens"] == 40_960
        assert ctx["kv_token_bytes"] == 114_688
        assert ctx["weights_bytes"] == WEIGHTS
        assert ctx["spawn_memory_bound"] is False
        assert ctx["vram_total_bytes"] == 16 * GIB
        assert ctx["fa_proven"] is True
        assert ctx["tok_embd_bytes"] == 300 * MIB
        assert ctx["exps_bytes"] == 0
        assert ctx["kv_classify_bytes"] == 40_960 * 114_688
        assert ctx["gpu_uuid"] == UUID
        assert any("Window sized at spawn" in r.getMessage() for r in caplog.records)
        assert not _warnings(caplog)
        argv = _argv(**ctx)
        assert _flag(argv, "-c") == "40960"
        assert _flag(argv, "-ngl") == "-1"
        assert _flag(argv, "-fitt") == str(cuda_memory.CUDA_FIT_TARGET_MIB) == "1024"
        assert "--fit" not in argv

    def test_a_memory_bound_window_on_a_crowded_card(self, gpu):
        gpu.device = _device(total=8 * GIB, free=int(6.5 * GIB))
        ctx = CUDA_Engine._prepare_spawn_context(Path("/m.gguf"), "erudi-9")
        assert ctx["ctx_size"] == 20_736
        assert ctx["spawn_memory_bound"] is True

    def test_the_alias_window_memory_caps_a_load(self, gpu):
        CUDA_Engine._window_memory["erudi-9"] = {"cap": 20_480, "faulty": None, "at": time.time()}
        ctx = CUDA_Engine._prepare_spawn_context(Path("/m.gguf"), "erudi-9")
        assert ctx["ctx_size"] == 20_480
        assert ctx["static_window_tokens"] == 40_960

    def test_a_floor_that_does_not_fit_falls_back_to_fit_with_the_brake_off(self, gpu, caplog):
        gpu.inputs = (QWEN3, int(7.5 * GIB))
        gpu.device = _device(total=8 * GIB, free=int(6.5 * GIB))
        with caplog.at_level(logging.WARNING, logger="erudi"):
            ctx = CUDA_Engine._prepare_spawn_context(Path("/m.gguf"), "erudi-9")
        assert ctx["ctx_size"] is None
        assert ctx["gpu_layers"] == -1
        assert ctx["window_sized"] is False
        assert len(_warnings(caplog)) == 1
        argv = _argv(**ctx)
        assert "-c" not in argv
        assert _flag(argv, "-ngl") == "-1"
        assert _flag(argv, "-fitt") == "1024"

    def test_an_unreadable_shape_falls_back(self, gpu):
        gpu.inputs = None
        ctx = CUDA_Engine._prepare_spawn_context(Path("/m.gguf"), "erudi-9")
        assert (ctx["ctx_size"], ctx["gpu_layers"], ctx["window_sized"]) == (None, -1, False)

    def test_mla_falls_back(self, gpu, caplog):
        gpu.inputs = (dataclasses.replace(QWEN3, kv_lora_rank=512), WEIGHTS)
        with caplog.at_level(logging.WARNING, logger="erudi"):
            ctx = CUDA_Engine._prepare_spawn_context(Path("/m.gguf"), "erudi-9")
        assert (ctx["ctx_size"], ctx["window_sized"]) == (None, False)
        assert "MLA" in _warnings(caplog)[0].getMessage()

    def test_unreadable_nvml_falls_back(self, gpu, caplog):
        gpu.device = cuda_memory.DeviceMemory(
            index=None, uuid=None, total_bytes=0, free_bytes=0, unavailable="pynvml missing"
        )
        with caplog.at_level(logging.WARNING, logger="erudi"):
            ctx = CUDA_Engine._prepare_spawn_context(Path("/m.gguf"), "erudi-9")
        assert (ctx["ctx_size"], ctx["gpu_layers"], ctx["window_sized"]) == (None, -1, False)
        assert len(_warnings(caplog)) == 1
        assert "pynvml missing" in _warnings(caplog)[0].getMessage()

    def test_no_gpu_runs_on_the_processor(self, gpu, caplog):
        gpu.device = cuda_memory.DeviceMemory(index=None, uuid=None, total_bytes=0, free_bytes=0)
        with caplog.at_level(logging.WARNING, logger="erudi"):
            ctx = CUDA_Engine._prepare_spawn_context(Path("/m.gguf"), "erudi-9")
        assert (ctx["ctx_size"], ctx["gpu_layers"], ctx["window_sized"]) == (None, 0, False)
        assert "NVML reports no GPU" in _warnings(caplog)[0].getMessage()
        assert _flag(_argv(**ctx), "-ngl") == "0"

    def test_erudi_ctx_is_passed_verbatim_and_pinned(self, gpu, monkeypatch):
        monkeypatch.setenv("ERUDI_CTX", "8192")
        ctx = CUDA_Engine._prepare_spawn_context(Path("/m.gguf"), "erudi-9")
        assert ctx["ctx_size"] == 8192
        assert ctx["window_pinned"] is True
        assert ctx["window_sized"] is True
        assert ctx["gpu_layers"] == -1
        assert ctx["kv_classify_bytes"] == 8192 * 114_688
        argv = _argv(**ctx)
        assert _flag(argv, "-c") == "8192"
        assert _flag(argv, "-ngl") == "-1"

    def test_the_vram_override_feeds_the_sizing(self, gpu, monkeypatch, caplog):
        monkeypatch.setenv("ERUDI_VRAM_OVERRIDE_GB", "6.5")
        gpu.device = _device(total=8 * GIB, free=8 * GIB)
        with caplog.at_level(logging.INFO, logger="erudi"):
            ctx = CUDA_Engine._prepare_spawn_context(Path("/m.gguf"), "erudi-9")
        # free = the override, total = max(override, the real total).
        assert ctx["ctx_size"] == 20_736
        assert ctx["vram_total_bytes"] == 8 * GIB
        assert any("ERUDI_VRAM_OVERRIDE_GB" in r.getMessage() for r in caplog.records)

    def test_an_invalid_vram_override_is_one_warning_and_ignored(self, gpu, monkeypatch, caplog):
        monkeypatch.setenv("ERUDI_VRAM_OVERRIDE_GB", "lots")
        with caplog.at_level(logging.WARNING, logger="erudi"):
            ctx = CUDA_Engine._prepare_spawn_context(Path("/m.gguf"), "erudi-9")
        assert ctx["ctx_size"] == 40_960
        assert len(_warnings(caplog)) == 1

    def test_an_empty_vram_override_is_unset(self, gpu, monkeypatch, caplog):
        monkeypatch.setenv("ERUDI_VRAM_OVERRIDE_GB", "")
        with caplog.at_level(logging.INFO, logger="erudi"):
            ctx = CUDA_Engine._prepare_spawn_context(Path("/m.gguf"), "erudi-9")
        assert ctx["ctx_size"] == 40_960
        assert not _warnings(caplog)
        assert not [r for r in caplog.records if "ERUDI_VRAM_OVERRIDE_GB" in r.getMessage()]

    @pytest.mark.parametrize("unavailable, gpu_layers", [(None, 0), ("pynvml missing", -1)])
    def test_a_vram_override_never_invents_a_gpu(self, gpu, monkeypatch, unavailable, gpu_layers):
        """NVML failed or saw no GPU: the override changes nothing."""
        monkeypatch.setenv("ERUDI_VRAM_OVERRIDE_GB", "12")
        gpu.device = cuda_memory.DeviceMemory(
            index=None, uuid=None, total_bytes=0, free_bytes=0, unavailable=unavailable
        )
        ctx = CUDA_Engine._prepare_spawn_context(Path("/m.gguf"), "erudi-9")
        assert (ctx["ctx_size"], ctx["gpu_layers"], ctx["window_sized"]) == (
            None,
            gpu_layers,
            False,
        )

    def test_several_visible_gpus_take_the_unsized_shape(self, gpu, caplog):
        """A user's CUDA_VISIBLE_DEVICES naming several cards: llama.cpp
        splits the layers across them while the sizing reads one card."""
        gpu.device = dataclasses.replace(_device(), visible_devices=2)
        with caplog.at_level(logging.INFO, logger="erudi"):
            ctx = CUDA_Engine._prepare_spawn_context(Path("/m.gguf"), "erudi-9")
        assert (ctx["ctx_size"], ctx["gpu_layers"], ctx["window_sized"]) == (None, -1, False)
        assert "gpu_uuid" not in ctx
        (info,) = [r for r in caplog.records if "several GPUs" in r.getMessage()]
        assert info.levelno == logging.INFO
        assert "-c" not in _argv(**ctx)

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
            gpu_layers=-1,
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
        assert "-ngl -1" in joined
        assert "-fitt 1024" in joined

    def test_build_spawn_argv_omits_c_without_a_window(self):
        """No window (the fallback shape) -> no ``-c``: llama-server's fit
        (ON by default in the pinned b10883) resolves the window and the
        layers at load."""
        assert "-c" not in _argv(ctx_size=None, threads=4, gpu_layers=-1)

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

    def test_the_selected_gpu_is_pinned_by_uuid(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        with patch.object(CUDA_Engine, "_resolve_cuda_bin_dir", return_value=None):
            env = CUDA_Engine._build_spawn_env(gpu_uuid=UUID)
        assert env["CUDA_VISIBLE_DEVICES"] == UUID

    def test_a_user_cuda_visible_devices_is_never_overridden(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
        with patch.object(CUDA_Engine, "_resolve_cuda_bin_dir", return_value=None):
            env = CUDA_Engine._build_spawn_env(gpu_uuid=UUID)
        assert env["CUDA_VISIBLE_DEVICES"] == "1"

    def test_an_empty_cuda_visible_devices_is_replaced_by_the_pin(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        with patch.object(CUDA_Engine, "_resolve_cuda_bin_dir", return_value=None):
            env = CUDA_Engine._build_spawn_env(gpu_uuid=UUID)
        assert env["CUDA_VISIBLE_DEVICES"] == UUID

    def test_no_selected_gpu_sets_nothing(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        with patch.object(CUDA_Engine, "_resolve_cuda_bin_dir", return_value=None):
            env = CUDA_Engine._build_spawn_env()
        assert "CUDA_VISIBLE_DEVICES" not in env


# =====================================================================
# UNIT — the brake gate
# =====================================================================


@pytest.mark.unit
def test_a_child_whose_window_was_not_sized_is_never_sampled(monkeypatch):
    """Explicit brake-off: the pressure reader is not even called, so the
    law's unload branch can never run on a fallback-shaped child."""
    calls = []
    monkeypatch.setattr(memory_pressure, "read_pressure", lambda: calls.append(1))
    monkeypatch.setattr(memory_pressure, "read_vram", lambda pid: calls.append(2))
    proc = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(CUDA_Engine, "_model", {"proc": proc, "window_sized": False, "pid": 1})
    monkeypatch.setattr(CUDA_Engine, "_model_id", 7)
    for point in ("turn_start", "turn_end", "idle"):
        assert CUDA_Engine.brake_point(point, llm_id=7) == BrakeOutcome()
    assert calls == []
