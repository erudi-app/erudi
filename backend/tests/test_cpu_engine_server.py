"""Tests for `CPU_Engine` post-migration to `BaseLlamaCppEngine`.

Covers the CPU-specific hooks (`_build_spawn_argv`, `_prepare_spawn_context`)
and verifies the inherited llama-cpp-shared behaviour
(`_select_gguf` quant priority, `_translate_payload_kwargs` rename).
Shared subprocess + SSE lifecycle is covered by `test_base_chat_server_engine.py`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.core.exceptions import EngineException
from src.engines.cpu_engine import CPU_Engine


# =====================================================================
# UNIT — hierarchy
# =====================================================================


@pytest.mark.unit
class TestCpuEngineHierarchy:
    def test_mro_includes_base_llama_cpp_and_chat_server(self):
        names = [c.__name__ for c in CPU_Engine.__mro__]
        assert "BaseLlamaCppEngine" in names
        assert "BaseChatServerEngine" in names
        assert "BaseEngine" in names

    def test_uses_cpu_build_artifact_dir(self):
        # CPU_Engine inherits _use_cuda_build=False from BaseLlamaCppEngine.
        assert CPU_Engine._use_cuda_build is False
        assert CPU_Engine._default_install_dir().name == "bin"
        # 'cpu' segment in the path
        assert "cpu" in str(CPU_Engine._default_install_dir())


# =====================================================================
# UNIT — spawn context / argv
# =====================================================================


@pytest.mark.unit
class TestSpawnContextAndArgv:
    def test_prepare_spawn_context_forces_zero_gpu_layers(self, monkeypatch):
        monkeypatch.delenv("ERUDI_CTX", raising=False)
        ctx = CPU_Engine._prepare_spawn_context()
        assert ctx["gpu_layers"] == 0
        assert ctx["threads"] >= 1
        # Deliberate inversion of the old `>= 1` assertion: without ERUDI_CTX
        # the engine no longer chooses a number — llama-server's own fit
        # resolves the window at load (trained window, reduced only when the
        # machine's memory demands it).
        assert ctx["ctx_size"] is None

    def test_prepare_spawn_context_honours_erudi_ctx_env(self, monkeypatch):
        monkeypatch.setenv("ERUDI_CTX", "8192")
        ctx = CPU_Engine._prepare_spawn_context()
        assert ctx["ctx_size"] == 8192

    def test_build_spawn_argv_emits_required_flags(self):
        llama_server = Path("/bin/llama-server")
        model_gguf = Path("/m.gguf")
        argv = CPU_Engine._build_spawn_argv(
            llama_server=llama_server,
            model_gguf=model_gguf,
            alias="erudi-7",
            port=8123,
            ctx_size=4096,
            threads=8,
            gpu_layers=0,
        )
        joined = " ".join(str(x) for x in argv)
        # str(Path(...)) uses the OS-native separator (backslash on Windows),
        # so the expected substrings must go through the same conversion
        # rather than hardcoding POSIX slashes (#357 CI finding).
        assert str(llama_server) in joined
        assert f"-m {model_gguf}" in joined
        assert "--host 127.0.0.1" in joined
        assert "--port 8123" in joined
        assert "--alias erudi-7" in joined
        assert "-c 4096" in joined
        assert "--threads 8" in joined
        assert "-ngl 0" in joined  # CPU forces 0

    def test_build_spawn_argv_omits_c_without_a_pinned_window(self):
        """No ERUDI_CTX -> no ``-c`` at all: llama-server's own fit (ON by
        default in the pinned b10883) then resolves the window to the model's
        trained window and reduces it only against measured free memory.
        Passing any number here would either shrink the catalog (the old
        hardcoded 4096) or pay KV memory for nothing."""
        argv = CPU_Engine._build_spawn_argv(
            llama_server=Path("/bin/llama-server"),
            model_gguf=Path("/m.gguf"),
            alias="erudi-7",
            port=8123,
            ctx_size=None,
            threads=8,
            gpu_layers=0,
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
            for x in CPU_Engine._build_spawn_argv(
                llama_server=Path("/bin/llama-server"),
                model_gguf=Path("/m.gguf"),
                alias="erudi-7",
                port=8123,
            )
        ]
        assert "--reasoning-format" not in argv


# =====================================================================
# UNIT — _select_gguf (inherited from BaseLlamaCppEngine)
# =====================================================================


@pytest.mark.unit
class TestSelectGguf:
    def test_explicit_gguf_file_returned_as_is(self, tmp_path):
        f = tmp_path / "model-q4_k_m.gguf"
        f.write_bytes(b"\x00" * 16)
        assert CPU_Engine._select_gguf(f) == f.resolve()

    def test_non_gguf_file_raises(self, tmp_path):
        f = tmp_path / "model.safetensors"
        f.write_bytes(b"\x00" * 16)
        with pytest.raises(EngineException, match=".gguf"):
            CPU_Engine._select_gguf(f)

    def test_directory_picks_q4_k_m_first(self, tmp_path):
        d = tmp_path / "mdl"
        d.mkdir()
        for name in ["model-q8_0.gguf", "model-q4_k_m.gguf", "model-f16.gguf"]:
            (d / name).write_bytes(b"\x00" * 1000)
        picked = CPU_Engine._select_gguf(d)
        assert picked.name == "model-q4_k_m.gguf"

    def test_directory_empty_raises(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(EngineException, match="No .gguf"):
            CPU_Engine._select_gguf(d)


# =====================================================================
# UNIT — _translate_payload_kwargs (HF → llama.cpp wire names)
# =====================================================================


@pytest.mark.unit
class TestTranslatePayloadKwargs:
    def test_repetition_penalty_translated(self):
        out = CPU_Engine._translate_payload_kwargs(
            {
                "repetition_penalty": 1.2,
                "repetition_context_size": 96,
            }
        )
        assert out == {"repeat_penalty": 1.2, "repeat_last_n": 96}

    def test_other_kwargs_passthrough(self):
        out = CPU_Engine._translate_payload_kwargs(
            {
                "top_k": 50,
                "min_p": 0.05,
                "seed": 7,
            }
        )
        assert out == {"top_k": 50, "min_p": 0.05, "seed": 7}

    def test_mixed_pass_and_translate(self):
        out = CPU_Engine._translate_payload_kwargs(
            {
                "top_k": 50,
                "repetition_penalty": 1.2,
            }
        )
        assert out == {"top_k": 50, "repeat_penalty": 1.2}

    def test_enable_thinking_becomes_chat_template_kwargs(self):
        # #266: llama-server has no top-level enable_thinking field; it must
        # travel via chat_template_kwargs to reach the Jinja chat template
        # (templates without the kwarg ignore it harmlessly).
        out = CPU_Engine._translate_payload_kwargs(
            {
                "repetition_penalty": 1.1,
                "repetition_context_size": 64,
                "enable_thinking": False,
            }
        )
        assert out == {
            "repeat_penalty": 1.1,
            "repeat_last_n": 64,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def test_no_seed_is_injected(self):
        # llama-server samples randomly by default (seed -1); only the MLX
        # translation stamps a per-request seed, because mlx_vlm.server would
        # otherwise replay DEFAULT_SEED on every generation.
        out = CPU_Engine._translate_payload_kwargs(
            {
                "repetition_penalty": 1.1,
                "repetition_context_size": 64,
            }
        )
        assert "seed" not in out


# =====================================================================
# UNIT — config attrs
# =====================================================================


@pytest.mark.unit
class TestCpuEngineConfig:
    def test_server_name(self):
        assert CPU_Engine._server_name == "llama-server"

    def test_tokenizer_provider(self):
        assert CPU_Engine._tokenizer_provider == "llama-server"

    def test_port_range_starts_in_canonical_block(self):
        # llama.cpp owns 27200–27299 inside Erudi's canonical 271xx–273xx block,
        # clear of the backend (27182–27199) and MLX (27300–27399).
        assert CPU_Engine._port_range_start == 27200
        assert CPU_Engine._port_range_start + CPU_Engine._port_range_count <= 27300

    def test_payload_model_value_returns_handle_alias(self):
        """LlamaCpp engines use the handle's alias (not the MLX sentinel)."""
        assert CPU_Engine._payload_model_value({"alias": "erudi-x"}) == "erudi-x"


# =====================================================================
# UNIT — _read_server_properties (the allocated window, read after boot)
# =====================================================================


def _props_handle() -> dict:
    return {
        "pid": 1,
        "proc": MagicMock(),
        "port": 27200,
        "base_url": "http://127.0.0.1:27200",
        "alias": "erudi-7",
        "model_path": "/m.gguf",
        "api_key": "spawn-key",
    }


def _props_payload() -> dict:
    """The `/props` shape the pinned b10883 answers (verified on a real spawn
    of the bundled binary against SmolLM2-135M without `-c`)."""
    return {
        "default_generation_settings": {"n_ctx": 40960},
        "chat_template_caps": {
            "supports_reasoning_effort": False,
            "supports_system_role": True,
        },
        "chat_template": "{{ bos_token }}...",
    }


@pytest.mark.unit
class TestReadServerProperties:
    """After the probe, ONE bounded `GET /props` reads what llama-server
    actually allocated: `default_generation_settings.n_ctx` is the window the
    engine's fit resolved (trained window, or less when memory demanded it) —
    the value every percentage, warning and budget downstream must use, which
    the declared ceiling (ERUDI_CTX) cannot substitute for."""

    def test_success_stamps_window_caps_and_template_on_the_handle(self):
        handle = _props_handle()
        resp = MagicMock(status_code=200)
        resp.json.return_value = _props_payload()
        with patch("src.engines.base_llama_cpp_engine.requests.get", return_value=resp) as mock_get:
            CPU_Engine._read_server_properties(handle)
        assert handle["context_tokens"] == 40960
        assert handle["chat_template_caps"] == {
            "supports_reasoning_effort": False,
            "supports_system_role": True,
        }
        assert handle["chat_template"] == "{{ bos_token }}..."
        url = mock_get.call_args.args[0]
        assert url == "http://127.0.0.1:27200/props"

    def test_request_carries_the_spawn_key_and_a_bounded_timeout(self):
        """The child only answers its own per-spawn key, and a wedged server
        must not stall the load: the call is authenticated and bounded."""
        handle = _props_handle()
        resp = MagicMock(status_code=200)
        resp.json.return_value = _props_payload()
        with patch("src.engines.base_llama_cpp_engine.requests.get", return_value=resp) as mock_get:
            CPU_Engine._read_server_properties(handle)
        kwargs = mock_get.call_args.kwargs
        assert kwargs["headers"] == {"Authorization": "Bearer spawn-key"}
        assert 0 < kwargs["timeout"] <= 5.0

    def test_timeout_degrades_to_none_with_one_warning(self, caplog):
        handle = _props_handle()
        with patch(
            "src.engines.base_llama_cpp_engine.requests.get",
            side_effect=requests.Timeout("no answer"),
        ):
            with caplog.at_level(logging.WARNING):
                CPU_Engine._read_server_properties(handle)
        assert handle["context_tokens"] is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1

    def test_http_500_degrades_to_none_with_one_warning(self, caplog):
        handle = _props_handle()
        resp = MagicMock(status_code=500, text="boom")
        with patch("src.engines.base_llama_cpp_engine.requests.get", return_value=resp):
            with caplog.at_level(logging.WARNING):
                CPU_Engine._read_server_properties(handle)
        assert handle["context_tokens"] is None
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1

    def test_garbage_body_degrades_to_none_with_one_warning(self, caplog):
        handle = _props_handle()
        resp = MagicMock(status_code=200)
        resp.json.side_effect = ValueError("not json")
        with patch("src.engines.base_llama_cpp_engine.requests.get", return_value=resp):
            with caplog.at_level(logging.WARNING):
                CPU_Engine._read_server_properties(handle)
        assert handle["context_tokens"] is None
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1

    def test_missing_n_ctx_degrades_to_none_with_one_warning(self, caplog):
        handle = _props_handle()
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"chat_template": "t"}
        with patch("src.engines.base_llama_cpp_engine.requests.get", return_value=resp):
            with caplog.at_level(logging.WARNING):
                CPU_Engine._read_server_properties(handle)
        assert handle["context_tokens"] is None
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1

    def test_failure_never_raises_so_the_model_still_loads(self):
        """Degradation is mandatory: a /props hiccup costs the window metadata,
        never the model."""
        handle = _props_handle()
        with patch(
            "src.engines.base_llama_cpp_engine.requests.get",
            side_effect=requests.ConnectionError("refused"),
        ):
            CPU_Engine._read_server_properties(handle)  # must not raise
        assert handle["context_tokens"] is None
