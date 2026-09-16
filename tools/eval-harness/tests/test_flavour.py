from pathlib import Path

import pytest

from erudi_eval import flavour

HARNESS = Path(__file__).resolve().parent.parent


def test_every_shipped_flavour_loads_and_is_consistent():
    names = sorted(p.stem for p in (HARNESS / "flavours").glob("*.json"))
    assert names == ["linux-cpu", "linux-cuda", "mac-mlx", "win-cpu", "win-cuda"]
    for name in names:
        f = flavour.load(HARNESS, name)
        assert f.name == name and f.os in ("darwin", "windows", "linux")
        assert f.backend_type in ("mlx", "cuda", "cpu") and f.default_model_link
        assert f.per_process_gpu_memory in ("nvml", "unified", "none")
        if f.backend_type == "mlx":
            assert f.inference_role == "mlx_child" and f.per_process_gpu_memory == "unified"
        else:
            assert f.inference_role == "llama_server" and "llama-server" in f.inference_process["executable_basenames"][0]
    assert flavour.load(HARNESS, "mac-mlx").default_model_link.endswith("MLX-4bit")
    assert flavour.load(HARNESS, "win-cuda").default_model_link.endswith("GGUF")
    assert "ngl" in flavour.load(HARNESS, "win-cuda").extra_checks
    assert flavour.load(HARNESS, "win-cpu").extra_checks == ["threads", "ngl_zero"]
    with pytest.raises(ValueError, match="unknown flavour"):
        flavour.load(HARNESS, "mac-cuda")


def test_default_for_os_and_artifact():
    assert flavour.default_name("darwin", "mlx") == "mac-mlx"
    assert flavour.default_name("windows", "cuda") == "win-cuda"
    assert flavour.default_name("windows", "cpu") == "win-cpu"
    assert flavour.default_name("windows", "unknown") == "win-cpu"  # the CPU build is the safe default
    assert flavour.default_name("linux", "cuda") == "linux-cuda"


def test_detect_from_the_running_app():
    startup = {"backend_type": "cuda", "global_inference_label": "fast"}
    env = {"engine": "CUDA_Engine", "gpu_name": "RTX 4070"}
    d = flavour.detect("windows", startup, env)
    assert d["backend_type"] == "cuda" and d["engine"] == "CUDA_Engine" and d["flavour"] == "win-cuda"
    d2 = flavour.detect("darwin", {"backend_type": "mlx"}, {"engine": "MLX_Engine"})
    assert d2["flavour"] == "mac-mlx"
    d3 = flavour.detect("windows", {}, {"engine": "CPU_Engine"})
    assert d3["flavour"] == "win-cpu" and d3["backend_type"] is None


def test_mismatch_is_reported_with_both_sides():
    f = flavour.load(HARNESS, "win-cuda")
    ok, message = flavour.check(f, {"backend_type": "cuda", "engine": "CUDA_Engine", "flavour": "win-cuda"})
    assert ok and "cuda" in message
    ok, message = flavour.check(f, {"backend_type": "cpu", "engine": "CPU_Engine", "flavour": "win-cpu"})
    assert not ok and "win-cuda" in message and "win-cpu" in message and "CPU_Engine" in message
    ok, message = flavour.check(f, {"backend_type": None, "engine": None, "flavour": None})
    assert ok and "could not be detected" in message  # nothing to contradict: not a failure


def test_inference_process_and_gpu_expectations():
    mac, cuda, cpu = (flavour.load(HARNESS, n) for n in ("mac-mlx", "win-cuda", "win-cpu"))
    assert flavour.inference_matches(mac, [{"category": "inference", "role": "mlx_child", "name": "backend"}]) == (True, "mlx_child")
    assert flavour.inference_matches(cuda, [{"category": "inference", "role": "llama_server", "name": "llama-server.exe"}])[0] is True
    ok, seen = flavour.inference_matches(cuda, [{"category": "inference", "role": "mlx_child", "name": "backend"}])
    assert not ok and seen == "mlx_child"
    assert flavour.inference_matches(cpu, []) == (None, None)  # nothing resident: nothing to check
    assert flavour.gpu_expectation(cuda) == "nvml"
    assert flavour.extra_check_values(cpu, "llama-server.exe --port 27201 --threads 8 -ngl 0 --api-key <redacted>") == {"threads": "8", "ngl": "0"}
    assert flavour.extra_check_values(cuda, "llama-server.exe --port 27201 --threads 8 -ngl 37") == {"threads": "8", "ngl": "37"}
    assert flavour.extra_check_values(mac, "backend --multiprocessing-fork tracker_fd=30") == {}
