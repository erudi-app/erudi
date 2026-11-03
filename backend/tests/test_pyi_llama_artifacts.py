"""The llama.cpp artifacts the PyInstaller specs bundle (`pyi_llama_artifacts`).

The specs only run at build time, on the platform they target, so the decision
of WHAT goes into the frozen tree lives in a pure helper that every CI leg can
exercise: the CUDA variant ships both llama-server flavours (the processor
engine runs the CPU build, never the CUDA one), the CPU variant ships the CPU
build alone, and a missing flavour fails the build by name unless the boot-only
smoke opted out.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyi_llama_artifacts import collect_llama_artifacts


def _flavour(root: Path, flavour: str, *, exe: str, dlls: tuple[str, ...] = ()) -> Path:
    bindir = root / "artifacts" / "llama-cpp" / flavour / "bin"
    bindir.mkdir(parents=True)
    (bindir / exe).write_bytes(b"\x00")
    for dll in dlls:
        (bindir / dll).write_bytes(b"\x00")
    return bindir


CRT = ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")
CUDA_RUNTIME = ("cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll")


@pytest.mark.unit
class TestCudaVariant:
    def test_windows_ships_both_flavours_to_distinct_destinations(self, tmp_path):
        cpu = _flavour(tmp_path, "cpu", exe="llama-server.exe", dlls=CRT)
        cuda = _flavour(tmp_path, "cuda", exe="llama-server.exe", dlls=CRT + CUDA_RUNTIME)

        out = collect_llama_artifacts(variant="cuda", root=tmp_path, os_tag="win")

        assert sorted(out.datas) == sorted(
            [
                (str(cpu / "llama-server.exe"), "artifacts/llama-cpp/cpu/bin"),
                (str(cuda / "llama-server.exe"), "artifacts/llama-cpp/cuda/bin"),
            ]
        )
        cpu_files = {n for n, _ in out.runtime_files if n.startswith("artifacts/llama-cpp/cpu/")}
        cuda_files = {n for n, _ in out.runtime_files if n.startswith("artifacts/llama-cpp/cuda/")}
        assert cpu_files == {f"artifacts/llama-cpp/cpu/bin/{d}" for d in CRT}
        assert cuda_files == {f"artifacts/llama-cpp/cuda/bin/{d}" for d in CRT + CUDA_RUNTIME}
        # Every runtime file is listed from its own flavour's directory: the
        # CPU copy never picks up the CUDA runtime.
        for name, src in out.runtime_files:
            flavour = name.split("/")[2]
            assert Path(src).parent == tmp_path / "artifacts" / "llama-cpp" / flavour / "bin"
        assert out.skipped == []

    def test_linux_ships_both_flavours(self, tmp_path):
        cpu = _flavour(tmp_path, "cpu", exe="llama-server")
        cuda = _flavour(tmp_path, "cuda", exe="llama-server")

        out = collect_llama_artifacts(variant="cuda", root=tmp_path, os_tag="linux")

        assert sorted(out.datas) == sorted(
            [
                (str(cpu / "llama-server"), "artifacts/llama-cpp/cpu/bin"),
                (str(cuda / "llama-server"), "artifacts/llama-cpp/cuda/bin"),
            ]
        )
        assert out.runtime_files == []

    @pytest.mark.parametrize("missing", ["cpu", "cuda"])
    def test_a_missing_flavour_fails_the_build_naming_it(self, tmp_path, missing):
        present = "cuda" if missing == "cpu" else "cpu"
        _flavour(tmp_path, present, exe="llama-server.exe")

        with pytest.raises(RuntimeError) as excinfo:
            collect_llama_artifacts(variant="cuda", root=tmp_path, os_tag="win")

        message = str(excinfo.value)
        assert f"artifacts/llama-cpp/{missing}/bin" in message.replace("\\", "/")
        assert f"build-llamacpp-{missing}-win.ps1" in message
        assert f"build-llamacpp-{present}-win.ps1" not in message

    def test_both_missing_names_both_scripts(self, tmp_path):
        with pytest.raises(RuntimeError) as excinfo:
            collect_llama_artifacts(variant="cuda", root=tmp_path, os_tag="linux")

        message = str(excinfo.value)
        assert "build-llamacpp-cuda-linux.sh" in message
        assert "build-llamacpp-cpu-linux.sh" in message

    def test_allow_missing_tolerates_absent_flavours(self, tmp_path):
        cuda = _flavour(tmp_path, "cuda", exe="llama-server.exe")

        out = collect_llama_artifacts(
            variant="cuda", root=tmp_path, os_tag="win", allow_missing=True
        )

        assert out.datas == [(str(cuda / "llama-server.exe"), "artifacts/llama-cpp/cuda/bin")]
        assert [Path(p).parent.name for p in out.skipped] == ["bin"]
        assert "cpu" in out.skipped[0].replace("\\", "/").split("/")


@pytest.mark.unit
class TestCpuVariant:
    def test_ships_the_cpu_flavour_only_even_when_cuda_is_built(self, tmp_path):
        cpu = _flavour(tmp_path, "cpu", exe="llama-server.exe", dlls=CRT)
        _flavour(tmp_path, "cuda", exe="llama-server.exe", dlls=CRT + CUDA_RUNTIME)

        out = collect_llama_artifacts(variant="cpu", root=tmp_path, os_tag="win")

        assert out.datas == [(str(cpu / "llama-server.exe"), "artifacts/llama-cpp/cpu/bin")]
        assert sorted(n for n, _ in out.runtime_files) == sorted(
            f"artifacts/llama-cpp/cpu/bin/{d}" for d in CRT
        )

    def test_does_not_require_the_cuda_flavour(self, tmp_path):
        _flavour(tmp_path, "cpu", exe="llama-server")

        out = collect_llama_artifacts(variant="cpu", root=tmp_path, os_tag="linux")

        assert len(out.datas) == 1

    def test_missing_cpu_flavour_fails_unless_allowed(self, tmp_path):
        with pytest.raises(RuntimeError, match="build-llamacpp-cpu-win.ps1"):
            collect_llama_artifacts(variant="cpu", root=tmp_path, os_tag="win")

        out = collect_llama_artifacts(
            variant="cpu", root=tmp_path, os_tag="win", allow_missing=True
        )
        assert out.datas == [] and out.runtime_files == [] and len(out.skipped) == 1


@pytest.mark.unit
def test_macos_bundles_no_llama_server(tmp_path):
    # MLX inference: no llama-server, and nothing to fail on.
    out = collect_llama_artifacts(variant="cuda", root=tmp_path, os_tag="mac")
    assert out.datas == [] and out.runtime_files == [] and out.skipped == []


@pytest.mark.unit
def test_an_unknown_variant_is_refused(tmp_path):
    with pytest.raises(ValueError, match="ERUDI_BUILD_VARIANT"):
        collect_llama_artifacts(variant="rocm", root=tmp_path, os_tag="linux")
