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

from pyi_llama_artifacts import (
    assert_cublas_only_in_cuda_bin,
    collect_llama_artifacts,
    drop_toplevel_cublas,
)


def _flavour(root: Path, flavour: str, *, exe: str, dlls: tuple[str, ...] = ()) -> Path:
    bindir = root / "artifacts" / "llama-cpp" / flavour / "bin"
    bindir.mkdir(parents=True)
    (bindir / exe).write_bytes(b"\x00")
    for dll in dlls:
        (bindir / dll).write_bytes(b"\x00")
    return bindir


CRT = ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")
CUDA_RUNTIME = ("cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll")


def _gguf(root: Path) -> Path:
    """The KB embedding GGUF (#594) the Windows/Linux installers bundle."""
    path = root / "artifacts" / "kb-embedder" / "multilingual-e5-small-q8_0.gguf"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"GGUF")
    return path


@pytest.mark.unit
class TestCudaVariant:
    def test_windows_ships_both_flavours_to_distinct_destinations(self, tmp_path):
        cpu = _flavour(tmp_path, "cpu", exe="llama-server.exe", dlls=CRT)
        cuda = _flavour(tmp_path, "cuda", exe="llama-server.exe", dlls=CRT + CUDA_RUNTIME)
        gguf = _gguf(tmp_path)

        out = collect_llama_artifacts(variant="cuda", root=tmp_path, os_tag="win")

        assert sorted(out.datas) == sorted(
            [
                (str(cpu / "llama-server.exe"), "artifacts/llama-cpp/cpu/bin"),
                (str(cuda / "llama-server.exe"), "artifacts/llama-cpp/cuda/bin"),
                (str(gguf), "artifacts/kb-embedder"),
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
        gguf = _gguf(tmp_path)

        out = collect_llama_artifacts(variant="cuda", root=tmp_path, os_tag="linux")

        assert sorted(out.datas) == sorted(
            [
                (str(cpu / "llama-server"), "artifacts/llama-cpp/cpu/bin"),
                (str(cuda / "llama-server"), "artifacts/llama-cpp/cuda/bin"),
                (str(gguf), "artifacts/kb-embedder"),
            ]
        )
        assert out.runtime_files == []

    @pytest.mark.parametrize("missing", ["cpu", "cuda"])
    def test_a_missing_flavour_fails_the_build_naming_it(self, tmp_path, missing):
        present = "cuda" if missing == "cpu" else "cpu"
        _flavour(tmp_path, present, exe="llama-server.exe")
        _gguf(tmp_path)

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
        _gguf(tmp_path)

        out = collect_llama_artifacts(
            variant="cuda", root=tmp_path, os_tag="win", allow_missing=True
        )

        assert sorted(out.datas) == sorted(
            [
                (str(cuda / "llama-server.exe"), "artifacts/llama-cpp/cuda/bin"),
                (
                    str(tmp_path / "artifacts" / "kb-embedder" / "multilingual-e5-small-q8_0.gguf"),
                    "artifacts/kb-embedder",
                ),
            ]
        )
        assert [Path(p).parent.name for p in out.skipped] == ["bin"]
        assert "cpu" in out.skipped[0].replace("\\", "/").split("/")


@pytest.mark.unit
class TestCpuVariant:
    def test_ships_the_cpu_flavour_only_even_when_cuda_is_built(self, tmp_path):
        cpu = _flavour(tmp_path, "cpu", exe="llama-server.exe", dlls=CRT)
        _flavour(tmp_path, "cuda", exe="llama-server.exe", dlls=CRT + CUDA_RUNTIME)
        gguf = _gguf(tmp_path)

        out = collect_llama_artifacts(variant="cpu", root=tmp_path, os_tag="win")

        assert sorted(out.datas) == sorted(
            [
                (str(cpu / "llama-server.exe"), "artifacts/llama-cpp/cpu/bin"),
                (str(gguf), "artifacts/kb-embedder"),
            ]
        )
        assert sorted(n for n, _ in out.runtime_files) == sorted(
            f"artifacts/llama-cpp/cpu/bin/{d}" for d in CRT
        )

    def test_does_not_require_the_cuda_flavour(self, tmp_path):
        _flavour(tmp_path, "cpu", exe="llama-server")
        _gguf(tmp_path)

        out = collect_llama_artifacts(variant="cpu", root=tmp_path, os_tag="linux")

        assert len(out.datas) == 2

    def test_missing_cpu_flavour_fails_unless_allowed(self, tmp_path):
        _gguf(tmp_path)
        with pytest.raises(RuntimeError, match="build-llamacpp-cpu-win.ps1"):
            collect_llama_artifacts(variant="cpu", root=tmp_path, os_tag="win")

        out = collect_llama_artifacts(
            variant="cpu", root=tmp_path, os_tag="win", allow_missing=True
        )
        assert [d for _, d in out.datas] == ["artifacts/kb-embedder"]
        assert out.runtime_files == [] and len(out.skipped) == 1


@pytest.mark.unit
def test_macos_bundles_no_llama_server(tmp_path):
    # MLX inference: no llama-server, and nothing to fail on.
    out = collect_llama_artifacts(variant="cuda", root=tmp_path, os_tag="mac")
    assert out.datas == [] and out.runtime_files == [] and out.skipped == []


@pytest.mark.unit
def test_an_unknown_variant_is_refused(tmp_path):
    with pytest.raises(ValueError, match="ERUDI_BUILD_VARIANT"):
        collect_llama_artifacts(variant="rocm", root=tmp_path, os_tag="linux")


@pytest.mark.unit
class TestEmbedderGguf:
    """The KB embedding GGUF (#594, F3) ships in EVERY Windows/Linux
    installer; a build without it would gate the Knowledge Base on a
    download that does not exist."""

    @pytest.mark.parametrize("variant", ["cpu", "cuda"])
    def test_a_missing_gguf_fails_the_build_naming_its_script(self, tmp_path, variant):
        _flavour(tmp_path, "cpu", exe="llama-server")
        if variant == "cuda":
            _flavour(tmp_path, "cuda", exe="llama-server")

        with pytest.raises(RuntimeError) as excinfo:
            collect_llama_artifacts(variant=variant, root=tmp_path, os_tag="linux")

        message = str(excinfo.value).replace("\\", "/")
        assert "artifacts/kb-embedder" in message
        assert "scripts/dev/backend/build_e5_gguf.py" in message

    def test_allow_missing_tolerates_an_absent_gguf(self, tmp_path):
        cpu = _flavour(tmp_path, "cpu", exe="llama-server")

        out = collect_llama_artifacts(
            variant="cpu", root=tmp_path, os_tag="linux", allow_missing=True
        )

        assert out.datas == [(str(cpu / "llama-server"), "artifacts/llama-cpp/cpu/bin")]
        assert any("kb-embedder" in p.replace("\\", "/") for p in out.skipped)

    def test_macos_bundles_no_gguf(self, tmp_path):
        # The macOS embedding child serves the downloadable snapshot (F2).
        out = collect_llama_artifacts(variant="cuda", root=tmp_path, os_tag="mac")
        assert out.datas == []


@pytest.mark.unit
class TestCublasOnlyInCudaBin:
    def test_drops_the_toplevel_pair_and_keeps_everything_else(self):
        toc = [
            ("cublas64_12.dll", "C:/cuda/bin/cublas64_12.dll", "BINARY"),
            ("cublasLt64_12.dll", "C:/cuda/bin/cublasLt64_12.dll", "BINARY"),
            ("python312.dll", "C:/py/python312.dll", "BINARY"),
            ("numpy\\core\\_x.pyd", "C:/np/_x.pyd", "EXTENSION"),
        ]
        kept, dropped = drop_toplevel_cublas(toc)
        assert [e[0] for e in kept] == ["python312.dll", "numpy\\core\\_x.pyd"]
        assert dropped == ["cublas64_12.dll", "cublasLt64_12.dll"]

    def test_the_guard_accepts_the_cuda_bin_copies(self):
        datas = [(f"artifacts/llama-cpp/cuda/bin/{n}", f"/s/{n}", "DATA") for n in CUDA_RUNTIME]
        assert_cublas_only_in_cuda_bin([("python312.dll", "x", "BINARY")], datas)

    @pytest.mark.parametrize(
        "dest",
        ["cublasLt64_12.dll", "CUBLAS64_12.DLL", "artifacts/llama-cpp/cpu/bin/cublas64_12.dll"],
    )
    def test_the_guard_rejects_a_cublas_dll_anywhere_else(self, dest):
        with pytest.raises(RuntimeError, match="cuBLAS"):
            assert_cublas_only_in_cuda_bin([], [(dest, "x", "DATA")])

    def test_the_guard_ignores_unrelated_names(self):
        assert_cublas_only_in_cuda_bin(
            [("cublas_helper.pyd", "x", "EXTENSION"), ("mycublas64.dll", "x", "BINARY")]
        )
