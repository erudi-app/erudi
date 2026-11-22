"""Which llama.cpp artifacts a PyInstaller build bundles.

Imported by `backend.spec` (and so by `backend-cpu.spec`, which executes it),
and kept free of PyInstaller so every CI leg can unit-test it
(tests/test_pyi_llama_artifacts.py): the specs themselves only run at build
time, on the platform they target.

What ships, per build variant (`ERUDI_BUILD_VARIANT`):

- ``cpu`` (backend-cpu.spec): ``artifacts/llama-cpp/cpu/bin`` alone.
- ``cuda`` (backend.spec, the Windows and Linux CUDA installers): BOTH
  ``cuda/bin`` and ``cpu/bin``. ``CUDA_Engine`` runs the CUDA build; the
  processor engine (``CPU_Engine``, picked by the "Processor only" setting)
  runs the CPU build, which loads no NVIDIA library at all -- the CUDA build
  initialises the card even with ``-ngl 0`` and does not start without a
  driver.

Each flavour lands in its own directory: a Windows executable loads the DLLs
beside it first, so the CPU copy never sees the CUDA runtime.

Both variants also bundle ``artifacts/kb-embedder/multilingual-e5-small-q8_0.gguf``
(#594): the Knowledge Base embedding child on Windows/Linux is the CPU
``llama-server --embedding`` serving that file, produced reproducibly by
``scripts/dev/backend/build_e5_gguf.py`` and never downloaded at runtime. A
build without it would gate every Knowledge Base on a download that does not
exist, so its absence fails the build like a missing server, under the same
``allow_missing`` opt-out (the boot-only smoke). macOS bundles neither: its
embedding child serves the downloadable snapshot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

_FLAVOURS = {"cuda": ("cuda", "cpu"), "cpu": ("cpu",)}

#: The KB embedding GGUF every Windows/Linux installer carries (#594) and
#: the script that produces it (named by the build failure).
_E5_GGUF = Path("artifacts") / "kb-embedder" / "multilingual-e5-small-q8_0.gguf"
_E5_GGUF_SCRIPT = "scripts/dev/backend/build_e5_gguf.py"


@dataclass
class LlamaArtifacts:
    # (source path, destination directory) pairs for the spec's `datas`.
    datas: List[Tuple[str, str]] = field(default_factory=list)
    # (destination file name, source path) pairs the spec appends to `a.datas`
    # verbatim AFTER Analysis (see backend.spec).
    runtime_files: List[Tuple[str, str]] = field(default_factory=list)
    # Server paths that were absent and tolerated (allow_missing).
    skipped: List[str] = field(default_factory=list)


def _build_script(flavour: str, os_tag: str) -> str:
    suffix = ".ps1" if os_tag == "win" else ".sh"
    return f"scripts/dev/backend/build-llamacpp-{flavour}-{os_tag}{suffix}"


def collect_llama_artifacts(
    *, variant: str, root: Path, os_tag: str, allow_missing: bool = False
) -> LlamaArtifacts:
    """List the llama-server files to bundle for ``variant`` on ``os_tag``.

    ``root`` is the backend directory (the spec's ``SPECPATH``); ``os_tag`` is
    ``"win"``, ``"linux"`` or ``"mac"`` (MLX: nothing to bundle). A flavour
    whose server is missing raises ``RuntimeError`` naming its build script,
    unless ``allow_missing`` (the boot-only smoke build) is set.
    """
    if variant not in _FLAVOURS:
        raise ValueError(
            f"Unknown ERUDI_BUILD_VARIANT {variant!r}: expected one of {sorted(_FLAVOURS)}"
        )
    out = LlamaArtifacts()
    if os_tag == "mac":
        return out

    exe = "llama-server.exe" if os_tag == "win" else "llama-server"
    # (what is missing, its path, the command that produces it)
    missing: List[Tuple[str, Path, str]] = []

    gguf = Path(root) / _E5_GGUF
    if gguf.exists():
        out.datas.append((str(gguf), str(_E5_GGUF.parent).replace("\\", "/")))
    else:
        missing.append(("the KB embedding GGUF", gguf, f"python {_E5_GGUF_SCRIPT}"))
    for flavour in _FLAVOURS[variant]:
        bindir = Path(root) / "artifacts" / "llama-cpp" / flavour / "bin"
        server = bindir / exe
        if not server.exists():
            missing.append((f"the {flavour} llama-server", server, _build_script(flavour, os_tag)))
            continue
        dest = f"artifacts/llama-cpp/{flavour}/bin"
        # Only the inference server ships. The HF -> GGUF conversion toolchain
        # was removed with #408: the app downloads pre-built quants only.
        out.datas.append((str(server), dest))
        # The runtime DLLs the build scripts copy beside the server: the MSVC
        # C++ runtime for both flavours (#144, so it loads without the VC++
        # redistributable), plus cudart/cublas/cublasLt for CUDA (so the user
        # needs only a driver). None on Linux, where the builds are static.
        out.runtime_files += [(f"{dest}/{f.name}", str(f)) for f in sorted(bindir.glob("*.dll"))]

    if missing and not allow_missing:
        # A bundle without these files is an app that loads a model and then
        # fails at the first message (or, for the CPU flavour, whose
        # "Processor only" setting runs the CUDA build; or, for the GGUF,
        # whose Knowledge Base is gated on a download that does not exist),
        # and nothing downstream would catch it. Fail here, where the cause
        # is legible.
        lines = "\n".join(f"    {path}  <- {producer}" for _what, path, producer in missing)
        names = ", ".join(what for what, _path, _producer in missing)
        raise RuntimeError(
            f"Missing for the {variant} build: {names}. The frozen backend would ship "
            f"without them. Produce them first:\n{lines}\n"
            f"Set ERUDI_ALLOW_MISSING_INFERENCE_BINARY=1 only for a build that is never "
            f"released (the boot-only smoke gate)."
        )
    out.skipped = [str(path) for _what, path, _producer in missing]
    return out


#: Where the CUDA llama-server loads its cuBLAS from: beside itself.
CUDA_BIN_DIR = "artifacts/llama-cpp/cuda/bin"

#: cublas64_12.dll / cublasLt64_12.dll (Windows). Linux builds are static and
#: out of scope.
_CUBLAS_DLL = re.compile(r"^cublas(lt)?(64)?(_\d+)?\.dll$", re.IGNORECASE)


def _norm(dest: str) -> str:
    return str(dest).replace("\\", "/")


def is_cublas_dll(dest: str) -> bool:
    return bool(_CUBLAS_DLL.match(_norm(dest).rsplit("/", 1)[-1]))


def drop_toplevel_cublas(toc: Sequence[tuple]) -> Tuple[list, List[str]]:
    """Remove cuBLAS DLLs from a PyInstaller binaries TOC; return (kept, dropped).

    PyInstaller reclassifies the PE files found in ``datas`` -- the
    ``llama-server.exe`` entries -- as binaries and scans their imports. On the
    CUDA leg the server imports ``cublas64_12.dll``, which resolves on PATH to
    the CUDA toolkit the runner installed, and that DLL imports
    ``cublasLt64_12.dll`` (643 MB): both land at the top of ``_internal``. The
    frozen backend never loads cuBLAS (only the llama-server child does, from
    ``cuda/bin``, which ``_llama_runtime_dlls`` ships verbatim after Analysis),
    so those copies are dead weight. Only ``a.binaries`` is filtered: the
    ``cuda/bin`` copies live in ``a.datas`` and are never touched.
    """
    kept, dropped = [], []
    for entry in toc:
        (dropped if is_cublas_dll(entry[0]) else kept).append(entry)
    return kept, [_norm(e[0]) for e in dropped]


def assert_cublas_only_in_cuda_bin(*tocs: Sequence[tuple]) -> None:
    """Fail the build when a cuBLAS DLL is bundled anywhere but ``cuda/bin``."""
    stray = sorted(
        {
            _norm(e[0])
            for toc in tocs
            for e in toc
            if is_cublas_dll(e[0]) and not _norm(e[0]).startswith(CUDA_BIN_DIR + "/")
        }
    )
    if stray:
        raise RuntimeError(
            f"cuBLAS DLL(s) bundled outside {CUDA_BIN_DIR}: {', '.join(stray)}. Only "
            f"llama-server uses cuBLAS and it loads its own copies; these would add "
            f"~750 MB to the CUDA installer."
        )
