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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

_FLAVOURS = {"cuda": ("cuda", "cpu"), "cpu": ("cpu",)}


@dataclass
class LlamaArtifacts:
    # (source path, destination directory) pairs for the spec's `datas`.
    datas: List[Tuple[str, str]] = field(default_factory=list)
    # (destination file name, source path) pairs the spec appends to `a.datas`
    # verbatim AFTER Analysis and the torch-CUDA strip (see backend.spec).
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
    missing: List[Tuple[str, Path]] = []
    for flavour in _FLAVOURS[variant]:
        bindir = Path(root) / "artifacts" / "llama-cpp" / flavour / "bin"
        server = bindir / exe
        if not server.exists():
            missing.append((flavour, server))
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
        # A bundle without these binaries is an app that loads a model and
        # then fails at the first message (or, for the CPU flavour, whose
        # "Processor only" setting runs the CUDA build), and nothing
        # downstream would catch it. Fail here, where the cause is legible.
        lines = "\n".join(
            f"    {server}  <- {_build_script(flavour, os_tag)}" for flavour, server in missing
        )
        raise RuntimeError(
            f"llama-server missing for the {variant} build. The frozen backend would ship "
            f"without its inference engine. Compile it first:\n{lines}\n"
            f"Set ERUDI_ALLOW_MISSING_INFERENCE_BINARY=1 only for a build that is never "
            f"released (the boot-only smoke gate)."
        )
    out.skipped = [str(server) for _flavour, server in missing]
    return out
