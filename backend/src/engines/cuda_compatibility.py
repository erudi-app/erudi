"""Can this NVIDIA GPU actually run the CUDA ``llama-server`` Erudi ships?

``BaseEngine.get_engine()`` picks ``CUDA_Engine`` as soon as NVML reports one
NVIDIA device. That is the right default, but it is not sufficient: the CUDA
build we ship targets a fixed architecture list, so a card below it -- or a
driver too old to JIT the PTX in it -- crashes on the first chat message with a
raw ``GGML_ABORT`` instead of running on the CPU it would run on perfectly well.

This module holds the two halves of the answer, both free of NVML imports at
module scope so it stays importable everywhere:

- :func:`cuda_preflight_notice` -- two NVML reads at startup (compute
  capability, CUDA driver version), turned into an ``engine_notice`` payload
  the frontend renders as a decision. It never raises.
- :func:`classify_cuda_failure` -- the runtime half: the tail a dead child
  printed, matched against the fixed strings ggml emits, so the chat stream's
  error event carries a code instead of a wall of text.

Neither half ever switches the engine on its own: the CPU fallback is
PROPOSED, and only the persisted ``user_settings.inference_backend``
preference (or a reinstall with the CPU build) applies it.
"""

from __future__ import annotations

from typing import Optional

from src.core.logging import logger

# ===================== Compatibility floors =====================
# Every number below is read off the architecture list the CUDA build passes to
# CMake -- ``scripts/dev/backend/build-llamacpp-cuda-linux.sh`` and
# ``scripts/dev/backend/build-llamacpp-cuda-win.ps1``, both of which compile:
#
#     50-virtual;61-virtual;70-virtual;75-virtual;80-virtual;86-real;89-real
#     (+ 120-real when the toolkit is CUDA 12.8 or newer, which the release is)
#
# ``-virtual`` emits PTX only, JIT-compiled by the driver on first run;
# ``-real`` emits native SASS the driver runs as-is.

# Lowest ``-virtual`` entry in that list, and also the floor of CUDA 12 itself:
# the toolkit dropped Kepler (CC 3.x) and CC 2.x entirely, so nothing below 5.0
# can be targeted at all -- no driver update helps.
MIN_COMPUTE_CAPABILITY = (5, 0)

# The ``-real`` entries: these architectures have native code in the binary and
# never go through the driver's PTX JIT. Everything else in the supported range
# runs from the 80-virtual PTX (as does anything newer than Blackwell).
NATIVE_SASS_ARCHITECTURES = frozenset({(8, 6), (8, 9), (12, 0)})

# A card with native SASS only needs a driver that supports the CUDA runtime the
# binary was linked against. NVIDIA's minor-version compatibility makes any
# 12.x driver good enough for a 12.x runtime, so the floor is the 12.0 family.
CUDA_RUNTIME_BASELINE = 12000

# A card WITHOUT native SASS needs the driver to JIT our PTX, and NVIDIA's
# minor-version compatibility explicitly excludes PTX JIT: the driver must be at
# least the one shipped with the toolkit that produced the PTX. The release
# builds with CUDA 12.8, whose driver family is 570.
CUDA_PTX_DRIVER_FLOOR = 12080

# ===================== Failure codes =====================
# The vocabulary shared with the renderer (frontend/src/utils/engineNotice.js).
# Order is significant only as documentation: most specific first.
CUDA_FAILURE_CODES = (
    "CUDA_COMPUTE_CAPABILITY_TOO_LOW",
    "CUDA_DRIVER_TOO_OLD",
    "CUDA_OUT_OF_MEMORY",
    "CUDA_ERROR",
)

# What ggml prints, verbatim, from
# ``backend/forks/llama-cpp/ggml/src/ggml-cuda/ggml-cuda.cu`` -- one
# ``CUDA error: <cudaGetErrorString>`` line, then device/function/file, then
# ``GGML_ABORT``. Matched case-sensitively: these are fixed C strings, and a
# case-insensitive match would start catching unrelated prose.
_CUDA_ERROR_MARKER = "CUDA error:"
_SIGNATURES: tuple[tuple[str, str], ...] = (
    # The card is below the binary's architecture floor: no SASS matches it and
    # no PTX can be JIT-compiled for it either.
    ("no kernel image is available for execution on the device", "CUDA_COMPUTE_CAPABILITY_TOO_LOW"),
    # The driver refused to JIT our PTX because it predates the toolkit.
    ("the provided PTX was compiled with an unsupported toolchain", "CUDA_DRIVER_TOO_OLD"),
    # Printed by ggml_cuda_init, which does NOT abort -- llama.cpp carries on
    # with zero devices and silently runs on the CPU. The pre-flight is what
    # normally catches this case; classify it here too if it ever surfaces.
    ("CUDA driver version is insufficient for CUDA runtime version", "CUDA_DRIVER_TOO_OLD"),
    # VRAM exhaustion. Kept out of the generic bucket because the remedy is
    # different: a smaller model or a smaller context, not a driver update.
    ("cudaMalloc failed", "CUDA_OUT_OF_MEMORY"),
    ("out of memory", "CUDA_OUT_OF_MEMORY"),
)


def classify_cuda_failure(child_output: Optional[str]) -> Optional[str]:
    """Map a dead child's captured output to a CUDA failure code.

    Args:
        child_output: The tail the drainer collected before the child exited.
            ``None`` or empty is fine.

    Returns:
        One of :data:`CUDA_FAILURE_CODES`, or ``None`` when the output shows no
        CUDA failure at all -- a missing GGUF, a corrupt model, a port clash.
        ``None`` is what keeps a non-CUDA crash on the generic error path.
    """
    if not child_output:
        return None
    for signature, code in _SIGNATURES:
        if signature in child_output:
            return code
    if _CUDA_ERROR_MARKER in child_output:
        return "CUDA_ERROR"
    return None


def _format_compute_capability(capability: tuple[int, int]) -> str:
    return f"{capability[0]}.{capability[1]}"


def format_cuda_version(version: int) -> str:
    """Render an NVML CUDA version int (``12080``) for display (``"12.8"``)."""
    major, minor = divmod(version, 1000)
    return f"{major}.{minor // 10}"


def required_cuda_version(capability: tuple[int, int]) -> int:
    """The driver's CUDA version this card needs to run the bundled binary.

    A card with native SASS in the binary only needs the runtime baseline; a
    card that depends on the PTX needs the toolkit's own driver family.
    """
    if capability in NATIVE_SASS_ARCHITECTURES:
        return CUDA_RUNTIME_BASELINE
    return CUDA_PTX_DRIVER_FLOOR


def cuda_preflight_notice() -> Optional[dict]:
    """Verdict on the selected NVIDIA GPU, as an ``engine_notice`` payload.

    Reads the first NVML device's compute capability and the system's CUDA
    driver version through ``CUDA_Engine``'s existing helpers, and returns:

    - ``CUDA_COMPUTE_CAPABILITY_TOO_LOW`` when the card is below CUDA 12's own
      floor -- there is no driver that fixes it;
    - ``CUDA_DRIVER_TOO_OLD`` when the card is fine but the driver is below the
      version it needs (see :func:`required_cuda_version`);
    - ``None`` when the machine can run the bundled build, when a reading is
      unavailable, or when NVML fails.

    An unreadable value yields ``None`` on purpose: guessing from a failed read
    is how a healthy card gets told its driver is too old. Never raises --
    startup must not fail because a diagnostic could not run.
    """
    try:
        from src.engines.cuda_engine import CUDA_Engine

        gpus = CUDA_Engine._get_nvml_gpus()
        if not gpus:
            logger.warning("CUDA pre-flight skipped: NVML reported no device.")
            return None
        gpu = gpus[0]
        capability = tuple(CUDA_Engine._get_compute_capability(gpu.get("handle")))
        driver_version = CUDA_Engine._get_cuda_driver_version_int()
    except Exception as e:
        logger.warning(f"CUDA pre-flight skipped: {e}")
        return None

    if capability == (0, 0):
        logger.warning("CUDA pre-flight skipped: compute capability is unreadable.")
        return None
    if not driver_version:
        logger.warning("CUDA pre-flight skipped: CUDA driver version is unreadable.")
        return None

    gpu_name = gpu.get("name") or "NVIDIA GPU"
    needed = required_cuda_version(capability)
    notice = {
        "event": "engine_notice",
        "gpu_name": gpu_name,
        "compute_capability": _format_compute_capability(capability),
        "driver_cuda_version": format_cuda_version(driver_version),
        "required_cuda_version": format_cuda_version(needed),
    }

    if capability < MIN_COMPUTE_CAPABILITY:
        floor = _format_compute_capability(MIN_COMPUTE_CAPABILITY)
        notice["code"] = "CUDA_COMPUTE_CAPABILITY_TOO_LOW"
        notice["raw"] = (
            f"{gpu_name} has compute capability {notice['compute_capability']}; "
            f"the bundled CUDA build requires {floor} or newer."
        )
        logger.warning(
            f"CUDA pre-flight: compute capability {notice['compute_capability']} "
            f"is below the {floor} floor of the bundled CUDA build."
        )
        return notice

    if driver_version < needed:
        notice["code"] = "CUDA_DRIVER_TOO_OLD"
        notice["raw"] = (
            f"{gpu_name} (compute capability {notice['compute_capability']}) needs a driver "
            f"providing CUDA {notice['required_cuda_version']} or newer; this system reports "
            f"CUDA {notice['driver_cuda_version']}."
        )
        logger.warning(
            f"CUDA pre-flight: driver CUDA {notice['driver_cuda_version']} is below the "
            f"{notice['required_cuda_version']} this GPU needs."
        )
        return notice

    logger.info(
        f"CUDA pre-flight ok: compute capability {notice['compute_capability']}, "
        f"driver CUDA {notice['driver_cuda_version']}."
    )
    return None
