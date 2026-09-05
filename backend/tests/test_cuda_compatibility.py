"""Pre-flight verdicts and crash classification for the bundled CUDA build.

Two independent concerns live in ``src.engines.cuda_compatibility`` and are
tested here:

1. The startup PRE-FLIGHT: two NVML reads (compute capability, CUDA driver
   version) decide whether the GPU the app just selected can actually run the
   CUDA ``llama-server`` we ship. The risk this file exists to guard is not "do
   we fail loudly" but "do we deliver a WRONG verdict" -- a healthy RTX 3060 on
   a current driver must never be told its driver is too old. Every boundary is
   therefore pinned exactly.
2. The RUNTIME classification: the child's captured tail is matched against the
   fixed strings ggml prints, so an early crash carries a code the UI can act
   on instead of an untyped wall of text.

NVML is fully mocked: none of this needs an NVIDIA GPU.
"""

import pytest

from src.engines import cuda_compatibility as cc


# --------------------------------------------------------------------------
# Pre-flight
# --------------------------------------------------------------------------


def _fake_gpu(name="NVIDIA GeForce RTX 3060"):
    return {"id": 0, "handle": object(), "name": name}


def _patch_nvml(monkeypatch, *, gpus, compute_capability, driver_version):
    """Stand in for the three CUDA_Engine NVML readers the pre-flight uses."""
    from src.engines.cuda_engine import CUDA_Engine

    monkeypatch.setattr(CUDA_Engine, "_get_nvml_gpus", classmethod(lambda cls: gpus))
    monkeypatch.setattr(
        CUDA_Engine,
        "_get_compute_capability",
        classmethod(lambda cls, handle: compute_capability),
    )
    monkeypatch.setattr(
        CUDA_Engine,
        "_get_cuda_driver_version_int",
        classmethod(lambda cls: driver_version),
    )


@pytest.mark.unit
def test_preflight_flags_a_pre_maxwell_card(monkeypatch):
    """Compute capability below 5.0: CUDA 12 cannot target the card at all."""
    _patch_nvml(
        monkeypatch,
        gpus=[_fake_gpu("NVIDIA GeForce GTX 780")],
        compute_capability=(3, 5),
        driver_version=12080,
    )
    notice = cc.cuda_preflight_notice()

    assert notice is not None
    assert notice["event"] == "engine_notice"
    assert notice["code"] == "CUDA_COMPUTE_CAPABILITY_TOO_LOW"
    assert notice["gpu_name"] == "NVIDIA GeForce GTX 780"
    assert notice["compute_capability"] == "3.5"


@pytest.mark.unit
def test_preflight_flags_a_ptx_only_card_on_an_old_driver(monkeypatch):
    """CC 6.1 has no native code in our binary: the driver must JIT the PTX,
    and NVIDIA's minor-version compatibility does not cover PTX -- the driver
    has to be at least the one shipped with CUDA 12.8."""
    _patch_nvml(
        monkeypatch,
        gpus=[_fake_gpu("NVIDIA GeForce GTX 1080")],
        compute_capability=(6, 1),
        driver_version=12010,
    )
    notice = cc.cuda_preflight_notice()

    assert notice is not None
    assert notice["code"] == "CUDA_DRIVER_TOO_OLD"
    assert notice["driver_cuda_version"] == "12.1"
    assert notice["required_cuda_version"] == "12.8"


@pytest.mark.unit
def test_preflight_stays_silent_for_a_native_card_on_the_baseline_driver(monkeypatch):
    """CC 8.9 is compiled as native SASS (89-real), so no PTX JIT happens and
    the plain CUDA 12.0 runtime baseline is enough. A 12.1 driver is FINE."""
    _patch_nvml(
        monkeypatch,
        gpus=[_fake_gpu("NVIDIA GeForce RTX 4090")],
        compute_capability=(8, 9),
        driver_version=12010,
    )
    assert cc.cuda_preflight_notice() is None


@pytest.mark.unit
def test_preflight_stays_silent_for_a_ptx_card_on_a_current_driver(monkeypatch):
    """CC 6.1 with exactly the 12.8 driver: the PTX floor is met, no notice."""
    _patch_nvml(
        monkeypatch,
        gpus=[_fake_gpu("NVIDIA GeForce GTX 1080")],
        compute_capability=(6, 1),
        driver_version=12080,
    )
    assert cc.cuda_preflight_notice() is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "compute_capability, driver_version, expected",
    [
        # --- the compute-capability boundary, 5.0 ---
        ((4, 0), 12080, "CUDA_COMPUTE_CAPABILITY_TOO_LOW"),  # below the floor
        ((5, 0), 12080, None),  # exactly the floor, current driver
        ((5, 0), 12070, "CUDA_DRIVER_TOO_OLD"),  # at the floor, PTX-only
        # --- the native-SASS boundary, 8.6 (8.0 JITs, 8.6 does not) ---
        ((8, 0), 12010, "CUDA_DRIVER_TOO_OLD"),  # 80-virtual: PTX, needs 12.8
        ((8, 6), 12010, None),  # 86-real: native, 12.0 baseline is enough
        ((8, 6), 11080, "CUDA_DRIVER_TOO_OLD"),  # native but below the runtime baseline
        # --- the driver boundaries, 12000 and 12080 ---
        ((6, 1), 12079, "CUDA_DRIVER_TOO_OLD"),  # one below the PTX floor
        ((6, 1), 12080, None),  # exactly the PTX floor
        ((8, 9), 12000, None),  # exactly the runtime baseline, native card
        ((8, 9), 11990, "CUDA_DRIVER_TOO_OLD"),  # one below it
        # --- a healthy current machine must never be nagged ---
        ((8, 6), 12040, None),  # RTX 3060 on driver 550 / CUDA 12.4
        ((12, 0), 12080, None),  # RTX 50, native 120-real
    ],
)
def test_preflight_boundaries(monkeypatch, compute_capability, driver_version, expected):
    _patch_nvml(
        monkeypatch,
        gpus=[_fake_gpu()],
        compute_capability=compute_capability,
        driver_version=driver_version,
    )
    notice = cc.cuda_preflight_notice()
    assert (notice or {}).get("code") == expected


@pytest.mark.unit
def test_preflight_never_crashes_when_nvml_fails(monkeypatch):
    """NVML raising is not a startup failure: no notice, a WARNING in the log."""
    from src.engines.cuda_engine import CUDA_Engine

    def _boom(cls):
        raise RuntimeError("NVML is not loaded")

    monkeypatch.setattr(CUDA_Engine, "_get_nvml_gpus", classmethod(_boom))
    assert cc.cuda_preflight_notice() is None


@pytest.mark.unit
def test_preflight_is_silent_when_nvml_reports_no_usable_reading(monkeypatch):
    """No device, or an unreadable compute capability ((0, 0)) -> no verdict.

    Guessing from a failed read is exactly how a healthy card gets a wrong
    verdict, so an unreadable capability produces nothing at all.
    """
    _patch_nvml(monkeypatch, gpus=[], compute_capability=(8, 6), driver_version=12080)
    assert cc.cuda_preflight_notice() is None

    _patch_nvml(monkeypatch, gpus=[_fake_gpu()], compute_capability=(0, 0), driver_version=12080)
    assert cc.cuda_preflight_notice() is None

    _patch_nvml(monkeypatch, gpus=[_fake_gpu()], compute_capability=(6, 1), driver_version=0)
    assert cc.cuda_preflight_notice() is None


# --------------------------------------------------------------------------
# Runtime classification
# --------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "tail, expected",
    [
        (
            "CUDA error: no kernel image is available for execution on the device\n"
            "  current device: 0, in function ggml_cuda_op_mul_mat",
            "CUDA_COMPUTE_CAPABILITY_TOO_LOW",
        ),
        (
            "CUDA error: the provided PTX was compiled with an unsupported toolchain.",
            "CUDA_DRIVER_TOO_OLD",
        ),
        (
            "ggml_cuda_init: failed to initialize CUDA: "
            "CUDA driver version is insufficient for CUDA runtime version",
            "CUDA_DRIVER_TOO_OLD",
        ),
        (
            "CUDA error: out of memory\n  cudaMalloc failed",
            "CUDA_OUT_OF_MEMORY",
        ),
        (
            "CUDA error: an illegal memory access was encountered",
            "CUDA_ERROR",
        ),
        # Not a CUDA failure at all -> no code, the generic path stays generic.
        ("error: unable to load model: no .gguf file found", None),
        ("", None),
        (None, None),
    ],
)
def test_classify_cuda_failure(tail, expected):
    assert cc.classify_cuda_failure(tail) == expected


@pytest.mark.unit
def test_classification_is_case_sensitive():
    """ggml prints these strings verbatim; a lowercased look-alike is not it."""
    assert cc.classify_cuda_failure("cuda error: NO KERNEL IMAGE IS AVAILABLE") is None


@pytest.mark.unit
def test_out_of_memory_wins_over_the_generic_cuda_code():
    """A VRAM exhaustion has its own remedy (a smaller model), so it must not
    collapse into the generic CUDA_ERROR bucket."""
    tail = "CUDA error: out of memory\ncurrent device: 0, in function ggml_backend_cuda_buffer"
    assert cc.classify_cuda_failure(tail) == "CUDA_OUT_OF_MEMORY"


@pytest.mark.unit
def test_the_known_codes_are_the_ones_the_frontend_maps():
    """The code vocabulary is a wire contract shared with the renderer."""
    assert cc.CUDA_FAILURE_CODES == (
        "CUDA_COMPUTE_CAPABILITY_TOO_LOW",
        "CUDA_DRIVER_TOO_OLD",
        "CUDA_OUT_OF_MEMORY",
        "CUDA_ERROR",
    )


@pytest.mark.unit
def test_ptx_and_native_architectures_match_the_build_scripts():
    """The floors are derived from CMAKE_CUDA_ARCHITECTURES, not invented."""
    assert cc.MIN_COMPUTE_CAPABILITY == (5, 0)
    assert cc.NATIVE_SASS_ARCHITECTURES == frozenset({(8, 6), (8, 9), (12, 0)})
    assert cc.CUDA_RUNTIME_BASELINE == 12000
    assert cc.CUDA_PTX_DRIVER_FLOOR == 12080
