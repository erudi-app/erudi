"""Pydantic validation schemas for backend-agnostic hardware information.

Defines response schemas for hardware detection endpoints across all backends
(MLX/CUDA/CPU). Uses discriminated unions and inheritance to reduce duplication
while maintaining type safety for backend-specific fields.

Architecture:
    - BaseHardwareInfo: Common fields shared by all backends
    - MLXHardwareInfo/CUDAHardwareInfo/CPUHardwareInfo: Backend-specific extensions
    - PerformanceBreakdown: Typed performance metrics (replaces Dict[str, Any])
    - Endpoint-specific views: Training, AppStartup, Detailed

Example:
    Training endpoint returns full backend-specific schema::

        {
            "backend_type": "mlx",
            "mlx_chip_model": "M3 Max",
            "mlx_gpu_cores": 40,
            ...
        }

    App startup returns minimal UI data::

        {
            "backend_type": "cuda",
            "raw_inference_score": 82.5,
            "boosted_inference_score": 100.0,
            ...
        }
"""

from pydantic import BaseModel, Field
from typing import Optional, Union, Literal


# ======================= PERFORMANCE BREAKDOWN =======================


class PerformanceBreakdown(BaseModel):
    """Typed performance breakdown replacing Dict[str, Any].

    Scores are normalized 0-100 scale representing component contributions
    to overall performance.
    """

    compute_score: float = Field(..., description="GPU/CPU compute capability score (0-100)")
    memory_bandwidth_score: float = Field(..., description="Memory bandwidth score (0-100)")
    memory_capacity_score: float = Field(..., description="Memory capacity score (0-100)")
    cpu_performance_score: float = Field(..., description="CPU performance score (0-100)")
    disk_score: Optional[float] = Field(None, description="Disk storage score (0-100)")


# ======================= BASE SCHEMAS =======================


class BaseHardwareInfo(BaseModel):
    """Common fields shared by all backends.

    All engines must provide these fields. Backend-specific fields
    defined in subclasses.
    """

    backend_type: Literal["mlx", "cuda", "cpu"] = Field(..., description="Backend discriminator")

    # CPU/System
    cpu_model: str = Field(..., description="CPU model name")
    total_memory_gb: float = Field(..., description="Total system RAM in GB")
    available_memory_gb: float = Field(..., description="Available system RAM in GB")

    # Storage
    disk_total_gb: float = Field(..., description="Total disk space in GB")
    disk_available_gb: float = Field(..., description="Available disk space in GB")

    # Performance scores (raw, without UI boost)
    raw_inference_score: float = Field(..., description="Raw inference score (0-100)")
    global_inference_label: str = Field(..., description="Qualitative inference rating")

    # Component scores
    cpu_score: float = Field(..., description="CPU component score (0-100)")
    memory_score: float = Field(..., description="Memory component score (0-100)")

    # Optional common fields
    architecture: Optional[str] = Field(None, description="CPU/GPU architecture")
    system_platform: Optional[str] = Field(None, description="OS platform")


# ======================= BACKEND-SPECIFIC SCHEMAS =======================


class MLXHardwareInfo(BaseHardwareInfo):
    """Apple Silicon (MLX) specific hardware fields."""

    backend_type: Literal["mlx"] = "mlx"

    mlx_chip_model: str = Field(..., description="Apple chip model (e.g., M3 Max)")
    mlx_gpu_cores: int = Field(..., description="Number of GPU cores")
    mps_available: bool = Field(..., description="Metal Performance Shaders availability")
    neural_engine_tops: float = Field(..., description="Neural Engine TOPS")
    unified_memory: bool = Field(True, description="Unified memory architecture")
    estimated_tflops: Optional[float] = Field(None, description="Estimated GPU TFLOPS")
    memory_bandwidth_gbs: Optional[float] = Field(None, description="Memory bandwidth in GB/s")
    gpu_score: float = Field(..., description="GPU component score (0-100)")


class CUDAHardwareInfo(BaseHardwareInfo):
    """NVIDIA CUDA specific hardware fields."""

    backend_type: Literal["cuda"] = "cuda"

    gpu_name: str = Field(..., description="NVIDIA GPU model name")
    cuda_cores: int = Field(..., description="Number of CUDA cores")
    cuda_version: str = Field(..., description="CUDA runtime version")
    compute_capability: str = Field(..., description="GPU compute capability (e.g., 8.6)")
    vram_total_gb: float = Field(..., description="Total VRAM in GB")
    vram_available_gb: float = Field(..., description="Available VRAM in GB")
    estimated_tflops: float = Field(..., description="Estimated GPU TFLOPS")
    memory_bandwidth_gbs: Optional[float] = Field(None, description="Memory bandwidth in GB/s")
    gpu_score: float = Field(..., description="GPU component score (0-100)")
    unified_memory: bool = Field(False, description="Unified memory (always False for CUDA)")


class CPUHardwareInfo(BaseHardwareInfo):
    """CPU-only backend (fallback) hardware fields."""

    backend_type: Literal["cpu"] = "cpu"

    compute_units: int = Field(..., description="Number of CPU cores")
    cpu_performance_units: int = Field(..., description="Logical CPU cores")
    accelerator_available: bool = Field(False, description="GPU accelerator availability")
    gpu_score: float = Field(0.0, description="GPU score (always 0 for CPU)")
    unified_memory: bool = Field(False, description="Unified memory (always False for CPU)")


# ======================= ENDPOINT-SPECIFIC VIEWS =======================


class HardwareAppStartupInfo(BaseModel):
    """Essential performance metrics for application startup dashboard.

    Returns minimal UI data with transparent score boosting for user-friendly display.
    Raw scores preserved for debugging.
    """

    backend_type: Literal["mlx", "cuda", "cpu"]

    # Boosted score for UI display (raw + 20 points, capped at 100)
    global_inference_score: float = Field(..., description="UI-boosted inference score")
    global_inference_label: str

    # Raw score for comparison/debugging
    raw_inference_score: float = Field(..., description="Actual hardware score without boost")

    # Hardware-fit model size window (billions of params) — drives the UI's
    # "Models For You" recommendations (#86).
    recommended_param_min: float = Field(
        ..., description="Smallest recommended model size (B params)"
    )
    recommended_param_max: float = Field(
        ..., description="Largest recommended model size (B params)"
    )


class DetailedHardwareInfo(BaseModel):
    """Comprehensive hardware profile for debugging and diagnostics.

    Returns full backend-specific data plus performance breakdown and
    raw/boosted score comparison.
    """

    hardware: Union[MLXHardwareInfo, CUDAHardwareInfo, CPUHardwareInfo] = Field(
        ..., discriminator="backend_type"
    )
    performance_breakdown: PerformanceBreakdown

    # Include both raw and boosted for transparency
    boosted_inference_score: float = Field(..., description="UI-boosted inference score")


class GpuContentionStatus(BaseModel):
    """Whether another app is taking the graphics card from the loaded model
    now: the last cached reading of the CUDA engine's eviction gauge
    (Windows). Not contended, with no numbers, on every other engine and OS
    and without a loaded model."""

    active: bool = Field(..., description="Another app is taking the card from the model")
    evicted_bytes: Optional[int] = Field(
        None, description="The model's GPU memory resident nowhere, in bytes"
    )
    device_free_bytes: Optional[int] = Field(
        None, description="The card's free memory at that reading (NVML), in bytes"
    )
    read_at: Optional[float] = Field(None, description="When it was read (Unix time, seconds)")
