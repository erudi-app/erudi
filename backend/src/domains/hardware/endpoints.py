"""REST API endpoints for backend-agnostic hardware information.

Provides hardware detection, scoring, and monitoring endpoints across all backends
(MLX/Apple Silicon, CUDA/NVIDIA, CPU fallback). Uses discriminated unions and
transparent score boosting for optimal type safety and user experience.

Endpoints:
    GET /hardware/app_startup: Minimal UI data with boosted scores
    GET /hardware/detailed: Comprehensive diagnostics with raw/boosted scores
    POST /hardware/refresh: Force hardware re-detection

Architecture:
    Endpoints → Service → Repository → Entity → Database
"""

from fastapi import Depends, APIRouter, HTTPException
from sqlalchemy.orm import Session

from src.database.core import get_db
from src.domains.hardware.repository import Hardware_Repository
from src.domains.hardware.services import Hardware_Service
from src.domains.hardware.schemas import (
    HardwareAppStartupInfo,
    DetailedHardwareInfo,
    PerformanceBreakdown,
    MLXHardwareInfo,
    CUDAHardwareInfo,
    CPUHardwareInfo,
)
from src.core.logging import logger
from src.core.exceptions import HardwareException, DatabaseException
from src.entities.HardwareProfile import HardwareProfile

router = APIRouter(prefix="/hardware", tags=["hardware"])


def _get_service(db: Session) -> Hardware_Service:
    """Initialize service with repository dependency."""
    repository = Hardware_Repository(db)
    return Hardware_Service(repository)


def _build_backend_specific_schema(profile: HardwareProfile, scores: dict):
    """Build backend-specific schema instance based on backend_type.

    Args:
        profile: HardwareProfile entity
        scores: Dict with raw scores from service.calculate_boosted_scores()

    Returns:
        Union[MLXHardwareInfo, CUDAHardwareInfo, CPUHardwareInfo]
    """
    # Common base fields
    base_data = {
        "backend_type": profile.backend_type,
        "cpu_model": profile.cpu_model,
        "total_memory_gb": profile.total_memory_gb,
        "available_memory_gb": profile.available_memory_gb,
        "disk_total_gb": profile.disk_total_gb,
        "disk_available_gb": profile.disk_available_gb,
        "raw_inference_score": scores["raw_inference_score"],
        "global_inference_label": profile.global_inference_label,
        "cpu_score": scores["cpu_score"],
        "memory_score": scores["memory_score"],
        "architecture": profile.architecture,
        "system_platform": profile.system_platform,
    }

    if profile.backend_type == "mlx":
        return MLXHardwareInfo(
            **base_data,
            mlx_chip_model=profile.mlx_chip_model,
            mlx_gpu_cores=profile.mlx_gpu_cores or 0,
            mps_available=profile.mps_available or False,
            neural_engine_tops=profile.neural_engine_tops or 0.0,
            unified_memory=True,
            estimated_tflops=profile.estimated_tflops,
            memory_bandwidth_gbs=profile.memory_bandwidth_gbs,
            gpu_score=scores["gpu_score"],
        )
    elif profile.backend_type == "cuda":
        return CUDAHardwareInfo(
            **base_data,
            gpu_name=profile.gpu_name or "Unknown NVIDIA GPU",
            cuda_cores=profile.cuda_cores or 0,
            cuda_version=profile.cuda_version or "Unknown",
            compute_capability=profile.compute_capability or "Unknown",
            vram_total_gb=profile.vram_total_gb or 0.0,
            vram_available_gb=profile.vram_available_gb or 0.0,
            estimated_tflops=profile.estimated_tflops or 0.0,
            memory_bandwidth_gbs=profile.memory_bandwidth_gbs,
            gpu_score=scores["gpu_score"],
            unified_memory=False,
        )
    else:  # cpu
        # The entity has no compute_units column — the CPU count lives in
        # cpu_performance_units (a Float; the schema wants an int).
        cpu_units = int(profile.cpu_performance_units or 1)
        return CPUHardwareInfo(
            **base_data,
            compute_units=cpu_units,
            cpu_performance_units=cpu_units,
            accelerator_available=False,
            gpu_score=0.0,
            unified_memory=False,
        )


def _build_performance_breakdown(profile: HardwareProfile) -> PerformanceBreakdown:
    """Build typed PerformanceBreakdown from entity's JSON field."""
    pb = profile.performance_breakdown or {}
    return PerformanceBreakdown(
        compute_score=pb.get("compute_score", 0.0),
        memory_bandwidth_score=pb.get("memory_bandwidth_score", 0.0),
        memory_capacity_score=pb.get("memory_capacity_score", 0.0),
        cpu_performance_score=pb.get("cpu_performance_score", 0.0),
        disk_score=pb.get("disk_score"),
    )


@router.get("/app_startup", response_model=HardwareAppStartupInfo)
def get_app_startup_info(db: Session = Depends(get_db)):
    """Get minimal performance scores for application startup dashboard.

    Returns boosted scores (+20 points, capped at 100) for user-friendly display
    alongside raw scores for transparency. This endpoint is optimized for quick
    loading on app startup.

    Response structure:
        {
            "backend_type": "cuda",
            "global_inference_score": 100.0,     # Boosted, capped at 100
            "global_inference_label": "Excellent",
            "raw_inference_score": 82.0          # Actual hardware score
        }

    Raises:
        HTTPException: 500 if hardware detection fails
    """
    try:
        service = _get_service(db)
        profile = service.get_or_create_profile()
        scores = service.calculate_boosted_scores(profile)

        response = HardwareAppStartupInfo(
            backend_type=profile.backend_type,
            global_inference_score=scores["boosted_inference_score"],
            global_inference_label=scores["global_inference_label"],
            raw_inference_score=scores["raw_inference_score"],
            recommended_param_min=scores["recommended_param_min"],
            recommended_param_max=scores["recommended_param_max"],
        )

        db.commit()
        logger.info(
            f"App startup info retrieved: backend={profile.backend_type}, "
            f"boosted_inf={scores['boosted_inference_score']:.1f}, "
            f"raw_inf={scores['raw_inference_score']:.1f}"
        )
        return response

    except (HardwareException, DatabaseException) as e:
        # Already logged with its traceback where it was raised (services /
        # repository); a second record here would show one failure twice.
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        db.rollback()
        logger.exception(f"Unexpected error in get_app_startup_info: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/detailed", response_model=DetailedHardwareInfo)
def get_detailed_hardware_info(db: Session = Depends(get_db)):
    """Get comprehensive hardware diagnostics with raw/boosted score comparison.

    Returns complete backend-specific hardware profile plus performance breakdown
    and both raw/boosted scores for debugging and diagnostics.

    Response structure:
        {
            "hardware": { ...full backend-specific fields... },
            "performance_breakdown": { ...typed scores... },
            "boosted_inference_score": 85.0
        }

    Note:
        Raw score available in hardware.raw_inference_score

    Raises:
        HTTPException: 500 if hardware detection fails
    """
    try:
        service = _get_service(db)
        profile = service.get_or_create_profile()
        scores = service.calculate_boosted_scores(profile)

        backend_schema = _build_backend_specific_schema(profile, scores)
        perf_breakdown = _build_performance_breakdown(profile)

        response = DetailedHardwareInfo(
            hardware=backend_schema,
            performance_breakdown=perf_breakdown,
            boosted_inference_score=scores["boosted_inference_score"],
        )

        db.commit()
        logger.info(f"Detailed hardware info retrieved: backend={profile.backend_type}")
        return response

    except (HardwareException, DatabaseException) as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        db.rollback()
        logger.exception(f"Unexpected error in get_detailed_hardware_info: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/refresh")
def refresh_hardware_profile(db: Session = Depends(get_db)):
    """Force hardware re-detection and update cached profile.

    Performs fresh hardware detection through engine and updates database profile.
    Useful for detecting hardware changes (RAM upgrade, new GPU) or refreshing
    dynamic fields (available_memory_gb, disk_available_gb) without app restart.

    Returns:
        dict: Confirmation message with backend_type

    Example response:
        {
            "message": "Hardware profile refreshed successfully",
            "backend_type": "mlx"
        }

    Raises:
        HTTPException: 500 if refresh fails
    """
    try:
        service = _get_service(db)
        profile = service.refresh_profile()
        db.commit()

        logger.info(f"Hardware profile refreshed: backend={profile.backend_type}")
        return {
            "message": "Hardware profile refreshed successfully",
            "backend_type": profile.backend_type,
        }

    except (HardwareException, DatabaseException) as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        db.rollback()
        logger.exception(f"Unexpected error in refresh_hardware_profile: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
