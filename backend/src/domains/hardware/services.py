"""Business logic layer for hardware domain.

Implements Service pattern for hardware operations. Handles hardware detection,
profile management, score calculation, and labeling logic.

Architecture:
    Endpoints → Service → Repository → Entity → Database

Key Responsibilities:
    - Orchestrate hardware detection through LLM_Engine
    - Manage hardware profile lifecycle (get_or_create pattern)
    - Calculate boosted scores for UI display (+20 points)
    - Generate performance labels (Excellent, Good, Fair, Poor, Weak)
    - Format performance_breakdown structure

Example:
    from src.domains.hardware.services import Hardware_Service
    from src.domains.hardware.repository import Hardware_Repository
    from src.database.core import SessionLocal

    db = SessionLocal()
    repo = Hardware_Repository(db)
    service = Hardware_Service(repo)

    profile = service.get_or_create_profile()
    boosted = service.calculate_boosted_scores(profile)
    db.commit()
"""

from typing import Dict, Any, Optional
from src.domains.hardware.repository import Hardware_Repository
from src.entities.HardwareProfile import HardwareProfile
from src.core.logging import logger
from src.core.exceptions import HardwareException
from src.core import config


# Revision of the hardware-profiling logic (detection + scoring). The profile is
# written once at first boot and read forever after, so any fix to either is
# invisible on a machine that already has a row: #365 corrected a 448 GB/s card
# profiled at 13 GB/s, and the corrected code still served the stored 13 until
# the profile was recomputed by hand. Nothing in the app offered that -- the only
# path was Clear All Data, which also destroys the user's models, conversations
# and knowledge bases.
#
# BUMP THIS whenever detection or scoring changes in a way that should reach
# existing installs. Startup re-profiles when the stored value differs.
#
#   1 -- everything up to and including #365 (rated clocks, breakdown key names)
PROFILING_LOGIC_VERSION = 1


# Recommended model size window (billions of params) — drives the hardware-fit
# "Models For You" filter in the UI (#86). Two INDEPENDENT physical constraints,
# whichever binds first (#199 Part 2):
#
#   capacity  — can the weights fit in memory at all?
#   bandwidth — will generation be fast enough to be usable?
#
# Both matter and they fail differently. A score-tier lookup collapsed them into
# one opaque 0-100 bucket, so a 16GB card and a 48GB card both landed on 7-12B.
# Capacity alone is no better: two 16GB cards at 200 and 900 GB/s fit the same
# model and generate at wildly different speeds.
#
# Quantized (4-bit) footprint: ~0.6 GB per billion params, matching the same
# ratio the frontend uses for per-model fit (hardwareFit.js estimateFootprintGb).
_GB_PER_BILLION_PARAMS_Q4 = 0.6
# KV cache and runtime overhead for a modest context window.
_INFERENCE_OVERHEAD_RESERVE_GB = 1.5

# Unified/system memory is shared with the OS and other apps, unlike dedicated
# VRAM. The share the OS takes is roughly FIXED, not proportional, so a pure
# fraction over-promises badly on small machines: 0.75 x 16GB suggests ~17B on a
# 16GB Mac where 12B measurably lags. Fraction for headroom pressure, then a flat
# reserve for the OS itself.
_MLX_UNIFIED_MEMORY_USABLE_FRACTION = 0.75
_MLX_OS_RESERVE_GB = 4.0
_CPU_RAM_USABLE_FRACTION = 0.5

# Token generation is memory-bandwidth-bound: every token streams the whole
# weight set, so tok/s ~= bandwidth_gbs / (GB_PER_B x params). Inverting at a
# comfort floor gives the largest model that still generates at a usable speed.
# 20 tok/s is comfortable reading pace. (TFLOPs govern prompt processing rather
# than generation, and are unset on CPU profiles, so bandwidth is the honest
# proxy here.)
_COMFORT_FLOOR_TOKENS_PER_SEC = 20.0

_MIN_PARAM_FLOOR = 1.0
_MAX_PARAM_CEILING = 120.0
# Keep the excellent 7-14B models in "Models For You" on every machine: an
# uncapped max/2 floor pushes a 48GB box to a 38B minimum and empties the row.
_MAX_RECOMMENDED_MIN_PARAMS = 8.0


def _usable_inference_memory_gb(profile: HardwareProfile) -> float:
    """Memory actually available to load model weights into, per backend.

    CUDA: dedicated VRAM, not shared with the OS. MLX: unified memory less the
    OS's roughly fixed share. CPU: system RAM fraction, left unchanged (no
    validated baseline exists for it, and it is outside the reported regression).
    """
    if profile.backend_type == "cuda" and profile.vram_total_gb:
        return profile.vram_total_gb
    if profile.unified_memory and profile.total_memory_gb:
        usable = profile.total_memory_gb * _MLX_UNIFIED_MEMORY_USABLE_FRACTION
        return max(0.0, usable - _MLX_OS_RESERVE_GB)
    return (profile.total_memory_gb or 0.0) * _CPU_RAM_USABLE_FRACTION


def _bandwidth_param_cap(profile: HardwareProfile) -> Optional[float]:
    """Largest model (billions of params) that still generates at the comfort
    floor on this machine's memory bandwidth.

    Returns None when bandwidth is unknown (it is nullable, and CPU fallback
    paths leave it unset) so the caller degrades to capacity-only rather than
    capping the window to zero.
    """
    try:
        bandwidth = float(getattr(profile, "memory_bandwidth_gbs", None) or 0.0)
    except (TypeError, ValueError):
        return None  # unreadable value: degrade to capacity-only, never raise
    if bandwidth <= 0:
        return None
    return bandwidth / (_GB_PER_BILLION_PARAMS_Q4 * _COMFORT_FLOOR_TOKENS_PER_SEC)


def recommended_param_range(profile: HardwareProfile) -> tuple[float, float]:
    """Recommended model size window (billions of params) this machine runs
    comfortably at 4-bit: the smaller of what fits and what runs fast enough."""
    usable_gb = _usable_inference_memory_gb(profile)
    weight_budget_gb = max(0.0, usable_gb - _INFERENCE_OVERHEAD_RESERVE_GB)
    max_params = weight_budget_gb / _GB_PER_BILLION_PARAMS_Q4

    bandwidth_cap = _bandwidth_param_cap(profile)
    if bandwidth_cap is not None:
        max_params = min(max_params, bandwidth_cap)

    max_params = max(_MIN_PARAM_FLOOR, min(max_params, _MAX_PARAM_CEILING))
    min_params = min(max_params * 0.5, _MAX_RECOMMENDED_MIN_PARAMS)
    min_params = max(_MIN_PARAM_FLOOR, min(min_params, max_params))
    return (round(min_params, 1), round(max_params, 1))


class Hardware_Service:
    """Service layer for hardware domain business logic.

    Orchestrates hardware detection, profile management, and score calculations.
    Separates business logic from data access and API concerns.

    Attributes:
        repository: Hardware_Repository for data access operations.

    Note:
        Service methods do NOT commit. Commits are handled at endpoint level.
        Uses repository for all database operations.

    Example:
        >>> service = Hardware_Service(Hardware_Repository(db))
        >>> profile = service.get_or_create_profile()
        >>> boosted = service.calculate_boosted_scores(profile)
        >>> print(boosted["global_inference_score"])  # +20 boost
        85.0
    """

    def __init__(self, repository: Hardware_Repository):
        """Initialize service with repository.

        Args:
            repository: Hardware_Repository instance for data access.
        """
        self.repository = repository
        logger.debug("Initializing Hardware_Service")

    def get_or_create_profile(self) -> HardwareProfile:
        """Get cached hardware profile or detect new one.

        Retrieves existing hardware profile from database. If none exists or
        if the cached profile backend doesn't match the current engine,
        performs hardware detection and creates new profile.

        Returns:
            HardwareProfile: Existing or newly created profile.

        Raises:
            HardwareException: If hardware detection or profile creation fails.

        Note:
            Does NOT commit. Caller must commit transaction.

        Example:
            >>> service = Hardware_Service(repo)
            >>> profile = service.get_or_create_profile()
            >>> print(profile.backend_type)  # "mlx", "cuda", or "cpu"
        """
        try:
            logger.info("Getting or creating hardware profile")

            # Try to get existing profile
            profile = self.repository.get_profile()

            # Extract backend type from current engine name
            current_backend = config.LLM_Engine.__name__.lower().replace("_engine", "")

            if profile:
                stored_version = getattr(profile, "profiling_version", None)
                if profile.backend_type != current_backend:
                    # No profile exists or backend mismatch, detect hardware
                    logger.info(
                        f"Backend mismatch: cached={profile.backend_type}, current={current_backend}. Re-detecting."
                    )
                    self.repository.delete_profile(profile)
                elif stored_version != PROFILING_LOGIC_VERSION:
                    # Same machine, but the numbers were produced by profiling
                    # logic we have since corrected (#365). Redo them, otherwise
                    # the fix never reaches anyone who already ran the app.
                    logger.info(
                        f"Profiling logic changed: cached=v{stored_version}, "
                        f"current=v{PROFILING_LOGIC_VERSION}. Re-detecting."
                    )
                    self.repository.delete_profile(profile)
                else:
                    logger.info(f"Using cached hardware profile: backend={profile.backend_type}")
                    return profile
            else:
                logger.info("No cached profile found, detecting hardware")

            hardware_data = self._detect_hardware()

            # Create new profile
            profile = self.repository.create_profile(hardware_data)
            logger.info(f"Hardware profile created: backend={profile.backend_type}")

            return profile

        except HardwareException:
            raise  # logged where it was raised
        except Exception as e:
            logger.exception(f"Failed to get or create hardware profile: {e}")
            raise HardwareException("Failed to retrieve hardware information", trace=str(e))

    def _detect_hardware(self) -> Dict[str, Any]:
        """Perform hardware detection through LLM_Engine.

        Calls engine's get_flat_hardware_data() method to retrieve hardware
        specifications in flat format ready for HardwareProfile entity creation.

        Returns:
            Dict[str, Any]: Flat hardware data ready for entity creation.

        Raises:
            HardwareException: If hardware detection fails.

        Note:
            Private method. Use get_or_create_profile() from endpoints.
        """
        try:
            logger.debug("Starting hardware detection via LLM_Engine")

            if not config.LLM_Engine:
                raise HardwareException("LLM_Engine not initialized")

            # Get flat hardware data from engine
            hardware_data = config.LLM_Engine.get_flat_hardware_data()
            # Stamp the logic that produced these numbers, so a later fix to
            # detection or scoring can tell this row is stale and redo it.
            hardware_data["profiling_version"] = PROFILING_LOGIC_VERSION

            logger.debug(
                f"Hardware detection complete: backend={hardware_data.get('backend_type')}"
            )
            return hardware_data

        except Exception as e:
            logger.exception(f"Hardware detection failed: {e}")
            raise HardwareException("Failed to detect hardware", trace=str(e))

    def warm_up(self, duration_seconds: int = 5) -> bool:
        """Warm up hardware accelerator (GPU/MPS/CUDA).

        Delegates to LLM_Engine's warm_up_accelerator() method. Useful before
        benchmarking or first inference to ensure accurate performance.

        Args:
            duration_seconds: How long to run warm-up routine.

        Returns:
            bool: True if warm-up succeeded, False otherwise.

        Raises:
            HardwareException: If warm-up fails critically.

        Example:
            >>> service = Hardware_Service(repo)
            >>> success = service.warm_up(duration_seconds=5)
            >>> if success:
            ...     print("Accelerator ready")
        """
        try:
            logger.info(f"Starting hardware warm-up: duration={duration_seconds}s")

            if not config.LLM_Engine:
                logger.warning("LLM_Engine not initialized, skipping warm-up")
                return False

            success = config.LLM_Engine.warm_up_accelerator(duration_seconds)

            if success:
                logger.info("Hardware warm-up completed successfully")
            else:
                logger.warning("Hardware warm-up failed or not available")

            return success

        except Exception as e:
            logger.exception(f"Hardware warm-up failed: {e}")
            raise HardwareException("Failed to warm up hardware accelerator", trace=str(e))

    def calculate_boosted_scores(self, profile: HardwareProfile) -> Dict[str, Any]:
        """Calculate UI-friendly scores with +20 boost for frontend display.

        Returns both raw scores (actual hardware capability) and boosted scores
        (UI-friendly display with +20 point boost, capped at 100). This makes
        the boost transparent while maintaining user-friendly presentation.

        Args:
            profile: HardwareProfile entity with original scores.

        Returns:
            Dict[str, Any]: Dictionary with both raw and boosted scores/labels.

        Example:
            >>> profile = service.get_or_create_profile()
            >>> scores = service.calculate_boosted_scores(profile)
            >>> print(scores)
            {
                "raw_inference_score": 65.0,
                "boosted_inference_score": 85.0,  # min(65 + 20, 100)
                "global_inference_label": "Excellent",  # Based on boosted
                ...
            }
        """
        logger.debug("Calculating raw and boosted scores")

        # Raw score (actual hardware capability)
        raw_inf = profile.global_inference_score

        # Boosted score for UI (+ 20 points, capped at 100)
        boosted_inf = min(100.0, raw_inf + 20.0)

        result = {
            # Raw score (engine output, no modification)
            "raw_inference_score": raw_inf,
            # Boosted score for UI display
            "boosted_inference_score": boosted_inf,
            "global_inference_score": boosted_inf,  # Alias for backward compat
            # Label based on boosted score
            "global_inference_label": self._get_label(boosted_inf),
            # Component scores (raw, no boost needed for internal metrics)
            "cpu_score": profile.cpu_score,
            "memory_score": profile.memory_score,
            "gpu_score": profile.gpu_score if profile.gpu_score is not None else 0.0,
        }

        # Hardware-fit model size window (billions of params) for the UI's
        # "Models For You" recommendations (#86), derived from real usable memory.
        param_min, param_max = recommended_param_range(profile)
        result["recommended_param_min"] = param_min
        result["recommended_param_max"] = param_max

        logger.debug(f"Scores calculated: raw_inf={raw_inf:.1f}, boosted_inf={boosted_inf:.1f}")
        return result

    def _get_label(self, score: float) -> str:
        """Convert numeric score to performance label.

        Applies thresholds to categorize performance:
        - 80-100: Excellent
        - 60-79:  Good
        - 40-59:  Fair
        - 20-39:  Poor
        - 0-19:   Weak

        Args:
            score: Numeric performance score (0-100).

        Returns:
            str: Performance label.

        Example:
            >>> service._get_label(85.0)
            'Excellent'
            >>> service._get_label(45.0)
            'Fair'
        """
        if score >= 80:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Fair"
        elif score >= 20:
            return "Poor"
        else:
            return "Weak"

    def refresh_profile(self) -> HardwareProfile:
        """Re-detect hardware and update existing profile.

        Performs fresh hardware detection and updates existing profile with
        new data. Useful for detecting hardware changes or refreshing dynamic
        fields (available_memory_gb, disk_available_gb).

        Returns:
            HardwareProfile: Updated profile entity.

        Raises:
            HardwareException: If refresh fails.

        Note:
            Does NOT commit. Caller must commit transaction.

        Example:
            >>> service = Hardware_Service(repo)
            >>> profile = service.refresh_profile()
            >>> db.commit()
        """
        try:
            logger.info("Refreshing hardware profile")

            # Detect current hardware state
            hardware_data = self._detect_hardware()

            # Get existing profile
            profile = self.repository.get_profile()
            if not profile:
                # No profile exists, create new
                logger.info("No profile to refresh, creating new")
                profile = self.repository.create_profile(hardware_data)
            else:
                # Update existing profile
                logger.info(f"Updating existing profile: id={profile.id}")
                profile = self.repository.update_profile(profile, hardware_data)

            logger.info("Hardware profile refreshed successfully")
            return profile

        except HardwareException:
            raise  # logged where it was raised
        except Exception as e:
            logger.exception(f"Failed to refresh hardware profile: {e}")
            raise HardwareException("Failed to refresh hardware profile", trace=str(e))
