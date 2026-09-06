"""Abstract base engine for multi-backend LLM inference.

This module provides the foundation for Erudi's multi-engine architecture,
supporting MLX (Apple Silicon), CUDA (NVIDIA GPUs), and CPU fallback.
The engine abstraction enables consistent model loading, quantization,
and streaming inference across different hardware backends.

Fonctionnalités:
- Automatic engine selection based on platform and hardware
- Singleton pattern with shared model state across requests
- Automatic memory cleanup after idle timeout
- Streaming token generation API
- Engine-format gate (FORMAT_TAG): only pre-built artefacts are ever loaded

Architecture:
    BaseEngine (ABC)
    ├── MLX_Engine (Apple Silicon)
    ├── CUDA_Engine (NVIDIA GPUs)
    └── CPU_Engine (Fallback)

Examples:
    Get appropriate engine for current platform:
        from src.engines.base_engine import BaseEngine
        engine_class = BaseEngine.get_engine()
        model, tokenizer = engine_class.get_model_and_tokenizer(
            llm_id="1",
            llm_local_path="backend/data/models/llama-7b"
        )

    Stream tokens (driven by the agent layer, not the engine):
        the model handle's ``base_url`` is consumed by ``ChatOpenAI`` in
        ``src.agents.model_factory``; the engine only spawns / probes /
        reaps the OpenAI-compatible server.

"""

import asyncio
import platform
import os
import time
from datetime import datetime, timedelta
from typing import Any, ClassVar, Optional, Tuple, Union, Type, Dict
from abc import ABC, abstractmethod, ABCMeta
from contextlib import asynccontextmanager
from pathlib import Path

from src.core.exceptions import EngineException
from src.core.logging import logger


class EngineMeta(ABCMeta):
    """Metaclass for Engine classes with custom repr."""

    def __repr__(cls):
        """Return human-readable engine class name.

        Returns:
            String representation like "LLM Engine: MLX_Engine".

        """
        return f"LLM Engine: {cls.__name__}"


# Exception class names pynvml raises on a machine that simply has no NVIDIA
# driver. Compared by name so the CPU build, which may not ship pynvml at all,
# never has to import it to know them.
_NO_NVIDIA_DRIVER_ERRORS = frozenset({"NVMLError_LibraryNotFound", "NVMLError_DriverNotLoaded"})


def _log_monitor_death(task: "asyncio.Task") -> None:
    """Done callback of the idle-cleanup task: a death nobody awaits is logged."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(f"Idle cleanup monitor died: {exc}", exc_info=exc)


class BaseEngine(ABC, metaclass=EngineMeta):
    """Abstract base class for all LLM inference engines.

    Implements singleton pattern with shared model state, automatic cleanup,
    and platform-aware engine selection. Subclasses must implement backend-
    specific model loading and inference methods.

    Class Attributes:
        _model: Currently loaded model instance (shared across all engines).
        _tokenizer: Currently loaded tokenizer instance.
        _model_id: ID of the currently loaded model.
        _last_used: Timestamp of last model access for cleanup tracking.
        _cleanup_task: Async task monitoring idle time.
        _max_idle_time: Seconds before automatic memory cleanup (default: 300).
        MODEL_MAPPING: Dictionary mapping model architectures to classes.

    Note:
        Do not instantiate directly. Use get_engine() to obtain the appropriate
        engine class, then call class methods for operations.

    """

    # --- Global shared state (unique across all engines) ---
    _model: Optional[Any] = None
    _tokenizer: Optional[Any] = None
    _model_id: Optional[int] = None
    _last_used: Optional[datetime] = None

    # Serializes a full generation (model resolution + stream) on the
    # single-model engine. Lazily created and rebound per running loop so each
    # test (own loop) stays isolated; production has one loop -> one lock.
    _generation_lock = None
    _generation_lock_loop = None

    # --- Lifecycle management ---
    _cleanup_task = None
    _max_idle_time = 300  # 5 min

    # HF library tag identifying THIS engine's runnable format ("mlx" / "gguf").
    # The catalog is built ONLY from repos carrying this tag (searched via
    # filter=FORMAT_TAG), so every catalog entry is runnable by construction —
    # no hand-maintained allowlist. Set per engine family.
    FORMAT_TAG = None

    @classmethod
    def max_context_tokens(cls) -> Optional[int]:
        """The engine-imposed context window in tokens, or ``None`` when the engine
        sets none (MLX). llama.cpp engines spawn with a fixed ``-c`` and override
        this; the per-model sampling resolver (#388) caps ``max_tokens`` on it."""
        return None

    # Stored model links that download fine but FAIL TO RUN on this engine
    # (e.g. a quantized checkpoint the loader can't read). Overridden per engine.
    # is_runnable() uses it to ban such models from the catalog for this hardware.
    KNOWN_BROKEN: frozenset = frozenset()

    @classmethod
    def community_search_kwargs(cls, term: str) -> dict:
        """`HfApi.list_models` kwargs to find models in THIS engine's runnable
        format across ALL of HF (any author), so community fine-tunes surface and
        are runnable by construction. Searches by the format TAG, not by org —
        e.g. ``filter="mlx"`` catches non-mlx-community quants the org never holds.
        An empty ``term`` is the global pass: filter by format only, ranked by
        downloads.

        ``expand`` pulls pipeline_tag + tags (alongside the downloads/likes the
        quality floor needs) in the one call, so the non-chat task filter (#242)
        can gate ASR/embedding/OCR/media-gen repos out of the derived catalog,
        plus ``gated`` so a repo the token-less app cannot download is dropped
        (``model_resolver.is_gated``).
        """
        kwargs = {
            "filter": cls.FORMAT_TAG,
            "expand": ["pipeline_tag", "tags", "downloads", "likes", "gated"],
        }
        if term:
            kwargs["search"] = term
        return kwargs

    @classmethod
    def is_runnable(cls, link: str) -> bool:
        """Whether a catalog model (already engine-format by construction, since it
        came from a filter=FORMAT_TAG search) can actually run. The only exclusion
        is KNOWN_BROKEN — quants that load-crash on this engine."""
        return link not in cls.KNOWN_BROKEN

    def __init__(self):
        """Prevent direct instantiation.

        Raises:
            RuntimeError: Always raised as engines use class methods only.

        """
        raise RuntimeError("Use the methods instead of instantiating")

    @classmethod
    def __repr__(cls):
        return f"LLM Engine: {cls.__name__}"

    # ======================= COMMON API CONTRACT =======================
    @classmethod
    @abstractmethod
    def get_model_and_tokenizer(
        cls, llm_id: str, llm_local_path: Union[str, Path], *args
    ) -> Tuple[Any, Any]:
        """Load or retrieve cached model and tokenizer for inference.

        Implements singleton pattern: returns cached model if already loaded
        for the given llm_id, otherwise loads from disk. Thread-safe.

        Args:
            llm_id: Unique identifier for the model (used for caching).
            llm_local_path: Path to the model directory on disk.
            *args: Engine-specific loading arguments (e.g., device, dtype).

        Returns:
            Tuple of (model, tokenizer) ready for inference.

        Raises:
            EngineException: If model loading fails.
            FileNotFoundError: If model path doesn't exist.

        Examples:
            from src.engines.base_engine import BaseEngine
            engine = BaseEngine.get_engine()
            model, tokenizer = engine.get_model_and_tokenizer(
                llm_id="1",
                llm_local_path="backend/data/models/llama-7b"
            )

        """
        pass

    # ======================= TOOL-CALLING CAPABILITY =======================
    @classmethod
    def _load_capability_tokenizer(cls, local_path: Union[str, Path]) -> Any:
        """Load a tokenizer for static tool-calling detection (engine-specific).

        Returns a tokenizer-like object exposing ``apply_chat_template``, or
        raises if the artifact cannot be read. Overridden per engine: MLX reads
        the HuggingFace directory, llama.cpp reads the GGUF via ``gguf_file=``.
        ``compute_supports_tools`` treats any failure here as "not tool-capable".
        """
        raise NotImplementedError

    @classmethod
    def compute_supports_tools(cls, local_path: Union[str, Path]) -> bool:
        """Static tool-calling capability of the model at ``local_path``.

        Computed once at post-download time from the model's chat template by
        differential rendering (see ``engines.tool_capability``): no inference,
        no runtime probe. Any failure (unreadable artifact, missing template)
        returns ``False`` so the model routes through the systematic KB path —
        the capability is never assumed.
        """
        from src.engines.tool_capability import tokenizer_declares_tools

        try:
            tokenizer = cls._load_capability_tokenizer(local_path)
        except Exception:
            # exc_info matters: a missing dependency in the frozen build (the
            # gguf package, #171) is indistinguishable from an unreadable model
            # without the underlying exception in the log.
            logger.warning(
                f"[{cls.__name__}] tool-calling detection: could not load a "
                f"tokenizer for {local_path}; treating as not tool-capable",
                exc_info=True,
            )
            return False
        if tokenizer is None:
            return False
        return tokenizer_declares_tools(tokenizer)

    @classmethod
    def compute_wire_tools(cls, local_path: Union[str, Path]) -> Optional[bool]:
        """Verified tool-call WIRE capability of the model at ``local_path`` (#298).

        ``compute_supports_tools`` says the chat template DECLARES tools; this
        says the engine's local server actually parses the model's tool-call
        output into a structured call the agent executes — a per-model wire
        property (#273 matrix): a template matching no parser streams the call
        as raw text (swallowed answer, #295) or leaks JSON to the user.

        Engine-specific: MLX runs mlx-vlm's own parser inference on the chat
        template; llama.cpp engines trust the ``--jinja`` generic fallback for
        any usable template. The base default is ``None`` (unverified) so an
        engine without an implementation never routes a KB turn agentic.
        """
        return None

    @classmethod
    def model_supports_vision(cls, local_path: Union[str, Path]) -> Optional[bool]:
        """Whether the model at ``local_path`` accepts image input (#133).

        Deterministic, no model load: llama.cpp engines look for an ``mmproj``
        projector, MLX reads ``config.json``. Returns ``None`` when the engine
        cannot decide; the runtime treats ``None`` as not vision-capable (#212):
        images are stripped unless the capability is an explicit ``True``,
        and the user is notified when the current turn carried an image.
        """
        return None

    # ======================= HARDWARE DETECTION & EVALUATION =======================
    @classmethod
    @abstractmethod
    def get_hardware_info(cls) -> Dict[str, Any]:
        """Get comprehensive hardware information for this backend.

        Returns detailed hardware specifications including CPU, GPU/accelerator,
        memory, and platform-specific details. Implementation varies by backend:
        - MLX: Apple Silicon chip model, unified memory, MPS availability
        - CUDA: NVIDIA GPU model, CUDA cores, separate VRAM
        - CPU: Processor model, core count, system RAM

        Returns:
            Dict containing hardware specifications with the following structure:
            {
                "system": {
                    "platform": str,  # "Darwin", "Linux", "Windows"
                    "platform_version": Optional[str],
                    "machine": str,  # "arm64", "x86_64", etc.
                    "processor": str
                },
                "cpu": {
                    "model": str,
                    "architecture": str,
                    "total_cores": int,
                    "logical_cores": int,
                    "is_apple_silicon": bool,  # MLX only
                    "performance_cores": Optional[int],  # MLX only
                    "efficiency_cores": Optional[int],  # MLX only
                },
                "memory": {
                    "total_memory_gb": float,
                    "available_memory_gb": float,
                    "memory_pressure": float,  # 0.0-1.0
                    "memory_type": str,  # "unified" (MLX) or "system" (CPU/CUDA)
                },
                "gpu": {
                    "gpu_name": str,
                    "gpu_cores": Optional[int],  # MLX: GPU cores, CUDA: CUDA cores
                    "memory_bandwidth_gbs": Optional[float],
                    "vram_total_gb": Optional[float],  # CUDA only
                    "vram_available_gb": Optional[float],  # CUDA only
                    "compute_capability": Optional[str],  # CUDA only
                    "cuda_version": Optional[str],  # CUDA only
                    "mps_supported": Optional[bool],  # MLX only
                    "unified_memory": bool,  # True for MLX, False for CUDA/CPU
                },
                "accelerator": {  # MLX only
                    "neural_engine_tops": Optional[float],
                    "architecture": Optional[str],  # "3nm", "5nm", etc.
                },
                "storage": {
                    "total_gb": float,
                    "available_gb": float,
                    "usage_percentage": float
                },
                "backend_type": str,  # "mlx", "cuda", or "cpu"
                "timestamp": float
            }

        Raises:
            EngineException: If hardware detection fails critically.

        Note:
            Implementation should handle errors gracefully and return fallback
            values rather than raising exceptions for non-critical failures.

        Examples:
            >>> engine = BaseEngine.get_engine()
            >>> hw_info = engine.get_hardware_info()
            >>> print(f"GPU: {hw_info['gpu']['gpu_name']}")
            >>> print(f"Total Memory: {hw_info['memory']['total_memory_gb']} GB")

        """
        pass

    @classmethod
    @abstractmethod
    def warm_up_accelerator(cls, duration_seconds: float = 1.0) -> bool:
        """Warm up the hardware accelerator (GPU/Neural Engine) for optimal performance.

        Runs compute-intensive operations to bring the accelerator to optimal
        performance state before benchmarking or inference. Implementation varies:
        - MLX: Matrix operations on MPS device
        - CUDA: CUDA kernel warm-up on GPU
        - CPU: CPU cache warm-up (minimal effect)

        Args:
            duration_seconds: How long to run warm-up operations (default: 1.0).

        Returns:
            bool: True if warm-up completed successfully, False otherwise.

        Raises:
            EngineException: If warm-up fails critically (rare, usually returns False).

        Note:
            This is particularly important for GPUs that dynamically adjust clocks.
            MLX/MPS benefits significantly from warm-up due to power management.
            CPU backend may implement minimal warm-up or skip entirely.

        Examples:
            >>> engine = BaseEngine.get_engine()
            >>> success = engine.warm_up_accelerator(1.5)
            >>> if success:
            ...     print("Accelerator ready for benchmarking")

        """
        pass

    @classmethod
    @abstractmethod
    def get_performance_evaluation(cls) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics and scores for this backend.

        Evaluates hardware capabilities and returns performance scores for
        inference workloads. Scoring methodology varies by backend but results
        are normalized to 0-100 scale for cross-platform comparison.

        Scoring Components:
            - **Inference Score**: Optimized for generation speed and latency.
              Weights: GPU/accelerator compute (35-60%), memory bandwidth (20-30%),
              memory capacity (10-30%), CPU (5-10%).

        Returns:
            Dict containing performance metrics and scores:
            {
                # Hardware identification
                "backend_type": str,  # "mlx", "cuda", "cpu"
                "accelerator_name": str,  # GPU/chip model
                "cpu_model": str,

                # Memory metrics
                "total_memory_gb": float,
                "available_memory_gb": float,
                "memory_bandwidth_gbs": Optional[float],

                # Storage metrics
                "disk_total_gb": float,
                "disk_available_gb": float,

                # Compute metrics
                "estimated_tflops": Optional[float],  # GPU compute power
                "compute_units": Optional[int],  # GPU cores or CUDA cores
                "cpu_performance_units": float,

                # Backend-specific metrics
                "neural_engine_tops": Optional[float],  # MLX only
                "cuda_version": Optional[str],  # CUDA only
                "compute_capability": Optional[str],  # CUDA only
                "architecture": Optional[str],  # MLX: "3nm", CUDA: "Ampere"

                # Performance scores (0-100)
                "global_inference_score": float,
                "global_inference_label": str,  # "Very Good", "Good", "Medium", "Poor"
                "gpu_score": float,
                "cpu_score": float,
                "memory_score": float,

                # Technical details
                "unified_memory": bool,
                "accelerator_available": bool,
                "system_platform": str,

                # Performance breakdown for debugging
                "performance_breakdown": {
                    "compute_score": float,
                    "memory_bandwidth_score": float,
                    "memory_capacity_score": float,
                    "cpu_performance_score": float,
                    # Backend-specific breakdown
                }
            }

        Raises:
            EngineException: If evaluation fails critically.

        Note:
            Scores are platform-specific estimates based on hardware specs.
            Should call warm_up_accelerator() before evaluation for accuracy.
            Returns fallback scores if evaluation fails rather than raising.

        Examples:
            >>> engine = BaseEngine.get_engine()
            >>> engine.warm_up_accelerator(1.5)
            >>> eval_result = engine.get_performance_evaluation()
            >>> print(f"Inference: {eval_result['global_inference_score']}/100")

        """
        pass

    @classmethod
    @abstractmethod
    def get_flat_hardware_data(cls) -> Dict[str, Any]:
        """Get hardware data in flat format compatible with HardwareProfile entity.

        Returns hardware specifications as a flat dictionary ready for direct
        insertion into the HardwareProfile database entity. Combines and flattens
        data from get_hardware_info() and get_performance_evaluation().

        Returns:
            Dict with keys matching HardwareProfile columns:
            {
                # Common fields (all backends)
                "backend_type": str,  # "mlx", "cuda", "cpu"
                "cpu_model": str,
                "total_memory_gb": float,
                "available_memory_gb": float,
                "disk_total_gb": float,
                "disk_available_gb": float,
                "global_inference_score": float,
                "global_inference_label": str,
                "cpu_score": float,
                "memory_score": float,
                "gpu_score": float,
                "system_platform": str,
                "architecture": Optional[str],
                "estimated_tflops": Optional[float],
                "memory_bandwidth_gbs": Optional[float],
                "cpu_performance_units": Optional[float],
                "performance_breakdown": dict,

                # MLX-specific fields
                "mlx_chip_model": Optional[str],
                "mlx_gpu_cores": Optional[int],
                "mps_available": Optional[bool],
                "neural_engine_tops": Optional[float],
                "unified_memory": Optional[bool],
                "gpu_name": Optional[str],

                # CUDA-specific fields
                "cuda_cores": Optional[int],
                "cuda_version": Optional[str],
                "compute_capability": Optional[str],
                "vram_total_gb": Optional[float],
                "vram_available_gb": Optional[float],
            }

        Raises:
            EngineException: If hardware data collection fails.

        Note:
            This method provides a single point of access for hardware data
            in the format expected by the database layer, eliminating the need
            for manual flattening in service layers.

        Examples:
            >>> engine = BaseEngine.get_engine()
            >>> data = engine.get_flat_hardware_data()
            >>> profile = HardwareProfile(**data)  # Direct instantiation
            >>> db.add(profile)

        """
        pass

    # ======================= COMMON INFRASTRUCTURE =======================
    @classmethod
    def get_engine(cls) -> Type["BaseEngine"]:
        """Select and return appropriate engine class for current platform.

        Automatically detects OS and hardware to choose the optimal backend:
        - macOS ARM (M1/M2/M3): MLX_Engine
        - macOS Intel: CPU_Engine
        - Linux/Windows with CUDA: CUDA_Engine
        - Linux/Windows without CUDA: CPU_Engine

        Returns:
            The engine class (not an instance) appropriate for this system.

        Raises:
            EngineException: If no suitable engine can be determined.

        Examples:
            from src.engines.base_engine import BaseEngine
            engine_class = BaseEngine.get_engine()
            print(f"Using: {engine_class}")  # e.g., "LLM Engine: MLX_Engine"

        """
        from src.engines.mlx_engine import MLX_Engine
        from src.engines.cpu_engine import CPU_Engine
        from src.engines.cuda_engine import CUDA_Engine

        system = platform.system().lower()  # mac, linux, windows
        machine = platform.machine().lower()  # arm64, x86_64, etc.
        llm_engine = None

        try:
            # Testing override: ERUDI_FORCE_CPU=1 bypasses GPU detection entirely.
            # Use this to validate CPU fallback on a GPU machine without disabling hardware.
            if os.environ.get("ERUDI_FORCE_CPU"):
                logger.info("[ERUDI_FORCE_CPU] Forcing CPU_Engine - GPU detection skipped.")
                return CPU_Engine

            if system == "darwin":  # MacOS
                if "arm" in machine:
                    llm_engine = MLX_Engine
                elif "x86" in machine:
                    llm_engine = CPU_Engine
            elif system in ("linux", "windows"):
                cuda_present = False
                try:
                    import pynvml as nv

                    nv.nvmlInit()
                    cuda_present = nv.nvmlDeviceGetCount() > 0
                except ImportError:
                    # A CPU build ships without pynvml: expected, not a defect.
                    logger.info("pynvml is not installed; using CPU_Engine.")
                except Exception as e:
                    if type(e).__name__ in _NO_NVIDIA_DRIVER_ERRORS:
                        # No NVIDIA driver on this machine: the ordinary way a
                        # CPU-only PC boots, not a fallback worth a warning.
                        logger.info(f"No NVIDIA driver ({e}); using CPU_Engine.")
                    else:
                        # A driver that is present but broken. The user lands
                        # on the processor without being told why unless this
                        # record says so.
                        logger.warning(
                            f"NVIDIA detection failed ({type(e).__name__}: {e}); "
                            f"using CPU_Engine.",
                            exc_info=True,
                        )
                if cuda_present:
                    llm_engine = CUDA_Engine
                else:
                    llm_engine = CPU_Engine
                    logger.info(f"System: {system} and CUDA not available.")
            logger.info(f"Engine chosen: {llm_engine}")
            if llm_engine is None:
                raise RuntimeError(f"no engine matches system={system!r} machine={machine!r}")
            return llm_engine
        except Exception as e:
            raise EngineException(
                message="Error selecting the LLM Engine.",
                trace=f"{type(e).__name__}: {e}",
            )

    @classmethod
    def _should_cleanup(cls) -> bool:
        # Active-marker contract: `_last_used = None` means a generation is
        # in flight (set by `generation_guard`). Returning False here is what
        # blocks the idle monitor from reaping the model mid-generation.
        if cls._last_used is None or cls._model is None:
            return False
        idle_time = datetime.now() - cls._last_used
        return idle_time > timedelta(seconds=cls._max_idle_time)

    @classmethod
    def _should_not_reload_model(cls, llm_id: str) -> bool:
        return llm_id == cls._model_id and cls._model is not None and cls._tokenizer is not None

    @classmethod
    def _return_cached_model_and_tokenizer(cls) -> Tuple[Any, Any]:
        cls._last_used = datetime.now()
        logger.info(f"Using cached model {cls._model_id}")
        return cls._model, cls._tokenizer

    @classmethod
    def _generation_lock_for_running_loop(cls) -> "asyncio.Lock":
        """Return an ``asyncio.Lock`` bound to the current running loop.

        Recreated if the running loop changed so tests (each with their own
        loop) stay isolated; in production there is exactly one loop, so the
        lock is created once and shared across all engine classes.
        """
        loop = asyncio.get_running_loop()
        if BaseEngine._generation_lock is None or BaseEngine._generation_lock_loop is not loop:
            BaseEngine._generation_lock = asyncio.Lock()
            BaseEngine._generation_lock_loop = loop
        return BaseEngine._generation_lock

    @classmethod
    @asynccontextmanager
    async def generation_guard(cls):
        """Serialize a full generation and suppress idle cleanup for its duration.

        The agent layer wraps model resolution + the entire token stream in this
        guard so that:
          - concurrent requests for different models can't thrash the
            single-model engine subprocess (they serialize on one asyncio lock);
          - the idle-cleanup tick (``_cleanup_tick``) shares this SAME lock, so
            it cannot run — let alone reap the model — while a generation is in
            flight. It waits until the guard exits, by which point the idle
            clock has been refreshed, so no mid-generation teardown is possible.

        This is the engine-level home of the invariant that ``generate_stream``
        used to carry; it lives here (not in the agent layer) to keep subprocess
        lifecycle and concurrency inside the engine encapsulation.
        """
        lock = cls._generation_lock_for_running_loop()
        # Contention diagnostic: another generation (or the cleanup tick)
        # holds the lock right now — measure how long this request waits.
        wait_start_s = time.perf_counter()
        contended = lock.locked()
        async with lock:
            if contended:
                wait_ms = (time.perf_counter() - wait_start_s) * 1000
                logger.debug(
                    f"[{cls.__name__}] generation_guard waited {wait_ms:.0f}ms "
                    f"for the generation lock (model={cls._model_id})"
                )
            logger.info(f"[{cls.__name__}] generation_guard acquired (model={cls._model_id})")
            held_start_s = time.perf_counter()
            try:
                yield
            finally:
                # Restart the idle clock from the end of the generation.
                cls._last_used = datetime.now()
                held_ms = (time.perf_counter() - held_start_s) * 1000
                logger.info(
                    f"[{cls.__name__}] generation_guard released "
                    f"(model={cls._model_id}, held {held_ms:.0f}ms)"
                )

    @classmethod
    def cleanup(cls) -> None:
        """Free model and tokenizer from memory, reset state.

        Releases GPU/CPU memory occupied by the model and tokenizer,
        resets all cached state. Called automatically after idle timeout
        or manually when switching models.

        Note:
            Thread-safe. Can be called explicitly or by cleanup monitor.
            Calls garbage collector to ensure memory is freed.

        """
        if cls._model or cls._tokenizer:
            import gc

            logger.info(f"Cleaning up model {cls._model_id}")
            cls._model = cls._tokenizer = cls._model_id = cls._last_used = None
            gc.collect()

    @classmethod
    async def _cleanup_tick(cls):
        """One idle-check pass: reap the model iff it has been idle too long.

        Acquires the SAME asyncio lock as ``generation_guard`` (so it can never
        run during a generation) and runs the blocking ``cleanup()`` — which
        kills the child subprocess — off the event loop via ``asyncio.to_thread``.
        """
        async with cls._generation_lock_for_running_loop():
            if cls._should_cleanup():
                await asyncio.to_thread(cls.cleanup)

    # Seconds between idle-cleanup ticks. A class attribute so tests can run
    # the monitor loop at speed.
    _cleanup_interval_s: ClassVar[float] = 300.0

    @classmethod
    async def _cleanup_monitor(cls):
        """Background task monitoring idle time and triggering cleanup.

        Runs every ``_cleanup_interval_s`` seconds, checks if the model has
        been idle longer than ``_max_idle_time``, and calls ``cleanup()`` if
        the threshold is exceeded. A tick that raises is logged at ERROR with
        its traceback and the loop goes on: the task is awaited by nobody, so
        an exception that escaped it would end idle cleanup for the life of
        the process without a single record.

        Note:
            Internal method. Do not call directly. Use start_cleanup_task().

        """
        while True:
            await asyncio.sleep(cls._cleanup_interval_s)
            try:
                await cls._cleanup_tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    f"Idle cleanup tick failed; the model stays loaded: {exc}", exc_info=True
                )

    @classmethod
    def start_cleanup_task(cls):
        """Start the automatic cleanup monitoring task.

        Creates an async task that periodically checks for idle models
        and frees memory. Should be called once during application startup.

        Note:
            Idempotent - calling multiple times has no effect if task already running.

        Examples:
            from src.engines.base_engine import BaseEngine
            engine = BaseEngine.get_engine()
            engine.start_cleanup_task()

        """
        if cls._cleanup_task is None:
            cls._cleanup_task = asyncio.create_task(cls._cleanup_monitor(), name="idle-cleanup")
            cls._cleanup_task.add_done_callback(_log_monitor_death)
            logger.info("Started cleanup monitor")

    @classmethod
    def stop_cleanup_task(cls):
        """Stop the automatic cleanup monitoring task.

        Cancels the cleanup monitor task. Should be called during application
        shutdown to gracefully stop background tasks.

        Note:
            Idempotent - calling when no task is running has no effect.

        Examples:
            from src.engines.base_engine import BaseEngine
            engine = BaseEngine.get_engine()
            engine.stop_cleanup_task()

        """
        if cls._cleanup_task is not None:
            cls._cleanup_task.cancel()
            cls._cleanup_task = None
            logger.info("Stopped cleanup monitor")
