"""CPU engine for fallback inference using llama.cpp HTTP server.

This module provides a CPU-only backend for systems without GPU acceleration:
- Uses llama.cpp server (OpenAI-compatible API) for optimized CPU inference
- GGUF model format for quantization (4-bit, 5-bit, 8-bit)
- Threading optimization for multi-core CPUs
- Fallback when MLX (Mac Silicon) or CUDA (NVIDIA GPU) unavailable
- Rosetta support for x86_64 binaries on Apple Silicon

Current Status:
    FUNCTIONAL - Core features implemented. Performance evaluation pending refinement.

Features:
    - Load GGUF quantized models via llama-server subprocess
    - Multi-threaded inference with configurable thread count
    - Context window optimization (configurable via ERUDI_CTX)
    - Streaming generation via OpenAI-compatible /v1/chat/completions
    - Automatic port selection and server lifecycle management
    - Cross-platform (macOS, Linux, Windows) with Rosetta support

Architecture:
    CPU Engine (Singleton):
    ┌───────────────────────────────────────────────────────────┐
    │ get_model_and_tokenizer()                                 │
    │  └─> Start llama-server → return handle                  │
    └───────────────────────────────────────────────────────────┘
                            ↓
    ┌───────────────────────────────────────────────────────────┐
    │ token streaming lives in the agent layer, not the engine: │
    │   AgentRunner → ChatOpenAI(base_url) → POST /v1/chat/...  │
    │   the engine only spawns / probes / reaps the server      │
    └───────────────────────────────────────────────────────────┘
                            ↓
    ┌───────────────────────────────────────────────────────────┐
    │ cleanup()                                                 │
    │  └─> Terminate server process → free memory              │
    └───────────────────────────────────────────────────────────┘

Example:
    ::

        from src.engines.cpu_engine import CPU_Engine

        model, tokenizer = CPU_Engine.get_model_and_tokenizer(
            llm_id="mistral-7b-q4",
            llm_local_path="/path/to/model.gguf"
        )

        # Token streaming is driven by the agent layer (ChatOpenAI(base_url=...)),
        # not the engine; ``model["base_url"]`` is what it connects to.

Note:
    BaseEngine.get_engine() will automatically select CPU_Engine if:
    - System is not Mac Silicon (no MLX)
    - No CUDA-capable GPU detected (no CUDA_Engine)

    This ensures graceful degradation for unsupported hardware.

Warning:
    CPU inference is significantly slower than GPU (10-50x depending on model size).
    Use only as last resort fallback when GPU acceleration is unavailable.
    For production deployments, prefer MLX (Mac) or CUDA (NVIDIA) engines.
"""

# src/engines/cpu_engine.py
# CPU_Engine: llama.cpp HTTP server backend
# - Overrides only abstract methods from BaseEngine
# - Keeps common infra (singleton, cleanup monitor) untouched
# - OS-proof (Darwin/Windows/Linux) with guarded, minimal OS-specific code
# - Uses llama-server (OpenAI-compatible /v1/chat/completions)
# - No subprocess-per-request; one local server process per loaded model

from __future__ import annotations
import os
import platform
import time  # used by hardware warm-up loop
from pathlib import Path
from typing import Any, Dict, List

from src.engines.base_llama_cpp_engine import BaseLlamaCppEngine
from src.engines.cpu_brand import get_cpu_brand
from src.core.logging import logger


class CPU_Engine(BaseLlamaCppEngine):
    """
    CPU Engine using llama.cpp's HTTP server (llama-server).

    Inherits the subprocess + SSE lifecycle from BaseChatServerEngine and the
    llama-cpp specifics (Popen, GGUF picking, llama-server location) from
    BaseLlamaCppEngine. Only `_build_spawn_argv` + `_prepare_spawn_context`
    + class attrs are CPU-specific here.

    Notes:
      * "model" is a dict handle for the running server (pid, port, alias, base_url)
      * "tokenizer" is a placeholder (HTTP server abstracts tokenization)
    """

    # --- BaseChatServerEngine config overrides ---
    _server_name = "llama-server"
    _tokenizer_provider = "llama-server"
    # _port_range_start = 27200 (inherited from BaseLlamaCppEngine)
    # _use_cuda_build = False (inherited from BaseLlamaCppEngine)
    # USES_GGUF + MODEL_MAPPING (public-GGUF catalog) inherited from BaseLlamaCppEngine.

    # ---------- CPU-specific spawn hooks ----------

    @classmethod
    def _prepare_spawn_context(cls) -> Dict[str, Any]:
        """Resolve the per-spawn CPU context: context window, thread count,
        and the fixed `gpu_layers=0` (CPU-only inference)."""
        return {
            "ctx_size": cls.max_context_tokens(),
            "threads": max(1, os.cpu_count() or 1),
            "gpu_layers": 0,
        }

    @classmethod
    def _build_spawn_argv(
        cls,
        *,
        llama_server: Path,
        model_gguf: Path,
        alias: str,
        port: int,
        ctx_size: int = 4096,
        threads: int = 1,
        gpu_layers: int = 0,
        **_ignored: Any,
    ) -> List[Any]:
        """CPU CLI for llama-server: forces `-ngl 0`, sized context, native threads.

        ``--jinja`` enables the model's own chat template and with it
        OpenAI-style function calling (the agent's calculator tool) —
        without it llama-server never emits ``tool_calls``.

        ``--reasoning-format`` is deliberately NOT passed (#554): the default
        (``auto``) is llama-server's per-family reasoning extraction into the
        dedicated ``delta.reasoning_content`` stream field -- it knows each
        family's markers, re-injects a prompt-opened thinking block on every
        post-tool hop, and closes an unterminated block cleanly at EOS.
        ``Erudi_Chat_OpenAI`` (``src.agents.chat_model``) carries the field to
        the runner's ``thinking`` events; the runner's ThinkSplitter stays as
        the inline fallback for families the server parser does not know.
        """
        return [
            str(llama_server),
            "-m",
            str(model_gguf),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--alias",
            alias,
            "-c",
            str(ctx_size),
            "--threads",
            str(threads),
            "-ngl",
            str(gpu_layers),
            "--jinja",
        ]

    @classmethod
    def get_hardware_info(cls) -> Dict[str, Any]:
        """Get comprehensive hardware information for CPU-only backend.

        Returns detailed hardware specifications including CPU model, system memory,
        storage capacity, and platform information. No GPU/accelerator data.

        Returns:
            Dict containing hardware specifications following BaseEngine contract:
            {
                "system": {"platform": str, "platform_version": str, ...},
                "cpu": {"model": str, "total_cores": int, "architecture": str, ...},
                "memory": {"total_memory_gb": float, "available_memory_gb": float, ...},
                "storage": {"total_gb": float, "available_gb": float, "usage_percentage": float},
                "backend_type": "cpu",
                "timestamp": float
            }

        Note:
            GPU fields return None/False since this is CPU-only backend.
            Falls back gracefully if psutil unavailable.
        """
        try:
            # Import optional dependencies for hardware detection
            try:
                import psutil
            except ImportError as e:
                logger.warning(f"Optional hardware detection dependency missing: {e}")
                psutil = None

            # System information
            system = platform.system()
            machine = platform.machine()
            processor = platform.processor() or platform.uname().processor

            # CPU information
            total_cores = psutil.cpu_count(logical=False) if psutil else (os.cpu_count() or 1)
            logical_cores = psutil.cpu_count(logical=True) if psutil else (os.cpu_count() or 1)

            # Enhanced CPU model detection from OS-native sources (no py-cpuinfo
            # multiprocessing child; see src/engines/cpu_brand.py and #282)
            cpu_model = processor
            cpu_brand = get_cpu_brand()
            if cpu_brand:
                cpu_model = cpu_brand

            # Check if Apple Silicon (shouldn't normally reach CPU_Engine on Mac, but handle it)
            is_apple_silicon = system == "Darwin" and "arm" in machine.lower()

            # Memory information
            total_mem = avail_mem = memory_pressure = None
            if psutil:
                vm = psutil.virtual_memory()
                total_mem = round(vm.total / (1024**3), 2)
                avail_mem = round(vm.available / (1024**3), 2)
                memory_pressure = round(1.0 - (vm.available / vm.total), 3) if vm.total > 0 else 0.0

            # Storage information
            disk_total = disk_available = disk_usage_pct = None
            if psutil:
                try:
                    disk = psutil.disk_usage(os.path.abspath(os.sep))
                    disk_total = round(disk.total / (1024**3), 2)
                    disk_available = round(disk.free / (1024**3), 2)
                    disk_usage_pct = round(disk.percent, 2)
                except Exception as e:
                    logger.warning(f"Failed to get disk info: {e}")

            # Build hardware info dictionary
            info = {
                "system": {
                    "platform": system,
                    "platform_version": platform.version(),
                    "machine": machine,
                    "processor": processor,
                },
                "cpu": {
                    "model": cpu_model,
                    "architecture": machine,
                    "total_cores": total_cores,
                    "logical_cores": logical_cores,
                    "is_apple_silicon": is_apple_silicon,
                    "performance_cores": None,  # Not available without Apple Silicon API
                    "efficiency_cores": None,
                },
                "memory": {
                    "total_memory_gb": total_mem,
                    "available_memory_gb": avail_mem,
                    "memory_pressure": memory_pressure,
                    "memory_type": "system",
                },
                "gpu": {
                    "gpu_name": "CPU Only",
                    "gpu_cores": None,
                    "memory_bandwidth_gbs": None,
                    "vram_total_gb": None,
                    "vram_available_gb": None,
                    "compute_capability": None,
                    "cuda_version": None,
                    "mps_supported": False,
                    "unified_memory": False,
                },
                "accelerator": {
                    "neural_engine_tops": None,
                    "architecture": None,
                },
                "storage": {
                    "total_gb": disk_total,
                    "available_gb": disk_available,
                    "usage_percentage": disk_usage_pct,
                },
                "backend_type": "cpu",
                "timestamp": time.time(),
            }

            logger.info(
                f"CPU hardware detected: {cpu_model}, {total_cores} cores, {total_mem}GB RAM"
            )
            return info

        except Exception as e:
            logger.exception(f"CPU hardware detection failed: {e}")
            # Return minimal fallback info
            return {
                "system": {
                    "platform": platform.system(),
                    "platform_version": platform.version(),
                    "machine": platform.machine(),
                    "processor": "Unknown",
                },
                "cpu": {
                    "model": "Unknown CPU",
                    "architecture": platform.machine(),
                    "total_cores": os.cpu_count() or 1,
                    "logical_cores": os.cpu_count() or 1,
                    "is_apple_silicon": False,
                    "performance_cores": None,
                    "efficiency_cores": None,
                },
                "memory": {
                    "total_memory_gb": None,
                    "available_memory_gb": None,
                    "memory_pressure": None,
                    "memory_type": "system",
                },
                "gpu": {
                    "gpu_name": "CPU Only",
                    "gpu_cores": None,
                    "memory_bandwidth_gbs": None,
                    "vram_total_gb": None,
                    "vram_available_gb": None,
                    "compute_capability": None,
                    "cuda_version": None,
                    "mps_supported": False,
                    "unified_memory": False,
                },
                "accelerator": {"neural_engine_tops": None, "architecture": None},
                "storage": {"total_gb": None, "available_gb": None, "usage_percentage": None},
                "backend_type": "cpu",
                "timestamp": time.time(),
            }

    @classmethod
    def warm_up_accelerator(cls, duration_seconds: float = 1.0) -> bool:
        """Warm up CPU with compute workload.

        Runs lightweight computation loop to bring CPU to stable performance
        state. Less critical than GPU warm-up but still useful for consistent
        benchmarking results.

        Args:
            duration_seconds: How long to run warm-up operations (default: 1.0).

        Returns:
            bool: True if warm-up completed successfully, False otherwise.

        Note:
            CPU warm-up is minimal compared to GPU since CPUs don't have
            dynamic clock management like modern GPUs.
        """
        try:
            logger.info(f"Warming up CPU for {duration_seconds}s...")
            t0 = time.time()

            # Run bounded compute loop
            n = 0
            iterations = 0
            while time.time() - t0 < max(0.05, float(duration_seconds)):
                # Lightweight integer arithmetic to engage CPU
                for _ in range(10000):
                    n += (_ * 7) % 13
                iterations += 1

                # Small sleep to prevent complete CPU saturation
                if iterations % 10 == 0:
                    time.sleep(0.001)

            logger.info(f"CPU warm-up completed successfully ({iterations} iterations)")
            return True

        except Exception as e:
            logger.warning(f"CPU warm-up failed: {e}")
            return False

    @classmethod
    def get_performance_evaluation(cls) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics for CPU-only backend.

        Evaluates hardware capabilities and returns performance scores for
        inference workloads. Scoring acknowledges CPU limitations compared to
        GPU-accelerated backends.

        Scoring methodology:
            - Inference: CPU cores (40%), memory capacity (30%), memory bandwidth est. (20%), disk (10%)

        Normalization:
            - CPU cores: 64 cores = 100 points
            - Memory: 128GB = 100 points
            - Memory bandwidth: Estimated from CPU specs, 100GB/s = 100 points
            - Disk: 500GB available = 100 points

        Returns:
            Dict containing performance metrics and scores (0-100 scale):
            {
                "backend_type": "cpu",
                "cpu_model": str,
                "total_memory_gb": float,
                "available_memory_gb": float,
                "compute_units": int,  # CPU cores
                "cpu_performance_units": int,
                "global_inference_score": float,
                "global_inference_label": str,
                "cpu_score": float,
                "memory_score": float,
                "performance_breakdown": PerformanceBreakdown,
                ...
            }

        Note:
            CPU backend scores are generally lower than GPU backends due to
            lack of parallel processing capabilities. Scores are calibrated
            to be realistic about CPU limitations while still differentiating
            between high-end and low-end CPU systems.
        """
        try:
            # Import optional dependencies
            try:
                import psutil
            except ImportError as e:
                logger.warning(f"Optional hardware detection dependency missing: {e}")
                psutil = None

            # Get hardware info
            hw_info = cls.get_hardware_info()

            # Extract key metrics
            total_cores = hw_info["cpu"]["total_cores"] or 1
            total_memory_gb = hw_info["memory"]["total_memory_gb"] or 0
            available_memory_gb = hw_info["memory"]["available_memory_gb"] or 0
            disk_available_gb = hw_info["storage"]["available_gb"] or 0
            cpu_model = hw_info["cpu"]["model"]

            # === SCORING WEIGHTS ===
            # Inference: CPU (40%), Memory Capacity (30%), Memory BW (20%), Disk (10%)
            INF_WEIGHTS = {
                "cpu": 0.40,
                "memory_capacity": 0.30,
                "memory_bandwidth": 0.20,
                "disk": 0.10,
            }

            # === NORMALIZATION FACTORS ===
            NORM_CPU_CORES = 64.0  # 64 cores = 100 points
            NORM_MEMORY_GB = 128.0  # 128GB RAM = 100 points
            NORM_MEM_BW_GBS = 100.0  # 100GB/s = 100 points (estimated)
            NORM_DISK_GB = 500.0  # 500GB available = 100 points

            # === COMPONENT SCORES (0-100 scale) ===

            # CPU Score: Based on core count
            cpu_score = min(100.0, (total_cores / NORM_CPU_CORES) * 100.0)

            # Memory Score: Based on total memory capacity
            memory_capacity_score = min(100.0, (total_memory_gb / NORM_MEMORY_GB) * 100.0)

            # Memory Bandwidth Score: Estimate based on CPU architecture
            # Modern CPUs: ~50-100 GB/s, older CPUs: ~20-40 GB/s
            estimated_mem_bw = total_cores * 1.5  # Rough heuristic: 1.5 GB/s per core
            memory_bandwidth_score = min(100.0, (estimated_mem_bw / NORM_MEM_BW_GBS) * 100.0)

            # Disk Score: Based on available storage
            disk_score = min(100.0, (disk_available_gb / NORM_DISK_GB) * 100.0)

            # === GLOBAL SCORES ===

            # Inference Score
            inference_score = (
                cpu_score * INF_WEIGHTS["cpu"]
                + memory_capacity_score * INF_WEIGHTS["memory_capacity"]
                + memory_bandwidth_score * INF_WEIGHTS["memory_bandwidth"]
                + disk_score * INF_WEIGHTS["disk"]
            )

            # Round score
            inference_score = round(inference_score, 2)

            # === LABELS ===
            def score_to_label(score: float) -> str:
                """Convert 0-100 score to qualitative label."""
                if score >= 85:
                    return "Amazing"
                elif score >= 70:
                    return "Excellent"
                elif score >= 55:
                    return "Very Good"
                elif score >= 40:
                    return "Good"
                elif score >= 25:
                    return "Medium"
                elif score >= 10:
                    return "Poor"
                else:
                    return "Terrible"

            inference_label = score_to_label(inference_score)

            # === BUILD RESULT ===
            result = {
                "backend_type": "cpu",
                "gpu_name": "CPU Only",
                "cpu_model": cpu_model,
                "total_memory_gb": total_memory_gb,
                "available_memory_gb": available_memory_gb,
                "memory_bandwidth_gbs": round(estimated_mem_bw, 2),
                "disk_total_gb": hw_info["storage"]["total_gb"],
                "disk_available_gb": disk_available_gb,
                "estimated_tflops": None,  # Not applicable for CPUs
                "cpu_performance_units": total_cores,
                "neural_engine_tops": None,
                "cuda_version": None,
                "compute_capability": None,
                "architecture": hw_info["cpu"]["architecture"],
                "global_inference_score": inference_score,
                "global_inference_label": inference_label,
                "gpu_score": 0.0,  # No GPU
                "cpu_score": round(cpu_score, 2),
                "memory_score": round(memory_capacity_score, 2),
                "unified_memory": False,
                "system_platform": hw_info["system"]["platform"],
                "performance_breakdown": {
                    "compute_score": round(cpu_score, 2),
                    "memory_bandwidth_score": round(memory_bandwidth_score, 2),
                    "memory_capacity_score": round(memory_capacity_score, 2),
                    "cpu_performance_score": round(cpu_score, 2),
                    "disk_score": round(disk_score, 2),
                },
            }

            logger.info(
                f"CPU performance evaluated: Inference={inference_score:.1f} ({inference_label})"
            )

            return result

        except Exception as e:
            logger.exception(f"CPU performance evaluation failed: {e}")
            # Return minimal fallback
            cores = os.cpu_count() or 1
            base_score = min(100.0, cores * 5.0)  # 5 points per core
            return {
                "backend_type": "cpu",
                "gpu_name": "CPU Only",
                "cpu_model": platform.processor() or "Unknown CPU",
                "total_memory_gb": None,
                "available_memory_gb": None,
                "memory_bandwidth_gbs": None,
                "disk_total_gb": None,
                "disk_available_gb": None,
                "estimated_tflops": None,
                "cpu_performance_units": cores,
                "neural_engine_tops": None,
                "cuda_version": None,
                "compute_capability": None,
                "architecture": platform.machine(),
                "global_inference_score": round(base_score, 2),
                "global_inference_label": "Poor",
                "gpu_score": 0.0,
                "cpu_score": round(base_score, 2),
                "memory_score": 0.0,
                "unified_memory": False,
                "system_platform": platform.system(),
                "performance_breakdown": {
                    "compute_score": round(base_score, 2),
                    "memory_bandwidth_score": 0.0,
                    "memory_capacity_score": 0.0,
                    "cpu_performance_score": round(base_score, 2),
                    "disk_score": 0.0,
                },
            }

    @classmethod
    def get_flat_hardware_data(cls) -> Dict[str, Any]:
        """Get hardware data in flat format compatible with HardwareProfile entity.

        Returns hardware specifications as a flat dictionary ready for database
        insertion. For CPU backend, get_performance_evaluation() already returns
        data in the correct flat format.

        Returns:
            Flat dict with all fields matching HardwareProfile columns.

        Raises:
            HardwareException: If hardware data collection fails.
        """
        # get_performance_evaluation() already returns flat structure
        return cls.get_performance_evaluation()
