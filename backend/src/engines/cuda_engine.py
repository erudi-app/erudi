"""CUDA engine for NVIDIA GPU inference.

This module implements the CUDA backend for local LLM inference on systems
with NVIDIA GPUs. It provides:
- Hardware detection via PyTorch CUDA and NVML
- Performance evaluation for inference
- GPU warm-up for optimal benchmarking
- Cross-platform support (Windows and Linux)

Architecture:
    CUDA Engine → PyTorch CUDA + NVML
    ┌───────────────────────────────────────────────────────────┐
    │ get_hardware_info()                                       │
    │  └─> Detect GPU, VRAM, compute capability via NVML       │
    └───────────────────────────────────────────────────────────┘
                            ↓
    ┌───────────────────────────────────────────────────────────┐
    │ get_performance_evaluation()                              │
    │  1. Calculate TFLOPS (FP32, FP16, BF16)                   │
    │  2. Measure memory bandwidth                              │
    │  3. Score inference performance                           │
    └───────────────────────────────────────────────────────────┘
                            ↓
    ┌───────────────────────────────────────────────────────────┐
    │ warm_up_accelerator()                                     │
    │  └─> Matrix operations to boost GPU clocks               │
    └───────────────────────────────────────────────────────────┘

Hardware Detection:
    - CUDA availability: torch.cuda.is_available()
    - GPU information: NVML (pynvml library)
    - Best GPU selection: highest VRAM capacity
    - OS-agnostic: Works on Windows and Linux

Performance Scoring:
    Inference weights:
        - GPU compute (40%): Tensor TFLOPS (BF16/FP16)
        - Memory bandwidth (30%): GB/s throughput
        - VRAM capacity (15%): GPU memory size
        - CPU performance (5%): Multi-core GHz
        - System RAM (5%): Host memory
        - PCIe bandwidth (5%): Bus capacity

Example:
    Get CUDA hardware information::

        from src.engines.cuda_engine import CUDA_Engine

        # Detect hardware
        hw_info = CUDA_Engine.get_hardware_info()
        print(f"GPU: {hw_info['gpu']['gpu_name']}")
        print(f"VRAM: {hw_info['gpu']['vram_total_gb']} GB")

        # Warm up GPU
        CUDA_Engine.warm_up_accelerator(1.5)

        # Evaluate performance
        perf = CUDA_Engine.get_performance_evaluation()
        print(f"Inference score: {perf['global_inference_score']}/100")
        print(f"Label: {perf['global_inference_label']}")

Note:
    - Requires torch with CUDA support
    - Requires pynvml for GPU monitoring
    - All methods return fallback values if CUDA unavailable
    - Thread-safe through BaseEngine._lock
    - Inference uses llama.cpp compiled with CUDA (llama-server subprocess)
    - GPU layer count auto-detected from available VRAM at model load time
"""

import os
import time  # used by hardware methods
import platform
from typing import Any, Dict, List, Optional
from pathlib import Path

from src.engines.base_llama_cpp_engine import BaseLlamaCppEngine
from src.engines.cpu_brand import get_cpu_brand
from src.engines.cuda_compatibility import format_cuda_version
from src.core.logging import logger
from src.core.exceptions import HardwareException


class CUDA_Engine(BaseLlamaCppEngine):
    """Engine for NVIDIA CUDA GPU inference.

    Provides hardware detection and performance evaluation for NVIDIA GPUs
    using PyTorch CUDA backend and NVML monitoring library.

    Hardware Specs Reference Tables:
        CUDA_PER_SM: CUDA cores and Tensor cores per SM by compute capability
        TC_OPS: Tensor core operations per precision by compute capability

    Note:
        Inference uses a CUDA-compiled llama-server subprocess (same
        architecture as CPU_Engine). GPU layers are auto-detected from VRAM.
    """

    # --- BaseChatServerEngine / BaseLlamaCppEngine config overrides ---
    _server_name = "llama-server"
    _tokenizer_provider = "llama-server-cuda"
    _use_cuda_build = True  # → BaseLlamaCppEngine._default_install_dir picks the cuda/ artifact

    # --- CUDA-specific spawn hooks ----
    # `_compute_gpu_layers` is defined further below and reads NVML at spawn time.

    # NVIDIA GPU architecture specifications
    # Maps compute capability major version to (CUDA cores/SM, Tensor cores/SM)
    _CUDA_PER_SM = {
        7: (64, 8),  # Turing (SM 7.5): RTX 20xx, Quadro RTX
        8: (128, 4),  # Ampere (SM 8.x): RTX 30xx, A100
        9: (128, 4),  # Ada Lovelace / Hopper (SM 9.x): RTX 40xx, H100
        10: (128, 4),  # Blackwell (SM 10.x): RTX 50xx (data-centre B100/B200)
        12: (128, 4),  # Blackwell (SM 12.0): RTX 5060 Ti / consumer desktop
    }

    # Tensor core operations per clock per precision
    # Format: precision → (ops_per_TC_per_clock, minimum_compute_capability)
    _TC_OPS = {
        "fp16": (256 * 2, 7),  # 256 FMA = 512 FLOP, requires CC 7.0+
        "bf16": (256 * 2, 8),  # 256 FMA = 512 FLOP, requires CC 8.0+
    }

    # Performance scoring weights for inference workload (sum = 1.0)
    _WEIGHTS_INFERENCE = {
        "gpu_compute": 0.40,  # Tensor TFLOPS (BF16/FP16)
        "gpu_bw": 0.30,  # Memory bandwidth GB/s
        "gpu_vram": 0.15,  # VRAM capacity
        "cpu_single": 0.05,  # CPU performance units
        "sys_ram": 0.05,  # System RAM
        "pcie": 0.05,  # PCIe bus capacity
    }

    # Normalization factors for inference scoring
    _NORM_INFERENCE = {
        "tflops": 80,  # 80 TFLOPS FP16 reference
        "bandwidth": 500,  # 500 GB/s (7B models saturate beyond this)
        "vram": 12,  # 12 GB (sufficient for 7B INT4)
        "cpu_ghz": 3.6,  # 3.6 GHz single-core reference
        "ram": 24,  # 24 GB (comfortable for multi-tasking)
        "pcie": 32,  # Gen3 x16 or Gen4 x8
    }

    # USES_GGUF and MODEL_MAPPING (the public-GGUF catalog) are inherited from
    # BaseLlamaCppEngine — shared with CPU_Engine so both stay in sync.

    # ---------- CUDA-specific spawn hooks (BaseLlamaCppEngine contract) ----------

    @classmethod
    def _prepare_spawn_context(cls) -> Dict[str, Any]:
        """Resolve per-spawn CUDA context: context window, thread count,
        and GPU layers computed from current VRAM (NVML)."""
        return {
            "ctx_size": cls.max_context_tokens(),
            "threads": max(1, os.cpu_count() or 1),
            "gpu_layers": cls._compute_gpu_layers(),
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
        gpu_layers: int = -1,
        **_ignored: Any,
    ) -> List[Any]:
        """CUDA CLI for llama-server: injects computed `-ngl <gpu_layers>`.

        ``--jinja`` enables the model's own chat template and with it
        OpenAI-style function calling (the agent's calculator tool) —
        without it llama-server never emits ``tool_calls``.

        ``--reasoning-format none`` keeps ``<think>...</think>`` INLINE in the
        answer stream (#90). The default (``auto``) extracts reasoning into a
        dedicated ``delta.reasoning_content`` field that ChatOpenAI drops, so the
        thinking would be silently lost. Inline, the runner's single streaming
        splitter separates thinking from answer uniformly across engines.
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
            "--reasoning-format",
            "none",
        ]

    @classmethod
    def _build_spawn_env(cls) -> Dict[str, str]:
        """Prepend the CUDA toolkit `bin/` to PATH so llama-server finds
        the runtime DLLs (cuBLAS, cuDNN, etc.)."""
        env = os.environ.copy()
        cuda_bin = cls._resolve_cuda_bin_dir()
        if cuda_bin:
            env["PATH"] = str(cuda_bin) + os.pathsep + env.get("PATH", "")
            logger.debug(f"[CUDA_Engine] Prepended CUDA bin to PATH: {cuda_bin}")
        return env

    # ---------- CUDA-specific helpers (kept from pre-refactor) ----------

    @classmethod
    def _resolve_cuda_bin_dir(cls) -> Optional[Path]:
        """Locate CUDA toolkit bin directory for runtime DLLs.

        Checks CUDA_PATH environment variable first, then the default
        NVIDIA GPU Computing Toolkit install locations on Windows.

        Returns:
            Path to CUDA bin directory, or None if not found.
        """
        # 1. Check CUDA_PATH env (set by CUDA installer)
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path:
            bin_dir = Path(cuda_path) / "bin"
            if bin_dir.is_dir():
                return bin_dir

        # 2. Windows default install locations
        if os.name == "nt":
            base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
            if base.is_dir():
                # Pick highest version available
                versions = sorted(base.iterdir(), reverse=True)
                for ver in versions:
                    candidate = ver / "bin"
                    if candidate.is_dir():
                        return candidate

        # 3. Linux: CUDA usually in /usr/local/cuda/bin and already on PATH
        linux_default = Path("/usr/local/cuda/bin")
        if linux_default.is_dir():
            return linux_default

        return None

    @classmethod
    def _compute_gpu_layers(cls) -> int:
        """Determine GPU layer count based on available VRAM.

        Returns:
            -1 for full offload (≥10 GB free), else a partial count, or
            0 if no GPU detected.

        Override for testing:
            Set ERUDI_VRAM_OVERRIDE_GB=<float> to simulate a specific free VRAM amount.
            Examples:
                ERUDI_VRAM_OVERRIDE_GB=2   → 0  (CPU fallback)
                ERUDI_VRAM_OVERRIDE_GB=4.5 → 20 (partial offload)
                ERUDI_VRAM_OVERRIDE_GB=7   → 32 (partial offload)
                ERUDI_VRAM_OVERRIDE_GB=12  → -1 (full GPU)
        """
        import os

        vram_override = os.environ.get("ERUDI_VRAM_OVERRIDE_GB")
        if vram_override is not None:
            try:
                vram_free_gb = float(vram_override)
                logger.info(f"[ERUDI_VRAM_OVERRIDE_GB] Simulating {vram_free_gb:.1f} GB free VRAM.")
            except ValueError:
                logger.warning(
                    f"[ERUDI_VRAM_OVERRIDE_GB] Invalid value '{vram_override}', ignoring."
                )
                vram_override = None

        if vram_override is None:
            gpus = cls._get_nvml_gpus()
            if not gpus:
                # The CUDA engine was selected, yet NVML now sees no device:
                # every generation runs on the processor from here on, which
                # the user will experience as a slow model and nothing else.
                logger.warning(
                    "[CUDA_Engine] NVML reports no GPU; llama-server starts with 0 GPU "
                    "layers (processor only)"
                )
                return 0
            best = cls._select_best_gpu(gpus)
            vram_free_gb = best["vram_free_mb"] / 1024

        if vram_free_gb < 3:
            return 0
        if vram_free_gb < 6:
            return 20
        if vram_free_gb < 10:
            return 32
        return -1  # Full offload

    # ======================= HARDWARE DETECTION =======================

    @classmethod
    def _cuda_available(cls) -> bool:
        """Check if CUDA is available via NVML (pynvml).

        Returns:
            bool: True if NVML initialises and at least one GPU is detected.

        Note:
            Uses pynvml so that a CPU-only torch installation does not affect
            GPU detection.
        """
        try:
            if not cls._init_nvml():
                return False
            import pynvml as nv

            return nv.nvmlDeviceGetCount() > 0
        except Exception as e:
            logger.warning(f"CUDA availability check failed: {e}")
            return False

    @classmethod
    def _get_compute_capability(cls, handle) -> tuple[int, int]:
        """Return (major, minor) compute capability for a GPU handle via NVML."""
        try:
            import pynvml as nv

            return nv.nvmlDeviceGetCudaComputeCapability(handle)
        except Exception as e:
            logger.warning(f"Could not get compute capability: {e}")
            return (0, 0)

    @staticmethod
    def _rated_clock_mhz(nv, handle, domain) -> int:
        """Return a clock domain's RATED (maximum) frequency in MHz.

        The hardware profile rates what the card *can* do, so every clock it
        reads must be the rated one. nvmlDeviceGetClockInfo reports the
        INSTANTANEOUS frequency instead, and an idle GPU sits deep in a low
        power state: measured on an RTX 5060 Ti at idle, memory reads 405 MHz
        against a rated 14001, and SM reads 300 MHz against a rated 3090. Since
        the profile is built once at startup, on an idle machine by definition,
        that turned a 448 GB/s card into a 13 GB/s one and collapsed the
        recommended model window to 1-1.1B on 16 GB of VRAM.

        nvmlDeviceGetMaxClockInfo is supported by every driver that supports
        nvmlDeviceGetClockInfo, but fall back to the instantaneous reading
        rather than lose the field entirely if it ever raises.
        """
        try:
            return nv.nvmlDeviceGetMaxClockInfo(handle, domain)
        except Exception as e:
            logger.warning(f"Max clock unavailable, falling back to current clock: {e}")
            return nv.nvmlDeviceGetClockInfo(handle, domain)

    @classmethod
    def _get_sm_count(cls, device_id: int) -> int:
        """Return the SM count for a device using the CUDA Driver API (nvcuda.dll).

        nvcuda.dll is installed with every NVIDIA driver — no CUDA toolkit
        or torch required.  Falls back to 0 on any error.
        """
        try:
            import ctypes

            nvcuda = ctypes.CDLL("nvcuda.dll")
            nvcuda.cuInit(0)
            device = ctypes.c_int()
            nvcuda.cuDeviceGet(ctypes.byref(device), device_id)
            sm_count = ctypes.c_int()
            # CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
            nvcuda.cuDeviceGetAttribute(ctypes.byref(sm_count), 16, device)
            return sm_count.value
        except Exception as e:
            logger.warning(f"Could not get SM count via nvcuda.dll: {e}")
            return 0

    @classmethod
    def _get_cuda_driver_version_int(cls) -> int:
        """Return the driver's CUDA version as NVML reports it (e.g. 12080).

        This is the raw comparable form the startup pre-flight needs
        (``src/engines/cuda_compatibility.py``): the driver must reach a given
        CUDA version for the bundled binary to run, and "12.1" as a string
        does not compare. 0 means the reading failed.
        """
        try:
            import pynvml as nv

            return int(nv.nvmlSystemGetCudaDriverVersion())
        except Exception as e:
            logger.warning(f"Could not get CUDA driver version: {e}")
            return 0

    @classmethod
    def _get_cuda_driver_version(cls) -> str:
        """Return CUDA driver version string via NVML."""
        # nvmlSystemGetCudaDriverVersion returns an int like 12010 -> "12.1"
        v = cls._get_cuda_driver_version_int()
        if not v:
            return "Unknown"
        return format_cuda_version(v)

    @classmethod
    def _init_nvml(cls) -> bool:
        """Initialize NVIDIA Management Library (NVML).

        Returns:
            bool: True if NVML initialized successfully, False otherwise.

        Note:
            Safe to call multiple times. Returns True if already initialized.
        """
        try:
            import pynvml as nv

            try:
                nv.nvmlInit()
                return True
            except nv.NVMLError as e:
                # Already initialized is not an error
                if "Already initialized" in str(e) or "Initialized" in str(e):
                    return True
                logger.warning(f"NVML initialization failed: {e}")
                return False
        except ImportError:
            logger.warning("pynvml not installed, GPU monitoring unavailable")
            return False

    @classmethod
    def _get_nvml_gpus(cls) -> list[dict]:
        """Get list of GPUs detected by NVML.

        Returns:
            List of dictionaries describing each GPU:
            [
                {
                    "id": int,
                    "handle": nvmlDevice,
                    "name": str,
                    "vram_total_mb": float,
                    "vram_free_mb": float
                },
                ...
            ]
            Empty list if NVML unavailable or no GPUs found.

        Note:
            Automatically initializes NVML if needed.
        """
        if not cls._init_nvml():
            return []

        try:
            import pynvml as nv

            gpus = []
            device_count = nv.nvmlDeviceGetCount()

            for idx in range(device_count):
                try:
                    handle = nv.nvmlDeviceGetHandleByIndex(idx)
                    name = nv.nvmlDeviceGetName(handle)
                    mem_info = nv.nvmlDeviceGetMemoryInfo(handle)

                    gpus.append(
                        {
                            "id": idx,
                            "handle": handle,
                            "name": name,
                            "vram_total_mb": mem_info.total / (1024**2),
                            "vram_free_mb": mem_info.free / (1024**2),
                        }
                    )
                except nv.NVMLError as e:
                    logger.warning(f"Failed to get info for GPU {idx}: {e}")
                    continue

            return gpus

        except Exception as e:
            # The caller treats an empty list as "no GPU" and carries on, so
            # this is a degradation with a fallback, and the traceback is the
            # only way to tell a broken driver from an absent one.
            logger.warning(f"Failed to enumerate GPUs: {e}", exc_info=True)
            return []

    @classmethod
    def _select_best_gpu(cls, gpus: list[dict]) -> Optional[dict]:
        """Select best GPU based on total VRAM capacity.

        Args:
            gpus: List of GPU dictionaries from _get_nvml_gpus().

        Returns:
            GPU dictionary with highest VRAM, or None if list empty.
        """
        if not gpus:
            return None
        return max(gpus, key=lambda g: g["vram_total_mb"])

    @classmethod
    def _get_cpu_performance_units(cls) -> float:
        """Calculate CPU performance units (cores × GHz).

        Returns:
            float: Performance units (physical_cores * max_frequency_ghz).
                  Fallback: 10.0 if detection fails.

        Note:
            Uses psutil for cross-platform CPU detection.
        """
        try:
            import psutil

            freq = psutil.cpu_freq()
            max_freq_mhz = freq.max if freq and freq.max else (freq.current if freq else 2500)
            physical_cores = psutil.cpu_count(logical=False) or 4

            return physical_cores * max_freq_mhz / 1000  # Convert MHz to GHz

        except Exception as e:
            logger.warning(f"CPU performance detection failed: {e}")
            return 10.0  # Fallback

    @classmethod
    def _get_pcie_capacity(cls, handle) -> float:
        """Calculate PCIe bus capacity units (generation × width).

        Args:
            handle: NVML device handle.

        Returns:
            float: PCIe capacity (gen * width). E.g., Gen3 x16 = 48.
                  Fallback: 16.0 if detection fails.

        Note:
            Higher values indicate better host-device bandwidth.
        """
        try:
            import pynvml as nv

            gen = nv.nvmlDeviceGetMaxPcieLinkGeneration(handle)
            width = nv.nvmlDeviceGetMaxPcieLinkWidth(handle)

            return float(gen * width)

        except Exception as e:
            logger.warning(f"PCIe capacity detection failed: {e}")
            return 16.0  # Fallback (Gen3 x16 / Gen4 x8)

    @classmethod
    def get_hardware_info(cls) -> Dict[str, Any]:
        """Get comprehensive hardware information for NVIDIA CUDA backend.

        Returns detailed hardware specifications including GPU model, VRAM,
        compute capability, system resources, and CUDA availability.

        Returns:
            Dict containing hardware specifications:
            {
                "system": {
                    "platform": str,
                    "platform_version": str,
                    "machine": str,
                    "processor": str
                },
                "cpu": {
                    "model": str,
                    "architecture": str,
                    "total_cores": int,
                    "logical_cores": int
                },
                "memory": {
                    "total_memory_gb": float,
                    "available_memory_gb": float,
                    "memory_pressure": float,
                    "memory_type": "system"
                },
                "gpu": {
                    "gpu_name": str,
                    "gpu_index": int,
                    "cuda_cores": int,
                    "vram_total_gb": float,
                    "vram_available_gb": float,
                    "compute_capability": str,
                    "memory_bandwidth_gbs": float,
                    "cuda_available": bool,
                    "unified_memory": False
                },
                "storage": {
                    "total_gb": float,
                    "available_gb": float,
                    "usage_percentage": float
                },
                "backend_type": "cuda",
                "timestamp": float
            }

        Raises:
            HardwareException: If critical hardware detection fails.

        Note:
            Returns fallback values if CUDA unavailable rather than raising.
            Safe to call even on systems without NVIDIA GPUs.

        Examples:
            >>> hw_info = CUDA_Engine.get_hardware_info()
            >>> print(f"GPU: {hw_info['gpu']['gpu_name']}")
            >>> print(f"VRAM: {hw_info['gpu']['vram_total_gb']} GB")
            >>> print(f"Compute: {hw_info['gpu']['compute_capability']}")
        """
        try:
            # Import required modules
            try:
                import psutil
            except ImportError as e:
                logger.error(f"Required hardware detection dependency missing: {e}")
                raise HardwareException(
                    f"Missing dependency for hardware detection: {e}", trace=str(e)
                )

            # Check CUDA availability
            cuda_available = cls._cuda_available()

            # Get system info
            vm = psutil.virtual_memory()
            total_memory_gb = vm.total / (1024**3)
            available_memory_gb = vm.available / (1024**3)
            memory_pressure = 1.0 - (vm.available / vm.total)

            disk = psutil.disk_usage(os.path.abspath(os.sep))
            disk_total_gb = disk.total / (1024**3)
            disk_available_gb = disk.free / (1024**3)
            disk_usage_pct = disk.percent

            cpu_model = get_cpu_brand() or "Unknown CPU"
            total_cores = psutil.cpu_count(logical=False) or 4
            logical_cores = psutil.cpu_count(logical=True) or 8

            # GPU information (if available)
            gpu_info = {
                "gpu_name": "No NVIDIA GPU",
                "gpu_index": -1,
                "cuda_cores": 0,
                "vram_total_gb": 0.0,
                "vram_available_gb": 0.0,
                "compute_capability": "N/A",
                "memory_bandwidth_gbs": 0.0,
                "cuda_available": cuda_available,
                "unified_memory": False,
            }

            if cuda_available:
                gpus = cls._get_nvml_gpus()
                if gpus:
                    best_gpu = cls._select_best_gpu(gpus)
                    if best_gpu:
                        # Get basic GPU info
                        gpu_info["gpu_name"] = best_gpu["name"]
                        gpu_info["gpu_index"] = best_gpu["id"]
                        gpu_info["vram_total_gb"] = best_gpu["vram_total_mb"] / 1024
                        gpu_info["vram_available_gb"] = best_gpu["vram_free_mb"] / 1024

                        # Get compute capability and CUDA cores
                        device_id = best_gpu["id"]
                        major, minor = cls._get_compute_capability(best_gpu["handle"])
                        compute_cap = f"{major}.{minor}"
                        gpu_info["compute_capability"] = compute_cap

                        cores_per_sm, _ = cls._CUDA_PER_SM.get(major, (64, 0))
                        sm_count = cls._get_sm_count(device_id)
                        gpu_info["cuda_cores"] = sm_count * cores_per_sm

                        # Get memory bandwidth (requires NVML)
                        try:
                            import pynvml as nv

                            handle = best_gpu["handle"]
                            mem_clock_mhz = cls._rated_clock_mhz(nv, handle, nv.NVML_CLOCK_MEM)
                            bus_bits = nv.nvmlDeviceGetMemoryBusWidth(handle)
                            # Bandwidth = 2 * clock * bus_width / 8 (DDR, convert bits to bytes)
                            bandwidth_gbs = 2 * mem_clock_mhz / 1000 * bus_bits / 8
                            gpu_info["memory_bandwidth_gbs"] = round(bandwidth_gbs, 1)
                        except Exception as e:
                            logger.warning(f"Could not get memory bandwidth: {e}")

            # Build complete hardware info
            hardware_info = {
                "system": {
                    "platform": platform.system(),
                    "platform_version": platform.version(),
                    "machine": platform.machine(),
                    "processor": platform.processor() or cpu_model,
                },
                "cpu": {
                    "model": cpu_model,
                    "architecture": platform.machine(),
                    "total_cores": total_cores,
                    "logical_cores": logical_cores,
                },
                "memory": {
                    "total_memory_gb": round(total_memory_gb, 2),
                    "available_memory_gb": round(available_memory_gb, 2),
                    "memory_pressure": round(memory_pressure, 3),
                    "memory_type": "system",
                },
                "gpu": gpu_info,
                "storage": {
                    "total_gb": round(disk_total_gb, 2),
                    "available_gb": round(disk_available_gb, 2),
                    "usage_percentage": round(disk_usage_pct, 2),
                },
                "backend_type": "cuda",
                "timestamp": time.time(),
            }

            logger.info(
                f"CUDA hardware detected: {gpu_info['gpu_name']}, "
                f"{gpu_info['cuda_cores']} CUDA cores, "
                f"{gpu_info['vram_total_gb']:.1f}GB VRAM"
            )

            return hardware_info

        except Exception as e:
            logger.exception(f"CUDA hardware detection failed: {e}")
            raise HardwareException("Failed to detect CUDA hardware", trace=str(e))

    @classmethod
    def warm_up_accelerator(cls, duration_seconds: float = 1.0) -> bool:
        """Warm up NVIDIA GPU to boost clock speeds for benchmarking.

        Runs matrix multiplication operations on GPU to bring it to optimal
        performance state before measurements. Important for GPUs with
        dynamic clock management.

        Args:
            duration_seconds: How long to run warm-up operations (default: 1.0).

        Returns:
            bool: True if warm-up completed successfully, False otherwise.

        Note:
            Silently returns False if CUDA unavailable rather than raising.
            Uses 4096x4096 matrix multiplications on primary GPU.

        Examples:
            >>> success = CUDA_Engine.warm_up_accelerator(1.5)
            >>> if success:
            ...     print("GPU ready for benchmarking")
        """
        if not cls._cuda_available():
            logger.warning("CUDA not available, skipping GPU warm-up")
            return False

        # GPU warm-up via torch matmul is skipped — the packaged build uses
        # CPU-only torch.  llama-server warms up its own CUDA context on first
        # inference, so this has no effect on LLM performance.
        logger.info("GPU warm-up skipped (not required for llama-server inference)")
        return False

    @classmethod
    def get_performance_evaluation(cls) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics for NVIDIA CUDA backend.

        Evaluates hardware capabilities and returns normalized performance scores
        for inference workloads. Scoring based on TFLOPS, memory bandwidth, VRAM
        capacity, and system resources.

        Scoring methodology:
            Inference (0-100 scale):
                - GPU compute (40%): Tensor TFLOPS (BF16/FP16)
                - Memory bandwidth (30%): GB/s throughput
                - VRAM capacity (15%): GPU memory size
                - CPU performance (5%): Multi-core units
                - System RAM (5%): Host memory
                - PCIe capacity (5%): Bus bandwidth

        Returns:
            Dict containing performance metrics and scores:
            {
                "backend_type": "cuda",
                "gpu_name": str,
                "cpu_model": str,
                "total_memory_gb": float,
                "available_memory_gb": float,
                "memory_bandwidth_gbs": float,
                "disk_total_gb": float,
                "disk_available_gb": float,
                "estimated_tflops": float,
                "compute_units": int,
                "cpu_performance_units": float,
                "cuda_version": str,
                "compute_capability": str,
                "architecture": str,
                "global_inference_score": float,
                "global_inference_label": str,
                "gpu_score": float,
                "cpu_score": float,
                "memory_score": float,
                "unified_memory": False,
                "accelerator_available": bool,
                "system_platform": str,
                "performance_breakdown": dict
            }

        Raises:
            HardwareException: If evaluation fails critically.

        Note:
            Returns fallback scores if CUDA unavailable.
            Should call warm_up_accelerator() before evaluation for accuracy.

        Examples:
            >>> CUDA_Engine.warm_up_accelerator(1.5)
            >>> eval_result = CUDA_Engine.get_performance_evaluation()
            >>> print(f"Inference: {eval_result['global_inference_score']}/100")
            >>> print(f"Label: {eval_result['global_inference_label']}")
        """
        try:
            # Get base hardware info
            hw_info = cls.get_hardware_info()

            cuda_available = hw_info["gpu"]["cuda_available"]

            # Initialize scores with fallback values
            gpu_score = 0.0
            mem_bandwidth_score = 0.0
            vram_score = 0.0
            estimated_tflops = 0.0
            cuda_version = "N/A"
            architecture = "N/A"
            tensor_tflops = {}
            sm_clock_ghz = 0.0

            if cuda_available:
                gpus = cls._get_nvml_gpus()
                if gpus:
                    best_gpu = cls._select_best_gpu(gpus)
                    device_id = best_gpu["id"]
                    handle = best_gpu["handle"]

                    # Get GPU clocks and specs
                    try:
                        import pynvml as nv

                        sm_clock_mhz = cls._rated_clock_mhz(nv, handle, nv.NVML_CLOCK_SM)
                        sm_clock_ghz = sm_clock_mhz / 1000
                        mem_clock_mhz = cls._rated_clock_mhz(nv, handle, nv.NVML_CLOCK_MEM)
                        bus_bits = nv.nvmlDeviceGetMemoryBusWidth(handle)
                        bandwidth_gbs = 2 * mem_clock_mhz / 1000 * bus_bits / 8
                    except Exception as e:
                        logger.warning(f"Failed to get GPU clocks: {e}")
                        sm_clock_ghz = 1.5  # Fallback
                        bandwidth_gbs = hw_info["gpu"]["memory_bandwidth_gbs"]

                    # Get compute capability and SM count (no torch required)
                    compute_cap_major, compute_cap_minor = cls._get_compute_capability(handle)
                    sm_count = cls._get_sm_count(device_id)

                    # Calculate CUDA cores and tensor cores
                    cores_per_sm, tc_per_sm = cls._CUDA_PER_SM.get(compute_cap_major, (64, 0))
                    total_cuda_cores = sm_count * cores_per_sm

                    # Calculate FP32 TFLOPS
                    # 2 ops per CUDA core per clock (FMA = mul + add)
                    fp32_tflops = 2 * sm_count * cores_per_sm * sm_clock_ghz / 1000

                    # Calculate Tensor TFLOPS (FP16, BF16)
                    for precision, (ops_per_tc, min_cc) in cls._TC_OPS.items():
                        if compute_cap_major >= min_cc:
                            tflops = (sm_count * tc_per_sm * ops_per_tc * sm_clock_ghz) / 1000
                            tensor_tflops[precision] = round(tflops, 2)
                        else:
                            tensor_tflops[precision] = 0.0

                    # Use best tensor performance for scoring
                    estimated_tflops = tensor_tflops.get(
                        "bf16", tensor_tflops.get("fp16", fp32_tflops)
                    )

                    # Get CUDA version from NVML driver info
                    cuda_version = cls._get_cuda_driver_version()

                    # Determine architecture
                    if compute_cap_major == 7:
                        architecture = "Turing"
                    elif compute_cap_major == 8:
                        architecture = "Ampere"
                    elif compute_cap_major == 9:
                        architecture = "Ada Lovelace / Hopper"
                    else:
                        architecture = f"Compute {compute_cap_major}.{compute_cap_minor}"

                    # Calculate component scores
                    vram_gb = hw_info["gpu"]["vram_total_gb"]
                    vram_score = min(100, (vram_gb / cls._NORM_INFERENCE["vram"]) * 100)
                    mem_bandwidth_score = min(
                        100, (bandwidth_gbs / cls._NORM_INFERENCE["bandwidth"]) * 100
                    )
                    gpu_score = min(100, (estimated_tflops / cls._NORM_INFERENCE["tflops"]) * 100)

            # CPU and system scores
            cpu_perf_units = cls._get_cpu_performance_units()
            cpu_score = min(100, (cpu_perf_units / (cls._NORM_INFERENCE["cpu_ghz"] * 12)) * 100)

            sys_ram_gb = hw_info["memory"]["total_memory_gb"]
            ram_score = min(100, (sys_ram_gb / cls._NORM_INFERENCE["ram"]) * 100)

            # PCIe score
            pcie_score = 50.0  # Default fallback
            if cuda_available and gpus:
                pcie_units = cls._get_pcie_capacity(handle)
                pcie_score = min(100, (pcie_units / cls._NORM_INFERENCE["pcie"]) * 100)

            # Calculate weighted inference score
            inference_score = (
                cls._WEIGHTS_INFERENCE["gpu_compute"] * gpu_score
                + cls._WEIGHTS_INFERENCE["gpu_bw"] * mem_bandwidth_score
                + cls._WEIGHTS_INFERENCE["gpu_vram"] * vram_score
                + cls._WEIGHTS_INFERENCE["cpu_single"] * cpu_score
                + cls._WEIGHTS_INFERENCE["sys_ram"] * ram_score
                + cls._WEIGHTS_INFERENCE["pcie"] * pcie_score
            )

            # Generate labels
            def get_label(score: float) -> str:
                if score >= 90:
                    return "Amazing"
                elif score >= 80:
                    return "Excellent"
                elif score >= 70:
                    return "Very High"
                elif score >= 60:
                    return "High"
                elif score >= 50:
                    return "Good"
                elif score >= 40:
                    return "Medium"
                elif score >= 30:
                    return "Bad"
                elif score >= 20:
                    return "Very Bad"
                elif score >= 10:
                    return "Poor"
                else:
                    return "Terrible"

            inference_label = get_label(inference_score)

            # Build performance breakdown
            performance_breakdown = {
                "compute_score": round(gpu_score, 2),
                "memory_bandwidth_score": round(mem_bandwidth_score, 2),
                "memory_capacity_score": round(vram_score, 2),
                "cpu_performance_score": round(cpu_score, 2),
                "system_ram_score": round(ram_score, 2),
                "pcie_score": round(pcie_score, 2),
                "weights_inference": cls._WEIGHTS_INFERENCE,
                "normalization_inference": cls._NORM_INFERENCE,
            }

            # Build complete evaluation result
            eval_result = {
                # Hardware identification
                "backend_type": "cuda",
                "gpu_name": hw_info["gpu"]["gpu_name"],
                "cpu_model": hw_info["cpu"]["model"],
                # Memory metrics
                "total_memory_gb": hw_info["memory"]["total_memory_gb"],
                "available_memory_gb": hw_info["memory"]["available_memory_gb"],
                "memory_bandwidth_gbs": hw_info["gpu"]["memory_bandwidth_gbs"],
                # Storage metrics
                "disk_total_gb": hw_info["storage"]["total_gb"],
                "disk_available_gb": hw_info["storage"]["available_gb"],
                # Compute metrics
                "estimated_tflops": round(estimated_tflops, 2),
                "tensor_tflops": tensor_tflops,
                "compute_units": hw_info["gpu"]["cuda_cores"],
                "cuda_cores": hw_info["gpu"]["cuda_cores"],  # Entity field name
                "cpu_performance_units": round(cpu_perf_units, 1),
                # CUDA specific
                "cuda_version": cuda_version,
                "compute_capability": hw_info["gpu"]["compute_capability"],
                "architecture": architecture,
                "sm_clock_ghz": round(sm_clock_ghz, 3),
                # VRAM (CUDA-specific, from hw_info)
                "vram_total_gb": hw_info["gpu"]["vram_total_gb"],
                "vram_available_gb": hw_info["gpu"]["vram_available_gb"],
                # Performance scores (0-100)
                "global_inference_score": round(inference_score, 2),
                "global_inference_label": inference_label,
                "gpu_score": round(gpu_score, 2),
                "cpu_score": round(cpu_score, 2),
                "memory_score": round(vram_score, 2),
                # Technical details
                "unified_memory": False,
                "accelerator_available": cuda_available,
                "system_platform": hw_info["system"]["platform"],
                # Performance breakdown
                "performance_breakdown": performance_breakdown,
            }

            logger.info(
                f"Performance evaluation: Inference={inference_score:.1f}/100 ({inference_label})"
            )

            return eval_result

        except Exception as e:
            logger.exception(f"Performance evaluation failed: {e}")
            raise HardwareException("Failed to evaluate CUDA performance", trace=str(e))

    @classmethod
    def get_flat_hardware_data(cls) -> Dict[str, Any]:
        """Get hardware data in flat format compatible with HardwareProfile entity.

        Calls get_performance_evaluation() and strips keys that have no matching
        column in the HardwareProfile ORM model so the result can be splatted
        directly into the entity constructor.

        Returns:
            Flat dict with fields matching HardwareProfile columns.

        Raises:
            HardwareException: If hardware data collection fails.
        """
        data = cls.get_performance_evaluation()
        # Keys produced by get_performance_evaluation() that are NOT columns on
        # HardwareProfile — strip them to avoid "invalid keyword argument".
        _EXTRA_KEYS = {
            "tensor_tflops",
            "sm_clock_ghz",
            "compute_units",
            "accelerator_available",
        }
        for key in _EXTRA_KEYS:
            data.pop(key, None)
        return data
