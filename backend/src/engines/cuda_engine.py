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

Inference (llama.cpp compiled with CUDA, a llama-server subprocess):
    - The window (-c) is sized by Erudi at spawn for a FULL offload, against
      the selected GPU's memory at load (cuda_memory); llama.cpp's fit places
      the layers (-ngl -1, "auto" while fit is on) with Erudi's margin
      (-fitt). When the floor window does not fit or the GGUF cannot be
      modelled, no -c: fit sizes the window and places the layers.
    - After load the placement is classified (full or partial offload); on
      Windows a full-offload child carries a memory brake driven by the WDDM
      spill gauge alone -- its Shared GPU memory above the healthy level of
      its window (memory_brake.llama_step); RAM pressure never drives it. On
      Linux, which has no spill, the brake has no signal and never acts. A
      partial or fit-sized child runs brake-off.
    - One GPU, pinned by UUID (CUDA_VISIBLE_DEVICES), never over a value the
      user set.

Note:
    - Requires torch with CUDA support
    - Requires pynvml for GPU monitoring and the window sizing
    - All methods return fallback values if CUDA unavailable
    - Thread-safe through BaseEngine._lock
"""

import dataclasses
import os
import time  # used by hardware methods
import platform
from typing import Any, ClassVar, Dict, List, Optional, Tuple
from pathlib import Path

from src.engines import cuda_memory, gguf_memory, memory_pressure
from src.engines.base_llama_cpp_engine import BaseLlamaCppEngine, LlamaSignals
from src.engines.cpu_brand import get_cpu_brand
from src.engines.cuda_compatibility import format_cuda_version
from src.engines.memory_brake import BrakeOutcome
from src.engines.working_window import MIN_KV_BOUND_TOKENS
from src.core.logging import logger
from src.core.exceptions import EngineException, HardwareException

_MIB = 1024**2
# How much of the child's output the post-load checks scan (the offload line,
# the Flash-Attention WARN).
_LOAD_TAIL_CHARS = 20000


class CUDA_Engine(BaseLlamaCppEngine):
    """Engine for NVIDIA CUDA GPU inference.

    Provides hardware detection and performance evaluation for NVIDIA GPUs
    using PyTorch CUDA backend and NVML monitoring library.

    Hardware Specs Reference Tables:
        CUDA_PER_SM: CUDA cores and Tensor cores per SM by compute capability
        TC_OPS: Tensor core operations per precision by compute capability

    Note:
        Inference uses a CUDA-compiled llama-server subprocess (same
        architecture as CPU_Engine). The window is sized for a full offload
        against the GPU's memory; llama.cpp's fit places the layers.
    """

    # --- BaseChatServerEngine / BaseLlamaCppEngine config overrides ---
    _server_name = "llama-server"
    _tokenizer_provider = "llama-server-cuda"
    _use_cuda_build = True  # → BaseLlamaCppEngine._default_install_dir picks the cuda/ artifact

    # The last spawn context was a load sized above the floor: an
    # out-of-memory death at its load is retried once at the floor.
    _oom_retryable: ClassVar[bool] = False

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
    def _prepare_spawn_context(
        cls,
        model_path: Path,
        alias: str,
        *,
        brake_cap: Optional[int] = None,
        respawn_from: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Resolve the per-spawn CUDA context -- the spawn matrix:

        * a brake respawn (``respawn_from``, the live handle of the same
          alias): ``-c = min(its static window, max(floor, cap))``, nothing
          re-read from NVML, every layer pinned (``-ngl -2``: fit then keeps
          the placement and WDDM spills instead of failing) -- Windows only,
          the spill being the one thing that drives a respawn;
        * ``ERUDI_CTX`` pinned: ``-c`` verbatim, ``-ngl -1``, never resized;
        * a load: ``-c`` sized for a full offload against the GPU's memory
          now (``cuda_memory.window_for_vram``), folded with ``brake_cap`` or
          the cap remembered for ``alias``, ``-ngl -1``;
        * the floor does not fit, the GGUF cannot be modelled (MLA) or read,
          or NVML is unreadable: no ``-c``, ``-ngl -1`` -- fit sizes the
          window and places the layers -- and the brake off
          (``window_sized=False``), ONE WARNING;
        * the user's ``CUDA_VISIBLE_DEVICES`` names several GPUs (llama.cpp
          splits the layers across them, which a one-card sizing cannot
          model): no ``-c``, ``-ngl -1``, the brake off, ONE INFO;
        * NVML sees no GPU: no ``-c``, ``-ngl 0``, the brake off.

        ``-ngl -1`` is "auto" while fit is on (llama.cpp ``common.h:473``);
        at the model level any negative value offloads every layer."""
        ctx: Dict[str, Any] = {"threads": max(1, os.cpu_count() or 1)}
        cls._oom_retryable = False
        if (
            respawn_from is not None
            and respawn_from.get("alias") == alias
            and respawn_from.get("window_sized")
            and not respawn_from.get("window_pinned")
        ):
            ctx.update(cls._respawn_context(model_path, respawn_from, brake_cap))
            return ctx
        pinned = cls.max_context_tokens()
        if pinned is not None:
            ctx.update(ctx_size=pinned, window_pinned=True)
        device = cls._device_for_sizing()
        if not device.present:
            if device.unavailable is None:
                # The CUDA engine was selected, yet NVML now sees no device:
                # every generation runs on the processor from here on, which
                # the user will experience as a slow model and nothing else.
                logger.warning(
                    f"[{cls.__name__}] NVML reports no GPU; llama-server starts with 0 GPU "
                    f"layers (processor only)"
                )
                ctx.update(gpu_layers=0)
            else:
                logger.warning(
                    f"[{cls.__name__}] GPU memory unreadable ({device.unavailable}); "
                    f"llama-server's fit places the layers and sizes the window itself; "
                    f"the memory brake is off"
                )
                ctx.update(gpu_layers=-1)
            ctx.setdefault("ctx_size", None)
            ctx.update(window_sized=False)
            return ctx
        if device.visible_devices > 1:
            # The user's CUDA_VISIBLE_DEVICES names several cards: llama.cpp
            # splits the layers across them, which a one-card sizing and
            # measurement cannot model.
            logger.info(
                f"[{cls.__name__}] CUDA_VISIBLE_DEVICES leaves several GPUs visible "
                f"({device.visible_devices}): llama-server's fit places the model across them "
                f"and sizes the window; the memory brake is off for this model"
            )
            ctx.setdefault("ctx_size", None)
            ctx.update(gpu_layers=-1, window_sized=False)
            return ctx
        ctx.update(gpu_layers=-1, gpu_uuid=device.uuid)
        mmproj = cls._find_mmproj(model_path)
        inputs = gguf_memory.read_sizing_inputs(model_path, mmproj)
        if pinned is not None:
            ctx.update(window_sized=True)
            window = (
                cuda_memory.pinned_window(
                    inputs[0],
                    window=pinned,
                    weights_bytes=inputs[1],
                    vram_total_bytes=device.total_bytes,
                    vram_free_bytes=device.free_bytes,
                )
                if inputs is not None
                else None
            )
            if window is not None:
                ctx.update(
                    cls._window_facts(
                        window, **cls._load_facts(model_path, inputs[0], mmproj is not None)
                    )
                )
            return ctx
        if inputs is None:
            ctx.update(ctx_size=None, window_sized=False)
            return ctx
        shape, weights = inputs
        if brake_cap is None:
            entry = cls._window_memory_entry(alias)
            brake_cap = entry.get("cap") if entry else None
        window = cuda_memory.window_for_vram(
            shape,
            weights_bytes=weights,
            vram_total_bytes=device.total_bytes,
            vram_free_bytes=device.free_bytes,
            brake_cap=brake_cap,
        )
        if window is None:
            if shape.kv_lora_rank is not None:
                reason = "MLA"
            elif gguf_memory.kv_bytes_per_token(shape) is None:
                reason = "no attention KV cache"
            else:
                reason = (
                    f"the {MIN_KV_BOUND_TOKENS}-token floor does not fit "
                    f"{device.free_bytes // _MIB} MiB of free VRAM"
                )
            logger.warning(
                f"[{cls.__name__}] Window not sized for {model_path} (architecture "
                f"{shape.architecture}, {reason}); llama-server's fit sizes the window and "
                f"places the layers; the memory brake is off"
            )
            ctx.update(ctx_size=None, window_sized=False)
            return ctx
        logger.info(
            f"[{cls.__name__}] Window sized at spawn: window={window.window}, "
            f"trained={window.trained}, memory_bound={window.memory_bound}, "
            f"vram_free_mib={device.free_bytes // _MIB}, "
            f"vram_total_mib={device.total_bytes // _MIB}, fa_slope={window.fa_proven}"
            + (f", brake_cap={brake_cap}" if brake_cap else "")
        )
        ctx.update(ctx_size=window.window, window_sized=True)
        ctx.update(
            cls._window_facts(window, **cls._load_facts(model_path, shape, mmproj is not None))
        )
        cls._oom_retryable = window.window > MIN_KV_BOUND_TOKENS
        return ctx

    # The handle facts carried over from the load to every respawn: they do
    # not depend on the window.
    _CARRIED_FACTS = ("block_count", "mmproj_loaded", "tok_embd_bytes", "exps_bytes")

    @staticmethod
    def _window_facts(window: cuda_memory.VramWindow, **carried: Any) -> Dict[str, Any]:
        """The facts the brake and the post-load checks read back off the
        handle: the window's, and ``carried`` (``_CARRIED_FACTS``)."""
        return {
            "trained_context_tokens": window.trained,
            "static_window_tokens": window.static_window,
            "kv_token_bytes": window.kv_token_bytes,
            "weights_bytes": window.weights_bytes,
            "spawn_memory_bound": window.memory_bound,
            "vram_total_bytes": window.vram_total_bytes,
            "device_budget_bytes": window.device_budget_bytes,
            "static_device_bytes": window.static_device_bytes,
            "fa_proven": window.fa_proven,
            "kv_classify_bytes": window.kv_classify_bytes,
            **carried,
        }

    @classmethod
    def _load_facts(
        cls, model_path: Path, shape: gguf_memory.GgufShape, mmproj_loaded: bool
    ) -> Dict[str, Any]:
        """The ``_CARRIED_FACTS`` of a load. The tensor bytes of the
        classification's floor are None when the tensor infos cannot be read
        (the bytes then classify nothing)."""
        try:
            tensors: Optional[gguf_memory.TensorBytes] = gguf_memory.read_tensor_bytes(model_path)
        except Exception as exc:
            # Degraded, not failed: the placement then cannot be read from
            # the bytes and is assumed full (DEBUG at the classification).
            logger.debug(
                f"[{cls.__name__}] GGUF tensor infos unreadable at {model_path} "
                f"({type(exc).__name__})"
            )
            tensors = None
        return {
            "block_count": shape.block_count,
            "mmproj_loaded": mmproj_loaded,
            "tok_embd_bytes": tensors.tok_embd_bytes if tensors else None,
            "exps_bytes": tensors.exps_bytes if tensors else None,
        }

    @classmethod
    def _respawn_context(
        cls, model_path: Path, live: Dict[str, Any], cap: Optional[int]
    ) -> Dict[str, Any]:
        """A brake respawn's context: the live child's static window folded
        with the cap -- no NVML sizing, so a series of respawns is
        deterministic and never reads the free VRAM a hog left -- and its
        other facts carried over (the GGUF header is re-read for the
        window-dependent ones: deterministic). Every layer is pinned on the
        GPU (``-ngl -2``): fit cannot change an explicit ``-ngl`` and keeps
        the placement, and WDDM spills instead of failing. Only a spill
        drives a respawn, so it happens on Windows only; Linux, where an
        allocation that does not fit fails, never respawns. Never the
        fallback shape: at worst the floor. Raises when the GGUF cannot be
        read (the respawn then fails like an unload)."""
        shape = gguf_memory.read_shape(model_path)
        window = cuda_memory.respawn_window(
            shape,
            static_window=live["static_window_tokens"],
            cap=cap,
            weights_bytes=live["weights_bytes"],
            vram_total_bytes=live["vram_total_bytes"],
            device_budget_bytes=live.get("device_budget_bytes"),
            memory_bound=bool(live.get("spawn_memory_bound")),
        )
        if window is None:
            raise EngineException(message=f"KV cache of {model_path} not modelled at respawn")
        gpu_layers = -2
        logger.info(
            f"[{cls.__name__}] Window sized at respawn: window={window.window}, "
            f"static_window={window.static_window}, brake_cap={cap}, gpu_layers={gpu_layers}"
        )
        ctx = cls._window_facts(window, **{key: live.get(key) for key in cls._CARRIED_FACTS})
        ctx.update(
            ctx_size=window.window,
            gpu_layers=gpu_layers,
            window_sized=True,
            brake_respawn=True,
            gpu_uuid=live.get("gpu_uuid"),
        )
        return ctx

    @classmethod
    def _start_server(
        cls,
        *,
        model_path: Path,
        alias: str,
        port: int,
        probe_timeout_s: Optional[float] = None,
        brake_cap: Optional[int] = None,
        respawn_from: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """The shared spawn, plus ONE retry at the floor window when a load
        sized above the floor dies out of memory (``CUDA_OUT_OF_MEMORY``).
        A respawn, a retry, a pinned or fit-sized child is never retried;
        a second out-of-memory death is the load error."""
        try:
            return super()._start_server(
                model_path=model_path,
                alias=alias,
                port=port,
                probe_timeout_s=probe_timeout_s,
                brake_cap=brake_cap,
                respawn_from=respawn_from,
            )
        except EngineException as exc:
            if (
                exc.engine_code != "CUDA_OUT_OF_MEMORY"
                or respawn_from is not None
                or brake_cap is not None
                or not cls._oom_retryable
            ):
                raise
            logger.warning(
                f"[{cls.__name__}] llama-server ran out of memory at load (on the GPU, or on "
                f"the host: a pinned-buffer cudaHostAlloc failure prints the same); retrying "
                f"once at the {MIN_KV_BOUND_TOKENS}-token floor window"
            )
            return super()._start_server(
                model_path=model_path,
                alias=alias,
                port=cls._pick_free_port(),
                probe_timeout_s=probe_timeout_s,
                brake_cap=MIN_KV_BOUND_TOKENS,
            )

    @classmethod
    def _build_spawn_argv(
        cls,
        *,
        llama_server: Path,
        model_gguf: Path,
        alias: str,
        port: int,
        ctx_size: Optional[int] = None,
        threads: int = 1,
        gpu_layers: int = -1,
        **_ignored: Any,
    ) -> List[Any]:
        """CUDA CLI for llama-server: the window and the layers the spawn
        context chose, and fit's per-device margin.

        ``-c`` is the window ``_prepare_spawn_context`` chose (sized for a
        full offload, a brake respawn's, or ``ERUDI_CTX``); llama.cpp's fit
        never changes an explicit ``-c``. With ``ctx_size=None`` (the
        fallback shape) the flag is omitted and fit resolves the window at
        load; ``_read_server_properties`` reads the allocated value back
        after the probe. ``-ngl`` is ``-1`` (auto: fit places the layers),
        ``-2`` (all: a Windows brake respawn) or ``0`` (no GPU). ``-fitt``
        pins fit's per-device margin to ``CUDA_FIT_TARGET_MIB`` on every
        spawn.

        ``--jinja`` enables the model's own chat template and with it
        OpenAI-style function calling (the agent's calculator tool) —
        without it llama-server never emits ``tool_calls``.

        ``-np 1 --cache-ram 0`` (one slot, no prompt cache) is not here: the
        shared ``BaseLlamaCppEngine._spawn_child`` appends it, with the
        authentication flags, to every llama-server it spawns.

        ``--reasoning-format`` is deliberately NOT passed (#554): the default
        (``auto``) is llama-server's per-family reasoning extraction into the
        dedicated ``delta.reasoning_content`` stream field -- it knows each
        family's markers, re-injects a prompt-opened thinking block on every
        post-tool hop, and closes an unterminated block cleanly at EOS.
        ``Erudi_Chat_OpenAI`` (``src.agents.chat_model``) carries the field to
        the runner's ``thinking`` events; the runner's ThinkSplitter stays as
        the inline fallback for families the server parser does not know.
        """
        argv: List[Any] = [
            str(llama_server),
            "-m",
            str(model_gguf),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--alias",
            alias,
        ]
        if ctx_size is not None:
            argv += ["-c", str(ctx_size)]
        argv += [
            "--threads",
            str(threads),
            "-ngl",
            str(gpu_layers),
            "-fitt",
            str(cuda_memory.CUDA_FIT_TARGET_MIB),
            "--jinja",
        ]
        return argv

    @classmethod
    def _build_spawn_env(cls, *, gpu_uuid: Optional[str] = None, **_ignored: Any) -> Dict[str, str]:
        """Prepend the CUDA toolkit `bin/` to PATH so llama-server finds
        the runtime DLLs (cuBLAS, cuDNN, etc.), and pin the GPU the window
        was sized on: ``CUDA_VISIBLE_DEVICES=GPU-<uuid>`` (immune to any
        ordering difference between NVML and the CUDA runtime) -- never over
        a value the user set, which the selection already honoured (an
        empty value is unset)."""
        env = os.environ.copy()
        cuda_bin = cls._resolve_cuda_bin_dir()
        if cuda_bin:
            env["PATH"] = str(cuda_bin) + os.pathsep + env.get("PATH", "")
            logger.debug(f"[CUDA_Engine] Prepended CUDA bin to PATH: {cuda_bin}")
        if gpu_uuid and not env.get("CUDA_VISIBLE_DEVICES"):
            env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
        return env

    # ---------- the GPU's memory (the ONE NVML seam of the spawn path) ----------

    @classmethod
    def _host_is_windows(cls) -> bool:
        """WDDM spills an allocation that does not fit; Linux fails it."""
        return platform.system() == "Windows"

    @staticmethod
    def _nvml_memory(nv: Any, handle: Any) -> Any:
        """``nvmlDeviceGetMemoryInfo``, v2 when the binding and the driver
        support it (its ``free`` excludes the driver's reserved memory)."""
        version = getattr(nv, "nvmlMemory_v2", None)
        if version is not None:
            try:
                return nv.nvmlDeviceGetMemoryInfo(handle, version=version)
            except Exception:
                # A driver without v2: v1 answers the same total and free.
                pass
        return nv.nvmlDeviceGetMemoryInfo(handle)

    @staticmethod
    def _nvml_processes(nv: Any, handle: Any) -> Optional[Tuple[Tuple[int, Optional[int]], ...]]:
        """The device's compute processes as ``(pid, usedGpuMemory)``; None
        when NVML cannot list them (WDDM, an old driver)."""
        try:
            return tuple(
                (
                    int(process.pid),
                    process.usedGpuMemory if isinstance(process.usedGpuMemory, int) else None,
                )
                for process in nv.nvmlDeviceGetComputeRunningProcesses(handle)
            )
        except Exception:
            # Unknown, not empty: the readers fall back on the device totals.
            return None

    @classmethod
    def _read_vram_info(cls) -> cuda_memory.DeviceMemory:
        """The selected GPU's memory now: the largest card among those the
        child may see (a user's ``CUDA_VISIBLE_DEVICES`` restricts them; its
        integer entries are taken as NVML indexes), its UUID, total and free
        bytes, and its compute processes. Silent: an unreadable NVML is an
        ``unavailable`` reason the caller logs once."""
        no_gpu = cuda_memory.DeviceMemory(index=None, uuid=None, total_bytes=0, free_bytes=0)
        try:
            import pynvml as nv
        except ImportError:
            return cuda_memory.DeviceMemory(
                index=None, uuid=None, total_bytes=0, free_bytes=0, unavailable="pynvml missing"
            )
        try:
            nv.nvmlInit()
            devices = []
            for idx in range(nv.nvmlDeviceGetCount()):
                handle = nv.nvmlDeviceGetHandleByIndex(idx)
                uuid = nv.nvmlDeviceGetUUID(handle)
                if isinstance(uuid, bytes):
                    uuid = uuid.decode("ascii", errors="replace")
                devices.append({"id": idx, "uuid": uuid, "handle": handle})
            # Empty is unset: backend/.env.example lists the variable empty.
            visible = os.environ.get("CUDA_VISIBLE_DEVICES") or None
            if visible and any(entry.strip().isdigit() for entry in visible.split(",")):
                logger.debug(
                    f"[{cls.__name__}] CUDA_VISIBLE_DEVICES={visible}: integer entries are "
                    f"taken as NVML indexes (CUDA's order may differ)"
                )
            candidates = cuda_memory.visible_devices(devices, visible)
            if candidates is None:
                return cuda_memory.DeviceMemory(
                    index=None,
                    uuid=None,
                    total_bytes=0,
                    free_bytes=0,
                    unavailable="CUDA_VISIBLE_DEVICES names no device NVML lists",
                )
            if not candidates:
                return no_gpu
            for device in candidates:
                memory = cls._nvml_memory(nv, device["handle"])
                device.update(
                    total=int(memory.total),
                    free=int(memory.free),
                    vram_total_mb=int(memory.total) / _MIB,
                )
            best = cls._select_best_gpu(candidates)
            return cuda_memory.DeviceMemory(
                index=best["id"],
                uuid=best["uuid"],
                total_bytes=best["total"],
                free_bytes=best["free"],
                processes=cls._nvml_processes(nv, best["handle"]),
                visible_devices=len(candidates) if visible else 1,
            )
        except Exception as exc:
            return cuda_memory.DeviceMemory(
                index=None,
                uuid=None,
                total_bytes=0,
                free_bytes=0,
                unavailable=f"NVML read failed ({type(exc).__name__})",
            )

    @classmethod
    def _device_for_sizing(cls) -> cuda_memory.DeviceMemory:
        """The GPU memory a load is sized against: NVML's, or the QA seam
        ``ERUDI_VRAM_OVERRIDE_GB`` standing in for the free VRAM (the total
        is then the larger of the override and the real total). The override
        never invents a GPU: when NVML failed or sees none, it is ignored.
        An empty value is unset."""
        device = cls._read_vram_info()
        raw = os.environ.get("ERUDI_VRAM_OVERRIDE_GB", "").strip()
        if not raw or not device.present:
            return device
        try:
            free = int(float(raw) * 1024**3)
        except ValueError:
            logger.warning(f"[ERUDI_VRAM_OVERRIDE_GB] Invalid value '{raw}', ignoring.")
            return device
        logger.info(f"[ERUDI_VRAM_OVERRIDE_GB] Simulating {float(raw):.1f} GB free VRAM.")
        return dataclasses.replace(
            device, total_bytes=max(free, device.total_bytes), free_bytes=free
        )

    # ---------- after load: the placement and the post-load checks ----------

    @classmethod
    def _read_server_properties(cls, handle: Dict[str, Any]) -> None:
        """The shared ``/props`` read, then -- on a child whose window Erudi
        sized -- the placement classification BEFORE the brake seed, the
        spill baseline and the memory-bound stamp: a partial offload (some
        KV off the device) runs brake-off. A full child is seeded, its spill
        baseline stamped and its device bytes checked against the window
        budget; the Flash-Attention WARN is looked for in the child's
        output. Whether Flash Attention runs (proven for the model and not
        disabled by llama.cpp) is stamped first: the healthy Shared GPU
        memory the device bytes, the baseline and every later spill reading
        are counted from depends on it. Degrades, never raises."""
        cls._read_props(handle)
        drainer = cls._output_drainer_for(handle.get("proc"))
        tail = drainer.tail(max_chars=_LOAD_TAIL_CHARS) if drainer is not None else None
        handle["fa_on"] = bool(handle.get("fa_proven")) and not cuda_memory.fa_disabled_in(tail)
        device_bytes = spill_bytes = None
        if handle.get("window_sized"):
            try:
                device_bytes, spill_bytes = cls._measure_device(handle)
                cls._classify_placement(handle, tail, device_bytes)
            except Exception as exc:
                # The model is loaded and serves: an unreadable placement is
                # assumed full, the brake keeps its spill evidence.
                logger.warning(
                    f"[{cls.__name__}] Placement check failed ({type(exc).__name__}); "
                    f"assuming a full offload (port {handle.get('port')})"
                )
        if handle.get("window_sized"):
            cls._llama_seed(handle)
            cls._stamp_spill_baseline(handle, spill_bytes)
        else:
            # Brake-off: the child forgets an unload's record like any spawn.
            cls._brake_reload = None
        # The budget bound the WINDOW; fit dropping layers says nothing of it.
        handle["memory_bound_tokens"] = (
            handle["context_tokens"] if handle.get("spawn_memory_bound") else None
        )
        if handle.get("window_sized"):
            cls._llama_check_footprint(handle)
            cls._check_device(handle, device_bytes)
        cls._check_flash_attention(handle, tail)

    @classmethod
    def _measure_device(cls, handle: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
        """The child's device bytes and its raw Shared GPU memory reading:
        Windows counts its dedicated VRAM plus its shared memory above the
        healthy level of its window (PDH; NVML has no per-process memory
        under WDDM), Linux reads NVML's process list (and has no spill)."""
        pid = handle.get("pid")
        if not cls._host_is_windows():
            return memory_pressure.child_vram_bytes(pid, cls._read_vram_info().processes), None
        healthy = cls._healthy_shared_bytes(handle)
        vram = memory_pressure.read_vram(pid, healthy_bytes=healthy)
        if vram.dedicated_bytes is None or vram.spill_bytes is None:
            return None, vram.spill_bytes
        device = cuda_memory.windows_device_bytes(
            dedicated_bytes=vram.dedicated_bytes,
            shared_bytes=vram.spill_bytes,
            healthy_bytes=healthy,
        )
        return device, vram.spill_bytes

    @staticmethod
    def _healthy_shared_bytes(handle: Dict[str, Any]) -> int:
        """The Shared GPU memory the loaded child holds when healthy: the
        fixed part plus the host slope of its window, Flash Attention on or
        off (``fa_on``, stamped at load)."""
        return cuda_memory.healthy_shared_bytes(
            handle["context_tokens"], fa_on=bool(handle.get("fa_on"))
        )

    @classmethod
    def _classify_placement(
        cls, handle: Dict[str, Any], tail: Optional[str], device_bytes: Optional[int]
    ) -> None:
        """Full or partial offload: llama.cpp's ``offloaded N/M`` line when
        it reached the drainer, else the device bytes against the floor of
        a full offload. Partial (some KV off the device) turns the brake off
        for this child -- ONE INFO; the window memory keeps its cap. Neither
        signal: assumed full, one DEBUG."""
        port = handle.get("port")
        after = " after a brake respawn" if handle.get("brake_respawn") else ""
        line = cuda_memory.placement_from_tail(tail)
        if line is not None:
            partial = line[0] < line[1]
            signal = f"signal=line, offloaded {line[0]}/{line[1]} layers"
        else:
            terms = [
                handle.get(key)
                for key in ("weights_bytes", "tok_embd_bytes", "exps_bytes", "block_count")
            ]
            if device_bytes is None or any(term is None for term in terms):
                logger.debug(
                    f"[{cls.__name__}] GPU placement unknown (no offload line, no device "
                    f"reading); assuming a full offload (port {port})"
                )
                return
            weights, tok_embd, exps, blocks = terms
            predicted = cuda_memory.predicted_device_floor_bytes(
                weights_bytes=weights,
                tok_embd_bytes=tok_embd,
                exps_bytes=exps,
                kv_classify_bytes=handle.get("kv_classify_bytes"),
            )
            partial = cuda_memory.placement_partial(
                predicted_bytes=predicted, block_count=blocks, measured_bytes=device_bytes
            )
            signal = (
                f"signal=bytes, measured_mib={device_bytes // _MIB}, "
                f"predicted_floor_mib={predicted // _MIB}"
            )
        if partial:
            handle["window_sized"] = False
            logger.info(
                f"[{cls.__name__}] llama-server placed the model partly on the GPU{after} "
                f"({signal}); the memory brake is off for this child (port {port})"
            )
        else:
            logger.info(
                f"[{cls.__name__}] GPU placement: full offload{after} ({signal}) (port {port})"
            )

    @classmethod
    def _stamp_spill_baseline(cls, handle: Dict[str, Any], spill_bytes: Optional[int]) -> None:
        """The child's spill baseline: its Shared GPU memory at load,
        bounded by the healthy level of its window
        (``cuda_memory.healthy_shared_bytes``) -- that level itself when the
        load reading is missing, never 0: every later reading counts the
        spill above it. A load reading above the healthy level is spill
        already present -- a WARNING at a load (the budget was too large or
        the GPU crowded), an INFO after a brake respawn (expected under spill
        pressure)."""
        healthy = cls._healthy_shared_bytes(handle)
        handle["spill_baseline"] = healthy if spill_bytes is None else min(spill_bytes, healthy)
        if spill_bytes is None or spill_bytes <= healthy:
            return
        respawned = bool(handle.get("brake_respawn"))
        (logger.info if respawned else logger.warning)(
            f"[{cls.__name__}] Shared GPU memory above the healthy level at load: "
            f"shared_mib={spill_bytes // _MIB}, healthy_mib={healthy // _MIB} "
            f"(window={handle.get('context_tokens')}, flash_attention={handle.get('fa_on')}); "
            + (
                "expected after a brake respawn under spill pressure"
                if respawned
                else "the window budget was too large or the GPU was crowded"
            )
            + f" (port {handle.get('port')})"
        )

    @classmethod
    def _check_device(cls, handle: Dict[str, Any], device_bytes: Optional[int]) -> None:
        """Right after load: the child's device bytes against what the
        window predicts (KV, compute, the CUDA context, the weights), ONE
        WARNING above it -- a mis-calibrated CUDA constant or a
        Flash-Attention mis-prediction. Never an action; never raises."""
        try:
            if handle.get("mmproj_loaded"):
                logger.debug(
                    f"[{cls.__name__}] Post-load device check skipped: a vision projector "
                    f"is loaded (its compute buffer is not modelled)"
                )
                return
            static, weights = handle.get("static_device_bytes"), handle.get("weights_bytes")
            if device_bytes is None or not static or not weights:
                return
            limit = cuda_memory.device_check_limit_bytes(
                static_device_bytes=static, weights_bytes=weights
            )
            if device_bytes > limit:
                logger.warning(
                    f"[{cls.__name__}] llama-server holds more device memory than the window "
                    f"budget predicted: device_bytes={device_bytes}, "
                    f"predicted_limit_bytes={limit}, window={handle.get('context_tokens')} "
                    f"(port {handle.get('port')})"
                )
        except Exception as exc:
            # The check is advisory: the model is loaded and serves.
            logger.warning(f"[{cls.__name__}] Post-load device check failed ({type(exc).__name__})")

    @classmethod
    def _check_flash_attention(cls, handle: Dict[str, Any], tail: Optional[str]) -> None:
        """The window was sized on the Flash-Attention slope, yet llama.cpp
        disabled FA at context creation (its WARN in the child's output):
        ONE WARNING -- the compute buffer is larger than planned."""
        if handle.get("fa_proven") and cuda_memory.fa_disabled_in(tail):
            logger.warning(
                f"[{cls.__name__}] llama-server disabled Flash Attention for this model: the "
                f"window was sized on the Flash-Attention compute slope, its compute buffer "
                f"is larger than planned (port {handle.get('port')})"
            )

    # ---------- the memory brake (bodies on BaseLlamaCppEngine) ----------

    @classmethod
    def _llama_read_signals(cls, handle: Dict[str, Any]) -> Optional[LlamaSignals]:
        """The Windows spill gauge in the law's terms, and nothing else: the
        Shared GPU memory above the child's baseline is the level (and, at
        warning or worse, the damage), the window's KV over the VRAM the
        share a smaller ``-c`` returns, the child's dedicated + shared bytes
        over the VRAM its footprint. No RAM sensor is read: on a full
        offload the KV and the compute buffer are on the card, so RAM
        pressure is not the child's to relieve. None -- the brake off for
        this sample, ONE WARNING per process -- when the gauge is
        unreadable. Reached on Windows only (``brake_point``)."""
        healthy = cls._healthy_shared_bytes(handle)
        vram = memory_pressure.read_vram(handle.get("pid"), healthy_bytes=healthy)
        if vram.spill_bytes is None:
            cls._llama_warn_sensor_off(vram.unavailable, sensor="GPU spill gauge")
            return None
        signals = cuda_memory.spill_signals(max(0, vram.spill_bytes - handle["spill_baseline"]))
        device = (
            vram.dedicated_bytes + vram.spill_bytes if vram.dedicated_bytes is not None else None
        )
        vram_total = handle.get("vram_total_bytes")
        window, kv = handle.get("context_tokens"), handle.get("kv_token_bytes")
        return LlamaSignals(
            level=signals.level,
            damage_bytes=signals.damage_bytes,
            damage_available=signals.damage_available,
            kv_share=window * kv / vram_total if window and kv and vram_total else None,
            footprint_share=device / vram_total if device is not None and vram_total else None,
            footprint_bytes=device,
            ram_damage_total=None,
        )

    @classmethod
    def brake_point(cls, point: str, *, llm_id: Any = None) -> BrakeOutcome:
        """Take the brake's sample at ``point`` and execute its decision --
        on Windows only, and only on a child whose window Erudi sized and
        that loaded in full: a fit-sized or partial child is never sampled
        (nothing for the law's unload to act on either). Off Windows there
        is no spill, so no signal: nothing is read and nothing acts (the idle
        monitor still ticks; its samples return here). Never raises: a
        failure is ONE WARNING with the exception's class name, and the
        sample is lost."""
        handle = cls._model
        if (
            not cls._host_is_windows()
            or not isinstance(handle, dict)
            or not handle.get("window_sized")
        ):
            return BrakeOutcome()
        try:
            return cls._llama_brake_point(point, llm_id)
        except Exception as exc:
            logger.warning(
                f"[{cls.__name__}] Memory brake sample failed ({type(exc).__name__}): "
                f"point={point}"
            )
            return BrakeOutcome()

    @classmethod
    def memory_bound_tokens(cls) -> Optional[int]:
        return cls._llama_memory_bound_tokens()

    @classmethod
    def memory_bound_footprint_bytes(cls) -> Optional[int]:
        return cls._llama_memory_bound_footprint_bytes()

    @classmethod
    def begin_memory_window(cls) -> Any:
        return cls._llama_begin_memory_window()

    @classmethod
    def brake_reload_pending(cls, llm_id: Any) -> bool:
        """An unload's record for ``llm_id`` (the shared answer), or a brake
        respawn of its loaded child whose reload state is still owed."""
        return super().brake_reload_pending(llm_id) or cls._llama_respawn_pending(llm_id)

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
