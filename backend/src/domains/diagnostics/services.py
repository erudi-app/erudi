"""Assembling the environment summary a bug report needs.

Everything here is best-effort and local. A field the machine cannot answer
comes back as ``None`` rather than failing the request: the panel that reads
this endpoint is opened when something is already broken, and a summary missing
its GPU name is far more useful than a 500.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from src.core import config
from src.core.health import _db_state
from src.core.logging import LOG_FILE_NAME, logger
from src.domains.diagnostics import log_reader
from src.launcher import ensure_runtime_paths_initialized


def backend_log_path() -> Path:
    """Absolute path of ``backend.log`` for this installation.

    Dev builds keep it under ``backend/logs``; packaged builds put it in the
    per-OS user log directory resolved by ``src.launcher.runtime_paths``.
    """
    return ensure_runtime_paths_initialized().log_dir / LOG_FILE_NAME


def _engine_hardware() -> Dict[str, Any]:
    """GPU and CPU identity from the selected engine, or Nones.

    ``get_hardware_info`` reaches out to NVML, ``cpuinfo`` and ``psutil``; any
    of those can be absent or throw on a machine that is already misbehaving,
    which is exactly when this endpoint is called.
    """
    blank = {
        "cpu_model": None,
        "gpu_name": None,
        "compute_capability": None,
        "vram_total_gb": None,
    }
    engine = getattr(config, "LLM_Engine", None)
    if engine is None:
        return blank
    try:
        info = engine.get_hardware_info() or {}
        cpu = info.get("cpu") or {}
        gpu = info.get("gpu") or {}
        return {
            "cpu_model": cpu.get("model"),
            "gpu_name": gpu.get("gpu_name"),
            "compute_capability": gpu.get("compute_capability"),
            "vram_total_gb": gpu.get("vram_total_gb"),
        }
    except Exception as error:
        logger.warning(f"Diagnostics could not read hardware info: {error}")
        return blank


def _loaded_model_name(db: Optional[Session], model_id: Optional[int]) -> Optional[str]:
    """Repository name of the model held by the engine singleton, if any.

    The engine tracks the database id; a bug report needs the name a
    maintainer can recognise.
    """
    if db is None or model_id is None:
        return None
    try:
        from src.entities.Llm import Llm

        llm = db.get(Llm, model_id)
        return llm.name if llm is not None else None
    except Exception as error:
        logger.warning(f"Diagnostics could not resolve the loaded model name: {error}")
        return None


def build_environment(db: Optional[Session] = None) -> Dict[str, Any]:
    """Describe this backend: platform, runtime, engine, model, database, log.

    Args:
        db: Optional session used only to turn the engine's loaded model id
            into a repository name. Omitted, the name comes back as None.

    Returns:
        dict: Flat summary; every value is a plain JSON type or None.
    """
    engine = getattr(config, "LLM_Engine", None)
    model_id = getattr(engine, "_model_id", None) if engine is not None else None

    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "frozen": getattr(sys, "frozen", False),
        "engine": getattr(engine, "__name__", None) if engine is not None else None,
        "loaded_model_id": model_id,
        "loaded_model": _loaded_model_name(db, model_id),
        "db": _db_state(),
        "backend_log_path": str(backend_log_path()),
        **_engine_hardware(),
    }


def build_recent_errors(limit: int = log_reader.DEFAULT_LIMIT) -> List[Dict[str, Any]]:
    """Last ``limit`` WARNING-or-worse records of ``backend.log``, newest last."""
    return log_reader.recent_errors(backend_log_path(), limit=limit)
