"""Response models for ``GET /erudi/diagnostics/``."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class DiagnosticsEnvironment(BaseModel):
    """What this backend is running on, right now.

    Every field but ``platform`` is optional: the panel that reads this is
    opened when something is broken, and a partial answer beats an error.
    """

    platform: str = Field(description='Operating system family ("Darwin", "Windows", "Linux").')
    platform_release: Optional[str] = Field(default=None, description="OS release string.")
    architecture: Optional[str] = Field(default=None, description='CPU architecture ("arm64").')
    python_version: Optional[str] = Field(default=None, description="Backend Python version.")
    frozen: bool = Field(default=False, description="True in a PyInstaller build.")
    engine: Optional[str] = Field(
        default=None, description='Selected engine class ("MLX_Engine", "CUDA_Engine", ...).'
    )
    loaded_model_id: Optional[int] = Field(
        default=None, description="Database id of the model held in memory, if any."
    )
    loaded_model: Optional[str] = Field(
        default=None, description="Repository name of that model, when it can be resolved."
    )
    cpu_model: Optional[str] = Field(default=None, description="CPU as the engine reports it.")
    gpu_name: Optional[str] = Field(default=None, description="GPU as the engine reports it.")
    compute_capability: Optional[str] = Field(default=None, description="CUDA only.")
    vram_total_gb: Optional[float] = Field(default=None, description="CUDA only.")
    db: str = Field(description='Embedded database state: "ok", "recovering" or "failed".')
    backend_log_path: str = Field(description="Absolute path of backend.log on this machine.")


class DiagnosticsLogRecord(BaseModel):
    """One WARNING-or-worse record read from ``backend.log``."""

    timestamp: str = Field(description="UTC ISO-8601 with milliseconds, as written in the file.")
    level: str = Field(description='"WARNING", "ERROR" or "CRITICAL".')
    request_id: Optional[str] = Field(
        default=None, description="Correlation id of the request that produced it, if any."
    )
    message: str = Field(description="Record body, continuation lines included.")


class DiagnosticsResponse(BaseModel):
    """Everything the backend can tell a bug report about itself."""

    environment: DiagnosticsEnvironment
    recent_errors: List[DiagnosticsLogRecord] = Field(
        default_factory=list, description="Oldest first; empty when the log is unreadable."
    )
