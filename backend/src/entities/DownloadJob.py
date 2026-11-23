"""SQLAlchemy entity for LLM download job tracking with progress and ETA.

Represents a background download task for fetching and quantizing LLMs from HuggingFace.
Tracks progress, timing, file paths, and error states.

Example:
    from src.entities.DownloadJob import DownloadJobModel

    job = DownloadJobModel(
        remote_model_id="meta-llama/Llama-3-8B-Instruct",
        remote_model_link="https://huggingface.co/meta-llama/Llama-3-8B-Instruct",
        local_model_id=42,
        status="pending"
    )
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, Float, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import validates
from src.database.core import Base


class DownloadJobModel(Base):
    """SQLAlchemy model for LLM download jobs with progress tracking and file paths.

    Tracks the lifecycle of downloading and quantizing an LLM from HuggingFace, including
    progress percentage, ETA, temp/final file paths, and error messages.

    Attributes:
        id: Primary key (auto-increment).
        remote_model_id: HuggingFace repo ID (e.g., "meta-llama/Llama-3-8B-Instruct").
        local_model_id: Database ID of temp Llm entry created during download.
        remote_model_link: HuggingFace URL or API endpoint.
        temp_local_model_link: Temp directory for full-precision download.
        final_local_model_link: Final directory for quantized model.
        status: Job state - "pending", "running", "completed", "failed", "cancelled".
        error_message: Exception message if status="failed" (English, for the
            log and Diagnostics; never shown to the user).
        error_code: Stable code of the failure if status="failed" (one of
            ``DOWNLOAD_ERROR_CODES`` in ``src/domains/llms/services.py``); the
            interface shows its sentence for it. None on a job that failed
            before the column existed.
        created_at: Job creation timestamp (server default).
        updated_at: Last status update timestamp (auto-updated).
        total_bytes: Total download size in bytes.
        progress: Download progress percentage (0.0-100.0).
        total_time_elapsed: Seconds elapsed since job start.
        time_left: Estimated seconds remaining.

    Constraints:
        - status must be one of: pending, running, completed, failed, cancelled.
        - progress must be between 0.0 and 100.0.
        - total_bytes, total_time_elapsed, time_left must be non-negative.
        - remote_model_id and remote_model_link must not be empty.

    Example:
        >>> job = DownloadJobModel(remote_model_id="Qwen/Qwen2.5-7B-Instruct", status="pending")
        >>> job.progress = 50.0
        >>> job.time_left = 120.5
    """

    __tablename__ = "download_jobs"

    id = Column(Integer, primary_key=True, index=True)
    remote_model_id = Column(String, nullable=False)
    # Audit record: survives deletion of the temp Llm (server-side SET NULL).
    local_model_id = Column(Integer, ForeignKey("llms.id", ondelete="SET NULL"), nullable=True)
    remote_model_link = Column(String, nullable=False)
    temp_local_model_link = Column(String, nullable=True)
    final_local_model_link = Column(String, nullable=True)
    status = Column(String, default="pending", nullable=False)
    error_message = Column(Text, nullable=True)
    error_code = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    total_bytes = Column(Float, default=0.0, nullable=False)
    progress = Column(Float, default=0.0, nullable=False)
    total_time_elapsed = Column(Float, default=0.0, nullable=False)
    time_left = Column(Float, default=0.0, nullable=False)

    @validates("status")
    def validate_status(self, key, value):
        """Ensure status is one of the allowed values.

        Args:
            key: Column name being validated ('status').
            value: Proposed string value for job status.

        Returns:
            str: The validated status value.

        Raises:
            ValueError: If status is not in allowed list.
        """
        allowed = ["pending", "running", "completed", "failed", "cancelled"]
        if value not in allowed:
            raise ValueError(f"Invalid status: {value}. Must be one of {allowed}")
        return value

    @validates("progress")
    def validate_progress(self, key, value):
        """Ensure progress is between 0.0 and 100.0.

        Args:
            key: Column name being validated ('progress').
            value: Proposed float value for download progress percentage.

        Returns:
            float: The validated progress value.

        Raises:
            ValueError: If progress is outside 0.0-100.0 range.
        """
        if not (0.0 <= value <= 100.0):
            raise ValueError(f"Progress must be between 0.0 and 100.0, got {value}")
        return value

    @validates("total_bytes", "total_time_elapsed", "time_left")
    def validate_non_negative(self, key, value):
        """Ensure numeric fields are non-negative.

        Args:
            key: Column name being validated (one of: total_bytes, total_time_elapsed, time_left).
            value: Proposed numeric value.

        Returns:
            float: The validated non-negative value.

        Raises:
            ValueError: If value is negative.
        """
        if value < 0:
            raise ValueError(f"{key} must be non-negative, got {value}")
        return value

    @validates("remote_model_id", "remote_model_link")
    def validate_not_empty(self, key, value):
        """Ensure required string fields are not empty.

        Args:
            key: Column name being validated (one of: remote_model_id, remote_model_link).
            value: Proposed string value.

        Returns:
            str: The validated stripped string value.

        Raises:
            ValueError: If value is empty or whitespace-only.
        """
        if not value or not value.strip():
            raise ValueError(f"{key} cannot be empty")
        return value.strip() if isinstance(value, str) else value
