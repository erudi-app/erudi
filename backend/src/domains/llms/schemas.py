"""Pydantic validation schemas for LLM management and download jobs.

This module defines data transfer objects (DTOs) for the LLMs domain, separating:
- **LLM metadata**: Model catalog entries (name, link, quantization status).
- **Download jobs**: Background tasks for downloading and quantizing models from HuggingFace.

Schema Hierarchy:
- LLMBase → LLMCreate, LLMResponse (inheritance pattern).
- DownloadJobResponse (standalone, aliased fields for API compatibility).

Model State Encoding (local field):
- 0: Remote only (browsable but not downloaded).
- 1: Local (downloaded and ready for inference).
- 2: Downloading (temporary placeholder during download).

Quantization Status (quantized field):
- False: Not quantized (full precision or unknown).
- True: Pre-quantized (model already in 4-bit/8-bit format).

Example:
    from src.domains.llms.schemas import LLMCreate, DownloadJobResponse
    from fastapi import FastAPI

    app = FastAPI()

    @app.post("/llms", response_model=LLMResponse)
    def create_llm(llm: LLMCreate):
        return db.create(llm)
"""

from pydantic import BaseModel, Field, computed_field
from typing import Any, Dict, List, Optional
from datetime import datetime


class SamplingDefaultsResponse(BaseModel):
    """Resolved per-model sampling defaults (#388), the shape
    ``src.database.generation_hints.SamplingDefaults.to_dict`` produces.

    ``source`` says which capture stage the values came from
    (``base_generation_config`` / ``quant_generation_config`` / ``model_card``)
    or ``none`` when the publisher gives no usable recommendation and the
    neutral defaults apply -- the UI tells the user in that case. ``evidence``
    is the model-card sentence the values were read from (``model_card`` only;
    debugging aid, not displayed). ``top_k`` / ``min_p`` / ``presence_penalty``
    are None unless the captured block defines them (and are then the only case
    the backend puts them on the wire). ``max_tokens_cap`` is the UI ceiling for
    the max-tokens field: min(model context window, engine context window).
    """

    temperature: float
    top_p: float
    max_tokens: int
    repetition_penalty: float
    repetition_context_size: int
    max_tokens_cap: int
    source: str
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    base_repo: Optional[str] = None
    evidence: Optional[str] = None


class LLMBase(BaseModel):
    """Base schema for LLM metadata with minimal required fields.

    Attributes:
        name: Human-readable model name (e.g., "llama-3-8b-instruct").
        local: Download state - 0=remote, 1=local/ready, 2=downloading.
        link: HuggingFace model ID or local filesystem path.
    """

    name: str = Field(..., min_length=1, description="Model name must not be empty")
    local: int = Field(
        ..., ge=0, le=2, description="Must be 0 (remote), 1 (ready), or 2 (downloading)"
    )
    link: str


class LLMCreate(LLMBase):
    """Request schema for creating a new LLM catalog entry.

    Inherits all fields from LLMBase. Used when seeding database from HuggingFace
    or manually registering a model.

    Example:
        >>> llm_data = LLMCreate(name="Qwen2.5-7B", local=0, link="Qwen/Qwen2.5-7B-Instruct")
        >>> response = requests.post("/llms", json=llm_data.model_dump())
    """

    pass


class LLMResponse(LLMBase):
    """Response schema with full LLM metadata including database ID.

    Extends LLMBase with optional fields for detailed model information. Used by
    all read endpoints (list, get, search).

    Attributes:
        id: Database primary key.
        type: Model family (e.g., "llama", "qwen", "mistral").
        description: Model description from HuggingFace or user annotation.
        model_metadata: JSON string with additional metadata (vocab size, context length).
        quantized: Boolean - False=not quantized, True=pre-quantized (already in 4-bit/8-bit format).
        param_size: Model size in billions of parameters (positive), or None if unmeasured (#201).
    """

    id: int
    type: Optional[str] = None
    description: Optional[str] = None
    model_metadata: Optional[str] = None
    quantized: Optional[bool] = False
    supports_tools: Optional[bool] = None
    supports_tools_wire: Optional[bool] = Field(
        default=None,
        description="True=tool calls verified to parse into structured calls on the active engine's wire (#298); False=verified unreliable; None=unverified (routes systematic)",
    )
    param_size: Optional[float] = Field(
        default=None, gt=0, description="Parameter size in billions; None when unmeasured (#201)"
    )
    is_base: bool = Field(
        default=False,
        description="True=curated foundation/base model, False=derived/community quant",
    )
    conversational: Optional[bool] = Field(
        default=None,
        description="True=instruction-tuned/chat model; drives IT-first recommendations and list order (#182). None=unknown",
    )
    category: Optional[str] = Field(
        default="general",
        description="Capability category: general/code/reasoning/math/vision/medical/function/safety (#122)",
    )
    # KB-assistant identity (#225/#208): the cards need to tell assistants apart
    # from regular models to show the weights-of-<base> wording, the orphan badge
    # and the rebind affordance. Plain column pass-throughs from the entity.
    is_attached_to_kb: Optional[bool] = Field(
        default=None,
        description="True when this row is a KB assistant (specialized copy bound to a Knowledge Base)",
    )
    kb_id: Optional[int] = Field(
        default=None, description="The assistant's KnowledgeBase id (None for regular models)"
    )
    # Captured sampling facts (#388): raw column pass-through, mostly for
    # debugging; the UI reads the resolved ``sampling_defaults`` below.
    generation_hints: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Captured base-repo generation facts (generation_config subset, context_length, supports_thinking); None = no hints",
    )
    # Real artifact size: the bytes the downloader fetches (catalog rows, from
    # the snapshot) or the measured on-disk footprint (installed rows). The UI
    # reads it before its per-parameter estimate, which misses e.g. a VLM's
    # vision tower by 25-35 %. None = unknown -> the estimate stands.
    artifact_size_bytes: Optional[int] = Field(
        default=None,
        gt=0,
        description="Real download / on-disk size in bytes; None when unknown (the UI falls back to its estimate)",
    )

    @computed_field
    @property
    def sampling_defaults(self) -> SamplingDefaultsResponse:
        """Per-model sampling defaults resolved at read time (#388): the values a
        new conversation / arena panel seeds its sliders from. Pure function over
        this row (captured generation_config, else the neutral constants), no DB
        column."""
        from src.database.generation_hints import resolve_sampling_defaults

        return SamplingDefaultsResponse(**resolve_sampling_defaults(self).to_dict())

    @computed_field
    @property
    def context_window(self) -> Optional[int]:
        """The model's TRAINED context window in tokens, or None when unknown.

        A fact of the artifact (``generation_hints.context_length``: the GGUF
        metadata for llama.cpp entries, ``config.json`` elsewhere) -- it
        survives restarts and machines, unlike ``allocated_context_window``
        below. First-class here so the UI does not dig through the hints blob.
        """
        hints = self.generation_hints
        if isinstance(hints, dict):
            window = hints.get("context_length")
            if isinstance(window, int) and not isinstance(window, bool) and window > 0:
                return window
        return None

    @computed_field
    @property
    def allocated_context_window(self) -> Optional[int]:
        """The window the engine's loaded child actually runs with, or None.

        Non-None ONLY when this row is the CURRENTLY LOADED model: the value
        (``BaseEngine.effective_context_tokens``) belongs to a running child,
        depends on the machine and its memory state (llama-server's fit can
        shrink it below ``context_window``), and is never persisted. Computed
        on the fly like ``runnable``: cheap attribute reads, no I/O, and any
        failure degrades to None (window unknown).
        """
        if self.local != 1:
            return None
        from src.core import config

        engine = getattr(config, "LLM_Engine", None)
        if engine is None:
            return None
        try:
            loaded_id = getattr(engine, "_model_id", None)
            # get_model_and_tokenizer's llm_id is typed str while rows carry
            # int ids; compare their string forms so neither shape lies.
            if loaded_id is None or str(loaded_id) != str(self.id):
                return None
            probe = getattr(engine, "effective_context_tokens", None)
            window = probe() if callable(probe) else None
        except Exception:
            return None
        if isinstance(window, int) and not isinstance(window, bool) and window > 0:
            return window
        return None

    @computed_field
    @property
    def runnable(self) -> bool:
        """Whether this model can run on the active engine's hardware.

        Computed on the fly (no DB column). The catalog is built only from repos in
        the engine's format (filter=FORMAT_TAG), so every entry is runnable by
        construction; the lone exception is a KNOWN_BROKEN quant that load-crashes
        on this engine (see BaseEngine.is_runnable). Defaults to True when the engine
        is unknown, so nothing is hidden by accident.
        """
        if self.local != 0:
            return True
        from src.core import config

        engine = getattr(config, "LLM_Engine", None)
        if engine is None:
            return True
        try:
            return bool(engine.is_runnable(self.link or ""))
        except Exception:
            return True

    @computed_field
    @property
    def supports_vision(self) -> Optional[bool]:
        """Image-input capability, derived for downloaded models (#133).

        Computed on the fly (no DB column), like ``runnable``: the engine reads
        the artifact (mmproj projector for llama.cpp, ``config.json`` for MLX).
        Only meaningful once downloaded, so remote rows stay None. None = unknown
        and the UI treats it as permissive — it disables the image attach button
        only on an explicit False, never blocking a real VLM by accident.

        Short-circuits to None when the weights are known to be gone
        (``weights_available`` False, #376): probing a path that cannot exist
        would make the engine raise a 500-class exception whose constructor
        logs an ERROR on every listing poll, for a request that returns 200.
        """
        if self.local != 1 or not self.link:
            return None
        if self.weights_available is False:
            return None
        from src.domains.llms.repository import detect_supports_vision

        return detect_supports_vision(self.link)

    @computed_field
    @property
    def weights_available(self) -> Optional[bool]:
        """Whether the model's weights still exist on disk (#208).

        Computed on the fly (no DB column), like ``supports_vision``: it is
        self-healing, so a base model whose files were removed to orphan its KB
        assistants reports False here without any migration. Only meaningful for
        a downloaded model, so remote/downloading rows stay None. A local model
        with a missing/empty link, or a link that no longer exists on disk,
        reports False -- the UI reads this to prompt a rebind onto a new base.
        """
        if self.local != 1:
            return None
        if not self.link:
            return False
        from pathlib import Path

        try:
            return Path(self.link).exists()
        except Exception:
            return False

    class Config:
        """Pydantic configuration for LLMResponse model.

        Enables ORM mode to directly convert SQLAlchemy Llm entities to Pydantic models.
        """

        from_attributes = True


class HFSearchResult(BaseModel):
    """A live HuggingFace search hit (not a persisted catalog row).

    Returned by GET /search/huggingface. The frontend renders these and POSTs the
    chosen one to POST /download/huggingface (by link), so results never pollute the
    curated catalog.
    """

    link: str = Field(..., description="HuggingFace repo id, e.g. mlx-community/Foo-4bit")
    name: str
    param_size: Optional[float] = Field(
        default=None, gt=0, description="Billions of params; None when unmeasured (#201)"
    )
    category: str = "general"
    downloads: int = 0
    likes: int = 0
    gated: bool = False
    pipeline_tag: Optional[str] = None
    quantized: bool = True


class HFDownloadRequest(BaseModel):
    """Body for POST /download/huggingface — download a model picked from HF search
    by its repo id, without it having to exist in the catalog first."""

    link: str = Field(..., min_length=1, description="HuggingFace repo id to download")
    name: Optional[str] = None
    type: Optional[str] = None
    param_size: Optional[float] = Field(
        default=None, gt=0, description="Billions of params; None when unmeasured (#201)"
    )
    quantized: bool = True
    category: str = "general"


class DependentAssistant(BaseModel):
    """One KB assistant that shares a base model's weights (COPIED link, #209).

    Deleting the base would leave this assistant's link dangling (orphaned),
    hence it is surfaced before the delete so the user can decide.
    """

    id: int
    name: str
    kb_id: int
    conversation_count: int = Field(..., ge=0)


class DependentsResponse(BaseModel):
    """What a base-model deletion would affect (GET /llms/{id}/dependents, and
    the payload of the 409 raised by DELETE without ``orphan_dependents``).

    Attributes:
        assistants: KB assistants sharing the model's weights (empty when none).
        own_conversation_count: Conversations bound to the model itself.
        total_conversation_count: own + every dependent assistant's count.
    """

    assistants: List[DependentAssistant]
    own_conversation_count: int = Field(..., ge=0)
    total_conversation_count: int = Field(..., ge=0)


class RebindRequest(BaseModel):
    """Body for POST /llms/{assistant_id}/rebind -- re-point an orphaned KB
    assistant at a new base model whose weights exist."""

    new_base_llm_id: int = Field(..., description="Id of the local base model to rebind onto")


class DownloadJobResponse(BaseModel):
    """Response schema for download job status with progress tracking.

    Encodes the state of a background download task, including progress metrics,
    file paths, and error information. The job_id field is aliased from 'id' to
    avoid conflicts with LLM IDs in API responses.

    Attributes:
        job_id: Database primary key (aliased from 'id').
        remote_model_id: HuggingFace model ID (e.g., "meta-llama/Llama-3-8B").
        local_model_id: ID of temp LLM entry created during download, None once
            that entry is deleted (cleanup nulls the FK server-side).
        remote_model_link: HuggingFace URL or API endpoint.
        local_model_link: Filesystem path to final model after quantization.
        status: Job state - "pending", "running", "completed", "failed", "cancelled".
        total_bytes: Total download size in bytes.
        progress: Download progress percentage (0.0-100.0).
        total_time_elapsed: Seconds elapsed since job start.
        time_left: Estimated seconds remaining (updated each poll).
        error_message: Exception message if status="failed", None otherwise.
        created_at: Job creation timestamp.
        updated_at: Last status update timestamp (used for stale job detection).

    Example:
        >>> job = DownloadJobResponse(
        ...     id=42, remote_model_id="Qwen/Qwen2.5-7B-Instruct", status="running",
        ...     progress=65.0, time_left=120.5, ...
        ... )
        >>> print(job.job_id)  # 42 (aliased from 'id')
    """

    job_id: int = Field(..., alias="id")
    remote_model_id: str = Field(..., min_length=1)
    local_model_id: Optional[int] = None
    remote_model_link: str = Field(..., min_length=1)
    local_model_link: Optional[str] = None
    status: str = Field(..., pattern="^(pending|running|completed|failed|cancelled)$")
    total_bytes: float = Field(default=0.0, ge=0)
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    total_time_elapsed: float = Field(default=0.0, ge=0)
    time_left: float = Field(default=0.0, ge=0)
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        """Pydantic configuration for DownloadJobResponse model.

        Enables ORM mode and validates field names by alias (job_id aliased from id).
        """

        from_attributes = True
        validate_by_name = True
