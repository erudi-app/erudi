"""Pydantic validation schemas for the user-settings domain (issue #310).

One singleton resource: the app-wide user settings. It carries the global
web-search default (#310), the interface language (#385), the
automatic-update preference, the inference backend, the default reasoning
effort and the default compaction threshold (#607); new settings slot in as
additional fields.
"""

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from src.engines.working_window import (
    COMPACTION_THRESHOLD_MAX,
    COMPACTION_THRESHOLD_MIN,
    COMPACTION_THRESHOLD_STEP,
)

LanguageCode = Literal["en", "fr", "es", "zh"]
InferenceBackend = Literal["auto", "cpu"]
# Mirrors src.agents.reasoning_effort.REASONING_EFFORT_LEVELS (a Literal cannot
# be built from a runtime tuple without losing the OpenAPI enum); the entity
# validator enforces the same list on the way to the database.
ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh"]
# The share of the working window, in whole percent, at which a conversation
# compacts (#607): 50 to 90 in steps of 5. Strict: a bool, a float or a
# numeric string is refused, not coerced. The entity validators enforce the
# same scale on the way to the database.
CompactionThreshold = Annotated[
    int,
    Field(
        strict=True,
        ge=COMPACTION_THRESHOLD_MIN,
        le=COMPACTION_THRESHOLD_MAX,
        multiple_of=COMPACTION_THRESHOLD_STEP,
    ),
]


class UserSettingsResponse(BaseModel):
    """Response schema for the user-settings singleton.

    Attributes:
        web_search_enabled: Global default for the web_search agent tool.
            New conversations copy this value at creation; the per-conversation
            toggle owns it afterwards.
        language: Interface language code the frontend renders in.
        auto_update_enabled: Whether the Electron main process may check for,
            download and install a new version on its own.
        inference_backend: "auto" to let hardware detection pick the engine,
            "cpu" to run models on the CPU build regardless of the GPU.
        default_reasoning_effort: Global default for how much a model may
            deliberate before answering. New conversations copy it at
            creation; arena turns read it directly.
        default_compaction_threshold: Global default for the share of the
            working window, in percent, at which a conversation compacts. New
            conversations copy it at creation.
    """

    web_search_enabled: bool = Field(
        ...,
        description="Global default for the web_search agent tool (new conversations inherit it)",
    )
    language: LanguageCode = Field(
        ...,
        description="Interface language code (en, fr, es, zh)",
    )
    auto_update_enabled: bool = Field(
        ...,
        description="Whether the app may check for and install updates on its own",
    )
    inference_backend: InferenceBackend = Field(
        ...,
        description="Inference backend preference (auto: detect hardware, cpu: force the CPU build)",
    )
    default_reasoning_effort: ReasoningEffort = Field(
        ...,
        description="Default reasoning effort new conversations inherit (none, low, medium, high, xhigh)",
    )
    default_compaction_threshold: CompactionThreshold = Field(
        ...,
        description="Default compaction threshold new conversations inherit, in percent (50-90, step 5)",
    )

    class Config:
        from_attributes = True


class UserSettingsUpdate(BaseModel):
    """Request schema for updating the user-settings singleton (PUT).

    Partial update: every field is optional and an omitted field is left
    untouched, but an empty payload is rejected (nothing to update).
    """

    web_search_enabled: Optional[bool] = Field(
        None,
        description="Enable or disable the global web-search default",
    )
    language: Optional[LanguageCode] = Field(
        None,
        description="Interface language code (en, fr, es, zh)",
    )
    auto_update_enabled: Optional[bool] = Field(
        None,
        description="Allow or refuse automatic update checks, downloads and installs",
    )
    inference_backend: Optional[InferenceBackend] = Field(
        None,
        description="Inference backend preference (auto or cpu); applies on the next app start",
    )
    default_reasoning_effort: Optional[ReasoningEffort] = Field(
        None,
        description="Default reasoning effort for new conversations (none, low, medium, high, xhigh)",
    )
    default_compaction_threshold: Optional[CompactionThreshold] = Field(
        None,
        description="Default compaction threshold for new conversations, in percent (50-90, step 5)",
    )

    @model_validator(mode="after")
    def require_at_least_one_field(self):
        if (
            self.web_search_enabled is None
            and self.language is None
            and self.auto_update_enabled is None
            and self.inference_backend is None
            and self.default_reasoning_effort is None
            and self.default_compaction_threshold is None
        ):
            raise ValueError("At least one setting must be provided")
        return self
