"""Pydantic validation schemas for the user-settings domain (issue #310).

One singleton resource: the app-wide user settings. It carries the global
web-search default (#310), the interface language (#385), the
automatic-update preference, the inference backend and the default reasoning
effort; new settings slot in as additional fields.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

LanguageCode = Literal["en", "fr", "es", "zh"]
InferenceBackend = Literal["auto", "cpu"]
# Mirrors src.agents.reasoning_effort.REASONING_EFFORT_LEVELS (a Literal cannot
# be built from a runtime tuple without losing the OpenAPI enum); the entity
# validator enforces the same list on the way to the database.
ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh"]


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

    @model_validator(mode="after")
    def require_at_least_one_field(self):
        if (
            self.web_search_enabled is None
            and self.language is None
            and self.auto_update_enabled is None
            and self.inference_backend is None
            and self.default_reasoning_effort is None
        ):
            raise ValueError("At least one setting must be provided")
        return self
