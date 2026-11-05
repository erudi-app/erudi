"""Pydantic validation schemas for startup state management.

This module defines data transfer objects (DTOs) for the startup domain, handling
UI state persistence flags across application restarts.

Schema Purpose:
- WelcomePopupResponse: Response indicating if welcome popup should be shown.
- InstallStateResponse: Whether the database holds the user's data.

Example:
    from src.domains.startup.schemas import WelcomePopupResponse
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/startup/welcome-popup", response_model=WelcomePopupResponse)
    def check_welcome_popup():
        return {"has_already_displayed": True}
"""

from pydantic import BaseModel, Field


class WelcomePopupResponse(BaseModel):
    """Response schema for welcome popup display status.

    Indicates whether the welcome popup has already been shown to the user.
    Used by frontend to determine if popup should be displayed.

    Attributes:
        has_already_displayed: True if popup was already shown, False if first time.

    Example:
        >>> response = WelcomePopupResponse(has_already_displayed=False)
        >>> print(response.has_already_displayed)
        False
    """

    has_already_displayed: bool = Field(
        ..., description="True if welcome popup already shown, False if first time"
    )

    class Config:
        """Pydantic configuration for WelcomePopupResponse model.

        Enables JSON schema generation for OpenAPI documentation.
        """

        json_schema_extra = {"example": {"has_already_displayed": False}}


class InstallStateResponse(BaseModel):
    """Response schema for ``GET /startup/install_state`` (#572).

    Attributes:
        has_user_data: True when the database holds at least one
            conversation, knowledge base or downloaded model. The renderer reads it at the first launch
            of a version that recorded no "last seen version": True means the
            install was updated from an earlier version, False a fresh install
            (or one after "Clear all data").
    """

    has_user_data: bool = Field(
        ..., description="True if a conversation, a knowledge base or a downloaded model exists"
    )
