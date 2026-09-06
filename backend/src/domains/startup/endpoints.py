"""FastAPI endpoints for startup state management.

This module provides REST API endpoints for managing application startup state,
including UI flags that persist across app restarts (e.g., welcome popup status).

Architecture:
    - Endpoints use repository pattern via dependency injection.
    - Pydantic schemas validate all request/response data.
    - Structured logging for traceability.
    - Proper error handling with domain-specific exceptions.

Endpoints:
    GET /startup/welcome-popup - Check and update welcome popup display status.

Example:
    curl http://127.0.0.1:27182/erudi/startup/welcome-popup
    {"has_already_displayed": false}
"""

from fastapi import Depends, APIRouter
from sqlalchemy.orm import Session

from src.database.core import get_db
from src.domains.startup.repository import Startup_Variables_Repository
from src.domains.startup.schemas import WelcomePopupResponse
from src.core.logging import logger
from src.core.exceptions import DatabaseException

router = APIRouter(prefix="/startup", tags=["startup"])


# ============ Dependency Injection ============


def get_startup_repository(db: Session = Depends(get_db)) -> Startup_Variables_Repository:
    """FastAPI dependency injection factory for Startup_Variables_Repository.

    Args:
        db: SQLAlchemy session (injected by FastAPI via Depends(get_db)).

    Returns:
        Startup_Variables_Repository: Configured repository instance.
    """
    return Startup_Variables_Repository(db)


# ============ Endpoints ============


@router.get("/welcome-popup", response_model=WelcomePopupResponse)
async def get_welcome_popup_status(
    startup_repo: Startup_Variables_Repository = Depends(get_startup_repository),
    db: Session = Depends(get_db),
):
    """Check welcome popup display status and mark as shown if first time.

    Implements "check-and-set" logic: returns False on first call (popup should show),
    then sets flag to True. Subsequent calls return True (popup already shown).

    Args:
        startup_repo: Injected startup variables repository.
        db: Database session for transaction control.

    Returns:
        WelcomePopupResponse: {"has_already_displayed": bool}

    Raises:
        DatabaseException: If database operation fails.

    Example:
        First call:  GET /startup/welcome-popup → {"has_already_displayed": false}
        Second call: GET /startup/welcome-popup → {"has_already_displayed": true}
    """
    try:
        # Get or create singleton startup variables
        vars = startup_repo.get_or_create()

        # Check current status
        already_displayed = vars.welcome_popup_has_already_displayed

        # If not displayed yet, mark as displayed now
        if not already_displayed:
            startup_repo.mark_welcome_popup_displayed(vars)
            db.commit()
            logger.info("Welcome popup marked as displayed (first time)")
            return WelcomePopupResponse(has_already_displayed=False)

        # Already displayed before
        logger.debug("Welcome popup status: already displayed")
        return WelcomePopupResponse(has_already_displayed=True)

    except Exception as e:
        db.rollback()
        raise DatabaseException("Failed to get welcome popup status", trace=str(e))
