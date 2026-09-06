"""FastAPI endpoints for the user-settings singleton (issue #310).

GET/PUT ``/erudi/user_settings/``: the app-wide settings the frontend's
Settings page binds to: the global web-search default (#310), the
interface language (#385), the automatic-update preference and the inference
backend. Follows the startup domain's layering (endpoints -> repository) — the resource is a
one-row singleton with no business logic beyond get-or-create. The inference
backend rides here too: it is read once per boot by the lifespan, so changing
it needs a backend restart, which the frontend triggers explicitly.
"""

from fastapi import Depends, APIRouter
from sqlalchemy.orm import Session

from src.database.core import get_db
from src.domains.user_settings.repository import User_Settings_Repository
from src.domains.user_settings.schemas import UserSettingsResponse, UserSettingsUpdate
from src.core.logging import logger
from src.core.exceptions import DatabaseException

router = APIRouter(prefix="/user_settings", tags=["user_settings"])


def get_user_settings_repository(
    db: Session = Depends(get_db),
) -> User_Settings_Repository:
    """FastAPI dependency injection factory for User_Settings_Repository."""
    return User_Settings_Repository(db)


@router.get("/", response_model=UserSettingsResponse)
async def get_user_settings(
    settings_repo: User_Settings_Repository = Depends(get_user_settings_repository),
    db: Session = Depends(get_db),
):
    """Fetch the user-settings singleton (created with defaults on first read).

    Example:
        GET /erudi/user_settings/
        -> {"web_search_enabled": false, "language": "en", "auto_update_enabled": true,
            "inference_backend": "auto"}
    """
    try:
        settings = settings_repo.get_or_create()
        db.commit()
        return settings
    except Exception as e:
        db.rollback()
        raise DatabaseException("Failed to get user settings", trace=str(e))


@router.put("/", response_model=UserSettingsResponse)
async def update_user_settings(
    payload: UserSettingsUpdate,
    settings_repo: User_Settings_Repository = Depends(get_user_settings_repository),
    db: Session = Depends(get_db),
):
    """Update the user-settings singleton (partial: omitted fields are kept).

    Example:
        PUT /erudi/user_settings/ {"language": "fr"}
        -> {"web_search_enabled": false, "language": "fr", "auto_update_enabled": true,
            "inference_backend": "auto"}
    """
    try:
        settings = settings_repo.get_or_create()
        if payload.web_search_enabled is not None:
            settings_repo.set_web_search_enabled(settings, payload.web_search_enabled)
        if payload.language is not None:
            settings_repo.set_language(settings, payload.language)
        if payload.auto_update_enabled is not None:
            settings_repo.set_auto_update_enabled(settings, payload.auto_update_enabled)
        if payload.inference_backend is not None:
            settings_repo.set_inference_backend(settings, payload.inference_backend)
        db.commit()
        logger.info(
            "User settings updated: "
            f"web_search_enabled={settings.web_search_enabled} language={settings.language} "
            f"auto_update_enabled={settings.auto_update_enabled} "
            f"inference_backend={settings.inference_backend}"
        )
        return settings
    except Exception as e:
        db.rollback()
        raise DatabaseException("Failed to update user settings", trace=str(e))
