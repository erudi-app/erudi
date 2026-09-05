"""REST layer of the diagnostics domain.

Endpoint:
    GET /erudi/diagnostics/ -- environment summary plus the recent
    WARNING-or-worse records of ``backend.log``.

The response is built for a human to paste into a bug report. It is assembled
on this machine from this machine and returned to the app window; nothing is
sent anywhere, and the reader excludes INFO records, which carry conversation
content on purpose (see ``docs/privacy.md``).
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.database.core import get_db
from src.domains.diagnostics import services
from src.domains.diagnostics.log_reader import DEFAULT_LIMIT
from src.domains.diagnostics.schemas import DiagnosticsResponse

router = APIRouter(prefix="/diagnostics", tags=["diagnostics"])


@router.get("/", response_model=DiagnosticsResponse)
async def get_diagnostics(
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=500, description="Recent records to return."),
    db: Optional[Session] = Depends(get_db),
) -> DiagnosticsResponse:
    """Return what a bug report needs about this installation.

    Args:
        limit: Maximum number of recent log records to include.
        db: Session used only to name the model the engine holds in memory.

    Returns:
        DiagnosticsResponse: Environment summary and recent errors, oldest first.
    """
    return DiagnosticsResponse(
        environment=services.build_environment(db),
        recent_errors=services.build_recent_errors(limit=limit),
    )
