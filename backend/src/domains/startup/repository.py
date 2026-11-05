"""Data access layer for StartupVariables entity.

This module provides the repository class following the Repository pattern for database
operations on startup state variables. The repository encapsulates all SQLAlchemy queries
and database interactions for the singleton StartupVariables entity.

Architecture:
    - Startup_Variables_Repository: Singleton operations for startup state management.

Repository Pattern Benefits:
    - Single source of truth for data access logic.
    - Easy to mock for testing.
    - Clear separation between business logic (services) and data access.
    - Consistent error handling and logging.

Example:
    from src.domains.startup.repository import Startup_Variables_Repository

    # In endpoint or service
    startup_repo = Startup_Variables_Repository(db)
    vars = startup_repo.get_or_create()
"""

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from src.entities.Conversation import Conversation
from src.entities.Llm import Llm
from src.entities.StartupVariables import StartupVariables
from src.core.logging import logger


class Startup_Variables_Repository:
    """Repository for StartupVariables entity database operations.

    Handles all operations for the singleton StartupVariables entity, which stores
    persistent UI state flags across app restarts (e.g., welcome popup display status).

    Attributes:
        db: SQLAlchemy database session (injected by FastAPI).

    Example:
        >>> startup_repo = Startup_Variables_Repository(db)
        >>> vars = startup_repo.get_or_create()
        >>> startup_repo.mark_welcome_popup_displayed(vars)
    """

    def __init__(self, db: Session):
        """Initialize repository with database session.

        Args:
            db: SQLAlchemy session for database operations.
        """
        self.db = db
        logger.debug("Initializing Startup_Variables_Repository")

    def get_or_create(self) -> StartupVariables:
        """Retrieve the singleton StartupVariables record, creating if absent.

        Returns:
            StartupVariables: The singleton startup state entity.

        Note:
            This is a singleton entity - only one row should exist in the table.
            Creates with default values (welcome_popup_has_already_displayed=False) if missing.
        """
        logger.debug("Retrieving or creating StartupVariables singleton")
        vars = self.db.query(StartupVariables).first()

        if not vars:
            logger.info("StartupVariables not found, creating new record")
            vars = StartupVariables(welcome_popup_has_already_displayed=False)
            self.db.add(vars)
            self.db.flush()
            self.db.refresh(vars)
            logger.info(f"Created StartupVariables {vars.id}")

        return vars

    def has_user_data(self) -> bool:
        """Whether the database holds anything the user made (#572).

        True when at least one conversation exists or one model was downloaded
        (``llms.local = 1``). The catalog seeds ``local = 0`` rows on every
        boot and an unfinished download is ``local = 2``, so neither counts.
        Read-only, and independent of the welcome flag, which reading marks.

        Returns:
            bool: True for an install that was used, False for a fresh one.
        """
        statement = select(exists().select_from(Conversation) | exists().where(Llm.local == 1))
        return bool(self.db.execute(statement).scalar())

    def get_welcome_popup_status(self) -> bool:
        """Check if welcome popup has already been displayed to user.

        Returns:
            bool: True if popup was already shown, False if first time.
        """
        logger.debug("Checking welcome popup display status")
        vars = self.get_or_create()
        return vars.welcome_popup_has_already_displayed

    def mark_welcome_popup_displayed(self, vars: StartupVariables) -> StartupVariables:
        """Mark welcome popup as displayed (set flag to True).

        Args:
            vars: StartupVariables entity to update.

        Returns:
            StartupVariables: Updated entity (not yet committed, use flush()).
        """
        logger.info("Marking welcome popup as displayed")
        vars.welcome_popup_has_already_displayed = True
        self.db.flush()
        self.db.refresh(vars)
        return vars

    def update_field(self, vars: StartupVariables, field: str, value) -> StartupVariables:
        """Update a specific field on the StartupVariables entity.

        Args:
            vars: StartupVariables entity to update.
            field: Name of the field to update.
            value: New value for the field.

        Returns:
            StartupVariables: Updated entity (not yet committed, use flush()).

        Raises:
            AttributeError: If field does not exist on entity.
        """
        logger.info(f"Updating StartupVariables field: {field} = {value}")
        if not hasattr(vars, field):
            raise AttributeError(f"StartupVariables has no field '{field}'")

        setattr(vars, field, value)
        self.db.flush()
        self.db.refresh(vars)
        return vars

    def reset(self, vars: StartupVariables) -> StartupVariables:
        """Reset all startup variables to default values.

        Args:
            vars: StartupVariables entity to reset.

        Returns:
            StartupVariables: Reset entity (not yet committed, use flush()).
        """
        logger.info("Resetting StartupVariables to defaults")
        vars.welcome_popup_has_already_displayed = False
        self.db.flush()
        self.db.refresh(vars)
        return vars
