"""Repository layer for arena database operations.

Provides data access methods for the arena domain, following the repository pattern.
Isolates SQLAlchemy queries from business logic, handles database errors gracefully.

Example:
    from src.domains.arena.repository import ArenaRepository

    repo = ArenaRepository(db)
    llm = repo.get_llm_by_id(42)
"""

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from src.entities.Llm import Llm
from src.core.logging import logger
from src.core.exceptions import ModelNotFoundException, DatabaseException


class ArenaRepository:
    """Repository for arena database queries with error handling."""

    def __init__(self, db: Session):
        """Initialize repository with database session.

        Args:
            db: SQLAlchemy session for executing queries.
        """
        logger.debug("Initializing ArenaRepository")
        self.db = db

    def get_llm_by_id(self, llm_id: int) -> Llm:
        """Retrieve LLM entity by database ID with error handling.

        Args:
            llm_id: Database primary key of the LLM to retrieve.

        Returns:
            Llm entity with all metadata fields.

        Raises:
            HTTPException: 404 if LLM not found, 500 on SQLAlchemy errors.

        Example:
            >>> repo = ArenaRepository(db)
            >>> llm = repo.get_llm_by_id(42)
            >>> print(llm.name)
            "Llama-3-8B-Instruct"
        """
        try:
            logger.debug(f"Retrieving LLM {llm_id}")
            llm = self.db.query(Llm).filter(Llm.id == llm_id).first()

            if not llm:
                raise ModelNotFoundException(f"LLM {llm_id}")

            logger.debug(f"Retrieved LLM {llm_id}: {llm.name}")
            return llm

        except ModelNotFoundException:
            raise
        except SQLAlchemyError as e:
            raise DatabaseException("Could not retrieve LLM", trace=str(e))
