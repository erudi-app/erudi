"""Data access layer for hardware profile entity.

Implements Repository pattern for hardware profile database operations. Handles
singleton pattern enforcement and provides clean data access API.

Architecture:
    Endpoints → Service → Repository → Entity → Database

Singleton Pattern:
    - Hardware profile is singleton (one row, id=1)
    - get_profile() returns existing or None
    - create_profile() creates new profile
    - update_profile() modifies existing

Example:
    from src.domains.hardware.repository import Hardware_Repository
    from src.database.core import SessionLocal

    db = SessionLocal()
    repo = Hardware_Repository(db)

    profile = repo.get_profile()
    if not profile:
        profile = repo.create_profile({
            "backend_type": "mlx",
            "cpu_model": "Apple M3 Max",
            ...
        })
    db.commit()
"""

from typing import Optional, Dict, Any
from sqlalchemy.orm import Session

from src.entities.HardwareProfile import HardwareProfile
from src.core.logging import logger
from src.core.exceptions import DatabaseException


class Hardware_Repository:
    """Repository for hardware profile data access operations.

    Encapsulates all database operations for HardwareProfile entity. Follows
    Repository pattern with clean separation from business logic.

    Attributes:
        db: SQLAlchemy session for database operations.

    Note:
        Does NOT commit transactions. Commits are handled at endpoint level.
        Uses flush() to make changes visible within transaction.

    Example:
        >>> repo = Hardware_Repository(db)
        >>> profile = repo.get_profile()
        >>> if not profile:
        ...     profile = repo.create_profile(hardware_data)
        >>> db.commit()  # Commit at endpoint level
    """

    def __init__(self, db: Session):
        """Initialize repository with database session.

        Args:
            db: SQLAlchemy session injected from endpoint.
        """
        self.db = db
        logger.debug("Initializing Hardware_Repository")

    def get_profile(self) -> Optional[HardwareProfile]:
        """Retrieve hardware profile from database (singleton).

        Returns existing hardware profile if present. Since hardware profile
        is singleton, returns first (and only) row. -> Delete less recent if
        more than one exists.

        Returns:
            Optional[HardwareProfile]: Existing profile or None if not found.

        Raises:
            DatabaseException: If query fails critically.

        Example:
            >>> repo = Hardware_Repository(db)
            >>> profile = repo.get_profile()
            >>> if profile:
            ...     print(f"Backend: {profile.backend_type}")
        """
        try:
            logger.debug("Querying hardware profile")
            profiles = (
                self.db.query(HardwareProfile)
                .order_by(
                    HardwareProfile.updated_at.desc(),
                    HardwareProfile.created_at.desc(),
                    HardwareProfile.id.desc(),
                )
                .all()
            )
            profile = profiles[0] if profiles else None

            if len(profiles) > 1:
                logger.warning(
                    f"Multiple hardware profiles detected ({len(profiles)}); pruning older entries"
                )
                for stale_profile in profiles[1:]:
                    self.delete_profile(stale_profile)

            if profile:
                logger.debug(f"Found existing hardware profile: backend={profile.backend_type}")
            else:
                logger.debug("No hardware profile found in database")

            return profile

        except Exception as e:
            raise DatabaseException("Failed to retrieve hardware profile", trace=str(e))

    def create_profile(self, hardware_data: Dict[str, Any]) -> HardwareProfile:
        """Create new hardware profile from engine data.

        Creates HardwareProfile entity from dictionary returned by engine's
        get_performance_evaluation(). Uses flush() to make changes visible
        within transaction without committing.

        Args:
            hardware_data: Dictionary with hardware specs from engine.
                Must include: backend_type, cpu_model, total_memory_gb,
                global_inference_score, etc.

        Returns:
            HardwareProfile: Newly created profile entity.

        Raises:
            DatabaseException: If profile creation fails.

        Note:
            Does NOT commit. Caller must commit transaction.

        Example:
            >>> hw_data = engine.get_performance_evaluation()
            >>> profile = repo.create_profile(hw_data)
            >>> db.commit()  # Commit at endpoint level
        """
        try:
            logger.info(f"Creating hardware profile: backend={hardware_data.get('backend_type')}")

            # Create entity from engine data
            profile = HardwareProfile(**hardware_data)

            self.db.add(profile)
            self.db.flush()  # Make visible in transaction, but don't commit

            logger.info(f"Hardware profile created successfully: id={profile.id}")
            return profile

        except Exception as e:
            raise DatabaseException("Failed to create hardware profile", trace=str(e))

    def update_profile(self, profile: HardwareProfile, updates: Dict[str, Any]) -> HardwareProfile:
        """Update existing hardware profile with new data.

        Updates profile fields from dictionary. Useful for refreshing hardware
        info or updating dynamic fields (available_memory_gb, disk_available_gb).

        Args:
            profile: Existing HardwareProfile entity to update.
            updates: Dictionary with fields to update.

        Returns:
            HardwareProfile: Updated profile entity.

        Raises:
            DatabaseException: If update fails.

        Note:
            Does NOT commit. Caller must commit transaction.

        Example:
            >>> profile = repo.get_profile()
            >>> updated = repo.update_profile(profile, {
            ...     "available_memory_gb": 64.5,
            ...     "disk_available_gb": 120.0
            ... })
            >>> db.commit()
        """
        try:
            logger.debug(f"Updating hardware profile id={profile.id}")

            for key, value in updates.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)
                else:
                    logger.warning(f"Ignoring unknown field: {key}")

            self.db.flush()  # Make visible in transaction

            logger.debug(f"Hardware profile updated successfully: id={profile.id}")
            return profile

        except Exception as e:
            raise DatabaseException("Failed to update hardware profile", trace=str(e))

    def delete_profile(self, profile: HardwareProfile) -> None:
        """Delete hardware profile from database.

        Removes profile entity. Useful for testing or re-detection scenarios.

        Args:
            profile: HardwareProfile entity to delete.

        Raises:
            DatabaseException: If deletion fails.

        Note:
            Does NOT commit. Caller must commit transaction.

        Example:
            >>> profile = repo.get_profile()
            >>> if profile:
            ...     repo.delete_profile(profile)
            ...     db.commit()
        """
        try:
            logger.info(f"Deleting hardware profile id={profile.id}")

            self.db.delete(profile)
            self.db.flush()

            logger.info("Hardware profile deleted successfully")

        except Exception as e:
            raise DatabaseException("Failed to delete hardware profile", trace=str(e))
