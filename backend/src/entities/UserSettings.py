"""SQLAlchemy entity for global user settings.

Singleton table (one row) holding app-wide user preferences, mirroring the
``StartupVariables`` singleton pattern. First occupant: the global web-search
toggle (issue #310) — the DEFAULT for new conversations. Each conversation
copies this value at creation and owns its flag afterwards; changing the
global setting never retro-affects existing conversations. It also carries the
interface language, the automatic-update preference and the inference
backend the app runs models on.

Example:
    from src.entities.UserSettings import UserSettings

    settings = UserSettings(web_search_enabled=False)
"""

from sqlalchemy import Column, Integer, Boolean, String
from sqlalchemy.orm import validates
from src.database.core import Base

# The four interface languages the frontend ships translations for (#385).
# The list is the single source of truth for the schema Literal and the
# entity validator; the frontend mirrors it in src/i18n/languages.js.
SUPPORTED_LANGUAGES = ("en", "fr", "es", "zh")
DEFAULT_LANGUAGE = "en"

# Which backend inference runs on. "auto" keeps the hardware detection of
# ``BaseEngine.get_engine()``; "cpu" pins the CPU build on a machine whose GPU
# the bundled CUDA binary cannot drive. The fallback is never applied on the
# app's own initiative -- the user opts in here, or reinstalls the CPU build.
INFERENCE_BACKENDS = ("auto", "cpu")
DEFAULT_INFERENCE_BACKEND = "auto"


class UserSettings(Base):
    """SQLAlchemy model for the user-settings singleton.

    Attributes:
        id: Primary key (singleton - only one row).
        web_search_enabled: Boolean - global default for the web_search agent
            tool (#310). False by default: a web search egresses the user's
            query, so the local-first product keeps it strictly opt-in.
        language: Interface language code (#385), one of SUPPORTED_LANGUAGES.
            "en" by default; the frontend derives the first value from the OS
            locale and persists it here so it survives restarts.
        auto_update_enabled: Boolean - whether the Electron main process may
            check for, download and install a new version on its own. True by
            default, so an install that never touches the setting behaves as it
            always has; turning it off stops the update traffic entirely.
        inference_backend: One of INFERENCE_BACKENDS. "auto" by default, so
            hardware detection decides; "cpu" makes the app run models on the
            CPU build even when an NVIDIA GPU is present. Read once per boot,
            after the migrations, so the choice needs an app restart to apply.

    Constraints:
        - web_search_enabled must be a Boolean (enforced by validator).
        - auto_update_enabled must be a Boolean (enforced by validator).
        - language must be one of SUPPORTED_LANGUAGES (enforced by validator).
        - inference_backend must be one of INFERENCE_BACKENDS (validator).
    """

    __tablename__ = "user_settings"

    id = Column(Integer, primary_key=True, index=True)
    web_search_enabled = Column(Boolean, default=False, nullable=False)
    language = Column(String(8), default=DEFAULT_LANGUAGE, nullable=False)
    auto_update_enabled = Column(Boolean, default=True, nullable=False)
    inference_backend = Column(String(8), default=DEFAULT_INFERENCE_BACKEND, nullable=False)

    @validates("language")
    def validate_language(self, key, value):
        """Ensure the language is one of the supported interface languages.

        Raises:
            ValueError: If value is not in SUPPORTED_LANGUAGES.
        """
        if value not in SUPPORTED_LANGUAGES:
            raise ValueError(f"{key} must be one of {SUPPORTED_LANGUAGES}, got {value!r}")
        return value

    @validates("inference_backend")
    def validate_inference_backend(self, key, value):
        """Ensure the inference backend is one of the supported values.

        Raises:
            ValueError: If value is not in INFERENCE_BACKENDS.
        """
        if value not in INFERENCE_BACKENDS:
            raise ValueError(f"{key} must be one of {INFERENCE_BACKENDS}, got {value!r}")
        return value

    @validates("web_search_enabled", "auto_update_enabled")
    def validate_boolean_flags(self, key, value):
        """Ensure boolean flags are actually Boolean type.

        Args:
            key: Column name being validated.
            value: Proposed Boolean value.

        Returns:
            bool: The validated Boolean value.

        Raises:
            ValueError: If value is not a Boolean.
        """
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be a Boolean, got {type(value)}")
        return value
