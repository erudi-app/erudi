"""SQLAlchemy entity for conversation sessions with LLM generation parameters.

Represents a chat session with a specific LLM, storing messages, generation settings,
and conversation metadata. Enforces parameter validation via SQLAlchemy validators.

Relationships:
    - messages: One-to-many with Message (cascade delete).
    - llm: Many-to-one with Llm (which model is used).

Example:
    from src.entities.Conversation import Conversation

    conv = Conversation(
        llm_id=42,
        name="Python Programming Help",
        temperature=0.7,
        top_p=0.9,
        max_tokens=2048
    )
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column,
    Integer,
    SmallInteger,
    String,
    ForeignKey,
    DateTime,
    Float,
    Text,
    Boolean,
)
from sqlalchemy.orm import relationship, validates
from src.agents.reasoning_effort import (
    DEFAULT_REASONING_EFFORT,
    REASONING_EFFORT_LEVELS,
)
from src.engines.working_window import (
    COMPACTION_THRESHOLDS,
    DEFAULT_COMPACTION_THRESHOLD,
    is_compaction_threshold,
)
from src.database.generation_hints import (
    FALLBACK_MAX_TOKENS,
    FALLBACK_TEMPERATURE,
    FALLBACK_TOP_P,
)
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql import func
from src.database.core import Base


# Bounds of the sampling values a conversation runs with. The request schemas
# (``src.domains.conversations.schemas.Temperature`` / ``TopP``) read them too.
TEMPERATURE_MIN, TEMPERATURE_MAX = 0.0, 2.0
TOP_P_MIN, TOP_P_MAX = 0.0, 1.0


# The rules of the per-conversation values, shared by every row that stores
# one: a conversation, and the defaults saved for a model (``ModelDefaults``).
def check_temperature(temperature: float) -> float:
    """Return ``temperature``, or raise ValueError outside [0.0, 2.0]."""
    if not TEMPERATURE_MIN <= temperature <= TEMPERATURE_MAX:
        raise ValueError(f"Temperature must be between {TEMPERATURE_MIN} and {TEMPERATURE_MAX}")
    return temperature


def check_top_p(top_p: float) -> float:
    """Return ``top_p``, or raise ValueError outside [0.0, 1.0]."""
    if not TOP_P_MIN <= top_p <= TOP_P_MAX:
        raise ValueError(f"Top_p must be between {TOP_P_MIN} and {TOP_P_MAX}")
    return top_p


def check_reasoning_effort(reasoning_effort: str) -> str:
    """Return the level, or raise ValueError when it is not in REASONING_EFFORT_LEVELS."""
    if reasoning_effort not in REASONING_EFFORT_LEVELS:
        raise ValueError(f"Reasoning effort must be one of {REASONING_EFFORT_LEVELS}")
    return reasoning_effort


def check_compaction_threshold(compaction_threshold: int) -> int:
    """Return the percentage, or raise ValueError when it is not an int in
    COMPACTION_THRESHOLDS (50-90, step 5); a bool, a float or a string is
    rejected."""
    if not is_compaction_threshold(compaction_threshold):
        raise ValueError(f"Compaction threshold must be one of {COMPACTION_THRESHOLDS}")
    return compaction_threshold


class Conversation(Base):
    """SQLAlchemy model for conversation sessions with generation parameter validation.

    Stores chat sessions with LLM-specific generation settings. Each conversation maintains
    its own temperature, top_p, and max_tokens parameters. Messages are stored separately
    with cascade delete.

    Attributes:
        id: Primary key (auto-increment).
        llm_id: Foreign key to Llm, NULLABLE (server-side SET NULL on model
            delete). A conversation survives the deletion of its model and can
            be switched to another one (#225).
        created_at: Conversation creation timestamp (server-stamped).
        updated_at: Last modification timestamp (server-stamped, auto-updated).
        temperature: Sampling temperature (0.0-2.0, validated).
        top_p: Nucleus sampling threshold (0.0-1.0, validated).
        max_tokens: Maximum tokens to generate (1-32768, validated).
        custom_prompt: Optional additional system instructions.
        name: Conversation title (default "New Conversation").
        web_search_enabled: Per-conversation web-search toggle.
        reasoning_effort: How much the model may deliberate (validated).
        compaction_threshold: Share of the working window, in percent, at
            which the conversation compacts (50-90 step 5, validated).
        messages: Relationship to Message entities (ordered by id = insertion order).
        llm: Relationship to Llm entity (which model).
        message_count: Hybrid property returning number of messages.
        last_message_time: Hybrid property returning last message timestamp.

    Example:
        >>> conv = Conversation(llm_id=42, name="Debug Session", temperature=0.3, max_tokens=4096)
        >>> print(conv.message_count)  # 0 (messages are added via MessageRepository)
    """

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    # NULLABLE + ON DELETE SET NULL: a conversation is never permanently bound
    # to one model (#225). Deleting the model nulls this FK server-side (the
    # conversation survives, unbound) instead of cascading the conversation away.
    llm_id = Column(Integer, ForeignKey("llms.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    # Conversation parameters. Column defaults = the ONE fallback (#388); the
    # service normally resolves the model's own defaults before the row exists.
    temperature = Column(Float, default=FALLBACK_TEMPERATURE)
    top_p = Column(Float, default=FALLBACK_TOP_P)
    max_tokens = Column(Integer, default=FALLBACK_MAX_TOKENS)
    custom_prompt = Column(Text, default="")
    name = Column(String(255), nullable=False, index=True, default="New Conversation")
    # Per-conversation web-search toggle (#310): set at creation from the
    # value saved for the model (#609), else the GLOBAL user-settings default;
    # the conversation owns it afterwards (a later change never retro-affects
    # existing conversations).
    web_search_enabled = Column(Boolean, default=False, nullable=False)
    # How much the model may deliberate before answering (1.1.2), one of
    # REASONING_EFFORT_LEVELS. Same lifecycle as the web-search toggle: the
    # model's saved value, else the GLOBAL user-settings default, at creation;
    # owned by the conversation afterwards.
    reasoning_effort = Column(String(8), default=DEFAULT_REASONING_EFFORT, nullable=False)
    # The share of the working window, in whole percent, at which this
    # conversation compacts (#607), one of COMPACTION_THRESHOLDS. Same
    # lifecycle as the reasoning effort: the model's saved value, else the
    # GLOBAL user-settings default, at creation; owned by the conversation
    # afterwards.
    compaction_threshold = Column(
        SmallInteger, default=DEFAULT_COMPACTION_THRESHOLD, nullable=False
    )

    # Relationships — ordered by pk, NOT timestamp: PostgreSQL's now() is
    # frozen per transaction, so a user/assistant pair written in the same
    # request shares a timestamp.
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.id",
    )
    llm = relationship("Llm", back_populates="conversations")

    @validates("temperature")
    def validate_temperature(self, key: str, temperature: float) -> float:
        """Validate the temperature (``check_temperature``)."""
        return check_temperature(temperature)

    @validates("top_p")
    def validate_top_p(self, key: str, top_p: float) -> float:
        """Validate top_p (``check_top_p``)."""
        return check_top_p(top_p)

    @validates("max_tokens")
    def validate_max_tokens(self, key: str, max_tokens: int) -> int:
        """Validate max_tokens parameter is within allowed range.

        Args:
            key: Column name ("max_tokens").
            max_tokens: Max_tokens value to validate.

        Returns:
            Validated max_tokens value.

        Raises:
            ValueError: If max_tokens not in range [1, 32768].
        """
        if not 1 <= max_tokens <= 32768:
            raise ValueError("Max tokens must be between 1 and 32768")
        return max_tokens

    @validates("reasoning_effort")
    def validate_reasoning_effort(self, key: str, reasoning_effort: str) -> str:
        """Validate the reasoning effort (``check_reasoning_effort``)."""
        return check_reasoning_effort(reasoning_effort)

    @validates("compaction_threshold")
    def validate_compaction_threshold(self, key: str, compaction_threshold: int) -> int:
        """Validate the compaction threshold (``check_compaction_threshold``)."""
        return check_compaction_threshold(compaction_threshold)

    @hybrid_property
    def message_count(self) -> int:
        """Get total number of messages in this conversation.

        Returns:
            Message count (user + assistant messages).
        """
        return len(self.messages)

    @hybrid_property
    def last_message_time(self) -> Optional[datetime]:
        """Get timestamp of most recent message.

        Returns:
            Last message timestamp, or conversation created_at if no messages.
        """
        if self.messages:
            return max(msg.timestamp for msg in self.messages)
        return self.created_at

    def __repr__(self) -> str:
        """String representation of the conversation."""
        return (
            f"<Conversation(id={self.id}, name='{self.name}', " f"messages={self.message_count})>"
        )
