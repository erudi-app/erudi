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
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Float, Text, Boolean
from sqlalchemy.orm import relationship, validates
from src.agents.reasoning_effort import (
    DEFAULT_REASONING_EFFORT,
    REASONING_EFFORT_LEVELS,
)
from src.database.generation_hints import (
    FALLBACK_MAX_TOKENS,
    FALLBACK_TEMPERATURE,
    FALLBACK_TOP_P,
)
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql import func
from src.database.core import Base


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
    # Per-conversation web-search toggle (#310): copied from the GLOBAL
    # user-settings default at creation; the conversation owns it afterwards
    # (a later global change never retro-affects existing conversations).
    web_search_enabled = Column(Boolean, default=False, nullable=False)
    # How much the model may deliberate before answering (1.1.2), one of
    # REASONING_EFFORT_LEVELS. Same lifecycle as the web-search toggle: copied
    # from the GLOBAL user-settings default at creation, owned by the
    # conversation afterwards.
    reasoning_effort = Column(String(8), default=DEFAULT_REASONING_EFFORT, nullable=False)

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
        """Validate temperature parameter is within allowed range.

        Args:
            key: Column name ("temperature").
            temperature: Temperature value to validate.

        Returns:
            Validated temperature value.

        Raises:
            ValueError: If temperature not in range [0.0, 2.0].
        """
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return temperature

    @validates("top_p")
    def validate_top_p(self, key: str, top_p: float) -> float:
        """Validate top_p parameter is within allowed range.

        Args:
            key: Column name ("top_p").
            top_p: Top_p value to validate.

        Returns:
            Validated top_p value.

        Raises:
            ValueError: If top_p not in range [0.0, 1.0].
        """
        if not 0.0 <= top_p <= 1.0:
            raise ValueError("Top_p must be between 0.0 and 1.0")
        return top_p

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
        """Validate the reasoning effort is one of the five levels.

        Args:
            key: Column name ("reasoning_effort").
            reasoning_effort: Level to validate.

        Returns:
            Validated level.

        Raises:
            ValueError: If the level is not in REASONING_EFFORT_LEVELS.
        """
        if reasoning_effort not in REASONING_EFFORT_LEVELS:
            raise ValueError(f"Reasoning effort must be one of {REASONING_EFFORT_LEVELS}")
        return reasoning_effort

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
