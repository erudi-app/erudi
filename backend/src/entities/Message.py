"""SQLAlchemy entity for individual messages within conversations.

Represents a single user or assistant message with metadata for starring and
timestamp tracking. Messages belong to a Conversation; insertion order is the
primary key (PostgreSQL's ``now()`` is frozen per transaction, so timestamps
can collide within one request — never order by them).

Relationships:
    - conversation: Many-to-one with Conversation (parent session).

Example:
    from src.entities.Message import Message

    msg = Message(
        conversation_id=42,
        sender="user",
        content="What is quantum computing?",
        starred=False
    )
"""

from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Boolean, Text, JSON
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func
from src.database.core import Base


class Message(Base):
    """SQLAlchemy model for conversation messages with starring support.

    Stores individual messages within conversations, tracking sender, content,
    server-stamped timestamp, and starred status (for memory injection).

    Attributes:
        id: Primary key (auto-increment) — also the insertion order.
        conversation_id: Foreign key to Conversation (server-side CASCADE).
        sender: Message sender - "user" or "llm" (validated).
        content: Message text content (non-empty, no length cap).
        timestamp: Message creation timestamp (server-stamped).
        starred: True if user starred for importance (used in memory injection).
        trace: Optional JSON list of the assistant turn's non-answer stream
            events (thinking / tool_call / tool_result, and the compaction
            ``done`` events), for replay on reload.
        conversation: Relationship to Conversation entity.

    Example:
        >>> msg = Message(conversation_id=42, sender="user", content="Hello!")
        >>> print(msg.starred)  # False (starring is done via MessageRepository)
    """

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(
        Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    sender = Column(String(50), nullable=False)  # "user" or "llm"
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, server_default=func.now(), nullable=False)
    starred = Column(Boolean, default=False, nullable=False)
    # Ordered list of non-answer stream events for an assistant turn (#90):
    # thinking / tool_call / tool_result (and a truncated marker when capped).
    # Nullable: user messages and turns with neither thinking nor tools have none.
    trace = Column(JSON, nullable=True)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")

    @validates("sender")
    def validate_sender(self, key: str, sender: str) -> str:
        """Validate sender is either 'user' or 'llm'.

        Args:
            key: Column name ("sender").
            sender: Sender value to validate.

        Returns:
            Validated sender value.

        Raises:
            ValueError: If sender not in ['user', 'llm'].
        """
        if sender not in ["user", "llm"]:
            raise ValueError("Sender must be either 'user' or 'llm'")
        return sender

    @validates("content")
    def validate_content(self, key: str, content: str) -> str:
        """Validate content is non-empty.

        Args:
            key: Column name ("content").
            content: Content value to validate.

        Returns:
            Validated content value.

        Raises:
            ValueError: If content is empty or whitespace-only.
        """
        if not content or len(content.strip()) == 0:
            raise ValueError("Message content cannot be empty")
        return content

    def __repr__(self) -> str:
        """String representation of the message."""
        return (
            f"<Message(id={self.id}, sender='{self.sender}', "
            f"content='{self.content[:50]}...', starred={self.starred})>"
        )
