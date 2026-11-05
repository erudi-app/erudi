"""SQLAlchemy entity for the conversation defaults saved for one model (#609).

``/config global`` saves a conversation's values as the defaults of its model:
a new conversation on that model starts from them, ``/model`` brings them
along, and an Arena panel runs at the saved effort, temperature and top_p
(the Arena has no threshold or instructions, and its web search follows the
global setting). One row per model at most, keyed by
the model's id; every value column is nullable and NULL means "not saved"
(the value then comes from the Settings or the model, see
``src.domains.llms.conversation_defaults``).

The row belongs to one downloaded model: the foreign key cascades, so deleting
the model deletes its row in the database. ``Llm`` has no relationship to this
entity, so a ``db.delete(llm)`` issues a single DELETE and leaves the cascade
to PostgreSQL. A re-download is a new ``llms`` row and starts with nothing
saved; a knowledge-base assistant is its own row and keeps its values when it
is re-bound to another base.

The value rules are the conversation's own (``src.entities.Conversation``
``check_*`` helpers), applied to every value that is not NULL.
"""

from __future__ import annotations

from sqlalchemy import Boolean, Column, Float, ForeignKey, Integer, SmallInteger, String, Text
from sqlalchemy.orm import validates

from src.database.core import Base
from src.entities.Conversation import (
    check_compaction_threshold,
    check_reasoning_effort,
    check_temperature,
    check_top_p,
)


class ModelDefaults(Base):
    """The conversation values saved for one model; NULL = not saved.

    Attributes:
        llm_id: The model (primary key, foreign key to ``llms.id``, ON DELETE
            CASCADE).
        reasoning_effort: One of REASONING_EFFORT_LEVELS.
        temperature: 0.0-2.0.
        top_p: 0.0-1.0.
        compaction_threshold: 50-90 in steps of 5 (percent).
        web_search_enabled: The web-search switch.
        custom_prompt: Custom instructions; never empty (empty instructions
            clear the saved ones, see the PUT route).
    """

    __tablename__ = "model_defaults"

    llm_id = Column(Integer, ForeignKey("llms.id", ondelete="CASCADE"), primary_key=True)
    reasoning_effort = Column(String(8), nullable=True)
    temperature = Column(Float, nullable=True)
    top_p = Column(Float, nullable=True)
    compaction_threshold = Column(SmallInteger, nullable=True)
    web_search_enabled = Column(Boolean, nullable=True)
    custom_prompt = Column(Text, nullable=True)

    @validates("temperature")
    def validate_temperature(self, key, value):
        return None if value is None else check_temperature(value)

    @validates("top_p")
    def validate_top_p(self, key, value):
        return None if value is None else check_top_p(value)

    @validates("reasoning_effort")
    def validate_reasoning_effort(self, key, value):
        return None if value is None else check_reasoning_effort(value)

    @validates("compaction_threshold")
    def validate_compaction_threshold(self, key, value):
        return None if value is None else check_compaction_threshold(value)

    def __repr__(self) -> str:
        return f"<ModelDefaults(llm_id={self.llm_id})>"
