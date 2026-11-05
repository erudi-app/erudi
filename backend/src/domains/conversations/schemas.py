"""Pydantic schemas for conversation and message data validation.

This module defines request/response schemas for:
- Message creation and responses (user/assistant messages)
- Conversation CRUD operations (create, update, read)
- Query parameters for streaming generation
- Bulk operations (delete multiple conversations)
- Message starring (bookmarking)

Schema Hierarchy:
    ::

        MessageBase → MessageResponse
        ConversationBase → ConversationCreate, ConversationUpdate, ConversationResponse
        ConversationResponse → ConversationWithMessagesResponse

Validation Rules:
    - Message content: non-empty, no length cap
    - Message sender: Must be "user" or "assistant"
    - Temperature: 0.0-2.0
    - Top-p: 0.0-1.0
    - Max tokens: 1-32768
    - Custom prompt: no character cap (the model's context window bounds it)
    - A field left out of a creation is resolved per model
      (``src.domains.llms.conversation_defaults``)

Example:
    Create conversation request::

        payload = ConversationCreate(
            llm_id=5,
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
            custom_prompt="You are a helpful coding assistant."
        )

    Query conversation with message::

        query = ConversationQuery(
            question="Explain asyncio in Python",
            temperature=0.5,
        )

Note:
    All schemas use Pydantic v2 field validation with clear error messages.
    from_attributes=True allows ORM model conversion to schema.
"""

"""
Pydantic schemas for conversation-related data validation.
"""
from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import Annotated, List, Optional

from src.domains.user_settings.schemas import CompactionThreshold, ReasoningEffort
from src.entities.Conversation import TEMPERATURE_MAX, TEMPERATURE_MIN, TOP_P_MAX, TOP_P_MIN

# The conversation values a request may carry, with the bounds the entity
# enforces (``src.entities.Conversation``), one bound per value on every path:
# conversation creation, update and turn query, the Arena query and the
# per-model defaults (``src.domains.llms.schemas.ConversationDefaultsUpdate``).
Temperature = Annotated[float, Field(ge=TEMPERATURE_MIN, le=TEMPERATURE_MAX)]
TopP = Annotated[float, Field(ge=TOP_P_MIN, le=TOP_P_MAX)]
# Custom instructions have no character cap: the model's context window is
# their bound (an oversized prompt gets the honest overflow answer, #599), and
# instructions stored before the per-model defaults may be of any length.
CustomPrompt = str


class MessageBase(BaseModel):
    """Base schema for messages with sender and content validation.

    Attributes:
        sender: Message sender, must be "user" or "assistant".
        content: Message content, non-empty, no length cap.
    """

    """Base schema for messages."""
    sender: str = Field(..., description="Message sender (user or assistant)")
    content: str = Field(..., min_length=1, description="Message content")

    @validator("sender")
    def validate_sender(cls, v):
        """Ensure sender is either 'user', 'assistant', or 'llm'.

        Args:
            v: Sender value to validate.

        Returns:
            str: Validated sender value.

        Raises:
            ValueError: If sender is not "user", "assistant", or "llm".
        """
        if v not in ["user", "assistant", "llm"]:
            raise ValueError('Sender must be either "user", "assistant", or "llm"')
        return v


class MessageResponse(MessageBase):
    """Schema for message responses with database fields.

    Extends MessageBase with id, conversation_id, timestamp, starred.

    Attributes:
        id: Database message ID.
        conversation_id: Parent conversation ID.
        timestamp: Message creation timestamp.
        starred: Whether message is bookmarked (default False).
        trace: Optional list of the assistant turn's non-answer stream events
            (thinking / tool_call / tool_result, and the compaction ``done``
            events, #614) for replay on reload (#90).
    """

    """Schema for message responses."""
    id: int = Field(..., description="Message ID")
    conversation_id: int = Field(..., description="Parent conversation ID")
    timestamp: datetime = Field(..., description="Message creation timestamp")
    starred: bool = Field(default=False, description="Whether message is starred")
    trace: Optional[List[dict]] = Field(
        default=None,
        description=(
            "Assistant turn's non-answer stream events (thinking/tool activity, "
            "compaction markers), for replay (#90, #614)"
        ),
    )

    class Config:
        from_attributes = True


class ConversationBase(BaseModel):
    """Base schema for conversations with required LLM ID.

    Attributes:
        llm_id: ID of the LLM model to use for this conversation.
    """

    """Base schema for conversations."""
    llm_id: int = Field(..., description="ID of the LLM to use")


class ConversationCreate(ConversationBase):
    """Schema for creating new conversations with generation parameters.

    Attributes:
        llm_id: Inherited from ConversationBase.
        temperature: Sampling temperature (0.0-2.0).
        top_p: Nucleus sampling threshold (0.0-1.0).
        max_tokens: Legacy fallback ceiling, used ONLY when the engine
            cannot report a context window; with a window, the output
            budget is automatic and this value is ignored.
        custom_prompt: Custom system prompt (no character cap).
        web_search_enabled, reasoning_effort, compaction_threshold: The
            conversation's own settings.

        Every field but ``llm_id`` is optional; one left out (None) is
        resolved per model, see the field comments below.

    Example:
        ::

            conv = ConversationCreate(
                llm_id=3,
                temperature=0.8,
                top_p=0.95,
                max_tokens=2048,
                custom_prompt="You are a creative storyteller."
            )
    """

    """Schema for creating new conversations."""
    # None = "resolve it" (#388, #609): every field the client leaves out is
    # resolved by ``src.domains.llms.conversation_defaults`` -- the value saved
    # for the model with ``/config global``, else the user settings (effort,
    # web search, threshold) or the model's recommendation (temperature,
    # top_p), else a constant. An explicit value always wins.
    temperature: Optional[Temperature] = Field(
        default=None,
        description="Sampling temperature; None resolves the model's saved or recommended value",
    )
    top_p: Optional[TopP] = Field(
        default=None,
        description="Nucleus sampling threshold; None resolves the model's saved or recommended value",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=32768,
        description="Maximum number of tokens to generate; None resolves the model's default (#388)",
    )
    custom_prompt: Optional[CustomPrompt] = Field(
        default=None,
        description=(
            "Custom system prompt. None resolves the instructions saved for the "
            "model (#609), else empty; an explicit value, empty included, wins."
        ),
    )
    web_search_enabled: Optional[bool] = Field(
        default=None,
        description=(
            "Per-conversation web-search toggle (#310). None resolves the value "
            "saved for the model, else the global user-settings default; an "
            "explicit value wins."
        ),
    )
    reasoning_effort: Optional[ReasoningEffort] = Field(
        default=None,
        description=(
            "How much the model may deliberate before answering. None resolves "
            "the level saved for the model, else the global user-settings "
            "default; an explicit value wins."
        ),
    )
    compaction_threshold: Optional[CompactionThreshold] = Field(
        default=None,
        description=(
            "Share of the working window, in percent (50-90, step 5), at which "
            "the conversation compacts (#607). None resolves the threshold saved "
            "for the model, else the global user-settings default; an explicit "
            "value wins."
        ),
    )


class ConversationUpdate(ConversationBase):
    """Schema for partial conversation updates (PATCH operations).

    All fields are optional to support partial updates. Only provided
    fields will be updated in the database. A different ``llm_id`` switches
    the model and applies its saved values and recommended sampling
    (``ConversationService.update_conversation``).

    Attributes:
        llm_id: Optional LLM ID override.
        name: Optional conversation name.
        temperature: Optional temperature override.
        top_p: Optional top_p override.
        max_tokens: Legacy fallback ceiling (window-less engines only);
            persisted but ignored whenever the engine reports a window.
        custom_prompt: Optional system prompt override.

    Example:
        ::

            update = ConversationUpdate(
                name="Python Best Practices",
                temperature=0.3  # Only update name and temperature
            )
    """

    """Schéma utilisé pour les mises à jour partielles (PATCH)."""

    llm_id: Optional[int] = None
    name: Optional[str] = None
    temperature: Optional[Temperature] = None
    top_p: Optional[TopP] = None
    max_tokens: Optional[int] = None
    custom_prompt: Optional[CustomPrompt] = None
    web_search_enabled: Optional[bool] = None
    reasoning_effort: Optional[ReasoningEffort] = None
    compaction_threshold: Optional[CompactionThreshold] = None


class ConversationResponse(ConversationBase):
    """Schema for conversation responses with full metadata.

    Attributes:
        id: Conversation database ID.
        created_at: Conversation creation timestamp.
        last_message_time: Timestamp of most recent message.
        name: Conversation display name.
        temperature: Current temperature setting.
        top_p: Current top_p setting.
        max_tokens: Current max_tokens setting.
        custom_prompt: Current system prompt.
    """

    id: int
    # Overrides the required llm_id from ConversationBase: NULL once the bound
    # model is deleted (#225). The conversation survives, unbound, awaiting a
    # switch to another model -- so reads must not 500 on a null FK.
    llm_id: Optional[int] = None
    created_at: datetime
    last_message_time: datetime
    name: str
    temperature: float
    top_p: float
    max_tokens: int
    custom_prompt: str
    web_search_enabled: bool
    reasoning_effort: ReasoningEffort
    compaction_threshold: CompactionThreshold

    class Config:
        from_attributes = True


class ConversationWithMessagesResponse(ConversationResponse):
    """Extended conversation response including full message history.

    Attributes:
        messages: List of all MessageResponse objects in conversation.

    Example:
        ::

            GET /conversations/42
            → ConversationWithMessagesResponse {
                id: 42,
                name: "Python Tutorial",
                messages: [
                  MessageResponse(...),
                  MessageResponse(...)
                ]
              }
    """

    messages: List[MessageResponse]


class ConversationQuery(BaseModel):
    """Schema for querying conversations with streaming generation parameters.

    Attributes:
        question: User message/query to send to LLM.
        images: Optional base64 data-URL images attached to the question (vision models).
        image_paths: Local filesystem path of each image, parallel to ``images``.
        attachments: Local filesystem paths of documents/folders attached to the question.
        temperature: Optional temperature override for this query only (0.0-2.0).
        top_p: Optional top_p override for this query only (0.0-1.0).
        max_new_tokens: Accepted and ignored -- the output budget is computed per
            model call from the context window the model runs in
            (``src.agents.output_budget``), so nothing on the client sets it.
            The field stays on the schema so an older client is answered, not
            rejected with a 422.
        custom_prompt: Optional system prompt override for this query only;
            left out (None), the conversation's stored instructions apply, and
            an explicit "" runs the turn without any.

    Example:
        ::

            query = ConversationQuery(
                question="Explain list comprehensions",
                temperature=0.3,
            )
    """

    question: str
    images: Optional[List[str]] = Field(
        default=None,
        description="Optional base64 data-URL images attached to the question (vision models)",
    )
    image_paths: Optional[List[str]] = Field(
        default=None,
        description="Local filesystem paths for each image (parallel to images, empty string when unavailable)",
    )
    attachments: Optional[List[str]] = Field(
        default=None,
        description=(
            "Local filesystem paths of documents (or folders of documents) attached "
            "to the question; read on this machine, never uploaded (#492)"
        ),
    )
    temperature: Optional[Temperature] = None
    top_p: Optional[TopP] = None
    max_new_tokens: Optional[int] = None
    custom_prompt: Optional[CustomPrompt] = None


class MessageStarRequest(BaseModel):
    """Schema for starring/unstarring messages.

    Attributes:
        message_id: ID of the message to toggle star state.

    Example:
        ::

            POST /conversations/star_message
            {
              "message_id": 345
            }
    """

    message_id: int = Field(..., description="ID of the message to toggle star state")


class CompactionResponse(BaseModel):
    """What a manual compaction did (``POST /conversations/{id}/compact``).

    ``summarized`` counts the questions and answers the summary replaced
    (0: nothing to compact, and then nothing else is said); ``reason`` is
    ``length`` or ``memory``, as on the automatic compaction's separator;
    ``summary`` is the summary's text. The separator itself is saved in the
    last answer's trace.
    """

    summarized: int
    reason: Optional[str] = None
    summary: Optional[str] = None


class ContextCategories(BaseModel):
    """The context panel's blocks, in estimated tokens.

    ``system_prompt + tools + messages + summary`` is what the next request
    would carry (``used``); ``free`` is what is left below the compaction
    threshold and ``autocompaction_buffer`` the reserve above it, so the six
    add up to the working window while ``used`` fits in it.
    """

    system_prompt: int
    tools: int
    messages: int
    summary: int
    free: int
    autocompaction_buffer: int


class ContextWindow(BaseModel):
    """The window the counts are a share of.

    ``working`` is the denominator: the allocated window folded with the
    memory ceiling when this conversation's model is loaded
    (``model_loaded``), the trained window otherwise; ``None`` when neither is
    known. ``allocated`` exists only while the model is loaded;
    ``reduced_by_memory`` says this computer's memory bounds it.
    """

    working: Optional[int] = None
    allocated: Optional[int] = None
    trained: Optional[int] = None
    reduced_by_memory: bool
    model_loaded: bool


class ContextUsageResponse(BaseModel):
    """``GET /conversations/{id}/context``: a snapshot, never stored.

    ``kb_excerpts_excluded``: an assistant that searches its knowledge base on
    every question is counted without the excerpts a question retrieves.
    ``over_window``: the next request would not fit the window (it will be
    summarized first).
    """

    categories: ContextCategories
    window: ContextWindow
    kb_excerpts_excluded: bool
    over_window: bool
