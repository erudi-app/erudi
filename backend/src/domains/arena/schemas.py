"""Pydantic validation schemas for arena stateless queries.

Defines request schemas for the arena domain, validating query parameters and
generation settings for stateless LLM testing.

Example:
    from src.domains.arena.schemas import ArenaQueryPayload

    query = ArenaQueryPayload(
        question="What is quantum computing?",
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=512,
        custom_prompt="Use simple language"
    )
"""

from pydantic import BaseModel, Field, model_validator
from typing import List, Optional


class ArenaQueryPayload(BaseModel):
    """Request schema for stateless arena queries with generation parameters.

    Validates user input for arena LLM queries, enforcing ranges for sampling parameters
    and ensuring non-empty questions. Includes optional custom instructions for specialized
    prompting.

    Attributes:
        question: The question/prompt to send to the model (trimmed; may be empty
            only when images or documents are attached).
        images: Optional base64 data-URL images attached to the question (vision models).
        attachments: Local filesystem paths of documents/folders attached to the question.
        temperature: Sampling temperature (0.0=deterministic, 2.0=creative, default=0.1).
        top_p: Nucleus sampling threshold (0.0-1.0, default=0.5).
        max_new_tokens: Maximum tokens to generate (1-8192, default=1024).
        custom_prompt: Optional additional instructions appended to system prompt.

    Example:
        >>> payload = ArenaQueryPayload(
        ...     question="Explain relativity",
        ...     temperature=0.7,
        ...     top_p=0.9,
        ...     max_new_tokens=512,
        ...     custom_prompt="Use analogies"
        ... )
    """

    question: str = Field(..., description="The question to ask the model")
    images: Optional[List[str]] = Field(
        default=None,
        description="Optional base64 data-URL images attached to the question (vision models)",
    )
    attachments: Optional[List[str]] = Field(
        default=None,
        description=(
            "Local filesystem paths of documents (or folders of documents) attached "
            "to the question; read on this machine, never uploaded (#492)"
        ),
    )
    # None = "use the model's own defaults" (#388), resolved by the service from
    # the Llm row (captured generation_config, else the neutral constants);
    # explicit wins.
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0-2.0); None resolves the model's default (#388)",
    )
    top_p: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold (0.0-1.0); None resolves the model's default (#388)",
    )
    max_new_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=32768,
        description="Maximum number of tokens to generate; None resolves the model's default (#388)",
    )
    custom_prompt: Optional[str] = Field(
        default=None, description="Optional additional instructions for the model"
    )

    @model_validator(mode="after")
    def validate_question_or_attachments(self):
        """Ensure the turn carries text, at least one image, or a document.

        Image-only asks are valid vision-model turns (mirrors conversations,
        #136 C) and a document-only ask is a valid "what does this say?" turn
        (#492); a fully empty ask is still rejected.

        Returns:
            The payload with a trimmed question.

        Raises:
            ValueError: If question is empty/whitespace and nothing is attached.
        """
        self.question = self.question.strip()
        if not self.question and not self.images and not self.attachments:
            raise ValueError("Question cannot be empty or whitespace")
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the capital of France?",
                "temperature": 0.7,
                "top_p": 0.9,
                "max_new_tokens": 512,
                "custom_prompt": "Please answer concisely.",
            }
        }
