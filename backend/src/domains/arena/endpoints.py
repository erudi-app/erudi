"""REST API endpoints for stateless LLM testing in the arena.

The arena provides a lightweight query interface for testing models without creating
conversations or storing history. Useful for quick model comparisons, experimentation,
and benchmarking.

Key Features:
- **Stateless**: No conversation history or message storage.
- **Streaming**: Real-time token generation via Server-Sent Events.
- **KB-aware**: Automatically injects relevant context from attached Knowledge Bases.
- **Customizable**: Supports temperature, top_p, and custom instructions. The output
  budget is not a request parameter: the backend derives it from the context window
  the model runs in (see docs/guides/conversations.md).

Architecture:
    ┌──────────────┐
    │ POST /arena/ │
    │ {llm_id}/query│
    └───────┬──────┘
            │ (1) Validate llm_id + payload (ArenaQueryPayload)
            ↓
    ┌──────────────┐
    │ ArenaService │ ← get_prompting_strategy(param_size)
    │.query_llm_   │ ← build_agent_system_prompt() + KB context
    │ stream()     │ ← AgentRunner (stateless) → ChatOpenAI(base_url)
    └───────┬──────┘
            │ (2) Yield tokens via StreamingResponse
            ↓
    ┌──────────────┐
    │ Client       │ ← Receives text/plain stream
    └──────────────┘

Use Cases:
    - Model comparison: Query 2+ models with identical prompt, compare outputs.
    - Quick testing: Test custom prompts without creating conversations.
    - Benchmarking: Measure generation speed and quality for different models.

Endpoints:
    - POST /arena/{llm_id}/query → Stream stateless response for given question.

Example:
    POST /arena/42/query
    {
        "question": "Explain quantum entanglement in simple terms.",
        "temperature": 0.7,
        "top_p": 0.9,
        "custom_prompt": "Use analogies suitable for a 10-year-old."
    }
    Response: StreamingResponse(text/plain) → "Imagine two magic coins..."
"""

from fastapi import Depends, APIRouter
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from src.database.core import get_db
from src.domains.arena.schemas import ArenaQueryPayload
from src.domains.arena.services import ArenaService
from src.core.logging import logger

router = APIRouter(prefix="/arena", tags=["arena"])


def get_arena_service(db: Session = Depends(get_db)) -> ArenaService:
    """Dependency injection provider for ArenaService.

    Args:
        db: Database session injected by FastAPI.

    Returns:
        Configured ArenaService instance with database access.
    """
    return ArenaService(db)


@router.post("/{llm_id}/query")
async def query_arena(
    llm_id: int, payload: ArenaQueryPayload, service: ArenaService = Depends(get_arena_service)
):
    """Stream stateless LLM response without conversation history.

    Queries a model with a single question, no context from previous messages. Useful
    for quick testing, benchmarking, and model comparison. Automatically injects KB
    context if the model has a Knowledge Base attached.

    Args:
        llm_id: Database ID of the LLM to query.
        payload: Query request with question, temperature, top_p, custom_prompt.
        service: ArenaService instance injected by FastAPI.

    Returns:
        StreamingResponse: Real-time token stream (text/plain).

    Raises:
        HTTPException: 404 if llm_id not found, 500 on model loading or generation errors.

    Example:
        POST /arena/42/query
        {
            "question": "What is the Heisenberg uncertainty principle?",
            "temperature": 0.7,
            "top_p": 0.9,
            "custom_prompt": "Explain like I'm 5 years old."
        }
        Response: StreamingResponse → "Imagine you have a tiny ball..."
    """
    logger.info(f"Arena query request for LLM {llm_id}")

    # Validate LLM exists BEFORE starting StreamingResponse
    # (StreamingResponse returns 200 immediately, exceptions inside async generator are lost)
    service._get_llm(llm_id)

    return StreamingResponse(
        service.query_llm_stream(llm_id, payload),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
