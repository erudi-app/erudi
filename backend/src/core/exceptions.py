"""Core exception handling for Erudi.

This module defines the exception hierarchy used throughout the application.
All business exceptions inherit from AppBaseException and provide structured
error reporting with HTTP status codes and custom error codes.

Exception Hierarchy:
    AppBaseException (base class)
    ├── ModelNotFoundException (404, MODEL_NOT_FOUND)
    ├── InvalidInputException (422, INVALID_INPUT)
    ├── StateConflictException (400, STATE_CONFLICT)
    ├── DatabaseException (500, DATABASE_ERROR)
    ├── FileSystemException (500, FILESYSTEM_ERROR)
    ├── HuggingFaceAPIException (503, HUGGINGFACE_API_ERROR)
    ├── ModelLoadingException (500, MODEL_LOADING_ERROR)
    ├── GenerationException (500, GENERATION_ERROR)
    ├── KnowledgeBaseNotFoundException (404, KB_NOT_FOUND)
    ├── KnowledgeBaseCorruptedException (500, KB_CORRUPTED)
    ├── ConversationNotFoundException (404, CONVERSATION_NOT_FOUND)
    ├── MessageNotFoundException (404, MESSAGE_NOT_FOUND)
    ├── InsufficientMemoryException (507, INSUFFICIENT_MEMORY)
    ├── UnsupportedPlatformException (501, UNSUPPORTED_PLATFORM)
    ├── DownloadJobNotFoundException (404, DOWNLOAD_JOB_NOT_FOUND)
    ├── TokenizationException (500, TOKENIZATION_ERROR)
    ├── ConfigurationException (500, CONFIGURATION_ERROR)
    └── EngineException (500, LLM_ENGINE_FAILURE)

Fonctionnalités:
- Structured exception hierarchy with status codes
- Automatic error logging via structured logger
- FastAPI integration with JSON response handlers
- Custom Erudi error codes for client diagnostics
- Consistent error messages with remediation hints

Usage Patterns:
    **Creating New Exceptions:**
    All exceptions inherit from AppBaseException and provide:
    1. Appropriate HTTP status code (use fastapi.status constants)
    2. Custom Erudi error code (e.g., "KB_NOT_FOUND", "GENERATION_ERROR")
    3. Clear error messages with remediation hints
    4. Optional trace parameter for debugging context

    **Raising Exceptions:**
        from src.core.exceptions import ModelNotFoundException

        llm = db.query(Llm).filter(Llm.id == llm_id).first()
        if not llm:
            raise ModelNotFoundException(f"LLM {llm_id}")

    **Catching Specific Exceptions:**
        from src.core.exceptions import DatabaseException

        try:
            db.commit()
        except Exception as e:
            raise DatabaseException(f"Failed to persist changes: {e}", trace=str(e))

    **FastAPI Integration:**
        from fastapi import FastAPI
        from src.core.exceptions import app_base_exception_handler, AppBaseException

        app = FastAPI()
        app.add_exception_handler(AppBaseException, app_base_exception_handler)

Best Practices:
    - Never use bare `except:` - always catch specific exception types
    - All custom exceptions must be defined in this module
    - Include clear error messages with remediation hints
    - Use trace parameter to include original error context
    - Let exceptions bubble up to FastAPI exception handler
    - Log at appropriate levels (ERROR for exceptions)

Exception Categories:
    **Resource Not Found (404):**
    - ModelNotFoundException
    - KnowledgeBaseNotFoundException
    - ConversationNotFoundException
    - MessageNotFoundException
    - DownloadJobNotFoundException

    **Client Errors (400, 422):**
    - InvalidInputException (422)
    - StateConflictException (400)

    **Server Errors (500):**
    - DatabaseException
    - FileSystemException
    - ModelLoadingException
    - GenerationException
    - KnowledgeBaseCorruptedException
    - TokenizationException
    - ConfigurationException
    - EngineException

    **Service Unavailable (503):**
    - HuggingFaceAPIException

    **Not Implemented (501):**
    - UnsupportedPlatformException

    **Insufficient Storage (507):**
    - InsufficientMemoryException

See Also:
    - src.core.logging: Structured logging for error tracking
    - src.core.api: FastAPI application with exception handlers

"""

from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi import status
from typing import Any, Optional
from src.core.logging import logger
from src.core.request_context import get_request_id


class AppBaseException(Exception):
    """Base class for all Erudi business exceptions.

    Provides structured error handling with HTTP status codes, custom error codes,
    and automatic logging. All application-specific exceptions should inherit from
    this class to ensure consistent error reporting.

    Attributes:
        message: Human-readable error description, verbatim as raised.
        status_code: HTTP status code for the error response.
        erudi_code: Custom error code for client-side diagnostics.
        detail: Optional structured payload surfaced verbatim in the JSON
            response so a client can render a decision UI (e.g. the dependent
            KB assistants of a base model targeted for deletion). None omits it.

    """

    def __init__(
        self,
        message: str = "An unexpected error occured.",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        erudi_code: Optional[str] = None,
        trace: Optional[str] = None,
        detail: Optional[Any] = None,
    ):
        """Initialize the exception with error details.

        Args:
            message: Human-readable error description.
            status_code: HTTP status code (default: 500).
            erudi_code: Custom error code for diagnostics (default: "INTERNAL_SERVER_ERROR").
            trace: Optional stack trace or additional context.
            detail: Optional JSON-serializable payload returned in the response
                body (under error.detail) for the client to act on.

        Note:
            The message is exactly what the raiser wrote. Telling the user
            where to report a bug is the interface's job -- the Diagnostics
            panel in Settings owns that path -- so no support address or
            reporting instruction is appended to a payload that domain code
            also logs, matches on and shows in three other languages.

            All errors are logged via the structured logger. Severity follows
            the HTTP class: WARNING for client errors (< 500), ERROR for
            server errors (>= 500).

        """
        self.message = message
        self.status_code = status_code
        self.erudi_code = erudi_code or "INTERNAL_SERVER_ERROR"
        self.detail = detail
        log = logger.error if status_code >= 500 else logger.warning
        log(
            f"- Status Code: {status_code}\n- Erudi Custom Code: {erudi_code}\n- Message: {message}\n- Trace: {trace}"
        )
        super().__init__(message)

    def __repr__(self):
        return super().__repr__()


class ModelNotFoundException(AppBaseException):
    """Exception raised when a requested model cannot be found.

    Raised when attempting to load, access, or operate on a model that
    doesn't exist in the models directory or database.

    Examples:
        from src.core.exceptions import ModelNotFoundException
        raise ModelNotFoundException("llama-2-7b")

    """

    def __init__(self, model_name: str, trace: Optional[str] = None):
        """Initialize exception for missing model.

        Args:
            model_name: Name or ID of the model that was not found.
            trace: Optional stack trace or additional context.

        """
        super().__init__(
            message=f"Model '{model_name}' not found",
            status_code=status.HTTP_404_NOT_FOUND,
            erudi_code="MODEL_NOT_FOUND",
            trace=trace,
        )


class InvalidInputException(AppBaseException):
    """Exception raised for invalid user input or request parameters.

    Used when request validation fails beyond Pydantic schema validation,
    such as business logic constraints or cross-field validation.

    Examples:
        from src.core.exceptions import InvalidInputException
        if temperature < 0 or temperature > 2:
            raise InvalidInputException("temperature")

    """

    def __init__(self, field_name: str, trace: Optional[str] = None):
        """Initialize exception for invalid input field.

        Args:
            field_name: Name of the field that contains invalid data.
            trace: Optional stack trace or additional context.

        """
        super().__init__(
            f"Invalid input for '{field_name}'",
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            erudi_code="INVALID_INPUT",
            trace=trace,
        )


class StateConflictException(AppBaseException):
    """Exception raised when operation conflicts with current entity state.

    Used for operations that cannot proceed due to the current state of a resource,
    such as attempting to delete a model that is currently downloading, or starting
    a download when one is already in progress.

    Examples:
        from src.core.exceptions import StateConflictException
        if llm.local == 2:  # Downloading
            raise StateConflictException("Cannot delete LLM while downloading")

    """

    def __init__(
        self,
        message: str,
        trace: Optional[str] = None,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        detail: Optional[Any] = None,
    ):
        """Initialize state conflict exception.

        Args:
            message: Description of the state conflict.
            trace: Optional stack trace or additional context.
            status_code: HTTP status (default 400). Pass 409 for a resource
                conflict the client can resolve by opting in (e.g. deleting a
                base model that has dependent KB assistants).
            detail: Optional structured payload for the client (e.g. the
                dependents to render in a confirmation dialog).

        """
        super().__init__(
            message=message,
            status_code=status_code,
            erudi_code="STATE_CONFLICT",
            trace=trace,
            detail=detail,
        )


class DatabaseException(AppBaseException):
    """Exception raised for database operation failures.

    Raised when SQLAlchemy operations fail, including connection errors,
    transaction conflicts, constraint violations, or query execution failures.

    Examples:
        from src.core.exceptions import DatabaseException
        try:
            db.add(entity)
            db.commit()
        except SQLAlchemyError as e:
            raise DatabaseException(f"Failed to save entity: {e}", trace=str(e))

    """

    def __init__(self, message: str, trace: Optional[str] = None):
        """Initialize database exception with error details.

        Args:
            message: Description of the database failure.
            trace: Optional stack trace or additional context.

        """
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            erudi_code="DATABASE_ERROR",
            trace=trace,
        )


class FileSystemException(AppBaseException):
    """Exception raised for file system operation failures.

    Covers file not found, permission denied, disk full, I/O errors, and
    other filesystem-related failures during model loading, KB creation,
    or index persistence.

    Examples:
        from src.core.exceptions import FileSystemException
        import os
        if not os.path.exists(model_path):
            raise FileSystemException(f"Model directory not found: {model_path}")

    """

    def __init__(self, message: str, trace: Optional[str] = None):
        """Initialize filesystem exception with error details.

        Args:
            message: Description of the filesystem failure.
            trace: Optional stack trace or additional context.

        """
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            erudi_code="FILESYSTEM_ERROR",
            trace=trace,
        )


class HuggingFaceAPIException(AppBaseException):
    """Exception raised for HuggingFace Hub API operation failures.

    Covers network errors, authentication failures, rate limiting, model
    not found on Hub, and other API-related issues during model downloads
    or metadata fetching.

    Examples:
        from src.core.exceptions import HuggingFaceAPIException
        try:
            model_info = HF_API.model_info(model_id)
        except Exception as e:
            raise HuggingFaceAPIException(f"Failed to fetch model info: {e}")

    """

    def __init__(self, message: str, trace: Optional[str] = None):
        """Initialize HuggingFace API exception with error details.

        Args:
            message: Description of the API failure.
            trace: Optional stack trace or additional context.

        """
        super().__init__(
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            erudi_code="HUGGINGFACE_API_ERROR",
            trace=trace,
        )


class ModelLoadingException(AppBaseException):
    """Exception raised for model loading failures.

    Raised when model cannot be loaded into memory due to corrupted weights,
    incompatible format, insufficient memory, or missing configuration files.
    Covers failures across all engine backends (MLX, CUDA, CPU).

    Examples:
        from src.core.exceptions import ModelLoadingException
        try:
            model, tokenizer = mlx_vlm.load(model_path)
        except Exception as e:
            raise ModelLoadingException(f"Failed to load model: {e}", trace=str(e))

    """

    def __init__(self, message: str, trace: Optional[str] = None):
        """Initialize model loading exception with error details.

        Args:
            message: Description of the model loading failure.
            trace: Optional stack trace or additional context.

        """
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            erudi_code="MODEL_LOADING_ERROR",
            trace=trace,
        )


class GenerationException(AppBaseException):
    """Exception raised for text generation failures.

    Covers failures during LLM inference including tokenization errors,
    generation timeouts, context window overflow, and model execution errors.

    Examples:
        from src.core.exceptions import GenerationException
        try:
            async for chunk in chat_model.astream(messages):
                yield chunk.content
        except Exception as e:
            raise GenerationException(f"Generation failed: {e}", trace=str(e))

    """

    def __init__(self, message: str, trace: Optional[str] = None):
        """Initialize generation exception with error details.

        Args:
            message: Description of the generation failure.
            trace: Optional stack trace or additional context.

        """
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            erudi_code="GENERATION_ERROR",
            trace=trace,
        )


class KnowledgeBaseNotFoundException(AppBaseException):
    """Exception raised when a Knowledge Base cannot be found.

    Raised when attempting to access a KB that doesn't exist in the database
    or when associated resources (index, vector store) are missing.

    Examples:
        from src.core.exceptions import KnowledgeBaseNotFoundException
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
        if not kb:
            raise KnowledgeBaseNotFoundException(kb_id)

    """

    def __init__(self, kb_id: int, trace: Optional[str] = None):
        """Initialize exception for missing KB.

        Args:
            kb_id: ID of the KB that was not found.
            trace: Optional stack trace or additional context.

        """
        super().__init__(
            message=f"Knowledge Base with ID {kb_id} not found",
            status_code=status.HTTP_404_NOT_FOUND,
            erudi_code="KB_NOT_FOUND",
            trace=trace,
        )


class KnowledgeBaseCorruptedException(AppBaseException):
    """Exception raised when Knowledge Base retrieval is in a broken state.

    Raised when hybrid retrieval over the KB's chunks fails unexpectedly
    (vector store unreachable, inconsistent chunk data, …).

    Examples:
        from src.core.exceptions import KnowledgeBaseCorruptedException
        raise KnowledgeBaseCorruptedException(kb_id, "retrieval error", trace=str(e))

    """

    def __init__(self, kb_id: int, reason: str, trace: Optional[str] = None):
        """Initialize exception for corrupted KB.

        Args:
            kb_id: ID of the corrupted KB.
            reason: Description of the corruption issue.
            trace: Optional stack trace or additional context.

        """
        super().__init__(
            message=f"Knowledge Base {kb_id} is corrupted: {reason}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            erudi_code="KB_CORRUPTED",
            trace=trace,
        )


class ConversationNotFoundException(AppBaseException):
    """Exception raised when a conversation cannot be found.

    Raised when attempting to access, update, or delete a conversation that
    doesn't exist in the database.

    Examples:
        from src.core.exceptions import ConversationNotFoundException
        conv = db.query(Conversation).filter(Conversation.id == conv_id).first()
        if not conv:
            raise ConversationNotFoundException(conv_id)

    """

    def __init__(self, conversation_id: int, trace: Optional[str] = None):
        """Initialize exception for missing conversation.

        Args:
            conversation_id: ID of the conversation that was not found.
            trace: Optional stack trace or additional context.

        """
        super().__init__(
            message=f"Conversation with ID {conversation_id} not found",
            status_code=status.HTTP_404_NOT_FOUND,
            erudi_code="CONVERSATION_NOT_FOUND",
            trace=trace,
        )


class MessageNotFoundException(AppBaseException):
    """Exception raised when a message cannot be found by ID.

    Raised by MessageRepository when attempting to retrieve, update, or delete
    a message that doesn't exist in the database.

    Examples:
        from src.core.exceptions import MessageNotFoundException
        message = db.query(Message).filter(Message.id == msg_id).first()
        if not message:
            raise MessageNotFoundException(msg_id)

    """

    def __init__(self, message_id: int, trace: Optional[str] = None):
        """Initialize exception for missing message.

        Args:
            message_id: ID of the message that was not found.
            trace: Optional stack trace or additional context.

        """
        super().__init__(
            message=f"Message with ID {message_id} not found",
            status_code=status.HTTP_404_NOT_FOUND,
            erudi_code="MESSAGE_NOT_FOUND",
            trace=trace,
        )


class InsufficientMemoryException(AppBaseException):
    """Exception raised when system runs out of memory.

    Raised when model loading, inference, or KB creation fails due to
    insufficient RAM/VRAM. Provides remediation hints for reducing memory usage.

    Examples:
        from src.core.exceptions import InsufficientMemoryException
        try:
            model = load_large_model()
        except MemoryError as e:
            raise InsufficientMemoryException("Model loading", trace=str(e))

    """

    def __init__(self, operation: str, trace: Optional[str] = None):
        """Initialize exception for memory exhaustion.

        Args:
            operation: Description of operation that failed (e.g., "model loading").
            trace: Optional stack trace or additional context.

        """
        message = (
            f"Insufficient memory for {operation}. "
            f"Try: 1) Close other applications, 2) Use smaller model, "
            f"3) Increase system RAM/VRAM."
        )
        super().__init__(
            message=message,
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
            erudi_code="INSUFFICIENT_MEMORY",
            trace=trace,
        )


class UnsupportedPlatformException(AppBaseException):
    """Exception raised when operation is not supported on current platform.

    Raised when attempting to use engine-specific features on incompatible
    hardware (e.g., MLX on non-Mac, CUDA without NVIDIA GPU).

    Examples:
        from src.core.exceptions import UnsupportedPlatformException
        if not torch.cuda.is_available():
            raise UnsupportedPlatformException("CUDA", "No NVIDIA GPU detected")

    """

    def __init__(self, feature: str, reason: str, trace: Optional[str] = None):
        """Initialize exception for unsupported platform.

        Args:
            feature: Feature that is not supported (e.g., "MLX", "CUDA").
            reason: Explanation of why it's not supported.
            trace: Optional stack trace or additional context.

        """
        super().__init__(
            message=f"{feature} not supported on this platform: {reason}",
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            erudi_code="UNSUPPORTED_PLATFORM",
            trace=trace,
        )


class DownloadJobNotFoundException(AppBaseException):
    """Exception raised when a download job cannot be found.

    Raised when attempting to access or update a download job that doesn't
    exist in the database.

    Examples:
        from src.core.exceptions import DownloadJobNotFoundException
        job = db.query(DownloadJobModel).filter(DownloadJobModel.id == job_id).first()
        if not job:
            raise DownloadJobNotFoundException(job_id)

    """

    def __init__(self, job_id: int, trace: Optional[str] = None):
        """Initialize exception for missing download job.

        Args:
            job_id: ID of the download job that was not found.
            trace: Optional stack trace or additional context.

        """
        super().__init__(
            message=f"Download job with ID {job_id} not found",
            status_code=status.HTTP_404_NOT_FOUND,
            erudi_code="DOWNLOAD_JOB_NOT_FOUND",
            trace=trace,
        )


class TokenizationException(AppBaseException):
    """Exception raised for tokenization failures.

    Raised when tokenizer fails to encode text, apply chat template, or
    decode tokens. Covers tokenizer loading failures and encoding errors.

    Examples:
        from src.core.exceptions import TokenizationException
        try:
            tokens = tokenizer.encode(text)
        except Exception as e:
            raise TokenizationException(f"Failed to tokenize: {e}", trace=str(e))

    """

    def __init__(self, message: str, trace: Optional[str] = None):
        """Initialize tokenization exception with error details.

        Args:
            message: Description of the tokenization failure.
            trace: Optional stack trace or additional context.

        """
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            erudi_code="TOKENIZATION_ERROR",
            trace=trace,
        )


class ConfigurationException(AppBaseException):
    """Exception raised for configuration errors.

    Raised when environment variables are missing, configuration files are
    invalid, or system is misconfigured in a way that prevents operation.

    Examples:
        from src.core.exceptions import ConfigurationException
        if not os.getenv("DATABASE_URL"):
            raise ConfigurationException("DATABASE_URL environment variable not set")

    """

    def __init__(self, message: str, trace: Optional[str] = None):
        """Initialize configuration exception with error details.

        Args:
            message: Description of the configuration issue.
            trace: Optional stack trace or additional context.

        """
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            erudi_code="CONFIGURATION_ERROR",
            trace=trace,
        )


class EngineException(AppBaseException):
    """Exception raised for LLM engine failures during inference.

    Covers model loading errors, inference failures, out-of-memory conditions,
    and other runtime issues during model execution. A failure whose cause is
    identifiable carries an ``engine_code`` the UI acts on (a GPU the bundled
    CUDA build cannot drive, a driver too old to JIT its PTX, VRAM exhaustion);
    everything else leaves it unset and keeps the generic error turn.

    Examples:
        from src.core.exceptions import EngineException
        try:
            model.generate(prompt)
        except RuntimeError as e:
            raise EngineException(f"Inference failed: {e}")

    """

    def __init__(
        self,
        message: str,
        trace: Optional[str] = None,
        engine_code: Optional[str] = None,
    ):
        """Initialize engine exception with error details.

        Args:
            message: Description of the engine failure.
            trace: Optional stack trace or additional context.
            engine_code: Optional machine-readable cause, from
                ``src.engines.cuda_compatibility.CUDA_FAILURE_CODES``. It is a
                SECOND axis, orthogonal to ``erudi_code`` (which stays
                ``LLM_ENGINE_FAILURE`` for the HTTP layer): it says WHY the
                engine failed, so the renderer can offer the matching remedy
                instead of a generic apology. ``None`` -- the default -- is the
                generic path every other engine failure keeps taking.

        Attributes:
            engine_code: As above.
            engine_trace: The ``trace`` argument, kept on the instance so the
                caller can surface it verbatim (the child's captured output).
                ``AppBaseException`` only logs it.
        """
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            erudi_code="LLM_ENGINE_FAILURE",
            trace=trace,
        )
        self.engine_code = engine_code
        self.engine_trace = trace


class HardwareException(AppBaseException):
    """Exception raised for hardware detection and profiling failures.

    Covers GPU detection errors, hardware capability assessment failures,
    and issues during hardware profile creation or updates.

    Examples:
        from src.core.exceptions import HardwareException
        try:
            profile = detect_hardware()
        except RuntimeError as e:
            raise HardwareException(f"Hardware detection failed: {e}")

    """

    def __init__(self, message: str, trace: Optional[str] = None):
        """Initialize hardware exception with error details.

        Args:
            message: Description of the hardware failure.
            trace: Optional stack trace or additional context.

        """
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            erudi_code="HARDWARE_ERROR",
            trace=trace,
        )


async def app_base_exception_handler(request: Request, exc: AppBaseException):
    """Global exception handler for AppBaseException and subclasses.

    Converts application exceptions into structured JSON responses with
    appropriate HTTP status codes. Logs request details for debugging.

    Args:
        request: Incoming FastAPI request object.
        exc: The raised AppBaseException or subclass instance.

    Returns:
        JSONResponse with structured error information.

    Examples:
        Register in FastAPI application:
            from fastapi import FastAPI
            from src.core.exceptions import app_base_exception_handler, AppBaseException

            app = FastAPI()
            app.add_exception_handler(AppBaseException, app_base_exception_handler)

    """
    log = logger.error if exc.status_code >= 500 else logger.warning
    log(f"- Path: {request.url}")
    error: dict = {
        "type": exc.erudi_code,
        "message": exc.message,
    }
    # Structured payload for clients to act on (e.g. the choice dialog on a
    # guarded base-model delete). Omitted entirely when unset.
    if getattr(exc, "detail", None) is not None:
        error["detail"] = exc.detail
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": error,
        },
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Fallback handler for exceptions that escaped every domain handler.

    Logs the full traceback tagged with the current request id and returns a
    500 JSON body shaped exactly like app_base_exception_handler responses.
    Registered for plain ``Exception``: Starlette wires it into
    ServerErrorMiddleware, which sits OUTSIDE the request-logging middleware,
    so this response must carry the X-Request-ID header itself.

    Args:
        request: Incoming FastAPI request object.
        exc: The unhandled exception instance.

    Returns:
        JSONResponse with structured error information (HTTP 500).
    """
    request_id = get_request_id()
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: {exc}",
        exc_info=exc,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "type": "INTERNAL_SERVER_ERROR",
                "message": f"Unexpected server error: {exc}",
            },
        },
        headers={"X-Request-ID": request_id},
    )
