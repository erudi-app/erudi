# Exception Handling

How errors are raised in the backend, what they look like on the wire, and how the
frontend consumes them.

## Principles

- Domain code never raises a bare `Exception`. It raises an `AppBaseException` subclass
  from `backend/src/core/exceptions.py`.
- Every exception carries an HTTP status code and an `erudi_code`, so a client can branch
  on the error type instead of matching on message text.
- Constructing an exception logs it: `WARNING` for client errors (status < 500), `ERROR`
  for server errors (status >= 500). There is no separate "log then raise" step.
- Never use a bare `except:`. Catch the specific exceptions you can handle, and let the
  rest bubble up to the global handlers.

## The base class

```python
class AppBaseException(Exception):
    def __init__(
        self,
        message: str = "An unexpected error occured.",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        erudi_code: str | None = None,
        trace: str | None = None,
        detail: Any | None = None,
    ): ...
```

| Parameter | Meaning |
|---|---|
| `message` | Human-readable description, exactly as the raiser wrote it. Nothing is appended: telling the user where to report a bug belongs to the interface (the Diagnostics page), not to a payload that is also logged, matched on and shown in four languages. |
| `status_code` | HTTP status of the response. Defaults to 500. |
| `erudi_code` | Machine-readable code surfaced as `error.type`. Defaults to `INTERNAL_SERVER_ERROR`. |
| `trace` | Extra debugging context. **Logged only** — never returned to the client. |
| `detail` | JSON-serializable payload returned verbatim under `error.detail`, so a client can render a decision UI. Omitted entirely when `None`. |

## Exception hierarchy

All classes below live in `backend/src/core/exceptions.py`.

| Class | Constructor | Status | `erudi_code` |
|---|---|---|---|
| `ModelNotFoundException` | `(model_name, trace=None)` | 404 | `MODEL_NOT_FOUND` |
| `KnowledgeBaseNotFoundException` | `(kb_id, trace=None)` | 404 | `KB_NOT_FOUND` |
| `ConversationNotFoundException` | `(conversation_id, trace=None)` | 404 | `CONVERSATION_NOT_FOUND` |
| `MessageNotFoundException` | `(message_id, trace=None)` | 404 | `MESSAGE_NOT_FOUND` |
| `DownloadJobNotFoundException` | `(job_id, trace=None)` | 404 | `DOWNLOAD_JOB_NOT_FOUND` |
| `InvalidInputException` | `(field_name, trace=None)` | 422 | `INVALID_INPUT` |
| `StateConflictException` | `(message, trace=None, status_code=400, detail=None)` | 400, or 409 on request | `STATE_CONFLICT` |
| `DatabaseException` | `(message, trace=None)` | 500 | `DATABASE_ERROR` |
| `FileSystemException` | `(message, trace=None)` | 500 | `FILESYSTEM_ERROR` |
| `ModelLoadingException` | `(message, trace=None)` | 500 | `MODEL_LOADING_ERROR` |
| `GenerationException` | `(message, trace=None)` | 500 | `GENERATION_ERROR` |
| `KnowledgeBaseCorruptedException` | `(kb_id, reason, trace=None)` | 500 | `KB_CORRUPTED` |
| `TokenizationException` | `(message, trace=None)` | 500 | `TOKENIZATION_ERROR` |
| `ConfigurationException` | `(message, trace=None)` | 500 | `CONFIGURATION_ERROR` |
| `EngineException` | `(message, trace=None)` | 500 | `LLM_ENGINE_FAILURE` |
| `HardwareException` | `(message, trace=None)` | 500 | `HARDWARE_ERROR` |
| `UnsupportedPlatformException` | `(feature, reason, trace=None)` | 501 | `UNSUPPORTED_PLATFORM` |
| `HuggingFaceAPIException` | `(message, trace=None)` | 503 | `HUGGINGFACE_API_ERROR` |
| `InsufficientMemoryException` | `(operation, trace=None)` | 507 | `INSUFFICIENT_MEMORY` |

Note the argument shapes: the "not found" exceptions take the **identifier**, not a
sentence, and build their own message. `InvalidInputException` takes a **field name**.

Adding a new exception means adding it to this module — nowhere else.

## Raising exceptions

```python
from src.core.exceptions import (
    ModelNotFoundException,
    InvalidInputException,
    DatabaseException,
    StateConflictException,
)

# 404 — pass the identifier, not a message
llm = db.query(Llm).filter(Llm.id == llm_id).first()
if not llm:
    raise ModelNotFoundException(f"LLM {llm_id}")

# 422 — pass the offending field name
if not payload.question or not payload.question.strip():
    raise InvalidInputException("question")

# 500 — the original error goes in `trace`, which is logged and not returned
try:
    db.add(conversation)
    db.commit()
except SQLAlchemyError as e:
    raise DatabaseException("Failed to create conversation", trace=str(e))
```

### Conflicts the client can resolve

`StateConflictException` defaults to 400. Pass `status_code=409` together with a `detail`
payload when the client can opt into resolving the conflict — the guarded delete of a
base model that has dependent Knowledge Base assistants is the canonical case:

```python
raise StateConflictException(
    "Deleting this base model would break its Knowledge Base assistants",
    status_code=409,
    detail={"dependents": [{"id": a.id, "name": a.name} for a in dependents]},
)
```

The client reads `error.detail` and re-issues the delete with the opt-in query
parameter.

### Catching

```python
try:
    llm = self.get_llm_by_id(llm_id)
    self.update_metadata(llm)
except (ModelNotFoundException, DatabaseException):
    raise                      # already structured — let it bubble up
except Exception as e:
    raise DatabaseException("Unexpected error during update", trace=str(e))
```

## The wire format

Two handlers are registered in `add_exception_handlers`
(`backend/src/core/api.py`):

```python
app.add_exception_handler(AppBaseException, app_base_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)
```

`app_base_exception_handler` responds with the exception's own status code:

```json
{
  "success": false,
  "error": {
    "type": "MODEL_NOT_FOUND",
    "message": "Model 'qwen2.5-7b' not found",
    "detail": { "…": "present only when the exception carries one" }
  }
}
```

`error.detail` is omitted entirely when the exception has none. `trace` never appears in
the response.

`unhandled_exception_handler` is the fallback for anything that escaped the domain
handlers. It logs the full traceback tagged with the current request id and returns the
same shape with HTTP 500 and `"type": "INTERNAL_SERVER_ERROR"`. Starlette wires plain
`Exception` handlers into `ServerErrorMiddleware`, which sits **outside** the
request-logging middleware, so this handler sets the `X-Request-ID` response header
itself.

### Status codes in use

| Status | When |
|---|---|
| 400 | `StateConflictException` — operation conflicts with the current state |
| 404 | resource not found (model, KB, conversation, message, download job) |
| 409 | `StateConflictException` raised with `status_code=409` — resolvable conflict, carries `detail` |
| 422 | `InvalidInputException` — business validation beyond the Pydantic schema |
| 500 | database, filesystem, model loading, generation, tokenization, configuration, engine, hardware, and any unhandled exception |
| 501 | `UnsupportedPlatformException` |
| 503 | `HuggingFaceAPIException` |
| 507 | `InsufficientMemoryException` |

FastAPI's own request validation still returns its standard 422 body; that path does not
go through these handlers.

## Frontend

All error handling on the renderer side goes through
`frontend/src/services/api/client.js`. It exports two things:

- **`apiClient`**, a singleton with `get`, `post`, `put`, `patch` and `delete`. Every
  call gets a fresh `X-Request-ID` of the form `fe-<base36 timestamp>-<counter>`, a
  30-second timeout via `AbortController`, and up to 3 attempts with exponential backoff
  on transient failures (network `TypeError`, `ECONNREFUSED`, `ENOTFOUND`, `ETIMEDOUT`,
  `EHOSTUNREACH`). Non-OK responses are normalized by `handleErrorResponse` into an
  `Error` carrying `status`, `code` (`HTTP_<status>`) and `data` (the parsed body). An
  abort throws an error with `code = "TIMEOUT"`.
- **`tracedFetch`**, a drop-in `fetch` that adds the same `X-Request-ID` header and the
  same request/response/failure log entries but touches nothing else — no retry, no
  timeout, no JSON parsing — so streaming readers keep working. Use it for the NDJSON
  conversation stream.

Two consequences worth knowing:

- `handleErrorResponse` looks for a top-level `detail` or `message`. The backend nests
  both under `error`, so a caller that needs the `erudi_code` or the structured `detail`
  reads them from `err.data.error.type` and `err.data.error.detail`.
- The backend's own message text is passed through untranslated; only the generic
  fallbacks (`errors:api.timeout`, `errors:api.httpStatus`) go through i18n.

Both the request id on the frontend and the one injected into every backend log line for
that request are the same string, so one click can be followed end to end. See
[Logging & Traceability](../logging.md).

### Writing frontend examples

Any JSX in documentation or code must go through i18n: no hardcoded user-facing string,
every one of them a `t('ns:key')` over `frontend/src/locales/<lang>/*.json`. The rule is
enforced by the `i18next/no-literal-string` ESLint rule and by `locales.test.js`.

```jsx
{error && <p className="text-red-500">{t("errors:model.loadFailed")}</p>}
```

See [Internationalization](../i18n.md).

## Testing

Backend, with pytest:

```python
import pytest
from src.core.exceptions import ModelNotFoundException

def test_missing_model_raises():
    with pytest.raises(ModelNotFoundException) as exc:
        service.get_llm(999)
    assert exc.value.status_code == 404
    assert exc.value.erudi_code == "MODEL_NOT_FOUND"
```

Through the API client:

```python
def test_missing_model_response(client):
    response = client.get("/erudi/llms/999")
    assert response.status_code == 404
    body = response.json()
    assert body["success"] is False
    assert body["error"]["type"] == "MODEL_NOT_FOUND"
```

Frontend tests run on vitest (`cd frontend && npm test`).

## See also

- [Architecture](../architecture.md#error-handling)
- [Core Reference](../reference/core.md)
- [Logging & Traceability](../logging.md)
