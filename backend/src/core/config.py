"""Global application configuration and environment management.

This module centralizes configuration constants, environment variables,
and directory paths used across Erudi. It ensures consistent path resolution
in both editable source checkouts and PyInstaller bundles.

Configuration Categories:
    - **Authentication**: HuggingFace token for model downloads.
    - **Directories**: Data storage, model cache, logs, vector indexes.
    - **Database**: embedded PostgreSQL cluster data directory.
    - **Engine**: Runtime LLM engine instance (MLX/CUDA/CPU).

Directory Structure:
    Development mode (backend/)::

        backend/
        ├── data/
        │   ├── postgres/             # Embedded PostgreSQL cluster (pgserver)
        │   ├── models/               # Downloaded models
        │   ├── models_cache/         # HuggingFace download cache
        │   └── training_datasets/    # Fine-tuning datasets
        └── logs/                     # Structured application logs

    Production mode (PyInstaller bundle):
        Runtime directories are resolved dynamically by
        ``src.launcher.runtime_paths`` so packaged builds write to
        user-writable locations (AppData on Windows, Library folders on macOS,
        XDG paths on Linux).

Environment Variables:
    HF_TOKEN (optional):
        HuggingFace API token for downloading gated models.
        If not set, only public models are accessible.

Path Resolution:
    ROOT_DIR:
        Populated from the runtime path registry. In development it points to
        the ``backend/`` folder; in bundled builds it matches the PyInstaller
        resource directory.

Global State:
    LLM_Engine:
        Runtime engine instance initialized during FastAPI lifespan startup.
        Set by BaseEngine.get_engine() based on platform detection.
        Type is Optional[Type[BaseEngine]] before initialization.

Example:
    Access configuration in application code::

        from src.core import config

        # Use paths
        model_path = config.LLM_DIR / "Llama-3-8B"

        # Check HuggingFace authentication
        if config.HF_TOKEN:
            api = config.get_hf_api()
            api.whoami()  # Verify token validity

        # Use engine (after lifespan startup): resolve the model server,
        # then stream from it via the agent layer (ChatOpenAI(base_url=...)).
        model, tokenizer = config.LLM_Engine.get_model_and_tokenizer(llm_id, path)

Note:
    All directories are created automatically at module import time.
    Do not manually modify LLM_Engine; it is managed by the lifespan context.

Warning:
    Changing ROOT_DIR after import will not update derived paths (LLM_DIR,
    CACHE_DIR, etc.). Ensure paths are configured before first import.
"""

from dotenv import load_dotenv
import os
import time
from typing import Optional, Type

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

from src.core import hf_http
from src.engines.base_engine import BaseEngine
from src.launcher import ensure_runtime_paths_initialized

# ============ Environment Variables ============

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", None)

# Agentic KB mode exposes the knowledge base as a ``search_knowledge_base`` tool
# the model decides to call. Whether the local server turns the model's
# tool-call output into a structured call is a PER-MODEL wire property (#273
# matrix, #298): some templates parse and execute, others leak raw JSON or get
# swallowed entirely (#295). ERUDI_KB_AGENTIC is therefore tri-state:
#   unset      -> None:  per-model routing (the default). A KB turn goes
#                        agentic only for models whose wire capability was
#                        VERIFIED (llms.supports_tools_wire is True).
#   1 / true   -> True:  force agentic for every KB turn (debug).
#   0 / false  -> False: force systematic for every KB turn (kill switch,
#                        the pre-#298 behavior of #288).


def parse_kb_agentic_flag(raw: Optional[str]) -> Optional[bool]:
    """Parse the tri-state ERUDI_KB_AGENTIC value (#298).

    Unset/empty/unrecognized -> None (per-model routing); "1"/"true" -> True
    (force agentic); "0"/"false" -> False (force systematic). Case-insensitive,
    surrounding whitespace ignored.
    """
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in ("1", "true"):
        return True
    if value in ("0", "false"):
        return False
    return None


KB_AGENTIC_MODE: Optional[bool] = parse_kb_agentic_flag(os.getenv("ERUDI_KB_AGENTIC"))

# Restrict LangGraph checkpoint (msgpack) deserialization to known-safe types.
# Set before any langgraph serializer is constructed.
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

# ============ Runtime Paths ============

_RUNTIME_PATHS = ensure_runtime_paths_initialized()
ROOT_DIR = _RUNTIME_PATHS.backend_root
DATA_ROOT = _RUNTIME_PATHS.data_dir
LOG_DIR = _RUNTIME_PATHS.log_dir

# ============ Directory Paths ============

LLM_DIR = DATA_ROOT / "models"
CACHE_DIR = DATA_ROOT / "models_cache"

# ============ Database Configuration ============

DATA_ROOT.mkdir(parents=True, exist_ok=True)
# Embedded PostgreSQL cluster data directory (pgserver). The SQLAlchemy
# engine is initialized explicitly once the cluster is up — see core/api.py
# lifespan (step 0: start_postgres, step 1: database.core.init_database).
POSTGRES_DATA_DIR = DATA_ROOT / "postgres"

# ============ Directory Creation ============

LLM_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ============ HuggingFace API Client ============

# Every huggingface_hub request goes through Erudi's client (#747): a shared
# client another thread's retry closed hands the request on, and a server's
# wait is capped (src/core/hf_http.py).
hf_http.install_client_factory()

# Lazy-loaded HuggingFace API client singleton
_HF_API: Optional[HfApi] = None


def _retry_after_seconds(err: HfHubHTTPError) -> Optional[float]:
    """Read the Retry-After header off a 429, if HF sent one."""
    try:
        value = err.response.headers.get("Retry-After")
        return float(value) if value else None
    except Exception:
        return None


def _call_with_429_retry(fn, *args, _max_retries: int = 5, _max_backoff: float = 30.0, **kwargs):
    """Call ``fn`` and retry on HTTP 429 with exponential backoff (honoring
    Retry-After). The catalog resync fires hundreds of anonymous HF metadata
    calls in a burst; without this a 429 either aborts the whole resync or is
    silently swallowed as "no result" (dropping models). Runs inside the resync's
    threadpool, so the blocking sleep stays off the event loop.

    ``_max_retries``/``_max_backoff`` bound the total wall-clock budget. The bulk
    resync wants a long ladder (survive rate-limiting), but an interactive,
    user-facing search must fail fast: a caller waiting on the request has a
    client-side timeout, and a ladder that outlives it (#210) guarantees failure.
    Both the exponential delay and any ``Retry-After`` are capped at ``_max_backoff``."""
    for attempt in range(_max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except HfHubHTTPError as err:
            status = getattr(getattr(err, "response", None), "status_code", None)
            if status == 429 and attempt < _max_retries:
                delay = _retry_after_seconds(err)
                if delay is None:
                    delay = 2**attempt
                delay = min(delay, _max_backoff)
                from src.core.logging import logger

                logger.warning(
                    f"HF rate-limited (429); backing off {delay:.0f}s "
                    f"(attempt {attempt + 1}/{_max_retries})"
                )
                time.sleep(delay)
                continue
            raise


class _RetryingHfApi(HfApi):
    """HfApi that survives anonymous rate limiting during the catalog resync:
    paces metadata calls and retries them on 429. ``list_models`` is materialized
    inside the retry because it paginates lazily — a 429 otherwise surfaces during
    iteration (uncaught), not at call time. Callers therefore MUST pass a bounded
    ``limit`` (they do). Downloads (hf_hub_download / AutoTokenizer) already retry
    on their own path."""

    _PACE_SECONDS = 0.1

    def list_models(self, *args, _max_retries: int = 5, _max_backoff: float = 30.0, **kwargs):
        time.sleep(self._PACE_SECONDS)
        _super_list_models = super().list_models
        return _call_with_429_retry(
            lambda: list(_super_list_models(*args, **kwargs)),
            _max_retries=_max_retries,
            _max_backoff=_max_backoff,
        )

    def model_info(self, *args, **kwargs):
        time.sleep(self._PACE_SECONDS)
        return _call_with_429_retry(super().model_info, *args, **kwargs)

    def hf_hub_download(self, *args, **kwargs):
        # Small per-repo file fetches (generation_config.json & co for the
        # sampling hints, #388) ride the same pacing + 429 ladder as the
        # metadata calls: the snapshot generator fires ~300 of them in a burst.
        time.sleep(self._PACE_SECONDS)
        return _call_with_429_retry(super().hf_hub_download, *args, **kwargs)


def get_hf_api() -> HfApi:
    """Get or initialize HuggingFace API client (lazy-loaded singleton).

    Provides thread-safe lazy initialization of HfApi client. The client is
    created only when first requested, avoiding unnecessary initialization
    during imports or when working offline.

    Returns:
        HfApi: Configured HuggingFace API client with authentication token.

    Note:
        Thread-safe through Python's GIL. Multiple calls return same instance.
        Token is loaded from HF_TOKEN environment variable.

    Example:
        >>> from src.core import config
        >>> api = config.get_hf_api()
        >>> models = api.list_models(search="gemma")
    """
    global _HF_API
    if _HF_API is None:
        _HF_API = _RetryingHfApi(token=HF_TOKEN)
    return _HF_API


# ============ LLM Engine (Runtime State) ============

# Runtime engine instance initialized during FastAPI lifespan startup
# Set by BaseEngine.get_engine() based on platform detection
LLM_Engine: Optional[Type[BaseEngine]] = None
