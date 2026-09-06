"""On-demand availability + download of the KB embedding model (#146).

The Knowledge Base needs ``intfloat/multilingual-e5-small`` for BOTH embeddings
(``E5Embeddings``) and token-accurate chunking (``chunking._get_tokenizer``).
Both load it from ``config.CACHE_DIR`` (``data/models_cache``), so a single
download serves both consumers.

This module gates that download behind an explicit user action instead of the
implicit lazy Hugging Face fallback, so a fresh/offline install no longer fails
its first Knowledge Base use silently.

Design invariants:
- **Presence is filesystem-driven, never a DB flag.** ``embedding_model_available``
  checks EVERY file a ``SentenceTransformer`` load needs (#164) — huggingface_hub
  only links a file into ``snapshots/`` once it is fully downloaded, so a
  partial/interrupted download reads as unavailable and the gate re-appears.
  This also makes the state self-heal across backend restarts (in-memory flags
  are lost, the file check is not).
- **A complete cache means OFFLINE loads** (#164). Without ``local_files_only=True``
  huggingface_hub HEAD-revalidates every file against the hub on each load —
  offline, the failed DNS probe surfaces as a bogus load error despite a fully
  pre-downloaded model, defeating the whole point of the gate. Consumers
  (``E5Embeddings``, ``chunking._get_tokenizer``) apply the same rule.
- **Download == the real load path** (``SentenceTransformer(..., cache_folder=CACHE_DIR)``),
  not a parallel ``snapshot_download`` — so the download target is ISO with where
  the app loads from, by construction, and it warms the resident singleton.
- **The download runs in a background thread**, decoupled from the HTTP request,
  so leaving the KB page never loses it; the UI polls a boolean status.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from huggingface_hub import try_to_load_from_cache

from src.core import config
from src.core.logging import logger

EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-small"
# Everything a SentenceTransformer load of multilingual-e5-small touches, as
# present in its hub snapshot: the ST architecture (modules.json), the
# transformer config + weights, the pooling module, and the tokenizer files.
# Checking only the weights false-positived on a partial download (#164).
# NB: config_sentence_transformers.json is NOT required — the upstream repo
# does not ship it (it lives in the cache's .no_exist/ markers).
REQUIRED_FILES = (
    "modules.json",
    "config.json",
    "model.safetensors",
    "sentence_bert_config.json",
    "1_Pooling/config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
)

OFFLINE_ERROR_MESSAGE = "no internet connection — the embedding model could not be downloaded"

# Lowercased substrings that mark a network-down failure anywhere in the
# exception chain. transformers/hub re-wrap connection errors liberally, so
# matching by message (defensively, alongside types) is deliberate.
_OFFLINE_MESSAGE_MARKERS = (
    "getaddrinfo",
    "nodename nor servname",
    "name or service not known",
    "temporary failure in name resolution",
    "network is unreachable",
    "offline mode is enabled",
    "outgoing traffic has been disabled",
    "connection error",
    "couldn't connect to 'https://huggingface.co'",
)

_lock = threading.Lock()
_state = {"downloading": False, "error": None}


def embedding_model_available() -> bool:
    """True iff EVERY required file is fully present in ``CACHE_DIR`` (no network)."""
    for filename in REQUIRED_FILES:
        cached = try_to_load_from_cache(
            EMBEDDING_MODEL_ID, filename, cache_dir=str(config.CACHE_DIR)
        )
        if not isinstance(cached, str):
            return False
    return True


def download_state() -> dict:
    """The gate's view: on-disk presence + transient background-task flags."""
    return {
        "available": embedding_model_available(),
        "downloading": _state["downloading"],
        "error": _state["error"],
    }


# Cached verdict for "MPS says available but cannot actually allocate" (#335).
# Sticky for the process: once an allocation has failed, retrying it on every
# load would pay the failure again on a hot path (the resident singleton is
# rebuilt whenever it is dropped), and nothing about the machine has changed.
_MPS_UNUSABLE = False


def _mps_is_available() -> bool:
    """True when torch reports a usable-looking MPS backend (never raises)."""
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001 - no torch, no backends.mps, probe error
        return False


def resolve_embedding_device() -> Optional[str]:
    """Device for the embedding model. ``None`` = let sentence-transformers pick.

    Apple Silicon is the primary desktop platform, so MPS stays preferred
    whenever Metal genuinely works. ``is_available()`` is only a support flag
    though: a virtualised macOS host (the ``macos-14`` CI runner) answers True
    and then refuses the model's ~366 MiB allocation, and a real Mac under
    memory pressure fails the same way (#335). ``load_sentence_transformer``
    turns that failure into this cached CPU verdict.

    Off Apple Silicon the choice is left untouched (``None``), so a CUDA host
    keeps auto-selecting its GPU.
    """
    if _MPS_UNUSABLE:
        return "cpu"
    if _mps_is_available():
        return "mps"
    return None


def _is_mps_allocation_failure(exc: BaseException) -> bool:
    """True when the load died inside the Metal backend rather than elsewhere."""
    return "mps" in str(exc).lower()


def load_sentence_transformer(model_name: str, *, cache_folder: str, local_files_only: bool):
    """Load a SentenceTransformer on the best device the machine can actually use.

    Single entry point for every embedding-model load (the #146 download gate
    and the resident ``E5Embeddings`` singleton), so the MPS fallback is decided
    once and applies to both.
    """
    global _MPS_UNUSABLE
    from sentence_transformers import SentenceTransformer

    device = resolve_embedding_device()
    try:
        return SentenceTransformer(
            model_name,
            cache_folder=cache_folder,
            local_files_only=local_files_only,
            device=device,
        )
    except RuntimeError as exc:
        if device != "mps" or not _is_mps_allocation_failure(exc):
            raise
        _MPS_UNUSABLE = True
        logger.warning(
            f"MPS is available but the embedding model could not be allocated on it; "
            f"falling back to CPU: {exc}"
        )

    return SentenceTransformer(
        model_name,
        cache_folder=cache_folder,
        local_files_only=local_files_only,
        device="cpu",
    )


def _load_model(local_files_only: Optional[bool] = None):
    """The real load path — downloads into ``CACHE_DIR`` if absent (ISO with runtime).

    ``local_files_only`` defaults to the availability check: once the cache is
    complete the load NEVER touches the network (#164). The download path
    passes ``False`` explicitly — that IS the fetch.
    """
    if local_files_only is None:
        local_files_only = embedding_model_available()
    return load_sentence_transformer(
        EMBEDDING_MODEL_ID,
        cache_folder=str(config.CACHE_DIR),
        local_files_only=local_files_only,
    )


def _is_offline_error(exc: BaseException) -> bool:
    """True if the exception chain looks like a no-network failure (#164)."""
    from huggingface_hub.errors import OfflineModeIsEnabled
    from requests.exceptions import ConnectionError as RequestsConnectionError

    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (OfflineModeIsEnabled, RequestsConnectionError)):
            return True
        message = str(current).lower()
        if any(marker in message for marker in _OFFLINE_MESSAGE_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


def _describe_download_error(exc: BaseException) -> str:
    """Map raw transformers/hub failures to a message the user can act on."""
    if _is_offline_error(exc):
        return OFFLINE_ERROR_MESSAGE
    return str(exc)


def _spawn(target) -> None:
    threading.Thread(target=target, name="e5-download", daemon=True).start()


def _run_download() -> None:
    start_s = time.perf_counter()
    try:
        logger.info(f"Downloading embedding model {EMBEDDING_MODEL_ID} -> {config.CACHE_DIR}")
        _load_model(local_files_only=False)
        _state["error"] = None
        logger.info(f"Embedding model download complete ({time.perf_counter() - start_s:.1f}s)")
    except Exception as exc:  # noqa: BLE001 - any failure is surfaced to the UI
        # Runs on a thread nobody joins: this record is the only trace.
        logger.error(f"Embedding model download failed: {exc}", exc_info=True)
        _state["error"] = _describe_download_error(exc)
    finally:
        _state["downloading"] = False


def start_download() -> dict:
    """Idempotently kick off a background download. Returns the current state.

    No-op (returns current state) if the model is already present or a download
    is already running — guards double-clicks and concurrent callers.
    """
    with _lock:
        if not _state["downloading"] and not embedding_model_available():
            _state["downloading"] = True
            _state["error"] = None
            _spawn(_run_download)
    return download_state()


def _reset_state_for_tests(downloading: bool = False, error: Optional[str] = None) -> None:
    _state["downloading"] = downloading
    _state["error"] = error
