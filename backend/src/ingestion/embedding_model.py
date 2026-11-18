"""On-demand availability + download of the e5 snapshot the macOS KB
embedding child serves (#146, #594).

On macOS the Knowledge Base embeddings come from a dedicated `mlx_vlm.server`
child (``mlx_embedder``) serving ``intfloat/multilingual-e5-small`` BY PATH
from ``config.CACHE_DIR`` (``data/models_cache``). Token-accurate chunking
does not wait for it: its tokenizer ships frozen in the app
(``chunking.E5_TOKENIZER_FILE``). On Windows and Linux the embedding child
serves a GGUF that ships with the installers instead, and nothing here is
used to produce it.

This module gates that download behind an explicit user action instead of an
implicit lazy fetch, so a fresh/offline install never fails its first
Knowledge Base use silently.

Design invariants:
- **Presence is filesystem-driven, never a DB flag.** ``embedding_model_available``
  checks EVERY file the child's load reads (#164) — huggingface_hub only links
  a file into ``snapshots/`` once it is fully downloaded, so a
  partial/interrupted download reads as unavailable and the gate re-appears.
  This also makes the state self-heal across backend restarts (in-memory flags
  are lost, the file check is not).
- **The child never touches the network.** It loads the local snapshot
  directory (``embedding_snapshot_path``), so an offline machine with a
  complete cache embeds as well as an online one.
- **The download fetches exactly the required files** into the hub cache
  layout under ``CACHE_DIR`` (``snapshot_download`` with ``allow_patterns``),
  so what the gate checks, what it fetches and what the child loads are the
  same list by construction.
- **The download runs in a background thread**, decoupled from the HTTP request,
  so leaving the KB page never loses it; the UI polls a boolean status.
"""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path
from typing import Optional

from huggingface_hub import try_to_load_from_cache

from src.core import config
from src.core.logging import logger

EMBEDDING_MODEL_ID = "intfloat/multilingual-e5-small"
# The files of the multilingual-e5-small snapshot the embedding child reads:
# the model config and weights, the pooling module (mlx_vlm's
# ``read_pooling_config`` takes the mean pooling from ``1_Pooling/config.json``)
# and the tokenizer files. Checking only the weights false-positived on a
# partial download (#164).
REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "1_Pooling/config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
)

OFFLINE_ERROR_MESSAGE = "no internet connection — the embedding model could not be downloaded"

# Lowercased substrings that mark a network-down failure anywhere in the
# exception chain. huggingface_hub re-wraps connection errors liberally, so
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


def embedding_snapshot_path() -> Optional[Path]:
    """The LOCAL snapshot directory of the embedding model, or ``None``.

    The dedicated embedding child (#594) serves the snapshot BY PATH, never by
    repo id: a path load never touches the network and cannot resolve to
    another revision than the one the gate downloaded. ``None`` whenever the
    gate says the cache is incomplete — the caller surfaces the
    missing-artifact state instead of spawning a child that cannot load.
    """
    if not embedding_model_available():
        return None
    cached = try_to_load_from_cache(
        EMBEDDING_MODEL_ID, "config.json", cache_dir=str(config.CACHE_DIR)
    )
    if not isinstance(cached, str):
        return None
    return Path(cached).parent


def download_state() -> dict:
    """The gate's view: on-disk presence + transient background-task flags."""
    return {
        "available": embedding_model_available(),
        "downloading": _state["downloading"],
        "error": _state["error"],
    }


def remove_snapshot() -> bool:
    """Delete the e5 snapshot from ``CACHE_DIR`` (its hub repo folder and its
    download locks). Returns whether there was one to delete.

    Never raises: a failed delete is one WARNING and the snapshot stays for
    the next start to retry. The caller decides when the snapshot is
    obsolete (``embedder_child.remove_obsolete_e5_snapshot``).
    """
    folder = "models--" + EMBEDDING_MODEL_ID.replace("/", "--")
    repo = Path(config.CACHE_DIR) / folder
    if not repo.exists():
        return False
    try:
        freed = sum(f.stat().st_size for f in repo.rglob("*") if f.is_file())
        shutil.rmtree(repo)
        shutil.rmtree(Path(config.CACHE_DIR) / ".locks" / folder, ignore_errors=True)
    except OSError as exc:
        logger.warning(
            f"Could not delete the obsolete embedding model snapshot {repo} "
            f"(the next start retries): {type(exc).__name__}: {exc}"
        )
        return False
    logger.info(
        f"Deleted the obsolete embedding model snapshot {repo} "
        f"({freed / 1e6:.0f} MB): the knowledge base uses the bundled model file"
    )
    return True


def _fetch() -> None:
    """Download the required snapshot files into ``CACHE_DIR`` (the network
    call; resumes a partial download, skips files already complete)."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        EMBEDDING_MODEL_ID,
        cache_dir=str(config.CACHE_DIR),
        allow_patterns=list(REQUIRED_FILES),
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
    """Map raw hub failures to a message the user can act on."""
    if _is_offline_error(exc):
        return OFFLINE_ERROR_MESSAGE
    return str(exc)


def _spawn(target) -> None:
    threading.Thread(target=target, name="e5-download", daemon=True).start()


def _run_download() -> None:
    start_s = time.perf_counter()
    try:
        logger.info(f"Downloading embedding model {EMBEDDING_MODEL_ID} -> {config.CACHE_DIR}")
        _fetch()
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
