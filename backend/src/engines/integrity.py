"""Post-download / pre-load artifact integrity checks (#88).

A model must never become selectable (``local=1``) or reach the spawn/probe
path without its ESSENTIAL files on disk. These validators check exactly what
each engine family needs to run and raise :class:`EngineException` with an
explicit, ASCII-only, user-facing message on the FIRST missing or corrupt file:

- **GGUF** (llama.cpp CPU/CUDA): one non-``mmproj`` ``.gguf`` that is non-empty
  and starts with the GGUF magic (``b"GGUF"``). ``llama-server`` reads the
  tokenizer + chat template out of the GGUF container itself, so that single
  weights file is the whole essential set (the ``mmproj`` projector and the tiny
  aux JSONs the downloader also fetches are optional).
- **HF / MLX snapshot** (``mlx_vlm`` / ``transformers``): ``config.json`` + a
  tokenizer file + at least one weights file, each present and non-empty. That
  is the minimum a snapshot load needs before it can even build the model.

The wording is deliberately model-blaming, not network-blaming: a file that is
missing AFTER a completed transfer almost always means the source repo is odd,
not that the connection dropped. The same validators back both the
download-completion gate (``services``/``endpoints``) and the pre-spawn load
gate (``get_model_and_tokenizer``), so the two paths can never disagree on what
"complete" means.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from src.core.exceptions import EngineException

# First four bytes of every GGUF container (spec magic, "GGUF").
GGUF_MAGIC = b"GGUF"

# Any ONE of these filenames satisfies the tokenizer requirement for a snapshot:
# a fast tokenizer (tokenizer.json), a sentencepiece model (tokenizer.model /
# spiece.model), or the config that pairs with a vocab file.
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "spiece.model",
)

# Weight containers a snapshot load can consume. mmproj/gguf are intentionally
# excluded here: an MLX/HF snapshot ships safetensors (or legacy bin/npz).
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".npz")

_INCOMPLETE = "The downloaded files for this model are incomplete or invalid"
_RETRY = (
    "This is usually a problem with the model itself, not your connection. "
    "You can retry the download or choose another model."
)


def incomplete_message(detail: str) -> str:
    """Build the shared, user-facing incomplete-artifact message.

    ``detail`` names the specific missing/corrupt piece (e.g. ``"missing
    tokenizer"``); the surrounding sentence tells the user it is the model's
    fault, not the network, and offers the two real remedies.
    """
    return f"{_INCOMPLETE} ({detail}). {_RETRY}"


def _nonempty_file(path: Path) -> bool:
    """True if ``path`` is a regular file with size > 0 (never raises)."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def validate_gguf_file(gguf_path: Union[str, Path]) -> None:
    """Validate a single GGUF file: exists, non-empty, GGUF magic bytes.

    Raises :class:`EngineException` (with a curated user-facing message) on the
    first problem; returns None when the file is a plausible GGUF container.
    """
    # The message is curated for the user; the path goes in the trace so the
    # log record says which file failed the check.
    path = Path(gguf_path)
    if not path.exists():
        raise EngineException(
            message=incomplete_message("the GGUF weights file is missing"), trace=str(path)
        )
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise EngineException(
            message=incomplete_message("the GGUF weights file is unreadable"),
            trace=f"{path}: {exc}",
        )
    if size == 0:
        raise EngineException(
            message=incomplete_message("the GGUF weights file is empty"), trace=str(path)
        )
    try:
        with open(path, "rb") as handle:
            head = handle.read(len(GGUF_MAGIC))
    except OSError as exc:
        raise EngineException(
            message=incomplete_message("the GGUF weights file is unreadable"),
            trace=f"{path}: {exc}",
        )
    if head != GGUF_MAGIC:
        raise EngineException(
            message=incomplete_message("the GGUF weights file is corrupted"), trace=str(path)
        )


def validate_hf_snapshot(model_dir: Union[str, Path]) -> None:
    """Validate an HF/MLX snapshot directory has its essential, non-empty files.

    Essential set (minimum a ``transformers``/``mlx_vlm`` load needs):
    ``config.json`` + one tokenizer file + at least one weights file. Raises
    :class:`EngineException` on the first missing/empty essential; returns None
    when all three are present.
    """
    path = Path(model_dir)
    if not path.exists() or not path.is_dir():
        raise EngineException(
            message=incomplete_message("the model folder is missing"), trace=str(path)
        )
    if not _nonempty_file(path / "config.json"):
        raise EngineException(message=incomplete_message("missing config.json"), trace=str(path))
    if not any(_nonempty_file(path / name) for name in _TOKENIZER_FILES):
        raise EngineException(message=incomplete_message("missing tokenizer"), trace=str(path))
    has_weights = any(
        _nonempty_file(child)
        for child in path.iterdir()
        if child.suffix.lower() in _WEIGHT_SUFFIXES
    )
    if not has_weights:
        raise EngineException(
            message=incomplete_message("no model weights were found"), trace=str(path)
        )
