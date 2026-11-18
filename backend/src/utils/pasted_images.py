"""The pasted images a conversation owns, and their removal (#428).

A user message stores one ``[image_path:<path>]`` marker per image it carried
(``ConversationService._user_display_content``, the path percent-encoded by
``_encode_marker_path``). The path is either a file the user picked from the
disk, which is theirs, or a clipboard image the Electron main process wrote
into the app's pasted-images folder (``PASTED_IMAGES_FOLDER`` in the data
folder, ``src.launcher.runtime_paths``), which is the app's.

Deleting a conversation removes a file only when its real path (symlinks
resolved) is inside that folder. A picked file is never touched, wherever it
is, and neither is a link placed in the folder that leads out of it.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from src.core.logging import logger

# One image marker; the path inside has its "]" encoded as "%5D".
_IMAGE_MARKER = re.compile(r"\[image_path:([^\]]*)\]")


@dataclass(frozen=True)
class PastedImage:
    """A pasted image one conversation references.

    ``marker`` is the stored marker, as it appears in a message (what another
    conversation referencing the same file would hold too); ``path`` is the
    file's real path, inside the pasted-images folder.
    """

    marker: str
    path: Path


def decode_marker_path(encoded: str) -> str:
    """The path inside a marker, decoded: ``%5D`` first, then ``%25`` (the
    reverse of the encoding order; the frontend's ``decodeMarkerPath`` does the
    same)."""
    return encoded.replace("%5D", "]").replace("%25", "%")


def _real(path: str | Path) -> Path:
    return Path(os.path.normcase(os.path.realpath(path)))


def pasted_images_in(contents: Iterable[str], folder: Path) -> List[PastedImage]:
    """The images of ``contents`` (message texts) whose real path lies inside
    ``folder``, each once, in order of first appearance."""
    root = _real(folder)
    found: dict[Path, PastedImage] = {}
    for content in contents:
        for match in _IMAGE_MARKER.finditer(content or ""):
            encoded = match.group(1)
            if not encoded:
                continue
            real = _real(decode_marker_path(encoded))
            if root in real.parents and real not in found:
                found[real] = PastedImage(marker=match.group(0), path=real)
    return list(found.values())


def remove_pasted_images(images: Iterable[PastedImage], conversation_id: int) -> int:
    """Delete each image file; return how many were deleted.

    A file already gone is not a failure. A file that cannot be deleted stays
    on the disk and is one WARNING: the conversation itself is deleted.
    """
    removed = 0
    for image in images:
        try:
            image.path.unlink()
            removed += 1
        except FileNotFoundError:
            # Already gone (deleted by hand, or Clear All Data): nothing to do.
            continue
        except OSError as exc:
            logger.warning(
                f"Could not delete a pasted image of conversation {conversation_id}; "
                f"the file stays on disk: {exc}"
            )
    return removed
