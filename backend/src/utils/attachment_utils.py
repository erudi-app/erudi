"""Question attachments: local files and folders turned into model-visible text.

A chat or arena question can carry local filesystem paths (#492). Erudi runs on
the same machine as the files, so nothing is uploaded: the renderer sends the
paths and this module reads them.

Contract:
  * every file goes through the Knowledge Base's ``DocumentReader`` facade -
    there is exactly one parser in the app, and this is not a second one;
  * a directory is expanded into the supported files it holds;
  * the extracted text becomes one delimited block per file, injected BEFORE
    the user's own words in the turn;
  * the total is capped per question, and an overflowing block is truncated
    with an explicit marker rather than silently blowing the context;
  * a file that cannot be read never fails the turn: it is reported back to the
    caller and its block says, in the model's own view, that it was unreadable.

Layering: services call ``resolve_attachments``; the reader stays the only
parser. The work is blocking (pypdf, python-docx, openpyxl), so callers on the
event loop must run it through ``run_in_threadpool``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from src.core.exceptions import InvalidInputException
from src.core.logging import logger
from src.ingestion.reader import IMAGE_EXTENSIONS, DocumentReader

# One question's attachment text is capped so a single large document cannot
# crowd the model's context out of the answer. The cap is expressed in
# CHARACTERS on purpose: the only tokenizer resident in the backend is the e5
# retrieval tokenizer, which is not the chat model's, so counting with it would
# buy a precision the number does not have while adding work to the request
# path. At the ~4 characters/token rule of thumb that holds for the Latin-script
# Markdown these extractors produce, 24 000 characters is roughly 6 000 tokens -
# about 60 % of the 8 192-token window the smallest bundled models expose, and a
# small fraction of the larger ones.
ATTACHMENT_CHAR_BUDGET = 24_000

# Upper bound on the files one question may carry once folders are expanded.
# It bounds the wall-clock cost of a dropped folder; the character budget above
# already bounds what reaches the model.
MAX_ATTACHMENT_FILES = 50

# How deep a dropped folder is walked: the folder itself plus one level below
# it. Deeper trees are a Knowledge Base, which is the tool for that job.
ATTACHMENT_DIR_MAX_DEPTH = 2

TRUNCATION_MARKER = "[truncated: file too large to include fully]"

_BLOCK_HEADER = "[Attached file: {name}]"
_BLOCK_FOOTER = "[End of attached file]"

# What the model sees in place of a file that could not be read. Kept explicit
# so the model does not invent content for a document it never received.
_UNREADABLE_BODY = "[This file could not be read: {reason}]"

_REASON_TEXT = {
    "missing": "the file no longer exists at that location",
    "unsupported": "this file type is not supported",
    "pending_vision": (
        "it holds no extractable text (a scan or an image), and text recognition "
        "is not bundled yet"
    ),
    "empty": "it holds no text",
    "unreadable": "the extractor failed on it",
}


@dataclass(frozen=True)
class AttachmentIssue:
    """One file that could not be turned into text."""

    name: str
    path: str
    # One of: missing, unsupported, pending_vision, empty, unreadable,
    # too_many, empty_folder.
    reason: str
    message: str


@dataclass(frozen=True)
class ResolvedAttachments:
    """What the service needs after resolution.

    Attributes:
        block: The delimited attachment text, empty when nothing was attached.
        paths: Absolute path of every file the question carries, in order,
            including the ones that failed - they are what the conversation
            records as attached.
        issues: One entry per file that could not be read, for the UI.
        truncated_names: Files whose text was cut to fit the budget.
    """

    block: str = ""
    paths: list[str] = field(default_factory=list)
    issues: list[AttachmentIssue] = field(default_factory=list)
    truncated_names: list[str] = field(default_factory=list)


def _supported_extensions() -> set[str]:
    """Extensions worth expanding a folder for: everything the reader parses,
    minus images (they have their own vision path and hold no text)."""
    return DocumentReader().supported_extensions - IMAGE_EXTENSIONS


def _walk_directory(directory: Path) -> list[Path]:
    """Supported files in ``directory`` and in its immediate subdirectories.

    Sorted so the same folder always yields the same order, and hidden entries
    (dotfiles, dot-directories) are skipped: a dropped folder is a document
    folder, not a checkout.
    """
    supported = _supported_extensions()
    found: list[Path] = []

    def scan(current: Path, depth: int) -> None:
        try:
            entries = sorted(current.iterdir(), key=lambda p: p.name)
        except OSError as exc:
            logger.warning(f"Attachment folder {current.name} could not be listed: {exc}")
            return
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                if depth + 1 < ATTACHMENT_DIR_MAX_DEPTH:
                    scan(entry, depth + 1)
            elif entry.suffix.lower() in supported:
                found.append(entry)

    scan(directory, 0)
    return found


def _expand(paths: Sequence[str]) -> tuple[list[Path], list[AttachmentIssue]]:
    """Turn the attached paths into a de-duplicated, capped list of files."""
    files: list[Path] = []
    seen: set[str] = set()
    issues: list[AttachmentIssue] = []
    overflow = False

    def add(candidate: Path) -> None:
        nonlocal overflow
        key = str(candidate)
        if key in seen:
            return
        if len(files) >= MAX_ATTACHMENT_FILES:
            overflow = True
            return
        seen.add(key)
        files.append(candidate)

    for raw in paths:
        if not raw or not raw.strip():
            continue
        path = Path(raw.strip())
        if path.is_dir():
            walked = _walk_directory(path)
            if not walked:
                issues.append(
                    AttachmentIssue(
                        name=path.name,
                        path=str(path),
                        reason="empty_folder",
                        message="no supported document was found in this folder",
                    )
                )
                continue
            for entry in walked:
                add(entry)
        else:
            add(path)

    if overflow:
        issues.append(
            AttachmentIssue(
                name="",
                path="",
                reason="too_many",
                message=f"only the first {MAX_ATTACHMENT_FILES} files were attached",
            )
        )
    return files, issues


def _read_one(reader: DocumentReader, path: Path) -> tuple[str, Optional[str]]:
    """Extract one file. Returns ``(text, reason)``; ``reason`` is None on
    success and a key of ``_REASON_TEXT`` otherwise. Never raises."""
    if not path.exists() or not path.is_file():
        return "", "missing"
    try:
        document = reader.read(path)
    except InvalidInputException:
        # Expected outcome of dropping, say, a .zip: the app degraded on its
        # own and tells the user, so a WARNING with no traceback is the record.
        logger.warning(f"Attachment {path.name}: unsupported document type, skipped")
        return "", "unsupported"
    except Exception:
        logger.exception(f"Attachment {path.name}: extraction failed")
        return "", "unreadable"

    if document.status == "pending_vision":
        logger.info(f"Attachment {path.name}: no text layer (pending_vision), skipped")
        return "", "pending_vision"
    text = document.markdown.strip()
    if not text:
        logger.warning(f"Attachment {path.name}: extraction returned no text")
        return "", "empty"
    return text, None


def resolve_attachments(paths: Optional[Iterable[str]]) -> ResolvedAttachments:
    """Read every attached path into one delimited, budgeted text block.

    Blocking (real file I/O and parsing): call it through
    ``run_in_threadpool`` from async code.

    Args:
        paths: Local filesystem paths of the attached files and folders.

    Returns:
        A ``ResolvedAttachments``; never raises on a per-file problem.
    """
    if not paths:
        return ResolvedAttachments()

    files, issues = _expand(list(paths))
    if not files:
        return ResolvedAttachments(issues=issues)

    reader = DocumentReader()
    blocks: list[str] = []
    truncated: list[str] = []
    remaining = ATTACHMENT_CHAR_BUDGET

    for path in files:
        text, reason = _read_one(reader, path)
        if reason is not None:
            issues.append(
                AttachmentIssue(
                    name=path.name,
                    path=str(path),
                    reason=reason,
                    message=_REASON_TEXT[reason],
                )
            )
            body = _UNREADABLE_BODY.format(reason=_REASON_TEXT[reason])
        else:
            if len(text) > remaining:
                text = text[: max(remaining, 0)]
                truncated.append(path.name)
                body = f"{text}\n{TRUNCATION_MARKER}" if text else TRUNCATION_MARKER
                remaining = 0
            else:
                remaining -= len(text)
                body = text
        blocks.append(f"{_BLOCK_HEADER.format(name=path.name)}\n{body}\n{_BLOCK_FOOTER}")

    logger.info(
        f"Attachments resolved: files={len(files)}, chars={ATTACHMENT_CHAR_BUDGET - remaining}, "
        f"issues={len(issues)}, truncated={len(truncated)}"
    )
    return ResolvedAttachments(
        block="\n\n".join(blocks),
        paths=[str(p) for p in files],
        issues=issues,
        truncated_names=truncated,
    )


def build_attachment_notice(resolved: ResolvedAttachments) -> str:
    """Markdown notice prepended to the answer when an attachment was not used
    in full. Empty string when every attached file was read whole.

    Mirrors ``IMAGES_IGNORED_NOTICE``: the user is told what the model did not
    get instead of the attachment being dropped in silence.
    """
    lines: list[str] = []
    if resolved.issues:
        named = [
            f"{issue.name} ({issue.message})" if issue.name else issue.message
            for issue in resolved.issues
        ]
        lines.append(f"*Some attachments could not be read: {', '.join(named)}.*")
    if resolved.truncated_names:
        lines.append(
            "*Attached content was shortened to fit this model's context: "
            f"{', '.join(resolved.truncated_names)}.*"
        )
    if not lines:
        return ""
    return "\n".join(lines) + "\n\n"


def prepend_attachment_block(block: str, question: str) -> str:
    """The text the model sees: attachment blocks first, the question last, so
    the question is the most recent thing in the turn."""
    if not block:
        return question
    return f"{block}\n\n{question}" if question else block
