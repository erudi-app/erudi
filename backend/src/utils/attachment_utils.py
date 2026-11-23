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
  * the total is capped per question (``ATTACHMENT_TOKEN_BUDGET``, in the
    compaction counter's tokens), and an overflowing file keeps its longest
    head within what is left, with an explicit marker, rather than silently
    blowing the context;
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

from src.agents.token_accounting import counter_head, counter_text_tokens
from src.core.exceptions import InvalidInputException, UnreadableDocumentException
from src.core.logging import logger
from src.ingestion.reader import IMAGE_EXTENSIONS, DocumentReader

# One question's attachment text is capped so a single large document cannot
# crowd the model's context out of the answer. The cap is in TOKENS as the
# compaction counter counts them (``token_accounting.counter_text_tokens``:
# chars/4 times the text's own script weight -- an ASCII digit one token, a
# CJK character one token -- and never under 1.2 times chars/4), because
# characters are not a size: 24 000 characters of short French lines dense in
# numbers are about 13 000 tokens on a tokenizer that splits digits one by
# one, where 24 000 characters of prose are about 6 000. The counter needs no
# tokenizer (the only one the backend loads is the e5 retrieval tokenizer, not
# the chat model's), and it is an estimate, not a bound: on those digit-dense
# lines it still reads about 30 % under Qwen's count (6 000 counted, about
# 7 900 real), on English prose about 50 % over.
#
# The cap is FIXED, not derived from the loaded model's window: the shortened
# text stays in the conversation's state, which every later turn reads back,
# so the same file must give the same text whatever model reads it. 6 000
# tokens is about 60 % of the 8 192-token window the smallest bundled models
# expose: about 20 000 characters of Latin prose, about 14 000 of digit-dense
# lines. On a small window the shortened message can still not fit; the turn
# is then refused as "this message alone" (``runner``), never sent truncated
# further.
ATTACHMENT_TOKEN_BUDGET = 6_000

# Upper bound on the files one question may carry once folders are expanded.
# It bounds the wall-clock cost of a dropped folder; the token budget above
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
    "unreadable": "it is damaged, protected by a password, or not the format its extension says",
    "failed": "the app could not extract its text",
}


@dataclass(frozen=True)
class AttachmentIssue:
    """One file that could not be turned into text."""

    name: str
    path: str
    # One of: missing, unsupported, pending_vision, empty, unreadable, failed,
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
    except UnreadableDocumentException as e:
        # The user's file is not what its extension says (a text renamed
        # .docx, a PDF locked by a password): the same WARNING, no traceback.
        logger.warning(f"Attachment {path.name}: skipped: {e.reason}")
        return "", "unreadable"
    except Exception:
        # Anything else is the extractor's own failure: a defect.
        logger.exception(f"Attachment {path.name}: extraction failed")
        return "", "failed"

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
    remaining = ATTACHMENT_TOKEN_BUDGET
    kept_chars = 0
    kept_tokens = 0

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
            cost = counter_text_tokens(text)
            if cost > remaining:
                # The longest head of the file the budget left holds.
                text = counter_head(text, remaining)
                cost = counter_text_tokens(text)
                truncated.append(path.name)
                body = f"{text}\n{TRUNCATION_MARKER}" if text else TRUNCATION_MARKER
                remaining = 0
            else:
                remaining -= cost
                body = text
            kept_chars += len(text)
            kept_tokens += cost
        blocks.append(f"{_BLOCK_HEADER.format(name=path.name)}\n{body}\n{_BLOCK_FOOTER}")

    logger.info(
        f"Attachments resolved: files={len(files)}, chars={kept_chars}, "
        f"tokens={kept_tokens}, "
        f"issues={len(issues)}, truncated={len(truncated)}"
    )
    return ResolvedAttachments(
        block="\n\n".join(blocks),
        paths=[str(p) for p in files],
        issues=issues,
        truncated_names=truncated,
    )


def prepend_attachment_block(block: str, question: str) -> str:
    """The text the model sees: attachment blocks first, the question last, so
    the question is the most recent thing in the turn."""
    if not block:
        return question
    return f"{block}\n\n{question}" if question else block
