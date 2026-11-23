"""Answer notices: the app's own notes that open an answer.

A notice tells the user what the model did not get from the question -- an
image it cannot see, an attached file it could not read, attached text
shortened to fit -- instead of dropping it in silence. The conversation and
Arena services stream each one as an ``answer`` event before the model's
reply.

Each kind is one entry of ``NOTICE_KINDS``, its English line with ``{name}``
placeholders. That line, in Markdown italics, is what the message content
stores: what the model reads back on the next turn, what an older interface
shows. The answer event carries ``notice: {kind, params, text}`` next to it
(``text`` is the line without the italics); the conversation service persists
it in the message's trace as ``{"t": "notice", "kind", "params", "text"}``,
pinned so the trace cap never drops it, and the interface shows the sentence
of ``chat:notices.<kind>`` in its language in place of exactly ``text`` in the
stored content. A message can hold several notices, one event each.

Params:

* ``images_ignored``: none;
* ``attachments_unreadable``: ``files``, one ``{name, reason}`` per file, the
  reason a code of ``common:fileReasons`` (``missing``, ``unsupported``,
  ``pending_vision``, ``empty``, ``unreadable``, ``failed``, ``empty_folder``,
  ``too_many``; ``too_many`` has no name and carries ``limit``, the file cap);
* ``attachments_truncated``: ``files``, the names of the shortened files.

Adding a kind is one entry here and ``chat:notices.<kind>`` in the four
locales; a kind the interface does not know shows its stored English line.
"""

from __future__ import annotations

from src.utils.attachment_utils import MAX_ATTACHMENT_FILES, ResolvedAttachments

# The answer-event field and the trace event of a notice.
NOTICE_EVENT = "notice"

NOTICE_KINDS: dict[str, str] = {
    # The current turn carries images but the model is not positively
    # vision-capable (#212): the runner's ``_StripImagesForTextModel`` drops them.
    "images_ignored": "This model doesn't support images — your image was ignored.",
    # A file the reader could not turn into text (#492).
    "attachments_unreadable": "Some attachments could not be read: {files}.",
    # Text cut to fit the per-question budget (``ATTACHMENT_TOKEN_BUDGET``).
    "attachments_truncated": "Attached content was shortened to fit this model's context: {files}.",
}


def notice_answer(kind: str, params: dict, **values: str) -> dict:
    """The answer event of one notice: its English line, italic, then a blank
    line, and the notice the interface translates it from."""
    text = NOTICE_KINDS[kind].format(**values)
    return {
        "t": "answer",
        "text": f"*{text}*\n\n",
        NOTICE_EVENT: {"kind": kind, "params": params, "text": text},
    }


def images_ignored_answer() -> dict:
    """The notice of images the model will not see."""
    return notice_answer("images_ignored", {})


def attachment_notice_answers(resolved: ResolvedAttachments) -> list[dict]:
    """The notices of attachments not used in full, in order: the unreadable
    files, then the shortened ones. Empty when every file was read whole."""
    answers: list[dict] = []
    if resolved.issues:
        files = []
        for issue in resolved.issues:
            entry: dict = {"name": issue.name, "reason": issue.reason}
            if issue.reason == "too_many":
                entry["limit"] = MAX_ATTACHMENT_FILES
            files.append(entry)
        named = [
            f"{issue.name} ({issue.message})" if issue.name else issue.message
            for issue in resolved.issues
        ]
        answers.append(
            notice_answer("attachments_unreadable", {"files": files}, files=", ".join(named))
        )
    if resolved.truncated_names:
        answers.append(
            notice_answer(
                "attachments_truncated",
                {"files": list(resolved.truncated_names)},
                files=", ".join(resolved.truncated_names),
            )
        )
    return answers


def notice_trace_event(notice: dict) -> dict:
    """The trace event persisted for an answer event's ``notice``."""
    return {
        "t": NOTICE_EVENT,
        "kind": notice["kind"],
        "params": notice["params"],
        "text": notice["text"],
    }
