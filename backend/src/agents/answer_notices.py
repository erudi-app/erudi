"""Answer notices: the app's own notes that open an answer.

A notice tells the user what the model did not get from the question -- an
image it cannot see, an attached file it could not read, attached text
shortened to fit -- instead of dropping it in silence. The conversation and
Arena services stream each one as an ``answer`` event before the model's
reply. The model never sees a notice: it is added to the answer the database
stores, not to the conversation's state.

Each kind is one entry of ``NOTICE_KINDS``, its English line with ``{files}``
where it names files. That line, in Markdown italics followed by a blank
line, is what the message content stores: what an older interface and a plain
read of the database show. The answer event carries ``notice: {kind,
params}`` next to it; the conversation service persists it in the message's
trace as ``{"t": "notice", "kind", "params"}``, pinned so the trace cap never
drops it, and the interface rebuilds the English line from the ``en`` locale
(which says exactly what ``notice_text`` says, ``tests/fixtures/
answer_notices.json`` holds both sides to it), finds it in the stored content
and shows ``chat:notices.<kind>`` in its language in its place. A message can
hold several notices, one event each.

Params:

* ``images_ignored``: none;
* ``attachments_unreadable``: ``files``, one ``{name, reason}`` per file, the
  reason a key of ``attachment_utils.REASON_TEXT`` (``too_many`` has no name
  and carries ``limit``, the file cap);
* ``attachments_truncated``: ``files``, the names of the shortened files.

``files`` holds at most ``NOTICE_FILES_SHOWN`` entries, then ``more`` counts
the others ("and 3 more"), so a dropped folder of long names cannot fill the
trace.

Adding a kind is one entry here and ``chat:notices.<kind>`` in the four
locales; a kind the interface does not know shows its stored English line.
"""

from __future__ import annotations

from src.utils.attachment_utils import MAX_ATTACHMENT_FILES, ResolvedAttachments, reason_text

# The answer-event field and the trace event of a notice.
NOTICE_EVENT = "notice"

# How many files a notice names before it counts the rest.
NOTICE_FILES_SHOWN = 20

NOTICE_KINDS: dict[str, str] = {
    # The current turn carries images but the model is not positively
    # vision-capable (#212): the runner's ``_StripImagesForTextModel`` drops them.
    "images_ignored": "This model doesn't support images — your image was ignored.",
    # A file the reader could not turn into text (#492).
    "attachments_unreadable": "Some attachments could not be read: {files}.",
    # Text cut to fit the per-question budget (``ATTACHMENT_TOKEN_BUDGET``).
    "attachments_truncated": "Attached content was shortened to fit this model's context: {files}.",
}


def english_list(items: list[str]) -> str:
    """Items joined as English joins a list, the way the interface's
    ``Intl.ListFormat("en-US")`` does: "a", "a and b", "a, b, and c"."""
    if len(items) <= 2:
        return " and ".join(items)
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _unreadable_item(file: dict) -> str:
    reason = reason_text(file["reason"], limit=file.get("limit"))
    return f"{file['name']} ({reason})" if file["name"] else reason


def notice_text(kind: str, params: dict) -> str:
    """The English line of a notice, without its italics."""
    template = NOTICE_KINDS[kind]
    if "files" not in params:
        return template
    if kind == "attachments_unreadable":
        items = [_unreadable_item(file) for file in params["files"]]
    else:
        items = list(params["files"])
    if params.get("more"):
        items.append(f"{params['more']} more")
    return template.format(files=english_list(items))


def notice_answer(kind: str, params: dict) -> dict:
    """The answer event of one notice: its English line, italic, then a blank
    line, and the notice the interface translates it from."""
    return {
        "t": "answer",
        "text": f"*{notice_text(kind, params)}*\n\n",
        NOTICE_EVENT: {"kind": kind, "params": params},
    }


def images_ignored_answer() -> dict:
    """The notice of images the model will not see."""
    return notice_answer("images_ignored", {})


def _files_params(files: list) -> dict:
    params: dict = {"files": files[:NOTICE_FILES_SHOWN]}
    if len(files) > NOTICE_FILES_SHOWN:
        params["more"] = len(files) - NOTICE_FILES_SHOWN
    return params


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
        answers.append(notice_answer("attachments_unreadable", _files_params(files)))
    if resolved.truncated_names:
        answers.append(
            notice_answer("attachments_truncated", _files_params(list(resolved.truncated_names)))
        )
    return answers


def notice_trace_event(notice: dict) -> dict:
    """The trace event persisted for an answer event's ``notice``."""
    return {"t": NOTICE_EVENT, "kind": notice["kind"], "params": notice["params"]}
