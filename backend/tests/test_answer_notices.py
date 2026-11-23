"""#764 - the notes that open an answer, carried with a kind the interface
translates them from.

A note (an image the model cannot see, an attachment it could not read or got
shortened) is an ``answer`` event whose English line goes into the message
content, and whose ``notice`` (kind, params, that line) becomes a
``{"t": "notice", ...}`` trace event the interface renders in its language.
"""

import pytest

from src.agents.answer_notices import (
    NOTICE_EVENT,
    NOTICE_KINDS,
    attachment_notice_answers,
    images_ignored_answer,
    notice_trace_event,
)
from src.utils.attachment_utils import (
    MAX_ATTACHMENT_FILES,
    AttachmentIssue,
    ResolvedAttachments,
    resolve_attachments,
)

pytestmark = pytest.mark.unit


def _issue(name, reason, message="why"):
    return AttachmentIssue(name=name, path=f"/docs/{name}", reason=reason, message=message)


class TestImagesIgnored:
    def test_the_answer_carries_the_english_line_and_its_kind(self):
        event = images_ignored_answer()

        assert event["t"] == "answer"
        assert event["text"] == (
            "*This model doesn't support images — your image was ignored.*\n\n"
        )
        assert event[NOTICE_EVENT] == {
            "kind": "images_ignored",
            "params": {},
            "text": "This model doesn't support images — your image was ignored.",
        }

    def test_the_trace_event_is_the_notice(self):
        event = notice_trace_event(images_ignored_answer()[NOTICE_EVENT])

        assert event == {
            "t": "notice",
            "kind": "images_ignored",
            "params": {},
            "text": "This model doesn't support images — your image was ignored.",
        }


class TestAttachmentNotices:
    def test_a_clean_resolution_produces_no_notice(self, tmp_path):
        doc = tmp_path / "notes.txt"
        doc.write_text("Quarterly notes.", encoding="utf-8")

        assert attachment_notice_answers(resolve_attachments([str(doc)])) == []

    def test_an_unreadable_file_is_named_with_its_reason(self):
        resolved = ResolvedAttachments(
            issues=[_issue("scan.pdf", "pending_vision", "it holds no extractable text")]
        )

        [event] = attachment_notice_answers(resolved)

        assert event["text"] == (
            "*Some attachments could not be read: scan.pdf (it holds no extractable text).*\n\n"
        )
        assert event[NOTICE_EVENT] == {
            "kind": "attachments_unreadable",
            "params": {"files": [{"name": "scan.pdf", "reason": "pending_vision"}]},
            "text": "Some attachments could not be read: scan.pdf (it holds no extractable text).",
        }

    def test_the_file_cap_carries_its_limit(self):
        resolved = ResolvedAttachments(issues=[_issue("", "too_many", "only the first 50")])

        [event] = attachment_notice_answers(resolved)

        assert event[NOTICE_EVENT]["params"] == {
            "files": [{"name": "", "reason": "too_many", "limit": MAX_ATTACHMENT_FILES}]
        }
        assert (
            event[NOTICE_EVENT]["text"] == "Some attachments could not be read: only the first 50."
        )

    def test_a_shortened_file_is_its_own_note(self, tmp_path):
        bad = tmp_path / "archive.zip"
        bad.write_bytes(b"PK\x03\x04")
        huge = tmp_path / "huge.txt"
        huge.write_text("A" * 30_000, encoding="utf-8")

        unreadable, truncated = attachment_notice_answers(
            resolve_attachments([str(bad), str(huge)])
        )

        assert unreadable[NOTICE_EVENT]["kind"] == "attachments_unreadable"
        assert unreadable[NOTICE_EVENT]["params"] == {
            "files": [{"name": "archive.zip", "reason": "unsupported"}]
        }
        assert truncated[NOTICE_EVENT] == {
            "kind": "attachments_truncated",
            "params": {"files": ["huge.txt"]},
            "text": "Attached content was shortened to fit this model's context: huge.txt.",
        }
        assert truncated["text"] == f"*{truncated[NOTICE_EVENT]['text']}*\n\n"

    def test_no_kind_carries_markup_in_its_line(self):
        """The stored ``text`` is what the interface replaces: no Markdown in
        it, the italics stay around it in the content."""
        for kind in NOTICE_KINDS:
            assert "*" not in NOTICE_KINDS[kind]
