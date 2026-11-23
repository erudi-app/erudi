"""#764 - the notes that open an answer, carried with a kind the interface
translates them from.

A note (an image the model cannot see, an attachment it could not read or got
shortened) is an ``answer`` event whose English line goes into the message
content, and whose ``notice`` (kind, params) becomes a ``{"t": "notice", ...}``
trace event. The interface rebuilds the English line from its ``en`` locale to
find it in the content, so both sides say exactly the lines of
``fixtures/answer_notices.json``.
"""

import json
import re
from pathlib import Path

import pytest

from src.agents.answer_notices import (
    NOTICE_EVENT,
    NOTICE_FILES_SHOWN,
    NOTICE_KINDS,
    attachment_notice_answers,
    english_list,
    images_ignored_answer,
    notice_text,
    notice_trace_event,
)
from src.domains.conversations.services import TRACE_MAX_BYTES, _cap_trace
from src.utils.attachment_utils import (
    MAX_ATTACHMENT_FILES,
    REASON_TEXT,
    AttachmentIssue,
    ResolvedAttachments,
    resolve_attachments,
)

pytestmark = pytest.mark.unit

_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "answer_notices.json").read_text(encoding="utf-8")
)
_LOCALES = Path(__file__).resolve().parents[2] / "frontend" / "src" / "locales"
_LANGUAGES = ["en", "fr", "es", "zh"]


def _issue(name, reason):
    return AttachmentIssue(name=name, path=f"/docs/{name}", reason=reason)


def _locale(language, namespace):
    return json.loads((_LOCALES / language / f"{namespace}.json").read_text(encoding="utf-8"))


def _placeholders(text, braces=r"\{\{\s*(\w+)\s*\}\}"):
    return set(re.findall(braces, text))


# ===== One wording: the backend's line is the interface's English =====


@pytest.mark.parametrize("case", _FIXTURE["cases"], ids=lambda c: c["english"][:40])
def test_the_english_line_is_the_shared_one(case):
    assert notice_text(case["kind"], case["params"]) == case["english"]


def test_the_shared_lines_cover_every_kind_and_reason():
    cases = _FIXTURE["cases"]
    assert {case["kind"] for case in cases} == set(NOTICE_KINDS)
    reasons = {
        file["reason"]
        for case in cases
        if case["kind"] == "attachments_unreadable"
        for file in case["params"]["files"]
    }
    assert reasons == set(REASON_TEXT)


@pytest.mark.parametrize(
    ("items", "joined"),
    [([], ""), (["a"], "a"), (["a", "b"], "a and b"), (["a", "b", "c"], "a, b, and c")],
)
def test_english_list_joins_as_the_interface_does(items, joined):
    assert english_list(items) == joined


# ===== Every kind and reason has its sentence in every interface language =====


@pytest.mark.parametrize("language", _LANGUAGES)
def test_every_kind_has_its_sentence_with_exactly_its_placeholders(language):
    """A kind the interface has no key for shows its stored English line: a
    hole here is a note left untranslated."""
    notices = _locale(language, "chat")["notices"]

    assert set(notices) == set(NOTICE_KINDS)
    for kind, template in NOTICE_KINDS.items():
        assert _placeholders(notices[kind]) == _placeholders(template, r"\{(\w+)\}"), kind


@pytest.mark.parametrize("language", _LANGUAGES)
def test_every_reason_has_its_text_with_exactly_its_placeholders(language):
    reasons = _locale(language, "common")["fileReasons"]

    assert set(REASON_TEXT) <= set(reasons)
    for reason, text in REASON_TEXT.items():
        assert _placeholders(reasons[reason]) == _placeholders(text, r"\{(\w+)\}"), reason


def test_the_english_reasons_are_the_backend_texts():
    reasons = _locale("en", "common")["fileReasons"]
    for reason, text in REASON_TEXT.items():
        assert reasons[reason] == text.replace("{limit}", "{{limit}}"), reason


# ===== The events =====


class TestImagesIgnored:
    def test_the_answer_carries_the_english_line_and_its_kind(self):
        event = images_ignored_answer()

        assert event["t"] == "answer"
        assert event["text"] == (
            "*This model doesn't support images — your image was ignored.*\n\n"
        )
        assert event[NOTICE_EVENT] == {"kind": "images_ignored", "params": {}}

    def test_the_trace_event_is_the_kind_and_params_alone(self):
        event = notice_trace_event(images_ignored_answer()[NOTICE_EVENT])

        assert event == {"t": "notice", "kind": "images_ignored", "params": {}}


class TestAttachmentNotices:
    def test_a_clean_resolution_produces_no_notice(self, tmp_path):
        doc = tmp_path / "notes.txt"
        doc.write_text("Quarterly notes.", encoding="utf-8")

        assert attachment_notice_answers(resolve_attachments([str(doc)])) == []

    def test_an_unreadable_file_is_named_with_its_reason(self):
        resolved = ResolvedAttachments(issues=[_issue("scan.pdf", "pending_vision")])

        [event] = attachment_notice_answers(resolved)

        assert event[NOTICE_EVENT] == {
            "kind": "attachments_unreadable",
            "params": {"files": [{"name": "scan.pdf", "reason": "pending_vision"}]},
        }
        assert event["text"] == (
            "*Some attachments could not be read: scan.pdf (no extractable text (a scan or "
            "an image), and text recognition is not bundled yet).*\n\n"
        )

    def test_the_file_cap_carries_its_limit(self):
        resolved = ResolvedAttachments(issues=[AttachmentIssue("", "", "too_many")])

        [event] = attachment_notice_answers(resolved)

        assert event[NOTICE_EVENT]["params"] == {
            "files": [{"name": "", "reason": "too_many", "limit": MAX_ATTACHMENT_FILES}]
        }

    def test_a_shortened_file_is_its_own_note(self, tmp_path):
        bad = tmp_path / "archive.zip"
        bad.write_bytes(b"PK\x03\x04")
        huge = tmp_path / "huge.txt"
        huge.write_text("A" * 30_000, encoding="utf-8")

        unreadable, truncated = attachment_notice_answers(
            resolve_attachments([str(bad), str(huge)])
        )

        assert unreadable[NOTICE_EVENT] == {
            "kind": "attachments_unreadable",
            "params": {"files": [{"name": "archive.zip", "reason": "unsupported"}]},
        }
        assert truncated == {
            "t": "answer",
            "text": "*Attached content was shortened to fit this model's context: huge.txt.*\n\n",
            NOTICE_EVENT: {"kind": "attachments_truncated", "params": {"files": ["huge.txt"]}},
        }

    def test_a_long_list_names_the_first_files_and_counts_the_rest(self):
        names = [f"{i:02d}-{'n' * 200}.pdf" for i in range(MAX_ATTACHMENT_FILES)]
        resolved = ResolvedAttachments(
            issues=[_issue(name, "pending_vision") for name in names], truncated_names=names
        )

        unreadable, truncated = attachment_notice_answers(resolved)

        more = MAX_ATTACHMENT_FILES - NOTICE_FILES_SHOWN
        assert [f["name"] for f in unreadable[NOTICE_EVENT]["params"]["files"]] == names[
            :NOTICE_FILES_SHOWN
        ]
        assert unreadable[NOTICE_EVENT]["params"]["more"] == more
        assert truncated[NOTICE_EVENT]["params"] == {
            "files": names[:NOTICE_FILES_SHOWN],
            "more": more,
        }
        assert unreadable["text"].endswith(f", and {more} more.*\n\n")

    def test_the_notices_of_a_full_folder_leave_room_for_the_reasoning(self):
        """Pinned, the notices are never dropped: bounded, they leave most of
        the trace to the reasoning that follows them."""
        names = [f"{i:02d}-{'n' * 200}.pdf" for i in range(MAX_ATTACHMENT_FILES)]
        resolved = ResolvedAttachments(
            issues=[_issue(name, "pending_vision") for name in names], truncated_names=names
        )
        notices = [notice_trace_event(a[NOTICE_EVENT]) for a in attachment_notice_answers(resolved)]
        thinking = [{"t": "thinking", "text": f"{i}-" + "x" * 1000} for i in range(50)]

        capped = _cap_trace([*notices, *thinking])

        assert len(json.dumps(capped)) <= TRACE_MAX_BYTES
        assert capped[:2] == notices
        assert len(json.dumps(notices)) < TRACE_MAX_BYTES // 3
        assert sum(1 for e in capped if e["t"] == "thinking") >= 15


def test_no_kind_carries_markup_in_its_line():
    """The line is what the interface replaces: the italics stay around it in
    the content, never inside it."""
    for template in NOTICE_KINDS.values():
        assert "*" not in template
