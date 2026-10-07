"""_FoldSystemIntoUserMiddleware folds the system prompt into the first user turn.

Added by the runner only when the model is positively detected as not
system-role-capable (Gemma 2). ``create_agent`` carries the prompt as
``request.system_message`` and the model node prepends it; the middleware clears
it and merges its text into the first human message so the chat template never
sees a system role.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.middleware import _FoldSystemIntoUserMiddleware


class _FakeRequest:
    def __init__(self, messages, system_message):
        self.messages = messages
        self.system_message = system_message

    def override(self, **kw):
        new = _FakeRequest(list(self.messages), self.system_message)
        for k, v in kw.items():
            setattr(new, k, v)
        return new


def _fold(messages, system_text):
    sys_msg = SystemMessage(content=system_text) if system_text is not None else None
    req = _FakeRequest(messages, sys_msg)
    return _FoldSystemIntoUserMiddleware()._fold(req)


@pytest.mark.unit
class TestFoldSystemIntoUser:
    def test_folds_into_first_human_and_clears_system(self):
        out = _fold([HumanMessage(content="hello")], "You are Erudi.")
        assert out.system_message is None
        assert out.messages[0].content == "You are Erudi.\n\nhello"

    def test_folds_into_first_human_across_history(self):
        msgs = [
            HumanMessage(content="first"),
            AIMessage(content="reply"),
            HumanMessage(content="second"),
        ]
        out = _fold(msgs, "SYS")
        assert out.system_message is None
        assert out.messages[0].content == "SYS\n\nfirst"  # first human only
        assert out.messages[2].content == "second"  # later turns untouched

    def test_no_system_message_is_passthrough(self):
        msgs = [HumanMessage(content="hello")]
        out = _fold(msgs, None)
        assert out.messages[0].content == "hello"

    def test_empty_system_message_just_cleared(self):
        out = _fold([HumanMessage(content="hi")], "   ")
        assert out.system_message is None
        assert out.messages[0].content == "hi"  # nothing prepended

    def test_no_human_message_becomes_user_message(self):
        out = _fold([AIMessage(content="orphan")], "SYS")
        assert out.system_message is None
        assert out.messages[0].type == "human"
        assert out.messages[0].content == "SYS"

    def test_multimodal_first_human_preserves_images(self):
        content = [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:x"}},
        ]
        out = _fold([HumanMessage(content=content)], "SYS")
        assert out.system_message is None
        parts = out.messages[0].content
        assert parts[0] == {"type": "text", "text": "SYS\n\nlook"}
        assert parts[1]["type"] == "image_url"  # image kept for the VLM

    def test_the_fold_keeps_the_messages_metadata_and_id(self):
        """The summary's ``lc_source`` and any other metadata (and the
        message id) survive into the request on system-role-less models."""
        summary = HumanMessage(
            content="Here is a summary of the conversation to date:\n\nfacts",
            additional_kwargs={"lc_source": "summarization", "note": "kept"},
            id="summary-id",
        )
        out = _fold([summary, AIMessage(content="a"), HumanMessage(content="q")], "SYS")
        folded = out.messages[0]
        assert folded.content.startswith("SYS\n\nHere is a summary")
        assert folded.additional_kwargs == {
            "lc_source": "summarization",
            "note": "kept",
        }
        assert folded.id == "summary-id"
