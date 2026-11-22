"""``reduce_data_urls``: base64 never reaches a log line or an error's Details (#760)."""

import pytest

from src.core.logutils import reduce_data_urls

pytestmark = pytest.mark.unit


def test_reduces_the_payload_and_keeps_the_mime():
    assert (
        reduce_data_urls("x data:image/jpeg;base64,/9j/4AAQSkZJRg== y")
        == "x data:image/jpeg;base64,<16 chars> y"
    )


def test_leaves_other_text_alone():
    text = "see data:text/plain,hello and base64, nothing"
    assert reduce_data_urls(text) == text
