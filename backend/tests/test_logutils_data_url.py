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


def test_reduces_a_data_url_with_parameters_before_base64():
    assert (
        reduce_data_urls("data:image/png;charset=utf-8;name=a.b;base64,QUJD end")
        == "data:image/png;charset=utf-8;name=a.b;base64,<4 chars> end"
    )


def test_record_factory_reduces_records_of_any_logger_and_handler(monkeypatch):
    # The MLX child builds its handlers after the factory is installed.
    import io
    import logging

    from src.core.logutils import install_data_url_record_factory

    monkeypatch.setattr(logging, "_logRecordFactory", logging.getLogRecordFactory())
    install_data_url_record_factory()
    install_data_url_record_factory()  # idempotent

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    log = logging.getLogger("mlx-child-test")
    log.propagate = False
    log.addHandler(handler)
    try:
        try:
            raise ValueError("Failed to load image from data:image/png;base64,Tm90ZSBpcyBub3Q=")
        except ValueError:
            log.exception("boom")
    finally:
        log.removeHandler(handler)
    out = stream.getvalue()
    assert "Tm90" not in out
    assert "data:image/png;base64,<16 chars>" in out
