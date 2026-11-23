"""Offline download errors (issue #109 P2).

An offline model download must surface the friendly ``OFFLINE_DOWNLOAD_MESSAGE``
instead of the raw hf_hub "Cannot reach https://..." string, mirroring the e5
embedding-download offline path.
"""

import httpx
import pytest
import requests

from src.core.exceptions import HuggingFaceAPIException
from src.domains.llms.services import (
    OFFLINE_DOWNLOAD_MESSAGE,
    _is_offline_download_error,
)

pytestmark = pytest.mark.unit


class TestOfflineDownloadErrorMapping:
    def test_requests_connection_error_is_offline(self):
        exc = requests.exceptions.ConnectionError("Max retries exceeded with url")
        assert _is_offline_download_error(exc) is True

    def test_hf_cannot_reach_message_is_offline(self):
        exc = Exception("Cannot reach https://huggingface.co: (ConnectionError) network down")
        assert _is_offline_download_error(exc) is True

    def test_chained_connection_error_is_offline(self):
        # hf_hub/transformers re-wrap the original error; the chain must be walked.
        outer = None
        try:
            try:
                raise requests.exceptions.ConnectionError("boom")
            except Exception as inner:
                raise RuntimeError("download failed") from inner
        except RuntimeError as e:
            outer = e
        assert _is_offline_download_error(outer) is True

    @pytest.mark.parametrize(
        "exc",
        [
            httpx.ConnectError("All connection attempts failed"),
            httpx.ReadTimeout("The read operation timed out"),
            httpx.ReadError("[Errno 54] Connection reset by peer"),
            httpx.RemoteProtocolError("peer closed connection without sending complete message"),
        ],
        ids=["connect", "read-timeout", "read-reset", "cut-mid-body"],
    )
    def test_network_failure_after_the_retries_is_offline(self, exc):
        # huggingface_hub (httpx) retries these, then raises the last one: a
        # download the network let down for good, not an Erudi defect (#747).
        assert _is_offline_download_error(exc) is True

    def test_closed_client_is_not_offline(self):
        # src.core.hf_http hands a request on a closed shared client to the
        # live one: if this error still reaches a download, it is a defect.
        exc = RuntimeError("Cannot send a request, as the client has been closed.")
        assert _is_offline_download_error(exc) is False

    def test_generic_errors_are_not_offline(self):
        assert _is_offline_download_error(ValueError("no space left on device")) is False
        assert _is_offline_download_error(Exception("403 Client Error: Forbidden")) is False

    def test_offline_message_is_ascii_and_friendly(self):
        # ASCII-safe (no em dash) so it is log-safe on cp1252 consoles.
        OFFLINE_DOWNLOAD_MESSAGE.encode("ascii")
        assert "offline" in OFFLINE_DOWNLOAD_MESSAGE.lower()

    def test_mapped_exception_carries_friendly_message_verbatim(self):
        # The download except-block raises this; endpoints store str(e) as the
        # job's error_message, so str() must be the friendly text (no support
        # suffix, which AppBaseException only appends to .message).
        exc = HuggingFaceAPIException(OFFLINE_DOWNLOAD_MESSAGE, trace="raw hub error")
        assert str(exc) == OFFLINE_DOWNLOAD_MESSAGE
