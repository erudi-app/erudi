"""A model download that hits a connection error retries on a fresh HTTP client (#747).

huggingface_hub 1.10.1 took the shared httpx client once, before its retry loop,
closed it on an ``httpx.ConnectError`` and retried with the same, now closed,
object: the download died with "Cannot send a request, as the client has been
closed" instead of retrying. Fixed upstream in 1.27.0 (the client is taken again
on each attempt). The download path (``HfFileSystem.get_file`` ->
``http_backoff``) goes through the same loop, so these offline tests drive it with
a fake transport that fails once, then answers."""

import httpx
import pytest
from huggingface_hub.utils import _http, http_backoff, http_stream_backoff

URL = "https://huggingface.co/org/model/resolve/main/config.json"


@pytest.fixture
def flaky_hub(monkeypatch):
    """Route huggingface_hub's shared client through a transport that raises
    ConnectError on its first request, then answers 200. Yields the call count
    and the clients the factory handed out."""
    state = {"calls": 0, "clients": []}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if state["calls"] == 1:
            raise httpx.ConnectError("network unreachable", request=request)
        return httpx.Response(200, content=b"{}", request=request)

    def factory() -> httpx.Client:
        client = httpx.Client(transport=httpx.MockTransport(handler))
        state["clients"].append(client)
        return client

    # The retry loop sleeps between attempts: no real wait in tests.
    monkeypatch.setattr(_http.time, "sleep", lambda *_: None)
    _http.set_client_factory(factory)
    yield state
    _http.set_client_factory(_http.default_client_factory)
    _http.close_session()


def test_http_backoff_retries_on_a_fresh_client_after_connect_error(flaky_hub):
    response = http_backoff("GET", URL, max_retries=3, base_wait_time=0)

    assert response.status_code == 200
    assert flaky_hub["calls"] == 2
    # The first client was closed by the retry loop; the second attempt used another.
    assert len(flaky_hub["clients"]) == 2
    assert flaky_hub["clients"][0].is_closed


def test_http_stream_backoff_retries_on_a_fresh_client_after_connect_error(flaky_hub):
    with http_stream_backoff("GET", URL, max_retries=3, base_wait_time=0) as response:
        assert response.status_code == 200

    assert flaky_hub["calls"] == 2
    assert len(flaky_hub["clients"]) == 2
