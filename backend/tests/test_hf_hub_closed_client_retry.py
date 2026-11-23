"""A model download on a flaky network never dies on a closed HTTP client (#747).

huggingface_hub 1.10.1 took the shared httpx client once, before its retry loop,
closed it on an ``httpx.ConnectError`` and retried with the same, now closed,
object: the download died with "Cannot send a request, as the client has been
closed" instead of retrying. Fixed upstream in 1.27.0 (the client is taken again
on each attempt). The download path (``HfFileSystem.get_file`` ->
``http_backoff``) goes through the same loop, so these offline tests drive it with
a fake transport that fails once, then answers.

The client is still ONE object shared by every thread, and the retry loop that
sees a ConnectError closes it (``close_session``). A transfer in another thread
(the shards of one model download, run side by side) that already took that
client from ``get_session()`` but has not sent its request yet then dies with
the same RuntimeError, which the retry loop does not catch. Erudi installs its
own client factory (``src.core.hf_http``) whose client hands such a request to
the live shared client instead; the race tests below pin both halves."""

import threading

import httpx
import pytest
from huggingface_hub.utils import _http, http_backoff, http_stream_backoff

from src.core import hf_http

URL = "https://huggingface.co/org/model/resolve/main/config.json"


@pytest.fixture
def restore_hub_client(monkeypatch):
    """No real wait in the retry loop; Erudi's factory back in place afterwards."""
    monkeypatch.setattr(_http.time, "sleep", lambda *_: None)
    yield
    hf_http.install_client_factory()


@pytest.fixture
def flaky_hub(restore_hub_client):
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
        client = hf_http.HubClient(transport=httpx.MockTransport(handler))
        state["clients"].append(client)
        return client

    _http.set_client_factory(factory)
    yield state


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


def _race_two_transfers(client_class, *, stream: bool = False):
    """Drive two concurrent transfers through huggingface_hub's shared client.

    Thread B takes the shared client, then waits (inside ``build_request``, the
    step between ``get_session()`` and the client's closed check) until thread A
    hit a ConnectError and the retry loop closed that very client. Returns each
    thread's outcome: the status code, or the exception it raised."""
    a_closed = threading.Event()
    b_has_client = threading.Event()
    state = {"a_failed": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/a") and not state["a_failed"]:
            state["a_failed"] = True
            # A fails only once B holds the client A's retry loop then closes.
            assert b_has_client.wait(5), "thread B never took the shared client"
            raise httpx.ConnectError("network unreachable", request=request)
        return httpx.Response(200, content=b"ok", request=request)

    class Gated(client_class):
        def build_request(self, *args, **kwargs):
            if threading.current_thread().name == "transfer-b" and not b_has_client.is_set():
                b_has_client.set()
                assert a_closed.wait(5), "thread A never closed the shared client"
            return super().build_request(*args, **kwargs)

        def close(self):
            super().close()
            a_closed.set()

    _http.set_client_factory(lambda: Gated(transport=httpx.MockTransport(handler)))
    outcome = {}

    def transfer(name: str) -> None:
        url = f"{URL}/{name[-1]}"
        try:
            if stream:
                with http_stream_backoff("GET", url, max_retries=3) as response:
                    outcome[name] = response.status_code
            else:
                outcome[name] = http_backoff("GET", url, max_retries=3).status_code
        except Exception as exc:  # the outcome under test, read by the caller
            outcome[name] = exc

    threads = [
        threading.Thread(target=transfer, args=(name,), name=name)
        for name in ("transfer-a", "transfer-b")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    return outcome


def test_a_plain_shared_client_closed_by_another_thread_kills_the_transfer(restore_hub_client):
    """The race exists in the installed huggingface_hub. If this ever stops
    failing, upstream fixed it and ``src.core.hf_http``'s rerouting can go."""
    outcome = _race_two_transfers(httpx.Client)

    assert outcome["transfer-a"] == 200  # A retried on a fresh client
    assert isinstance(outcome["transfer-b"], RuntimeError)
    assert "client has been closed" in str(outcome["transfer-b"])


@pytest.mark.parametrize("stream", [False, True], ids=["request", "stream"])
def test_erudi_client_closed_by_another_thread_hands_the_request_on(restore_hub_client, stream):
    outcome = _race_two_transfers(hf_http.HubClient, stream=stream)

    assert outcome == {"transfer-a": 200, "transfer-b": 200}


def test_erudi_factory_is_the_one_installed():
    """``src.core.config`` installs it at import, before any Hub request."""
    import src.core.config  # noqa: F401

    assert _http._GLOBAL_CLIENT_FACTORY is hf_http.hub_client_factory


def test_erudi_client_keeps_huggingface_hub_defaults():
    """Same request hook, redirects and timeout as huggingface_hub's own
    factory: only the closed-client handover and the wait cap are added."""
    default = _http.default_client_factory()
    ours = hf_http.hub_client_factory()
    try:
        assert set(default.event_hooks["request"]) <= set(ours.event_hooks["request"])
        assert ours.follow_redirects == default.follow_redirects
        assert ours.timeout == default.timeout
    finally:
        default.close()
        ours.close()


@pytest.mark.parametrize(
    "headers, expected",
    [
        ({"Retry-After": "86400"}, {"retry-after": "30"}),
        ({"Retry-After": "12"}, {"retry-after": "12"}),
        ({"RateLimit": '"resolvers";r=0;t=3600'}, {"ratelimit": '"resolvers";r=0;t=30'}),
        ({"RateLimit": '"api";r=0;t=55'}, {"ratelimit": '"api";r=0;t=30'}),
        ({"RateLimit": '"api";r=0;t=7'}, {"ratelimit": '"api";r=0;t=7'}),
        ({"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}, {}),
    ],
)
def test_server_wait_is_capped(headers, expected):
    """A server's Retry-After or RateLimit reset never makes huggingface_hub's
    retry loop sleep longer than ``MAX_SERVER_WAIT_SECONDS``."""
    response = httpx.Response(429, headers=headers, request=httpx.Request("GET", URL))
    hf_http.cap_server_wait(response)

    for name, value in expected.items():
        assert response.headers[name] == value
    assert hf_http.MAX_SERVER_WAIT_SECONDS == 30


def test_retry_loop_waits_at_most_the_cap(restore_hub_client, monkeypatch):
    """End to end through huggingface_hub's loop: a 429 asking for a day waits
    the cap (plus the loop's own 1 s margin), then the retry answers."""
    sleeps = []
    monkeypatch.setattr(_http.time, "sleep", sleeps.append)
    answers = iter(
        [
            httpx.Response(429, headers={"Retry-After": "86400"}),
            httpx.Response(200, content=b"ok"),
        ]
    )
    transport = httpx.MockTransport(lambda request: next(answers))
    _http.set_client_factory(lambda: hf_http.HubClient(transport=transport))

    assert http_backoff("GET", URL, max_retries=3).status_code == 200
    assert sleeps == [hf_http.MAX_SERVER_WAIT_SECONDS + 1]
