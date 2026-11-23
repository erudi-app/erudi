"""The HTTP client every huggingface_hub call in Erudi goes through (#747).

huggingface_hub keeps ONE ``httpx.Client`` for the whole process
(``huggingface_hub.utils.get_session``), shared by every thread: the shards of a
model download fetched side by side, the catalog's metadata calls, the Hugging
Face search. Its retry loop (``_http_backoff_base``) closes that client when a
request meets an ``httpx.ConnectError`` (``close_session``) and retries on a new
one. A request another thread had already started on the old client, between
``get_session()`` and the client's own closed check, then fails with
``RuntimeError: Cannot send a request, as the client has been closed`` -- an
error the retry loop does not catch, so a flaky network killed a download as if
Erudi were broken.

``HubClient`` hands such a request to the live shared client instead: the
closing thread has already swapped it in (``close_session`` clears the global
before it closes the old one), so the request goes out on the client a retry
would use. A request already inside the transport when the client closes loses
its connection, which httpx raises as a network error the retry loop does retry.

It also caps how long a server can make the retry loop wait: the loop sleeps
for a 429's ``RateLimit`` reset or any retried response's ``Retry-After``
without a bound, so a response hook clamps both to ``MAX_SERVER_WAIT_SECONDS``,
the same cap Erudi's own 429 ladder applies (``config._call_with_429_retry``).

``install_client_factory`` puts the factory in place; ``src.core.config`` calls
it at import, before any Hub request.
"""

import re

import httpx
from huggingface_hub.utils import get_session, set_client_factory
from huggingface_hub.utils._http import hf_request_event_hook

# Longest wait, in seconds, a server's Retry-After or RateLimit reset can impose
# on one retry of huggingface_hub's loop (it adds 1 s of its own).
MAX_SERVER_WAIT_SECONDS = 30

# The reset field of a RateLimit header, e.g. '"api";r=0;t=55' (the format
# huggingface_hub's parse_ratelimit_headers reads).
_RATELIMIT_RESET = re.compile(r"(;\s*t\s*=\s*)(\d+)")


def cap_server_wait(response: httpx.Response) -> None:
    """Clamp the response's Retry-After and RateLimit reset to the cap.

    Only the numeric forms are touched, the ones huggingface_hub reads (an
    HTTP-date Retry-After it ignores, so it is left as it is)."""
    retry_after = response.headers.get("retry-after", "").strip()
    if retry_after.isdigit() and int(retry_after) > MAX_SERVER_WAIT_SECONDS:
        response.headers["retry-after"] = str(MAX_SERVER_WAIT_SECONDS)
    ratelimit = response.headers.get("ratelimit")
    if ratelimit:
        response.headers["ratelimit"] = _RATELIMIT_RESET.sub(
            lambda m: m.group(1) + str(min(int(m.group(2)), MAX_SERVER_WAIT_SECONDS)),
            ratelimit,
        )


class HubClient(httpx.Client):
    """huggingface_hub's default client, plus the closed-client handover and
    the server-wait cap (see the module docstring)."""

    def __init__(self, **kwargs):
        # huggingface_hub's default_client_factory settings, kept identical.
        kwargs.setdefault(
            "event_hooks", {"request": [hf_request_event_hook], "response": [cap_server_wait]}
        )
        kwargs.setdefault("follow_redirects", True)
        kwargs.setdefault("timeout", None)
        super().__init__(**kwargs)

    def send(self, request: httpx.Request, **kwargs) -> httpx.Response:
        if self.is_closed:
            # Another thread's retry loop closed this shared client after this
            # thread took it: the live one goes in its place.
            live = get_session()
            if live is not self:
                return live.send(request, **kwargs)
        return super().send(request, **kwargs)


def hub_client_factory() -> httpx.Client:
    """The factory huggingface_hub builds its shared client with."""
    return HubClient()


def install_client_factory() -> None:
    """Make ``hub_client_factory`` the one huggingface_hub uses (it closes the
    current shared client, so the next request builds a ``HubClient``)."""
    set_client_factory(hub_client_factory)
