"""How Erudi talks to the Hugging Face Hub over huggingface_hub (#747).

The shared client
-----------------
huggingface_hub keeps ONE ``httpx.Client`` for the whole process
(``huggingface_hub.utils.get_session``), shared by every thread: the shards of a
model download fetched side by side, the embedding model's snapshot workers, the
catalog's metadata calls, the Hugging Face search, the sampling hints. Its retry
loop (``_http_backoff_base``) closes that client when a request meets an
``httpx.ConnectError`` (``close_session``; a timeout or a cut-off read is
retried on the same client) and retries on a new one. A request another thread
had already started on the old client, between ``get_session()`` and the
client's own closed check, then fails with ``RuntimeError: Cannot send a
request, as the client has been closed`` -- an error the retry loop does not
catch, so a flaky network killed a download as if Erudi were broken.

``HubClient`` hands such a request to the live shared client instead: the
closing thread has already swapped it in (``close_session`` clears the global
before it closes the old one), so the request goes out on the client a retry
would use. A request already inside the transport when the client closes loses
its connection, which httpx raises as a network error the retry loop does retry.

It also bounds how long a server can make the retry loop wait: the loop sleeps
for a 429's ``RateLimit`` reset or any retried response's ``Retry-After``
without a limit, so a response hook clamps both to ``MAX_SERVER_WAIT_SECONDS``.

``install_client_factory`` puts the factory in place; ``src.core.config`` calls
it at import, before any Hub request.

Network failures
----------------
``is_offline_error`` is the one predicate both Hub downloads (a model, the
embedding model) use to tell a network failure from a defect.
"""

import re
from typing import Optional

import httpx
from huggingface_hub.errors import OfflineModeIsEnabled
from huggingface_hub.utils import get_session, set_client_factory

# Longest wait, in seconds, a server's Retry-After or RateLimit reset can impose
# on one retry of huggingface_hub's loop (it adds 1 s of its own). Above the
# Hub's own rate-limit window (300 s), so a reset the Hub sends is never
# shortened: the cap only stops an absurd value (a day) from parking a transfer.
MAX_SERVER_WAIT_SECONDS = 330

# The reset field of a RateLimit header, e.g. '"api";r=0;t=55' (the format
# huggingface_hub's parse_ratelimit_headers reads).
_RATELIMIT_RESET = re.compile(r"(;\s*t\s*=\s*)(\d+)")

# The settings of huggingface_hub's own client, read off one its default
# factory builds (``install_client_factory``); HubClient adds the wait cap.
_hub_settings: dict = {"event_hooks": {}}


def cap_server_wait(response: httpx.Response) -> None:
    """Clamp the response's Retry-After and RateLimit reset to the cap.

    Only the numeric forms are touched, the ones huggingface_hub reads (an
    HTTP-date Retry-After it ignores, so it is left as it is). The header is
    rewritten because it is the only input of the loop's sleep; the 429 message
    huggingface_hub builds from it ("Retry after N seconds") then states the
    wait Erudi applies, and differs from the server's only past the cap."""
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
        hooks = _hub_settings["event_hooks"]
        kwargs.setdefault(
            "event_hooks",
            {
                "request": list(hooks.get("request", [])),
                "response": [*hooks.get("response", []), cap_server_wait],
            },
        )
        for name in ("follow_redirects", "timeout"):
            if name in _hub_settings:
                kwargs.setdefault(name, _hub_settings[name])
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


def _read_hub_settings() -> dict:
    """The request hooks, redirect policy and timeout of the client
    huggingface_hub's default factory builds (a private name: any change to
    it raises here, and ``install_client_factory`` then installs nothing)."""
    from huggingface_hub.utils import _http

    reference = _http.default_client_factory()
    try:
        return {
            "event_hooks": {name: list(hooks) for name, hooks in reference.event_hooks.items()},
            "follow_redirects": reference.follow_redirects,
            "timeout": reference.timeout,
        }
    finally:
        reference.close()


def install_client_factory() -> bool:
    """Make ``hub_client_factory`` the one huggingface_hub uses (it closes the
    current shared client, so the next request builds a ``HubClient``).

    Returns False, with one WARNING, when huggingface_hub's own client settings
    cannot be read: the app then runs on huggingface_hub's client unchanged
    rather than on a client that would drop its request hook."""
    try:
        settings = _read_hub_settings()
    except Exception as exc:
        # Degraded, not broken: downloads work, without the handover and the cap.
        from src.core.logging import logger

        logger.warning(
            "Could not read huggingface_hub's client settings; keeping its own client "
            f"(no closed-client handover, no server wait cap): {type(exc).__name__}: {exc}"
        )
        return False
    _hub_settings.clear()
    _hub_settings.update(settings)
    set_client_factory(hub_client_factory)
    return True


# Lowercased substrings marking a no-network failure anywhere in the exception
# chain, for the shapes huggingface_hub wraps without keeping the httpx error
# as a cause. Mirrored by OFFLINE_NETWORK_MARKERS in
# frontend/src/utils/bugCounter.js.
OFFLINE_MESSAGE_MARKERS = (
    "cannot reach https://huggingface.co",
    "couldn't connect to 'https://huggingface.co'",
    "we couldn't connect to 'https://huggingface.co'",
    "offline mode is enabled",
    "outgoing traffic has been disabled",
    "getaddrinfo",
    "nodename nor servname",
    "name or service not known",
    "temporary failure in name resolution",
    "network is unreachable",
    "max retries exceeded",
    "connection error",
)

# What huggingface_hub's retry loop retries, then raises once its retries are
# spent: the connection let the transfer down, not Erudi.
_NETWORK_ERRORS = (
    OfflineModeIsEnabled,
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)


def is_offline_error(exc: BaseException) -> bool:
    """True if the exception chain looks like a network failure reaching the Hub
    (#109, #164, #747). A ``RuntimeError`` about a closed client is not one:
    ``HubClient`` hands those requests on, so one reaching here is a defect."""
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, _NETWORK_ERRORS):
            return True
        message = str(current).lower()
        if any(marker in message for marker in OFFLINE_MESSAGE_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False
