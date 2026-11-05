"""``error_detail``: the cleaned, one-line cause shown under a generic failure (#615)."""

import pytest

from src.core.error_detail import MAX_DETAIL_CHARS, error_detail
from src.core.exceptions import EngineException

pytestmark = pytest.mark.unit


def _detail(message: str) -> str:
    return error_detail(ValueError(message))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # paths collapse to their file name
        ("cannot open /Users/jane/models/q4.gguf now", "cannot open q4.gguf now"),
        (
            "missing /Users/jane/Library/Application Support/Erudi/model.gguf",
            "missing model.gguf",
        ),
        ("cannot open C:\\Users\\jane\\models\\q4.gguf: denied", "cannot open q4.gguf: denied"),
        ("cannot open C:/Users/jane/q4.gguf", "cannot open q4.gguf"),
        ("cannot open C:\\Users\\Jane Doe\\q4.gguf", "cannot open q4.gguf"),
        ("share \\\\nas\\models\\team\\q4.gguf is gone", "share q4.gguf is gone"),
        ("loading file:///Users/jane/q4.gguf failed", "loading q4.gguf failed"),
        ("loading file:///C:/Users/jane/q4.gguf failed", "loading q4.gguf failed"),
        ('File "/opt/erudi/src/Message.py", line 103', 'File "Message.py", line 103'),
        # not paths: left alone
        ("and/or either", "and/or either"),
        ("ratio 1/2 reached", "ratio 1/2 reached"),
        ("relative dir/file.txt stays", "relative dir/file.txt stays"),
        ("Message content too long (max 32K chars)", "Message content too long (max 32K chars)"),
        (
            "Server error for url 'http://127.0.0.1:27300/v1/chat/completions'",
            "Server error for url 'http://127.0.0.1:27300/v1/chat/completions'",
        ),
        # credentials
        ("GET https://bob:hunter2@host.example/x", "GET https://***@host.example/x"),
        ("GET https://bob@host.example/x", "GET https://***@host.example/x"),
        ("Authorization: Bearer abc.DEF-123_xyz== rejected", "Authorization: Bearer *** rejected"),
        ("bearer abc123 rejected", "Bearer *** rejected"),
        (
            "spawn llama-server --api-key s3cret --port 1",
            "spawn llama-server --api-key *** --port 1",
        ),
        (
            "spawn llama-server --api-key=s3cret --port 1",
            "spawn llama-server --api-key=*** --port 1",
        ),
        ("cmd ['llama-server', '--api-key', 's3cret']", "cmd ['llama-server', '--api-key', '***']"),
        ("GET /x?api_key=s3cret&page=2", "GET /x?api_key=***&page=2"),
        ("apikey=s3cret failed", "apikey=*** failed"),
        ("x-api-key: s3cret failed", "x-api-key: *** failed"),
        (
            "connect postgresql+psycopg://postgres:pw@127.0.0.1:5432/erudi failed",
            "connect postgresql://*** failed",
        ),
        ("could not open sqlite:////home/jane/app.db", "could not open sqlite://***"),
        # mixed
        (
            "open /Users/jane/m.gguf via https://u:p@h/x with Bearer tok123",
            "open m.gguf via https://***@h/x with Bearer ***",
        ),
    ],
)
def test_sanitisation_table(raw, expected):
    assert _detail(raw) == expected


def test_secrets_never_survive():
    secrets = ["hunter2", "s3cret", "tok123", "pw"]
    raw = (
        "https://bob:hunter2@h/ --api-key s3cret Bearer tok123 "
        "postgresql://postgres:pw@127.0.0.1/erudi"
    )
    out = _detail(raw)
    assert not any(secret in out for secret in secrets)


def test_one_line():
    assert _detail("first\n  second\r\n\tthird") == "first second third"


def test_unicode_survives():
    assert _detail("modele introuvable \u00e9\u00e8 \u6a21\u578b /tmp/\u6a21\u578b.gguf") == (
        "modele introuvable \u00e9\u00e8 \u6a21\u578b \u6a21\u578b.gguf"
    )


def test_truncated_after_sanitising():
    out = _detail("x" * 1000)
    assert len(out) == MAX_DETAIL_CHARS and out.endswith("...")
    # A secret straddling the cut is already masked: the cut never exposes a prefix.
    padded = "y" * (MAX_DETAIL_CHARS - 10) + " --api-key abcdefghijklmnop"
    out = _detail(padded)
    assert "abcdef" not in out and len(out) <= MAX_DETAIL_CHARS


def test_very_long_path_is_handled():
    out = _detail("/" + "a/" * 5000 + "end.gguf")
    assert out == "end.gguf"


def test_leaf_cause_through_explicit_chain():
    try:
        try:
            raise OSError("disk full at /var/data/db")
        except OSError as inner:
            raise RuntimeError("wrapper") from inner
    except RuntimeError as exc:
        assert error_detail(exc) == "disk full at db"


def test_leaf_cause_through_implicit_context():
    try:
        try:
            raise KeyError("slot")
        except KeyError:
            raise RuntimeError("while handling")
    except RuntimeError as exc:
        assert error_detail(exc) == "'slot'"


def test_suppressed_context_is_not_walked():
    try:
        try:
            raise KeyError("slot")
        except KeyError:
            raise RuntimeError("clean") from None
    except RuntimeError as exc:
        assert error_detail(exc) == "clean"


def test_empty_message_falls_back_to_the_type_name():
    assert error_detail(TimeoutError()) == "TimeoutError"
    try:
        try:
            raise TimeoutError()
        except TimeoutError as inner:
            raise RuntimeError("wrapper") from inner
    except RuntimeError as exc:
        assert error_detail(exc) == "TimeoutError"


def test_engine_exception_message_is_the_detail_and_stops_the_walk():
    try:
        try:
            raise OSError("raw child failure")
        except OSError as inner:
            raise EngineException(message="The model server stopped") from inner
    except EngineException as exc:
        assert error_detail(exc) == "The model server stopped"


def test_cyclic_chain_terminates():
    a, b = ValueError("a"), ValueError("b")
    a.__cause__, b.__cause__ = b, a
    assert error_detail(a) in {"a", "b"}
