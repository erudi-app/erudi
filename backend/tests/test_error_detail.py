"""``error_detail``: the cleaned, one-line cause shown under a generic failure (#615)."""

import subprocess

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
        assert error_detail(exc) == "KeyError: 'slot'"


def test_suppressed_context_is_not_walked():
    try:
        try:
            raise KeyError("slot")
        except KeyError:
            raise RuntimeError("clean") from None
    except RuntimeError as exc:
        assert error_detail(exc) == "clean"


def test_a_bare_exception_falls_back_to_its_type_name():
    assert error_detail(TimeoutError()) == "TimeoutError"


def test_nothing_with_a_message_in_the_chain_names_the_outermost_type():
    try:
        try:
            raise TimeoutError()
        except TimeoutError as inner:
            raise RuntimeError() from inner
    except RuntimeError as exc:
        assert error_detail(exc) == "RuntimeError"


def test_an_empty_leaf_does_not_replace_a_meaningful_wrapper():
    # openai.APITimeoutError("Request timed out.") <- httpx.ReadTimeout("") <- TimeoutError()
    try:
        try:
            try:
                raise TimeoutError()
            except TimeoutError as deepest:
                raise ConnectionError("") from deepest
        except ConnectionError as middle:
            raise RuntimeError("Request timed out.") from middle
    except RuntimeError as exc:
        assert error_detail(exc) == "Request timed out."


def test_a_cancellation_leaf_does_not_replace_the_wrapper():
    import asyncio

    try:
        try:
            raise asyncio.CancelledError()
        except asyncio.CancelledError as inner:
            raise RuntimeError("Stream ended unexpectedly") from inner
    except RuntimeError as exc:
        assert error_detail(exc) == "Stream ended unexpectedly"


# --- real exception strings (the form a path actually takes in a message) ---

_WIN_MODEL = r"C:\Users\John Doe\AppData\Roaming\Erudi\models\x.gguf"
_POSIX_MODEL = "/Users/John Doe/Library/Application Support/Erudi/models/x.gguf"


@pytest.mark.parametrize("model", [_WIN_MODEL, _POSIX_MODEL])
def test_a_file_not_found_never_leaks_the_user_name(model):
    out = error_detail(FileNotFoundError(2, "No such file or directory", model))
    assert out == "[Errno 2] No such file or directory: 'x.gguf'"


@pytest.mark.parametrize(
    "path", [r"C:\Users\jdoe\AppData\Local\Temp\x", "/home/jdoe/.cache/erudi/x"]
)
def test_a_permission_error_never_leaks_the_user_name(path):
    out = error_detail(PermissionError(13, "Permission denied", path))
    assert out == "[Errno 13] Permission denied: 'x'"


@pytest.mark.parametrize(
    ("binary", "shown"),
    [
        (r"C:\Users\john\llama-server.exe", "llama-server.exe"),
        ("/home/john/llama-server", "llama-server"),
    ],
)
def test_a_failed_command_masks_the_key_and_the_user_name(binary, shown):
    out = error_detail(subprocess.CalledProcessError(1, [binary, "--api-key", "k3y"]))
    assert out == f"Command '['{shown}', '--api-key', '***']' returned non-zero exit status 1."


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("OPENAI_API_KEY=sk-1 failed", "OPENAI_API_KEY=*** failed"),
        ("password=hunter2 refused", "password=*** refused"),
        ("PGPASSWORD=pw host=h", "PGPASSWORD=*** host=h"),
        ("GET /x?token=abcdefghij1234&y=1", "GET /x?token=***&y=1"),
        ("access_token=abc secret=def", "access_token=*** secret=***"),
        ("bad hf_" + "a" * 34 + " given", "bad hf_*** given"),
        ("Authorization: Basic dXNlcjpwdw== sent", "Authorization: Basic *** sent"),
        ("headers {'authorization': 'abc def'}", "headers {'authorization': '***'}"),
        ("GET https://bob:p@ss@host.example/x", "GET https://***@host.example/x"),
        # prose keeps its words
        (
            "Failed to open /Users/rayan/models/a.gguf and/or fallback 3/4 shards",
            "Failed to open a.gguf and/or fallback 3/4 shards",
        ),
        ("HTTP 404 /v1/chat/completions not found", "HTTP 404 /v1/chat/completions not found"),
        # a home directory alone is never the user name
        ("in '/home/john'", "in '~'"),
        ("in /Users/john/ now", "in ~ now"),
        ("in C:\\Users\\john now", "in ~ now"),
        ("cwd:/Users/john/work/x.gguf", "cwd:x.gguf"),
    ],
)
def test_more_secrets_and_path_shapes(raw, expected):
    assert _detail(raw) == expected


def test_adversarial_input_stays_fast():
    import time

    start = time.perf_counter()
    for payload in ("a-" * 150_000, "/ " * 100_000, "\\\\" * 100_000, "C:\\" + "a b\\" * 50_000):
        _detail(payload)
    assert time.perf_counter() - start < 5


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


_TABLE_R2 = [
    # Windows paths do not swallow the prose after them
    (
        r"Failed to open C:\Users\jane\a.gguf and/or fallback 3/4 shards",
        "Failed to open a.gguf and/or fallback 3/4 shards",
    ),
    (
        r"cannot load C:\Users\jane\a.gguf (use the CPU/GPU toggle)",
        "cannot load a.gguf (use the CPU/GPU toggle)",
    ),
    (
        r"Failed to open C:\Users\jane\a.gguf, see docs/errors.md for help",
        "Failed to open a.gguf, see docs/errors.md for help",
    ),
    (r"C:\Users\O'Brien\x.gguf", "x.gguf"),
    (r"C:\Users\Jane Doe\x.gguf", "x.gguf"),
    (r"C:\Users/jane\x.gguf", "x.gguf"),
    (r"C:/Users/Jane Doe/models/x.gguf", "x.gguf"),
    (r"C:\Users\Jane Doe\OneDrive - Contoso Ltd\x.gguf", "x.gguf"),
    # file:// URLs reduce like any path
    ("file:///Users/jane", "~"),
    ("file:///home/jane", "~"),
    ("file:///C:/Users/jane/", "~"),
    ("file:///Users/Jane%20Doe/x.gguf", "x.gguf"),
    (
        "cannot load file:///Users/Jane Doe/Library/Application Support/erudi/x.gguf",
        "cannot load x.gguf",
    ),
    # tokenizer diagnostics stay readable, secret-like tokens do not
    (
        "special token: <|im_end|> not found in vocab",
        "special token: <|im_end|> not found in vocab",
    ),
    (
        "chat template error: eos_token='</s>' is missing",
        "chat template error: eos_token='</s>' is missing",
    ),
    ("Unexpected token: '<' in JSON at position 0", "Unexpected token: '<' in JSON at position 0"),
    ("token=abcdefghijkl1234 sent", "token=*** sent"),
    # bare sk- keys, driveless and encoded paths
    (
        "Incorrect API key provided: sk-abcDEF1234567890secret.",
        "Incorrect API key provided: sk-***.",
    ),
    (r"opening \Users\jane\x.gguf", "opening x.gguf"),
    ("opening %2FUsers%2Fjane%2Fx.gguf", "opening x.gguf"),
]


@pytest.mark.parametrize(("raw", "expected"), _TABLE_R2)
def test_second_round_table(raw, expected):
    assert _detail(raw) == expected


def test_the_argv_of_a_failed_windows_command_keeps_the_binary_and_flags():
    out = error_detail(
        subprocess.CalledProcessError(
            1, [r"C:\Program Files\Erudi\llama-server.exe", "-m", r"models\x.gguf"]
        )
    )
    assert (
        out
        == "Command '['llama-server.exe', '-m', 'models\\\\x.gguf']' returned non-zero exit status 1."
    )


def test_the_input_is_cut_before_it_is_sanitised():
    import time

    start = time.perf_counter()
    out = _detail("x-" * 2_000_000)
    assert time.perf_counter() - start < 2
    assert len(out) <= MAX_DETAIL_CHARS


def test_the_generic_apology_is_the_text_the_frontend_matches():
    """The renderer recognises a generic failure by this exact line
    (frontend/src/utils/messageContent.js, GENERIC_ERROR_LINE): rewording
    ERROR_MESSAGE without the constant there stops the headline translation."""
    from pathlib import Path

    from src.agents.runner import ERROR_MESSAGE, ERROR_SENTINEL

    frontend = Path(__file__).resolve().parents[2] / "frontend" / "src" / "utils"
    source = (frontend / "messageContent.js").read_text(encoding="utf-8")
    # The line is written there as `${ERROR_SENTINEL} <apology>`.
    assert f'export const ERROR_SENTINEL = "{ERROR_SENTINEL}";' in source
    apology = ERROR_MESSAGE.removeprefix(f"{ERROR_SENTINEL} ")
    assert f"`${{ERROR_SENTINEL}} {apology}`" in source


def test_a_failing_str_never_raises():
    class Unprintable(Exception):
        def __str__(self):
            raise RuntimeError("no str")

    assert error_detail(Unprintable()) == "Unprintable"


def test_a_quoted_key_names_its_exception_type():
    assert error_detail(KeyError("choices")) == "KeyError: 'choices'"
    assert error_detail(KeyError("")) == "KeyError"
    try:
        try:
            raise KeyError("")
        except KeyError as inner:
            raise RuntimeError("meaningful wrapper") from inner
    except RuntimeError as exc:
        assert error_detail(exc) == "meaningful wrapper"
