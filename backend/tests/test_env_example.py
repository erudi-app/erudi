"""Keep ``backend/.env.example`` in sync with the environment variables the
backend actually reads (#409).

The example file is the single place a contributor looks to learn which
variables exist. Two things must hold for it to stay trustworthy:

* every variable read from the environment in ``src/``, ``run.py`` and the
  Alembic env is listed there (a new ``os.getenv("X")`` without a matching
  line fails here, not in a user's setup), and
* it never carries a value -- it is a template, not a config file, and the
  only secret the backend knows (``HF_TOKEN``) must not land in git by way of
  a "harmless" example.

The scan is AST-based so docstrings and comments that merely mention a
variable (e.g. the ``DATABASE_URL`` example in ``ConfigurationException``)
are not mistaken for real reads.
"""

import ast
import re
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
ENV_EXAMPLE = BACKEND / ".env.example"

_SCANNED_FILES = [BACKEND / "run.py", BACKEND / "alembic" / "env.py"] + sorted(
    (BACKEND / "src").rglob("*.py")
)

_LINE = re.compile(r"^([A-Z][A-Z0-9_]*)=$")


def _is_os_environ(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _first_str_arg(call: ast.Call) -> str | None:
    if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
        return call.args[0].value
    return None


def _env_reads(tree: ast.AST) -> set[str]:
    """Names read via os.getenv / os.environ.get / os.environ[...] / `in os.environ`."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            func = node.func
            is_getenv = (
                func.attr == "getenv" and isinstance(func.value, ast.Name) and func.value.id == "os"
            )
            is_environ_get = func.attr == "get" and _is_os_environ(func.value)
            if is_getenv or is_environ_get:
                name = _first_str_arg(node)
                if name:
                    names.add(name)
        elif isinstance(node, ast.Subscript) and _is_os_environ(node.value):
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                names.add(node.slice.value)
        elif isinstance(node, ast.Compare) and any(_is_os_environ(c) for c in node.comparators):
            if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
                names.add(node.left.value)
    return names


def _documented_names() -> set[str]:
    names: set[str] = set()
    for raw in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE.match(line)
        assert match, f".env.example line must be `NAME=` with no value, got: {line!r}"
        names.add(match.group(1))
    return names


@pytest.mark.unit
def test_env_example_exists():
    assert ENV_EXAMPLE.is_file(), "backend/.env.example is missing (#409)"


@pytest.mark.unit
def test_env_example_carries_no_values():
    # Every non-comment line is `NAME=`; _documented_names asserts the shape.
    assert _documented_names()


@pytest.mark.unit
def test_every_env_read_is_documented():
    read: dict[str, set[str]] = {}
    for path in _SCANNED_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for name in _env_reads(tree):
            read.setdefault(name, set()).add(str(path.relative_to(BACKEND)))

    assert read, "the scan found no environment reads at all -- is the AST matcher broken?"
    assert {"HF_TOKEN", "ERUDI_FORCE_CPU", "ERUDI_LOG_LEVEL"} <= read.keys()

    missing = {
        name: sorted(files) for name, files in read.items() if name not in _documented_names()
    }
    assert (
        not missing
    ), f"environment variables read but absent from backend/.env.example: {missing}"


@pytest.mark.unit
def test_the_qa_seams_are_documented():
    # QA seams are read rarely and set by hand: each has its line saying what
    # it forces and on which engine (ERUDI_FORCE_MEMORY_PRESSURE's behaviour
    # is tested in test_memory_pressure.py).
    assert {"ERUDI_MEMORY_PRIOR_SCALE", "ERUDI_FORCE_MEMORY_PRESSURE"} <= _documented_names()


@pytest.mark.unit
def test_scan_matches_the_supported_read_forms():
    tree = ast.parse(
        "import os\n"
        "a = os.getenv('A')\n"
        "b = os.getenv('B', 'default')\n"
        "c = os.environ.get('C')\n"
        "d = os.environ['D']\n"
        "e = 'E' in os.environ\n"
        "os.environ.setdefault('NOT_A_READ', '1')\n"
        "'''docstring: os.getenv(\"DOCSTRING\")'''\n"
    )
    assert _env_reads(tree) == {"A", "B", "C", "D", "E"}


@pytest.mark.unit
def test_secrets_scaffold_is_gone():
    # The `src.config.secrets` module was a never-imported scaffold whose
    # docstring promised a token "embedded in the binary". HF_TOKEN is read
    # from the environment at runtime and must never be baked into a build.
    assert not (BACKEND / "src" / "config").exists()
    for spec in BACKEND.glob("*.spec"):
        assert "src.config.secrets" not in spec.read_text(encoding="utf-8"), spec.name
