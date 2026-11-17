"""Backend memory at rest (#621): what boot must not load.

The document parsers (and, with them, lxml) stay out of ``sys.modules`` until
a document is read.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]

PARSER_MODULES = ["pypdf", "docx", "openpyxl", "lxml"]


def _run_isolated(code: str) -> str:
    """Run ``code`` in a fresh interpreter and return the last stdout line.

    A subprocess, so nothing a previous test imported can leak into the answer.
    """
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=BACKEND_DIR,
        env={"PYTHONPATH": str(BACKEND_DIR), "PATH": "", "SYSTEMROOT": "C:\\Windows"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()[-1]


def _loaded_after(import_statement: str, modules: list[str]) -> dict[str, bool]:
    code = (
        "import json, sys\n"
        f"{import_statement}\n"
        f"print(json.dumps({{m: m in sys.modules for m in {modules!r}}}))\n"
    )
    return json.loads(_run_isolated(code))


def test_importing_the_app_does_not_load_the_document_parsers():
    loaded = _loaded_after("from src.main import app", PARSER_MODULES)
    assert loaded == dict.fromkeys(PARSER_MODULES, False)


def test_importing_the_reader_does_not_load_the_document_parsers():
    loaded = _loaded_after("import src.ingestion.reader", PARSER_MODULES)
    assert loaded == dict.fromkeys(PARSER_MODULES, False)


@pytest.mark.parametrize(
    "extension, module",
    [(".pdf", "pypdf"), (".docx", "docx"), (".xlsx", "openpyxl")],
)
def test_a_parser_loads_on_the_first_document_of_its_kind(extension, module):
    code = (
        "import sys, tempfile, pathlib\n"
        "from src.ingestion.reader import DocumentReader\n"
        f"assert {module!r} not in sys.modules\n"
        f"p = pathlib.Path(tempfile.mkdtemp()) / 'x{extension}'\n"
        "p.write_bytes(b'not a real document')\n"
        "try:\n"
        "    DocumentReader().read(p)\n"
        "except Exception:\n"
        "    pass\n"
        f"print({module!r} in sys.modules)\n"
    )
    assert _run_isolated(code) == "True"
