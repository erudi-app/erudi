"""Backend memory at rest (#621): what boot must not load, and the GC freeze.

The document parsers (and, with them, lxml) stay out of ``sys.modules`` until
a document is read; ``gc.freeze()`` runs once, when startup is over.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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


async def test_gc_freeze_runs_once_after_startup(monkeypatch):
    import gc

    from fastapi import FastAPI

    from src.core import api, config

    calls: list[str] = []

    class _Engine:
        @staticmethod
        def start_cleanup_task():
            pass

        @staticmethod
        def start_pressure_task():
            pass

        @staticmethod
        def set_idle_unload_minutes(minutes):
            pass

        @staticmethod
        def stop_cleanup_task():
            pass

        @staticmethod
        async def stop_pressure_task():
            pass

        @staticmethod
        def cleanup():
            pass

    class _Checkpointer:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *exc):
            return False

    handle = SimpleNamespace(sqlalchemy_url="x", psycopg_url="y")
    monkeypatch.setattr(api, "start_postgres", lambda *_: handle)
    monkeypatch.setattr(api, "stop_postgres", lambda *_: None)
    monkeypatch.setattr(api, "init_database", lambda *_: None)
    monkeypatch.setattr(api, "run_migrations", lambda *_: None)
    monkeypatch.setattr(api, "init_kb_store", lambda *_: None)
    monkeypatch.setattr(api, "close_kb_store", lambda *_a, **_k: None)
    monkeypatch.setattr(api, "open_checkpointer", lambda *_: _Checkpointer())
    monkeypatch.setattr(api, "start_watchdog", lambda *_: calls.append("watchdog"))
    monkeypatch.setattr(api, "stop_watchdog", lambda *_: None)
    monkeypatch.setattr(api, "backfill_wire_tools_startup", lambda: None)

    async def _populate():
        calls.append("catalog")
        return {}

    monkeypatch.setattr(api, "startup_populate_database", _populate)
    monkeypatch.setattr(api, "BaseEngine", SimpleNamespace(get_engine=lambda: _Engine))
    monkeypatch.setattr(config, "LLM_Engine", None, raising=False)
    monkeypatch.setattr(gc, "freeze", lambda: calls.append("freeze"))

    async with api.lifespan(FastAPI()):
        assert calls.count("freeze") == 1
        assert calls.index("catalog") < calls.index("watchdog") < calls.index("freeze")

    assert calls.count("freeze") == 1
