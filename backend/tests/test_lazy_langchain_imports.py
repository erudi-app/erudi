"""Boot must not pay for deferred heavyweight stacks (issue #160, partial scope).

``import src.main`` (what uvicorn does before serving anything) must not pull:

1. The agent-layer LangChain packages — ``langchain`` (create_agent,
   middleware, tools) and ``langchain_openai`` (ChatOpenAI). Nothing needs
   them before the FIRST conversation/arena turn, so their imports are
   deferred to function scope in ``src.agents.{runner,model_factory,kb_mode}``.

2. The ingestion stack — ``langchain_text_splitters`` (it imports every
   splitter it ships at module level) and ``transformers``. They are reached
   at boot through seed → src.utils → kb_utils → src.ingestion.chunking, and
   are only needed at KB ingestion time, so ``chunking.py`` imports the
   splitters in function scope. ``transformers`` (AutoTokenizer) is likewise
   function-scoped in the engines.

Deliberately OUT of this guard (still boot-loaded, out of #160-partial scope):
  - ``langchain_core`` — cheap (~10 ms), pulled by ``src.ingestion.embeddings``
    for the ``Embeddings`` interface.
  - ``langchain_postgres`` — pulled by ``src.ingestion.vector_store``, whose
    ``init_kb_store`` runs IN the lifespan, so deferring gains nothing.
  - ``langgraph`` — the checkpointer is opened IN the lifespan, same reason.

The check runs in a subprocess so this test is immune to whatever the pytest
session itself has already imported.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_BACKEND = Path(__file__).resolve().parents[1]
_MARKER = "LOADED_MODULES_JSON:"

_PROBE = f"""
import json, sys
import src.main  # noqa: F401  (what uvicorn imports before serving)
print({_MARKER!r} + json.dumps(sorted(sys.modules)))
"""


def _boot_modules() -> list[str]:
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        cwd=_BACKEND,
        timeout=180,
    )
    assert proc.returncode == 0, f"import src.main failed:\n{proc.stderr}"
    payload = [ln for ln in proc.stdout.splitlines() if ln.startswith(_MARKER)]
    assert payload, f"probe marker line missing in stdout:\n{proc.stdout}"
    return json.loads(payload[-1][len(_MARKER) :])


def _offenders(modules: list[str], packages: tuple[str, ...]) -> list[str]:
    return [m for m in modules if any(m == p or m.startswith(p + ".") for p in packages)]


def test_import_src_main_does_not_load_agent_langchain_stack():
    offenders = _offenders(_boot_modules(), ("langchain", "langchain_openai"))
    assert offenders == [], (
        "boot (import src.main) must not load the agent LangChain stack; " f"loaded: {offenders}"
    )


def test_import_src_main_does_not_load_the_ingestion_stack():
    offenders = _offenders(
        _boot_modules(),
        (
            "langchain_text_splitters",
            "transformers",
        ),
    )
    assert offenders == [], (
        "boot (import src.main) must not load the ingestion stack "
        "(splitters/tokenizer are KB-ingestion-time only); "
        f"loaded: {offenders}"
    )
