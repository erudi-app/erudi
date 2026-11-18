"""``src.utils`` package init must stay boot-cheap (issue #160 P1).

Importing ANY ``src.utils.*`` submodule runs ``src/utils/__init__.py``. The seed
does exactly that at boot (``from src.utils.hf_model_metadata import ...``). The
package used to eagerly re-export from ``kb_utils``, which pulls the ingestion
import tree (``src.ingestion.{chunking,vector_store}`` -> ``langchain_postgres``,
~3.4 s measured, ~5-6 s wall on this path) for a caller that only wanted the
~4 ms metadata helpers -- structurally bypassing the #198 lazy-loading work.

The re-exports are now resolved lazily via PEP 562 ``__getattr__``, so:

1. A metadata-only consumer of ``src.utils`` must NOT drag the KB / ingestion /
   ML stack into ``sys.modules`` (the regression this PR fixes).
2. The convenient ``from src.utils import <name>`` surface must still work, and
   a KB name must still resolve (lazily, pulling ``kb_utils`` only then).

The stack-drag check runs in a subprocess so it is immune to whatever the pytest
session has already imported. NB: this guards the *package-init* boot cost only;
``src.ingestion.chunking`` legitimately loads at app boot through the
conversations/arena routers and ``core.api``'s ``init_kb_store`` -- that path is
out of scope here and covered by ``test_lazy_langchain_imports`` for the heavy
ML libraries.
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_BACKEND = Path(__file__).resolve().parents[1]

# Nothing a metadata-only ``src.utils`` consumer should ever pull at import time.
_FORBIDDEN = (
    "src.utils.kb_utils",
    "src.ingestion.chunking",
    "src.ingestion.vector_store",
    "langchain_postgres",
    "tokenizers",
    "transformers",
)

_PROBE = f"""
import json, sys
# What the seed imports at boot -- must resolve WITHOUT the KB/ingestion tree.
from src.utils.hf_model_metadata import format_model_info_metadata  # noqa: F401
forbidden = {list(_FORBIDDEN)!r}
loaded = [m for m in forbidden if m in sys.modules]
print("DRAGGED_JSON:" + json.dumps(loaded))
"""


def test_seed_metadata_import_does_not_drag_kb_stack():
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        cwd=_BACKEND,
        timeout=180,
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stderr}"
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("DRAGGED_JSON:")]
    assert line, f"probe marker missing:\n{proc.stdout}"
    import json

    dragged = json.loads(line[-1][len("DRAGGED_JSON:") :])
    assert dragged == [], (
        "importing src.utils.hf_model_metadata (the seed's boot import) must not "
        f"pull the KB/ingestion/ML stack; dragged: {dragged}"
    )


def test_lazy_reexports_still_resolve():
    """The PEP 562 surface keeps working for every documented name."""
    import src.utils as u

    # Metadata + prompt names resolve without the KB tree.
    assert callable(u.format_model_info_metadata)
    assert callable(u.build_system_prompt)
    # A KB name still resolves (pulls kb_utils lazily, on this access).
    assert u.KbExcerpt is not None
    assert callable(u.retrieve_kb_excerpts)
    # __all__ advertises exactly the documented surface.
    assert set(u.__all__) == {
        "build_system_prompt",
        "get_prompting_strategy",
        "KbExcerpt",
        "retrieve_kb_excerpts",
        "get_disk_size_after_quant",
        "get_parameter_count_from_name",
        "format_model_info_metadata",
    }


def test_unknown_attribute_raises_attribute_error():
    import src.utils as u

    with pytest.raises(AttributeError):
        u.does_not_exist
