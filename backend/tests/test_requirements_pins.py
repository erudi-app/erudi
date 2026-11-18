"""Dependency-pin guard for the mlx-vlm swap bumps.

The swap moves the MLX engine onto ``mlx-vlm``, which forces ``numpy`` to
2.x and therefore ``py3langid`` to 0.3.x (0.3.0 declares ``numpy>=2``).
The ``numpy`` pin lives in the shared ``base.txt``, so the bump reaches
*every* platform's CI, not just macOS. These tests run on the CPU/Ubuntu
job and are the safety net: they fail loudly if the common stack is not
importable under numpy 2, or if the language-detection contract that
``src.agents.language`` relies on regresses.
"""

import pytest


@pytest.mark.unit
def test_numpy_is_2x():
    import numpy as np

    assert int(np.__version__.split(".")[0]) >= 2, np.__version__


@pytest.mark.unit
def test_py3langid_contract_under_numpy2():
    # The exact import + call path src.agents.language depends on; 0.3.x must
    # keep it (it does) so language.py stays untouched.
    from py3langid.langid import MODEL_FILE, LanguageIdentifier

    identifier = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)
    language, probability = identifier.classify("This is an English sentence about cats and dogs.")
    assert language == "en"
    assert 0.0 <= float(probability) <= 1.0


@pytest.mark.unit
def test_detect_language_still_works_under_numpy2():
    from src.agents.language import detect_language

    assert detect_language("Ceci est une phrase clairement rédigée en français.") == "fr"
    assert detect_language("This sentence is unambiguously written in English.") == "en"


@pytest.mark.unit
def test_common_stack_imports_under_numpy2():
    # Deps present on EVERY platform (base.txt). Excludes datasets/mlx_vlm
    # (macOS/CUDA-only pins) so this stays valid on the CPU/Ubuntu CI job.
    import langchain  # noqa: F401
    import langchain_postgres  # noqa: F401
    import pgserver  # noqa: F401
    import psycopg  # noqa: F401
    import jinja2  # noqa: F401
    import transformers  # noqa: F401
