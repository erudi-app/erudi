"""transformers' "PyTorch was not found" advisory stays out of the app's logs.

The app ships no torch, and transformers prints that advisory on the stderr
of every process importing it: the backend's (captured into
erudi-backend.log) and the MLX children's (their log files). ``src``
silences advisory warnings, and only those, before anything imports it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_BACKEND = Path(__file__).resolve().parents[1]


def test_importing_src_silences_transformers_advisories():
    import src  # noqa: F401

    assert os.environ.get("TRANSFORMERS_NO_ADVISORY_WARNINGS") == "1"


def test_a_process_importing_transformers_through_src_prints_no_advisory():
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRANSFORMERS_")}
    result = subprocess.run(
        [sys.executable, "-c", "import src; import transformers"],
        cwd=_BACKEND,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "PyTorch was not found" not in result.stderr


def test_real_transformers_warnings_still_print():
    # Only the advisory channel is muted: transformers' own logger keeps its
    # WARNING level.
    env = {k: v for k, v in os.environ.items() if not k.startswith("TRANSFORMERS_")}
    code = (
        "import src\n"
        "from transformers.utils import logging\n"
        "logging.get_logger('transformers').warning('a real warning')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_BACKEND,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "a real warning" in result.stderr
