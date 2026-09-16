# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "psutil>=5.9",
#     "websocket-client>=1.6",
#     "nvidia-ml-py>=12; sys_platform != 'darwin'",
# ]
# ///
"""erudi-eval: measure memory and storage of the installed Erudi app, phase by phase.

    uv run erudi_eval.py selftest
    uv run erudi_eval.py run [options]
    uv run erudi_eval.py compare results/<run-a> results/<run-b>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from erudi_eval.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
