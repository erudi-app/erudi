"""Erudi Backend Package.

This module sets up environment variables required for thread-safe operation
of numerical libraries (NumPy and the BLAS it links) before any heavy imports occur.

CRITICAL: These must be set BEFORE any import of numpy.
Python's import system guarantees __init__.py runs first when importing from this package.
"""

import os
import sys

# macOS-specific: Prevent OpenMP conflicts with Accelerate/vecLib
# Must use single thread for BLAS stability on macOS
if sys.platform == "darwin":
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # Accelerate/vecLib (macOS)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# The app ships no torch: transformers serves it tokenizers, processors and
# configs, and says so with an advisory line ("PyTorch was not found. Models
# won't be available ...") on the stderr of every process that imports it,
# the backend's included (erudi-backend.log). Advisory warnings only:
# transformers' real warnings still print. (mlx_vlm sets the same variable,
# and TRANSFORMERS_VERBOSITY=error, in the processes that import it: the MLX
# children.)
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
