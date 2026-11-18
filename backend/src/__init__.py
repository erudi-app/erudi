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
