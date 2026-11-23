"""Where one ingestion job spends its time, phase by phase.

An ingestion job runs on one thread: `collect` opens a tally on it, and every
`timed` block or `add` call made on that thread while the tally is open adds
its seconds to the phase it names (``read``, ``chunk``, ``tokenizer``,
``embed``, ``db``). Outside a tally (a search query, a test) they cost one
attribute read and record nothing. The job's closing INFO line
(``KB job N timings: ...``) is written from the tally, so the next run on a
user's machine tells where a slow job went.

Thread-local on purpose: the code that measures (``chunking``,
``vector_store``) is shared with the retrieval path, which must not add to a
job's tally running beside it.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Iterator

_local = threading.local()


@contextlib.contextmanager
def collect() -> Iterator[dict[str, float]]:
    """Tally the phases timed on this thread for the length of the block.

    Yields the tally: phase name to seconds, summed over the block."""
    tally: dict[str, float] = {}
    previous = getattr(_local, "tally", None)
    _local.tally = tally
    try:
        yield tally
    finally:
        _local.tally = previous


def add(phase: str, seconds: float) -> None:
    """Add ``seconds`` to ``phase`` in this thread's open tally, if any."""
    tally = getattr(_local, "tally", None)
    if tally is not None:
        tally[phase] = tally.get(phase, 0.0) + seconds


@contextlib.contextmanager
def timed(phase: str) -> Iterator[None]:
    """Add the block's duration to ``phase`` (raised or not)."""
    start = time.perf_counter()
    try:
        yield
    finally:
        add(phase, time.perf_counter() - start)
