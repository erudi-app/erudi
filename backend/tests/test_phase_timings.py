"""Where an ingestion job's time goes: the thread-local phase tally."""

from __future__ import annotations

import threading

import pytest

from src.ingestion import chunking, phase_timings

pytestmark = pytest.mark.unit


def test_timed_blocks_add_up_by_phase_inside_a_tally():
    with phase_timings.collect() as tally:
        with phase_timings.timed("read"):
            pass
        phase_timings.add("read", 0.5)
        phase_timings.add("db", 0.25)
    assert tally["read"] >= 0.5
    assert tally["db"] == 0.25


def test_outside_a_tally_nothing_is_recorded():
    phase_timings.add("embed", 1.0)
    with phase_timings.timed("embed"):
        pass
    with phase_timings.collect() as tally:
        pass
    assert tally == {}


def test_a_block_that_raises_is_still_counted():
    with phase_timings.collect() as tally:
        with pytest.raises(ValueError):
            with phase_timings.timed("chunk"):
                raise ValueError("boom")
    assert "chunk" in tally


def test_another_threads_phases_never_reach_this_tally():
    """A search query beside a running job times its own embedding: the
    job's tally must not count it."""
    with phase_timings.collect() as tally:
        thread = threading.Thread(target=lambda: phase_timings.add("embed", 3.0))
        thread.start()
        thread.join()
    assert tally == {}


def test_the_tokenizer_load_is_a_phase_of_its_own():
    chunking.shutdown_tokenizer()
    try:
        with phase_timings.collect() as tally:
            chunking.count_tokens("Bonjour le monde")
        assert tally["tokenizer"] > 0.0
        # Already loaded: no second load, no second count.
        with phase_timings.collect() as again:
            chunking.count_tokens("Bonjour le monde")
        assert "tokenizer" not in again
    finally:
        chunking.shutdown_tokenizer()
