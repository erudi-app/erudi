"""The arena phase reads the Arena's NDJSON stream: answer text only, a failed turn fails the phase."""

import json
from types import SimpleNamespace

import pytest

from erudi_eval.phases import phase_arena
from erudi_eval.runner import PhaseFailed


class _Api:
    def __init__(self, events):
        self.events = events
        self.paths = []

    def stream_lines(self, path, body, read_timeout=600.0):
        self.paths.append(path)
        yield 0.0, "\x00request-id:be-test"
        for i, event in enumerate(self.events, start=1):
            yield float(i), json.dumps(event) + "\n"


def _run(events):
    recorded = []
    run = SimpleNamespace(
        api=_Api(events),
        cfg=SimpleNamespace(turn_timeout=60),
        last_use_t=None,
        require_app=lambda: None,
        require_model=lambda: {"id": 3},
        event=lambda kind, **data: recorded.append((kind, data)),
        settle=lambda seconds: None,
    )
    return run, recorded


def test_counts_the_answer_text_only():
    run, recorded = _run([{"t": "answer", "text": "Photosynthesis "}, {"t": "answer", "text": "turns light into sugar."}])

    data = phase_arena(run)

    assert run.api.paths == ["/arena/3/query"]
    assert data["chars"] == len("Photosynthesis turns light into sugar.")
    assert recorded[0][0] == "arena_turn"


def test_a_failed_turn_fails_the_phase():
    run, _ = _run([{"t": "answer", "text": "Half"}, {"t": "error", "text": "[ERROR_MESSAGE_SYSTEM] I apologize.", "detail": "boom"}])

    with pytest.raises(PhaseFailed, match="arena turn failed"):
        phase_arena(run)


def test_a_stream_without_answer_text_fails_the_phase():
    run, _ = _run([])

    with pytest.raises(PhaseFailed, match="no text"):
        phase_arena(run)
