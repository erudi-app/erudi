"""Streaming ``<think>...</think>`` splitter -- the FALLBACK reasoning path (#554).

The primary reasoning path does not run through here: both local inference
servers extract each family's chain-of-thought server-side (llama-server's
default ``--reasoning-format auto``, mlx_vlm.server's native split) into a
dedicated delta field that ``Erudi_Chat_OpenAI`` carries to the runner's
``thinking`` events. This splitter stays on the CONTENT channel as the
fallback for families whose markers the server parser does not know: inline
``<think>`` tags that slip through still become ``thinking`` events instead of
leaking into the answer. Arena's plain-text projection (``emit_events=False``)
depends on it -- it is what keeps stray inline reasoning off Arena's wire. It
is fed the model's answer deltas and emits event dicts: text outside the tags
becomes ``answer`` events, text inside becomes ``thinking`` events.

Robust to tags split across chunks (``<th`` then ``ink>``): only a trailing
partial prefix of the tag currently being scanned for is buffered; everything
before it is emitted immediately. ``flush()`` drains the buffer at stream end (an
unclosed ``<think>`` flushes its remainder as thinking).

Pure string logic, no LangChain import -- unit-tested with adversarial splits.
"""

from __future__ import annotations

OPEN_TAG = "<think>"
CLOSE_TAG = "</think>"


def _partial_suffix_len(buf: str, tag: str) -> int:
    """Length of the longest suffix of ``buf`` that is a proper prefix of ``tag``.

    That tail is exactly what must be held back: it *might* complete into ``tag``
    on the next chunk. Callers scan for a full occurrence first, so ``tag`` is
    never wholly contained here and the result is always ``< len(tag)``. Returns
    ``0`` when no suffix of ``buf`` starts ``tag``.
    """
    max_len = min(len(buf), len(tag) - 1)
    for k in range(max_len, 0, -1):
        if buf.endswith(tag[:k]):
            return k
    return 0


class ThinkSplitter:
    """Stateful streaming splitter.

    Call :meth:`feed` per answer delta, then :meth:`flush` once at stream end.
    Both return a list of ``{"t": "answer"|"thinking", "text": ...}`` event dicts
    (empty-text events are never produced).
    """

    def __init__(self) -> None:
        self._inside = False
        self._buf = ""

    def _label(self) -> str:
        return "thinking" if self._inside else "answer"

    def feed(self, text: str) -> list[dict]:
        """Consume a delta; return the events it completes (may be empty)."""
        if not text:
            return []
        self._buf += text
        events: list[dict] = []
        while self._buf:
            target = CLOSE_TAG if self._inside else OPEN_TAG
            idx = self._buf.find(target)
            if idx != -1:
                before = self._buf[:idx]
                if before:
                    events.append({"t": self._label(), "text": before})
                # Consume the tag and flip mode; keep scanning the remainder.
                self._buf = self._buf[idx + len(target) :]
                self._inside = not self._inside
                continue
            # No complete tag: emit everything except a trailing partial tag that
            # could still complete on the next feed().
            keep = _partial_suffix_len(self._buf, target)
            emit_upto = len(self._buf) - keep
            if emit_upto > 0:
                events.append({"t": self._label(), "text": self._buf[:emit_upto]})
                self._buf = self._buf[emit_upto:]
            break
        return events

    def flush(self) -> list[dict]:
        """Drain the buffer in the current mode at stream end.

        A leftover partial tag flushes as literal text; an unclosed ``<think>``
        flushes its remainder as thinking (never leaks into an answer event).
        """
        if not self._buf:
            return []
        event = {"t": self._label(), "text": self._buf}
        self._buf = ""
        return [event]
