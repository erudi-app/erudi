"""The work in flight on each conversation, and the deletes that wait for it.

A turn (``ConversationService.query_and_respond_stream``), a title
(``generate_title_stream``) and a manual compaction (``compact_conversation``)
write a conversation's rows and its checkpointer thread. Some of those writes
happen after the generation lock is released: the ``finally`` of a turn
persists its answer once the runner has let the model go, and a client that
left mid-answer (an ephemeral conversation the user just left) reaches that
``finally`` only when the server has seen the disconnect -- possibly after
the DELETE that followed the abort has arrived.

So each of them is registered here from its first statement to its last, and
``ConversationService.delete_conversation`` closes the conversation before it
deletes anything:

* ``closing`` marks the conversation as closing, then waits for every
  registered piece of work on it to end;
* while it is closing, ``begin`` refuses new work on it (``None``), which then
  writes nothing;
* the delete runs inside ``closing``, so once it returns every write that
  could have touched the conversation is behind it, and the next one finds no
  row.

One registry per process, on the event loop: no lock is needed, since
``begin`` and the start of ``closing`` run without awaiting. Each turn's
``asyncio.Event`` is created by the turn itself, on the loop that runs it.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import AsyncIterator, Dict, Optional, Set

from src.core.logging import logger


class ConversationTurns:
    """Registry of the work in flight per conversation id."""

    def __init__(self) -> None:
        self._open: Dict[int, Set[asyncio.Event]] = {}
        # Conversations a delete is closing, with the number of deletes
        # closing each (two DELETEs of one id may overlap).
        self._closing: Dict[int, int] = {}

    def begin(self, conversation_id: int) -> Optional[asyncio.Event]:
        """Register work on the conversation; ``None`` while it is closing."""
        if conversation_id in self._closing:
            return None
        turn = asyncio.Event()
        self._open.setdefault(conversation_id, set()).add(turn)
        return turn

    def end(self, conversation_id: int, turn: asyncio.Event) -> None:
        """The work registered by ``begin`` is over (idempotent)."""
        turn.set()
        open_turns = self._open.get(conversation_id)
        if open_turns is None:
            return
        open_turns.discard(turn)
        if not open_turns:
            del self._open[conversation_id]

    @contextlib.asynccontextmanager
    async def closing(self, conversation_id: int) -> AsyncIterator[None]:
        """Refuse new work on the conversation and wait for the work in
        flight to end; the body runs once nothing writes to it any more."""
        self._closing[conversation_id] = self._closing.get(conversation_id, 0) + 1
        try:
            in_flight = list(self._open.get(conversation_id, ()))
            if in_flight:
                logger.info(
                    f"Conversation {conversation_id}: waiting for {len(in_flight)} "
                    f"turn(s) in flight to end before deleting it"
                )
                for turn in in_flight:
                    await turn.wait()
            yield
        finally:
            remaining = self._closing[conversation_id] - 1
            if remaining:
                self._closing[conversation_id] = remaining
            else:
                del self._closing[conversation_id]


# The process-wide registry the conversation service uses.
conversation_turns = ConversationTurns()
