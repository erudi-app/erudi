"""The sweeps of ephemeral conversations (#428): at startup and at shutdown.

The renderer deletes an ephemeral conversation when the user leaves it. What
it cannot delete -- the app quit with the conversation open, a crash, a
force-quit, a backend killed -- is deleted here:

* at shutdown (``purge_at_shutdown``), in the lifespan before the
  checkpointer closes, bounded by ``SHUTDOWN_PURGE_SECONDS``. Best effort:
  uvicorn first waits for the requests in flight and Electron kills the
  backend 8 s after asking it to stop, so this may not run or not finish;
* at the next start (``sweep_at_start``), in ``_start_services`` once the
  checkpointer is open: every ephemeral conversation left over, then every
  checkpointer thread no conversation owns. A delete that stopped between the
  rows and the thread leaves such a thread, and once its row is gone nothing
  tells whether the conversation was ephemeral; a thread without its
  conversation is never read again and would be handed to a later
  conversation that got the same id, so every one is deleted.

Each conversation goes through ``ConversationService.delete_conversation``
(rows, thread, pasted images, after any turn in flight). One INFO line per
sweep, ASCII only.
"""

from __future__ import annotations

import asyncio
from typing import List

from fastapi.concurrency import run_in_threadpool
from sqlalchemy import text

from src.core.exceptions import ConversationNotFoundException
from src.core.logging import logger
from src.database.core import SessionLocal
from src.domains.conversations.services import ConversationService

# How long the shutdown waits for the purge before it closes the
# checkpointer; what is left is deleted at the next start.
SHUTDOWN_PURGE_SECONDS = 10.0

# The thread ids LangGraph's tables hold that are a conversation's id (every
# thread of the app is one, ``thread_id=str(conversation_id)``) and match no
# conversation row. The three tables are the ones ``adelete_thread`` clears.
_ORPHAN_THREADS_SQL = text(
    """
    SELECT t.thread_id FROM (
        SELECT thread_id FROM checkpoints
        UNION SELECT thread_id FROM checkpoint_writes
        UNION SELECT thread_id FROM checkpoint_blobs
    ) AS t
    WHERE t.thread_id ~ '^[0-9]+$'
      AND NOT EXISTS (SELECT 1 FROM conversations c WHERE c.id::text = t.thread_id)
    ORDER BY t.thread_id
    """
)


async def purge_ephemeral_conversations(db, checkpointer) -> int:
    """Delete every ephemeral conversation; return how many were deleted."""
    service = ConversationService(db, checkpointer)
    ids = await run_in_threadpool(service.conversation_repo.get_ephemeral_conversation_ids)
    purged = 0
    for conversation_id in ids:
        try:
            await service.delete_conversation(conversation_id)
            purged += 1
        except ConversationNotFoundException:
            # The renderer's own delete of it got there first.
            continue
    return purged


def _orphan_thread_ids(db) -> List[str]:
    return [row[0] for row in db.execute(_ORPHAN_THREADS_SQL)]


async def purge_orphan_threads(db, checkpointer) -> int:
    """Delete every checkpointer thread whose conversation is gone; return
    how many were deleted."""
    thread_ids = await run_in_threadpool(_orphan_thread_ids, db)
    for thread_id in thread_ids:
        await checkpointer.adelete_thread(thread_id)
    return len(thread_ids)


async def sweep_at_start(checkpointer) -> None:
    """The startup sweep: ephemeral conversations left over, then orphan
    threads. Raises on failure (the lifespan logs it and boots on)."""
    db = SessionLocal()
    try:
        purged = await purge_ephemeral_conversations(db, checkpointer)
        logger.info(f"Ephemeral conversations purged at startup: {purged}")
        orphans = await purge_orphan_threads(db, checkpointer)
        logger.info(f"Orphan conversation threads purged at startup: {orphans}")
    finally:
        db.close()


async def purge_at_shutdown(checkpointer) -> None:
    """The shutdown purge, bounded by ``SHUTDOWN_PURGE_SECONDS``; what it
    could not delete in time is left to the next start's sweep."""
    db = SessionLocal()
    try:
        purged = await asyncio.wait_for(
            purge_ephemeral_conversations(db, checkpointer), SHUTDOWN_PURGE_SECONDS
        )
        logger.info(f"Ephemeral conversations purged at shutdown: {purged}")
    except asyncio.TimeoutError:
        logger.warning(
            f"Ephemeral conversations not all purged within {SHUTDOWN_PURGE_SECONDS:.0f} s "
            f"at shutdown; the next start deletes the rest"
        )
    finally:
        db.close()
