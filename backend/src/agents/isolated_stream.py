"""Consume an async stream from a child task, so ONE cancellation reaches it.

Why the agent's LangGraph stream is not iterated in the request's own task
---------------------------------------------------------------------------
Starlette (1.3.1, anyio task group) delivers a client disconnect as an anyio
cancellation, and anyio RE-DELIVERS it at every await of the cancelled scope.
langgraph (1.2.11) unwinds a run in ``AsyncPregelLoop.__aexit__`` by running
its exit stack in a separate task that it awaits; a re-delivered cancellation
cancels that task before it even starts, so ``AsyncBackgroundExecutor`` never
cancels the in-flight node. The model call inside that node then keeps
streaming from the inference child after the generation guard is released --
while the next turn may already reset the child's prefix cache.

``isolated_stream`` iterates the source in a child task created with
``asyncio.create_task``: that task is outside the anyio cancel scope, so when
the consumer goes away it receives exactly ONE native cancellation, LangGraph's
exit runs to completion, and the node is cancelled and awaited. The consumer
waits for the child with ``wait_shielded`` (an anyio shield scope around an
``asyncio.shield`` loop), so this all happens before the consumer -- and the
generation guard it holds -- unwinds.

Exit paths, all of which end with the child task finished:

* the source is exhausted: the child enqueues an end marker, the consumer
  returns after waiting for the child to finish;
* the source raises: the child task ends with that exception, and the
  consumer re-raises the SAME exception object (type and traceback kept)
  after every item produced before it;
* the consumer is closed (``aclose``/GeneratorExit, e.g. a caller's
  ``aclosing``) or cancelled (anyio or native) or raises: the child is
  cancelled once and awaited, shielded; a native cancellation of the consumer
  that arrived during that wait is re-raised afterwards.

The queue holds ONE item: the child runs at most one item ahead of the
consumer, as backpressure. The child runs in a copy of the consumer's
context (``create_task``), so context variables set before the call -- the
request id the logging filter reads -- are visible inside the source.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, AsyncIterator, Callable

from src.engines.base_engine import wait_shielded

# Marks the end of a source that was exhausted normally.
_END = object()


async def isolated_stream(make_source: Callable[[], AsyncIterator[Any]]) -> AsyncIterator[Any]:
    """Yield the items of ``make_source()``, iterated in a child task.

    ``make_source`` is called inside the child task, so the source (an async
    generator) is created, iterated and closed there.
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)

    async def _produce() -> None:
        async with contextlib.aclosing(make_source()) as source:
            async for item in source:
                await queue.put(item)
        await queue.put(_END)

    producer = asyncio.create_task(_produce())
    pending_get: "asyncio.Future | None" = None
    finished = False
    try:
        while True:
            if pending_get is None:
                pending_get = asyncio.ensure_future(queue.get())
            await asyncio.wait({pending_get, producer}, return_when=asyncio.FIRST_COMPLETED)
            if pending_get.done():
                item = pending_get.result()
                pending_get = None
                if item is _END:
                    finished = True
                    return
                yield item
                continue
            # The child ended without the end marker: it raised (or was
            # cancelled from elsewhere). Deliver what it queued first --
            # cancelling the pending get leaves a queued item in the queue.
            pending_get.cancel()
            pending_get = None
            while not queue.empty():
                item = queue.get_nowait()
                if item is not _END:
                    yield item
            if producer.cancelled():
                raise asyncio.CancelledError()
            exc = producer.exception()
            if exc is not None:
                raise exc
            finished = True
            return
    finally:
        if pending_get is not None:
            pending_get.cancel()
        if not finished and not producer.done():
            producer.cancel()
        cancelled = await wait_shielded(producer)
        if not producer.cancelled():
            # Mark the outcome retrieved: a failure was already re-raised
            # above, or the consumer is leaving for its own reason, which is
            # the one that propagates.
            producer.exception()
        if cancelled:
            raise asyncio.CancelledError()
