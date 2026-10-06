"""Consuming an async stream from a child task (``src.agents.isolated_stream``).

The agent's LangGraph stream is consumed in a child ``asyncio`` task so that a
client disconnect -- an anyio cancellation Starlette RE-DELIVERS at every await
of the cancelled scope -- reaches LangGraph as ONE native cancellation. LangGraph
unwinds its own exit in a separate task it awaits; a re-delivered cancellation
would cancel that task before it runs and leave the in-flight node running.
These tests pin the wrapper's contract on a plain async generator.
"""

from __future__ import annotations

import asyncio
import traceback

import anyio
import pytest

from src.agents.isolated_stream import isolated_stream
from src.core.request_context import request_id_var

pytestmark = pytest.mark.unit


def _other_tasks():
    current = asyncio.current_task()
    return {t for t in asyncio.all_tasks() if t is not current}


async def test_a_full_consumption_yields_every_item_in_order_and_leaves_no_task():
    before = _other_tasks()

    async def source():
        for i in range(5):
            yield i

    got = [item async for item in isolated_stream(source)]

    assert got == [0, 1, 2, 3, 4]
    assert _other_tasks() == before


async def test_an_exception_in_the_source_surfaces_unchanged_with_its_traceback():
    class _Boom(Exception):
        pass

    error = _Boom("raised inside the producer")

    async def failing_source():
        yield "first"
        raise error

    got = []
    with pytest.raises(_Boom) as excinfo:
        async for item in isolated_stream(failing_source):
            got.append(item)

    assert got == ["first"]
    assert excinfo.value is error
    frames = [frame.name for frame in traceback.extract_tb(excinfo.value.__traceback__)]
    assert "failing_source" in frames


async def test_items_produced_before_an_exception_are_all_delivered_first():
    async def source():
        yield 1
        yield 2
        raise ValueError("after two")

    got = []
    with pytest.raises(ValueError):
        async for item in isolated_stream(source):
            got.append(item)

    assert got == [1, 2]


async def test_closing_the_consumer_closes_the_source_before_aclose_returns():
    events = []

    async def source():
        try:
            yield "a"
            await asyncio.sleep(30)
            yield "never"
        finally:
            events.append("source-closed")

    before = _other_tasks()
    stream = isolated_stream(source)
    assert await stream.__anext__() == "a"
    await asyncio.wait_for(stream.aclose(), timeout=5)

    assert events == ["source-closed"]
    assert _other_tasks() == before


async def test_an_anyio_cancellation_reaches_the_source_once_and_its_cleanup_completes():
    """The source's cleanup awaits (like LangGraph's exit task); a re-delivered
    cancellation would interrupt it. It must run to the end, and before the
    consumer's cancellation finishes unwinding."""
    events = []

    async def source():
        try:
            yield "a"
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            events.append("source-cancelled")
            raise
        finally:
            await asyncio.sleep(0.05)
            events.append("source-cleanup-done")

    got_first = asyncio.Event()
    holder: dict = {}

    async def consume():
        with anyio.CancelScope() as scope:
            holder["scope"] = scope
            try:
                async for _ in isolated_stream(source):
                    got_first.set()
            finally:
                events.append("consumer-unwound")

    before = _other_tasks()
    task = asyncio.create_task(consume())
    await asyncio.wait_for(got_first.wait(), timeout=5)
    holder["scope"].cancel()
    await asyncio.wait_for(task, timeout=5)

    assert events == ["source-cancelled", "source-cleanup-done", "consumer-unwound"]
    assert _other_tasks() - {task} == before - {task}


async def test_a_native_cancellation_of_the_consumer_is_re_raised_after_the_source_closed():
    events = []

    async def source():
        try:
            yield "a"
            await asyncio.sleep(30)
        finally:
            events.append("source-closed")

    async def consume(started):
        async for _ in isolated_stream(source):
            started.set()

    started = asyncio.Event()
    task = asyncio.create_task(consume(started))
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert events == ["source-closed"]


async def test_a_failure_while_the_source_unwinds_after_the_consumer_left_is_logged_once(caplog):
    """The consumer is leaving for its own reason, which is the one that
    propagates; an error raised during the source's teardown (a checkpointer
    write failing while LangGraph unwinds) is still one WARNING record with
    its traceback, never discarded in silence."""
    import logging

    teardown_error = RuntimeError("checkpoint write failed during unwind")

    async def source():
        try:
            yield "a"
            await asyncio.sleep(30)
        finally:
            raise teardown_error

    stream = isolated_stream(source)
    assert await stream.__anext__() == "a"
    with caplog.at_level(logging.WARNING):
        await asyncio.wait_for(stream.aclose(), timeout=5)

    records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(records) == 1
    assert records[0].exc_info is not None
    assert records[0].exc_info[1] is teardown_error
    assert records[0].getMessage().isascii()


async def test_a_failure_already_raised_to_the_consumer_is_not_logged_again(caplog):
    import logging

    async def source():
        yield "a"
        raise ValueError("delivered to the consumer")

    with caplog.at_level(logging.WARNING), pytest.raises(ValueError):
        async for _ in isolated_stream(source):
            pass

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


async def test_the_source_runs_in_a_copy_of_the_callers_context():
    """``create_task`` copies the context: the request id the HTTP middleware
    set is what the logging filter reads inside the node."""
    seen = []

    async def source():
        seen.append(request_id_var.get())
        yield "x"

    token = request_id_var.set("req-123")
    try:
        _ = [item async for item in isolated_stream(source)]
    finally:
        request_id_var.reset(token)

    assert seen == ["req-123"]
