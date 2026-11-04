"""Shared helpers for the test suite.

Currently exposes platform detection helpers used by both `conftest.py`
fixtures and per-file `pytest.mark.skipif` guards. Mirrors the dispatch
logic of `src.engines.base_engine.BaseEngine.get_engine()` without
re-evaluating it on every call site.

Imported explicitly by test files (not auto-discovered by pytest).
"""

from __future__ import annotations


def is_mlx_platform() -> bool:
    """Return True iff `BaseEngine.get_engine()` resolves to `MLX_Engine`."""
    try:
        from src.engines.base_engine import BaseEngine

        return BaseEngine.get_engine().__name__ == "MLX_Engine"
    except Exception:
        return False


def is_cuda_platform() -> bool:
    """Return True iff `BaseEngine.get_engine()` resolves to `CUDA_Engine`."""
    try:
        from src.engines.base_engine import BaseEngine

        return BaseEngine.get_engine().__name__ == "CUDA_Engine"
    except Exception:
        return False


def is_cpu_platform() -> bool:
    """Return True iff `BaseEngine.get_engine()` resolves to `CPU_Engine`."""
    try:
        from src.engines.base_engine import BaseEngine

        return BaseEngine.get_engine().__name__ == "CPU_Engine"
    except Exception:
        return False


import json
import re

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk


class ToolableFakeChatModel(GenericFakeChatModel):
    """GenericFakeChatModel usable under ``create_agent(tools=[...])``.

    Two gaps in the stock fake are filled:
    - ``bind_tools`` raises NotImplementedError → no-op here (the scripted
      messages already decide whether a tool gets called);
    - ``_stream`` only splits ``content`` and ignores modern ``tool_calls``
      (a tool-call-only message yields zero chunks → "No generations found
      in stream") → emitted as a single chunk with ``tool_call_chunks``.

    A message carrying BOTH content and ``tool_calls`` streams its content
    tokens first, THEN the ``tool_call_chunks`` chunk — mirroring the real
    pre-tool narration order observed on local models (#297).
    """

    def bind_tools(self, tools, **kwargs):
        return self

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        result = self._generate(messages, stop=stop, run_manager=None, **kwargs)
        message = result.generations[0].message
        tool_calls = getattr(message, "tool_calls", None) or []
        if message.content or not tool_calls:
            tokens = re.split(r"(\s)", message.content)
            for index, token in enumerate(tokens):
                is_last = index == len(tokens) - 1 and not tool_calls
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=token, chunk_position="last" if is_last else None
                    )
                )
                if run_manager:
                    run_manager.on_llm_new_token(token, chunk=chunk)
                yield chunk
        if tool_calls:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    chunk_position="last",
                    tool_call_chunks=[
                        {
                            "name": call["name"],
                            "args": json.dumps(call["args"]),
                            "id": call["id"],
                            "index": index,
                            "type": "tool_call_chunk",
                        }
                        for index, call in enumerate(message.tool_calls)
                    ],
                )
            )


class FakeDownloadQuery:
    """In-memory stand-in for the queries the download task runs
    (``_run_download_task``) over one job row and one model row: ``get``
    returns the row, and the job's conditional UPDATE (``filter`` on
    ``status``, then ``update``) and its column read (``one_or_none``) apply
    to the job row the way Postgres would."""

    def __init__(self, model, job, llm):
        from src.entities.DownloadJob import DownloadJobModel

        self._job, self._llm = job, llm
        self._is_job = model is DownloadJobModel or (
            getattr(model, "class_", None) is DownloadJobModel
        )
        self._criteria = []

    def get(self, _id):
        return self._job if self._is_job else self._llm

    def filter(self, *criteria):
        self._criteria.extend(criteria)
        return self

    def update(self, values, **_):
        for criterion in self._criteria:
            key = criterion.left.key
            if key != "id" and not criterion.operator(
                getattr(self._job, key), criterion.right.value
            ):
                return 0
        for key, value in values.items():
            setattr(self._job, key, value)
        return 1

    def one_or_none(self):
        return self._job
