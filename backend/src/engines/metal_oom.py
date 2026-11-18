"""Metal out-of-memory detection for the MLX inference child (#504).

When the GPU runs out of memory, MLX aborts the command buffer and writes, to
the child's stderr (`mlx_child_log`)::

    [METAL] Command buffer execution failed: Insufficient Memory
    (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)

The line reaches the app by whichever way the failure takes: in the message of
the exception the stream raises, or only in the child's log (the child can
survive the error, or die with it). ``find_metal_out_of_memory`` looks for it
in any number of texts and returns that line.

It matches on the IOGPU error token, or on an ``Insufficient Memory`` that the
same line attributes to ``[METAL]``; never on a vague word. A Python
``MemoryError``, ``Cannot allocate memory``, a CUDA or llama.cpp allocation
failure and the other ``[METAL]`` errors (a GPU timeout) are not it: the turn
that names the GPU's memory must not be shown for host-RAM or other causes.

Pure and dependency-free.
"""

from __future__ import annotations

from typing import Optional

_TOKEN = "kIOGPUCommandBufferCallbackErrorOutOfMemory"
_METAL_TAG = "[METAL]"
_INSUFFICIENT = "Insufficient Memory"

# What is read of one text: the end of a log tail, where the failure is.
_MAX_TEXT_CHARS = 64 * 1024


def _is_metal_oom_line(line: str) -> bool:
    if _TOKEN in line:
        return True
    return _METAL_TAG in line and _INSUFFICIENT in line


def find_metal_out_of_memory(*texts: Optional[str]) -> Optional[str]:
    """The line of ``texts`` that reports a Metal out-of-memory, else ``None``.

    The line is stripped; when several match, the last one of the last text
    that has one is returned (the freshest). In a text without line breaks
    (an exception message) the line is the whole text.
    """
    found: Optional[str] = None
    for text in texts:
        if not text or not isinstance(text, str):
            continue
        for line in text[-_MAX_TEXT_CHARS:].splitlines():
            if _is_metal_oom_line(line):
                found = line.strip()
    return found
