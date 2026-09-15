"""Pure helpers for the dedicated reasoning channel of the local servers (#554).

Both inference children extract chain-of-thought server-side and stream it in
a dedicated delta field of the raw OpenAI chunk, NOT inside ``delta.content``:

- llama-server (default ``--reasoning-format auto``): per-family extraction
  into ``delta.reasoning_content`` -- ``<think>`` and friends, the thinking
  block a chat template reopens on every post-tool hop, and an unterminated
  block closed cleanly at EOS.
- mlx_vlm.server (native split): ``delta.reasoning``, mirrored into
  ``delta.reasoning_content`` on the pinned 0.6.17.

Stock ``ChatOpenAI`` drops both fields on the floor, so ``Erudi_Chat_OpenAI``
(``src.agents.chat_model``) re-attaches them to the message chunk from inside
its ``_convert_chunk_to_generation_chunk`` override. The extraction itself
lives here, LangChain-free, so it stays unit-testable anywhere and importable
at boot (issue #160: nothing in ``src.agents`` may pull LangChain at module
import time).
"""

from __future__ import annotations

from typing import Any, Optional

# The ``additional_kwargs`` key the runner reads on the LangChain side. It
# mirrors llama-server's wire name (the more widespread of the two fields).
REASONING_KWARG = "reasoning_content"

# Wire fields checked, in order of precedence.
_REASONING_FIELDS = ("reasoning_content", "reasoning")


def extract_reasoning_delta(raw_chunk: Any) -> Optional[str]:
    """The reasoning text carried by one raw chat-completion chunk.

    Reads the first choice's ``delta`` and returns ``reasoning_content``
    (llama-server, and mlx_vlm 0.6.17's mirror) or, failing that,
    ``reasoning`` (mlx_vlm's own name). Returns ``None`` when the chunk
    carries no reasoning -- absent fields, ``None`` values, empty strings and
    non-string values alike -- so callers only ever test the result for
    truthiness. Defensive about shape on purpose: this runs on every streamed
    chunk, and a malformed chunk must read as "no reasoning", never raise.
    """
    if not isinstance(raw_chunk, dict):
        return None
    choices = raw_chunk.get("choices")
    if not isinstance(choices, (list, tuple)) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    delta = first.get("delta")
    if not isinstance(delta, dict):
        return None
    for field in _REASONING_FIELDS:
        value = delta.get(field)
        if isinstance(value, str) and value:
            return value
    return None
