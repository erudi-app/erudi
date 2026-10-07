"""Real-token accounting: one estimator, scaled by a ratio the server measured.

The local servers own the tokenizer; the backend estimates with chars/4
(``count_tokens_approximately`` with its defaults) and corrects the estimate
with what the server reports. Every model call sends
``stream_options.include_usage`` (``build_chat_model``), so the server's last
chunk carries the real ``prompt_tokens`` of the request. ``Erudi_Chat_OpenAI``
stamps that chunk with the estimate of THE SAME request
(``erudi_request_est``: system prompt, KB block, history after the strippers,
tool schemas), so

    r = input_tokens / erudi_request_est

is a pure measurement of how much chars/4 under-counts this model's
tokenizer on this conversation -- injected overhead is on both sides and
cancels out. r is clamped to [0.8, 6.0]; a request that carried images is
skipped (the estimator charges 85 tokens per image, the server the real ones).

Which hop's r is used matters. A FIRST hop (the request ended with a user
message, stamped ``erudi_request_first_hop``) has the composition of the next
first hop: past tool results are already markers. The LAST hop of a KB or web
turn does not -- its request carried this turn's results, often in another
script (O_est 1500 at 1.1 + history 1000 + 3000 tokens of CJK results at 3
gives 2.14 while the first hop measured 1.1). So:

* ``first_hop_ratio``: the r of the last stamped first hop, else the value the
  summary message carries (written at compaction), else ``None``;
* ``current_turn_ratio``: the r of a later hop of the CURRENT turn (an AI
  message after the last user message), else ``None``.

The compaction counter uses ``current_turn_ratio``, else ``first_hop_ratio``,
else ``max(1.5, script_ratio)``; the output budget uses
``current_turn_ratio``, else the client's ``prompt_ratio`` (the runner's
``first_hop_ratio`` of the raw state at turn start), else ``script_ratio``.
Nothing is learned across sessions: every number here is read off the
conversation's own messages.

Pure module: no I/O, ``langchain_core`` only.
"""

from __future__ import annotations

import math
import unicodedata
from typing import Any, Iterable, Optional

from langchain_core.messages.utils import count_tokens_approximately

# Bounds of a measured ratio. Below 0.8 the estimate over-counts more than any
# tokenizer this app runs; above 6.0 a request is all CJK and dense symbols.
RATIO_FLOOR = 0.8
RATIO_CEILING = 6.0

# The compaction counter's floor when nothing is measured: counting high only
# compacts a little earlier, which is the safe side for compaction (never for
# the output budget, which uses ``script_ratio`` alone).
COUNTER_FALLBACK_RATIO_FLOOR = 1.5

# The stamp ``Erudi_Chat_OpenAI._astream`` writes on the usage chunk's
# ``response_metadata`` (one chunk only: LangChain's ``merge_dicts`` sums two
# ints under one key and rejects two differing bools).
REQUEST_EST_KEY = "erudi_request_est"
REQUEST_HAS_IMAGES_KEY = "erudi_request_has_images"
REQUEST_FIRST_HOP_KEY = "erudi_request_first_hop"

# What a compaction writes on the summary message (``additional_kwargs``): the
# first-hop ratio of the state it summarized, so a state whose kept tail holds
# only a turn's LAST hop still has a first-hop measurement.
FIRST_HOP_RATIO_KWARG = "erudi_first_hop_ratio"

# ``additional_kwargs["lc_source"]`` of the summary LangChain inserts.
SUMMARY_SOURCE_MARKER = "summarization"

# chars/4: ``count_tokens_approximately``'s default.
APPROX_CHARS_PER_TOKEN = 4

# The script-aware fallback's per-character costs (see ``script_ratio``).
_DENSE_SCRIPT_TOKENS_PER_CHAR = 1.0
_OTHER_LETTER_TOKENS_PER_CHAR = 0.4

_IMAGE_PART_TYPES = frozenset({"image", "image_url"})


def openai_tools(tools: Optional[Iterable[Any]]) -> Optional[list]:
    """``tools`` in the OpenAI schema shape, or ``None`` when there are none.

    The stamp sees the request's dict schemas (``bind_tools`` already
    converted them), the overhead sees ``BaseTool`` objects: converting both
    with ``convert_to_openai_tool`` (idempotent on a converted dict) makes the
    two sides count the same characters.
    """
    if not tools:
        return None
    from langchain_core.utils.function_calling import convert_to_openai_tool

    return [convert_to_openai_tool(t) for t in tools]


def request_tokens_est(messages: Iterable[Any], tools: Optional[Iterable[Any]] = None) -> int:
    """THE request-size estimator: chars/4 over the messages and the tool
    schemas (``count_tokens_approximately`` with its defaults, no usage
    scaling). Raises on a message shape it cannot read, like the counter it
    wraps."""
    return count_tokens_approximately(list(messages or ()), tools=openai_tools(tools))


def request_overhead_est(
    system_prompt: Optional[str],
    kb_context_block: Optional[str] = None,
    kb_language_line: str = "",
    tools: Optional[Iterable[Any]] = None,
) -> int:
    """``O_est``: what a request carries beyond the conversation state, unscaled.

    The system prompt and the tool schemas (``create_agent`` adds both at
    request time), plus the KB additions exactly as
    ``_KbContextMiddleware._merge`` adds them to the last user message (the
    block, two blank-line joins and the language line; the question itself is
    already in the state and is not counted again).
    """
    from langchain_core.messages import SystemMessage

    messages = [SystemMessage(system_prompt)] if system_prompt else []
    estimate = request_tokens_est(messages, tools)
    if kb_context_block:
        added_chars = len(kb_context_block) + 4 + len(kb_language_line or "")
        estimate += math.ceil(added_chars / APPROX_CHARS_PER_TOKEN)
    return estimate


def messages_have_images(messages: Iterable[Any]) -> bool:
    """Whether any message carries an image content part."""
    for message in messages or ():
        content = getattr(message, "content", None)
        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") in _IMAGE_PART_TYPES for part in content
        ):
            return True
    return False


def _positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def _clamp_ratio(value: float) -> float:
    return min(RATIO_CEILING, max(RATIO_FLOOR, value))


def hop_ratio(message: Any) -> Optional[float]:
    """r of one stamped AI message, clamped; ``None`` when it carries no usage
    or no stamp, or when its request carried images."""
    if getattr(message, "type", None) != "ai":
        return None
    metadata = getattr(message, "response_metadata", None) or {}
    if metadata.get(REQUEST_HAS_IMAGES_KEY) is True:
        return None
    estimate = _positive_int(metadata.get(REQUEST_EST_KEY))
    usage = getattr(message, "usage_metadata", None) or {}
    input_tokens = _positive_int(usage.get("input_tokens"))
    if estimate is None or input_tokens is None:
        return None
    return _clamp_ratio(input_tokens / estimate)


def _is_summary(message: Any) -> bool:
    kwargs = getattr(message, "additional_kwargs", None) or {}
    return kwargs.get("lc_source") == SUMMARY_SOURCE_MARKER


def _carried_ratio(message: Any) -> Optional[float]:
    value = (getattr(message, "additional_kwargs", None) or {}).get(FIRST_HOP_RATIO_KWARG)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return _clamp_ratio(float(value))


def first_hop_ratio(messages: Iterable[Any]) -> Optional[float]:
    """The r of the last stamped FIRST hop, else the summary-carried value,
    else ``None``. Never a last hop's r, never a fallback value."""
    messages = list(messages or ())
    for message in reversed(messages):
        metadata = getattr(message, "response_metadata", None) or {}
        if metadata.get(REQUEST_FIRST_HOP_KEY) is not True:
            continue
        ratio = hop_ratio(message)
        if ratio is not None:
            return ratio
    for message in messages:
        if _is_summary(message):
            carried = _carried_ratio(message)
            if carried is not None:
                return carried
    return None


def _last_human_index(messages: list) -> Optional[int]:
    return next(
        (
            i
            for i in range(len(messages) - 1, -1, -1)
            if getattr(messages[i], "type", None) == "human"
        ),
        None,
    )


def current_turn_ratio(messages: Iterable[Any]) -> Optional[float]:
    """The r of the latest stamped hop of the CURRENT turn (an AI message after
    the last user message), else ``None``."""
    messages = list(messages or ())
    start = _last_human_index(messages)
    tail = messages[start + 1 :] if start is not None else messages
    for message in reversed(tail):
        ratio = hop_ratio(message)
        if ratio is not None:
            return ratio
    return None


def _is_dense_script(char: str) -> bool:
    """CJK ideographs, Kana and Hangul: about one token per character."""
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF  # CJK Extension A
        or 0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0xF900 <= code <= 0xFAFF  # CJK Compatibility Ideographs
        or 0x20000 <= code <= 0x3FFFF  # CJK Extensions B and later
        or 0x3040 <= code <= 0x30FF  # Hiragana, Katakana
        or 0x31F0 <= code <= 0x31FF  # Katakana Phonetic Extensions
        or 0xFF66 <= code <= 0xFF9F  # Halfwidth Katakana
        or 0x1100 <= code <= 0x11FF  # Hangul Jamo
        or 0x3130 <= code <= 0x318F  # Hangul Compatibility Jamo
        or 0xAC00 <= code <= 0xD7AF  # Hangul Syllables
    )


def _message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def script_ratio(messages: Iterable[Any]) -> float:
    """The fallback ratio when nothing is measured: a script-aware estimate
    over chars/4, on the messages' text.

    CJK ideographs, Kana and Hangul count one token per character; other
    non-ASCII letters AND combining marks (Unicode L* and M*: Cyrillic,
    Greek, Arabic, Devanagari and its vowel signs) 0.4; everything else
    chars/4. English and code read 1.0, French with 4 % accented letters
    about 1.02, Russian about 1.5, Hindi about 1.55, CJK 3 to 4. Code and
    numbers (which chars/4 also under-counts) are not detected.
    """
    chars = 0
    estimate = 0.0
    for message in messages or ():
        for char in _message_text(message):
            chars += 1
            if char.isascii():
                estimate += 1.0 / APPROX_CHARS_PER_TOKEN
            elif _is_dense_script(char):
                estimate += _DENSE_SCRIPT_TOKENS_PER_CHAR
            elif unicodedata.category(char)[0] in ("L", "M"):
                estimate += _OTHER_LETTER_TOKENS_PER_CHAR
            else:
                estimate += 1.0 / APPROX_CHARS_PER_TOKEN
    if chars == 0:
        return 1.0
    return estimate / (chars / APPROX_CHARS_PER_TOKEN)


def counter_ratio(messages: Iterable[Any], as_sent: Optional[Iterable[Any]] = None) -> float:
    """The compaction counter's ratio for this state: ``current_turn_ratio``,
    else ``first_hop_ratio``, else ``max(1.5, script_ratio)`` over the
    messages as sent (``as_sent``; the messages themselves by default)."""
    messages = list(messages or ())
    measured = current_turn_ratio(messages)
    if measured is None:
        measured = first_hop_ratio(messages)
    if measured is not None:
        return measured
    sample = messages if as_sent is None else list(as_sent)
    return max(COUNTER_FALLBACK_RATIO_FLOOR, script_ratio(sample))
