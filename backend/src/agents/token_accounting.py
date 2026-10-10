"""Real-token accounting: what a request costs, message by message.

The local servers own the tokenizer; the backend estimates with chars/4
(``count_tokens_approximately`` with its defaults) and corrects the estimate
with what the server reported -- PER MESSAGE, because one ratio measured on a
request of one composition is wrong on a request of another (live run: short
English turns measured r = 0.84-0.94, the next 7k-token paste then needed
1.0-1.2; a 24.8k-token looping answer counted at the previous ratio pushed the
next output budget to its floor).

Every model call sends ``stream_options.include_usage`` (``build_chat_model``)
and ``Erudi_Chat_OpenAI`` stamps the usage chunk with the estimate of the
request it answers (``erudi_request_est``), so a stamped AI message carries
``r = input_tokens / erudi_request_est`` (clamped to [0.8, 6.0]; a request that
carried images has none: the estimator charges 85 tokens per image, the server
the real ones). The cost of a list of messages is then decided once, from the
whole list:

1. k = the last stamped hop AFTER the last user message (the current turn's
   latest request is an exact prefix of the next one), else the last stamped
   FIRST hop (a request that ended with a user message). Every message BEFORE
   k was inside that measured request and costs ``r * est(m)`` -- except a
   compaction summary (``lc_source``), written after that request.
2. Everything else was never measured as input. An AI message that carries
   ``usage_metadata.output_tokens`` and no reasoning costs exactly that, plus
   the estimator's per-message overhead: the server generated those tokens
   (with reasoning, ``output_tokens`` includes tokens the history never
   replays, so its text is weighed instead). Any other message -- a question,
   a paste, a tool result of the current turn, a summary, the KB block --
   costs ``w * est(m)``, ``w`` the script weight of ITS OWN text: ASCII digits
   one token each for the counter and, for the budget, as the loaded
   tokenizer says (``digit_tokens_of``: 1.0 when it splits numbers per digit
   like Qwen, Gemma, Mistral, DeepSeek; 0.34 when it groups them like Llama 3,
   gpt-oss, Phi-4, or when unknown), CJK/Kana/Hangul ``dense`` tokens each,
   other non-ASCII letters and marks 0.4, everything else chars/4.
3. Nothing measured (the first turn, the Arena, right after a compaction that
   kept no stamped hop): rule 2 for every message.

The request overhead O that is not in the conversation state -- the system
prompt and the tool schemas -- costs ``r * est`` when k exists (it was in that
request too), else its own script weight. The KB block is new every turn and
is weighed like any new text.

The compaction counter and the output budget share this pure function
(``real_tokens_est`` / ``message_weights``) with two settings: the counter
uses ``dense = 1.0`` and a floor of 1.2 on script weights (code and numbers
are only partly detected: counting high only compacts earlier); the budget
uses ``dense = 0.65`` (large-vocabulary tokenizers spend 0.6-0.8 token per CJK
character) and no floor (an under-estimate is recovered by the preflight
retry, an over-estimate silently truncates the answer). Nothing is learned
across sessions: every number here is read off the conversation's own
messages.

The script densities are tokenizer-dependent CONSTANTS, applied only to the
unmeasured tail (what was never inside a measured request). They can be
wrong for a given model: Japanese kana, small-vocabulary tokenizers,
Devanagari or Thai can be under-estimated -- the budget is then caught by the
preflight retry on the allocated window, and compaction may come a little
late; a CJK paste of tens of thousands of characters is over-counted by the
counter (one token per character) and can trigger one compaction that was not
needed. Once a request has been answered, its share is measured and these
constants no longer apply to it.

A KB turn's request carries a block the history does not keep
(``_KbContextMiddleware`` adds it to the request only), so the client stamps
the block's size (``erudi_request_kb_est`` and its real estimate) and a hop's
ratio is measured on the rest of the request: the next request, which no
longer carries the block, is costed on what it still holds.

Pure module: no I/O, ``langchain_core`` only.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, List, NamedTuple, Optional, Sequence, Tuple

from langchain_core.messages.utils import count_tokens_approximately

# Bounds of a measured ratio. Below 0.8 the estimate over-counts more than any
# tokenizer this app runs; above 6.0 a request is all CJK and dense symbols.
RATIO_FLOOR = 0.8
RATIO_CEILING = 6.0

# The two settings of the script weight (see the module docstring).
COUNTER_DENSE_TOKENS = 1.0
COUNTER_WEIGHT_FLOOR = 1.2
BUDGET_DENSE_TOKENS = 0.65

# Per-character costs of the script weight.
DIGIT_TOKENS = 1.0
# A tokenizer that groups digits (Llama 3, gpt-oss, Phi-4: ``\p{N}{1,3}``)
# spends about a third of a token per digit. The budget uses it unless the
# loaded tokenizer is known to split digits one by one (``digit_tokens_of``):
# for the budget an over-estimate silently truncates the answer.
GROUPED_DIGIT_TOKENS = 0.34
OTHER_LETTER_TOKENS = 0.4

# The stamp ``Erudi_Chat_OpenAI._astream`` writes on the usage chunk's
# ``response_metadata`` (one chunk only: LangChain's ``merge_dicts`` sums two
# ints under one key and rejects two differing bools).
REQUEST_EST_KEY = "erudi_request_est"
REQUEST_HAS_IMAGES_KEY = "erudi_request_has_images"
REQUEST_FIRST_HOP_KEY = "erudi_request_first_hop"
# The KB additions a request carried (``_KbContextMiddleware`` adds them to the
# request only, never to the state): their chars/4 estimate and their real
# cost at the budget's weights. A hop's ratio is measured WITHOUT them, so the
# next request -- which no longer carries that block -- is costed on what it
# still holds.
REQUEST_KB_EST_KEY = "erudi_request_kb_est"
REQUEST_KB_REAL_KEY = "erudi_request_kb_real_est"

# ``additional_kwargs["lc_source"]`` of the summary LangChain inserts.
SUMMARY_SOURCE_MARKER = "summarization"

# The dedicated reasoning field the chat client re-attaches to AI chunks.
REASONING_KWARG = "reasoning_content"
_INLINE_REASONING_TAG = "<think>"
_INLINE_REASONING_END = "</think>"

# chars/4: ``count_tokens_approximately``'s default.
APPROX_CHARS_PER_TOKEN = 4

_IMAGE_PART_TYPES = frozenset({"image", "image_url"})

# Each stripped tool's directive marker (#310): past turns' KB and web results
# are sent as these. Tools not listed here (e.g. the tiny calculator results)
# pass through untouched.
STALE_TOOL_RESULT_MARKERS = {
    "search_knowledge_base": (
        "[knowledge base results from an earlier turn omitted - call "
        "search_knowledge_base again if this turn needs facts from the "
        "documents]"
    ),
    "web_search": (
        "[web search results from an earlier turn omitted - call "
        "web_search again if this turn needs fresh web facts]"
    ),
}


# --------------------------------------------------------------------------- shared helpers


def last_human_index(messages: Sequence[Any]) -> Optional[int]:
    """Index of the last user message, or ``None``."""
    for i in range(len(messages) - 1, -1, -1):
        if getattr(messages[i], "type", None) == "human":
            return i
    return None


def stale_result_copy(message: Any) -> Any:
    """``message`` as a past turn sends it: a KB/web result becomes its marker
    (a copy; any other message is returned as is)."""
    marker = STALE_TOOL_RESULT_MARKERS.get(getattr(message, "name", None))
    if getattr(message, "type", None) == "tool" and marker is not None:
        return message.model_copy(update={"content": marker})
    return message


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
    """THE estimator: chars/4 over the messages and the tool schemas
    (``count_tokens_approximately`` with its defaults, no usage scaling).
    Raises on a message shape it cannot read, like the counter it wraps."""
    return count_tokens_approximately(list(messages or ()), tools=openai_tools(tools))


def estimate(message: Any) -> int:
    """chars/4 of one message, its per-message overhead included."""
    return count_tokens_approximately([message])


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


# --------------------------------------------------------------------------- measurements


def hop_ratio(message: Any) -> Optional[float]:
    """r of one stamped AI message, clamped; ``None`` when it carries no usage
    or no stamp, or when its request carried images."""
    if getattr(message, "type", None) != "ai":
        return None
    metadata = getattr(message, "response_metadata", None) or {}
    if metadata.get(REQUEST_HAS_IMAGES_KEY) is True:
        return None
    estimated = _positive_int(metadata.get(REQUEST_EST_KEY))
    usage = getattr(message, "usage_metadata", None) or {}
    input_tokens = _positive_int(usage.get("input_tokens"))
    if estimated is None or input_tokens is None:
        return None
    ratio = input_tokens / estimated
    kb_est = _positive_int(metadata.get(REQUEST_KB_EST_KEY))
    kb_real = _positive_int(metadata.get(REQUEST_KB_REAL_KEY))
    if kb_est is not None and kb_real is not None:
        rest_est, rest_real = estimated - kb_est, input_tokens - kb_real
        if rest_est < 1 or rest_real <= 0:
            # The hop measured its KB block, not the history: unmeasurable.
            return None
        ratio = rest_real / rest_est
    return min(RATIO_CEILING, max(RATIO_FLOOR, ratio))


def _is_first_hop(message: Any) -> bool:
    metadata = getattr(message, "response_metadata", None) or {}
    return metadata.get(REQUEST_FIRST_HOP_KEY) is True


def measured_anchor(messages: Sequence[Any]) -> Tuple[Optional[int], Optional[float]]:
    """``(k, r)``: the last stamped hop after the last user message, else the
    last stamped first hop; ``(None, None)`` when nothing is measured."""
    messages = list(messages)
    start = last_human_index(messages)
    if start is not None:
        for i in range(len(messages) - 1, start, -1):
            ratio = hop_ratio(messages[i])
            if ratio is not None:
                return i, ratio
    for i in range(len(messages) - 1, -1, -1):
        if _is_first_hop(messages[i]):
            ratio = hop_ratio(messages[i])
            if ratio is not None:
                return i, ratio
    return None, None


# --------------------------------------------------------------------------- script weight


_GROUPED_DIGITS = re.compile(r"(\\p\{N\}|\\d)(\{\d*,?\d*\}|\+|\*)")
_SINGLE_DIGIT = re.compile(r"(\\p\{N\}|\\d)(?![{+*])")


def _pre_tokenizers(node: Any) -> list:
    if not isinstance(node, dict):
        return []
    if node.get("type") == "Sequence":
        out = []
        for child in node.get("pretokenizers") or []:
            out.extend(_pre_tokenizers(child))
        return out
    return [node]


def digit_tokens_of(tokenizer: Any) -> Optional[float]:
    """Tokens per digit of a ``tokenizer.json`` (a parsed dict), read off its
    pre-tokenizer: digits split one by one (a ``Digits`` pre-tokenizer with
    ``individual_digits``, or a ``Split`` regex matching a single
    ``\\p{N}``) cost 1.0; grouped (``\\p{N}{1,3}``, ``\\p{N}+``, ``Digits``
    without ``individual_digits``) cost ``GROUPED_DIGIT_TOKENS``; ``None``
    when the pre-tokenizer says nothing about digits."""
    if not isinstance(tokenizer, dict):
        return None
    verdict: Optional[float] = None
    for node in _pre_tokenizers(tokenizer.get("pre_tokenizer")):
        kind = node.get("type")
        if kind == "Digits":
            if node.get("individual_digits") is True:
                return DIGIT_TOKENS
            verdict = GROUPED_DIGIT_TOKENS
        elif kind == "Split":
            pattern = (node.get("pattern") or {}).get("Regex")
            if not isinstance(pattern, str):
                continue
            if _GROUPED_DIGITS.search(pattern):
                verdict = GROUPED_DIGIT_TOKENS
            elif _SINGLE_DIGIT.search(pattern):
                return DIGIT_TOKENS
    return verdict


def _is_dense_script(char: str) -> bool:
    """CJK ideographs, Kana and Hangul."""
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


def message_text(message: Any) -> str:
    """The text of a message's content parts, plus its tool calls (what the
    estimator counts besides the role)."""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                value = part.get("text")
                if isinstance(value, str):
                    parts.append(value)
        text = "".join(parts)
    else:
        text = ""
    calls = getattr(message, "tool_calls", None)
    if calls:
        text += json.dumps(calls, ensure_ascii=False, default=str)
    return text


def script_weight(text: str, *, dense: float, digit_tokens: float = DIGIT_TOKENS) -> float:
    """Real tokens per chars/4 token of ``text``: digits 1.0 token each,
    CJK/Kana/Hangul ``dense``, other non-ASCII letters and marks 0.4,
    everything else chars/4. English prose reads about 1.0, numbers and
    CJK several times more; an empty text reads 1.0."""
    if not text:
        return 1.0
    tokens = 0.0
    for char in text:
        if char.isascii():
            tokens += digit_tokens if char.isdigit() else 1.0 / APPROX_CHARS_PER_TOKEN
        elif _is_dense_script(char):
            tokens += dense
        elif unicodedata.category(char)[0] in ("L", "M"):
            tokens += OTHER_LETTER_TOKENS
        else:
            tokens += 1.0 / APPROX_CHARS_PER_TOKEN
    return tokens / (len(text) / APPROX_CHARS_PER_TOKEN)


# --------------------------------------------------------------------------- per-message costs


def history_view(message: Any) -> Any:
    """The message as the history replays it: an inline ``<think>`` block is
    dropped by the chat templates, so only the text after the last
    ``</think>`` remains (a copy; any other message is returned as is)."""
    content = getattr(message, "content", None)
    if (
        getattr(message, "type", None) == "ai"
        and isinstance(content, str)
        and _INLINE_REASONING_END in content
    ):
        tail = content.rsplit(_INLINE_REASONING_END, 1)[1].lstrip()
        return message.model_copy(update={"content": tail})
    return message


def has_reasoning(message: Any) -> bool:
    """An AI message whose generation included reasoning tokens the history
    does not replay (the dedicated field, or inline ``<think>``)."""
    if (getattr(message, "additional_kwargs", None) or {}).get(REASONING_KWARG):
        return True
    content = getattr(message, "content", None)
    return isinstance(content, str) and (
        _INLINE_REASONING_TAG in content or _INLINE_REASONING_END in content
    )


_AI_OVERHEAD: Optional[int] = None


def _ai_overhead() -> int:
    """The estimator's per-message overhead for an AI message (role + 3)."""
    global _AI_OVERHEAD
    if _AI_OVERHEAD is None:
        from langchain_core.messages import AIMessage

        _AI_OVERHEAD = estimate(AIMessage(content=""))
    return _AI_OVERHEAD


def exact_tokens(message: Any) -> Optional[int]:
    """What an AI message costs exactly -- its ``output_tokens`` plus the
    per-message overhead -- when it carries usage and no reasoning."""
    if getattr(message, "type", None) != "ai" or has_reasoning(message):
        return None
    usage = getattr(message, "usage_metadata", None) or {}
    output_tokens = _positive_int(usage.get("output_tokens"))
    if output_tokens is None:
        return None
    return output_tokens + _ai_overhead()


def fresh_weight(
    message: Any,
    *,
    dense: float,
    weight_floor: float = 0.0,
    digit_tokens: float = DIGIT_TOKENS,
) -> float:
    """Rule 2: the weight of a message that was never measured as input."""
    exact = exact_tokens(message)
    if exact is not None:
        return exact / max(1, estimate(message))
    viewed = history_view(message)
    return max(
        weight_floor,
        script_weight(message_text(viewed), dense=dense, digit_tokens=digit_tokens),
    )


def _is_summary(message: Any) -> bool:
    kwargs = getattr(message, "additional_kwargs", None) or {}
    return kwargs.get("lc_source") == SUMMARY_SOURCE_MARKER


def _measured_before_anchor(index: int, message: Any, k: Optional[int]) -> bool:
    return k is not None and index < k and not _is_summary(message)


def message_weights(
    messages: Sequence[Any],
    *,
    dense: float,
    weight_floor: float = 0.0,
    digit_tokens: float = DIGIT_TOKENS,
) -> Tuple[List[float], Optional[float]]:
    """``(weights, r)``: real tokens per estimated token for each message
    (position-aligned), decided once from the whole list, and the measured
    ratio of the anchor (``None`` when nothing is measured)."""
    messages = list(messages)
    k, ratio = measured_anchor(messages)
    weights = [
        ratio
        if _measured_before_anchor(i, message, k)
        else fresh_weight(
            message, dense=dense, weight_floor=weight_floor, digit_tokens=digit_tokens
        )
        for i, message in enumerate(messages)
    ]
    return weights, ratio


def weighted_cost(message: Any, weight: float) -> int:
    """``ceil(weight * est(message))`` over the message as the history
    replays it: a partial copy costs its own share at the weight of the
    message it came from."""
    return math.ceil(weight * estimate(history_view(message)))


def kb_stamp(added_text: str, digit_tokens: float = GROUPED_DIGIT_TOKENS) -> dict:
    """The stamp keys of a request that carried ``added_text`` as its KB
    additions (empty when it carried none)."""
    if not added_text:
        return {}
    kb_est = math.ceil(len(added_text) / APPROX_CHARS_PER_TOKEN)
    weight = script_weight(added_text, dense=BUDGET_DENSE_TOKENS, digit_tokens=digit_tokens)
    return {REQUEST_KB_EST_KEY: kb_est, REQUEST_KB_REAL_KEY: math.ceil(weight * kb_est)}


@dataclass(frozen=True)
class RequestOverhead:
    """What a request carries beyond the conversation state.

    ``fixed_est``/``fixed_text``: the system prompt and the tool schemas
    (chars/4, and their text for the script weight when nothing is measured);
    ``added_text``: the KB additions ``_KbContextMiddleware._merge`` adds to
    the last user message (the block, two blank-line joins, the language
    line) -- new content every turn.
    """

    fixed_est: int = 0
    fixed_text: str = ""
    added_text: str = ""


def request_overhead(
    system_prompt: Optional[str],
    kb_context_block: Optional[str] = None,
    kb_language_line: str = "",
    tools: Optional[Iterable[Any]] = None,
) -> RequestOverhead:
    """The overhead of this turn's requests, unscaled."""
    from langchain_core.messages import SystemMessage

    system = [SystemMessage(system_prompt)] if system_prompt else []
    converted = openai_tools(tools) or []
    fixed_text = (system_prompt or "") + "".join(
        json.dumps(t, ensure_ascii=False) for t in converted
    )
    added_text = f"{kb_context_block}\n\n\n\n{kb_language_line or ''}" if kb_context_block else ""
    return RequestOverhead(
        fixed_est=request_tokens_est(system, tools),
        fixed_text=fixed_text,
        added_text=added_text,
    )


def overhead_tokens(
    overhead: Optional[RequestOverhead],
    ratio: Optional[float],
    *,
    dense: float,
    weight_floor: float = 0.0,
    digit_tokens: float = DIGIT_TOKENS,
) -> int:
    """O in real tokens: the fixed part at the anchor's r (else its own script
    weight), the KB additions at their own script weight."""
    if overhead is None:
        return 0
    fixed_weight = (
        ratio
        if ratio is not None
        else max(
            weight_floor,
            script_weight(overhead.fixed_text, dense=dense, digit_tokens=digit_tokens),
        )
    )
    total = math.ceil(fixed_weight * overhead.fixed_est)
    if overhead.added_text:
        added_est = math.ceil(len(overhead.added_text) / APPROX_CHARS_PER_TOKEN)
        added_weight = max(
            weight_floor,
            script_weight(overhead.added_text, dense=dense, digit_tokens=digit_tokens),
        )
        total += math.ceil(added_weight * added_est)
    return total


class RealTokens(NamedTuple):
    """A request's size in real tokens, and the part of it that is exact
    (AI messages costed at their ``output_tokens``)."""

    total: int
    exact: int


def real_tokens_est(
    messages: Sequence[Any],
    *,
    dense: float,
    weight_floor: float = 0.0,
    overhead: Optional[RequestOverhead] = None,
    tools: Optional[Iterable[Any]] = None,
    digit_tokens: float = DIGIT_TOKENS,
) -> RealTokens:
    """THE real-token estimate of a list of messages (the module rules),
    plus the request overhead and the tool schemas it carries."""
    messages = list(messages)
    weights, ratio = message_weights(
        messages, dense=dense, weight_floor=weight_floor, digit_tokens=digit_tokens
    )
    k, _ = measured_anchor(messages)
    total = 0
    exact = 0
    for i, (message, weight) in enumerate(zip(messages, weights)):
        cost = weighted_cost(message, weight)
        total += cost
        if not _measured_before_anchor(i, message, k) and exact_tokens(message) is not None:
            exact += cost
    converted = openai_tools(tools)
    if converted:
        tools_est = count_tokens_approximately([], tools=converted)
        tools_text = "".join(json.dumps(t, ensure_ascii=False) for t in converted)
        tools_weight = (
            ratio
            if ratio is not None
            else max(
                weight_floor,
                script_weight(tools_text, dense=dense, digit_tokens=digit_tokens),
            )
        )
        total += math.ceil(tools_weight * tools_est)
    total += overhead_tokens(
        overhead, ratio, dense=dense, weight_floor=weight_floor, digit_tokens=digit_tokens
    )
    return RealTokens(total=total, exact=exact)
