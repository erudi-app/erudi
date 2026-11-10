"""Curated turns: the app's own sentence for a turn it could not answer normally.

A curated turn names why a turn has no regular answer -- a prompt that does
not fit the window or the machine's memory, a model that did not start
answering in time or went silent, the tool-loop limit, a turn that ended
during reasoning -- in words the app chose, never a traceback.

Each kind is one entry of ``CURATED_KINDS``:

* its English line, ASCII, with ``{name}`` placeholders for its numbers. That
  line is what the database stores as the message content (what an older
  interface shows) and, for the empty-answer kinds, what the model reads back
  on the next request; the failed kinds are closed in the thread state with
  the generic ``ERROR_MESSAGE`` (``AgentRunner._repair_alternation``);
* whether it is a failed turn (red, behind ``ERROR_SENTINEL``) or a normal
  answer line (the empty-answer kinds, whose reasoning trace is kept).

The answer event of a curated turn carries ``curated_error: {kind, params,
text}`` next to the English line (``text`` is that line without the sentinel);
the conversation service persists it in the message's trace as
``{"t": "curated_error", "kind", "params", "text"}`` and the interface
translates the turn at display from ``chat:curated.<kind>`` in its language,
the numbers formatted for it, replacing exactly ``text`` in the stored content
(whatever came after it in the same turn, a failure or an interrupted stream,
stays). Adding a kind is one entry here and ``chat:curated.<kind>`` in the four
locales; a kind the interface does not know shows its stored English line.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field

# Frontend detects this prefix (substring match) to render an error turn in red.
# Keep it; the message intentionally carries NO traceback (avoids info leak).
ERROR_SENTINEL = "[ERROR_MESSAGE_SYSTEM]"

# The trace event of a curated turn, persisted with its message.
CURATED_ERROR_EVENT = "curated_error"


@dataclass(frozen=True)
class CuratedKind:
    """One curated sentence: its English line and whether it is a failure."""

    template: str
    error: bool = True
    params: tuple = field(init=False)

    def __post_init__(self) -> None:
        names = tuple(
            name for _, name, _, _ in string.Formatter().parse(self.template) if name is not None
        )
        object.__setattr__(self, "params", names)


CURATED_KINDS: dict[str, CuratedKind] = {
    # The engine refused the prompt on the model's window (a window the model
    # itself binds, one llama-server's fit chose, an inherited MAX_KV_SIZE).
    "context_overflow": CuratedKind(
        "This conversation no longer fits the model's context window "
        "(the request needs about {prompt_tokens} tokens; the window holds {context_tokens}). "
        "Start a new conversation or send less at once."
    ),
    # The same refusal when its numbers did not parse.
    "context_overflow_unsized": CuratedKind(
        "This conversation no longer fits the model's context window. "
        "Start a new conversation or send less at once."
    ),
    # The limit refused on is the loaded child's memory bound
    # (``BaseEngine.memory_bound_tokens``).
    "memory_overflow": CuratedKind(
        "This conversation no longer fits what your machine's memory can "
        "hold for this model (the request needs about {prompt_tokens} tokens; memory allows "
        "{context_tokens}). Start a new conversation, send less at once, or switch to a "
        "smaller model."
    ),
    # No first chunk within the budget (#573): retrying makes it worse, the
    # history only grows.
    "prefill_timeout": CuratedKind(
        "This model did not start answering within {minutes} minutes on this "
        "machine. Before writing a word it has to read everything this turn sends it -- "
        "the whole conversation so far -- and it did not get through that in time. "
        "Sending the same thing again will take longer, not less: start a new "
        "conversation, send less at once, or pick a smaller model."
    ),
    # Tokens flowed, then silence past the inter-chunk budget (#573).
    "decode_timeout": CuratedKind(
        "This model started answering, then went silent for "
        "{seconds} seconds, so the turn was stopped. Anything it had "
        "already written is kept above. Please try asking your question again."
    ),
    # A runaway agent hit AGENT_RECURSION_LIMIT with nothing to fall back to (#277).
    "loop_limit": CuratedKind(
        "I couldn't reach a final answer for this request "
        "(the model kept retrying without making progress). Please try rephrasing "
        "your question."
    ),
    # No answer text and nothing to fall back to (#554). Normal answers, not
    # failures: a failed turn is persisted without its reasoning, which is
    # exactly what explains these. ``length``: cut mid-reasoning (the line
    # never names the removed Max Tokens control); ``stop``: the model closed
    # its turn without writing an answer.
    "empty_answer_length": CuratedKind(
        "Generation stopped during reasoning, before an answer was written. "
        "Send a follow-up asking for the final answer directly, or lower the reasoning effort.",
        error=False,
    ),
    "empty_answer_stop": CuratedKind(
        "The model finished its turn without writing an answer. "
        "Send a follow-up asking it to continue.",
        error=False,
    ),
}


def _checked_params(kind: str, params: dict) -> dict:
    spec = CURATED_KINDS[kind]
    if set(params) != set(spec.params):
        raise ValueError(f"Curated kind {kind} takes {sorted(spec.params)}, got {sorted(params)}")
    return dict(params)


def _english_line(kind: str, params: dict) -> str:
    return CURATED_KINDS[kind].template.format(**_checked_params(kind, params))


def curated_text(kind: str, **params: int) -> str:
    """The English line of ``kind`` with its numbers, sentinel-prefixed for a
    failed turn."""
    text = _english_line(kind, params)
    return f"{ERROR_SENTINEL} {text}" if CURATED_KINDS[kind].error else text


def curated_answer(kind: str, **params: int) -> dict:
    """The answer event of a curated turn: the English line and its trace.

    ``curated: True`` marks a failed kind (it rides to the wire ``error``
    event, where the renderer logs it at WARNING instead of ERROR); an
    empty-answer line is a normal answer and carries only ``curated_error``.
    """
    spec = CURATED_KINDS[kind]
    event: dict = {
        "t": "answer",
        "text": curated_text(kind, **params),
        CURATED_ERROR_EVENT: {
            "kind": kind,
            "params": _checked_params(kind, params),
            "text": _english_line(kind, params),
        },
    }
    if spec.error:
        event["curated"] = True
    return event


def curated_trace_event(curated_error: dict) -> dict:
    """The trace event persisted for a curated turn's ``curated_error``."""
    return {
        "t": CURATED_ERROR_EVENT,
        "kind": curated_error["kind"],
        "params": dict(curated_error.get("params") or {}),
        "text": curated_error["text"],
    }
