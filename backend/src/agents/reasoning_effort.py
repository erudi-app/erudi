"""Reasoning effort: five levels, one lever verdict, one degradation table (1.1.2).

The user picks HOW MUCH the model may deliberate before answering --
``none / low / medium / high / xhigh``, ``medium`` by default. What that costs
on the wire depends entirely on the ARTIFACT, so the level is resolved against
a per-artifact lever verdict (``src.engines.reasoning_lever``) into an
``EffortPlan`` carrying one of two mechanisms:

* ``wire_effort`` -- the native OpenAI ``reasoning_effort`` field. Both local
  servers read it: llama-server maps ``"none"`` to ``enable_thinking=false``
  and forwards any other value to the chat template as a kwarg
  (``server-common.cpp``), mlx_vlm normalizes it the same way
  (``_DISABLED_REASONING_EFFORTS``) and binds both ``reasoning_effort`` and
  its ``reasoning_strength`` alias.
* ``prompt_section`` -- a graded system-prompt instruction, used when no native
  lever can carry the level. Two tiers: a model that already reasons is told
  how DEEPLY to reason; a model that does not is taught to write its
  chain-of-thought between ``<think>`` tags, which our ``ThinkSplitter``
  separates from the answer exactly as it does a native reasoning trace.

They are alternatives: wiring a level natively AND instructing the model about
it would normally say the same thing twice. One cell rides both, on purpose --
``none`` on a template that grades its own reasoning, where llama-server erases
the effort kwarg it just read as "none" and the instruction is the only lever
left (see ``resolve_effort_plan``).

``degraded_from`` names the level a prompt section is standing in for. The
degradation is SILENT to the user (the picker UX lands in 1.1.3) and shows up
once per turn in the INFO log, which is what a field report is read against.

``medium`` is the pivot: on any model that reasons naturally it adds NOTHING
(no wire value, no instruction), so the default level leaves the request
byte-identical to what it was before this module existed. The one deliberate
exception is a NON-reasoning model, where every level above ``none`` induces a
chain of thought that did not exist before -- that is the feature. When the
lever verdict is ``UNKNOWN`` (the probe could not run), NO level changes
anything: acting on ignorance is the one failure mode worth designing against.

This module is pure: constants, texts and one table. The composition with the
engine-side verdict (``plan_reasoning_effort``) is the only impure entry point,
and it defers its engine import to call time.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

# The five levels, ordered from least to most deliberation. "xhigh" (not
# "extra-high"): it is the spelling the upstream servers and templates use.
REASONING_EFFORT_LEVELS = ("none", "low", "medium", "high", "xhigh")
DEFAULT_REASONING_EFFORT = "medium"


class ReasoningLever(str, Enum):
    """What the model's own chat template can be told about reasoning.

    Decided PER ARTIFACT from the template itself -- never from a family name
    or ``llm.type``, which a community fine-tune inherits from its parent while
    shipping a different template.
    """

    #: The template reads ``reasoning_effort``: every level goes on the wire.
    NATIVE_EFFORT = "native_effort"
    #: The template only honours on/off (``enable_thinking``).
    NATIVE_TOGGLE = "native_toggle"
    #: No lever at all: only the prompt can carry the level.
    NONE = "none"
    #: The probe FAILED -- which is not the same as ``NONE``. An unrenderable
    #: template or an unreadable artifact says nothing about the model, and
    #: acting on that ignorance is the dangerous direction: at the default
    #: level it would teach a real reasoner, whose own protocol we merely
    #: failed to read, a second one. An unknown verdict changes nothing.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EffortPlan:
    """How one turn delivers the requested effort level.

    Attributes:
        level: The normalized level the turn runs at.
        wire_effort: Value for the native ``reasoning_effort`` request field,
            or None when nothing goes on the wire.
        prompt_section: Graded instruction appended to the system prompt, or
            None when the level needs no instruction.
        degraded_from: The level ``prompt_section`` is standing in for, or None
            when the level was honoured natively (or needed nothing).
    """

    level: str
    wire_effort: Optional[str] = None
    prompt_section: Optional[str] = None
    degraded_from: Optional[str] = None


# Utility calls (conversation titles, the compaction summary) always run at
# ``none``: they are one-shot machine work on a tiny budget, and a reasoning
# model would spend all of it inside <think> (#266).
#
# The wire value is all these paths can say -- they compose no system prompt of
# ours, so a ``prompt_section`` here would be inert -- and it is not a
# guarantee. mlx_vlm honours it (thinking off, plus the value bound for the
# template). llama-server maps "none" to ``enable_thinking=false`` AND ERASES
# the ``reasoning_effort`` template kwarg, so a template that only grades its
# own reasoning hears neither and falls back to its built-in default. On that
# combination the suppression rests on what is left: the deliberately tiny
# output budget, and the ThinkSplitter keeping the reasoning out of the title.
NO_REASONING_PLAN = EffortPlan(level="none", wire_effort="none")


# --------------------------------------------------------------------------
# Graded instructions, tier 1: the model ALREADY reasons, but its template
# exposes no granular lever. The instruction shapes the DEPTH; it never
# teaches the <think> protocol (the model has its own). ASCII, impersonal,
# addressed to the model, one sentence -- long rule sheets leak verbatim into
# sub-4B prose (#129).
# --------------------------------------------------------------------------
_THINKER_SECTIONS = {
    "none": (
        "Answer directly, without deliberating first: give the final answer "
        "straight away, as briefly as the question allows."
    ),
    "low": ("Keep your thinking short: a few lines at most before you give the " "final answer."),
    "high": (
        "Think thoroughly before answering: work through the question step by "
        "step, check each intermediate result, and consider the cases that "
        "would change the answer."
    ),
    "xhigh": (
        "Think exhaustively before answering: approach the question from "
        "several angles, weigh the alternatives against each other, verify "
        "every intermediate result, and only then write the final answer."
    ),
}

# --------------------------------------------------------------------------
# Graded instructions, tier 2: the model does NOT reason. The chain of thought
# is INDUCED and must be delimited -- the tags are what let the app show the
# trace apart from the answer instead of printing deliberation at the user.
# --------------------------------------------------------------------------
_COT_PREFIX = "Before answering, think inside a thinking block: write <think>, "
_COT_SUFFIX = (
    "then </think>. Write the final answer after the closing tag, and do not "
    "repeat the reasoning in it."
)
_COT_SECTIONS = {
    "low": _COT_PREFIX + "then a line or two of reasoning, " + _COT_SUFFIX,
    "medium": _COT_PREFIX + "reason step by step through the question, " + _COT_SUFFIX,
    "high": (
        _COT_PREFIX + "reason step by step, check each intermediate result "
        "and the cases that would change the answer, " + _COT_SUFFIX
    ),
    "xhigh": (
        _COT_PREFIX + "explore several approaches, weigh them against each "
        "other, verify every intermediate result, " + _COT_SUFFIX
    ),
}


def normalize_effort(value) -> str:
    """The stored value as a usable level, falling back to the default.

    Anything unusable -- None, an empty string, a level a newer client wrote,
    a non-string -- resolves to ``medium`` rather than failing the turn: the
    effort level is a preference, never a precondition for answering.
    """
    if isinstance(value, str) and value in REASONING_EFFORT_LEVELS:
        return value
    return DEFAULT_REASONING_EFFORT


def resolve_effort_plan(level, lever: ReasoningLever, is_thinker: bool) -> EffortPlan:
    """The plan for ``level`` on a model with this ``lever`` verdict.

    The whole table, in one place:

    ==============  =========================  ================  ==============
    lever           level                      wire_effort       prompt_section
    ==============  =========================  ================  ==============
    UNKNOWN         any                        -                 -
    NATIVE_EFFORT   none                       "none"            thinker tier
    NATIVE_EFFORT   low / medium / high/xhigh  the level         -
    NATIVE_TOGGLE   none                       "none"            -
    NATIVE_TOGGLE   medium                     -                 -
    NATIVE_TOGGLE   low / high / xhigh         -                 thinker tier
    NONE, thinker   none                       -                 thinker tier
    NONE, thinker   medium                     -                 -
    NONE, thinker   low / high / xhigh         -                 thinker tier
    NONE, other     none                       -                 -
    NONE, other     low / medium / high/xhigh  -                 induced CoT
    ==============  =========================  ================  ==============

    Three "nothing at all" cells are load-bearing: ``medium`` on any reasoning
    model (its natural behaviour IS medium), ``none`` on a model that never
    reasons (there is nothing to turn off), and EVERY level on an unknown
    verdict. All leave the request exactly as it was before this feature.

    ``none`` is best-effort on every model that reasons, and says so twice on a
    NATIVE_EFFORT template: llama-server erases the effort kwarg when it reads
    "none", so a template that only grades its own reasoning never hears the
    request -- the instruction is then the only lever left, and a harmless
    prompt line on the engines where the native path does work. An always-on
    reasoner that reasons anyway is within its rights.
    """
    level = normalize_effort(level)

    if lever is ReasoningLever.UNKNOWN:
        # We failed to read this artifact. Do nothing rather than guess.
        return EffortPlan(level=level)

    if lever is ReasoningLever.NATIVE_EFFORT:
        if level == "none":
            return EffortPlan(
                level=level,
                wire_effort="none",
                prompt_section=_THINKER_SECTIONS["none"],
                degraded_from="none",
            )
        return EffortPlan(level=level, wire_effort=level)

    if lever is ReasoningLever.NATIVE_TOGGLE:
        if level == "none":
            # Both servers read "none" as thinking off -- that IS the toggle.
            return EffortPlan(level=level, wire_effort="none")
        if level == DEFAULT_REASONING_EFFORT:
            return EffortPlan(level=level)
        return EffortPlan(level=level, prompt_section=_THINKER_SECTIONS[level], degraded_from=level)

    # No lever: the prompt is the only channel left.
    if is_thinker:
        if level == DEFAULT_REASONING_EFFORT:
            return EffortPlan(level=level)
        return EffortPlan(level=level, prompt_section=_THINKER_SECTIONS[level], degraded_from=level)

    if level == "none":
        # A model that does not reason, asked not to reason: today's behaviour.
        return EffortPlan(level=level)
    return EffortPlan(level=level, prompt_section=_COT_SECTIONS[level], degraded_from=level)


def plan_reasoning_effort(llm, level) -> EffortPlan:
    """Resolve ``level`` against the artifact behind ``llm`` and log the result.

    The ONE composition point for a turn: reads the per-artifact lever verdict
    (engine handle caps for llama.cpp, template probe otherwise) and folds it
    with the requested level. Blocking (the probe may read a tokenizer off
    disk), so services call it through ``run_in_threadpool``.

    Emits exactly one INFO line per turn: with the picker still invisible in
    1.1.2, this log is the only place a field report can be read against to see
    which mechanism actually carried the level.
    """
    # Deferred: keeps this module importable (and pure) at boot, and keeps the
    # agents -> engines direction one-way.
    from src.core.logging import logger
    from src.engines.reasoning_lever import model_reasoning_lever

    verdict = model_reasoning_lever(getattr(llm, "link", None), llm_id=getattr(llm, "id", None))
    plan = resolve_effort_plan(level, verdict.lever, verdict.is_thinker)
    logger.info(
        # 'unset' vs the literal level "none": collapsing both to 'none' made
        # the field unreadable in the release recette (an xhigh plan that wires
        # NOTHING logged the same as a none plan that wires "none").
        f"Reasoning effort: level={plan.level}, lever={verdict.lever.value}, "
        f"thinker={verdict.is_thinker}, wire_effort={plan.wire_effort or 'unset'}, "
        f"degraded_from={plan.degraded_from or 'unset'}"
    )
    return plan
