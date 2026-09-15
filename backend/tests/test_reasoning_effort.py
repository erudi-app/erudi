"""Reasoning-effort levels and their degradation table (1.1.2).

Five levels -- ``none / low / medium / high / xhigh``, default ``medium`` --
resolved against the PER-ARTIFACT lever verdict into an ``EffortPlan``:

* ``wire_effort``    -> the native ``reasoning_effort`` field sent to the child
                        server (both servers map ``"none"`` to
                        ``enable_thinking=false``);
* ``prompt_section`` -> the graded system-prompt instruction used when no
                        native lever can carry the level;
* ``degraded_from``  -> the level the plan could not honour natively (silent to
                        the user, logged once per turn at INFO).

The table itself is the contract, so it is asserted exhaustively: every level
against every lever, for a thinking and a non-thinking model.
"""

from __future__ import annotations

import pytest

from src.agents.reasoning_effort import (
    DEFAULT_REASONING_EFFORT,
    REASONING_EFFORT_LEVELS,
    EffortPlan,
    ReasoningLever,
    normalize_effort,
    resolve_effort_plan,
)

pytestmark = pytest.mark.unit


# ===================== vocabulary =====================


def test_the_five_levels_in_order():
    assert REASONING_EFFORT_LEVELS == ("none", "low", "medium", "high", "xhigh")


def test_the_default_is_medium():
    assert DEFAULT_REASONING_EFFORT == "medium"
    assert DEFAULT_REASONING_EFFORT in REASONING_EFFORT_LEVELS


@pytest.mark.parametrize("level", REASONING_EFFORT_LEVELS)
def test_every_level_normalizes_to_itself(level):
    assert normalize_effort(level) == level


@pytest.mark.parametrize("value", [None, "", "extra-high", "MEDIUM", "off", 3, object()])
def test_an_unusable_value_falls_back_to_the_default(value):
    # A row written by an older client, or a value that lost its meaning: the
    # turn must run at the default rather than fail.
    assert normalize_effort(value) == DEFAULT_REASONING_EFFORT


# ===================== the degradation table =====================


def _plan(level, lever, is_thinker):
    return resolve_effort_plan(level, lever, is_thinker)


class TestNativeEffortLever:
    """A template that READS ``reasoning_effort``: every level goes on the wire."""

    @pytest.mark.parametrize("level", ["low", "medium", "high", "xhigh"])
    def test_every_graded_level_is_wired_natively_and_says_nothing_else(self, level):
        plan = _plan(level, ReasoningLever.NATIVE_EFFORT, True)
        assert plan == EffortPlan(
            level=level, wire_effort=level, prompt_section=None, degraded_from=None
        )

    def test_none_is_wired_AND_instructed(self):
        # The one cell where both mechanisms ride together. llama-server reads
        # "none" as enable_thinking=false and ERASES the reasoning_effort
        # kwarg, so a template that only grades its own reasoning never hears
        # the request and falls back to its built-in default. The instruction
        # is the only lever left there, and a harmless prompt line where the
        # native path does work.
        plan = _plan("none", ReasoningLever.NATIVE_EFFORT, True)
        assert plan.wire_effort == "none"
        assert plan.prompt_section and "directly" in plan.prompt_section
        assert plan.degraded_from == "none"


class TestNativeToggleLever:
    """A template that only honours on/off (``enable_thinking``)."""

    def test_none_is_wired_because_both_servers_map_it_to_thinking_off(self):
        plan = _plan("none", ReasoningLever.NATIVE_TOGGLE, True)
        assert plan == EffortPlan(
            level="none", wire_effort="none", prompt_section=None, degraded_from=None
        )

    def test_medium_adds_nothing_at_all(self):
        # medium IS the model's natural behaviour: nothing on the wire, nothing
        # in the prompt -- the turn stays byte-identical to today's.
        plan = _plan("medium", ReasoningLever.NATIVE_TOGGLE, True)
        assert plan == EffortPlan(
            level="medium", wire_effort=None, prompt_section=None, degraded_from=None
        )

    @pytest.mark.parametrize("level", ["low", "high", "xhigh"])
    def test_the_other_levels_degrade_to_a_graded_instruction(self, level):
        plan = _plan(level, ReasoningLever.NATIVE_TOGGLE, True)
        assert plan.wire_effort is None
        assert plan.degraded_from == level
        assert plan.prompt_section and plan.prompt_section.isascii()
        # The toggle model already thinks: the instruction shapes HOW MUCH, it
        # never teaches the <think> protocol (that is the non-thinker tier).
        assert "<think>" not in plan.prompt_section


class TestNoLeverThinker:
    """A reasoning model whose template exposes no lever at all."""

    def test_none_is_a_best_effort_instruction_to_answer_directly(self):
        plan = _plan("none", ReasoningLever.NONE, True)
        assert plan.wire_effort is None
        assert plan.degraded_from == "none"
        assert plan.prompt_section and "directly" in plan.prompt_section

    def test_medium_adds_nothing_at_all(self):
        plan = _plan("medium", ReasoningLever.NONE, True)
        assert plan == EffortPlan(
            level="medium", wire_effort=None, prompt_section=None, degraded_from=None
        )

    @pytest.mark.parametrize("level", ["low", "high", "xhigh"])
    def test_the_other_levels_get_the_graded_instruction(self, level):
        plan = _plan(level, ReasoningLever.NONE, True)
        assert plan.wire_effort is None
        assert plan.degraded_from == level
        assert plan.prompt_section and "<think>" not in plan.prompt_section

    def test_the_four_graded_instructions_are_all_different(self):
        sections = {
            level: _plan(level, ReasoningLever.NONE, True).prompt_section
            for level in ("none", "low", "high", "xhigh")
        }
        assert len(set(sections.values())) == 4


class TestNoLeverNonThinker:
    """A model that does not reason: chain-of-thought is INDUCED, between tags."""

    def test_none_keeps_todays_behaviour_untouched(self):
        plan = _plan("none", ReasoningLever.NONE, False)
        assert plan == EffortPlan(
            level="none", wire_effort=None, prompt_section=None, degraded_from=None
        )

    @pytest.mark.parametrize("level", ["low", "medium", "high", "xhigh"])
    def test_every_other_level_induces_chain_of_thought_between_think_tags(self, level):
        plan = _plan(level, ReasoningLever.NONE, False)
        assert plan.wire_effort is None
        assert plan.degraded_from == level
        section = plan.prompt_section
        assert section and section.isascii()
        # The induced trace must be delimited: our ThinkSplitter separates it
        # from the answer, so the tags are the whole point.
        assert "<think>" in section and "</think>" in section

    def test_the_four_induced_instructions_are_all_different(self):
        sections = {
            level: _plan(level, ReasoningLever.NONE, False).prompt_section
            for level in ("low", "medium", "high", "xhigh")
        }
        assert len(set(sections.values())) == 4


class TestUnknownVerdict:
    """The probe FAILED -- which is not the same as "the template has no lever".

    An unrenderable template or an unreadable artifact says nothing about the
    model. Injecting on that ignorance is the dangerous direction: at the
    DEFAULT level it would teach a real reasoner, whose own protocol we simply
    failed to read, a second one. So an unknown verdict does nothing at all,
    at every level.
    """

    @pytest.mark.parametrize("level", REASONING_EFFORT_LEVELS)
    @pytest.mark.parametrize("is_thinker", [True, False])
    def test_every_level_is_a_full_no_op(self, level, is_thinker):
        assert resolve_effort_plan(level, ReasoningLever.UNKNOWN, is_thinker) == EffortPlan(
            level=level, wire_effort=None, prompt_section=None, degraded_from=None
        )


# ===================== cross-cutting properties =====================


@pytest.mark.parametrize("lever", list(ReasoningLever))
@pytest.mark.parametrize("level", REASONING_EFFORT_LEVELS)
@pytest.mark.parametrize("is_thinker", [True, False])
def test_the_two_mechanisms_ride_together_only_for_a_best_effort_none(lever, level, is_thinker):
    # Wiring a level natively and ALSO instructing the model about it would
    # normally say the same thing twice. The single exception is "none" on a
    # template that grades its own reasoning, where the native channel is not
    # reliably honoured end to end and the instruction is the backstop.
    plan = resolve_effort_plan(level, lever, is_thinker)
    both = bool(plan.wire_effort and plan.prompt_section)
    assert both == (level == "none" and lever is ReasoningLever.NATIVE_EFFORT)
    assert plan.level == level


@pytest.mark.parametrize("lever", list(ReasoningLever))
@pytest.mark.parametrize("level", REASONING_EFFORT_LEVELS)
@pytest.mark.parametrize("is_thinker", [True, False])
def test_degraded_from_is_set_exactly_when_a_prompt_section_carries_the_level(
    lever, level, is_thinker
):
    plan = resolve_effort_plan(level, lever, is_thinker)
    assert (plan.degraded_from is not None) == (plan.prompt_section is not None)


def test_an_unusable_level_is_normalized_by_the_resolver_too(monkeypatch):
    plan = resolve_effort_plan("extra-high", ReasoningLever.NATIVE_EFFORT, True)
    assert plan.level == DEFAULT_REASONING_EFFORT
    assert plan.wire_effort == DEFAULT_REASONING_EFFORT


def test_the_summary_plan_is_always_none_on_the_wire():
    from src.agents.reasoning_effort import NO_REASONING_PLAN

    assert NO_REASONING_PLAN.wire_effort == "none"
    # The utility paths compose no system prompt of ours, so a section there
    # would be inert: the wire value is all they can say.
    assert NO_REASONING_PLAN.prompt_section is None
