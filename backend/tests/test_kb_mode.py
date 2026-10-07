"""Tests for the per-turn KB mode routing (issue #84, step 7).

The mode is derived from the model, never a user toggle. These cover the three
modes (plain / systematic / agentic), the NULL-capability fallback, and the
per-model verified wire routing (#298): with the tri-state flag unset, a KB
turn goes agentic iff the model declares tools (``supports_tools``) AND its
tool calls were verified to parse on this engine's wire
(``supports_tools_wire is True``).
"""

from types import SimpleNamespace

import pytest

from src.agents.kb_mode import plan_turn, should_use_kb
from src.agents.tools import calculator, search_knowledge_base
from src.core.config import parse_kb_agentic_flag
from src.utils.kb_utils import KbExcerpt

pytestmark = pytest.mark.unit


def _llm(**kw):
    base = dict(
        name="M",
        param_size=7.0,
        is_attached_to_kb=False,
        kb_id=None,
        supports_tools=None,
        supports_tools_wire=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _must_not_retrieve():
    raise AssertionError("agentic mode must not retrieve up front")


class TestShouldUseKb:
    def test_false_without_kb(self):
        assert should_use_kb(_llm()) is False

    def test_true_with_kb_and_medium_tier(self):
        assert should_use_kb(_llm(is_attached_to_kb=True)) is True


class TestPlanTurn:
    def test_plain_mode_no_kb_is_zero_tool_for_tool_capable_model(self):
        # #129: plain chat (no KB attached) carries NO tools at all, even when
        # the model supports native function calling.
        plan = plan_turn(_llm(supports_tools=True), question="hi", retrieve=lambda: [])
        assert plan.tools == []
        assert plan.kb_context_block is None and plan.context is None
        assert "search_knowledge_base" not in plan.system_prompt

    def test_plain_mode_no_kb_is_zero_tool_for_non_tool_capable_model(self):
        # #129: same zero-tool policy for models without tool support.
        plan = plan_turn(_llm(supports_tools=False), question="hi", retrieve=lambda: [])
        assert plan.tools == []
        assert plan.kb_context_block is None and plan.context is None

    def test_agentic_mode_when_enabled_and_tool_capable(self, monkeypatch):
        # Flag True = force-agentic (#288-era opt-in, kept as the debug state
        # of the #298 tri-state flag): the wire capability is bypassed.
        from src.core import config

        monkeypatch.setattr(config, "KB_AGENTIC_MODE", True)
        plan = plan_turn(
            _llm(is_attached_to_kb=True, kb_id=5, supports_tools=True),
            question="q",
            retrieve=_must_not_retrieve,
        )
        # Non-regression (#129): the agentic KB branch keeps BOTH tools.
        assert plan.tools == [calculator, search_knowledge_base]
        assert plan.context.kb_id == 5 and plan.context.kb_token_budget == 1000
        assert plan.kb_context_block is None
        # Composed prompt (#129): tier persona base + agentic KB section.
        assert "You are Erudi" in plan.system_prompt
        assert "call search_knowledge_base before answering" in plan.system_prompt

    def test_tool_capable_but_unverified_wire_defaults_to_systematic(self):
        # #298: with the flag unset (per-model routing), a model that declares
        # tools but whose wire capability is unverified (NULL) does NOT get the
        # search tool; it takes the systematic context-injection path.
        excerpts = [KbExcerpt(source_file="d.pdf", text="Le preavis est de 90 jours.")]
        plan = plan_turn(
            _llm(is_attached_to_kb=True, kb_id=5, supports_tools=True),
            question="preavis ?",
            retrieve=lambda: excerpts,
        )
        assert plan.tools == []  # #288: systematic path is zero-tool
        assert search_knowledge_base not in plan.tools
        assert plan.context is None
        assert plan.kb_context_block and "[Document: d.pdf]" in plan.kb_context_block

    def test_systematic_mode_when_not_tool_capable(self):
        excerpts = [KbExcerpt(source_file="d.pdf", text="Le préavis est de 90 jours.")]
        plan = plan_turn(
            _llm(is_attached_to_kb=True, kb_id=5, supports_tools=False),
            question="préavis ?",
            retrieve=lambda: excerpts,
        )
        # #288: systematic KB is zero-tool (the calculator caused tool-JSON
        # leaks on some models and added nothing for document Q&A).
        assert plan.tools == []
        assert plan.context is None
        assert plan.kb_context_block and "[Document: d.pdf]" in plan.kb_context_block
        # Composed prompt (#129): tier persona base + systematic KB section.
        assert "You are Erudi" in plan.system_prompt
        assert "excerpts from the user's documents" in plan.system_prompt

    def test_systematic_empty_pool_falls_back_to_plain(self):
        plan = plan_turn(
            _llm(is_attached_to_kb=True, kb_id=5, supports_tools=False),
            question="q",
            retrieve=lambda: [],
        )
        assert plan.kb_context_block is None and plan.context is None
        # The empty-pool fallback IS the plain mode: zero tools (#129).
        assert plan.tools == []

    def test_null_supports_tools_routes_systematic_never_agentic(self):
        # NULL (unknown capability) must behave like not-tool-capable.
        excerpts = [KbExcerpt(source_file="d.pdf", text="x")]
        plan = plan_turn(
            _llm(is_attached_to_kb=True, kb_id=5, supports_tools=None),
            question="q",
            retrieve=lambda: excerpts,
        )
        assert plan.context is None
        assert search_knowledge_base not in plan.tools


class TestPerModelWireRouting:
    """#298 routing truth table.

    agentic iff should_use_kb AND (flag is True
                                   OR (flag is None AND supports_tools
                                       AND supports_tools_wire is True))
    """

    def _plan(self, monkeypatch, *, flag, tools, wire, attached=True):
        from src.core import config

        monkeypatch.setattr(config, "KB_AGENTIC_MODE", flag)
        excerpts = [KbExcerpt(source_file="d.pdf", text="x")]
        return plan_turn(
            _llm(
                is_attached_to_kb=attached,
                kb_id=5,
                supports_tools=tools,
                supports_tools_wire=wire,
            ),
            question="q",
            retrieve=lambda: excerpts,
        )

    def _is_agentic(self, plan):
        return plan.context is not None and search_knowledge_base in plan.tools

    # ---- flag None: per-model routing (the new default) ----

    def test_flag_none_tools_and_verified_wire_is_agentic(self, monkeypatch):
        plan = self._plan(monkeypatch, flag=None, tools=True, wire=True)
        assert self._is_agentic(plan)
        assert plan.kb_context_block is None

    def test_flag_none_wire_false_is_systematic(self, monkeypatch):
        # Verified unreliable (e.g. Llama 3.1 8B leaks raw JSON on mlx wire).
        plan = self._plan(monkeypatch, flag=None, tools=True, wire=False)
        assert not self._is_agentic(plan)
        assert plan.kb_context_block is not None

    def test_flag_none_wire_null_is_systematic(self, monkeypatch):
        # Unverified (pre-#298 rows before the backfill lands) -> systematic.
        plan = self._plan(monkeypatch, flag=None, tools=True, wire=None)
        assert not self._is_agentic(plan)
        assert plan.kb_context_block is not None

    def test_flag_none_no_tool_support_is_systematic_even_with_wire(self, monkeypatch):
        # Wire True cannot outrank the template gate: no declared tools, no agent.
        for tools in (False, None):
            plan = self._plan(monkeypatch, flag=None, tools=tools, wire=True)
            assert not self._is_agentic(plan)

    # ---- flag True: force agentic (debug) ----

    def test_flag_true_forces_agentic_whatever_the_wire(self, monkeypatch):
        for wire in (True, False, None):
            plan = self._plan(monkeypatch, flag=True, tools=True, wire=wire)
            assert self._is_agentic(plan), f"wire={wire}"

    def test_flag_true_forces_agentic_even_without_declared_tools(self, monkeypatch):
        # Debug override: the flag exists to exercise the agentic path at will.
        plan = self._plan(monkeypatch, flag=True, tools=False, wire=False)
        assert self._is_agentic(plan)

    def test_flag_true_still_requires_a_kb(self, monkeypatch):
        from src.core import config

        monkeypatch.setattr(config, "KB_AGENTIC_MODE", True)
        plan = plan_turn(
            _llm(supports_tools=True, supports_tools_wire=True),
            question="q",
            retrieve=lambda: [],
        )
        assert plan.tools == [] and plan.context is None  # plain mode

    # ---- flag False: kill switch ----

    def test_flag_false_forces_systematic_even_fully_verified(self, monkeypatch):
        plan = self._plan(monkeypatch, flag=False, tools=True, wire=True)
        assert not self._is_agentic(plan)
        assert plan.kb_context_block is not None


class TestKbAgenticFlagParsing:
    """Tri-state ERUDI_KB_AGENTIC parsing (#298). Unset -> None (per-model
    routing, the default); 1/true -> force agentic; 0/false -> kill switch."""

    def test_unset_is_none(self):
        assert parse_kb_agentic_flag(None) is None

    def test_truthy_values(self):
        for raw in ("1", "true", "True", "TRUE", " 1 "):
            assert parse_kb_agentic_flag(raw) is True, raw

    def test_falsy_values(self):
        for raw in ("0", "false", "False", "FALSE", " 0 "):
            assert parse_kb_agentic_flag(raw) is False, raw

    def test_empty_and_garbage_fall_back_to_per_model(self):
        for raw in ("", "  ", "yes", "on", "2"):
            assert parse_kb_agentic_flag(raw) is None, raw


class TestWebSearchGate:
    """#310: the web_search tool joins the tools list iff the conversation's
    toggle is on AND the model is wire-capable (supports_tools AND
    supports_tools_wire is True — the SAME gate as agentic KB, #301). It
    applies on plain turns (plain+web becomes a tool turn) and on agentic-KB
    turns (both tools). Systematic-KB turns stay zero-tool: a non-wire model
    cannot execute tools, toggle or not."""

    def test_plain_toggle_on_wire_capable_gets_the_tool(self):
        from src.agents.tools import web_search

        plan = plan_turn(
            _llm(supports_tools=True, supports_tools_wire=True),
            question="latest news?",
            retrieve=lambda: [],
            web_search_enabled=True,
        )
        assert plan.tools == [web_search]
        assert plan.context is not None
        assert plan.context.kb_id is None
        assert plan.context.web_max_results == 5
        assert plan.context.web_token_budget == 1000  # medium tier budget
        assert "web_search" in plan.system_prompt

    def test_plain_toggle_off_stays_zero_tool(self):
        plan = plan_turn(
            _llm(supports_tools=True, supports_tools_wire=True),
            question="hi",
            retrieve=lambda: [],
            web_search_enabled=False,
        )
        assert plan.tools == [] and plan.context is None
        assert "web_search" not in plan.system_prompt

    @pytest.mark.parametrize(
        "tools,wire", [(True, None), (True, False), (False, True), (None, None)]
    )
    def test_plain_toggle_on_not_wire_capable_stays_zero_tool(self, tools, wire):
        plan = plan_turn(
            _llm(supports_tools=tools, supports_tools_wire=wire),
            question="hi",
            retrieve=lambda: [],
            web_search_enabled=True,
        )
        assert plan.tools == [] and plan.context is None
        assert "web_search" not in plan.system_prompt

    def test_agentic_kb_toggle_on_carries_both_tools(self):
        from src.agents.tools import web_search

        plan = plan_turn(
            _llm(
                is_attached_to_kb=True,
                kb_id=5,
                supports_tools=True,
                supports_tools_wire=True,
            ),
            question="q",
            retrieve=_must_not_retrieve,
            web_search_enabled=True,
        )
        assert plan.tools == [calculator, search_knowledge_base, web_search]
        assert plan.context.kb_id == 5 and plan.context.kb_token_budget == 1000
        assert plan.context.web_token_budget == 1000
        assert "search_knowledge_base" in plan.system_prompt
        assert "web_search" in plan.system_prompt

    def test_agentic_kb_toggle_off_keeps_kb_tools_only(self):
        plan = plan_turn(
            _llm(
                is_attached_to_kb=True,
                kb_id=5,
                supports_tools=True,
                supports_tools_wire=True,
            ),
            question="q",
            retrieve=_must_not_retrieve,
            web_search_enabled=False,
        )
        assert plan.tools == [calculator, search_knowledge_base]
        assert "web_search" not in plan.system_prompt

    def test_systematic_kb_stays_zero_tool_even_with_toggle_on(self):
        # Non-wire model with a KB: systematic injection; the toggle cannot
        # hand tools to a model that cannot execute them.
        excerpts = [KbExcerpt(source_file="d.pdf", text="x")]
        plan = plan_turn(
            _llm(is_attached_to_kb=True, kb_id=5, supports_tools=True),
            question="q",
            retrieve=lambda: excerpts,
            web_search_enabled=True,
        )
        assert plan.tools == []
        assert plan.kb_context_block is not None
        assert "web_search" not in plan.system_prompt

    def test_web_budget_follows_the_size_tier(self):
        from src.agents.tools import web_search

        plan = plan_turn(
            _llm(param_size=0.5, supports_tools=True, supports_tools_wire=True),
            question="q",
            retrieve=lambda: [],
            web_search_enabled=True,
        )
        assert plan.tools == [web_search]
        assert plan.context.web_token_budget == 400  # tiny tier budget

    def test_decision_is_logged(self, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="erudi"):
            plan_turn(
                _llm(supports_tools=True, supports_tools_wire=True),
                question="q",
                retrieve=lambda: [],
                web_search_enabled=True,
            )
        joined = " ".join(r.message for r in caplog.records)
        assert "web_search" in joined and "decided_by" in joined

    def test_unavailable_decision_is_logged(self, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="erudi"):
            plan_turn(
                _llm(supports_tools=True, supports_tools_wire=None),
                question="q",
                retrieve=lambda: [],
                web_search_enabled=True,
            )
        joined = " ".join(r.message for r in caplog.records)
        assert "web_search" in joined and "unavailable" in joined


@pytest.mark.parametrize("flag", [None, True, False])
def test_a_kb_context_block_always_comes_with_a_zero_tool_agent(monkeypatch, flag):
    """Invariant the first-hop ratio rests on: ``_KbContextMiddleware._merge``
    rewrites the LAST message as a user message, so a KB block on a request
    ending with a tool result would turn it into a user turn. A block
    therefore never rides a tool-carrying turn, and a first hop is exactly a
    request that ends with a user message."""
    import itertools

    from src.core import config

    monkeypatch.setattr(config, "KB_AGENTIC_MODE", flag)
    excerpts = [KbExcerpt(source_file="d.pdf", text="x")]
    for attached, tools, wire, web, found in itertools.product(
        (False, True), (None, False, True), (None, False, True), (False, True), (False, True)
    ):
        plan = plan_turn(
            _llm(
                is_attached_to_kb=attached,
                kb_id=5 if attached else None,
                supports_tools=tools,
                supports_tools_wire=wire,
            ),
            question="q",
            retrieve=(lambda: excerpts) if found else (lambda: []),
            web_search_enabled=web,
        )
        if plan.kb_context_block is not None:
            assert plan.tools == [], (attached, tools, wire, web, found)
