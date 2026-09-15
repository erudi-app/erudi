"""Per-turn KB mode routing for conversation/arena (issue #84).

The KB mode is DERIVED from the model, never a user toggle:
  - KB attached & the size tier allows context & the model's tool calls are
    VERIFIED to work on this engine's wire (#298)
    -> AGENTIC: the KB is exposed as the ``search_knowledge_base`` tool and the
       model decides when to consult it (no systematic injection).
  - KB attached & tier allows & not verified -> SYSTEMATIC: excerpts retrieved
    up front and merged request-time (unchanged).
  - otherwise -> PLAIN: zero tools (#129) — plain chat never pays the
    tool-scaffolding cost, whatever the model's capability.

Agentic gating (#298): tool-calling reliability is a per-model WIRE property
(#273 matrix — same server, some templates parse, others leak raw JSON or get
swallowed, #295), so the blanket #288 kill switch became the tri-state
``config.KB_AGENTIC_MODE``:

    agentic iff should_use_kb(llm) AND (flag is True
                                        OR (flag is None AND supports_tools
                                            AND supports_tools_wire is True))

Flag None (default) = per-model routing on the verified capability; True =
force agentic (debug); False = force systematic (kill switch). A NULL/False
wire verdict always routes systematic.

Factored here so conversation and arena share one decision. Retrieval is
injected as a callable so each caller keeps its own failure policy
(conversation degrades to no-context, arena raises); it is only invoked in
systematic mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from src.agents.prompts import (
    answer_language_line,
    build_agent_system_prompt,
    build_kb_agentic_system_prompt,
    build_kb_context_block,
    build_kb_system_prompt,
)
from src.core.logging import logger
from src.utils.kb_utils import KbExcerpt
from src.utils.prompt_utils import get_prompting_strategy


@dataclass(frozen=True)
class TurnPlan:
    """The runner bundle for one turn: prompt, tools, KB block, tool context."""

    system_prompt: str
    tools: list
    kb_context_block: Optional[str]
    kb_language_line: str
    context: Optional[Any]  # TurnToolContext on tool-carrying turns, else None


def _param_size(llm) -> float:
    return llm.param_size if getattr(llm, "param_size", None) is not None else 2


def should_use_kb(llm) -> bool:
    """True when the model has a KB attached AND its size tier allows context."""
    if not getattr(llm, "is_attached_to_kb", False):
        return False
    return get_prompting_strategy(_param_size(llm)).get("use_kb_context", False)


def plan_turn(
    llm,
    *,
    question: str,
    retrieve: Callable[[], List[KbExcerpt]],
    custom_prompt: Optional[str] = None,
    starred_messages: Optional[List[str]] = None,
    web_search_enabled: bool = False,
    effort_section: Optional[str] = None,
) -> TurnPlan:
    """Decide the turn's mode and build the runner bundle.

    ``retrieve`` is only called in systematic mode: agentic mode defers
    retrieval to the model's tool call, and plain mode has no KB.

    ``effort_section`` is the graded reasoning instruction for this turn
    (1.1.2, ``src.agents.reasoning_effort``) or ``None`` when the level needs
    no instruction. Every mode takes it in the same slot -- after the tool
    regime, before the user's own instructions -- so the level means the same
    thing on a plain, a systematic and an agentic turn.

    ``web_search_enabled`` is the conversation's toggle (#310, copied from the
    global default at creation; arena passes the global setting directly). The
    web_search tool joins the turn iff the toggle is on AND the model is
    wire-capable — the SAME gate as agentic KB (#301): supports_tools AND
    supports_tools_wire is True. It rides plain turns (plain+web becomes a
    tool turn) and agentic-KB turns (both tools); the systematic path stays
    zero-tool, toggle or not — a non-wire model cannot execute tools, and the
    debug force-systematic flag deliberately keeps its zero-tool contract.
    The tool is ALWAYS exposed when gated in, online or not: offline turns get
    the tool's honest deterministic failure text instead of a missing tool.
    """
    # Deferred (#160): importing the tools module pulls the LangChain ``@tool``
    # machinery, needed on turns only — never at boot. Deterministic tools are
    # carried by KB turns only (the KB tool is added in agentic mode); plain
    # chat is zero-tool (#129) unless the web toggle turns it into a tool turn.
    from src.agents.tools import (
        TurnToolContext,
        calculator,
        search_knowledge_base,
        web_search,
    )
    from src.core import config

    base_tools = [calculator]

    # #298 agentic gate: the tri-state flag arbitrates, the per-model verified
    # wire capability decides in the default (flag None) state.
    flag = config.KB_AGENTIC_MODE
    supports_tools = bool(getattr(llm, "supports_tools", False))
    wire = getattr(llm, "supports_tools_wire", None)
    agentic = should_use_kb(llm) and (
        flag is True or (flag is None and supports_tools and wire is True)
    )

    # #310 web gate: strictly the verified wire capability — the KB tri-state
    # flag never bypasses it (a forced-agentic debug run on an unverified wire
    # still gets no web tool).
    wire_capable = supports_tools and wire is True
    web = web_search_enabled and wire_capable

    def _log_web(mode_allows_tools: bool) -> None:
        if not web_search_enabled:
            return
        if web and mode_allows_tools:
            logger.info(
                "Turn tools: web_search enabled "
                "(decided_by=web toggle on + verified wire capability)"
            )
        elif not wire_capable:
            logger.info(
                "Turn tools: web_search unavailable "
                "(model tool-call wire capability not verified)"
            )
        else:
            logger.info("Turn tools: web_search unavailable (systematic KB path is zero-tool)")

    if agentic:
        if flag is True:
            decided_by = "flag force-agentic (ERUDI_KB_AGENTIC=1)"
        else:
            decided_by = (
                "per-model verified wire capability "
                "(supports_tools=True, supports_tools_wire=True)"
            )
        budget = get_prompting_strategy(_param_size(llm))["kb_token_budget"]
        _log_web(mode_allows_tools=True)
        logger.info(
            f"Turn mode: agentic KB (kb_id={getattr(llm, 'kb_id', None)}, "
            f"decided_by={decided_by}, web_search={'on' if web else 'off'})"
        )
        return TurnPlan(
            system_prompt=build_kb_agentic_system_prompt(
                llm,
                custom_prompt=custom_prompt,
                starred_messages=starred_messages,
                web_search=web,
                effort_section=effort_section,
            ),
            tools=[*base_tools, search_knowledge_base, *([web_search] if web else [])],
            kb_context_block=None,
            kb_language_line="",
            context=TurnToolContext(
                kb_id=llm.kb_id,
                kb_token_budget=budget,
                web_token_budget=budget,
            ),
        )

    # Systematic: retrieve() encapsulates is_attached + tier + failure policy.
    excerpts = retrieve()
    if excerpts:
        # Name the gate that kept this KB turn off the agentic path (#298).
        if flag is False:
            decided_by = "flag force-systematic (ERUDI_KB_AGENTIC=0)"
        elif not supports_tools:
            decided_by = "model does not declare tool support"
        elif wire is None:
            decided_by = "tool-call wire capability unverified (NULL)"
        else:
            decided_by = "tool-call wire capability verified unreliable (False)"
        _log_web(mode_allows_tools=False)
        logger.info(
            f"Turn mode: systematic KB (kb_id={getattr(llm, 'kb_id', None)}, "
            f"excerpts={len(excerpts)}, decided_by={decided_by})"
        )
        return TurnPlan(
            system_prompt=build_kb_system_prompt(
                llm,
                custom_prompt=custom_prompt,
                starred_messages=starred_messages,
                effort_section=effort_section,
            ),
            # Zero tools on the systematic path (#288): the context is injected
            # directly, so the model only has to answer. Carrying the calculator
            # here bought nothing for document Q&A and made leak-prone models
            # (e.g. Llama 3.1 8B) wrap the answer in a raw calculator tool-call
            # JSON. Same #129 rationale that made plain chat zero-tool.
            tools=[],
            kb_context_block=build_kb_context_block(excerpts=excerpts, question=question),
            kb_language_line=answer_language_line(question),
            context=None,
        )

    if not getattr(llm, "is_attached_to_kb", False):
        plain_reason = "no KB attached"
    elif not should_use_kb(llm):
        plain_reason = "size tier disables KB context"
    else:
        plain_reason = "KB retrieval returned no excerpts"
    _log_web(mode_allows_tools=True)
    logger.info(
        f"Turn mode: plain (reason={plain_reason}, " f"web_search={'on' if web else 'off'})"
    )
    # Zero tools in plain mode (#129): the calculator was a demo tool whose
    # scaffolding cost hurt every model (and wrecked small ones) for no product
    # value outside KB turns. The web toggle is the ONE exception (#310):
    # a wire-capable model with the toggle on carries web_search alone.
    if web:
        budget = get_prompting_strategy(_param_size(llm))["kb_token_budget"]
        return TurnPlan(
            system_prompt=build_agent_system_prompt(
                llm,
                custom_prompt=custom_prompt,
                starred_messages=starred_messages,
                web_search=True,
                effort_section=effort_section,
            ),
            tools=[web_search],
            kb_context_block=None,
            kb_language_line="",
            context=TurnToolContext(web_token_budget=budget),
        )
    return TurnPlan(
        system_prompt=build_agent_system_prompt(
            llm,
            custom_prompt=custom_prompt,
            starred_messages=starred_messages,
            effort_section=effort_section,
        ),
        tools=[],
        kb_context_block=None,
        kb_language_line="",
        context=None,
    )
