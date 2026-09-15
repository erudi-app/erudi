"""Agent system-prompt construction.

- build_agent_system_prompt: size-tier prompt (reuses prompt_utils tiers).
- build_kb_system_prompt / build_kb_agentic_system_prompt: KB prompts
  COMPOSE instead of replacing (#129): the size-tier persona from
  build_system_prompt stays as the base — byte-identical to the plain
  path for the same model — and the KB regime is an APPENDED section
  (scoped agentic trigger / systematic excerpts contract), followed by
  the custom instructions then the starred messages. The old "document
  analyst" replacement prompt made models refuse everyday questions and
  over-trigger searches (48-conversation baseline eval).
- build_kb_context_block: the PER-TURN block (excerpts + grounding
  reminder + dynamic answer-language line) that the runner's middleware
  merges into the model request's last user message — system instructions
  dissolve over turn depth on small local models, the block stays glued
  to generation.
"""

import pytest

from src.agents.prompts import (
    answer_language_line,
    build_agent_system_prompt,
    build_kb_agentic_system_prompt,
    build_kb_context_block,
    build_kb_system_prompt,
)
from src.utils.kb_utils import KbExcerpt
from src.utils.prompt_utils import build_system_prompt

pytestmark = pytest.mark.unit


class _Llm:
    def __init__(self, name="Test 7B", param_size=7.0):
        self.name = name
        self.param_size = param_size


def test_system_prompt_uses_erudi_persona_and_drops_ltm():
    p = build_agent_system_prompt(_Llm(name="Qwen 7B", param_size=7.0))
    # The reworked tiers speak as Erudi (#129), not as the model itself.
    assert "You are Erudi" in p
    # Long-term memory now lives in the checkpointer, never injected here.
    assert "Summary of the conversation" not in p


def test_starred_messages_injected():
    p = build_agent_system_prompt(_Llm(param_size=7.0), starred_messages=["use async def"])
    assert "Important points" in p
    assert "use async def" in p


def test_custom_prompt_appended():
    p = build_agent_system_prompt(_Llm(param_size=7.0), custom_prompt="Speak like a pirate")
    assert "Additional instructions: Speak like a pirate" in p


def test_param_size_none_falls_back():
    # m4: defensive fallback when a seeded model has no param_size.
    p = build_agent_system_prompt(_Llm(param_size=None))
    assert isinstance(p, str) and len(p) > 0


def test_plain_prompt_is_unchanged_by_kb_composition():
    # Regression guard: composing the KB prompts on the persona base must
    # leave the plain path byte-identical to the bare tier persona — no KB
    # text may leak into KB-less conversations.
    p = build_agent_system_prompt(_Llm(name="Qwen 7B", param_size=7.0))
    assert p == build_system_prompt(model_name="Qwen 7B", size_category="medium")
    assert "search_knowledge_base" not in p
    assert "not in the documents" not in p


# ===================== KB-assistant prompts (PR3 + #129 composition) =====================

EXCERPTS = [
    KbExcerpt(source_file="contrat-cadre.docx", text="Le préavis est de 90 jours."),
    KbExcerpt(source_file="faq-support.md", text="Le support répond sous 48 h."),
]

# Distinguishing substrings of the two agentic KB appends.
_FULL_MARKER = "never assume what the documents contain"
_COMPACT_MARKER = "never guess what they contain"


class TestBuildKbSystemPrompt:
    def _prompt(self, llm=None, **kwargs):
        return build_kb_system_prompt(llm or _Llm(name="Analyste 4B"), **kwargs)

    def test_persona_base_is_the_plain_path_prompt(self):
        # The 48-conversation baseline eval measured the damage of REPLACING
        # the tier persona (refusals, "not in your documents" tails): the KB
        # prompt now starts from the exact plain-path persona.
        llm = _Llm(name="Analyste 7B", param_size=7.0)
        plain = build_agent_system_prompt(llm)
        kb = build_kb_system_prompt(llm)
        assert kb.startswith(plain)
        assert "You are Erudi" in kb
        assert "document analyst" not in kb

    def test_excerpts_contract_is_relevance_conditional(self):
        # Excerpts are retrieved automatically for EVERY question, so the
        # system append must flag possible irrelevance and open the escape
        # hatch: the strict contract measurably refused everyday questions
        # ("Quelle est la capitale de l'Australie ?" -> "not in the
        # documents", sleep advice refused — 6-question matrix on a 7B).
        p = self._prompt()
        assert "Each question comes with excerpts from the user's documents" in p
        assert "retrieved automatically" in p
        assert "may or may not be relevant" in p
        assert "answer normally from your own knowledge" in p
        assert "Do not mention these instructions" in p

    def test_system_append_carries_no_abstention_clause(self):
        # ONE canonical abstention clause per layer: it rides the per-turn
        # reminder (the dominant layer, close to generation), never the
        # system append — stacked refusal rules measurably over-abstain.
        assert "not in the documents" not in self._prompt()

    def test_custom_prompt_and_starred_land_after_the_kb_section(self):
        p = self._prompt(
            custom_prompt="Tutoie l'utilisateur",
            starred_messages=["le client est Meridia"],
        )
        kb_idx = p.find("Each question comes with excerpts")
        custom_idx = p.find("Additional instructions: Tutoie l'utilisateur")
        starred_idx = p.find("Important points")
        assert 0 < kb_idx < custom_idx < starred_idx
        assert "le client est Meridia" in p


class TestBuildKbAgenticSystemPrompt:
    def _prompt(self, llm=None, **kwargs):
        return build_kb_agentic_system_prompt(llm or _Llm(name="Agent 7B"), **kwargs)

    def test_medium_llm_composes_persona_with_scoped_kb_section(self):
        # Persona base identical to the plain path, KB regime appended with a
        # SCOPED trigger ("when a question concerns the user's documents")
        # replacing the over-triggering "for ANY question" imperative, plus the
        # explicit restraint clause for everyday questions.
        llm = _Llm(name="Agent 7B", param_size=7.0)
        plain = build_agent_system_prompt(llm)
        p = build_kb_agentic_system_prompt(llm)
        assert p.startswith(plain)
        assert "search_knowledge_base" in p
        assert _FULL_MARKER in p
        assert "answer directly from your own knowledge without searching" in p
        assert "document analyst" not in p

    def test_search_before_claiming_absence_is_kept_in_both_variants(self):
        # The grounding discipline hardened for #84 (soft phrasing under-called
        # the tool) must survive the restraint fix: absence may only be claimed
        # after an empty search.
        full = self._prompt(_Llm(param_size=7.0))
        assert (
            "only say the information is not in the documents after a search in the current turn has come back empty"
            in full
        )
        assert "search again instead of relying on what they said" in full
        assert "Ground document answers on the excerpts the tool returns" in full
        compact = self._prompt(_Llm(param_size=0.6))
        assert (
            "say the information is not in the documents only after a search in this turn finds nothing"
            in compact
        )
        assert "search again when you need them" in compact

    def test_tiny_tier_gets_the_compact_variant(self):
        p = self._prompt(_Llm(name="Petit 0.6B", param_size=0.6))
        assert _COMPACT_MARKER in p
        assert _FULL_MARKER not in p
        assert "search_knowledge_base" in p

    def test_medium_tier_gets_the_full_variant(self):
        p = self._prompt(_Llm(name="Agent 7B", param_size=7.0))
        assert _FULL_MARKER in p
        assert _COMPACT_MARKER not in p

    @pytest.mark.parametrize(
        "param_size,marker",
        [
            (0.6, _COMPACT_MARKER),  # tiny
            (3.0, _COMPACT_MARKER),  # small
            (7.0, _FULL_MARKER),  # medium
            (12.0, _FULL_MARKER),  # large
            (32.0, _FULL_MARKER),  # xlarge
        ],
    )
    def test_variant_follows_the_prompt_tier(self, param_size, marker):
        assert marker in self._prompt(_Llm(param_size=param_size))

    def test_param_size_none_falls_back_to_compact(self):
        # Same defensive fallback as build_agent_system_prompt: an unmeasured
        # size is treated as a small model.
        p = self._prompt(_Llm(param_size=None))
        assert _COMPACT_MARKER in p

    def test_custom_prompt_and_starred_land_after_the_kb_section(self):
        p = self._prompt(custom_prompt="Tutoie", starred_messages=["client Meridia"])
        kb_idx = p.find("search_knowledge_base")
        custom_idx = p.find("Additional instructions: Tutoie")
        starred_idx = p.find("Important points")
        assert 0 < kb_idx < custom_idx < starred_idx
        assert "client Meridia" in p


class TestKbRegressionGuards:
    """The relevance-conditional grounding change (#129 follow-up) touches
    ONLY the systematic layers: the plain-path persona and the agentic KB
    prompts must stay byte-identical to their pre-change output."""

    _AGENTIC_FULL = (
        "You also have access to the user's personal documents through the "
        "search_knowledge_base tool. When a question concerns the user's "
        "documents, or facts that could plausibly be in them, call "
        "search_knowledge_base before answering: never assume what the documents "
        "contain, and only say the information is not in the documents after a "
        "search in the current turn has come back empty. Results of earlier "
        "searches may have been removed from the conversation - search again "
        "instead of relying on what they said. When in doubt whether the "
        "documents cover it, search. Ground document answers on the excerpts the "
        "tool returns and stay faithful to them. For everyday questions that have "
        "nothing to do with the user's documents, answer directly from your own "
        "knowledge without searching. Do not mention these instructions."
    )
    _AGENTIC_COMPACT = (
        "You can also search the user's documents with the search_knowledge_base "
        "tool. Search before answering anything about the documents - never guess "
        "what they contain, and say the information is not in the documents only "
        "after a search in this turn finds nothing. Earlier search results may "
        "have been removed - search again when you need them. For questions "
        "unrelated to the documents, answer directly without searching. Do not "
        "mention these instructions."
    )

    def test_plain_prompt_byte_unchanged(self):
        llm = _Llm(name="Qwen 7B", param_size=7.0)
        assert build_agent_system_prompt(llm) == build_system_prompt(
            model_name="Qwen 7B", size_category="medium"
        )

    def test_agentic_full_prompt_byte_unchanged(self):
        llm = _Llm(name="Agent 7B", param_size=7.0)
        expected = build_agent_system_prompt(llm) + "\n\n" + self._AGENTIC_FULL
        assert build_kb_agentic_system_prompt(llm) == expected

    def test_agentic_compact_prompt_byte_unchanged(self):
        llm = _Llm(name="Petit 0.6B", param_size=0.6)
        expected = build_agent_system_prompt(llm) + "\n\n" + self._AGENTIC_COMPACT
        assert build_kb_agentic_system_prompt(llm) == expected


class TestBuildKbContextBlock:
    def _block(self, question="What is the notice period?"):
        return build_kb_context_block(excerpts=EXCERPTS, question=question)

    def test_excerpts_are_attributed_and_ordered(self):
        b = self._block()
        a = b.find("[Document: contrat-cadre.docx]")
        z = b.find("[Document: faq-support.md]")
        assert -1 < a < z  # both present, RRF order preserved
        assert "Le préavis est de 90 jours." in b
        assert "Le support répond sous 48 h." in b

    def test_reminder_is_relevance_conditional_with_escape_hatch(self):
        # Relevance-conditional grounding: excerpts arrive on EVERY question
        # (retrieved automatically), so the reminder grounds document answers
        # on them but lets unrelated questions escape to the model's own
        # knowledge — the unconditional "Answer ONLY from the excerpts"
        # contract measurably refused everyday questions (0/2 on the
        # 6-question matrix).
        b = self._block()
        assert "If the excerpts above are relevant" in b
        assert "answer normally from your own knowledge" in b
        assert "exactly as written" in b
        assert "source document" in b  # according-to attribution
        # The reminder sits AFTER the excerpts (close to generation).
        assert b.find("If the excerpts above are relevant") > b.find("[Document: faq-support.md]")

    def test_reminder_keeps_exactly_one_scoped_abstention_clause(self):
        # The abstention clause stays SINGLE and canonical (stacked refusal
        # rules over-abstain) and is now SCOPED to document questions: an
        # everyday question must never trigger "not in the documents".
        en = self._block()
        assert en.count("not in the documents") == 1
        assert "asks about the user's documents" in en
        fr = self._block(question="Quel est le préavis de résiliation du contrat ?")
        assert fr.count("ne figure pas dans les documents") == 1
        assert "porte sur les documents de l'utilisateur" in fr

    def test_calculator_tool_reference_gone_arithmetic_caution_stays(self):
        # #289 removed the calculator from the systematic path: the reminder
        # must not point the model at a tool that no longer exists here.
        # The write-out-the-operation caution survives.
        en = self._block()
        fr = self._block(question="Quel est le préavis de résiliation du contrat ?")
        assert "calculator" not in en
        assert "calculator" not in fr
        assert "Never do mental arithmetic" in en
        assert "write out the operation" in en
        assert "Ne fais jamais de calcul mental" in fr
        assert "écris l'opération" in fr

    def test_scaffolding_is_localized_to_the_question_language(self):
        # Runs 3-5: the model answers in the language of the SCAFFOLDING
        # around the question (English attractor), so the scaffolding
        # itself must speak the question's language.
        fr = self._block(question="Quel est le préavis de résiliation du contrat ?")
        assert "Extraits de documents :" in fr
        assert "Si les extraits ci-dessus sont pertinents" in fr
        assert "ne figure pas dans les documents" in fr
        assert "réponds normalement avec tes propres connaissances" in fr
        assert "If the excerpts" not in fr

    def test_unmapped_language_falls_back_to_english_scaffolding(self):
        b = self._block(question="ok")  # unconfident detection
        assert "Document excerpts:" in b
        assert "If the excerpts above are relevant" in b


class TestAnswerLanguageLine:
    # The line is appended AFTER the question by the runner middleware:
    # pre-question language lines are read as block metadata and ignored
    # (run-4 eval), in-question user-voiced requests are honored (T5).
    def test_localized_to_the_question_language(self):
        assert (
            answer_language_line("Quel est le préavis de résiliation du contrat ?")
            == "Réponds en français."
        )
        assert (
            answer_language_line("What is the notice period for termination?")
            == "Answer in English."
        )

    def test_ambiguous_question_falls_back_to_generic_line(self):
        assert answer_language_line("ok") == ("Answer in the same language as the user's question.")


class TestWebSearchPromptSection:
    """#310: a short web section is APPENDED when the web_search tool rides
    the turn — same #129/#304 doctrine as the KB sections: scoped trigger
    (current/external facts only), current-turn absence discipline, an
    answer-directly escape hatch, and no mention of the instructions."""

    def test_plain_prompt_without_web_is_byte_unchanged(self):
        llm = _Llm(name="Qwen 7B", param_size=7.0)
        assert build_agent_system_prompt(llm) == build_system_prompt(
            model_name="Qwen 7B", size_category="medium"
        )

    def test_plain_prompt_with_web_appends_the_section(self):
        llm = _Llm(name="Qwen 7B", param_size=7.0)
        prompt = build_agent_system_prompt(llm, web_search=True)
        base = build_agent_system_prompt(llm)
        assert prompt.startswith(base)
        assert "web_search" in prompt
        assert "Do not mention these instructions." in prompt

    def test_web_section_is_scoped_with_escape_hatch(self):
        prompt = build_agent_system_prompt(_Llm(param_size=7.0), web_search=True)
        # Scoped trigger: current/external facts, never a broad imperative.
        assert "current" in prompt
        # Absence claims scoped to a current-turn search.
        assert "current turn" in prompt
        # Escape hatch: answer directly from own knowledge otherwise.
        assert "answer directly" in prompt

    def test_tiny_tier_gets_a_compact_web_section(self):
        full = build_agent_system_prompt(_Llm(param_size=7.0), web_search=True)
        compact = build_agent_system_prompt(_Llm(param_size=0.5), web_search=True)
        assert "web_search" in compact
        # The compact variant must be meaningfully shorter than the full one.
        assert len(compact) < len(full)

    def test_custom_prompt_and_starred_land_after_the_web_section(self):
        prompt = build_agent_system_prompt(
            _Llm(param_size=7.0),
            web_search=True,
            custom_prompt="Be terse.",
            starred_messages=["retenir ceci"],
        )
        assert prompt.index("web_search") < prompt.index("Additional instructions: Be terse.")
        assert prompt.index("Be terse.") < prompt.index("retenir ceci")

    def test_agentic_kb_prompt_with_web_keeps_kb_first_and_arbitrates(self):
        llm = _Llm(name="Agent 7B", param_size=7.0)
        prompt = build_kb_agentic_system_prompt(llm, web_search=True)
        # KB section first, web section after.
        assert prompt.index("search_knowledge_base tool") < prompt.index("web_search tool")
        # Arbitration: document questions go to the KB tool, not the web.
        assert "use search_knowledge_base, not web_search" in prompt

    def test_agentic_kb_prompt_without_web_is_byte_unchanged(self):
        llm = _Llm(name="Agent 7B", param_size=7.0)
        expected = build_agent_system_prompt(llm) + "\n\n" + TestKbRegressionGuards._AGENTIC_FULL
        assert build_kb_agentic_system_prompt(llm) == expected

    def test_web_only_prompt_carries_no_arbitration_clause(self):
        prompt = build_agent_system_prompt(_Llm(param_size=7.0), web_search=True)
        assert "search_knowledge_base" not in prompt


# ===================== reasoning-effort section (1.1.2) =====================


class TestReasoningEffortSection:
    """Where the graded reasoning instruction sits, and what it costs when absent.

    The section is the fallback channel for an effort level the model's chat
    template carries no native lever for. It takes ONE slot everywhere -- after
    the tool regime (how to use the tools comes first), before the user's own
    instructions (which stay the last word) -- so a level means the same thing
    on a plain, a systematic and an agentic turn.
    """

    _SECTION = "Think thoroughly before answering."

    def test_no_section_leaves_the_plain_prompt_byte_identical(self):
        llm = _Llm(name="Qwen 7B", param_size=7.0)
        assert build_agent_system_prompt(llm, effort_section=None) == build_agent_system_prompt(llm)

    def test_no_section_leaves_every_composed_prompt_byte_identical(self):
        llm = _Llm(name="Qwen 7B", param_size=7.0)
        for build in (build_kb_system_prompt, build_kb_agentic_system_prompt):
            assert build(llm, effort_section=None) == build(llm)
        assert build_agent_system_prompt(
            llm, web_search=True, effort_section=None
        ) == build_agent_system_prompt(llm, web_search=True)

    def test_the_plain_prompt_appends_the_section_after_the_persona(self):
        llm = _Llm(name="Qwen 7B", param_size=7.0)
        prompt = build_agent_system_prompt(llm, effort_section=self._SECTION)
        assert prompt == build_agent_system_prompt(llm) + "\n\n" + self._SECTION

    def test_the_plain_prompt_keeps_the_custom_instructions_last(self):
        prompt = build_agent_system_prompt(
            _Llm(param_size=7.0), effort_section=self._SECTION, custom_prompt="Be terse."
        )
        assert prompt.index(self._SECTION) < prompt.index("Additional instructions: Be terse.")

    def test_the_section_sits_between_the_tool_regime_and_the_custom_prompt(self):
        prompt = build_kb_agentic_system_prompt(
            _Llm(name="Agent 7B", param_size=7.0),
            web_search=True,
            effort_section=self._SECTION,
            custom_prompt="Be terse.",
            starred_messages=["retenir ceci"],
        )
        assert prompt.index("search_knowledge_base tool") < prompt.index(self._SECTION)
        assert prompt.index("web_search tool") < prompt.index(self._SECTION)
        assert prompt.index(self._SECTION) < prompt.index("Additional instructions: Be terse.")
        assert prompt.index("Be terse.") < prompt.index("retenir ceci")

    def test_the_systematic_prompt_takes_the_same_slot(self):
        prompt = build_kb_system_prompt(
            _Llm(name="Analyste 4B", param_size=4.0),
            effort_section=self._SECTION,
            custom_prompt="Be terse.",
        )
        assert prompt.index("excerpts") < prompt.index(self._SECTION)
        assert prompt.index(self._SECTION) < prompt.index("Additional instructions: Be terse.")


class TestPlanTurnCarriesTheEffortSection:
    """Every turn mode hands the section to its prompt builder."""

    def _plan(self, llm, **kwargs):
        from src.agents.kb_mode import plan_turn

        return plan_turn(llm, question="q", retrieve=lambda: [], **kwargs)

    def test_plain_turn(self):
        plan = self._plan(_Llm(param_size=7.0), effort_section="SECTION-MARKER")
        assert "SECTION-MARKER" in plan.system_prompt

    def test_plain_turn_without_a_section_is_unchanged(self):
        llm = _Llm(param_size=7.0)
        assert self._plan(llm).system_prompt == build_agent_system_prompt(llm)
