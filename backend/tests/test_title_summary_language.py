"""Titles and compaction summaries follow the language of the user's messages
(#525, #690 items 4 and 5).

Real py3langid model (as in ``test_agent_language.py``); no LLM: prompts are
built and inspected, and the title stream runs on a scripted fake model.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents import runner as runner_module
from src.agents.language import is_greeting, language_name, strip_foreign_scripts
from src.domains.conversations.services import ConversationService, _sanitize_title

pytestmark = pytest.mark.unit

FRENCH = "Explique-moi comment fonctionne la photosynthèse chez les plantes vertes."
ENGLISH = "Explain how photosynthesis works in green plants, please."
SPANISH = "Explícame cómo funciona la fotosíntesis en las plantas verdes."
CHINESE = "请解释一下绿色植物是如何进行光合作用的。"


class TestLanguageName:
    @pytest.mark.parametrize(
        "text,name",
        [(FRENCH, "French"), (ENGLISH, "English"), (SPANISH, "Spanish"), (CHINESE, "Chinese")],
    )
    def test_names_the_language(self, text, name):
        assert language_name(text) == name

    def test_several_messages_are_read_together(self):
        assert language_name([FRENCH, "Et pour les algues ?", None]) == "French"

    def test_unconfident_text_has_no_name(self):
        assert language_name("ok") is None
        assert language_name("") is None
        assert language_name([]) is None


class TestTitlePromptLanguage:
    @pytest.mark.parametrize(
        "question,name",
        [(FRENCH, "French"), (ENGLISH, "English"), (SPANISH, "Spanish"), (CHINESE, "Chinese")],
    )
    @pytest.mark.parametrize("model_type", ["mistral", "qwen"])
    def test_names_the_language(self, test_db_session, question, name, model_type):
        prompt = ConversationService(test_db_session)._build_title_prompt_text(question, model_type)
        assert f"Write the title in {name}." in prompt

    @pytest.mark.parametrize("model_type", ["mistral", "qwen"])
    def test_unconfident_detection_keeps_the_generic_line_never_english(
        self, test_db_session, model_type
    ):
        prompt = ConversationService(test_db_session)._build_title_prompt_text("ok", model_type)
        assert "same language as the user's message" in prompt
        assert "Write the title in English." not in prompt

    def test_mistral_variant_is_kept(self, test_db_session):
        service = ConversationService(test_db_session)
        mistral = service._build_title_prompt_text(FRENCH, "mistral")
        generic = service._build_title_prompt_text(FRENCH, "qwen")
        assert "TITLE generator" in mistral
        assert "TITLE generator" not in generic


class TestGreeting:
    @pytest.mark.parametrize(
        "text",
        [
            "Bonjour !",
            "Hi",
            "hello!!",
            "Salut, ça va ?",
            "Hola, ¿qué tal?",
            "你好",
            "你好，你好吗？",
            "Bonjour ! Présente-toi en deux phrases.",
            "Hi, introduce yourself in one sentence.",
        ],
    )
    def test_greetings(self, text):
        assert is_greeting(text)

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "Bonjour, quel est le préavis de résiliation du contrat ?",
            "Hi, can you explain quantum entanglement?",
            "Hola, ¿cómo funciona la fotosíntesis?",
            FRENCH,
            "Hi " + "word " * 30,
        ],
    )
    def test_questions_are_not_greetings(self, text):
        assert not is_greeting(text)


class TestForeignScripts:
    def test_drops_han_from_a_french_title(self):
        assert (
            strip_foreign_scripts("Photosynthèse Explication简单", FRENCH)
            == "Photosynthèse Explication"
        )

    def test_keeps_a_title_in_the_scripts_of_the_message(self):
        assert strip_foreign_scripts("Photosynthèse 2", FRENCH) == "Photosynthèse 2"
        assert strip_foreign_scripts("光合作用 Photosynthesis", CHINESE + " Photosynthesis") == (
            "光合作用 Photosynthesis"
        )

    def test_a_title_wholly_in_another_script_becomes_empty(self):
        assert strip_foreign_scripts("光合作用解释", FRENCH) == ""

    def test_japanese_message_allows_kana_and_han(self):
        assert strip_foreign_scripts("光合成のしくみ", "光合成の仕組みを教えて") == "光合成のしくみ"


class TestTitleStream:
    """The stream's final title goes through the same checks as the stored one."""

    @staticmethod
    async def _run(test_db_session, mock_llm, monkeypatch, question, *model_output):
        from src.agents import runner as agent_runner
        from src.core import config
        from tests.test_conversations import _FakeEngine, _fake_chat_model

        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(agent_runner, "build_chat_model", _fake_chat_model(*model_output))
        service = ConversationService(test_db_session)
        conversation = service.create_conversation(
            llm_id=mock_llm.id, temperature=0.5, top_p=0.8, max_tokens=512
        )
        chunks = [c async for c in service.generate_title_stream(conversation.id, question)]
        stored = service.conversation_repo.get_conversation_by_id(conversation.id).name
        return "".join(chunks), stored

    async def test_mixed_script_title_is_cleaned_in_stream_and_storage(
        self, test_db_session, mock_llm, monkeypatch
    ):
        streamed, stored = await self._run(
            test_db_session, mock_llm, monkeypatch, FRENCH, "Photosynthèse Explication简单"
        )
        assert streamed == stored == "Photosynthèse Explication"

    async def test_greeting_gets_the_neutral_marker_without_a_model_call(
        self, test_db_session, mock_llm, monkeypatch
    ):
        # No scripted output: a model call would fail the scripted fake.
        streamed, stored = await self._run(
            test_db_session, mock_llm, monkeypatch, "Bonjour !", "Prénom Nom"
        )
        assert streamed == ""
        assert stored == "New Conversation"

    async def test_title_wholly_in_a_foreign_script_keeps_the_default(
        self, test_db_session, mock_llm, monkeypatch
    ):
        streamed, stored = await self._run(
            test_db_session, mock_llm, monkeypatch, FRENCH, "光合作用解释"
        )
        assert streamed == ""
        assert stored == "New Conversation"


def test_sanitize_title_is_unchanged_for_plain_titles():
    assert _sanitize_title("Compte De Pommes") == "Compte De Pommes"


class TestSummaryPromptLanguage:
    def test_template_has_a_language_slot_and_no_fixed_headings(self):
        template = runner_module.SUMMARY_PROMPT
        assert "{language_line}" in template
        assert "{messages}" in template
        assert "concrete fact" in template
        assert "**" not in template

    @pytest.mark.parametrize(
        "text,name",
        [(FRENCH, "French"), (ENGLISH, "English"), (SPANISH, "Spanish"), (CHINESE, "Chinese")],
    )
    def test_names_the_language(self, text, name):
        prompt = runner_module.build_summary_prompt([HumanMessage(content=text)])
        assert f"Write the whole summary in {name}" in prompt
        assert "{messages}" in prompt

    def test_unconfident_falls_back_to_the_generic_line(self):
        prompt = runner_module.build_summary_prompt([HumanMessage(content="ok")])
        assert "same language as the user's messages" in prompt
        assert "in English" not in prompt

    def test_the_previous_summary_and_the_model_do_not_decide_the_language(self):
        previous = HumanMessage(
            content=runner_module.SUMMARY_MESSAGE_PREFIX + "The user is called Anna."
        )
        messages = [previous, HumanMessage(content=FRENCH), AIMessage(content=ENGLISH * 3)]
        prompt = runner_module.build_summary_prompt(messages)
        assert "Write the whole summary in French" in prompt

    async def test_the_summary_call_carries_the_instruction(self):
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        seen = {}

        class _Model(FakeListChatModel):
            def _call(self, messages, *args, **kwargs):
                seen["prompt"] = messages if isinstance(messages, str) else str(messages)
                return super()._call(messages, *args, **kwargs)

        middleware = object.__new__(runner_module._summarization_middleware_class())
        middleware.model = _Model(responses=["Resume."])
        middleware.summary_prompt = runner_module.SUMMARY_PROMPT
        middleware._attempt_failures = []
        out = await middleware._summarize(None, [HumanMessage(content=FRENCH)], attempt="first")
        assert out
        assert "Write the whole summary in French" in str(seen["prompt"])
