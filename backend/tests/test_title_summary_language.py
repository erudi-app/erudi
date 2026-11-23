"""Titles and compaction summaries follow the language of the user's messages
(#525, #690 items 4 and 5).

Real py3langid model (as in ``test_agent_language.py``); no LLM: prompts are
built and inspected, and the title stream runs on a scripted fake model.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents import runner as runner_module
from src.agents.language import (
    TITLE_LANGUAGE_LINES,
    is_greeting,
    language_name,
    strip_foreign_scripts,
)
from src.domains.conversations.services import ConversationService, _sanitize_title

pytestmark = pytest.mark.unit

FRENCH = "Explique-moi comment fonctionne la photosynthèse chez les plantes vertes."
ENGLISH = "Explain how photosynthesis works in green plants, please."
SPANISH = "Explícame cómo funciona la fotosíntesis en las plantas verdes."
CHINESE = "请解释一下绿色植物是如何进行光合作用的。"
GERMAN = "Erkläre mir bitte, wie die Photosynthese bei grünen Pflanzen funktioniert."
ITALIAN = "Spiegami come funziona la fotosintesi nelle piante verdi, per favore."
PORTUGUESE = "Explique-me como funciona a fotossíntese nas plantas verdes, por favor."
DUTCH = "Leg me alsjeblieft uit hoe de fotosynthese bij groene planten werkt."
RUSSIAN = "Объясни, пожалуйста, как работает фотосинтез у зелёных растений."


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

    @pytest.mark.parametrize("model_type", ["mistral", "qwen"])
    @pytest.mark.parametrize(
        "question,line",
        [
            (FRENCH, "Écris le titre en français."),
            (SPANISH, "Escribe el título en español."),
            (CHINESE, "用中文写标题。"),
            (GERMAN, "Schreibe den Titel auf Deutsch."),
            (ITALIAN, "Scrivi il titolo in italiano."),
            (PORTUGUESE, "Escreva o título em português."),
            (DUTCH, "Schrijf de titel in het Nederlands."),
        ],
    )
    def test_title_prompt_carries_the_instruction_in_the_target_language(
        self, test_db_session, question, line, model_type
    ):
        prompt = ConversationService(test_db_session)._build_title_prompt_text(question, model_type)
        assert line in prompt
        # The last instruction: only the question itself follows it.
        tail = prompt.split(line, 1)[1]
        assert question in tail
        assert tail.replace(question, "").strip(" \n:") in ("", "User message", "Message")
        assert "Examples" not in tail

    def test_mistral_french_separates_the_examples_from_the_line(self, test_db_session):
        prompt = ConversationService(test_db_session)._build_title_prompt_text(FRENCH, "mistral")
        assert "Compte De Pommes\n\nÉcris le titre en français.\n" in prompt

    def test_the_default_prompt_labels_the_question_after_the_line(self, test_db_session):
        prompt = ConversationService(test_db_session)._build_title_prompt_text(FRENCH, "qwen")
        assert prompt.endswith(f"Écris le titre en français.\nMessage:\n{FRENCH}")

    @pytest.mark.parametrize("model_type", ["mistral", "qwen"])
    def test_a_language_without_a_line_keeps_every_example(self, test_db_session, model_type):
        prompt = ConversationService(test_db_session)._build_title_prompt_text(RUSSIAN, model_type)
        assert "Write the title in Russian." in prompt
        assert "Pizza Recipe" in prompt
        assert "Compte De Pommes" in prompt
        assert not any(line in prompt for line in TITLE_LANGUAGE_LINES.values())

    def test_an_english_prompt_is_unchanged(self, test_db_session):
        service = ConversationService(test_db_session)
        examples = (
            "Examples (user question -> title):\n"
            "give me pizza recipe -> Pizza Recipe\n"
            "google founding team members -> Google Founding Team\n"
            "what's the capital of japan -> Japan Capital\n"
            "female of the pig -> Pig Female Name\n"
            "combien de pommes me reste-t-il -> Compte De Pommes\n"
        )
        language = (
            "Write the title in the same language as the user's message.\n"
            "Write the title in English.\n"
        )
        assert service._build_title_prompt_text(ENGLISH, "mistral") == (
            "You are a TITLE generator. Produce ONLY a very short title (2–4 words maximum).\n"
            "Rules: only the title text; Title Case; no question mark; no quotes; no emojis; "
            "no hashtags; no code; no trailing punctuation; never answer the question; "
            "if empty/URL/noise => output nothing.\n"
            f"{language}"
            f"User message: {ENGLISH}\n"
            "Do not answer the question, only create a relevant title. "
            "Do NOT add quotes around the title.\n"
            f"{examples}"
        )
        assert service._build_title_prompt_text(ENGLISH, "qwen") == (
            "You are a very-short-title generator. Return ONLY a concise title. "
            "No punctuation (except apostrophes in possessives), no quotes, no hashtags, "
            "no emojis, no trailing filler words. Capitalize important words. "
            "The title shouldn't be a question.\n"
            "Do not answer the question, only create a relevant title.\n"
            f"{language}"
            "If the message is empty or meaningless, return nothing.\n"
            f"{examples}"
            "Format: just the title, nothing else.\n\n"
            f"Create a 2-to-4-word title for:\n{ENGLISH}"
        )

    @pytest.mark.parametrize("model_type", ["mistral", "qwen"])
    def test_title_prompt_has_no_english_examples_for_a_french_question(
        self, test_db_session, model_type
    ):
        prompt = ConversationService(test_db_session)._build_title_prompt_text(FRENCH, model_type)
        for english in ("Pizza Recipe", "Google Founding Team", "Japan Capital", "Pig Female"):
            assert english not in prompt
        assert "Compte De Pommes" in prompt

    @pytest.mark.parametrize("model_type", ["mistral", "qwen"])
    def test_no_examples_next_to_another_named_language(self, test_db_session, model_type):
        prompt = ConversationService(test_db_session)._build_title_prompt_text(SPANISH, model_type)
        assert "Pizza Recipe" not in prompt
        assert "Compte De Pommes" not in prompt

    @pytest.mark.parametrize("model_type", ["mistral", "qwen"])
    def test_an_english_question_keeps_its_examples(self, test_db_session, model_type):
        prompt = ConversationService(test_db_session)._build_title_prompt_text(ENGLISH, model_type)
        assert "Pizza Recipe" in prompt
        assert "Compte De Pommes" in prompt
        assert "Écris le titre" not in prompt

    @pytest.mark.parametrize("model_type", ["mistral", "qwen"])
    def test_unconfident_detection_keeps_examples_and_no_localized_line(
        self, test_db_session, model_type
    ):
        prompt = ConversationService(test_db_session)._build_title_prompt_text("ok", model_type)
        assert "Pizza Recipe" in prompt
        assert "Écris le titre" not in prompt

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
        from tests.test_conversations import _fake_chat_model, _FakeEngine

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
        # A model call would store the scripted "Prénom Nom".
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
        middleware.model = _Model(
            responses=[
                "L'utilisateur a pose une question en francais sur son contrat de travail ; l'assistant a explique les conditions de resiliation et les delais de preavis applicables."
            ]
        )
        middleware.summary_prompt = runner_module.SUMMARY_PROMPT
        middleware._attempt_failures = []
        out = await middleware._summarize(None, [HumanMessage(content=FRENCH)], attempt="first")
        assert out
        assert "Write the whole summary in French" in str(seen["prompt"])


ENGLISH_BIO = (
    "John Smith was born in Chicago and studied physics at the university. "
    "He worked for many years at a laboratory where he published several papers "
    "about the properties of light and of the materials that carry it."
)
GERMAN_TEXT = "Der Vertrag endet am Ende des Monats, wenn keine Kuendigung eingeht."
TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "main.py", line 3, in <module>\n'
    "    print(undefined_name)\nNameError: name 'undefined_name' is not defined"
)


class TestOwnWordsOnly:
    """Pasted or attached content never decides the language (review item 1)."""

    def test_attachment_block_is_ignored_in_a_summary(self):
        message = HumanMessage(
            content=f"[Attached file: bio.txt]\n{ENGLISH_BIO}\n[End of attached file]\n\n"
            "Résume ce document en trois points."
        )
        prompt = runner_module.build_summary_prompt([message])
        assert "Write the whole summary in French" in prompt

    def test_a_pasted_block_after_the_instruction_is_ignored(self):
        assert language_name("Résume ce texte : " + ENGLISH_BIO) in (None, "French")
        assert language_name("Traduis en anglais : " + GERMAN_TEXT) in (None, "French")

    def test_a_traceback_is_ignored_in_a_title(self):
        assert language_name("Pourquoi cette erreur ?\n" + TRACEBACK) in (None, "French")

    def test_fenced_code_urls_and_quotes_are_ignored(self):
        fenced = "Pourquoi cette fonction échoue-t-elle ?\n```python\n" + ENGLISH_BIO + "\n```"
        assert language_name(fenced) == "French"
        url = "Que contient cette page https://en.wikipedia.org/wiki/The_English_Language ?"
        assert language_name(url) == "French"
        quoted = "Que penses-tu de ce passage ?\n> " + ENGLISH_BIO.replace(". ", ".\n> ")
        assert language_name(quoted) in (None, "French")

    def test_never_the_attachments_language_when_own_words_are_short(self):
        short = "[Attached file: a.txt]\n" + ENGLISH_BIO + "\n[End of attached file]\n\nok"
        assert language_name(short) is None

    def test_summary_takes_the_majority_of_user_messages(self):
        messages = [
            HumanMessage(content=FRENCH),
            HumanMessage(content="Et pour les algues, comment cela se passe-t-il exactement ?"),
            HumanMessage(content="Thanks"),
        ]
        assert "Write the whole summary in French" in runner_module.build_summary_prompt(messages)

    def test_one_long_english_attachment_does_not_outvote_french_messages(self):
        messages = [
            HumanMessage(content=FRENCH),
            HumanMessage(
                content=f"[Attached file: bio.txt]\n{ENGLISH_BIO * 3}\n[End of attached file]\n\n"
                "Résume ce document en trois points."
            ),
        ]
        assert "Write the whole summary in French" in runner_module.build_summary_prompt(messages)


class TestGreetingFalsePositives:
    @pytest.mark.parametrize(
        "text",
        [
            "Hi, introduce yourself and explain quantum physics",
            "Hello, who are you? Tell me about Paris",
            "Hey, who are you voting for?",
            "Hello there, who are you in Star Wars?",
            "Bonjour ! Présente-toi et explique la photosynthèse",
            "Salut, qui es-tu ? Résume ce PDF",
        ],
    )
    def test_real_content_after_a_greeting_is_titled(self, text):
        assert not is_greeting(text)

    @pytest.mark.parametrize(
        "text",
        [
            "Bonjour ! Présente-toi en deux phrases.",
            "Hi, introduce yourself briefly please",
            "Hola, preséntate en dos frases por favor",
            "Salut, qui es-tu ?",
            "Hello, who are you?",
            "Bonjour, présente-toi brièvement s'il te plaît",
        ],
    )
    def test_short_introduction_requests_are_greetings(self, text):
        assert is_greeting(text)


class TestCheckedTitle:
    @pytest.mark.parametrize(
        "title,question,expected",
        [
            ("数学：基础", FRENCH, ""),
            ("« 数学 »", FRENCH, ""),
            ("Λ-Calcul Expliqué", FRENCH, "Λ-Calcul Expliqué"),
            ("µ-Controller", FRENCH, "µ-Controller"),
            ("Photosynthèse：简单", FRENCH, "Photosynthèse"),
            ("«Photosynthèse»简单", FRENCH, "Photosynthèse"),
        ],
    )
    def test_checked_title(self, title, question, expected):
        from src.domains.conversations.services import _checked_title

        assert _checked_title(title, question) == expected

    @pytest.mark.parametrize(
        "title,question,expected",
        [
            ("iPhone 截图方法", "如何在手机上截图？", "iPhone 截图方法"),
            ("Docker 容器入门", "请介绍一下容器是什么", "Docker 容器入门"),
            ("Пример Python", "Объясни мне как работает цикл в языке", "Пример Python"),
            ("Pi π Explained", ENGLISH, "Pi π Explained"),
        ],
    )
    def test_latin_and_lone_greek_letters_are_kept(self, title, question, expected):
        assert strip_foreign_scripts(title, question) == expected

    def test_a_greek_word_is_still_dropped_from_a_french_title(self):
        assert strip_foreign_scripts("Photosynthèse φωτοσύνθεση", FRENCH) == "Photosynthèse"


class TestTitleStreamFailure:
    async def test_a_failure_mid_generation_stores_the_default_name(
        self, test_db_session, mock_llm, monkeypatch
    ):
        from src.agents import runner as agent_runner
        from src.core import config
        from tests.test_conversations import _FakeEngine

        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)

        async def _boom(self, **kwargs):
            yield "Photosynthèse"
            raise RuntimeError("model died")

        monkeypatch.setattr(agent_runner.AgentRunner, "astream_oneshot", _boom)
        service = ConversationService(test_db_session)
        conversation = service.create_conversation(
            llm_id=mock_llm.id, temperature=0.5, top_p=0.8, max_tokens=512
        )
        with pytest.raises(RuntimeError):
            async for _ in service.generate_title_stream(conversation.id, FRENCH):
                pass
        assert (
            service.conversation_repo.get_conversation_by_id(conversation.id).name
            == "New Conversation"
        )


class TestTitlePromptExampleBias:
    @pytest.mark.parametrize("model_type", ["mistral", "qwen"])
    def test_the_french_example_sentence_is_dropped_when_a_language_is_named(
        self, test_db_session, model_type
    ):
        service = ConversationService(test_db_session)
        named = service._build_title_prompt_text(ENGLISH, model_type)
        assert "a French message gets a French title" not in named
        generic = service._build_title_prompt_text("ok", model_type)
        assert "a French message gets a French title" in generic
