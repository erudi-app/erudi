"""Language of the user's messages: detection, and what is built on it.

Also here: the language name the conversation title and compaction summary
prompts state (``language_name``), the greeting rule that leaves a
conversation untitled (``is_greeting``), and the script check on a generated
title (``strip_foreign_scripts``).

Question-language detection for the KB prompt's answer-language line.

Multilingual RAG drifts toward English (the "semantic attractor" — measured
in the language-drift literature), and a generic "same language" line is the
weakest counter-measure. The strongest prompt-level move is a dynamic
instruction written IN the target language ("Réponds en français."), which
needs the question's language: py3langid (resident ~2 MB model, BSD), with
a confidence floor so short/ambiguous turns fall back to the generic line
instead of guessing.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from py3langid.langid import MODEL_FILE, LanguageIdentifier

from src.core.logging import logger

# Below this normalized probability the signal is noise ("ok" scores ~0.17):
# the caller falls back to the generic same-language instruction.
MIN_CONFIDENCE = 0.7

_identifier: Optional[LanguageIdentifier] = None
# Cache a load failure so a missing/unreadable model file is not retried on
# every turn (it would never recover within a process).
_load_failed = False


def _get_identifier() -> Optional[LanguageIdentifier]:
    """Resident classifier (lazy singleton — same pattern as E5Embeddings).

    Returns None if the pickled model cannot be loaded (e.g. the data file is
    missing from a packaged build); callers then fall back to the generic
    answer-language line instead of crashing the query.
    """
    global _identifier, _load_failed
    if _identifier is None and not _load_failed:
        try:
            _identifier = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)
        except Exception:
            # Logged once per process: from here on every KB turn falls back
            # to the generic answer-language line, which a maintainer would
            # otherwise only discover through answers drifting to English.
            _load_failed = True
            logger.warning(
                f"Language detection is unavailable (could not load {MODEL_FILE}); "
                f"KB prompts fall back to the generic answer-language line",
                exc_info=True,
            )
            return None
    return _identifier


def detect_language(text: str) -> Optional[str]:
    """ISO 639-1 code of ``text``'s language, or None when unconfident."""
    text = (text or "").strip()
    if not text:
        return None
    identifier = _get_identifier()
    if identifier is None:
        return None
    try:
        language, probability = identifier.classify(text)
    except Exception:
        # Per-turn and best-effort: the generic line is the fallback, and the
        # text itself is already in the request's INFO records.
        logger.debug("Language detection failed for this turn", exc_info=True)
        return None
    return language if probability >= MIN_CONFIDENCE else None


# English names of the languages an instruction can name. The instruction is
# written in English (the prompts around it are), so the model gets
# "Write the title in French." A detected code outside this table is treated
# as unconfident: the generic same-language line stands in.
LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "zh": "Chinese",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "pl": "Polish",
    "tr": "Turkish",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "cs": "Czech",
    "uk": "Ukrainian",
    "hi": "Hindi",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "he": "Hebrew",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
}

# What goes into the detector for a conversation's messages: the classifier
# needs no more than this, and the tail is the freshest signal.
_DETECTION_MAX_CHARS = 2000


def language_name(texts: "str | list[Optional[str]]") -> Optional[str]:
    """English name of the language of the user's text(s), or None.

    ``texts`` is one message (a title: the first message) or several (a
    summary: the recent user messages, joined). None when detection is
    unconfident or the language has no entry in ``LANGUAGE_NAMES``: the caller
    then writes the generic "same language as the user's messages" line --
    never a default language.
    """
    if not isinstance(texts, str):
        texts = "\n".join(t for t in texts if t)
    code = detect_language(texts[-_DETECTION_MAX_CHARS:])
    return LANGUAGE_NAMES.get(code) if code else None


# ---------------------------------------------------------------------------
# Greeting detection (a first message that says nothing a title could use)
# ---------------------------------------------------------------------------

_GREETING_PHRASES = (
    "good morning",
    "good afternoon",
    "good evening",
    "good day",
    "buenos dias",
    "buenas tardes",
    "buenas noches",
    "guten tag",
    "guten morgen",
    "guten abend",
    "bonjour",
    "bonsoir",
    "salut",
    "coucou",
    "hello",
    "hi",
    "hey",
    "hola",
    "ciao",
    "hallo",
    "yo",
    "buenas",
    "你好",
    "您好",
    "嗨",
    "哈喽",
    "早上好",
    "晚上好",
    "下午好",
    "早安",
    "晚安",
)
_SMALL_TALK_PHRASES = (
    "comment ca va",
    "comment vas tu",
    "comment allez vous",
    "comment tu vas",
    "tu vas bien",
    "vous allez bien",
    "ca va",
    "ca roule",
    "how are you doing",
    "how are you",
    "hows it going",
    "how is it going",
    "whats up",
    "how do you do",
    "como estas",
    "como esta usted",
    "como va",
    "que tal",
    "que onda",
    "tout le monde",
    "a tous",
    "everyone",
    "everybody",
    "there",
    "todos",
    "你好吗",
    "你还好吗",
    "最近怎么样",
    "怎么样",
    "大家",
)
_INTRO_PHRASES = (
    "presente toi",
    "presentez vous",
    "qui es tu",
    "qui etes vous",
    "introduce yourself",
    "who are you",
    "tell me about yourself",
    "presentate",
    "quien eres",
    "你是谁",
    "介绍一下你自己",
    "自我介绍",
)


def _normalize(text: str) -> str:
    """Lowercase, accents and apostrophes removed, other non-word runs a space."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("'", "").replace("’", "")
    return re.sub(r"[\W_]+", " ", text).strip()


def _alternation(phrases) -> "re.Pattern[str]":
    ordered = sorted({_normalize(p) for p in phrases}, key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(re.escape(p) for p in ordered) + r")(?!\w)")


_GREETING_RE = _alternation(_GREETING_PHRASES)
_SMALL_TALK_RE = _alternation(_SMALL_TALK_PHRASES)
_INTRO_RE = _alternation(_INTRO_PHRASES)
# Words a "present yourself" request may keep ("in two sentences").
_INTRO_TAIL_MAX_WORDS = 4


def is_greeting(text: str) -> bool:
    """True when ``text`` is a greeting a title cannot be drawn from.

    Two shapes, both from a small multilingual word list (fr, en, es, de, zh):
    a message made only of greetings and small talk ("Hi", "Salut, ca va ?",
    "How are you?"), and a greeting followed by a request to introduce
    oneself ("Bonjour ! Presente-toi en deux phrases."), which says nothing
    about a subject either. A greeting followed by a real question is not one:
    that message has a topic and gets a title.
    """
    normalized = _normalize(text or "")
    if not normalized or _GREETING_RE.match(normalized) is None:
        return False
    rest = _SMALL_TALK_RE.sub(" ", _GREETING_RE.sub(" ", normalized)).strip()
    if not rest:
        return True
    if not _INTRO_RE.search(rest):
        return False
    tail = _INTRO_RE.sub(" ", rest).split()
    return len(tail) <= _INTRO_TAIL_MAX_WORDS


# ---------------------------------------------------------------------------
# Script validation of a generated title
# ---------------------------------------------------------------------------

_SCRIPT_PREFIXES = (
    "LATIN",
    "CJK",
    "HIRAGANA",
    "KATAKANA",
    "HANGUL",
    "CYRILLIC",
    "ARABIC",
    "GREEK",
    "HEBREW",
    "DEVANAGARI",
    "THAI",
)


def _script(char: str) -> Optional[str]:
    """Script of a letter (None for digits, punctuation, spaces)."""
    if not char.isalpha():
        return None
    try:
        name = unicodedata.name(char)
    except ValueError:
        return None
    for prefix in _SCRIPT_PREFIXES:
        if name.startswith(prefix):
            return prefix
    return "OTHER"


def strip_foreign_scripts(title: str, source: str) -> str:
    """Drop from ``title`` the letters of a script ``source`` does not use.

    A small model sometimes slips a word of another script into a title
    (two Han characters after a French title, for a French message). The
    script is the deterministic part of "wrong language": a title letter of a
    script absent from the user's message cannot belong there. A Japanese
    message writes Han and kana both. Digits and punctuation are kept; the
    leftover spaces are collapsed.
    """
    allowed = {_script(c) for c in source} - {None}
    if allowed & {"HIRAGANA", "KATAKANA"}:
        allowed |= {"HIRAGANA", "KATAKANA", "CJK"}
    kept = [c for c in title if (s := _script(c)) is None or s in allowed]
    return re.sub(r"\s+", " ", "".join(kept)).strip()
