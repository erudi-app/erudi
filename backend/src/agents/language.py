"""Question-language detection for the KB prompt's answer-language line.

Multilingual RAG drifts toward English (the "semantic attractor" — measured
in the language-drift literature), and a generic "same language" line is the
weakest counter-measure. The strongest prompt-level move is a dynamic
instruction written IN the target language ("Réponds en français."), which
needs the question's language: py3langid (resident ~2 MB model, BSD), with
a confidence floor so short/ambiguous turns fall back to the generic line
instead of guessing.
"""

from __future__ import annotations

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
