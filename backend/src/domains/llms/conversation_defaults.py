"""The values a conversation on a given model starts from (#609, #627).

One pure resolver, ``resolve_conversation_defaults(llm, pins, settings)``,
answers every conversation value a user can save for a model with
``/config global``, and where it came from:

=========================  ================================================
Field                      First value that exists
=========================  ================================================
``reasoning_effort``       saved > Settings default > ``medium``
``compaction_threshold``   saved > Settings default > 80
``web_search_enabled``     saved > Settings default > off
``temperature``            saved > the model's recommendation > 0.2
``top_p``                  saved > the model's recommendation > 0.95
``custom_prompt``          saved > empty
=========================  ================================================

Sources: ``saved`` (the ``model_defaults`` row), ``settings`` (the
``user_settings`` row), ``model`` (``resolve_sampling_defaults``: the
publisher's captured generation config) and ``fallback`` (the constants; for
the settings fields, only when no settings row exists yet). There is no
per-model effort recommendation: a model without saved values opens at the
Settings effort whether it reasons or not.

Read by the conversation creation and the model switch
(``src.domains.conversations.services``), the Arena
(``src.domains.arena.services``) and ``GET|PUT
/llms/{llm_id}/conversation_defaults``.
"""

from __future__ import annotations

from typing import Any, Dict, NamedTuple, Optional

from sqlalchemy.orm import Session

from src.agents.reasoning_effort import DEFAULT_REASONING_EFFORT
from src.database.generation_hints import SOURCE_NONE, resolve_sampling_defaults
from src.domains.llms.repository import Model_Defaults_Repository
from src.domains.user_settings.repository import User_Settings_Repository
from src.engines.working_window import DEFAULT_COMPACTION_THRESHOLD

SOURCE_SAVED = "saved"
SOURCE_SETTINGS = "settings"
SOURCE_MODEL = "model"
SOURCE_FALLBACK = "fallback"

# The values ``/config global`` saves, in the order the resolver answers them.
CONVERSATION_DEFAULT_FIELDS = (
    "reasoning_effort",
    "temperature",
    "top_p",
    "compaction_threshold",
    "web_search_enabled",
    "custom_prompt",
)

# Field -> (user_settings column, constant used without a settings row).
_SETTINGS_FIELDS = {
    "reasoning_effort": ("default_reasoning_effort", DEFAULT_REASONING_EFFORT),
    "compaction_threshold": ("default_compaction_threshold", DEFAULT_COMPACTION_THRESHOLD),
    "web_search_enabled": ("web_search_enabled", False),
}


class ResolvedValue(NamedTuple):
    """One resolved field: the value and the source it came from."""

    value: Any
    source: str


def resolve_conversation_defaults(
    llm: Any, pins: Optional[Any], settings: Optional[Any]
) -> Dict[str, ResolvedValue]:
    """Resolve every field of ``CONVERSATION_DEFAULT_FIELDS`` for ``llm``.

    Pure: reads attributes only, never the database or the network.

    Args:
        llm: The model (``Llm`` row or any object with ``generation_hints``).
        pins: Its ``ModelDefaults`` row, or None when nothing was saved. A
            NULL column is a value not saved.
        settings: The ``UserSettings`` row, or None when it does not exist.

    Returns:
        ``{field: ResolvedValue(value, source)}`` in
        ``CONVERSATION_DEFAULT_FIELDS`` order.
    """
    sampling = resolve_sampling_defaults(llm)
    sampling_source = SOURCE_FALLBACK if sampling.source == SOURCE_NONE else SOURCE_MODEL
    resolved: Dict[str, ResolvedValue] = {}
    for field in CONVERSATION_DEFAULT_FIELDS:
        saved = getattr(pins, field, None) if pins is not None else None
        if saved is not None:
            resolved[field] = ResolvedValue(saved, SOURCE_SAVED)
        elif field in _SETTINGS_FIELDS:
            column, constant = _SETTINGS_FIELDS[field]
            if settings is not None:
                resolved[field] = ResolvedValue(getattr(settings, column), SOURCE_SETTINGS)
            else:
                resolved[field] = ResolvedValue(constant, SOURCE_FALLBACK)
        elif field in ("temperature", "top_p"):
            resolved[field] = ResolvedValue(getattr(sampling, field), sampling_source)
        else:  # custom_prompt
            resolved[field] = ResolvedValue("", SOURCE_FALLBACK)
    return resolved


def read_conversation_defaults(db: Session, llm: Any) -> Dict[str, ResolvedValue]:
    """``resolve_conversation_defaults`` for ``llm`` with its saved values and
    the settings read from ``db``. Read-only and blocking (run it in a
    threadpool from async code)."""
    pins = Model_Defaults_Repository(db).get(llm.id)
    settings = User_Settings_Repository(db).get()
    return resolve_conversation_defaults(llm, pins, settings)
