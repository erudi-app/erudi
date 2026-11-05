"""Per-model conversation defaults (#609, #627).

``/config global`` saves a conversation's values as the defaults of its model.
They live in ``model_defaults`` (one nullable column per value, NULL = not
saved, deleted with the model) and one pure resolver turns a model, its saved
values and the user settings into the value of every field and where it came
from:

- effort, compaction threshold, web search: saved > Settings > constant;
- temperature, top_p: saved > the model's recommendation > constant;
- custom instructions: saved > empty.

Covered here: the resolver, the entity, the two routes and the deletion of a
model. Conversation creation, the model switch, the turn log line and the
Arena are in ``test_conversation_model_defaults.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.routing import APIRoute

from src.database.generation_hints import FALLBACK_TEMPERATURE, FALLBACK_TOP_P
from src.domains.llms.conversation_defaults import (
    CONVERSATION_DEFAULT_FIELDS,
    resolve_conversation_defaults,
)
from src.entities.Conversation import Conversation
from src.entities.Llm import Llm
from src.entities.ModelDefaults import ModelDefaults

pytestmark = pytest.mark.unit

# Qwen2.5's published values: both differ from the neutral constants, so a
# "model" source can never be confused with a "fallback" one.
RECOMMENDED = {"temperature": 0.7, "top_p": 0.8, "do_sample": True}

HINTED = SimpleNamespace(generation_hints={"generation_config": RECOMMENDED})
UNHINTED = SimpleNamespace(generation_hints=None)

SETTINGS = SimpleNamespace(
    default_reasoning_effort="low",
    default_compaction_threshold=55,
    web_search_enabled=True,
)

PINS = SimpleNamespace(
    reasoning_effort="high",
    temperature=1.1,
    top_p=0.4,
    compaction_threshold=65,
    web_search_enabled=False,
    custom_prompt="Answer in French.",
)

EMPTY_PINS = SimpleNamespace(**{field: None for field in vars(PINS)})


# ===================== the resolver =====================


def test_the_resolver_answers_exactly_the_six_fields():
    resolved = resolve_conversation_defaults(UNHINTED, None, None)
    assert tuple(resolved) == CONVERSATION_DEFAULT_FIELDS
    assert set(CONVERSATION_DEFAULT_FIELDS) == {
        "reasoning_effort",
        "temperature",
        "top_p",
        "compaction_threshold",
        "web_search_enabled",
        "custom_prompt",
    }


@pytest.mark.parametrize(
    "field, llm, pins, settings, expected",
    [
        # effort: saved > Settings > constant
        ("reasoning_effort", HINTED, PINS, SETTINGS, ("high", "saved")),
        ("reasoning_effort", HINTED, None, SETTINGS, ("low", "settings")),
        ("reasoning_effort", HINTED, EMPTY_PINS, SETTINGS, ("low", "settings")),
        ("reasoning_effort", HINTED, None, None, ("medium", "fallback")),
        # compaction threshold: same chain
        ("compaction_threshold", UNHINTED, PINS, SETTINGS, (65, "saved")),
        ("compaction_threshold", UNHINTED, None, SETTINGS, (55, "settings")),
        ("compaction_threshold", UNHINTED, EMPTY_PINS, None, (80, "fallback")),
        # web search: a saved False wins over a Settings True
        ("web_search_enabled", UNHINTED, PINS, SETTINGS, (False, "saved")),
        ("web_search_enabled", UNHINTED, EMPTY_PINS, SETTINGS, (True, "settings")),
        ("web_search_enabled", UNHINTED, None, None, (False, "fallback")),
        # temperature: saved > the model's recommendation > constant
        ("temperature", HINTED, PINS, SETTINGS, (1.1, "saved")),
        ("temperature", HINTED, EMPTY_PINS, SETTINGS, (0.7, "model")),
        ("temperature", UNHINTED, None, SETTINGS, (FALLBACK_TEMPERATURE, "fallback")),
        # top_p: same chain
        ("top_p", UNHINTED, PINS, None, (0.4, "saved")),
        ("top_p", HINTED, None, None, (0.8, "model")),
        ("top_p", UNHINTED, EMPTY_PINS, SETTINGS, (FALLBACK_TOP_P, "fallback")),
        # custom instructions: saved > empty (the settings play no part)
        ("custom_prompt", UNHINTED, PINS, SETTINGS, ("Answer in French.", "saved")),
        ("custom_prompt", HINTED, EMPTY_PINS, SETTINGS, ("", "fallback")),
        ("custom_prompt", HINTED, None, None, ("", "fallback")),
    ],
)
def test_every_field_from_every_source(field, llm, pins, settings, expected):
    resolved = resolve_conversation_defaults(llm, pins, settings)[field]
    assert (resolved.value, resolved.source) == expected
    # A plain (value, source) pair for callers that unpack it.
    value, source = resolved
    assert (value, source) == expected


def test_saved_values_of_one_field_leave_the_others_to_their_chain():
    pins = SimpleNamespace(**{**vars(EMPTY_PINS), "top_p": 0.3})
    resolved = resolve_conversation_defaults(HINTED, pins, SETTINGS)
    assert resolved["top_p"] == (0.3, "saved")
    assert resolved["temperature"] == (0.7, "model")
    assert resolved["reasoning_effort"] == (SETTINGS.default_reasoning_effort, "settings")


# ===================== the entity =====================


def test_a_row_with_nothing_saved_is_valid(test_db_session, mock_llm):
    row = ModelDefaults(llm_id=mock_llm.id)
    test_db_session.add(row)
    test_db_session.flush()
    test_db_session.refresh(row)
    assert all(getattr(row, field) is None for field in CONVERSATION_DEFAULT_FIELDS)


@pytest.mark.parametrize(
    "field, value",
    [
        ("reasoning_effort", "max"),
        ("temperature", 2.5),
        ("temperature", -0.1),
        ("top_p", 1.5),
        ("compaction_threshold", 82),
        ("compaction_threshold", True),
    ],
)
def test_the_entity_refuses_what_a_conversation_refuses(field, value):
    # One set of rules for both rows: the helpers are shared, not copied.
    with pytest.raises(ValueError):
        ModelDefaults(**{field: value})
    with pytest.raises(ValueError):
        Conversation(**{field: value})


# ===================== the routes =====================


def _url(llm_id: int) -> str:
    return f"/erudi/llms/{llm_id}/conversation_defaults"


def _flat(body: dict) -> dict:
    return {
        field: (body[field]["value"], body[field]["source"]) for field in body if field != "llm_id"
    }


@pytest.fixture
def hinted_llm(test_db_session):
    llm = Llm(
        name="Qwen2.5 0.5B",
        local=1,
        link="/models/does-not-exist/qwen25",
        type="qwen",
        param_size=0.5,
        quantized=True,
        generation_hints={"generation_config": RECOMMENDED},
    )
    test_db_session.add(llm)
    test_db_session.commit()
    test_db_session.refresh(llm)
    return llm


class TestTheRoutes:
    def test_the_routes_are_declared_before_the_bare_id_route(self, client):
        # A literal segment after the id cannot be swallowed by /{llm_id}, but
        # the declaration order keeps that true whatever is added later.
        paths = [r.path for r in client.app.routes if isinstance(r, APIRoute)]
        assert paths.index("/erudi/llms/{llm_id}/conversation_defaults") < paths.index(
            "/erudi/llms/{llm_id}"
        )

    def test_get_without_saved_values_or_settings_row(self, client, hinted_llm):
        response = client.get(_url(hinted_llm.id))
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["llm_id"] == hinted_llm.id
        assert _flat(body) == {
            "reasoning_effort": ("medium", "fallback"),
            "temperature": (0.7, "model"),
            "top_p": (0.8, "model"),
            "compaction_threshold": (80, "fallback"),
            "web_search_enabled": (False, "fallback"),
            "custom_prompt": ("", "fallback"),
        }

    def test_get_follows_the_settings(self, client, hinted_llm):
        client.put(
            "/erudi/user_settings/",
            json={
                "default_reasoning_effort": "xhigh",
                "default_compaction_threshold": 60,
                "web_search_enabled": True,
            },
        )
        flat = _flat(client.get(_url(hinted_llm.id)).json())
        assert flat["reasoning_effort"] == ("xhigh", "settings")
        assert flat["compaction_threshold"] == (60, "settings")
        assert flat["web_search_enabled"] == (True, "settings")

    def test_the_bare_id_route_still_answers(self, client, hinted_llm):
        response = client.get(f"/erudi/llms/{hinted_llm.id}")
        assert response.status_code == 200
        assert response.json()["name"] == "Qwen2.5 0.5B"

    def test_an_unknown_model_is_404(self, client):
        assert client.get(_url(987654)).status_code == 404
        assert client.put(_url(987654), json={"temperature": 0.5}).status_code == 404

    def test_put_saves_every_field_and_returns_the_resolved_values(self, client, hinted_llm):
        payload = {
            "reasoning_effort": "high",
            "temperature": 1.1,
            "top_p": 0.4,
            "compaction_threshold": 65,
            "web_search_enabled": False,
            "custom_prompt": "Answer in French.",
        }
        response = client.put(_url(hinted_llm.id), json=payload)
        assert response.status_code == 200, response.text
        expected = {field: (value, "saved") for field, value in payload.items()}
        assert _flat(response.json()) == expected
        assert _flat(client.get(_url(hinted_llm.id)).json()) == expected

    def test_an_absent_field_is_left_unchanged(self, client, hinted_llm):
        client.put(_url(hinted_llm.id), json={"reasoning_effort": "high"})
        flat = _flat(client.put(_url(hinted_llm.id), json={"temperature": 1.1}).json())
        assert flat["reasoning_effort"] == ("high", "saved")
        assert flat["temperature"] == (1.1, "saved")

    def test_an_explicit_null_clears_the_field(self, client, hinted_llm):
        client.put(_url(hinted_llm.id), json={"reasoning_effort": "high", "temperature": 1.1})
        flat = _flat(client.put(_url(hinted_llm.id), json={"temperature": None}).json())
        assert flat["temperature"] == (0.7, "model")
        assert flat["reasoning_effort"] == ("high", "saved")
        flat = _flat(client.put(_url(hinted_llm.id), json={"reasoning_effort": None}).json())
        assert flat["reasoning_effort"] == ("medium", "fallback")

    @pytest.mark.parametrize("empty", ["", "   ", "\n\t "])
    def test_empty_instructions_clear_the_saved_ones(self, client, hinted_llm, empty):
        client.put(_url(hinted_llm.id), json={"custom_prompt": "Answer in French."})
        flat = _flat(client.put(_url(hinted_llm.id), json={"custom_prompt": empty}).json())
        assert flat["custom_prompt"] == ("", "fallback")

    def test_instructions_are_saved_as_typed(self, client, hinted_llm):
        flat = _flat(
            client.put(_url(hinted_llm.id), json={"custom_prompt": "  Be brief.\n"}).json()
        )
        assert flat["custom_prompt"] == ("  Be brief.\n", "saved")

    def test_an_empty_payload_changes_nothing(self, client, hinted_llm):
        client.put(_url(hinted_llm.id), json={"top_p": 0.3})
        flat = _flat(client.put(_url(hinted_llm.id), json={}).json())
        assert flat["top_p"] == (0.3, "saved")

    @pytest.mark.parametrize(
        "payload",
        [
            {"reasoning_effort": "max"},
            {"temperature": 2.5},
            {"top_p": -0.1},
            {"top_p": 1.01},
            {"compaction_threshold": 82},
            {"compaction_threshold": True},
            {"compaction_threshold": 45},
            {"web_search_enabled": "sometimes"},
            {"custom_prompt": "x" * 4097},
            {"custom_prompt": "x" * 5000},
            {"max_tokens": 512},
        ],
    )
    def test_an_invalid_value_is_422_and_saves_nothing(self, client, hinted_llm, payload):
        client.put(_url(hinted_llm.id), json={"temperature": 1.1})
        response = client.put(_url(hinted_llm.id), json={"top_p": 0.3, **payload})
        assert response.status_code == 422, response.text
        flat = _flat(client.get(_url(hinted_llm.id)).json())
        assert flat["top_p"] == (0.8, "model")
        assert flat["temperature"] == (1.1, "saved")

    @pytest.mark.parametrize("local", [0, 2])
    def test_put_refuses_a_model_that_is_not_downloaded(
        self, client, test_db_session, hinted_llm, local
    ):
        # A catalog row (0) or a download placeholder (2) never backs a
        # conversation and is replaced by a new row once downloaded.
        hinted_llm.local = local
        test_db_session.commit()

        response = client.put(_url(hinted_llm.id), json={"reasoning_effort": "high"})

        assert response.status_code == 400, response.text
        assert response.json()["error"]["type"] == "STATE_CONFLICT"
        assert _saved_rows(test_db_session, hinted_llm.id) == 0
        # Reading stays open: it answers what a conversation would start from.
        assert client.get(_url(hinted_llm.id)).status_code == 200

    def test_saving_one_model_never_moves_another(self, client, hinted_llm, mock_llm):
        client.put(_url(hinted_llm.id), json={"reasoning_effort": "high", "temperature": 1.1})
        flat = _flat(client.get(_url(mock_llm.id)).json())
        assert flat["reasoning_effort"] == ("medium", "fallback")
        assert flat["temperature"] == (FALLBACK_TEMPERATURE, "fallback")


# ===================== deleted with the model =====================


def _saved_rows(db, llm_id: int) -> int:
    return db.query(ModelDefaults).filter(ModelDefaults.llm_id == llm_id).count()


class TestDeletedWithTheModel:
    def test_deleting_a_model_deletes_its_saved_values(self, client, test_db_session, hinted_llm):
        llm_id = hinted_llm.id
        client.put(_url(llm_id), json={"reasoning_effort": "high"})
        assert _saved_rows(test_db_session, llm_id) == 1

        response = client.delete(f"/erudi/llms/{llm_id}")

        assert response.status_code == 200, response.text
        test_db_session.expire_all()
        assert _saved_rows(test_db_session, llm_id) == 0

    def test_deleting_a_kb_assistant_deletes_its_saved_values(
        self, client, test_db_session, mock_llm_with_kb
    ):
        assistant, _kb = mock_llm_with_kb
        llm_id = assistant.id
        client.put(_url(llm_id), json={"custom_prompt": "Cite the documents."})
        assert _saved_rows(test_db_session, llm_id) == 1

        response = client.delete(f"/erudi/llms/{llm_id}")

        assert response.status_code == 200, response.text
        test_db_session.expire_all()
        assert _saved_rows(test_db_session, llm_id) == 0

    def test_an_orm_delete_with_the_row_in_the_session(self, test_db_session, mock_llm):
        # The seed's and the repository's ``db.delete(llm)``: no relationship
        # from Llm, so the ORM issues one DELETE and the database cascades.
        test_db_session.add(ModelDefaults(llm_id=mock_llm.id, top_p=0.3))
        test_db_session.flush()
        llm_id = mock_llm.id

        test_db_session.delete(mock_llm)
        test_db_session.flush()

        test_db_session.expire_all()
        assert _saved_rows(test_db_session, llm_id) == 0
