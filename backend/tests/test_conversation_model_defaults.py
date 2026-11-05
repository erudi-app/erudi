"""Where the per-model conversation defaults apply (#609, #627).

- ``POST /conversations/``: every field the client leaves out is resolved
  through the model's saved values, then the Settings or the model's
  recommendation, then the constants. What the client sends wins.
- ``PATCH /conversations/{id}`` with ANOTHER ``llm_id`` (the ``/model``
  command): the new model brings its saved values and, for creativity and
  diversity it has not saved, its recommendation; creativity and diversity in
  the same payload are ignored. The same ``llm_id`` re-applies nothing.
- The Arena: a panel that sends no effort, creativity or diversity runs at the
  model's saved values, else the Settings effort and the model's
  recommendation.
- Every conversation turn logs the values it runs with.
"""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from src.agents.reasoning_effort import plan_reasoning_effort
from src.core import config
from src.domains.arena.schemas import ArenaQueryPayload
from src.domains.arena.services import ArenaService
from src.domains.conversations.schemas import ConversationQuery
from src.domains.conversations.services import ConversationService
from src.engines.base_engine import BaseEngine
from src.entities.Llm import Llm
from src.entities.ModelDefaults import ModelDefaults
from tests.test_user_settings import _plan_spy, _stream_spy, _stub_stream

pytestmark = pytest.mark.unit


class _FakeEngine(BaseEngine):
    """generation_guard without a real model."""


@pytest.fixture(autouse=True)
def _engine(monkeypatch):
    monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)


def _llm(db, name: str, temperature: float, top_p: float, context_length=None) -> Llm:
    hints = {"generation_config": {"temperature": temperature, "top_p": top_p, "do_sample": True}}
    if context_length is not None:
        hints["context_length"] = context_length
    llm = Llm(
        name=name,
        local=1,
        link=f"/models/does-not-exist/{name}",
        type="qwen",
        param_size=1.0,
        quantized=True,
        generation_hints=hints,
    )
    db.add(llm)
    db.commit()
    db.refresh(llm)
    return llm


@pytest.fixture
def model_a(test_db_session):
    return _llm(test_db_session, "model-a", 0.7, 0.8)


@pytest.fixture
def model_b(test_db_session):
    return _llm(test_db_session, "model-b", 0.6, 0.95)


def _save(client, llm_id: int, **values) -> None:
    response = client.put(f"/erudi/llms/{llm_id}/conversation_defaults", json=values)
    assert response.status_code == 200, response.text


def _create(client, llm_id: int, **values) -> dict:
    response = client.post("/erudi/conversations/", json={"llm_id": llm_id, **values})
    assert response.status_code == 201, response.text
    return response.json()


def _patch(client, conversation_id: int, **values):
    return client.patch(f"/erudi/conversations/{conversation_id}", json=values)


SAVED_B = {
    "reasoning_effort": "high",
    "temperature": 1.1,
    "top_p": 0.4,
    "compaction_threshold": 65,
    "web_search_enabled": True,
    "custom_prompt": "Answer in French.",
}


def _settings_of(conversation: dict) -> dict:
    return {field: conversation[field] for field in SAVED_B}


# ===================== POST /conversations/ =====================


class TestCreation:
    def test_omitted_fields_take_the_saved_values(self, client, model_b):
        _save(client, model_b.id, **SAVED_B)
        conversation = _create(client, model_b.id)
        assert _settings_of(conversation) == SAVED_B
        # max_tokens is not a saved value: still the model's fallback budget.
        assert conversation["max_tokens"] == 1024

    def test_without_saved_values_the_settings_and_the_model_apply(self, client, model_a):
        client.put(
            "/erudi/user_settings/",
            json={
                "default_reasoning_effort": "low",
                "default_compaction_threshold": 55,
                "web_search_enabled": True,
            },
        )
        assert _settings_of(_create(client, model_a.id)) == {
            "reasoning_effort": "low",
            "temperature": 0.7,
            "top_p": 0.8,
            "compaction_threshold": 55,
            "web_search_enabled": True,
            "custom_prompt": "",
        }

    def test_some_saved_values_leave_the_others_to_their_chain(self, client, model_a):
        _save(client, model_a.id, temperature=1.3, custom_prompt="Be brief.")
        conversation = _create(client, model_a.id)
        assert conversation["temperature"] == 1.3
        assert conversation["top_p"] == 0.8
        assert conversation["custom_prompt"] == "Be brief."
        assert conversation["reasoning_effort"] == "medium"

    def test_what_the_client_sends_wins(self, client, model_b):
        _save(client, model_b.id, **SAVED_B)
        sent = {
            "reasoning_effort": "none",
            "temperature": 0.2,
            "top_p": 0.9,
            "compaction_threshold": 90,
            "web_search_enabled": False,
            "custom_prompt": "",
        }
        assert _settings_of(_create(client, model_b.id, **sent)) == sent

    def test_another_models_saved_values_never_apply(self, client, model_a, model_b):
        _save(client, model_b.id, **SAVED_B)
        conversation = _create(client, model_a.id)
        assert conversation["reasoning_effort"] == "medium"
        assert (conversation["temperature"], conversation["top_p"]) == (0.7, 0.8)
        assert conversation["custom_prompt"] == ""

    def test_an_unknown_model_is_404(self, client):
        assert client.post("/erudi/conversations/", json={"llm_id": 987654}).status_code == 404


# ===================== PATCH: the model switch =====================


class TestModelSwitch:
    def _start_on_a(self, client, model_a) -> dict:
        return _create(
            client,
            model_a.id,
            reasoning_effort="low",
            temperature=0.3,
            top_p=0.5,
            compaction_threshold=70,
            web_search_enabled=False,
            custom_prompt="Mine.",
        )

    def test_the_new_model_brings_its_saved_values(self, client, model_a, model_b):
        conversation = self._start_on_a(client, model_a)
        _save(client, model_b.id, **SAVED_B)

        response = _patch(client, conversation["id"], llm_id=model_b.id)

        assert response.status_code == 200, response.text
        assert response.json()["llm_id"] == model_b.id
        assert _settings_of(response.json()) == SAVED_B
        stored = client.get(f"/erudi/conversations/{conversation['id']}").json()
        assert _settings_of(stored) == SAVED_B

    def test_without_saved_values_only_creativity_and_diversity_move(
        self, client, model_a, model_b
    ):
        conversation = self._start_on_a(client, model_a)

        switched = _patch(client, conversation["id"], llm_id=model_b.id).json()

        assert _settings_of(switched) == {
            "reasoning_effort": "low",
            "temperature": 0.6,
            "top_p": 0.95,
            "compaction_threshold": 70,
            "web_search_enabled": False,
            "custom_prompt": "Mine.",
        }

    def test_creativity_and_diversity_sent_with_the_switch_are_ignored(
        self, client, model_a, model_b
    ):
        conversation = self._start_on_a(client, model_a)
        _save(client, model_b.id, temperature=1.1)

        switched = _patch(
            client, conversation["id"], llm_id=model_b.id, temperature=0.33, top_p=0.44
        ).json()

        assert switched["temperature"] == 1.1  # saved
        assert switched["top_p"] == 0.95  # recommended, not the payload's 0.44

    def test_other_payload_values_lose_only_to_saved_ones(self, client, model_a, model_b):
        conversation = self._start_on_a(client, model_a)
        _save(client, model_b.id, reasoning_effort="high", custom_prompt="Theirs.")

        switched = _patch(
            client,
            conversation["id"],
            llm_id=model_b.id,
            reasoning_effort="none",
            compaction_threshold=60,
            web_search_enabled=True,
            custom_prompt="Sent.",
        ).json()

        assert switched["reasoning_effort"] == "high"  # saved wins
        assert switched["custom_prompt"] == "Theirs."  # saved wins
        assert switched["compaction_threshold"] == 60  # not saved: payload
        assert switched["web_search_enabled"] is True  # not saved: payload

    def test_the_same_model_reapplies_nothing(self, client, model_a, model_b):
        conversation = _create(client, model_b.id)
        _save(client, model_b.id, **SAVED_B)

        patched = _patch(
            client, conversation["id"], llm_id=model_b.id, temperature=0.33, name="Renamed"
        ).json()

        assert patched["temperature"] == 0.33
        assert patched["name"] == "Renamed"
        assert patched["reasoning_effort"] == "medium"
        assert patched["custom_prompt"] == ""
        assert patched["top_p"] == 0.95

    def test_a_patch_without_llm_id_applies_its_values(self, client, model_a, model_b):
        conversation = self._start_on_a(client, model_a)
        _save(client, model_a.id, **SAVED_B)

        patched = _patch(client, conversation["id"], temperature=0.25).json()

        assert patched["temperature"] == 0.25
        assert patched["reasoning_effort"] == "low"

    def test_an_orphaned_conversation_takes_the_new_models_values(
        self, client, test_db_session, model_a, model_b
    ):
        conversation = self._start_on_a(client, model_a)
        assert client.delete(f"/erudi/llms/{model_a.id}").status_code == 200
        test_db_session.expire_all()
        assert client.get(f"/erudi/conversations/{conversation['id']}").json()["llm_id"] is None
        _save(client, model_b.id, reasoning_effort="xhigh")

        switched = _patch(client, conversation["id"], llm_id=model_b.id).json()

        assert switched["llm_id"] == model_b.id
        assert switched["reasoning_effort"] == "xhigh"
        assert (switched["temperature"], switched["top_p"]) == (0.6, 0.95)

    def test_switching_to_an_unknown_model_is_404_and_changes_nothing(self, client, model_a):
        conversation = self._start_on_a(client, model_a)

        response = _patch(client, conversation["id"], llm_id=987654, temperature=1.9)

        assert response.status_code == 404
        stored = client.get(f"/erudi/conversations/{conversation['id']}").json()
        assert stored["llm_id"] == model_a.id
        assert stored["temperature"] == 0.3

    def test_the_switch_logs_what_it_applied(self, client, model_a, model_b, caplog):
        conversation = self._start_on_a(client, model_a)
        _save(client, model_b.id, reasoning_effort="high", temperature=1.1)

        with caplog.at_level(logging.INFO):
            _patch(client, conversation["id"], llm_id=model_b.id, temperature=0.33, top_p=0.44)

        messages = [r.getMessage() for r in caplog.records]
        (switch,) = [m for m in messages if "switched to model" in m]
        assert switch.isascii()
        assert switch == (
            f"Conversation {conversation['id']} switched to model {model_b.id}: "
            "saved=[reasoning_effort, temperature], temperature=1.1 (saved), "
            "top_p=0.95 (model), max_tokens=1024"
        )
        # The update line names what was written: the saved effort the PATCH
        # did not carry, and creativity/diversity with the model's values.
        (updated,) = [m for m in messages if m.startswith("Conversation updated")]
        assert updated == (
            f"Conversation updated: id={conversation['id']}, fields=['llm_id', 'temperature', "
            "'top_p', 'max_tokens', 'reasoning_effort']"
        )

    def test_a_plain_update_logs_the_fields_it_carried(self, client, model_a, caplog):
        conversation = self._start_on_a(client, model_a)

        with caplog.at_level(logging.INFO):
            _patch(client, conversation["id"], llm_id=model_a.id, name="Renamed", top_p=0.6)

        messages = [r.getMessage() for r in caplog.records]
        assert not [m for m in messages if "switched to model" in m]
        (updated,) = [m for m in messages if m.startswith("Conversation updated")]
        assert updated == (
            f"Conversation updated: id={conversation['id']}, fields=['name', 'llm_id', 'top_p']"
        )

    def test_a_failed_write_logs_no_switch(
        self, test_db_session, model_a, model_b, monkeypatch, caplog
    ):
        from src.core.exceptions import DatabaseException

        service = ConversationService(test_db_session)
        conversation = service.create_conversation(llm_id=model_a.id)
        test_db_session.commit()

        def _refuse(**kwargs):
            raise DatabaseException("Could not update conversation")

        monkeypatch.setattr(service.conversation_repo, "update_conversation", _refuse)
        with caplog.at_level(logging.INFO), pytest.raises(DatabaseException):
            service.update_conversation(conversation.id, llm_id=model_b.id)

        messages = [r.getMessage() for r in caplog.records]
        assert not [m for m in messages if "switched to model" in m]
        assert not [m for m in messages if m.startswith("Conversation updated")]

    def test_the_switch_resolves_the_fallback_budget_of_the_new_model(
        self, client, test_db_session, model_a
    ):
        small = _llm(test_db_session, "model-small", 0.5, 0.9, context_length=512)
        conversation = self._start_on_a(client, model_a)
        assert conversation["max_tokens"] == 1024

        switched = _patch(client, conversation["id"], llm_id=small.id).json()

        assert switched["max_tokens"] == 512

    def test_a_budget_sent_with_the_switch_wins(self, client, test_db_session, model_a):
        small = _llm(test_db_session, "model-small", 0.5, 0.9, context_length=512)
        conversation = self._start_on_a(client, model_a)

        switched = _patch(client, conversation["id"], llm_id=small.id, max_tokens=300).json()

        assert switched["max_tokens"] == 300


# ===================== the bounds, on every path =====================


class TestValueBounds:
    """Temperature and top_p have one bound on every path. Custom instructions
    have none: the model's context window is their bound (an oversized prompt
    gets the honest overflow answer), and instructions stored before 1.1.4
    may already be longer than any fixed cap."""

    LONG = "x" * 5000

    def test_creation_keeps_long_instructions_whole(self, client, model_a):
        conversation = _create(client, model_a.id, custom_prompt=self.LONG)
        assert conversation["custom_prompt"] == self.LONG

    def test_a_patch_keeps_long_instructions_whole(self, client, model_a):
        conversation = _create(client, model_a.id)
        patched = _patch(client, conversation["id"], custom_prompt=self.LONG)
        assert patched.status_code == 200, patched.text
        stored = client.get(f"/erudi/conversations/{conversation['id']}").json()
        assert stored["custom_prompt"] == self.LONG

    def test_saved_long_instructions_open_a_conversation_whole(self, client, model_a):
        _save(client, model_a.id, custom_prompt=self.LONG)
        assert _create(client, model_a.id)["custom_prompt"] == self.LONG

    def test_the_turn_payload_accepts_long_instructions(self):
        assert ConversationQuery(question="hi", custom_prompt=self.LONG).custom_prompt == self.LONG

    async def test_a_turn_with_long_stored_instructions_uses_them_whole(
        self, test_db_session, model_a, monkeypatch, caplog
    ):
        service = ConversationService(test_db_session)
        conversation = service.create_conversation(llm_id=model_a.id, custom_prompt=self.LONG)
        test_db_session.commit()
        plan_kwargs: dict = {}
        monkeypatch.setattr("src.domains.conversations.services.plan_turn", _plan_spy(plan_kwargs))
        monkeypatch.setattr(
            "src.domains.conversations.services.detect_supports_vision", lambda link: False
        )
        monkeypatch.setattr(service.runner, "astream_text", _stream_spy({}))

        with caplog.at_level(logging.INFO):
            async for _ in service.query_and_respond_stream(
                conversation.id,
                ConversationQuery(question="hi", custom_prompt=conversation.custom_prompt),
            ):
                pass

        assert plan_kwargs["custom_prompt"] == self.LONG
        assert _turn_line(caplog).endswith("custom_prompt_chars=5000")

    @pytest.mark.parametrize(
        "values",
        [
            {"temperature": 5.0},
            {"temperature": -0.1},
            {"top_p": 1.5},
            {"top_p": -0.1},
        ],
    )
    def test_a_patch_off_the_bounds_is_422_and_changes_nothing(self, client, model_a, values):
        conversation = _create(client, model_a.id, custom_prompt="Mine.", temperature=0.3)

        response = _patch(client, conversation["id"], **values)

        assert response.status_code == 422, response.text
        stored = client.get(f"/erudi/conversations/{conversation['id']}").json()
        assert (stored["custom_prompt"], stored["temperature"]) == ("Mine.", 0.3)
        assert stored["top_p"] == 0.8

    def test_a_patch_at_the_bounds_is_accepted(self, client, model_a):
        conversation = _create(client, model_a.id)
        patched = _patch(client, conversation["id"], temperature=2.0, top_p=1.0)
        assert patched.status_code == 200, patched.text

    @pytest.mark.parametrize("values", [{"temperature": 5.0}, {"top_p": 1.5}])
    def test_a_turn_off_the_bounds_is_422(self, client, model_a, values):
        conversation = _create(client, model_a.id)
        response = client.post(
            f"/erudi/conversations/{conversation['id']}/query", json={"question": "hi", **values}
        )
        assert response.status_code == 422, response.text

    def test_the_turn_payload_schema_carries_the_bounds(self):
        with pytest.raises(ValidationError):
            ConversationQuery(question="hi", temperature=2.5)
        with pytest.raises(ValidationError):
            ConversationQuery(question="hi", top_p=1.01)


# ===================== the turn settings line =====================


async def _run_turn(service, conversation_id, monkeypatch, **payload):
    monkeypatch.setattr("src.domains.conversations.services.plan_turn", _plan_spy({}))
    monkeypatch.setattr(
        "src.domains.conversations.services.detect_supports_vision", lambda link: False
    )
    stream_kwargs: dict = {}
    monkeypatch.setattr(service.runner, "astream_text", _stream_spy(stream_kwargs))
    async for _ in service.query_and_respond_stream(
        conversation_id, ConversationQuery(question="hi", **payload)
    ):
        pass
    return stream_kwargs


def _turn_line(caplog) -> str:
    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Turn settings")]
    return line


async def test_the_turn_logs_the_conversations_values(
    test_db_session, model_a, monkeypatch, caplog
):
    service = ConversationService(test_db_session)
    conversation = service.create_conversation(
        llm_id=model_a.id,
        reasoning_effort="high",
        temperature=0.4,
        top_p=0.6,
        compaction_threshold=75,
        web_search_enabled=True,
        custom_prompt="Stored but not sent.",
    )
    test_db_session.commit()

    with caplog.at_level(logging.INFO):
        await _run_turn(service, conversation.id, monkeypatch)

    line = _turn_line(caplog)
    assert line.isascii()
    assert line == (
        f"Turn settings: conversation={conversation.id}, model={model_a.id}, effort=high, "
        "temperature=0.4, top_p=0.6, threshold=75, web_search_toggle=True, custom_prompt_chars=0"
    )


async def test_the_turn_logs_the_payload_overrides(test_db_session, model_a, monkeypatch, caplog):
    service = ConversationService(test_db_session)
    conversation = service.create_conversation(llm_id=model_a.id, temperature=0.4, top_p=0.6)
    test_db_session.commit()

    with caplog.at_level(logging.INFO):
        stream_kwargs = await _run_turn(
            service,
            conversation.id,
            monkeypatch,
            temperature=1.2,
            top_p=0.3,
            custom_prompt="Réponds en français.",
        )

    line = _turn_line(caplog)
    assert "temperature=1.2, top_p=0.3," in line
    assert line.endswith("custom_prompt_chars=20")
    # The line says what the turn ran with.
    assert (stream_kwargs["params"].temperature, stream_kwargs["params"].top_p) == (1.2, 0.3)


# ===================== the Arena =====================


async def _arena_turn(db, llm_id, monkeypatch, **payload):
    efforts: list = []

    def _effort_spy(llm, level):
        efforts.append(level)
        return plan_reasoning_effort(llm, level)

    monkeypatch.setattr("src.domains.arena.services.plan_reasoning_effort", _effort_spy)
    monkeypatch.setattr("src.domains.arena.services.plan_turn", _plan_spy({}))
    monkeypatch.setattr("src.domains.arena.services.detect_supports_vision", lambda link: False)
    service = ArenaService(db)
    stream_kwargs: dict = {}
    monkeypatch.setattr(service.runner, "astream_text", _stream_spy(stream_kwargs))
    async for _ in service.query_llm_stream(llm_id, ArenaQueryPayload(question="hi", **payload)):
        pass
    params = stream_kwargs["params"]
    return efforts, (params.temperature, params.top_p)


class TestArena:
    def test_the_payload_accepts_null_and_refuses_an_unknown_effort(self):
        payload = ArenaQueryPayload(question="q")
        assert payload.reasoning_effort is None
        assert ArenaQueryPayload(question="q", reasoning_effort="xhigh").reasoning_effort == "xhigh"
        with pytest.raises(ValidationError):
            ArenaQueryPayload(question="q", reasoning_effort="max")

    async def test_null_values_take_the_saved_ones(self, test_db_session, model_b, monkeypatch):
        test_db_session.add(ModelDefaults(llm_id=model_b.id, reasoning_effort="high", top_p=0.4))
        test_db_session.commit()

        efforts, sampling = await _arena_turn(test_db_session, model_b.id, monkeypatch)

        assert efforts == ["high"]
        assert sampling == (0.6, 0.4)  # recommended creativity, saved diversity

    async def test_without_saved_values_the_settings_and_the_model_apply(
        self, test_db_session, model_a, monkeypatch
    ):
        from src.domains.user_settings.repository import User_Settings_Repository

        repo = User_Settings_Repository(test_db_session)
        repo.set_default_reasoning_effort(repo.get_or_create(), "xhigh")
        test_db_session.commit()

        efforts, sampling = await _arena_turn(test_db_session, model_a.id, monkeypatch)

        assert efforts == ["xhigh"]
        assert sampling == (0.7, 0.8)

    async def test_explicit_values_win(self, test_db_session, model_b, monkeypatch):
        test_db_session.add(
            ModelDefaults(llm_id=model_b.id, reasoning_effort="high", temperature=1.1, top_p=0.4)
        )
        test_db_session.commit()

        efforts, sampling = await _arena_turn(
            test_db_session,
            model_b.id,
            monkeypatch,
            reasoning_effort="none",
            temperature=0.3,
            top_p=0.9,
        )

        assert efforts == ["none"]
        assert sampling == (0.3, 0.9)

    async def test_web_search_stays_the_global_setting(self, test_db_session, model_b, monkeypatch):
        # Only effort and sampling follow the model in the Arena (#609).
        test_db_session.add(ModelDefaults(llm_id=model_b.id, web_search_enabled=True))
        test_db_session.commit()
        captured: dict = {}
        monkeypatch.setattr("src.domains.arena.services.plan_turn", _plan_spy(captured))
        monkeypatch.setattr("src.domains.arena.services.detect_supports_vision", lambda link: False)
        service = ArenaService(test_db_session)
        monkeypatch.setattr(service.runner, "astream_text", _stub_stream)

        async for _ in service.query_llm_stream(model_b.id, ArenaQueryPayload(question="hi")):
            pass

        assert captured["web_search_enabled"] is False
