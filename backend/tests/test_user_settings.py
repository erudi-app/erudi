"""User settings singleton + web-search toggles (issue #310).

Three layers, following the startup-domain pattern:
1. Entity: ``UserSettings`` singleton (mirrors ``StartupVariables``) with
   ``web_search_enabled`` defaulting to False (opt-in egress: nothing leaves
   the machine until the user says so).
2. Repository + endpoints: GET/PUT ``/erudi/user_settings/``.
3. Conversation wiring: ``web_search_enabled`` column copied from the GLOBAL
   default at creation (the conversation owns it afterwards — a later global
   change never retro-affects existing conversations), exposed on the
   conversation GET/POST/PATCH like temperature, and threaded into
   ``plan_turn`` on every turn (conversation: per-conversation flag; arena:
   the global setting — arena panels have no conversation row).
"""

import pytest

from src.agents import reasoning_effort as reasoning_effort_module
from src.agents.kb_mode import TurnPlan
from src.agents.reasoning_effort import ReasoningLever
from src.domains.user_settings.repository import User_Settings_Repository
from src.engines.reasoning_lever import LeverVerdict
from src.entities.UserSettings import UserSettings

pytestmark = pytest.mark.unit


# ============ Entity ============


class TestUserSettingsEntity:
    def test_default_web_search_disabled(self, test_db_session):
        settings = UserSettings()
        test_db_session.add(settings)
        test_db_session.commit()
        test_db_session.refresh(settings)
        assert settings.web_search_enabled is False

    def test_boolean_validator_rejects_non_boolean(self):
        settings = UserSettings()
        with pytest.raises(ValueError):
            settings.web_search_enabled = "yes"

    def test_default_language_is_english(self, test_db_session):
        settings = UserSettings()
        test_db_session.add(settings)
        test_db_session.commit()
        test_db_session.refresh(settings)
        assert settings.language == "en"

    def test_language_validator_rejects_unknown_code(self):
        settings = UserSettings()
        with pytest.raises(ValueError):
            settings.language = "de"

    def test_default_auto_update_enabled(self, test_db_session):
        # The shipped behaviour is unchanged for anyone who never opens the
        # setting: updates keep downloading on their own until refused.
        settings = UserSettings()
        test_db_session.add(settings)
        test_db_session.commit()
        test_db_session.refresh(settings)
        assert settings.auto_update_enabled is True

    def test_default_inference_backend_is_auto(self, test_db_session):
        settings = UserSettings()
        test_db_session.add(settings)
        test_db_session.commit()
        test_db_session.refresh(settings)
        assert settings.inference_backend == "auto"

    def test_inference_backend_validator_rejects_unknown_value(self):
        settings = UserSettings()
        with pytest.raises(ValueError):
            settings.inference_backend = "gpu"

    def test_auto_update_validator_rejects_non_boolean(self):
        settings = UserSettings()
        with pytest.raises(ValueError):
            settings.auto_update_enabled = "yes"

    def test_default_reasoning_effort_is_medium(self, test_db_session):
        # medium is the level at which a model that reasons naturally behaves
        # exactly as it did before the setting existed.
        settings = UserSettings()
        test_db_session.add(settings)
        test_db_session.commit()
        test_db_session.refresh(settings)
        assert settings.default_reasoning_effort == "medium"

    @pytest.mark.parametrize("level", ["none", "low", "medium", "high", "xhigh"])
    def test_every_level_is_accepted(self, level):
        settings = UserSettings()
        settings.default_reasoning_effort = level
        assert settings.default_reasoning_effort == level

    def test_reasoning_effort_validator_rejects_unknown_level(self):
        settings = UserSettings()
        with pytest.raises(ValueError):
            settings.default_reasoning_effort = "extra-high"


# ============ Repository ============


class TestUserSettingsRepository:
    def test_get_or_create_creates_singleton_with_default(self, test_db_session):
        repo = User_Settings_Repository(test_db_session)
        assert test_db_session.query(UserSettings).count() == 0
        settings = repo.get_or_create()
        assert settings.id is not None
        assert settings.web_search_enabled is False
        assert test_db_session.query(UserSettings).count() == 1

    def test_get_or_create_returns_existing(self, test_db_session):
        existing = UserSettings(web_search_enabled=True)
        test_db_session.add(existing)
        test_db_session.commit()
        repo = User_Settings_Repository(test_db_session)
        settings = repo.get_or_create()
        assert settings.id == existing.id
        assert settings.web_search_enabled is True

    def test_set_web_search_enabled(self, test_db_session):
        repo = User_Settings_Repository(test_db_session)
        settings = repo.get_or_create()
        repo.set_web_search_enabled(settings, True)
        test_db_session.commit()
        assert repo.get_or_create().web_search_enabled is True

    def test_get_web_search_enabled_default(self, test_db_session):
        repo = User_Settings_Repository(test_db_session)
        assert repo.get_web_search_enabled() is False

    def test_set_language(self, test_db_session):
        repo = User_Settings_Repository(test_db_session)
        settings = repo.get_or_create()
        repo.set_language(settings, "fr")
        test_db_session.commit()
        assert repo.get_or_create().language == "fr"

    def test_set_auto_update_enabled(self, test_db_session):
        repo = User_Settings_Repository(test_db_session)
        settings = repo.get_or_create()
        repo.set_auto_update_enabled(settings, False)
        test_db_session.commit()
        assert repo.get_or_create().auto_update_enabled is False

    def test_set_inference_backend(self, test_db_session):
        repo = User_Settings_Repository(test_db_session)
        settings = repo.get_or_create()
        repo.set_inference_backend(settings, "cpu")
        test_db_session.commit()
        assert repo.get_or_create().inference_backend == "cpu"

    def test_set_default_reasoning_effort(self, test_db_session):
        repo = User_Settings_Repository(test_db_session)
        settings = repo.get_or_create()
        repo.set_default_reasoning_effort(settings, "high")
        test_db_session.commit()
        assert repo.get_or_create().default_reasoning_effort == "high"

    def test_get_default_reasoning_effort_without_a_row(self, test_db_session):
        # A missing singleton IS the default: the read must not write one.
        repo = User_Settings_Repository(test_db_session)
        assert repo.get_default_reasoning_effort() == "medium"
        assert test_db_session.query(UserSettings).count() == 0


# ============ Endpoints ============


class TestUserSettingsEndpoints:
    def test_get_returns_defaults(self, client):
        response = client.get("/erudi/user_settings/")
        assert response.status_code == 200
        assert response.json() == {
            "web_search_enabled": False,
            "language": "en",
            "auto_update_enabled": True,
            "inference_backend": "auto",
            "default_reasoning_effort": "medium",
        }

    def test_put_updates_and_persists(self, client):
        response = client.put("/erudi/user_settings/", json={"web_search_enabled": True})
        assert response.status_code == 200
        assert response.json() == {
            "web_search_enabled": True,
            "language": "en",
            "auto_update_enabled": True,
            "inference_backend": "auto",
            "default_reasoning_effort": "medium",
        }
        assert client.get("/erudi/user_settings/").json() == {
            "web_search_enabled": True,
            "language": "en",
            "auto_update_enabled": True,
            "inference_backend": "auto",
            "default_reasoning_effort": "medium",
        }

    def test_put_back_to_false(self, client):
        client.put("/erudi/user_settings/", json={"web_search_enabled": True})
        response = client.put("/erudi/user_settings/", json={"web_search_enabled": False})
        assert response.json()["web_search_enabled"] is False

    def test_put_rejects_missing_field(self, client):
        response = client.put("/erudi/user_settings/", json={})
        assert response.status_code == 422

    @pytest.mark.parametrize("code", ["en", "fr", "es", "zh"])
    def test_put_language_persists(self, client, code):
        response = client.put("/erudi/user_settings/", json={"language": code})
        assert response.status_code == 200, response.text
        assert response.json()["language"] == code
        assert client.get("/erudi/user_settings/").json()["language"] == code

    def test_put_language_leaves_web_search_untouched(self, client):
        client.put("/erudi/user_settings/", json={"web_search_enabled": True})
        response = client.put("/erudi/user_settings/", json={"language": "es"})
        assert response.json() == {
            "web_search_enabled": True,
            "language": "es",
            "auto_update_enabled": True,
            "inference_backend": "auto",
            "default_reasoning_effort": "medium",
        }

    def test_put_web_search_leaves_language_untouched(self, client):
        client.put("/erudi/user_settings/", json={"language": "zh"})
        response = client.put("/erudi/user_settings/", json={"web_search_enabled": True})
        assert response.json() == {
            "web_search_enabled": True,
            "language": "zh",
            "auto_update_enabled": True,
            "inference_backend": "auto",
            "default_reasoning_effort": "medium",
        }

    @pytest.mark.parametrize("code", ["de", "EN", "fr-FR", "", 42])
    def test_put_rejects_unknown_language(self, client, code):
        response = client.put("/erudi/user_settings/", json={"language": code})
        assert response.status_code == 422
        assert client.get("/erudi/user_settings/").json()["language"] == "en"

    def test_get_reports_automatic_updates_on_by_default(self, client):
        # A fresh install must behave exactly as it did before the setting
        # existed, otherwise everyone silently stops receiving updates.
        assert client.get("/erudi/user_settings/").json()["auto_update_enabled"] is True

    def test_put_refuses_automatic_updates_and_it_survives(self, client):
        # The whole point of the setting: once refused, it stays refused across
        # reads -- a value that did not persist would let updates resume.
        response = client.put("/erudi/user_settings/", json={"auto_update_enabled": False})
        assert response.status_code == 200
        assert response.json()["auto_update_enabled"] is False
        assert client.get("/erudi/user_settings/").json()["auto_update_enabled"] is False

    def test_put_auto_update_leaves_the_other_settings_untouched(self, client):
        client.put("/erudi/user_settings/", json={"web_search_enabled": True})
        client.put("/erudi/user_settings/", json={"language": "fr"})
        response = client.put("/erudi/user_settings/", json={"auto_update_enabled": False})
        assert response.json() == {
            "web_search_enabled": True,
            "language": "fr",
            "auto_update_enabled": False,
            "inference_backend": "auto",
            "default_reasoning_effort": "medium",
        }

    def test_put_auto_update_back_on(self, client):
        client.put("/erudi/user_settings/", json={"auto_update_enabled": False})
        response = client.put("/erudi/user_settings/", json={"auto_update_enabled": True})
        assert response.json()["auto_update_enabled"] is True

    def test_put_pins_the_cpu_backend_and_it_survives(self, client):
        # The whole point of the setting: a user whose GPU the bundled CUDA
        # build cannot drive opts into CPU once and the app honours it on every
        # later boot.
        response = client.put("/erudi/user_settings/", json={"inference_backend": "cpu"})
        assert response.status_code == 200
        assert response.json()["inference_backend"] == "cpu"
        assert client.get("/erudi/user_settings/").json()["inference_backend"] == "cpu"

    def test_put_inference_backend_back_to_auto(self, client):
        # The choice is reversible -- a driver update makes the GPU usable again.
        client.put("/erudi/user_settings/", json={"inference_backend": "cpu"})
        response = client.put("/erudi/user_settings/", json={"inference_backend": "auto"})
        assert response.json()["inference_backend"] == "auto"

    def test_put_inference_backend_leaves_the_other_settings_untouched(self, client):
        client.put("/erudi/user_settings/", json={"web_search_enabled": True})
        client.put("/erudi/user_settings/", json={"language": "fr"})
        response = client.put("/erudi/user_settings/", json={"inference_backend": "cpu"})
        assert response.json() == {
            "web_search_enabled": True,
            "language": "fr",
            "auto_update_enabled": True,
            "inference_backend": "cpu",
            "default_reasoning_effort": "medium",
        }

    @pytest.mark.parametrize("value", ["gpu", "cuda", "CPU", "", 1])
    def test_put_rejects_an_unknown_inference_backend(self, client, value):
        response = client.put("/erudi/user_settings/", json={"inference_backend": value})
        assert response.status_code == 422
        assert client.get("/erudi/user_settings/").json()["inference_backend"] == "auto"

    @pytest.mark.parametrize("level", ["none", "low", "medium", "high", "xhigh"])
    def test_put_reasoning_effort_persists(self, client, level):
        response = client.put("/erudi/user_settings/", json={"default_reasoning_effort": level})
        assert response.status_code == 200, response.text
        assert response.json()["default_reasoning_effort"] == level
        assert client.get("/erudi/user_settings/").json()["default_reasoning_effort"] == level

    @pytest.mark.parametrize("level", ["extra-high", "MEDIUM", "off", "", 2])
    def test_put_rejects_an_unknown_reasoning_effort(self, client, level):
        response = client.put("/erudi/user_settings/", json={"default_reasoning_effort": level})
        assert response.status_code == 422
        assert client.get("/erudi/user_settings/").json()["default_reasoning_effort"] == "medium"

    def test_put_reasoning_effort_leaves_the_other_settings_untouched(self, client):
        client.put("/erudi/user_settings/", json={"language": "fr"})
        response = client.put("/erudi/user_settings/", json={"default_reasoning_effort": "none"})
        assert response.json() == {
            "web_search_enabled": False,
            "language": "fr",
            "auto_update_enabled": True,
            "inference_backend": "auto",
            "default_reasoning_effort": "none",
        }

    def test_put_rejects_a_value_that_is_not_a_boolean(self, client):
        # A payload the schema cannot read must leave the preference alone
        # rather than land as a truthy value the user never asked for.
        response = client.put("/erudi/user_settings/", json={"auto_update_enabled": "maybe"})
        assert response.status_code == 422
        assert client.get("/erudi/user_settings/").json()["auto_update_enabled"] is True


# ============ Conversation inheritance + PATCH ============


class TestConversationWebSearchField:
    def _create(self, client, llm_id, **extra):
        payload = {
            "llm_id": llm_id,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 512,
            "custom_prompt": "",
        }
        payload.update(extra)
        response = client.post("/erudi/conversations/", json=payload)
        assert response.status_code == 201, response.text
        return response.json()

    def test_creation_inherits_global_default_false(self, client, mock_llm):
        conv = self._create(client, mock_llm.id)
        assert conv["web_search_enabled"] is False

    def test_creation_inherits_global_default_true(self, client, mock_llm):
        client.put("/erudi/user_settings/", json={"web_search_enabled": True})
        conv = self._create(client, mock_llm.id)
        assert conv["web_search_enabled"] is True

    def test_creation_explicit_value_wins_over_global(self, client, mock_llm):
        client.put("/erudi/user_settings/", json={"web_search_enabled": True})
        conv = self._create(client, mock_llm.id, web_search_enabled=False)
        assert conv["web_search_enabled"] is False

    def test_global_change_does_not_retro_affect_existing(self, client, mock_llm):
        conv = self._create(client, mock_llm.id)
        assert conv["web_search_enabled"] is False
        client.put("/erudi/user_settings/", json={"web_search_enabled": True})
        fetched = client.get(f"/erudi/conversations/{conv['id']}").json()
        assert fetched["web_search_enabled"] is False

    def test_patch_toggles_the_conversation(self, client, mock_llm):
        conv = self._create(client, mock_llm.id)
        response = client.patch(
            f"/erudi/conversations/{conv['id']}", json={"web_search_enabled": True}
        )
        assert response.status_code == 200
        assert response.json()["web_search_enabled"] is True
        fetched = client.get(f"/erudi/conversations/{conv['id']}").json()
        assert fetched["web_search_enabled"] is True

    def test_patch_without_the_field_leaves_it_unchanged(self, client, mock_llm):
        conv = self._create(client, mock_llm.id, web_search_enabled=True)
        client.patch(f"/erudi/conversations/{conv['id']}", json={"name": "Renamed"})
        fetched = client.get(f"/erudi/conversations/{conv['id']}").json()
        assert fetched["web_search_enabled"] is True
        assert fetched["name"] == "Renamed"

    def test_list_response_exposes_the_field(self, client, mock_llm):
        self._create(client, mock_llm.id)
        rows = client.get("/erudi/conversations/").json()
        assert all("web_search_enabled" in row for row in rows)


class TestConversationReasoningEffortField:
    """Same lifecycle as the web toggle: copied at creation, owned afterwards."""

    def _create(self, client, llm_id, **extra):
        payload = {"llm_id": llm_id, "temperature": 0.7, "top_p": 0.9, "max_tokens": 512}
        payload.update(extra)
        response = client.post("/erudi/conversations/", json=payload)
        assert response.status_code == 201, response.text
        return response.json()

    def test_creation_inherits_the_global_default(self, client, mock_llm):
        assert self._create(client, mock_llm.id)["reasoning_effort"] == "medium"

    def test_creation_inherits_a_changed_global_default(self, client, mock_llm):
        client.put("/erudi/user_settings/", json={"default_reasoning_effort": "xhigh"})
        assert self._create(client, mock_llm.id)["reasoning_effort"] == "xhigh"

    def test_creation_explicit_value_wins_over_global(self, client, mock_llm):
        client.put("/erudi/user_settings/", json={"default_reasoning_effort": "xhigh"})
        conv = self._create(client, mock_llm.id, reasoning_effort="none")
        assert conv["reasoning_effort"] == "none"

    def test_a_later_global_change_does_not_retro_affect_existing(self, client, mock_llm):
        conv = self._create(client, mock_llm.id)
        client.put("/erudi/user_settings/", json={"default_reasoning_effort": "none"})
        fetched = client.get(f"/erudi/conversations/{conv['id']}").json()
        assert fetched["reasoning_effort"] == "medium"

    def test_patch_persists_the_level(self, client, mock_llm):
        conv = self._create(client, mock_llm.id)
        response = client.patch(
            f"/erudi/conversations/{conv['id']}", json={"reasoning_effort": "high"}
        )
        assert response.status_code == 200
        assert response.json()["reasoning_effort"] == "high"
        assert client.get(f"/erudi/conversations/{conv['id']}").json()["reasoning_effort"] == "high"

    def test_patch_rejects_an_unknown_level(self, client, mock_llm):
        conv = self._create(client, mock_llm.id)
        response = client.patch(
            f"/erudi/conversations/{conv['id']}", json={"reasoning_effort": "extra-high"}
        )
        assert response.status_code == 422
        assert (
            client.get(f"/erudi/conversations/{conv['id']}").json()["reasoning_effort"] == "medium"
        )

    def test_patch_without_the_field_leaves_it_unchanged(self, client, mock_llm):
        conv = self._create(client, mock_llm.id, reasoning_effort="low")
        client.patch(f"/erudi/conversations/{conv['id']}", json={"name": "Renamed"})
        fetched = client.get(f"/erudi/conversations/{conv['id']}").json()
        assert fetched["reasoning_effort"] == "low" and fetched["name"] == "Renamed"

    def test_list_response_exposes_the_field(self, client, mock_llm):
        self._create(client, mock_llm.id)
        rows = client.get("/erudi/conversations/").json()
        assert all("reasoning_effort" in row for row in rows)


# ============ Turn wiring: the flag reaches plan_turn ============


def _plan_spy(captured):
    def spy(llm, **kwargs):
        captured.update(kwargs)
        return TurnPlan(
            system_prompt="sys",
            tools=[],
            kb_context_block=None,
            kb_language_line="",
            context=None,
        )

    return spy


async def _stub_stream(**kwargs):
    # Honors the runner contract: event dicts with emit_events, str otherwise.
    if kwargs.get("emit_events"):
        yield {"t": "answer", "text": "ok"}
    else:
        yield "ok"


def _stream_spy(captured):
    """``astream_text`` stand-in that records what the service handed it."""

    async def spy(**kwargs):
        captured.update(kwargs)
        async for chunk in _stub_stream(**kwargs):
            yield chunk

    return spy


def _pin_lever(monkeypatch, lever, is_thinker):
    """Pin the per-artifact lever verdict and record the levels asked for.

    The verdict comes from the artifact on disk, which a wiring test has none
    of: pinning it keeps these tests about the PLUMBING (which level reaches
    which call), with the table itself covered in test_reasoning_effort.
    """
    levels: list = []
    original = reasoning_effort_module.resolve_effort_plan

    def _spy(level, _lever, _is_thinker):
        levels.append(level)
        return original(level, lever, is_thinker)

    monkeypatch.setattr(
        "src.engines.reasoning_lever.model_reasoning_lever",
        lambda local_path, llm_id=None: LeverVerdict(lever=lever, is_thinker=is_thinker),
    )
    monkeypatch.setattr(reasoning_effort_module, "resolve_effort_plan", _spy)
    return levels


class TestConversationTurnThreadsTheFlag:
    async def test_query_stream_passes_the_conversation_flag(
        self, test_db_session, mock_llm, monkeypatch
    ):
        from src.domains.conversations.schemas import ConversationQuery
        from src.domains.conversations.services import ConversationService

        service = ConversationService(test_db_session)
        conversation = service.create_conversation(llm_id=mock_llm.id, web_search_enabled=True)
        test_db_session.commit()

        captured = {}
        monkeypatch.setattr("src.domains.conversations.services.plan_turn", _plan_spy(captured))
        monkeypatch.setattr(
            "src.domains.conversations.services.detect_supports_vision",
            lambda link: False,
        )
        monkeypatch.setattr(service.runner, "astream_text", _stub_stream)

        payload = ConversationQuery(question="latest news?")
        async for _ in service.query_and_respond_stream(conversation.id, payload):
            pass

        assert captured["web_search_enabled"] is True

    async def test_create_without_explicit_value_copies_global(self, test_db_session, mock_llm):
        from src.domains.conversations.services import ConversationService

        repo = User_Settings_Repository(test_db_session)
        repo.set_web_search_enabled(repo.get_or_create(), True)
        test_db_session.commit()

        service = ConversationService(test_db_session)
        conversation = service.create_conversation(llm_id=mock_llm.id)
        assert conversation.web_search_enabled is True


class TestConversationTurnThreadsTheReasoningEffort:
    async def _run(self, db, mock_llm, monkeypatch, level, lever, is_thinker):
        from src.domains.conversations.schemas import ConversationQuery
        from src.domains.conversations.services import ConversationService

        service = ConversationService(db)
        conversation = service.create_conversation(llm_id=mock_llm.id, reasoning_effort=level)
        db.commit()

        levels = _pin_lever(monkeypatch, lever, is_thinker)
        plan_kwargs: dict = {}
        stream_kwargs: dict = {}
        monkeypatch.setattr("src.domains.conversations.services.plan_turn", _plan_spy(plan_kwargs))
        monkeypatch.setattr(
            "src.domains.conversations.services.detect_supports_vision", lambda link: False
        )
        monkeypatch.setattr(service.runner, "astream_text", _stream_spy(stream_kwargs))

        async for _ in service.query_and_respond_stream(
            conversation.id, ConversationQuery(question="how many primes under 100?")
        ):
            pass
        return levels, plan_kwargs, stream_kwargs

    async def test_the_conversations_level_reaches_the_wire_natively(
        self, test_db_session, mock_llm, monkeypatch
    ):
        levels, plan_kwargs, stream_kwargs = await self._run(
            test_db_session, mock_llm, monkeypatch, "high", ReasoningLever.NATIVE_EFFORT, True
        )
        assert levels == ["high"]
        # A native lever carries it: nothing is added to the prompt.
        assert plan_kwargs["effort_section"] is None
        assert stream_kwargs["effort_plan"].wire_effort == "high"

    async def test_a_level_without_a_lever_reaches_the_prompt_instead(
        self, test_db_session, mock_llm, monkeypatch
    ):
        levels, plan_kwargs, stream_kwargs = await self._run(
            test_db_session, mock_llm, monkeypatch, "xhigh", ReasoningLever.NONE, False
        )
        assert levels == ["xhigh"]
        assert "<think>" in plan_kwargs["effort_section"]
        assert stream_kwargs["effort_plan"].wire_effort is None
        assert stream_kwargs["effort_plan"].degraded_from == "xhigh"

    async def test_the_default_level_adds_nothing_to_a_reasoning_model(
        self, test_db_session, mock_llm, monkeypatch
    ):
        # The pinned regression: at the default level, a model that reasons
        # naturally gets exactly the turn it got before the feature existed.
        _levels, plan_kwargs, stream_kwargs = await self._run(
            test_db_session, mock_llm, monkeypatch, "medium", ReasoningLever.NATIVE_TOGGLE, True
        )
        assert plan_kwargs["effort_section"] is None
        assert stream_kwargs["effort_plan"].wire_effort is None

    async def test_a_patched_level_is_used_by_the_next_turn(
        self, test_db_session, mock_llm, monkeypatch
    ):
        from src.domains.conversations.schemas import ConversationQuery
        from src.domains.conversations.services import ConversationService

        service = ConversationService(test_db_session)
        conversation = service.create_conversation(llm_id=mock_llm.id)
        service.update_conversation(conversation.id, reasoning_effort="none")
        test_db_session.commit()

        levels = _pin_lever(monkeypatch, ReasoningLever.NATIVE_EFFORT, True)
        monkeypatch.setattr("src.domains.conversations.services.plan_turn", _plan_spy({}))
        monkeypatch.setattr(
            "src.domains.conversations.services.detect_supports_vision", lambda link: False
        )
        stream_kwargs: dict = {}
        monkeypatch.setattr(service.runner, "astream_text", _stream_spy(stream_kwargs))

        async for _ in service.query_and_respond_stream(
            conversation.id, ConversationQuery(question="hi")
        ):
            pass
        assert levels == ["none"]
        assert stream_kwargs["effort_plan"].wire_effort == "none"


class TestArenaFollowsTheGlobalSetting:
    async def test_arena_passes_the_global_flag(self, test_db_session, mock_llm, monkeypatch):
        from src.domains.arena.schemas import ArenaQueryPayload
        from src.domains.arena.services import ArenaService

        repo = User_Settings_Repository(test_db_session)
        repo.set_web_search_enabled(repo.get_or_create(), True)
        test_db_session.commit()

        captured = {}
        monkeypatch.setattr("src.domains.arena.services.plan_turn", _plan_spy(captured))
        monkeypatch.setattr("src.domains.arena.services.detect_supports_vision", lambda link: False)
        service = ArenaService(test_db_session)
        monkeypatch.setattr(service.runner, "astream_text", _stub_stream)

        payload = ArenaQueryPayload(question="latest news?")
        async for _ in service.query_llm_stream(mock_llm.id, payload):
            pass

        assert captured["web_search_enabled"] is True

    async def test_arena_reads_the_global_reasoning_effort(
        self, test_db_session, mock_llm, monkeypatch
    ):
        # An arena panel has no conversation row, so the level is the GLOBAL
        # setting, read on every turn (the web-search precedent).
        from src.domains.arena.schemas import ArenaQueryPayload
        from src.domains.arena.services import ArenaService

        repo = User_Settings_Repository(test_db_session)
        repo.set_default_reasoning_effort(repo.get_or_create(), "xhigh")
        test_db_session.commit()

        levels = _pin_lever(monkeypatch, ReasoningLever.NATIVE_EFFORT, True)
        captured = {}
        monkeypatch.setattr("src.domains.arena.services.plan_turn", _plan_spy(captured))
        monkeypatch.setattr("src.domains.arena.services.detect_supports_vision", lambda link: False)
        service = ArenaService(test_db_session)
        stream_kwargs = {}
        monkeypatch.setattr(service.runner, "astream_text", _stream_spy(stream_kwargs))

        async for _ in service.query_llm_stream(mock_llm.id, ArenaQueryPayload(question="hi")):
            pass

        assert levels == ["xhigh"]
        assert stream_kwargs["effort_plan"].wire_effort == "xhigh"

    async def test_arena_default_is_off(self, test_db_session, mock_llm, monkeypatch):
        from src.domains.arena.schemas import ArenaQueryPayload
        from src.domains.arena.services import ArenaService

        captured = {}
        monkeypatch.setattr("src.domains.arena.services.plan_turn", _plan_spy(captured))
        monkeypatch.setattr("src.domains.arena.services.detect_supports_vision", lambda link: False)
        service = ArenaService(test_db_session)
        monkeypatch.setattr(service.runner, "astream_text", _stub_stream)

        payload = ArenaQueryPayload(question="hi")
        async for _ in service.query_llm_stream(mock_llm.id, payload):
            pass

        assert captured["web_search_enabled"] is False
