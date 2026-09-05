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

from src.agents.kb_mode import TurnPlan
from src.domains.user_settings.repository import User_Settings_Repository
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
        }

    def test_put_updates_and_persists(self, client):
        response = client.put("/erudi/user_settings/", json={"web_search_enabled": True})
        assert response.status_code == 200
        assert response.json() == {
            "web_search_enabled": True,
            "language": "en",
            "auto_update_enabled": True,
            "inference_backend": "auto",
        }
        assert client.get("/erudi/user_settings/").json() == {
            "web_search_enabled": True,
            "language": "en",
            "auto_update_enabled": True,
            "inference_backend": "auto",
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
        }

    def test_put_web_search_leaves_language_untouched(self, client):
        client.put("/erudi/user_settings/", json={"language": "zh"})
        response = client.put("/erudi/user_settings/", json={"web_search_enabled": True})
        assert response.json() == {
            "web_search_enabled": True,
            "language": "zh",
            "auto_update_enabled": True,
            "inference_backend": "auto",
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
        }

    @pytest.mark.parametrize("value", ["gpu", "cuda", "CPU", "", 1])
    def test_put_rejects_an_unknown_inference_backend(self, client, value):
        response = client.put("/erudi/user_settings/", json={"inference_backend": value})
        assert response.status_code == 422
        assert client.get("/erudi/user_settings/").json()["inference_backend"] == "auto"

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
