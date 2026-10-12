"""The compaction threshold a user adjusts (#607).

A conversation compacts when its request reaches a share of the working
window. That share is the conversation's ``compaction_threshold``, a whole
percent between 50 and 90 in steps of 5, copied at creation from the global
``default_compaction_threshold`` (80) and owned by the conversation afterwards
-- the reasoning-effort lifecycle.

One integer helper, ``trigger_tokens(W, pct)``, owns the trigger for every
reader: the middleware's trigger, the keep budget's room, the futility guard,
the summarizer's trim budget and the memory brake's shrink test. It never
leaves less than the output budget's floor and its margin floor (512 + 256)
for the answer. At 80 % the trigger is unchanged for any window of 3836
tokens or more, the keep for any window of 2229 tokens or more.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately as approx_token_count
from pydantic import ValidationError

from src.agents.output_budget import MARGIN_FLOOR_TOKENS, OUTPUT_BUDGET_FLOOR_TOKENS
from src.agents.runner import (
    MEMORY_WARNING_FRACTION,
    AgentRunner,
    compaction_cutoff,
    keep_token_budget,
    summarization_triggers,
    summarize_trim_budget,
    summary_cap,
)
from src.agents.token_accounting import RequestOverhead
from src.domains.conversations.schemas import ConversationCreate, ConversationUpdate
from src.domains.user_settings.repository import User_Settings_Repository
from src.domains.user_settings.schemas import UserSettingsUpdate
from src.entities.Conversation import Conversation
from src.entities.UserSettings import UserSettings
from src.engines.working_window import (
    COMPACTION_THRESHOLD_MAX,
    COMPACTION_THRESHOLD_MIN,
    COMPACTION_THRESHOLD_STEP,
    COMPACTION_THRESHOLDS,
    DEFAULT_COMPACTION_THRESHOLD,
    TRIGGER_ANSWER_ROOM_TOKENS,
    trigger_tokens,
)
from tests._helpers import ToolableFakeChatModel
from tests.test_compaction_keep import (
    _ScriptedSummaryModel,
    _after,
    _conversation,
    _middleware,
    _text,
)
from tests.test_user_settings import _plan_spy, _stream_spy

pytestmark = pytest.mark.unit

WINDOWS = (2048, 4096, 8192, 40_960)
BOUNDS_AND_DEFAULT = (50, 80, 90)


@pytest.fixture
def window_engine(monkeypatch):
    """The bare engine whose loaded child reports the window ``_middleware``
    sets (the middleware reads the window from the configured engine)."""
    from src.core import config
    from tests.test_compaction_keep import _WindowEngine

    monkeypatch.setattr(config, "LLM_Engine", _WindowEngine)
    yield _WindowEngine
    _WindowEngine.window = None
    _WindowEngine._last_used = None


# ===================== the bounds =====================


def test_the_bounds_and_the_default():
    assert (COMPACTION_THRESHOLD_MIN, COMPACTION_THRESHOLD_MAX) == (50, 90)
    assert COMPACTION_THRESHOLD_STEP == 5
    assert DEFAULT_COMPACTION_THRESHOLD == 80
    assert COMPACTION_THRESHOLDS == (50, 55, 60, 65, 70, 75, 80, 85, 90)


def test_the_answer_room_is_the_output_budgets_floor_and_margin():
    # Mirrored, not imported: the engine layer must not pull the agent layer
    # (and LangChain) in at boot.
    assert TRIGGER_ANSWER_ROOM_TOKENS == OUTPUT_BUDGET_FLOOR_TOKENS + MARGIN_FLOOR_TOKENS == 768


# ===================== the trigger =====================


def test_at_the_default_the_trigger_is_unchanged_from_3836_tokens():
    assert all(trigger_tokens(window, 80) == int(0.8 * window) for window in range(3836, 140_000))
    assert trigger_tokens(3835, 80) < int(0.8 * 3835)


@pytest.mark.parametrize("pct", BOUNDS_AND_DEFAULT)
@pytest.mark.parametrize("window", WINDOWS)
def test_the_trigger_leaves_the_floor_and_margin_for_the_answer(window, pct):
    trigger = trigger_tokens(window, pct)
    assert 1 <= trigger <= window * pct // 100
    assert window - trigger >= OUTPUT_BUDGET_FLOOR_TOKENS + MARGIN_FLOOR_TOKENS


@pytest.mark.parametrize(
    ("window", "pct", "trigger"),
    [
        (2048, 50, 1024),
        (2048, 80, 1280),
        (2048, 90, 1280),
        (4096, 50, 2048),
        (4096, 80, 3276),
        (4096, 90, 3328),
        (8192, 90, 7372),
        (40_960, 50, 20_480),
        (40_960, 90, 36_864),
    ],
)
def test_the_trigger_values(window, pct, trigger):
    assert trigger_tokens(window, pct) == trigger


@pytest.mark.parametrize("window", [1, 300, 768])
def test_a_window_too_small_for_the_answer_room_compacts_as_early_as_it_can(window):
    assert trigger_tokens(window, 50) == 1


@pytest.mark.parametrize("pct", BOUNDS_AND_DEFAULT)
def test_the_middleware_trigger_is_the_helper(pct):
    assert summarization_triggers(10_000, pct) == [("tokens", trigger_tokens(10_000, pct))]


def test_without_a_window_the_message_fallback_ignores_the_threshold():
    assert summarization_triggers(None, 50) == [("messages", 20)]


# ===================== what a compaction keeps =====================


# headroom = trigger - (keep + summary cap): how far under the trigger the
# state a compaction leaves sits, with ordinary messages.
@pytest.mark.parametrize(
    ("window", "pct", "keep", "headroom"),
    [
        (2048, 50, 512, 256),
        (4096, 50, 1127, 409),
        (8192, 50, 2253, 819),
        (40_960, 50, 15_360, 4096),
        (2048, 80, 768, 256),
        (4096, 80, 1638, 1126),
        (8192, 80, 3276, 2253),
        (40_960, 80, 16_384, 15_360),
        (2048, 90, 768, 256),
        (4096, 90, 1638, 1178),
        (8192, 90, 3276, 3072),
        (40_960, 90, 16_384, 19_456),
    ],
)
def test_the_keep_leaves_a_pinned_headroom_under_the_trigger(window, pct, keep, headroom):
    assert keep_token_budget(window, pct) == keep
    trigger = trigger_tokens(window, pct)
    assert trigger - keep - summary_cap(window) == headroom
    # The margin under the trigger scales with it: a fifth of it, at least 256.
    assert headroom >= max(256, trigger // 5)


def test_at_the_default_the_keep_is_unchanged_from_2229_tokens():
    def old(window):
        return max(1, min(int(0.4 * window), int(0.8 * window) - summary_cap(window) - 256))

    assert all(keep_token_budget(window, 80) == old(window) for window in range(2229, 140_000))
    assert keep_token_budget(2228, 80) < old(2228)


@pytest.mark.parametrize("pct", BOUNDS_AND_DEFAULT)
@pytest.mark.parametrize("window", WINDOWS)
def test_a_compaction_of_ordinary_messages_does_not_refire_on_the_next_turn(window, pct):
    # Exchanges of ~100 tokens a message, enough of them to reach the trigger.
    trigger = trigger_tokens(window, pct)
    pairs = trigger // 200 + 2
    messages = _conversation(pairs, 100)
    assert approx_token_count(messages) >= trigger

    cut = compaction_cutoff(messages, window, approx_token_count, pct=pct)

    assert cut > 0
    after = approx_token_count(messages[cut:]) + summary_cap(window)
    # The next turn adds an answer and a question of the same size: still
    # under the trigger, so it does not compact again.
    assert after + 200 < trigger


@pytest.mark.parametrize("pct", BOUNDS_AND_DEFAULT)
def test_the_trim_budget_reads_the_threshold(pct):
    window = 40_960
    expected = max(4000, trigger_tokens(window, pct) - summary_cap(window))
    assert summarize_trim_budget(window, None, pct) == expected
    assert summarize_trim_budget(window, window, pct) > 0


def test_the_futility_guard_judges_against_the_threshold():
    # One small exchange, then a current question larger than the keep: the
    # cutoff can only summarize the small exchange, a gain under 256. Over a
    # 50 % trigger the guard runs and skips it; under the 80 % trigger the
    # guard is not asked (the trigger has not fired).
    window = 8192
    messages = [
        HumanMessage(_text(300), id="h0"),
        AIMessage(_text(300), id="a0"),
        HumanMessage(_text(3600), id="q"),
    ]
    total = approx_token_count(messages)
    assert trigger_tokens(window, 50) <= total < trigger_tokens(window, 80)

    assert compaction_cutoff(messages, window, approx_token_count, pct=50) == 0
    assert compaction_cutoff(messages, window, approx_token_count, pct=80) == 1


# ===================== the middleware =====================


async def test_a_middleware_at_50_compacts_a_state_the_default_does_not(window_engine):
    window = 8192
    state = _conversation(14, 150)  # ~4400 tokens: over 50 %, under 80 %

    default = _middleware(_ScriptedSummaryModel(messages=iter([])), window=window)
    default._freeze(state)
    count = default.token_counter(state)
    assert trigger_tokens(window, 50) <= count < trigger_tokens(window, 80)
    assert await default.abefore_model({"messages": state}, None) is None

    summary = _ScriptedSummaryModel(messages=iter([]))
    low = _middleware(summary, window=window, compaction_threshold=50)
    assert low.compaction_threshold == 50
    assert low.trigger == [("tokens", trigger_tokens(window, 50))]
    assert low.keep == ("tokens", keep_token_budget(window, 50))
    after = _after(await low.abefore_model({"messages": state}, None))
    assert len(after) < len(state)
    assert len(summary.prompts) == 1


def test_build_middleware_requires_the_threshold():
    with pytest.raises(TypeError):
        AgentRunner()._build_middleware(ToolableFakeChatModel(messages=iter([])))


async def test_a_summarizing_turn_without_a_threshold_is_refused():
    from src.agents.runner import GenParams

    stream = AgentRunner().astream_text(
        llm=object(),
        user_message="hi",
        system_prompt="s",
        params=GenParams(temperature=0.5, top_p=0.9, max_tokens=64),
        thread_id="1",
        summarize=True,
    )
    with pytest.raises(ValueError, match="compaction_threshold"):
        await stream.__anext__()


async def test_the_compaction_log_line_names_the_threshold(window_engine, caplog):
    import logging

    window = 8192
    state = _conversation(14, 150)
    low = _middleware(
        _ScriptedSummaryModel(messages=iter([])), window=window, compaction_threshold=50
    )
    with caplog.at_level(logging.INFO):
        await low.abefore_model({"messages": state}, None)
    (line,) = [r.getMessage() for r in caplog.records if "Conversation compacted" in r.getMessage()]
    assert "threshold=50%" in line
    assert line.isascii()


async def test_at_90_the_warning_never_fires_for_growth_the_threshold_allows():
    """Above the warning's 85 %, a next request between 85 % and a 90 %
    threshold is not compacted yet, but the projection counts the compaction
    the threshold will run: the warning speaks of what compaction cannot save,
    not of growth the user's threshold allows."""
    from tests.test_memory_warning import _Engine, _agent_with_state, _budget, _long_thread

    window = 20_000
    overhead = RequestOverhead(fixed_est=200, fixed_text="system prompt")
    runner = AgentRunner()

    async def _project(state):
        return await runner._project_next_request(
            _agent_with_state(state),
            {},
            window,
            overhead=overhead,
            allocated_window=window,
            compaction_threshold=90,
        )

    state = None
    for tokens_each in range(700, 1000, 5):
        candidate = _long_thread(pairs=10, tokens_each=tokens_each)
        _, conversation = await _project(candidate)
        if MEMORY_WARNING_FRACTION * window <= conversation < trigger_tokens(window, 90):
            state = candidate
            break
    assert state is not None
    projected, _ = await _project(state)
    assert projected < MEMORY_WARNING_FRACTION * window

    event = await runner._memory_warning_event(
        _agent_with_state(state),
        {},
        _budget(1.0),
        window,
        overhead=overhead,
        allocated_window=window,
        engine=_Engine(window),
        compaction_threshold=90,
    )
    assert event is None


# ===================== validation: schemas and entities =====================


@pytest.mark.parametrize("value", [45, 95, 82, True, 0, -5, 100, "80", 80.5])
def test_the_schemas_reject_a_threshold_off_the_scale(value):
    with pytest.raises(ValidationError):
        ConversationCreate(llm_id=1, compaction_threshold=value)
    with pytest.raises(ValidationError):
        ConversationUpdate(compaction_threshold=value)
    with pytest.raises(ValidationError):
        UserSettingsUpdate(default_compaction_threshold=value)


@pytest.mark.parametrize("value", COMPACTION_THRESHOLDS)
def test_the_schemas_accept_every_step(value):
    assert ConversationCreate(llm_id=1, compaction_threshold=value).compaction_threshold == value
    assert ConversationUpdate(compaction_threshold=value).compaction_threshold == value
    assert (
        UserSettingsUpdate(default_compaction_threshold=value).default_compaction_threshold == value
    )


def test_a_settings_update_with_only_the_threshold_is_a_valid_update():
    assert UserSettingsUpdate(default_compaction_threshold=60).default_compaction_threshold == 60


@pytest.mark.parametrize("value", [45, 95, 82, True, None, "80", 80.0])
def test_the_entities_reject_a_threshold_off_the_scale(value):
    with pytest.raises(ValueError):
        Conversation().compaction_threshold = value
    with pytest.raises(ValueError):
        UserSettings().default_compaction_threshold = value


def test_the_entities_default_to_80(test_db_session, mock_llm):
    settings = UserSettings()
    conversation = Conversation(llm_id=mock_llm.id, name="c")
    test_db_session.add_all([settings, conversation])
    test_db_session.commit()
    test_db_session.refresh(settings)
    test_db_session.refresh(conversation)
    assert settings.default_compaction_threshold == 80
    assert conversation.compaction_threshold == 80


# ===================== repository =====================


def test_the_global_default_reads_80_without_a_row(test_db_session):
    repo = User_Settings_Repository(test_db_session)
    assert repo.get_default_compaction_threshold() == 80
    assert test_db_session.query(UserSettings).count() == 0


def test_the_global_default_is_set(test_db_session):
    repo = User_Settings_Repository(test_db_session)
    repo.set_default_compaction_threshold(repo.get_or_create(), 65)
    test_db_session.commit()
    assert repo.get_default_compaction_threshold() == 65


# ===================== endpoints: the lifecycle =====================


class TestUserSettingsCompactionThreshold:
    def test_get_reports_80_by_default(self, client):
        assert client.get("/erudi/user_settings/").json()["default_compaction_threshold"] == 80

    @pytest.mark.parametrize("value", [50, 75, 90])
    def test_put_with_only_the_threshold_persists(self, client, value):
        response = client.put("/erudi/user_settings/", json={"default_compaction_threshold": value})
        assert response.status_code == 200, response.text
        assert response.json()["default_compaction_threshold"] == value
        assert client.get("/erudi/user_settings/").json()["default_compaction_threshold"] == value

    @pytest.mark.parametrize("value", [45, 95, 82, True])
    def test_put_rejects_a_threshold_off_the_scale(self, client, value):
        response = client.put("/erudi/user_settings/", json={"default_compaction_threshold": value})
        assert response.status_code == 422
        assert client.get("/erudi/user_settings/").json()["default_compaction_threshold"] == 80

    def test_put_threshold_leaves_the_other_settings_untouched(self, client):
        client.put("/erudi/user_settings/", json={"language": "fr"})
        response = client.put("/erudi/user_settings/", json={"default_compaction_threshold": 60})
        assert response.json() == {
            "web_search_enabled": False,
            "language": "fr",
            "auto_update_enabled": True,
            "inference_backend": "auto",
            "default_reasoning_effort": "medium",
            "default_compaction_threshold": 60,
        }


class TestConversationCompactionThreshold:
    def _create(self, client, llm_id, **extra):
        payload = {"llm_id": llm_id, "temperature": 0.7, "top_p": 0.9, "max_tokens": 512}
        payload.update(extra)
        response = client.post("/erudi/conversations/", json=payload)
        assert response.status_code == 201, response.text
        return response.json()

    def test_creation_copies_the_global_default(self, client, mock_llm):
        assert self._create(client, mock_llm.id)["compaction_threshold"] == 80
        client.put("/erudi/user_settings/", json={"default_compaction_threshold": 55})
        assert self._create(client, mock_llm.id)["compaction_threshold"] == 55

    def test_creation_explicit_value_wins_over_global(self, client, mock_llm):
        client.put("/erudi/user_settings/", json={"default_compaction_threshold": 55})
        assert (
            self._create(client, mock_llm.id, compaction_threshold=90)["compaction_threshold"] == 90
        )

    def test_creation_rejects_a_threshold_off_the_scale(self, client, mock_llm):
        response = client.post(
            "/erudi/conversations/", json={"llm_id": mock_llm.id, "compaction_threshold": 82}
        )
        assert response.status_code == 422

    def test_a_later_global_change_does_not_retro_affect_existing(self, client, mock_llm):
        conv = self._create(client, mock_llm.id)
        client.put("/erudi/user_settings/", json={"default_compaction_threshold": 50})
        fetched = client.get(f"/erudi/conversations/{conv['id']}").json()
        assert fetched["compaction_threshold"] == 80

    def test_patch_persists_the_threshold(self, client, mock_llm):
        conv = self._create(client, mock_llm.id)
        response = client.patch(
            f"/erudi/conversations/{conv['id']}", json={"compaction_threshold": 65}
        )
        assert response.status_code == 200
        assert response.json()["compaction_threshold"] == 65
        assert client.get(f"/erudi/conversations/{conv['id']}").json()["compaction_threshold"] == 65

    @pytest.mark.parametrize("value", [45, 95, 82, True])
    def test_patch_rejects_a_threshold_off_the_scale(self, client, mock_llm, value):
        conv = self._create(client, mock_llm.id)
        response = client.patch(
            f"/erudi/conversations/{conv['id']}", json={"compaction_threshold": value}
        )
        assert response.status_code == 422
        assert client.get(f"/erudi/conversations/{conv['id']}").json()["compaction_threshold"] == 80

    def test_patch_without_the_field_leaves_it_unchanged(self, client, mock_llm):
        conv = self._create(client, mock_llm.id, compaction_threshold=70)
        client.patch(f"/erudi/conversations/{conv['id']}", json={"name": "Renamed"})
        fetched = client.get(f"/erudi/conversations/{conv['id']}").json()
        assert fetched["compaction_threshold"] == 70 and fetched["name"] == "Renamed"

    def test_list_response_exposes_the_field(self, client, mock_llm):
        self._create(client, mock_llm.id)
        rows = client.get("/erudi/conversations/").json()
        assert all(row["compaction_threshold"] == 80 for row in rows)


# ===================== services: the turn runs at the conversation's value =====================


async def _stream_kwargs(service, conversation_id, monkeypatch) -> dict:
    from src.domains.conversations.schemas import ConversationQuery

    stream_kwargs: dict = {}
    monkeypatch.setattr("src.domains.conversations.services.plan_turn", _plan_spy({}))
    monkeypatch.setattr(
        "src.domains.conversations.services.detect_supports_vision", lambda link: False
    )
    monkeypatch.setattr(service.runner, "astream_text", _stream_spy(stream_kwargs))
    async for _ in service.query_and_respond_stream(
        conversation_id, ConversationQuery(question="hi")
    ):
        pass
    return stream_kwargs


async def test_the_turn_runs_at_the_conversations_threshold(test_db_session, mock_llm, monkeypatch):
    from src.domains.conversations.services import ConversationService

    service = ConversationService(test_db_session)
    conversation = service.create_conversation(llm_id=mock_llm.id, compaction_threshold=60)
    test_db_session.commit()

    stream_kwargs = await _stream_kwargs(service, conversation.id, monkeypatch)

    assert stream_kwargs["summarize"] is True
    assert stream_kwargs["compaction_threshold"] == 60


async def test_a_patched_threshold_is_used_by_the_next_turn(test_db_session, mock_llm, monkeypatch):
    from src.domains.conversations.services import ConversationService

    service = ConversationService(test_db_session)
    conversation = service.create_conversation(llm_id=mock_llm.id)
    assert conversation.compaction_threshold == 80
    service.update_conversation(conversation.id, compaction_threshold=50)
    test_db_session.commit()

    stream_kwargs = await _stream_kwargs(service, conversation.id, monkeypatch)

    assert stream_kwargs["compaction_threshold"] == 50


def test_creation_without_an_explicit_value_copies_the_global(test_db_session, mock_llm):
    from src.domains.conversations.services import ConversationService

    repo = User_Settings_Repository(test_db_session)
    repo.set_default_compaction_threshold(repo.get_or_create(), 85)
    test_db_session.commit()

    conversation = ConversationService(test_db_session).create_conversation(llm_id=mock_llm.id)
    assert conversation.compaction_threshold == 85
