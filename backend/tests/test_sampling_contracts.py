"""#388 contracts: sampling defaults on the API surface.

- ``GET /llms/*`` carries ``sampling_defaults`` (with ``source``) per row.
- ``POST /conversations/`` and the arena query resolve omitted sampling values
  from the model; explicit values win.
- A download copies the catalog row's hints to the local row and refreshes
  them best-effort after completion (never a precondition for ``completed``).
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status

from src.agents import runner as agent_runner
from src.core import config
from src.database import generation_hints as gh
from src.database.generation_hints import (
    FALLBACK_MAX_TOKENS,
    FALLBACK_TEMPERATURE,
    FALLBACK_TOP_P,
    UNBOUNDED_CONTEXT_TOKENS,
)
from src.domains.arena.schemas import ArenaQueryPayload
from src.domains.arena.services import ArenaService
from src.domains.conversations.schemas import ConversationCreate
from src.domains.conversations.services import ConversationService
from src.domains.llms import endpoints as llm_endpoints
from src.entities.Conversation import Conversation
from src.entities.DownloadJob import DownloadJobModel
from src.entities.Llm import Llm
from src.engines.base_engine import BaseEngine
from tests._helpers import ToolableFakeChatModel
from langchain_core.messages import AIMessage

pytestmark = pytest.mark.integration

_QWEN3_HINTS = {
    "base_repo": "Qwen/Qwen3-0.6B",
    "generation_config": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "do_sample": True},
    "supports_thinking": True,
    "context_length": 40960,
    "captured_at": "2026-08-28",
}


class _FakeEngine(BaseEngine):
    """generation_guard without a real model; MLX-like (no context cap)."""


@pytest.fixture(autouse=True)
def _engine(monkeypatch):
    monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
    gh.reset_capture_cache()
    yield
    gh.reset_capture_cache()


@pytest.fixture
def hinted_llm(test_db_session):
    llm = Llm(
        name="Qwen3 0.6B",
        local=1,
        link="/models/qwen3",
        type="qwen",
        param_size=0.6,
        quantized=True,
        generation_hints=_QWEN3_HINTS,
    )
    test_db_session.add(llm)
    test_db_session.commit()
    test_db_session.refresh(llm)
    return llm


# ---------------------------------------------------------------- GET /llms


class TestLlmListingCarriesSamplingDefaults:
    def test_row_with_hints(self, client, hinted_llm):
        body = client.get("/erudi/llms/local").json()
        row = next(r for r in body if r["id"] == hinted_llm.id)
        sd = row["sampling_defaults"]
        # Hints captured before the cascade carry no stage: they came from
        # the base repo's generation_config.json, which is what is reported.
        assert sd["source"] == "base_generation_config"
        assert sd["evidence"] is None
        assert (sd["temperature"], sd["top_p"], sd["top_k"]) == (0.6, 0.95, 20)
        assert sd["max_tokens"] == FALLBACK_MAX_TOKENS
        # The model's 40960-token window is clipped to the Conversation
        # validator's upper bound: the UI never gets a ceiling the API rejects.
        assert sd["max_tokens_cap"] == min(40960, UNBOUNDED_CONTEXT_TOKENS) == 32768
        assert sd["base_repo"] == "Qwen/Qwen3-0.6B"
        assert sd["min_p"] is None and sd["presence_penalty"] is None
        assert row["generation_hints"] == _QWEN3_HINTS

    def test_row_without_hints_is_none(self, client, mock_llm):
        body = client.get("/erudi/llms/local").json()
        row = next(r for r in body if r["id"] == mock_llm.id)
        sd = row["sampling_defaults"]
        assert sd["source"] == "none"
        assert sd["evidence"] is None
        assert (sd["temperature"], sd["top_p"], sd["max_tokens"]) == (
            FALLBACK_TEMPERATURE,
            FALLBACK_TOP_P,
            FALLBACK_MAX_TOKENS,
        )
        assert sd["max_tokens_cap"] == UNBOUNDED_CONTEXT_TOKENS
        assert sd["top_k"] is None
        assert row["generation_hints"] is None

    def test_row_with_facts_but_no_sampling_value_is_none(self, client, test_db_session):
        # The gated-base / no-numbers-in-the-card case: facts stored, source none.
        hints = {
            "base_repo": "meta-llama/Llama-3.2-1B-Instruct",
            "supports_thinking": False,
            "context_length": 131072,
            "captured_at": "2026-08-28",
            "source_stage": None,
            "evidence": None,
        }
        llm = Llm(
            name="Llama 3.2 1B",
            local=1,
            link="/models/llama",
            type="llama",
            param_size=1.0,
            quantized=True,
            generation_hints=hints,
        )
        test_db_session.add(llm)
        test_db_session.commit()
        sd = client.get(f"/erudi/llms/{llm.id}").json()["sampling_defaults"]
        assert sd["source"] == "none"
        assert sd["base_repo"] == "meta-llama/Llama-3.2-1B-Instruct"
        assert (sd["temperature"], sd["top_p"]) == (FALLBACK_TEMPERATURE, FALLBACK_TOP_P)

    def test_model_card_row_exposes_stage_and_evidence(self, client, test_db_session):
        hints = {
            "base_repo": "mistralai/Mistral-Small-24B-Instruct-2501",
            "generation_config": {"temperature": 0.15},
            "supports_thinking": False,
            "context_length": 32768,
            "captured_at": "2026-08-28",
            "source_stage": "model_card",
            "evidence": "We recommend using a relatively low temperature, such as `temperature=0.15`.",
        }
        llm = Llm(
            name="Mistral Small",
            local=1,
            link="/models/mistral",
            type="mistral",
            param_size=24.0,
            quantized=True,
            generation_hints=hints,
        )
        test_db_session.add(llm)
        test_db_session.commit()
        sd = client.get(f"/erudi/llms/{llm.id}").json()["sampling_defaults"]
        assert sd["source"] == "model_card"
        assert sd["temperature"] == 0.15
        assert sd["evidence"] == hints["evidence"]

    def test_by_id_and_full_listing_carry_it_too(self, client, hinted_llm):
        assert client.get(f"/erudi/llms/{hinted_llm.id}").json()["sampling_defaults"]["top_k"] == 20
        rows = client.get("/erudi/llms/").json()
        assert all("sampling_defaults" in r for r in rows)


# ---------------------------------------------------------------- conversations


class TestConversationCreateResolvesDefaults:
    def test_schema_values_are_optional(self):
        payload = ConversationCreate(llm_id=1)
        assert payload.temperature is None and payload.top_p is None and payload.max_tokens is None

    def test_omitted_values_come_from_the_model(self, client, hinted_llm):
        resp = client.post("/erudi/conversations/", json={"llm_id": hinted_llm.id})
        assert resp.status_code == status.HTTP_201_CREATED
        data = resp.json()
        assert (data["temperature"], data["top_p"], data["max_tokens"]) == (0.6, 0.95, 1024)

    def test_omitted_values_without_hints_are_the_fallback(self, client, mock_llm):
        data = client.post("/erudi/conversations/", json={"llm_id": mock_llm.id}).json()
        assert (data["temperature"], data["top_p"], data["max_tokens"]) == (
            FALLBACK_TEMPERATURE,
            FALLBACK_TOP_P,
            FALLBACK_MAX_TOKENS,
        )

    def test_explicit_values_win(self, client, hinted_llm):
        data = client.post(
            "/erudi/conversations/",
            json={
                "llm_id": hinted_llm.id,
                "temperature": 1.3,
                "top_p": 0.5,
                "max_tokens": 77,
            },
        ).json()
        assert (data["temperature"], data["top_p"], data["max_tokens"]) == (1.3, 0.5, 77)

    def test_partial_payload_resolves_only_the_missing_keys(self, client, hinted_llm):
        data = client.post(
            "/erudi/conversations/", json={"llm_id": hinted_llm.id, "temperature": 1.3}
        ).json()
        assert (data["temperature"], data["top_p"], data["max_tokens"]) == (1.3, 0.95, 1024)

    def test_the_stored_max_tokens_is_only_the_fallback_budget(self, client, hinted_llm):
        # The column stays (inert): it is what a turn runs with when the engine
        # cannot report the window it loaded with. Nothing writes it from the UI
        # any more -- the creation-time resolution is all that fills it.
        data = client.post("/erudi/conversations/", json={"llm_id": hinted_llm.id}).json()
        assert data["max_tokens"] == 1024

    def test_unknown_model_with_omitted_values_is_404(self, client):
        assert client.post("/erudi/conversations/", json={"llm_id": 987654}).status_code == 404

    def test_service_resolves_none(self, test_db_session, hinted_llm):
        conv = ConversationService(test_db_session).create_conversation(llm_id=hinted_llm.id)
        assert (conv.temperature, conv.top_p, conv.max_tokens) == (0.6, 0.95, 1024)

    def test_entity_and_repository_defaults_are_the_fallback(self, test_db_session, mock_llm):
        # The four divergent defaults (0.2/0.95, 0.2/0.5, 1.0/0.95, 1.0/0.95/512)
        # collapse onto the fallback constants: no stray default survives.
        conv = Conversation(llm_id=mock_llm.id, name="x")
        test_db_session.add(conv)
        test_db_session.commit()
        test_db_session.refresh(conv)
        assert (conv.temperature, conv.top_p, conv.max_tokens) == (
            FALLBACK_TEMPERATURE,
            FALLBACK_TOP_P,
            FALLBACK_MAX_TOKENS,
        )
        from src.domains.conversations.repository import ConversationRepository

        conv2 = ConversationRepository(test_db_session).create_conversation(llm_id=mock_llm.id)
        assert (conv2.temperature, conv2.top_p, conv2.max_tokens) == (
            FALLBACK_TEMPERATURE,
            FALLBACK_TOP_P,
            FALLBACK_MAX_TOKENS,
        )


# ---------------------------------------------------------------- arena


class _FakeEngine(BaseEngine):
    """Engine stub: generation_guard without spawning a model, no window."""


def _capturing_factory(captured):
    def factory(llm, **kw):
        captured.update(kw)
        return ToolableFakeChatModel(messages=iter([AIMessage(content="ok")]))

    return factory


class TestArenaResolvesDefaults:
    def test_schema_values_are_optional(self):
        payload = ArenaQueryPayload(question="q")
        assert payload.temperature is None and payload.top_p is None
        assert payload.max_new_tokens is None

    async def test_omitted_values_come_from_the_model(
        self, test_db_session, hinted_llm, monkeypatch
    ):
        captured = {}
        monkeypatch.setattr(agent_runner, "build_chat_model", _capturing_factory(captured))
        service = ArenaService(test_db_session)
        out = [
            t
            async for t in service.query_llm_stream(hinted_llm.id, ArenaQueryPayload(question="q"))
        ]
        assert "".join(out) == "ok"
        assert (captured["temperature"], captured["top_p"], captured["max_tokens"]) == (
            0.6,
            0.95,
            1024,
        )
        assert captured["sampling"].top_k == 20

    async def test_explicit_values_win(self, test_db_session, hinted_llm, monkeypatch):
        captured = {}
        monkeypatch.setattr(agent_runner, "build_chat_model", _capturing_factory(captured))
        payload = ArenaQueryPayload(question="q", temperature=1.1, top_p=0.4)
        async for _ in ArenaService(test_db_session).query_llm_stream(hinted_llm.id, payload):
            pass
        assert (captured["temperature"], captured["top_p"]) == (1.1, 0.4)

    async def test_a_max_new_tokens_from_an_old_client_is_accepted_and_ignored(
        self, test_db_session, hinted_llm, monkeypatch
    ):
        # The panel control is gone; the output budget comes from the window the
        # model runs in. An older client still gets an answer, not a 422, and
        # the number it sends changes nothing: the value the factory receives is
        # the model's own fallback budget.
        captured = {}
        monkeypatch.setattr(agent_runner, "build_chat_model", _capturing_factory(captured))
        payload = ArenaQueryPayload(question="q", max_new_tokens=33)
        async for _ in ArenaService(test_db_session).query_llm_stream(hinted_llm.id, payload):
            pass
        assert captured["max_tokens"] == 1024

    def test_endpoint_accepts_a_bare_question(self, client, hinted_llm):
        with patch.object(agent_runner, "build_chat_model", _capturing_factory({})):
            resp = client.post(f"/erudi/arena/{hinted_llm.id}/query", json={"question": "q"})
        assert resp.status_code == 200

    def test_endpoint_still_accepts_an_old_clients_max_new_tokens(self, client, hinted_llm):
        with patch.object(agent_runner, "build_chat_model", _capturing_factory({})):
            resp = client.post(
                f"/erudi/arena/{hinted_llm.id}/query",
                json={"question": "q", "max_new_tokens": 512},
            )
        assert resp.status_code == 200


# ------------------------------------------------- the conversation turn


class TestConversationTurnBudget:
    """What a conversation turn hands the model factory as ``max_tokens``.

    Only the FALLBACK budget: the real ceiling is computed per model call from
    the window the child reports (``src.agents.output_budget``, covered by
    ``tests/test_output_budget.py``). Here the contract is that nothing stale
    -- neither a per-turn payload field nor a client-chosen number -- reaches
    the request any more, and that an engine with no window still runs on the
    conversation row's value exactly as before.
    """

    async def _captured_kwargs(self, db, llm, monkeypatch, payload, **conversation_kwargs):
        from langgraph.checkpoint.memory import InMemorySaver

        from src.domains.conversations.schemas import ConversationQuery

        captured: dict = {}
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(agent_runner, "build_chat_model", _capturing_factory(captured))
        service = ConversationService(db, InMemorySaver())
        conversation = service.create_conversation(llm_id=llm.id, **conversation_kwargs)
        async for _ in service.query_and_respond_stream(
            conversation.id, ConversationQuery(**payload)
        ):
            pass
        return captured

    async def test_the_conversation_row_is_the_fallback_budget(
        self, test_db_session, mock_llm, monkeypatch
    ):
        captured = await self._captured_kwargs(
            test_db_session,
            mock_llm,
            monkeypatch,
            {"question": "q"},
            temperature=0.7,
            top_p=0.9,
            max_tokens=777,
        )
        assert captured["max_tokens"] == 777

    async def test_a_per_turn_max_new_tokens_is_accepted_and_ignored(
        self, test_db_session, mock_llm, monkeypatch
    ):
        captured = await self._captured_kwargs(
            test_db_session,
            mock_llm,
            monkeypatch,
            {"question": "q", "max_new_tokens": 4321},
            temperature=0.7,
            top_p=0.9,
            max_tokens=777,
        )
        assert captured["max_tokens"] == 777

    async def test_a_conversation_without_a_stored_value_falls_back_to_the_constant(
        self, test_db_session, mock_llm, monkeypatch
    ):
        captured = await self._captured_kwargs(
            test_db_session, mock_llm, monkeypatch, {"question": "q"}
        )
        assert captured["max_tokens"] == FALLBACK_MAX_TOKENS


# ---------------------------------------------------------------- downloads


class TestDownloadCarriesHints:
    @patch("src.domains.llms.endpoints.download_llm")
    @patch("pathlib.Path.exists")
    def test_catalog_download_copies_hints_to_the_local_row(
        self, mock_exists, mock_download, client, test_db_session
    ):
        mock_exists.return_value = False
        mock_download.return_value = AsyncMock()
        remote = Llm(
            name="Remote",
            local=0,
            type="qwen",
            link="Qwen/Qwen3-0.6B-MLX-4bit",
            param_size=0.6,
            generation_hints=_QWEN3_HINTS,
        )
        test_db_session.add(remote)
        test_db_session.commit()

        resp = client.post(f"/erudi/llms/{remote.id}/download")
        assert resp.status_code == 200
        local_id = resp.json()["local_model_id"]
        assert test_db_session.query(Llm).get(local_id).generation_hints == _QWEN3_HINTS

    def _run_task(self, monkeypatch, tmp_path, *, initial_hints, local_hints, remote_hints):
        """Drive _run_download_task over a MagicMock session (unit style, like
        TestRunDownloadTaskFailure) and return the Llm stand-in."""
        job = SimpleNamespace(status="pending", error_message=None, updated_at=None, progress=0.0)
        llm = SimpleNamespace(
            local=2,
            link=str(tmp_path / "final"),
            model_metadata=None,
            generation_hints=initial_hints,
            supports_tools=None,
            supports_tools_wire=None,
            supports_vision=None,
        )
        session = MagicMock()
        session.query.side_effect = lambda model: SimpleNamespace(
            get=lambda _id: job if model is DownloadJobModel else llm
        )
        monkeypatch.setattr(llm_endpoints, "SessionLocal", lambda: session)

        async def ok_download(**kwargs):
            return None

        monkeypatch.setattr(llm_endpoints, "download_llm", ok_download)
        monkeypatch.setattr(llm_endpoints, "_assert_downloaded_artifact_ok", lambda *a: None)
        monkeypatch.setattr(llm_endpoints, "measure_dir_size_bytes", lambda *a: 0)
        from src.domains.llms import repository as llm_repository

        for name in ("detect_supports_tools", "detect_wire_tools", "detect_supports_vision"):
            monkeypatch.setattr(llm_repository, name, lambda *a: None)
        monkeypatch.setattr(
            llm_endpoints,
            "read_local_generation_hints",
            lambda model_dir, base_repo=None: local_hints,
        )
        # ``remote_hints`` is either the dict the network capture returns or a
        # callable standing in for the capture itself (to make it raise).
        monkeypatch.setattr(
            llm_endpoints,
            "capture_generation_hints",
            remote_hints
            if callable(remote_hints)
            else (lambda base_repo, hf_api, quant_repo=None, quant_format=None: remote_hints),
        )
        # Pinned so the engine side threaded into the capture is the same on
        # every platform.
        monkeypatch.setattr(config, "LLM_Engine", SimpleNamespace(FORMAT_TAG="gguf"))
        monkeypatch.setattr(
            config,
            "get_hf_api",
            lambda: MagicMock(model_info=lambda *a, **k: SimpleNamespace(tags=[])),
        )

        llm_endpoints._run_download_task(
            "org/model", 1, tmp_path / "temp", tmp_path / "final", job_id=9
        )
        assert job.status == "completed"
        return llm

    def test_local_generation_config_is_read_first(self, monkeypatch, tmp_path):
        local = dict(_QWEN3_HINTS, captured_at="local")
        llm = self._run_task(
            monkeypatch,
            tmp_path,
            initial_hints=None,
            local_hints=local,
            remote_hints=dict(_QWEN3_HINTS, captured_at="remote"),
        )
        assert llm.generation_hints["captured_at"] == "local"

    def test_network_capture_when_the_artifact_has_none(self, monkeypatch, tmp_path):
        llm = self._run_task(
            monkeypatch,
            tmp_path,
            initial_hints=None,
            local_hints=None,
            remote_hints=dict(_QWEN3_HINTS, captured_at="remote"),
        )
        assert llm.generation_hints["captured_at"] == "remote"

    def test_local_facts_without_sampling_go_through_the_cascade(self, monkeypatch, tmp_path):
        # A Mistral MLX dir ships config.json but no usable generation_config:
        # the network cascade runs (base card) and the artifact's facts fill in
        # what the network could not read.
        local = {
            "base_repo": "org/model",
            "supports_thinking": False,
            "context_length": 32768,
            "captured_at": "local",
            "source_stage": None,
            "evidence": None,
        }
        seen = {}

        def capture(base_repo, hf_api, quant_repo=None, quant_format=None):
            seen["args"] = (base_repo, quant_repo, quant_format)
            return {
                "base_repo": base_repo,
                "generation_config": {"temperature": 0.15},
                "supports_thinking": None,
                "context_length": None,
                "captured_at": "remote",
                "source_stage": "model_card",
                "evidence": "We recommend `temperature=0.15`.",
            }

        llm = self._run_task(
            monkeypatch, tmp_path, initial_hints=None, local_hints=local, remote_hints=capture
        )
        # The downloaded repo is the quant repo, and the engine side rides along
        # (a GGUF download reads its window from the file's own metadata).
        assert seen["args"] == ("org/model", "org/model", "gguf")
        assert llm.generation_hints["source_stage"] == "model_card"
        assert llm.generation_hints["context_length"] == 32768
        assert llm.generation_hints["supports_thinking"] is False

    def test_local_facts_are_kept_when_the_network_has_nothing(self, monkeypatch, tmp_path):
        local = {
            "base_repo": "org/model",
            "supports_thinking": False,
            "context_length": 32768,
            "captured_at": "local",
            "source_stage": None,
            "evidence": None,
        }
        llm = self._run_task(
            monkeypatch, tmp_path, initial_hints=None, local_hints=local, remote_hints=None
        )
        assert llm.generation_hints == local

    def test_catalog_hints_are_kept(self, monkeypatch, tmp_path):
        llm = self._run_task(
            monkeypatch,
            tmp_path,
            initial_hints=_QWEN3_HINTS,
            local_hints=dict(_QWEN3_HINTS, captured_at="local"),
            remote_hints=None,
        )
        assert llm.generation_hints == _QWEN3_HINTS

    def test_capture_failure_never_blocks_completion(self, monkeypatch, tmp_path):
        def boom(*a, **k):
            raise RuntimeError("no network")

        llm = self._run_task(
            monkeypatch, tmp_path, initial_hints=None, local_hints=None, remote_hints=boom
        )
        assert llm.generation_hints is None
        assert Path(llm.link).name == "final"


# ------------------------------------------------- reasoning effort on the wire


class _WireEngine:
    """Engine stub with identity kwarg translation (MLX-shaped)."""

    @staticmethod
    def get_model_and_tokenizer(llm_id, link):
        return ({"base_url": "http://127.0.0.1:8080", "alias": f"erudi-{llm_id}"}, {})

    @staticmethod
    def _payload_model_value(handle):
        return "default_model"


class TestReasoningEffortReachesTheRequestBody:
    """``reasoning_effort`` is a NATIVE field of all three layers (1.1.2).

    It rides the ``ChatOpenAI`` constructor -- not ``extra_body`` -- because
    llama-server and mlx_vlm both read it off the top level of the request
    body, where langchain puts a set field and omits an unset one.
    """

    def _model(self, monkeypatch, **kwargs):
        from src.agents.model_factory import build_chat_model

        monkeypatch.setattr(config, "LLM_Engine", _WireEngine)
        return build_chat_model(
            SimpleNamespace(id=1, name="M", link="/models/m", param_size=7.0),
            temperature=0.3,
            top_p=0.8,
            max_tokens=12,
            **kwargs,
        )

    def test_without_a_plan_the_body_is_unchanged(self, monkeypatch):
        params = self._model(monkeypatch)._default_params
        assert "reasoning_effort" not in params

    def test_a_plan_with_no_wire_value_leaves_the_body_unchanged(self, monkeypatch):
        from src.agents.reasoning_effort import EffortPlan

        plan = EffortPlan(level="high", prompt_section="think hard", degraded_from="high")
        params = self._model(monkeypatch, effort_plan=plan)._default_params
        assert "reasoning_effort" not in params

    @pytest.mark.parametrize("level", ["none", "low", "medium", "high", "xhigh"])
    def test_a_native_level_lands_in_the_body(self, monkeypatch, level):
        from src.agents.reasoning_effort import EffortPlan

        plan = EffortPlan(level=level, wire_effort=level)
        model = self._model(monkeypatch, effort_plan=plan)
        assert model.reasoning_effort == level
        assert model._default_params["reasoning_effort"] == level
        # extra_body stays the sampling channel: the effort is NOT smuggled in.
        assert "reasoning_effort" not in model.extra_body

    async def test_the_one_shot_title_path_runs_at_none(self, monkeypatch):
        # A title is 2-4 words on a ~12-token budget: a reasoning model must not
        # spend it deliberating. ``enable_thinking`` alone does not reach a
        # template that only reads ``reasoning_effort``.
        captured: dict = {}
        monkeypatch.setattr(agent_runner, "build_chat_model", _capturing_factory(captured))
        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)

        async for _ in agent_runner.AgentRunner().astream_oneshot(
            llm=SimpleNamespace(id=1, name="M", link="/models/m", param_size=7.0),
            prompt_text="title this",
            temperature=0.3,
            top_p=0.8,
            max_tokens=12,
        ):
            pass
        assert captured["disable_thinking"] is True
        assert captured["effort_plan"].wire_effort == "none"


class TestTheCompactionSummaryRunsAtNone:
    """The summarizer gets its OWN client, pinned to ``reasoning_effort="none"``.

    Same child server, same sampling: only the reasoning field differs, so the
    second client costs a cached handle lookup and the summary gets written
    instead of deliberated about.
    """

    async def _calls(self, db, llm, monkeypatch, **conversation_kwargs):
        from langgraph.checkpoint.memory import InMemorySaver

        from src.domains.conversations.schemas import ConversationQuery
        from src.domains.conversations.services import ConversationService

        calls: list = []

        def factory(model, **kw):
            calls.append(kw)
            return ToolableFakeChatModel(messages=iter([AIMessage(content="ok")]))

        monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
        monkeypatch.setattr(agent_runner, "build_chat_model", factory)
        service = ConversationService(db, InMemorySaver())
        conversation = service.create_conversation(llm_id=llm.id, **conversation_kwargs)
        async for _ in service.query_and_respond_stream(
            conversation.id, ConversationQuery(question="q")
        ):
            pass
        return calls

    async def test_a_summarizing_turn_builds_a_second_none_client(
        self, test_db_session, mock_llm, monkeypatch
    ):
        calls = await self._calls(test_db_session, mock_llm, monkeypatch, reasoning_effort="xhigh")
        assert len(calls) == 2
        # Everything but the reasoning field matches the answering client.
        assert calls[1]["temperature"] == calls[0]["temperature"]
        assert calls[1]["max_tokens"] == calls[0]["max_tokens"]
        assert calls[1]["effort_plan"].wire_effort == "none"
