"""``GET /conversations/{id}/context`` and its pieces (#608).

The panel shows what the NEXT request of the conversation would carry, by
category, against the window the conversation works with: the projection
the amber warning uses (``project_next_request``), weighed once on the whole
set, then split. ``used`` = system prompt + tools + summary + messages; with
``T = trigger_tokens(W, pct)``, ``free = max(0, T - used)`` and the
auto-compaction reserve ``max(0, W - max(T, used))``. ``W`` is the working
window when this conversation's model is the loaded one, its trained window
otherwise. Read only: no lock, no model load, no commit, no re-assignment.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.agents import runner as runner_module
from src.agents.prompts import build_kb_system_prompt
from src.agents.runner import (
    AgentRunner,
    project_next_request,
)
from src.agents.token_accounting import (
    COUNTER_DENSE_TOKENS,
    COUNTER_WEIGHT_FLOOR,
    RequestOverhead,
    overhead_tokens,
    request_overhead,
)
from src.core import config
from src.core.exceptions import StateConflictException
from src.domains.conversations.services import ConversationService
from src.engines.base_engine import BaseEngine
from src.engines.working_window import trigger_tokens
from src.entities.Conversation import Conversation
from src.entities.Llm import Llm
from tests.test_compaction_keep import _alternating, _previous_summary_message, _seed

pytestmark = pytest.mark.unit

_TRAINED = 32_768


class _Engine(BaseEngine):
    """An engine whose loaded child is the handle ``_model`` serving
    ``_model_id``; its window and memory bound live on that handle, as on
    the real engines."""

    @classmethod
    def loaded_child_for(cls, llm_id):
        return isinstance(cls._model, dict) and cls._model_id == llm_id


def _load(llm_id=7, window=8192, bound=None):
    _Engine._model = {"context_tokens": window, "memory_bound_tokens": bound}
    _Engine._model_id = llm_id


class _NoBudget:
    def tokens_at_ceiling(self):
        return None


@pytest.fixture(autouse=True)
def _engine(monkeypatch):
    monkeypatch.setattr(_Engine, "_model", None)
    monkeypatch.setattr(_Engine, "_model_id", None)
    monkeypatch.setattr(config, "LLM_Engine", _Engine)

    def _no_build(*a, **kw):
        raise AssertionError("/context never builds a client (it would load the model)")

    monkeypatch.setattr(runner_module, "build_chat_model", _no_build)
    monkeypatch.setattr(
        runner_module,
        "MemoryBudget",
        SimpleNamespace(from_engine=lambda engine, handle=None: _NoBudget()),
    )


def _llm(**kw):
    base = dict(id=7, generation_hints={"context_length": _TRAINED})
    base.update(kw)
    return SimpleNamespace(**base)


async def _usage(state, *, llm=None, system_prompt="You are Erudi.", tools=None, pct=80):
    cp = InMemorySaver()
    if state:
        await _seed(cp, "x", state)
    return await AgentRunner(checkpointer=cp).acontext_usage(
        llm=llm or _llm(),
        thread_id="x",
        system_prompt=system_prompt,
        tools=tools or [],
        compaction_threshold=pct,
    )


def _used(categories) -> int:
    return sum(categories[k] for k in ("system_prompt", "tools", "messages", "summary"))


# ===================== the runner =====================


async def test_the_categories_are_the_projection_split():
    state = [_previous_summary_message()] + _alternating(9, 200)[1:]

    usage = await _usage(state, system_prompt="You are Erudi. " * 30)

    projection = project_next_request(
        state,
        _TRAINED,
        overhead=request_overhead("You are Erudi. " * 30),
        compaction_threshold=80,
    )
    categories = usage["categories"]
    assert _used(categories) == projection.conversation_tokens
    assert categories["summary"] == projection.summary_tokens > 0
    assert categories["messages"] == projection.message_tokens
    assert categories["system_prompt"] + categories["tools"] == projection.overhead_tokens
    assert categories["tools"] == 0


async def test_tool_schemas_are_their_own_category():
    from src.agents.tools import calculator, web_search

    state = _alternating(4, 100)
    usage = await _usage(state, tools=[calculator, web_search])

    overhead = request_overhead("You are Erudi.", tools=[calculator, web_search])
    projection = project_next_request(state, _TRAINED, overhead=overhead, compaction_threshold=80)
    system_only = request_overhead("You are Erudi.")
    expected_system = overhead_tokens(
        RequestOverhead(fixed_est=system_only.fixed_est, fixed_text=overhead.fixed_text),
        projection.ratio,
        dense=COUNTER_DENSE_TOKENS,
        weight_floor=COUNTER_WEIGHT_FLOOR,
    )
    assert usage["categories"]["system_prompt"] == expected_system
    assert usage["categories"]["tools"] == projection.overhead_tokens - expected_system
    assert usage["categories"]["tools"] > 0


async def test_under_the_threshold_the_blocks_fill_the_window():
    usage = await _usage(_alternating(4, 100))

    categories, window = usage["categories"], usage["window"]
    w = window["working"]
    t = trigger_tokens(w, 80)
    used = _used(categories)
    assert categories["free"] == t - used
    assert categories["autocompaction_buffer"] == w - t
    assert used + categories["free"] + categories["autocompaction_buffer"] == w
    assert usage["over_window"] is False


async def test_past_the_threshold_free_is_zero_and_the_reserve_shrinks():
    _load(window=4096)
    usage = await _usage(_alternating(8, 400))

    categories = usage["categories"]
    w = usage["window"]["working"]
    assert w == 4096
    used = _used(categories)
    assert trigger_tokens(w, 80) < used <= w
    assert categories["free"] == 0
    assert categories["autocompaction_buffer"] == w - used
    assert usage["over_window"] is False


async def test_past_the_window_every_block_is_used():
    _load(window=4096)
    usage = await _usage(_alternating(20, 400))

    categories = usage["categories"]
    assert _used(categories) > 4096
    assert categories["free"] == 0
    assert categories["autocompaction_buffer"] == 0
    assert usage["over_window"] is True


async def test_the_threshold_moves_the_trigger():
    low = await _usage(_alternating(4, 100), pct=50)
    high = await _usage(_alternating(4, 100), pct=90)
    assert low["categories"]["free"] < high["categories"]["free"]


async def test_not_loaded_the_trained_window_is_the_denominator():
    usage = await _usage(_alternating(4, 100))

    assert usage["window"] == {
        "working": _TRAINED,
        "allocated": None,
        "trained": _TRAINED,
        "reduced_by_memory": False,
        "model_loaded": False,
    }


async def test_another_loaded_model_is_not_this_conversations():
    _load(llm_id=99)
    usage = await _usage(_alternating(4, 100))
    assert usage["window"]["model_loaded"] is False
    assert usage["window"]["working"] == _TRAINED


async def test_loaded_the_allocated_window_is_the_denominator():
    _load()
    usage = await _usage(_alternating(4, 100))

    assert usage["window"] == {
        "working": 8192,
        "allocated": 8192,
        "trained": _TRAINED,
        "reduced_by_memory": False,
        "model_loaded": True,
    }


async def test_loaded_and_bounded_by_memory_says_so():
    _load(bound=8192)
    usage = await _usage(_alternating(4, 100))
    assert usage["window"]["reduced_by_memory"] is True


async def test_loaded_the_memory_ceiling_folds_the_working_window(monkeypatch):
    class _Ceiling:
        def tokens_at_ceiling(self):
            return 6000

    monkeypatch.setattr(
        runner_module,
        "MemoryBudget",
        SimpleNamespace(from_engine=lambda engine, handle=None: _Ceiling()),
    )
    _load()
    usage = await _usage(_alternating(4, 100))

    assert usage["window"]["working"] == 6000
    assert usage["window"]["allocated"] == 8192
    assert usage["window"]["reduced_by_memory"] is True


async def test_a_model_swapped_during_the_check_is_not_this_conversations(monkeypatch):
    _load()
    real_probe = _Engine.loaded_child_for.__func__

    def _swapping_probe(cls, llm_id):
        loaded = real_probe(cls, llm_id)
        _load(llm_id=99, window=2048)  # another model replaces it right after
        return loaded

    monkeypatch.setattr(_Engine, "loaded_child_for", classmethod(_swapping_probe))
    usage = await _usage(_alternating(4, 100))

    assert usage["window"]["model_loaded"] is False
    assert usage["window"]["allocated"] is None
    assert usage["window"]["working"] == _TRAINED


async def test_the_window_comes_from_one_snapshot_of_the_handle(monkeypatch):
    _load(window=8192, bound=None)
    seen: list = []

    def _swapping_budget(engine, handle=None):
        seen.append(handle)
        _load(llm_id=7, window=2048, bound=2048)  # a reload after the snapshot
        return _NoBudget()

    monkeypatch.setattr(
        runner_module, "MemoryBudget", SimpleNamespace(from_engine=_swapping_budget)
    )
    usage = await _usage(_alternating(4, 100))

    assert seen[0] == {"context_tokens": 8192, "memory_bound_tokens": None}
    assert usage["window"]["allocated"] == 8192
    assert usage["window"]["working"] == 8192
    assert usage["window"]["reduced_by_memory"] is False


# ----- the real engines' read path (no mocked probe nor budget) -----


class _AliveProc:
    """A child that is running, for ``poll()`` (llama.cpp) and ``is_alive()`` (MLX)."""

    def poll(self):
        return None

    def is_alive(self):
        return True


async def test_the_real_cpu_engine_read_path(monkeypatch):
    from src.engines.cpu_engine import CPU_Engine
    from src.engines.memory_budget import MemoryBudget

    class _Cpu(CPU_Engine):
        pass

    monkeypatch.setattr(runner_module, "MemoryBudget", MemoryBudget)
    monkeypatch.setattr(config, "LLM_Engine", _Cpu)
    handle = {"proc": _AliveProc(), "context_tokens": 12_288, "memory_bound_tokens": 12_288}
    monkeypatch.setattr(_Cpu, "_model", handle)
    monkeypatch.setattr(_Cpu, "_model_id", 7)

    usage = await _usage(_alternating(4, 100))

    assert usage["window"] == {
        "working": 12_288,
        "allocated": 12_288,
        "trained": _TRAINED,
        "reduced_by_memory": True,
        "model_loaded": True,
    }
    assert _Cpu._model is handle and set(handle) == {
        "proc",
        "context_tokens",
        "memory_bound_tokens",
    }


async def test_the_real_cpu_engine_with_a_dead_child_is_not_loaded(monkeypatch):
    from src.engines.cpu_engine import CPU_Engine
    from src.engines.memory_budget import MemoryBudget

    class _Cpu(CPU_Engine):
        pass

    class _DeadProc:
        def poll(self):
            return 1

    monkeypatch.setattr(runner_module, "MemoryBudget", MemoryBudget)
    monkeypatch.setattr(config, "LLM_Engine", _Cpu)
    monkeypatch.setattr(_Cpu, "_model", {"proc": _DeadProc(), "context_tokens": 12_288})
    monkeypatch.setattr(_Cpu, "_model_id", 7)

    usage = await _usage(_alternating(4, 100))

    assert usage["window"]["model_loaded"] is False
    assert usage["window"]["working"] == _TRAINED


async def test_the_real_mlx_engine_folds_its_memory_ceiling(monkeypatch):
    from src.engines.memory_budget import MLX_PREFILL_STEP_TOKENS, MemoryBudget
    from src.engines.mlx_engine import MLX_Engine

    gib = 1024**3

    class _Mlx(MLX_Engine):
        @classmethod
        def max_recommended_working_set_bytes(cls):
            return 12 * gib

    handle = {
        "proc": _AliveProc(),
        "context_tokens": 32_768,
        "base_footprint_bytes": 6 * gib,
        "memory_facts": {
            "weights_bytes": 5 * gib,
            "kv_token_bytes": 160 * 1024,
            "vocab_size": 151_936,
            "prefill_step": MLX_PREFILL_STEP_TOKENS,
            "working_set_bytes": 12 * gib,
            "model_type": "qwen3",
        },
    }
    monkeypatch.setattr(runner_module, "MemoryBudget", MemoryBudget)
    monkeypatch.setattr(config, "LLM_Engine", _Mlx)
    monkeypatch.setattr(_Mlx, "_model", handle)
    monkeypatch.setattr(_Mlx, "_model_id", 7)
    ceiling = MemoryBudget.from_handle(_Mlx, handle).tokens_at_ceiling()
    assert ceiling is not None and 1 < ceiling < 32_768

    usage = await _usage(_alternating(4, 100))

    assert usage["window"]["model_loaded"] is True
    assert usage["window"]["allocated"] == 32_768
    assert usage["window"]["working"] == ceiling
    assert usage["window"]["reduced_by_memory"] is True


async def test_an_unknown_window_reports_the_usage_alone():
    usage = await _usage(_alternating(4, 100), llm=_llm(generation_hints=None))

    assert usage["window"]["working"] is None
    assert usage["categories"]["free"] == 0
    assert usage["categories"]["autocompaction_buffer"] == 0
    assert usage["over_window"] is False


async def test_an_empty_conversation_counts_its_system_prompt():
    usage = await _usage([])
    assert usage["categories"]["system_prompt"] > 0
    assert usage["categories"]["summary"] == 0


async def test_it_never_waits_for_the_engine():
    async with _Engine.generation_guard():
        usage = await asyncio.wait_for(_usage(_alternating(4, 100)), timeout=1)
    assert usage["categories"]["messages"] > 0


# ===================== the service =====================


def _conversation(db, llm_id, **kw) -> Conversation:
    service = ConversationService(db, InMemorySaver())
    conversation = service.create_conversation(llm_id=llm_id, **kw)
    db.commit()
    return conversation


class _Spy:
    def __init__(self, monkeypatch):
        self.kwargs: dict = {}

        async def _capture(runner, **kwargs):
            self.kwargs = kwargs
            return {
                "categories": {},
                "window": {},
                "over_window": False,
            }

        monkeypatch.setattr(AgentRunner, "acontext_usage", _capture)


async def test_the_service_counts_the_conversations_own_prompt(
    test_db_session, mock_llm, monkeypatch
):
    conversation = _conversation(
        test_db_session, mock_llm.id, custom_prompt="Always answer in French."
    )
    from src.entities.Message import Message

    test_db_session.add(
        Message(
            conversation_id=conversation.id, sender="user", content="My cat is Tom.", starred=True
        )
    )
    test_db_session.commit()
    spy = _Spy(monkeypatch)

    result = await ConversationService(test_db_session, InMemorySaver()).context_usage(
        conversation.id
    )

    assert "Always answer in French." in spy.kwargs["system_prompt"]
    assert "My cat is Tom." in spy.kwargs["system_prompt"]
    assert spy.kwargs["compaction_threshold"] == conversation.compaction_threshold
    assert spy.kwargs["thread_id"] == str(conversation.id)
    assert result["kb_excerpts_excluded"] is False


async def test_a_systematic_assistant_counts_its_kb_prompt_without_excerpts(
    test_db_session, mock_llm_with_kb, monkeypatch
):
    llm, _kb = mock_llm_with_kb
    conversation = _conversation(test_db_session, llm.id)
    spy = _Spy(monkeypatch)

    def _no_retrieval(*a, **kw):
        raise AssertionError("/context never retrieves")

    monkeypatch.setattr("src.domains.conversations.services.retrieve_kb_excerpts", _no_retrieval)

    result = await ConversationService(test_db_session, InMemorySaver()).context_usage(
        conversation.id
    )

    assert result["kb_excerpts_excluded"] is True
    assert spy.kwargs["tools"] == []
    assert spy.kwargs["system_prompt"].startswith(
        build_kb_system_prompt(llm, custom_prompt="")[:40]
    )
    assert "excerpts from the user's documents" in spy.kwargs["system_prompt"]


async def test_an_agentic_assistant_counts_its_tools(
    test_db_session, mock_llm_with_kb, monkeypatch
):
    llm, _kb = mock_llm_with_kb
    llm.supports_tools = True
    llm.supports_tools_wire = True
    test_db_session.commit()
    monkeypatch.setattr(config, "KB_AGENTIC_MODE", None)
    conversation = _conversation(test_db_session, llm.id)
    spy = _Spy(monkeypatch)

    result = await ConversationService(test_db_session, InMemorySaver()).context_usage(
        conversation.id
    )

    assert result["kb_excerpts_excluded"] is False
    assert "search_knowledge_base" in [t.name for t in spy.kwargs["tools"]]


async def test_the_service_never_commits_nor_rebinds(test_db_session, mock_llm, monkeypatch):
    conversation = _conversation(test_db_session, mock_llm.id)
    test_db_session.query(Conversation).filter(Conversation.id == conversation.id).update(
        {"llm_id": None}
    )
    test_db_session.commit()
    commits: list = []
    monkeypatch.setattr(test_db_session, "commit", lambda: commits.append(1))
    _Spy(monkeypatch)

    with pytest.raises(StateConflictException) as raised:
        await ConversationService(test_db_session, InMemorySaver()).context_usage(conversation.id)

    assert raised.value.detail == {"reason": "no_model"}
    assert commits == []
    test_db_session.expire_all()
    assert test_db_session.get(Conversation, conversation.id).llm_id is None
    assert test_db_session.query(Llm).filter(Llm.local == 1).count() >= 1


async def test_a_normal_read_commits_nothing(test_db_session, mock_llm, monkeypatch):
    conversation = _conversation(test_db_session, mock_llm.id)
    commits: list = []
    monkeypatch.setattr(test_db_session, "commit", lambda: commits.append(1))
    _Spy(monkeypatch)

    await ConversationService(test_db_session, InMemorySaver()).context_usage(conversation.id)

    assert commits == []


# ===================== the endpoint =====================


def test_the_endpoint_answers_the_panel(client, mock_llm, test_db_session):
    test_db_session.query(Llm).filter(Llm.id == mock_llm.id).update(
        {"generation_hints": {"context_length": _TRAINED}}
    )
    test_db_session.commit()
    conversation_id = client.post("/erudi/conversations/", json={"llm_id": mock_llm.id}).json()[
        "id"
    ]
    asyncio.run(
        _seed(
            client.app.state.checkpointer,
            str(conversation_id),
            [HumanMessage("hi", id="h"), AIMessage("hello", id="a")],
        )
    )

    response = client.get(f"/erudi/conversations/{conversation_id}/context")

    assert response.status_code == 200
    body = response.json()
    assert set(body["categories"]) == {
        "system_prompt",
        "tools",
        "messages",
        "summary",
        "free",
        "autocompaction_buffer",
    }
    assert body["window"]["model_loaded"] is False
    assert body["window"]["working"] == _TRAINED
    assert body["kb_excerpts_excluded"] is False
    assert body["over_window"] is False
    assert (
        sum(body["categories"].values()) == _TRAINED
    ), "the blocks add up to the window while the request fits"


def test_the_endpoint_on_an_unknown_conversation_is_404(client):
    assert client.get("/erudi/conversations/999999/context").status_code == 404


def test_the_endpoint_says_to_pick_a_model_first(client, mock_llm, test_db_session):
    conversation_id = client.post("/erudi/conversations/", json={"llm_id": mock_llm.id}).json()[
        "id"
    ]
    test_db_session.query(Conversation).filter(Conversation.id == conversation_id).update(
        {"llm_id": None}
    )
    test_db_session.commit()

    response = client.get(f"/erudi/conversations/{conversation_id}/context")

    assert response.status_code == 409
    assert response.json()["error"]["detail"] == {"reason": "no_model"}
