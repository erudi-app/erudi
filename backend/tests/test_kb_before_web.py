"""A Knowledge Base assistant searches its documents before the web (#707).

With web search on, a small model (Qwen3 4B) called ``web_search`` on
questions its documents answer, then said the information "is not in the
documents" without having searched them. The prompt already said which tool
comes first; the lever is deterministic: on a turn that carries both tools,
a ``web_search`` call made before any ``search_knowledge_base`` call of the
turn first searches the Knowledge Base with the same query, and the model
gets both results in the one tool result that answers its call.

Two layers are pinned here:
- the guard itself (``_KbBeforeWebMiddleware``) against a scripted tool
  executor: when it acts, when it stays out of the way, what it returns;
- the whole turn through ``AgentRunner`` on the fixed question set
  (``tests/fixtures/kb/kb_before_web_questions.json``) with a scripted model
  that reproduces the #707 failure -- it calls ``web_search`` first on every
  question -- a scripted retrieval over ``nimbus_contract.txt`` and scripted
  web results. Deterministic, no network, no model.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from src.agents import runner as runner_module
from src.agents import tools as tools_module
from src.agents.middleware import KB_FIRST_ARTIFACT, _KbBeforeWebMiddleware
from src.agents.runner import AgentRunner, GenParams
from src.agents.tools import (
    KB_NO_EXCERPTS_TEXT,
    TurnToolContext,
    calculator,
    search_knowledge_base,
    web_search,
)
from src.core import config
from src.engines.base_engine import BaseEngine
from src.utils.kb_utils import KbExcerpt
from tests._helpers import ToolableFakeChatModel

pytestmark = pytest.mark.unit

_FIXTURES = Path(__file__).parent / "fixtures" / "kb"
_QUESTION_SET = json.loads((_FIXTURES / "kb_before_web_questions.json").read_text("utf-8"))
_QUESTIONS = _QUESTION_SET["questions"]
_DOCUMENT = _QUESTION_SET["document"]
_DOCUMENT_TEXT = (_FIXTURES / _DOCUMENT).read_text("utf-8")

_GUARD_LOG = "KB-first guard"


# ===================== The question set itself =====================


class TestQuestionSet:
    def test_it_holds_at_least_ten_questions_of_every_kind(self):
        kinds = [q["kind"] for q in _QUESTIONS]
        assert len(_QUESTIONS) >= 10
        assert {"kb", "web", "both"} <= set(kinds)

    def test_ids_are_unique(self):
        ids = [q["id"] for q in _QUESTIONS]
        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize("q", _QUESTIONS, ids=lambda q: q["id"])
    def test_each_kind_says_where_its_answer_lives(self, q):
        if q["kind"] in ("kb", "both"):
            assert q["kb_fact"] and q["kb_fact"] in _DOCUMENT_TEXT
        else:
            assert q["kb_fact"] is None
        if q["kind"] in ("web", "both"):
            assert q["web_result"]["href"].startswith("https://")
        else:
            assert q["web_result"] is None


# ===================== The guard against a scripted executor =====================


def _executor(results: dict, calls: list):
    """A stand-in for the tool node's executor: records every call it runs
    (name, args, id, tool name) and answers with the scripted text."""

    async def execute(request):
        call = request.tool_call
        calls.append((call["name"], dict(call["args"]), call["id"], request.tool.name))
        return ToolMessage(
            content=results[call["name"]], name=call["name"], tool_call_id=call["id"]
        )

    return execute


def _request(name: str, args: dict, messages: list, call_id: str = "call-1"):
    tool = {"web_search": web_search, "search_knowledge_base": search_knowledge_base}.get(
        name, calculator
    )
    return ToolCallRequest(
        tool_call={"name": name, "args": args, "id": call_id, "type": "tool_call"},
        tool=tool,
        state={"messages": messages},
        runtime=None,
    )


def _ai_call(name: str, call_id: str, query: str = "q") -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": {"query": query}, "id": call_id}]
    )


_RESULTS = {
    "search_knowledge_base": "[Document: nimbus_contract.txt]\nKB-EXCERPT",
    "web_search": "Web search results:\n\n[t - https://x]\nWEB-SNIPPET",
    "calculator": "4",
}


class TestGuard:
    async def test_a_first_web_search_searches_the_documents_first(self, caplog):
        calls: list = []
        messages = [
            HumanMessage("Quel est le préavis ?"),
            _ai_call("web_search", "call-1", "préavis Nimbus"),
        ]
        guard = _KbBeforeWebMiddleware(search_knowledge_base)

        with caplog.at_level(logging.INFO, logger="erudi"):
            result = await guard.awrap_tool_call(
                _request("web_search", {"query": "préavis Nimbus"}, messages),
                _executor(_RESULTS, calls),
            )

        # The Knowledge Base ran first, with the model's query, then the web.
        assert [c[0] for c in calls] == ["search_knowledge_base", "web_search"]
        assert calls[0][1] == {"query": "préavis Nimbus"}
        assert calls[0][3] == "search_knowledge_base"
        # One result answers the model's one call: same id, same tool name.
        assert isinstance(result, ToolMessage)
        assert result.tool_call_id == "call-1"
        assert result.name == "web_search"
        assert result.artifact == KB_FIRST_ARTIFACT
        # Both results reach the model, the documents first.
        text = result.text
        assert text.index("KB-EXCERPT") < text.index("WEB-SNIPPET")
        acted = [r for r in caplog.records if _GUARD_LOG in r.message]
        assert len(acted) == 1
        assert acted[0].levelno == logging.INFO

    async def test_the_sync_path_acts_the_same(self):
        calls: list = []
        messages = [HumanMessage("q"), _ai_call("web_search", "call-1")]

        def execute(request):
            call = request.tool_call
            calls.append(call["name"])
            return ToolMessage(
                content=_RESULTS[call["name"]], name=call["name"], tool_call_id=call["id"]
            )

        result = _KbBeforeWebMiddleware(search_knowledge_base).wrap_tool_call(
            _request("web_search", {"query": "q"}, messages), execute
        )
        assert calls == ["search_knowledge_base", "web_search"]
        assert result.text.index("KB-EXCERPT") < result.text.index("WEB-SNIPPET")

    @pytest.mark.parametrize(
        "messages",
        [
            # The model searched its documents on an earlier hop of this turn.
            [
                HumanMessage("q"),
                _ai_call("search_knowledge_base", "kb-1"),
                ToolMessage(content="kb", name="search_knowledge_base", tool_call_id="kb-1"),
                _ai_call("web_search", "call-1"),
            ],
            # The model called both tools in the same message.
            [
                HumanMessage("q"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "web_search", "args": {"query": "q"}, "id": "call-1"},
                        {"name": "search_knowledge_base", "args": {"query": "q"}, "id": "kb-1"},
                    ],
                ),
            ],
            # The guard already searched the documents on this turn.
            [
                HumanMessage("q"),
                _ai_call("web_search", "w-0"),
                ToolMessage(
                    content="both",
                    name="web_search",
                    tool_call_id="w-0",
                    artifact=KB_FIRST_ARTIFACT,
                ),
                _ai_call("web_search", "call-1"),
            ],
        ],
        ids=["kb-searched-earlier-hop", "kb-in-same-message", "guard-already-acted"],
    )
    async def test_a_web_search_after_the_documents_runs_untouched(self, messages, caplog):
        calls: list = []
        with caplog.at_level(logging.INFO, logger="erudi"):
            result = await _KbBeforeWebMiddleware(search_knowledge_base).awrap_tool_call(
                _request("web_search", {"query": "q"}, messages), _executor(_RESULTS, calls)
            )
        assert [c[0] for c in calls] == ["web_search"]
        assert result.text == _RESULTS["web_search"]
        assert not [r for r in caplog.records if _GUARD_LOG in r.message]

    async def test_a_search_on_an_earlier_turn_does_not_count(self):
        calls: list = []
        messages = [
            HumanMessage("first question"),
            _ai_call("search_knowledge_base", "kb-1"),
            ToolMessage(content="kb", name="search_knowledge_base", tool_call_id="kb-1"),
            AIMessage(content="answer"),
            HumanMessage("second question"),
            _ai_call("web_search", "call-1"),
        ]
        await _KbBeforeWebMiddleware(search_knowledge_base).awrap_tool_call(
            _request("web_search", {"query": "q"}, messages), _executor(_RESULTS, calls)
        )
        assert [c[0] for c in calls] == ["search_knowledge_base", "web_search"]

    async def test_other_tools_run_untouched(self):
        calls: list = []
        messages = [HumanMessage("q"), _ai_call("calculator", "call-1")]
        result = await _KbBeforeWebMiddleware(search_knowledge_base).awrap_tool_call(
            _request("calculator", {"expression": "2+2"}, messages), _executor(_RESULTS, calls)
        )
        assert [c[0] for c in calls] == ["calculator"]
        assert result.text == "4"

    async def test_a_web_call_without_a_query_runs_untouched(self):
        # Nothing to search the documents with: the tool's own validation
        # answers the malformed call, as it does without the guard.
        calls: list = []
        messages = [HumanMessage("q"), _ai_call("web_search", "call-1")]
        await _KbBeforeWebMiddleware(search_knowledge_base).awrap_tool_call(
            _request("web_search", {}, messages), _executor(_RESULTS, calls)
        )
        assert [c[0] for c in calls] == ["web_search"]

    async def test_documents_with_nothing_relevant_do_not_order_a_refusal(self):
        # Alone, an empty search tells the model to say "not in the
        # documents"; beside web results that order would contradict them.
        calls: list = []
        results = {**_RESULTS, "search_knowledge_base": KB_NO_EXCERPTS_TEXT}
        messages = [HumanMessage("q"), _ai_call("web_search", "call-1")]
        result = await _KbBeforeWebMiddleware(search_knowledge_base).awrap_tool_call(
            _request("web_search", {"query": "q"}, messages), _executor(results, calls)
        )
        assert "Tell the user the information is not in their documents" not in result.text
        assert "No relevant excerpts were found in the user's documents" in result.text
        assert "WEB-SNIPPET" in result.text


# ===================== The whole turn on the question set =====================


class _FakeEngine(BaseEngine):
    """Supplies generation_guard without touching real engine state."""


class _Llm:
    id = 7
    link = "/fake/path"
    name = "Qwen3 4B"
    param_size = 4.0


_PARAMS = GenParams(temperature=0.5, top_p=0.9, max_tokens=64)


@pytest.fixture(autouse=True)
def _engine(monkeypatch):
    monkeypatch.setattr(config, "LLM_Engine", _FakeEngine)
    yield
    _FakeEngine._last_used = None


class _ScriptedModel(ToolableFakeChatModel):
    """A model whose every reply is decided by ``policy(messages)`` and that
    records the message lists it receives."""

    policy: Any = None
    received: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.policy(messages))])


def _question_of(messages) -> str:
    human = [m for m in messages if m.type == "human"][-1]
    return human.text


def _web_first(messages) -> AIMessage:
    """The #707 failure: the model's first move on every question is
    ``web_search``; once it has a result it answers."""
    if messages[-1].type == "tool":
        return AIMessage(content="answer")
    return AIMessage(
        content="",
        tool_calls=[{"name": "web_search", "args": {"query": _question_of(messages)}, "id": "w-1"}],
    )


def _kb_then_web(messages) -> AIMessage:
    """A model that does it right: documents first, then the web."""
    tool_results = [m for m in messages if m.type == "tool"]
    query = _question_of(messages)
    if not tool_results:
        return AIMessage(
            content="",
            tool_calls=[{"name": "search_knowledge_base", "args": {"query": query}, "id": "k-1"}],
        )
    if len(tool_results) == 1:
        return AIMessage(
            content="",
            tool_calls=[{"name": "web_search", "args": {"query": query}, "id": "w-1"}],
        )
    return AIMessage(content="answer")


def _script_sources(monkeypatch):
    """Scripted retrieval over the fixture document (the sections holding the
    question's ``kb_fact``, else the first section: retrieval always returns
    its best pool) and scripted web results (the question's ``web_result``)."""
    by_question = {q["question"]: q for q in _QUESTIONS}
    sections = [s.strip() for s in _DOCUMENT_TEXT.split("\n## ") if s.strip()]

    def retrieve(query, kb_id, token_budget):
        fact = by_question[query]["kb_fact"]
        hits = [s for s in sections if fact and fact in s] or sections[1:2]
        return [KbExcerpt(source_file=_DOCUMENT, text=s) for s in hits]

    def ddgs(query, max_results):
        result = by_question[query]["web_result"]
        return [result] if result else []

    monkeypatch.setattr(tools_module, "retrieve_kb_excerpts", retrieve)
    monkeypatch.setattr(tools_module, "_run_ddgs_text", ddgs)


async def _run_turn(monkeypatch, policy, question, tools, thread_id):
    model = _ScriptedModel(messages=iter([]), policy=policy)
    monkeypatch.setattr(runner_module, "build_chat_model", lambda llm, **kw: model)
    checkpointer = InMemorySaver()
    runner = AgentRunner(checkpointer=checkpointer)
    events = [
        e
        async for e in runner.astream_text(
            llm=_Llm(),
            user_message=question,
            system_prompt="sys",
            params=_PARAMS,
            thread_id=thread_id,
            tools=tools,
            context=TurnToolContext(kb_id=1, kb_token_budget=500, web_token_budget=1000),
        )
    ]
    state = await checkpointer.aget({"configurable": {"thread_id": thread_id}})
    return model, events, state["channel_values"]["messages"]


def _invoked(caplog) -> list:
    return [
        r.message.split("(")[0].removeprefix("Tool invoked: ")
        for r in caplog.records
        if r.message.startswith("Tool invoked: ")
    ]


def _assert_paired(messages):
    """Every tool call the model made is answered by exactly one tool result."""
    call_ids = [c["id"] for m in messages if m.type == "ai" for c in m.tool_calls]
    result_ids = [m.tool_call_id for m in messages if m.type == "tool"]
    assert sorted(call_ids) == sorted(result_ids)


_BOTH_TOOLS = [calculator, search_knowledge_base, web_search]


@pytest.mark.parametrize("q", _QUESTIONS, ids=lambda q: q["id"])
async def test_web_first_model_gets_its_documents_first(monkeypatch, caplog, q):
    _script_sources(monkeypatch)
    with caplog.at_level(logging.INFO, logger="erudi"):
        model, events, messages = await _run_turn(
            monkeypatch, _web_first, q["question"], _BOTH_TOOLS, f"t-{q['id']}"
        )

    # The log a release QA reads: the documents were searched, before the web.
    assert _invoked(caplog) == ["search_knowledge_base", "web_search"]
    assert len([r for r in caplog.records if _GUARD_LOG in r.message]) == 1
    # The model's next request carries both results in the one tool result.
    tool_results = [m for m in model.received[-1] if m.type == "tool"]
    assert len(tool_results) == 1
    text = tool_results[0].text
    assert f"[Document: {_DOCUMENT}]" in text
    if q["kb_fact"]:
        assert q["kb_fact"] in text
    if q["web_result"]:
        assert q["web_result"]["body"] in text
        assert text.index(f"[Document: {_DOCUMENT}]") < text.index(q["web_result"]["body"])
    # The stream and the history stay consistent: one call, one result.
    assert [e["name"] for e in events if e["t"] == "tool_call"] == ["web_search"]
    assert [e["name"] for e in events if e["t"] == "tool_result"] == ["web_search"]
    _assert_paired(messages)


@pytest.mark.parametrize("q", _QUESTIONS, ids=lambda q: q["id"])
async def test_documents_first_model_is_left_alone(monkeypatch, caplog, q):
    _script_sources(monkeypatch)
    with caplog.at_level(logging.INFO, logger="erudi"):
        _, _, messages = await _run_turn(
            monkeypatch, _kb_then_web, q["question"], _BOTH_TOOLS, f"k-{q['id']}"
        )
    assert _invoked(caplog) == ["search_knowledge_base", "web_search"]
    assert not [r for r in caplog.records if _GUARD_LOG in r.message]
    _assert_paired(messages)


async def test_web_search_without_a_knowledge_base_is_untouched(monkeypatch, caplog):
    # A plain assistant with web search on carries web_search alone: there
    # is nothing to search first.
    _script_sources(monkeypatch)
    question = next(q for q in _QUESTIONS if q["kind"] == "web")["question"]
    with caplog.at_level(logging.INFO, logger="erudi"):
        await _run_turn(monkeypatch, _web_first, question, [web_search], "plain-web")
    assert _invoked(caplog) == ["web_search"]
    assert not [r for r in caplog.records if _GUARD_LOG in r.message]
