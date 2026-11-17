"""A Knowledge Base assistant searches its documents before the web (#707).

With web search on, a small model (Qwen3 4B) called ``web_search`` on
questions its documents answer, then said the information "is not in the
documents" without having searched them. The prompt already said which tool
comes first; the lever is deterministic: on a turn that carries both tools,
a ``web_search`` call made before any ``search_knowledge_base`` call of the
turn first searches the Knowledge Base with the same query, and the model
gets both results in the one tool result that answers its call.

Two layers are pinned here:
- the guard itself (``_KbBeforeWebMiddleware``) with scripted retrieval and
  web search: when it acts, when it stays out of the way, what it returns;
- the whole turn through ``AgentRunner`` on the fixed question set
  (``tests/fixtures/kb/kb_before_web_questions.json``) with a scripted model
  that reproduces the #707 failure -- it calls ``web_search`` first on every
  question -- a scripted retrieval over ``nimbus_contract.txt`` and scripted
  web results. Deterministic, no network, no model.
"""

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
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
from src.agents.reasoning_effort import NO_REASONING_PLAN
from src.agents.runner import AgentRunner, GenParams
from src.agents.tools import (
    KB_NO_EXCERPTS_FACT,
    KB_SEARCH_FAILED_TEXT,
    KB_THEN_WEB_ARBITRATION,
    KB_THEN_WEB_DOCUMENTS_HEADER,
    KB_THEN_WEB_WEB_HEADER,
    WEB_SEARCH_ERROR_PREFIX,
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


def _invoked(caplog) -> list:
    return [
        r.message.split("(")[0].removeprefix("Tool invoked: ")
        for r in caplog.records
        if r.message.startswith("Tool invoked: ")
    ]


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


# ===================== The guard, tool call by tool call =====================


_EXCERPT = KbExcerpt(source_file="nimbus_contract.txt", text="KB-EXCERPT: préavis de 90 jours")
_WEB = {"title": "t", "href": "https://example.org/x", "body": "WEB-SNIPPET"}


class _Sources:
    """Scripted retrieval and web search, recording every query they get.
    ``excerpts`` is what retrieval returns, or the exception it raises."""

    def __init__(self, monkeypatch, excerpts=(_EXCERPT,), web=(_WEB,)):
        self.kb_queries: list = []
        self.web_queries: list = []
        self.excerpts, self.web = excerpts, web
        monkeypatch.setattr(tools_module, "retrieve_kb_excerpts", self._retrieve)
        monkeypatch.setattr(tools_module, "_run_ddgs_text", self._ddgs)

    def _retrieve(self, query, kb_id, token_budget):
        self.kb_queries.append(query)
        if isinstance(self.excerpts, BaseException):
            raise self.excerpts
        return list(self.excerpts)

    def _ddgs(self, query, max_results):
        self.web_queries.append(query)
        return list(self.web)


def _executor(calls: list):
    """The tool node's executor, for the calls the guard leaves alone:
    records each and answers with a marker text."""

    async def execute(request):
        call = request.tool_call
        calls.append(call["name"])
        return ToolMessage(
            content=f"UNTOUCHED {call['name']}", name=call["name"], tool_call_id=call["id"]
        )

    return execute


_CTX = TurnToolContext(kb_id=1, kb_token_budget=500, web_token_budget=1000)


def _request(name: str, args: dict, messages: list, call_id: str = "call-1"):
    tool = {"web_search": web_search, "search_knowledge_base": search_knowledge_base}.get(
        name, calculator
    )
    return ToolCallRequest(
        tool_call={"name": name, "args": args, "id": call_id, "type": "tool_call"},
        tool=tool,
        state={"messages": messages},
        runtime=SimpleNamespace(context=_CTX),
    )


def _ai_call(name: str, call_id: str, query: str = "q") -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": {"query": query}, "id": call_id}]
    )


def _parallel_web_calls(*queries) -> list:
    """A turn whose model called ``web_search`` once per query, in one message."""
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "web_search", "args": {"query": q}, "id": f"w-{i}"}
            for i, q in enumerate(queries)
        ],
    )
    return [HumanMessage("q"), message]


async def _guarded(question: str, query: str = None):
    query = query or question
    messages = [HumanMessage(question), _ai_call("web_search", "call-1", query)]
    calls: list = []
    result = await _KbBeforeWebMiddleware().awrap_tool_call(
        _request("web_search", {"query": query}, messages), _executor(calls)
    )
    return result, calls


class TestGuard:
    async def test_a_first_web_search_searches_the_documents_first(self, monkeypatch, caplog):
        sources = _Sources(monkeypatch)
        with caplog.at_level(logging.INFO, logger="erudi"):
            result, calls = await _guarded("Quel est le préavis ?", "préavis Nimbus")

        # The Knowledge Base, then the web, both with the model's query; the
        # guard runs them, the tool node runs nothing else.
        assert sources.kb_queries == ["préavis Nimbus"]
        assert sources.web_queries == ["préavis Nimbus"]
        assert calls == []
        assert _invoked(caplog) == ["search_knowledge_base", "web_search"]
        # One result answers the model's one call: same id, same tool name.
        assert isinstance(result, ToolMessage)
        assert result.tool_call_id == "call-1"
        assert result.name == "web_search"
        assert result.artifact == KB_FIRST_ARTIFACT
        # Both results reach the model, the documents first.
        assert result.text.index("KB-EXCERPT") < result.text.index("WEB-SNIPPET")
        assert "https://example.org/x" in result.text
        acted = [r for r in caplog.records if _GUARD_LOG in r.message]
        assert len(acted) == 1
        assert acted[0].levelno == logging.INFO

    async def test_the_language_line_comes_once_and_last(self, monkeypatch):
        _Sources(monkeypatch)
        result, _ = await _guarded("Quel est le préavis de résiliation du contrat Nimbus ?")
        text = result.text
        assert text.count("Réponds en français.") == 1
        assert text.endswith("Réponds en français.")
        # The arbitration is the last instruction before it.
        assert text.removesuffix("Réponds en français.").rstrip().endswith(KB_THEN_WEB_ARBITRATION)
        assert "Answer in English." not in text

    async def test_the_arbitration_is_the_only_instruction(self, monkeypatch):
        # The KB tool's reminder (answer from your own knowledge when the
        # excerpts are unrelated, say "not in the documents") and the web
        # tool's grounding line would compete with the arbitration.
        _Sources(monkeypatch)
        both = [q for q in _QUESTIONS if q["kind"] == "both"]
        for question in (q["question"] for q in both):
            result, _ = await _guarded(question)
            text = result.text
            assert "answer normally from your own knowledge" not in text
            assert "réponds normalement avec tes propres connaissances" not in text
            assert "say that the information is not in the documents" not in text
            assert "ne figure pas dans les documents" not in text
            assert "Ground your answer on these results" not in text
            assert text.count(KB_THEN_WEB_ARBITRATION) == 1
            assert "[Document: nimbus_contract.txt]" in text
            assert "[t - https://example.org/x]" in text

    async def test_two_identical_web_searches_share_one_document_search(self, monkeypatch, caplog):
        # Two web_search calls in one message run side by side; with the
        # same query, the documents are searched once and both results carry
        # the excerpts.
        sources = _Sources(monkeypatch)
        guard = _KbBeforeWebMiddleware()
        messages = _parallel_web_calls("q", "q")
        with caplog.at_level(logging.INFO, logger="erudi"):
            results = await asyncio.gather(
                guard.awrap_tool_call(
                    _request("web_search", {"query": "q"}, messages, "w-0"), _executor([])
                ),
                guard.awrap_tool_call(
                    _request("web_search", {"query": "q"}, messages, "w-1"), _executor([])
                ),
            )
        assert sources.kb_queries == ["q"]
        assert sources.web_queries == ["q", "q"]
        assert [r.tool_call_id for r in results] == ["w-0", "w-1"]
        assert all("KB-EXCERPT" in r.text and "WEB-SNIPPET" in r.text for r in results)
        assert _invoked(caplog).count("search_knowledge_base") == 1

    async def test_two_different_web_searches_each_search_the_documents(self, monkeypatch):
        sources = _Sources(monkeypatch)
        guard = _KbBeforeWebMiddleware()
        messages = _parallel_web_calls("a", "b")
        await asyncio.gather(
            guard.awrap_tool_call(
                _request("web_search", {"query": "a"}, messages, "w-0"), _executor([])
            ),
            guard.awrap_tool_call(
                _request("web_search", {"query": "b"}, messages, "w-1"), _executor([])
            ),
        )
        assert sorted(sources.kb_queries) == ["a", "b"]

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
    async def test_a_web_search_after_the_documents_runs_untouched(
        self, monkeypatch, messages, caplog
    ):
        sources = _Sources(monkeypatch)
        calls: list = []
        with caplog.at_level(logging.INFO, logger="erudi"):
            result = await _KbBeforeWebMiddleware().awrap_tool_call(
                _request("web_search", {"query": "q"}, messages), _executor(calls)
            )
        assert calls == ["web_search"]
        assert result.text == "UNTOUCHED web_search"
        assert sources.kb_queries == []
        assert not [r for r in caplog.records if _GUARD_LOG in r.message]

    async def test_a_search_on_an_earlier_turn_does_not_count(self, monkeypatch):
        sources = _Sources(monkeypatch)
        messages = [
            HumanMessage("first question"),
            _ai_call("search_knowledge_base", "kb-1"),
            ToolMessage(content="kb", name="search_knowledge_base", tool_call_id="kb-1"),
            AIMessage(content="answer"),
            HumanMessage("second question"),
            _ai_call("web_search", "call-1"),
        ]
        await _KbBeforeWebMiddleware().awrap_tool_call(
            _request("web_search", {"query": "q"}, messages), _executor([])
        )
        assert sources.kb_queries == ["q"]

    async def test_other_tools_run_untouched(self, monkeypatch):
        sources = _Sources(monkeypatch)
        calls: list = []
        messages = [HumanMessage("q"), _ai_call("calculator", "call-1")]
        result = await _KbBeforeWebMiddleware().awrap_tool_call(
            _request("calculator", {"expression": "2+2"}, messages), _executor(calls)
        )
        assert calls == ["calculator"]
        assert result.text == "UNTOUCHED calculator"
        assert sources.kb_queries == []

    async def test_a_web_call_without_a_query_runs_untouched(self, monkeypatch):
        # Nothing to search the documents with: the tool's own validation
        # answers the malformed call, as it does without the guard.
        sources = _Sources(monkeypatch)
        calls: list = []
        messages = [HumanMessage("q"), _ai_call("web_search", "call-1")]
        await _KbBeforeWebMiddleware().awrap_tool_call(
            _request("web_search", {}, messages), _executor(calls)
        )
        assert calls == ["web_search"]
        assert sources.kb_queries == []

    async def test_documents_with_nothing_relevant_do_not_order_a_refusal(self, monkeypatch):
        # Alone, an empty search tells the model to say "not in the
        # documents"; beside web results that order would contradict them.
        _Sources(monkeypatch, excerpts=())
        result, _ = await _guarded("q")
        assert KB_NO_EXCERPTS_FACT in result.text
        assert "Tell the user the information is not in their documents" not in result.text
        assert "WEB-SNIPPET" in result.text

    async def test_a_knowledge_base_without_chunks_searches_empty(self, monkeypatch):
        # The real retrieval over a Knowledge Base that holds no chunk: no
        # document uploaded, or only images waiting for a reader
        # (``pending_vision``, never chunked).
        from src.utils import kb_utils

        monkeypatch.setattr(kb_utils, "search_kb_chunks_scored", lambda query, kb_id: [])
        monkeypatch.setattr(tools_module, "retrieve_kb_excerpts", kb_utils.retrieve_kb_excerpts)
        monkeypatch.setattr(tools_module, "_run_ddgs_text", lambda q, n: [_WEB])
        result, _ = await _guarded("q")
        assert KB_NO_EXCERPTS_FACT in result.text
        assert "WEB-SNIPPET" in result.text

    async def test_a_failing_document_search_still_returns_the_web(self, monkeypatch, caplog):
        _Sources(monkeypatch, excerpts=RuntimeError("vector store down"))
        with caplog.at_level(logging.INFO, logger="erudi"):
            result, _ = await _guarded("q")
        assert KB_SEARCH_FAILED_TEXT in result.text
        assert "WEB-SNIPPET" in result.text
        assert result.tool_call_id == "call-1"
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(errors) == 1 and errors[0].exc_info

    async def test_a_failing_web_search_still_returns_the_documents(self, monkeypatch):
        def offline(query, max_results):
            raise ConnectionError("no route")

        _Sources(monkeypatch)
        monkeypatch.setattr(tools_module, "_run_ddgs_text", offline)
        result, _ = await _guarded("q")
        assert "KB-EXCERPT" in result.text
        assert f"{WEB_SEARCH_ERROR_PREFIX}no internet connection" in result.text


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


def _web_first(answer: str = "answer"):
    """The #707 failure: the model's first move on every question is
    ``web_search``; once it has a result it answers ``answer``."""

    def policy(messages) -> AIMessage:
        if messages[-1].type == "tool":
            return AIMessage(content=answer)
        query = _question_of(messages)
        return AIMessage(
            content="",
            tool_calls=[{"name": "web_search", "args": {"query": query}, "id": "w-1"}],
        )

    return policy


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


def _script_sources(monkeypatch, *, pad: int = 0):
    """Scripted retrieval over the fixture document (the sections holding the
    question's ``kb_fact``, else the first section: retrieval always returns
    its best pool) and scripted web results (the question's ``web_result``).
    ``pad`` characters lengthen the web snippet."""
    by_question = {q["question"]: q for q in _QUESTIONS}
    sections = [s.strip() for s in _DOCUMENT_TEXT.split("\n## ") if s.strip()]

    def retrieve(query, kb_id, token_budget):
        fact = by_question[query]["kb_fact"]
        hits = [s for s in sections if fact and fact in s] or sections[1:2]
        return [KbExcerpt(source_file=_DOCUMENT, text=s) for s in hits]

    def ddgs(query, max_results):
        result = by_question[query]["web_result"]
        if not result:
            return []
        return [{**result, "body": result["body"] + " " + "w" * pad}]

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
            monkeypatch, _web_first(), q["question"], _BOTH_TOOLS, f"t-{q['id']}"
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
        await _run_turn(monkeypatch, _web_first(), question, [web_search], "plain-web")
    assert _invoked(caplog) == ["web_search"]
    assert not [r for r in caplog.records if _GUARD_LOG in r.message]


async def test_an_empty_answer_shows_the_results_without_their_frame(monkeypatch):
    # #90: a model that closes its turn with no text after a tool gets the
    # tool's result as its answer. For a result the guard framed, the user
    # sees what the documents and the web returned, never the headers or the
    # instructions written for the model.
    _script_sources(monkeypatch)
    q = next(q for q in _QUESTIONS if q["id"] == "both-notice-law")
    _, events, _ = await _run_turn(monkeypatch, _web_first(""), q["question"], _BOTH_TOOLS, "e")
    answer = "".join(e["text"] for e in events if e["t"] == "answer")
    assert q["kb_fact"] in answer
    assert q["web_result"]["body"] in answer
    assert q["web_result"]["href"] in answer
    for frame in (
        KB_THEN_WEB_DOCUMENTS_HEADER,
        KB_THEN_WEB_WEB_HEADER,
        KB_THEN_WEB_ARBITRATION,
        "Réponds en français.",
        "Extraits de documents :",
    ):
        assert frame not in answer


async def test_a_compaction_between_the_guarded_result_and_the_answer(monkeypatch, caplog):
    # The guard's result is large enough to push the conversation over its
    # compaction threshold: the summary runs before the model reads it. The
    # turn completes, the guard acted once, and the history the next turn
    # starts from still answers every tool call it holds.
    from tests.test_compaction_keep import (
        _ScriptedSummaryModel,
        _WindowEngine,
        _alternating,
        _seed,
    )

    _WindowEngine.window = 10_000
    _WindowEngine.claims, _WindowEngine.rewrites = [], []
    monkeypatch.setattr(config, "LLM_Engine", _WindowEngine)
    _script_sources(monkeypatch, pad=16_000)
    q = next(q for q in _QUESTIONS if q["id"] == "both-sla-compare")
    model = _ScriptedModel(messages=iter([]), policy=_web_first())
    summary = _ScriptedSummaryModel(messages=iter([]))

    def build(llm, **kw):
        return summary if kw.get("effort_plan") is NO_REASONING_PLAN else model

    monkeypatch.setattr(runner_module, "build_chat_model", build)
    checkpointer = InMemorySaver()
    # 5,000 tokens of history: under the threshold before the search, over
    # it once the result (about 4,000 tokens) joins.
    await _seed(checkpointer, "c", _alternating(10, 500))
    runner = AgentRunner(checkpointer=checkpointer)
    with caplog.at_level(logging.INFO, logger="erudi"):
        events = [
            e
            async for e in runner.astream_text(
                llm=_Llm(),
                user_message=q["question"],
                system_prompt="sys",
                params=_PARAMS,
                thread_id="c",
                summarize=True,
                compaction_threshold=80,
                tools=_BOTH_TOOLS,
                context=TurnToolContext(kb_id=1, kb_token_budget=500, web_token_budget=8000),
            )
        ]
    kinds = [(e["t"], e.get("phase")) for e in events]
    # The summary ran after the guarded result, before the answer.
    assert kinds.index(("tool_result", None)) < kinds.index(("compaction", "done"))
    assert "".join(e["text"] for e in events if e["t"] == "answer") == "answer"
    assert len([r for r in caplog.records if _GUARD_LOG in r.message]) == 1
    assert _invoked(caplog) == ["search_knowledge_base", "web_search"]
    state = await checkpointer.aget({"configurable": {"thread_id": "c"}})
    _assert_paired(state["channel_values"]["messages"])
