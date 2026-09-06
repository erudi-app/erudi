"""Agent tools — deterministic calculator (issue #81, problem #4).

LLMs predict tokens, they don't compute: small quantized models add
multi-digit numbers wrong with full confidence (measured: a different
wrong total on every eval run, correct operands displayed alongside).
The agent therefore carries a deterministic ``calculator`` tool; models
with native function calling (Qwen, Mistral, Llama 3.1+…) invoke it
through the standard agent loop. Models without it (e.g. Gemma 3 — no
tool format in its chat template, never emits ``tool_calls``) fall back
to the KB prompt's no-mental-math rule.

The evaluator is a strict AST whitelist — NEVER ``eval``: numbers and
arithmetic operators only (no names, calls, attributes, subscripts).
"""

from __future__ import annotations

import asyncio
import ast
import operator
from dataclasses import dataclass
from typing import List, Optional

from fastapi.concurrency import run_in_threadpool
from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from src.agents.prompts import answer_language_line, build_kb_context_block
from src.core.logging import logger
from src.core.logutils import truncate_for_log
from src.ingestion.chunking import count_tokens
from src.utils.kb_utils import KbExcerpt, retrieve_kb_excerpts

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Bounds against degenerate inputs (9**9**9 would hang/explode memory).
_MAX_EXPRESSION_LENGTH = 200
_MAX_POWER_EXPONENT = 1000
_MAX_ABS_VALUE = 1e15


def _evaluate_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POWER_EXPONENT:
            raise ValueError("exponent too large")
        try:
            result = _BINARY_OPERATORS[type(node.op)](left, right)
        except ZeroDivisionError:
            raise ValueError("division by zero")
        if abs(result) > _MAX_ABS_VALUE:
            raise ValueError("result too large")
        return result
    raise ValueError(f"unsupported syntax: {type(node).__name__}")


def evaluate_arithmetic(expression: str) -> str:
    """Evaluate a pure arithmetic expression deterministically.

    Returns the result as a string (integers without a trailing ``.0``).

    Raises:
        ValueError: empty input, non-arithmetic syntax, division by zero,
            or out-of-bounds magnitude.
    """
    expression = (expression or "").strip()
    if not expression:
        raise ValueError("empty expression")
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ValueError("expression too long")
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise ValueError("invalid arithmetic expression")
    result = _evaluate_node(parsed.body)
    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)


@tool
def calculator(expression: str) -> str:
    """Compute an arithmetic expression exactly. Use this for EVERY
    calculation instead of doing mental math: additions, subtractions,
    multiplications, divisions, percentages, powers.

    Args:
        expression: A pure arithmetic expression, e.g. "1240 + 1378 + 1456"
            or "(290 - 89) * 12". Numbers and + - * / // % ** ( ) only.
    """
    logger.info(f"Tool invoked: calculator(expression={truncate_for_log(expression, 200)})")
    try:
        return evaluate_arithmetic(expression)
    except ValueError as exc:
        # Text the model can react to — never crash the agent loop.
        return f"Error: {exc}. Provide a pure arithmetic expression."


# ===================== Shared per-turn tool context =====================

# Web search bounds (issue #310): a handful of attributed snippets is enough
# grounding for a small local model; more results just burn its context.
WEB_SEARCH_MAX_RESULTS = 5
# Per-request ddgs timeout (seconds) — the tool must never hang a turn.
WEB_SEARCH_TIMEOUT_S = 5
# Belt-and-braces overall guard around the threadpool call: ddgs may retry
# several engine backends, each with its own timeout.
WEB_SEARCH_OVERALL_TIMEOUT_S = 20
# Locked error contract (#310): every failure text starts with this exact
# prefix; the tail names the cause. The tool NEVER raises.
WEB_SEARCH_ERROR_PREFIX = "Error during Web Search: "
_WEB_NO_RESULTS_TEXT = "No results found for this query."
# Default web snippet budget (e5 tokens) — plan_turn overrides it with the
# model's size-tier budget; this default keeps a bare context bounded too.
WEB_SEARCH_DEFAULT_TOKEN_BUDGET = 1000


@dataclass
class TurnToolContext:
    """Per-turn context for ALL runtime-context tools, hidden from the model.

    ``create_agent`` accepts ONE ``context_schema`` (see ``runner.py``), so
    the KB and web tools share this single dataclass instead of one context
    type per tool: passed via ``create_agent(context_schema=TurnToolContext)``
    + ``astream(..., context=...)`` and read through ``ToolRuntime``; the
    model only ever sees each tool's ``query`` argument. KB fields stay at
    their defaults on turns without the KB tool (issue #310).
    """

    kb_id: Optional[int] = None
    kb_token_budget: int = 0
    web_max_results: int = WEB_SEARCH_MAX_RESULTS
    web_token_budget: int = WEB_SEARCH_DEFAULT_TOKEN_BUDGET


def format_kb_tool_result(excerpts: List[KbExcerpt], query: str) -> str:
    """Grounded ToolMessage payload: attributed excerpts + grounding reminder +
    the localized answer-language line, LAST (close to generation — the spot
    small local models honor, per the issue #81 findings). Reuses the systematic
    block builder so both KB paths ground identically. Empty pool -> an explicit
    "not in the documents" instruction rather than silence."""
    if not excerpts:
        return (
            "No relevant excerpts were found in the user's documents for this "
            "query. Tell the user the information is not in their documents."
        )
    block = build_kb_context_block(excerpts=excerpts, question=query)
    return f"{block}\n\n{answer_language_line(query)}"


@tool
async def search_knowledge_base(query: str, runtime: ToolRuntime[TurnToolContext]) -> str:
    """Search the user's personal knowledge base (their uploaded documents).

    Use this whenever the question concerns the content of those documents.
    Answer only from what this tool returns; if it returns no excerpts, tell the
    user the information is not in their documents.

    Args:
        query: A focused natural-language search query for the documents.
    """
    ctx = runtime.context
    logger.info(
        f"Tool invoked: search_knowledge_base(query={truncate_for_log(query, 200)}, "
        f"kb_id={ctx.kb_id}, token_budget={ctx.kb_token_budget})"
    )
    try:
        excerpts = await run_in_threadpool(
            retrieve_kb_excerpts, query, ctx.kb_id, ctx.kb_token_budget
        )
    except Exception:
        # A broken vector store must not crash the agent loop.
        logger.exception("search_knowledge_base tool failed (degrading gracefully)")
        return "The knowledge base could not be searched right now."
    logger.info(
        f"Tool search_knowledge_base returned {len(excerpts)} excerpt(s) " f"(kb_id={ctx.kb_id})"
    )
    return format_kb_tool_result(excerpts, query)


# ===================== Web search (issue #310) =====================


def _run_ddgs_text(query: str, max_results: int) -> list:
    """Blocking ddgs metasearch call — always run via ``run_in_threadpool``.

    Deferred import (#160 discipline): ddgs only loads on the first web
    search, never at boot. ``backend="auto"`` lets ddgs fail over across
    engines when one blocks/throttles.
    """
    from ddgs import DDGS

    return DDGS(timeout=WEB_SEARCH_TIMEOUT_S).text(query, max_results=max_results, backend="auto")


def _chain_has_network_error(exc: BaseException) -> bool:
    """True when the exception or anything in its cause/context chain is an
    ``OSError`` (covers ConnectionError, socket.gaierror, DNS failures…)."""
    seen: set = set()
    node: Optional[BaseException] = exc
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if isinstance(node, OSError):
            return True
        node = node.__cause__ or node.__context__
    return False


def map_web_search_error(exc: BaseException) -> str:
    """Deterministic error text for a failed web search (locked #310 contract).

    Exact prefix ``Error during Web Search: `` + a cause-specific tail:
      - ddgs TimeoutException / builtin TimeoutError -> timed out
      - ddgs RatelimitException -> rate-limited
      - OSError/ConnectionError anywhere in the chain -> no internet
      - anything else -> unexpected failure (<short cause>)

    ``TimeoutError`` is checked BEFORE the OSError chain walk: since 3.10 the
    builtin is an OSError subclass and would otherwise map to "no internet".
    """
    try:
        from ddgs.exceptions import RatelimitException, TimeoutException
    except ImportError:
        # The failure being mapped may BE the missing package: the mapper
        # must still return text, never raise into the agent loop.
        RatelimitException = TimeoutException = ()  # type: ignore[assignment]

    if isinstance(exc, (TimeoutException, TimeoutError)):
        tail = "the request timed out"
    elif isinstance(exc, RatelimitException):
        tail = "search engines are rate-limiting requests, try again later"
    elif _chain_has_network_error(exc):
        tail = "no internet connection"
    else:
        message = str(exc).strip()
        summary = message.splitlines()[0][:120] if message else type(exc).__name__
        tail = f"unexpected failure ({summary})"
    return f"{WEB_SEARCH_ERROR_PREFIX}{tail}"


def format_web_tool_result(results: list, query: str, token_budget: int) -> str:
    """ToolMessage payload for a web search: attributed snippets with SOURCE
    URLS (``title - href`` + snippet) so the model can cite, kept whole and
    best-first within ``token_budget`` e5 tokens (the first result always
    survives — mirroring the KB budget rule), a citation reminder, and the
    localized answer-language line LAST (the spot small local models honor,
    same as ``format_kb_tool_result``). Empty results are NOT an error."""
    if not results:
        return _WEB_NO_RESULTS_TEXT
    blocks: List[str] = []
    spent = 0
    for result in results:
        title = (result.get("title") or "").strip()
        href = (result.get("href") or "").strip()
        body = (result.get("body") or "").strip()
        block = f"[{title} - {href}]\n{body}"
        cost = count_tokens(block)
        if blocks and spent + cost > token_budget:
            break
        blocks.append(block)
        spent += cost
    joined = "\n\n".join(blocks)
    return (
        "Web search results:\n\n"
        f"{joined}\n\n"
        "Ground your answer on these results and cite the source URLs of the "
        "results you use.\n\n"
        f"{answer_language_line(query)}"
    )


@tool
async def web_search(query: str, runtime: ToolRuntime[TurnToolContext]) -> str:
    """Search the web for current or external information.

    Use this when the question needs recent events, live data, or specific
    external facts you do not reliably know. Results include source URLs to
    cite in your answer.

    Args:
        query: A focused natural-language web search query.
    """
    ctx = runtime.context
    logger.info(
        f"Tool invoked: web_search(query={truncate_for_log(query, 200)}, "
        f"max_results={ctx.web_max_results}, token_budget={ctx.web_token_budget})"
    )
    try:
        results = await asyncio.wait_for(
            run_in_threadpool(_run_ddgs_text, query, ctx.web_max_results),
            timeout=WEB_SEARCH_OVERALL_TIMEOUT_S,
        )
    except Exception as exc:
        # Locked contract (#310): NEVER raise — return deterministic text the
        # model can read and react to (offline, rate-limits, timeouts...).
        # The traceback is attached only when the cause is not one of the
        # expected network outcomes, whose text says everything.
        expected = isinstance(exc, TimeoutError) or _chain_has_network_error(exc)
        logger.warning(
            f"web_search tool failed: {type(exc).__name__}: {truncate_for_log(str(exc), 300)}",
            exc_info=not expected,
        )
        return map_web_search_error(exc)
    logger.info(f"Tool web_search returned {len(results)} result(s)")
    return format_web_tool_result(results, query, ctx.web_token_budget)
