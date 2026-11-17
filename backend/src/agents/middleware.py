"""Per-turn agent middleware (KB merge, image/tool-result hygiene, KB before web).

Factored out of ``runner.py`` so the runner module itself stays free of
module-level LangChain imports: this module subclasses ``AgentMiddleware`` at
class-definition time, so it is imported LAZILY (inside the runner's methods)
and the whole LangChain agent stack only loads on the first turn, not at boot
(issue #160).
"""

from __future__ import annotations

from typing import Annotated, Optional

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import AgentState, PrivateStateAttr
from langchain_core.messages import HumanMessage, ToolMessage
from typing_extensions import NotRequired

from src.core.logging import logger
from src.core.logutils import truncate_for_log

# One substitution helper and one marker table, shared with the real-token
# counters (``src.agents.token_accounting``); re-exported here for the
# middleware's own users.
from src.agents.token_accounting import (
    STALE_TOOL_RESULT_MARKERS,
    last_human_index,
    stale_result_copy,
)


class CompactionState(AgentState):
    """The agent state of a compacting turn: the messages, plus the
    compaction pause (``src.agents.runner.COMPACTION_PAUSE_KEY``), the id of
    the user message whose automatic compaction failed. Private to the
    compaction middleware, persisted with the thread by the checkpointer."""

    erudi_compaction_paused_at: NotRequired[Annotated[Optional[str], PrivateStateAttr]]


def _split_multimodal(content):
    """Split message content into (joined_text, image_parts).

    For plain-string content, returns (content, []). For OpenAI multimodal
    content (a list of ``{"type": "text"|"image_url", ...}`` parts), returns
    the joined text of the text parts and the list of image parts.
    """
    if isinstance(content, str):
        return content, []
    text = " ".join(
        p["text"]
        for p in content
        if isinstance(p, dict) and p.get("type") == "text" and p.get("text")
    )
    images = [p for p in content if isinstance(p, dict) and p.get("type") == "image_url"]
    return text, images


def _flatten_without_images(content) -> str:
    """Plain-text rendering of multimodal content; each image -> ``[image]``."""
    if isinstance(content, str):
        return content
    out = []
    for p in content:
        if isinstance(p, dict):
            if p.get("type") == "text" and p.get("text"):
                out.append(p["text"])
            elif p.get("type") == "image_url":
                out.append("[image]")
    return " ".join(out).strip()


class _KbContextMiddleware(AgentMiddleware):
    """Merge the per-turn KB block into the model request's LAST user message.

    Request-time only (``request.override``): the checkpointer keeps the
    clean question, so past turns never re-expose stale excerpts (no
    context pollution, no parroting fuel). Rationale: on small local
    models, grounding/language instructions dissolve with turn depth when
    they live in the system prompt (chat templates prepend it before the
    whole history) — the tail of the last user message is the one spot
    that always stays inside the effective window.

    Layout: excerpts+rules block, then the question, then the answer-
    language request LAST in the user's voice — pre-question language
    lines are ignored as block metadata (run-4 eval), in-question
    requests are honored (T5).
    """

    def __init__(self, context_block: str, language_line: str):
        super().__init__()
        self.context_block = context_block
        self.language_line = language_line

    def _merge(self, request):
        messages = list(request.messages)
        last = messages[-1]
        # No "Question:" label: any English structural string near the
        # question feeds the English attractor (run-5 eval finding).
        question_text, image_parts = _split_multimodal(last.content)
        merged_text = f"{self.context_block}\n\n{question_text}\n\n{self.language_line}"
        if image_parts:
            # Multimodal turn: merge the KB block into the text part and keep
            # the screenshot(s) attached for the VLM.
            merged = HumanMessage(content=[{"type": "text", "text": merged_text}, *image_parts])
        else:
            merged = HumanMessage(content=merged_text)
        return request.override(messages=[*messages[:-1], merged])

    def wrap_model_call(self, request, handler):
        return handler(self._merge(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._merge(request))


class _StripStaleImagesMiddleware(AgentMiddleware):
    """Carry only the MOST RECENT image forward in the model request.

    The checkpointer stores each turn's multimodal ``HumanMessage``, so without
    this every past screenshot would be re-sent on each follow-up and blow the
    (small, local) VLM context. Instead we keep the images of the most recent
    image-bearing turn and collapse every OLDER image to an ``[image]`` text
    marker. So a user can ask follow-ups about the image they just sent ("what
    colour is his hair?") without re-attaching it, while the context still carries
    at most one turn's images. When the current turn itself has an image, that is
    the most recent one, so it is the one kept (the previous single-turn case).
    """

    @staticmethod
    def _has_images(m):
        return isinstance(m.content, list) and any(
            isinstance(p, dict) and p.get("type") == "image_url" for p in m.content
        )

    def _strip(self, request):
        messages = list(request.messages)
        img_idxs = [i for i, m in enumerate(messages) if self._has_images(m)]
        if not img_idxs:
            return request
        keep = img_idxs[-1]  # most recent image-bearing turn: carry it forward
        changed = False
        for i in img_idxs:
            if i == keep:
                continue
            messages[i] = messages[i].model_copy(
                update={"content": _flatten_without_images(messages[i].content)}
            )
            changed = True
        return request.override(messages=messages) if changed else request

    def wrap_model_call(self, request, handler):
        return handler(self._strip(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._strip(request))


class _StripImagesForTextModel(AgentMiddleware):
    """Flatten ALL image content when the model is not known to see images (#133).

    A text-only model (no MLX ``vision_config`` / no llama.cpp ``mmproj``) would
    either crash or silently ignore image parts, so every ``image_url`` part —
    the current turn included — collapses to an ``[image]`` text marker before
    the request reaches the model server. The answer stays clean text instead of
    broken inference. The caller adds this whenever ``supports_vision is not
    True`` (#212): unknown capability (None) strips too — the services prepend a
    user-facing notice — and only a positively-detected vision model keeps its
    images.
    """

    def _strip(self, request):
        messages = list(request.messages)
        changed = False
        for i, m in enumerate(messages):
            if isinstance(m.content, list):
                messages[i] = m.model_copy(update={"content": _flatten_without_images(m.content)})
                changed = True
        return request.override(messages=messages) if changed else request

    def wrap_model_call(self, request, handler):
        return handler(self._strip(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._strip(request))


class _FoldSystemIntoUserMiddleware(AgentMiddleware):
    """Fold the system prompt into the first user turn for models with no system role.

    Some chat templates reject the ``system`` role outright (Gemma's raises
    ``System role not supported``), so passing our system prompt as a real
    ``SystemMessage`` 500s every turn. The caller adds this middleware ONLY when
    the model is positively detected as not system-role-capable
    (``model_supports_system_role`` is False) — system-role-capable models keep
    the intended behavior (a proper system message per their template).

    ``create_agent`` carries the prompt as ``request.system_message`` and the
    model node prepends it to the messages; we clear it and prepend its text to
    the first human message instead, so the template never sees a system role.
    Runs innermost (added last) so it folds the FINAL message list, after the KB
    merge has shaped the last user message.
    """

    _JOIN = "\n\n"

    def _fold(self, request):
        sys_msg = getattr(request, "system_message", None)
        if sys_msg is None:
            return request
        sys_text = (sys_msg.text or "").strip()
        if not sys_text:
            return request.override(system_message=None)

        messages = list(request.messages)
        first_human = next((i for i, m in enumerate(messages) if m.type == "human"), None)
        if first_human is None:
            # No user turn to attach to (rare): keep the instruction as a plain
            # user message so the model still receives it, just not as a system role.
            return request.override(
                system_message=None, messages=[HumanMessage(content=sys_text), *messages]
            )

        target = messages[first_human]
        text, image_parts = _split_multimodal(target.content)
        folded_text = f"{sys_text}{self._JOIN}{text}" if text else sys_text
        content = (
            [{"type": "text", "text": folded_text}, *image_parts] if image_parts else folded_text
        )
        # A copy, not a new message: the target's metadata and id (the
        # summary's ``lc_source``) survive the fold.
        messages[first_human] = target.model_copy(update={"content": content})
        return request.override(system_message=None, messages=messages)

    def wrap_model_call(self, request, handler):
        return handler(self._fold(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._fold(request))


def strip_stale_tool_results(messages, *, all_past: bool = False) -> list:
    """``messages`` as sent: past turns' KB/web results replaced by their markers.

    Pure (copies, never mutates). A result is past when it lies before the
    last user message -- the positional rule ``_StripStaleToolResults``
    applies to every request. ``all_past=True`` replaces every such result:
    the summarizer's pool holds only past turns, and the amber-warning
    projection judges the NEXT request, where the turn just answered is past.
    """
    messages = list(messages)
    if all_past:
        keep = len(messages)
    else:
        keep = last_human_index(messages)
        if keep is None:
            return messages
    for i, message in enumerate(messages[:keep]):
        messages[i] = stale_result_copy(message)
    return messages


def past_tool_result_ids(messages) -> frozenset:
    """Message ids of the tool results ``strip_stale_tool_results`` marks.

    Message ids, not tool-call ids: a parser with deterministic call ids
    (mlx_vlm's kimi_k2 parser emits ``functions.<name>:0``) reuses the same
    call id every turn, while message ids are unique once
    ``_ensure_message_ids`` ran. The compaction counter designates past
    results through this set, so it counts any list LangChain hands it --
    suffixes, the reversed pool of ``trim_messages``, partial copies -- as
    sent, without knowing where the list came from.
    """
    messages = list(messages)
    keep = last_human_index(messages)
    if keep is None:
        return frozenset()
    return frozenset(
        message.id
        for message in messages[:keep]
        if message.type == "tool"
        and message.id is not None
        and getattr(message, "name", None) in STALE_TOOL_RESULT_MARKERS
    )


class _StripStaleToolResults(AgentMiddleware):
    """Placeholder PAST turns' bulky tool results (KB search, web search).

    The checkpointer persists every ToolMessage, so without this each
    follow-up would re-send every past turn's (bulky) excerpts/snippets and
    re-introduce the multi-turn context pollution the request-time design of
    issue #81 had eliminated. The CURRENT turn's results stay intact (the
    model just fetched them and must read them); only past ones shrink to a
    short PER-TOOL directive marker telling the model to search again. We
    rewrite content only, never dropping the message, so the
    ``AIMessage(tool_calls) -> ToolMessage`` pairing the chat template requires
    stays valid. The checkpointer keeps the full result, so the UI is
    unaffected — symmetric to ``_StripStaleImagesMiddleware`` for images.
    Name-keyed (#310): each stripped tool carries its own marker
    (``STALE_TOOL_RESULT_MARKERS``). The substitution is the pure
    ``strip_stale_tool_results``, which the compaction counter, the
    summarizer's pool and the amber-warning projection share, so what they
    count is what this middleware sends.
    """

    _MARKERS = STALE_TOOL_RESULT_MARKERS

    def _strip(self, request):
        messages = list(request.messages)
        stripped = strip_stale_tool_results(messages)
        changed = any(new is not old for new, old in zip(stripped, messages))
        return request.override(messages=stripped) if changed else request

    def wrap_model_call(self, request, handler):
        return handler(self._strip(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._strip(request))


# The artifact of a web result the guard preceded with a Knowledge Base
# search: never sent to the model, it tells a later ``web_search`` of the
# same turn that the documents were already searched.
KB_FIRST_ARTIFACT = {"kb_searched_first": True}

_KB_TOOL = "search_knowledge_base"
_WEB_TOOL = "web_search"


def _kb_searched_this_turn(messages) -> bool:
    """Whether the current turn (what follows the last user message) holds a
    ``search_knowledge_base`` call, or a web result the guard preceded with
    one. The tool node runs after the model's message joined the state, so a
    call made beside ``web_search`` in that same message counts."""
    start = last_human_index(messages) or 0
    for message in messages[start:]:
        if message.type == "ai" and any(
            call.get("name") == _KB_TOOL for call in getattr(message, "tool_calls", None) or []
        ):
            return True
        if message.type == "tool" and getattr(message, "artifact", None) == KB_FIRST_ARTIFACT:
            return True
    return False


class _KbBeforeWebMiddleware(AgentMiddleware):
    """A Knowledge Base assistant searches its documents before the web (#707).

    On a turn that carries both ``search_knowledge_base`` and ``web_search``,
    a ``web_search`` call made before any Knowledge Base search of the turn
    runs ``search_knowledge_base`` first, with the same query, then the web
    search, and answers the model's call with ONE tool result holding both
    (``kb_then_web_result``: the documents first, the arbitration last). The
    model's one call gets its one result under its own id and name, so the
    ``AIMessage(tool_calls) -> ToolMessage`` pairing the chat template needs
    is untouched and the history holds no call the model did not make.

    Never a block: the web search the model asked for always runs, and once
    the documents were searched in the turn -- by the model, beside the web
    call or on an earlier hop, or by this guard -- every ``web_search`` runs
    as it is. A call with no query has nothing to search the documents with
    and runs as it is too.

    Why a guard and not words: the system prompt already says document
    questions go to ``search_knowledge_base`` (``_WEB_KB_ARBITRATION`` in
    ``src.agents.prompts``), and Qwen3 4B called ``web_search`` instead, then
    said the information was not in the documents it had not searched.
    """

    def __init__(self, kb_tool):
        super().__init__()
        self.kb_tool = kb_tool

    def _kb_request(self, request):
        """The Knowledge Base search to run first, or ``None`` when the call
        runs as it is."""
        call = request.tool_call
        if call.get("name") != _WEB_TOOL:
            return None
        query = (call.get("args") or {}).get("query")
        if not isinstance(query, str) or not query.strip():
            return None
        if _kb_searched_this_turn(list((request.state or {}).get("messages", []))):
            return None
        logger.info(
            "KB-first guard: web_search called before any knowledge base search "
            f"this turn, searching the knowledge base first (query={truncate_for_log(query, 200)})"
        )
        return request.override(
            tool_call={**call, "name": _KB_TOOL, "args": {"query": query}},
            tool=self.kb_tool,
        )

    @staticmethod
    def _combine(kb_result, web_result):
        """The web result, carrying the documents' result before its own."""
        from src.agents.tools import kb_then_web_result

        if not isinstance(kb_result, ToolMessage) or not isinstance(web_result, ToolMessage):
            # Not a tool's text (a ``Command``, which neither tool returns):
            # nothing to fold, the web call keeps its own result.
            return web_result
        return web_result.model_copy(
            update={
                "content": kb_then_web_result(kb_result.text, web_result.text),
                "artifact": KB_FIRST_ARTIFACT,
            }
        )

    def wrap_tool_call(self, request, handler):
        kb_request = self._kb_request(request)
        if kb_request is None:
            return handler(request)
        kb_result = handler(kb_request)
        return self._combine(kb_result, handler(request))

    async def awrap_tool_call(self, request, handler):
        kb_request = self._kb_request(request)
        if kb_request is None:
            return await handler(request)
        kb_result = await handler(kb_request)
        return self._combine(kb_result, await handler(request))
