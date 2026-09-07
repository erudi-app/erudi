---
title: Conversations
description: Creating conversations, the NDJSON streaming contract, and how conversation state is kept.
---

# Conversations

A conversation binds one installed model to a thread of messages and a set of generation parameters.
Answers are produced by a LangChain agent whose state lives in a LangGraph checkpointer; the
`messages` table keeps the full history for display.

All routes are mounted under `/erudi/conversations` on `127.0.0.1:27182`.

## Creating a conversation

```bash
curl -X POST http://127.0.0.1:27182/erudi/conversations/ \
  -H 'Content-Type: application/json' \
  -d '{"llm_id": 57}'
```

`ConversationCreate` has one required field, `llm_id`. Everything else is optional and, when left
out, is resolved rather than hard-coded:

| Field | Type | When omitted |
|-------|------|--------------|
| `llm_id` | int | required |
| `temperature` | float, 0.0–2.0 | resolved from the model's own defaults (`sampling_defaults`) |
| `top_p` | float, 0.0–1.0 | resolved from the model's own defaults |
| `max_tokens` | int, 1–32768 | resolved from the model's own defaults |
| `custom_prompt` | string, ≤ 4096 chars | empty |
| `web_search_enabled` | bool | copies the global default from `GET /erudi/user_settings/` |

There is no `name` field: the conversation is created as `"New Conversation"` and renamed later.
Once created, the conversation **owns** its values — a later change to the model's defaults or to the
global web-search setting never retro-affects it.

The response (`ConversationResponse`) carries `id`, `llm_id`, `name`, `created_at`,
`last_message_time`, `temperature`, `top_p`, `max_tokens`, `custom_prompt`, `web_search_enabled`.
`llm_id` is nullable: deleting the bound model nulls it server-side and the conversation survives,
unbound, until it is pointed at another model.

## Asking a question

`POST /erudi/conversations/{id}/query` streams the turn as **NDJSON** (`application/x-ndjson`): one
JSON object per line, no SSE framing.

```bash
curl -N -X POST http://127.0.0.1:27182/erudi/conversations/42/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the capital of France?"}'
```

```
{"t": "thinking", "text": "The user asks for a capital city."}
{"t": "answer", "text": "The capital of France"}
{"t": "answer", "text": " is Paris."}
{"t": "done"}
```

Request body (`ConversationQuery`):

| Field | Type | Purpose |
|-------|------|---------|
| `question` | string | required |
| `images` | list of strings | base64 data-URL images attached to the question (vision models) |
| `image_paths` | list of strings | local paths parallel to `images`, empty string when unavailable |
| `attachments` | list of strings | local paths of documents, or of folders of documents, attached to this question |
| `temperature`, `top_p`, `max_new_tokens`, `custom_prompt` | optional | per-turn overrides of the conversation's settings |

### Attached documents

`attachments` carries filesystem paths, not bytes: the backend runs on the same machine as the
files, so nothing is uploaded. Each path is read through the same `DocumentReader` the Knowledge
Base uses (`.pdf`, `.docx`, `.xlsx`, `.csv`, `.txt`, `.md`), and a path that names a directory is
walked - the folder itself and one level below it, hidden entries skipped - for those extensions.
Up to 50 files are read per question.

The extracted text is injected into the user turn ahead of the question, one delimited block per
file:

```
[Attached file: report.pdf]
Revenue grew steadily during the first quarter.
[End of attached file]

What does it say?
```

The total is capped at 24 000 characters per question (`ATTACHMENT_CHAR_BUDGET` in
`src/utils/attachment_utils.py`, roughly 6 000 tokens). A block that overflows the remaining budget
is cut and ends with `[truncated: file too large to include fully]`.

A file that cannot be read never fails the turn: its block says so in the model's view, and the
answer opens with an italic notice naming the file and why - unsupported type, missing file, no
extractable text (Erudi bundles no OCR tier, so images and scanned PDFs land here), or an extractor
failure. Truncated files are named in the same notice.

Both marker paths are percent-encoded when stored — `%` as `%25`, then `]` as `%5D` — so a path
holding a closing bracket cannot end its own marker; the renderer decodes them in
`frontend/src/utils/messageContent.js`.

What the conversation *stores* is only a `[file_path:/abs/path]` marker per attachment, next to the
`[image_path:...]` markers images use. The extracted text rides the live turn and is never persisted
as message content, so a reloaded conversation shows which documents a turn carried without dragging
their contents through history forever.

### Event types

| `t` | Payload | Meaning |
|-----|---------|---------|
| `answer` | `text` | A chunk of the answer, streamed |
| `thinking` | `text` | A chunk of the model's reasoning, extracted from `<think>` blocks |
| `tool_call` | `name`, `args` | The agent called a tool (`search_knowledge_base`, `web_search`, `calculator`) |
| `tool_result` | `name`, `text` | What that tool returned |
| `error` | `text` | The turn failed; the text is the curated error message |
| `done` | — | Terminal event, always sent, including after an `error` |

`done` is also the refetch signal: the assistant message is committed to the database *before* it is
emitted, so a client that refetches on `done` never races the insert.

Non-ASCII text is `\uXXXX`-escaped on the wire (`json.dumps` defaults) and decoded by `JSON.parse`.

### Parsing the stream

Lines can be split across network chunks, so buffer and only parse complete lines:

```js
const response = await fetch(`${API_BASE_URL}/conversations/${id}/query`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ question }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = "";

for (;;) {
  const { value, done } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });

  const lines = buffer.split("\n");
  buffer = lines.pop() ?? ""; // keep the trailing partial line

  for (const line of lines) {
    if (!line.trim()) continue;
    const event = JSON.parse(line);
    switch (event.t) {
      case "answer":
        appendAnswer(event.text);
        break;
      case "thinking":
        appendThinking(event.text);
        break;
      case "tool_call":
        showToolCall(event.name, event.args);
        break;
      case "tool_result":
        showToolResult(event.name, event.text);
        break;
      case "error":
        showError(event.text);
        break;
      case "done":
        refetchMessages();
        break;
    }
  }
}
```

The renderer's own implementation lives in `frontend/src/` and goes through
`frontend/src/services/api/client.js`, which adds the retry, timeout and `X-Request-ID` correlation
described in [Logging & Traceability](../logging.md).

### Persistence of a turn

The user message is persisted immediately, so it shows up before generation starts. When the stream
ends, the accumulated `answer` text becomes the assistant message's `content`, and the non-answer
events (`thinking`, `tool_call`, `tool_result`) are stored in order as that message's `trace`, so the
reasoning and tool panel can be replayed on reload. The serialized trace is capped at 32 KB,
drop-oldest, with a `{"t": "truncated"}` marker prepended when events were dropped. Error turns
persist no trace.

## Other endpoints

### Generate a title

```bash
curl -X POST http://127.0.0.1:27182/erudi/conversations/42/generate_title \
  -H 'Content-Type: application/json' \
  -d '{"question": "Explain quantum entanglement in simple terms"}'
```

Streams `text/plain` (not NDJSON) — the title, token by token. The result is sanitized before use:
code fences, backticks, wrapping quotes and list markers are stripped, only the first line is kept,
consecutive duplicate words are collapsed, and the length is capped. If nothing usable remains, the
caller keeps the default name.

### Read messages

```bash
curl http://127.0.0.1:27182/erudi/conversations/42/fetch_messages
```

Each `MessageResponse` carries `id`, `conversation_id`, `sender` (`user`, `assistant` or `llm`),
`content`, `timestamp`, `starred`, and the optional `trace` described above.

`GET /erudi/conversations/` lists conversations without their messages;
`GET /erudi/conversations/{id}` returns the conversation with its full message list.

### Update a conversation

```bash
curl -X PATCH http://127.0.0.1:27182/erudi/conversations/42 \
  -H 'Content-Type: application/json' \
  -d '{"name": "Physics questions", "temperature": 0.3}'
```

`ConversationUpdate` accepts `name`, `llm_id`, `temperature`, `top_p`, `max_tokens`, `custom_prompt`
and `web_search_enabled`; omitted fields are left unchanged. Switching `llm_id` is how an unbound
conversation (its model was deleted) is put back to work.

### Star and unstar a message

Both are POSTs at the collection level, taking the message id in the body:

```bash
curl -X POST http://127.0.0.1:27182/erudi/conversations/star_message \
  -H 'Content-Type: application/json' -d '{"message_id": 345}'

curl -X POST http://127.0.0.1:27182/erudi/conversations/unstar_message \
  -H 'Content-Type: application/json' -d '{"message_id": 345}'
```

Starred messages are read back when a turn is planned and appended to the system prompt under
"Important points from the conversation so far" (`build_system_prompt` in
`backend/src/utils/prompt_utils.py`).

### Record a failed generation

```bash
curl -X POST http://127.0.0.1:27182/erudi/conversations/42/store_error_message
```

Used by the frontend when generation fails outside the stream, so the history stays consistent.
Returns `{"message": ..., "error_message_id": <id>}`.

### Delete

```bash
curl -X DELETE http://127.0.0.1:27182/erudi/conversations/42
```

Permanent; messages cascade.

## How conversation state is kept

There is **no multi-tier memory**. Two mechanisms, and only two:

1. **The LangGraph checkpointer.** An `AsyncPostgresSaver` (`backend/src/agents/checkpoint.py`)
   persists the agent's message history in the same embedded PostgreSQL database as the business
   tables. It is opened once for the whole application lifetime by the FastAPI lifespan and reached
   by endpoints through the `get_checkpointer` dependency. The conversation id is the thread id, so a
   turn restores its own prior history without the caller replaying anything.

2. **Summarization middleware.** `SummarizationMiddleware`
   (`backend/src/agents/runner.py`, `_build_middleware`) runs on the **same local model**. It
   triggers at 20 messages and keeps the last 10, rewriting the checkpointer state: old turns are
   dropped and replaced by a summary, so the agent's context stays bounded. The `messages` table is
   untouched — the UI still shows the whole conversation.

Two more middlewares run alongside it: stale images and stale tool results are stripped from the
replayed state before the model is called.

If a turn fails mid-super-step and leaves a dangling user message in the checkpointer, the runner
appends an error assistant message so the thread keeps alternating roles — otherwise the next turn
would send two consecutive user messages and the chat template would reject it.

## The system prompt and the KB budget

The system prompt is picked from the model's parameter count by `get_prompting_strategy`
(`backend/src/utils/prompt_utils.py`), which returns a tier (`tiny`, `small`, `medium`, `large`,
`xlarge`) and, when the model has a Knowledge Base attached, the token budget for KB context:

| Parameters | Tier | KB token budget (e5 tokens) |
|------------|------|-----------------------------|
| unknown or ≤ 2 B | `tiny` | 400 |
| ≤ 4 B | `small` | 700 |
| < 8 B | `medium` | 1000 |
| ≤ 16 B | `large` | 1400 |
| > 16 B | `xlarge` | 2000 |

The budget is a ceiling, not a chunk count: retrieval decides per query how much of it to consume.
See [Knowledge Base](knowledge_base.md#token-budget-per-model-size) for the selection algorithm and
for how a KB turn routes between agentic and systematic mode.

## Web search

`web_search_enabled` is a per-conversation toggle, copied at creation from the global default in
`GET /erudi/user_settings/` (itself `false` until changed). The `web_search` tool is actually carried
on a turn only when the toggle is on **and** the model's tool calls are verified to parse on the
active engine's wire (`supports_tools` and `supports_tools_wire is True`) — the same gate as agentic
Knowledge Base turns. A systematic-KB turn stays zero-tool whatever the toggle says.

When the tool is gated in, it is exposed whether or not the machine is online: an offline turn gets
the tool's explicit failure text instead of a silently missing tool.

## Model residency

Only one model is resident at a time, and it is unloaded after about five minutes without a request:
`BaseEngine._max_idle_time` is 300 seconds and the cleanup monitor started by the FastAPI lifespan
ticks every 300 seconds. The next question reloads it, which costs the usual load time. See
[LLMs](llms.md#memory-and-the-idle-unload).

## Related pages

- [LLMs](llms.md) — installing and managing the models a conversation binds to.
- [Knowledge Base](knowledge_base.md) — chatting against your own documents.
- [Logging & Traceability](../logging.md) — following one request across the renderer and the backend.
- [API reference](../reference/conversations.md) — generated from the docstrings.
