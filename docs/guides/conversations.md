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
| `max_tokens` | int, 1–32768 | resolved from the model's own defaults — see [Output budget](#output-budget) |
| `custom_prompt` | string, ≤ 4096 chars | empty |
| `web_search_enabled` | bool | copies the global default from `GET /erudi/user_settings/` |
| `reasoning_effort` | `none` \| `low` \| `medium` \| `high` \| `xhigh` | copies the global default from `GET /erudi/user_settings/` |
| `compaction_threshold` | int, 50–90 in steps of 5 (percent) | copies the global default from `GET /erudi/user_settings/` — see [compaction](#how-conversation-state-is-kept) |

There is no `name` field: the conversation is created as `"New Conversation"` and renamed later.
Once created, the conversation **owns** its values — a later change to the model's defaults, to the
global web-search setting, to the global reasoning effort or to the global compaction threshold
never retro-affects it.

The response (`ConversationResponse`) carries `id`, `llm_id`, `name`, `created_at`,
`last_message_time`, `temperature`, `top_p`, `max_tokens`, `custom_prompt`, `web_search_enabled`,
`reasoning_effort`, `compaction_threshold`.
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
| `temperature`, `top_p`, `custom_prompt` | optional | per-turn overrides of the conversation's settings |
| `max_new_tokens` | optional | accepted and ignored — see [Output budget](#output-budget) |

### Output budget

How long an answer may get is not a setting. There is no Max Tokens control, and neither
`max_new_tokens` on a turn nor `max_tokens` on the conversation decides it in normal operation.

The reason is what the parameter actually does: it is a server-side cut the model never sees. It
cannot make an answer shorter or more focused — it can only end one mid-sentence. A number chosen
in advance is therefore either too small (a truncated answer) or irrelevant, and the model itself
stops when it is done.

So every model call gets its budget computed from the window it is running in:

```
max_tokens = max(512, window − prompt − margin)      margin = max(256, 10 % of the estimated part)
```

The window is the working window the loaded model runs with (see
[Context window](llms.md) and the working window below); the prompt is the whole request as it is
about to be sent — system prompt, knowledge-base block, history, and the tool schemas the call
carries — so a long conversation leaves a smaller budget than a fresh one, and each hop of a
tool-calling turn is budgeted again against its own longer history. The margin covers what the
estimate still cannot see, such as the chat template's own tokens; it applies to the estimated part
of the prompt only, never to the answers counted exactly (below).

There is no fixed ceiling above that: the window is the ceiling.

The prompt in that formula is counted in **real tokens** without tokenizing anything in the
backend, **message by message** (`backend/src/agents/token_accounting.py`): one ratio measured on
a request of one composition is wrong on a request of another (short English turns measure about
0.85, a long paste after them needs 1.0–1.2). Every call asks the server for its usage; the last
streamed chunk then carries the exact prompt size, and the client stamps that chunk with its own
`chars / 4` estimate of the same request, so `ratio = real prompt tokens / estimate` (clamped to
0.8–6; requests that carried images are skipped; the knowledge-base block a request carried is
taken out of both sides, since the next request no longer carries it). Then, for the request
about to be sent:

- the messages that were inside the last measured request — the latest request of the current
  turn, else the last turn's first request — cost `chars / 4 × ratio`;
- an answer the server generated (without reasoning) costs exactly its `output_tokens`; with
  reasoning, only the text the history replays is weighed;
- everything new — the question, a paste, a tool result of this turn, the knowledge-base block, a
  summary — costs `chars / 4` × the **script weight** of its own text: digits one token each when
  the loaded model's tokenizer splits numbers digit by digit (read once from the model's
  `tokenizer.json`, Qwen-style) and about a third of a token otherwise (Llama 3-style groups of
  three, and whenever the tokenizer is unknown or the engine is llama.cpp), Chinese, Japanese and
  Korean characters 0.65 token each, other non-Latin letters 0.4 token each, everything else
  `chars / 4`;
- the system prompt and the tool schemas cost the measured ratio, or their own script weight
  when nothing is measured yet (the first turn, the Arena).

The script weights are constants that depend on the tokenizer: on a model whose tokenizer splits
more finely (Japanese kana, a small vocabulary, Devanagari or Thai) the new text can be
under-counted until it has been sent once. This applies on every engine, llama.cpp included.

`llama-server` trims its own generation against what is actually left of the window.
`mlx_vlm.server` checks `prompt + budget ≤ window` against the real tokenised prompt before
generating and rejects the whole call otherwise. Where the ratio lags — the first turn in another
script, the first turn after a model swap — that rejection carries its own cure: it names the
exact prompt count. So the call is retried once, with a budget built from that number instead of
an estimate. Nothing is lost — the rejection arrives before any generation — and the turn then
answers normally. The retry checks the **allocated** window only — on Apple Silicon, the memory
bound the child enforces when memory binds. A prompt that fills the window on its own is not retried: no budget makes it fit, and it is
reported as a genuine overflow with the real numbers (see [When the prompt no longer fits the
context window](#when-the-prompt-no-longer-fits-the-context-window)). The compaction summary call
is never retried this way (see below).

Two values still matter at the edges:

- `Conversation.max_tokens` (and the model's `sampling_defaults.max_tokens`, clamped to
  `max_tokens_cap`) is the **fallback**: what a turn runs with when the engine cannot report the
  window it loaded with. The column is kept and resolved at creation; nothing writes it from the
  interface.
- `ERUDI_MAX_TOKENS` pins the budget for every call and wins over all of the above. It is a QA and
  development escape hatch, documented in `backend/.env.example`.

`max_new_tokens` stays on the request schema so an older client gets an answer instead of a 422.

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
| `thinking` | `text` | A chunk of the model's reasoning (see below) |
| `tool_call` | `name`, `args` | The agent called a tool (`search_knowledge_base`, `web_search`, `calculator`) |
| `tool_result` | `name`, `text` | What that tool returned |
| `compaction` | `phase`; with `phase: "done"`: `summarized`, `reason`, `summary`, `summary_lost` | Earlier messages are summarized (see [compaction](#what-the-thread-shows-of-a-compaction)). `phase: "start"`: the summary call is about to run; never persisted. `phase: "done"`: the history was rewritten — `summarized` = the user messages and answers summarized (tool calls, tool results and the previous summary not counted), `reason` = `"memory"` (the computer's memory bounds the window) or `"length"`, `summary` = the summary text, `summary_lost: true` when no summary could be written (`summary` is then the previous summary the model keeps, possibly empty). Persisted in the turn's trace. Several per turn are possible: one before each model call that needs it |
| `memory_reload` | `reason` (`"idle"` only) | Without `reason`, Apple Silicon and the CPU engine: the memory brake unloaded the model (or, on the CPU engine, restarted it with a smaller window) while the computer's memory was under pressure, and this turn reloads it or re-reads the conversation (see below). `reason: "idle"` (any engine): the model was unloaded after about five minutes without use, and this turn reloads it; the renderer says so instead of blaming memory. Only a conversation that already has messages gets it (a new one shows its first-reply hint), a model switch never sends it, and a turn announces at most one reload. Sent before anything else the model produces; never persisted |
| `memory_warning` | `reason`, `footprint_bytes` (`bound` only) | At the end of a turn, at most one. `reason: "pressure"`: the computer's memory is under pressure and the memory brake shrank the window, so the conversation is summarized earlier. `reason: "pressure_external"`: the computer's memory is under pressure from other apps and this conversation uses little of it. `reason: "bound"`: the machine's memory bounds the model's window and, even after compacting, the next request would fill 85 % of the working window (see below); `footprint_bytes` = the loaded model and the projected conversation together, the size the notice quotes |
| `error` | `text` | The turn failed; the text is the curated error message |
| `done` | — | Terminal event, always sent, including after an `error` |

`done` is also the refetch signal: the assistant message is committed to the database *before* it is
emitted, so a client that refetches on `done` never races the insert.

Non-ASCII text is `\uXXXX`-escaped on the wire (`json.dumps` defaults) and decoded by `JSON.parse`.

### Where `thinking` comes from

The inference server extracts each model family's chain-of-thought markers itself and streams the
reasoning in a dedicated field of its OpenAI chunks — llama-server (default `--reasoning-format
auto`) sends `delta.reasoning_content`, mlx_vlm.server sends `delta.reasoning`. Erudi's chat client
(`Erudi_Chat_OpenAI`, `backend/src/agents/chat_model.py`) carries that field through LangChain, and
the runner emits it as `thinking` events. The server-side parser also handles the cases inline tag
matching gets wrong: the thinking block a chat template reopens after every tool call, and a block
cut by the token budget before its closing tag.

For families whose markers the server parser does not know, the runner's `ThinkSplitter`
(`backend/src/agents/think_splitter.py`) stays as a fallback on the content channel: inline
`<think>...</think>` that slips through still becomes `thinking` events instead of leaking into the
answer.

### How much the model may think: reasoning effort

A conversation carries a reasoning effort — one of `none`, `low`, `medium`, `high`, `xhigh` — that
says how much the model may deliberate before answering. `medium` is the default: a conversation
copies the global default (`user_settings.default_reasoning_effort`) when it is created and owns its
level afterwards, so changing the global setting only affects the next conversation. Arena panels
have no conversation row and read the global level on every turn.

The chat header's settings panel carries a Reasoning effort control next to the Web search toggle,
seeded from the conversation's level and persisted through the same one-field PATCH the moment it
changes; the pre-conversation panel shows the same control seeded from the global default. The
Settings page's "Default reasoning effort" card sets that global default.

What a level costs on the wire depends on the model's own chat template, never on its family name: a
community fine-tune keeps its parent's name and ships whatever template its author baked in. The
verdict is read per artifact (`backend/src/engines/reasoning_lever.py`), from the
`chat_template_caps` llama-server reports for the running child and from a differential render of the
template otherwise, and it takes one of three values:

| the template… | how a level is delivered |
|---|---|
| reads `reasoning_effort` (gpt-oss and friends) | the level goes on the wire, natively |
| only honours `enable_thinking` (Qwen3 and friends) | `none` turns thinking off; `medium` is the natural behaviour; the other levels become a graded instruction |
| exposes no lever, but the model reasons | `medium` is the natural behaviour; every other level becomes a graded instruction |
| exposes no lever and the model does not reason | `none` changes nothing; every other level asks for a chain of thought written between `<think>` tags, which the splitter above separates from the answer |
| could not be read at all | nothing changes, at any level |

The last row is a deliberate asymmetry: a probe that *failed* is not a model
that *has no lever*. Filing one as the other would, at the default level, teach a
real reasoner whose own protocol we merely failed to read a second one, so an
unreadable template leaves the turn exactly as it was and says so in the log.

Two consequences are deliberate. At `medium`, a model that reasons naturally gets exactly the request
it got before the feature existed — nothing is added to the prompt and nothing to the body. And
`none` is best-effort on every model that reasons: the instruction asks it to answer directly, and a
model that reasons anyway is within its rights. That is why `none` on a template that grades its own
reasoning sends the native value *and* the instruction — llama-server maps `none` to
`enable_thinking=false` and then erases the `reasoning_effort` kwarg, so such a template hears
neither and falls back to its own default; the instruction is the only lever left there.

Induced reasoning has one accepted cost. When the server has no parser for the `<think>` block the
model was asked to write, the split happens in Erudi rather than server-side, so the tags and the
reasoning stay inside the assistant message the agent state keeps and are replayed as history on the
following turns — bounded, like everything else in the thread, by compaction.

Falling back from a level to an instruction is silent in the interface and recorded once per turn in
the backend log (`Reasoning effort: level=… lever=… wire_effort=… degraded_from=…`), which is what a
field report is read against.

Utility calls never reason, whatever the conversation's level: the conversation title and the
compaction summary both run at `none` — the summary through a second client on the same child server.

### When a turn ends without an answer

A thinking model can spend its whole turn reasoning and never write an answer. Instead of a generic
error (which would also discard the reasoning trace), the stream then ends with a curated `answer`
line picked by the model's `finish_reason`, persisted like any other answer and replayed to the
model on the next turn:

- generation was cut mid-reasoning (`length`):
  `Generation stopped during reasoning, before an answer was written. Send a follow-up asking for the final answer directly, or lower the reasoning effort.`
- the model closed its turn without an answer (`stop`):
  `The model finished its turn without writing an answer. Send a follow-up asking it to continue.`

On agentic turns a successful tool result takes precedence: it is delivered as the answer instead of
either line.

### When the prompt no longer fits the context window

Both local engines reject an over-budget prompt with a precise 400 that names the real numbers --
llama-server as `n_prompt_tokens`/`n_ctx` in its error body, mlx_vlm.server as a `MAX_KV_SIZE` detail
string. `src.agents.overflow.parse_context_overflow` discriminates that specific failure from any
other 400 (strictly on the llama body's `type`, never on message text) and the runner surfaces it as
an honest `error` turn instead of the generic one:

`This conversation no longer fits the model's context window (the request needs about N tokens; the
window holds W). Start a new conversation or send less at once.`

When the limit the engine refused on is the memory bound the Apple-Silicon child enforces
(`memory_bound_tokens()`, see [Engines](../dev/architecture/engines.md#the-memory-bound-mlx)), the
turn names memory instead:

`This conversation no longer fits what your machine's memory can hold for this model (the request
needs about N tokens; memory allows W). Start a new conversation, send less at once, or switch to a
smaller model.`

When the wire error matched but its numbers didn't parse, the turn drops the figures but keeps the
same honest wording. There is no silent truncation or shifting of the conversation to make it fit --
the user always gets a turn that says why and what to do next.

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
events (`thinking`, `tool_call`, `tool_result`, and `compaction` with `phase: "done"`) are stored in
order as that message's `trace`, so the reasoning and tool panel and the compaction markers can be
replayed on reload. The serialized trace is capped at 32 KB, drop-oldest, with a
`{"t": "truncated"}` marker prepended when events were dropped; compaction events are never dropped
(they lead the capped trace, ahead of the marker, which follows them only when a reasoning or tool
step was dropped). Error turns persist their compaction events only —
a compaction is checkpointed even when the model call after it fails — and no trace when they have
none. `compaction` with `phase: "start"`, `memory_warning` and `memory_reload` reach the wire but
are **never** persisted: the first describes the turn in flight, and the memory events describe the
machine's memory at that moment — an old conversation reopened on a bigger machine must not replay
them.

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

The app requests the title on the first message of a conversation, **before** the chat request:
the chat starts once the title stream has ended, whether it succeeded or failed (a failed title
never delays the answer). The llama.cpp engines keep one conversation's KV in a single slot, and a
title run after the first turn would overwrite it. The first message therefore waits for the title's own time (the prefill of the
question, plus the model load when the model is cold) before the chat request starts its attachment
and Knowledge Base preparation.

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

`ConversationUpdate` accepts `name`, `llm_id`, `temperature`, `top_p`, `max_tokens`, `custom_prompt`,
`web_search_enabled`, `reasoning_effort` and `compaction_threshold`; omitted fields are left
unchanged, and a `compaction_threshold` off the 50–90 scale (or not a multiple of 5) is a 422. Switching `llm_id` is how an unbound
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

2. **Summarization middleware (compaction).** `SummarizationMiddleware`
   (`backend/src/agents/runner.py`, `_build_middleware`) runs on the **same local model**. It fires
   when the request — the conversation plus the request overhead *O* (system prompt,
   knowledge-base block, tool schemas: what every request carries beyond the stored history) —
   reaches the conversation's **compaction threshold**, a share of the **working context window**
   `W` recomputed on every turn (the middleware is built after the inference child spawned, so the
   window is fresh for this model on this machine). Token pressure is the **only** trigger while a
   window is known: the number of messages says nothing about the window — every tool round adds
   two messages, so an agentic conversation holds many of them long before it fills the window.

   The threshold is a whole percent from **50 to 90 in steps of 5, 80 by default**. Each
   conversation owns its own (`compaction_threshold`, set from the chat header's settings panel or
   the new-conversation panel); the Settings page's "Default compaction threshold" card sets the
   default a new conversation copies (`default_compaction_threshold`). One helper turns it into the trigger every part of
   compaction reads (`trigger_tokens` in `backend/src/engines/working_window.py`):
   `T = max(1, min(⌊pct × W / 100⌋, W − 768))` — the second term always leaves the answer the output
   budget's 512-token floor and its 256-token margin, and at 80 % it only binds below 3836 tokens.
   A lower threshold summarizes sooner and more often and keeps the conversation smaller; a
   higher one keeps more of it word for word and summarizes less often. Lowering it in an open
   conversation compacts on the next turn when the conversation is already over the new share;
   raising it changes nothing until the conversation grows to it.

   The **working context window** is the single canonical window the memory-aware parts of a turn
   share (`backend/src/engines/working_window.py`, `working_context_tokens`): the **smaller** of

   - **the allocated context window** (`BaseEngine.effective_context_tokens()`, the window the
     loaded child actually runs with), and
   - **the memory ceiling** (**Apple Silicon only**) — the conversation token count at which the
     inference child's predicted footprint reaches the memory budget
     (`backend/src/engines/memory_budget.py`). The budget is 0.9 × the GPU's recommended working
     set (well below total RAM, and not net of the app's other processes). The prediction is a
     **measured prior**: the child's own footprint, measured right after it starts (`base`), plus
     `m × (t0 + step × vocab × 2 bytes + c0 × kv × N)` for a conversation of `N` tokens — `kv` the
     per-token KV-cache cost (`2 × layers × kv_heads × head_dim × 2 bytes f16`, read from the local
     artifact's `config.json`), `step × vocab × 2 bytes` the last prefill chunk's logits, and three
     named priors measured on an M4 16 GB (`c0 = 3.5` KV units per token — the cache, its
     prefix-cache copy and MLX's buffer cache as measured with its default limit; the child now bounds it at 0, which only lowers the peak —, `t0 = 1 GiB`, a margin `m = 1.10`; provisional
     until the release measurement campaign). The ceiling reserves the output budget's 512-token
     floor and stacks no further margin. Nothing is learned while the app runs: every turn's real
     peak is recorded next to its prediction in `memory_calibration.json` in the data folder, and a
     peak above the prediction writes one warning to `backend.log`, the memory the previous turn left in the child at the start of the turn being set aside (see
     [Privacy](../privacy.md)). Requests that carry images are outside the prediction (a vision
     tower's activations do not scale with the conversation). The ceiling exists only on MLX because only
     MLX grows its KV cache lazily with usage; on both llama.cpp engines (CPU and CUDA) the cache is
     allocated **in full at load**, for a window sized to fit at spawn (the CPU engine's own RAM
     budget, the CUDA engine's VRAM budget) — memory use does not grow with the conversation, so there
     is nothing per-token to measure (and on a
     discrete card, partial layer offload would make any single-pool accounting dishonest anyway).
     The accounting deliberately never reads the OS's "available" memory (macOS compression and swap
     make it non-deterministic); a model whose facts cannot be read simply runs without the memory
     ceiling. When the ceiling is smaller than the model's trained window, the child itself is
     bounded to it right after it loads, so the allocated window is that bound (see
     [Engines](../dev/architecture/engines.md#the-memory-bound-mlx)).

   over whichever of the two is known. When only the allocated window is known — the common case,
   and every llama.cpp engine — the working window is the allocated window; when the model's
   fixed part alone already fills the memory budget, compaction fires as early as it can. When
   neither window is readable (a llama.cpp `/props` read that failed, an MLX artifact without a
   context length, and no memory ceiling), there is no threshold to count tokens against, so a
   **message-count fallback** takes over: compaction fires at **20 messages** and keeps the last
   **10** — without it, the conversation would grow until every turn overflows. Compaction
   rewrites the checkpointer state: old turns are dropped and replaced by a summary, so the
   agent's context stays bounded. The `messages` table is untouched — the UI still shows the whole
   conversation.

   **What a compaction keeps** (`compaction_cutoff` in `runner.py`) starts from a **token budget**
   of `max(1, min(0.4 × W, T − summary_cap(W) − max(256, T / 5)) − O)` tokens, `W` being the
   working window and `T` the trigger: the longest recent part that fits it, however many messages
   that is. The margin under the trigger scales with it, so a low threshold still leaves room to
   grow: at 50 % a compaction leaves about 410 tokens under the trigger on a 4096-token window —
   the next compaction comes a turn or two later — and about 820 on an 8192-token window; at 80 %
   the keep is unchanged for a window of 2229 tokens or more. The bound is in
   tokens only — with token pressure the only trigger, the number of kept messages cannot re-fire
   anything, so a message cap on top would only discard recent context that fits the budget and
   bring the next compaction sooner. Three rules then adjust the cut:

   - the **current question** — the last user message — is never summarized away, even mid-turn
     when a large tool result follows it;
   - the kept part always **starts with an assistant message**. The summary is inserted as a user
     message, and two user messages in a row make strict chat templates (Gemma, Mistral) reject
     every later turn, so the state always reads `[summary, answer, question, …]`. A cut that lands
     on an older user message moves **forward** to the answer after it (keeping less, so the budget
     holds); only a cut that lands on the current question itself moves **back** to the answer
     before it;
   - an assistant message is never separated from its **tool results**. A cut that lands inside a
     tool round of an earlier turn moves **forward** past it, so the call and its results are
     summarized together and the budget holds; only a round that follows the current question
     (mid-turn) is kept whole with its call.

   When only the previous summary would be summarized, nothing is compacted. With a known window,
   a compaction that would gain less than **256 tokens**
   (`count(all) − count(kept) − summary_cap(W)`) is skipped — a small gain is not worth a summary
   call and a prefix-cache reset — unless the request would overflow without it (`O + count + 512`
   above the allocated window, or `O + count` above the working window); the message-count
   fallback has no such guard. What is guaranteed:
   with ordinary messages, the state left behind sits under `T − max(256, T / 5)` and the next
   model call does not compact again. The documented exceptions keep more than the budget: the answer before
   the current question when the cut falls on that question, a current message or a previous
   answer larger than the budget on its own (kept whole, summarized on a later turn, never
   truncated), and a tool round kept whole with its call because it follows the current question,
   or precedes it with no answer in between — the next call can then compact again.
   In the corner where the loaded model alone already exceeds the memory floor (`W = 1`), a model
   call compacts down to the current turn and the answer before it; once the state is
   `[summary, answer, current question, …]` only the summary would remain to summarize, so later
   calls of the same turn do not compact again. Without a known window the keep is the last 10
   messages, adjusted the same way.

   **Every count is in real tokens, as sent.** The trigger, the keep, the summarizer's trim and the
   warning projection cost each message as the output budget does (see
   [Output budget](#output-budget)) over the history as the model receives it: past turns'
   knowledge-base and web results count as the short markers that replace them in the request,
   not as the full results the checkpoint keeps. The weights are frozen once per model call from
   the whole state, per message, so a partial copy the summarizer trims costs its share. The
   counter counts new text a little higher than the budget — Chinese, Japanese and Korean
   characters one token each, and no script weight below 1.2 — because counting high only
   compacts a little earlier (a paste of tens of thousands of Chinese characters can trigger one
   compaction that was not needed). The usage total a model call reports is never used to
   trigger a compaction directly (a kept answer's report counts messages that are gone). Images
   are still counted at the estimator's 85 tokens each.

   **The summary itself is bounded**: the summary client is capped at
   `summary_cap(W) = clamp(W / 8, 128, 1024)` tokens, with no automatic output budget and no
   preflight retry (a smaller cap would silently truncate the summary; a rejection takes the size
   path below). It reads up to
   `max(256, min(max(4000, T − cap), W − 2 × cap − 256, (allocated − 2 × cap − 512) / 2))`
   tokens of the history being summarized — sized from the working window, and kept inside the
   allocated window with a factor 2 for the error left in the ratio (it is a whole-request average:
   an English system prompt around a Chinese history under-counts the history). Past
   knowledge-base and web results reach the summarizer as their markers, exactly as the model last
   saw them. The **previous summary is always part of what the summarizer reads**, prepended on top
   of that budget, so the facts it carries survive every compaction.

   With a large window, compaction is **rare and reads a large pool**. On a 40,960-token window at
   the default 80 % the first compaction fires near 32,768 tokens, keeps up to 16,384, and the summarizer reads up to
   19,200 tokens of history in one call, so that one turn waits for a long prefill before its
   answer starts — slowest on the CPU engine.

   **A failed summary never erases memory.** A transient failure of the summary call (the child
   unreachable or dead, a client timeout, a timeout while the summary was being written, HTTP
   408/409/429/5xx) fails the turn with the history intact: no summary is written and the next turn
   retries. A deterministic failure — a context overflow, another 4xx, a timeout while the model was
   still reading the input (deterministic for that input size), or an **empty summary** — and a
   history where no user turn fits the summarizer's budget because one answer alone exceeds it are
   retried once with oversized messages truncated (and the budget halved after a failed call); if
   that fails too, the compaction writes a placeholder that **carries the previous summary** —
   capped at `summary_cap(W)` real tokens, so a summary written under a larger window cannot keep the
   conversation above a smaller window's trigger — followed by "Later messages of this
   conversation could not be summarized." The conversation keeps working, and `backend.log`
   carries one warning per failed attempt.

   **Warn only when memory limits the conversation and compaction cannot save it**: the memory
   notice concerns only a model whose window the machine's memory bounds — on Apple Silicon, when
   the memory ceiling is smaller than the model's trained window, the inference child is lowered to
   that bound right after it loads (see [Engines](../dev/architecture/engines.md#the-memory-bound-mlx));
   on the CPU engine, when what 0.75 × the RAM affords is smaller than the model's trained window
   (never for a sliding-window model such as Gemma 3, whose window the budget over-estimates; see
   [Engines](../dev/architecture/engines.md#the-window-on-the-cpu-engine)) — and
   `memory_bound_tokens()` reports it. When the model's own window is the smaller one, memory is
   not the limiting factor and no memory notice is shown, however close to the budget the model
   runs in bytes. For a memory-bound model, the middleware compacts in `before_model`, so
   this turn's growth is compacted on the *next* turn. At the end of each turn the runner therefore
   projects the next request in real tokens — the turn just answered counted as past, an empty
   next question appended (so the projection keeps what the next request's compaction keeps), the
   system prompt and tool schemas on top (not the knowledge-base block of the turn just answered,
   which the next request does not carry): when a compaction would happen, `O` + the tail
   `compaction_cutoff` would keep + a `summary_cap(W)`-token summary; otherwise `O` + the whole
   conversation, with no summary added — and emits one `memory_warning` event only if, even after
   compacting to the kept budget, that projection reaches **85 % of the working window** `W` — for
   a memory-bound model, 85 % of the memory ceiling. Only a model that alone fills the budget (a
   memory ceiling at or below zero: the working window is clamped to 1 token) gets the notice from
   the first turn; any other memory-bound model gets it as soon as the projected next request
   reaches 85 % of the ceiling. At a 90 % threshold, a next request between 85 % and the trigger
   is not compacted yet, but the projection still counts the compaction the threshold will run:
   the notice speaks of what compaction cannot save, never of growth the threshold allows. The
   renderer then shows an amber notice above the
   composer with the approximate memory size of the model and the conversation together
   (`footprint_bytes`; on the CPU engine, what the child allocated at load for its window and
   its weights, since that does not grow with the conversation); the projection's numbers go to `backend.log` in one
   `Memory-bound conversation over 85 % of the working window ...` warning. The warning is about
   the machine *now*, so it is never persisted and clears on the next turn that carries none;
   `backend.log` records one `Memory warning started` / `Memory warning ended` line per
   conversation at each transition. A child whose memory bound could not be applied (the GPU
   working set readable only after the load, a request the child did not apply, an unreadable
   base footprint or model facts) stays at its spawn window, and no `bound` notice can show for it.

   **Under memory pressure, the memory brake acts first** (Apple Silicon and the CPU engine; see
   [Engines](../dev/architecture/engines.md#the-memory-brake-mlx) and
   [the CPU brake](../dev/architecture/engines.md#the-memory-brake-cpu)). It samples the
   computer's memory pressure at the start and the end of every turn and every minute while the
   model is idle, never during an answer. When it shrinks the working window, the next turn
   compacts earlier, and the end of the turn carries `memory_warning` with `reason: "pressure"`;
   when the pressure comes from other apps (this conversation uses little), `reason:
   "pressure_external"`. Either takes precedence over the `bound` projection above, which is then
   not evaluated. Only a conversation turn consumes the brake's notice; an Arena turn never does.
   When the brake unloaded the model while the user was away, the next turn first sends
   `memory_reload` (without `reason`; after the idle timeout's own unload it carries
   `reason: "idle"` and the copy names the pause instead of memory) and the renderer shows a reload state in the pending answer until the model
   speaks, an error arrives or the turn ends; the model reloads with the reduced window. On the CPU
   engine a smaller window means restarting the model with it (about 1-2 s, then the
   conversation is read again): under critical pressure at the next turn boundary — a restart at
   the start of a turn sends `memory_reload` in that same turn —, under warning once the user has
   been away two minutes (not when the idle unload is less than two minutes away: the next load
   then carries the smaller window), and the turn that re-reads the conversation shows the same
   reload state. The `pressure` notice is shown once the smaller window applies.

Two more middlewares run alongside it: stale images and stale tool results are stripped from the
replayed state before the model is called.

If a turn fails mid-super-step and leaves it open in the checkpointer — its question last, or
tool calls without their results — the runner appends a result for each unanswered tool call and
an error assistant message, so the thread keeps alternating roles: otherwise the next turn would
send two consecutive user messages (or unanswered calls) and the chat template would reject it.

A turn the client **interrupts** — Stop, a closed tab, the app's own client timeout — is closed the
same way. The disconnect cancels the model before its answer is committed to the checkpointer, so
the runner closes the turn itself, before it releases the model: a `[The tool call was
interrupted.]` result for every tool call left unanswered, then one assistant message — the
answer text already streamed to the client (the same text the conversation stores and shows), or,
when nothing was streamed yet, the line `[The answer was interrupted.]`. The decision reads the
**committed** checkpoint: an answer the model produced while the client was no longer reading is
only a pending write there, and the next question would discard it. Nothing is written when the
committed state already ends with an answer. The closing message carries no token usage (it is
counted from its text like any new message). The write is bounded (10 s); past that, one warning
in `backend.log` and the turn is released. The next question therefore never follows another
question, and strict templates keep working.

### What the thread shows of a compaction

The compaction middleware announces itself on the turn's stream: LangGraph's `custom` stream mode,
which the runner reads beside the `messages` one (`stream_mode=["messages", "custom"]`) and turns
into `compaction` events. The summarizer's own tokens never reach the answer: the runner reads
answer text from the `model` node only.

- **While it runs.** `phase: "start"` goes out right before the summary call. The pending answer
  shows a typing indicator and "Summarizing earlier messages to make room…" in the slot the reload
  state uses (a compaction start replaces a reload state), until the model's first answer, thinking
  or tool event, an error, or the end of the turn. The compaction's `done` drops the text but
  keeps the typing indicator: the model still reads the shortened conversation again before it
  answers, and the separator already says what was summarized.
- **Once it is done.** `phase: "done"` goes out after the history is rewritten (and the prefix
  cache reset). It renders as a thin separator above the turn's reasoning strip and answer: "N
  earlier messages were summarized to keep the conversation within the model's context", or "…to
  fit this computer's memory" when `reason` is `memory` — the engine reports the window as bound by
  memory (`memory_bound_tokens()`, the criterion of the amber notice), the memory ceiling made
  the working window smaller than the allocated one, or the allocated window is unknown and the
  working window comes from the memory ceiling alone. N counts the user messages and the answers
  summarized; with N at 0 the line names no number. "Show summary" reveals the summary as plain
  text, its inline `<think>` blocks removed when the model wrote any. A lost summary says "Earlier messages could not be summarized and were dropped from what
  the model sees." and offers the previous summary the model keeps, when there is one. A turn
  that compacts several times shows one separator per compaction, in order.
- **After a reload.** The `done` events are part of the persisted trace, error turns included, so
  the separators come back with the conversation; a trace holding only compactions shows no
  reasoning strip.

The announcements never cost a turn: a direct call of the middleware (no runtime) has no channel
and stays silent, and a channel that fails is one warning in `backend.log` per turn.

### The prefix cache holds one conversation (Apple Silicon)

On Apple Silicon the inference child keeps a **prefix cache**: a turn re-reads only the part of the
prompt the previous turn did not already process. That cache belongs to **one conversation at a
time**. Every turn claims it for its conversation (the Arena claims it for itself; titles claim
nothing), and it is **reset** when another conversation claims it and when compaction rewrites the
history — a cached prefix is then never another conversation's, and never a history that no longer
exists. `backend.log` records each reset as `Prefix cache reset: reason=<conversation_change|arena|compaction>`.

The cost is deliberate: **switching conversation re-reads the history** of the conversation you
switch to on its first turn (a longer wait before the first word), and alternating between two
conversations pays that on every switch. Staying in one conversation keeps the cache warm.

A reset never fails a turn: when the child cannot be reached it is deferred (one warning) and
retried on the next turn. When a turn was stopped mid-generation, the next reset first sends a
one-token request so the stopped request is fully settled before the cache is cleared. When the
client goes away mid-turn, the model call and any reset in progress are finished before the next
turn, title or idle unload starts. The llama.cpp
engines (CPU, CUDA) keep the same shape in their own way: one slot and no prompt cache, so the
current conversation's follow-up reuses its prefix and an older conversation is re-read once when
you return to it.

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
ticks every 30 seconds, so the unload lands about 300 to 330 seconds after the last request.
The next question reloads it, which costs the usual load time; the next turn of an existing
conversation says why it reloads (`memory_reload` with `reason: "idle"`), while a new conversation
shows its usual first-reply hint. See
[LLMs](llms.md#memory-and-the-idle-unload).

## Related pages

- [LLMs](llms.md) — installing and managing the models a conversation binds to.
- [Knowledge Base](knowledge_base.md) — chatting against your own documents.
- [Logging & Traceability](../logging.md) — following one request across the renderer and the backend.
- [API reference](../reference/conversations.md) — generated from the docstrings.
