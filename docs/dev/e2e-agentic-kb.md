# Test strategy — Agentic Knowledge Base (#84)

> **Historical — reviewed 2026-09-03.** This is the execution record of the #84
> test campaign, kept for the scenarios and the reasoning; the mechanics it
> describes have moved on. The agentic path is now gated on
> `supports_tools_wire` plus the tri-state `config.KB_AGENTIC_MODE`, decided in
> `src/agents/kb_mode.py`; the UI surfaces tool calls in
> `frontend/src/components/TraceStrip.jsx`; and the repository has real
> Playwright and vitest suites rather than the ad-hoc harness assumed below.
> Re-read the code before acting on any step here.

> **Status: DRAFT for agreement.** Validates the agentic KB feature (#84) on top of
> the PR3 RAG stack. Builds on the PR1 conversation/arena harness
> (`e2e-conversation-langchain.md`) — same pyramid, same E2E rule.
>
> **Models under test**
> - **Agentic path** — `Qwen2.5-0.5B (E2E)`, `Llm` id=105, `supports_tools=True`,
>   on disk at `backend/data/models/105`. Tool-capable → exercises the
>   `search_knowledge_base` path.
> - **Systematic path** — a **non**-tool-capable model (Gemma-270M was purged from
>   this machine). See [Prerequisites](#prerequisites-test-data): either re-download
>   it for a real systematic E2E, or cover that path at Layer 1 only.

## What #84 changed (recap, code-grounded)

Per-turn KB mode is **derived from the model**, never a user toggle
(`src/agents/kb_mode.py::plan_turn`):

| Mode | Condition | Behaviour |
|------|-----------|-----------|
| **AGENTIC** | KB attached **and** size tier allows context **and** `supports_tools` | KB exposed as the `search_knowledge_base` tool + `KbToolContext`; model decides when to consult. **No** up-front injection. |
| **SYSTEMATIC** | KB attached **and** tier allows **and not** `supports_tools` | Legacy path: excerpts retrieved up front, merged request-time via `_KbContextMiddleware`. |
| **PLAIN** | otherwise | No KB. |

- `should_use_kb(llm)` = `is_attached_to_kb` **and**
  `get_prompting_strategy(param_size)["use_kb_context"]`. The tier table treats every
  size `<= 2B` as `tiny` with `use_kb_context=True, kb_token_budget=400` — so **the
  0.5B qwen does qualify** (this was the main viability risk; it is cleared).
- `supports_tools` is detected from the chat template at post-download (#64) and
  persisted on `Llm` (#65); `NULL`/`False` → not tool-capable → systematic path.
- The KB tool returns a **grounded** `ToolMessage` (`tools.py::format_kb_tool_result`):
  attributed excerpts + answer-language line last; empty pool → an explicit
  "not in the documents" instruction.
- `runner.py::_StripStaleKbToolMessages` replaces **past-turn** `search_knowledge_base`
  `ToolMessage`s with a placeholder so the checkpointer history stays small while
  keeping the `AIMessage(tool_calls) → ToolMessage` pairing the chat template requires.

## The pyramid, applied (same rule as PR1)

- **Unit** — done with #64–#70 (engine/model faked).
- **Integration** (pytest, **no browser**): real components across a boundary — service
  + runner + real checkpointer + real `create_agent` graph + real `rag.kb_chunks`.
  Asserts on **checkpointer/DB state**. This is where *"the model actually called the
  tool"*, mode routing, grounding, and the placeholder rewrite are proven.
- **E2E** (Playwright, **browser, black-box**): the whole app through the real UI.
  Asserts **only on what the user observes** — never reaches into the DB. The proof of
  KB consultation is **the bot's own answer containing a fact that exists only in the
  uploaded document**.

> **Rule we will not break (from PR1):** an E2E test asserts observable behaviour; it
> does not open the DB to check internals.
>
> **Consequence specific to #84:** the UI has **no tool-use indicator** (no
> sources/citations/tool badge — confirmed in `frontend/src`). Therefore *"`search_
> knowledge_base` was invoked"* is an **integration** assertion. E2E proves **grounding**
> (the answer's content), not the tool mechanics.

## Prerequisites & test data

- **Agentic assistant.** Create a KB assistant over the qwen (id=105) and ingest the
  Nimbus corpus. This yields an `Llm` with `is_attached_to_kb=True`, a `kb_id`, and a
  `rag.kb_chunks` corpus.
- **Systematic model.** Needs a non-tool-capable model. Gemma-270M (~181 MB) was removed
  during the local purge. Options:
  - **(A)** Re-download Gemma-270M → a real systematic-path E2E (E-KB7).
  - **(B)** Keep the systematic path at **Layer 1 only**, with a fabricated
    `Llm(supports_tools=False)` driving `plan_turn` and the runner. No download.
- **Corpus** — `backend/data/e2e-kb-doc.md` ("Contrat de service Nimbus"). Facts that are
  **not** in any model's weights, used as grounding assertions:

  | Question intent | Expected fact in the answer |
  |-----------------|------------------------------|
  | Préavis de résiliation | **90 jours** (calendaires) |
  | SLA garanti | **99,7 %** |
  | Pénalité en cas de manquement | **5 %** |
  | Délai support / tickets | **48 heures** (ouvrées) |
  | Incidents critiques | **4 heures** |

## Layer 0 — Pre-flight probes (do FIRST)

Rule out a capability/plumbing bug before any scenario.

- **P0.1 — `supports_tools` is set.** Confirm qwen id=105 exposes `supports_tools=True`
  (API `/erudi/llms/...` + DB). If a Gemma is present, confirm it is `False`/`NULL`.
- **P0.2 — the model *actually* function-calls.** A detected flag is necessary, not
  sufficient. With the `calculator` tool only, ask the qwen `(290 - 89) * 12` → it must
  emit a real `tool_call` (and the agent loop must return `2412`). A 0.5B model emitting
  reliable `tool_calls` is the core risk of the agentic path; if it fails here, agentic
  KB cannot work and we fall back to systematic for this model.
- **P0.3 — mode routing is correct.** `plan_turn`:
  - qwen + KB attached → **AGENTIC** (`tools` include `search_knowledge_base`,
    `context` is a `KbToolContext`, `kb_context_block is None`);
  - non-tool model + KB attached → **SYSTEMATIC** (`kb_context_block` set, KB tool
    absent);
  - any model, no KB → **PLAIN**.

## Layer 1 — Integration tests (pytest, backend, no browser)

> Where *"the tool was called"* and all internals live. Asserts on checkpointer/DB.

- **IT-KB1 — `supports_tools` persisted (#64/#65).** Post-download detection writes the
  column; the llms API surfaces it.
- **IT-KB2 — Agentic routing (#70).** KB assistant + `supports_tools=True` → `plan_turn`
  returns `[calculator, search_knowledge_base]`, a `KbToolContext(kb_id, budget)`, and
  `kb_context_block is None`.
- **IT-KB3 — Systematic routing.** Same KB assistant but `supports_tools=False` →
  up-front `kb_context_block` set, KB tool **absent**, `retrieve()` invoked once.
- **IT-KB4 — Tool retrieval + grounding (#66/#67/#68).** Calling `search_knowledge_base`
  with a Nimbus query returns attributed excerpts; `format_kb_tool_result` includes the
  excerpt block + the answer-language line; an empty pool yields the explicit
  "not in the documents" string.
- **IT-KB5 — End-to-end tool call through the runner.** Run a KB turn with a tool-capable
  model (real qwen, or a fake that emits `tool_calls`) → the checkpointer thread contains
  `AIMessage(tool_calls=search_knowledge_base) → ToolMessage(grounded) → AIMessage(final)`,
  and the final answer reflects the excerpt.
- **IT-KB6 — Placeholder of stale KB ToolMessages (#69).** Two KB turns → the **past**
  turn's `search_knowledge_base` `ToolMessage` is replaced by the placeholder in the
  rewritten state, the `AIMessage(tool_calls) → ToolMessage` pairing is preserved, and
  the **current** turn's tool result is intact.
- **IT-KB7 — Arena shares the routing (#70).** An arena turn with the KB assistant routes
  through `plan_turn` identically and stays stateless (`AgentRunner(checkpointer=None)`).

## Layer 2 — E2E (Playwright, black-box, UI only)

> Proof = the app's visible behaviour + the bot's answers. Streaming = the bubble
> **growing across successive snapshots**. Setup identical to PR1 (see below).

- **E-KB1 — Create the KB assistant.** KnowledgeBasePage → pick qwen in `ModelLibrary` →
  name it (e.g. "Analyste Nimbus") + lock → drop `e2e-kb-doc.md` into `DragDropArea` →
  **Create Assistant** → confirm modal → spinner runs then clears → the assistant appears
  in the conversation/arena model pickers.
- **E-KB2 — Agentic grounding (the heart).** New conversation with the Nimbus assistant →
  *"Quel est le préavis de résiliation du contrat Nimbus ?"* → reply streams and contains
  **"90 jours"**. The fact exists only in the document → proves the KB was consulted.
- **E-KB3 — Second grounded fact.** *"Quel est le SLA garanti et la pénalité ?"* →
  contains **"99,7"** (and ideally **"5"**).
- **E-KB4 — Out-of-corpus honesty.** *"Quelle est la capitale de l'Australie ?"* (absent
  from the doc) → the assistant answers from general knowledge **or** says it is not in
  the documents, but does **not** invent a Nimbus clause. (Soft assertion — 0.5B model.)
- **E-KB5 — Multi-turn keeps a valid history (#69 observable).** Ask E-KB2 then E-KB3 in
  the **same** conversation → both answer correctly with **no** template/alternation error
  ("roles must alternate"). Observable proof the placeholder middleware preserves a valid
  `tool_calls ↔ ToolMessage` history across turns.
- **E-KB6 — Arena with the assistant.** Arena panel set to the Nimbus assistant → same
  résiliation question → grounded answer streams into the panel.
- **E-KB7 — Systematic path *(only if Gemma re-downloaded)*.** KB assistant on Gemma-270M
  → same Nimbus question → grounded answer via the **injected-context** path (no tool
  call). Otherwise this path is covered by IT-KB3/IT-KB4.

## What cannot be E2E'd, and where it is covered

| Claim | Covered at |
|-------|-----------|
| `search_knowledge_base` was actually invoked | Layer 1 (IT-KB5) + Layer 0 P0.2 |
| Past-turn ToolMessage replaced by placeholder | Layer 1 (IT-KB6); E2E sees only "no error" (E-KB5) |
| Systematic injection path (no Gemma on disk) | Layer 1 (IT-KB3/IT-KB4), unless E-KB7 |
| `supports_tools` detection internals | Layer 1 (IT-KB1) + Layer 0 (P0.1) |

## Open decisions (for agreement before execution)

1. **Systematic path:** re-download Gemma-270M for a real E2E (E-KB7), or keep it at
   Layer 1 only? (Recommended: Layer 1 only first; add E-KB7 if we want UI proof.)
2. **Playwright infra is absent** — no `playwright.config`, no `@playwright/*`, no
   `data-testid` (confirmed). Reuse PR1's `.playwright/cli.config.json` approach
   (Chromium with `bypassCSP` + disabled web-security / Local-Network-Access). Consider
   adding a few `data-testid` to the KB controls (`ModelLibrary` item, `DragDropArea`
   file input, "Create Assistant") to stabilize selectors before the run.
3. **Probe ordering:** P0.2 (does the 0.5B qwen reliably emit `tool_calls`?) is a
   go/no-go gate for the whole agentic E2E. Run it first.

## Setup (same as PR1)

- **Backend**: `cd backend && source venv/bin/activate && python run.py --port 27182`.
- **Renderer**: webpack dev server `http://localhost:3000` (HashRouter `/#/erudi/...`);
  Playwright drives it; streaming is `fetch` to `127.0.0.1:27182`.
- **Browser security**: launch Chromium via `.playwright/cli.config.json` with
  `contextOptions.bypassCSP=true` and `launchOptions.args` disabling web-security +
  Local-Network-Access checks (replicates Electron's session; test-harness only).
- **Models**: qwen id=105 visible in the picker; KB assistant created in E-KB1.

## Out of scope

- RAG retrieval **quality** tuning (tracked separately in #81 / `rag-quality-eval.md`).
- Packaged PyInstaller build.
- Answer-quality beyond grounding on the listed facts (0.5B model — it must ground on the
  document, not write polished prose).
