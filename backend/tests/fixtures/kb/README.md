# KB test fixtures

Versioned corpora for Knowledge Base E2E and integration tests.

## `nimbus_contract.txt`

A small fictional service contract. Its facts exist **only** in this document
(not in any model's weights), so a grounded answer proves the KB was consulted.

Verifiable facts (use as assertion targets):

| Question intent | Expected fact |
|-----------------|---------------|
| Préavis de résiliation | **90 jours** (calendaires) |
| SLA garanti | **99,7 %** |
| Pénalité en cas de manquement | **5 %** |
| Délai support / tickets | **48 heures** (ouvrées) |
| Incidents critiques | **4 heures** |
| Jour de facturation | **premier jour ouvré** du mois |
| Suspension pour retard de paiement | au-delà de **30 jours** |

`.txt` (not `.md`) so it passes the uploader's `accept=".pdf,.txt"` filter while
remaining readable by the ingestion `text_extractor`.

## `kb_before_web_questions.json`

The fixed question set that judges whether a Knowledge Base assistant with web
search on looks in its documents before the web (#707). The Knowledge Base is
`nimbus_contract.txt` alone. Each question has a `kind`:

- `kb` (7): answered only by the document; `kb_fact` is the figure the answer
  must carry.
- `web` (3): answered only by the web (current rates, releases, weather).
- `both` (2): needs the document's figure and a web fact.

`web_result` is the scripted search result `tests/test_kb_before_web.py` returns
for the question: its body is placeholder text, not a real fact. That test runs
every question through the real agent loop with a scripted model that calls
`web_search` first (the #707 failure) and checks that the Knowledge Base is
searched first, with the same query, and that the model gets both results.

### Manual run on a real model (release QA)

Out of CI: it needs a real model, the embedding model and the network.

1. Install **Qwen3 4B** (`lmstudio-community/Qwen3-4B-MLX-4bit` on macOS,
   `Qwen/Qwen3-4B-GGUF` elsewhere) and create a Knowledge Base assistant on it
   whose only document is `nimbus_contract.txt`. Wait for the ingestion to
   finish.
2. Check the assistant runs agentic: the first turn logs
   `Turn mode: agentic KB (... web_search=on)` in `backend.log`.
3. **Web search on.** For each question, in order, open a new conversation on
   the assistant with the Web search switch on, ask the question, and note:
   the `Tool invoked:` lines of the turn in `backend.log` and their order,
   whether a `KB-first guard:` line was logged, and the answer.
   - Pass, every question: no `Tool invoked: web_search` before a
     `Tool invoked: search_knowledge_base` of the same turn.
   - Pass, `kb`: the answer carries `kb_fact` (the figure, in any wording) and
     never says the information is not in the documents.
   - Pass, `both`: the answer carries `kb_fact` and cites at least one URL.
   - Pass, `web`: the turn calls `web_search` and the answer cites a URL (the
     web part depends on the network; offline, the honest
     `Error during Web Search:` text is a pass).
   - A turn with no tool call at all that says the information is not in the
     documents is a failure to record against #375 (the guard acts only on a
     web search).
4. **Web search off.** Repeat the `kb` questions with the switch off: each is
   answered from the document, with `Tool invoked: search_knowledge_base` in
   the turn.
5. Record per question: kind, tools in order, guard line yes/no, pass/fail.
