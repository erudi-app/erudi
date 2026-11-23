---
title: Knowledge Base
description: How Erudi turns a base model plus your documents into a Knowledge Base assistant, and how the ingestion and retrieval pipeline works.
---

# Knowledge Base

A **Knowledge Base (KB) assistant** is a base model you have installed, plus a set of your own
documents. Erudi creates a second model row that reuses the base model's weights (the `link` is
copied, not duplicated on disk), attaches a `KnowledgeBase` to it, and indexes the documents into a
vector store. Chatting with that assistant answers from your documents; the base model stays
available untouched.

Everything runs locally: extraction, chunking, embeddings and search all happen in the backend
process against the embedded PostgreSQL cluster. No document ever leaves the machine.

## Supported formats

| Extension | Extractor | Result |
|-----------|-----------|--------|
| `.pdf` | `PdfExtractor` (pypdf) | Text layer extracted page by page; a scanned PDF with no text layer is accepted as `pending_vision` |
| `.docx` | `DocxExtractor` (python-docx) | Paragraphs and tables to Markdown |
| `.xlsx` | `XlsxExtractor` (openpyxl) | One Markdown section per sheet |
| `.csv` | `CsvExtractor` | Markdown table |
| `.txt`, `.md` | `TextExtractor` | Read as-is |
| `.png`, `.jpg`, `.jpeg`, `.webp` | none | Accepted and recorded as `pending_vision` |

Any other extension is rejected with an `InvalidInputException` listing the supported set.

**A file that is not what its extension says is the user's, not a bug.** Each extractor turns the
parser's verdict on the file itself into `UnreadableDocumentException` (`src/core/exceptions.py`): a
text renamed `.docx` (`PackageNotFoundError`), an `.xlsx` that is not a zip (`BadZipFile`), a zip
renamed `.docx`/`.xlsx` with no Office parts in it (`KeyError` from the archive lookup), another
Office file renamed (a workbook or a `.docm` renamed `.docx`: python-docx's `ValueError` "is not a
Word file"; a Word document renamed `.xlsx`: openpyxl's `OSError` "File contains no valid workbook
part", both matched on that wording so no other `ValueError` or `OSError` is hidden), a PDF that
is empty, damaged or not a PDF (`PdfReadError` and its subclasses `PdfStreamError`,
`EmptyFileError`), a PDF locked by a password (`FileNotDecryptedError`). The Office exceptions are
caught around the opening call only; the PDF ones around the page extraction too, so a page pypdf
cannot parse marks the whole file `unreadable`.
The document is recorded `unreadable` and the job logs one `WARNING` without a traceback. Every
other exception (the extractor's own code, a missing optional package such as pypdf's
`DependencyError`, the tokenizer, the embedder) stays a `failed` file logged at `ERROR` with its
traceback.

**Images are not read.** Erudi bundles no OCR or vision tier today, so images and scanned PDFs are
registered in the KB with the status `pending_vision` and contribute zero searchable chunks. A batch
made only of such files fails the job rather than reporting a misleading success.

The routing table lives in `backend/src/ingestion/reader.py`; `DocumentReader.read(path)` is the
single extraction surface the service layer sees.

## The ingestion pipeline

Ingestion runs as a FastAPI background task, one file at a time
(`backend/src/domains/knowledge_base/services.py`). A file that is not a valid document of its
type marks that document `unreadable`; any other failure on one file marks it `failed`. The run
continues either way.

1. **Hash and dedup.** The file bytes are SHA-256 hashed. A hash already present in this KB
   (`knowledge_documents.content_hash_sha256`, unique per `kb_id`) is skipped as a duplicate
   when its row is `active`. A row an earlier job left unindexed (`failed`, `empty`,
   `pending_vision`, `unreadable`) never blocks its own file: it is deleted (its chunks follow,
   `ON DELETE CASCADE`) and the file goes through the pipeline again as a new document, so the job
   names it again. Two files with the same bytes in one job share the first one's row and outcome:
   the second is counted and named with it (from the job's in-memory ledger), never replacing it.
2. **Register the document.** A `KnowledgeDocument` row is created in the `indexing` status and
   committed before indexing — the vector store writes through its own connection and needs the row
   for its foreign key. It becomes `active` once its chunks are stored. A failure while reading the
   file or registering the row fails that file alone (rolled back, no row).
3. **Extract.** `DocumentReader` routes by extension and returns an `ExtractedDocument` carrying
   Markdown (and per-page text for PDFs).
4. **Clean, non-destructively.** Each extractor passes its text through
   `clean_extracted_text` (`src/ingestion/cleaning.py`): NFC normalization, control/format characters
   and NUL bytes removed, PDF end-of-line hyphenation rejoined, whitespace collapsed. Accents,
   currency signs and CJK are **preserved** — stripping them would desynchronize the indexed corpus
   from real user queries and break the sparse branch of hybrid search.
5. **Chunk in three passes** (`src/ingestion/chunking.py`), token-accurate against the real
   `intfloat/multilingual-e5-small` tokenizer:
   split on Markdown headers `h1`–`h4`; sub-split oversized sections with a token-aware recursive
   splitter (paragraph > line > sentence > word); prefix each chunk with its heading breadcrumb
   (`# A > ## B`) and re-attach Markdown table headers lost mid-split.
   Defaults: `DEFAULT_TARGET_TOKENS = 180`, `DEFAULT_OVERLAP_TOKENS = 27` (about 15 %).
   The tokenizer ships with the app, frozen in `src/ingestion/e5_tokenizer.json.gz`: the e5
   tokenizer as `transformers` builds it (`backend_tokenizer.to_str()`, gzipped), loaded with the
   `tokenizers` package alone. It needs no download and no `transformers`;
   `tests/test_e5_tokenizer.py` checks it gives `transformers`' ids and the same chunks.
   Loaded, it holds about 250 MB, so it is loaded on demand (about half a second) and released
   60 seconds after its last use (`TOKENIZER_IDLE_SECONDS`, by a reaper thread that exists only while
   it is loaded and that the backend shutdown stops); on release the backend asks the allocator to
   hand the freed memory back to the system (`malloc_zone_pressure_relief` on macOS,
   `malloc_trim` on Linux, no call on Windows). A call in progress holds it: it is never released
   under one, and chunking a text holds it for the whole text. Bounding an input at the embedding
   window (`truncate_to_tokens`, step 6) does not load it for a text that provably fits: every
   token covers at least one character of the normalized, pre-tokenized text, which the
   tokenizer's normalizer and pre-tokenizer alone (about a megabyte, kept loaded) count. A
   question under about 500 characters (fewer when the normalizer expands some of them) never
   loads it.
6. **Embed.** `get_embeddings()` (`src/ingestion/embeddings.py`) serves `ChildE5Embeddings` for
   `multilingual-e5-small` (384 dimensions, 512-token window) on every platform. The two children
   produce equivalent vectors (the fp32 snapshot on macOS, the Q8_0 GGUF elsewhere: worst cosine
   0.99981), so a Knowledge Base never needs re-indexing across platforms or updates:
   - `ChildE5Embeddings`: a dedicated embedding child
     (`src/ingestion/embedder_child.py`) serves `POST /v1/embeddings` on its own port slice
     (27400-27499) with a per-spawn API key. On Apple Silicon the child is a second
     `mlx_vlm.server` over the local e5 snapshot (`src/ingestion/mlx_embedder.py`); on Windows and
     Linux it is the bundled CPU `llama-server --embedding` over the e5 Q8_0 GGUF that ships
     inside the installers (`src/ingestion/llama_embedder.py`; the GGUF is produced at build time
     — see BUILD.md — and never downloaded at runtime; worst measured cosine vs the fp32 model
     0.99981, so existing Knowledge Bases keep their vectors). Nothing stays
     resident in the backend: the child spawns on first use (or ahead, when a conversation with a
     Knowledge Base opens; an ingestion job starts it on a background thread at its first file
     with text to embed, so the spawn runs while that file is read and chunked and the e5
     tokenizer loads, and a failed start is reported once, by the file that needed the child), an
     ingestion holds a lease so it is never stopped mid-job, and it is
     released after a fixed 5 minutes unused (no setting). Inputs are bounded at e5's 512-token
     window Erudi-side (`chunking.truncate_to_tokens`, prefix included) before the call —
     llama-server refuses longer inputs rather than truncating them. On a CUDA install the child
     runs the CPU binary with every GPU masked and is read by none of the CUDA gauges (see
     `docs/dev/architecture/engines.md`).

   The app ships no torch and no sentence-transformers. The test suite embeds through
   `tests/_e5_reference.py`: `ChildE5Embeddings` with its transport replaced by vectors recorded
   from the real child (`tests/fixtures/e5_reference_vectors.json`). A test that asserts e5
   semantics (rankings, golden queries) uses the `strict_e5_reference` fixture, which fails on a
   text without a recorded vector; record new ones on macOS with
   `pytest <test files> --record-e5-reference`.

   The asymmetric prefixes are mandatory: `passage: ` for indexed chunks,
   `query: ` for searches. Vectors are L2-normalized, so cosine similarity is a dot product. The
   embedded text is
   `passage: [document_name:<file>]\n<breadcrumb>\n\n<chunk>` — the document-name prefix boosts
   retrieval but is **not** stored: the stored content is the clean chunk, which is what goes into
   the prompt.
7. **Store.** `add_kb_chunks` writes the vectors plus the metadata columns
   (`kb_id`, `document_id`, `source_file`, `page`, `chunk_index`) into `rag.kb_chunks`, and the
   chunk's e5 token count as `token_count` in the `langchain_metadata` JSON column.

A document that extracts fine but yields zero chunks (blank file, parser that returned no text) is
marked `empty`, never `active`.

Each job ends with a `KB job N timings:` line in `backend.log` giving the time spent starting the
child, reading, chunking (the tokenizer's load apart), embedding and writing to the database (see
[A slow Knowledge Base job](../logging.md#a-slow-knowledge-base-job)).

### Document statuses

`indexing` (registered, chunks not stored yet), `active` (indexed and retrievable), `empty` (no indexable content), `failed` (ingestion raised),
`pending_vision` (image or scanned PDF, awaiting a future OCR/vision tier), `unreadable` (not a
valid document of its type: damaged, locked by a password, or another format). Defined in
`backend/src/entities/KnowledgeDocument.py`. A quit mid-file leaves its row `indexing`; the
startup cleanup turns every such row into `failed` (and interrupted jobs into `failed`, see below),
so the file can be added again.

### Job outcome

The KB job reports `completed` only when at least one file was indexed, or when every file was a
duplicate of content already in the KB. A batch that indexed nothing new and had no duplicates fails
with `NothingIndexedException` (`src/core/exceptions.py`), whose English message counts the failed,
empty, pending-vision and unreadable files; the status route reports it as `error_code:
"nothing_indexed"`. The task boundary logs it at `INFO` when the user's files are the whole cause
(every file `pending_vision`, `empty` or `unreadable`), at `ERROR` without a traceback when a
file `failed` (that file's own `ERROR` carries it).

A job that completes with some files not indexed still reports `completed`; the status route names
those files (see [Poll the job](#poll-the-job)) and the Knowledge Base page shows them. A file that
`failed` replaces the plain success message with one worded differently for a created assistant and
an updated one, which points to Diagnostics for the reason. A file the user's own file kept out is
named under its reason, translated from `common:fileReasons` (`unreadable`, `empty`,
`pending_vision`), with no pointer to Diagnostics. A job that fails because nothing could be indexed
shows a translated message built from its `error_code` (the English `error_message` goes to the
renderer's log only, at `warn`; any other failure is logged at `error`), followed by the same
names. At most five names are listed per reason, then "and N more".

## Retrieval

All chunks of all knowledge bases share one table, `rag.kb_chunks`, managed by
`langchain-postgres` `PGVectorStore` and filtered by the typed `kb_id` column at query time
(`src/ingestion/vector_store.py`).

Search is hybrid from the first query:

- **Dense**: HNSW index, cosine distance, over the e5 vectors.
- **Sparse**: a `tsvector` column built with `pg_catalog.simple` — language-neutral, no stemming, so
  identifiers and product codes survive intact; the dense branch compensates for the missing
  stemming.
- **Fusion**: Reciprocal Rank Fusion with `k = 60`. RRF is rank algebra, not a reranker model.

The retrieval entry point is `search_kb_chunks_scored(query, kb_id=...)`. It returns the candidate
pool in RRF order, each candidate carrying its dense cosine similarity re-read from the stored
vectors (the fusion overwrites per-row scores with rank harmonics, which have no semantic scale).

`retrieve_kb_excerpts` (`src/utils/kb_utils.py`) then selects what actually reaches the model:

1. Fetch a wide pool (`POOL_K = 20`).
2. **Adaptive cut**: keep candidates above the largest similarity drop-off, so a factoid question
   injects a narrow context while a panorama question keeps its whole cluster. Two guards keep the
   purely relative cut from starving recall: a widest gap sitting right after the top hit is honored
   only when it is a true outlier, and a recall floor (`K_MIN_EXCERPTS = 2`) re-extends the pool from
   candidates within `RECALL_BAND = 0.05` of the top similarity.
3. **Token budget**: keep whole chunks best-first until the model's budget is spent. Each chunk
   costs its stored `token_count`; a chunk without one is counted on the fly.

Chunks ingested before Erudi stored token counts have none: they are counted with the tokenizer when
they are retrieved, and keep no count afterwards.

Each excerpt keeps its `source_file` so the prompt can attribute it (`[Document: report.pdf]`).

### Token budget per model size

There is no fixed top-k. The ceiling is a token budget chosen from the model's parameter count by
`get_prompting_strategy` (`backend/src/utils/prompt_utils.py`), measured in e5 tokens (roughly 180
per chunk):

| Parameters | Prompt tier | KB token budget |
|------------|-------------|-----------------|
| `None` or ≤ 2 B | `tiny` | 400 |
| ≤ 4 B | `small` | 700 |
| < 8 B | `medium` | 1000 |
| ≤ 16 B | `large` | 1400 |
| > 16 B | `xlarge` | 2000 |

An unmeasured parameter count is treated as the smallest tier — the conservative choice.

## Agentic vs systematic mode

The KB mode is derived per turn from the model, never from a user toggle
(`backend/src/agents/kb_mode.py`, `plan_turn`):

- **Agentic** — the KB is exposed as the `search_knowledge_base` tool and the model decides when to
  consult it. Requires a KB attached, a size tier that allows KB context, and a model whose tool
  calls are *verified* to parse on the active engine's wire (`supports_tools` **and**
  `supports_tools_wire is True`).
- **Systematic** — excerpts are retrieved up front and merged into the request. This path carries
  **zero tools**: leak-prone models otherwise wrap the answer in raw tool-call JSON.
- **Plain** — no KB attached, or the size tier disables KB context, or retrieval returned nothing.

`ERUDI_KB_AGENTIC` overrides the routing and is tri-state (`src/core/config.py`): unset means
per-model routing (the default), `1`/`true` forces agentic for every KB turn (debugging), `0`/`false`
forces systematic (kill switch). The forced-agentic value never grants the web-search tool to a model
whose wire capability is unverified.

Every turn logs the decision, including which gate decided it (`Turn mode: agentic KB (... decided_by=...)`).

### With web search on: the documents first

An agentic turn with web search on carries both `search_knowledge_base` and `web_search`. The
instructions say document questions go to the documents, but small models do not always follow
them, so the order is enforced in the agent loop (`_KbBeforeWebMiddleware`,
`backend/src/agents/middleware.py`): when the model calls `web_search` and the turn has not
searched the Knowledge Base yet, the Knowledge Base is searched first with the same query, then the
web search runs, and the model's `web_search` call gets one result holding both: the document
excerpts, then the web results, then a single instruction — answer from the documents when they
contain the answer, use the web for the rest and cite its URLs — and the answer-language line last.
Neither tool's own instructions are repeated in it. A search that finds nothing or fails is said in
its place ("No relevant excerpts were found…", "The knowledge base could not be searched right
now.", `Error during Web Search: …`), and the other result is still delivered. `backend.log` shows
`KB-first guard: web_search called before any knowledge base search this turn, ...` followed by
`Tool invoked: search_knowledge_base` and `Tool invoked: web_search`.

The guard never blocks the web: once the turn has searched the documents — the model called
`search_knowledge_base` earlier in the turn or in the same message, or the guard already did —
every `web_search` runs as it is. When the model calls `web_search` several times in one message,
each call searches the documents first, and calls with the same query share one search. It acts on
web searches only: a turn where the model calls no tool at all is answered without any search. When
the model ends such a turn with no text, the answer shown is the excerpts and web results with
their sources, without the instructions written for the model.

The cost: a Knowledge Base search (local: one query embedding and one database search, no network)
before the first web search of every turn that has not searched the documents, and a
tool result of up to the Knowledge Base budget plus the web budget of the model's size tier (both
are the tier's `kb_token_budget`, see [Token budget per model size](#token-budget-per-model-size))
instead of the web budget alone. Like every past web result, it shrinks to the `web_search`
placeholder on later turns.

The fixed question set that measures this behaviour, and the manual procedure for a real model,
are in `backend/tests/fixtures/kb/README.md`.

## The embedding-model gate

On Apple Silicon the embedding child serves the `intfloat/multilingual-e5-small` snapshot, which
is **not** bundled: it is downloaded on demand into the app's own cache directory
(`config.CACHE_DIR`, i.e. `data/models_cache`, the Hugging Face cache layout), by
`snapshot_download` restricted to the files the child reads (`REQUIRED_FILES` in
`src/ingestion/embedding_model.py`: config, weights, `1_Pooling/config.json`, tokenizer files).
Chunking does not use that download: its tokenizer ships with the app (see step 5 of the ingestion
pipeline).

Presence is filesystem-driven, never a database flag: `embedding_model_available()` checks every
required file, so a partial download reads as unavailable and the gate reappears. The child loads
the snapshot by local path and never touches the network — an offline machine works.

```bash
curl http://127.0.0.1:27182/erudi/knowledge_base/embedding-model/status
# {"available": false, "downloading": false, "error": null}

curl -X POST http://127.0.0.1:27182/erudi/knowledge_base/embedding-model/download
# {"available": false, "downloading": true, "error": null}
```

`POST .../download` is idempotent: it is a no-op if the model is already present or a download is
already running. The download runs in a background thread, so navigating away from the KB page does
not lose it; the UI polls the status route.

The child serves the snapshot by LOCAL path, never by repo id (`embedding_snapshot_path()`).

On Windows and Linux the artifact the child needs is the bundled GGUF, not the snapshot: the
status route answers its on-disk presence (always there in a packaged install — no download is
ever offered or needed), and the download route is a no-op for it. A development tree without the
file produces it with `python scripts/dev/backend/build_e5_gguf.py`; until then the missing-model
state below applies. A snapshot left in `models_cache` there by an earlier version is dead weight
(~470 MB): once the GGUF is on disk, every start deletes it if present
(`embedder_child.remove_obsolete_e5_snapshot`, one INFO line with the size freed; a failed delete
is one WARNING and the next start retries). macOS never deletes it: its child serves it.

A Knowledge Base whose embedding artifact has left the disk (a cleared `models_cache` on macOS, a
dev tree without the GGUF elsewhere) is a **state**, not a turn error: a conversation or Arena
panel whose assistant carries a Knowledge Base shows a banner naming it
(`KbEmbeddingBanner`); a turn asked meanwhile degrades to answering without the documents, as any
Knowledge Base search failure does.

## API

All routes are mounted under `/erudi` on `127.0.0.1:27182`.

### Create or extend a Knowledge Base

`POST /erudi/knowledge_base/create` takes a **JSON body with local file paths** — it is not a
multipart upload. In the desktop app, the drop area resolves each selected or dropped file to its
absolute path through Electron's `webUtils.getPathForFile` before posting.

```bash
curl -X POST http://127.0.0.1:27182/erudi/knowledge_base/create \
  -H 'Content-Type: application/json' \
  -d '{
        "selectedModel": 42,
        "modelName": "Financial Reports",
        "description": "Quarterly earnings, 2025",
        "paths": [
          "/Users/me/Documents/q1.pdf",
          "/Users/me/Documents/q2.docx"
        ]
      }'
```

```json
{ "msg": "Knowledge Base Assistant is being created.", "model_id": 108 }
```

Body fields (`KnowledgeBaseCreate`): `paths` (non-empty list of file paths), `selectedModel` (id of
the installed base model), `modelName` (name of the assistant), `description` (optional).

Behaviour depends on `selectedModel`:

- The base model has **no** KB attached: a new assistant is created and `model_id` is its new id.
  A name already used by another installed model is refused with **409**.
- The base model **already has** a KB attached: the same route adds the new documents to the existing
  KB and `model_id` is that model's own id. The name is not read on this path.

### One job at a time per KB

Adding documents to an assistant whose KB still has a `pending` or `running` job is refused with
**409** (`STATE_CONFLICT`): two jobs would race on the same documents.

### Poll the job

```bash
curl http://127.0.0.1:27182/erudi/knowledge_base/108/status
```

```json
{
  "status": "completed",
  "status_updated_at": "2026-09-03T09:12:44.310Z",
  "error_message": null,
  "error_code": null,
  "failed_files": ["maintenance.docx"],
  "files_by_reason": {
    "pending_vision": ["scan.pdf"],
    "empty": [],
    "unreadable": ["broken.docx"]
  }
}
```

`status` is `pending`, `running`, `completed` or `failed`; `error_message` (English, for the log)
is only populated on `failed`. `error_code` is `"nothing_indexed"` for a `failed` job none of whose
files was indexed, `null` otherwise (a job that failed before its first file included).
`failed_files` lists the names of the files of this job whose ingestion failed for an internal
cause (empty when none did): the documents of the job's KB left in the `failed` status that were
created from the job's start on, so no earlier job's failure appears, plus the files that failed
before they had a row (bytes that could not be read, a row that could not be registered), which the
running backend keeps in memory per job. The status carries no reason for them. `files_by_reason`
names the job's documents in each of the statuses the user's file causes, every key present. The
names are also returned for a `failed` job.
Polling a **failed creation** also cleans up: the half-built assistant and its
`KnowledgeBase` are deleted (a failed *update* never deletes an assistant that already works).

### Deleting a Knowledge Base

There is no `DELETE` route in the knowledge base domain. Deleting the assistant deletes its KB:

```bash
curl -X DELETE http://127.0.0.1:27182/erudi/llms/108
```

The assistant's `link` is a copy of the base model's, so the **weights are left on disk** — they
belong to the base model. The `KnowledgeBase` row is deleted instead, and server-side cascades sweep
its `knowledge_documents` and its `rag.kb_chunks`. The assistant's conversations survive with a null
`llm_id`.

Deleting the **base** model of an assistant is guarded: without `?orphan_dependents=true` it returns
**409** with the list of dependent assistants. See [LLMs](llms.md#deleting-a-model-with-dependents).

## Notes for contributors

- **Always search through `search_kb_chunks_scored`.** `langchain-postgres` 0.0.17 writes the first
  query's `fts_query` onto the *shared* `HybridSearchConfig`, so every later sparse search would
  silently reuse the first query's tsquery. `search_kb_chunks_scored` works around it by building a
  fresh config on every call. Do not "simplify" that away.
- `HybridSearchConfig.primary_top_k` / `secondary_top_k` default to 4 in the library. They are
  overridden explicitly, otherwise the wide pool silently degrades to 4 + 4 candidates.
- The adaptive cut runs on **dense cosine similarities**, never on RRF scores.
- `langchain_text_splitters` is imported in function scope only: its package `__init__` eagerly
  imports every splitter it ships, which would tax every backend start.
- `pypdf`, `python-docx` (with `lxml`) and `openpyxl` are imported in function scope only, inside
  the extractor that needs them: the backend loads none of them at boot, and the first PDF, DOCX or
  XLSX read (an ingestion or a file attached to a question) pays the import. The PyInstaller specs
  pin them as hidden imports as a precaution, like the other function-scope imports.
- Related pages: [LLMs](llms.md), [Conversations](conversations.md),
  [API reference](../reference/knowledge_base.md), and the retrieval-quality evaluation notes in
  [`docs/dev/rag-quality-eval.md`](../dev/rag-quality-eval.md).
