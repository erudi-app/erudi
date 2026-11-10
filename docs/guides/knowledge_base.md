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

**Images are not read.** Erudi bundles no OCR or vision tier today, so images and scanned PDFs are
registered in the KB with the status `pending_vision` and contribute zero searchable chunks. A batch
made only of such files fails the job rather than reporting a misleading success.

The routing table lives in `backend/src/ingestion/reader.py`; `DocumentReader.read(path)` is the
single extraction surface the service layer sees.

## The ingestion pipeline

Ingestion runs as a FastAPI background task, one file at a time
(`backend/src/domains/knowledge_base/services.py`). A failure on one file marks that document
`failed` and the run continues.

1. **Hash and dedup.** The file bytes are SHA-256 hashed. A hash already present in this KB
   (`knowledge_documents.content_hash_sha256`, unique per `kb_id`) is skipped as a duplicate,
   unless its row is `failed`: a failed document never blocks its own file. The failed row is
   deleted (its chunks follow, `ON DELETE CASCADE`) and the file goes through the pipeline again
   as a new document.
2. **Register the document.** A `KnowledgeDocument` row is created and committed before indexing —
   the vector store writes through its own connection and needs the row for its foreign key.
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
6. **Embed.** `E5Embeddings` (`src/ingestion/embeddings.py`) holds a resident
   `multilingual-e5-small` (384 dimensions, 512-token window). The asymmetric prefixes are
   mandatory: `passage: ` for indexed chunks, `query: ` for searches. Vectors are L2-normalized, so
   cosine similarity is a dot product. The embedded text is
   `passage: [document_name:<file>]\n<breadcrumb>\n\n<chunk>` — the document-name prefix boosts
   retrieval but is **not** stored: the stored content is the clean chunk, which is what goes into
   the prompt.
7. **Store.** `add_kb_chunks` writes the vectors plus the metadata columns
   (`kb_id`, `document_id`, `source_file`, `page`, `chunk_index`) into `rag.kb_chunks`.

A document that extracts fine but yields zero chunks (blank file, parser that returned no text) is
marked `empty`, never `active`.

### Document statuses

`active` (indexed and retrievable), `empty` (no indexable content), `failed` (ingestion raised),
`pending_vision` (image or scanned PDF, awaiting a future OCR/vision tier). Defined in
`backend/src/entities/KnowledgeDocument.py`.

### Job outcome

The KB job reports `completed` only when at least one file was indexed, or when every file was a
duplicate of content already in the KB. A batch that indexed nothing new and had no duplicates fails
with a message counting the failed, empty and pending-vision files.

A job that completes with some files failed still reports `completed`; the status route names those
files (see [Poll the job](#poll-the-job)) and the Knowledge Base page shows them instead of the plain
success message. The reason stays in the log and in Diagnostics.

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
3. **Token budget**: keep whole chunks best-first until the model's budget is spent.

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

## The embedding-model gate

The KB needs `intfloat/multilingual-e5-small` for both embeddings and token-accurate chunking. It is
**not** bundled: it is downloaded on demand into the app's own cache directory
(`config.CACHE_DIR`, i.e. `data/models_cache`), so a single download serves both consumers.

Presence is filesystem-driven, never a database flag: `embedding_model_available()` checks every file
a `SentenceTransformer` load touches, so a partial download reads as unavailable and the gate
reappears. Once the cache is complete, loads run with `local_files_only=True` and never touch the
network — an offline machine works.

```bash
curl http://127.0.0.1:27182/erudi/knowledge_base/embedding-model/status
# {"available": false, "downloading": false, "error": null}

curl -X POST http://127.0.0.1:27182/erudi/knowledge_base/embedding-model/download
# {"available": false, "downloading": true, "error": null}
```

`POST .../download` is idempotent: it is a no-op if the model is already present or a download is
already running. The download runs in a background thread, so navigating away from the KB page does
not lose it; the UI polls the status route.

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

### Poll the job

```bash
curl http://127.0.0.1:27182/erudi/knowledge_base/108/status
```

```json
{
  "status": "completed",
  "status_updated_at": "2026-09-03T09:12:44.310Z",
  "error_message": null,
  "failed_files": ["maintenance.docx"]
}
```

`status` is `pending`, `running`, `completed` or `failed`; `error_message` is only populated on
`failed`. `failed_files` lists the names of the files of this job whose ingestion failed (empty
when none did): the documents of the job's KB left in the `failed` status that were created from
the job's start on, so no earlier job's failure appears. The status carries no per-file reason.
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
  imports `sentence_transformers`, which would tax every backend start.
- Related pages: [LLMs](llms.md), [Conversations](conversations.md),
  [API reference](../reference/knowledge_base.md), and the retrieval-quality evaluation notes in
  [`docs/dev/rag-quality-eval.md`](../dev/rag-quality-eval.md).
