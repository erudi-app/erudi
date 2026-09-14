---
title: LLMs
description: The model catalog, the download flow, installed models, deletion and rebinding, and the idle unload.
---

# LLMs

Erudi runs open-weight models locally. It **never converts or quantizes weights on the machine**: it
downloads pre-built artefacts in the format the active engine can load, and refuses anything else.
Everything in this page is served under `/erudi/llms` on `127.0.0.1:27182`.

## Engine format

Each engine declares the Hugging Face library tag of the artefacts it loads, `FORMAT_TAG`:

| Engine | Platform | `FORMAT_TAG` |
|--------|----------|--------------|
| `MLX_Engine` | macOS on Apple Silicon | `mlx` |
| `CUDA_Engine`, `CPU_Engine` (both via `BaseLlamaCppEngine`) | Windows / Linux | `gguf` |

The tag is the single gate: the bundled catalog, the live Hugging Face search and the by-link
download check are all filtered on it. See the [engines notes](../dev/architecture/engines.md) for
what runs underneath, including the llama.cpp builds.

## The catalog

The remote catalog (rows with `local = 0`) comes from **build-time snapshots bundled with the app**,
one per engine format:

- `backend/src/database/catalog_snapshot_mlx.json`
- `backend/src/database/catalog_snapshot_gguf.json`

At every boot, `startup_populate_database` reconciles the catalog against the snapshot for the active
engine. This is a zero-network operation: the catalog follows app releases, and there is no live
Hugging Face resync. A fresh install therefore has a full browsable catalog on its first boot,
offline.

Every catalog row is a pre-built quant that the active engine can load, so `runnable` is true by
construction; the only exception is a quant known to crash on load for this engine
(`BaseEngine.is_runnable`). For the `gguf` format the snapshot build also reads each candidate
repo's file listing and keeps a repo only when the downloader selects a loadable file from it (see
[The format gate](#the-format-gate)): when several repos quantize the same base model, the
resolver takes the best-ranked one that passes. A row's size is the bytes of the files a download
fetches, or an estimate when Hugging Face does not answer.

### Base models vs community models

`LLMResponse.is_base` marks a curated foundation model (`true`) versus a derived or community quant
(`false`). The UI splits the catalog on that flag alone — there is no hand-maintained list of base
model names.

### Hardware-aware ordering

`GET /erudi/hardware/app_startup` returns the machine's recommended model-size window, in billions of
parameters:

```bash
curl http://127.0.0.1:27182/erudi/hardware/app_startup
```

```json
{
  "backend_type": "mlx",
  "global_inference_score": 92.0,
  "global_inference_label": "Excellent",
  "raw_inference_score": 72.0,
  "recommended_param_min": 3.5,
  "recommended_param_max": 7.0
}
```

`recommended_param_range` (`backend/src/domains/hardware/services.py`) derives the window from real
usable memory at 4-bit, capped by a memory-bandwidth comfort floor. The frontend scores each model
against that window (ideal / good / tight / heavy), leads with instruction-tuned models
(`conversational`), and builds the "recommended for your machine" rail from one flagship per
well-known family. See [Hardware Detection](hardware.md) for the rest of the hardware surface.

## Downloading a model

Two routes start a download; both create a placeholder row (`local = 2`), a `DownloadJob`, and run
the transfer in a background task.

**From the catalog, by id:**

```bash
curl -X POST http://127.0.0.1:27182/erudi/llms/57/download
```

```json
{
  "job_id": 123,
  "remote_model_id": "57",
  "remote_model_link": "lmstudio-community/Qwen3-4B-MLX-4bit",
  "status": "pending",
  "progress": 0.0,
  "total_bytes": 0.0,
  "created_at": "2026-09-03T09:04:11.882Z"
}
```

**From a Hugging Face repo id, by link** — used for results of the live search, which are never
persisted into the catalog:

```bash
curl -X POST http://127.0.0.1:27182/erudi/llms/download/huggingface \
  -H 'Content-Type: application/json' \
  -d '{
        "link": "lmstudio-community/Qwen3-4B-MLX-4bit",
        "name": "Qwen3 4B",
        "param_size": 4.0,
        "quantized": true,
        "category": "general"
      }'
```

`HFDownloadRequest` fields: `link` (required), `name`, `type`, `param_size`, `quantized`
(default `true`), `category` (default `general`).

### The format gate

Before any byte is transferred, `_assert_repo_has_engine_artifact` lists the repo's files and refuses
a repo that ships nothing this engine can run:

- llama.cpp engines: at least one text-model `.gguf` file must be present (`mmproj-*.gguf` vision
  projectors do not count).
- MLX: the repo must be tagged `mlx` or declare it as its `library_name`.

Otherwise the request fails with an `InvalidInputException` explaining that Erudi only downloads
pre-built artefacts and does not convert weights locally.

On llama.cpp engines the download takes one quant: `pick_best_gguf` picks it, and every part comes
along when it ships as a llama.cpp split (`<name>-00001-of-00003.gguf`, each part a complete GGUF
file). A split whose listing misses a part is refused before any transfer. Weights cut into raw
byte pieces (`<name>-chunk-001-of-018.gguf`, `<name>.gguf.part1of2`, `<name>.gguf.a`,
`<name>.gguf-split-a`) are never selected: only the first piece carries a GGUF header, and nothing
loads them until they are joined into one file, which Erudi does not do. A repo whose only GGUF
weights are such pieces is refused with a message saying so.

### What happens on completion

1. The downloaded directory is validated by the active engine (`validate_local_artifact`). On
   failure, the artefacts are removed and the job is marked `failed` — a model never reaches
   `local = 1` without passing this gate.
2. The row flips to `local = 1`, and the real on-disk size is measured into `artifact_size_bytes`.
3. The job is marked `completed`. Only *then* are capabilities probed, best-effort:
   `supports_tools`, `supports_tools_wire` (does the engine's server actually parse this model's tool
   calls — this is what gates agentic Knowledge Base turns), `supports_vision`, and the sampling
   defaults captured from the base repo's `generation_config` or model card, alongside the model's
   context window. On llama.cpp engines that window comes from the GGUF metadata Hugging Face
   exposes for the downloaded repo — the value the `.gguf` file itself declares, which is the one
   the inference server reads; elsewhere it comes from `config.json`.

A probe failure leaves the model usable with unset capabilities (`null` = unknown, which every
consumer treats conservatively).

### Polling and cancelling

```bash
# one job
curl http://127.0.0.1:27182/erudi/llms/downloads/123/status

# the most recent job still active in the last 60 seconds (single-download UI)
curl http://127.0.0.1:27182/erudi/llms/downloads/status

# cancel
curl -X POST http://127.0.0.1:27182/erudi/llms/downloads/123/cancel
```

`status` is one of `pending`, `running`, `completed`, `failed`, `cancelled`. Polling a `failed` or
`cancelled` job also cleans up: the placeholder row is deleted and the temporary files removed.

## Searching

Two different searches:

```bash
# the local catalog, by name (case-insensitive partial match)
curl 'http://127.0.0.1:27182/erudi/llms/search?name=qwen'

# live Hugging Face, in the active engine's format, chat/vision models only
curl 'http://127.0.0.1:27182/erudi/llms/search/huggingface?q=uncensored&limit=30'
```

Hugging Face hits (`HFSearchResult`: `link`, `name`, `param_size`, `category`, `downloads`, `likes`,
`gated`, `pipeline_tag`, `quantized`) are ephemeral. Download a chosen one through
`POST /llms/download/huggingface`.

## Installed models

```bash
curl http://127.0.0.1:27182/erudi/llms/local
```

`local` encodes the state: `0` remote, `1` installed and ready, `2` downloading. Beyond the stored
columns, `LLMResponse` computes six fields at read time (no database column, so they self-heal):

| Field | Meaning |
|-------|---------|
| `runnable` | Can the active engine run this catalog entry |
| `supports_vision` | Image input, read from the artefact (`mmproj` projector for llama.cpp, `config.json` for MLX); `null` until downloaded |
| `weights_available` | Do the weights still exist on disk — `false` marks an orphaned KB assistant |
| `sampling_defaults` | Resolved per-model temperature, `top_p`, `max_tokens`, repetition penalty and the `max_tokens_cap`, plus the `source` that produced them |
| `context_window` | The model's trained context window in tokens (`generation_hints.context_length`); `null` when unknown |
| `allocated_context_window` | The window the engine's loaded child actually runs with; non-null only for the currently loaded model, resolved live and never persisted (see [Context windows](../dev/architecture/engines.md#context-windows)) |

KB assistants are ordinary rows flagged with `is_attached_to_kb` and a `kb_id`. They carry a **copy**
of their base model's `link`, so they share the weights on disk.

## Deleting a model with dependents

`DELETE /erudi/llms/{id}` is permanent, and refuses to run while the model is downloading (400).

Deleting a **KB assistant** is a direct 200: its weights belong to its base model and are left
untouched, its `KnowledgeBase` is deleted, and cascades sweep its documents and `rag.kb_chunks`.

Deleting a **base model that has dependent assistants** is guarded. Without the opt-in, it returns
**409** carrying the dependents payload:

```bash
curl -X DELETE http://127.0.0.1:27182/erudi/llms/42
```

```json
{
  "success": false,
  "error": {
    "type": "STATE_CONFLICT",
    "message": "Base model 'Qwen3 4B' has 2 dependent KB assistant(s). Retry with orphan_dependents=true to delete it anyway; the assistants remain and can be rebound to another base.",
    "detail": {
      "assistants": [
        { "id": 108, "name": "Financial Reports", "kb_id": 7, "conversation_count": 3 },
        { "id": 111, "name": "Support Handbook", "kb_id": 8, "conversation_count": 0 }
      ],
      "own_conversation_count": 5,
      "total_conversation_count": 8
    }
  }
}
```

(Every error `message` is returned with a support-contact sentence appended by
`AppBaseException`; it is elided above for readability.)

The same payload is available up front:

```bash
curl http://127.0.0.1:27182/erudi/llms/42/dependents
```

To proceed anyway:

```bash
curl -X DELETE 'http://127.0.0.1:27182/erudi/llms/42?orphan_dependents=true'
```

The base is deleted, the assistants **remain** with a dangling link (`weights_available` becomes
`false`), and every conversation bound to the base has its `llm_id` nulled server-side rather than
being deleted.

### Rebinding an orphaned assistant

```bash
curl -X POST http://127.0.0.1:27182/erudi/llms/108/rebind \
  -H 'Content-Type: application/json' \
  -d '{"new_base_llm_id": 57}'
```

This re-copies the new base's `link` and its descriptive fields onto the assistant, keeping the
assistant's own name, description and KB wiring. The target must be a local, non-assistant model
whose weights exist on disk; anything else returns 409.

## Route list

Regenerated from `backend/src/domains/llms/endpoints.py`.

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/erudi/llms/` | Every model, local and remote |
| `GET` | `/erudi/llms/local` | Installed models (`local = 1`) |
| `GET` | `/erudi/llms/remote` | Catalog entries (`local = 0`) |
| `GET` | `/erudi/llms/search?name=` | Search the catalog by name |
| `GET` | `/erudi/llms/search/huggingface?q=&limit=` | Live Hugging Face search in the engine's format |
| `GET` | `/erudi/llms/{llm_id}` | One model |
| `GET` | `/erudi/llms/{llm_id}/dependents` | KB assistants and conversation counts a delete would affect |
| `PUT` | `/erudi/llms/{llm_id}` | Update metadata (`LLMCreate` body) |
| `DELETE` | `/erudi/llms/{llm_id}?orphan_dependents=` | Delete, guarded by the dependents check |
| `POST` | `/erudi/llms/{assistant_id}/rebind` | Re-point an orphaned KB assistant (`{"new_base_llm_id": ...}`) |
| `POST` | `/erudi/llms/{llm_id}/download` | Download a catalog entry (no body) |
| `POST` | `/erudi/llms/download/huggingface` | Download by repo id (`HFDownloadRequest`) |
| `POST` | `/erudi/llms/downloads/{job_id}/cancel` | Cancel a running download |
| `GET` | `/erudi/llms/downloads/{job_id}/status` | Poll one download job |
| `GET` | `/erudi/llms/downloads/status` | Most recent job active in the last 60 s |

There is no unload route, and no training route.

## Memory and the idle unload

Only **one model is resident at a time**. `BaseEngine` keeps `_model`, `_tokenizer`, `_model_id` and
`_last_used` as class attributes shared across requests, guarded by a lock, and a background monitor
started in the FastAPI lifespan (`start_cleanup_task`) ticks every 300 seconds and calls `cleanup()`
when the model has been idle longer than `_max_idle_time`, also 300 seconds
(`backend/src/engines/base_engine.py`).

In practice: after about five minutes without a request, the model is unloaded and its memory freed.
The next request reloads it, which costs the usual load time. Switching to another model swaps the
resident one.

Do not instantiate engines. Call class methods on the result of `BaseEngine.get_engine()`; the
selected engine is also available as `src.core.config.LLM_Engine`.

## Related pages

- [Knowledge Base](knowledge_base.md) — turning an installed model into a KB assistant.
- [Conversations](conversations.md) — using an installed model in a chat.
- [Hardware Detection](hardware.md) — the scores and the recommended size window.
- [API reference](../reference/llms.md) — generated from the docstrings.
