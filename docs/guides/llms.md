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

### Publication date

`LLMResponse.published_at` (and `published_at` on each Hugging Face search hit) is an ISO-8601
date, or `null` when it is unknown. Cards print it as "Published <month year>" in the interface
language, and print nothing for `null`. It is computed at response time from the `Created:` line of
the row's stored `model_metadata`, so there is no column and no network call at render time. The
card shows when the **publisher** released the model, never a quant's upload date, and the date is
the creation of the base repository on Hugging Face, which can precede the public release.

- A curated base row holds its base repo's creation date.
- A derived or community row resolves its base through the `base_model:*` card tag
  (`resolve_base_repo`) when the catalog is built (snapshot generation), with one `model_info` call,
  and stores the base's date. A base that cannot be read leaves no date; a repo with no
  `base_model:` tag is its own publisher.
- A Hugging Face search hit carries a date only when it is itself a base (no `base_model:` tag);
  a quant hit carries `null`, since resolving its base would cost a Hub call per hit.
- A by-link download sends the hit's `published_at` in `HFDownloadRequest`, and the placeholder row
  stores it as a `Created:` line, so the installed card keeps the date.

A row whose metadata has no `Created:` line, such as a snapshot entry built before the date was
resolved, shows no date; the next snapshot regeneration fills it in.

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
(default `true`), `category` (default `general`), `published_at` (ISO date of the hit, optional).

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

### On a flaky network

The files of a download go through `HfFileSystem`, the `.safetensors` shards of an MLX model side
by side in threads. huggingface_hub retries a request that meets a network error (connection,
timeout, cut-off read) up to 5 times. Its HTTP client is one object shared by every Hub request in
the process: the files of a download, the catalog's metadata calls, the Hugging Face search, the
sampling hints, the embedding model's download. A connection error closes that client and the retry
goes out on a new one (a timeout or a cut-off read is retried on the same client). Erudi's client
(`src/core/hf_http.py`, installed by `src/core/config.py` before any Hub request) hands a request
another thread had already started on the closed client to the live one, so one request's retry
never fails a concurrent one. The same client caps a server's `Retry-After` or `RateLimit` reset at
330 seconds per retry, above the Hub's own 300-second rate-limit window, so only an absurd wait is
shortened. A download whose retries run out fails with "you appear to be offline - check your
connection and retry" (the embedding model's download with its own no-connection message; both
decide through `is_offline_error`): an environmental failure, which Diagnostics leaves out of its
recent errors and its badge (see [logging](../logging.md)). If huggingface_hub's own client
settings cannot be read at start, the backend logs one WARNING and keeps huggingface_hub's client
as it is.

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
`cancelled` is final: the download stops, says so in one INFO line, and never turns the job
`failed`. The cancel commits at once (the job reads `cancelled`, the placeholder model is
gone), so a status poll meanwhile finds nothing left to clean. The transfer sees the cancel
at its next chunk; the cancel deletes the partial files once every transfer of the job has
stopped (Windows refuses to delete a file still open), waiting up to 10 seconds, and a
transfer slower than that deletes them itself when it stops, after the cancel has answered.
Once every file is transferred the download is being installed and a cancel is refused
with `409`: the job then ends `completed`, or `failed` if the integrity gate rejects the
files. A cancel of a job that ended meanwhile is refused (`400`) and touches nothing. A
cancel that lands before the download task started leaves the job `cancelled` and nothing
is downloaded: the task registers its transfer, then claims its job only while it is still
`pending`. A cancel that finds no transfer cancels a `pending` job only; if the task has
claimed it meanwhile, the cancel finds its transfer and stops it like any other. A
`running` job whose task is gone cannot be cancelled (`400`): the next start resolves it.
A cancel logs nothing above INFO. While the engine switch waits for a download being
installed, its control says so ("Waiting for the download to finish installing…").

Quitting the app is not a cancel: the backend interrupts the download, leaves the job
`running` and its partial files on disk, and the next start resolves them (the job ends
`failed` with "Download interrupted due to application shutdown" and the code
`interrupted`, and the files are removed, unless the transfer had already completed). The
cleanup logs one INFO line saying the download was interrupted; the integrity-check WARNING
is kept for a download whose transfer had reached 100 % yet whose files fail the engine's
check.

### Download failures

A `failed` job carries two fields: `error_message`, the English text of the failure, for
`backend.log` and Diagnostics; and `error_code`, a stable code the download widget turns into
a sentence in the interface language (`downloads:errors.codes.*`,
`frontend/src/utils/downloadErrors.js`). The widget never shows `error_message`. The code is
read off the exception where the failure is recorded (`download_error_code` in
`src/domains/llms/download_errors.py`): a refusal Erudi raises itself is marked with its code at
the raise site, a failure from below is recognised from its type anywhere in the exception
chain.

| `error_code` | When |
|--------------|------|
| `offline` | No network reaches the Hub: `OFFLINE_DOWNLOAD_MESSAGE`, or any error `hf_http.is_offline_error` reads as a network failure |
| `disk_full` | An `OSError` with `ENOSPC` while writing the files |
| `gated` | The repository needs an account (`GatedRepoError`, HTTP 401/403) |
| `not_found` | The repository or a file of it does not exist (`RepositoryNotFoundError`, `EntryNotFoundError`, HTTP 404); the Hub also answers 401 for a missing repository, which keeps this code |
| `rate_limited` | HTTP 429 once huggingface_hub's own retries are spent |
| `interrupted` | The app quit during the transfer (written by the next start's cleanup) |
| `no_engine_artifact` | The repository ships nothing this engine runs |
| `raw_gguf_chunks` | The GGUF weights are raw byte pieces only |
| `incomplete_split` | The listing misses a part of the chosen split quant |
| `unsafe_file_name` | A listed file name would land outside the model folder |
| `not_runnable` | The quant is known not to run on this engine |
| `integrity_failed` | The files arrived but the engine's integrity gate refused them |
| `unexpected` | Anything else; the job's ERROR record carries the traceback |

A cancelled job is not a failure and carries no code. A job that failed before codes were
stored has `error_code: null`, and the widget shows its generic sentence for it, as it does
for a code it does not know.

## Searching

Two different searches:

```bash
# the local catalog, by name (case-insensitive partial match)
curl 'http://127.0.0.1:27182/erudi/llms/search?name=qwen'

# live Hugging Face, in the active engine's format, chat/vision models only
curl 'http://127.0.0.1:27182/erudi/llms/search/huggingface?q=uncensored&limit=30'
```

Hugging Face hits (`HFSearchResult`: `link`, `name`, `param_size`, `category`, `downloads`, `likes`,
`gated`, `pipeline_tag`, `quantized`, `published_at`, `artifact_size_bytes`) are ephemeral. Download
a chosen one through `POST /llms/download/huggingface`.

`artifact_size_bytes` is the size a download of the hit installs, measured the way a catalog row's
is: after the hits are filtered, the search asks Hugging Face for each one's file listing
(`model_info(files_metadata=True)`) and sums the files the downloader would fetch, so for a GGUF
repo holding several quantizations it is the one quantization picked, not the whole repo. Those
requests are bounded so the search still answers quickly:

- at most four at once, one per hit (so at most `limit`), each without retry;
- one budget of 4 seconds for the whole batch; a request not sent by then is not sent;
- the first `429` from Hugging Face stops every request still to send for that search, and no
  search sends a size request for a cooldown after it: 60 seconds, or longer when the response's
  `Retry-After` or `RateLimit` header announces a later reset;
- a measured size is kept in memory for an hour (256 repos at most), so the same search typed again
  sends no size request for the hits already measured (the search itself still goes out);
- the requests run on daemon threads, so one still on the wire when the app quits does not hold the
  backend's exit.

On Windows and Linux (GGUF), a repo from which the downloader selects no loadable GGUF (only raw
byte pieces, only an `mmproj`, a split missing a part) carries `null`: the download refuses it, so
no size is shown as measured for it.

A hit whose size could not be read (timeout, rate limit, cooldown, a gated repo without a token, any
other failure) carries `null`, and its card shows the `~` estimate from `param_size`, as before;
the search itself never fails because of a size. Each search writes one `INFO` line to
`backend.log`,
`HF search sizes: enriched=<measured>/<hits> elapsed_ms=<ms> stopped_on_429=<True|False> cooldown=<True|False>`
(`cooldown=True`: an earlier `429`'s cooldown kept every size request from going out), plus a
`WARNING` when Hugging Face rate-limited the batch or when requests failed.

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
| `sampling_defaults` | Resolved per-model temperature, `top_p`, repetition penalty, the fallback `max_tokens` and the `max_tokens_cap` it is clamped to (both only matter when the engine cannot report its window — see [Output budget](conversations.md#output-budget)), plus the `source` that produced them |
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

A delete that goes through first unloads the resident model it would strand. For a base model
that is the loaded model whether it was loaded under the base's own id or under a dependent
assistant's, since the engine matches the weights folder the child runs on
(`BaseEngine.serves_artifact_under`). Deleting an assistant unloads it only when it is the
loaded model: the files stay, so a loaded base stays loaded. A refused delete (409) unloads
nothing.

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
| `GET` | `/erudi/llms/loaded` | The model in memory (`{"llm_id", "name"}`), or `null` |
| `POST` | `/erudi/llms/loaded/unload` | Unload the model named in the body (`{"llm_id": ...}`) now: `204` (also when nothing is loaded), `409` while busy or when another model is loaded |
| `GET` | `/erudi/llms/{llm_id}` | One model |
| `GET` | `/erudi/llms/{llm_id}/dependents` | KB assistants and conversation counts a delete would affect |
| `GET` | `/erudi/llms/{llm_id}/conversation_defaults` | The values a new conversation on the model starts from, with their source |
| `PUT` | `/erudi/llms/{llm_id}/conversation_defaults` | Save or clear values for the model (`/config global`) — see [Conversation defaults per model](conversations.md#conversation-defaults-per-model) |
| `PUT` | `/erudi/llms/{llm_id}` | Update metadata (`LLMCreate` body) |
| `DELETE` | `/erudi/llms/{llm_id}?orphan_dependents=` | Delete, guarded by the dependents check |
| `POST` | `/erudi/llms/{assistant_id}/rebind` | Re-point an orphaned KB assistant (`{"new_base_llm_id": ...}`) |
| `POST` | `/erudi/llms/{llm_id}/download` | Download a catalog entry (no body) |
| `POST` | `/erudi/llms/download/huggingface` | Download by repo id (`HFDownloadRequest`) |
| `POST` | `/erudi/llms/downloads/{job_id}/cancel` | Cancel a running download |
| `GET` | `/erudi/llms/downloads/{job_id}/status` | Poll one download job |
| `GET` | `/erudi/llms/downloads/status` | Most recent job active in the last 60 s |

The two `loaded` routes are declared before every `/{llm_id}` route, which would otherwise read
`loaded` as an id. There is no training route.

## Memory and the idle unload

Only **one model is resident at a time**. `BaseEngine` keeps `_model`, `_tokenizer`, `_model_id` and
`_last_used` as class attributes shared across requests, guarded by a lock, and a background monitor
started in the FastAPI lifespan (`start_cleanup_task`) ticks every 30 seconds and calls `cleanup()`
when the model has been idle longer than `_max_idle_time` (`backend/src/engines/base_engine.py`).

`_max_idle_time` is the Settings card **Unload the model when idle**: 5, 15, 30 or 60 minutes, or
**Never** (`user_settings.idle_unload_minutes`, `0` for Never, 5 by default). At the default an idle
model is released about 300 to 330 seconds after its last use; with Never it stays loaded however
long it is unused. The lifespan applies the stored delay to the engine that runs, after the
CUDA-to-CPU swap, and `PUT /erudi/user_settings/` applies a change at once, without a relaunch
(`BaseEngine.set_idle_unload_minutes`). It governs the chat model only: the knowledge-base embedding
model, once used, stays loaded until the backend stops. Nothing shorter than five minutes is offered, because the memory
brake's idle timings are calibrated against a reap at five minutes or later; with Never the brake
still acts when the computer runs short of memory.

The conversation header's settings panel (in a conversation and on the new-conversation page) shows
a small red **Unload <model>** button while a model is loaded, naming it whichever conversation
loaded it (`GET /erudi/llms/loaded`, which checks that the inference child is alive, read when the
panel opens, when an answer starts streaming, once a turn has completed, when the conversation's
model changes, and after an unload). Pressing it calls `POST /erudi/llms/loaded/unload` with the
`llm_id` it names, which runs `BaseEngine.unload` under `try_generation_guard`: while an answer, a reload, a memory-brake sample or the idle check
holds the engine (waited out for at most 2 seconds) it answers 409 "The model is busy; try again in
a moment." and nothing is interrupted; when another model is loaded than the one named it answers
409 "Another model is loaded now." and nothing is unloaded, not even the loaded model's idle clock
restarted (the button then names the loaded model);
otherwise the model goes through the same `cleanup()` as an idle unload, so on llama.cpp a window
the memory brake reduced recently stays reduced at the next load. The cleanup thread is shielded: a
request cancelled meanwhile keeps the lock until the thread is done, the record is written, then the
cancellation is raised.

The next request reloads an unloaded model, which costs the usual load time; the next turn of an
existing conversation says why it reloads (`memory_reload` with `reason: "idle"` after the idle
delay, `"manual"` after the Unload button; a new conversation shows its usual first-reply hint).
Switching to another model swaps the resident one.

Do not instantiate engines. Call class methods on the result of `BaseEngine.get_engine()`; the
selected engine is also available as `src.core.config.LLM_Engine`.

## Related pages

- [Knowledge Base](knowledge_base.md) — turning an installed model into a KB assistant.
- [Conversations](conversations.md) — using an installed model in a chat.
- [Hardware Detection](hardware.md) — the scores and the recommended size window.
- [API reference](../reference/llms.md) — generated from the docstrings.
