# Erudi — QA Acceptance Scenarios

Walk this list on every **release candidate** (the signed *draft* build — see
[`docs/dev/release-qa-checklist.md`](docs/dev/release-qa-checklist.md) for the
process) before promoting it to `latest`.

Each line reads: **on page X, when I do Y, then Z must happen.** Tick the box if
Z happens; if it doesn't, mark it **FAIL** and open an issue. The plain language
is deliberate so anyone — not just a developer — can run the pass.

Each screen lists the **happy path** first, then **edge cases & errors** — don't
skip the edge block, that's where regressions hide. Covered: the five app
screens, the shared chrome, and non-functional behavior.

---

## Models / Explore — `/erudi/models`

**Happy path**
- [ ] When I launch the app, then I land on the Models screen and my **machine readout** shows (chip name, runtime, unified memory, GPU cores, bandwidth, inference score, and a "Sweet spot" size range).
- [ ] When at least one base model fits my machine, then a **"Recommended for your machine"** row shows up to 3 fitting models.
- [ ] When the catalog is loaded, then the left rail lists each **capability category with a live count** (General, Reasoning, Code, Vision & Multimodal, Math, Medical, Function Calling, Safety) plus Community, and clicking an entry scrolls to it.
- [ ] When I have downloaded models, then the **Installed** section lists them with Chat / Info / Knowledge Base / Delete actions.
- [ ] When I click **Download** on a runnable model and confirm, then a progress widget shows percentage, time left, cancel, and collapse; on completion the model appears in Installed.
- [ ] When I type a query in **Search Hugging Face** and press Enter, then results render ranked best-fit-first.

**Edge cases & errors**
- [ ] When it is my very first launch, then the **Welcome** dialog appears once; on later launches it does not.
- [ ] When I have no installed models, then the Installed section shows "No models installed yet…" with my recommended size.
- [ ] When no base model fits my machine, then the "Recommended" section is hidden (not empty).
- [ ] When a model is **not runnable on my hardware**, then its card shows "Not supported on your hardware" and Download is disabled.
- [ ] When a model is **gated** (from a Hugging Face search hit), then the card shows a "gated" tag.
- [ ] When I browse the **bundled catalog**, then no card links to a gated repository at all — Erudi downloads anonymously, so a gated link would 401 whoever clicked it *(#392; gated repos are dropped at snapshot time, not flagged)*.
- [ ] When a model is **under ~4B parameters**, then its card — catalog, explore and installed alike — carries the note that tool use, knowledge-base search and multi-step reasoning are unreliable below ~4B *(#381)*; a 7B or unknown-size card carries no such note.
- [ ] When a category carousel has more than 4 models, then a "See all" control expands it to a grid (and back).
- [ ] When I apply a **size filter** or **"Fits my machine"** and nothing matches, then I see "No models match these filters. Widen the size range or turn off 'Fits my machine'."
- [ ] When there are no base models at all, then the browse area shows "No base models available" (not a crash).
- [ ] When a Hugging Face search returns nothing runnable, then I see "Nothing runnable matched…" (a helpful message, not an error).
- [ ] When I am **offline** and run a Hugging Face search, then I see "No internet connection for the moment." and no request is made.
- [ ] When a download **fails**, then the widget shows the error and a "Download failed. Please try again." message.
- [ ] When I **cancel** an in-progress download, then it stops and the model returns to a not-downloaded state (no "Download failed" dialog).
- [ ] When I delete an installed model and confirm, then it is removed and a success message shows; if the delete request fails, the list is left intact with an error.
- [ ] When I delete a base model that **powers KB assistants**, then the confirmation dialog lists the assistants and conversation count and warns they must be re-bound; **Delete anyway** removes the base while the assistants and conversations are kept.
- [ ] When an assistant's base model was deleted, then its card shows **"Model weights missing"**, Chat is disabled, and **Re-bind** to another installed model restores it (the card then reads "Uses the weights of <that model>").
- [ ] When the network drops, then the connection pill switches from "Connected" to "Offline" live.

## Chat — `/erudi/chat`

**Happy path**
- [ ] When I open Chat with at least one local model, then the first model is auto-selected in the "Chat with" picker.
- [ ] When I type a prompt and press Enter, then it sends; Shift+Enter inserts a newline.
- [ ] When I send a prompt, then a new conversation is created and I am taken to it, where the reply **streams token by token**.
- [ ] When I adjust Creativity / Diversity / Max Tokens or customize the prompt, then those settings carry into the conversation.

**Per-model sampling defaults (#388)**
- [ ] When I select a model whose publisher ships sampling values, then Creativity / Diversity start at **that model's** values rather than a global 0.2 / 0.95 (Qwen3 starts at 0.6 / 0.95, Qwen2.5 at 0.7 / 0.8).
- [ ] When the publisher ships a **greedy** temperature (Qwen2.5-VL ships `0.000001`), then the slider shows **0** and the model answers normally — no stream of `!` *(#395: greedy is sent as an exact 0)*.
- [ ] When I send the **same prompt in several fresh conversations** on an Apple Silicon model at a non-zero temperature, then the answers differ *(#402: a fresh seed per request; a short factual answer may still converge)*.
- [ ] When I switch model mid-setup, then the sliders **re-default** to the new model's values.
- [ ] When I open the Max Tokens control, then its ceiling is the model's own cap (`min(model context, engine context)`), not a fixed 1024.
- [ ] When the Creativity slider is dragged to the top, then it reaches **2**, not 1.
- [ ] When the selected model's publisher recommends **nothing**, then a muted one-liner under the sliders says so ("No sampling recommendation from this model's publisher; neutral defaults applied") — in the conversation header panel, the pre-conversation panel and the model info modal, and **never** on a card face or in the Arena.
- [ ] When the publisher **does** recommend values, then that one-liner is absent.

**Edge cases & errors**
- [ ] When I send **without any image** (plain text), then the model answers normally.
- [ ] When I attach image(s) on a **vision-capable** model (button, paste, or drag-and-drop) and send, then thumbnails show (up to 4) and the images are used in the answer.
- [ ] When the **selected model is NOT vision-capable**, then the image **attach button is not shown at all** (vision-only affordance) and pasting/dropping an image is ignored; if an image still reaches the backend it is stripped, so the answer is plain text (never broken).
- [ ] When I try to attach a **5th** image, then it is rejected (cap of 4) and the attach button is disabled at 4.
- [ ] When I drop a **non-image** file, then it is refused with "This format is not supported." and nothing is attached.
- [ ] When the input is **empty or whitespace** only, then the send button is disabled.
- [ ] When the model is loading on the first reply, then a "First response may take a bit longer while loading the model into memory…" hint shows.
- [ ] When I have **zero local models**, then the composer is replaced by "No current local models found, please add local models to proceed."
- [ ] When a model download **completes while I am sitting on Chat**, then the model list refreshes by itself and the composer unlocks — I don't have to navigate away and back.
- [ ] When I open Chat via `?model=<name|id>`, then that model is pre-selected (else the first model stays).
- [ ] When the **backend is unreachable**, then an error dialog "Failed to load models: …" shows.

## Conversation — `/erudi/conversations/:id`

**Happy path**
- [ ] When I open an existing conversation, then its **full history** renders in order (my messages right, assistant left as markdown) and the model/settings populate.
- [ ] When I send a follow-up, then a user bubble appears immediately and the assistant reply streams live; both are saved.
- [ ] When I send the first message of a new conversation, then a short (2–4 word) **title** appears in the sidebar, free of any reasoning fragments, ideally in the conversation's language.
- [ ] When I reload the page, then the full text history re-renders from the database.

**Reasoning / thinking models**
- [ ] When a thinking model (e.g. Qwen3) generates, then its reasoning streams into a **collapsible "Reasoning" strip** above the answer — never into the answer bubble itself.
- [ ] When the turn ends, then the strip settles to a collapsed "Reasoning — N steps" summary; expanding it shows the full trace, and the trace **survives a reload**.
- [ ] When an agentic model narrates **before calling a tool** ("Let me search the documents…"), then that narration lands in the reasoning strip, not in the answer bubble — the answer zone holds only the final grounded answer.

**Knowledge-Base / agentic behavior**
- [ ] When the model has a KB attached and is **tool-capable (agentic)**, then on a document question the model **calls the KB search tool itself** before answering, and the answer references the source.
- [ ] When an agentic, KB-attached model gets a **chit-chat / meta turn** (not about the documents), then it answers directly **without** searching the KB.
- [ ] When the model has a KB attached and is **not tool-capable (systematic)**, then relevant document excerpts are **injected up-front** every turn and the answer is grounded in them.
- [ ] When a small / uncooperative model is KB-attached, then the answer should still reference the source *(prompt-instructed only — no clickable source UI; acceptance = it mentions the doc when it complies)*.
- [ ] When KB retrieval **fails** (broken/empty vector store), then the turn **degrades to a no-context answer** instead of erroring.
- [ ] When I ask about something the documents **do not cover** (e.g. an undocumented product variant), then the agentic model searches, finds nothing relevant, and **says the documents don't cover it** — it never invents a value and never substitutes a nearby fact (e.g. another model's price).
- [ ] When a follow-up returns to a **topic searched earlier in the conversation**, then the model runs a **fresh search** rather than answering from its memory of old excerpts (old tool results are placeholder-stripped from context) — it must not claim "not in the documents" without having just searched.
- [ ] When ONE question spans **two subjects living in two different documents** ("what is the drone's payload, and how many remote days are allowed?"), then the answer grounds **both** facts — neither half is dropped or answered from world knowledge *(multi-subject coverage — see #85)*.
- [ ] When I inspect any agentic answer, then **no raw tool markup** (`<tool_call>`, JSON arguments, function-call syntax) appears in the answer bubble or in the persisted history; the search call and its excerpts appear only inside the reasoning strip.

**Web search (#310)**
- [ ] When I create a new conversation, then its **Web search** toggle (settings panel, next to Max Tokens) starts at the value of the global Settings-page default at creation time.
- [ ] When I flip the Web search toggle in an open conversation, then it persists immediately (survives a reload) and **takes effect on the next turn** — no Apply needed.
- [ ] When web search is ON with a tool-capable model and I ask a question needing a **current external fact**, then the reasoning strip shows a `web_search` call with its results, and the answer **cites source URLs** from those results.
- [ ] When web search is ON and I ask something the model already knows ("capital of France"), then it answers **directly with zero web calls**.
- [ ] When web search is ON but the machine is **offline**, then the turn completes with the model relaying the honest tool text ("Error during Web Search: no internet connection") — no hang, no invented answer.
- [ ] When the conversation's model is **not verified tool-capable**, then the model never receives the web tool, whatever the toggle says (the toggle stays visible; it simply has no effect on such models).
- [ ] When I change the **global** web-search default in Settings, then existing conversations keep their own toggle unchanged; only conversations created afterwards inherit the new default.

**Multimodal / multi-turn**
- [ ] When I send an image on a vision model, then it is used for that turn and **carried forward** on later turns so a follow-up ("what colour is his hair?") works without re-attaching; as soon as I send a **newer** image, every older one collapses to an `[image]` marker in the model's context (at most one turn's images ever reach the model), while the display keeps all images.
- [ ] When I reload a conversation with **file-attached** images, then the thumbnails re-render (for images still present on disk).
- [ ] When I reload a conversation whose image was **pasted from the clipboard**, then it shows an "image attachment" placeholder, not the image *(clipboard images aren't restorable yet — see #136)*.
- [ ] When an attached image's original file was **moved/deleted**, then that image quietly shows nothing on reload (no broken-image artifact).

**Edge cases & errors**
- [ ] When I hover a message, then copy and star controls appear; a starred message stays starred after reload and is fed back as context on later turns.
- [ ] When I delete the conversation I'm viewing, then it's removed and I'm redirected to `/erudi/chat`; deleting a different one keeps me in place.
- [ ] When I quit and relaunch and reopen the conversation, then its full history is intact.
- [ ] When generation **fails** or the connection **drops** mid-reply, then a red error message shows and any partial reply is kept.
- [ ] When an answer contains a **markdown image pointing at a web address** (ask the model to reply with exactly `![logo](https://example.com/logo.png)`), then no picture is fetched or shown — at most a broken-image placeholder — because the window loads no remote images, so no site learns my address from an answer on screen.
- [ ] When the conversation's assigned model was **deleted**, then the conversation survives with no model assigned: sending is **blocked**, the header model picker shows a red "Please select a model" attention state, and **explicitly picking** an installed model unblocks sending (no auto-fallback).

## Arena — `/erudi/arena`

**Happy path**
- [ ] When I open Arena with at least two local models, then two panels show, pre-filled with the first two models.
- [ ] When I pick a model or change settings/custom prompt in one panel, then only that panel changes.
- [ ] When I send one prompt, then it goes to **every** panel and each streams its own model's answer.
- [ ] When I click "+", then a panel is added (up to 4, layout reflows); the trash removes one (minimum 1).

**Edge cases & errors**
- [ ] When only **one** local model exists, then both panels default to it.
- [ ] When two panels use **different** models, then the answers are produced one model after another (single engine — not truly simultaneous), and the run still completes for every panel.
- [ ] When two panels use the **same** model, then the loaded model is reused (no reload between them).
- [ ] When a panel's model **errors**, then that panel shows "[Erreur]" in red while the others still resolve.
- [ ] When a panel's model has a **KB attached**, then KB context is auto-injected for that panel (no toggle).
- [ ] When I attach an **image** in Arena, then attaching is allowed as soon as **any** panel's model is vision-capable; vision panels use the image for that turn (Arena is stateless — the image lives for this turn only), and a non-vision panel answers text-only with a notice that the images were ignored.
- [ ] When a generation is running, then settings/model pickers are disabled; there is **no stop button** — the run must finish.
- [ ] When I submit an **empty** prompt, then it does not send.

## Knowledge Base / Create Assistant — `/erudi/attach_knowledge_base`

**Happy path**
- [ ] When I open the screen for the **first time** (embedding model not yet installed), then a dialog offers to download the embedding model (multilingual-e5-small) once; accepting downloads it and confirms "the Knowledge Base is ready to use"; declining ("Not now") returns to the Models page *(#146/#157)* and the offer returns on the next visit.
- [ ] When I open the screen, then I see the KB description, a chat-capabilities rating (my machine's inference label/score), the local-model library, a name field, and a drag-and-drop area.
- [ ] When I select a base model, type a name and **click Check to lock it**, add supported files (`.pdf`/`.txt`/`.docx`/`.xlsx`/`.csv`/`.md`), and click "Create Assistant" + confirm, then a spinner polls progress.
- [ ] When ingestion completes, then "Data attached to your Assistant successfully!" shows and the form resets.

**Edge cases & errors**
- [ ] When I leave the assistant name **unlocked** (didn't click Check), or pick no model, or add no files, then "Please fill in all required fields" shows and nothing is sent.
- [ ] When I add a **supported document** beyond `.pdf`/`.txt` (`.docx`, `.xlsx`, `.csv`, `.md`), then it is accepted; an **unsupported** file (e.g. `.png`, `.zip`) isn't offered by the picker and a dropped one is ignored.
- [ ] When I add the **same file twice**, then it is de-duplicated.
- [ ] When I submit a **scanned / image-only PDF** alongside readable files, then it is accepted as *pending vision* (no searchable content yet) and the job completes for the readable ones; a **pending-vision-only** upload fails with "no searchable content" (no OCR tier yet).
- [ ] When I submit an **empty / no-text** file (and nothing else indexes), then the job **fails** with a "no searchable content" message and the document is flagged *empty* — never a false success.
- [ ] When **every** submitted file is unreadable/unsupported, then the job fails with a clear error and the half-built assistant is auto-cleaned up.
- [ ] When **some** files fail but at least one ingests, then the job still completes for the good ones.
- [ ] When ingestion **fails** (network/HTTP), then an error dialog shows the reason.
- [ ] When the selected base model **already has a KB**, then submitting **updates** the existing KB with the new files instead of creating a new assistant.

## Settings — `/erudi/settings`

- [ ] When I click the **gear icon** at the bottom of the left rail, then the Settings page opens and the gear shows the active highlight.
- [ ] When I open Settings on a fresh install, then the **Web Search** toggle is **off** and the copy explains that enabling it sends the searched query to external search engines when the model decides to search.
- [ ] When I flip the Web Search toggle, then the change **persists across an app relaunch**.
- [ ] When the global toggle is on and I start a **new** conversation, then that conversation's own Web search toggle starts **on** (inheritance at creation; the conversation owns it afterwards).

**Automatic updates**
- [ ] When I open Settings on a fresh install, then the **Automatic updates** toggle is **on** and the copy says the request goes to this project's GitHub releases and carries nothing but my version and platform.
- [ ] When I turn Automatic updates **off** and relaunch, then it is still off and `erudi-backend.log` says `Updater: automatic updates are turned off; no check will run` — with it on, the same file says `checking now, then every 4 hours` instead.

**Inference engine**

*Everything in this block needs a Windows or Linux machine with an NVIDIA GPU — on
Apple Silicon the control shows but has nothing to switch, which is itself worth
one check.*

- [ ] When I open Settings, then an **Inference engine** card offers **Automatic** and **Processor only**, and the note says Erudi restarts its engine when the setting changes.
- [ ] When I open Settings on a fresh install, then the engine is **Automatic**.
- [ ] *(NVIDIA machine)* When I switch from Automatic to **Processor only**, then the backend restarts and comes back — the app is usable again within the usual boot time, not stuck on the loader.
- [ ] *(NVIDIA machine)* After that switch, when I send a chat message, then it answers, and `backend.log` shows `Engine chosen: <CUDA_Engine>` followed by the processor swap line — i.e. the model actually runs on the CPU build.
- [ ] *(NVIDIA machine)* When I relaunch the app, then the setting is still **Processor only** and inference is still on the processor.
- [ ] *(NVIDIA machine)* When I switch back to **Automatic** and relaunch, then the GPU is used again — the choice is reversible.
- [ ] *(Apple Silicon)* When I set **Processor only** on a Mac, then nothing about inference changes — MLX still runs the models (the setting only governs the NVIDIA path).

**Application language (#385)**
- [ ] When I open Settings, then an **Application language** card offers English, Français, Español and 中文, each named in its own language.
- [ ] When I pick another language, then the **whole interface** switches immediately — every screen, the live download widget included — with no English left behind and no reload.
- [ ] When a language is active, then numbers, percentages, sizes and dates follow it (French shows `10,8 %` and `31 Go`, not `10.8 %` and `31GB`).
- [ ] When I relaunch the app, then it comes back in the language I chose (the backend value wins over the local mirror).
- [ ] When it is my **first** launch, then the language is derived from my system locale.
- [ ] When I change the language, then the **native application menu** (Help → Clear All Data…) is rebuilt in that language.
- [ ] When I use **Clear All Data**, then the app comes back in English on the next boot (settings deleted; the backend default wins).

**Diagnostics page**
- [ ] When I open Diagnostics (the bug icon in the left rail) with the backend running, then the page shows my Erudi version, operating system, inference engine, CPU/GPU, the model in memory (or none), the backend's Python version and the database state — no log-file paths are listed on screen, and there is no text preview of the report.
- [ ] When the backend has recorded warnings or errors, then they are listed newest last with their timestamp, level, source and request id, and an identical error repeated many times appears **once** with a repeat count.
- [ ] When a recorded error merely describes the environment (no network, HuggingFace rate-limited or down, a gated repository with no token, the disk or the app's own port taken) rather than Erudi being wrong, then it does **not** appear in the list and is **not** in the copied report — the same exclusion the bug icon's badge uses, applied once so the two can never disagree. A `WARNING` is never excluded this way.
- [ ] When I read the recent-errors list, then no ordinary activity line appears — only `WARNING` and above.
- [ ] When there is at least one recent error, then a one-line hint says to paste the report into the bug form's Logs field, and a single **Copy the full report** button appears alongside **Report on GitHub** and the contact link, all headed **Report a problem**.
- [ ] When nothing was recorded, then the recent-errors area shows a check mark and **No warning or error recorded.** and nothing else; there is no copy button and no line about pasting a report, but **Open log folder**, **Report on GitHub** and the contact link are still there, headed **Report a problem**.
- [ ] When I click **Copy the full report**, then the button confirms *Copied* and the clipboard holds the setup summary plus the error list as plain text — including the absolute path of both log files, even though neither is shown on screen.
- [ ] When I click **Report on GitHub**, then my browser opens this repository's bug report form with **Erudi version**, **Operating system**, **Hardware** and **Model** already filled in, and pasting into the **Logs** field gives the text I just copied. *(Dropdown prefill is unverified upstream: if **Operating system** arrives empty, that is the known gap — every other field must be filled.)*
- [ ] When I click **Open log folder**, then the file manager opens with `backend.log` selected.
- [ ] When I use the **contact page** link instead, then `erudi.app/contact` opens in my browser and the copy tells me to include everything above plus my screenshots.
- [ ] When I kill the backend (or launch with the port blocked) and open the page, then it says **the backend did not answer**, still shows my version and platform, still lists the app-side errors, and still offers **Copy the full report** (when there is something to report) and **Report on GitHub** — it does **not** go blank.
- [ ] When I click the **bug icon** in the left rail, then I land on the Diagnostics page, the icon is highlighted like the other destinations, and Settings shows no diagnostics of its own — no web page opens.
- [ ] When something the app itself gets wrong happens silently (not offline, not a service outage) — for example the backend returns a real 500 — then a small badge appears on the bug icon showing **1**; opening Diagnostics lists that error and the badge disappears, with no popup, toast or sound at any point.
- [ ] When I trigger a **Hugging Face download while offline**, then the download modal still tells me it failed, but the failure appears on **neither** the Diagnostics page's recent-errors list **nor** the copied report **nor** the bug icon's badge — an unreachable network is not counted as a defect anywhere.
- [ ] When more than nine new errors accumulate before I next open Diagnostics, then the badge reads **9+** rather than the exact count.

## Shared chrome (sidebar, connection, downloads)

- [ ] When I click the sidebar icons, then I navigate to Models (Brain), Chat (Chat), Arena (Swords), and Knowledge Base (Book); the active screen is highlighted (Chat stays highlighted while in a conversation).
- [ ] When I click the bug icon, then the Diagnostics page opens (the web contact page is offered from inside that page, not by the icon).
- [ ] When a download is in progress, then the bug icon is hidden; navigation stays enabled and the progress widget follows me across screens.
- [ ] When I navigate to an unknown route, then I am redirected to the Models screen.

## Security — localhost hardening (#89)

*Merged and shipping in this candidate — no longer skippable.*

- [ ] When the packaged app is running and I send the API a request with a **foreign Origin** (e.g. `curl -H "Origin: https://evil.example" http://127.0.0.1:27182/erudi/health -i`), then the response carries **no** `access-control-allow-origin` header (a malicious website cannot read the local API).
- [ ] When I send a request with `Origin: null` (what the packaged renderer sends), then the response grants exactly `access-control-allow-origin: null` — and the app's own screens all load their data normally (proof the packaged renderer's requests still pass).
- [ ] When I send a request with a **non-local Host header** (`curl -H "Host: attacker.example" http://127.0.0.1:27182/erudi/health -i`), then the API answers **400** (DNS-rebinding guard).
- [ ] When I inspect any API response, then **no** `access-control-allow-credentials` header is present.
- [ ] When the app runs on **macOS or Windows**, then the Chromium renderer processes run **sandboxed** (no `--no-sandbox` in the renderer process arguments — check the process list); on Linux the flag is expected (user-namespace workaround).
- [ ] When the backend logs a request with a foreign Origin or Host, then the request id correlation (`X-Request-ID`) still works for allowed requests (tracing survives the tightening).

### The inference child requires a key (all platforms)

*Every inference child is spawned with a per-process `--api-key`: `llama-server` on Windows and Linux (also with `--no-slots` and `--no-webui`), `mlx_vlm.server` on Apple Silicon. The first scenario is the one that matters most in the whole pass: if the key wiring is wrong, **every** model load fails at readiness rather than degrading quietly, so run it before anything else.*

- [ ] When I download a model and send it a message, then the answer streams normally (proof the backend authenticates itself to its own child; a broken key shows up as a readiness timeout at load, never as a bad answer).

**Windows and Linux (`llama-server`)**

- [ ] When a model is loaded and I find the child's port in `%TEMP%\erudi-backend.log` (or `/tmp/erudi-backend.log`), then an **unauthenticated** request to it is refused: `curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:<port>/v1/chat/completions -d '{"model":"x","messages":[]}'` answers **401**.
- [ ] When I request `http://127.0.0.1:<port>/slots` on that same port, then it does **not** return the prompts of in-flight requests (the endpoint is disabled; a 404 or an error is the expected outcome, never a JSON list of slots carrying prompt text).
- [ ] When I open `http://127.0.0.1:<port>/` in a browser, then llama.cpp's bundled web interface does **not** load.
- [ ] When I unload the model and load it again, then the child's key has **changed** — grep the process arguments (`ps aux | grep llama-server` on macOS/Linux, Task Manager details on Windows) before and after; the two values must differ, which is what makes a leaked key worthless.
- [ ] When I read `%TEMP%\erudi-backend.log` after a load, then the key appears **nowhere** in it.

**Apple Silicon (`mlx_vlm.server`)**

*The child is an `mp.Process` of the backend: its arguments travel by pickle, not on a command line, so `ps` shows neither the `--api-key` flag nor the value, and `ps -E` does not show the `MLX_VLM_SERVER_API_KEY` variable either (the child sets it after start). Find the port with `lsof -nP -iTCP:27300-27399 -sTCP:LISTEN` or in `$TMPDIR/erudi-backend.log` (`Spawned mlx_vlm.server child: pid=..., port=...`).*

- [ ] When a model is loaded, then an **unauthenticated** chat request to the child is refused: `curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:<port>/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"x","messages":[]}'` answers **401**, and `curl -si http://127.0.0.1:<port>/v1/chat/completions -d '{}'` shows `WWW-Authenticate: Bearer`.
- [ ] When I request `http://127.0.0.1:<port>/health` without a key, then it answers **401** too — and the app keeps chatting normally, which proves the backend's own probe presents the key.
- [ ] When I send the same chat request with a made-up key (`-H 'Authorization: Bearer nope'`), then it is still **401**.
- [ ] When I grep `$TMPDIR/erudi-backend.log` and `~/Library/Logs/erudi/backend.log` for `api-key`, `api_key`, `MLX_VLM_SERVER_API_KEY` and `Bearer`, then no line carries a key value (the key exists only in the backend's and the child's memory; that it changes on every load is pinned by the unit tests in `backend/tests/test_mlx_engine_server.py`, `TestSpawnApiKey`, and cannot be observed from outside the processes).

### The embedded database requires a password (Windows)

*On Windows the embedded PostgreSQL listens on a loopback TCP port (there are no Unix sockets), so the per-cluster password is the only thing between the database and any other program running under the user's account. On macOS and Linux the cluster opens no port at all, so the first three scenarios do not apply there; the last one does.*

- [ ] When the app is running and I find the database port in `%LOCALAPPDATA%\erudi\backend\prod\data\postgres\postmaster.pid` (fourth line), then connecting **without a password** is refused: `psql -h 127.0.0.1 -p <port> -U postgres -d erudi -c "select 1"` (any client will do) fails with a password error, and connecting with a **wrong** password fails with `password authentication failed`.
- [ ] When I connect with the password read from `%LOCALAPPDATA%\erudi\backend\prod\data\postgres\erudi_db_password` (`set PGPASSWORD=<value>` first), then the same command succeeds — proof that the app's own credential works and nothing else does.
- [ ] When I quit and relaunch the app, then it boots, migrates and chats normally (the password is set and enforced on **every** start, and a second start on the same data folder changes nothing) — and the log contains neither the password nor a connection URL carrying it.
- [ ] When I open the data folder, then `postgres\erudi_db_password` exists, contains a single random value, and `postgres\pg_hba.conf` has **no** `host … trust` line left (every `host` rule reads `scram-sha-256`; the `local` lines stay `trust`).

## Non-functional (boot, offline, persistence, updates, errors)

**Boot & errors**
- [ ] When I launch the app, then the window opens immediately on a loading screen and switches to the app once the backend is healthy, landing on Models.
- [ ] When the **backend fails to start** (port in use, crash, timeout), then the app shows a clear error with the reason (code + log path) and Retry/Quit — **not** a perpetual spinner.
- [ ] When the backend dies **after** load, then API calls fail per-screen with a visible error.
- [ ] When the interface throws an **uncaught error** while rendering, then the window shows a recoverable screen (title, explanation, **Reload**, the report block) instead of going white, the error is in `erudi-backend.log` under `renderer:uncaught`, and it is listed on the Diagnostics page.
- [ ] When something fails **repeatedly** — a poll that keeps rejecting, a render loop — then the log gains **one** entry with a repeat count, not thousands of identical lines, and the app stays responsive.

**Graphics card Erudi cannot use**

*The whole point of this block is that a machine Erudi cannot drive on the GPU gets an
honest explanation and a working way forward, instead of a crash on the first message.*

*Two of these need hardware nobody on the team has — a pre-Maxwell card (GTX 700 or
older) for the "too old" verdict, and an NVIDIA machine kept on a driver older than the
570 family for the "driver too old" one. Mark them **NOT RUN** rather than guessing, and
say which hardware was missing. The rest run on any machine.*

- [ ] *(any machine, NVIDIA GPU that works)* When I launch the app on a supported card and driver, then **no** graphics-card dialog appears — a healthy machine is never nagged. (`backend.log` shows `CUDA pre-flight ok: ...`.)
- [ ] *(NVIDIA machine with the driver only — no CUDA toolkit installed)* When I send a chat message, then it answers on the graphics card (`backend.log` shows `Engine chosen: <CUDA_Engine>` and no processor swap line) — the installer carries the CUDA runtime, so the driver is the only prerequisite. An RTX 50-series card counts double here: it is the newest generation the build carries native code for.
- [ ] *(needs a pre-Maxwell card — likely NOT RUN)* When I launch the app on a card below compute capability 5.0, then once the app has loaded a dialog says the card is too old for GPU mode, names the card and its capability, and states that no driver update changes it.
- [ ] *(needs an old driver — likely NOT RUN)* When I launch the app on a supported card with a driver older than the 570 family, then the dialog says the **driver** is too old, names the CUDA version needed and the one installed, and says updating the driver is the fix.
- [ ] When that dialog is open, then the app behind it is fully usable — it is a decision, not an error screen, and it never replaces the loading screen.
- [ ] When I click **Not now**, then the dialog closes and nothing is saved; relaunching the app shows it again.
- [ ] When I click **Switch to processor mode**, then Erudi restarts its engine, comes back, and chat works — and Settings shows **Processor only**.
- [ ] After choosing processor mode, when I relaunch, then the dialog does **not** come back (a user who already decided is not nagged).
- [ ] When the dialog is open, then a **Technical details** block shows what the graphics driver reported, and the copy button puts it on the clipboard (paste it somewhere to confirm).
- [ ] When I click **Report on GitHub**, then the issue page opens in my **system browser**, not inside the app window; same for the **Erudi website** link and the processor-version download link.
- [ ] When my interface language is French, Spanish or Chinese, then every word of that dialog is in that language — no English left behind.
- [ ] *(NVIDIA machine)* When a chat turn dies because the graphics card could not run the model, then the red error bubble appears **and** the same dialog opens with the trace — the failure is explained, not just red.
- [ ] When a chat turn fails for an ordinary reason (a corrupt model, a missing file), then only the red bubble appears — no graphics-card dialog.

**Offline & persistence**
- [ ] When I launch **offline**, then my downloaded models still list and work, the catalog shows from the bundled snapshot, and Hugging Face search reports no connection.
- [ ] When I quit and relaunch, then my conversations, knowledge bases, downloaded models, and settings all persist.
- [ ] When the catalog refreshes on restart, then my downloaded and in-progress models are never altered (only remote suggestions reconcile, with stable IDs).
- [ ] When I **force-kill** the app and relaunch, then it recovers (stale DB locks pruned) and interrupted download/KB jobs are marked failed and cleaned up.
- [ ] When I close the window on **macOS**, then the app keeps running; on **Windows/Linux**, closing the last window quits and stops the backend.
- [ ] When I use Help → **"Clear All Data"** and confirm, then the backend stops, the data directory is deleted, and the app quits.

**Shutdown & orphans (#224, #341)**
- [ ] When I **hard-kill the app process** (Activity Monitor "Force Quit" / Task Manager "End task") rather than quitting cleanly, then the **backend stops by itself within a few seconds** — it does not survive holding port 27182 (parent-death watchdog).
- [ ] After that same hard kill, then **no `postgres` and no `llama-server` / `mlx_vlm` process is left running** (check the process list). *Known open question on Windows — see #341; record exactly what survives, and grab the tail of `backend.log` right after the kill: whether it shows a shutdown marker or just stops decides the cause.*
- [ ] When I hard-kill **during a generation** (not idle), then the same holds: backend gone, children gone, and relaunching immediately works (the port is free, the cluster is not locked).
- [ ] When I relaunch after any of the above, then the app boots normally and my conversations, models and knowledge bases are intact.

**Interrupted downloads (#314, #315, #291)**
- [ ] When a download **completes** but the app is killed before the job row is finalized, then on relaunch the model is **kept and marked installed** — it is never silently deleted (a multi-GB artifact must survive; if it *is* deleted, that is a data-loss regression).
- [ ] When a download is **genuinely truncated** and the app is killed, then on relaunch the incomplete files are removed and the log states the path and the size reclaimed (deletion is never silent).
- [ ] When a download finishes and I stay on the screen, then the progress widget resolves and the UI **never stays stuck at 100%** — if finalization wedges, the poll gives up after a few minutes, the sidebar and contact icon come back, and the message says the files were saved (not "Download failed", which would push me to re-download gigabytes I already have).

**Windows regression gate (#313, #321) — run these FIRST on any Windows candidate**

*Both were release blockers on an earlier draft (then numbered 2.0.0): the packaged Windows
build deadlocked on the first chat turn and on the KB embedding download. The
cause was a blocking stdin read parking a thread inside the Windows CRT, which
froze every off-main-thread native import. If either of these hangs, stop the
pass and report — the candidate is not shippable.*

- [ ] When I download a **GGUF model** on Windows and send my **first chat turn**, then the answer streams within the usual model-load time — it never hangs indefinitely.
- [ ] When I open **Knowledge Base** on a fresh Windows install and click Download on the embedding gate, then the embedding model downloads to completion — the UI never sits on "Downloading the embedding model…" forever.
- [ ] When either of those runs, then the backend keeps answering other requests throughout (the app is not wedged as a whole).

**Tool-call gate (llama.cpp backends) — run on every candidate**

*A turn that carries tools is the whole agentic knowledge base and the whole of
web search. On an earlier draft (then numbered 2.0.0) the bundled `llama-server` exited silently the
moment a model emitted a tool call, taking both features out on Windows; a
plain turn on the same model, in the same process, was unaffected. Test the
tool path explicitly — a working chat proves nothing about it.*

- [ ] When a **tool-capable** model answers a knowledge-base question, then the turn completes: the reasoning strip shows the search call and the answer arrives — the child process does not die mid-stream and the answer is not `[ERROR_MESSAGE_SYSTEM]`.
- [ ] When **web search** is on and the model decides to search, then the same holds.
- [ ] When either turn fails, then check whether the inference child is still alive: a client-side `ReadError` / `ECONNRESET` with no error in the child's output is a crash in the inference binary, not an app bug.

**Model sizes & recommendations (#316, #319)**
- [ ] When I look at a model's **Size** before downloading it and again once installed, then the two figures **match** — a model must not appear to shrink (or grow) the moment it finishes downloading.
- [ ] When I compare a model's displayed size with the figure on its Hugging Face page, then they agree (decimal GB, the unit HF quotes).
- [ ] When I read the machine readout's recommended size window, then it reflects **both** what fits in memory **and** what my memory bandwidth can stream at a usable speed — on a 16 GB Apple Silicon machine that lands around 5–10B, not the high teens.
- [ ] When my machine is large (high-VRAM card), then the recommended window still includes the excellent **7–14B** models rather than starting above them.

**Updates & first run**
- [ ] When I run a **packaged** build and a newer release is published, then a banner shows "downloading…", then "ready — restart to install", and it installs on click or next quit.
- [ ] When a release is still a **draft**, then my installed build is **not** offered that update.
- [ ] When **Automatic updates** is off in Settings and a newer release is published, then no banner appears, nothing is downloaded and quitting installs nothing; turning the toggle back on starts a check at once — the "downloading…" banner appears without a relaunch.
- [ ] When I do a **fresh install**, then the Welcome dialog shows once, the catalog seeds instantly from the bundled snapshot (then refreshes in the background), and the machine readout renders (even if hardware profiling falls back).
- [ ] When the app quits, then the backend and its inference child processes are stopped (none left orphaned).

---

### How to record a run

Per release candidate, note: build version, OS + hardware, who ran it, date, and
any **FAIL** with a linked issue. For a FAIL, grab both log files and the `fe-…`
request id of the failing action — locations and the tracing recipe are in
[docs/logging.md](docs/logging.md). Platform coverage (which OS/GPU each artifact
was tested on) is tracked in `docs/dev/release-qa-checklist.md`. Scenarios marked
*(see #136)* are known **P2** UX defects (not release-blocking), tracked in #136;
the release-blocking defects from the bug bash (#133) are fixed (PR #135).
