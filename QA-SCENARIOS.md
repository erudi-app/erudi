# Erudi — QA Acceptance Scenarios

Walk this list on every **release candidate** (the signed *draft* build — see
[`docs/dev/release-qa-checklist.md`](docs/dev/release-qa-checklist.md) for the
process) before promoting it to `latest`.

Each line reads: **on page X, when I do Y, then Z must happen.** Tick the box if
Z happens; if it doesn't, mark it **FAIL** and open an issue. The plain language
is deliberate so anyone — not just a developer — can run the pass.

Each screen lists the **happy path** first, then **edge cases & errors** — don't
skip the edge block, that's where regressions hide. Covered: the five app
screens, the session controls and slash commands, the shared chrome, and
non-functional behavior.

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
- [ ] When I browse the **bundled catalog**, then no card links to a gated repository at all — Erudi downloads anonymously, so a gated link would 401 whoever clicked it *(gated repos are dropped at snapshot time, not flagged)*.
- [ ] When a model is **under ~4B parameters**, then its card — catalog, explore and installed alike — carries the note that tool use, knowledge-base search and multi-step reasoning are unreliable below ~4B; a 7B or unknown-size card carries no such note.
- [ ] When I view a model card, then it shows the model's **publication date** (when the publisher released it); a model with no readable date simply shows none (not a broken or wrong date).
- [ ] When a category carousel has more than 4 models, then a "See all" control expands it to a grid (and back).
- [ ] When I apply a **size filter** or **"Fits my machine"** and nothing matches, then I see "No models match these filters. Widen the size range or turn off 'Fits my machine'."
- [ ] When there are no base models at all, then the browse area shows "No base models available" (not a crash).
- [ ] When a Hugging Face search returns nothing runnable, then I see "Nothing runnable matched…" (a helpful message, not an error).
- [ ] When I am **offline** and run a Hugging Face search, then I see "No internet connection for the moment." and no request is made.
- [ ] When a download **fails**, then the widget shows the error and a "Download failed. Please try again." message.
- [ ] When, on Windows or Linux, I start a by-link download of a repository whose GGUF weights are published as **raw chunks** (`curl -X POST http://127.0.0.1:27182/erudi/llms/download/huggingface -H 'Content-Type: application/json' -d '{"link":"stevescot1979/DeepSeek-V3.2-MXFP4-GGUF"}'`), then the job fails at once with nothing transferred, and `curl http://127.0.0.1:27182/erudi/llms/downloads/<job_id>/status` carries a message saying the model is published as raw chunks that must be joined before it can run, which Erudi does not do.
- [ ] When I **cancel** an in-progress download, then it stops and the model returns to a not-downloaded state (no "Download failed" dialog); `backend.log` says `Download job N was cancelled during transfer` once at INFO, with no ERROR for the job, and Diagnostics' recent errors get no entry for it.
- [ ] When I delete an installed model and confirm, then it is removed and a success message shows; if the delete request fails, the list is left intact with an error.
- [ ] When I delete a base model that **powers KB assistants**, then the confirmation dialog lists the assistants and conversation count and warns they must be re-bound; **Delete anyway** removes the base while the assistants and conversations are kept.
- [ ] When an assistant's base model was deleted, then its card shows **"Model weights missing"**, Chat is disabled, and **Re-bind** to another installed model restores it (the card then reads "Uses the weights of <that model>").
- [ ] When a card shows the **weights missing** badge (in French too, where the label is longer), then the badge sits beside the title without covering it; a long model name is cut with an ellipsis and shows in full on hover.
- [ ] When the network drops, then the connection pill switches from "Connected" to "Offline" live.

## Chat — `/erudi/chat`

**Happy path**
- [ ] When I open Chat with at least one local model, then the first model is auto-selected and the header reads "You are chatting with <model>"; the header's **Model** button changes it.
- [ ] When I type a prompt and press Enter, then it sends; Shift+Enter inserts a newline.
- [ ] When I send a prompt, then a new conversation is created and I am taken to it, where the reply **streams token by token**.
- [ ] When I adjust values through the header's **Config** or **Effort** before the first message, then the conversation is created with them and `/config` shows them once it opens.
- [ ] When I run `/config global` on the new-conversation page, then the values are saved as the defaults for the selected model, as in a conversation.
- [ ] *(not built yet: ephemeral chats, #428 — record as not applicable)* When I turn on the **ephemeral** toggle before starting a conversation, then it opens with a visible **ephemeral badge** marking that it will not be kept.

**Per-model sampling defaults**
- [ ] When I select a model whose publisher ships sampling values, then Creativity / Diversity start at the values saved for it with `/config global`, else at **that model's** values rather than a global 0.2 / 0.95 (Qwen3 starts at 0.6 / 0.95, Qwen2.5 at 0.7 / 0.8).
- [ ] When the publisher ships a **greedy** temperature (Qwen2.5-VL ships `0.000001`), then the slider shows **0** and the model answers normally — no stream of `!` *(greedy is sent as an exact 0)*.
- [ ] When I send the **same prompt in several fresh conversations** on an Apple Silicon model at a non-zero temperature, then the answers differ *(a fresh seed per request; a short factual answer may still converge)*.
- [ ] When I switch model mid-setup, then the new model brings its own values, as `/model` does: the defaults saved for it with `/config global`, otherwise its recommended creativity and diversity; the other values I set stay.
**Output budget (no Max Tokens control)**
- [ ] When I open `/config` (in a conversation or on the new-conversation page) or an Arena panel, then there is **no Max Tokens field anywhere** — the answer budget is assigned automatically from the model's context window and the space the conversation already uses.
- [ ] When I ask for a genuinely long answer (a long structured document, a big code file), then it **completes** — nothing is cut at the old 1024-token default.
- [ ] When a conversation created **before this version** (with a stored Max Tokens value) sends a turn, then it works normally — the stored value is ignored, nothing errors.
- [ ] When I search any of the four languages for a "Max Tokens" label, then none survives anywhere in the UI.
- [ ] When the Creativity slider is dragged to the top, then it reaches **2**, not 1.
- [ ] When the selected model's publisher recommends **nothing**, then a muted one-liner under the sliders says so ("No sampling recommendation from this model's publisher; neutral defaults applied") — in `/config` (in a conversation and on the new-conversation page) and the model info modal, and **never** on a card face or in the Arena.
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
- [ ] When I send the first message of a new conversation, then the **title appears in the sidebar before the first answer starts streaming** (the title request runs first, then the chat), and the answer still arrives if the title fails.
- [ ] When I send a second message in a new conversation on a `llama-server` engine (Windows, Linux), then it starts answering **much faster than the first one did** (the first message paid the whole prefill; the second reuses the first one's prefix; the title is requested before the first answer, so it never evicts that prefix).
- [ ] When I go back to an older conversation after working in another one, then its first answer is slower (the history is re-read once, expected), and the following ones are fast again.
- [ ] *(Windows, Linux)* When I open and answer several distinct long conversations in turn, then the memory of the `llama-server` child stays flat instead of growing with each conversation (one slot, no prompt cache: `-np 1 --cache-ram 0` in the process arguments).
- [ ] When I reload the page, then the full text history re-renders from the database.

**Reasoning / thinking models**
- [ ] When a thinking model (e.g. Qwen3) generates, then its reasoning streams into a **collapsible "Reasoning" strip** above the answer — never into the answer bubble itself, and **no thinking marker** (`<think>`, `</think>`, channel tags) ever appears as literal text anywhere.
- [ ] When the turn ends, then the strip settles to a collapsed "Reasoning — N steps" summary; expanding it shows the full trace, and the trace **survives a reload**.
- [ ] When an agentic model narrates **before calling a tool** ("Let me search the documents…"), then that narration lands in the reasoning strip, not in the answer bubble — the answer zone holds only the final grounded answer.
- [ ] When a thinking model reasons **again after a tool result** (the hop where the template re-opens the thinking block), then that post-tool reasoning also lands in the strip — never appended to the final answer *(the 1.1.x field report)*.
- [ ] When a model's reasoning is **cut off before it ever closes** (small thinking model, heavy question), then the partial reasoning shows as reasoning — the answer bubble never fills with raw chain-of-thought.
- [ ] When a turn ends with reasoning but **no answer text** because the model ran out of budget (`finish` = length), then the turn shows a **normal assistant message** (not a red error) saying the generation stopped during reasoning before an answer was produced — the reasoning strip is intact and both **survive a reload**.
- [ ] When a turn ends with reasoning but no answer because the model **chose to stop**, then the same normal-message treatment applies with its own wording, and simply asking the model to continue works.
- [ ] When the turn after an empty-answer turn is sent, then it behaves normally — no role-alternation error, no silent breakage.
- [ ] When conversation **titles and the automatic summary** are produced, then they never contain reasoning fragments or thinking markers, whatever the model.

**Reasoning effort**
- [ ] When I open a new conversation, then its **reasoning effort** starts at the default saved for its model with `/config global`, or else at the Settings **Default reasoning effort** (Medium on a fresh install) — whether the model reasons or not; it is adjustable as a **5-notch graduated slider** — None, Low, Medium, High, Xhigh — through `/effort` or `/config`.
- [ ] When I save a model's defaults with `/config global` and create conversations on that model and on another one, then the first opens at the saved effort and the other at the Settings default — saving one model's defaults never moves another's.
- [ ] When I change the effort in an open conversation, then it persists immediately (survives a reload) and **takes effect on the next turn** — no Apply needed.
- [ ] When I use a model whose own template understands effort levels, then the five levels visibly change how much the model reasons — None shows no strip at all, Xhigh reasons markedly longer than Low.
- [ ] When I use a thinking model that only knows **on/off** (e.g. Qwen3), then None genuinely suppresses the reasoning, Medium is the model's natural behavior with nothing added, and High/Xhigh lengthen the reasoning best-effort through instructions.
- [ ] When I use a **non-thinking** model at Medium or above, then it reasons step by step **inside the strip** (never in the answer) and the answer stays clean; at None it behaves exactly as before this version *(prompt-instructed: a very small model may not comply — acceptance = the strip appears when it complies, and nothing leaks into the answer when it does not)*.
- [ ] When a model that **always reasons** is set to None, then the app asks it to answer directly — best-effort: record what the model actually does; residual reasoning is not a FAIL *(documented limitation)*.
- [ ] When a long conversation is summarized or a title is generated, then those internal calls always run **without** reasoning, whatever the conversation's effort level — no thinking stall, no reasoning in the title.
- [ ] When I use the Arena, then each panel follows its own model's defaults: the effort, creativity and diversity saved for that model with `/config global`, else the Settings default effort and the model's recommended creativity and diversity (no per-panel effort control in this version).

**Compaction threshold**
- [ ] When I open a new conversation, then `/config` shows its **compaction threshold** at the default saved for its model, else the **global default** (80 % on a fresh install), offered from 50 % to 90 % in steps of 5, and its help icon says that older messages are summarized when the conversation fills that share of the model's context window.
- [ ] When I change the threshold in an open conversation, then it persists immediately (survives a reload) with no Apply needed, and the next turn compacts at the new share.
- [ ] When I lower the threshold below where a long conversation already is, then the **next turn compacts** — one `Conversation compacted ... threshold=<new>%` line in `backend.log`; when I raise it, nothing happens until the conversation grows to the new share.
- [ ] When I set **50 %** on a small-window model (about 4,096 tokens), then compaction comes about every turn or two, the turn right after a compaction does not compact again, and the answers are never cut short for lack of budget.
- [ ] When I set **90 %**, then the turn just before a compaction still answers with a real budget (never cut after a few tokens), on a small-window model too.
- [ ] When I pick a threshold through **Config** on the new-conversation page before the first message, then the conversation is created with it (`/config` shows the same value once it opens); before the global default has loaded, the control is disabled; if it cannot be loaded, the control shows 80 % and the conversation gets the global default unless I pick a value.
- [ ] When I use the Arena, then there is no threshold control (Arena panels never compact).
- [ ] When the interface is in French, then the percentages read `80 %`; in English, `80%`.

**Context window**
- [ ] When I open a model's **info card**, then it shows the model's **context window**; for the **currently loaded** model it also shows the window actually **allocated** on this machine.
- [ ] When I load a model whose full window **fits** my machine, then the allocated window equals the model's window — no gratuitous reduction.
- [ ] When I load a model whose full window **cannot fit** (a 128k-context model on a 16 GB machine), then the model still loads, the allocated window shown is smaller than the model's, and `backend.log` carries the engine's own `context size reduced from X to Y` line *(llama.cpp `--fit`)*.
- [ ] When a conversation grows, then it goes **far past the old 4096-token limit** before anything is compacted (the window is the model's, not a fixed constant).
- [ ] When a single turn is **larger than the window itself** (a huge attached document), then the turn fails with an honest message naming the prompt size and the window and suggesting what to do — nothing is silently truncated, no answer is invented.
- [ ] *(Apple Silicon)* When the same oversized turn is sent on MLX, then it is **refused cleanly before generation** with the same honest treatment — never accepted into minutes of silent gibberish.
- [ ] *(Windows, Linux)* When `ERUDI_CTX` is set in the environment, then the llama.cpp window is pinned to exactly that value (dev/QA escape hatch) and the engine log says the user's value was respected. *(Apple Silicon: the MLX engine does not read it; the window is sized as usual.)*
- [ ] When the app boots **offline** on a fresh install (catalog seeded from the bundled snapshot), then model info cards already show context windows — the values shipped with the snapshot.
- [ ] *(Apple Silicon)* When the machine's memory tightens during a long conversation, then the **working window** the app plans against for the next turn shrinks to the smaller of the model's allocated window and what memory can safely hold — visible in `/context` and in `backend.log`; it never silently plans past what the GPU can hold.
- [ ] *(llama.cpp)* When a llama.cpp model's window must shrink to free memory, then it does so by **respawning the child while I am idle** — the pause reloads the model transparently (a short reload; the prompt cache is rebuilt), the next turn uses the smaller window, and no in-flight turn is ever interrupted.
- [ ] When I leave an existing conversation idle a little longer than the Settings idle delay (5 minutes on a fresh install; the model is unloaded) and send a message, then the pending answer says the model was unloaded while unused and is reloading to resume the conversation (translated in fr/es/zh, and the copy names no fixed duration) until the answer starts; switching to another model and back shows no such notice, and a **new** conversation shows the usual first-reply hint instead.
- [ ] When a model reports **no readable window** through the usual path (a mis-packaged artifact), then it is treated as a **defective artifact with an honest load error** — never loaded silently with an unknown window; the app always knows the window it plans against.

**Compaction & memory**
- [ ] When a conversation reaches its **compaction threshold** (80 % of the working window by default), then it is compacted automatically — the on-screen history is untouched, later answers still refer to earlier turns, and `backend.log` records the summarization.
- [ ] *(Apple Silicon)* When a conversation grows toward what the machine's memory can hold, then compaction fires at the threshold's share (80 % by default) of the **memory ceiling** even well below that share of the allocated window — and the model's measured peak (`Memory prediction exceeded` absent from `backend.log`) never reaches the GPU budget; the memory ceiling exists only on MLX, whose cache grows with usage; the llama.cpp engines allocate theirs in full at load.
- [ ] *(Apple Silicon)* When the machine's memory bounds the model's window (`Window bounded by memory` in `backend.log` at load) and, even after compacting to the kept budget, the next request would fill **85 % of the working window**, then — and only then — an **amber notice** appears in the conversation quoting the memory the model and the conversation take together. For a memory-bound model it shows as soon as the projected next request reaches 85 % of the memory ceiling (the `ceiling=` value of that line); only a model that alone fills the budget (`ceiling=` zero or negative, e.g. an 8 GB Mac with a large model) gets it from the first turn. Record, per model and machine, the turn where it first appears.
- [ ] *(Apple Silicon)* When the **model's own window** binds (no `Window bounded by memory` line at load), then no amber notice ever appears, even as the model runs close to the memory budget — memory is not what limits that conversation *(intended: the notice says memory limits the conversation, not how full the memory budget is in bytes; a model whose trained window fits the budget can run close to 100 % of it with no notice)*.
- [ ] *(Apple Silicon)* When a model is bounded by memory, then its **model-info card** shows the bound as the window allocated on this machine (the value of `bound=` in the `Window bounded by memory` line), not the model's trained window.
- [ ] *(Apple Silicon)* When I paste in one message more than a memory-bound model can hold, then the turn is refused before generation with the memory message — "no longer fits what your machine's memory can hold for this model", naming the request size and what memory allows — instead of the machine starting to swap, and the model is **not** reloaded (same inference child; no new `Spawned mlx_vlm.server child` line).
- [ ] *(Apple Silicon)* When a long answer is done (a prompt of several thousand tokens) and I then send a short message in **another** conversation (its first turn resets the prefix cache), then the `mlx_vlm.server` process's memory in Activity Monitor drops back close to its size right after the model loaded (record both numbers), and `logs/mlx-child-<port>.log` shows `MLX buffer cache limit set to 0 bytes`. Record the long prompt's time to first token and its tokens per second: they are as fast as before the bound.
- [ ] *(Apple Silicon)* When a memory-bound conversation goes on for several turns while the Mac is not under memory pressure, then the bound holds on every turn without any reload of the model — one `Window bounded by memory` line per load, one inference child for the whole conversation.
- [ ] *(Apple Silicon)* When the memory bound cannot be applied at load (`Memory bound of the window not applied`, `Base footprint unknown` or `Base footprint unreadable` in `backend.log`), then the model still loads and answers, keeps the window it was spawned with, and no amber memory notice appears for it.
- [ ] *(Apple Silicon)* When I hold a few turns with a model, then `memory_calibration.json` appears in the data folder with one observation per turn (`n`, `y_bytes`, `predicted_bytes`), and `backend.log` carries no `Memory prediction exceeded` line (the memory the previous turn left in the child at the start of the turn is set aside, so a short turn right after a long one does not trip it); if it does, collect the line and the file — the app's memory estimate was wrong on that machine. With `ERUDI_MEMORY_PRIOR_SCALE=0.01`, exactly one such line appears per loaded model.
- [ ] *(Apple Silicon, packaged build)* When the app runs from the packaged build, then the observations are recorded too (the measured process is the inference child, not the backend).
- [ ] When I ask about something that only existed in a **compacted-away** turn, then the answer still knows it — the summary lists user-stated facts first, by design *(summary quality is model-bound: acceptance = the fact survives on a mid-size model; a very small summarizer may still lose it)*.
- [ ] When a conversation is fresh and short, then compaction never fires.
- [ ] When I paste a **long prompt** (several thousand tokens) on Apple Silicon — including as the first long message after a few short turns, in English and in Chinese — then the answer starts without a retry: `backend.log` shows no `Output budget overshot the engine's context check` line — the output budget counts each message in real tokens, tool schemas included.
- [ ] When an answer runs to its length limit (tens of thousands of tokens) and I send another message, then the next answer still gets a real budget (not cut after a few hundred tokens): the long answer is counted at its exact token count.
- [ ] When I hold a conversation in **Chinese or Japanese** past a few turns, then compaction fires at about the threshold (80 % by default) of the window in **real** tokens (not three times too late), the answers are never cut short for lack of budget, and the next turn after a knowledge-base or web search is budgeted like the turns before it.
- [ ] *(Windows CPU/CUDA, Linux)* When I send the first turn of a conversation on a llama.cpp engine, then it answers normally (the server is now asked for its token usage) — and a long conversation compacts at about its threshold (80 % by default) of the window counted in real tokens, a little earlier than before on text that `chars / 4` under-counts (code, numbers, other scripts).
- [ ] When an **automatic** compaction would gain almost nothing (the history before the last exchange is already tiny), then it is skipped — no summary call, no `Prefix cache reset: reason=compaction` — unless the turn would otherwise overflow the window.
- [ ] When the amber memory notice about the memory bound is shown, then it says your computer's memory limits this conversation, quotes the approximate size of the model and the conversation together, and suggests starting a new conversation, sending less at once or switching to a smaller model; it is **translated** (fr/es/zh — not English-only), it never lands in the answer bubble or in what the copy button copies, and it does **not** stick to the conversation after a reload once the pressure is gone.
- [ ] When a conversation crosses the **compaction threshold** (a percentage of the working window, 50 % to 90 % in steps of 5, set through the conversation's `/config`), then compaction fires — the trigger is counted **in tokens** against the working window, not in a fixed number of messages.
- [ ] When the engine cannot report a window at all (the rare defective-artifact case), then and only then does a **message-count fallback** guard compaction — never in the normal path, where the trigger is purely token-based.
- [ ] When I hold an **agentic conversation** (tools on, a web or knowledge-base search on most turns) for ten turns or more while it stays well under the window, then it is **not compacted** — no `Conversation compacted` line in `backend.log`, however many tool rounds the turns made. When it does reach the default 80 % of a large window (e.g. ~32,700 tokens on a 40,960-token model), the one turn that compacts reads a large history in one summary call (up to ~19,200 tokens there) and its answer starts noticeably later — slowest on the CPU engine.
- [ ] When compaction keeps the most recent turns, then it keeps them by a **token budget** — as many messages as fit, plus the answer before them when that keeps the roles alternating — and a **single message larger than that budget is kept whole** — never dropped or truncated to fit.
- [ ] When a long conversation of ordinary messages compacts once, then the **next turn does not compact again** — even when its last few answers were long, and even after many short exchanges — and `backend.log` shows one `Conversation compacted` line, not one per turn. *(Documented exceptions: a current message or previous answer larger than the keep budget on its own, or a tool round kept whole, can make the next call compact again.)*
- [ ] When compaction runs on a model with a strict chat template (Gemma, Mistral), then the **following turns keep working** — no "roles must alternate" error, ever — and the question being answered is never lost, even mid-turn after a large tool result.
- [ ] When I press **Stop** (or close the window) while a strict-template model (Gemma, Mistral) is answering — and once before its first word — then send another message in the same conversation, then the conversation keeps working: no "Conversation roles must alternate" error, and the model's history holds the interrupted answer as it was shown (or `[The answer was interrupted.]` when nothing had been written yet).
- [ ] When a conversation has been compacted several times, then a fact stated before the **first** compaction is still known — the previous summary is always part of what the next compaction summarizes *(summary quality is model-bound: acceptance on a mid-size model)*.
- [ ] When the summary cannot be written because the inference child goes away (e.g. it is killed mid-compaction), then the turn fails honestly with the history intact and the **next turn compacts normally** — no earlier memory is lost.
- [ ] *(Apple Silicon)* When I **switch to another conversation** and send a message, then its first answer starts after a longer pause (the history is re-read) and `backend.log` shows `Prefix cache reset: reason=conversation_change`; the following turns in that conversation start quickly again.
- [ ] *(Apple Silicon)* When I alternate between two conversations, then each switch costs that re-read (one reset per switch) — and staying in one conversation never triggers a reset.
- [ ] *(Apple Silicon)* When a conversation compacts, then `backend.log` shows `Prefix cache reset: reason=compaction` right after `Conversation compacted`, and the answer that follows is coherent.
- [ ] *(Apple Silicon)* When I press **Stop** while the model is still reading a long prompt (before the first word), then switch conversation and send, then the new turn answers normally — in the child log, `Generation cancelled` appears **before** the next request starts, and nothing in the log reports a corrupted cache.
- [ ] When compaction is running, then the pending answer shows a typing indicator and "Summarizing earlier messages to make room…"; once the summary is written the text goes (the separator appears) and the typing indicator alone stays until the model starts answering (or thinking, or calling a tool) — never in the answer text, never after a reload of the page.
- [ ] When the turn that compacted has answered, then a **thin separator above that answer** says "N earlier messages were summarized to keep the conversation within the model's context" (N the user messages and answers summarized), and **Show summary** reveals the summary as plain text, **Hide summary** hides it again; the separator and its summary are still there after reopening the conversation or restarting the app, and `backend.log` shows one `Conversation compacted` line for it.
- [ ] When a conversation compacts on a model whose window the computer's **memory** bounds (Apple Silicon: `Window bounded by memory` in `backend.log` at load; CPU/CUDA engines: `Window sized at spawn: ... memory_bound=True`), then the separator ends with "…to fit this computer's memory" instead.
- [ ] When the summary cannot be written (the summary call keeps being rejected, e.g. a context overflow), then the separator says "Earlier messages could not be summarized and were dropped from what the model sees." and, when an earlier summary exists, offers **Show the summary the model keeps**; the conversation keeps answering.
- [ ] When a turn compacts and then fails (red error bubble), then the separator still shows above that error, and after a reload of the page too.
- [ ] When I switch the interface to French, Spanish or Chinese, then the summarizing state and the separator (count, reason, toggle) are translated.
- [ ] *(Apple Silicon)* When I open a memory-heavy app in the middle of a long conversation (macOS memory pressure turns yellow or red in Activity Monitor), then at the end of the next answer an **amber notice** says my computer's memory is under pressure and the conversation will be summarized sooner to keep things fluid, `backend.log` shows one `Memory brake: ... action=shrink, window=A->B` line, and the next turns compact earlier (an earlier `Conversation compacted`) and keep answering — no error bubble.
- [ ] *(Apple Silicon)* When I walk away from a long conversation under **critical** memory pressure, then within about a minute the model's memory goes down in Activity Monitor (the `mlx_vlm.server` process) and `backend.log` shows `Memory brake: point=idle, level=4, action=release` with the excess before and after; under **warning** pressure the same happens after about 3 to 4 minutes away — a pause to read an answer (a minute or so) keeps the conversation's cache, and the next answer starts as quickly as before.
- [ ] *(Apple Silicon)* When critical pressure lasts while I am away, then after two idle samples (about two minutes) the model is unloaded — one `Memory brake: ... action=unload` WARNING in `backend.log` — and my next message shows "Memory was freed on your computer; reloading the model to resume the conversation…" in the pending answer until the model starts answering; the answer comes with the reduced window (`brake_cap=` in the `Window bounded by memory` line of the reload), translated in fr/es/zh, and the reload state never stays in the answer or comes back after a reload of the page.
- [ ] *(Apple Silicon)* When the memory brake has unloaded the model and I send the **first message of a NEW conversation**, then the usual first-reply hint ("First response may take a bit longer while loading the model into memory…") shows while the title request reloads the model, and the answer follows; the memory-freed reload notice is what an existing conversation shows when its next message reloads the model.
- [ ] *(Apple Silicon)* When other apps keep the Mac under memory pressure for a long time (a permanent warning or critical), then the app never loops: at most one unload per pressure episode (none again right after a reload of the model), at most two `action=shrink` lines per episode, and after that nothing but the paced idle samples — the conversation keeps working.
- [ ] *(Apple Silicon)* When the pressure is gone for a few minutes, then the window grows back step by step (`action=grow` lines: three calm samples each, one a minute while idle and one per turn while chatting), first never back to the size that hurt, then fully once the episode is long forgotten (ten calm samples). A fully idle user sees at most one idle `action=grow` before the idle unload (five minutes at the default idle delay), so keep chatting through the calm period to see the full recovery.
- [ ] *(Apple Silicon)* When a tiny conversation runs while other apps put the Mac under memory pressure, then the amber notice says the pressure comes from other apps and this conversation uses little of it, and no `action=shrink` line appears.
- [ ] *(Apple Silicon)* When the memory pressure sensor cannot be read, then the brake is off: one `Memory pressure level unreadable ... the memory brake is off` WARNING per launch, and conversations work as usual.
- [ ] *(Apple Silicon)* When memory pressure rises while an answer is streaming, then the answer is never cut or slowed by the brake: every `Memory brake` line is at `point=turn_start`, `point=turn_end` or `point=idle`.
- [ ] *(Apple Silicon)* When I quit the app while the model is loaded and idle, then the app closes cleanly within seconds: `Stopped memory pressure monitor` precedes the inference child's termination in `backend.log`, and no `mlx_vlm.server` process is left behind.
- [ ] *(Apple Silicon)* When a model sits **near the machine's usable GPU budget** (its weights close to the working-set limit), then the memory ceiling reflects that honestly — the accounting is against the GPU's usable working set (0.9 x the recommended working set), not total system RAM — so the model is bounded by memory at load (`Window bounded by memory`, down to a 4096-token floor), the working window shrinks (visible in `/context`), the amber notice follows the working window, and a model that is genuinely tight is caught before it fails.
- [ ] *(Windows CPU, Linux CPU — FIRST, before the scenarios below)* When the shipped build's `llama-server` loads a small GGUF with `-lv 5 -np 1 --cache-ram 0 -c 8192`, then again with `-c 32768` (on the packaged binary), then its `KV buffer size` is exactly `n_ctx` × the model's KV bytes per token, its `compute buffer size` grows by 1 KiB per token between the two (24 MiB over the 24,576 extra tokens), `flash_attn` resolves to enabled, and the `CPU_REPACK` line says whether this build repacks Erudi's quants (Q4_K_M, Q8_0) — record all four lines; the Windows build is the OpenBLAS one. On a vision model (a GGUF with its `mmproj`), record the same lines once, to size the projector's compute buffer.
- [ ] *(CPU engine)* When a model loads, then `backend.log` shows `Window sized at spawn: window=W, trained=T, memory_bound=...` with `W = min(T, what 0.75 x RAM affords)` and the `allocated context window: W tokens` line agrees; loading the same model again with very different free RAM (a heavy app open, then closed) gives the same `W` — the window no longer depends on the free memory of the moment (on a pre-3.5b build, the Windows window did).
- [ ] *(CPU engine)* When a model loads, then no `llama-server allocated more than the window budget predicted` WARNING appears; if one does, collect it with the `-lv 5` lines above.
- [ ] *(CPU engine)* When I open a memory-heavy app in the middle of a long conversation (Windows: commit charge above 85 % in Task Manager, Performance → Memory → Committed; Linux: `cat /proc/pressure/memory` shows `some avg10` above 10), then `backend.log` shows `Memory brake: ... action=shrink, ... cap=...` at the end of the next answer (no notice yet: nothing is summarized earlier until the cap applies); once I have been away two to three minutes the model restarts with the smaller window (`action=respawn, window=A->B`, B a multiple of 256, and the `allocated context window` line of the restart says B), and my next message shows the reload state ("Memory was freed on your computer; reloading the model…") in the pending answer, then answers, and the amber notice says my computer's memory is under pressure and the conversation will be summarized sooner. Away longer than three minutes before the first idle sample past two, no restart happens: the idle unload (five minutes at the default idle delay) comes first, and the next load carries the smaller window (`brake_cap=`). Every later turn runs without any further restart while the pressure stays as it is.
- [ ] *(CPU engine)* When memory goes **critical** in the middle of a session (Windows: committed above 93 %), then at the next turn boundary the model restarts with a smaller window (`point=turn_start` or `point=turn_end, ..., action=shrink+respawn`); a restart at the start of my turn shows the reload state in that same turn, before the answer.
- [ ] *(CPU engine, 14B model)* When the brake restarts a 14B model under real critical pressure, then time the restart (`action=shrink+respawn` follows within 30 s); past 30 s it degrades into an unload — one `action=respawn-failed (...)` WARNING, the model reloads at my next message with the reloading state and the reduced window.
- [ ] *(CPU engine)* When critical pressure lasts while I am away, then after the restart and two idle samples the model is unloaded (`action=unload` WARNING) and my next message reloads it with the reduced window (`brake_cap=` in the `Window sized at spawn` line).
- [ ] *(CPU engine)* When other apps put the computer under pressure while a small model runs a short conversation, then the amber notice says the pressure comes from other apps, and there is no `action=shrink` and no `action=respawn` line — no restart loop, however long the pressure lasts.
- [ ] *(CPU engine)* When the pressure has been gone a few minutes, then the window grows back **at the next load** of the model (a swap away and back, an idle unload's reload, a brake reload): `Window sized at spawn` carries a larger `brake_cap=` (or none), never back to the size that hurt while it is remembered (30 minutes); the brake never restarts a model just to grow it.
- [ ] *(CPU engine, Gemma 3)* When a Gemma 3 model loads, then it gets a conservative window (smaller than its trained 131k on a 16 GB machine) and no amber notice blames memory for that window.
- [ ] *(CPU engine)* With `ERUDI_CTX=8192` set, when memory pressure rises, then the window is never resized: `-c 8192` at every load, no `action=shrink` or `action=respawn` line; an unload under lasting critical pressure can still happen.
- [ ] *(CPU engine)* When the pressure sensor cannot be read (Linux without PSI and without `/proc/vmstat`, a container; Windows with PDH denied by policy), then `backend.log` carries one WARNING per launch (`Memory pressure unreadable (...); the memory brake is off`, or `Paging counter unavailable (PDH, ...)` when only the paging counter is missing), and the app works as usual.
- [ ] *(CPU engine, QA seam)* With `ERUDI_FORCE_MEMORY_PRESSURE=warning` on a calm machine, when I send a turn of a long conversation and then leave the app idle for two minutes, then I see the `pressure` notice, one `action=shrink` line, then one `action=respawn` line at the first idle sample past two minutes, and nothing more; with `critical`, the next turn boundary restarts the model and two idle samples later it is unloaded.
- [ ] *(Linux)* Before the scenarios above, record `cat /proc/pressure/memory` as a normal user, `uname -r` (4.20 or later), `grep CONFIG_PSI /boot/config-$(uname -r)`, and `cat /sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/memory.max` (`max` on a desktop: the system PSI is used); under `stress-ng --vm 4 --vm-bytes 85% -t 180`, record `watch -n1 cat /proc/pressure/memory` and `grep -E "pswpout|pgmajfault" /proc/vmstat`, and for the loaded child `grep -E "Anonymous|Swap" /proc/<pid>/smaps_rollup` — the brake's thresholds are calibrated from these.
- [ ] *(Windows)* Before the scenarios above, in a non-admin PowerShell record `typeperf "\Memory\Pages Output/sec" -sc 5` (it must answer without elevation, on a non-English Windows too), `Get-CimInstance Win32_OperatingSystem | Select TotalVirtualMemorySize,FreeVirtualMemory`, `Get-CimInstance Win32_PageFileSetting` (no page file: the commit tiers run hot at rest — check the 85 / 93 % tiers stay calm then), and `(Get-Process -Id <llama-server pid>).PrivateMemorySize64` before and after a working-set trim (it must not drop).

**Knowledge-Base / agentic behavior**
- [ ] When the model has a KB attached and is **tool-capable (agentic)**, then on a document question the model **calls the KB search tool itself** before answering, and the answer references the source.
- [ ] When an agentic, KB-attached model gets a **chit-chat / meta turn** (not about the documents), then it answers directly **without** searching the KB.
- [ ] When the model has a KB attached and is **not tool-capable (systematic)**, then relevant document excerpts are **injected up-front** every turn and the answer is grounded in them.
- [ ] When a small / uncooperative model is KB-attached, then the answer should still reference the source *(prompt-instructed only — no clickable source UI; acceptance = it mentions the doc when it complies)*.
- [ ] When KB retrieval **fails** (broken/empty vector store), then the turn **degrades to a no-context answer** instead of erroring.
- [ ] When I ask about something the documents **do not cover** (e.g. an undocumented product variant), then the agentic model searches, finds nothing relevant, and **says the documents don't cover it** — it never invents a value and never substitutes a nearby fact (e.g. another model's price).
- [ ] When a follow-up returns to a **topic searched earlier in the conversation**, then the model runs a **fresh search** rather than answering from its memory of old excerpts (old tool results are placeholder-stripped from context) — it must not claim "not in the documents" without having just searched.
- [ ] When ONE question spans **two subjects living in two different documents** ("what is the drone's payload, and how many remote days are allowed?"), then the answer grounds **both** facts — neither half is dropped or answered from world knowledge *(multi-subject coverage)*.
- [ ] When I inspect any agentic answer, then **no raw tool markup** (`<tool_call>`, JSON arguments, function-call syntax) appears in the answer bubble or in the persisted history; the search call and its excerpts appear only inside the reasoning strip.

**Web search**
- [ ] When I create a new conversation, then its **Web search** switch (in `/config`) starts at the value saved for its model with `/config global`, else at the global Settings-page default at creation time.
- [ ] When I flip the Web search toggle in an open conversation, then it persists immediately (survives a reload) and **takes effect on the next turn** — no Apply needed.
- [ ] When web search is ON with a tool-capable model and I ask a question needing a **current external fact**, then the reasoning strip shows a `web_search` call with its results, and the answer **cites source URLs** from those results.
- [ ] When web search is ON and I ask something the model already knows ("capital of France"), then it answers **directly with zero web calls**.
- [ ] When web search is ON but the machine is **offline**, then the turn completes with the model relaying the honest tool text ("Error during Web Search: no internet connection") — no hang, no invented answer.
- [ ] When the conversation's model is **not verified tool-capable**, then the Web search switch in `/config` is **disabled** (greyed, cannot be flipped) and its help explains why — one wording for a model that **can't** use tools, another for one whose tool support **hasn't been verified** yet. The model never receives the web tool either way.
- [ ] When the Settings page describes the **Web Search** control, then it reads as the starting value for **new** conversations (each conversation keeps its own switch in `/config`) — not as a global switch.
- [ ] When I change the **global** web-search default in Settings, then existing conversations keep their own switch unchanged; only conversations created afterwards on models without saved defaults inherit the new default.

**Multimodal / multi-turn**
- [ ] When I send an image on a vision model, then it is used for that turn and **carried forward** on later turns so a follow-up ("what colour is his hair?") works without re-attaching; as soon as I send a **newer** image, every older one collapses to an `[image]` marker in the model's context (at most one turn's images ever reach the model), while the display keeps all images.
- [ ] When I reload a conversation with **file-attached** images, then the thumbnails re-render (for images still present on disk).
- [ ] When I reload a conversation whose image was **pasted from the clipboard**, then it shows an "image attachment" placeholder, not the image *(clipboard images are not kept, so they cannot be restored)*.
- [ ] When an attached image's original file was **moved/deleted**, then that image quietly shows nothing on reload (no broken-image artifact).

**Attached documents**
- [ ] When I drag a `.pdf`, `.docx`, `.xlsx`, `.csv`, `.txt` or `.md` file onto the composer (or pick it with the attach button), then a chip with its name appears, and asking "what does it say?" produces an answer grounded in that file's content.
- [ ] When the model has **no vision support**, then documents still attach and are still read (only images are refused).
- [ ] When I drop a **folder**, then its supported files - those directly inside it and those one level below - are attached, and the answer covers them.
- [ ] When I attach a file the app cannot read (a `.zip`, a **scanned PDF**, an image), then the turn still completes and the answer opens with an italic notice naming the file and why it could not be read.
- [ ] When I attach a document far larger than the per-question budget, then the answer opens with a notice saying the content was shortened, and the model still answers on what it received.
- [ ] When I reload the conversation, then each attached document shows as a **named chip** on its turn, and no `[file_path:...]` marker ever appears in the readable text or in what the copy button copies.
- [ ] When I remove a chip before sending, then that file is not attached.

**Edge cases & errors**
- [ ] When I hover a message, then copy and star controls appear; a starred message stays starred after reload and is fed back as context on later turns.
- [ ] When I delete the conversation I'm viewing, then it's removed and I'm redirected to `/erudi/chat`; deleting a different one keeps me in place.
- [ ] *(not built yet: ephemeral chats, #428 — record as not applicable)* When a conversation is **ephemeral** and I leave it (navigate away, open another conversation, or quit), then it is **purged from the database and the checkpointer** on the way out — reopening its URL finds nothing, and nothing about it survives a relaunch.
- [ ] When I quit and relaunch and reopen the conversation, then its full history is intact.
- [ ] When generation **fails** or the connection **drops** mid-reply, then a red error message shows, its headline in the interface language, and any partial reply is kept.
- [ ] When a **long conversation** on a **big model / small machine** reaches the point where the model needs more than two minutes just to read the history before writing its first word (e.g. a 9B model on a 16 GB laptop, a dozen long turns in), then the answer still arrives — the turn is no longer cut at exactly two minutes of silence. If it genuinely never starts, the red turn says the model did not start answering for a prompt of that size and suggests sending less, instead of a generic error.
- [ ] When an answer contains a **markdown image pointing at a web address** (ask the model to reply with exactly `![logo](https://example.com/logo.png)`), then no picture is fetched or shown — at most a broken-image placeholder — because the window loads no remote images, so no site learns my address from an answer on screen.
- [ ] When the conversation's assigned model was **deleted**, then the conversation survives with no model assigned: sending is **blocked**, the header shows a red "Please select a model" attention state, and **explicitly picking** an installed model via the header's **Model** button unblocks sending (no auto-fallback); opening `/context` or running `/compact` on it says to pick a model first and never assigns one.
- [ ] When a model writes a **very long answer** (tens of thousands of characters, e.g. ask for a long detailed document), then after the answer ends and the conversation is **reloaded**, the answer is still there in full — what was shown is what was stored.
- [ ] When I paste a **very long message** (several tens of thousands of characters), then it is accepted and handled by the honest overflow path — no ugly low-level `422` error; if it exceeds the window, the same honest "prompt size vs window" message applies.
- [ ] When a turn fails with an unexpected error, then the red bubble's headline is in my interface language and a discreet **Details** control reveals the exact cause in one line, as the backend wrote it (not translated), with a request id — never a stack trace, and a file path shows as its file name only.
- [ ] When I reload the conversation or switch the interface language, then the failed turn keeps its translated headline and its **Details**.
- [ ] When I open a conversation whose failed turn dates from before this version, then its generic headline shows in my interface language, without **Details**.
- [ ] *(covered by the error-detail sanitizer's unit tests; no packaged recipe — record the test run)* When a failure's cause contains a web address with a user name and password, a `--api-key` value or a bearer token, then **Details** shows it masked.
- [ ] When I copy the request id from **Details** and open Diagnostics, then the matching entry carries the same request id with the full technical record.
- [ ] When a turn ends with a curated message (the prompt does not fit the window or this computer's memory, the model did not start answering in time, the tool-loop limit), then it reads as today, without **Details**, and it is **not** counted as a bug (no bug badge, no Diagnostics error for it).
- [ ] *(Apple Silicon)* When a model that is too tight for the machine hits a **GPU out-of-memory** during a tool-call turn, then the turn ends with an **honest message** that the model is too large for this machine's memory — never a silent crash, a red generic error, or minutes of gibberish.
- [ ] When I start conversations in **different languages**, then each **title** is in the language of my message — an English prompt gets an English title, not a French one (no language bias in the title generator).

**Long-conversation rendering (renderer)**
- [ ] *(not built yet: renderer virtualization, #618 — record as not applicable)* When I scroll a **very long conversation** (hundreds of turns), then scrolling stays smooth and renderer memory does not balloon — the thread is virtualized.
- [ ] *(not built yet: #618 — record as not applicable)* When a reply is **streaming**, then only the streaming bubble updates per token — the rest of the thread does not re-render, and the stream stays fluid even on a long conversation.
- [ ] *(not built yet: #618 — record as not applicable)* When I use browser **find (Cmd/Ctrl+F)**, **select-all**, or **copy** across a long conversation, then it works across the whole conversation, not only the on-screen part.
- [ ] When a new turn arrives at the bottom, then the view **auto-follows** to it as before, and I can scroll up without being yanked back down mid-read.
- [ ] When I have scrolled up into a long history and **send** a message, then the view jumps to the bottom, shows my message and follows the answer as it streams.
- [ ] *(not built yet: #618 — record as not applicable)* When I open the **catalog** with hundreds of model cards, then it scrolls smoothly (the card list is virtualized too).
- [ ] *(not built yet: #618 — record as not applicable)* When I navigate a long conversation with a **screen reader**, then messages remain reachable (a11y preserved through virtualization).
- [ ] *(Windows/Linux, llama.cpp)* When I **return to an older conversation** and send a turn, then the first token comes back quickly — the host-side prompt cache is kept (`--cache-ram`), not a cold re-read of the whole history.

## Session controls & slash commands — the chat composer

**Header**
- [ ] When I open a conversation, then the header reads a static **"You are chatting with <model name>"** — there is no model dropdown in the header any more.
- [ ] When I click the **settings icon** in the conversation header, then a panel opens with **one button per command** — Model, Config, Effort, Compact, Context — each opening that command's popup or panel, exactly as typing the command does.
- [ ] When the conversation's model was deleted, then the header shows the red "Please select a model" attention state and the **Model** button in its panel still works although the composer is blocked; picking a model there unblocks sending.
- [ ] When I open the **new-conversation page**, then it carries the same header and settings icon: the Model, Config and Effort buttons set the values the conversation will be created with, and Compact and Context are not offered (there is no conversation yet). The old inline settings block of that page is gone.
- [ ] When I use the **Arena**, then its header is unchanged by this version (its own model pickers and sliders).
- [ ] When I pick a model in an Arena panel, then that panel's creativity and diversity sliders start at the values saved for that model with `/config global`, else at its recommended values.

**Typing a command**
- [ ] When I type **`/`** at the start of the composer, then a styled **drop-up** lists the available commands (`/config`, `/context`, `/compact`, `/effort`, `/model`) with a one-line description each.
- [ ] When the drop-up is showing, then **Arrow Up/Down** move the highlight, **Enter** confirms the highlighted command (the first one when I have not moved), **Tab** completes the name in the composer, and **Escape** closes the drop-up and leaves my text alone.
- [ ] When I have typed a complete command (`/config global`) and press **Enter**, then exactly that command runs — never the highlighted suggestion; typing a space after a complete name closes the drop-up.
- [ ] When I type with an input method (Japanese, Chinese) and press Enter to confirm a character, then no command runs and nothing is sent.
- [ ] When I run a command while files are attached to the composer, then the attachments stay for my next message.
- [ ] When I type `/con`, then the drop-up narrows to the matching commands (`/config`, `/context`).
- [ ] When the message is **exactly** `/config`, `/config global`, `/context`, `/compact`, `/effort` or `/model` and I send it, then the command runs and **nothing is sent to the model**; the composer is cleared.
- [ ] When the message is anything else — a command inside a sentence ("please /compact this for me"), an unknown name (`/foo`), or an argument a command does not take (`/compact now`, `/config everything`) — then it is a **normal message** sent to the model.
- [ ] When I read the command names in another interface language, then the **names stay in English** (fixed identifiers); the drop-up descriptions, the popups and their help texts are translated.
- [ ] When any command popup opens, then it carries a **subtitle** and a **`?` help icon** next to each field; the help shows on hover **and** on keyboard focus.
- [ ] When a popup is open, then the keyboard stays inside it (Tab cycles through its fields), **Escape** closes it, and the focus returns to where it was before.

**/config**
- [ ] When I run **/config**, then one popup shows every adjustable value of the current conversation: reasoning effort as a **5-notch graduated slider** (None → Low → Medium → High → Xhigh), web search as a switch, creativity and diversity as sliders with their real bounds, the compaction threshold as a select (50 % to 90 % in steps of 5), and the custom instructions (the existing instructions editor). There is **no** "defaults for new conversations" section and **no Apply button**.
- [ ] When the model has no publisher recommendation for creativity and diversity, then the popup says the neutral values are in use (the note the old header panel showed); when web search cannot be enabled for this model, then the switch is disabled and its help says why.
- [ ] When I change the effort, the web search switch or the threshold, then the change is saved at once; when I move the creativity or diversity slider, it is saved as soon as I stop moving it. Sending a message right after any change (even within a second) uses the new value on that turn: the turn's `Turn settings: ...` line in `backend.log` shows it (effort, creativity, diversity, threshold, web search, and the length of the custom instructions).
- [ ] When a change cannot be saved (the backend is unreachable), then the control returns to its previous value and a short error says the change was not saved.
- [ ] When I close **/config** and open the header settings panel's Config button or type `/config` again, then both show the same values (one source, never two copies that disagree).
- [ ] When I run **/config global**, then the current values — effort, web search, creativity, diversity, threshold and custom instructions — are saved as the defaults **for the current model** (the popup says so and names the model), and a confirmation says they were saved. A **different** model's defaults are left untouched. Empty custom instructions are not saved as a default: they clear instructions saved earlier for this model, and never erase another conversation's instructions.
- [ ] When I save defaults for a knowledge-base assistant, then they belong to that assistant, not to its base model, and they survive a **Re-bind** of the assistant to another model.
- [ ] When I create a new conversation on a model whose defaults I saved, then it opens with exactly those values; on a model without saved defaults, it opens with the Settings defaults (effort, web search, threshold), the model's recommended creativity and diversity, and empty custom instructions.
- [ ] When I delete a model and download it again, then it starts without saved defaults (a fresh download is a new model for the app).

**/effort**
- [ ] When I run **/effort**, then a popup opens on the **5-notch slider** at the conversation's current effort; changing it is saved at once (survives a reload) and takes effect on the next turn.

**/model**
- [ ] When I run **/model**, then a **model-selection popup** lists my installed models (a model whose weights are missing is not offered); picking one switches the conversation's model and the header line updates to the new model's name.
- [ ] When I switch model, then the new model brings its own values: the defaults saved for it with `/config global`, and otherwise its recommended creativity and diversity; every other value of the conversation (effort, web search, threshold, custom instructions — unless the new model has saved ones) stays as it was.

**/compact**
- [ ] When I run **/compact** on a conversation with several exchanges, then a **confirmation popup** explains what compaction does; confirming summarizes the earlier messages now (the same summary as the automatic one): the pending state reads "Summarizing earlier messages to make room…", then a thin separator appears **below the last answer** saying "You summarized N earlier messages.", with **Show summary**; while it runs, a status line at the end of the thread says it is summarizing, and sending, the header buttons and commands are disabled. The on-screen history is untouched, the next answers still know the earlier facts, and `backend.log` shows one `Conversation compacted` line.
- [ ] When I send the next message after a manual compaction, then the model answers normally — on a strict-template model (Gemma, Mistral) too: no "roles must alternate" error.
- [ ] When I dismiss the confirmation, then nothing is compacted.
- [ ] When the conversation has a single exchange, or I just ran **/compact** and nothing was said since, then **/compact** says there is nothing to compact yet, without loading the model, and nothing is stored. With two exchanges or more (a summary counts as history), it compacts everything before the previous answer.
- [ ] When the summary cannot be written (the model fails or is stopped), then **/compact** says so and nothing changes: no separator, and the model still sees the whole history.
- [ ] When the last turn failed (red bubble, for example the model did not start answering in time on a long history), then **/compact** works and its separator shows below that turn; the next message is answered on the smaller history.
- [ ] When an answer is being generated (ask for a long answer, then run **/compact** from the header's Compact button before it ends), then **/compact** says "The model is busy; try again in a moment."; nothing is compacted and the answer is not interrupted.
- [ ] When I am on the new-conversation page, then **/compact** and **/context** are not offered in the drop-up, and typing either exactly sends nothing: a short notice says they are available once the conversation exists.
- [ ] When the model is not loaded, then **/compact** loads it first (the pending state shows) and then compacts.
- [ ] When I reopen the conversation or restart the app, then the manual separator is still below the same answer, and its summary still opens.

**/context**
- [ ] When I run **/context**, then a panel renders the conversation's context usage as a **grid of small blocks** colored by category, with a legend giving each category's token count and percentage — the free space as empty blocks and the auto-compaction reserve marked distinctly.
- [ ] When I read the legend, then the categories are **system prompt, tools, messages, summary, free space, auto-compaction reserve**, each with tokens and percent; the summary appears only when the conversation was compacted.
- [ ] When the model is loaded and the machine reduced its window, then the panel shows the **allocated** window with a short note that the model supports more but this computer's memory bounds it — never a smaller-than-the-card number left unexplained.
- [ ] When the model is not loaded, then the panel shows the model's trained window and says the window this computer allows is known once the model is loaded; opening **/context** never loads the model.
- [ ] When the conversation is with a knowledge-base assistant that searches on every question, then the panel counts its knowledge-base instructions and says the counts leave out the excerpts a question retrieves; with an assistant that searches through a tool, the excerpts of earlier searches count only as the short placeholder that replaces them in each request.
- [ ] When I read the counts, then they are presented as **estimates** and the percentages are of the window the conversation works with (the allocated window, or less when memory bounds it; the trained window when the model is not loaded); the blocks add up to that window: what the next request would use, the free space left below the compaction threshold, and the auto-compaction reserve above it.
- [ ] When a conversation is past its threshold, then free space shows as zero, the reserve shrinks by what the conversation already uses beyond the threshold, and the panel says the next turn will be summarized first.
- [ ] When I reload the page, then the **/context** panel is gone (it is a snapshot, never stored in the conversation).

**Unloading the model**
- [ ] When a model is loaded, then the header settings panel (in a conversation and on the new-conversation page) shows a small red **Unload <model name>** button naming the loaded model, even when another conversation loaded it; with no model loaded it is not shown.
- [ ] When I press **Unload <model name>**, then the model leaves memory (its inference process ends, `backend.log` shows the cleanup) and the button disappears; my next message shows "You unloaded the model; reloading it…" (translated in fr/es/zh, distinct from the idle-unload notice), then answers; a brand-new conversation shows the usual first-reply hint instead.
- [ ] When an answer is being generated (ask for a long answer, open the panel and press **Unload** before it ends), then it is refused with "The model is busy; try again in a moment."; nothing is interrupted.
- [ ] When I unload the model and send a message, then the model reloads with the same window it would get after an idle unload (on Windows and Linux a window the memory brake reduced recently stays reduced, as after an idle unload).

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
- [ ] When I attach a **document** in Arena, then it attaches whatever the panels' models are (documents need no vision support), every panel receives its text for that turn, and the chips show on the sent question.
- [ ] When a generation is running, then settings/model pickers are disabled; there is **no stop button** — the run must finish.
- [ ] When I submit an **empty** prompt, then it does not send.

## Knowledge Base / Create Assistant — `/erudi/attach_knowledge_base`

**Happy path**
- [ ] When I open the screen for the **first time** (embedding model not yet installed), then a dialog offers to download the embedding model (multilingual-e5-small) once; accepting downloads it and confirms "the Knowledge Base is ready to use"; declining ("Not now") returns to the Models page and the offer returns on the next visit.
- [ ] When I open the screen, then I see the KB description, a chat-capabilities rating (my machine's inference label/score), the local-model library, a name field, and a drag-and-drop area.
- [ ] When I select a base model, type a name, add supported files (`.pdf`/`.txt`/`.docx`/`.xlsx`/`.csv`/`.md`), and click "Create Assistant" + confirm, then a spinner polls progress.
- [ ] When ingestion completes, then "Data attached to your Assistant successfully!" shows and the form resets.

**Edge cases & errors**
- [ ] When I leave the assistant name **empty**, or pick no model, or add no files, then "Please fill in all required fields" shows and nothing is sent.
- [ ] When I add a **supported document** beyond `.pdf`/`.txt` (`.docx`, `.xlsx`, `.csv`, `.md`), then it is accepted; an **unsupported** file (e.g. `.png`, `.zip`) isn't offered by the picker and a dropped one is ignored.
- [ ] When I add the **same file twice**, then it is de-duplicated.
- [ ] When I submit a **scanned / image-only PDF** alongside readable files, then it is accepted as *pending vision* (no searchable content yet) and the job completes for the readable ones; a **pending-vision-only** upload fails with "no searchable content" (no OCR tier yet).
- [ ] When I submit an **empty / no-text** file (and nothing else indexes), then the job **fails** with a "no searchable content" message and the document is flagged *empty* — never a false success.
- [ ] When **every** submitted file is unreadable/unsupported, then the job fails with a clear error and the half-built assistant is auto-cleaned up.
- [ ] When **some** files fail but at least one ingests, then the job still completes for the good ones.
- [ ] When ingestion **fails** (network/HTTP), then an error dialog shows the reason.
- [ ] When the selected base model **already has a KB**, then submitting **updates** the existing KB with the new files instead of creating a new assistant.

**Embeddings run without the torch stack (footprint)**
- [ ] *(not built yet: embeddings outside the backend, #594 — record as not applicable)* When I create an assistant and ask a document question, then ingestion and retrieval work exactly as before — embeddings now come from a **dedicated child** (`llama-server --embeddings` on Windows/Linux, `mlx_vlm.server` on Apple Silicon), not from the in-process sentence-transformers stack.
- [ ] *(not built yet: embeddings outside the backend, #594 — record as not applicable)* When retrieval is compared to the previous version on the **same KB** (golden questions), then answers ground on the same passages — no regression from the engine swap.
- [ ] *(not built yet: embeddings outside the backend, #594 — record as not applicable)* When the embedding child has been **idle**, then it is unloaded to free memory, and the next KB operation transparently reloads it. *(The Settings idle delay governs the chat model only, not this one.)*
- [ ] *(not built yet: embeddings outside the backend, #594 — record as not applicable)* When I inspect the installed app, then on Windows/Linux the **torch/transformers tree is gone**, and on Apple Silicon torch/sklearn/sentence-transformers are gone (transformers stays as an mlx-vlm dependency) — with no loss of KB function.
- [ ] *(not built yet: embeddings outside the backend, #594 — record as not applicable)* *(Apple Silicon)* When the KB embeds, then the app serves `intfloat/multilingual-e5-small` **as-is** through mlx_vlm — vectors match the previous output, so **no reindex** is needed.
- [ ] *(not built yet: embeddings outside the backend, #594 — record as not applicable)* *(Windows/Linux)* When the KB embeds, then the app serves the **published GGUF (F16)** through `llama-server --embeddings` — 384-dim, mean-pooled, L2-normalized, with the `query:`/`passage:` prefixes applied.
- [ ] *(not built yet: embeddings outside the backend, #594 — record as not applicable)* When chunking counts tokens, then it uses the pure `tokenizers` package (no HF `transformers`) and produces the same chunk boundaries as before.

## Settings — `/erudi/settings`

- [ ] When I click the **gear icon** at the bottom of the left rail, then the Settings page opens and the gear shows the active highlight.
- [ ] When I open Settings on a fresh install, then the **Web Search** toggle is **off** and the copy explains that enabling it sends the searched query to external search engines when the model decides to search.
- [ ] When I flip the Web Search toggle, then the change **persists across an app relaunch**.
- [ ] When the global toggle is on and I start a **new** conversation, then that conversation's own Web search switch starts **on** unless its model has saved defaults (inheritance at creation; the conversation owns it afterwards).

**Default reasoning effort**
- [ ] When I open Settings on a fresh install, then a **Default reasoning effort** card shows **Medium** selected among None / Low / Medium / High / Xhigh, and the copy says it sets the starting value for new conversations (each conversation keeps its own control afterwards).
- [ ] When I change the default, then it persists across an app relaunch and only affects conversations created afterwards.
- [ ] When I have saved a model's defaults with `/config global`, then new conversations on that model open at the saved effort, and every model without saved defaults keeps opening at this card's value.

**Default compaction threshold**
- [ ] When I open Settings on a fresh install, then a **Default compaction threshold** card shows **80 %** selected among 50 % to 90 %, and the copy says it sets the starting value for new conversations on models without saved defaults (each conversation keeps its own value afterwards).
- [ ] When I change the default, then it persists across an app relaunch and only affects conversations created afterwards — an existing conversation keeps its threshold.

**Idle unload delay**
- [ ] When I open Settings on a fresh install, then an **Unload the model when idle** card shows **5 minutes** selected among 5, 15, 30 and 60 minutes and **Never**, and the copy says an unused model leaves memory after that delay and reloads at the next message.
- [ ] When I pick **15 minutes**, then it applies at once (no relaunch needed), persists across an app relaunch, and a model left unused is unloaded about 15 minutes after its last answer (within half a minute; `backend.log` shows the cleanup), not after 5.
- [ ] When I pick **Never**, then a model left unused stays loaded (an hour later it still answers without reloading); the memory brake still acts when the computer runs short of memory, and **Unload <model name>** still works.

**Updates**
- [ ] When I open Settings on a fresh install, then the **Updates** card shows **Check for updates** on, **Download updates automatically** off and **Install downloaded updates when I quit** off; the copy says the check asks this project's GitHub releases and carries nothing but my version and platform, that downloading fetches the installer, and that installing replaces the app when I quit.
- [ ] When I turn **Check for updates** off and relaunch, then it is still off, no check runs (`erudi-backend.log` says no check will run; with it on, `checking now, then every 4 hours`), and the two other switches are disabled; the **Check for updates** button of the Update Erudi card still works.
- [ ] *(verified at the next real update)* When a newer release is published and only **Check for updates** is on, then a banner names the version and offers **Download**; nothing is downloaded until I press it, and quitting installs nothing.
- [ ] *(verified at the next real update)* When **Download updates automatically** is on and **Install … when I quit** is off, then the update downloads in the background, the banner says it is ready and offers **Install now**, and quitting installs nothing.
- [ ] *(verified at the next real update)* When **Install downloaded updates when I quit** is on, then a downloaded update installs when I quit, with no dialog, and the next launch runs the new version — also when I turned the switch on **after** the download had finished (Windows, macOS, Linux).
- [ ] *(verified at the next real update)* *(macOS)* When **Install … when I quit** was on while an update downloaded and I turn it off afterwards, then the card says this update may still install when I quit (macOS has already received it); the same holds when I turn **Check for updates** off afterwards.
- [ ] When I update an earlier version to this one, then **Check for updates** keeps its previous value and the two other switches are **off**, whatever automatic updates did before.
- [ ] When I read the privacy page, then it describes each of the three switches and what each one sends.

**Update Erudi (the manual controls)**

*The whole block needs a packaged build: running from source there is no
updater, and the card says so.*

- [ ] When I open Settings, then an **Update Erudi** card names the version I am running and offers **Check for updates**.
- [ ] When I press **Check for updates** while already on the newest release, then the button shows it is checking, and the card then says I am on the latest version — no dialog, no popup.
- [ ] *(verified at the next real update)* When a newer release exists and **Download updates automatically** is off, then pressing Check names that version and offers **Download**; nothing is downloaded until I press Download. Once downloaded, **Install now** installs it; quitting installs it only when **Install downloaded updates when I quit** is on and **Check for updates** is on (with Check off, both other switches are disabled and inert).
- [ ] *(verified at the next real update)* When the download is running, then the card shows the percentage and offers no second Download button.
- [ ] *(verified at the next real update)* When the update is downloaded, then **Install now** closes Erudi and opens it again on the new version.
- [ ] *(verified at the next real update)* When an update was downloaded earlier and its installation never applied (the app was reopened before the installer finished), then it comes back without downloading anything again — this is the case the card exists for. With **Download updates automatically** on, the check at launch re-detects it within moments and the card offers **Install now** on its own; with it off, the card offers **Download**, which completes at once from the copy already on disk. With **Check for updates** off, pressing **Check for updates** then **Download** brings it back instantly, served from the copy already on disk.
- [ ] When the machine is offline and I press Check for updates, then one quiet line on the card says the check did not go through, and the button is usable again.
- [ ] When I run the app from source (`npm start`), then the card says updates are handled by the installed application and the button is disabled.

**Context window on a discrete GPU**

*Everything in this block needs a Windows or Linux machine with an NVIDIA GPU —
the memory the window must fit is the card's VRAM, a path a unified-memory Mac
cannot exercise at all. `win-cuda` runs on the Windows CUDA machine, `linux-cuda`
on a Linux+NVIDIA one. Record every number asked for: the CUDA engine's constants
are starting values calibrated from them.*

- [ ] *(win-cuda, linux-cuda — FIRST, before the scenarios below)* When the shipped CUDA `llama-server` loads a small GGUF with `-lv 5 -np 1 --cache-ram 0 -c 8192`, then again with `-c 32768`, then with `-c 8192 -fa off`, then its `CUDA0 KV buffer size` is exactly `n_ctx` × the model's KV bytes per token (every run prints the KV line twice: `0.00 MiB` first, from fit's probe pass, then the real figure — read the second); record the `CUDA0 compute buffer size` and `CUDA_Host compute buffer size` lines (their growth between the two windows is the compute slope, the `-fa off` run the non-Flash-Attention one), the Flash Attention line (`enabled` or `not supported, set to disabled`) and `offloaded N/M layers to GPU`. At the default verbosity, record whether the Flash Attention WARN and, at increasing `-lv`, the `offloaded N/M` line reach `erudi-backend.log`, and how many lines per second that level writes during an answer.
- [ ] *(win-cuda, linux-cuda)* With the server idle after load, record `nvidia-smi --query-gpu=memory.used --format=csv` (and on Linux `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`): the child's memory minus weights, KV and the compute buffer is the CUDA context overhead. Record the child's device memory before and after an 8k-token prompt on a Q4_K_M and on an F16 GGUF (the pool's growth), its idle host footprint (Linux: `grep -E "Anonymous|Swap" /proc/<pid>/smaps_rollup`; Windows: `(Get-Process -Id <pid>).PrivateMemorySize64`), and whether `nvmlDeviceGetMemoryInfo(h, version=nvmlMemory_v2)` answers (informational: the app falls back to v1 when v2 is refused). On Windows, also compare NVML's free VRAM before the spawn with the free memory the `-lv 5` fit trace reports: the gap must stay under 512 MiB.
- [ ] *(win-cuda, linux-cuda)* When a catalog model loads on a roomy card, then `backend.log` shows `Window sized at spawn: window=W, trained=T, memory_bound=..., vram_free_mib=..., vram_total_mib=...`, the child's command line carries `-c W -ngl -1 -fitt 1024`, the `allocated context window: W tokens` line agrees, and `GPU placement: full offload (...)` follows.
- [ ] *(win-cuda, linux-cuda)* When a model is loaded with the free VRAM inflated on purpose (`ERUDI_VRAM_OVERRIDE_GB` above the real free memory), then llama.cpp's fit keeps fewer layers on the GPU (the child's VRAM in `nvidia-smi` is well below the full prediction), `backend.log` says `llama-server placed the model partly on the GPU (...); the memory brake is off for this child`, and no `Memory brake` line ever appears for that child.
- [ ] *(win-cuda, linux-cuda)* When a model whose minimum window cannot fit the card loads (a 13B-class model on an 8 GB card), then one WARNING says `Window not sized for ... the 4096-token floor does not fit ...`, the child starts without `-c` (fit chooses the window and the layers), and it answers; a DeepSeek-lineage GGUF says `MLA` the same way. No `Memory brake` line appears for either. On win-cuda the child's command line carries `-fitt` above 1024 and one INFO says `fit's margin set from the free memory CUDA reports: cuda_free_mib=..., nvml_free_mib=..., fit_target_mib=...` (1024 plus the gap); on linux-cuda `-fitt 1024` and no such line.
- [ ] *(win-cuda)* The #661 hog-first load: start a VRAM hog that leaves about 3.6 GB free (10.85 GiB on a 16 GB card), then load Qwen3 4B Q4_K_M. `backend.log` shows the floor WARNING, then `cuda_free_mib=` near the card's budget (15172 on the RTX 5060 Ti), `nvml_free_mib=` near 3561, and `-fitt` about 12.6k on the child's command line; with `-lv 4` added by hand to that command line, fit now reports `offloaded N/37 layers` with N below 37 (or a window well under 40960). Answers come at normal speed (#661: 8.5 s for N=27, against 38-51 s before), and `typeperf` shows no eviction (Total Committed equal to Dedicated + Shared) during the answers. Time `llama-server --list-devices` run by hand under the same environment (`Measure-Command`), and the gap between the floor WARNING and the margin INFO in `backend.log`: record both.
- [ ] *(win-cuda)* On a calm desktop, a 27B-class model that takes the floor path loads as before: `fit_target_mib` within a few hundred MiB of 1024 (the gap between CUDA's free and NVML's), the same `offloaded N/M` as without the correction, and the same answer speed.
- [ ] *(win-cuda)* With `ERUDI_VRAM_OVERRIDE_GB=3.5` on a calm 16 GB card and a model whose floor then does not fit, the margin INFO reads `nvml_free_mib=3584` and fit places the model partly on the GPU: the hog-first path without a hog.
- [ ] *(win-cuda, linux-cuda)* With `ERUDI_CTX=8192` set, then every load passes `-c 8192 -ngl -1`, and the window is never resized whatever the pressure. On win-cuda the margin INFO above appears at every load and `-fitt` carries it.
- [ ] *(win-cuda)* Swap from a large model (one holding most of the card) to an MLA GGUF (DeepSeek lineage), then again with `ERUDI_CTX=8192` set, right after the first one stops (switch models with `/model` and send at once): the margin INFO of the new load reads `nvml_free_mib=` close to the card's free memory with no model loaded (record it beforehand), not a figure that still counts the old child's memory, and `-fitt` stays within a few hundred MiB of 1024 on a calm desktop. If WDDM still counts the stopped child, record `nvml_free_mib`, the time since the stop (`Stopping llama-server on port ...` in `backend.log`) and the `offloaded N/M` the new child reports.
- [ ] *(win-cuda)* Before the spill scenarios, record the driver's "CUDA - Sysmem Fallback Policy" (NVIDIA Control Panel > Manage 3D settings; default: fallback on since driver 536.40) and, in a non-admin PowerShell, `typeperf "\GPU Process Memory(pid_<pid>*)\Shared Usage" -sc 5` and `... \Dedicated Usage` for the loaded child: both must answer without elevation, on a non-English Windows too; note the instance names (`pid_<pid>_luid_..._phys_0`), how many per pid on an Optimus laptop and which luid is the discrete GPU. Record the Shared Usage right after load, with the window and the Flash Attention state of `Window sized at spawn` (`fa_slope=`), and compare it with the healthy level the brake counts the spill from: 128 MiB + 1 KiB per window token with Flash Attention, 2 KiB per token without (40960 tokens with FA: 168 MiB). It must stay under it, and no `Shared GPU memory above the healthy level at load` WARNING appears. Take one more reading on a raw `llama-server -c 32768 -fa off` (or larger) run by hand: it checks the non-Flash-Attention slope. Informational only, not read by the app: `typeperf "\GPU Adapter Memory(*)\Dedicated Limit" -sc 1` (the counter set may not exist), and `nvmlDeviceGetComputeRunningProcesses` under WDDM (empty, or entries whose `usedGpuMemory` is 0).
- [ ] *(win-cuda)* When a model loaded near the card's limit (`ERUDI_VRAM_OVERRIDE_GB` inflated) meets a VRAM-hungry app started mid-conversation (a game, FurMark), then Task Manager (Details, "Shared GPU memory") shows `llama-server.exe` spilling, answers slow down but nothing crashes, and `backend.log` shows `Memory brake: ... action=shrink` at the next turn end; once I have been away two to three minutes the model restarts with a smaller window (`action=respawn, window=A->B`, the child's command line with `-ngl -2`, Task Manager's dedicated memory of the new child unchanged while shared memory absorbs the overflow), my next message shows the reload state and then the amber pressure notice. Pushing harder (spill over 1 GiB) restarts the model at the next turn boundary in the same turn; keeping it up while I am away unloads it after two idle samples (`action=unload` WARNING). The reload state is transient: it shows while the answer is pending and goes when the first token arrives, so watch the pending answer (reading the page after the turn finds nothing). The amber notice driven by a spill uses the generic copy ("Your computer's memory is under pressure: this conversation will be summarized sooner…"): that is expected, not a failure.
- [ ] *(win-cuda)* When the VRAM-hungry app closes, then the spill falls back in Task Manager and the window grows back at the next load of the model (`brake_cap=` larger or absent), never back to 90 % of the size that hurt while it is remembered.
- [ ] *(win-cuda)* On the normally loaded desktop (browser open), load each catalog model three times and record, with the free VRAM at load, whether the placement line says `full offload` or `partly on the GPU`: partial loads should be rare; record the Sysmem Fallback Policy with the runs. With "Prefer No Sysmem Fallback" set, a respawn under a VRAM hog fails (`action=respawn-failed` WARNING), the next message reloads the model, and its placement may read partial with the brake off.
- [ ] *(linux-cuda)* When a CUDA app grabs most of the remaining VRAM while a conversation runs, then our child keeps serving (Linux never spills: the other app's allocation fails), and no `Memory brake ... action=shrink` line appears. `llama-server -m <model.gguf> -ngl 99 -c <huge>` run by hand dies with `cudaMalloc failed: out of memory` before `/health` answers; through Erudi, a load that dies that way is retried once at the 4096 floor (one WARNING naming the possible host-side failure), and a second failure is the readable `CUDA_OUT_OF_MEMORY` error.
- [ ] *(win-cuda)* When a healthy full-window model (its `Window sized at spawn` window equal to the trained one) is loaded, answers once, and is left idle 10 minutes while the commit charge stays at or above 93 % (Task Manager, Performance → Memory → Committed; another app holding memory), then the brake never unloads it: no `Memory brake` line at all, no amber notice, and the only unload is the idle release 300-330 s (default 5-minute idle delay) after the last answer (`Cleaning up model`). The load logs no `Shared GPU memory above the healthy level at load` WARNING.
- [ ] *(linux-cuda)* With a full-offload child loaded, `stress-ng --vm 4 --vm-bytes 85% -t 180` changes nothing on our side: no `Memory brake` line, no amber notice, no unload, the window never moves — RAM pressure does not drive the CUDA brake, and Linux has no spill, so the brake there has no signal. The idle release at 300-330 s (default 5-minute idle delay) still happens as usual.
- [ ] *(win-cuda, linux-cuda, multi-GPU or Optimus machine — else record as untested)* `llama-server --list-devices` lists the cards; with no `CUDA_VISIBLE_DEVICES` set, the child's environment carries `CUDA_VISIBLE_DEVICES=GPU-<uuid>` of the largest card (`nvidia-smi -L`) and `--list-devices` under it shows that one card; with my own `CUDA_VISIBLE_DEVICES` naming one card, the child keeps my value and the window is sized on that card; naming two cards, `backend.log` says `CUDA_VISIBLE_DEVICES leaves several GPUs visible (2) ... the memory brake is off for this model`, the child starts without `-c` and keeps my value; set but empty (as in a `.env` copied from `backend/.env.example`), it counts as unset and the child is pinned to the largest card.
- [ ] *(win-cuda, linux-cuda, packaged build)* The frozen backend loads a model on the CUDA engine (no `pynvml` warning in `backend.log`), and `/props` `n_ctx` equals the `W` of `Window sized at spawn`. With `ERUDI_FORCE_MEMORY_PRESSURE=warning` or `critical` on a calm machine, nothing happens on the CUDA engine, on a full-offload child as on a fit-sized one (a DeepSeek GGUF): no `Memory brake` line, no notice, no shrink, no unload, even after 10 idle minutes (the idle release at 300-330 s (default 5-minute idle delay) aside) — these values force the RAM reading, which the CUDA engine never takes.
- [ ] *(win-cuda, packaged build, QA seam)* With `ERUDI_FORCE_MEMORY_PRESSURE=vram-warning` on a calm machine and a model whose window's KV is at least 10 % of the card's VRAM, the load logs the forced reading (`Shared GPU memory above the healthy level at load: shared_mib=..., healthy_mib=...`, expected under the seam: the forced reading is the healthy level plus 256 MiB); at the end of a turn of a long conversation `backend.log` shows `Memory brake: ... level=2, action=shrink`, once I have been away two minutes `action=respawn, window=A->B` with the child's command line carrying `-ngl -2`, and my next message shows the reload state (transient: watch the pending answer) then the amber pressure notice (the generic "Your computer's memory is under pressure…" copy). With `vram-critical`, the next turn boundary shrinks and restarts the model in the same turn (`action=shrink+respawn`); on a model whose window is the 4096 floor, no shrink is possible and two idle samples while I am away unload it (`action=unload`). On linux-cuda the VRAM values change nothing. Time a 14B-class model's brake respawn under real spill pressure (page cache evicted): fit's probe pass included, it must stay well inside the 30 s respawn bound.
- [ ] *(win-cuda)* With a model fully loaded and one answer given, when a VRAM hog is started that leaves less free memory than the model holds (a `cuMemAlloc` + `cuMemset` program of 10 GiB on a 16 GB card, or a game), then within a minute of idle (or at my next message, when it comes 15 s or more after the previous answer: no sample is taken sooner) `backend.log` shows one INFO `GPU contention started: evicted_mib=..., device_free_mib=...`, and the end of my next answer shows the amber notice "Another app is taking your graphics card's memory: answers are slower until it frees some. Nothing is lost." — again after every answer while the hog runs, and it stays between answers while the hog runs (no `GPU contention ended` line within 15 s of an answer's end, `Query completed`); no `Memory brake` line, no unload. Record `typeperf "\GPU Process Memory(pid_<pid>*)\Total Committed" "\GPU Process Memory(pid_<pid>*)\Dedicated Usage" "\GPU Process Memory(pid_<pid>*)\Shared Usage" -si 1` through the run: the evicted gap (Total Committed minus the two others) at idle under the hog, to calibrate the 256 MiB floor, and the same gap on a healthy child (expected 0). Then kill the hog without sending anything: within about 20 s the notice disappears on its own (`GPU contention ended` INFO within 10 s, then the page's next poll), even though the gap stays until the next answer pages the model back in. Record the time between the kill and the notice going. Once more with the hog killed within a few seconds of an answer's end: the `GPU contention ended` line comes no earlier than 15 s and at most about 25 s after that answer's end, and the notice goes at the next poll. Repeat once on an Optimus laptop if one is available (several `luid` instances per pid), else record it as untested.
- [ ] *(win-cuda)* While the contention notice shows, `erudi-backend.log` carries no INFO access line for `/erudi/hardware/gpu_contention/status` (the poll is a DEBUG line), the renderer polls only while the notice shows (no such request with a `bound` notice or none), and the notice stays during an answer (the poll pauses while a turn runs). While the contention lasts and nothing is asked, the backend's CPU use in Task Manager stays negligible (one PDH and one NVML read every 10 s).
- [ ] *(win-cuda, packaged build, QA seam)* With `ERUDI_FORCE_MEMORY_PRESSURE=vram-contention` on a calm machine, the first answer after the first idle minute (or the second answer) ends with the contention notice, `GPU contention started: evicted_mib=384, device_free_mib=0` is logged once, no `Memory brake` line appears, and the notice stays while the seam is set. On linux-cuda the value changes nothing.
- [ ] When the machine has far more system RAM than VRAM (e.g. 64 GB RAM, 8 GB card), then the **memory saturation notice** never appears — the notice is Apple-Silicon-only in this release: on the llama.cpp engines the KV cache is allocated in full at load for a window sized to fit, so memory use does not grow with the conversation and there is nothing for a per-turn notice to measure (partial offload would also make any single-pool accounting dishonest on a discrete card); the compaction threshold on the window still protects long conversations.
- [ ] When a model is loaded at a large window, then the VRAM shown by the system monitor at ready matches the expected weights-plus-cache footprint (no `holds more device memory than the window budget predicted` WARNING), and no out-of-memory happens later mid-conversation.

**Inference engine**

*Everything in this block needs a Windows or Linux machine with an NVIDIA GPU —
the card only appears where the setting does something.*

- [ ] When I open Settings on the NVIDIA machine, then an **Inference engine** card offers **Automatic** and **Processor only**, and the note says Erudi restarts its engine when the setting changes.
- [ ] When I open Settings on a fresh install, then the engine is **Automatic**.
- [ ] *(NVIDIA machine)* When I switch from Automatic to **Processor only**, then the backend restarts and comes back — the app is usable again within the usual boot time, not stuck on the loader.
- [ ] *(NVIDIA machine)* After that switch, when I send a chat message, then it answers, and `backend.log` shows `Engine chosen: <CUDA_Engine>` followed by the processor swap line — i.e. the model actually runs on the CPU build.
- [ ] *(NVIDIA installer, Processor only, a model loaded)* The child's command line (`(Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'").CommandLine` on Windows, `ps -o args= -C llama-server` on Linux; redact the key after `--api-key`) runs the binary under `artifacts\llama-cpp\cpu\bin\` — the installer's own CPU build — and `backend.log` has no `falling back to the cuda build` WARNING. The child holds **0 MiB** of dedicated GPU memory (Task Manager's per-process GPU memory, or `typeperf "\GPU Process Memory(pid_<pid>*)\Dedicated Usage" -sc 1`; `nvidia-smi` on Linux does not list it).
- [ ] *(win-cuda, Processor only, an SMT CPU — e.g. 8 cores / 16 threads)* The child's command line carries no `--threads`, and with a browser and a game launcher open but idle a 4B model generates at about the speed of `--threads <physical cores>` run by hand (#661: 17.3 tokens/s on a Ryzen 7 7700), never the collapse of `--threads <logical CPUs>` (0.66). Record the generation and the prompt tokens/s, and the core counts.
- [ ] *(NVIDIA machine)* When I relaunch the app, then the setting is still **Processor only**, inference is still on the processor, and the **Inference engine** card is still there — it is the only way back to Automatic.
- [ ] *(NVIDIA machine)* When I switch back to **Automatic** and relaunch, then the GPU is used again — the choice is reversible.
- [ ] *(NVIDIA machine)* While a model download is running, when I change the engine, then a question appears first — **A download is in progress** — saying the switch cancels the download and that it starts over from the beginning; nothing is saved or restarted yet. **Keep downloading** closes it with the engine unchanged and the download still running. **Cancel the download and switch** cancels it (the download widget reads **Download cancelled**, with no error), then the backend restarts on the new engine; `backend.log` shows the cancel before the restart, and no `Download interrupted due to application shutdown` at the next boot.
- [ ] *(NVIDIA machine, pinned to Processor only)* When I switch back to **Automatic** without relaunching, then the **Inference engine** card stays on screen, its select greyed out while the backend restarts, and once the backend is back it is still there, enabled, on **Automatic** — never gone until the app is relaunched.
- [ ] *(Apple Silicon)* When I open Settings on a Mac, then there is no **Inference engine** card at all — the Apple silicon engine is the only one, so the setting would be a no-op.

**Application language**
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
- [ ] When the app log cannot be read (rename `$TMPDIR/erudi-backend.log` to a directory of the same name, then reopen the page), then the recent-errors area says **The app log could not be read** and does *not* show the check mark that means nothing happened; the backend's own errors are still listed.
- [ ] When nothing was recorded, then the recent-errors area shows a check mark and **No warning or error recorded.** and nothing else; there is no copy button and no line about pasting a report, but **Open log folder**, **Report on GitHub** and the contact link are still there, headed **Report a problem**.
- [ ] When I click **Copy the full report**, then the button confirms *Copied* and the clipboard holds the setup summary plus the error list as plain text — including the absolute path of both log files, even though neither is shown on screen.
- [ ] When I click **Report on GitHub**, then my browser opens this repository's bug report form with **Erudi version**, **Operating system**, **Hardware** and **Model** already filled in, and pasting into the **Logs** field gives the text I just copied. *(Dropdown prefill is unverified upstream: if **Operating system** arrives empty, that is the known gap — every other field must be filled.)*
- [ ] When I click **Open log folder**, then the file manager opens with `backend.log` selected.
- [ ] *(Apple Silicon)* When a model is loaded, then the same folder holds `mlx-child-<port>.log` carrying the inference child's own output (its startup banner at least); when I switch models or quit, the file for the stopped child is gone.
- [ ] When I use the **contact page** link instead, then `erudi.app/contact` opens in my browser and the copy tells me to include everything above plus my screenshots.
- [ ] When I kill the backend (or launch with the port blocked) and open the page, then it says **the backend did not answer**, still shows my version and platform, still lists the app-side errors, and still offers **Copy the full report** (when there is something to report) and **Report on GitHub** — it does **not** go blank.
- [ ] When I click the **bug icon** in the left rail, then I land on the Diagnostics page, the icon is highlighted like the other destinations, and Settings shows no diagnostics of its own — no web page opens.
- [ ] When something the app itself gets wrong happens silently (not offline, not a service outage) — for example the backend returns a real 500 — then a small badge appears on the bug icon showing **1**; opening Diagnostics lists that error and the badge disappears, with no popup, toast or sound at any point.
- [ ] When I trigger a **Hugging Face download while offline**, then the download modal still tells me it failed, but the failure appears on **neither** the Diagnostics page's recent-errors list **nor** the copied report **nor** the bug icon's badge — an unreachable network is not counted as a defect anywhere.
- [ ] When more than nine new errors accumulate before I next open Diagnostics, then the badge reads **9+** rather than the exact count.
- [ ] When the recent-errors list would show the **same error on consecutive rows**, then they collapse into a single row with a repeat count — no run of identical lines slips through.
- [ ] When I **update or reinstall** the app, then Diagnostics shows only the **current install's** recent history — stale entries from a previous install are not carried over as if they were current.

## Shared chrome (sidebar, connection, downloads)

- [ ] When I click the sidebar icons, then I navigate to Models (Brain), Chat (Chat), Arena (Swords), and Knowledge Base (Book); the active screen is highlighted (Chat stays highlighted while in a conversation).
- [ ] When I click the bug icon, then the Diagnostics page opens (the web contact page is offered from inside that page, not by the icon).
- [ ] When a download is in progress, then the bug icon is hidden; navigation stays enabled and the progress widget follows me across screens.
- [ ] When I navigate to an unknown route, then I am redirected to the Models screen.
- [ ] When I type in the **conversation search** field in the sidebar, then the conversation list filters by **title and content** as I type (a light ILIKE match, no index, no semantic search), with matched snippets highlighted; clearing the field restores the full list.

## Security — localhost hardening

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
- [ ] When I grep `$TMPDIR/erudi-backend.log`, `~/Library/Logs/erudi/backend.log` and the child's own log `~/Library/Logs/erudi/mlx-child-<port>.log` for `api-key`, `api_key`, `MLX_VLM_SERVER_API_KEY` and `Bearer`, then no line carries a key value (the key exists only in the backend's and the child's memory; that it changes on every load is pinned by the unit tests in `backend/tests/test_mlx_engine_server.py`, `TestSpawnApiKey`, and cannot be observed from outside the processes).

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
- [ ] When the app boots, then the embedded PostgreSQL comes up tuned for a **single desktop user** (modest shared_buffers / max_connections / WAL rather than server defaults), and an existing data folder from a **previous version** still boots and migrates without loss.

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
- [ ] When I click **Switch to processor mode** while a model download is running, then the same dialog turns into the **A download is in progress** question (no second window on top of it); **Keep downloading** brings the dialog back unchanged, **Cancel the download and switch** cancels the download and then switches.
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

**Shutdown & orphans**
- [ ] When I **hard-kill the app process** (Activity Monitor "Force Quit" / Task Manager "End task") rather than quitting cleanly, then the **backend stops by itself within a few seconds** — it does not survive holding port 27182 (parent-death watchdog).
- [ ] After that same hard kill, then **no `postgres` and no `llama-server` / `mlx_vlm` process is left running** (check the process list). *Known open question on Windows; record exactly what survives, and grab the tail of `backend.log` right after the kill: whether it shows a shutdown marker or just stops decides the cause.*
- [ ] When I hard-kill **during a generation** (not idle), then the same holds: backend gone, children gone, and relaunching immediately works (the port is free, the cluster is not locked).
- [ ] When I relaunch after any of the above, then the app boots normally and my conversations, models and knowledge bases are intact.
- [ ] *(Windows)* When I shut Windows down with the app open, then start the app, quit it normally and start it again, then the **second** start runs no crash recovery: the database's own log (`%LOCALAPPDATA%\erudi\backend\prod\data\postgres\log`) shows no new `database system was not properly shut down`, `backend.log` has no `Postgres crash recovery finished` for it, the app is ready in seconds, and the first quit logs no `pgserver's handle count did not allow a clean stop` either (the start before it pruned the dead backend's entry: `Pruned pgserver handle pids`).

**Interrupted downloads**
- [ ] When a download **completes** but the app is killed before the job row is finalized, then on relaunch the model is **kept and marked installed** — it is never silently deleted (a multi-GB artifact must survive; if it *is* deleted, that is a data-loss regression).
- [ ] When a download is **genuinely truncated** and the app is killed, then on relaunch the incomplete files are removed and the log states the path and the size reclaimed (deletion is never silent).
- [ ] When a download finishes and I stay on the screen, then the progress widget resolves and the UI **never stays stuck at 100%** — if finalization wedges, the poll gives up after a few minutes, the sidebar and contact icon come back, and the message says the files were saved (not "Download failed", which would push me to re-download gigabytes I already have).

**Windows regression gate — run these FIRST on any Windows candidate**

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

**Model sizes & recommendations**
- [ ] When I look at a model's **Size** before downloading it and again once installed, then the two figures **match** — a model must not appear to shrink (or grow) the moment it finishes downloading.
- [ ] When I compare a model's displayed size with the figure on its Hugging Face page, then they agree (decimal GB, the unit HF quotes).
- [ ] When I read the machine readout's recommended size window, then it reflects **both** what fits in memory **and** what my memory bandwidth can stream at a usable speed — on a 16 GB Apple Silicon machine that lands around 5–10B, not the high teens.
- [ ] When my machine is large (high-VRAM card), then the recommended window still includes the excellent **7–14B** models rather than starting above them.

**Updates & first run**
- [ ] *(verified at the next real update)* When I run a **packaged** build with **Download updates automatically** on and a newer release is published, then a banner shows "downloading…", then "ready — restart to install"; it installs on **Install now**, or at quit when **Install downloaded updates when I quit** is on.
- [ ] *(verified at the next real update)* When the app starts on a newly installed version after an update it downloaded, then a dismissible popup says **Updated to X.Y.Z** with that release's notes, kept from the update download (no request is made at launch); once dismissed it does not reappear on later launches of that version.
- [ ] *(verified at the next real update)* When the notes contain formatting or links, then they show as plain text (lists as lines, no images loaded); the only link is **Read the release notes**, which opens in the browser.
- [ ] *(verified at the next real update)* When an update was downloaded but never installed and I then install a later version by hand, then the popup does not show the older notes (version and link only).
- [ ] When the notes are not available (a manual install, or a release without notes), then the popup shows the version and a **Read the release notes** link that opens the release page in the browser only when clicked.
- [ ] When I first launch this version over an earlier one, then the popup says **Updated to** this version, with the link, and explains that downloading and installing updates are now separate choices in Settings, both off.
- [ ] When I launch a fresh install, or *(macOS, Windows)* the app after **Clear all data** (no conversation and no downloaded model), then no update popup appears and the welcome dialog shows as usual.
- [ ] *(verified at the next real update)* When I run an older version than the last one I ran, then no update popup appears.
- [ ] *(QA setup)* Lines marked *verified at the next real update* need a newer published release: on this version's draft, record them as covered by the updater policy's unit tests (one run per platform, attached to the pass) and run them on the first real update from this version to the next one. On macOS, record every update line NOT RUN while no notarized macOS build exists.
- [ ] When a release is still a **draft**, then my installed build is **not** offered that update.
- [ ] *(verified at the next real update)* When **Check for updates** is off in Settings and a newer release is published, then no banner appears, nothing is downloaded and quitting installs nothing; turning it back on starts a check at once — the banner appears without a relaunch (and downloads only when **Download updates automatically** is on).
- [ ] *(verified at the next real update)* When I dismiss the update banner, then it does not come back for that version until the next launch (the periodic check does not re-raise it), and the **Update Erudi** card in Settings still knows the update is there and offers **Download** or **Install now**, whichever applies.
- [ ] When I do a **fresh install**, then the Welcome dialog shows once, the catalog seeds instantly from the bundled snapshot (then refreshes in the background), and the machine readout renders (even if hardware profiling falls back).
- [ ] When the app quits, then the backend and its inference child processes are stopped (none left orphaned).

---

### How to record a run

Per release candidate, note: build version, OS + hardware, who ran it, date, and
any **FAIL** with a linked issue. For a FAIL, grab both log files and the `fe-…`
request id of the failing action — locations and the tracing recipe are in
[docs/logging.md](docs/logging.md). Platform coverage (which OS/GPU each artifact
was tested on) is tracked in `docs/dev/release-qa-checklist.md`.
