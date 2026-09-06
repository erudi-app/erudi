# What leaves your machine

Erudi's promise is simple to state and easy to get wrong: your prompts, your answers and your documents are processed on this computer and stay on it. This page is the audit of that promise. It lists every network request the application can make, what is exposed on your computer while it runs, where your data is written, and the gaps we know about. Each claim links to the code that backs it, on the `main` branch at the time of writing — if the code moves, the link still shows the file.

If you find this page to be wrong, that is a valid security report: see [SECURITY.md](https://github.com/erudi-app/erudi/blob/main/SECURITY.md).

## The short version

- **Nothing you type is sent anywhere.** Inference runs in a child process on this computer, started with a local file path and bound to `127.0.0.1`. There is no account, no telemetry, no crash reporter.
- **The application makes network requests in exactly two situations you did not ask for:** an update check against this repository's GitHub releases, which you can turn off in Settings, and — after a model download you started — a small fetch of that model's generation defaults. Everything else happens only when you press a button that says so.
- **The one path by which the substance of a question leaves the machine is web search**, which is off by default and, when on, sends the search query the model formulates to public search engines.

## What never leaves

Models are launched as a local server with a file path, not a repository name, and told to listen on `127.0.0.1` only: `llama-server` gets `-m <path> --host 127.0.0.1` ([cpu_engine.py](https://github.com/erudi-app/erudi/blob/main/backend/src/engines/cpu_engine.py#L143-L154), [cuda_engine.py](https://github.com/erudi-app/erudi/blob/main/backend/src/engines/cuda_engine.py#L183-L194)), `mlx_vlm.server` gets `--model <dir> --host 127.0.0.1` ([mlx_engine.py](https://github.com/erudi-app/erudi/blob/main/backend/src/engines/mlx_engine.py#L296-L317)). The agent layer talks to that server at `http://127.0.0.1:<port>/v1` with the key minted for that child ([model_factory.py](https://github.com/erudi-app/erudi/blob/main/backend/src/agents/model_factory.py#L113-L119)). There is no `http://` or `https://` URL in the engine or agent code other than loopback.

Tokenizers are loaded from disk with `local_files_only=True` ([mlx_engine.py](https://github.com/erudi-app/erudi/blob/main/backend/src/engines/mlx_engine.py#L210-L212)); the embedding model used by the knowledge base is loaded from its local cache once it has been downloaded ([embedding_model.py](https://github.com/erudi-app/erudi/blob/main/backend/src/ingestion/embedding_model.py#L82-L90)). The model catalog you see at launch comes from a snapshot bundled inside the app, with no network call ([api.py](https://github.com/erudi-app/erudi/blob/main/backend/src/core/api.py#L451-L455), [catalog_snapshot.py](https://github.com/erudi-app/erudi/blob/main/backend/src/database/catalog_snapshot.py#L87-L96)).

Documents you attach to a knowledge base are read in place and never copied; only their text chunks and embeddings are stored, in the embedded database ([services.py](https://github.com/erudi-app/erudi/blob/main/backend/src/domains/knowledge_base/services.py#L304-L327)).

Hugging Face's own client telemetry is disabled by the launcher before anything is imported: `HF_HUB_DISABLE_TELEMETRY=1` ([run.py](https://github.com/erudi-app/erudi/blob/main/backend/run.py#L156)). No analytics or crash-reporting library is present in the code; the list of what we searched for is at the end of this page.

The connection indicator in the corner of the window costs nothing to anybody. It reads whether this machine is on a network from the operating system — `navigator.onLine` and the `online` / `offline` events — and corrects that with the requests the app already makes on your behalf: one that dies on the wire says more than a link light does ([networkStatus.js](https://github.com/erudi-app/erudi/blob/main/frontend/src/utils/networkStatus.js#L1-L35), [client.js](https://github.com/erudi-app/erudi/blob/main/frontend/src/services/api/client.js#L31-L44)). Nothing is sent to find out, so no one learns when your machine is running Erudi.

The window loads no remote images. Its Content-Security-Policy allows `img-src 'self' data:` and no scheme beyond that, both for the header the main process sets and for the packaged document ([main.js](https://github.com/erudi-app/erudi/blob/main/frontend/src/main.js#L705), [webpack.config.js](https://github.com/erudi-app/erudi/blob/main/frontend/webpack.config.js#L110-L125)). An image link written by a model, or carried by a document you attached, is simply not fetched — so no host learns your IP address from an answer on screen.

Automatic updates can be turned off. The preference lives with your other settings ([UserSettings.py](https://github.com/erudi-app/erudi/blob/main/backend/src/entities/UserSettings.py#L67)) and is handed to the process that owns the updater, which makes no check at all until it has been told what you chose ([main.js](https://github.com/erudi-app/erudi/blob/main/frontend/src/main.js#L1067-L1109)). Off means off end to end: no check, no download, no install on quit.

## Every network request

| When | Where to | What is sent | Can you turn it off |
|---|---|---|---|
| **At launch, then every 4 hours**, unless you turned updates off | GitHub releases of `erudi-app/erudi` | A request for the update feed and, when a newer version exists, the installer download. No identifier beyond the client's version and platform. [main.js](https://github.com/erudi-app/erudi/blob/main/frontend/src/main.js#L1067-L1109) · [electron-builder.yml](https://github.com/erudi-app/erudi/blob/main/frontend/electron-builder.yml#L52-L56) | **Yes**, in Settings. Off means no check, no download, no install. [UserSettings.py](https://github.com/erudi-app/erudi/blob/main/backend/src/entities/UserSettings.py#L67) |
| **When you press Download** on a model | `huggingface.co` and its file CDN | The repository id of the model you chose; your `HF_TOKEN` if you set one. [services.py](https://github.com/erudi-app/erudi/blob/main/backend/src/domains/llms/services.py#L686-L704) | It only happens when you ask |
| **Right after a download finishes**, if the files did not include generation defaults | `huggingface.co` | The same repository id, to fetch the model's `generation_config`. [endpoints.py](https://github.com/erudi-app/erudi/blob/main/backend/src/domains/llms/endpoints.py#L333-L351) | Follows your download |
| **When you search Hugging Face** from the catalog | `huggingface.co` | The text you typed in the search box, plus the format filter (`mlx` or `gguf`); your `HF_TOKEN` if set. [hf_search.py](https://github.com/erudi-app/erudi/blob/main/backend/src/domains/llms/hf_search.py#L66-L77) | It only happens when you ask |
| **When you accept the embedding-model download** for the knowledge base | `huggingface.co` | The repository id `intfloat/multilingual-e5-small`. [embedding_model.py](https://github.com/erudi-app/erudi/blob/main/backend/src/ingestion/embedding_model.py#L42) | Once; declining leaves the knowledge base unavailable |
| **Web search**, when you have turned it on and a tool-capable model decides to search | Public search engines — see below | The search query the model formulated from your question. [tools.py](https://github.com/erudi-app/erudi/blob/main/backend/src/agents/tools.py#L203-L212) | **Off by default**, per conversation and globally in Settings. [UserSettings.py](https://github.com/erudi-app/erudi/blob/main/backend/src/entities/UserSettings.py#L65) |
| **When you click a link** in the About menu, in this page's link in Settings, or in the report block that Settings → Diagnostics and the engine dialog both show | GitHub, `erudi.app`, this documentation | Nothing beyond the URL, and the page opens in your system browser. The bug report form's URL carries the four fields the panel shows you first — version, operating system, hardware, model — so that you do not retype them; your logs are never in it, they go through the clipboard when you press Copy. [main.js](https://github.com/erudi-app/erudi/blob/main/frontend/src/main.js#L659-L664) | It only happens when you click |

**About web search.** The search tool uses the `ddgs` library with its `auto` backend. Despite the name, that is not only DuckDuckGo: the library queries a shuffled selection of public engines — Wikipedia and Grokipedia first, then any of DuckDuckGo, Google, Bing, Brave, Startpage, Mojeek, Yahoo and Yandex — two at a time until it has enough results, and it presents itself to them with a randomised browser fingerprint. No API key is involved and nothing identifies you beyond your IP address, but you cannot know in advance which engine will see a given query. That is why the toggle ships off, and why the Settings screen says so. We intend to pin the engine — see [known gaps](#known-gaps).

**About LangChain's tracing client.** The agent runs on LangChain, which ships with a cloud tracing client called LangSmith. That client uploads the entire exchange — system prompt, knowledge-base excerpts, your question, the model's answer — and it is switched on by an environment variable alone. Erudi holds it shut in two places. The launcher assigns `LANGSMITH_TRACING` and `LANGCHAIN_TRACING_V2` to `false` on every start, overwriting whatever the process inherited ([run.py](https://github.com/erudi-app/erudi/blob/main/backend/run.py#L126-L163)), and the Electron process strips every `LANGCHAIN_*` and `LANGSMITH_*` variable out of the environment before the backend is started at all ([backendSpawn.js](https://github.com/erudi-app/erudi/blob/main/frontend/src/utils/backendSpawn.js#L14-L23)). Both are covered by tests. The single exception is the escape hatch described below.

## What is exposed on this computer

The backend listens on `127.0.0.1` only, on the first free port from 27182 to 27199 ([run.py](https://github.com/erudi-app/erudi/blob/main/backend/run.py#L94-L98), [run.py](https://github.com/erudi-app/erudi/blob/main/backend/run.py#L571)). Nothing on your network can reach it. It refuses requests whose `Host` is not local, which blocks DNS-rebinding attacks from web pages, and it only accepts browser origins belonging to the app window ([api.py](https://github.com/erudi-app/erudi/blob/main/backend/src/core/api.py#L280-L326)).

The inference servers listen on `127.0.0.1` too: `llama-server` on 27200–27299, `mlx_vlm.server` on 27300–27399 ([base_llama_cpp_engine.py](https://github.com/erudi-app/erudi/blob/main/backend/src/engines/base_llama_cpp_engine.py#L93-L98), [mlx_engine.py](https://github.com/erudi-app/erudi/blob/main/backend/src/engines/mlx_engine.py#L137-L141)).

Every inference server is started with a random key that exists only for the life of that process, so nothing else on your machine can drive the loaded model — not another program, and not a web page in your browser, which can otherwise POST to a loopback port. On Windows and Linux, `llama-server` gets `--api-key`; its slots endpoint, which would report the prompt of every request in flight, and its bundled web interface are both switched off, because Erudi uses neither ([base_llama_cpp_engine.py](https://github.com/erudi-app/erudi/blob/main/backend/src/engines/base_llama_cpp_engine.py#L417-L429)). On Apple Silicon, `mlx_vlm.server` gets its own `--api-key` ([mlx_engine.py](https://github.com/erudi-app/erudi/blob/main/backend/src/engines/mlx_engine.py#L304-L343)); because mlx-vlm applies that key to its management endpoints only and leaves the chat, responses and image routes open, Erudi adds a layer inside the child that requires the key on every route, `/v1/chat/completions` and `/health` included ([_mlx_vlm_server_runner.py](https://github.com/erudi-app/erudi/blob/main/backend/src/engines/_mlx_vlm_server_runner.py#L91-L128), [_mlx_vlm_server_runner.py](https://github.com/erudi-app/erudi/blob/main/backend/src/engines/_mlx_vlm_server_runner.py#L144-L176)). The backend presents the key on its own readiness probe and on every inference call ([base_chat_server_engine.py](https://github.com/erudi-app/erudi/blob/main/backend/src/engines/base_chat_server_engine.py#L276), [model_factory.py](https://github.com/erudi-app/erudi/blob/main/backend/src/agents/model_factory.py#L119)); the key is never written to a log.

The embedded PostgreSQL database is reachable through a Unix socket only on macOS and Linux ([postgres_runtime.py](https://github.com/erudi-app/erudi/blob/main/backend/src/launcher/postgres_runtime.py#L12-L16)). On Windows, where Unix sockets are not available, it listens on a loopback TCP port without a password.

The app window runs with Chromium's sandbox, context isolation and no Node integration ([main.js](https://github.com/erudi-app/erudi/blob/main/frontend/src/main.js#L640-L646)); the packaged build cannot be started with remote debugging or Node inspection flags ([electron-builder.yml](https://github.com/erudi-app/erudi/blob/main/frontend/electron-builder.yml#L29-L35)).

**What this does and does not protect against.** Erudi is built for a computer you use yourself. It defends the backend against other machines and against web pages you visit. It does not defend against software already running under your account: such a program can read the data folder. The same is true of every local-AI tool we know of; we say it here so that you can decide what runs next to Erudi.

## Where your data is written

| | Data (database, models, knowledge base) | Backend log |
|---|---|---|
| macOS | `~/Library/Application Support/erudi/backend/prod/data` | `~/Library/Logs/erudi/backend.log` |
| Windows | `%LOCALAPPDATA%\erudi\backend\prod\data` | `%LOCALAPPDATA%\erudi\logs\backend.log` |
| Linux | `$XDG_DATA_HOME/erudi/backend/prod/data` (default `~/.local/share/…`) | `$XDG_STATE_HOME/erudi/logs/backend.log` (default `~/.local/state/…`) |

Source: [runtime_paths.py](https://github.com/erudi-app/erudi/blob/main/backend/src/launcher/runtime_paths.py#L173-L191). Inside the data folder: `models/` (the weights you downloaded), `models_cache/` (the embedding model), `postgres/` (conversations, knowledge-base chunks and embeddings), `db-backups/` (a copy taken before each database migration). *Open data folder* and *Clear all data* in the app act on this folder.

Two things are written outside it:

- **The app log**, `erudi-backend.log`, in your system's temporary directory (`$TMPDIR` on macOS, `%TEMP%` on Windows, `/tmp` on Linux), with one rotated copy ([main.js](https://github.com/erudi-app/erudi/blob/main/frontend/src/main.js#L113-L115)).
- **One Hugging Face cache file**: the generation defaults fetched after a download go to Hugging Face's own cache, `~/.cache/huggingface/hub` ([generation_hints.py](https://github.com/erudi-app/erudi/blob/main/backend/src/database/generation_hints.py#L556)).

**The logs contain your content.** Because nothing leaves the machine, Erudi deliberately logs what it processes, so that a bug can be diagnosed from the file alone: the question you asked (up to 2 000 characters), a preview of the answer, document names, the knowledge-base query, and the web-search query ([logutils.py](https://github.com/erudi-app/erudi/blob/main/backend/src/core/logutils.py#L3-L11), [logging.md](logging.md)). The app log also records what you typed into text fields, truncated to 200 characters, to reconstruct what the interface was doing ([InteractionLogger.jsx](https://github.com/erudi-app/erudi/blob/main/frontend/src/components/InteractionLogger.jsx#L53-L56)). Read a log before you attach it to a public issue; the bug-report form reminds you.

## Environment variables that change any of this

All of them are documented in [`backend/.env.example`](https://github.com/erudi-app/erudi/blob/main/backend/.env.example) and read from the environment at launch; none is embedded in a build.

- `HF_TOKEN` — the only secret. When set, it is sent to `huggingface.co` on downloads, catalog searches and the post-download fetch above, and nowhere else.
- `HF_HUB_OFFLINE=1` — honoured by the Hugging Face client: every request to `huggingface.co` fails immediately with an explicit offline message. A blunt but effective switch if you want to be sure.
- `ERUDI_DATA_ROOT` — moves the data and log folders.
- `ERUDI_LOG_LEVEL` — verbosity of the backend log (`INFO` by default).
- `ERUDI_ALLOW_LANGSMITH_TRACING=1` — the only way to let LangChain's tracing client send conversations off your machine, meant for a contributor debugging the agent layer. It has to be typed out in full, and it should never be set in a build anyone distributes.

Three more are **written** by the launcher rather than read from you, and setting them yourself has no effect: `HF_HUB_DISABLE_TELEMETRY` is forced to `1`, and `LANGSMITH_TRACING` and `LANGCHAIN_TRACING_V2` are forced to `false`. They are assignments, not defaults, precisely so that a value inherited from your shell cannot re-open them.

## Known gaps

These are the points on which the current build is weaker than this page would like it to be. They are tracked as issues; the list is here so that you do not have to find them yourself.

1. **Web search does not pin an engine.** The `auto` backend picks among ten engines. It should use one, named on this page.
2. **On Windows, the embedded database is reachable by any local process** on a loopback port without a password.

## How to check for yourself

Run Erudi behind an outbound firewall or a local proxy and watch what it asks for: with web search off and no download in progress, you should see the update feed on GitHub and nothing else — and nothing at all once you turn updates off in Settings. Load a model and chat: the only traffic is on `127.0.0.1`. The backend log lists every request it received.

## What we looked for and did not find

Searched across the backend, the Electron main process, the renderer and the packaging configuration: telemetry and analytics SDKs (Sentry, PostHog, Mixpanel, Amplitude, Segment, Google Analytics, Umami, Plausible, Datadog, New Relic, Bugsnag, OpenTelemetry, Electron's crash reporter), other network transports (WebSocket, EventSource, `sendBeacon`, `XMLHttpRequest`, mail or FTP clients), search providers requiring an API key, login-item or firewall manipulation, remote debugging switches, and external CDNs for fonts or scripts. None is present. The interface's font and the maths renderer are bundled with the app.

We also looked for a **stable identifier** — an install id, a device fingerprint, a machine id, a MAC address read, anything that would let two requests be tied to the same computer. There is none. The only identifier the code generates is an eight-character request id, regenerated for every request and used solely to correlate lines in the local log; it is never sent anywhere. And no log leaves the machine: the backend attaches exactly two handlers, one to standard output and one to a rotating local file, and the renderer's logs travel over an internal channel to that same file.

`electron-log` does contain a remote transport. It ships disabled, with no URL configured, and nothing in this repository configures one.
