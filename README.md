# Erudi

**Open models on your machine. No account, no server to run, no data leaving your device.**

Erudi is a desktop app that downloads open-source language models, runs them on your own hardware — Apple Silicon, NVIDIA GPU or CPU — and lets you chat with them and with your documents, entirely offline. It installs like any other app: no Docker, no terminal, no runtime to manage.

**[Download for macOS · Windows · Linux](https://github.com/erudi-app/erudi/releases)**

[![Latest release](https://img.shields.io/github/v/release/erudi-app/erudi?include_prereleases&label=release)](https://github.com/erudi-app/erudi/releases)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-erudi--app.github.io-1f7a5a)](https://erudi-app.github.io/erudi/)

![Chatting with Qwen3 4B in Erudi — the model's reasoning streams into a collapsible strip above the answer](docs/assets/screenshots/chat-reasoning.png)

<sub>Qwen3 4B (MLX, 4-bit) on a MacBook Air M4 with 16 GB, in the packaged 1.0.0 build. The model's reasoning streams into the collapsible strip; the answer below it is the model's own.</sub>

## What leaves your machine

- **Nothing you type.** Prompts, answers and documents are processed by a model running in a child process on this computer, started from a local file and bound to `127.0.0.1`, and stored in an embedded database in your user folder.
- **No account, no telemetry, no crash reporter.** The backend listens on `127.0.0.1` only; nothing on your network can reach it.
- **Two requests you did not ask for, neither carrying your data**: an update check against this repository's releases, which you can turn off in Settings, and a fetch of a model's generation defaults after you download it. Everything else — model downloads, Hugging Face search, the optional web search (off by default) — happens when you press the button.

Every request, when it happens and the code behind it: **[What leaves your machine →](https://erudi-app.github.io/erudi/privacy/)**

## What it does

- **A catalog that knows your machine.** Every model card shows whether the model fits — memory *and* the speed you can expect — and the readout gives you a recommended size window. Erudi downloads pre-built MLX and GGUF quantizations; it never converts or quantizes weights on your computer.
- **A knowledge base on your documents.** Attach `.pdf`, `.docx`, `.xlsx`, `.csv`, `.txt` or `.md` files to a model and get an assistant that searches them before answering, cites the excerpt it used, and says so when the documents do not cover the question.
- **Inference that fits the hardware.** [MLX](https://github.com/ml-explore/mlx) on Apple Silicon, [llama.cpp](https://github.com/ggerganov/llama.cpp) with CUDA on NVIDIA GPUs, llama.cpp on CPU otherwise — picked automatically at launch.
- **Reasoning and tool calls you can see.** Thinking models stream their reasoning into a strip above the answer; when a model searches your documents or the web, the call and its result are shown there, never mixed into the answer.
- **Conversations kept on your machine**, restored across restarts, summarized as they grow.
- **Fully offline** once a model is downloaded.

<table>
  <tr>
    <td width="50%"><img src="docs/assets/screenshots/models-catalog.png" alt="The model catalog: each card says whether the model fits the machine, and the machine readout shows a recommended size window"></td>
    <td width="50%"><img src="docs/assets/screenshots/knowledge-base.png" alt="The knowledge base screen: attach documents to a local model to build an assistant that answers from them"></td>
  </tr>
  <tr>
    <td align="center"><sub>Catalog ranked for your hardware</sub></td>
    <td align="center"><sub>Knowledge base: documents in, grounded answers out</sub></td>
  </tr>
</table>

## Install

1. **[Download](https://github.com/erudi-app/erudi/releases)** the installer for your platform — a notarized `.dmg` for macOS (Apple Silicon, macOS 14+), a `Setup.exe` for Windows, an `.AppImage` for Linux. Windows and Linux each ship two builds: take the plain one unless you have an NVIDIA GPU, in which case take the `cuda` one. Nothing else to install — the inference engine and everything it needs are inside the installer.
   The `cuda` build needs a card of compute capability 5.0 or newer (Maxwell, 2014, and up) and a recent NVIDIA driver: the 570 family for cards up to the RTX 20 series and for the H100, which run from the build's PTX, and any 12.0-era driver (525+) for the RTX 30, 40 and 50 series, which have native code in it. Erudi checks both at launch and offers to run on the processor instead when your card or driver falls short — it never switches on its own. The [hardware guide](https://erudi-app.github.io/erudi/guides/hardware/) has the per-generation table.
2. Open it. The first launch prepares the embedded database and the model catalog; it takes a few seconds.
3. Pick a model marked **Runs easily** or **Ideal fit** and press Download. Once it is on disk, you can unplug the network and keep working.

The app updates itself from the GitHub releases of this repository, and only from there.

---

## Platform Support

| Platform | Backend | Status |
|---|---|---|
| Windows (NVIDIA GPU) | CUDA via `llama-server` | ✅ Supported (compute capability 5.0+, driver 570+ — 525+ for RTX 30/40/50) |
| Windows (no GPU) | CPU via `llama-server` | ✅ Supported |
| macOS Apple Silicon (macOS 14+) | MLX | ✅ Supported |
| Linux (NVIDIA GPU) | CUDA via `llama-server` | 🚧 Builds and launches in CI, not yet tested on real hardware (same GPU and driver floors as Windows) |
| Linux (CPU) | CPU via `llama-server` | 🚧 Builds and launches in CI, not yet tested on real hardware |

✅ means the packaged app is built by CI and manually tested on that platform. 🚧 means every release
builds and boots there in CI, but no one has yet run a full manual pass on real hardware — try it and
tell us what breaks.

---

## How it compares

Most local-AI tools are either a runtime you drive from a terminal, or a web interface you host on top of one. Erudi is the whole thing in one desktop app: the inference engine is inside it, the catalog tells you what your machine can actually run, and your documents become an assistant without anything to host, configure or pay for. If you already run Ollama or LM Studio and like it, keep it; Erudi is for the people who would rather not.

---

## Getting Started (Development)

### Prerequisites

- **Node.js** >= 20
- **Python 3.12** exactly — `pgserver`, which ships the embedded PostgreSQL cluster, publishes wheels up to cp312 only
- **Git**
- Platform-specific requirements:
  - A CUDA toolkit to compile `llama-server` for an NVIDIA GPU on Windows or Linux — any 12.x works; releases are built with 12.8, which is the first that emits native code for RTX 50 cards
  - Xcode Command Line Tools for macOS

> These are **build** requirements. Running Erudi needs none of them: the installer
> carries the inference engine and its CUDA runtime, so an NVIDIA user only needs a
> driver, and everyone else needs nothing at all. Which driver depends on the card —
> 570+ for anything that runs from the build's PTX, 525+ for the RTX 30/40/50 series,
> which have native code in it. The
> [hardware guide](https://erudi-app.github.io/erudi/guides/hardware/) has the table, and
> the app checks it at launch.

### 1. Clone the repository

```bash
git clone https://github.com/erudi-app/erudi.git
cd erudi
```

### 2. Set up the backend

Run the setup script for your platform:

| Platform | Script |
|---|---|
| macOS Apple Silicon | `bash scripts/dev/backend/setup-mac-silicon.sh` |
| Windows CUDA | `.\scripts\dev\backend\setup-win-cuda.ps1` |
| Windows CPU | `.\scripts\dev\backend\setup-win-cpu.ps1` |
| Linux CUDA | `bash scripts/dev/backend/setup-linux-cuda.sh` |
| Linux CPU | `bash scripts/dev/backend/setup-linux-cpu.sh` |

### 3. Build llama.cpp

The inference engine must be compiled for your platform:

```bash
# macOS Apple Silicon
bash scripts/dev/backend/build-llamacpp-cpu-macos-silicon.sh

# Linux CPU
bash scripts/dev/backend/build-llamacpp-cpu-linux.sh

# Linux CUDA
bash scripts/dev/backend/build-llamacpp-cuda-linux.sh
```

On Windows, run one of the following commands in PowerShell:

```powershell
# Windows CPU
.\scripts\dev\backend\build-llamacpp-cpu-win.ps1

# Windows CUDA
.\scripts\dev\backend\build-llamacpp-cuda-win.ps1
```

### 4. Start the app

Start the backend in the first terminal:

```bash
cd backend
source venv/bin/activate        # macOS/Linux
# Windows: .\venv\Scripts\Activate
python run.py
```

Start the frontend in a second terminal:

```bash
cd frontend
npm install
npm start
```

The app opens automatically. The backend runs on `http://127.0.0.1:27182` by default. The port number 27182 corresponds to the first digits of *e*.

Every environment variable read by the backend, including `HF_TOKEN` for gated models, `ERUDI_FORCE_CPU`, and `ERUDI_LOG_LEVEL`, is documented in [`backend/.env.example`](backend/.env.example). Copy it to the git-ignored `backend/.env` file if you need custom values.

---

## Building for Distribution

### Windows (NVIDIA GPU)

```powershell
.\scripts\build\build-win-cuda.ps1
```

The installer is generated at:

```text
frontend/dist/Erudi Setup <version>.exe
```

### macOS (Apple Silicon)

```bash
bash scripts/build/build-mac-silicon.sh
```

Pushing a `vX.Y.Z` tag builds every platform in CI and publishes a draft release. macOS builds are
signed and notarized on that leg; **Windows and Linux artifacts are currently unsigned**, so those
installers show an unknown-publisher warning. See [`BUILD.md`](BUILD.md) for the signing setup.

---

## Project Structure

```text
erudi/
├── backend/                  # Python FastAPI backend
│   ├── src/
│   │   ├── engines/          # Hardware backends (CUDA, CPU, MLX)
│   │   ├── domains/          # API domains (conversations, llms, knowledge_base…)
│   │   └── entities/         # SQLAlchemy models
│   ├── backend.spec          # PyInstaller build spec (Windows and Linux CUDA)
│   └── run.py                # Entry point
├── frontend/                 # Electron + React frontend
│   ├── src/
│   │   ├── main.js           # Electron main process
│   │   ├── pages/            # React pages
│   │   └── components/       # React components
│   └── electron-builder.yml  # Packaging, signing and update-feed config
├── scripts/
│   ├── dev/backend/          # Development environment setup scripts
│   └── build/                # Distribution build scripts
└── docs/                     # Architecture and build notes
```

---

## Architecture

The app has two processes:

- **Electron frontend** — React UI running in a `BrowserWindow`
- **Python backend** — FastAPI server running as `backend.exe` or a `backend` binary in production, and through `python run.py` in development

The backend selects an inference engine at startup:

```text
macOS ARM                 → MLX_Engine  (MLX framework)
macOS x86                 → CPU_Engine  (llama-server, CPU only)
Windows/Linux + NVIDIA    → CUDA_Engine (llama-server with CUDA offload)
Windows/Linux, no NVIDIA  → CPU_Engine  (llama-server, CPU only)
```

All inference engines run an OpenAI-compatible HTTP server in a child process and communicate with it through `http://127.0.0.1:<port>/v1/chat/completions`. Windows and Linux NVIDIA backends and the CPU fallback use `llama-server` from llama.cpp. macOS Apple Silicon uses `mlx_vlm.server` for vision and tool calling. PyTorch is never used for inference — only for the Knowledge Base embedding model (`multilingual-e5-small`), which runs on the CPU on Windows and Linux (those builds install torch from the CPU-only index) and on Metal on Apple Silicon. Conversation history and its rolling summary are handled by the LangGraph checkpointer, not by embeddings.

---

## Logs

| Platform | Log location |
|---|---|
| Windows | `%TEMP%\erudi-backend.log` |
| macOS | `$TMPDIR/erudi-backend.log` (a per-user folder under `/var/folders/…`, not `/tmp`) |
| Linux | `/tmp/erudi-backend.log` |

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and check the open issues for good first tasks.

---

## License

Erudi is licensed under the [MIT License](LICENSE).
