# -*- mode: python ; coding: utf-8 -*-
"""
Erudi Backend - PyInstaller spec for macOS Apple Silicon (arm64)

Bundles the FastAPI backend with MLX inference support into a
one-directory executable at backend/dist/backend/.

Prerequisites:
    cd backend
    venv/bin/pip install pyinstaller
    venv/bin/pyinstaller backend-mac-silicon.spec

Output:
    backend/dist/backend/backend   <- spawned by Electron main.js

The bundle root (dist/backend/) is set as ROOT_DIR by run.py, so all
relative paths (data/) resolve correctly at runtime.
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

spec_root = Path(SPECPATH)  # resolves to backend/

# ── Collected data / binaries ─────────────────────────────────────────────────
datas = []
binaries = []
hiddenimports = []

# torch: CPU tensor ops for the embedder (sentence-transformers).
# MLX handles LLM inference independently; torch is CPU-only here.
datas += collect_data_files("torch", includes=[
    "**/*.json",
    "**/*.yaml",
    "**/*.pyi",
])

# transformers: tokenizer configs, model cards, generation configs
datas += collect_data_files("transformers")

# sentence_transformers: module configs, pooling configs
datas += collect_data_files("sentence_transformers")

# tqdm: locale data
datas += collect_data_files("tqdm")

# filelock (used by transformers / huggingface_hub)
datas += collect_data_files("filelock")

# py3langid: the language-ID model (model.plzma) ships as package data and is
# loaded by src/agents/language.py at runtime. Without it, the systematic KB
# path (build_kb_context_block -> detect_language) dies with
# FileNotFoundError: .../py3langid/data/model.plzma.
datas += collect_data_files("py3langid")

# Alembic migration scripts (#96): the startup runner resolves script_location to
# ROOT_DIR/alembic and the config to ROOT_DIR/alembic.ini. In the frozen build
# ROOT_DIR is the bundle root (sys._MEIPASS), so ship the whole alembic/ tree and
# alembic.ini there. Alembic reads these from the filesystem (not as Python package
# data), so they must be bundled as datas, structure preserved.
_alembic_dir = spec_root / "alembic"
for _af in _alembic_dir.rglob("*"):
    if _af.is_file() and "__pycache__" not in _af.parts:
        datas.append((str(_af), str(Path("alembic") / _af.relative_to(_alembic_dir).parent)))
datas.append((str(spec_root / "alembic.ini"), "."))
# Offline base-model fallback catalog: seed.load_base_models_fallback() reads it at
# ROOT_DIR/src/database/base_models_fallback.json on first boot (placeholder before
# the background HF refresh). Bundle it preserving that path, else boot can't seed.
datas.append((str(spec_root / "src" / "database" / "base_models_fallback.json"), "src/database"))

# Build-time catalog snapshot (#112): the full resolved remote catalog for this
# engine format, so first boot loads it instantly (zero HF calls). The macOS bundle
# uses the MLX snapshot. Guarded: a missing snapshot just falls back to the offline
# JSON instead of breaking the build.
_mlx_snapshot = spec_root / "src" / "database" / "catalog_snapshot_mlx.json"
if _mlx_snapshot.exists():
    datas.append((str(_mlx_snapshot), "src/database"))
# Alembic loads its dialect ddl (alembic.ddl.postgresql) and other submodules
# dynamically — collect them so the startup migration runs in the frozen build.
hiddenimports += collect_submodules("alembic")

# ── MLX: Apple Silicon inference framework ────────────────────────────────────
# collect_all captures compiled .so extensions, Metal shader kernels, and
# Python submodules that static analysis misses.
tmp_ret = collect_all("mlx")
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

tmp_ret = collect_all("mlx_vlm")
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# mlx_vlm loads the per-architecture model module by NAME at runtime
# (mlx_lm.utils._get_classes -> import mlx_lm.models.<arch>, e.g.
# gemma3_text), which static analysis misses. collect_all pulls every
# mlx_lm.models.* submodule so any downloaded model's arch resolves.
tmp_ret = collect_all("mlx_lm")
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# mlx-vlm 0.6.13 imports mlx_audio EAGERLY at server import time
# (mlx_vlm/server/audio.py: from mlx_audio.audio_io import read) - the 0.6.2-era
# exclusion made the frozen child crash before ready (#273 packaged smoke).
# 17 MB; collect_all also grabs its audio-codec binaries.
tmp_ret = collect_all("mlx_audio")
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# ── pgserver: bundle the embedded PostgreSQL binaries (pginstall/bin) ──────────
# The hiddenimport alone ships the Python module but NOT the postgres binaries it
# spawns; without this the frozen backend dies at startup with
# "No such file or directory: .../pgserver/pginstall/bin".
tmp_ret = collect_all("pgserver")
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# ── Analysis ──────────────────────────────────────────────────────────────────
a = Analysis(
    ["run.py"],
    pathex=[str(spec_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + [
        # ── uvicorn internals (loaded via importlib, missed by static analysis)
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.loops.asyncio",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.http.h11_impl",
        "uvicorn.protocols.http.httptools_impl",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.protocols.websockets.websockets_impl",
        "uvicorn.lifespan",
        "uvicorn.lifespan.off",
        "uvicorn.lifespan.on",
        # ── SQLAlchemy dialect (loaded by string at connect time)
        "sqlalchemy.dialects.postgresql",
        "sqlalchemy.dialects.postgresql.psycopg",
        "sqlalchemy.ext.asyncio",
        # ── Starlette / FastAPI internals
        "starlette.routing",
        "starlette.responses",
        "starlette.requests",
        "starlette.middleware.cors",
        "starlette.middleware.base",
        # ── Pydantic v2
        "pydantic",
        "pydantic.v1",
        "pydantic_core",
        # ── Async I/O helpers
        "anyio",
        "anyio._backends._asyncio",
        "anyio._backends._trio",
        "sniffio",
        "h11",
        "exceptiongroup",
        # ── Our own modules (some loaded dynamically via BaseEngine.get_engine)
        "src.main",
        "src.core.api",
        "src.core.config",
        "src.core.exceptions",
        "src.core.logging",
        "src.core.health",
        "src.database.core",
        "src.database.seed",
        "src.entities.Conversation",
        "src.entities.DownloadJob",
        "src.entities.HardwareProfile",
        "src.entities.KBJob",
        "src.entities.KnowledgeBase",
        "src.entities.Llm",
        "src.entities.Message",
        "src.entities.StartupVariables",
        "src.engines.base_engine",
        "src.engines.base_chat_server_engine",
        "src.engines.base_llama_cpp_engine",
        "src.engines.mlx_engine",
        "src.engines._mlx_vlm_server_runner",  # picklable target for mp.Process spawning mlx_vlm.server
        "src.engines.mlx_child_log",  # the child redirects its own stdout/stderr through it
        "mlx_vlm.server",  # imported lazily inside the runner (uvicorn loads "mlx_vlm.server:app")
        "mlx_vlm.server.app",  # the FastAPI app object uvicorn imports by string
        "mlx_vlm.server.cli",  # main() entrypoint the runner calls
        "src.engines.cpu_engine",
        "src.launcher",
        "src.launcher.runtime_paths",
        "src.utils.hf_model_metadata",
        "src.utils.kb_utils",
        "src.utils.prompt_utils",
        "src.domains.arena.endpoints",
        "src.domains.arena.repository",
        "src.domains.arena.services",
        "src.domains.arena.schemas",
        "src.domains.conversations.endpoints",
        "src.domains.conversations.repository",
        "src.domains.conversations.services",
        "src.domains.conversations.schemas",
        "src.agents.checkpoint",
        "src.agents.model_factory",
        "src.agents.prompts",
        "src.agents.runner",
        "src.agents.middleware",
        "src.agents.kb_mode",
        "src.agents.tools",
        # LangChain/LangGraph dynamic imports not always caught by static analysis.
        # The agent stack below is imported at FUNCTION scope only (#160 lazy
        # imports) — pinned explicitly so the first chat turn of a frozen build
        # can never lose it to an analysis change.
        "langchain.agents",
        "langchain.agents.middleware",
        "langchain.tools",
        "langchain_core.messages",
        "langchain_core.tools",
        "langchain_openai",
        # Ingestion splitters: imported at FUNCTION scope only (#160 lazy
        # imports, src/ingestion/chunking.py) — pinned for the same reason.
        "langchain_text_splitters",
        "langgraph.checkpoint.postgres",
        "langgraph.checkpoint.postgres.aio",
        "psycopg",
        "psycopg.pq",
        "psycopg_binary",
        "psycopg_pool",
        "pgserver",
        "langchain_postgres",
        "src.domains.hardware.endpoints",
        "src.domains.hardware.repository",
        "src.domains.hardware.services",
        "src.domains.hardware.schemas",
        "src.domains.knowledge_base.endpoints",
        "src.domains.knowledge_base.repository",
        "src.domains.knowledge_base.services",
        "src.domains.knowledge_base.schemas",
        "src.domains.llms.endpoints",
        "src.domains.llms.repository",
        "src.domains.llms.services",
        "src.domains.llms.schemas",
        "src.domains.startup.endpoints",
        "src.domains.startup.repository",
        "src.domains.startup.schemas",
        # ── ML / inference
        "numpy",
                "transformers",
        "sentence_transformers",
        "huggingface_hub",
        "tokenizers",
        "pypdf",
        "tqdm",
        "requests",
        "httpx",
        # ── stdlib occasionally missed by PyInstaller
        "multiprocessing",
        "multiprocessing.util",
        "multiprocessing.managers",
        "asyncio",
        "pkg_resources",
        "importlib_metadata",
        "email.mime.multipart",
        "email.mime.text",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(spec_root / "pyi_rth_libpq.py")],
    excludes=[
        # Windows-only / CUDA-only — not present on macOS
        "pynvml",
        "asyncio.windows_events",
        # llama-cpp is used for Windows inference only
        "llama_cpp",
        # Heavy GUI / data science packages not used at runtime
        "matplotlib",
        "IPython",
        "jupyter",
        "notebook",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "tkinter",
        "_tkinter",
        "cv2",
        "wx",
        # Test tooling
        "pytest",
        "black",
        "ruff",
        "mypy",
        # Training / quantization packages not used at inference runtime
        "bitsandbytes",
        "llmcompressor",
        "compressed_tensors",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX breaks Metal/MLX binaries on macOS
    console=True,       # Must be True: Electron reads stdout JSON events
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch="arm64",
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="backend",
)
