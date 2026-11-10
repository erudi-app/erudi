# -*- mode: python ; coding: utf-8 -*-
"""
Erudi Backend - PyInstaller spec (cross-platform: Windows CUDA / macOS Apple Silicon)

Windows / Linux output (CUDA variant; backend-cpu.spec ships cpu/bin alone):
    backend/dist/backend/backend[.exe]
    backend/dist/backend/artifacts/llama-cpp/cuda/bin/   <- llama-server for CUDA_Engine
    backend/dist/backend/artifacts/llama-cpp/cpu/bin/    <- llama-server for CPU_Engine

macOS output:
    backend/dist/backend/backend            <- spawned by Electron main.js
    (MLX inference via mlx_vlm, no llama-server needed)

Build from backend/:
    Windows:  venv\\Scripts\\pyinstaller backend.spec
    macOS:    venv/bin/pyinstaller backend.spec
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")

spec_root = Path(SPECPATH)  # resolves to backend/

# ── Collected data / binaries ─────────────────────────────────────────────────
datas = []
binaries = []
hiddenimports = []

# torch: collect only config data files, NOT the CUDA DLLs.
datas += collect_data_files("torch", includes=[
    "**/*.json",
    "**/*.yaml",
    "**/*.pyi",
])

# transformers: tokenizer configs, model cards, generation configs
datas += collect_data_files("transformers")

# sentence_transformers: module configs, pooling configs
datas += collect_data_files("sentence_transformers")

# tqdm: needs its locale data on some systems
datas += collect_data_files("tqdm")

# filelock (used by transformers / huggingface_hub)
datas += collect_data_files("filelock")

# py3langid: the language-ID model (model.plzma) ships as package data and is
# loaded by src/agents/language.py at runtime. Shared across all platforms:
# without it the systematic KB path (build_kb_context_block -> detect_language)
# dies with FileNotFoundError on .../py3langid/data/model.plzma.
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
# engine format, so first boot loads it instantly (zero HF calls). Windows/Linux
# bundles use the GGUF snapshot. Guarded: a missing snapshot just falls back to the
# offline JSON instead of breaking the build.
_gguf_snapshot = spec_root / "src" / "database" / "catalog_snapshot_gguf.json"
if _gguf_snapshot.exists():
    datas.append((str(_gguf_snapshot), "src/database"))

# pgserver: bundle the embedded PostgreSQL binaries (pginstall/bin). The
# hiddenimport alone ships the Python module but NOT the postgres binaries it
# spawns; without this the frozen backend dies at startup with a missing
# pgserver/pginstall/bin. collect_all is platform-agnostic — on the Windows
# runner it picks up the Windows postgres binaries. (Proven on the mac build;
# the libpq runtime hook used there is dyld-specific and is NOT wired here —
# Windows DLL resolution differs and needs its own validation on a Win runner.)
tmp_ret = collect_all("pgserver")
datas += tmp_ret[0]; hiddenimports += tmp_ret[2]
if IS_LINUX:
    # On Linux, PyInstaller's binary analysis drops/relocates postgres's loadable
    # backend modules ($libdir/*.so, e.g. dict_snowball — they depend on the server
    # binary, not standalone libs), so initdb fails to bootstrap ("could not access
    # file $libdir/dict_snowball"). Bundle the whole pginstall tree verbatim as DATA
    # (no dependency analysis): postgres runs as a subprocess, not linked into the
    # frozen Python, so its install is opaque data and $libdir resolves relative to
    # the bundled binary. (collect_all's pgserver BINARIES are skipped here.)
    import pgserver as _pgsrv
    _pginstall = Path(_pgsrv.__file__).resolve().parent / "pginstall"
    for _pf in _pginstall.rglob("*"):
        if _pf.is_file():
            datas.append((str(_pf), str(Path("pgserver/pginstall") / _pf.relative_to(_pginstall).parent)))
else:
    binaries += tmp_ret[1]

# ── llama.cpp inference artifacts (Windows + Linux — llama.cpp engines) ────────
# What ships follows the build variant, decided in pyi_llama_artifacts (unit
# tested on every CI leg): the cpu spec (backend-cpu.spec sets
# ERUDI_BUILD_VARIANT=cpu) ships artifacts/llama-cpp/cpu/bin; the standalone
# (CUDA) spec ships BOTH cuda/bin and cpu/bin, because CUDA_Engine runs the
# CUDA build and the processor engine (the "Processor only" setting) runs the
# CPU build, which needs no NVIDIA driver and touches no GPU. The same two specs
# serve Windows and Linux; mac is MLX (backend-mac-silicon.spec) and bundles
# nothing here. Every flavour is compiled from the llama.cpp submodule before
# PyInstaller runs (release.yml). A missing flavour fails the build, naming its
# build script, unless the boot-only merge smoke opts out with
# ERUDI_ALLOW_MISSING_INFERENCE_BINARY (it never compiles llama.cpp; inference
# is out of its scope).
#
# The runtime DLLs beside each server (MSVC C++ runtime, plus cudart/cublas/
# cublasLt for CUDA) are NOT appended to `datas`: PyInstaller reclassifies PE
# files found in `datas` as binaries, so these land in `a.binaries` — where the
# Windows torch-CUDA strip below removes every name matching "cublaslt", taking
# the inference runtime's cublasLt64_12.dll with torch's. These files are opaque
# cargo for the subprocess, not link targets of the frozen app: they are appended
# to `a.datas` AFTER Analysis and after the strip (`_llama_runtime_dlls`, below),
# which copies them verbatim and keeps them out of both mechanisms. The release
# workflow's "Verify the bundled inference binary" step guards the outcome.
#
# PyInstaller runs a spec through exec() and does not put its directory on
# sys.path, so the helper beside it is reached explicitly.
if str(spec_root) not in sys.path:
    sys.path.insert(0, str(spec_root))
from pyi_llama_artifacts import collect_llama_artifacts

_llama = collect_llama_artifacts(
    variant=os.environ.get("ERUDI_BUILD_VARIANT", "cuda"),
    root=spec_root,
    os_tag="win" if IS_WIN else "linux" if IS_LINUX else "mac",
    allow_missing=bool(os.environ.get("ERUDI_ALLOW_MISSING_INFERENCE_BINARY")),
)
datas += _llama.datas
_llama_runtime_dlls = _llama.runtime_files
for _absent in _llama.skipped:
    print(
        f"[backend.spec] llama-server absent at {_absent} - continuing because "
        f"ERUDI_ALLOW_MISSING_INFERENCE_BINARY is set. The bundle produced here "
        f"CANNOT run inference and must not be released."
    )

# ── Analysis ──────────────────────────────────────────────────────────────────
_hidden_common = [
    # ── uvicorn internals
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
    # ── SQLAlchemy dialect
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
    # ── NVIDIA GPU detection (used by base_engine; graceful no-op on macOS)
    "pynvml",
    # ── Async I/O helpers
    "anyio",
    "anyio._backends._asyncio",
    "anyio._backends._trio",
    "sniffio",
    "h11",
    "exceptiongroup",
    # ── Our own modules
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
    "src.engines.cuda_engine",
    "src.engines.cpu_engine",
    "src.launcher",
    "src.launcher.runtime_paths",
    "src.launcher.windows_job",
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
    # ── stdlib
    "multiprocessing",
    "multiprocessing.util",
    "multiprocessing.managers",
    "asyncio",
    "importlib_metadata",
    "email.mime.multipart",
    "email.mime.text",
]

_hidden_windows = [
    "asyncio.windows_events",   # Windows-specific event loop policy
]

_hidden_macos = [
    "src.engines.mlx_engine",   # Apple Silicon MLX inference engine
    "mlx",
    "mlx_vlm",
    "src.engines._mlx_vlm_server_runner",  # picklable mp.Process target
    "src.engines.mlx_child_log",  # the child redirects its own stdout/stderr through it
    "mlx_vlm.server",           # uvicorn loads "mlx_vlm.server:app"
]

# += (NOT =): hiddenimports already carries pgserver's collect_all submodules
# gathered above — a plain reassignment silently discarded them (#181).
hiddenimports += _hidden_common
if IS_WIN:
    hiddenimports += _hidden_windows
if IS_MAC:
    hiddenimports += _hidden_macos
else:
    # llama.cpp builds (Windows/Linux, CPU + CUDA): transformers reads the GGUF
    # chat template through the `gguf` package via a LAZY import
    # (AutoTokenizer(gguf_file=...)), invisible to PyInstaller's static analysis.
    # Without it, tool-calling detection fails for every downloaded GGUF and the
    # agentic KB mode can never activate (#171). Not on macOS: the MLX env does
    # not install gguf (requirements/meta/cpu.txt is not in mac-silicon-prod).
    hiddenimports += ["gguf"]
    # The module alone is NOT enough: transformers gates the GGUF path through
    # is_gguf_available(), which reads the package VERSION via importlib.metadata.
    # Without the dist-info the version resolves to 'N/A' and version.parse()
    # raises InvalidVersion — same product symptom as the missing module (#206).
    datas += copy_metadata("gguf")
# Alembic loads its dialect ddl (alembic.ddl.postgresql) and other submodules
# dynamically — collect them so the startup migration runs in the frozen build.
hiddenimports += collect_submodules("alembic")

_excludes_common = [
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
]

# MLX is Apple-Silicon only — exclude it on the llama.cpp platforms (Windows + Linux).
_excludes_llamacpp = [
    "mlx",
    "mlx_vlm",
]

_excludes_macos = [
    # CUDA packages are Windows-only; pynvml is imported but gracefully no-ops on Mac
]

excludes = _excludes_common
if IS_WIN or IS_LINUX:
    excludes += _excludes_llamacpp
if IS_MAC:
    excludes += _excludes_macos
# Variant-specific extras injected by a wrapper spec that exec()s this template in
# its own namespace (e.g. backend-cpu.spec sets ERUDI_EXTRA_EXCLUDES). Empty for
# the standalone specs. OS-independent, so a CPU build excludes mlx_vlm even off Windows.
excludes += globals().get("ERUDI_EXTRA_EXCLUDES", [])

# Linux: preload psycopg's own (newer) libpq so it does not bind to pgserver's
# older bundled libpq (the same @rpath/.so collision the mac spec fixes). No-op on
# Windows (the glob patterns match no .dll), so it is wired Linux-only here.
_runtime_hooks = [str(spec_root / "pyi_rth_libpq.py")] if IS_LINUX else []

a = Analysis(
    ["run.py"],
    pathex=[str(spec_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=_runtime_hooks,
    excludes=excludes,
    # python-docx opens its default parts (styles, settings, header, footer,
    # comments) through <docx/parts>/../templates/. "pyz" alone leaves no
    # docx/parts/ directory on disk, so a .docx saved without a styles part
    # (TextEdit) fails with ENOENT on macOS and Linux (#705); only Windows
    # normalises the `..` lexically. "pyz+py" adds the sources.
    module_collection_mode={"docx": "pyz+py"},
    noarchive=False,
    optimize=0,
)

# ── Strip CUDA DLLs (Windows only) ────────────────────────────────────────────
# PyInstaller's torch hook auto-collects CUDA DLLs into a.binaries; they add
# ~3.8 GB but the packaged embedder runs CPU-only. Strip them on Windows.
if IS_WIN:
    _cuda_fragments = (
        "torch_cuda", "cublaslt", "cufft", "curand",
        "cusolver", "cusolvermg", "cusparse", "cudnn",
        "nvrtc", "nvjitlink", "nvjpeg", "nvperf",
        "cupti", "nvtoolsext",
    )
    a.binaries = [
        b for b in a.binaries
        if not any(frag in b[0].replace("\\", "/").lower() for frag in _cuda_fragments)
    ]

# ── llama-server runtime DLLs, verbatim ───────────────────────────────────────
# Collected next to the compiled server by the build scripts and listed above
# (see the llama.cpp block for why they must not ride in `datas`): appended
# here, after Analysis and after the strip, as plain TOC data entries that
# nothing re-analyses or filters. COLLECT copies them beside llama-server,
# which is the only place Windows looks when the subprocess loads them.
for _name, _src in _llama_runtime_dlls:
    a.datas.append((_name, _src, "DATA"))

pyz = PYZ(a.pure)

# Interpreter options for the frozen runtime. PyInstaller's bootloader
# pre-initializes CPython itself and IGNORES PYTHONUTF8 from the environment
# (documented PyInstaller behavior), so UTF-8 mode must be forced here.
# Without it sys.stdout/stderr inherit the locale code page (cp1252 on
# Windows), and any Unicode character in a log line kills the console log
# handler with "--- Logging error ---" (#168). open() also defaults to UTF-8
# under this mode, which is the systemic guard #149/#150 intended.
_interpreter_options = [("X utf8_mode=1", None, "OPTION")]

exe = EXE(
    pyz,
    a.scripts,
    _interpreter_options,
    exclude_binaries=True,
    name="backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,      # Must be True: Electron reads stdout JSON events
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[
        "*.dll",
        "vcruntime*.dll",
        "msvcp*.dll",
        "api-ms-*.dll",
        "ucrtbase.dll",
    ],
    name="backend",
)
