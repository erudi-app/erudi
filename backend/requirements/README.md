# Erudi Backend — Requirements

Dependencies are split per platform/hardware to keep each build minimal and
reproducible. Composition is layered: an **entrypoint** pulls in shared **meta**
modules (`-r ...`).

## Layout

```
requirements/
├── entrypoints/
│   ├── dev/                  # prod + dev tools (pytest, ruff, …)
│   │   ├── mac-silicon.txt
│   │   ├── linux-cpu.txt
│   │   ├── linux-cuda.txt
│   │   ├── win-cpu.txt
│   │   └── win-cuda.txt
│   └── prod/                 # minimal runtime deps (one per platform, *-prod.txt)
├── meta/
│   ├── base.txt              # shared core: FastAPI, SQLAlchemy, pgserver,
│   │                         #   langchain, sentence-transformers, transformers (pinned 5.14.1)
│   ├── dev.txt               # pytest / ruff (linter + formatter) / mkdocs
│   ├── cpu.txt               # CPU torch (official CPU index) + gguf — REUSED by the CUDA entrypoints
│   ├── cuda-specs.txt        # CUDA-only non-torch bits (pynvml). No torch+cuXXX.
│   ├── cuda-win-specs.txt    # Windows CUDA build tools (cmake)
│   ├── linux-specs.txt
│   ├── mac-silicon-specs.txt # MLX (mlx-vlm + mlx, both pinned)
│   └── win-specs.txt         # Windows-only (currently none)
```

## Key design notes

- **Inference is llama.cpp / MLX, not torch.** torch is only pulled (CPU build)
  by sentence-transformers for the e5 KB embeddings. There is **no torch+CUDA**:
  the CUDA build uses **CPU torch** plus a CUDA-compiled `llama-server` binary
  (built by `scripts/dev/backend/build-llamacpp-cuda-*`). The CUDA toolkit version
  lives in the **binary build**, not in pip — which is why there is a single
  `cuda` entrypoint per OS (no 118/121 split anymore).
- **transformers is pinned once in `base.txt` (5.14.1)** for every platform. The
  floor comes from mlx-vlm 0.6.17 (`transformers>=5.14.0`); it is coupled to
  torch (imports `torch.float8_e8m0fnu`, needing torch>=2.7).
- **Transitive packages are pinned to the tested build in `base.txt`** (and
  `miniaudio` in `mac-silicon-specs.txt`). A build bundles whatever pip
  resolves on the day, and an unsigned `.pyd` published hours before a build
  has no reputation yet: Windows Smart App Control refuses to load it. A
  rebuild must bundle what was tested, so any transitive package that moves
  between the tested build and a rebuild is pinned back (compare the
  `Successfully installed` lines of the two release runs).
- Fine-tuning deps (peft/accelerate/datasets/bitsandbytes) were removed — the
  feature is unimplemented dead code (see the fine-tuning cleanup issue).

## Usage

```bash
# Dev (with testing/linting tools)
pip install -r requirements/entrypoints/dev/<platform>.txt
# Prod (minimal)
pip install -r requirements/entrypoints/prod/<platform>-prod.txt
```

Or use the platform setup scripts in `scripts/dev/backend/` (they default to prod
in CI; set `INSTALL_TYPE=dev|prod` to force).

## Adding a dependency

- Shared (all platforms) → `meta/base.txt`
- Dev tool → `meta/dev.txt`
- Platform / hardware-specific → the matching `meta/*-specs.txt`

## Changing a version

Every version here is an exact pin, so bumping one can make the whole set
unsolvable — a library's new release may tighten a range around a sibling that
nobody thought to touch.

**Installing into your existing venv does not prove the set resolves.** pip
patches what is already there: it will quietly upgrade a transitive dependency
to satisfy the package you asked for, leave the pin file describing a state no
environment ever ran, and report success. CI builds a venv from nothing and
resolves properly, so it fails where you did not.

Check a bump the way CI will, in a throwaway environment:

```bash
python3.12 -m venv /tmp/resolvenv
/tmp/resolvenv/bin/python -m pip install --dry-run \
    -r requirements/entrypoints/dev/<platform>.txt
```

`--dry-run` resolves and prints what it *would* install without building
anything, so it costs metadata rather than gigabytes. Read the resulting
versions: any package that comes out at something other than its pin is a pin
that needs updating in the same commit.
