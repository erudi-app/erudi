"""Produce the KB embedding GGUF the Windows/Linux installers bundle (#594).

The Knowledge Base embedding child on Windows and Linux is the bundled CPU
``llama-server --embedding`` serving ``multilingual-e5-small`` as a Q8_0
GGUF (worst cosine against the fp32 model 0.99981 over the spike-S4
corpus). Community e5 GGUFs carry a WRONG WordPiece vocab (cosine ~0.897),
so Erudi converts its own, reproducibly:

1. the e5 snapshot is fetched at the PINNED revision into the script's own
   cache (``backend/data/e5-build-cache``; already-present files are reused).
   Not the backend's ``models_cache``: on Windows and Linux the backend
   deletes the e5 snapshot it finds there at start, once the GGUF exists;
2. the conversion toolchain of the pinned llama.cpp submodule is copied to
   a scratch directory and ``e5-conversion.patch`` (kept beside this
   script) is applied THERE -- the submodule is never modified;
3. ``convert_hf_to_gguf.py --outtype q8_0 --model-name multilingual-e5-small``
   runs with the scratch ``gguf-py`` first on ``PYTHONPATH``;
4. the result's sha256 must equal the pinned ``E5_GGUF_SHA256`` -- any
   drift (snapshot, converter, numpy quantization) fails the build loudly;
5. the file lands at ``backend/artifacts/kb-embedder/`` where
   ``pyi_llama_artifacts`` bundles it and ``src/ingestion/llama_embedder.py``
   serves it (dev runs read the same path).

Run it from the repo root with the CONVERSION venv, never the app's: the
converter needs torch, transformers, sentencepiece and protobuf, and the
app's locks carry neither torch nor sentencepiece nor protobuf. The venv
comes from the per-OS build lock (hash-checked, reproducible on a bare
runner):

    python3.12 -m venv e5-gguf-venv
    e5-gguf-venv/bin/pip install -r backend/requirements/lock/linux-e5-gguf.txt \
        --require-hashes --no-deps          # Windows: Scripts\\pip, win-e5-gguf.txt
    e5-gguf-venv/bin/python scripts/dev/backend/build_e5_gguf.py   # build + verify
    ... build_e5_gguf.py --print-sha   # after a deliberate change (new
        revision/patch): print the new hash to pin

There is no macOS build lock: on a Mac, install
``backend/requirements/meta/e5-gguf.txt`` into a venv of its own. CI
(release.yml, app-build-smoke.yml) builds that venv and runs this script
behind an ``actions/cache`` keyed on this file and the patch, so the
conversion is paid once per change and OS.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND = REPO_ROOT / "backend"
SUBMODULE = BACKEND / "forks" / "llama-cpp"
PATCH_FILE = Path(__file__).resolve().parent / "e5-conversion.patch"

E5_REPO_ID = "intfloat/multilingual-e5-small"
# The snapshot the app always embedded with (hub `main` on 2026-11-10); the
# GGUF must come from the same weights so existing KBs keep their vectors.
E5_SNAPSHOT_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
# The script's own snapshot cache, apart from the backend's models_cache (see
# step 1 of the module docstring). backend/data/ is gitignored.
E5_BUILD_CACHE = BACKEND / "data" / "e5-build-cache"

OUTPUT_NAME = "multilingual-e5-small-q8_0.gguf"
OUTPUT_FILE = BACKEND / "artifacts" / "kb-embedder" / OUTPUT_NAME

# sha256 of the Q8_0 GGUF produced from the pinned snapshot with the pinned
# submodule + patch. Pinned on arm64 and verified on x86 by CI: the
# quantization is deterministic numpy float32 arithmetic, so the file is
# expected bit-identical across build hosts -- a mismatch means the inputs
# drifted (or that expectation broke, which release engineering must hear
# about either way).
E5_GGUF_SHA256 = "2cab2fb5e05ebc65c9db1e0b78e474f277e9afc1ceebb6b2ab521cafca97acb1"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_conversion_deps() -> None:
    """Fail up front, naming the venv to build, instead of mid-conversion."""
    import importlib.util

    missing = [
        name
        for name in ("torch", "transformers", "sentencepiece", "google.protobuf")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise SystemExit(
            f"[build_e5_gguf] missing conversion dependencies: {', '.join(missing)}. "
            f"Run this script with the dedicated conversion venv, installed from "
            f"backend/requirements/lock/<win|linux>-e5-gguf.txt (see the module "
            f"docstring and BUILD.md)."
        )


def fetch_snapshot() -> Path:
    from huggingface_hub import snapshot_download

    print(f"[build_e5_gguf] snapshot {E5_REPO_ID} @ {E5_SNAPSHOT_REVISION[:12]}")
    return Path(
        snapshot_download(
            E5_REPO_ID,
            revision=E5_SNAPSHOT_REVISION,
            cache_dir=str(E5_BUILD_CACHE),
        )
    )


def stage_converter(scratch: Path) -> Path:
    """Copy the conversion toolchain out of the submodule and patch the copy."""
    if not (SUBMODULE / "convert_hf_to_gguf.py").is_file():
        raise SystemExit(
            "[build_e5_gguf] the llama.cpp submodule is not initialized; run: "
            "git submodule update --init backend/forks/llama-cpp"
        )
    shutil.copy2(SUBMODULE / "convert_hf_to_gguf.py", scratch / "convert_hf_to_gguf.py")
    shutil.copytree(
        SUBMODULE / "conversion",
        scratch / "conversion",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(
        SUBMODULE / "gguf-py",
        scratch / "gguf-py",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    # The patch is stored LF (.gitattributes: `*.patch -text`); a Windows
    # checkout of the submodule can hold bert.py with CRLF endings
    # (core.autocrlf), where every context line would mismatch. Normalizing
    # the COPY to LF makes the apply end-of-line-proof and changes no code.
    target = scratch / "conversion" / "bert.py"
    target.write_bytes(target.read_bytes().replace(b"\r\n", b"\n"))
    subprocess.run(
        ["git", "apply", "--verbose", str(PATCH_FILE)],
        cwd=scratch,
        check=True,
    )
    return scratch / "convert_hf_to_gguf.py"


def convert(snapshot: Path, converter: Path, out_file: Path) -> None:
    env = os.environ.copy()
    gguf_py = str(converter.parent / "gguf-py")
    env["PYTHONPATH"] = gguf_py + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [
            sys.executable,
            str(converter),
            str(snapshot),
            "--outtype",
            "q8_0",
            "--outfile",
            str(out_file),
            # Without it the metadata picks the snapshot directory's name up
            # as general.finetune and config.json's "tmp/" as the org.
            "--model-name",
            "multilingual-e5-small",
        ],
        check=True,
        env=env,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--print-sha",
        action="store_true",
        help="build, print the result's sha256 and keep it WITHOUT verifying "
        "(for pinning after a deliberate revision/patch change)",
    )
    args = parser.parse_args()

    if OUTPUT_FILE.is_file() and not args.print_sha:
        found = sha256_of(OUTPUT_FILE)
        if found == E5_GGUF_SHA256:
            print(f"[build_e5_gguf] up to date: {OUTPUT_FILE}")
            return 0
        print(f"[build_e5_gguf] present but stale ({found[:12]}...); rebuilding")

    check_conversion_deps()
    snapshot = fetch_snapshot()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="e5-gguf-") as scratch_name:
        scratch = Path(scratch_name)
        converter = stage_converter(scratch)
        staged = scratch / OUTPUT_NAME
        convert(snapshot, converter, staged)
        found = sha256_of(staged)
        if args.print_sha:
            print(f"[build_e5_gguf] sha256 to pin: {found}")
        elif found != E5_GGUF_SHA256:
            raise SystemExit(
                f"[build_e5_gguf] sha256 mismatch for {OUTPUT_NAME}:\n"
                f"  expected {E5_GGUF_SHA256}\n"
                f"  got      {found}\n"
                f"The pinned snapshot, the submodule, the patch or the "
                f"quantization arithmetic changed. If the change is "
                f"deliberate, re-pin with --print-sha and re-measure parity "
                f"(scripts/dev/backend/capture_e5_golden.py, "
                f"tests/test_llama_embedder_real.py)."
            )
        # Atomic within the destination directory: the app never sees a
        # half-written file at the path it serves.
        staging = OUTPUT_FILE.with_suffix(".gguf.tmp")
        shutil.move(str(staged), str(staging))
        staging.replace(OUTPUT_FILE)
    size_mib = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"[build_e5_gguf] wrote {OUTPUT_FILE} ({size_mib:.1f} MiB, sha256 {found[:12]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
