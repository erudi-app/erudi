#!/usr/bin/env python3
"""Check a backend lock (backend/requirements/lock/) against this machine.

Usage:
  python scripts/ci/check_backend_lock.py installed <lock>
  python scripts/ci/check_backend_lock.py resolution <prod entrypoint>
  python scripts/ci/check_backend_lock.py constraints <lock>

installed   This interpreter must hold exactly the lock's packages at their
            locked versions, pip aside (the venv brings it).
            scripts/ci/install_backend_requirements.sh runs it in the venv it
            installs the lock into.
resolution  Run by backend-ci on each OS, for the prod entrypoints of that OS,
            with this machine's own pip and markers, no install: (1) the
            entrypoint and meta/build.txt, constrained to the lock's versions,
            resolve to exactly the lock's set; (2) installing the lock in
            hash-checking mode selects the locked file of every package. The
            CUDA locks are installed by a release tag only, so this is what
            proves them on every pull request.
constraints The lock's `name==version` lines, without hashes, for `pip -c`: a
            hash in a constraints file turns on hash-checking for the whole
            install. backend-ci installs the dev entrypoints within them.

A difference fails with the command that regenerates the locks.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "dev"))

import lock_backend_requirements as locks  # noqa: E402

# What a fresh venv holds before anything is installed into it.
VENV_TOOLS = {"pip"}


def compare(locked: dict[str, str], found: dict[str, str], what: str) -> list[str]:
    """Every difference between the locked versions and the ones found."""
    differences = []
    for name, version in sorted(locked.items()):
        if name not in found:
            differences.append(f"{name}: locked at {version}, not {what}")
        elif found[name] != version:
            differences.append(f"{name}: locked at {version}, {found[name]} {what}")
    for name in sorted(set(found) - set(locked)):
        differences.append(f"{name}: {found[name]} {what}, not in the lock")
    return differences


def _fail(subject: str, differences: list[str]) -> int:
    print(
        f"::error::{subject}. Regenerate the locks with `{locks.REGENERATE_COMMAND}` and "
        "commit them (backend/requirements/README.md)."
    )
    for line in differences:
        print(f"  {line}")
    return 1


def installed(lock_file: Path) -> int:
    lock = locks.read_lock(lock_file)
    found = {
        locks.canonical_name(dist.metadata["Name"]): dist.version
        for dist in metadata.distributions()
    }
    for tool in VENV_TOOLS - set(lock.pins):
        found.pop(tool, None)
    differences = compare(lock.pins, found, "installed")
    if differences:
        return _fail(f"The installed backend packages differ from {lock_file}", differences)
    print(f"{len(lock.pins)} packages installed exactly as {lock_file} locks them.")
    return 0


def _report(arguments: list[str], work: Path, name: str) -> list[dict]:
    report = work / f"{name}.json"
    command = [sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed"]
    command += ["--quiet", "--disable-pip-version-check", "--report", str(report), *arguments]
    subprocess.run(command, check=True, cwd=REPO_ROOT)
    return json.loads(report.read_text(encoding="utf-8"))["install"]


def resolution(entrypoint: Path) -> int:
    lock_file = locks.lock_path(entrypoint)
    lock = locks.read_lock(lock_file)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        constraints = work / "constraints.txt"
        constraints.write_text(_constraints(lock), encoding="utf-8")
        resolved = _report(
            ["-r", str(entrypoint), "-r", str(locks.build_requirements()), "-c", str(constraints)],
            work,
            "resolution",
        )
        found = {
            locks.canonical_name(item["metadata"]["name"]): item["metadata"]["version"]
            for item in resolved
        }
        differences = compare(lock.pins, found, "resolved here")
        if differences:
            return _fail(f"{entrypoint.name} does not resolve here to {lock_file}", differences)

        selected = _report(
            ["--require-hashes", "--no-deps", "--only-binary=:all:", "-r", str(lock_file)],
            work,
            "files",
        )
    differences = []
    for item in selected:
        name = locks.canonical_name(item["metadata"]["name"])
        hashes = item.get("download_info", {}).get("archive_info", {}).get("hashes", {})
        # An index that gives no hash (the PyTorch index's copy of Jinja2): pip
        # checks the downloaded file against the lock at install time instead.
        if "sha256" in hashes and hashes["sha256"] != lock.hashes[name]:
            differences.append(f"{name}: pip selects {item['download_info']['url']}")
    if differences:
        return _fail(f"Installing {lock_file} here takes other files than it locks", differences)
    print(f"{entrypoint.name}: resolves to {lock_file} here, and its locked files are selected.")
    return 0


def _constraints(lock) -> str:
    return "".join(f"{name}=={version}\n" for name, version in sorted(lock.pins.items()))


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] not in {"installed", "resolution", "constraints"}:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    mode, path = argv[0], Path(argv[1])
    if mode == "installed":
        return installed(path)
    if mode == "resolution":
        return resolution(path)
    sys.stdout.write(_constraints(locks.read_lock(path)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
