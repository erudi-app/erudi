#!/usr/bin/env python3
"""Fail unless this interpreter holds exactly the packages of a lock file.

Usage: python scripts/ci/check_installed_lock.py backend/requirements/lock/<entrypoint>.txt

scripts/ci/install_backend_requirements.sh runs it right after installing an
entrypoint constrained to its lock. Every locked package must be installed at
its locked version, and every installed package must be locked, except pip,
setuptools and wheel, which the install script brings itself (when the lock
names setuptools, as torch requires it, its version is checked like any other).
A difference means the lock no longer describes what the entrypoint installs:
a requirement changed without the locks being regenerated, or the lock was
generated with the wrong markers for this platform.
"""

from __future__ import annotations

import re
import sys
from importlib import metadata
from pathlib import Path

BOOTSTRAP_TOOLS = {"pip", "setuptools", "wheel"}
REGENERATE_COMMAND = "python3.12 scripts/dev/lock_backend_requirements.py"


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def read_lock(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            name, _, version = line.partition("==")
            pins[canonical_name(name)] = version
    return pins


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(f"usage: {Path(__file__).name} <lock file>", file=sys.stderr)
        return 2
    lock = Path(argv[0])
    locked = read_lock(lock)
    installed = {
        canonical_name(dist.metadata["Name"]): dist.version for dist in metadata.distributions()
    }

    differences = []
    for name, version in sorted(locked.items()):
        if name not in installed:
            differences.append(f"{name}: locked at {version}, not installed")
        elif installed[name] != version:
            differences.append(f"{name}: locked at {version}, {installed[name]} installed")
    for name in sorted(set(installed) - set(locked) - BOOTSTRAP_TOOLS):
        differences.append(f"{name}: {installed[name]} installed, not in the lock")

    if differences:
        print(
            f"::error::The installed backend packages differ from {lock}. The lock no longer "
            f"describes what its entrypoint installs: regenerate it with `{REGENERATE_COMMAND}` "
            "and commit the result (backend/requirements/README.md)."
        )
        for line in differences:
            print(f"  {line}")
        return 1
    print(f"{len(locked)} packages installed exactly as {lock} locks them.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
