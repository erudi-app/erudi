#!/usr/bin/env python3
"""Lock what a release build installs, one file per prod entrypoint.

    python3.12 scripts/dev/lock_backend_requirements.py
    python3.12 scripts/dev/lock_backend_requirements.py --upgrade-package regex
    python3.12 scripts/dev/lock_backend_requirements.py --upgrade

For every file in backend/requirements/entrypoints/prod/ it resolves the
entrypoint together with backend/requirements/meta/build.txt (PyInstaller, the
build tool) and writes backend/requirements/lock/<same name>: every package of
the resolution as `name==version`, sorted. scripts/ci/install_backend_requirements.sh
installs both files constrained to the lock and fails when the installed set
differs, so a release build bundles the locked versions and nothing else.

Each entrypoint is resolved for the platform its build runs on, from any host:
`pip install --dry-run --report` with `--only-binary=:all: --platform <tags>
--python-version 3.12`. The tags are the ones the build runner's own pip
accepts: win_amd64; manylinux up to the glibc of the oldest Linux release runner;
macOS arm64 at the app's floor (mac.minimumSystemVersion in
frontend/electron-builder.yml), which is what the macOS leg installs. `--platform`
only selects wheels: pip still evaluates environment markers (`sys_platform`,
`platform_machine`, ...) against the interpreter running it, so a Windows lock
generated on macOS would miss `colorama; sys_platform == "win32"`. pip is
therefore run with its marker environment replaced by the target's, and the
result is verified: the dependency graph walked from the requested packages with
the target's markers must reach exactly the resolved set, at versions every
dependent accepts. A marker on `platform_release` or `platform_version` cannot be
emulated off the target and stops the run.

The resolution is seeded with the existing lock: its versions are passed as
constraints, so a regeneration only moves what the change requires. A locked
version that the requirement files no longer accept leaves the seed on its own;
anything else moves only when asked, with `--upgrade-package NAME` (repeatable)
or `--upgrade` (every package, newest that resolves).

Each lock records the digest of the requirement lines it was generated from
(comments and blank lines excluded); backend/tests/test_requirements_lock.py
fails when a requirement file changes without the locks being regenerated.

Stdlib only, plus pip, which it drives (`pip._vendor.packaging` evaluates the
markers and specifiers). Run it with Python 3.12, the backend's version: the
`python_version` markers are evaluated against it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS_DIR = REPO_ROOT / "backend" / "requirements"
PROD_DIR = REQUIREMENTS_DIR / "entrypoints" / "prod"
LOCK_DIR = REQUIREMENTS_DIR / "lock"
# Resolved and locked with every prod entrypoint; installed beside it by
# scripts/ci/install_backend_requirements.sh.
BUILD_REQUIREMENTS = Path("meta") / "build.txt"
ELECTRON_BUILDER = REPO_ROOT / "frontend" / "electron-builder.yml"

PYTHON_VERSION = (3, 12)
REGENERATE_COMMAND = "python3.12 scripts/dev/lock_backend_requirements.py"
DIGEST_PREFIX = "# inputs-sha256: "

# glibc of the oldest runner that installs a Linux lock: release.yml builds on
# ubuntu-22.04 (glibc 2.35); app-build-smoke.yml on ubuntu-latest, newer. A wheel
# tagged above it would not install there.
LINUX_GLIBC_MINOR = 35

# Markers that describe the running kernel: they cannot be emulated for another
# machine, so meeting one stops the run.
UNEMULATED_MARKERS = ("platform_release", "platform_version")

# pip's own comment rule for requirement files.
_COMMENT_RE = re.compile(r"(^|\s+)#.*$")
_INCLUDE_RE = re.compile(r"^(-r|--requirement|-c|--constraint)(\s+|=)(?P<path>\S+)$")
_LOCK_LINE_RE = re.compile(r"^(?P<name>[a-z0-9][a-z0-9.-]*)==(?P<version>\S+)$")

# Runs pip with the marker environment of the target platform. Marker.evaluate
# looks up default_environment() at call time, and the report module imports it
# by name, so the patch lands before pip's internals are imported.
_PIP_BOOTSTRAP = """\
import json, sys
import pip._vendor.packaging.markers as markers
_host = markers.default_environment
_target = json.loads(sys.argv[1])
def _target_environment():
    environment = _host()
    environment.update(_target)
    return environment
markers.default_environment = _target_environment
from pip._internal.cli.main import main
sys.exit(main(sys.argv[2:]))
"""


class LockError(Exception):
    """A lock cannot be generated; the message says why."""


@dataclass(frozen=True)
class Target:
    description: str
    platforms: tuple[str, ...]
    markers: dict[str, str]


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def prod_entrypoints() -> list[Path]:
    return sorted(PROD_DIR.glob("*.txt"))


def lock_path(entrypoint: Path) -> Path:
    return LOCK_DIR / entrypoint.name


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REQUIREMENTS_DIR.resolve()).as_posix()


def requirement_lines(path: Path) -> list[str]:
    """The lines pip reads from a requirement file: comments and blanks dropped."""
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = _COMMENT_RE.sub("", raw).strip()
        if line:
            lines.append(line)
    return lines


def build_requirements() -> Path:
    return REQUIREMENTS_DIR / BUILD_REQUIREMENTS


def input_files(entrypoint: Path) -> list[Path]:
    """What a lock is resolved from, in order: the entrypoint, the build tool's
    file, and every file either includes with -r/-c, recursively."""
    seen: list[Path] = []

    def visit(path: Path) -> None:
        path = path.resolve()
        if path in seen:
            return
        seen.append(path)
        for line in requirement_lines(path):
            match = _INCLUDE_RE.match(line)
            if match:
                visit(path.parent / match.group("path"))

    visit(entrypoint)
    visit(build_requirements())
    return seen


def inputs_digest(entrypoint: Path) -> str:
    digest = hashlib.sha256()
    for path in input_files(entrypoint):
        digest.update(f"--- {_relative(path)}\n".encode())
        for line in requirement_lines(path):
            digest.update(f"{line}\n".encode())
    return digest.hexdigest()


def read_lock(path: Path) -> tuple[dict[str, str], str | None]:
    """The pins of a lock file (canonical name -> version) and its inputs digest."""
    pins: dict[str, str] = {}
    digest = None
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith(DIGEST_PREFIX):
            digest = line[len(DIGEST_PREFIX) :].strip()
            continue
        if not line or line.startswith("#"):
            continue
        match = _LOCK_LINE_RE.match(line)
        if not match:
            raise LockError(f"{path}:{number}: not a `name==version` line: {line!r}")
        pins[match.group("name")] = match.group("version")
    return pins, digest


def macos_floor() -> str:
    """mac.minimumSystemVersion from the top-level `mac:` block of electron-builder.yml.

    Same reading as scripts/ci/install_backend_requirements.sh, which installs
    the wheels built for this floor on the macOS leg.
    """
    floors = []
    in_mac = False
    for line in ELECTRON_BUILDER.read_text(encoding="utf-8").splitlines():
        if line and not line[0].isspace() and not line.startswith("#"):
            in_mac = line.startswith("mac:")
        elif in_mac and re.match(r"^\s+minimumSystemVersion:", line):
            value = re.sub(r"\s*#.*$", "", line.split(":", 1)[1]).strip().strip("\"'")
            floors.append(value)
    if len(floors) != 1 or not re.fullmatch(r"\d+(\.\d+)?", floors[0]):
        raise LockError(f"Expected one mac.minimumSystemVersion in {ELECTRON_BUILDER}: {floors}")
    major, _, minor = floors[0].partition(".")
    return f"{major}_{minor or 0}"


def target_for(entrypoint: Path) -> Target:
    name = entrypoint.name
    if name.startswith("win-"):
        return Target(
            "win_amd64",
            ("win_amd64",),
            {
                "os_name": "nt",
                "sys_platform": "win32",
                "platform_system": "Windows",
                "platform_machine": "AMD64",
            },
        )
    if name.startswith("linux-"):
        manylinux = [f"manylinux_2_{minor}_x86_64" for minor in range(LINUX_GLIBC_MINOR, 4, -1)] + [
            "manylinux2014_x86_64",
            "manylinux2010_x86_64",
            "manylinux1_x86_64",
        ]
        return Target(
            f"manylinux_2_{LINUX_GLIBC_MINOR}_x86_64 and older",
            (*manylinux, "linux_x86_64"),
            {
                "os_name": "posix",
                "sys_platform": "linux",
                "platform_system": "Linux",
                "platform_machine": "x86_64",
            },
        )
    if name.startswith("mac-"):
        platform = f"macosx_{macos_floor()}_arm64"
        return Target(
            platform,
            (platform,),
            {
                "os_name": "posix",
                "sys_platform": "darwin",
                "platform_system": "Darwin",
                "platform_machine": "arm64",
            },
        )
    raise LockError(f"{name}: no target platform for this entrypoint (win-, linux- or mac-).")


def _packaging():
    from pip._vendor.packaging import markers, requirements, specifiers, version

    return markers, requirements, specifiers, version


def declared_requirements(entrypoint: Path) -> list:
    """Every requirement line of the entrypoint's files, parsed."""
    _, requirements, _, _ = _packaging()
    declared = []
    for path in input_files(entrypoint):
        for line in requirement_lines(path):
            if not line.startswith("-"):
                declared.append(requirements.Requirement(line))
    return declared


def seed_constraints(entrypoint: Path, upgrade: bool, upgrade_packages: set[str]) -> list[str]:
    """The existing lock as constraints, minus what the requirement files now refuse."""
    path = lock_path(entrypoint)
    if upgrade or not path.exists():
        return []
    _, _, _, version = _packaging()
    pins, _ = read_lock(path)
    refused = set(upgrade_packages)
    for requirement in declared_requirements(entrypoint):
        name = canonical_name(requirement.name)
        locked = pins.get(name)
        if locked and not requirement.specifier.contains(version.Version(locked), prereleases=True):
            refused.add(name)
    return [f"{name}=={pin}" for name, pin in sorted(pins.items()) if name not in refused]


def resolve(entrypoint: Path, target: Target, constraints: list[str], work: Path) -> dict:
    report = work / f"{entrypoint.stem}.json"
    command = [
        sys.executable,
        "-c",
        _PIP_BOOTSTRAP,
        json.dumps(target.markers),
        "install",
        "--dry-run",
        "--ignore-installed",
        "--quiet",
        "--disable-pip-version-check",
        "--only-binary=:all:",
        "--python-version",
        f"{PYTHON_VERSION[0]}.{PYTHON_VERSION[1]}",
        "--implementation",
        "cp",
        "--abi",
        f"cp{PYTHON_VERSION[0]}{PYTHON_VERSION[1]}",
        *(arg for platform in target.platforms for arg in ("--platform", platform)),
        "--report",
        str(report),
        "-r",
        str(entrypoint),
        "-r",
        str(build_requirements()),
    ]
    if constraints:
        seed = work / f"{entrypoint.stem}.seed.txt"
        seed.write_text("\n".join(constraints) + "\n", encoding="utf-8")
        command += ["-c", str(seed)]
    if subprocess.run(command, cwd=REPO_ROOT).returncode != 0:
        raise LockError(
            f"{entrypoint.name}: pip could not resolve it for {target.description}. If it "
            "conflicts with a locked version, rerun with --upgrade-package <name> for the "
            "packages pip names above."
        )
    data = json.loads(report.read_text(encoding="utf-8"))
    environment = data["environment"]
    wrong = {k: environment.get(k) for k, v in target.markers.items() if environment.get(k) != v}
    if wrong:
        raise LockError(
            f"{entrypoint.name}: pip ran with the host's markers, not the target's: {wrong}"
        )
    return data


def verify(entrypoint: Path, target: Target, report: dict) -> list[str]:
    """Walk the dependency graph with the target's markers.

    Every package reachable from the requested ones must be resolved, at a
    version its dependents accept, and every resolved package must be reachable.
    Returns the requirements the target's markers decide otherwise than the
    host's: what a resolution without the emulation would have got wrong.
    """
    markers, requirements, _, version = _packaging()
    host = markers.default_environment()
    environment = {**host, **target.markers}

    resolved = {canonical_name(item["metadata"]["name"]): item for item in report["install"]}
    flipped: set[str] = set()
    problems: list[str] = []

    def applies(owner: str, requirement, extras: set[str]) -> bool:
        if requirement.marker is None:
            return True
        text = str(requirement.marker)
        if any(name in text for name in UNEMULATED_MARKERS):
            problems.append(f"{owner}: `{text}` depends on the running kernel")

        def holds(base: dict) -> bool:
            return any(
                requirement.marker.evaluate({**base, "extra": extra}) for extra in {"", *extras}
            )

        on_target = holds(environment)
        if on_target != holds(host):
            flipped.add(f"{owner}: {requirement} ({'kept' if on_target else 'dropped'})")
        return on_target

    for path in input_files(entrypoint):
        for line in requirement_lines(path):
            if not line.startswith("-"):
                applies(_relative(path), requirements.Requirement(line), set())

    reached: dict[str, set[str]] = {}
    pending = [
        (name, set(item.get("requested_extras", [])))
        for name, item in resolved.items()
        if item.get("requested")
    ]
    while pending:
        name, extras = pending.pop()
        if name in reached and extras <= reached[name]:
            continue
        extras = reached.setdefault(name, set()) | extras
        reached[name] = extras
        item = resolved.get(name)
        if item is None:
            problems.append(f"{name} is required on the target but missing from the resolution")
            continue
        for line in item["metadata"].get("requires_dist", []):
            requirement = requirements.Requirement(line)
            if not applies(name, requirement, extras):
                continue
            dependency = canonical_name(requirement.name)
            dependency_item = resolved.get(dependency)
            if dependency_item is not None and not requirement.specifier.contains(
                version.Version(dependency_item["metadata"]["version"]), prereleases=True
            ):
                problems.append(
                    f"{name} requires {requirement}, resolved "
                    f"{dependency}=={dependency_item['metadata']['version']}"
                )
            pending.append((dependency, set(requirement.extras)))

    for name in sorted(set(resolved) - set(reached)):
        problems.append(f"{name} is resolved but nothing on the target requires it")
    if problems:
        raise LockError(
            f"{entrypoint.name}: the resolution does not hold for {target.description}:\n  "
            + "\n  ".join(sorted(set(problems)))
        )
    return sorted(flipped)


def render_lock(entrypoint: Path, target: Target, report: dict) -> str:
    pins = sorted(
        (canonical_name(item["metadata"]["name"]), item["metadata"]["version"])
        for item in report["install"]
    )
    header = [
        f"# Locked set of {_relative(entrypoint)} and {BUILD_REQUIREMENTS.as_posix()}",
        f"# for CPython {PYTHON_VERSION[0]}.{PYTHON_VERSION[1]}, {target.description}.",
        "# Generated by scripts/dev/lock_backend_requirements.py; do not edit by hand.",
        "# Regenerate after any change to the entrypoint or a file it includes:",
        f"#   {REGENERATE_COMMAND}",
        "# See backend/requirements/README.md.",
        f"{DIGEST_PREFIX}{inputs_digest(entrypoint)}",
    ]
    return "\n".join(header + [f"{name}=={pin}" for name, pin in pins]) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "entrypoints",
        nargs="*",
        metavar="ENTRYPOINT",
        help="prod entrypoint file names to lock (default: all of them)",
    )
    parser.add_argument(
        "--upgrade-package",
        action="append",
        default=[],
        metavar="NAME",
        help="let this package move to the newest version that resolves (repeatable)",
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="ignore the existing locks: every package moves to the newest that resolves",
    )
    args = parser.parse_args(argv)

    if sys.version_info[:2] != PYTHON_VERSION or sys.implementation.name != "cpython":
        print(
            f"Run this with CPython {PYTHON_VERSION[0]}.{PYTHON_VERSION[1]}, the backend's "
            f"interpreter: python_version markers are evaluated against it.",
            file=sys.stderr,
        )
        return 2

    entrypoints = prod_entrypoints()
    if args.entrypoints:
        by_name = {path.name: path for path in entrypoints}
        unknown = sorted(set(args.entrypoints) - set(by_name))
        if unknown:
            print(f"Not prod entrypoints: {', '.join(unknown)}", file=sys.stderr)
            return 2
        entrypoints = [by_name[name] for name in args.entrypoints]
    else:
        wanted = {lock_path(path) for path in entrypoints}
        for stale in sorted(set(LOCK_DIR.glob("*.txt")) - wanted):
            stale.unlink()
            print(f"Removed {stale.relative_to(REPO_ROOT)}: no such prod entrypoint.")

    upgrade_packages = {canonical_name(name) for name in args.upgrade_package}
    LOCK_DIR.mkdir(exist_ok=True)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for entrypoint in entrypoints:
                target = target_for(entrypoint)
                print(f"Resolving {entrypoint.name} for {target.description}...", flush=True)
                constraints = seed_constraints(entrypoint, args.upgrade, upgrade_packages)
                report = resolve(entrypoint, target, constraints, Path(tmp))
                flipped = verify(entrypoint, target, report)
                path = lock_path(entrypoint)
                path.write_text(render_lock(entrypoint, target, report), encoding="utf-8")
                print(f"  {path.relative_to(REPO_ROOT)}: {len(report['install'])} packages")
                if flipped:
                    print("  Decided by the target's markers, not this host's:")
                for line in flipped:
                    print(f"    {line}")
    except LockError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
