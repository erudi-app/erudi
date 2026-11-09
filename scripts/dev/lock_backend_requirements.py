#!/usr/bin/env python3
"""Lock what a release build installs, one file per prod entrypoint.

    python3.12 scripts/dev/lock_backend_requirements.py
    python3.12 scripts/dev/lock_backend_requirements.py --upgrade-package regex
    python3.12 scripts/dev/lock_backend_requirements.py --upgrade
    py -3.12 scripts\\dev\\lock_backend_requirements.py          (Windows)

For every file in backend/requirements/entrypoints/prod/ it resolves the
entrypoint together with backend/requirements/meta/build.txt (PyInstaller, the
build tool) and writes backend/requirements/lock/<same name>: the index options
of the requirement files, then every package of the resolution as
`name==version --hash=sha256:<hash>`, sorted. The hash is the one of the file
pip selected for the target, so the lock names files, not just versions: a wheel
published later for a locked version (a new build tag or platform tag) cannot
enter a build. scripts/ci/install_backend_requirements.sh installs the lock alone
in hash-checking mode (`-r <lock> --require-hashes --no-deps`) into a fresh venv
and fails when the installed set differs from it.

Each entrypoint is resolved for the platform its build runs on, from any host:
`pip install --dry-run --report` with `--only-binary=:all: --platform <tags>
--python-version 3.12`. The tags are the ones the build runner's own pip
accepts, in its order of preference: win_amd64; manylinux up to the glibc of the
oldest Linux release runner, so the file is the one that runner takes (a newer
runner accepts it too, and the hash makes it take the same one); macOS arm64 at
the app's floor (mac.minimumSystemVersion in frontend/electron-builder.yml), so
the frozen backend loads on every macOS the installer accepts. `--platform`
only selects wheels: pip still evaluates environment markers (`sys_platform`,
`platform_machine`, ...) against the interpreter running it, so a Windows lock
generated on macOS would miss `colorama; sys_platform == "win32"`. pip is
therefore run with its marker environment replaced by the target's, and the
result is verified: the dependency graph walked from the declared requirements
with the target's markers must reach exactly the resolved set, at versions every
dependent accepts. A marker on `platform_release` or `platform_version` cannot be
emulated off the target and stops the run.

The resolution is seeded with the existing lock: its versions are passed as
constraints, and a package whose version does not move keeps its locked file,
so a regeneration only moves what the change requires. A locked version that the
requirement files no longer accept leaves the seed on its own; anything else
moves only when asked, with `--upgrade-package NAME` (repeatable) or `--upgrade`
(every package, newest that resolves).

Each lock records a digest of the requirement lines it was generated from
(comments and blank lines excluded), of the target's platform tags and of its
own body; backend/tests/test_requirements_lock.py fails when a requirement file
or the macOS floor changes without the locks being regenerated, and when a lock
is edited by hand.

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
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS_DIR = REPO_ROOT / "backend" / "requirements"
PROD_DIR = REQUIREMENTS_DIR / "entrypoints" / "prod"
LOCK_DIR = REQUIREMENTS_DIR / "lock"
# Resolved and locked with every prod entrypoint.
BUILD_REQUIREMENTS = Path("meta") / "build.txt"
ELECTRON_BUILDER = REPO_ROOT / "frontend" / "electron-builder.yml"

PYTHON_VERSION = (3, 12)
REGENERATE_COMMANDS = (
    "python3.12 scripts/dev/lock_backend_requirements.py",
    "py -3.12 scripts\\dev\\lock_backend_requirements.py   (Windows)",
)
REGENERATE_COMMAND = f"{REGENERATE_COMMANDS[0]} (Windows: py -3.12 scripts\\dev\\...)"
DIGEST_PREFIX = "# sha256: "

# glibc of the oldest runner that installs a Linux lock: release.yml builds on
# ubuntu-22.04 (glibc 2.35); app-build-smoke.yml on ubuntu-latest, newer. A wheel
# tagged above it would not install there.
LINUX_GLIBC_MINOR = 35
# The legacy manylinux names, at the glibc they stand for.
_MANYLINUX_ALIASES = {17: "manylinux2014", 12: "manylinux2010", 5: "manylinux1"}

# Markers that describe the running kernel: they cannot be emulated for another
# machine, so meeting one stops the run.
UNEMULATED_MARKERS = ("platform_release", "platform_version")

# The requirement-file options a lock carries over, so it installs on its own.
_INDEX_OPTIONS = ("--index-url", "--extra-index-url", "--find-links", "--trusted-host")

# pip's own comment rule for requirement files.
_COMMENT_RE = re.compile(r"(^|\s+)#.*$")
_INCLUDE_RE = re.compile(r"^(-r|--requirement|-c|--constraint)(\s+|=)(?P<path>\S+)$")
_LOCK_LINE_RE = re.compile(
    r"^(?P<name>[a-z0-9][a-z0-9.-]*)==(?P<version>\S+) --hash=sha256:(?P<hash>[0-9a-f]{64})$"
)

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
    """A lock cannot be generated or read; the message says why."""


@dataclass(frozen=True)
class Target:
    description: str
    platforms: tuple[str, ...]
    markers: dict[str, str]


@dataclass
class Lock:
    """A parsed lock file."""

    options: list[str] = field(default_factory=list)
    pins: dict[str, str] = field(default_factory=dict)
    hashes: dict[str, str] = field(default_factory=dict)
    body: list[str] = field(default_factory=list)
    digest: str | None = None


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


def index_options(entrypoint: Path) -> list[str]:
    """The index options of the entrypoint's files, in order, once each."""
    options: list[str] = []
    for path in input_files(entrypoint):
        for line in requirement_lines(path):
            if line.split()[0].split("=")[0] in _INDEX_OPTIONS and line not in options:
                options.append(line)
    return options


def lock_digest(entrypoint: Path, target: Target, body: list[str]) -> str:
    """Digest of what a lock is generated from and of what it says."""
    digest = hashlib.sha256()
    for path in input_files(entrypoint):
        digest.update(f"--- {_relative(path)}\n".encode())
        for line in requirement_lines(path):
            digest.update(f"{line}\n".encode())
    digest.update(f"--- target {' '.join(target.platforms)}\n".encode())
    digest.update("--- lock\n".encode())
    for line in body:
        digest.update(f"{line}\n".encode())
    return digest.hexdigest()


def read_lock(path: Path) -> Lock:
    lock = Lock()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith(DIGEST_PREFIX):
            lock.digest = line[len(DIGEST_PREFIX) :].strip()
            continue
        if not line or line.startswith("#"):
            continue
        lock.body.append(line)
        if line.startswith("--"):
            lock.options.append(line)
            continue
        match = _LOCK_LINE_RE.match(line)
        if not match:
            raise LockError(f"{path}:{number}: not a `name==version --hash=sha256:...` line")
        lock.pins[match.group("name")] = match.group("version")
        lock.hashes[match.group("name")] = match.group("hash")
    return lock


def macos_floor() -> str:
    """mac.minimumSystemVersion from the top-level `mac:` block of electron-builder.yml."""
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


def _linux_platforms() -> tuple[str, ...]:
    """The tags of a glibc 2.LINUX_GLIBC_MINOR x86_64 runner, in its pip's order."""
    platforms = []
    for minor in range(LINUX_GLIBC_MINOR, 4, -1):
        platforms.append(f"manylinux_2_{minor}_x86_64")
        if minor in _MANYLINUX_ALIASES:
            platforms.append(f"{_MANYLINUX_ALIASES[minor]}_x86_64")
    return (*platforms, "linux_x86_64")


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
        return Target(
            f"manylinux_2_{LINUX_GLIBC_MINOR}_x86_64 and older",
            _linux_platforms(),
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


def declared_requirements(entrypoint: Path) -> list[tuple[str, object]]:
    """Every requirement line of the entrypoint's files, parsed, with its file."""
    _, requirements, _, _ = _packaging()
    declared = []
    for path in input_files(entrypoint):
        for line in requirement_lines(path):
            if not line.startswith("-"):
                declared.append((_relative(path), requirements.Requirement(line)))
    return declared


def seed_constraints(entrypoint: Path, upgrade: bool, upgrade_packages: set[str]) -> list[str]:
    """The existing lock as constraints, minus what the requirement files now refuse."""
    path = lock_path(entrypoint)
    if upgrade or not path.exists():
        return []
    _, _, _, version = _packaging()
    pins = read_lock(path).pins
    refused = set(upgrade_packages)
    for _, requirement in declared_requirements(entrypoint):
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
    """Walk the dependency graph from the declared requirements with the target's markers.

    Every declared requirement that applies to the target, and every package it
    reaches, must be resolved at a version its dependents accept, and every
    resolved package must be reached. Returns the requirements the target's
    markers decide otherwise than the host's: what a resolution without the
    emulation would have got wrong.
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
            # Not evaluated: the host's kernel version may not even parse.
            problems.append(f"{owner}: `{text}` depends on the running kernel")
            return False

        def holds(base: dict) -> bool:
            return any(
                requirement.marker.evaluate({**base, "extra": extra}) for extra in {"", *extras}
            )

        on_target = holds(environment)
        if on_target != holds(host):
            flipped.add(f"{owner}: {requirement} ({'kept' if on_target else 'dropped'})")
        return on_target

    def check(owner: str, requirement) -> str | None:
        name = canonical_name(requirement.name)
        item = resolved.get(name)
        if item is None:
            problems.append(f"{owner} requires {requirement}: missing from the resolution")
            return None
        if not requirement.specifier.contains(
            version.Version(item["metadata"]["version"]), prereleases=True
        ):
            problems.append(
                f"{owner} requires {requirement}, resolved {name}=={item['metadata']['version']}"
            )
        return name

    reached: dict[str, set[str]] = {}
    pending: list[tuple[str, set[str]]] = []
    for owner, requirement in declared_requirements(entrypoint):
        if applies(owner, requirement, set()):
            name = check(owner, requirement)
            if name:
                pending.append((name, set(requirement.extras)))
    while pending:
        name, extras = pending.pop()
        if name in reached and extras <= reached[name]:
            continue
        extras = reached.setdefault(name, set()) | extras
        reached[name] = extras
        for line in resolved[name]["metadata"].get("requires_dist", []):
            requirement = requirements.Requirement(line)
            if applies(name, requirement, extras):
                dependency = check(name, requirement)
                if dependency:
                    pending.append((dependency, set(requirement.extras)))

    for name in sorted(set(resolved) - set(reached)):
        problems.append(f"{name} is resolved but nothing on the target requires it")
    if problems:
        raise LockError(
            f"{entrypoint.name}: the resolution does not hold for {target.description}:\n  "
            + "\n  ".join(sorted(set(problems)))
        )
    return sorted(flipped)


def selected_hash(item: dict) -> str:
    """sha256 of the file pip selected for this package.

    The index gives it beside the link, except the PyTorch index for the copies
    of PyPI packages it serves (Jinja2): that file is downloaded and hashed.
    """
    info = item.get("download_info", {})
    archive = info.get("archive_info", {})
    digest = archive.get("hashes", {}).get("sha256")
    if not digest and archive.get("hash", "").startswith("sha256="):
        digest = archive["hash"].split("=", 1)[1]
    if digest:
        return digest
    url = info.get("url", "")
    if not url.startswith("https://"):
        raise LockError(f"{item['metadata']['name']}: no sha256 and no https URL to hash")
    sha256 = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as response:
        for chunk in iter(lambda: response.read(1 << 20), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def render_lock(
    entrypoint: Path, target: Target, report: dict, previous: Lock | None, upgrade: set[str]
) -> tuple[str, list[str]]:
    """The lock text, and the packages that keep a locked file pip no longer prefers."""
    kept = []
    pins = []
    for item in report["install"]:
        name = canonical_name(item["metadata"]["name"])
        pinned = item["metadata"]["version"]
        digest = selected_hash(item)
        if (
            previous
            and name not in upgrade
            and previous.pins.get(name) == pinned
            and previous.hashes[name] != digest
        ):
            # Same version, another file preferred today (a wheel added since):
            # the build keeps the file it was locked to.
            digest = previous.hashes[name]
            kept.append(name)
        pins.append((name, f"{name}=={pinned} --hash=sha256:{digest}"))
    body = index_options(entrypoint) + [line for _, line in sorted(pins)]
    header = [
        f"# Locked set of {_relative(entrypoint)} and {BUILD_REQUIREMENTS.as_posix()}",
        f"# for CPython {PYTHON_VERSION[0]}.{PYTHON_VERSION[1]}, {target.description}.",
        "# Generated by scripts/dev/lock_backend_requirements.py; do not edit by hand.",
        "# Regenerate after any change to the entrypoint or a file it includes:",
        *(f"#   {command}" for command in REGENERATE_COMMANDS),
        "# Install: pip install -r <this file> --require-hashes --no-deps",
        "# See backend/requirements/README.md.",
        f"{DIGEST_PREFIX}{lock_digest(entrypoint, target, body)}",
    ]
    return "\n".join(header + body) + "\n", sorted(kept)


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
                path = lock_path(entrypoint)
                previous = None if args.upgrade or not path.exists() else read_lock(path)
                constraints = seed_constraints(entrypoint, args.upgrade, upgrade_packages)
                report = resolve(entrypoint, target, constraints, Path(tmp))
                flipped = verify(entrypoint, target, report)
                text, kept = render_lock(entrypoint, target, report, previous, upgrade_packages)
                path.write_text(text, encoding="utf-8")
                print(f"  {path.relative_to(REPO_ROOT)}: {len(report['install'])} packages")
                if flipped:
                    print("  Decided by the target's markers, not this host's:")
                for line in flipped:
                    print(f"    {line}")
                if kept:
                    print(
                        "  Kept their locked file although pip now prefers another file of "
                        f"the same version (--upgrade-package to take it): {', '.join(kept)}"
                    )
    except LockError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
