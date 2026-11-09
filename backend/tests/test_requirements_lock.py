"""Guards on the locked production requirements (backend/requirements/lock/).

A release build installs each prod entrypoint's lock in hash-checking mode and
fails when the installed set differs from it (scripts/ci/install_backend_requirements.sh).
That only runs on the legs that build, so these tests catch the common mistakes
on every pull request and every OS, without the network: a requirement file or
the macOS floor changed without the locks being regenerated, a lock edited by
hand, a declared requirement and its lock disagreeing. They also pin down the
pieces of scripts/dev/lock_backend_requirements.py and
scripts/ci/check_backend_lock.py that decide whether a lock is right for its
platform.
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(relative: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


lock_tool = _load("scripts/dev/lock_backend_requirements.py")
lock_check = _load("scripts/ci/check_backend_lock.py")

ENTRYPOINTS = lock_tool.prod_entrypoints()
REGENERATE = lock_tool.REGENERATE_COMMAND
HASH = "0" * 64


@pytest.mark.unit
def test_every_prod_entrypoint_has_a_lock_and_every_lock_an_entrypoint():
    locks = {path.name for path in lock_tool.LOCK_DIR.glob("*.txt")}
    entrypoints = {path.name for path in ENTRYPOINTS}
    assert entrypoints, "no prod entrypoint found"
    assert locks == entrypoints, (
        f"locks without entrypoint: {sorted(locks - entrypoints)}, entrypoints without "
        f"lock: {sorted(entrypoints - locks)}. Run `{REGENERATE}`."
    )


@pytest.mark.unit
@pytest.mark.parametrize("entrypoint", ENTRYPOINTS, ids=lambda path: path.name)
def test_lock_is_options_then_sorted_hashed_canonical_pins(entrypoint):
    lock = lock_tool.read_lock(lock_tool.lock_path(entrypoint))
    assert lock.pins, "empty lock"
    options = [line for line in lock.body if line.startswith("-")]
    assert lock.body[: len(options)] == options, "index options come before the pins"
    assert options == lock_tool.index_options(entrypoint)
    names = list(lock.pins)
    assert all(lock_tool.canonical_name(name) == name for name in names)
    assert names == sorted(names), "lock is not sorted by name"
    assert len(names) == len(lock.body) - len(options), "a package is locked twice"


@pytest.mark.unit
@pytest.mark.parametrize("entrypoint", ENTRYPOINTS, ids=lambda path: path.name)
def test_lock_matches_its_requirement_files_and_was_not_edited(entrypoint):
    lock = lock_tool.read_lock(lock_tool.lock_path(entrypoint))
    target = lock_tool.target_for(entrypoint)
    assert lock.digest == lock_tool.lock_digest(entrypoint, target, lock.body), (
        f"{entrypoint.name}, a file it includes or the macOS floor changed since its lock "
        f"was generated, or the lock was edited by hand. Run `{REGENERATE}` and commit "
        "the locks with the change."
    )


@pytest.mark.unit
@pytest.mark.parametrize("entrypoint", ENTRYPOINTS, ids=lambda path: path.name)
def test_every_declared_requirement_is_locked_at_a_version_it_accepts(entrypoint):
    from pip._vendor.packaging.markers import default_environment

    environment = {**default_environment(), **lock_tool.target_for(entrypoint).markers}
    pins = lock_tool.read_lock(lock_tool.lock_path(entrypoint)).pins
    for _, requirement in lock_tool.declared_requirements(entrypoint):
        if requirement.marker is not None and not requirement.marker.evaluate(
            {**environment, "extra": ""}
        ):
            continue
        name = lock_tool.canonical_name(requirement.name)
        assert name in pins, f"{requirement} is declared but not locked. Run `{REGENERATE}`."
        assert requirement.specifier.contains(
            pins[name], prereleases=True
        ), f"{requirement} is declared, the lock has {name}=={pins[name]}. Run `{REGENERATE}`."


@pytest.mark.unit
def test_the_macos_lock_targets_the_floor_the_installer_declares():
    text = (REPO_ROOT / "frontend/electron-builder.yml").read_text(encoding="utf-8")
    floor = re.search(r"^\s+minimumSystemVersion:\s*\"?([\d.]+)", text, re.M).group(1)
    major, _, minor = floor.partition(".")
    (mac,) = [path for path in ENTRYPOINTS if path.name.startswith("mac-")]
    assert lock_tool.target_for(mac).platforms == (f"macosx_{major}_{minor or 0}_arm64",)


@pytest.mark.unit
def test_linux_tags_follow_the_runner_pip_order():
    # The order pip prefers on a glibc 2.35 x86_64 runner (ubuntu-22.04): the
    # lock's file is the one that runner takes.
    platforms = lock_tool.target_for(lock_tool.PROD_DIR / "linux-cpu-prod.txt").platforms
    assert platforms[0] == "manylinux_2_35_x86_64"
    assert platforms[-1] == "linux_x86_64"
    assert platforms.index("manylinux2014_x86_64") == platforms.index("manylinux_2_17_x86_64") + 1
    assert platforms.index("manylinux1_x86_64") == platforms.index("manylinux_2_5_x86_64") + 1


@pytest.mark.unit
@pytest.mark.parametrize("entrypoint", ENTRYPOINTS, ids=lambda path: path.name)
def test_lock_pins_the_build_tool(entrypoint):
    # PyInstaller's bootloader becomes the backend executable and its hooks
    # decide what the bundle carries: a rebuild must not pick up a new release.
    pins = lock_tool.read_lock(lock_tool.lock_path(entrypoint)).pins
    assert {"pyinstaller", "pyinstaller-hooks-contrib"} <= set(pins)


def _workflow(name):
    import yaml  # a transformers dependency, installed on every leg

    return yaml.safe_load((REPO_ROOT / ".github/workflows" / name).read_text(encoding="utf-8"))


@pytest.mark.unit
@pytest.mark.parametrize("workflow", ["release.yml", "app-build-smoke.yml"])
def test_builds_install_only_through_the_locked_script_before_freezing(workflow):
    (job,) = _workflow(workflow)["jobs"].values()
    runs = [step.get("run", "") for step in job["steps"]]
    (install,) = [i for i, run in enumerate(runs) if "install_backend_requirements.sh" in run]
    (freeze,) = [i for i, run in enumerate(runs) if run.startswith("pyinstaller ")]
    # The script puts its venv on PATH for the steps after it, not its own.
    assert install < freeze
    assert not any(
        re.search(r"pip install\b", run) for run in runs
    ), "the backend and PyInstaller come from the lock, through install_backend_requirements.sh"


@pytest.mark.unit
def test_backend_ci_installs_within_the_prod_lock_and_checks_the_locks_of_its_os():
    (job,) = _workflow("backend-ci.yml")["jobs"].values()
    legs = job["strategy"]["matrix"]["include"]
    assert legs
    checked = set()
    for leg in legs:
        assert leg["lock"] == leg["reqs"].replace(".txt", "-prod.txt"), leg
        assert lock_tool.lock_path(lock_tool.PROD_DIR / leg["lock"]).exists(), leg
        prefix = leg["reqs"].split("-")[0]
        names = leg["prod_entrypoints"].split()
        assert names == [path.name for path in ENTRYPOINTS if path.name.startswith(prefix)]
        checked |= set(names)
    assert checked == {path.name for path in ENTRYPOINTS}, "every lock is checked on its OS"
    runs = "\n".join(step.get("run", "") for step in job["steps"])
    assert "check_backend_lock.py constraints backend/requirements/lock/${{ matrix.lock }}" in runs
    assert "check_backend_lock.py resolution" in runs


def _report(*items):
    """A pip --report holding the given (name, version, requires_dist) items."""
    return {
        "install": [
            {"metadata": {"name": name, "version": version, "requires_dist": requires}}
            for name, version, requires in items
        ]
    }


@pytest.fixture
def windows_entrypoint(tmp_path, monkeypatch):
    return _requirements_tree(tmp_path, monkeypatch, "tqdm==4.67.1\n")


def _requirements_tree(root, monkeypatch, entrypoint_text, build_text=""):
    """A requirements directory holding one Windows prod entrypoint and meta/build.txt."""
    monkeypatch.setattr(lock_tool, "REQUIREMENTS_DIR", root)
    monkeypatch.setattr(lock_tool, "LOCK_DIR", root / "lock")
    (root / "lock").mkdir()
    (root / "meta").mkdir()
    (root / "meta" / "build.txt").write_text(build_text, encoding="utf-8")
    entrypoint = root / "win-test-prod.txt"
    entrypoint.write_text(entrypoint_text, encoding="utf-8")
    return entrypoint


@pytest.mark.unit
def test_verify_accepts_a_resolution_made_with_the_target_markers(windows_entrypoint):
    target = lock_tool.target_for(windows_entrypoint)
    report = _report(
        ("tqdm", "4.67.1", ['colorama; platform_system == "Windows"']),
        ("colorama", "0.4.6", []),
    )
    flipped = lock_tool.verify(windows_entrypoint, target, report)
    if sys.platform != "win32":
        assert flipped == ['tqdm: colorama; platform_system == "Windows" (kept)']


@pytest.mark.unit
def test_verify_refuses_a_resolution_made_with_the_host_markers(windows_entrypoint):
    # What pip produces for Windows from a mac or Linux host without the marker
    # emulation: tqdm's Windows-only dependency is missing.
    target = lock_tool.target_for(windows_entrypoint)
    report = _report(("tqdm", "4.67.1", ['colorama; platform_system == "Windows"']))
    with pytest.raises(lock_tool.LockError, match="colorama.*missing from the resolution"):
        lock_tool.verify(windows_entrypoint, target, report)


@pytest.mark.unit
def test_verify_starts_from_the_declared_requirements(tmp_path, monkeypatch):
    # A declared requirement the target's markers keep must be resolved at a
    # version it accepts; one they drop must not be resolved at all.
    entrypoint = _requirements_tree(
        tmp_path,
        monkeypatch,
        'tqdm==4.67.1\nuvloop==0.21.0; sys_platform != "win32"\n',
        "pyinstaller==6.22.3\n",
    )
    target = lock_tool.target_for(entrypoint)
    report = _report(("tqdm", "4.67.0", []), ("uvloop", "0.21.0", []))
    with pytest.raises(lock_tool.LockError) as error:
        lock_tool.verify(entrypoint, target, report)
    message = str(error.value)
    assert "win-test-prod.txt requires tqdm==4.67.1, resolved tqdm==4.67.0" in message
    assert "meta/build.txt requires pyinstaller==6.22.3: missing from the resolution" in message
    assert "uvloop is resolved but nothing on the target requires it" in message


@pytest.mark.unit
def test_verify_refuses_unreachable_packages_and_unmet_ranges(windows_entrypoint):
    target = lock_tool.target_for(windows_entrypoint)
    report = _report(
        ("tqdm", "4.67.1", ["colorama>=0.5"]),
        ("colorama", "0.4.6", []),
        ("uvloop", "0.21.0", []),
    )
    with pytest.raises(lock_tool.LockError) as error:
        lock_tool.verify(windows_entrypoint, target, report)
    assert "tqdm requires colorama>=0.5, resolved colorama==0.4.6" in str(error.value)
    assert "uvloop is resolved but nothing on the target requires it" in str(error.value)


@pytest.mark.unit
def test_verify_refuses_markers_on_the_running_kernel_without_evaluating_them(
    windows_entrypoint, monkeypatch
):
    # The host's kernel version need not parse as a version (Azure's
    # "7.0.0-1019-azure" does not): the marker is reported, never evaluated.
    from pip._vendor.packaging import markers

    host = markers.default_environment()
    monkeypatch.setattr(
        markers, "default_environment", lambda: {**host, "platform_release": "7.0.0-1019-azure"}
    )
    target = lock_tool.target_for(windows_entrypoint)
    report = _report(
        ("tqdm", "4.67.1", ['colorama; platform_release >= "10"']),
        ("colorama", "0.4.6", []),
    )
    with pytest.raises(lock_tool.LockError, match="depends on the running kernel"):
        lock_tool.verify(windows_entrypoint, target, report)


def _lock_text(*lines):
    return "# header\n# sha256: x\n" + "".join(f"{line}\n" for line in lines)


@pytest.mark.unit
def test_read_lock_refuses_a_pin_without_its_hash(tmp_path):
    path = tmp_path / "win-cpu-prod.txt"
    path.write_text(_lock_text("regex==2026.9.29"), encoding="utf-8")
    with pytest.raises(lock_tool.LockError, match="--hash=sha256"):
        lock_tool.read_lock(path)


@pytest.mark.unit
def test_seed_keeps_locked_versions_the_requirement_files_still_accept(tmp_path, monkeypatch):
    entrypoint = _requirements_tree(
        tmp_path, monkeypatch, "tqdm==4.68.0\ncmake>=4.1.2\n", "pyinstaller==6.22.3\n"
    )
    (tmp_path / "lock" / entrypoint.name).write_text(
        _lock_text(
            *(
                f"{pin} --hash=sha256:{HASH}"
                for pin in (
                    "cmake==4.4.4",
                    "colorama==0.4.6",
                    "pyinstaller==6.22.2",
                    "pyinstaller-hooks-contrib==2026.9",
                    "regex==2026.9.29",
                    "tqdm==4.67.1",
                )
            )
        ),
        encoding="utf-8",
    )
    # The pins of tqdm and of the build tool moved: their locked versions leave
    # the seed; the rest stays.
    assert lock_tool.seed_constraints(entrypoint, False, set()) == [
        "cmake==4.4.4",
        "colorama==0.4.6",
        "pyinstaller-hooks-contrib==2026.9",
        "regex==2026.9.29",
    ]
    assert lock_tool.seed_constraints(entrypoint, False, {"regex"}) == [
        "cmake==4.4.4",
        "colorama==0.4.6",
        "pyinstaller-hooks-contrib==2026.9",
    ]
    assert lock_tool.seed_constraints(entrypoint, True, set()) == []


def _item(name, version, sha256):
    return {
        "metadata": {"name": name, "version": version, "requires_dist": []},
        "download_info": {
            "url": f"https://x/{name}",
            "archive_info": {"hashes": {"sha256": sha256}},
        },
    }


@pytest.mark.unit
def test_render_keeps_the_locked_file_of_a_version_that_does_not_move(windows_entrypoint):
    target = lock_tool.target_for(windows_entrypoint)
    previous = lock_tool.Lock(
        pins={"tqdm": "4.67.1", "regex": "2026.9.29"},
        hashes={"tqdm": "a" * 64, "regex": "b" * 64},
    )
    report = {"install": [_item("tqdm", "4.67.1", "c" * 64), _item("regex", "2026.11.9", "d" * 64)]}
    text, kept = lock_tool.render_lock(windows_entrypoint, target, report, previous, set())
    assert f"tqdm==4.67.1 --hash=sha256:{'a' * 64}" in text  # a wheel added since: not taken
    assert f"regex==2026.11.9 --hash=sha256:{'d' * 64}" in text  # moved: its new file
    assert kept == ["tqdm"]
    text, kept = lock_tool.render_lock(windows_entrypoint, target, report, previous, {"tqdm"})
    assert f"tqdm==4.67.1 --hash=sha256:{'c' * 64}" in text and kept == []


@pytest.mark.unit
def test_lock_digest_covers_the_pins(windows_entrypoint):
    target = lock_tool.target_for(windows_entrypoint)
    body = [f"regex==2026.9.29 --hash=sha256:{HASH}"]
    edited = [f"regex==2026.11.8 --hash=sha256:{HASH}"]
    digest = lock_tool.lock_digest(windows_entrypoint, target, body)
    assert digest != lock_tool.lock_digest(windows_entrypoint, target, edited)


class _Dist:
    def __init__(self, name, version):
        self.metadata = {"Name": name}
        self.version = version


@pytest.mark.unit
def test_installed_check_names_every_difference(tmp_path, monkeypatch, capsys):
    lock = tmp_path / "win-cpu-prod.txt"
    lock.write_text(
        _lock_text(
            *(
                f"{pin} --hash=sha256:{HASH}"
                for pin in ("missing==1.0", "regex==2026.9.29", "setuptools==84.2.1")
            )
        ),
        encoding="utf-8",
    )
    installed = [
        _Dist("Regex", "2026.11.8"),
        _Dist("setuptools", "85.0.0"),
        _Dist("pip", "26.3"),
        _Dist("pipx", "1.18.12"),
        _Dist("Stray_Package", "0.1"),
    ]
    monkeypatch.setattr(lock_check.metadata, "distributions", lambda: installed)
    assert lock_check.main(["installed", str(lock)]) == 1
    out = capsys.readouterr().out
    assert "missing: locked at 1.0, not installed" in out
    assert "regex: locked at 2026.9.29, 2026.11.8 installed" in out
    assert "setuptools: locked at 84.2.1, 85.0.0 installed" in out
    assert "pipx: 1.18.12 installed, not in the lock" in out
    assert "stray-package: 0.1 installed, not in the lock" in out
    assert "pip:" not in out
    assert REGENERATE in out


@pytest.mark.unit
def test_installed_check_passes_on_the_exact_set(tmp_path, monkeypatch):
    lock = tmp_path / "win-cpu-prod.txt"
    lock.write_text(
        _lock_text(
            "--extra-index-url https://download.pytorch.org/whl/cpu",
            f"regex==2026.9.29 --hash=sha256:{HASH}",
            f"torch==2.13.0+cpu --hash=sha256:{HASH}",
        ),
        encoding="utf-8",
    )
    installed = [_Dist("regex", "2026.9.29"), _Dist("torch", "2.13.0+cpu"), _Dist("pip", "26.3")]
    monkeypatch.setattr(lock_check.metadata, "distributions", lambda: installed)
    assert lock_check.main(["installed", str(lock)]) == 0


@pytest.mark.unit
def test_constraints_are_the_versions_without_hashes_or_options(tmp_path, capsys):
    lock = tmp_path / "win-cpu-prod.txt"
    lock.write_text(
        _lock_text(
            "--extra-index-url https://download.pytorch.org/whl/cpu",
            f"regex==2026.9.29 --hash=sha256:{HASH}",
            f"torch==2.13.0+cpu --hash=sha256:{HASH}",
        ),
        encoding="utf-8",
    )
    assert lock_check.main(["constraints", str(lock)]) == 0
    assert capsys.readouterr().out == "regex==2026.9.29\ntorch==2.13.0+cpu\n"
