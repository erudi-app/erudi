"""Guards on the locked production requirements (backend/requirements/lock/).

A release build installs each prod entrypoint constrained to its lock and fails
when the installed set differs from it (scripts/ci/install_backend_requirements.sh).
That check only runs on the legs that build, so these tests catch the common
mistakes on every pull request and every OS, without the network: a requirement
file changed without the locks being regenerated, a lock edited by hand, a
declared pin and its lock disagreeing. They also pin down the two pieces of
scripts/dev/lock_backend_requirements.py and scripts/ci/check_installed_lock.py
that decide whether a lock is right for its platform.
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
installed_check = _load("scripts/ci/check_installed_lock.py")

ENTRYPOINTS = lock_tool.prod_entrypoints()
REGENERATE = lock_tool.REGENERATE_COMMAND


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
def test_lock_is_sorted_canonical_pins(entrypoint):
    lines = lock_tool.lock_path(entrypoint).read_text(encoding="utf-8").splitlines()
    pins = [line for line in lines if line and not line.startswith("#")]
    assert pins, "empty lock"
    names = []
    for line in pins:
        match = re.fullmatch(r"([a-z0-9][a-z0-9.-]*)==(\S+)", line)
        assert match, f"not a canonical `name==version` line: {line!r}"
        assert lock_tool.canonical_name(match.group(1)) == match.group(1), line
        names.append(match.group(1))
    assert names == sorted(names), "lock is not sorted by name"
    assert len(names) == len(set(names)), "a package is locked twice"


@pytest.mark.unit
@pytest.mark.parametrize("entrypoint", ENTRYPOINTS, ids=lambda path: path.name)
def test_lock_was_generated_from_the_current_requirement_files(entrypoint):
    _, digest = lock_tool.read_lock(lock_tool.lock_path(entrypoint))
    assert digest == lock_tool.inputs_digest(entrypoint), (
        f"{entrypoint.name} or a file it includes changed since its lock was generated. "
        f"Run `{REGENERATE}` and commit the locks with the change."
    )


@pytest.mark.unit
@pytest.mark.parametrize("entrypoint", ENTRYPOINTS, ids=lambda path: path.name)
def test_every_declared_requirement_is_locked_at_a_version_it_accepts(entrypoint):
    from pip._vendor.packaging.markers import default_environment

    environment = {**default_environment(), **lock_tool.target_for(entrypoint).markers}
    pins, _ = lock_tool.read_lock(lock_tool.lock_path(entrypoint))
    for requirement in lock_tool.declared_requirements(entrypoint):
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
def test_the_macos_lock_targets_the_floor_the_mac_leg_installs():
    text = (REPO_ROOT / "frontend/electron-builder.yml").read_text(encoding="utf-8")
    floor = re.search(r"^\s+minimumSystemVersion:\s*\"?([\d.]+)", text, re.M).group(1)
    major, _, minor = floor.partition(".")
    (mac,) = [path for path in ENTRYPOINTS if path.name.startswith("mac-")]
    assert lock_tool.target_for(mac).platforms == (f"macosx_{major}_{minor or 0}_arm64",)


@pytest.mark.unit
@pytest.mark.parametrize("entrypoint", ENTRYPOINTS, ids=lambda path: path.name)
def test_lock_pins_the_build_tool(entrypoint):
    # PyInstaller's bootloader becomes the backend executable and its hooks
    # decide what the bundle carries: a rebuild must not pick up a new release.
    pins, _ = lock_tool.read_lock(lock_tool.lock_path(entrypoint))
    assert {"pyinstaller", "pyinstaller-hooks-contrib"} <= set(pins)


def _workflow(name):
    import yaml  # a transformers dependency, installed on every leg

    return yaml.safe_load((REPO_ROOT / ".github/workflows" / name).read_text(encoding="utf-8"))


@pytest.mark.unit
@pytest.mark.parametrize("workflow", ["release.yml", "app-build-smoke.yml"])
def test_builds_install_pyinstaller_only_through_the_locked_script(workflow):
    (job,) = _workflow(workflow)["jobs"].values()
    runs = [step.get("run", "") for step in job["steps"]]
    assert any("scripts/ci/install_backend_requirements.sh" in run for run in runs)
    assert not any(
        re.search(r"pip install\b.*\bpyinstaller", run) for run in runs
    ), "PyInstaller is installed by install_backend_requirements.sh, locked (meta/build.txt)"


def _report(*items):
    """A pip --report holding the given (name, version, requires_dist, requested) items."""
    return {
        "install": [
            {
                "requested": requested,
                "metadata": {"name": name, "version": version, "requires_dist": requires},
            }
            for name, version, requires, requested in items
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
        ("tqdm", "4.67.1", ['colorama; platform_system == "Windows"'], True),
        ("colorama", "0.4.6", [], False),
    )
    flipped = lock_tool.verify(windows_entrypoint, target, report)
    if sys.platform != "win32":
        assert flipped == ['tqdm: colorama; platform_system == "Windows" (kept)']


@pytest.mark.unit
def test_verify_refuses_a_resolution_made_with_the_host_markers(windows_entrypoint):
    # What pip produces for Windows from a mac or Linux host without the marker
    # emulation: tqdm's Windows-only dependency is missing.
    target = lock_tool.target_for(windows_entrypoint)
    report = _report(("tqdm", "4.67.1", ['colorama; platform_system == "Windows"'], True))
    with pytest.raises(lock_tool.LockError, match="colorama is required on the target"):
        lock_tool.verify(windows_entrypoint, target, report)


@pytest.mark.unit
def test_verify_refuses_unreachable_packages_and_unmet_ranges(windows_entrypoint):
    target = lock_tool.target_for(windows_entrypoint)
    report = _report(
        ("tqdm", "4.67.1", ["colorama>=0.5"], True),
        ("colorama", "0.4.6", [], False),
        ("uvloop", "0.21.0", [], False),
    )
    with pytest.raises(lock_tool.LockError) as error:
        lock_tool.verify(windows_entrypoint, target, report)
    assert "tqdm requires colorama>=0.5, resolved colorama==0.4.6" in str(error.value)
    assert "uvloop is resolved but nothing on the target requires it" in str(error.value)


@pytest.mark.unit
def test_verify_refuses_markers_on_the_running_kernel(windows_entrypoint):
    target = lock_tool.target_for(windows_entrypoint)
    report = _report(
        ("tqdm", "4.67.1", ['colorama; platform_release >= "10"'], True),
        ("colorama", "0.4.6", [], False),
    )
    with pytest.raises(lock_tool.LockError, match="depends on the running kernel"):
        lock_tool.verify(windows_entrypoint, target, report)


@pytest.mark.unit
def test_seed_keeps_locked_versions_the_requirement_files_still_accept(tmp_path, monkeypatch):
    entrypoint = _requirements_tree(
        tmp_path, monkeypatch, "tqdm==4.68.0\ncmake>=4.1.2\n", "pyinstaller==6.22.3\n"
    )
    (tmp_path / "lock" / entrypoint.name).write_text(
        "# header\ncmake==4.4.4\ncolorama==0.4.6\npyinstaller==6.22.2\n"
        "pyinstaller-hooks-contrib==2026.9\nregex==2026.9.29\ntqdm==4.67.1\n",
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


class _Dist:
    def __init__(self, name, version):
        self.metadata = {"Name": name}
        self.version = version


@pytest.mark.unit
def test_installed_check_names_every_difference(tmp_path, monkeypatch, capsys):
    lock = tmp_path / "win-cpu-prod.txt"
    lock.write_text(
        "# header\nmissing==1.0\nregex==2026.9.29\nsetuptools==84.2.1\n", encoding="utf-8"
    )
    installed = [
        _Dist("Regex", "2026.11.8"),
        _Dist("setuptools", "85.0.0"),
        _Dist("pip", "26.3"),
        _Dist("wheel", "0.45.1"),
        _Dist("Stray_Package", "0.1"),
    ]
    monkeypatch.setattr(installed_check.metadata, "distributions", lambda: installed)
    assert installed_check.main([str(lock)]) == 1
    out = capsys.readouterr().out
    assert "missing: locked at 1.0, not installed" in out
    assert "regex: locked at 2026.9.29, 2026.11.8 installed" in out
    assert "setuptools: locked at 84.2.1, 85.0.0 installed" in out
    assert "stray-package: 0.1 installed, not in the lock" in out
    assert "pip:" not in out and "wheel:" not in out
    assert REGENERATE in out


@pytest.mark.unit
def test_installed_check_passes_on_the_exact_set(tmp_path, monkeypatch):
    lock = tmp_path / "win-cpu-prod.txt"
    lock.write_text("# header\nregex==2026.9.29\ntorch==2.13.0+cpu\n", encoding="utf-8")
    installed = [_Dist("regex", "2026.9.29"), _Dist("torch", "2.13.0+cpu"), _Dist("pip", "26.3")]
    monkeypatch.setattr(installed_check.metadata, "distributions", lambda: installed)
    assert installed_check.main([str(lock)]) == 0
