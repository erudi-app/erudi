"""Tests for launcher runtime path resolution (packaged vs. dev).

The prod branches of `src/launcher/runtime_paths.py` only ever ran inside a
bundled build, historically the worst family of release bugs. This file pins
every branch on a throwaway HOME: the module-level singleton contract, the
dev layout, the per-OS prod directory selection, the packaged-payload copy
semantics, and the macOS symlink swap.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.launcher import runtime_paths as rp


@pytest.fixture(autouse=True)
def _isolated_module_state(monkeypatch, tmp_path):
    """Reset the singleton and sandbox HOME/XDG so no test touches real dirs."""
    monkeypatch.setattr(rp, "_RUNTIME_PATHS", None)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.delenv("ERUDI_DATA_ROOT", raising=False)
    yield


# =====================================================================
# UNIT - singleton contract
# =====================================================================


@pytest.mark.unit
class TestSingletonContract:
    def test_get_before_init_raises(self):
        with pytest.raises(RuntimeError, match="not initialized"):
            rp.get_runtime_paths()

    def test_initialize_dev_and_get_roundtrip(self, tmp_path):
        root = tmp_path / "backend"
        root.mkdir()
        paths = rp.initialize_runtime_paths("dev", root)
        assert paths.mode == "dev"
        assert paths.data_dir == root.resolve() / "data"
        assert paths.log_dir == root.resolve() / "logs"
        assert paths.data_dir.is_dir()
        assert paths.log_dir.is_dir()
        assert rp.get_runtime_paths() is paths

    def test_double_initialize_raises(self, tmp_path):
        root = tmp_path / "backend"
        root.mkdir()
        rp.initialize_runtime_paths("dev", root)
        with pytest.raises(ValueError, match="already initialized"):
            rp.initialize_runtime_paths("dev", root)

    def test_unsupported_mode_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported runtime mode"):
            rp.initialize_runtime_paths("staging", tmp_path)

    def test_mode_is_case_insensitive(self, tmp_path):
        root = tmp_path / "backend"
        root.mkdir()
        paths = rp.initialize_runtime_paths("DEV", root)
        assert paths.mode == "dev"

    def test_ensure_defaults_to_dev_with_explicit_root(self, tmp_path):
        root = tmp_path / "backend"
        root.mkdir()
        paths = rp.ensure_runtime_paths_initialized(backend_root=root)
        assert paths.mode == "dev"
        assert paths.backend_root == root.resolve()
        # Idempotent: second call returns the same object
        assert rp.ensure_runtime_paths_initialized() is paths

    def test_ensure_defaults_to_repo_layout_without_root(self):
        paths = rp.ensure_runtime_paths_initialized()
        assert paths.mode == "dev"
        # parents[2] of src/launcher/runtime_paths.py is backend/
        assert paths.backend_root.name == "backend"


# =====================================================================
# UNIT - dev layout
# =====================================================================


@pytest.mark.unit
class TestDevPaths:
    def test_replaces_stale_data_symlink_with_real_dir(self, tmp_path):
        root = tmp_path / "backend"
        root.mkdir()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        os.symlink(elsewhere, root / "data")

        data_dir, log_dir = rp._setup_dev_paths(root)

        assert data_dir == root / "data"
        assert data_dir.is_dir()
        assert not data_dir.is_symlink()


# =====================================================================
# UNIT - prod directory selection per OS
# =====================================================================


@pytest.mark.unit
class TestDetermineProdDirectories:
    def test_darwin_layout(self, monkeypatch, tmp_path):
        monkeypatch.setattr(rp.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
        data_dir, log_dir = rp._determine_prod_directories()
        home = tmp_path / "home"
        assert data_dir == (
            home / "Library" / "Application Support" / "erudi" / "backend" / "prod" / "data"
        )
        assert log_dir == home / "Library" / "Logs" / "erudi"
        assert data_dir.is_dir()

    def test_windows_layout_uses_localappdata(self, monkeypatch, tmp_path):
        monkeypatch.setattr(rp.platform, "system", lambda: "Windows")
        local = tmp_path / "AppDataLocal"
        monkeypatch.setenv("LOCALAPPDATA", str(local))
        data_dir, log_dir = rp._determine_prod_directories()
        assert data_dir == local / "erudi" / "backend" / "prod" / "data"
        assert log_dir == local / "erudi" / "logs"

    def test_windows_layout_defaults_without_localappdata(self, monkeypatch, tmp_path):
        monkeypatch.setattr(rp.platform, "system", lambda: "Windows")
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
        data_dir, _log_dir = rp._determine_prod_directories()
        assert data_dir == (
            tmp_path / "home" / "AppData" / "Local" / "erudi" / "backend" / "prod" / "data"
        )

    def test_linux_layout_honours_xdg_vars(self, monkeypatch, tmp_path):
        monkeypatch.setattr(rp.platform, "system", lambda: "Linux")
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg-data"))
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg-state"))
        data_dir, log_dir = rp._determine_prod_directories()
        assert data_dir == tmp_path / "xdg-data" / "erudi" / "backend" / "prod" / "data"
        assert log_dir == tmp_path / "xdg-state" / "erudi" / "logs"

    def test_linux_layout_defaults_without_xdg_vars(self, monkeypatch, tmp_path):
        monkeypatch.setattr(rp.platform, "system", lambda: "Linux")
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
        data_dir, log_dir = rp._determine_prod_directories()
        home = tmp_path / "home"
        assert data_dir == home / ".local" / "share" / "erudi" / "backend" / "prod" / "data"
        assert log_dir == home / ".local" / "state" / "erudi" / "logs"


# =====================================================================
# UNIT - packaged payload copy
# =====================================================================


@pytest.mark.unit
class TestCopyPackagedPayload:
    def test_missing_source_is_noop(self, tmp_path):
        dest = tmp_path / "dest"
        rp._copy_packaged_payload(tmp_path / "missing", dest)
        assert not dest.exists()

    def test_source_equals_destination_is_noop(self, tmp_path):
        src = tmp_path / "payload"
        src.mkdir()
        (src / "seed.txt").write_text("x")
        rp._copy_packaged_payload(src, src)
        assert list(src.iterdir()) == [src / "seed.txt"]

    def test_copies_files_and_directories(self, tmp_path):
        src = tmp_path / "payload"
        (src / "models").mkdir(parents=True)
        (src / "models" / "weights.bin").write_bytes(b"\x00" * 8)
        (src / "catalog.json").write_text("{}")
        dest = tmp_path / "dest"

        rp._copy_packaged_payload(src, dest)

        assert (dest / "catalog.json").read_text() == "{}"
        assert (dest / "models" / "weights.bin").exists()

    def test_existing_targets_are_preserved(self, tmp_path):
        """User data must never be clobbered by the packaged payload."""
        src = tmp_path / "payload"
        src.mkdir()
        (src / "catalog.json").write_text("packaged")
        dest = tmp_path / "dest"
        dest.mkdir()
        (dest / "catalog.json").write_text("user-modified")

        rp._copy_packaged_payload(src, dest)

        assert (dest / "catalog.json").read_text() == "user-modified"


# =====================================================================
# UNIT - macOS symlink swap
# =====================================================================


@pytest.mark.unit
class TestEnsureMacosSymlink:
    def test_missing_packaged_path_is_noop(self, tmp_path):
        target = tmp_path / "writable"
        target.mkdir()
        rp._ensure_macos_symlink(tmp_path / "missing", target)
        assert not (tmp_path / "missing").exists()

    def test_already_linked_is_noop(self, tmp_path):
        target = tmp_path / "writable"
        target.mkdir()
        packaged = tmp_path / "packaged"
        os.symlink(target, packaged)
        rp._ensure_macos_symlink(packaged, target)
        assert packaged.resolve() == target.resolve()

    def test_stale_symlink_is_repointed(self, tmp_path):
        old_target = tmp_path / "old"
        old_target.mkdir()
        new_target = tmp_path / "new"
        new_target.mkdir()
        packaged = tmp_path / "packaged"
        os.symlink(old_target, packaged)

        rp._ensure_macos_symlink(packaged, new_target)

        assert packaged.is_symlink()
        assert packaged.resolve() == new_target.resolve()

    def test_real_directory_is_replaced_by_symlink(self, tmp_path):
        packaged = tmp_path / "packaged"
        packaged.mkdir()
        (packaged / "leftover.txt").write_text("x")
        target = tmp_path / "writable"
        target.mkdir()

        rp._ensure_macos_symlink(packaged, target)

        assert packaged.is_symlink()
        assert packaged.resolve() == target.resolve()

    def test_regular_file_is_replaced_by_symlink(self, tmp_path):
        packaged = tmp_path / "packaged"
        packaged.write_text("stray file")
        target = tmp_path / "writable"
        target.mkdir()

        rp._ensure_macos_symlink(packaged, target)

        assert packaged.is_symlink()
        assert packaged.resolve() == target.resolve()


# =====================================================================
# UNIT - explicit instance root (ERUDI_DATA_ROOT)
# =====================================================================


@pytest.mark.unit
class TestDataRootOverride:
    """ERUDI_DATA_ROOT points a whole instance (data + logs) at an explicit
    root, so an isolated backend (tests, QA side-by-side runs) never touches
    the default dev/prod directories or their log files."""

    def test_dev_mode_honours_erudi_data_root(self, monkeypatch, tmp_path):
        root = tmp_path / "backend"
        root.mkdir()
        instance = tmp_path / "instance"
        monkeypatch.setenv("ERUDI_DATA_ROOT", str(instance))

        paths = rp.initialize_runtime_paths("dev", root)

        assert paths.mode == "dev"
        assert paths.data_dir == instance.resolve() / "data"
        assert paths.log_dir == instance.resolve() / "logs"
        assert paths.data_dir.is_dir()
        assert paths.log_dir.is_dir()
        # The default dev layout must not have been created.
        assert not (root / "data").exists()
        assert not (root / "logs").exists()

    def test_prod_mode_honours_override_and_copies_payload(self, monkeypatch, tmp_path):
        monkeypatch.setattr(rp.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
        root = tmp_path / "bundle" / "backend"
        payload = root / "data"
        payload.mkdir(parents=True)
        (payload / "catalog.json").write_text("{}")
        instance = tmp_path / "instance"
        monkeypatch.setenv("ERUDI_DATA_ROOT", str(instance))

        paths = rp.initialize_runtime_paths("prod", root, packaged_data_dir=payload)

        assert paths.data_dir == instance.resolve() / "data"
        assert paths.log_dir == instance.resolve() / "logs"
        # Bundled payload still lands in the overridden data dir...
        assert (paths.data_dir / "catalog.json").exists()
        # ...but the bundle itself is never repointed at the override root.
        assert not payload.is_symlink()
        # And the default macOS prod location is untouched.
        assert not (tmp_path / "home" / "Library").exists()

    def test_empty_override_keeps_default_layout(self, monkeypatch, tmp_path):
        root = tmp_path / "backend"
        root.mkdir()
        monkeypatch.setenv("ERUDI_DATA_ROOT", "")

        paths = rp.initialize_runtime_paths("dev", root)

        assert paths.data_dir == root.resolve() / "data"
        assert paths.log_dir == root.resolve() / "logs"


# =====================================================================
# UNIT - full prod path assembly
# =====================================================================


@pytest.mark.unit
class TestProdMode:
    def _init_prod(self, monkeypatch, tmp_path, system: str):
        monkeypatch.setattr(rp.platform, "system", lambda: system)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
        root = tmp_path / "bundle" / "backend"
        payload = root / "data"
        payload.mkdir(parents=True)
        (payload / "catalog.json").write_text("{}")
        return rp.initialize_runtime_paths("prod", root, packaged_data_dir=payload)

    def test_linux_prod_copies_payload_into_user_dir(self, monkeypatch, tmp_path):
        paths = self._init_prod(monkeypatch, tmp_path, "Linux")
        assert paths.mode == "prod"
        assert (paths.data_dir / "catalog.json").exists()
        assert paths.log_dir.is_dir()
        # Payload dir untouched on Linux (no symlink swap)
        assert not (tmp_path / "bundle" / "backend" / "data").is_symlink()

    def test_darwin_prod_swaps_payload_for_symlink(self, monkeypatch, tmp_path):
        paths = self._init_prod(monkeypatch, tmp_path, "Darwin")
        packaged = tmp_path / "bundle" / "backend" / "data"
        assert (paths.data_dir / "catalog.json").exists()
        assert packaged.is_symlink()
        assert packaged.resolve() == paths.data_dir.resolve()


# =====================================================================
# UNIT - the layout the Electron side mirrors
# =====================================================================

# The Electron main process opens and clears these folders without asking the
# backend (which is stopped by then), so it computes them too
# (frontend/src/utils/dataLocations.js). Both sides are checked against this
# one table: a change to either fails a test until the table, and so the
# other side, follows.
_SHARED_TABLE = json.loads(
    (Path(__file__).parent / "fixtures" / "runtime_paths_layout.json").read_text(encoding="utf-8")
)
_SHARED_LAYOUT = _SHARED_TABLE["cases"]


def _table_env_value(home: Path, value: str) -> str:
    """An env value of the table: empty and ~ values as is, others under home."""
    if value == "" or value.startswith("~"):
        return value
    return str(home / value)


@pytest.mark.unit
@pytest.mark.parametrize("case", _SHARED_LAYOUT, ids=[c["name"] for c in _SHARED_LAYOUT])
def test_prod_layout_matches_the_table_shared_with_electron(case, monkeypatch, tmp_path):
    home = (tmp_path / "home").resolve()
    monkeypatch.setattr(rp.platform, "system", lambda: case["system"])
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    # expanduser reads HOME (POSIX) or USERPROFILE (Windows).
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    for name, value in case["env"].items():
        monkeypatch.setenv(name, _table_env_value(home, value))

    paths = rp._compute_runtime_paths(
        mode="prod", backend_root=tmp_path / "bundle", packaged_data_dir=tmp_path / "no-payload"
    )

    assert paths.data_dir.relative_to(home).as_posix() == case["data_dir"]
    assert paths.log_dir.relative_to(home).as_posix() == case["log_dir"]


# =====================================================================
# UNIT - the ownership sentinel Clear All Data checks
# =====================================================================

# Clear All Data deletes a backend folder only when it carries this file, so a
# folder Erudi did not create (an ERUDI_DATA_ROOT pointed at an existing
# folder holding a `data/` of the user's own) is never deleted.


def test_the_sentinel_name_is_the_one_shared_with_electron():
    assert rp.OWNERSHIP_SENTINEL == _SHARED_TABLE["ownership_sentinel"]


def test_the_pasted_images_folder_is_the_one_electron_writes_to():
    # Electron writes a pasted image there (frontend/src/utils/pastedImage.js);
    # a conversation's delete removes only the files inside it.
    assert rp.PASTED_IMAGES_FOLDER == _SHARED_TABLE["pasted_images_folder"]


def test_pasted_images_dir_is_in_the_data_folder(monkeypatch, tmp_path):
    paths = rp.RuntimePaths(
        mode="dev", backend_root=tmp_path, data_dir=tmp_path / "data", log_dir=tmp_path / "logs"
    )
    monkeypatch.setattr(rp, "_RUNTIME_PATHS", paths)

    assert rp.pasted_images_dir() == tmp_path / "data" / _SHARED_TABLE["pasted_images_folder"]


@pytest.mark.unit
class TestOwnershipSentinel:
    def test_prod_folders_are_marked_even_when_they_already_exist(self, monkeypatch, tmp_path):
        # An install from before the sentinel: its folders are Erudi's by name.
        monkeypatch.setattr(rp.platform, "system", lambda: "Linux")
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
        existing = tmp_path / "share" / "erudi" / "backend" / "prod" / "data"
        existing.mkdir(parents=True)

        paths = rp._compute_runtime_paths("prod", tmp_path / "bundle", tmp_path / "no-payload")

        assert (paths.data_dir / rp.OWNERSHIP_SENTINEL).is_file()
        assert (paths.log_dir / rp.OWNERSHIP_SENTINEL).is_file()

    def test_dev_folders_are_marked(self, tmp_path):
        root = tmp_path / "backend"
        root.mkdir()
        paths = rp._compute_runtime_paths("dev", root, None)
        assert (paths.data_dir / rp.OWNERSHIP_SENTINEL).is_file()
        assert (paths.log_dir / rp.OWNERSHIP_SENTINEL).is_file()

    def test_instance_folders_created_now_are_marked(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ERUDI_DATA_ROOT", str(tmp_path / "instance"))
        paths = rp._compute_runtime_paths("dev", tmp_path / "backend", None)
        assert (paths.data_dir / rp.OWNERSHIP_SENTINEL).is_file()
        assert (paths.log_dir / rp.OWNERSHIP_SENTINEL).is_file()

    def test_an_existing_folder_under_the_instance_root_is_never_marked(
        self, monkeypatch, tmp_path
    ):
        # ERUDI_DATA_ROOT pointed at a folder that already holds the user's own
        # data/ and logs/: they stay unmarked, so Clear All Data skips them.
        shared = tmp_path / "shared"
        (shared / "data").mkdir(parents=True)
        (shared / "data" / "thesis.docx").write_text("mine")
        (shared / "logs").mkdir()
        monkeypatch.setenv("ERUDI_DATA_ROOT", str(shared))

        paths = rp._compute_runtime_paths("dev", tmp_path / "backend", None)

        assert not (paths.data_dir / rp.OWNERSHIP_SENTINEL).exists()
        assert not (paths.log_dir / rp.OWNERSHIP_SENTINEL).exists()

    def test_a_marked_instance_folder_stays_marked(self, monkeypatch, tmp_path):
        instance = tmp_path / "instance"
        monkeypatch.setenv("ERUDI_DATA_ROOT", str(instance))
        rp._compute_runtime_paths("dev", tmp_path / "backend", None)

        paths = rp._compute_runtime_paths("dev", tmp_path / "backend", None)

        assert (paths.data_dir / rp.OWNERSHIP_SENTINEL).is_file()
