import os
import sys

import pytest

from erudi_eval import storage


def write(path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def test_usage_counts_files_and_does_not_follow_symlinks(tmp_path):
    outside = tmp_path / "outside"
    write(outside / "big.bin", 1_000_000)
    tree = tmp_path / "tree"
    write(tree / "a.bin", 10_000)
    write(tree / "sub" / "b.bin", 20_000)
    if sys.platform.startswith("win"):
        pytest.skip("symlink creation needs privileges on Windows")
    os.symlink(outside, tree / "data")  # like the macOS bundle's data -> data root link
    u = storage.usage(tree)
    assert u.files == 2
    assert u.symlinks == 1
    assert u.logical_bytes < 100_000  # the 1 MB behind the link is not counted
    if storage.HAS_BLOCKS:
        assert u.bytes >= 30_000  # allocated size, block-rounded


def test_hardlinks_counted_once(tmp_path):
    write(tmp_path / "f.bin", 50_000)
    os.link(tmp_path / "f.bin", tmp_path / "g.bin")
    assert storage.usage(tmp_path).files == 1


def test_snapshot_layout_and_delta(tmp_path):
    root = tmp_path / "prod"
    write(root / "data" / "models" / "Qwen3-4B" / "w.safetensors", 40_000)
    write(root / "data" / "postgres" / "base" / "1" / "t", 8_000)
    write(root / "data" / "postgres" / "pg_wal" / "0001", 16_000)
    write(root / "data" / "erudi_db_password", 10)
    write(root / "db-backups" / "erudi-x.dump", 5_000)
    app = tmp_path / "Erudi.app"
    write(app / "Contents" / "Resources" / "backend" / "_internal" / "torch" / "lib.dylib", 30_000)
    write(app / "Contents" / "Resources" / "app.asar", 7_000)
    install = {"app": app, "resources": app / "Contents" / "Resources", "backend_lib": app / "Contents" / "Resources" / "backend" / "_internal"}
    log = tmp_path / "erudi-backend.log"
    write(log, 100)

    s1 = storage.snapshot("start", install, root, {"stdout_capture": log})
    assert s1["size_kind"] == storage.SIZE_KIND
    assert [r["name"] for r in s1["data"]["models"]] == ["Qwen3-4B"]
    assert s1["data"]["postgres"]["pg_wal"]["logical_bytes"] == 16_000
    assert s1["install"]["backend_lib_top25"][0]["name"] == "torch"
    assert {r["name"] for r in s1["install"]["resources_top"]} == {"backend", "app.asar"}

    write(root / "data" / "models" / "Other" / "m.gguf", 90_000)
    s2 = storage.snapshot("after", install, root, {"stdout_capture": log})
    rows = {r["entry"]: r["delta"] for r in storage.delta(s1, s2)}
    assert "data/models/Other" in rows and rows["data/models/Other"] >= 90_000
    assert "install/total" not in rows
