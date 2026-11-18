"""The boot cleanup of the obsolete e5 snapshot on Windows and Linux (#594).

On Windows and Linux the KB embedding child serves the e5 GGUF that ships
with the installers; the fp32 sentence-transformers snapshot an earlier
version downloaded into ``models_cache`` is ~470 MB nothing reads anymore.
The start deletes it, once, and only where that is safe:

- only where the platform's artifact ships with the installation (the
  llama-server adapter): on macOS the snapshot IS the child's artifact;
- only once that bundled GGUF is on disk;
- idempotent: an absent snapshot is a silent no-op, a failed delete is one
  WARNING and the next start tries again.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from src.core import config
from src.ingestion import embedder_child, embedding_model

pytestmark = pytest.mark.unit


@pytest.fixture
def cache(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    return tmp_path


def _put_snapshot(cache) -> list:
    """The hub cache layout an earlier version left behind, plus its lock dir."""
    repo = cache / "models--intfloat--multilingual-e5-small"
    snapshot = repo / "snapshots" / "614241f6"
    (snapshot / "1_Pooling").mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (repo / "blobs").mkdir()
    (repo / "blobs" / "abc").write_bytes(b"x" * 2048)
    locks = cache / ".locks" / "models--intfloat--multilingual-e5-small"
    locks.mkdir(parents=True)
    other = cache / "models--someone--other-model"
    other.mkdir()
    return [repo, locks, other]


def _adapter(monkeypatch, *, downloadable: bool, available: bool):
    monkeypatch.setattr(
        embedder_child,
        "_adapter_override",
        SimpleNamespace(artifact_downloadable=downloadable, artifact_available=lambda: available),
    )


class TestRemoveObsoleteSnapshot:
    def test_bundled_platform_with_its_gguf_deletes_the_snapshot(self, cache, monkeypatch, caplog):
        repo, locks, other = _put_snapshot(cache)
        _adapter(monkeypatch, downloadable=False, available=True)

        with caplog.at_level(logging.INFO, logger="erudi"):
            assert embedder_child.remove_obsolete_e5_snapshot() is True

        assert not repo.exists()
        assert not locks.exists()
        assert other.exists()  # nothing else in the cache is touched
        records = [r for r in caplog.records if "obsolete" in r.message]
        assert len(records) == 1 and records[0].levelno == logging.INFO

    def test_macos_keeps_the_snapshot_its_child_serves(self, cache, monkeypatch):
        repo, _, _ = _put_snapshot(cache)
        _adapter(monkeypatch, downloadable=True, available=True)

        assert embedder_child.remove_obsolete_e5_snapshot() is False
        assert repo.exists()

    def test_kept_until_the_bundled_gguf_is_in_place(self, cache, monkeypatch):
        repo, _, _ = _put_snapshot(cache)
        _adapter(monkeypatch, downloadable=False, available=False)

        assert embedder_child.remove_obsolete_e5_snapshot() is False
        assert repo.exists()

    def test_a_disarmed_manager_touches_nothing(self, cache, monkeypatch):
        repo, _, _ = _put_snapshot(cache)
        monkeypatch.setattr(embedder_child, "_adapter_override", None)

        assert embedder_child.remove_obsolete_e5_snapshot() is False
        assert repo.exists()

    def test_idempotent_and_silent_once_gone(self, cache, monkeypatch, caplog):
        _put_snapshot(cache)
        _adapter(monkeypatch, downloadable=False, available=True)
        assert embedder_child.remove_obsolete_e5_snapshot() is True

        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="erudi"):
            assert embedder_child.remove_obsolete_e5_snapshot() is False
        assert not [r for r in caplog.records if "snapshot" in r.message]

    def test_a_failed_delete_is_one_warning_and_never_raises(self, cache, monkeypatch, caplog):
        repo, _, _ = _put_snapshot(cache)
        _adapter(monkeypatch, downloadable=False, available=True)

        def refuse(path, *a, **k):
            raise PermissionError(13, "Access is denied", str(path))

        monkeypatch.setattr(embedding_model.shutil, "rmtree", refuse)
        with caplog.at_level(logging.INFO, logger="erudi"):
            assert embedder_child.remove_obsolete_e5_snapshot() is False

        assert repo.exists()
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "next start" in warnings[0].message


class TestBootWiring:
    def test_the_start_runs_the_cleanup(self):
        # Wired into the lifespan's start sequence, off the event loop.
        import inspect

        from src.core import api

        source = inspect.getsource(api._start_services)
        assert "run_in_threadpool(remove_obsolete_e5_snapshot)" in source
