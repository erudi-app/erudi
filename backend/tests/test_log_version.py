"""#576 - ``backend.log`` belongs to the app version that is running.

The Diagnostics page reads ``backend.log``; a file that outlives updates shows
records of versions no longer installed as if they were today's problems. At a
start whose app version differs from the last run's, the launcher rolls the
file over into the numbered history before the logger opens it, and writes a
marker the Diagnostics reader cuts at -- which is also what hides the older
records when the file could not be rolled over (Windows: held by another
process).
"""

from __future__ import annotations

import logging

import pytest

from src.domains.diagnostics import log_reader
from src.launcher import log_version
from src.launcher.log_version import (
    LOG_BACKUP_COUNT,
    LOG_FILE_NAME,
    VERSION_FILE_NAME,
    VERSION_SWITCH_MARKER,
    app_version_from_env,
    record_log_version,
    switch_log_for_version,
)

OLD_RECORD = (
    "[WARNING] 2026-09-03T10:00:00.000Z [-] - erudi - llms.py:1 - "
    "a 404 an older version logged at WARNING\n"
)


def _write(path, text):
    path.write_text(text, encoding="utf-8")


class TestAppVersionFromEnv:
    @pytest.mark.unit
    def test_reads_the_version_electron_passes(self, monkeypatch):
        monkeypatch.setenv("ERUDI_APP_VERSION", "1.3.0")
        assert app_version_from_env() == "1.3.0"

    @pytest.mark.unit
    def test_a_prerelease_version_is_kept_whole(self, monkeypatch):
        monkeypatch.setenv("ERUDI_APP_VERSION", "1.3.0-beta.2+build.7")
        assert app_version_from_env() == "1.3.0-beta.2+build.7"

    @pytest.mark.unit
    @pytest.mark.parametrize("raw", ["", "   ", "1.3.0/../../x", "1.3.0\n[ERROR]", "versión"])
    def test_absent_or_unusable_means_no_version(self, monkeypatch, raw):
        """The value lands in a file name's neighbour and in an ASCII log
        line: anything but a plain version string is ignored."""
        monkeypatch.setenv("ERUDI_APP_VERSION", raw)
        assert app_version_from_env() is None

    @pytest.mark.unit
    def test_unset_means_no_version(self, monkeypatch):
        monkeypatch.delenv("ERUDI_APP_VERSION", raising=False)
        assert app_version_from_env() is None


class TestSwitchLogForVersion:
    @pytest.mark.unit
    def test_an_update_rolls_the_old_versions_log_into_the_history(self, tmp_path):
        _write(tmp_path / LOG_FILE_NAME, OLD_RECORD)
        _write(tmp_path / f"{LOG_FILE_NAME}.1", "older still\n")
        _write(tmp_path / VERSION_FILE_NAME, "1.2.0")

        switch = switch_log_for_version(tmp_path, "1.3.0")

        assert switch is not None
        assert switch.previous == "1.2.0"
        assert switch.rotated is True
        assert switch.error is None
        assert not (tmp_path / LOG_FILE_NAME).exists()
        assert (tmp_path / f"{LOG_FILE_NAME}.1").read_text(encoding="utf-8") == OLD_RECORD
        assert (tmp_path / f"{LOG_FILE_NAME}.2").read_text(encoding="utf-8") == "older still\n"
        assert (tmp_path / VERSION_FILE_NAME).read_text(encoding="utf-8") == "1.3.0"

    @pytest.mark.unit
    def test_the_history_stays_bounded(self, tmp_path):
        _write(tmp_path / LOG_FILE_NAME, OLD_RECORD)
        for index in range(1, LOG_BACKUP_COUNT + 1):
            _write(tmp_path / f"{LOG_FILE_NAME}.{index}", f"backup {index}\n")
        _write(tmp_path / VERSION_FILE_NAME, "1.2.0")

        switch_log_for_version(tmp_path, "1.3.0")

        names = sorted(p.name for p in tmp_path.iterdir() if p.name != VERSION_FILE_NAME)
        assert len(names) == LOG_BACKUP_COUNT
        assert (tmp_path / f"{LOG_FILE_NAME}.{LOG_BACKUP_COUNT}").read_text(
            encoding="utf-8"
        ) == f"backup {LOG_BACKUP_COUNT - 1}\n"

    @pytest.mark.unit
    def test_a_log_from_before_versioned_logs_is_rolled_over_too(self, tmp_path):
        """No version file: the log predates this feature, so it holds older
        versions' records."""
        _write(tmp_path / LOG_FILE_NAME, OLD_RECORD)

        switch = switch_log_for_version(tmp_path, "1.3.0")

        assert switch is not None and switch.previous is None and switch.rotated is True
        assert (tmp_path / f"{LOG_FILE_NAME}.1").exists()

    @pytest.mark.unit
    def test_the_same_version_keeps_its_log(self, tmp_path):
        """A restart, or a reinstall of the same version, keeps its records."""
        _write(tmp_path / LOG_FILE_NAME, OLD_RECORD)
        _write(tmp_path / VERSION_FILE_NAME, "1.3.0")

        assert switch_log_for_version(tmp_path, "1.3.0") is None
        assert (tmp_path / LOG_FILE_NAME).read_text(encoding="utf-8") == OLD_RECORD
        assert not (tmp_path / f"{LOG_FILE_NAME}.1").exists()

    @pytest.mark.unit
    def test_no_version_changes_nothing(self, tmp_path):
        """A backend started by hand (dev, smoke boot) has no app version."""
        _write(tmp_path / LOG_FILE_NAME, OLD_RECORD)

        assert switch_log_for_version(tmp_path, None) is None
        assert sorted(p.name for p in tmp_path.iterdir()) == [LOG_FILE_NAME]

    @pytest.mark.unit
    def test_a_fresh_install_has_nothing_to_roll_over(self, tmp_path):
        switch = switch_log_for_version(tmp_path, "1.3.0")

        assert switch is not None and switch.rotated is False and switch.error is None
        assert (tmp_path / VERSION_FILE_NAME).read_text(encoding="utf-8") == "1.3.0"

    @pytest.mark.unit
    def test_a_log_held_by_another_process_is_not_a_failed_start(self, tmp_path, monkeypatch):
        """Windows refuses to rename a file another process holds open."""
        _write(tmp_path / LOG_FILE_NAME, OLD_RECORD)
        _write(tmp_path / VERSION_FILE_NAME, "1.2.0")
        real_replace = log_version.os.replace

        def refuse_the_live_file(src, dst):
            if str(src).endswith(LOG_FILE_NAME):
                raise PermissionError(32, "The process cannot access the file")
            return real_replace(src, dst)

        monkeypatch.setattr(log_version.os, "replace", refuse_the_live_file)

        switch = switch_log_for_version(tmp_path, "1.3.0")

        assert switch is not None and switch.rotated is False
        assert "PermissionError" in switch.error
        assert (tmp_path / LOG_FILE_NAME).read_text(encoding="utf-8") == OLD_RECORD
        # Recorded anyway: the marker written next hides the older records,
        # and the next start of this version keeps what this one writes.
        assert (tmp_path / VERSION_FILE_NAME).read_text(encoding="utf-8") == "1.3.0"


class TestACorruptVersionFile:
    """The version file is ours, but a disk, a sync tool or a crash mid-write
    can leave anything in it: it must never stop the backend, and its content
    never reaches a log line unless it is a plain version."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "raw", [b"\xff\xfe1.2.0", b"\x00" * 64, b"1.2.0\n[ERROR] forged", b"   "]
    )
    def test_reads_as_unknown_and_the_log_is_rolled_over(self, tmp_path, raw):
        _write(tmp_path / LOG_FILE_NAME, OLD_RECORD)
        (tmp_path / VERSION_FILE_NAME).write_bytes(raw)

        switch = switch_log_for_version(tmp_path, "1.3.0")

        assert switch is not None
        assert switch.previous is None
        assert switch.rotated is True
        assert (tmp_path / VERSION_FILE_NAME).read_text(encoding="utf-8") == "1.3.0"

    @pytest.mark.unit
    def test_the_launcher_goes_on_when_the_switch_raises(self, tmp_path, monkeypatch, capsys):
        import run

        monkeypatch.setenv("ERUDI_APP_VERSION", "1.3.0")

        def boom(log_dir, version):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(log_version, "switch_log_for_version", boom)

        run.start_versioned_log(tmp_path)  # must not raise

        assert "unexpected" in capsys.readouterr().err


class TestRecordLogVersion:
    @pytest.mark.unit
    def test_a_switch_writes_the_marker_the_reader_cuts_at(self, tmp_path, caplog):
        switch = switch_log_for_version(tmp_path, "1.3.0")
        logger = logging.getLogger("erudi")

        with caplog.at_level(logging.INFO, logger="erudi"):
            record_log_version(logger, "1.3.0", switch)

        messages = [r.getMessage() for r in caplog.records if r.name == "erudi"]
        assert any(m.startswith(VERSION_SWITCH_MARKER) and "1.3.0" in m for m in messages)
        assert any("app version 1.3.0" in m for m in messages)
        assert all(m.isascii() for m in messages)

    @pytest.mark.unit
    def test_a_failed_roll_over_is_a_warning_after_the_marker(self, tmp_path, caplog):
        switch = log_version.LogVersionSwitch(
            version="1.3.0", previous="1.2.0", rotated=False, error="PermissionError: held"
        )
        logger = logging.getLogger("erudi")

        with caplog.at_level(logging.INFO, logger="erudi"):
            record_log_version(logger, "1.3.0", switch)

        records = [r for r in caplog.records if r.name == "erudi"]
        marker = next(
            i for i, r in enumerate(records) if r.getMessage().startswith(VERSION_SWITCH_MARKER)
        )
        warnings = [i for i, r in enumerate(records) if r.levelno == logging.WARNING]
        assert len(warnings) == 1 and warnings[0] > marker
        assert "PermissionError: held" in records[warnings[0]].getMessage()

    @pytest.mark.unit
    def test_the_same_version_writes_no_marker(self, caplog):
        logger = logging.getLogger("erudi")

        with caplog.at_level(logging.INFO, logger="erudi"):
            record_log_version(logger, "1.3.0", None)

        messages = [r.getMessage() for r in caplog.records if r.name == "erudi"]
        assert not [m for m in messages if m.startswith(VERSION_SWITCH_MARKER)]
        assert any("app version 1.3.0" in m for m in messages)


class TestDiagnosticsReadsTheRunningVersion:
    """The fallback when the old file could not be rolled over: the reader
    keeps only what follows the last version-switch marker."""

    @staticmethod
    def _marker(version="1.3.0"):
        return (
            f"[INFO] 2026-11-17T09:00:00.000Z [-] - erudi - run.py:1 - "
            f"{VERSION_SWITCH_MARKER} {version} (previous: 1.2.0)\n"
        )

    NEW_RECORD = "[ERROR] 2026-11-17T09:00:05.000Z [-] - erudi - llms.py:2 - a problem of today\n"

    @pytest.mark.unit
    def test_records_before_the_marker_are_not_listed(self):
        text = OLD_RECORD + self._marker() + self.NEW_RECORD

        records = log_reader.parse_records(text)

        assert [r["message"] for r in records] == ["a problem of today"]

    @pytest.mark.unit
    def test_the_last_marker_wins(self):
        text = OLD_RECORD + self._marker("1.2.0") + OLD_RECORD + self._marker() + self.NEW_RECORD

        assert [r["message"] for r in log_reader.parse_records(text)] == ["a problem of today"]

    @pytest.mark.unit
    def test_no_marker_lists_everything(self):
        text = OLD_RECORD + self.NEW_RECORD
        assert len(log_reader.parse_records(text)) == 2

    @pytest.mark.unit
    def test_another_loggers_line_is_not_a_marker(self):
        """Only the app's own logger writes the marker."""
        forged = (
            f"[INFO] 2026-11-17T09:00:00.000Z [-] - pgserver - x.py:1 - "
            f"{VERSION_SWITCH_MARKER} 9.9.9\n"
        )
        text = OLD_RECORD + forged + self.NEW_RECORD
        assert len(log_reader.parse_records(text)) == 2


class TestLauncherWiring:
    @pytest.mark.unit
    def test_run_switches_the_log_and_writes_the_marker(self, tmp_path, monkeypatch, caplog):
        import run

        _write(tmp_path / LOG_FILE_NAME, OLD_RECORD)
        _write(tmp_path / VERSION_FILE_NAME, "1.2.0")
        monkeypatch.setenv("ERUDI_APP_VERSION", "1.3.0")

        with caplog.at_level(logging.INFO, logger="erudi"):
            run.start_versioned_log(tmp_path)

        assert (tmp_path / f"{LOG_FILE_NAME}.1").read_text(encoding="utf-8") == OLD_RECORD
        assert any(
            r.getMessage().startswith(f"{VERSION_SWITCH_MARKER} 1.3.0") for r in caplog.records
        )
