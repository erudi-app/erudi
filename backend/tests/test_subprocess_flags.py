"""Unit tests for Windows console-window suppression on child spawns (#175)."""

from pathlib import Path

import pytest

from src.core.subprocess_flags import (
    _CREATE_NO_WINDOW,
    hidden_console_creationflags,
    signed_exit_code,
)


@pytest.mark.unit
class TestHiddenConsoleCreationflags:
    def test_windows_returns_create_no_window(self, monkeypatch):
        monkeypatch.setattr("src.core.subprocess_flags.platform.system", lambda: "Windows")
        assert hidden_console_creationflags() == _CREATE_NO_WINDOW

    @pytest.mark.parametrize("system", ["Linux", "Darwin"])
    def test_posix_returns_zero(self, monkeypatch, system):
        # 0 is required: Popen rejects a non-zero creationflags on POSIX.
        monkeypatch.setattr("src.core.subprocess_flags.platform.system", lambda: system)
        assert hidden_console_creationflags() == 0


@pytest.mark.unit
class TestSignedExitCode:
    """Windows reports a process exit status as an unsigned 32-bit value."""

    @pytest.mark.parametrize(
        "raw,expected",
        [(4294967295, -1), (3221225477, -1073741819), (1, 1), (0, 0), (2147483647, 2147483647)],
    )
    def test_windows_status_is_read_as_signed(self, monkeypatch, raw, expected):
        monkeypatch.setattr("src.core.subprocess_flags.platform.system", lambda: "Windows")
        assert signed_exit_code(raw) == expected

    @pytest.mark.parametrize("system", ["Linux", "Darwin"])
    def test_posix_status_is_left_alone(self, monkeypatch, system):
        monkeypatch.setattr("src.core.subprocess_flags.platform.system", lambda: system)
        assert signed_exit_code(-9) == -9
        assert signed_exit_code(4294967295) == 4294967295

    def test_none_stays_none(self):
        assert signed_exit_code(None) is None


@pytest.mark.unit
class TestLlamaServerSpawnHidesConsole:
    def test_spawn_child_passes_hidden_console_creationflags(self, monkeypatch, tmp_path):
        from src.engines import base_llama_cpp_engine as mod
        from src.engines.cpu_engine import CPU_Engine

        captured = {}

        class _FakeProc:
            pid = 4242
            # Part of the Popen contract `_spawn_child` relies on: it creates
            # the child with stdout=PIPE and hands the stream to the output
            # drainer (#361). None means "no output captured here".
            stdout = None

        def fake_popen(argv, **kwargs):
            captured["argv"] = argv
            captured.update(kwargs)
            return _FakeProc()

        monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(CPU_Engine, "_default_install_dir", classmethod(lambda cls: tmp_path))
        monkeypatch.setattr(
            CPU_Engine, "_find_llama_server", classmethod(lambda cls, d: tmp_path / "llama-server")
        )
        monkeypatch.setattr(
            CPU_Engine,
            "_build_spawn_argv",
            classmethod(lambda cls, **kw: [kw["llama_server"], "--port", kw["port"]]),
        )
        monkeypatch.setattr(CPU_Engine, "_find_mmproj", classmethod(lambda cls, p: None))
        monkeypatch.setattr(CPU_Engine, "_build_spawn_env", classmethod(lambda cls: {}))

        handle = CPU_Engine._spawn_child(
            model_path=tmp_path / "m.gguf", alias="erudi-x", port=27200
        )

        assert captured["creationflags"] == hidden_console_creationflags()
        assert handle["pid"] == 4242


@pytest.mark.unit
class TestPgDumpHidesConsole:
    def test_backup_database_passes_hidden_console_creationflags(self, monkeypatch, tmp_path):
        from src.database import backup as mod

        captured = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = [str(a) for a in argv]
            captured.update(kwargs)

        monkeypatch.setattr(mod.subprocess, "run", fake_run)
        monkeypatch.setattr(mod, "_pg_dump_bin", lambda: Path("pg_dump"))

        dump_path = mod.backup_database("postgresql://x", tmp_path / "postgres", "rev1")

        assert captured["creationflags"] == hidden_console_creationflags()
        assert dump_path.name == "erudi-rev1.dump"
