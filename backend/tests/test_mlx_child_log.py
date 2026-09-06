"""What the mlx-vlm child prints must survive the child.

`llama-server` is a `subprocess.Popen`, so its merged stdout+stderr is drained
into a tail the parent quotes when the child dies (`ChildOutputDrainer`). The
MLX child is an `mp.Process` with no pipe: until it captured its own output,
every mlx-vlm crash, Metal error and slow load reached the user as an exit
code and nothing else.

These tests exercise the capture helpers as plain functions -- no MLX, no
Apple Silicon -- plus two real child processes (a bare `python -c`, not
mlx-vlm) that prove the redirect catches writes made straight to the file
descriptors, which is where a C extension's last words appear.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.engines import mlx_child_log as child_log

BACKEND_ROOT = Path(__file__).resolve().parents[1]


def _run_child(body: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
    """Run `body` in a real python child with `backend/` importable."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(BACKEND_ROOT)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        cwd=str(BACKEND_ROOT),
    )


# =====================================================================
# Rolling: one live file plus one previous, per port
# =====================================================================


@pytest.mark.unit
class TestRollChildLog:
    def test_rolls_the_live_file_to_dot_one(self, tmp_path):
        path = tmp_path / "mlx-child-27300.log"
        path.write_text("first spawn", encoding="utf-8")

        child_log.roll_child_log(path)

        assert not path.exists()
        assert Path(f"{path}.1").read_text(encoding="utf-8") == "first spawn"

    def test_keeps_only_the_last_two_per_port(self, tmp_path):
        path = tmp_path / "mlx-child-27300.log"
        for spawn in ("one", "two", "three"):
            path.write_text(spawn, encoding="utf-8")
            child_log.roll_child_log(path)

        assert Path(f"{path}.1").read_text(encoding="utf-8") == "three"
        assert not Path(f"{path}.2").exists()

    def test_is_a_no_op_when_nothing_was_written_yet(self, tmp_path):
        child_log.roll_child_log(tmp_path / "mlx-child-27300.log")  # must not raise

    def test_prepare_creates_the_directory_and_rolls(self, tmp_path):
        log_dir = tmp_path / "logs"
        first = child_log.prepare_child_log(27300, log_dir=log_dir)
        assert first == log_dir / "mlx-child-27300.log"
        first.write_text("previous spawn", encoding="utf-8")

        second = child_log.prepare_child_log(27300, log_dir=log_dir)

        assert second == first
        assert not second.exists()
        assert Path(f"{first}.1").read_text(encoding="utf-8") == "previous spawn"

    def test_rolling_a_file_that_is_open_copies_and_truncates(self, tmp_path):
        """The live file is the one the child writes through descriptors 1 and
        2. Renaming it works on POSIX and is REFUSED on Windows, where a file
        with open handles cannot be renamed -- which disabled the size cap
        there silently. `copytruncate` needs no rename, so the same code holds
        the cap on every platform.
        """
        path = tmp_path / "mlx-child-27300.log"
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, b"the first megabyte\n")

            child_log.roll_open_log(path, fd=fd)

            assert Path(f"{path}.1").read_text(encoding="utf-8") == "the first megabyte\n"
            assert path.stat().st_size == 0  # same file, same descriptor
            # And the descriptor still writes into it, at the new end (O_APPEND
            # means no sparse hole where the old bytes were).
            os.write(fd, b"and then some more\n")
            assert path.read_text(encoding="utf-8") == "and then some more\n"
        finally:
            os.close(fd)

    def test_rolling_an_open_file_keeps_only_the_last_two(self, tmp_path):
        path = tmp_path / "mlx-child-27300.log"
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            for text in (b"one\n", b"two\n", b"three\n"):
                os.write(fd, text)
                child_log.roll_open_log(path, fd=fd)
        finally:
            os.close(fd)

        assert Path(f"{path}.1").read_text(encoding="utf-8") == "three\n"
        assert not Path(f"{path}.2").exists()

    def test_discard_removes_every_file_of_that_port(self, tmp_path):
        path = tmp_path / "mlx-child-27300.log"
        path.write_text("live", encoding="utf-8")
        Path(f"{path}.1").write_text("previous", encoding="utf-8")

        child_log.discard_child_log(path)

        assert not path.exists()
        assert not Path(f"{path}.1").exists()


# =====================================================================
# Reading the tail
# =====================================================================


@pytest.mark.unit
class TestReadChildLogTail:
    def test_returns_the_last_lines(self, tmp_path):
        path = tmp_path / "mlx-child-27300.log"
        path.write_text("\n".join(f"line {i}" for i in range(500)), encoding="utf-8")

        tail = child_log.read_child_log_tail(path, max_chars=200)

        assert len(tail) <= 200
        assert "line 499" in tail
        assert "line 0\n" not in tail

    def test_is_empty_when_the_child_wrote_nothing(self, tmp_path):
        path = tmp_path / "mlx-child-27300.log"
        path.touch()
        assert child_log.read_child_log_tail(path) == ""

    def test_is_empty_when_there_is_no_file(self, tmp_path):
        assert child_log.read_child_log_tail(tmp_path / "absent.log") == ""

    def test_spans_a_rollover_so_the_cap_never_hides_the_last_words(self, tmp_path):
        """A file rolled a moment before the crash would otherwise leave a tail
        of two lines; the previous file completes it."""
        path = tmp_path / "mlx-child-27300.log"
        Path(f"{path}.1").write_text("older output\n", encoding="utf-8")
        path.write_text("newest output\n", encoding="utf-8")

        tail = child_log.read_child_log_tail(path, max_chars=2000)

        assert tail.index("older output") < tail.index("newest output")

    def test_a_previous_childs_words_are_not_read_as_this_ones(self, tmp_path):
        """Ports are reused, and a child that died keeps its file: the next
        spawn rolls it to `.1`. Without the spawn's own start time, that dead
        child's errors would be quoted at the top of the NEXT child's crash
        report -- two failures read as one, the wrong one first."""
        path = tmp_path / "mlx-child-27300.log"
        previous = Path(f"{path}.1")
        previous.write_text("the first child died here\n", encoding="utf-8")
        os.utime(previous, (1_000_000, 1_000_000))  # long before this spawn
        path.write_text("the second child died here\n", encoding="utf-8")

        tail = child_log.read_child_log_tail(path, max_chars=2000, since=2_000_000)

        assert "the second child died here" in tail
        assert "the first child died here" not in tail

    def test_a_rollover_of_this_spawn_is_still_included(self, tmp_path):
        """The `.1` written by THIS child's size guard is its own output and
        belongs in its report."""
        path = tmp_path / "mlx-child-27300.log"
        rolled = Path(f"{path}.1")
        rolled.write_text("earlier in this run\n", encoding="utf-8")
        os.utime(rolled, (3_000_000, 3_000_000))  # after the spawn started
        path.write_text("and then it died\n", encoding="utf-8")

        tail = child_log.read_child_log_tail(path, max_chars=2000, since=2_000_000)

        assert tail.index("earlier in this run") < tail.index("and then it died")

    def test_never_raises_on_an_unreadable_path(self, tmp_path):
        assert child_log.read_child_log_tail(tmp_path) == ""  # a directory


# =====================================================================
# The redirect itself, in a real child process
# =====================================================================


@pytest.mark.unit
class TestRedirectStdioInARealChild:
    def test_captures_python_prints_and_raw_descriptor_writes(self, tmp_path):
        """dup2, not `contextlib.redirect_stdout`: a rebound `sys.stdout` would
        miss everything mlx and Metal write straight to fd 1/2, which is where
        a native failure's last words are."""
        path = tmp_path / "mlx-child-27300.log"
        result = _run_child(
            f"""
            from src.engines.mlx_child_log import redirect_stdio_to
            import os, sys

            redirect_stdio_to({str(path)!r})
            print("python stdout")
            print("python stderr", file=sys.stderr)
            os.write(1, b"raw fd 1\\n")
            os.write(2, b"raw fd 2\\n")
            """
        )

        assert result.returncode == 0, result.stderr
        captured = path.read_text(encoding="utf-8")
        for expected in ("python stdout", "python stderr", "raw fd 1", "raw fd 2"):
            assert expected in captured
        # Nothing leaks back to the parent's own stdout/stderr, which in the
        # real backend is the JSON launcher channel run.py parses.
        assert result.stdout == ""
        assert result.stderr == ""

    def test_lines_are_readable_before_the_child_exits(self, tmp_path):
        """Line buffering is what makes a probe timeout diagnosable: the parent
        reads the tail while the child is still (not) starting up."""
        path = tmp_path / "mlx-child-27300.log"
        proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    f"""
                    from src.engines.mlx_child_log import redirect_stdio_to
                    import time
                    redirect_stdio_to({str(path)!r})
                    print("loading the model")
                    time.sleep(30)
                    """
                ),
            ],
            env={**os.environ, "PYTHONPATH": str(BACKEND_ROOT)},
            cwd=str(BACKEND_ROOT),
        )
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if "loading the model" in child_log.read_child_log_tail(path):
                    break
                time.sleep(0.1)
            else:
                pytest.fail("the child's output never became readable")
        finally:
            proc.kill()
            proc.wait(timeout=10)

    def test_the_size_cap_rolls_instead_of_growing_without_bound(self, tmp_path):
        """A child that logs a line per token would otherwise fill the disk."""
        path = tmp_path / "mlx-child-27300.log"
        result = _run_child(
            f"""
            from src.engines.mlx_child_log import redirect_stdio_to
            import time

            redirect_stdio_to({str(path)!r}, max_bytes=4096, check_interval=0.05)
            for i in range(4000):
                print("noisy line %05d" % i)
                if i % 500 == 0:
                    time.sleep(0.06)
            time.sleep(0.3)
            print("the very last word")
            time.sleep(0.2)
            """
        )

        assert result.returncode == 0, result.stderr
        assert Path(f"{path}.1").exists(), "the cap never rolled the file"
        assert path.stat().st_size < 512 * 1024
        assert not Path(f"{path}.2").exists(), "more than two files per port"
        assert "the very last word" in child_log.read_child_log_tail(path)

    def test_a_child_record_carries_its_level_and_lands_in_the_file(self, tmp_path):
        path = tmp_path / "mlx-child-27300.log"
        result = _run_child(
            f"""
            from src.engines.mlx_child_log import redirect_stdio_to, child_warning

            redirect_stdio_to({str(path)!r})
            child_warning("a patch did not apply")
            """
        )

        assert result.returncode == 0, result.stderr
        captured = path.read_text(encoding="utf-8")
        assert "[WARNING]" in captured
        assert "a patch did not apply" in captured
        assert "erudi.mlx-child" in captured


# =====================================================================
# The parent side: MLX_Engine reads that file
# =====================================================================


@pytest.mark.unit
class TestEngineReadsTheChildLog:
    def test_read_child_output_returns_the_tail(self, tmp_path):
        from src.engines.mlx_engine import MLX_Engine

        path = tmp_path / "mlx-child-27300.log"
        path.write_text("ValueError: Expected shape (262144, 640)\n", encoding="utf-8")
        proc = MagicMock()
        setattr(proc, MLX_Engine._CHILD_LOG_ATTR, str(path))

        output = MLX_Engine._read_child_output(proc)

        assert "Expected shape (262144, 640)" in output

    def test_read_child_output_quotes_only_this_spawn(self, tmp_path):
        """Two crashes on one port: the second report must carry the second
        child's words and not the first's."""
        from src.engines.mlx_engine import MLX_Engine

        path = tmp_path / "mlx-child-27300.log"
        previous = Path(f"{path}.1")
        previous.write_text("first child: Metal command buffer error\n", encoding="utf-8")
        os.utime(previous, (1_000_000, 1_000_000))
        path.write_text("second child: model file is corrupt\n", encoding="utf-8")
        proc = MagicMock()
        setattr(proc, MLX_Engine._CHILD_LOG_ATTR, str(path))
        setattr(proc, MLX_Engine._CHILD_LOG_STARTED_ATTR, 2_000_000.0)

        output = MLX_Engine._read_child_output(proc)

        assert "model file is corrupt" in output
        assert "Metal command buffer error" not in output

    def test_spawn_records_when_this_child_started(self, tmp_path):
        from src.engines.mlx_engine import MLX_Engine

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        with (
            patch("src.engines.mlx_engine.mp.Process", return_value=MagicMock(pid=4321)),
            patch.object(child_log, "_log_dir", return_value=tmp_path / "logs"),
        ):
            before = time.time()
            handle = MLX_Engine._spawn_child(model_path=model_dir, alias="erudi-x", port=9087)
            after = time.time()

        started = getattr(handle["proc"], MLX_Engine._CHILD_LOG_STARTED_ATTR)
        assert before <= started <= after

    def test_read_child_output_explains_itself_when_there_is_no_file(self, tmp_path):
        from src.engines.mlx_engine import MLX_Engine

        proc = MagicMock()
        setattr(proc, MLX_Engine._CHILD_LOG_ATTR, str(tmp_path / "absent.log"))

        assert "no output" in MLX_Engine._read_child_output(proc).lower()

    def test_read_child_output_is_safe_without_a_captured_path(self):
        from src.engines.mlx_engine import MLX_Engine

        assert isinstance(MLX_Engine._read_child_output(None), str)
        assert isinstance(MLX_Engine._read_child_output(object()), str)

    def test_spawn_hands_the_path_to_the_child_and_keeps_it_on_the_proc(self, tmp_path):
        """The child cannot resolve the log directory itself: in a frozen build
        its runtime paths are uninitialized, so the parent -- which knows --
        passes the resolved path down."""
        from src.engines.mlx_engine import MLX_Engine

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        log_dir = tmp_path / "logs"
        captured: dict = {}

        def _fake_process(*, target, args, daemon):
            captured["args"] = args
            return MagicMock(pid=4321)

        with (
            patch("src.engines.mlx_engine.mp.Process", side_effect=_fake_process),
            patch.object(child_log, "_log_dir", return_value=log_dir),
        ):
            MLX_Engine._spawn_child(model_path=model_dir, alias="erudi-x", port=9087)

        expected = str(log_dir / "mlx-child-9087.log")
        assert captured["args"][1] == expected

    def test_an_unusable_log_directory_does_not_stop_the_spawn(self, tmp_path):
        """Capture is a diagnostic, never a precondition: a read-only log dir
        costs the tail, not the model."""
        from src.engines.mlx_engine import MLX_Engine

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        captured: dict = {}

        def _fake_process(*, target, args, daemon):
            captured["args"] = args
            return MagicMock(pid=4321)

        with (
            patch("src.engines.mlx_engine.mp.Process", side_effect=_fake_process),
            patch.object(child_log, "prepare_child_log", side_effect=OSError("read-only")),
        ):
            handle = MLX_Engine._spawn_child(model_path=model_dir, alias="erudi-x", port=9087)

        assert handle["port"] == 9087
        assert captured["args"][1] is None

    def test_an_orderly_stop_removes_the_file(self, tmp_path):
        from src.engines.mlx_engine import MLX_Engine

        path = tmp_path / "mlx-child-27300.log"
        path.write_text("goodbye\n", encoding="utf-8")
        proc = MagicMock()
        proc.is_alive.side_effect = [True, False]
        proc.exitcode = -15
        setattr(proc, MLX_Engine._CHILD_LOG_ATTR, str(path))

        MLX_Engine._terminate_process(proc)

        assert not path.exists()

    def test_a_child_that_died_on_its_own_keeps_its_last_words(self, tmp_path):
        """The next request respawns and reports the death; deleting the file
        here would throw away the only account of it."""
        from src.engines.mlx_engine import MLX_Engine

        path = tmp_path / "mlx-child-27300.log"
        path.write_text("Metal: command buffer error\n", encoding="utf-8")
        proc = MagicMock()
        proc.is_alive.return_value = False
        proc.exitcode = 1
        setattr(proc, MLX_Engine._CHILD_LOG_ATTR, str(path))

        MLX_Engine._terminate_process(proc)

        assert path.exists()


# =====================================================================
# The child entry point wires it up before mlx-vlm runs
# =====================================================================


@pytest.mark.unit
class TestRunnerRedirectsBeforeStarting:
    def test_redirect_happens_before_main(self, monkeypatch):
        from src.engines import _mlx_vlm_server_runner as runner

        order: list[str] = []
        monkeypatch.setattr(
            runner, "redirect_stdio_to", lambda path, **kw: order.append(f"redirect:{path}")
        )
        monkeypatch.setattr(
            runner, "_import_mlx_vlm_server_main", lambda: lambda: order.append("main")
        )

        runner.run_mlx_vlm_server(["mlx_vlm.server"], "/tmp/mlx-child-1.log")

        assert order == ["redirect:/tmp/mlx-child-1.log", "main"]

    def test_no_path_means_no_redirect(self, monkeypatch):
        from src.engines import _mlx_vlm_server_runner as runner

        calls: list[str] = []
        monkeypatch.setattr(
            runner, "redirect_stdio_to", lambda path, **kw: calls.append("redirect")
        )
        monkeypatch.setattr(runner, "_import_mlx_vlm_server_main", lambda: lambda: None)

        runner.run_mlx_vlm_server(["mlx_vlm.server"])

        assert calls == []

    def test_a_failed_capture_never_stops_the_server(self, monkeypatch, tmp_path):
        from src.engines import _mlx_vlm_server_runner as runner

        started: list[str] = []

        def _boom(path, **kwargs):
            raise OSError("no such log directory")

        monkeypatch.setattr(runner, "redirect_stdio_to", _boom)
        monkeypatch.setattr(
            runner, "_import_mlx_vlm_server_main", lambda: lambda: started.append("main")
        )

        runner.run_mlx_vlm_server(["mlx_vlm.server"], str(tmp_path / "nope" / "x.log"))

        assert started == ["main"]

    @pytest.mark.parametrize(
        "patch_name",
        [
            "_patch_gemma3_tied_lm_head_quant",
            "_patch_gemma_end_of_turn_stop",
            "_patch_inline_thinking",
        ],
    )
    def test_a_patch_that_did_not_apply_is_recorded(self, monkeypatch, patch_name):
        """Each patch fixes a user-visible defect (a Gemma3 quant that will not
        load, `<end_of_turn>` streamed as text, reasoning dropped on the floor).
        Returning False silently shipped a degraded product."""
        from src.engines import _mlx_vlm_server_runner as runner

        warnings: list[str] = []
        monkeypatch.setattr(runner, "child_warning", warnings.append)
        monkeypatch.setattr(runner, "redirect_stdio_to", lambda path, **kw: None)
        monkeypatch.setattr(runner, "_import_mlx_vlm_server_main", lambda: lambda: None)
        for name in (
            "_patch_gemma3_tied_lm_head_quant",
            "_patch_gemma_end_of_turn_stop",
            "_patch_inline_thinking",
        ):
            monkeypatch.setattr(runner, name, lambda *, applied=name != patch_name: applied)

        runner.run_mlx_vlm_server(["mlx_vlm.server"])

        assert len(warnings) == 1
        assert patch_name in warnings[0]

    def test_nothing_is_recorded_when_every_patch_applies(self, monkeypatch):
        from src.engines import _mlx_vlm_server_runner as runner

        warnings: list[str] = []
        monkeypatch.setattr(runner, "child_warning", warnings.append)
        monkeypatch.setattr(runner, "redirect_stdio_to", lambda path, **kw: None)
        monkeypatch.setattr(runner, "_import_mlx_vlm_server_main", lambda: lambda: None)
        for name in (
            "_patch_gemma3_tied_lm_head_quant",
            "_patch_gemma_end_of_turn_stop",
            "_patch_inline_thinking",
        ):
            monkeypatch.setattr(runner, name, lambda: True)

        runner.run_mlx_vlm_server(["mlx_vlm.server"])

        assert warnings == []
