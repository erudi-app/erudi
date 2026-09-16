import json
import time
from pathlib import Path

import pytest

from erudi_eval import lifecycle
from erudi_eval.app_control import clean_env, quit_command
from erudi_eval.cdp import CdpClient, CdpError, pick_page_target
from erudi_eval.layout import resolve_layout
from erudi_eval.phases import Config, Run
from erudi_eval.platform_info import disk_allows_download
from erudi_eval.util import utc_iso

GB = 1024**3


def test_pick_page_target_skips_devtools_and_non_pages():
    targets = [
        {"type": "page", "url": "devtools://devtools/bundled/inspector.html", "webSocketDebuggerUrl": "ws://a"},
        {"type": "service_worker", "url": "file:///x", "webSocketDebuggerUrl": "ws://b"},
        {"type": "page", "url": "file:///Applications/Erudi.app/Contents/Resources/app.asar/.webpack/renderer/main_window/index.html#/erudi/models", "webSocketDebuggerUrl": "ws://c"},
    ]
    assert pick_page_target(targets)["webSocketDebuggerUrl"] == "ws://c"
    assert pick_page_target([targets[0]]) is None


class FakeWs:
    def __init__(self, replies):
        self.sent, self.replies = [], list(replies)

    def send(self, raw):
        self.sent.append(json.loads(raw))

    def settimeout(self, t):
        pass

    def recv(self):
        return json.dumps(self.replies.pop(0))


def test_cdp_call_ignores_events_and_matches_ids():
    c = CdpClient()
    c.ws = FakeWs([{"method": "Page.frameNavigated", "params": {}}, {"id": 1, "result": {"metrics": [{"name": "Nodes", "value": 812}, {"name": "Other", "value": 1}]}}])
    assert c.call("Performance.getMetrics") == {"metrics": [{"name": "Nodes", "value": 812}, {"name": "Other", "value": 1}]}
    c.ws = FakeWs([{"id": 2, "error": {"message": "boom"}}])
    with pytest.raises(CdpError):
        c.call("Runtime.evaluate")


def test_cdp_ui_input_sequence():
    c = CdpClient()
    c.ws = FakeWs([{"id": 1, "result": {"result": {"value": True}}}, {"id": 2, "result": {}}, {"id": 3, "result": {}}, {"id": 4, "result": {}}])
    c.type_into_composer("hello")
    c.press_enter()
    methods = [m["method"] for m in c.ws.sent]
    assert methods == ["Runtime.evaluate", "Input.insertText", "Input.dispatchKeyEvent", "Input.dispatchKeyEvent"]
    assert "lucide-arrow-right" in c.ws.sent[0]["params"]["expression"]
    assert c.ws.sent[2]["params"]["key"] == "Enter" and c.ws.sent[2]["params"]["type"] == "rawKeyDown"


def test_layouts_per_os(tmp_path):
    mac = resolve_layout(os_name="darwin", env={"TMPDIR": "/var/folders/x/T"})
    assert str(mac.main_exe) == "/Applications/Erudi.app/Contents/MacOS/Erudi"
    assert mac.capture_logs[0] == Path("/var/folders/x/T/erudi-backend.log")
    assert str(mac.backend_lib).endswith("Contents/Resources/backend/_internal")
    win = resolve_layout(os_name="windows", env={"LOCALAPPDATA": "C:/Users/u/AppData/Local", "APPDATA": "C:/Users/u/AppData/Roaming", "TEMP": "C:/Temp"})
    assert win.main_exe == Path("C:/Users/u/AppData/Local/Programs/Erudi/Erudi.exe")
    assert win.data_root == Path("C:/Users/u/AppData/Local/erudi/backend/prod")
    appimage = tmp_path / "Erudi-1.1.2-cuda.AppImage"
    appimage.write_text("")
    lin = resolve_layout(str(appimage), os_name="linux", env={"XDG_DATA_HOME": "/d", "XDG_STATE_HOME": "/s"})
    assert lin.main_exe == appimage and lin.data_root == Path("/d/erudi/backend/prod") and lin.backend_log_dir == Path("/s/erudi/logs")
    mount = tmp_path / ".mount_ErudiX"
    (mount / "resources").mkdir(parents=True)
    lin.learn_main_exe(str(mount / "erudi"))
    assert lin.resources_dir == mount / "resources" and str(mount) in lin.install_dirs


def test_quit_command_never_targets_a_foreign_app():
    mac = resolve_layout(os_name="darwin")
    assert quit_command(mac, 42, "/Applications/Erudi.app/Contents/MacOS/Erudi") == ["osascript", "-e", 'quit app "Erudi"']
    assert quit_command(mac, 42, "/usr/bin/python3") is None  # falls back to SIGTERM on the PID
    win = resolve_layout(os_name="windows", env={})
    assert quit_command(win, 42, "C:/x/Erudi.exe") == ["taskkill", "/PID", "42"]


def test_disk_gate():
    assert disk_allows_download(10 * GB, 3 * GB, 5)[0] is True
    ok, why = disk_allows_download(7 * GB, 3 * GB, 5)
    assert not ok and "headroom" in why
    assert disk_allows_download(None, 1, 5)[0] is False and disk_allows_download(100 * GB, None, 5)[0] is False


def test_phase_selection(tmp_path):
    run = Run(Config(results_dir=tmp_path, corpus_dir=tmp_path, phases=["chat_cold", "arena", "cleanup"], skip=["chat_cold"], opt_in=[]), resolve_layout(os_name="darwin"), run_id="sel")
    try:
        assert run.selected("chat_cold") == (False, "skipped by --skip")
        assert run.selected("model_ready") == (False, "not selected by --phases")
        assert run.selected("arena")[0] is False and "opt-in" in run.selected("arena")[1]
        assert run.selected("cleanup")[0] is False and "--cleanup" in run.selected("cleanup")[1]
    finally:
        run.events.close()


def test_copy_logs_only_this_runs_lines(tmp_path):
    t0 = time.time()
    capture = tmp_path / "erudi-backend.log"
    capture.write_text("[2020-01-01T00:00:00.000Z] Backend stdout: old\n" f"[{utc_iso(t0 + 1)}] Backend stdout: new\n")
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "backend.log").write_text("[INFO] 2020-01-01T00:00:00.000Z [be-1] - old\n" f"[INFO] {utc_iso(t0 + 1)} [eval-x-1] - Turn mode: agentic KB (kb_id=3)\n")
    (logs / "mlx-child-27300.log").write_text("loading weights\nserver up\n")
    counts = lifecycle.copy_logs(tmp_path / "out", t0, [capture], logs)
    assert counts == {"erudi-backend.log": 1, "backend.log": 1, "mlx-child-27300.log": 2}
    assert "old" not in (tmp_path / "out" / "backend.log").read_text()
    assert lifecycle.find_log_lines(logs, "Turn mode:", t0) and not lifecycle.find_log_lines(logs, "Turn mode:", t0 + 5)


def test_clean_env_drops_the_harness_virtualenv():
    env = clean_env({"VIRTUAL_ENV": "/h/.venv", "PATH": "/h/.venv/bin:/usr/bin", "PYTHONPATH": "/x", "UV_CACHE_DIR": "/c", "HOME": "/Users/u"})
    assert env == {"PATH": "/usr/bin", "HOME": "/Users/u"}


def test_attach_on_linux_without_install_is_refused(monkeypatch, capsys):
    """With no install dir to scope discovery, any process named erudi could be
    elected main and quit-signalled: attach must demand --app-path instead."""
    from erudi_eval import cli
    from erudi_eval import layout as layout_mod

    real = layout_mod.resolve_layout
    monkeypatch.setattr(layout_mod, "resolve_layout", lambda app_path=None: real(app_path, os_name="linux", env={}))
    assert cli.main(["run", "--attach"]) == 2
    assert "--app-path" in capsys.readouterr().err
