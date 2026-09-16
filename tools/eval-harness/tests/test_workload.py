import json
from pathlib import Path

import pytest

from erudi_eval import workload
from erudi_eval.discovery import ProcInfo

HARNESS = Path(__file__).resolve().parent.parent
CLONE = "/private/var/folders/fw/x/X/com.google.Chrome.code_sign_clone/code_sign_clone.Kpe/Google Chrome.app.bundle/Contents/MacOS/Google Chrome"
AGENT_EXE = "/Users/u/.local/share/agent-cli/versions/2.1.272"


def shipped_examples():
    """The files that go to the repository; local-*.json belong to a machine and are gitignored."""
    return [p for p in (HARNESS / "workloads").glob("*.json") if not p.name.startswith("local-")]


def P(pid, ppid, name, exe, *cmd):
    return ProcInfo(pid, ppid, name, exe, tuple(cmd) or (exe,))


# Modelled on a read-only look at this Mac (2026-09-16): the browser's main process runs from a code-sign
# clone, the coding-agent CLI's executable basename is its version while argv[0] is the CLI name, and MCP
# servers it spawned are left orphaned to launchd.
PROCS = [
    P(1, 0, "launchd", "/sbin/launchd"),
    P(100, 1, "Google Chrome", CLONE),
    P(101, 100, "Google Chrome Helper (Renderer)", "/Applications/Google Chrome.app/Contents/Frameworks/Google Chrome Framework.framework/Helpers/Google Chrome Helper (Renderer).app/Contents/MacOS/Google Chrome Helper (Renderer)", "x", "--type=renderer"),
    P(102, 100, "chrome-native-host", "/Applications/Some Extension.app/Contents/Helpers/chrome-native-host"),
    P(200, 1, "Discord", "/Applications/Discord.app/Contents/MacOS/Discord"),
    P(300, 1, "ghostty", "/Applications/Ghostty.app/Contents/MacOS/ghostty"),
    P(301, 300, "login", "/usr/bin/login"),
    P(302, 301, "zsh", "/bin/zsh"),
    P(310, 302, "2.1.272", AGENT_EXE, "agent-cli", "--resume", "A"),
    P(311, 310, "node", "/opt/homebrew/bin/node", "node", "npm exec shadcn@latest mcp"),
    P(312, 310, "2.1.272", AGENT_EXE, "agent-cli", "-p", "subagent"),  # nested CLI process: not a session
    P(320, 302, "2.1.272", AGENT_EXE, "agent-cli"),
    P(321, 320, "zsh", "/bin/zsh", "/bin/zsh", "-c", "uv run erudi_eval.py run"),
    P(322, 321, "python3.12", "/usr/bin/python3", "python", "erudi_eval.py"),  # the harness
    P(323, 322, "Erudi", "/Applications/Erudi.app/Contents/MacOS/Erudi"),  # launched by the harness
    P(400, 1, "node", "/opt/homebrew/bin/node", "node", "/Users/u/.agent-cli/plugins/cache/x/server.js"),
]

WL = {
    "profile": "nominal",
    "groups": [
        {"name": "chrome", "exe_contains": ["/Google Chrome.app/", "com.google.Chrome.code_sign_clone"], "include_descendants": True,
         "declared": {"chrome_tabs": 4, "tabs": ["Gmail", "Google Docs", "YouTube paused", "GitHub"]}},
        {"name": "discord", "exe_prefixes": ["/Applications/Discord.app/"]},
        {"name": "slack", "exe_prefixes": ["/Applications/Slack.app/"]},
        {"name": "ghostty", "exe_prefixes": ["/Applications/Ghostty.app/"]},
        {"name": "coding_agent_cli", "argv0_basenames": ["agent-cli"], "exe_contains": ["/agent-cli/versions/"], "sessions": True, "expected_count": 3},
        {"name": "coding_agent_children", "descendants_of": "coding_agent_cli"},
        {"name": "coding_agent_orphans", "cmdline_contains": ["/.agent-cli/plugins/"], "orphaned": True, "optional": True},
    ],
}


def test_load_validates_and_expands(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", "C:/Users/u/AppData/Local")
    f = tmp_path / "w.json"
    f.write_text(json.dumps({"profile": "nominal", "groups": [{"name": "discord", "exe_prefixes": ["%LOCALAPPDATA%/Discord/"]}]}))
    wl = workload.load_workload(f)
    assert wl.groups[0].exe_prefixes == ("C:/Users/u/AppData/Local/Discord/",)
    f.write_text(json.dumps({"groups": [{"name": "x", "exe_prefix": ["typo"]}]}))
    with pytest.raises(ValueError, match="exe_prefix"):
        workload.load_workload(f)
    f.write_text(json.dumps({"groups": [{"name": "kids", "descendants_of": "nobody"}]}))
    with pytest.raises(ValueError, match="nobody"):
        workload.load_workload(f)


def test_assign_groups_rules_sessions_descendants_and_exclusions():
    wl = workload.parse_workload(WL)
    erudi = {323}
    a = workload.assign_groups(PROCS, wl, erudi_pids=erudi, harness_pid=322)
    assert a["chrome"].pids == {100, 101, 102}  # clone path main + helper + its native host child
    assert a["discord"].pids == {200}
    assert a["slack"].pids == set()
    assert a["ghostty"].pids == {300}  # shells are not ghostty (no include_descendants)
    assert a["coding_agent_cli"].pids == {310, 312, 320} and a["coding_agent_cli"].session_pids == {310, 320}
    assert a["coding_agent_children"].pids == {311, 321}  # the harness and the Erudi it launched are never captured
    assert a["coding_agent_orphans"].pids == {400}
    everything = set().union(*(g.pids for g in a.values()))
    assert not everything & {322, 323}


def test_verify_presence_warnings():
    wl = workload.parse_workload(WL)
    a = workload.assign_groups(PROCS, wl, erudi_pids={323}, harness_pid=322)
    checks = {c["group"]: c for c in workload.verify_presence(a, wl)}
    assert checks["coding_agent_cli"]["status"] == "warning" and "expected 3 sessions, found 2" in checks["coding_agent_cli"]["message"]
    assert checks["slack"]["status"] == "warning" and "not running" in checks["slack"]["message"]
    assert checks["discord"]["status"] == "ok"
    assert checks["coding_agent_orphans"]["status"] == "ok"  # optional
    clean = workload.parse_workload({"profile": "clean", "groups": [{"name": "chrome", "exe_contains": ["com.google.Chrome.code_sign_clone"], "expect_absent": True},
                                                                   {"name": "slack", "exe_prefixes": ["/Applications/Slack.app/"], "expect_absent": True}]})
    c2 = {c["group"]: c for c in workload.verify_presence(workload.assign_groups(PROCS, clean, set(), 322), clean)}
    assert c2["chrome"]["status"] == "warning" and "expected absent" in c2["chrome"]["message"]
    assert c2["slack"]["status"] == "ok"


def rec(phase, count, mem_mb, cpu, rss_mb=None, primary=True):
    return {"phase": phase, "workload": {"g": {"count": count, "primary_mb": mem_mb if primary else None, "rss_mb": rss_mb, "cpu_percent": cpu, "unreadable_primary": 0}}}


def test_workload_drift_count_memory_cpu_and_reference():
    samples = [rec("baseline", 2, 100, 1), rec("baseline", 2, 102, 3), rec("cold_boot", 2, 110, 2), rec("chat_cold", 3, 140, 5), rec("chat_cold", 3, 150, 45)]
    out = workload.workload_drift(samples, drift_pct=25, busy_cpu=20)
    assert out["reference_phase"] == "baseline"
    kinds = {(w["kind"], w["phase"]) for w in out["warnings"]}
    assert ("count_changed", "chat_cold") in kinds
    assert ("memory_drift", "chat_cold") in kinds and ("memory_drift", "cold_boot") not in kinds
    assert ("busy", "chat_cold") in kinds and ("busy", "baseline") not in kinds
    msg = next(w for w in out["warnings"] if w["kind"] == "memory_drift")["message"]
    assert "g" in msg and "chat_cold" in msg and "+44" in msg
    # No baseline (attach): the first phase is the reference; RSS is compared to RSS when primary is unreadable.
    s2 = [rec("preflight", 1, None, 0, rss_mb=50, primary=False), rec("chat_cold", 1, None, 0, rss_mb=80, primary=False)]
    out2 = workload.workload_drift(s2, 25, 20)
    assert out2["reference_phase"] == "preflight" and out2["warnings"][0]["metric"] == "rss_mb"


def test_shipped_workload_files_load():
    names = sorted(p.name for p in shipped_examples())
    assert names == ["clean-mac.json", "clean-windows.json", "nominal-mac.json", "nominal-windows.json"]
    for p in shipped_examples():
        wl = workload.load_workload(p)
        assert wl.profile in ("clean", "nominal") and wl.groups
    nominal = workload.load_workload(HARNESS / "workloads" / "nominal-mac.json")
    groups = {g.name: g for g in nominal.groups}
    assert groups["coding_agent_cli"].expected_count is None  # the shipped file is an example: counts belong to a local copy
    assert "<" in groups["coding_agent_cli"].raw_rules["exe_contains"][0]  # a placeholder, not a real path
    # On a machine that has a local file the default resolves to it; otherwise to the shipped example.
    assert workload.default_workload_path(HARNESS, "nominal", "darwin").name in ("local-nominal-mac.json", "nominal-mac.json")
    assert workload.default_workload_path(HARNESS, "stress", "darwin") is None


def test_placeholders_make_a_group_not_configured_instead_of_matching_everything(tmp_path):
    f = tmp_path / "example.json"
    f.write_text(json.dumps({"profile": "nominal", "groups": [
        {"name": "browser", "exe_contains": ["<path fragment of your browser install>"]},
        {"name": "coding_agent_cli", "argv0_basenames": ["<basename of your agent CLI>"], "exe_contains": ["/agent-cli/versions/"], "sessions": True},
        {"name": "coding_agent_children", "descendants_of": "coding_agent_cli"},
    ]}))
    wl = workload.load_workload(f)
    browser, cli = wl.groups[0], wl.groups[1]
    assert browser.unconfigured and not browser.has_rules  # no rule left: it can never match
    assert not cli.unconfigured and cli.exe_contains == ("/agent-cli/versions/",)  # the filled rule still works
    a = workload.assign_groups(PROCS, wl, erudi_pids=set(), harness_pid=322)
    assert a["browser"].pids == set() and a["coding_agent_cli"].pids == {310, 312, 320}
    checks = {c["group"]: c for c in workload.verify_presence(a, wl)}
    assert checks["browser"]["status"] == "not configured" and "local" in checks["browser"]["message"]
    assert checks["coding_agent_cli"]["status"] == "ok"


def test_every_shipped_example_is_loadable_and_uses_neutral_group_names():
    for p in shipped_examples():
        wl = workload.load_workload(p)
        names = {g.name for g in wl.groups}
        assert names <= {"browser", "chat_app", "terminal", "coding_agent_cli", "coding_agent_children", "coding_agent_orphans"}, p.name
        assert any(g.unconfigured for g in wl.groups), f"{p.name} should ship placeholders, not real machine paths"


def test_a_local_workload_file_wins_over_the_shipped_example(tmp_path):
    (tmp_path / "workloads").mkdir()
    example = tmp_path / "workloads" / "nominal-mac.json"
    example.write_text('{"profile": "nominal", "groups": [{"name": "browser", "exe_contains": ["<fragment>"]}]}')
    assert workload.default_workload_path(tmp_path, "nominal", "darwin") == example
    local = tmp_path / "workloads" / "local-nominal-mac.json"
    local.write_text('{"profile": "nominal", "groups": [{"name": "browser", "exe_contains": ["/Applications/Some.app/"]}]}')
    assert workload.default_workload_path(tmp_path, "nominal", "darwin") == local
