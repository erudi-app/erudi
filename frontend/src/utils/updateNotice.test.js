// @vitest-environment jsdom
/**
 * The "Updated to X.Y.Z" dialog (#572): when it shows, what it shows, and when
 * the version is recorded as seen.
 *
 * The page decides at startup from what main reports (the running version,
 * the last version seen, the notes kept from the download) and what the
 * backend reports (whether the database holds user data), then tells main to
 * record the version once the dialog is dismissed, or at once when there is
 * nothing to show. Notes are only ever text.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  compareVersions,
  decideUpdateNotice,
  loadUpdateNotice,
  releaseNotesToLines,
  releasePageUrl,
} from "./updateNotice";

describe("compareVersions", () => {
  it.each([
    ["1.1.4", "1.1.3", 1],
    ["1.1.3", "1.1.4", -1],
    ["1.1.4", "1.1.4", 0],
    ["1.10.0", "1.9.9", 1],
    ["2.0.0", "1.99.99", 1],
    ["1.2.0", "1.2.0-beta.1", 1],
    ["1.2.0-beta.1", "1.2.0", -1],
    ["v1.2.0", "1.2.0", 0],
  ])("%s vs %s", (a, b, expected) => {
    expect(Math.sign(compareVersions(a, b))).toBe(expected);
  });

  it("answers null for something that is not a version", () => {
    expect(compareVersions("garbage", "1.1.4")).toBeNull();
    expect(compareVersions("1.1.4", "")).toBeNull();
  });
});

describe("decideUpdateNotice", () => {
  const notes = (version, text = "<ul><li>Faster loads</li></ul>") => ({ version, notes: text });

  it("does nothing when the running version was already seen", () => {
    expect(
      decideUpdateNotice({
        runningVersion: "1.1.4",
        lastSeenVersion: "1.1.4",
        notes: notes("1.1.4"),
        hasUserData: true,
      })
    ).toEqual({ action: "none" });
  });

  it("marks a fresh install as seen without a word", () => {
    expect(
      decideUpdateNotice({
        runningVersion: "1.1.4",
        lastSeenVersion: null,
        notes: null,
        hasUserData: false,
      })
    ).toEqual({ action: "mark" });
  });

  it("explains the opt-ins on the first launch of this version over an older one", () => {
    // 1.1.3 recorded no last-seen version and kept no notes.
    expect(
      decideUpdateNotice({
        runningVersion: "1.1.4",
        lastSeenVersion: null,
        notes: null,
        hasUserData: true,
      })
    ).toEqual({ action: "show", version: "1.1.4", lines: null, explainOptIns: true });
  });

  it("waits for another launch when it cannot tell a fresh install from an update", () => {
    expect(
      decideUpdateNotice({
        runningVersion: "1.1.4",
        lastSeenVersion: null,
        notes: null,
        hasUserData: null,
      })
    ).toEqual({ action: "none" });
  });

  it("shows the notes kept from the download when they are this version's", () => {
    expect(
      decideUpdateNotice({
        runningVersion: "1.1.5",
        lastSeenVersion: "1.1.4",
        notes: notes("1.1.5"),
        hasUserData: true,
      })
    ).toEqual({
      action: "show",
      version: "1.1.5",
      lines: ["• Faster loads"],
      explainOptIns: false,
    });
  });

  it("shows the version and the link only after an upgrade without notes", () => {
    expect(
      decideUpdateNotice({
        runningVersion: "1.1.5",
        lastSeenVersion: "1.1.4",
        notes: null,
        hasUserData: true,
      })
    ).toEqual({ action: "show", version: "1.1.5", lines: null, explainOptIns: false });
  });

  it("never shows the notes of another version (downloaded, never installed)", () => {
    expect(
      decideUpdateNotice({
        runningVersion: "1.1.6",
        lastSeenVersion: "1.1.4",
        notes: notes("1.1.5"),
        hasUserData: true,
      })
    ).toEqual({ action: "show", version: "1.1.6", lines: null, explainOptIns: false });
  });

  it("says nothing on a downgrade and keeps the higher version recorded", () => {
    // Recording 1.1.3 would bring back a dialog already dismissed for 1.1.4
    // once the user upgrades again.
    expect(
      decideUpdateNotice({
        runningVersion: "1.1.3",
        lastSeenVersion: "1.1.4",
        notes: notes("1.1.4"),
        hasUserData: true,
      })
    ).toEqual({ action: "none" });
  });

  it("treats an unreadable last-seen version like a missing one", () => {
    expect(
      decideUpdateNotice({
        runningVersion: "1.1.4",
        lastSeenVersion: "not a version",
        notes: null,
        hasUserData: false,
      })
    ).toEqual({ action: "mark" });
  });

  it("does nothing without a running version", () => {
    expect(
      decideUpdateNotice({
        runningVersion: null,
        lastSeenVersion: "1.1.4",
        notes: null,
        hasUserData: true,
      })
    ).toEqual({ action: "none" });
  });
});

describe("releaseNotesToLines: notes are text only", () => {
  it("strips GitHub's HTML to text, one line per block and per list item", () => {
    const html =
      '<h2>What\'s Changed</h2>\n<ul>\n<li>feat: faster loads by <a href="https://github.com/x">@x</a> in <a href="https://github.com/erudi-app/erudi/pull/1">#1</a></li>\n<li>fix: <strong>no</strong> crash</li>\n</ul>\n<p><strong>Full Changelog</strong>: <a href="https://github.com/erudi-app/erudi/compare/v1.1.4...v1.1.5">v1.1.4...v1.1.5</a></p>';
    expect(releaseNotesToLines(html)).toEqual([
      "What's Changed",
      "• feat: faster loads by @x in #1",
      "• fix: no crash",
      "Full Changelog: v1.1.4...v1.1.5",
    ]);
  });

  it("keeps no image, script or style, only their surrounding text", () => {
    expect(
      releaseNotesToLines(
        '<p>Before<img src="https://tracker.example/pixel.png" alt="pixel"> after</p><script>alert(1)</script><style>p{}</style>'
      )
    ).toEqual(["Before after"]);
  });

  it("decodes entities into plain characters", () => {
    expect(
      releaseNotesToLines("<p>Fish &amp; chips &lt;3 &#233;t&#xE9; &quot;ok&quot;&nbsp;!</p>")
    ).toEqual(['Fish & chips <3 été "ok" !']);
  });

  it("keeps the lines of plain-text notes", () => {
    expect(releaseNotesToLines("First change\n\n- second change\n")).toEqual([
      "First change",
      "- second change",
    ]);
  });

  it("flattens structured notes, each under its version", () => {
    expect(
      releaseNotesToLines([
        { version: "1.1.5", note: "<ul><li>One</li></ul>" },
        { version: "1.1.4", note: null },
        { version: "1.1.3", note: "Two" },
      ])
    ).toEqual(["1.1.5", "• One", "1.1.3", "Two"]);
  });

  it("answers null when there is nothing to show", () => {
    for (const notes of [
      null,
      undefined,
      "",
      "   ",
      "<p> </p>",
      [],
      [{ version: "1", note: "" }],
      42,
    ]) {
      expect(releaseNotesToLines(notes), JSON.stringify(notes)).toBeNull();
    }
  });
});

describe("releaseNotesToLines: bounded work on any input (P3-3)", () => {
  const fast = (input) => {
    const started = performance.now();
    releaseNotesToLines(input);
    return performance.now() - started;
  };

  it("handles 200k whitespace characters inside HTML quickly", () => {
    expect(fast(`<p>a${" \n\t".repeat(70000)}b</p>`)).toBeLessThan(300);
    expect(fast(" ".repeat(200000))).toBeLessThan(300);
  });

  it("handles unclosed scripts, comments and tags quickly", () => {
    expect(fast("<script>".repeat(30000))).toBeLessThan(300);
    expect(fast("<!--".repeat(50000))).toBeLessThan(300);
    expect(fast("<".repeat(200000))).toBeLessThan(300);
    expect(fast("<a".repeat(100000) + ">")).toBeLessThan(300);
    expect(fast("&a".repeat(100000))).toBeLessThan(300);
  });

  it("reads at most 64 KB of notes", () => {
    const lines = releaseNotesToLines(`<p>${"x".repeat(200000)}</p>`);
    expect(lines.join("").length).toBeLessThanOrEqual(64 * 1024);
  });

  it("drops an unclosed script and everything after it", () => {
    expect(releaseNotesToLines("<p>Kept</p><script>alert(1)<p>gone</p>")).toEqual(["Kept"]);
  });

  it("reads tags named like built-in properties as plain tags", () => {
    expect(releaseNotesToLines("<p><constructor>a</constructor><toString>b</p>")).toEqual(["ab"]);
  });

  it("keeps a lone angle bracket as text", () => {
    expect(releaseNotesToLines("<p>1 < 2 and <3</p>")).toEqual(["1 < 2 and <3"]);
  });
});

describe("releasePageUrl", () => {
  it("points at the tag of the version on the project's releases", () => {
    expect(releasePageUrl("1.1.4")).toBe("https://github.com/erudi-app/erudi/releases/tag/v1.1.4");
  });
});

describe("loadUpdateNotice", () => {
  let getLaunchState;
  let markVersionSeen;
  let client;

  beforeEach(() => {
    getLaunchState = vi.fn();
    markVersionSeen = vi.fn().mockResolvedValue({ ok: true });
    window.updaterAPI = { getLaunchState, markVersionSeen };
    client = { get: vi.fn() };
  });

  afterEach(() => {
    delete window.updaterAPI;
  });

  it("asks nothing of the backend and records nothing when the version was seen", async () => {
    getLaunchState.mockResolvedValue({
      running_version: "1.1.4",
      last_seen_version: "1.1.4",
      notes: null,
    });

    await expect(loadUpdateNotice(client)).resolves.toBeNull();
    expect(client.get).not.toHaveBeenCalled();
    expect(markVersionSeen).not.toHaveBeenCalled();
  });

  it("records a fresh install as seen at once and shows nothing", async () => {
    getLaunchState.mockResolvedValue({
      running_version: "1.1.4",
      last_seen_version: null,
      notes: null,
    });
    client.get.mockResolvedValue({ has_user_data: false });

    await expect(loadUpdateNotice(client)).resolves.toBeNull();
    expect(client.get).toHaveBeenCalledWith("/startup/install_state");
    expect(markVersionSeen).toHaveBeenCalledTimes(1);
  });

  it("returns the dialog for an updated install and leaves the recording to the dismissal", async () => {
    getLaunchState.mockResolvedValue({
      running_version: "1.1.4",
      last_seen_version: null,
      notes: null,
    });
    client.get.mockResolvedValue({ has_user_data: true });

    await expect(loadUpdateNotice(client)).resolves.toEqual({
      action: "show",
      version: "1.1.4",
      lines: null,
      explainOptIns: true,
    });
    expect(markVersionSeen).not.toHaveBeenCalled();
  });

  it("does not ask the backend when the last-seen version is known", async () => {
    getLaunchState.mockResolvedValue({
      running_version: "1.1.5",
      last_seen_version: "1.1.4",
      notes: { version: "1.1.5", notes: "Faster" },
    });

    await expect(loadUpdateNotice(client)).resolves.toMatchObject({
      action: "show",
      lines: ["Faster"],
    });
    expect(client.get).not.toHaveBeenCalled();
  });

  it("shows nothing and records nothing when the backend cannot answer", async () => {
    getLaunchState.mockResolvedValue({
      running_version: "1.1.4",
      last_seen_version: null,
      notes: null,
    });
    client.get.mockRejectedValue(new Error("backend down"));

    await expect(loadUpdateNotice(client)).resolves.toBeNull();
    expect(markVersionSeen).not.toHaveBeenCalled();
  });

  it("shows nothing when main cannot answer, or outside Electron", async () => {
    getLaunchState.mockRejectedValue(new Error("no handler"));
    await expect(loadUpdateNotice(client)).resolves.toBeNull();

    delete window.updaterAPI;
    await expect(loadUpdateNotice(client)).resolves.toBeNull();
  });
});
