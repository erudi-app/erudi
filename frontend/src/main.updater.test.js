/**
 * The main-process half of the manual update controls (#571).
 *
 * An update staged to install on quit is cancelled by the native installer
 * when the app reopens before the swap finishes, and nothing in the app is
 * told. The Settings card is the recovery path, and it can only work if main
 * forwards every updater event and answers a check, a download and a state
 * request on demand.
 *
 * main.js runs in the Electron main process and cannot be imported here, so
 * the contract is read from the source text, the way the CSP and crash-handler
 * tests read theirs.
 */
import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const source = readFileSync(fileURLToPath(new URL("./main.js", import.meta.url)), "utf8");
const preload = readFileSync(fileURLToPath(new URL("./preload.js", import.meta.url)), "utf8");

/** The body of `setupAutoUpdater`, where every updater event is wired. */
function setupBody() {
  const start = source.indexOf("function setupAutoUpdater()");
  expect(start, "setupAutoUpdater not found").toBeGreaterThan(-1);
  return source.slice(start, source.indexOf('ipcMain.on("updater:set-enabled"', start));
}

/** The body of one `ipcMain.handle("<channel>", ...)` block. */
function handlerBody(channel) {
  const start = source.indexOf(`ipcMain.handle("${channel}"`);
  expect(start, `${channel} handler not found`).toBeGreaterThan(-1);
  return source.slice(start, source.indexOf("\n});", start));
}

describe("updater events reaching the renderer", () => {
  it("forwards every event, including the two that used to be logged and dropped", () => {
    const body = setupBody();
    for (const event of [
      "checking-for-update",
      "update-available",
      "update-not-available",
      "download-progress",
      "update-downloaded",
      "error",
    ]) {
      expect(body, event).toMatch(new RegExp(`send\\("${event}"`));
    }
  });

  it("sends them all on the one channel the renderer already listens to", () => {
    expect(setupBody()).not.toMatch(/webContents\.send\("(?!updater-event)/);
    expect(source).toMatch(
      /mainWindow\.webContents\.send\("updater-event",\s*\{ event, \.\.\.payload \}\)/
    );
  });

  it("keeps the updater's own error message out of the payload", () => {
    // The detail belongs in the log. The card shows one translated line, and
    // an updater error message is a URL and an HTTP status, not copy.
    const errorHandler = setupBody().slice(setupBody().indexOf('autoUpdater.on("error"'));
    expect(errorHandler).toMatch(/logWarn\(/);
    expect(errorHandler).toMatch(/send\("error", \{\}\)/);
  });

  it("records the phase so a window that mounts late can catch up", () => {
    expect(source).toMatch(/let updaterState = \{ phase: "idle", version: null, percent: 0 \}/);
    const body = setupBody();
    for (const phase of ["checking", "available", "up-to-date", "downloading", "downloaded"]) {
      expect(body, phase).toMatch(new RegExp(`updaterState = \\{ phase: "${phase}"`));
    }
  });
});

describe("the manual check", () => {
  it("answers something the renderer can render when there is no updater", () => {
    expect(handlerBody("updater:check-now")).toMatch(
      /if \(!autoUpdater\) \{\s*return \{ ok: false, reason: "unavailable" \}/
    );
    expect(handlerBody("updater:download-now")).toMatch(
      /if \(!autoUpdater\) \{\s*return \{ ok: false, reason: "unavailable" \}/
    );
  });

  it("runs even when the automatic-updates preference is off", () => {
    const body = handlerBody("updater:check-now");
    expect(body).toMatch(/autoUpdater\.checkForUpdates\(\)/);
    // No early return on the preference: the button must work for someone who
    // refused automatic updates -- that is who needs it most.
    expect(body).not.toMatch(/if \(autoUpdateEnabled[^)]*\) \{\s*return \{ ok: false/);
  });

  it("does not download behind the back of a user who refused automatic updates", () => {
    expect(handlerBody("updater:check-now")).toMatch(
      /if \(autoUpdateEnabled !== true\) \{\s*autoUpdater\.autoDownload = false;/
    );
  });

  it("never re-arms the install-on-quit path the preference turned off", () => {
    expect(handlerBody("updater:check-now")).not.toMatch(/autoInstallOnAppQuit/);
    expect(handlerBody("updater:download-now")).not.toMatch(/autoInstallOnAppQuit/);
  });

  it("swallows the rejection instead of throwing it at the renderer", () => {
    for (const channel of ["updater:check-now", "updater:download-now"]) {
      const body = handlerBody(channel);
      expect(body, channel).toMatch(/catch \(err\) \{/);
      expect(body, channel).toMatch(/logWarn\(/);
      expect(body, channel).toMatch(/return \{ ok: false, reason: "error" \}/);
    }
  });
});

describe("the state request", () => {
  it("says whether an updater exists at all, and what it last did", () => {
    expect(handlerBody("updater:get-state")).toMatch(/available: autoUpdater !== null/);
    expect(handlerBody("updater:get-state")).toMatch(/\.\.\.updaterState/);
  });
});

describe("what #571 must not disturb", () => {
  it("leaves the install-on-quit path and the 4-hour cadence alone (#563 owns those)", () => {
    expect(source).toMatch(/autoUpdater\.autoInstallOnAppQuit = autoUpdateEnabled;/);
    expect(source).toMatch(/4 \* 60 \* 60 \* 1000/);
    expect(source).toMatch(/autoUpdater\.quitAndInstall\(false, true\)/);
  });

  it("reports an update failure on the card, never in a dialog", () => {
    const body = setupBody();
    expect(body).not.toMatch(/dialog\./);
    for (const channel of ["updater:check-now", "updater:download-now", "updater:get-state"]) {
      expect(handlerBody(channel), channel).not.toMatch(/dialog\./);
    }
  });
});

describe("the preload bridge", () => {
  it("exposes the three new calls the card needs", () => {
    expect(preload).toMatch(/checkNow: \(\) => ipcRenderer\.invoke\("updater:check-now"\)/);
    expect(preload).toMatch(/downloadNow: \(\) => ipcRenderer\.invoke\("updater:download-now"\)/);
    expect(preload).toMatch(/getState: \(\) => ipcRenderer\.invoke\("updater:get-state"\)/);
  });

  it("removes only its own listener, so the banner and the card can coexist", () => {
    expect(preload).toMatch(/ipcRenderer\.removeListener\("updater-event", handler\)/);
    expect(preload).not.toMatch(/removeAllListeners\("updater-event"\)/);
  });
});
