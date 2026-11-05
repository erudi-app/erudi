/**
 * The main-process half of the update controls (#571, #563, #572).
 *
 * main.js runs in the Electron main process and cannot be imported here, so
 * the contract is read from the source text, the way the CSP and crash-handler
 * tests read theirs. The decisions themselves live in utils/updaterPolicy.js
 * and are tested there per platform; these tests pin that main.js wires them
 * and nothing else.
 */
import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const source = readFileSync(fileURLToPath(new URL("./main.js", import.meta.url)), "utf8");
const preload = readFileSync(fileURLToPath(new URL("./preload.js", import.meta.url)), "utf8");

/** The body of a top-level `function <name>(`. */
function functionBody(name) {
  const start = source.indexOf(`function ${name}(`);
  expect(start, `${name} not found`).toBeGreaterThan(-1);
  return source.slice(start, source.indexOf("\n}\n", start));
}

/** The body of one `ipcMain.handle("<channel>", ...)` block. */
function handlerBody(channel) {
  const start = source.indexOf(`ipcMain.handle("${channel}"`);
  expect(start, `${channel} handler not found`).toBeGreaterThan(-1);
  return source.slice(start, source.indexOf("\n});", start));
}

describe("updater events reaching the renderer", () => {
  it("forwards every event, including the two that used to be logged and dropped", () => {
    const body = functionBody("setupAutoUpdater");
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
    expect(source).not.toMatch(/webContents\.send\("(?!updater-event|backend-event)/);
    expect(functionBody("sendUpdaterEvent")).toMatch(
      /mainWindow\.webContents\.send\("updater-event",\s*\{ event, \.\.\.payload \}\)/
    );
  });

  it("keeps the updater's own error message out of the payload", () => {
    const body = functionBody("setupAutoUpdater");
    const errorHandler = body.slice(body.indexOf('autoUpdater.on("error"'));
    expect(errorHandler).toMatch(/logWarn\(/);
    expect(errorHandler).toMatch(/send\("error", \{\}\)/);
  });

  it("records the phase so a window that mounts late can catch up", () => {
    expect(source).toMatch(/let updaterState = \{ phase: "idle", version: null, percent: 0 \}/);
    const body = functionBody("setupAutoUpdater");
    for (const phase of ["checking", "available", "up-to-date", "downloading", "downloaded"]) {
      expect(body, phase).toMatch(new RegExp(`updaterState = \\{ phase: "${phase}"`));
    }
  });

  it("tells the banner whether a download follows the check on its own", () => {
    expect(functionBody("setupAutoUpdater")).toMatch(
      /send\("update-available", \{ version: info\.version, downloading: autoUpdater\.autoDownload \}\)/
    );
  });
});

describe("the library flags (#563)", () => {
  it("lowers both right after loading the library, whose defaults are on", () => {
    const load = source.slice(source.indexOf('require("electron-updater")'));
    const lowered = load.slice(0, load.indexOf("} catch"));
    expect(lowered).toMatch(/autoUpdater\.autoDownload = false;/);
    expect(lowered).toMatch(/autoUpdater\.autoInstallOnAppQuit = false;/);
  });

  it("raises them only from the policy's plan for a known snapshot", () => {
    const body = functionBody("applyUpdatePreferences");
    expect(body).toMatch(/if \(!autoUpdater \|\| updatePreferences === null\) \{\s*return;/);
    expect(body).toMatch(/planPreferenceChange\(/);
    expect(body).toMatch(/autoUpdater\.autoDownload = plan\.autoDownload;/);
    expect(body).toMatch(/autoUpdater\.autoInstallOnAppQuit = plan\.autoInstallOnAppQuit;/);
    const assignments = source.match(/autoUpdater\.autoInstallOnAppQuit = /g);
    expect(assignments).toHaveLength(2); // lowered at load, then the plan
  });

  it("resolves every snapshot the renderer sends through the policy", () => {
    const start = source.indexOf('ipcMain.on("updater:set-preferences"');
    expect(start).toBeGreaterThan(-1);
    const body = source.slice(start, source.indexOf("\n});", start));
    expect(body).toMatch(/updatePreferences = resolveUpdatePreferences\(snapshot\);/);
    expect(body).toMatch(/applyUpdatePreferences\(\);/);
  });

  it("logs whether a check will run, in the words QA reads", () => {
    const body = functionBody("applyUpdatePreferences");
    expect(body).toMatch(/no check will run/);
    expect(body).toMatch(/checking now, then every 4 hours/);
    expect(body).toMatch(/4 \* 60 \* 60 \* 1000/);
  });
});

describe("install turned on after a finished download (#563)", () => {
  it("registers the library's quit handler on Windows and Linux", () => {
    expect(functionBody("armDownloadedInstall")).toMatch(/autoUpdater\.addQuitHandler\(\);/);
  });

  it("hands the update to Squirrel.Mac again on macOS", () => {
    expect(functionBody("armDownloadedInstall")).toMatch(
      /require\("electron"\)\.autoUpdater\.checkForUpdates\(\);/
    );
    expect(functionBody("armDownloadedInstall")).toMatch(/sendUpdaterEvent\("install-handed-off"/);
  });

  it("never installs from the quit path, which would run a visible installer", () => {
    expect(functionBody("armDownloadedInstall")).not.toMatch(/quitAndInstall/);
    const beforeQuit = source.slice(source.indexOf('app.on("before-quit"'));
    expect(beforeQuit.slice(0, beforeQuit.indexOf("\n});"))).not.toMatch(/quitAndInstall/);
    const installs = source.match(/quitAndInstall\(/g);
    expect(installs).toHaveLength(1); // the Install now button only
    expect(handlerBody("updater:install-now")).toMatch(
      /autoUpdater\.quitAndInstall\(false, true\)/
    );
  });

  it("records what the library armed at download from the flag at that moment", () => {
    const body = functionBody("setupAutoUpdater");
    const downloaded = body.slice(body.indexOf('autoUpdater.on("update-downloaded"'));
    expect(downloaded).toMatch(/installArmedAfterDownload\(\{/);
    expect(downloaded).toMatch(/autoInstallOnAppQuit: autoUpdater\.autoInstallOnAppQuit/);
    expect(downloaded).toMatch(/downloadedVersion = info\.version;/);
  });
});

describe("the manual check", () => {
  it("answers something the renderer can render when there is no updater", () => {
    for (const channel of ["updater:check-now", "updater:download-now"]) {
      expect(handlerBody(channel), channel).toMatch(
        /if \(!autoUpdater\) \{\s*return \{ ok: false, reason: "unavailable" \}/
      );
    }
  });

  it("runs even when Check for updates is off", () => {
    const body = handlerBody("updater:check-now");
    expect(body).toMatch(/autoUpdater\.checkForUpdates\(\)/);
    expect(body).not.toMatch(/if \(!?updatePreferences[^)]*\) \{\s*return \{ ok: false/);
  });

  it("downloads what it finds only when Download updates automatically is on", () => {
    expect(handlerBody("updater:check-now")).toMatch(
      /autoUpdater\.autoDownload = updatePreferences\?\.download === true;/
    );
  });

  it("never arms an install at quit", () => {
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

  it("reports an update failure on the card, never in a dialog", () => {
    expect(functionBody("setupAutoUpdater")).not.toMatch(/dialog\./);
    for (const channel of ["updater:check-now", "updater:download-now", "updater:get-state"]) {
      expect(handlerBody(channel), channel).not.toMatch(/dialog\./);
    }
  });
});

describe("the state request", () => {
  it("says whether an updater exists, what it last did, and whether macOS has the update", () => {
    const body = handlerBody("updater:get-state");
    expect(body).toMatch(/available: autoUpdater !== null/);
    expect(body).toMatch(/\.\.\.updaterState/);
    expect(body).toMatch(/handedOff: isHandedOff\(installArmed\)/);
  });
});

describe("the update notice files (#572)", () => {
  it("live in the user data folder", () => {
    expect(functionBody("userDataFile")).toMatch(/app\.getPath\("userData"\)/);
    expect(source).toMatch(/const UPDATE_NOTES_FILE = "update-notes\.json";/);
    expect(source).toMatch(/const LAST_SEEN_VERSION_FILE = "last-seen-version";/);
  });

  it("keep the notes received with each download", () => {
    const body = functionBody("setupAutoUpdater");
    const downloaded = body.slice(body.indexOf('autoUpdater.on("update-downloaded"'));
    expect(downloaded).toMatch(/keepReleaseNotes\(info\);/);
    expect(functionBody("keepReleaseNotes")).toMatch(
      /JSON\.stringify\(\{ version: info\.version, notes: info\.releaseNotes \?\? null \}\)/
    );
  });

  it("report the running version, the last one seen and the notes", () => {
    const body = handlerBody("updater:get-launch-state");
    expect(body).toMatch(/running_version: app\.getVersion\(\)/);
    expect(body).toMatch(/last_seen_version: lastSeen/);
    expect(body).toMatch(/notes/);
    expect(body).not.toMatch(/https?:|net\.|fetch\(/);
  });

  it("record the running version, never one the renderer names", () => {
    const body = handlerBody("updater:mark-version-seen");
    expect(body).toMatch(/ipcMain\.handle\("updater:mark-version-seen", \(\) => \{/);
    expect(body).toMatch(
      /writeFileSync\(userDataFile\(LAST_SEEN_VERSION_FILE\), app\.getVersion\(\)\)/
    );
  });
});

describe("the preload bridge", () => {
  it("exposes the calls the cards need", () => {
    expect(preload).toMatch(/checkNow: \(\) => ipcRenderer\.invoke\("updater:check-now"\)/);
    expect(preload).toMatch(/downloadNow: \(\) => ipcRenderer\.invoke\("updater:download-now"\)/);
    expect(preload).toMatch(/getState: \(\) => ipcRenderer\.invoke\("updater:get-state"\)/);
  });

  it("carries the full snapshot of the three switches", () => {
    expect(preload).toMatch(
      /setPreferences: \(snapshot\) => ipcRenderer\.send\("updater:set-preferences", snapshot\)/
    );
    expect(preload).not.toMatch(/setAutoUpdateEnabled|updater:set-enabled/);
  });

  it("exposes the update notice calls", () => {
    expect(preload).toMatch(
      /getLaunchState: \(\) => ipcRenderer\.invoke\("updater:get-launch-state"\)/
    );
    expect(preload).toMatch(
      /markVersionSeen: \(\) => ipcRenderer\.invoke\("updater:mark-version-seen"\)/
    );
  });

  it("removes only its own listener, so the banner and the cards can coexist", () => {
    expect(preload).toMatch(/ipcRenderer\.removeListener\("updater-event", handler\)/);
    expect(preload).not.toMatch(/removeAllListeners\("updater-event"\)/);
  });
});
