/**
 * The main-process half of the update controls (#571, #563, #572).
 *
 * main.js runs in the Electron main process and cannot be imported here. What
 * the updater does lives in utils/updaterWiring.js, tested behaviourally
 * against fakes of the library; these tests pin, from the source text, that
 * main.js only configures the updater right after loading it, hands it to the
 * wiring and routes the IPC there, and that the "Updated to" files stay in
 * the user data folder.
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

/** The body of one `ipcMain.handle|on("<channel>", ...)` block. */
function handlerBody(channel, kind = "handle") {
  const start = source.indexOf(`ipcMain.${kind}("${channel}"`);
  expect(start, `${channel} handler not found`).toBeGreaterThan(-1);
  const end = source.indexOf("\n});", start);
  const endExpr = source.indexOf("\n);", start);
  return source.slice(start, Math.min(...[end, endExpr].filter((i) => i > -1)));
}

describe("loading the updater", () => {
  it("configures it right after the require, before anything can run", () => {
    const load = source.slice(source.indexOf('require("electron-updater")'));
    expect(load.slice(0, load.indexOf("} catch"))).toMatch(/configureUpdater\(autoUpdater\);/);
  });

  it("leaves every flag and header to the wiring", () => {
    expect(source).not.toMatch(/autoUpdater\.(autoDownload|autoInstallOnAppQuit|requestHeaders) =/);
    expect(source).not.toMatch(
      /autoUpdater\.(checkForUpdates|downloadUpdate|quitAndInstall|addQuitHandler)\(/
    );
  });

  it("hands the updater, and Squirrel.Mac on macOS, to the wiring", () => {
    const body = functionBody("setupAutoUpdater");
    expect(body).toMatch(/createUpdaterWiring\(\{/);
    expect(body).toMatch(/updater: autoUpdater/);
    expect(body).toMatch(
      /nativeUpdater: process\.platform === "darwin" \? require\("electron"\)\.autoUpdater : null/
    );
    expect(body).toMatch(/send: sendUpdaterEvent/);
    expect(body).toMatch(/onDownloaded: keepReleaseNotes/);
  });
});

describe("the IPC routes to the wiring", () => {
  it("answers unavailable without an updater", () => {
    for (const channel of ["updater:check-now", "updater:download-now"]) {
      expect(handlerBody(channel), channel).toMatch(
        /if \(!updaterWiring\) \{\s*return \{ ok: false, reason: "unavailable" \}/
      );
    }
    expect(handlerBody("updater:get-state")).toMatch(/available: false/);
  });

  it("routes each call", () => {
    expect(handlerBody("updater:check-now")).toMatch(/updaterWiring\.checkNow\(\)/);
    expect(handlerBody("updater:download-now")).toMatch(/updaterWiring\.downloadNow\(\)/);
    expect(handlerBody("updater:install-now")).toMatch(/updaterWiring\?\.installNow\(\)/);
    expect(handlerBody("updater:get-state")).toMatch(/updaterWiring\.getState\(\)/);
    expect(handlerBody("updater:set-preferences", "on")).toMatch(
      /updaterWiring\?\.setPreferences\(snapshot\)/
    );
  });

  it("sends every updater event on the one channel the renderer listens to", () => {
    expect(functionBody("sendUpdaterEvent")).toMatch(
      /mainWindow\.webContents\.send\("updater-event",\s*\{ event, \.\.\.payload \}\)/
    );
  });

  it("never installs from the quit path", () => {
    const beforeQuit = source.slice(source.indexOf('app.on("before-quit"'));
    expect(beforeQuit.slice(0, beforeQuit.indexOf("\n});"))).not.toMatch(/install/i);
  });
});

describe("the update notice files (#572)", () => {
  it("live in the user data folder", () => {
    expect(functionBody("userDataFile")).toMatch(/app\.getPath\("userData"\)/);
    expect(source).toMatch(/const UPDATE_NOTES_FILE = "update-notes\.json";/);
    expect(source).toMatch(/const LAST_SEEN_VERSION_FILE = "last-seen-version";/);
  });

  it("keep the notes received with each download", () => {
    expect(functionBody("keepReleaseNotes")).toMatch(
      /JSON\.stringify\(\{ version: info\.version, notes: info\.releaseNotes \?\? null \}\)/
    );
  });

  it("report the running version, the last one seen and the notes, with no network", () => {
    const body = handlerBody("updater:get-launch-state");
    expect(body).toMatch(/running_version: app\.getVersion\(\)/);
    expect(body).toMatch(/last_seen_version: lastSeen/);
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
