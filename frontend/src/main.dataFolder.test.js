/**
 * Open Data Folder and Clear All Data act on the folders the backend uses, on
 * every platform (#676), and the image reader only reads images.
 *
 * The decisions live in utils/dataLocations.js, utils/clearAllData.js and
 * utils/imageReader.js (tested on their own); main.js runs in the Electron
 * main process and cannot be imported here, so its wiring is read from the
 * source text, as the other main.*.test.js files do.
 */
import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const source = readFileSync(fileURLToPath(new URL("./main.js", import.meta.url)), "utf8");
const preload = readFileSync(fileURLToPath(new URL("./preload.js", import.meta.url)), "utf8");

const indexOf = (pattern) => {
  const match = source.match(pattern);
  expect(match, String(pattern)).not.toBeNull();
  return match.index;
};

describe("where main looks for Erudi's folders", () => {
  it("resolves them through utils/dataLocations.js, never a per-OS path of its own", () => {
    expect(source).toMatch(/const dataLocations = resolveDataLocations\(\{/);
    expect(source).not.toMatch(/Application Support/);
    expect(source).not.toMatch(/LOCALAPPDATA/);
    expect(source).not.toMatch(/XDG_/);
    expect(source).not.toMatch(/function getDataDirectory/);
  });

  it("moves the Electron folders of an ERUDI_DATA_ROOT instance before electron-updater loads", () => {
    const setUserData = indexOf(/app\.setPath\("userData", dataLocations\.electronDir\)/);
    const setLogs = indexOf(/app\.setAppLogsPath\(/);
    const updater = indexOf(/require\("electron-updater"\)/);
    const lock = indexOf(/app\.requestSingleInstanceLock\(\)/);
    expect(setUserData).toBeLessThan(updater);
    expect(setLogs).toBeLessThan(updater);
    expect(setUserData).toBeLessThan(lock);
  });

  it("writes electron-log's file in Electron's logs folder, the one Clear All Data deletes", () => {
    expect(source).toMatch(
      /transports\.file\.resolvePathFn = \(vars\) =>\s*path\.join\(vars\.electronDefaultDir, vars\.fileName\)/
    );
  });

  it("opens the backend's data folder", () => {
    expect(source).toMatch(/label: t\("menu\.help\.openDataFolder"\),\s*click: openDataFolder/);
    const body = source.match(/async function openDataFolder\(\) \{[\s\S]*?\n\}/);
    expect(body).not.toBeNull();
    expect(body[0]).toMatch(/const dataDir = dataLocations\.dataDir;/);
    expect(body[0]).toMatch(/await shell\.openPath\(dataDir\)/);
  });

  it("writes the app log where dataLocations puts it (the instance's own folder under ERUDI_DATA_ROOT)", () => {
    expect(source).toMatch(
      /const logFile = path\.join\(dataLocations\.appLogDir, "erudi-backend\.log"\);/
    );
    expect(source).not.toMatch(/os\.tmpdir\(\), "erudi-backend/);
  });

  it("keeps pasted images in the data folder", () => {
    expect(source).toMatch(
      /fs\.mkdirSync\(dataLocations\.pastedImagesDir, \{ recursive: true \}\)/
    );
  });

  it("allows revealing a log only in the folders the backend and main write logs to", () => {
    const dirs = source.match(/function knownLogDirectories\(\) \{[\s\S]*?\n\}/);
    expect(dirs).not.toBeNull();
    expect(dirs[0]).toMatch(/dataLocations\.backendLogDir/);
  });
});

describe("Clear All Data", () => {
  const handler = () => {
    const body = source.match(/async function clearAllDataFromMenu\(\) \{[\s\S]*?\n\}/);
    expect(body).not.toBeNull();
    return body[0];
  };

  it("is reached from the Help menu only: the dead renderer bridges are gone", () => {
    expect(source).toMatch(/label: t\("menu\.help\.clearAllData"\),\s*click: clearAllDataFromMenu/);
    expect(source).not.toMatch(/"data:clearAll"/);
    expect(source).not.toMatch(/"data:openFolder"/);
    expect(preload).not.toMatch(/electronAPI/);
  });

  it("marks the app as quitting, then stops the backend the way a quit does", () => {
    const body = handler();
    expect(body.indexOf("shuttingDown = true")).toBeGreaterThan(-1);
    expect(body.indexOf("shuttingDown = true")).toBeLessThan(body.indexOf("clearAllData({"));
    expect(body).toMatch(
      /requestStop\(backendProcess\);\s*await gracefulShutdown\(backendProcess, \{ killFn: killBackend \}\);/
    );
  });

  it("blanks the window, clears the interface's storage through the session, then deletes the folders", () => {
    const body = handler();
    expect(body).toMatch(/quietRenderer: async \(\) => \{[\s\S]*loadURL\("about:blank"\)/);
    expect(body).toMatch(/clearStorage: clearInterfaceStorage/);
    expect(source).toMatch(/session\.defaultSession\.clearStorageData\(\)/);
    expect(body).toMatch(/targets: clearTargets\(\)/);
    expect(body).toMatch(/isSafe: isSafeClearTarget/);
    expect(body).toMatch(/app\.quit\(\);/);
  });

  it("finishes an unfinished clearing at the next start, before the window and the backend", () => {
    const ready = source.match(
      /\.whenReady\(\)\s*\.then\(async \(\) => \{([\s\S]*?)createWindow\(\);/
    );
    expect(ready).not.toBeNull();
    expect(ready[1]).toMatch(/await finishPendingClear\(\);/);
  });

  it("never deletes the instance root or the Electron user data folder itself", () => {
    expect(source).toMatch(
      /protectedDirs: \[dataLocations\.instanceRoot, app\.getPath\("userData"\)\]/
    );
  });
});

describe("fs:readImageAsDataURL", () => {
  it("goes through the bounded reader and answers {ok, ...}, never a raw read", () => {
    const body = source.match(/ipcMain\.handle\("fs:readImageAsDataURL"[\s\S]*?\n\}\);/);
    expect(body).not.toBeNull();
    expect(body[0]).toMatch(/await readImageAsDataUrl\(filePath\)/);
    expect(body[0]).not.toMatch(/readFileSync/);
  });
});
