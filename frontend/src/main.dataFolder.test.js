/**
 * Open Data Folder and Clear All Data act on the folders the backend uses, on
 * every platform (#676), and the image reader only reads images.
 *
 * What happens, and in which order, is tested on fakes where it lives:
 * utils/dataLocations.js, utils/clearAllData.js (the clearing and the start
 * that finishes it), utils/appStartup.js (Electron paths before the updater,
 * the pending clear before the window), utils/imageReader.js,
 * utils/pastedImage.js. main.js runs in the Electron main process and cannot
 * be imported here, so this file only checks, from the source text, that
 * main.js hands its work to those functions and keeps no path of its own.
 */
import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const source = readFileSync(fileURLToPath(new URL("./main.js", import.meta.url)), "utf8");
const preload = readFileSync(fileURLToPath(new URL("./preload.js", import.meta.url)), "utf8");

describe("main.js delegates Erudi's folders", () => {
  it("resolves them through utils/dataLocations.js, never a per-OS path of its own", () => {
    expect(source).toMatch(/const dataLocations = resolveDataLocations\(\{/);
    expect(source).not.toMatch(/Application Support/);
    expect(source).not.toMatch(/LOCALAPPDATA/);
    expect(source).not.toMatch(/XDG_/);
  });

  it("hands the backend the ERUDI_DATA_ROOT it resolved", () => {
    expect(source).toMatch(/process\.env\.ERUDI_DATA_ROOT = dataLocations\.instanceRoot;/);
  });

  it("sets the Electron folders and loads the updater through setUpElectronPaths, before the lock", () => {
    const setUp = source.indexOf("setUpElectronPaths({");
    expect(setUp).toBeGreaterThan(-1);
    expect(setUp).toBeLessThan(source.indexOf("app.requestSingleInstanceLock()"));
    // The updater is loaded nowhere else.
    expect(source.match(/require\("electron-updater"\)/g)).toHaveLength(1);
    expect(source.indexOf('require("electron-updater")')).toBeGreaterThan(setUp);
    expect(source).not.toMatch(/app\.setPath\("userData"/);
  });

  it("writes the app log and the pasted images where dataLocations puts them", () => {
    expect(source).toMatch(
      /const logFile = path\.join\(dataLocations\.appLogDir, "erudi-backend\.log"\);/
    );
    expect(source).toMatch(/savePastedImage\(dataUrl, dataLocations\.pastedImagesDir\)/);
  });

  it("opens the backend's data folder", () => {
    expect(source).toMatch(/label: t\("menu\.help\.openDataFolder"\),\s*click: openDataFolder/);
    expect(source).toMatch(/const dataDir = dataLocations\.dataDir;/);
  });
});

describe("main.js delegates Clear All Data", () => {
  it("runs it through createClearAllDataFlow, from the Help menu only", () => {
    expect(source).toMatch(/const clearAllDataFlow = createClearAllDataFlow\(\{/);
    expect(source).toMatch(
      /click: clearAllDataFromMenu,\s*(\/\/[^\n]*\n\s*)*visible: clearAllDataAvailable/
    );
    expect(source).not.toMatch(/"data:clearAll"/);
    expect(source).not.toMatch(/"data:openFolder"/);
    expect(preload).not.toMatch(/electronAPI/);
  });

  it("is hidden in a dev run without ERUDI_DATA_ROOT", () => {
    expect(source).toMatch(
      /const clearAllDataAvailable = app\.isPackaged \|\| Boolean\(dataLocations\.instanceRoot\);/
    );
  });

  it("opens the first window through openFirstWindow, which finishes a pending clear first", () => {
    const ready = source.match(
      /\.whenReady\(\)\s*\.then\(async \(\) => \{([\s\S]*?)\n {4}if \(!app\.isPackaged\)/
    );
    expect(ready).not.toBeNull();
    expect(ready[1]).toMatch(/await openFirstWindow\(\{/);
    expect(ready[1]).toMatch(/finishPendingClear: \(\) => clearAllDataFlow\.finishPending\(\)/);
    expect(ready[1]).not.toMatch(/\bcreateWindow\(\)/);
  });

  it("guards every deletion with the home folder, the instance root and the profile folder", () => {
    expect(source).toMatch(
      /protectedDirs: \[dataLocations\.instanceRoot, app\.getPath\("userData"\)\]/
    );
    expect(source).toMatch(/isOwned: isOwnedFolder/);
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
