import { describe, it, expect } from "vitest";
import path from "node:path";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { resolveDataLocations, listClearTargets, isSafeToDelete } from "./dataLocations.js";

// The table the backend's runtime_paths.py is checked against too
// (backend/tests/test_runtime_paths.py): the folder Open data folder shows and
// Clear All Data empties is the backend's, on every platform.
const LAYOUT = JSON.parse(
  readFileSync(
    fileURLToPath(
      new URL("../../../backend/tests/fixtures/runtime_paths_layout.json", import.meta.url)
    ),
    "utf8"
  )
).cases;

const homeFor = (platform) => (platform === "win32" ? "C:\\Users\\me" : "/home/me");
const pathFor = (platform) => (platform === "win32" ? path.win32 : path.posix);

function locate(platform, env, extra = {}) {
  return resolveDataLocations({
    platform,
    env,
    home: homeFor(platform),
    tmpdir: platform === "win32" ? "C:\\Temp" : "/tmp",
    isPackaged: true,
    devBackendRoot: null,
    ...extra,
  });
}

describe("resolveDataLocations: the backend's layout", () => {
  it.each(LAYOUT.map((c) => [c.name, c]))("%s", (_name, c) => {
    const p = pathFor(c.platform);
    const home = homeFor(c.platform);
    const env = Object.fromEntries(
      Object.entries(c.env).map(([key, relative]) => [key, p.join(home, relative)])
    );
    const relative = (target) => p.relative(home, target).split(p.sep).join("/");

    const locations = locate(c.platform, env);

    expect(relative(locations.dataDir)).toBe(c.data_dir);
    expect(relative(locations.backendLogDir)).toBe(c.log_dir);
  });

  it("puts pasted images inside the data folder", () => {
    const locations = locate("linux", {});
    expect(locations.pastedImagesDir).toBe(
      "/home/me/.local/share/erudi/backend/prod/data/pasted-images"
    );
  });

  it("uses the repository's backend folder in a dev run", () => {
    const locations = resolveDataLocations({
      platform: "darwin",
      env: {},
      home: "/home/me",
      tmpdir: "/tmp",
      isPackaged: false,
      devBackendRoot: "/src/erudi/backend",
    });
    expect(locations.dataDir).toBe("/src/erudi/backend/data");
    expect(locations.backendLogDir).toBe("/src/erudi/backend/logs");
  });

  it("names the Electron folder of an ERUDI_DATA_ROOT instance", () => {
    const locations = locate("linux", { ERUDI_DATA_ROOT: "/srv/qa" });
    expect(locations.instanceRoot).toBe("/srv/qa");
    expect(locations.electronDir).toBe("/srv/qa/electron");
  });

  it("has no instance root by default, and keeps the app log in the temporary folder", () => {
    const locations = locate("darwin", {});
    expect(locations.instanceRoot).toBeNull();
    expect(locations.electronDir).toBeNull();
    expect(locations.appLogDir).toBe("/tmp");
  });

  it("keeps an ERUDI_DATA_ROOT instance's app log in its own log folder", () => {
    expect(locate("linux", { ERUDI_DATA_ROOT: "/srv/qa" }).appLogDir).toBe("/srv/qa/logs");
  });

  it("resolves a relative or ~ ERUDI_DATA_ROOT to an absolute folder", () => {
    expect(locate("linux", { ERUDI_DATA_ROOT: "~/qa" }).instanceRoot).toBe("/home/me/qa");
    expect(locate("linux", { ERUDI_DATA_ROOT: "qa" }).instanceRoot).toBe(path.posix.resolve("qa"));
  });
});

describe("listClearTargets", () => {
  const electron = (userData, electronLogsDir) => ({
    userData,
    electronLogsDir,
    appLogFiles: ["/tmp/erudi-backend.log", "/tmp/erudi-backend.old.log"],
  });

  it("macOS: Erudi's own subfolders, never the shared Application Support/erudi root", () => {
    const targets = listClearTargets(
      locate("darwin", {}),
      electron("/home/me/Library/Application Support/Erudi", "/home/me/Library/Logs/Erudi")
    );
    expect(targets).toEqual([
      "/home/me/Library/Application Support/erudi/backend",
      "/home/me/Library/Logs/erudi",
      "/home/me/Library/Application Support/erudi/pasted-images",
      "/home/me/Library/Logs/Erudi",
      "/home/me/Library/Application Support/Erudi/last-seen-version",
      "/home/me/Library/Application Support/Erudi/update-notes.json",
      "/tmp/erudi-backend.log",
      "/tmp/erudi-backend.old.log",
    ]);
    expect(targets).not.toContain("/home/me/Library/Application Support/erudi");
    expect(targets).not.toContain("/home/me/Library/Application Support/Erudi");
  });

  it("Windows: the backend folder and logs under LOCALAPPDATA, the Electron logs under APPDATA", () => {
    const targets = listClearTargets(
      locate("win32", { LOCALAPPDATA: "C:\\Users\\me\\AppData\\Local" }),
      {
        userData: "C:\\Users\\me\\AppData\\Roaming\\Erudi",
        electronLogsDir: "C:\\Users\\me\\AppData\\Roaming\\Erudi\\logs",
        appLogFiles: ["C:\\Temp\\erudi-backend.log"],
      }
    );
    expect(targets).toEqual([
      "C:\\Users\\me\\AppData\\Local\\erudi\\backend",
      "C:\\Users\\me\\AppData\\Local\\erudi\\logs",
      "C:\\Users\\me\\AppData\\Local\\erudi\\pasted-images",
      "C:\\Users\\me\\AppData\\Roaming\\Erudi\\logs",
      "C:\\Users\\me\\AppData\\Roaming\\Erudi\\last-seen-version",
      "C:\\Users\\me\\AppData\\Roaming\\Erudi\\update-notes.json",
      "C:\\Temp\\erudi-backend.log",
    ]);
  });

  it("Linux: the XDG data and state folders, and the pasted images an earlier version put under ~/Library", () => {
    const targets = listClearTargets(
      locate("linux", {}),
      electron("/home/me/.config/Erudi", "/home/me/.config/Erudi/logs")
    );
    expect(targets.slice(0, 4)).toEqual([
      "/home/me/.local/share/erudi/backend",
      "/home/me/.local/state/erudi/logs",
      "/home/me/Library/Application Support/erudi/pasted-images",
      "/home/me/.config/Erudi/logs",
    ]);
  });

  it("ERUDI_DATA_ROOT: the instance's data, logs and Electron files, never the root itself", () => {
    const targets = listClearTargets(
      locate("linux", { ERUDI_DATA_ROOT: "/srv/qa" }),
      electron("/srv/qa/electron", "/srv/qa/electron/logs")
    );
    expect(targets).toEqual([
      "/srv/qa/data",
      "/srv/qa/logs",
      "/srv/qa/electron/logs",
      "/srv/qa/electron/last-seen-version",
      "/srv/qa/electron/update-notes.json",
      "/tmp/erudi-backend.log",
      "/tmp/erudi-backend.old.log",
    ]);
    expect(targets).not.toContain("/srv/qa");
    expect(targets).not.toContain("/srv/qa/electron");
  });

  it("leaves out the Electron logs folder when there is none to clear (dev run)", () => {
    const targets = listClearTargets(
      resolveDataLocations({
        platform: "darwin",
        env: {},
        home: "/home/me",
        tmpdir: "/tmp",
        isPackaged: false,
        devBackendRoot: "/src/erudi/backend",
      }),
      electron("/home/me/Library/Application Support/erudi", null)
    );
    expect(targets).not.toContain("/home/me/Library/Logs/erudi");
    expect(targets.slice(0, 2)).toEqual(["/src/erudi/backend/data", "/src/erudi/backend/logs"]);
  });
});

describe("isSafeToDelete", () => {
  const guard = {
    platform: "linux",
    home: "/home/me",
    protectedDirs: ["/srv/qa", "/srv/qa/electron"],
  };

  it("accepts a folder Erudi owns", () => {
    expect(isSafeToDelete("/srv/qa/data", guard)).toBe(true);
    expect(isSafeToDelete("/home/me/.local/share/erudi/backend", guard)).toBe(true);
  });

  it("refuses the file system root, the home folder and anything above it", () => {
    expect(isSafeToDelete("/", guard)).toBe(false);
    expect(isSafeToDelete("/home", guard)).toBe(false);
    expect(isSafeToDelete("/home/me", guard)).toBe(false);
    expect(isSafeToDelete("/home/me/", guard)).toBe(false);
  });

  it("refuses a root it does not own and anything above it", () => {
    expect(isSafeToDelete("/srv/qa", guard)).toBe(false);
    expect(isSafeToDelete("/srv/qa/electron", guard)).toBe(false);
    expect(isSafeToDelete("/srv", guard)).toBe(false);
  });

  it("refuses a relative or empty path", () => {
    expect(isSafeToDelete("data", guard)).toBe(false);
    expect(isSafeToDelete("", guard)).toBe(false);
  });

  it("refuses a Windows drive root", () => {
    expect(
      isSafeToDelete("C:\\", { platform: "win32", home: "C:\\Users\\me", protectedDirs: [] })
    ).toBe(false);
    expect(
      isSafeToDelete("C:\\Users\\me\\AppData\\Local\\erudi\\backend", {
        platform: "win32",
        home: "C:\\Users\\me",
        protectedDirs: [],
      })
    ).toBe(true);
  });
});
