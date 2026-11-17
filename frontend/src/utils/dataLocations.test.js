import { describe, it, expect, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  resolveDataLocations,
  listClearTargets,
  isSafeToDelete,
  isUnsafeInstanceRoot,
  instanceRootFrom,
  OWNERSHIP_SENTINEL,
  PASTED_IMAGES_FOLDER,
} from "./dataLocations.js";

// The table the backend's runtime_paths.py is checked against too
// (backend/tests/test_runtime_paths.py): the folder Open data folder shows and
// Clear All Data empties is the backend's, on every platform.
const TABLE = JSON.parse(
  fs.readFileSync(
    fileURLToPath(
      new URL("../../../backend/tests/fixtures/runtime_paths_layout.json", import.meta.url)
    ),
    "utf8"
  )
);

const homeFor = (platform) => (platform === "win32" ? "C:\\Users\\me" : "/home/me");
const pathFor = (platform) => (platform === "win32" ? path.win32 : path.posix);
// Pure path arithmetic: nothing of these made-up paths exists on this machine.
const noRealpath = () => {
  throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
};

function locate(platform, env, extra = {}) {
  return resolveDataLocations({
    platform,
    env,
    home: homeFor(platform),
    tmpdir: platform === "win32" ? "C:\\Temp" : "/tmp",
    cwd: platform === "win32" ? "C:\\work" : "/work",
    isPackaged: true,
    devBackendRoot: null,
    ...extra,
  });
}

describe("resolveDataLocations: the backend's layout", () => {
  it("uses the ownership file name the backend writes", () => {
    expect(OWNERSHIP_SENTINEL).toBe(TABLE.ownership_sentinel);
  });

  it("writes pasted images in the folder whose files the backend deletes with a conversation", () => {
    expect(PASTED_IMAGES_FOLDER).toBe(TABLE.pasted_images_folder);
    const locations = locate("darwin", {});
    expect(locations.pastedImagesDir).toBe(
      path.posix.join(locations.dataDir, TABLE.pasted_images_folder)
    );
  });

  it.each(TABLE.cases.map((c) => [c.name, c]))("%s", (_name, c) => {
    const p = pathFor(c.platform);
    const home = homeFor(c.platform);
    // As in the table: empty and ~ values as is, the others under home.
    const env = Object.fromEntries(
      Object.entries(c.env).map(([key, value]) => [
        key,
        value === "" || value.startsWith("~") ? value : p.join(home, value),
      ])
    );
    const relative = (target) => p.relative(home, target).split(p.sep).join("/");

    const locations = locate(c.platform, env);

    expect(relative(locations.dataDir)).toBe(c.data_dir);
    expect(relative(locations.backendLogDir)).toBe(c.log_dir);
  });

  it("puts pasted images inside the data folder", () => {
    expect(locate("linux", {}).pastedImagesDir).toBe(
      "/home/me/.local/share/erudi/backend/prod/data/pasted-images"
    );
  });

  it("uses the repository's backend folder in a dev run", () => {
    const locations = locate("darwin", {}, { isPackaged: false, devBackendRoot: "/src/backend" });
    expect(locations.dataDir).toBe("/src/backend/data");
    expect(locations.backendLogDir).toBe("/src/backend/logs");
  });

  it("names the Electron folder of an ERUDI_DATA_ROOT instance and keeps its app log there", () => {
    const locations = locate("linux", { ERUDI_DATA_ROOT: "/srv/qa" });
    expect(locations.instanceRoot).toBe("/srv/qa");
    expect(locations.electronDir).toBe("/srv/qa/electron");
    expect(locations.appLogDir).toBe("/srv/qa/logs");
  });

  it("has no instance root by default, and keeps the app log in the temporary folder", () => {
    const locations = locate("darwin", {});
    expect(locations.instanceRoot).toBeNull();
    expect(locations.electronDir).toBeNull();
    expect(locations.appLogDir).toBe("/tmp");
  });
});

describe("instanceRootFrom: ~ and relative roots, as the backend reads them", () => {
  const resolve = (platform, raw) =>
    instanceRootFrom(
      { ERUDI_DATA_ROOT: raw },
      { platform, home: homeFor(platform), cwd: platform === "win32" ? "C:\\work" : "/work" }
    );

  it.each([
    ["linux", "~", "/home/me"],
    ["linux", "~/qa", "/home/me/qa"],
    ["linux", "~bob/qa", "/home/bob/qa"],
    ["linux", "qa", "/work/qa"],
    ["linux", "../qa", "/qa"],
    ["darwin", "~/qa", "/home/me/qa"],
    ["win32", "~\\qa", "C:\\Users\\me\\qa"],
    ["win32", "~/qa", "C:\\Users\\me\\qa"],
    ["win32", "~bob\\qa", "C:\\Users\\bob\\qa"],
    ["win32", "qa", "C:\\work\\qa"],
  ])("%s: %s -> %s", (platform, raw, expected) => {
    expect(resolve(platform, raw)).toBe(expected);
  });

  it("is null when unset or empty", () => {
    expect(instanceRootFrom({}, { platform: "linux", home: "/home/me", cwd: "/" })).toBeNull();
    expect(
      instanceRootFrom({ ERUDI_DATA_ROOT: "" }, { platform: "linux", home: "/home/me", cwd: "/" })
    ).toBeNull();
  });
});

describe("listClearTargets", () => {
  const electron = (userData, electronLogsDir) => ({
    userData,
    electronLogsDir,
    appLogFiles: ["/tmp/erudi-backend.log", "/tmp/erudi-backend.old.log"],
  });

  it("macOS: Erudi's own folders, each guarded by its ownership file, never the shared root", () => {
    const targets = listClearTargets(
      locate("darwin", {}),
      electron("/home/me/Library/Application Support/Erudi", "/home/me/Library/Logs/Erudi")
    );
    const data = "/home/me/Library/Application Support/erudi/backend/prod/data";
    expect(targets).toEqual([
      { path: "/home/me/Library/Application Support/erudi/backend", owner: data },
      { path: "/home/me/Library/Logs/erudi", owner: "/home/me/Library/Logs/erudi" },
      { path: "/home/me/Library/Application Support/erudi/pasted-images", owner: null },
      { path: "/home/me/Library/Logs/Erudi", owner: "/home/me/Library/Logs/Erudi" },
      { path: "/home/me/Library/Application Support/Erudi/last-seen-version", owner: null },
      { path: "/home/me/Library/Application Support/Erudi/update-notes.json", owner: null },
      { path: "/tmp/erudi-backend.log", owner: null },
      { path: "/tmp/erudi-backend.old.log", owner: null },
    ]);
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
    expect(targets.map((t) => t.path)).toEqual([
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
    expect(targets.slice(0, 4).map((t) => t.path)).toEqual([
      "/home/me/.local/share/erudi/backend",
      "/home/me/.local/state/erudi/logs",
      "/home/me/Library/Application Support/erudi/pasted-images",
      "/home/me/.config/Erudi/logs",
    ]);
  });

  it("ERUDI_DATA_ROOT: every target needs its ownership file, the root is never one", () => {
    const targets = listClearTargets(locate("linux", { ERUDI_DATA_ROOT: "/srv/qa" }), {
      userData: "/srv/qa/electron",
      electronLogsDir: "/srv/qa/electron/logs",
      appLogFiles: ["/srv/qa/logs/erudi-backend.log"],
    });
    expect(targets).toEqual([
      { path: "/srv/qa/data", owner: "/srv/qa/data" },
      { path: "/srv/qa/logs", owner: "/srv/qa/logs" },
      { path: "/srv/qa/electron/logs", owner: "/srv/qa/electron/logs" },
      { path: "/srv/qa/electron/last-seen-version", owner: "/srv/qa/electron" },
      { path: "/srv/qa/electron/update-notes.json", owner: "/srv/qa/electron" },
      { path: "/srv/qa/logs/erudi-backend.log", owner: "/srv/qa/logs" },
    ]);
  });
});

describe("isUnsafeInstanceRoot", () => {
  const check = (root, platform = "linux") =>
    isUnsafeInstanceRoot(root, { platform, home: homeFor(platform), realpath: noRealpath });

  it("refuses the home folder, a folder above it, and a file system or drive root", () => {
    expect(check("/home/me")).toBe(true);
    expect(check("/home")).toBe(true);
    expect(check("/")).toBe(true);
    expect(check("C:\\", "win32")).toBe(true);
    expect(check("C:\\Users", "win32")).toBe(true);
  });

  it("refuses the home folder spelled with another case where the file system ignores it", () => {
    expect(check("/HOME/ME", "darwin")).toBe(true);
    expect(check("c:\\users\\ME", "win32")).toBe(true);
  });

  it("accepts a folder inside the home folder, or elsewhere", () => {
    expect(check("/home/me/qa")).toBe(false);
    expect(check("/srv/qa")).toBe(false);
  });
});

describe("isSafeToDelete", () => {
  const guard = {
    platform: "linux",
    home: "/home/me",
    protectedDirs: ["/srv/qa", "/srv/qa/electron"],
    realpath: noRealpath,
  };

  it("accepts a folder Erudi owns", () => {
    expect(isSafeToDelete("/srv/qa/data", guard)).toBe(true);
    expect(isSafeToDelete("/home/me/.local/share/erudi/backend", guard)).toBe(true);
  });

  it("refuses the file system root, the home folder and anything above it", () => {
    for (const target of ["/", "/home", "/home/me", "/home/me/"]) {
      expect(isSafeToDelete(target, guard)).toBe(false);
    }
  });

  it("refuses a root it does not own and anything above it", () => {
    for (const target of ["/srv/qa", "/srv/qa/electron", "/srv"]) {
      expect(isSafeToDelete(target, guard)).toBe(false);
    }
  });

  it("refuses a relative or empty path", () => {
    expect(isSafeToDelete("data", guard)).toBe(false);
    expect(isSafeToDelete("", guard)).toBe(false);
  });

  it("refuses a Windows drive root and the home folder in another case", () => {
    const win = {
      platform: "win32",
      home: "C:\\Users\\me",
      protectedDirs: [],
      realpath: noRealpath,
    };
    expect(isSafeToDelete("C:\\", win)).toBe(false);
    expect(isSafeToDelete("c:\\USERS\\me", win)).toBe(false);
    expect(isSafeToDelete("C:\\Users\\me\\AppData\\Local\\erudi\\backend", win)).toBe(true);
  });

  describe("on the real file system", () => {
    let dir;
    beforeEach(() => {
      dir = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "erudi-guard-")));
    });
    afterEach(() => {
      fs.rmSync(dir, { recursive: true, force: true });
    });

    it("sees through a symlink that leads to the home folder", () => {
      const home = path.join(dir, "home");
      fs.mkdirSync(home);
      const alias = path.join(dir, "alias");
      fs.symlinkSync(home, alias);
      const realGuard = { platform: process.platform, home, protectedDirs: [] };
      expect(isSafeToDelete(alias, realGuard)).toBe(false);
      expect(isSafeToDelete(path.join(home, "erudi"), realGuard)).toBe(true);
    });
  });
});
