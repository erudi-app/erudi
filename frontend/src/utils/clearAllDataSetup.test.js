import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  isClearAllDataAvailable,
  isInstanceRootRefused,
  createClearTargetGuard,
  isForeignFile,
  claimInstalledLogsFolder,
} from "./clearAllDataSetup.js";
import { OWNERSHIP_SENTINEL } from "./dataLocations.js";

let dir;
beforeEach(() => {
  dir = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "erudi-setup-")));
});
afterEach(() => {
  fs.rmSync(dir, { recursive: true, force: true });
});

describe("isClearAllDataAvailable", () => {
  it("is offered by the installed app only", () => {
    expect(isClearAllDataAvailable({ isPackaged: true })).toBe(true);
  });

  it("is never offered by a run from source, with or without ERUDI_DATA_ROOT", () => {
    // Such a run spawns no backend, so it could not stop the one using the
    // data it would delete.
    expect(isClearAllDataAvailable({ isPackaged: false })).toBe(false);
  });
});

describe("isInstanceRootRefused", () => {
  it("is false without an instance root", () => {
    expect(
      isInstanceRootRefused({ instanceRoot: null, platform: process.platform, home: dir })
    ).toBe(false);
  });

  it("refuses the home folder, a folder above it, and a symlink to the home folder", () => {
    const home = path.join(dir, "home");
    fs.mkdirSync(home);
    fs.symlinkSync(home, path.join(dir, "alias"));
    const refused = (instanceRoot) =>
      isInstanceRootRefused({ instanceRoot, platform: process.platform, home });
    expect(refused(home)).toBe(true);
    expect(refused(dir)).toBe(true);
    expect(refused(path.join(dir, "alias"))).toBe(true);
    expect(refused(path.parse(dir).root)).toBe(true);
    expect(refused(path.join(home, "qa"))).toBe(false);
  });
});

describe("createClearTargetGuard", () => {
  it("protects the home folder, the instance root and the user data folder", () => {
    const home = path.join(dir, "home");
    const instanceRoot = path.join(dir, "qa");
    const userData = path.join(instanceRoot, "electron");
    fs.mkdirSync(userData, { recursive: true });
    const isSafe = createClearTargetGuard({
      platform: process.platform,
      home,
      instanceRoot,
      userData,
    });
    expect(isSafe(path.join(instanceRoot, "data"))).toBe(true);
    expect(isSafe(path.join(userData, "logs"))).toBe(true);
    expect(isSafe(instanceRoot)).toBe(false);
    expect(isSafe(userData)).toBe(false);
    expect(isSafe(home)).toBe(false);
    expect(isSafe(dir)).toBe(false);
  });
});

describe("isForeignFile", () => {
  it("is true for a file another user owns", () => {
    const file = path.join(dir, "erudi-backend.log");
    fs.writeFileSync(file, "x");
    expect(isForeignFile(file, { getuid: () => fs.statSync(file).uid + 1 })).toBe(true);
  });

  it("is false for one's own file", () => {
    const file = path.join(dir, "erudi-backend.log");
    fs.writeFileSync(file, "x");
    expect(isForeignFile(file, { getuid: () => fs.statSync(file).uid })).toBe(false);
  });

  it("is false where the system has no user ids (Windows), and for a missing file", () => {
    expect(isForeignFile(path.join(dir, "x"), { getuid: undefined })).toBe(false);
    expect(isForeignFile(path.join(dir, "gone"), { getuid: () => 1 })).toBe(false);
  });
});

describe("claimInstalledLogsFolder", () => {
  it("marks the installed app's Electron logs folder, existing or not", () => {
    const logsDir = path.join(dir, "Logs", "Erudi");
    fs.mkdirSync(logsDir, { recursive: true });
    expect(claimInstalledLogsFolder({ isPackaged: true, electronDir: null, logsDir })).toBeNull();
    expect(fs.existsSync(path.join(logsDir, OWNERSHIP_SENTINEL))).toBe(true);
  });

  it("leaves it to setUpElectronPaths under ERUDI_DATA_ROOT, and does nothing from source", () => {
    const logsDir = path.join(dir, "logs");
    claimInstalledLogsFolder({ isPackaged: true, electronDir: path.join(dir, "e"), logsDir });
    claimInstalledLogsFolder({ isPackaged: false, electronDir: null, logsDir });
    expect(fs.existsSync(logsDir)).toBe(false);
  });

  it("answers the error instead of throwing", () => {
    const claim = vi.fn(() => {
      throw new Error("read-only disk");
    });
    const error = claimInstalledLogsFolder({
      isPackaged: true,
      electronDir: null,
      logsDir: path.join(dir, "logs"),
      claim,
    });
    expect(error.message).toBe("read-only disk");
  });
});
