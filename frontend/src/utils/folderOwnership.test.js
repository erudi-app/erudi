import { describe, it, expect, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { claimFolder, isOwnedFolder } from "./folderOwnership.js";
import { OWNERSHIP_SENTINEL } from "./dataLocations.js";

let dir;
beforeEach(() => {
  dir = fs.mkdtempSync(path.join(os.tmpdir(), "erudi-own-"));
});
afterEach(() => {
  fs.rmSync(dir, { recursive: true, force: true });
});

describe("claimFolder", () => {
  it("creates a missing folder and marks it", () => {
    const target = path.join(dir, "electron", "logs");
    expect(claimFolder(target, { namespaced: false })).toBe(true);
    expect(fs.existsSync(path.join(target, OWNERSHIP_SENTINEL))).toBe(true);
    expect(isOwnedFolder(target)).toBe(true);
  });

  it("never marks an existing folder under a root the user chose", () => {
    const target = path.join(dir, "electron");
    fs.mkdirSync(target);
    fs.writeFileSync(path.join(target, "mine.txt"), "x");
    expect(claimFolder(target, { namespaced: false })).toBe(false);
    expect(isOwnedFolder(target)).toBe(false);
  });

  it("marks an existing folder whose path Erudi fixes (an install from before the file)", () => {
    const target = path.join(dir, "Logs", "Erudi");
    fs.mkdirSync(target, { recursive: true });
    expect(claimFolder(target, { namespaced: true })).toBe(true);
    expect(isOwnedFolder(target)).toBe(true);
  });

  it("keeps a folder marked on an earlier start", () => {
    const target = path.join(dir, "electron");
    claimFolder(target, { namespaced: false });
    expect(claimFolder(target, { namespaced: false })).toBe(true);
  });
});

describe("isOwnedFolder", () => {
  it("is false for a missing folder", () => {
    expect(isOwnedFolder(path.join(dir, "nothing"))).toBe(false);
  });
});
