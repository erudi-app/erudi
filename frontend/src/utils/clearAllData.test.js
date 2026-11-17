import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { clearAllData, removeTargets, PENDING_CLEAR_FILE } from "./clearAllData.js";

const busy = () => Object.assign(new Error("resource busy or locked"), { code: "EBUSY" });

let root;
beforeEach(() => {
  root = fs.mkdtempSync(path.join(os.tmpdir(), "erudi-clear-"));
});
afterEach(() => {
  fs.rmSync(root, { recursive: true, force: true });
});

function seed(relative, content = "x") {
  const file = path.join(root, relative);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, content);
  return file;
}

const allowAll = () => true;

describe("removeTargets", () => {
  it("deletes folders and files, and a target already gone is not a failure", async () => {
    seed("data/postgres/base/1");
    seed("logs/backend.log");
    const result = await removeTargets(
      [path.join(root, "data"), path.join(root, "logs"), path.join(root, "never-created")],
      { isSafe: allowAll }
    );
    expect(result.pending).toEqual([]);
    expect(result.refused).toEqual([]);
    expect(fs.existsSync(path.join(root, "data"))).toBe(false);
    expect(fs.existsSync(path.join(root, "logs"))).toBe(false);
  });

  it("asks fs.rm to retry a file another process still holds (Windows EBUSY)", async () => {
    const rm = vi.fn(async () => {});
    await removeTargets(["/x/data"], { isSafe: allowAll, rm });
    expect(rm).toHaveBeenCalledWith("/x/data", {
      recursive: true,
      force: true,
      maxRetries: expect.any(Number),
      retryDelay: expect.any(Number),
    });
    expect(rm.mock.calls[0][1].maxRetries).toBeGreaterThan(0);
  });

  it("goes on with the other targets when one is still held, and reports it", async () => {
    const rm = vi.fn(async (target) => {
      if (target === "/x/logs") throw busy();
    });
    const result = await removeTargets(["/x/data", "/x/logs", "/x/more"], {
      isSafe: allowAll,
      rm,
    });
    expect(rm).toHaveBeenCalledTimes(3);
    expect(result.removed).toEqual(["/x/data", "/x/more"]);
    expect(result.pending).toEqual([{ target: "/x/logs", code: "EBUSY" }]);
  });

  it("never deletes a target the guard refuses", async () => {
    const rm = vi.fn(async () => {});
    const result = await removeTargets(["/srv/qa", "/srv/qa/data"], {
      isSafe: (target) => target !== "/srv/qa",
      rm,
    });
    expect(rm).toHaveBeenCalledTimes(1);
    expect(rm).toHaveBeenCalledWith("/srv/qa/data", expect.any(Object));
    expect(result.refused).toEqual(["/srv/qa"]);
  });
});

describe("clearAllData", () => {
  function deps(overrides = {}) {
    const order = [];
    return {
      order,
      stopBackend: vi.fn(async () => order.push("stopBackend")),
      quietRenderer: vi.fn(async () => order.push("quietRenderer")),
      clearStorage: vi.fn(async () => order.push("clearStorage")),
      targets: [path.join(root, "data"), path.join(root, "logs")],
      isSafe: allowAll,
      pendingMarker: path.join(root, "electron", PENDING_CLEAR_FILE),
      ...overrides,
    };
  }

  it("stops the backend and quiets the window before deleting anything", async () => {
    seed("data/postgres/base/1");
    seed("logs/backend.log");
    const d = deps({
      rm: vi.fn(async (target, opts) => {
        d.order.push(`rm ${path.basename(target)}`);
        await fs.promises.rm(target, opts);
      }),
    });

    const result = await clearAllData(d);

    expect(d.order).toEqual(["stopBackend", "quietRenderer", "clearStorage", "rm data", "rm logs"]);
    expect(result).toEqual({ complete: true, pending: [], refused: [], storageError: null });
    expect(fs.existsSync(path.join(root, "data"))).toBe(false);
    expect(fs.existsSync(d.pendingMarker)).toBe(false);
  });

  it("leaves a marker so the next start finishes what was still held", async () => {
    const d = deps({
      rm: vi.fn(async (target) => {
        if (target.endsWith("logs")) throw busy();
      }),
    });

    const result = await clearAllData(d);

    expect(result.complete).toBe(false);
    expect(result.pending).toEqual([{ target: path.join(root, "logs"), code: "EBUSY" }]);
    expect(fs.existsSync(d.pendingMarker)).toBe(true);
  });

  it("deletes the files even when the storage could not be cleared, and retries it next start", async () => {
    seed("data/a");
    const failure = new Error("storage service gone");
    const d = deps({
      clearStorage: vi.fn(async () => {
        throw failure;
      }),
    });

    const result = await clearAllData(d);

    expect(fs.existsSync(path.join(root, "data"))).toBe(false);
    expect(result.complete).toBe(false);
    expect(result.storageError).toBe(failure);
    expect(fs.existsSync(d.pendingMarker)).toBe(true);
  });

  it("removes the marker once a later run clears everything", async () => {
    const d = deps();
    seed(path.join("electron", PENDING_CLEAR_FILE), "");

    const result = await clearAllData({ ...d, stopBackend: null, quietRenderer: null });

    expect(result.complete).toBe(true);
    expect(fs.existsSync(d.pendingMarker)).toBe(false);
  });
});
