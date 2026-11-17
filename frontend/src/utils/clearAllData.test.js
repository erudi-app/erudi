import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  createClearAllDataFlow,
  removeTargets,
  MAX_PENDING_ATTEMPTS,
  PENDING_CLEAR_FILE,
} from "./clearAllData.js";
import { OWNERSHIP_SENTINEL } from "./dataLocations.js";

const fsError = (code) => Object.assign(new Error(code), { code });

let root;
beforeEach(() => {
  root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "erudi-clear-")));
});
afterEach(() => {
  fs.rmSync(root, { recursive: true, force: true });
});

const at = (...parts) => path.join(root, ...parts);
function seed(relative, content = "x") {
  const file = at(relative);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, content);
  return file;
}
const own = (relative) => seed(path.join(relative, OWNERSHIP_SENTINEL), "");
const exists = (relative) => fs.existsSync(at(relative));
const allowAll = () => true;
const isOwned = (dir) => fs.existsSync(path.join(dir, OWNERSHIP_SENTINEL));
const notForeign = () => false;

describe("removeTargets", () => {
  it("deletes an owned folder and a file, and a target already gone is not a failure", async () => {
    seed("data/postgres/PG_VERSION");
    own("data");
    seed("erudi-backend.log");
    const result = await removeTargets(
      [
        { path: at("data"), owner: at("data") },
        { path: at("erudi-backend.log"), owner: null },
        { path: at("never-created"), owner: null },
      ],
      { isSafe: allowAll, isOwned, isForeign: notForeign }
    );
    expect(result).toMatchObject({ pending: [], refused: [], skipped: [] });
    expect(exists("data")).toBe(false);
    expect(exists("erudi-backend.log")).toBe(false);
  });

  it("skips a folder without Erudi's ownership file (ERUDI_DATA_ROOT at the user's own folder)", async () => {
    seed("data/thesis.docx");
    const result = await removeTargets([{ path: at("data"), owner: at("data") }], {
      isSafe: allowAll,
      isOwned,
      isForeign: notForeign,
    });
    expect(result.skipped).toEqual([at("data")]);
    expect(exists("data/thesis.docx")).toBe(true);
  });

  it("asks fs.rm to retry a file another process still holds (Windows EBUSY)", async () => {
    const rm = vi.fn(async () => {});
    await removeTargets([{ path: "/x/data", owner: null }], {
      isSafe: allowAll,
      isOwned,
      isForeign: notForeign,
      rm,
    });
    expect(rm.mock.calls[0][1]).toMatchObject({ recursive: true, force: true });
    expect(rm.mock.calls[0][1].maxRetries).toBeGreaterThan(0);
  });

  it("goes on when one target is still held, and reports it as pending", async () => {
    const rm = vi.fn(async (target) => {
      if (target === "/x/logs") throw fsError("EBUSY");
    });
    const result = await removeTargets(
      ["/x/data", "/x/logs", "/x/more"].map((p) => ({ path: p, owner: null })),
      { isSafe: allowAll, isOwned, isForeign: notForeign, rm }
    );
    expect(rm).toHaveBeenCalledTimes(3);
    expect(result.removed).toEqual(["/x/data", "/x/more"]);
    expect(result.pending).toEqual([{ path: "/x/logs", code: "EBUSY" }]);
  });

  it("refuses, rather than retries, a file another user owns that it may not delete", async () => {
    const rm = vi.fn(async () => {
      throw fsError("EPERM");
    });
    const result = await removeTargets([{ path: "/tmp/erudi-backend.log", owner: null }], {
      isSafe: allowAll,
      isOwned,
      isForeign: (p) => p === "/tmp/erudi-backend.log",
      rm,
    });
    expect(result.pending).toEqual([]);
    expect(result.refused).toEqual([{ path: "/tmp/erudi-backend.log", why: "foreign" }]);
  });

  it("never deletes a target the guard refuses", async () => {
    const rm = vi.fn(async () => {});
    const result = await removeTargets(
      [
        { path: "/srv/qa", owner: null },
        { path: "/srv/qa/data", owner: null },
      ],
      { isSafe: (p) => p !== "/srv/qa", isOwned, isForeign: notForeign, rm }
    );
    expect(rm).toHaveBeenCalledTimes(1);
    expect(result.refused).toEqual([{ path: "/srv/qa", why: "unsafe" }]);
  });
});

// The flow behind Help > Clear All Data and the start that finishes it, with
// fakes for everything Electron does.
function makeFlow(overrides = {}) {
  const order = [];
  const record = (name) =>
    vi.fn(async () => {
      order.push(name);
    });
  const deps = {
    context: { isPackaged: true, instanceRoot: null },
    available: true,
    rootRefused: false,
    confirm: vi.fn(async () => {
      order.push("confirm");
      return true;
    }),
    markQuitting: vi.fn(() => order.push("markQuitting")),
    stopBackend: record("stopBackend"),
    quietRenderer: record("quietRenderer"),
    clearStorage: record("clearStorage"),
    silenceAppLog: vi.fn(() => order.push("silenceAppLog")),
    targets: () => [
      { path: at("data"), owner: at("data") },
      { path: at("logs"), owner: at("logs") },
    ],
    isSafe: allowAll,
    isOwned,
    isForeign: notForeign,
    markerPath: at("userData", PENDING_CLEAR_FILE),
    showResult: vi.fn(async () => order.push("showResult")),
    showRootRefused: vi.fn(async () => order.push("showRootRefused")),
    showError: vi.fn(async () => order.push("showError")),
    quit: vi.fn(() => order.push("quit")),
    log: vi.fn(),
    logWarn: vi.fn(),
    logError: vi.fn(),
    ...overrides,
  };
  const rm = deps.rm;
  if (rm) {
    deps.rm = vi.fn(async (target, opts) => {
      order.push(`rm ${path.basename(target)}`);
      return rm(target, opts);
    });
  } else {
    deps.rm = vi.fn(async (target, opts) => {
      order.push(`rm ${path.basename(target)}`);
      return fs.promises.rm(target, opts);
    });
  }
  return { flow: createClearAllDataFlow(deps), deps, order };
}

const readMarker = (deps) => JSON.parse(fs.readFileSync(deps.markerPath, "utf8"));

describe("clearFromMenu", () => {
  beforeEach(() => {
    seed("data/postgres/PG_VERSION");
    own("data");
    seed("logs/backend.log");
    own("logs");
  });

  it("confirms, marks the app quitting, stops the backend, blanks the window, clears the storage, silences the log, deletes, reports, quits", async () => {
    const { flow, order, deps } = makeFlow();
    await flow.clearFromMenu();
    expect(order).toEqual([
      "confirm",
      "markQuitting",
      "stopBackend",
      "quietRenderer",
      "clearStorage",
      "silenceAppLog",
      "rm data",
      "rm logs",
      "showResult",
      "quit",
    ]);
    expect(exists("data")).toBe(false);
    expect(exists("logs")).toBe(false);
    expect(fs.existsSync(deps.markerPath)).toBe(false);
    expect(deps.showResult.mock.calls[0][0]).toMatchObject({ complete: true, leftInPlace: [] });
  });

  it("does nothing when the user cancels", async () => {
    const { flow, order } = makeFlow({ confirm: vi.fn(async () => false) });
    await flow.clearFromMenu();
    expect(order).toEqual([]);
    expect(exists("data")).toBe(true);
  });

  it("does nothing where it is not available (a dev run without ERUDI_DATA_ROOT)", async () => {
    const { flow, deps } = makeFlow({ available: false });
    await flow.clearFromMenu();
    expect(deps.confirm).not.toHaveBeenCalled();
    expect(exists("data")).toBe(true);
  });

  it("deletes nothing when ERUDI_DATA_ROOT is the home folder or above, and says why", async () => {
    const { flow, order } = makeFlow({ rootRefused: true });
    await flow.clearFromMenu();
    expect(order).toEqual(["showRootRefused"]);
    expect(exists("data")).toBe(true);
  });

  it("names a folder left in place because it is not Erudi's", async () => {
    fs.rmSync(at("logs", OWNERSHIP_SENTINEL));
    const { flow, deps } = makeFlow();
    await flow.clearFromMenu();
    expect(exists("logs/backend.log")).toBe(true);
    expect(deps.showResult.mock.calls[0][0].leftInPlace).toEqual([at("logs")]);
  });

  it("records exactly the targets still held, with its context, for the next start", async () => {
    const { flow, deps } = makeFlow({
      rm: async (target, opts) => {
        if (target.endsWith("logs")) throw fsError("EBUSY");
        return fs.promises.rm(target, opts);
      },
    });
    await flow.clearFromMenu();
    expect(readMarker(deps)).toEqual({
      version: 1,
      context: { isPackaged: true, instanceRoot: null },
      attempts: 0,
      storage: false,
      targets: [at("logs")],
    });
    expect(deps.showResult.mock.calls[0][0].complete).toBe(false);
  });

  it("still deletes the files when the storage cannot be cleared, and records the storage", async () => {
    const { flow, deps } = makeFlow({
      clearStorage: vi.fn(async () => {
        throw new Error("storage service gone");
      }),
    });
    await flow.clearFromMenu();
    expect(exists("data")).toBe(false);
    expect(readMarker(deps)).toMatchObject({ storage: true, targets: [] });
  });

  it("writes nothing to the log once the deletion started", async () => {
    const { flow, deps } = makeFlow();
    deps.log.mockClear();
    let silenced = false;
    deps.silenceAppLog.mockImplementation(() => {
      silenced = true;
    });
    const written = [];
    deps.log.mockImplementation((line) => written.push({ line, silenced }));
    deps.logWarn.mockImplementation((line) => written.push({ line, silenced }));
    await flow.clearFromMenu();
    expect(written.filter((w) => w.silenced)).toEqual([]);
  });

  it("quits even when a step throws, after saying so", async () => {
    const { flow, order } = makeFlow({
      stopBackend: vi.fn(async () => {
        throw new Error("boom");
      }),
    });
    await flow.clearFromMenu();
    expect(order.slice(-2)).toEqual(["showError", "quit"]);
    expect(exists("data")).toBe(true);
  });
});

describe("finishPending (next start)", () => {
  const writeMarker = (deps, marker) => {
    fs.mkdirSync(path.dirname(deps.markerPath), { recursive: true });
    fs.writeFileSync(deps.markerPath, JSON.stringify(marker));
  };
  const marker = (extra = {}) => ({
    version: 1,
    context: { isPackaged: true, instanceRoot: null },
    attempts: 0,
    storage: false,
    targets: [],
    ...extra,
  });

  beforeEach(() => {
    seed("data/postgres/PG_VERSION");
    own("data");
    seed("logs/backend.log");
    own("logs");
  });

  it("does nothing without a marker", async () => {
    const { flow, order } = makeFlow();
    await flow.finishPending();
    expect(order).toEqual([]);
    expect(exists("data")).toBe(true);
  });

  it("deletes only the targets the marker names, then removes the marker", async () => {
    const { flow, deps, order } = makeFlow();
    writeMarker(deps, marker({ targets: [at("logs")] }));
    await flow.finishPending();
    expect(exists("logs")).toBe(false);
    expect(exists("data/postgres/PG_VERSION")).toBe(true);
    expect(order).not.toContain("clearStorage");
    expect(fs.existsSync(deps.markerPath)).toBe(false);
  });

  it("ignores and removes a marker another kind of run left (dev and installed app share it)", async () => {
    const { flow, deps } = makeFlow();
    writeMarker(
      deps,
      marker({ context: { isPackaged: false, instanceRoot: null }, targets: [at("data")] })
    );
    await flow.finishPending();
    expect(exists("data/postgres/PG_VERSION")).toBe(true);
    expect(fs.existsSync(deps.markerPath)).toBe(false);
    expect(deps.logWarn).toHaveBeenCalled();
  });

  it("ignores a marker from another ERUDI_DATA_ROOT", async () => {
    const { flow, deps } = makeFlow({ context: { isPackaged: true, instanceRoot: "/srv/a" } });
    writeMarker(
      deps,
      marker({ context: { isPackaged: true, instanceRoot: "/srv/b" }, targets: [at("data")] })
    );
    await flow.finishPending();
    expect(exists("data/postgres/PG_VERSION")).toBe(true);
  });

  it("never deletes a path the marker names that is not one of this run's targets", async () => {
    seed("elsewhere/precious.txt");
    const { flow, deps } = makeFlow();
    writeMarker(deps, marker({ targets: [at("elsewhere"), at("logs")] }));
    await flow.finishPending();
    expect(exists("elsewhere/precious.txt")).toBe(true);
    expect(exists("logs")).toBe(false);
  });

  it("clears the storage again only when the marker says it was not cleared", async () => {
    const { flow, deps, order } = makeFlow();
    writeMarker(deps, marker({ storage: true }));
    await flow.finishPending();
    expect(order).toContain("clearStorage");
    expect(fs.existsSync(deps.markerPath)).toBe(false);
  });

  it("keeps the marker, counting the attempt, while a target is still held", async () => {
    const { flow, deps } = makeFlow({
      rm: async () => {
        throw fsError("EBUSY");
      },
    });
    writeMarker(deps, marker({ targets: [at("logs")] }));
    await flow.finishPending();
    expect(readMarker(deps)).toMatchObject({ attempts: 1, targets: [at("logs")] });
  });

  it(`gives up after ${MAX_PENDING_ATTEMPTS} starts with a warning, so no start re-wipes new data forever`, async () => {
    const { flow, deps } = makeFlow({
      rm: async () => {
        throw fsError("EBUSY");
      },
    });
    writeMarker(deps, marker({ attempts: MAX_PENDING_ATTEMPTS - 1, targets: [at("logs")] }));
    await flow.finishPending();
    expect(fs.existsSync(deps.markerPath)).toBe(false);
    expect(deps.logWarn).toHaveBeenCalledWith(expect.stringContaining("giving up"));
  });

  it("drops a marker it cannot read", async () => {
    const { flow, deps } = makeFlow();
    fs.mkdirSync(path.dirname(deps.markerPath), { recursive: true });
    fs.writeFileSync(deps.markerPath, "not json");
    await flow.finishPending();
    expect(fs.existsSync(deps.markerPath)).toBe(false);
    expect(exists("data/postgres/PG_VERSION")).toBe(true);
  });

  it("drops the marker without deleting anything when the instance root is refused", async () => {
    const { flow, deps } = makeFlow({ rootRefused: true });
    writeMarker(deps, marker({ targets: [at("data")] }));
    await flow.finishPending();
    expect(exists("data/postgres/PG_VERSION")).toBe(true);
    expect(fs.existsSync(deps.markerPath)).toBe(false);
  });
});
