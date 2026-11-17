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

// Every test deletes real files in a temporary folder with the real fs.rm. A
// folder the test makes read-only stands for a file Windows still holds: its
// contents cannot be deleted until the permission comes back. Read-only
// folders do not stop root, nor anyone on Windows.
const canHoldFiles =
  process.platform !== "win32" && !(typeof process.getuid === "function" && process.getuid() === 0);

const fsError = (code) => Object.assign(new Error(code), { code });

let root;
beforeEach(() => {
  root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "erudi-clear-")));
});
afterEach(() => {
  // Give back whatever a test made read-only, then clean up.
  for (const dir of held) fs.chmodSync(dir, 0o755);
  held.length = 0;
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
const held = [];
const hold = (relative) => {
  fs.chmodSync(at(relative), 0o555);
  held.push(at(relative));
};
const release = (relative) => {
  fs.chmodSync(at(relative), 0o755);
  held.splice(held.indexOf(at(relative)), 1);
};
const allowAll = () => true;
const isOwned = (dir) => fs.existsSync(path.join(dir, OWNERSHIP_SENTINEL));
const notForeign = () => false;
const options = { isSafe: allowAll, isOwned, isForeign: notForeign };

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
      options
    );
    expect(result).toMatchObject({ pending: [], refused: [], skipped: [] });
    expect(exists("data")).toBe(false);
    expect(exists("erudi-backend.log")).toBe(false);
  });

  it("deletes a folder whose ownership file sits deeper (backend/prod/data)", async () => {
    seed("backend/prod/data/postgres/PG_VERSION");
    own("backend/prod/data");
    const result = await removeTargets(
      [{ path: at("backend"), owner: at("backend/prod/data") }],
      options
    );
    expect(result.pending).toEqual([]);
    expect(exists("backend")).toBe(false);
  });

  it("skips a folder without Erudi's ownership file (ERUDI_DATA_ROOT at the user's own folder)", async () => {
    seed("data/thesis.docx");
    const result = await removeTargets([{ path: at("data"), owner: at("data") }], options);
    expect(result.skipped).toEqual([at("data")]);
    expect(exists("data/thesis.docx")).toBe(true);
  });

  it.skipIf(!canHoldFiles)(
    "keeps the ownership file while something in the folder is still held",
    async () => {
      seed("data/postgres/base/1");
      seed("data/models/weights.bin");
      own("data");
      hold("data/postgres/base");
      const result = await removeTargets([{ path: at("data"), owner: at("data") }], options);
      expect(result.pending).toEqual([{ path: at("data"), code: "EACCES" }]);
      expect(exists("data/models")).toBe(false);
      expect(exists(`data/${OWNERSHIP_SENTINEL}`)).toBe(true);
    }
  );

  it("passes fs.rm the retry options for a file another process holds (Windows EBUSY)", async () => {
    seed("erudi-backend.log");
    const rm = vi.fn((target, opts) => fs.promises.rm(target, opts));
    await removeTargets([{ path: at("erudi-backend.log"), owner: null }], { ...options, rm });
    expect(rm.mock.calls[0][1]).toMatchObject({ recursive: true, force: true });
    expect(rm.mock.calls[0][1].maxRetries).toBeGreaterThan(0);
  });

  it("goes on when one target is still held, and reports it as pending", async () => {
    for (const name of ["a.log", "b.log", "c.log"]) seed(name);
    const rm = vi.fn((target, opts) =>
      target.endsWith("b.log") ? Promise.reject(fsError("EBUSY")) : fs.promises.rm(target, opts)
    );
    const result = await removeTargets(
      ["a.log", "b.log", "c.log"].map((name) => ({ path: at(name), owner: null })),
      { ...options, rm }
    );
    expect(result.removed).toEqual([at("a.log"), at("c.log")]);
    expect(result.pending).toEqual([{ path: at("b.log"), code: "EBUSY" }]);
    expect(exists("b.log")).toBe(true);
  });

  it("refuses, rather than retries, a file another user owns that it may not delete", async () => {
    seed("erudi-backend.log");
    const result = await removeTargets([{ path: at("erudi-backend.log"), owner: null }], {
      ...options,
      isForeign: () => true,
      rm: () => Promise.reject(fsError("EPERM")),
    });
    expect(result.pending).toEqual([]);
    expect(result.refused).toEqual([{ path: at("erudi-backend.log"), why: "foreign" }]);
  });

  it("never deletes a target the guard refuses", async () => {
    seed("qa/data/x");
    own("qa/data");
    const result = await removeTargets(
      [
        { path: at("qa"), owner: null },
        { path: at("qa/data"), owner: at("qa/data") },
      ],
      { ...options, isSafe: (target) => target !== at("qa") }
    );
    expect(result.refused).toEqual([{ path: at("qa"), why: "unsafe" }]);
    expect(exists("qa")).toBe(true);
    expect(exists("qa/data")).toBe(false);
  });
});

// The flow behind Help > Clear All Data and the start that finishes it, with
// fakes for everything Electron does and the real fs.rm.
function makeFlow(overrides = {}) {
  const order = [];
  const record = (name) =>
    vi.fn(async () => {
      order.push(name);
    });
  const realRm = overrides.rm ?? ((target, opts) => fs.promises.rm(target, opts));
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
    showGaveUp: vi.fn(async () => order.push("showGaveUp")),
    showError: vi.fn(async () => order.push("showError")),
    quit: vi.fn(() => order.push("quit")),
    log: vi.fn(),
    logWarn: vi.fn(),
    logError: vi.fn(),
    ...overrides,
  };
  deps.rm = vi.fn(async (target, opts) => {
    if (!order.includes(`rm ${path.basename(target)}`)) order.push(`rm ${path.basename(target)}`);
    return realRm(target, opts);
  });
  return { flow: createClearAllDataFlow(deps), deps, order };
}

const readMarker = (deps) => JSON.parse(fs.readFileSync(deps.markerPath, "utf8"));
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
    expect(order.filter((step) => !step.startsWith("rm "))).toEqual([
      "confirm",
      "markQuitting",
      "stopBackend",
      "quietRenderer",
      "clearStorage",
      "silenceAppLog",
      "showResult",
      "quit",
    ]);
    expect(order.indexOf("silenceAppLog")).toBeLessThan(
      order.findIndex((s) => s.startsWith("rm "))
    );
    expect(order.findLastIndex((s) => s.startsWith("rm "))).toBeLessThan(
      order.indexOf("showResult")
    );
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

  it("does nothing where it is not available (a run from source)", async () => {
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

  it.skipIf(!canHoldFiles)(
    "records exactly the targets still held, with its context, for the next start",
    async () => {
      seed("logs/held/backend.log.1");
      hold("logs/held");
      const { flow, deps } = makeFlow();
      await flow.clearFromMenu();
      expect(readMarker(deps)).toEqual({
        version: 1,
        context: { isPackaged: true, instanceRoot: null },
        attempts: 0,
        storage: false,
        targets: [at("logs")],
      });
      expect(exists("data")).toBe(false);
      expect(deps.showResult.mock.calls[0][0].complete).toBe(false);
    }
  );

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
    let silenced = false;
    const written = [];
    const { flow } = makeFlow({
      silenceAppLog: vi.fn(() => {
        silenced = true;
      }),
      log: vi.fn((line) => written.push({ line, silenced })),
      logWarn: vi.fn((line) => written.push({ line, silenced })),
    });
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

  it.skipIf(!canHoldFiles)(
    "deletes at the next start what the clear could not, before the backend marks the folder again",
    async () => {
      seed("logs/held/backend.log.1");
      hold("logs/held");
      const first = makeFlow();
      await first.flow.clearFromMenu();
      expect(exists("logs/held/backend.log.1")).toBe(true);

      // Whatever the first pass managed to delete -- including, at worst, the
      // ownership file -- the marker's entry is still deleted at the next start.
      fs.rmSync(at("logs", OWNERSHIP_SENTINEL), { force: true });
      release("logs/held");
      const next = makeFlow();
      await next.flow.finishPending();

      expect(exists("logs")).toBe(false);
      expect(fs.existsSync(next.deps.markerPath)).toBe(false);
    }
  );

  it("deletes only the targets the marker names, then removes the marker", async () => {
    const { flow, deps, order } = makeFlow();
    writeMarker(deps, marker({ targets: [at("logs")] }));
    await flow.finishPending();
    expect(exists("logs")).toBe(false);
    expect(exists("data/postgres/PG_VERSION")).toBe(true);
    expect(order).not.toContain("clearStorage");
    expect(fs.existsSync(deps.markerPath)).toBe(false);
  });

  it("ignores, but keeps, a marker another kind of run left (dev and installed app share it)", async () => {
    const { flow, deps } = makeFlow();
    const theirs = marker({
      context: { isPackaged: false, instanceRoot: null },
      targets: [at("data")],
    });
    writeMarker(deps, theirs);
    await flow.finishPending();
    expect(exists("data/postgres/PG_VERSION")).toBe(true);
    expect(readMarker(deps)).toEqual(theirs);
    expect(deps.logWarn).toHaveBeenCalled();
  });

  it("ignores, but keeps, a marker from another ERUDI_DATA_ROOT", async () => {
    const { flow, deps } = makeFlow({ context: { isPackaged: true, instanceRoot: "/srv/a" } });
    writeMarker(
      deps,
      marker({ context: { isPackaged: true, instanceRoot: "/srv/b" }, targets: [at("data")] })
    );
    await flow.finishPending();
    expect(exists("data/postgres/PG_VERSION")).toBe(true);
    expect(fs.existsSync(deps.markerPath)).toBe(true);
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
    const { flow, deps } = makeFlow({ rm: () => Promise.reject(fsError("EBUSY")) });
    writeMarker(deps, marker({ targets: [at("logs")] }));
    await flow.finishPending();
    expect(readMarker(deps)).toMatchObject({ attempts: 1, targets: [at("logs")] });
    expect(deps.showGaveUp).not.toHaveBeenCalled();
  });

  it(`gives up after ${MAX_PENDING_ATTEMPTS} starts: tells the user once what is left, and the log`, async () => {
    const { flow, deps } = makeFlow({ rm: () => Promise.reject(fsError("EBUSY")) });
    writeMarker(deps, marker({ attempts: MAX_PENDING_ATTEMPTS - 1, targets: [at("logs")] }));
    await flow.finishPending();
    expect(fs.existsSync(deps.markerPath)).toBe(false);
    expect(deps.logWarn).toHaveBeenCalledWith(expect.stringContaining("giving up"));
    expect(deps.showGaveUp).toHaveBeenCalledTimes(1);
    expect(deps.showGaveUp).toHaveBeenCalledWith([at("logs")]);
    expect(exists("logs/backend.log")).toBe(true);
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
