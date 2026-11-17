import { describe, it, expect, vi } from "vitest";
import { setUpElectronPaths, openFirstWindow } from "./appStartup.js";

// A fake Electron `app` that records what was set, and in which order.
function fakeApp() {
  const calls = [];
  const paths = { userData: "/default/userData" };
  return {
    calls,
    paths,
    setPath: vi.fn((name, value) => {
      calls.push(`setPath ${name}`);
      paths[name] = value;
    }),
    setAppLogsPath: vi.fn((value) => {
      calls.push("setAppLogsPath");
      paths.logs = value ?? "<default logs>";
    }),
  };
}

describe("setUpElectronPaths", () => {
  it("moves userData and the logs under the instance root before the updater loads", () => {
    const app = fakeApp();
    const seenByUpdater = {};
    const updater = setUpElectronPaths({
      app,
      electronDir: "/srv/qa/electron",
      claim: (dir) => {
        app.calls.push(`claim ${dir}`);
        return true;
      },
      loadUpdater: () => {
        app.calls.push("loadUpdater");
        Object.assign(seenByUpdater, app.paths);
        return "updater";
      },
    });
    // Claimed before Electron creates them (setAppLogsPath creates its folder),
    // so the ownership file says Erudi created them.
    expect(app.calls).toEqual([
      "claim /srv/qa/electron",
      "claim /srv/qa/electron/logs",
      "setPath userData",
      "setAppLogsPath",
      "loadUpdater",
    ]);
    expect(seenByUpdater).toEqual({ userData: "/srv/qa/electron", logs: "/srv/qa/electron/logs" });
    expect(updater).toBe("updater");
  });

  it("never claims the logs folder inside an electron folder that is not Erudi's", () => {
    const app = fakeApp();
    const claim = vi.fn(() => false);
    setUpElectronPaths({ app, electronDir: "/srv/qa/electron", claim, loadUpdater: () => null });
    expect(claim.mock.calls).toEqual([["/srv/qa/electron"]]);
  });

  it("keeps Electron's default folders without an instance root, logs path defined", () => {
    const app = fakeApp();
    setUpElectronPaths({
      app,
      electronDir: null,
      loadUpdater: () => app.calls.push("loadUpdater"),
    });
    expect(app.setPath).not.toHaveBeenCalled();
    expect(app.setAppLogsPath).toHaveBeenCalledWith(undefined);
    expect(app.calls).toEqual(["setAppLogsPath", "loadUpdater"]);
  });
});

describe("openFirstWindow", () => {
  it("finishes a pending Clear All Data before the menu and the window", async () => {
    const order = [];
    let release;
    const pending = new Promise((resolve) => {
      release = resolve;
    });
    const opening = openFirstWindow({
      finishPendingClear: async () => {
        order.push("finishPendingClear started");
        await pending;
        order.push("finishPendingClear done");
      },
      createApplicationMenu: () => order.push("createApplicationMenu"),
      createWindow: () => order.push("createWindow"),
    });
    await Promise.resolve();
    expect(order).toEqual(["finishPendingClear started"]);
    release();
    await opening;
    expect(order).toEqual([
      "finishPendingClear started",
      "finishPendingClear done",
      "createApplicationMenu",
      "createWindow",
    ]);
  });

  it("opens the window even when finishing the clear fails", async () => {
    const createWindow = vi.fn();
    const logError = vi.fn();
    await openFirstWindow({
      finishPendingClear: async () => {
        throw new Error("disk gone");
      },
      createApplicationMenu: () => {},
      createWindow,
      logError,
    });
    expect(createWindow).toHaveBeenCalled();
    expect(logError).toHaveBeenCalled();
  });
});
