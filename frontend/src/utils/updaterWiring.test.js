/**
 * The updater wiring (#563, #572): what main.js does with electron-updater,
 * run against fakes that behave like the library.
 *
 * The fakes copy the parts of electron-updater 6.8 the wiring depends on:
 * - `BaseUpdater` (Windows, Linux): at the end of a download it emits
 *   `update-downloaded`, then calls `addQuitHandler()`, which registers the
 *   install at quit only while `autoInstallOnAppQuit` is on and records it in
 *   `quitHandlerAdded` (BaseUpdater.js `executeDownload`, `addQuitHandler`).
 * - `MacUpdater`: it emits `update-downloaded`, then asks Squirrel.Mac (the
 *   native updater) to fetch the update only while `autoInstallOnAppQuit` is
 *   on; Squirrel emits its own `update-downloaded` once it holds it.
 * - `checkForUpdates()` emits `update-available` and starts a download only
 *   while `autoDownload` is on, returning its cancellation token.
 * - `downloadUpdate()` joins a download already in flight and ignores the
 *   token it is given (AppUpdater `downloadUpdate`).
 */
import { EventEmitter } from "node:events";
import { describe, it, expect, vi } from "vitest";
import { NEUTRAL_STAGING_ID, configureUpdater, createUpdaterWiring } from "./updaterWiring";

class FakeToken {
  constructor() {
    this.cancelled = false;
    this.handlers = [];
  }
  cancel() {
    this.cancelled = true;
    this.handlers.forEach((handler) => handler());
  }
  onCancel(handler) {
    this.handlers.push(handler);
  }
}

class FakeUpdater extends EventEmitter {
  constructor({ mac = false, nativeUpdater = null } = {}) {
    super();
    // The library's defaults.
    this.autoDownload = true;
    this.autoInstallOnAppQuit = true;
    this.requestHeaders = null;
    this.mac = mac;
    this.nativeUpdater = nativeUpdater;
    this.quitHandlerAdded = false;
    this.available = null;
    this.pending = null;
    this.checkForUpdates = vi.fn(() => this.check());
    this.downloadUpdate = vi.fn((token = new FakeToken()) => this.download(token));
    this.quitAndInstall = vi.fn();
    if (!mac) {
      this.addQuitHandler = vi.fn(() => {
        if (this.quitHandlerAdded || !this.autoInstallOnAppQuit) return;
        this.quitHandlerAdded = true;
      });
    }
  }

  check() {
    this.emit("checking-for-update");
    if (!this.available) {
      this.emit("update-not-available", {});
      return Promise.resolve({ isUpdateAvailable: false });
    }
    this.emit("update-available", this.available);
    const cancellationToken = new FakeToken();
    return Promise.resolve({
      isUpdateAvailable: true,
      cancellationToken,
      downloadPromise: this.autoDownload ? this.download(cancellationToken) : null,
    });
  }

  download(token) {
    if (this.pending && !this.pending.settled) {
      // The library returns the download in flight; the new token is unused.
      return this.pending.promise;
    }
    let resolve;
    let reject;
    const promise = new Promise((res, rej) => {
      resolve = res;
      reject = rej;
    });
    const pending = { info: this.available, promise, settled: false };
    pending.resolve = (value) => {
      pending.settled = true;
      resolve(value);
    };
    pending.reject = (error) => {
      pending.settled = true;
      reject(error);
    };
    this.pending = pending;
    token.onCancel(() => pending.reject(new Error("cancelled")));
    return promise;
  }

  progress(percent) {
    this.emit("download-progress", { percent });
  }

  finishDownload() {
    const { info, resolve } = this.pending;
    this.emit("update-downloaded", info);
    if (this.mac) {
      if (this.autoInstallOnAppQuit) this.nativeUpdater.checkForUpdates();
    } else {
      this.addQuitHandler();
    }
    resolve([]);
  }
}

class FakeSquirrel extends EventEmitter {
  constructor() {
    super();
    this.checkForUpdates = vi.fn();
  }
  received() {
    this.emit("update-downloaded");
  }
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

function setup({
  platform = "win32",
  available = { version: "1.2.0", releaseNotes: "<p>x</p>" },
} = {}) {
  const nativeUpdater = platform === "darwin" ? new FakeSquirrel() : null;
  const updater = new FakeUpdater({ mac: platform === "darwin", nativeUpdater });
  configureUpdater(updater);
  updater.available = available;
  const send = vi.fn();
  const log = vi.fn();
  const logWarn = vi.fn();
  const onDownloaded = vi.fn();
  const timers = { setInterval: vi.fn(() => 42), clearInterval: vi.fn() };
  const wiring = createUpdaterWiring({
    updater,
    nativeUpdater,
    platform,
    send,
    log,
    logWarn,
    onDownloaded,
    timers,
  });
  const sent = (event) => send.mock.calls.filter(([name]) => name === event).map(([, p]) => p);
  return { updater, nativeUpdater, wiring, send, sent, log, logWarn, onDownloaded, timers };
}

const CHECK_ONLY = { check: true, download: false, install: false };
const WITH_DOWNLOAD = { check: true, download: true, install: false };
const WITH_INSTALL = { check: true, download: false, install: true };

describe("configureUpdater", () => {
  it("lowers both library flags, whose defaults are on", () => {
    const updater = new FakeUpdater();
    configureUpdater(updater);
    expect(updater.autoDownload).toBe(false);
    expect(updater.autoInstallOnAppQuit).toBe(false);
  });

  it("replaces the per-install identifier the library sends with a constant (P2-1)", () => {
    const updater = new FakeUpdater();
    updater.requestHeaders = { authorization: "kept" };
    configureUpdater(updater);
    expect(NEUTRAL_STAGING_ID).toBe("00000000-0000-0000-0000-000000000000");
    expect(updater.requestHeaders).toEqual({
      authorization: "kept",
      "x-user-staging-id": NEUTRAL_STAGING_ID,
    });
  });
});

describe("the switches", () => {
  it("checks nothing before the renderer reports them", () => {
    const { updater } = setup();
    expect(updater.checkForUpdates).not.toHaveBeenCalled();
  });

  it("checks nothing from an unknown snapshot (P3-1)", () => {
    const { updater, wiring, timers, log } = setup();
    wiring.setPreferences({ check: null, download: null, install: null });
    expect(updater.checkForUpdates).not.toHaveBeenCalled();
    expect(timers.setInterval).not.toHaveBeenCalled();
    expect(updater.autoDownload).toBe(false);
    expect(updater.autoInstallOnAppQuit).toBe(false);
    expect(log.mock.calls.flat().join("\n")).toMatch(/no check will run/);
  });

  it("checks at once and every 4 hours when Check is on, and finds without downloading", async () => {
    const { updater, wiring, timers, sent, log } = setup();
    wiring.setPreferences(CHECK_ONLY);
    await flush();

    expect(updater.checkForUpdates).toHaveBeenCalledTimes(1);
    expect(timers.setInterval).toHaveBeenCalledWith(expect.any(Function), 4 * 60 * 60 * 1000);
    expect(log.mock.calls.flat().join("\n")).toMatch(/checking now, then every 4 hours/);
    expect(updater.downloadUpdate).not.toHaveBeenCalled();
    expect(updater.pending).toBeNull();
    expect(sent("update-available")).toEqual([{ version: "1.2.0", downloading: false }]);
    expect(wiring.getState()).toMatchObject({ phase: "available", version: "1.2.0" });
  });

  it("stops the periodic check when Check is turned off", async () => {
    const { wiring, timers } = setup();
    wiring.setPreferences(CHECK_ONLY);
    await flush();
    wiring.setPreferences({ check: false, download: true, install: true });
    expect(timers.clearInterval).toHaveBeenCalledWith(42);
  });

  it("downloads on its own only with Download on, and says so to the banner", async () => {
    const { updater, wiring, sent } = setup();
    wiring.setPreferences(WITH_DOWNLOAD);
    await flush();
    expect(updater.pending).not.toBeNull();
    expect(sent("update-available")).toEqual([{ version: "1.2.0", downloading: true }]);
  });

  it("keeps the release notes and reports the download", async () => {
    const { updater, wiring, onDownloaded, sent } = setup();
    wiring.setPreferences(WITH_DOWNLOAD);
    await flush();
    updater.finishDownload();
    expect(onDownloaded).toHaveBeenCalledWith(expect.objectContaining({ version: "1.2.0" }));
    expect(sent("update-downloaded")).toEqual([{ version: "1.2.0", handedOff: false }]);
    expect(wiring.getState()).toMatchObject({ phase: "downloaded", version: "1.2.0" });
  });
});

describe("a downloaded update and the 4-hourly check (P2-2)", () => {
  it("stays downloaded when the check finds the same version again", async () => {
    const { updater, wiring, timers } = setup();
    wiring.setPreferences(CHECK_ONLY);
    await flush();
    const download = wiring.downloadNow();
    updater.finishDownload();
    await download;
    await flush();

    // The periodic check, with Download off, re-announces the version on disk.
    await timers.setInterval.mock.calls[0][0]();
    await flush();

    expect(wiring.getState()).toMatchObject({ phase: "downloaded", version: "1.2.0" });
  });
});

describe("a download that fails (P3-4)", () => {
  it("goes back to available", async () => {
    const { updater, wiring } = setup();
    wiring.setPreferences(WITH_DOWNLOAD);
    await flush();
    updater.progress(30);
    updater.emit("error", new Error("network"));
    expect(wiring.getState()).toMatchObject({ phase: "available", version: "1.2.0" });
  });
});

describe("turning Check or Download off during an automatic download (P3-9)", () => {
  it.each([
    ["Download", CHECK_ONLY],
    ["Check", { check: false, download: true, install: false }],
  ])("cancels it when %s is turned off", async (_name, next) => {
    const { updater, wiring, sent } = setup();
    wiring.setPreferences(WITH_DOWNLOAD);
    await flush();
    updater.progress(20);

    wiring.setPreferences(next);
    await flush();

    expect(sent("download-cancelled")).toEqual([{ version: "1.2.0" }]);
    expect(wiring.getState()).toMatchObject({ phase: "available", version: "1.2.0" });
  });

  it("still cancels it after a periodic check landed during the download (P3-A)", async () => {
    const { updater, wiring, sent, timers } = setup();
    wiring.setPreferences(WITH_DOWNLOAD);
    await flush();
    updater.progress(20);

    // The 4-hourly tick: the library joins the download in flight.
    await timers.setInterval.mock.calls[0][0]();
    await flush();
    expect(wiring.getState()).toMatchObject({ phase: "downloading", version: "1.2.0" });
    expect(sent("update-available").at(-1)).toEqual({ version: "1.2.0", downloading: true });

    wiring.setPreferences(CHECK_ONLY);
    await flush();

    expect(updater.pending.settled).toBe(true);
    expect(sent("download-cancelled")).toEqual([{ version: "1.2.0" }]);
    expect(wiring.getState()).toMatchObject({ phase: "available", version: "1.2.0" });
  });

  it("leaves a download the user started with the button alone", async () => {
    const { updater, wiring, sent } = setup();
    wiring.setPreferences(CHECK_ONLY);
    await flush();
    const pressed = wiring.downloadNow();
    updater.progress(20);

    wiring.setPreferences({ check: true, download: false, install: true });
    await flush();

    expect(sent("download-cancelled")).toEqual([]);
    updater.finishDownload();
    await expect(pressed).resolves.toEqual({ ok: true });
  });
});

describe("install turned on after a finished download: Windows and Linux", () => {
  it.each(["win32", "linux"])(
    "registers the library's quit handler, flags first (%s)",
    async (platform) => {
      const { updater, wiring, log } = setup({ platform });
      wiring.setPreferences(CHECK_ONLY);
      await flush();
      const download = wiring.downloadNow();
      updater.finishDownload();
      await download;
      expect(updater.quitHandlerAdded).toBe(false); // install was off at download

      wiring.setPreferences(WITH_INSTALL);

      // addQuitHandler returns early while autoInstallOnAppQuit is off: this
      // only holds when the flags were set before arming.
      expect(updater.quitHandlerAdded).toBe(true);
      expect(log.mock.calls.flat().join("\n")).toMatch(/v1\.2\.0 will install when the app quits/);
    }
  );

  it("does not register it again once the library did at download", async () => {
    const { updater, wiring } = setup();
    wiring.setPreferences(WITH_INSTALL);
    await flush();
    const download = wiring.downloadNow();
    updater.finishDownload();
    await download;
    expect(updater.quitHandlerAdded).toBe(true);
    updater.addQuitHandler.mockClear();

    wiring.setPreferences(WITH_INSTALL);
    expect(updater.addQuitHandler).not.toHaveBeenCalled();
  });

  it("warns instead of crashing when the library has no addQuitHandler", async () => {
    const { updater, wiring, logWarn } = setup();
    wiring.setPreferences(CHECK_ONLY);
    await flush();
    const download = wiring.downloadNow();
    updater.finishDownload();
    await download;
    delete updater.addQuitHandler;

    expect(() => wiring.setPreferences(WITH_INSTALL)).not.toThrow();
    expect(logWarn).toHaveBeenCalledTimes(1);
  });

  it("never installs from the switches: Install now only", async () => {
    const { updater, wiring } = setup();
    wiring.setPreferences(WITH_INSTALL);
    await flush();
    const download = wiring.downloadNow();
    updater.finishDownload();
    await download;
    wiring.setPreferences(CHECK_ONLY);
    expect(updater.quitAndInstall).not.toHaveBeenCalled();
    wiring.installNow();
    expect(updater.quitAndInstall).toHaveBeenCalledWith(false, true);
  });
});

describe("install and macOS (P3-5)", () => {
  it("hands the update to Squirrel when install is turned on, and records it once Squirrel has it", async () => {
    const { updater, nativeUpdater, wiring, sent } = setup({ platform: "darwin" });
    wiring.setPreferences(CHECK_ONLY);
    await flush();
    const download = wiring.downloadNow();
    updater.finishDownload();
    await download;
    expect(nativeUpdater.checkForUpdates).not.toHaveBeenCalled();

    wiring.setPreferences(WITH_INSTALL);
    expect(nativeUpdater.checkForUpdates).toHaveBeenCalledTimes(1);
    expect(wiring.getState().handedOff).toBe(false);
    expect(sent("install-handed-off")).toEqual([]);

    // Asked again while Squirrel is fetching: nothing more.
    wiring.setPreferences(WITH_INSTALL);
    expect(nativeUpdater.checkForUpdates).toHaveBeenCalledTimes(1);

    nativeUpdater.received();
    expect(wiring.getState().handedOff).toBe(true);
    expect(sent("install-handed-off")).toEqual([{ version: "1.2.0" }]);
  });

  it("records the library's own hand-off at download only once Squirrel has it", async () => {
    const { updater, nativeUpdater, wiring, sent } = setup({ platform: "darwin" });
    wiring.setPreferences({ check: true, download: true, install: true });
    await flush();
    updater.finishDownload();
    expect(nativeUpdater.checkForUpdates).toHaveBeenCalledTimes(1); // by the library
    expect(wiring.getState().handedOff).toBe(false);

    nativeUpdater.received();
    expect(wiring.getState().handedOff).toBe(true);
    expect(sent("install-handed-off")).toEqual([{ version: "1.2.0" }]);

    // Already handed off: turning install off and on asks Squirrel nothing more.
    wiring.setPreferences(CHECK_ONLY);
    wiring.setPreferences(WITH_INSTALL);
    expect(nativeUpdater.checkForUpdates).toHaveBeenCalledTimes(1);
  });

  it("does not record a hand-off Squirrel refused", async () => {
    const { updater, nativeUpdater, wiring } = setup({ platform: "darwin" });
    wiring.setPreferences(CHECK_ONLY);
    await flush();
    const download = wiring.downloadNow();
    updater.finishDownload();
    await download;
    wiring.setPreferences(WITH_INSTALL);
    updater.emit("error", new Error("Code signature did not match"));
    expect(nativeUpdater.checkForUpdates).toHaveBeenCalledTimes(1);
    expect(wiring.getState().handedOff).toBe(false);
  });
});

describe("the manual controls", () => {
  it("check-now runs whatever Check says, and downloads only with Download on", async () => {
    const { updater, wiring } = setup();
    wiring.setPreferences({ check: false, download: false, install: false });
    await expect(wiring.checkNow()).resolves.toEqual({ ok: true });
    expect(updater.checkForUpdates).toHaveBeenCalledTimes(1);
    expect(updater.pending).toBeNull();
  });

  it("check-now before any snapshot downloads nothing", async () => {
    const { updater, wiring } = setup();
    await wiring.checkNow();
    expect(updater.pending).toBeNull();
  });

  it("answers an error instead of throwing", async () => {
    const { updater, wiring, logWarn } = setup();
    updater.checkForUpdates.mockRejectedValueOnce(new Error("offline"));
    updater.downloadUpdate.mockRejectedValueOnce(new Error("offline"));
    await expect(wiring.checkNow()).resolves.toEqual({ ok: false, reason: "error" });
    await expect(wiring.downloadNow()).resolves.toEqual({ ok: false, reason: "error" });
    expect(logWarn).toHaveBeenCalledTimes(2);
  });

  it("never touches autoInstallOnAppQuit", async () => {
    const { updater, wiring } = setup();
    wiring.setPreferences(WITH_INSTALL);
    await flush();
    updater.autoInstallOnAppQuit = "sentinel";
    await wiring.checkNow();
    await flush();
    wiring.downloadNow();
    expect(updater.autoInstallOnAppQuit).toBe("sentinel");
  });
});
