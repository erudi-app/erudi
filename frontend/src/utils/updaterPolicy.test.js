/**
 * The updater policy (#563): what the main process does with electron-updater
 * for each preference snapshot, platform and updater state.
 *
 * main.js runs in the Electron main process and is not covered by tests, so
 * every decision it takes about the updater is made here, per platform:
 * - the two library flags (`autoDownload`, `autoInstallOnAppQuit`), which are
 *   raised only from a known snapshot and only while the check is on;
 * - the periodic check;
 * - installing an update that was ALREADY downloaded when the install switch
 *   is turned on: Windows and Linux register the library's quit handler,
 *   macOS hands the update to Squirrel again;
 * - the banner's phase for each updater event.
 */
import { describe, it, expect } from "vitest";
import {
  NO_INSTALL_ARMED,
  bannerStateAfter,
  installMayStillHappen,
  isHandedOff,
  planPreferenceChange,
  resolveUpdatePreferences,
  updaterStateAfter,
} from "./updaterPolicy";

const PLATFORMS = ["win32", "linux", "darwin"];

const ALL_ON = { check: true, download: true, install: true };
const CHECK_ONLY = { check: true, download: false, install: false };

function plan(overrides = {}) {
  return planPreferenceChange({
    preferences: resolveUpdatePreferences(CHECK_ONLY),
    platform: "win32",
    downloadedVersion: null,
    armed: NO_INSTALL_ARMED,
    polling: false,
    automaticDownload: false,
    ...overrides,
  });
}

describe("resolveUpdatePreferences: the effective switches", () => {
  it.each([
    // check, download, install -> effective download, effective install
    [true, false, false, false, false],
    [true, true, false, true, false],
    [true, false, true, false, true],
    [true, true, true, true, true],
    // Check off: the two opt-ins are disabled and inert, whatever is stored.
    [false, true, true, false, false],
    [false, false, false, false, false],
  ])("check=%s download=%s install=%s", (check, download, install, effDownload, effInstall) => {
    expect(resolveUpdatePreferences({ check, download, install })).toEqual({
      check,
      download: effDownload,
      install: effInstall,
    });
  });

  it("checks, downloads and installs nothing when the preferences could not be read", () => {
    for (const snapshot of [
      null,
      undefined,
      {},
      { check: null, download: null, install: null },
      { check: "yes", download: "yes", install: 1 },
    ]) {
      expect(resolveUpdatePreferences(snapshot), JSON.stringify(snapshot)).toEqual({
        check: false,
        download: false,
        install: false,
      });
    }
  });

  it("checks only on an explicit yes: a later launch reads the settings again", () => {
    expect(resolveUpdatePreferences({ check: true }).check).toBe(true);
    expect(resolveUpdatePreferences({ check: null }).check).toBe(false);
    expect(resolveUpdatePreferences({ check: false }).check).toBe(false);
  });
});

describe("planPreferenceChange: the library flags", () => {
  it.each(PLATFORMS)("raises nothing for check-only on %s", (platform) => {
    const p = plan({ platform });
    expect(p.autoDownload).toBe(false);
    expect(p.autoInstallOnAppQuit).toBe(false);
  });

  it.each(PLATFORMS)("raises both flags when both opt-ins are on, on %s", (platform) => {
    const p = plan({ platform, preferences: resolveUpdatePreferences(ALL_ON) });
    expect(p.autoDownload).toBe(true);
    expect(p.autoInstallOnAppQuit).toBe(true);
  });

  it.each(PLATFORMS)("lowers both flags when Check is turned off, on %s", (platform) => {
    const p = plan({
      platform,
      preferences: resolveUpdatePreferences({ check: false, download: true, install: true }),
    });
    expect(p.autoDownload).toBe(false);
    expect(p.autoInstallOnAppQuit).toBe(false);
  });

  it("follows each opt-in on its own", () => {
    const download = plan({
      preferences: resolveUpdatePreferences({ check: true, download: true, install: false }),
    });
    expect([download.autoDownload, download.autoInstallOnAppQuit]).toEqual([true, false]);
    const install = plan({
      preferences: resolveUpdatePreferences({ check: true, download: false, install: true }),
    });
    expect([install.autoDownload, install.autoInstallOnAppQuit]).toEqual([false, true]);
  });

  it("raises nothing and starts no check from an unknown snapshot", () => {
    const p = plan({ preferences: resolveUpdatePreferences(null) });
    expect(p.autoDownload).toBe(false);
    expect(p.autoInstallOnAppQuit).toBe(false);
    expect(p.startPolling).toBe(false);
  });
});

describe("planPreferenceChange: the periodic check", () => {
  it("starts it when Check is on and it is not running", () => {
    expect(plan({ polling: false })).toMatchObject({ startPolling: true, stopPolling: false });
  });

  it("leaves a running check alone", () => {
    expect(plan({ polling: true })).toMatchObject({ startPolling: false, stopPolling: false });
  });

  it("stops it when Check is turned off", () => {
    const off = resolveUpdatePreferences({ check: false });
    expect(plan({ preferences: off, polling: true })).toMatchObject({
      startPolling: false,
      stopPolling: true,
    });
    expect(plan({ preferences: off, polling: false })).toMatchObject({
      startPolling: false,
      stopPolling: false,
    });
  });
});

describe("planPreferenceChange: install turned on after a finished download", () => {
  const install = resolveUpdatePreferences({ check: true, download: false, install: true });

  it.each(["win32", "linux"])("registers the library's quit handler on %s", (platform) => {
    expect(plan({ platform, preferences: install, downloadedVersion: "1.2.0" }).armInstall).toBe(
      "quit-handler"
    );
  });

  it("hands the update to Squirrel again on macOS", () => {
    expect(
      plan({ platform: "darwin", preferences: install, downloadedVersion: "1.2.0" }).armInstall
    ).toBe("squirrel");
  });

  it.each(PLATFORMS)("arms nothing before any download on %s", (platform) => {
    expect(plan({ platform, preferences: install, downloadedVersion: null }).armInstall).toBe(null);
  });

  it.each(PLATFORMS)("arms nothing while install is off on %s", (platform) => {
    expect(plan({ platform, downloadedVersion: "1.2.0" }).armInstall).toBe(null);
  });

  it.each(PLATFORMS)(
    "arms nothing while Check is off, even with install stored on, on %s",
    (platform) => {
      const checkOff = resolveUpdatePreferences({ check: false, download: false, install: true });
      expect(plan({ platform, preferences: checkOff, downloadedVersion: "1.2.0" }).armInstall).toBe(
        null
      );
    }
  );

  it.each(["win32", "linux"])("does not register the quit handler twice on %s", (platform) => {
    expect(
      plan({
        platform,
        preferences: install,
        downloadedVersion: "1.2.0",
        armed: { ...NO_INSTALL_ARMED, quitHandler: true },
      }).armInstall
    ).toBe(null);
  });

  it("does not hand the same version to Squirrel twice on macOS", () => {
    expect(
      plan({
        platform: "darwin",
        preferences: install,
        downloadedVersion: "1.2.0",
        armed: { ...NO_INSTALL_ARMED, handedOffVersion: "1.2.0" },
      }).armInstall
    ).toBe(null);
  });

  it("does not ask Squirrel again while it is fetching that version (macOS)", () => {
    expect(
      plan({
        platform: "darwin",
        preferences: install,
        downloadedVersion: "1.2.0",
        armed: { ...NO_INSTALL_ARMED, handOffRequestedVersion: "1.2.0" },
      }).armInstall
    ).toBe(null);
  });

  it("hands a newer download to Squirrel on macOS even after an older hand-off", () => {
    expect(
      plan({
        platform: "darwin",
        preferences: install,
        downloadedVersion: "1.3.0",
        armed: { ...NO_INSTALL_ARMED, handedOffVersion: "1.2.0" },
      }).armInstall
    ).toBe("squirrel");
  });
});

describe("planPreferenceChange: an automatic download in progress (P3-9)", () => {
  it.each([
    ["Download", { check: true, download: false, install: false }],
    ["Check", { check: false, download: true, install: false }],
  ])("cancels it when %s is turned off", (_name, snapshot) => {
    expect(
      plan({ preferences: resolveUpdatePreferences(snapshot), automaticDownload: true })
        .cancelDownload
    ).toBe(true);
  });

  it("lets it run while the effective Download switch stays on", () => {
    expect(
      plan({ preferences: resolveUpdatePreferences(ALL_ON), automaticDownload: true })
        .cancelDownload
    ).toBe(false);
  });

  it("has nothing to cancel when no automatic download runs", () => {
    expect(plan({ automaticDownload: false }).cancelDownload).toBe(false);
  });
});

describe("installMayStillHappen: the macOS note", () => {
  it("warns when install was handed off and is now off", () => {
    expect(
      installMayStillHappen({
        handedOff: true,
        preferences: resolveUpdatePreferences(CHECK_ONLY),
      })
    ).toBe(true);
  });

  it("warns when Check is turned off after the hand-off", () => {
    expect(
      installMayStillHappen({
        handedOff: true,
        preferences: resolveUpdatePreferences({ check: false, download: false, install: true }),
      })
    ).toBe(true);
  });

  it("says nothing while install is on, or when nothing was handed off", () => {
    expect(
      installMayStillHappen({
        handedOff: true,
        preferences: resolveUpdatePreferences({ check: true, download: false, install: true }),
      })
    ).toBe(false);
    expect(
      installMayStillHappen({
        handedOff: false,
        preferences: resolveUpdatePreferences(CHECK_ONLY),
      })
    ).toBe(false);
  });
});

describe("isHandedOff", () => {
  it("is true only once Squirrel holds a version (macOS)", () => {
    expect(isHandedOff({ ...NO_INSTALL_ARMED, handedOffVersion: "1.2.0" })).toBe(true);
    expect(isHandedOff(NO_INSTALL_ARMED)).toBe(false);
  });

  it("is false for the Windows and Linux quit handler, which reads the flag at quit", () => {
    expect(isHandedOff({ ...NO_INSTALL_ARMED, quitHandler: true })).toBe(false);
    // Asked, not yet received: Squirrel may still fail.
    expect(isHandedOff({ ...NO_INSTALL_ARMED, handOffRequestedVersion: "1.2.0" })).toBe(false);
  });
});

describe("bannerStateAfter: the banner's phase per event", () => {
  it("offers Download when the update was found and nothing downloads", () => {
    expect(
      bannerStateAfter(null, { event: "update-available", version: "1.2.0", downloading: false })
    ).toEqual({ phase: "available", version: "1.2.0" });
  });

  it("says downloading, not available, when a download follows the check", () => {
    expect(
      bannerStateAfter(null, { event: "update-available", version: "1.2.0", downloading: true })
    ).toEqual({ phase: "downloading", version: "1.2.0", percent: 0 });
  });

  it("follows the progress and keeps the version", () => {
    expect(
      bannerStateAfter(
        { phase: "available", version: "1.2.0" },
        { event: "download-progress", percent: 40 }
      )
    ).toEqual({ phase: "downloading", version: "1.2.0", percent: 40 });
  });

  it("is ready once downloaded", () => {
    expect(
      bannerStateAfter(
        { phase: "downloading", version: "1.2.0", percent: 99 },
        { event: "update-downloaded", version: "1.2.0" }
      )
    ).toEqual({ phase: "ready", version: "1.2.0" });
  });

  it("stays ready when a later check finds the same version again", () => {
    const ready = { phase: "ready", version: "1.2.0" };
    expect(
      bannerStateAfter(ready, { event: "update-available", version: "1.2.0", downloading: false })
    ).toBe(ready);
  });

  it("moves on to a newer version found after a download", () => {
    expect(
      bannerStateAfter(
        { phase: "ready", version: "1.2.0" },
        { event: "update-available", version: "1.3.0", downloading: false }
      )
    ).toEqual({ phase: "available", version: "1.3.0" });
  });

  it.each(["error", "download-cancelled"])(
    "offers Download again when a download ends in %s",
    (event) => {
      expect(
        bannerStateAfter({ phase: "downloading", version: "1.2.0", percent: 40 }, { event })
      ).toEqual({ phase: "available", version: "1.2.0" });
    }
  );

  it("ignores the events the banner has nothing to say about", () => {
    const ready = { phase: "ready", version: "1.2.0" };
    for (const event of [
      "checking-for-update",
      "update-not-available",
      "error",
      "install-handed-off",
      "unknown",
    ]) {
      expect(bannerStateAfter(ready, { event }), event).toBe(ready);
      expect(bannerStateAfter(null, { event }), event).toBe(null);
    }
  });
});

describe("updaterStateAfter: the state main and the Update Erudi card keep", () => {
  const IDLE = { phase: "idle", version: null, percent: 0, downloadedVersion: null };
  const DOWNLOADED = {
    phase: "downloaded",
    version: "1.2.0",
    percent: 100,
    downloadedVersion: "1.2.0",
  };

  it("walks check, available, progress and download", () => {
    let state = updaterStateAfter(IDLE, { event: "checking-for-update" });
    expect(state.phase).toBe("checking");
    state = updaterStateAfter(state, { event: "update-available", version: "1.2.0" });
    expect(state).toMatchObject({ phase: "available", version: "1.2.0", percent: 0 });
    state = updaterStateAfter(state, { event: "download-progress", percent: 30 });
    expect(state).toMatchObject({ phase: "downloading", version: "1.2.0", percent: 30 });
    state = updaterStateAfter(state, { event: "update-downloaded", version: "1.2.0" });
    expect(state).toEqual(DOWNLOADED);
  });

  it("says up to date when nothing is newer", () => {
    expect(updaterStateAfter(IDLE, { event: "update-not-available" })).toMatchObject({
      phase: "up-to-date",
      version: null,
    });
  });

  it("keeps a downloaded update when the 4-hourly check finds the same version (P2-2)", () => {
    for (const event of [
      { event: "checking-for-update" },
      { event: "update-available", version: "1.2.0" },
      { event: "update-not-available" },
      { event: "error" },
    ]) {
      expect(updaterStateAfter(DOWNLOADED, event), event.event).toEqual(DOWNLOADED);
    }
  });

  it("keeps a download in flight when a check finds the same version (P3-A)", () => {
    const downloading = {
      phase: "downloading",
      version: "1.2.0",
      percent: 40,
      downloadedVersion: null,
    };
    expect(updaterStateAfter(downloading, { event: "update-available", version: "1.2.0" })).toBe(
      downloading
    );
    expect(updaterStateAfter(downloading, { event: "checking-for-update" })).toBe(downloading);
    expect(
      updaterStateAfter(downloading, { event: "update-available", version: "1.3.0" })
    ).toMatchObject({ phase: "available", version: "1.3.0" });
  });

  it("moves on to a newer version found after a download", () => {
    expect(
      updaterStateAfter(DOWNLOADED, { event: "update-available", version: "1.3.0" })
    ).toMatchObject({ phase: "available", version: "1.3.0", downloadedVersion: "1.2.0" });
  });

  it.each(["error", "download-cancelled"])(
    "offers Download again when a download ends in %s (P3-4)",
    (event) => {
      const downloading = {
        phase: "downloading",
        version: "1.2.0",
        percent: 40,
        downloadedVersion: null,
      };
      expect(updaterStateAfter(downloading, { event })).toMatchObject({
        phase: "available",
        version: "1.2.0",
        percent: 0,
      });
    }
  );

  it("reports a failed check as an error", () => {
    expect(updaterStateAfter({ ...IDLE, phase: "checking" }, { event: "error" }).phase).toBe(
      "error"
    );
  });

  it("ignores events it does not know", () => {
    expect(updaterStateAfter(DOWNLOADED, { event: "install-handed-off" })).toBe(DOWNLOADED);
  });
});
