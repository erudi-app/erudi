// @vitest-environment jsdom
/**
 * Handing the persisted update switches to the Electron main process (#563).
 *
 * Main owns electron-updater but has no database; the renderer owns the three
 * switches (Check for updates, Download automatically, Install when I quit).
 * Main is told the full snapshot on every path, and a value that could not be
 * read travels as `null`: main's policy (`updaterPolicy.js`) then downloads
 * and installs nothing.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  notifyUpdatePreferences,
  syncUpdatePreferencesWithMain,
  updatePreferencesFrom,
} from "./autoUpdate";

let setPreferences;

beforeEach(() => {
  setPreferences = vi.fn();
  window.updaterAPI = { setPreferences };
});

afterEach(() => {
  delete window.updaterAPI;
  vi.clearAllMocks();
});

describe("updatePreferencesFrom", () => {
  it("reads the three switches from the user settings", () => {
    expect(
      updatePreferencesFrom({
        auto_update_enabled: true,
        auto_download_updates: false,
        auto_install_updates: true,
      })
    ).toEqual({ check: true, download: false, install: true });
  });

  it("carries null for a switch it cannot read", () => {
    expect(updatePreferencesFrom({ auto_update_enabled: false })).toEqual({
      check: false,
      download: null,
      install: null,
    });
    expect(updatePreferencesFrom(null)).toEqual({ check: null, download: null, install: null });
    expect(updatePreferencesFrom({ auto_download_updates: "yes" }).download).toBeNull();
  });
});

describe("syncUpdatePreferencesWithMain", () => {
  it("hands main the full snapshot it read", async () => {
    const client = {
      get: vi.fn().mockResolvedValue({
        auto_update_enabled: true,
        auto_download_updates: true,
        auto_install_updates: false,
      }),
    };

    await expect(syncUpdatePreferencesWithMain(client)).resolves.toEqual({
      check: true,
      download: true,
      install: false,
    });

    expect(client.get).toHaveBeenCalledWith("/user_settings/");
    expect(setPreferences).toHaveBeenCalledWith({ check: true, download: true, install: false });
  });

  it("tells main it knows nothing when the settings request fails", async () => {
    const client = { get: vi.fn().mockRejectedValue(new Error("backend down")) };

    await syncUpdatePreferencesWithMain(client);

    expect(setPreferences).toHaveBeenCalledWith({ check: null, download: null, install: null });
  });

  it("does not throw outside Electron, where there is no bridge", async () => {
    delete window.updaterAPI;
    const client = { get: vi.fn().mockResolvedValue({ auto_update_enabled: false }) };

    await expect(syncUpdatePreferencesWithMain(client)).resolves.toEqual({
      check: false,
      download: null,
      install: null,
    });
  });
});

describe("notifyUpdatePreferences", () => {
  it("pushes the snapshot to main as it is", () => {
    notifyUpdatePreferences({ check: true, download: false, install: true });
    expect(setPreferences).toHaveBeenCalledWith({ check: true, download: false, install: true });
  });
});
