// @vitest-environment jsdom
/**
 * The Settings card that is the only manual way out of a stuck update (#571).
 *
 * An update staged to install on quit is cancelled by the native installer
 * when the app reopens before the swap finishes, and the app is never told.
 * Once the banner is dismissed nothing else in the interface mentions
 * updates, so the version never moves. These tests drive the card through the
 * same `updater-event` payloads the main process sends and pin every state a
 * user can land in, including the one where the card mounts after the events
 * already fired.
 */
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, act, waitFor } from "@testing-library/react";

import UpdatesCard from "./UpdatesCard.jsx";
import i18n from "../i18n";

let emit;

/**
 * Wire the preload bridge the way main.js does: `onUpdaterEvent` hands the
 * component a callback, and `emit` plays an event through it.
 */
function mockUpdaterAPI(overrides = {}) {
  let handler = null;
  const api = {
    onUpdaterEvent: vi.fn((callback) => {
      handler = callback;
      return vi.fn();
    }),
    getState: vi.fn().mockResolvedValue({
      available: true,
      phase: "idle",
      version: null,
      percent: 0,
    }),
    checkNow: vi.fn().mockResolvedValue({ ok: true }),
    downloadNow: vi.fn().mockResolvedValue({ ok: true }),
    installNow: vi.fn().mockResolvedValue(undefined),
    setAutoUpdateEnabled: vi.fn(),
    ...overrides,
  };
  window.updaterAPI = api;
  emit = (payload) => act(() => handler?.(payload));
  return api;
}

beforeEach(() => {
  window.diagnosticsAPI = { getAppInfo: vi.fn().mockResolvedValue({ version: "1.0.0" }) };
  mockUpdaterAPI();
});

afterEach(async () => {
  cleanup();
  vi.clearAllMocks();
  delete window.updaterAPI;
  delete window.diagnosticsAPI;
  emit = null;
  await i18n.changeLanguage("en");
});

describe("UpdatesCard — the idle card", () => {
  it("names the version the user is actually running", async () => {
    render(<UpdatesCard />);
    expect(await screen.findByText(/version 1\.0\.0/)).toBeTruthy();
  });

  it("offers a check as its only action", async () => {
    render(<UpdatesCard />);
    const button = await screen.findByRole("button", { name: "Check for updates" });
    expect(button.disabled).toBe(false);
  });

  it("asks the main process for the state it may have missed before mounting", async () => {
    render(<UpdatesCard />);
    await waitFor(() => expect(window.updaterAPI.getState).toHaveBeenCalled());
  });
});

describe("UpdatesCard — checking", () => {
  it("runs the check through the bridge and holds the button while it runs", async () => {
    render(<UpdatesCard />);
    fireEvent.click(await screen.findByRole("button", { name: "Check for updates" }));

    await waitFor(() => expect(window.updaterAPI.checkNow).toHaveBeenCalledTimes(1));
    const button = screen.getByRole("button", { name: /Checking/ });
    expect(button.disabled).toBe(true);
  });

  it("says so, quietly, when there is nothing to install", async () => {
    render(<UpdatesCard />);
    fireEvent.click(await screen.findByRole("button", { name: "Check for updates" }));
    await waitFor(() => expect(window.updaterAPI.checkNow).toHaveBeenCalled());

    emit({ event: "update-not-available" });

    expect(screen.getByText("You are on the latest version.")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Check for updates" }).disabled).toBe(false);
  });
});

describe("UpdatesCard — an update that exists", () => {
  it("names the new version and offers the download", async () => {
    render(<UpdatesCard />);
    await screen.findByRole("button", { name: "Check for updates" });

    emit({ event: "update-available", version: "1.1.0" });

    expect(screen.getByText("Version 1.1.0 is available.")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Download" })).toBeTruthy();
  });

  it("downloads on click through the bridge", async () => {
    render(<UpdatesCard />);
    await screen.findByRole("button", { name: "Check for updates" });
    emit({ event: "update-available", version: "1.1.0" });

    fireEvent.click(screen.getByRole("button", { name: "Download" }));

    await waitFor(() => expect(window.updaterAPI.downloadNow).toHaveBeenCalledTimes(1));
  });

  it("offers no second download once one is running (the automatic flow already started it)", async () => {
    render(<UpdatesCard />);
    await screen.findByRole("button", { name: "Check for updates" });

    emit({ event: "update-available", version: "1.1.0" });
    emit({ event: "download-progress", percent: 42 });

    expect(screen.queryByRole("button", { name: "Download" })).toBeNull();
    expect(screen.getByText(/42/)).toBeTruthy();
  });

  it("offers no download once the update is already on disk", async () => {
    render(<UpdatesCard />);
    await screen.findByRole("button", { name: "Check for updates" });

    emit({ event: "update-downloaded", version: "1.1.0" });

    expect(screen.queryByRole("button", { name: "Download" })).toBeNull();
  });
});

describe("UpdatesCard — installing", () => {
  it("offers the install and quits, installs and relaunches through the bridge", async () => {
    render(<UpdatesCard />);
    await screen.findByRole("button", { name: "Check for updates" });

    emit({ event: "update-downloaded", version: "1.1.0" });

    expect(screen.getByText("Version 1.1.0 is ready to install.")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Install now" }));
    expect(window.updaterAPI.installNow).toHaveBeenCalledTimes(1);
  });

  it("recovers a staged update the card never saw arrive", async () => {
    // The exact shape of #571: the update was downloaded hours ago, the
    // banner was dismissed, the install-on-quit swap was cancelled. Mounting
    // the card must still find it.
    mockUpdaterAPI({
      getState: vi.fn().mockResolvedValue({
        available: true,
        phase: "downloaded",
        version: "1.1.0",
        percent: 100,
      }),
    });
    render(<UpdatesCard />);

    expect(await screen.findByRole("button", { name: "Install now" })).toBeTruthy();
    expect(screen.getByText("Version 1.1.0 is ready to install.")).toBeTruthy();
  });

  it("recovers a download that was running when the card mounted", async () => {
    mockUpdaterAPI({
      getState: vi.fn().mockResolvedValue({
        available: true,
        phase: "downloading",
        version: "1.1.0",
        percent: 70,
      }),
    });
    render(<UpdatesCard />);

    expect(await screen.findByText(/70/)).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Download" })).toBeNull();
  });
});

describe("UpdatesCard — failures stay on the card", () => {
  it("shows an inline line when the updater fails during a check the user asked for", async () => {
    render(<UpdatesCard />);
    fireEvent.click(await screen.findByRole("button", { name: "Check for updates" }));
    await waitFor(() => expect(window.updaterAPI.checkNow).toHaveBeenCalled());

    emit({ event: "error" });

    expect(screen.getByText(/could not be checked/i)).toBeTruthy();
    // The product decision for #571: no dialog, no popup, ever.
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(screen.getByRole("button", { name: "Check for updates" }).disabled).toBe(false);
  });

  it("stays quiet when a background check fails behind the user's back", async () => {
    render(<UpdatesCard />);
    await screen.findByRole("button", { name: "Check for updates" });

    emit({ event: "error" });

    expect(screen.queryByText(/could not be checked/i)).toBeNull();
  });

  it("reports a refused check even when no event follows", async () => {
    mockUpdaterAPI({ checkNow: vi.fn().mockResolvedValue({ ok: false, reason: "error" }) });
    render(<UpdatesCard />);
    fireEvent.click(await screen.findByRole("button", { name: "Check for updates" }));

    expect(await screen.findByText(/could not be checked/i)).toBeTruthy();
  });

  it("reports a refused download", async () => {
    mockUpdaterAPI({ downloadNow: vi.fn().mockResolvedValue({ ok: false, reason: "error" }) });
    render(<UpdatesCard />);
    await screen.findByRole("button", { name: "Check for updates" });
    emit({ event: "update-available", version: "1.1.0" });

    fireEvent.click(screen.getByRole("button", { name: "Download" }));

    expect(await screen.findByText(/could not be checked/i)).toBeTruthy();
  });
});

describe("UpdatesCard — builds that have no updater", () => {
  it("says updates are not handled here instead of offering a button that does nothing", async () => {
    mockUpdaterAPI({
      getState: vi.fn().mockResolvedValue({
        available: false,
        phase: "idle",
        version: null,
        percent: 0,
      }),
    });
    render(<UpdatesCard />);

    expect(await screen.findByText(/cannot check for them/i)).toBeTruthy();
    expect(screen.getByRole("button", { name: "Check for updates" }).disabled).toBe(true);
  });

  it("survives a renderer with no preload bridge at all", async () => {
    delete window.updaterAPI;
    delete window.diagnosticsAPI;
    render(<UpdatesCard />);

    expect(await screen.findByText(/cannot check for them/i)).toBeTruthy();
    expect(screen.getByRole("button", { name: "Check for updates" }).disabled).toBe(true);
  });

  it("does not crash when the main process refuses the state request", async () => {
    mockUpdaterAPI({ getState: vi.fn().mockRejectedValue(new Error("no bridge")) });
    render(<UpdatesCard />);

    expect(await screen.findByRole("button", { name: "Check for updates" })).toBeTruthy();
  });
});

describe("UpdatesCard — translation", () => {
  it("is translated like the rest of the page", async () => {
    await i18n.changeLanguage("fr");
    render(<UpdatesCard />);

    expect(await screen.findByRole("button", { name: "Rechercher des mises à jour" })).toBeTruthy();
  });

  it("unsubscribes from the bridge on unmount", async () => {
    const cleanupIPC = vi.fn();
    mockUpdaterAPI({ onUpdaterEvent: vi.fn(() => cleanupIPC) });
    const { unmount } = render(<UpdatesCard />);
    await waitFor(() => expect(window.updaterAPI.onUpdaterEvent).toHaveBeenCalled());

    unmount();

    expect(cleanupIPC).toHaveBeenCalledTimes(1);
  });
});
