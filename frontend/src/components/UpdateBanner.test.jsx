// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, act } from "@testing-library/react";

import UpdateBanner from "./UpdateBanner.jsx";

// The component wires a window-level "__test_updater__" CustomEvent channel in
// addition to the preload IPC bridge; these tests drive the state machine
// through that channel and pin every user-visible banner phase.
const emit = (detail) => {
  act(() => {
    window.dispatchEvent(new CustomEvent("__test_updater__", { detail }));
  });
};

afterEach(() => {
  cleanup();
  delete window.updaterAPI;
});

describe("UpdateBanner state machine", () => {
  it("renders nothing until an updater event arrives", () => {
    const { container } = render(<UpdateBanner />);
    expect(container.firstChild).toBeNull();
  });

  it("ignores unknown updater events", () => {
    const { container } = render(<UpdateBanner />);
    emit({ event: "something-else", version: "9.9.9" });
    expect(container.firstChild).toBeNull();
  });

  it("stays silent on the events added for the Settings card (#571)", () => {
    // Main forwards "checking-for-update", "update-not-available" and "error"
    // on the same channel so the Settings card can answer a manual check. The
    // banner is for an update that exists, and must not turn any of them into
    // a notification.
    const { container } = render(<UpdateBanner />);
    emit({ event: "checking-for-update" });
    emit({ event: "update-not-available" });
    emit({ event: "error" });
    expect(container.firstChild).toBeNull();
  });

  it("keeps showing a downloaded update when a later background check errors", () => {
    render(<UpdateBanner />);
    emit({ event: "update-downloaded", version: "1.2.0" });
    emit({ event: "error" });
    expect(screen.getByText(/ready/)).toBeDefined();
  });

  it("shows the available phase with the version and a dismiss button", () => {
    render(<UpdateBanner />);
    emit({ event: "update-available", version: "1.2.0" });

    expect(screen.getByText("v1.2.0")).toBeDefined();
    expect(screen.getByText(/available/)).toBeDefined();
    expect(screen.getByLabelText("Dismiss update notification")).toBeDefined();
  });

  it("shows download progress with a percent bar and no dismiss button", () => {
    render(<UpdateBanner />);
    emit({ event: "update-available", version: "1.2.0" });
    emit({ event: "download-progress", percent: 60 });

    expect(screen.getByText(/Downloading/)).toBeDefined();
    expect(screen.getByText("60%")).toBeDefined();
    // The version is carried over from the "available" phase.
    expect(screen.getByText("v1.2.0")).toBeDefined();
    const bar = document.querySelector('[style*="width: 60%"]');
    expect(bar).not.toBeNull();
    expect(screen.queryByLabelText("Dismiss update notification")).toBeNull();
  });

  it("falls back to an empty version when progress arrives without a prior event", () => {
    render(<UpdateBanner />);
    emit({ event: "download-progress", percent: 15 });

    expect(screen.getByText(/Downloading/)).toBeDefined();
    expect(screen.getByText("15%")).toBeDefined();
    expect(screen.getByText("v")).toBeDefined();
  });

  it("shows the ready phase and installs on click through the preload bridge", () => {
    const installNow = vi.fn();
    window.updaterAPI = {
      onUpdaterEvent: vi.fn(() => null),
      installNow,
    };
    render(<UpdateBanner />);
    emit({ event: "update-downloaded", version: "1.2.0" });

    expect(screen.getByText(/ready/)).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "restart to install" }));
    expect(installNow).toHaveBeenCalledTimes(1);
  });

  it("does not crash on install when no preload bridge is present", () => {
    render(<UpdateBanner />);
    emit({ event: "update-downloaded", version: "1.2.0" });
    fireEvent.click(screen.getByRole("button", { name: "restart to install" }));
    expect(screen.getByText(/ready/)).toBeDefined();
  });

  it("dismiss hides the banner and a new update makes it reappear", () => {
    const { container } = render(<UpdateBanner />);
    emit({ event: "update-available", version: "1.2.0" });

    fireEvent.click(screen.getByLabelText("Dismiss update notification"));
    expect(container.firstChild).toBeNull();

    emit({ event: "update-available", version: "1.3.0" });
    expect(screen.getByText("v1.3.0")).toBeDefined();
  });
});

describe("UpdateBanner IPC bridge wiring", () => {
  it("subscribes to updater events via the preload bridge and cleans up on unmount", () => {
    let bridgeHandler = null;
    const cleanupIPC = vi.fn();
    window.updaterAPI = {
      onUpdaterEvent: vi.fn((handler) => {
        bridgeHandler = handler;
        return cleanupIPC;
      }),
      installNow: vi.fn(),
    };

    const { unmount } = render(<UpdateBanner />);
    expect(window.updaterAPI.onUpdaterEvent).toHaveBeenCalledTimes(1);

    act(() => {
      bridgeHandler({ event: "update-available", version: "2.0.0" });
    });
    expect(screen.getByText("v2.0.0")).toBeDefined();

    unmount();
    expect(cleanupIPC).toHaveBeenCalledTimes(1);
  });
});
