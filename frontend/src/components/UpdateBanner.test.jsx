// @vitest-environment jsdom
/**
 * The update banner (#563): one phase per updater state, driven through the
 * preload bridge only.
 *
 * - available: a version was found and nothing downloads on its own -- the
 *   banner names it and offers Download;
 * - downloading: a progress bar, no dismiss;
 * - ready: the update is on disk -- Install now.
 *
 * A dismissed banner stays dismissed for that version until the next launch:
 * the periodic check does not bring it back.
 */
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, act, waitFor } from "@testing-library/react";

import UpdateBanner from "./UpdateBanner.jsx";

let handler;
let cleanupIPC;

beforeEach(() => {
  handler = null;
  cleanupIPC = vi.fn();
  window.updaterAPI = {
    onUpdaterEvent: vi.fn((callback) => {
      handler = callback;
      return cleanupIPC;
    }),
    installNow: vi.fn(),
    downloadNow: vi.fn().mockResolvedValue({ ok: true }),
  };
});

afterEach(() => {
  cleanup();
  delete window.updaterAPI;
});

const emit = (payload) => {
  act(() => {
    handler(payload);
  });
};

const DISMISS = "Dismiss update notification";

describe("UpdateBanner phases", () => {
  it("renders nothing until an updater event arrives", () => {
    const { container } = render(<UpdateBanner />);
    expect(container.firstChild).toBeNull();
  });

  it("stays silent on the events it has nothing to say about", () => {
    const { container } = render(<UpdateBanner />);
    emit({ event: "checking-for-update" });
    emit({ event: "update-not-available" });
    emit({ event: "error" });
    emit({ event: "something-else", version: "9.9.9" });
    expect(container.firstChild).toBeNull();
  });

  it("names the version and offers Download when nothing downloads on its own", () => {
    render(<UpdateBanner />);
    emit({ event: "update-available", version: "1.2.0", downloading: false });

    expect(screen.getByText("v1.2.0")).toBeDefined();
    expect(screen.getByText(/available/)).toBeDefined();
    expect(document.body.textContent).not.toMatch(/downloading/i);
    expect(screen.getByRole("button", { name: "Download" })).toBeDefined();
    expect(screen.getByLabelText(DISMISS)).toBeDefined();
  });

  it("downloads only when Download is pressed", async () => {
    render(<UpdateBanner />);
    emit({ event: "update-available", version: "1.2.0", downloading: false });
    expect(window.updaterAPI.downloadNow).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Download" }));

    expect(window.updaterAPI.downloadNow).toHaveBeenCalledTimes(1);
    expect(await screen.findByText(/Downloading/)).toBeDefined();
  });

  it("goes back to Download when the download could not start", async () => {
    window.updaterAPI.downloadNow.mockResolvedValue({ ok: false, reason: "error" });
    render(<UpdateBanner />);
    emit({ event: "update-available", version: "1.2.0", downloading: false });

    fireEvent.click(screen.getByRole("button", { name: "Download" }));

    await waitFor(() => expect(screen.getByRole("button", { name: "Download" })).toBeDefined());
  });

  it("says downloading straight away when a download follows the check", () => {
    render(<UpdateBanner />);
    emit({ event: "update-available", version: "1.2.0", downloading: true });

    expect(screen.getByText(/Downloading/)).toBeDefined();
    expect(screen.queryByRole("button", { name: "Download" })).toBeNull();
  });

  it("shows progress with a bar and no dismiss button", () => {
    render(<UpdateBanner />);
    emit({ event: "update-available", version: "1.2.0", downloading: true });
    emit({ event: "download-progress", percent: 60 });

    expect(screen.getByText(/Downloading/)).toBeDefined();
    expect(screen.getByText("60%")).toBeDefined();
    expect(screen.getByText("v1.2.0")).toBeDefined();
    expect(document.querySelector('[style*="width: 60%"]')).not.toBeNull();
    expect(screen.queryByLabelText(DISMISS)).toBeNull();
  });

  it("offers Install now once the update is ready", () => {
    render(<UpdateBanner />);
    emit({ event: "update-downloaded", version: "1.2.0" });

    expect(screen.getByText(/ready/)).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Install now" }));
    expect(window.updaterAPI.installNow).toHaveBeenCalledTimes(1);
  });

  it.each(["error", "download-cancelled"])(
    "offers Download again, with a dismiss, when a download ends in %s",
    (event) => {
      render(<UpdateBanner />);
      emit({ event: "update-available", version: "1.2.0", downloading: true });
      emit({ event: "download-progress", percent: 30 });

      emit({ event });

      expect(screen.getByRole("button", { name: "Download" })).toBeDefined();
      expect(screen.getByLabelText(DISMISS)).toBeDefined();
    }
  );

  it("keeps showing a downloaded update when a later background check errors", () => {
    render(<UpdateBanner />);
    emit({ event: "update-downloaded", version: "1.2.0" });
    emit({ event: "error" });
    expect(screen.getByText(/ready/)).toBeDefined();
  });

  it("stays ready when a later check finds the same version again", () => {
    render(<UpdateBanner />);
    emit({ event: "update-downloaded", version: "1.2.0" });
    emit({ event: "update-available", version: "1.2.0", downloading: false });
    expect(screen.getByRole("button", { name: "Install now" })).toBeDefined();
  });
});

describe("UpdateBanner dismissal", () => {
  it("stays dismissed for that version when the periodic check finds it again", () => {
    const { container } = render(<UpdateBanner />);
    emit({ event: "update-available", version: "1.2.0", downloading: false });

    fireEvent.click(screen.getByLabelText(DISMISS));
    expect(container.firstChild).toBeNull();

    emit({ event: "update-available", version: "1.2.0", downloading: false });
    expect(container.firstChild).toBeNull();
    emit({ event: "update-downloaded", version: "1.2.0" });
    expect(container.firstChild).toBeNull();
  });

  it("comes back for a newer version", () => {
    const { container } = render(<UpdateBanner />);
    emit({ event: "update-available", version: "1.2.0", downloading: false });
    fireEvent.click(screen.getByLabelText(DISMISS));
    expect(container.firstChild).toBeNull();

    emit({ event: "update-available", version: "1.3.0", downloading: false });
    expect(screen.getByText("v1.3.0")).toBeDefined();
  });
});

describe("UpdateBanner bridge wiring", () => {
  it("subscribes through the preload bridge and cleans up on unmount", () => {
    const { unmount } = render(<UpdateBanner />);
    expect(window.updaterAPI.onUpdaterEvent).toHaveBeenCalledTimes(1);

    unmount();
    expect(cleanupIPC).toHaveBeenCalledTimes(1);
  });

  it("renders nothing and does not throw outside Electron", () => {
    delete window.updaterAPI;
    const { container } = render(<UpdateBanner />);
    expect(container.firstChild).toBeNull();
  });

  it("listens to no window-level test channel", () => {
    const { container } = render(<UpdateBanner />);
    act(() => {
      window.dispatchEvent(
        new CustomEvent("__test_updater__", {
          detail: { event: "update-available", version: "1.2.0" },
        })
      );
    });
    expect(container.firstChild).toBeNull();
  });
});
