// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";

// The widget is a pure renderer driven by the context (#292): every phase is
// exercised here from props alone, so the layout per state is pinned without
// driving the poll. The state machine itself is covered by the context tests.

import DownloadWidget from "./DownloadWidget";

// Decimal, like the model card and Hugging Face (#753).
const GB = 1_000_000_000;
const MB = 1_000_000;

const onToggleCollapse = vi.fn();
const onCancel = vi.fn();
const onDismiss = vi.fn();

const base = {
  modelName: "Base Model",
  phase: "downloading",
  progress: 42.35,
  timeLeft: 3700,
  totalBytes: 4 * GB,
  downloadedBytes: 1 * GB,
  speedBytesPerSec: 12 * MB,
  message: "",
  collapsed: false,
  onToggleCollapse,
  onCancel,
  onDismiss,
};

const renderWidget = (overrides = {}) => render(<DownloadWidget {...base} {...overrides} />);

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("DownloadWidget while downloading", () => {
  it("shows the model, the phase line, a real progress bar with its percent, bytes, speed and ETA", () => {
    renderWidget();

    expect(screen.getByText("Base Model")).toBeTruthy();
    expect(screen.getByRole("status").textContent).toBe("Downloading");

    const bar = screen.getByRole("progressbar");
    expect(bar.getAttribute("aria-valuenow")).toBe("42");
    expect(bar.getAttribute("aria-valuemin")).toBe("0");
    expect(bar.getAttribute("aria-valuemax")).toBe("100");
    expect(screen.getByText("42.4%")).toBeTruthy(); // formatPercent, locale-aware (#385)

    expect(screen.getByText("1 GB of 4 GB")).toBeTruthy();
    expect(screen.getByText("12 MB/s")).toBeTruthy();
    expect(screen.getByText("1h 1m left")).toBeTruthy(); // 3700s
  });

  it("shows the total the model card shows (#753)", () => {
    renderWidget({ totalBytes: 2_279_000_000, downloadedBytes: 1_000_000_000 });
    expect(screen.getByText("1 GB of 2.3 GB")).toBeTruthy();
  });

  it("exposes Cancel and Collapse as labelled buttons and wires them to the context", () => {
    renderWidget();

    fireEvent.click(screen.getByLabelText("Cancel"));
    expect(onCancel).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByLabelText("Collapse"));
    expect(onToggleCollapse).toHaveBeenCalledTimes(1);

    expect(screen.queryByText("Dismiss")).toBeNull();
  });

  it("omits the readouts the API has not measured yet instead of inventing them", () => {
    renderWidget({ totalBytes: 0, downloadedBytes: null, speedBytesPerSec: null, timeLeft: 0 });

    expect(screen.queryByText(/of /)).toBeNull();
    expect(screen.queryByText(/\/s$/)).toBeNull();
    expect(screen.queryByText(/left$/)).toBeNull();
    expect(screen.getByText("42.4%")).toBeTruthy();
  });

  it("is anchored bottom-left, clear of the 56px navigation rail and above the connection pill", () => {
    // The old strip floated a spinner over the Settings gear (#347) and had to
    // dodge the connection pill at the bottom of the models sidebar (#303). The
    // panel now starts right of the rail and above that pill by construction.
    renderWidget();
    const root = screen.getByLabelText("Model download");
    expect(root.className).toContain("fixed");
    // Tailwind emits `.relative` after `.fixed`, so the class must not be on the
    // root at all or the panel silently falls into the document flow.
    expect(root.className.split(" ")).not.toContain("relative");
    expect(root.className).toContain("left-[4.25rem]");
    expect(root.className).toContain("bottom-14");
  });
});

describe("DownloadWidget collapsed", () => {
  it("shrinks to a pill with the percent and an Expand control, keeping the phase line for assistive tech", () => {
    renderWidget({ collapsed: true });

    expect(screen.queryByText("Base Model")).toBeNull();
    expect(screen.getByText("42.4%")).toBeTruthy();
    expect(screen.queryByLabelText("Cancel")).toBeNull();
    expect(screen.getByRole("status").textContent).toBe("Downloading");

    fireEvent.click(screen.getByLabelText("Expand"));
    expect(onToggleCollapse).toHaveBeenCalledTimes(1);
  });
});

describe("DownloadWidget queued and finalizing", () => {
  it("says the download is starting, with no percent yet but a way to cancel", () => {
    renderWidget({ phase: "queued", progress: 0, timeLeft: null, totalBytes: 0 });

    expect(screen.getByRole("status").textContent).toBe("Starting download…");
    expect(screen.queryByText(/%/)).toBeNull();
    expect(screen.getByLabelText("Cancel")).toBeTruthy();
  });

  it("holds the bar full while finalizing and drops the ETA", () => {
    renderWidget({ phase: "finalizing", progress: 100, timeLeft: 0 });

    expect(screen.getByRole("status").textContent).toBe("Finalizing…");
    expect(screen.getByRole("progressbar").getAttribute("aria-valuenow")).toBe("100");
    expect(screen.getByText("100%")).toBeTruthy();
    expect(screen.queryByText(/left$/)).toBeNull();
  });
});

describe("DownloadWidget terminal states", () => {
  it("confirms completion inline with no cancel and no dismiss (it auto-dismisses)", () => {
    renderWidget({ phase: "completed", progress: 100, timeLeft: 0 });

    expect(screen.getByRole("status").textContent).toBe("Download complete");
    expect(screen.getByText("Available in your installed models")).toBeTruthy();
    expect(screen.queryByLabelText("Cancel")).toBeNull();
    expect(screen.queryByText("Dismiss")).toBeNull();
  });

  it("shows the failure message with a Dismiss action and no progress bar", () => {
    renderWidget({ phase: "failed", progress: 10, message: "disk full" });

    expect(screen.getByRole("status").textContent).toBe("Download failed");
    expect(screen.getByText("disk full")).toBeTruthy();
    expect(screen.queryByRole("progressbar")).toBeNull();
    expect(screen.queryByLabelText("Cancel")).toBeNull();

    fireEvent.click(screen.getByText("Dismiss"));
    expect(onDismiss).toHaveBeenCalledTimes(1);
  });

  it("does not call a stalled finalization a failure (#315)", () => {
    renderWidget({ phase: "stalled", progress: 100, message: "The files have been saved." });

    expect(screen.getByRole("status").textContent).toBe("Finalization is taking too long");
    expect(screen.queryByText(/failed/i)).toBeNull();
    expect(screen.getByText("The files have been saved.")).toBeTruthy();
    expect(screen.getByText("Dismiss")).toBeTruthy();
  });

  it("acknowledges a cancellation briefly with nothing left to act on", () => {
    renderWidget({ phase: "cancelled", progress: 0 });

    expect(screen.getByRole("status").textContent).toBe("Download cancelled");
    expect(screen.queryByRole("progressbar")).toBeNull();
    expect(screen.queryByLabelText("Cancel")).toBeNull();
    expect(screen.queryByText("Dismiss")).toBeNull();
  });

  it("keeps the cancel detail when the backend could not be reached", () => {
    renderWidget({ phase: "cancelled", progress: 0, message: "Failed to cancel download: boom" });

    expect(screen.getByText("Failed to cancel download: boom")).toBeTruthy();
  });
});
