// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, act } from "@testing-library/react";

// Facets not covered by DownloadModalContext.test.jsx: the progress widget as
// driven by the poll (time-left formatting across units, progress percent, the
// inline terminal states and their auto-dismiss -- #292), dismissing the
// confirmation, and the two cancel fallbacks (no job id yet; cancel endpoint
// failing -> local cleanup with the CANCELLED sentinel).

const { tracedFetchMock } = vi.hoisted(() => ({ tracedFetchMock: vi.fn() }));

vi.mock("../services/api/client", () => ({
  default: { get: vi.fn() },
  apiClient: { get: vi.fn() },
  tracedFetch: tracedFetchMock,
}));

vi.mock("../components/modals/ConfirmationModal", () => ({
  default: ({ onConfirm, onCancel }) => (
    <div>
      <button onClick={onConfirm}>CONFIRM</button>
      <button onClick={onCancel}>DISMISS</button>
    </div>
  ),
}));

import { DownloadModalProvider, useDownloadModal } from "./DownloadModalContext";
import { DOWNLOAD_CANCELLED } from "../utils/downloadStatus";

let postResponder;
let statusResponder;
let cancelResponder;

const onComplete = vi.fn();
const onError = vi.fn();

function Consumer() {
  const ctx = useDownloadModal();
  return (
    <>
      <button onClick={() => ctx.open({ id: 1, name: "Base Model" }, { onComplete, onError })}>
        OPEN
      </button>
      <span data-testid="downloading">{String(ctx.isDownloading)}</span>
    </>
  );
}

const renderProvider = () =>
  render(
    <DownloadModalProvider>
      <Consumer />
    </DownloadModalProvider>
  );

const widget = () => screen.queryByLabelText("Model download");
const phaseLine = () => screen.getByRole("status").textContent;

beforeEach(() => {
  vi.useFakeTimers();
  const root = document.createElement("div");
  root.setAttribute("id", "modal-root");
  document.body.appendChild(root);

  onComplete.mockReset();
  onError.mockReset();

  postResponder = () => ({ ok: true, json: async () => ({ id: "job1" }) });
  statusResponder = () => ({
    ok: true,
    json: async () => ({ status: "running", progress: 42.35, time_left: 3700 }),
  });
  cancelResponder = () => ({ ok: true });

  tracedFetchMock.mockReset();
  tracedFetchMock.mockImplementation(async (url, opts = {}) => {
    const u = String(url);
    const method = opts.method || "GET";
    if (method === "POST" && u.endsWith("/download")) return postResponder();
    if (u.includes("/downloads/") && u.endsWith("/cancel")) return cancelResponder();
    if (u.includes("/downloads/") && u.endsWith("/status")) return statusResponder();
    return { ok: true, json: async () => ({}) };
  });
});

afterEach(() => {
  cleanup();
  document.getElementById("modal-root")?.remove();
  vi.runOnlyPendingTimers();
  vi.useRealTimers();
});

// open -> confirm -> flush the POST and one 2s poll tick. The widget expands
// by itself at the 2s mark (the auto-uncollapse timeout), so no Expand click.
const startPollAndExpand = async () => {
  await act(async () => {
    fireEvent.click(screen.getByText("OPEN"));
  });
  await act(async () => {
    fireEvent.click(screen.getByText("CONFIRM"));
    await vi.advanceTimersByTimeAsync(2000);
  });
};

const advance = async (ms) => {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
};

describe("Download widget progress readout", () => {
  it("shows the model, hours-scale time left and the progress percent while running", async () => {
    renderProvider();
    await startPollAndExpand();

    expect(screen.getByText("Base Model")).toBeTruthy();
    expect(phaseLine()).toBe("Downloading");
    expect(screen.getByText("1h 1m left")).toBeTruthy(); // 3700s
    expect(screen.getByText("42.4%")).toBeTruthy(); // formatPercent, locale-aware (#385)
  });

  it("derives bytes and speed from the status reply's total_bytes and time_left", async () => {
    const GIB = 1_000_000_000; // decimal, like the model card (#753)
    statusResponder = () => ({
      ok: true,
      // 25% of 8 GB done, 6 GB left in 600s -> 10 MB/s
      json: async () => ({ status: "running", progress: 25, time_left: 600, total_bytes: 8 * GIB }),
    });
    renderProvider();
    await startPollAndExpand();

    expect(screen.getByText("2 GB of 8 GB")).toBeTruthy();
    expect(screen.getByText("10 MB/s")).toBeTruthy();
  });

  it("formats day-scale and minute-scale times and collapses back on demand", async () => {
    statusResponder = () => ({
      ok: true,
      json: async () => ({ status: "running", progress: 1, time_left: 90000 }),
    });
    renderProvider();
    await startPollAndExpand();
    expect(screen.getByText("1d 1h left")).toBeTruthy(); // 90000s

    statusResponder = () => ({
      ok: true,
      json: async () => ({ status: "running", progress: 2, time_left: 95 }),
    });
    await advance(2000);
    expect(screen.getByText("1m 35s left")).toBeTruthy();

    statusResponder = () => ({
      ok: true,
      json: async () => ({ status: "running", progress: 3, time_left: 40 }),
    });
    await advance(2000);
    expect(screen.getByText("40s left")).toBeTruthy();

    // Collapse shrinks the panel to the percent pill: the name goes, the
    // percent and the live phase line stay.
    await act(async () => {
      fireEvent.click(screen.getByLabelText("Collapse"));
    });
    await advance(500); // let the cross-fade settle
    expect(screen.queryByText("Base Model")).toBeNull();
    expect(screen.getByText("3%")).toBeTruthy();
    expect(phaseLine()).toBe("Downloading");
    expect(screen.getByLabelText("Expand")).toBeTruthy();
  });

  it("says the download is starting while the job is still pending", async () => {
    statusResponder = () => ({
      ok: true,
      json: async () => ({ status: "pending", progress: 0, time_left: null }),
    });
    renderProvider();
    await startPollAndExpand();

    expect(phaseLine()).toBe("Starting download…");
    expect(screen.queryByText(/%/)).toBeNull();
    expect(screen.queryByText(/left$/)).toBeNull();
  });

  it("switches to finalizing once the bytes are in and the job is still running", async () => {
    statusResponder = () => ({
      ok: true,
      json: async () => ({ status: "running", progress: 100, time_left: 0 }),
    });
    renderProvider();
    await startPollAndExpand();

    expect(phaseLine()).toBe("Finalizing…");
    expect(screen.getByText("100%")).toBeTruthy();
  });
});

describe("Download widget terminal states (#292)", () => {
  it("confirms completion inline, releases isDownloading at once, then dismisses itself", async () => {
    let polls = 0;
    statusResponder = () => {
      polls += 1;
      return {
        ok: true,
        json: async () =>
          polls < 2
            ? { status: "running", progress: 60, time_left: 30 }
            : { status: "completed", progress: 100, time_left: 0 },
      };
    };
    renderProvider();
    await startPollAndExpand();
    await advance(2000);

    expect(phaseLine()).toBe("Download complete");
    expect(screen.getByTestId("downloading").textContent).toBe("false");
    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(screen.queryByLabelText("Cancel")).toBeNull();
    expect(widget()).toBeTruthy();

    await advance(4000); // auto-dismiss fires
    await advance(1000); // exit transition plays out
    expect(widget()).toBeNull();
  });

  it("shows a failure inline with its message and stays until dismissed", async () => {
    statusResponder = () => ({
      ok: true,
      json: async () => ({ status: "failed", error_message: "disk full", progress: 10 }),
    });
    renderProvider();
    await startPollAndExpand();

    expect(phaseLine()).toBe("Download failed");
    expect(screen.getByText("disk full")).toBeTruthy();
    expect(screen.getByTestId("downloading").textContent).toBe("false");
    expect(onError).toHaveBeenCalledWith("disk full");

    await advance(30000);
    expect(widget()).toBeTruthy(); // no auto-dismiss on failure

    await act(async () => {
      fireEvent.click(screen.getByText("Dismiss"));
    });
    await advance(1000);
    expect(widget()).toBeNull();
  });

  it("shows a failed START inline instead of vanishing", async () => {
    postResponder = () => ({
      ok: false,
      status: 500,
      text: async () => JSON.stringify({ detail: "Model 1 not found" }),
    });
    renderProvider();
    await startPollAndExpand();

    expect(phaseLine()).toBe("Download failed");
    // The FastAPI envelope is unwrapped: the sentence, not the raw JSON.
    expect(screen.getByText("Failed to start download (500): Model 1 not found")).toBeTruthy();
    expect(screen.getByText("Dismiss")).toBeTruthy();
  });

  it("opens the panel at once when the start fails, without waiting for the auto-expand", async () => {
    postResponder = () => ({ ok: false, status: 500, text: async () => "boom" });
    renderProvider();
    await act(async () => {
      fireEvent.click(screen.getByText("OPEN"));
    });
    await act(async () => {
      fireEvent.click(screen.getByText("CONFIRM"));
      await vi.advanceTimersByTimeAsync(0);
    });

    // Before the 2 s auto-expand: the detail and Dismiss are already there.
    expect(screen.getByText("Base Model")).toBeTruthy();
    expect(screen.getByText("Failed to start download (500): boom")).toBeTruthy();
    expect(screen.getByText("Dismiss")).toBeTruthy();
  });

  it("acknowledges a user cancel inline, then dismisses itself", async () => {
    renderProvider();
    await startPollAndExpand();

    await act(async () => {
      fireEvent.click(screen.getByLabelText("Cancel"));
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(phaseLine()).toBe("Download cancelled");
    expect(screen.getByTestId("downloading").textContent).toBe("false");
    expect(onError).toHaveBeenCalledWith(DOWNLOAD_CANCELLED);

    await advance(2500); // auto-dismiss fires
    await advance(1000); // exit transition plays out
    expect(widget()).toBeNull();
  });

  it("resets to a fresh queued panel when another download starts after a terminal one", async () => {
    statusResponder = () => ({
      ok: true,
      json: async () => ({ status: "failed", error_message: "disk full", progress: 10 }),
    });
    renderProvider();
    await startPollAndExpand();
    expect(phaseLine()).toBe("Download failed");

    statusResponder = () => ({
      ok: true,
      json: async () => ({ status: "running", progress: 5, time_left: 100 }),
    });
    await startPollAndExpand();

    expect(phaseLine()).toBe("Downloading");
    expect(screen.queryByText("disk full")).toBeNull();
    expect(screen.getByTestId("downloading").textContent).toBe("true");
  });
});

describe("Confirmation dismissal", () => {
  it("dismissing the confirmation starts nothing", async () => {
    renderProvider();
    await act(async () => {
      fireEvent.click(screen.getByText("OPEN"));
    });
    await act(async () => {
      fireEvent.click(screen.getByText("DISMISS"));
    });

    expect(screen.queryByText("CONFIRM")).toBeNull();
    expect(tracedFetchMock).not.toHaveBeenCalled();
    expect(screen.getByTestId("downloading").textContent).toBe("false");
    expect(widget()).toBeNull();
  });
});

describe("Cancel fallbacks", () => {
  it("cancels locally with the CANCELLED sentinel when no job id exists yet", async () => {
    // POST never resolves: downloading is on, but jobId was never captured.
    postResponder = () => new Promise(() => {});
    renderProvider();

    await act(async () => {
      fireEvent.click(screen.getByText("OPEN"));
    });
    await act(async () => {
      fireEvent.click(screen.getByText("CONFIRM"));
      await vi.advanceTimersByTimeAsync(0);
    });
    await act(async () => {
      fireEvent.click(screen.getByLabelText("Expand"));
    });
    await advance(500);
    await act(async () => {
      fireEvent.click(screen.getByLabelText("Cancel"));
    });

    expect(onError).toHaveBeenCalledWith(DOWNLOAD_CANCELLED);
    expect(screen.getByTestId("downloading").textContent).toBe("false");
    // No cancel endpoint was hit — there was no job to cancel.
    expect(tracedFetchMock.mock.calls.some(([u]) => String(u).endsWith("/cancel"))).toBe(false);
  });

  it("cleans up locally with the CANCELLED sentinel when the cancel endpoint fails", async () => {
    cancelResponder = () => ({ ok: false, status: 500, statusText: "boom" });
    renderProvider();
    await startPollAndExpand();

    await act(async () => {
      fireEvent.click(screen.getByLabelText("Cancel"));
    });

    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError).toHaveBeenCalledWith(DOWNLOAD_CANCELLED);
    expect(screen.getByTestId("downloading").textContent).toBe("false");
    // Still a cancellation, with the detail kept visible.
    expect(phaseLine()).toBe("Download cancelled");
    expect(screen.getByText(/Failed to cancel download/)).toBeTruthy();
  });
});
