// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, act } from "@testing-library/react";

// The download modal owns the full download lifecycle and had zero tests. These
// drive it through its public surface (open -> confirm -> 2s poll) and assert the
// observable results a consumer relies on: the completion counter/timestamp
// (#205), the confirmed callback fan-out, and — critically — the cancel-vs-fail
// split (#133), where a cancelled job must reach onError with the CANCELLED
// sentinel and must NOT bump the completion counter or look like a failure.
//
// tracedFetch is mocked; fake timers drive the setInterval poll deterministically.

const { tracedFetchMock } = vi.hoisted(() => ({ tracedFetchMock: vi.fn() }));

vi.mock("../services/api/client", () => ({
  default: { get: vi.fn() },
  apiClient: { get: vi.fn() },
  tracedFetch: tracedFetchMock,
}));

// Child UI is not under test — expose the two lifecycle triggers as plain buttons.
vi.mock("../components/modals/ConfirmationModal", () => ({
  default: ({ onConfirm, onCancel }) => (
    <div>
      <button onClick={onConfirm}>CONFIRM</button>
      <button onClick={onCancel}>DISMISS</button>
    </div>
  ),
}));

import { DownloadModalProvider, useDownloadModal } from "./DownloadModalContext";
import { DOWNLOAD_CANCELLED, DOWNLOAD_STALLED } from "../utils/downloadStatus";

// Per-test configurable responders for the POST-start / status-poll / cancel URLs.
let postResponder;
let statusResponder;
let cancelResponder;

const onComplete = vi.fn();
const onError = vi.fn();
let currentModel;

// The latest context value, for tests that await a call (cancelDownload).
let seenContext;

function Consumer() {
  const ctx = useDownloadModal();
  React.useEffect(() => {
    seenContext = ctx;
  });
  return (
    <>
      <button onClick={() => ctx.open(currentModel, { onComplete, onError })}>OPEN</button>
      <button onClick={() => ctx.cancelDownload()}>CANCEL_FROM_APP</button>
      <span data-testid="downloading">{String(ctx.isDownloading)}</span>
      <span data-testid="completions">{String(ctx.completionCount)}</span>
      <span data-testid="lastCompleted">{ctx.lastCompletedAt === null ? "null" : "set"}</span>
    </>
  );
}

const renderProvider = () =>
  render(
    <DownloadModalProvider>
      <Consumer />
    </DownloadModalProvider>
  );

const val = (id) => screen.getByTestId(id).textContent;

// open -> confirm -> let the POST resolve and fire exactly one 2s poll tick.
const startAndPollOnce = async () => {
  await act(async () => {
    fireEvent.click(screen.getByText("OPEN"));
  });
  await act(async () => {
    fireEvent.click(screen.getByText("CONFIRM"));
    await vi.advanceTimersByTimeAsync(2000);
  });
};

beforeEach(() => {
  vi.useFakeTimers();
  const root = document.createElement("div");
  root.setAttribute("id", "modal-root");
  document.body.appendChild(root);

  onComplete.mockReset();
  onError.mockReset();
  currentModel = { id: 1, name: "Base Model" };

  postResponder = () => ({ ok: true, json: async () => ({ id: "job1" }) });
  statusResponder = () => ({
    ok: true,
    json: async () => ({ status: "completed", progress: 100, time_left: 0 }),
  });
  cancelResponder = () => ({ ok: true });

  tracedFetchMock.mockReset();
  tracedFetchMock.mockImplementation(async (url, opts = {}) => {
    const u = String(url);
    const method = opts.method || "GET";
    if (method === "POST" && (u.endsWith("/download") || u.includes("/download/huggingface"))) {
      return postResponder();
    }
    if (u.includes("/downloads/") && u.endsWith("/cancel")) {
      return cancelResponder();
    }
    if (u.includes("/downloads/") && u.endsWith("/status")) {
      return statusResponder();
    }
    return { ok: true, json: async () => ({}) };
  });
});

afterEach(() => {
  cleanup();
  document.getElementById("modal-root")?.remove();
  vi.runOnlyPendingTimers();
  vi.useRealTimers();
});

const statusCalls = () =>
  tracedFetchMock.mock.calls.filter(
    ([u]) => String(u).includes("/downloads/") && String(u).endsWith("/status")
  );

describe("DownloadModalContext completion (#205)", () => {
  it("bumps the completion counter and timestamp and fires onComplete once", async () => {
    renderProvider();
    expect(val("completions")).toBe("0");
    expect(val("lastCompleted")).toBe("null");

    await startAndPollOnce();

    expect(val("completions")).toBe("1");
    expect(val("lastCompleted")).toBe("set");
    expect(val("downloading")).toBe("false");
    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(onError).not.toHaveBeenCalled();
  });

  it("downloads a catalog model by numeric id", async () => {
    renderProvider();
    await startAndPollOnce();

    const post = tracedFetchMock.mock.calls.find(([, o]) => o?.method === "POST");
    expect(String(post[0])).toMatch(/\/llms\/1\/download$/);
  });

  it("downloads a HuggingFace search hit by link, preserving an unmeasured size as null", async () => {
    currentModel = {
      name: "HF Hit",
      link: "org/repo",
      type: "chat",
      param_size: null,
      quantized: true,
      category: "general",
    };
    renderProvider();
    await startAndPollOnce();

    const post = tracedFetchMock.mock.calls.find(
      ([u, o]) => o?.method === "POST" && String(u).includes("/llms/download/huggingface")
    );
    expect(post).toBeTruthy();
    const body = JSON.parse(post[1].body);
    expect(body.link).toBe("org/repo");
    expect(body.param_size).toBeNull(); // not laundered into a plausible default (#201)
  });
});

describe("DownloadModalContext failure and cancel routing", () => {
  it("routes a failed job to onError with the backend message and does not count it", async () => {
    statusResponder = () => ({
      ok: true,
      json: async () => ({ status: "failed", error_message: "disk full", progress: 0 }),
    });
    renderProvider();

    await startAndPollOnce();

    expect(onError).toHaveBeenCalledWith("disk full");
    expect(onComplete).not.toHaveBeenCalled();
    expect(val("completions")).toBe("0");
    expect(val("downloading")).toBe("false");
  });

  it("routes a cancelled job to onError with the CANCELLED sentinel, not a failure (#133)", async () => {
    statusResponder = () => ({ ok: true, json: async () => ({ status: DOWNLOAD_CANCELLED }) });
    renderProvider();

    await startAndPollOnce();

    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError).toHaveBeenCalledWith(DOWNLOAD_CANCELLED);
    expect(onComplete).not.toHaveBeenCalled();
    expect(val("completions")).toBe("0");
  });

  it("stops silently on a 404 (job already cleaned up) without firing any callback", async () => {
    statusResponder = () => ({ ok: false, status: 404, statusText: "Not Found" });
    renderProvider();

    await startAndPollOnce();

    expect(onComplete).not.toHaveBeenCalled();
    expect(onError).not.toHaveBeenCalled();
    expect(val("downloading")).toBe("false");
  });

  it("surfaces a poll network error through onError and stops downloading", async () => {
    statusResponder = () => {
      throw new TypeError("Failed to fetch");
    };
    renderProvider();

    await startAndPollOnce();

    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError.mock.calls[0][0]).toContain("An error occured during download");
    expect(val("downloading")).toBe("false");
  });

  it("reports a failed download START through onError and never begins polling", async () => {
    postResponder = () => ({ ok: false, status: 500, text: async () => "gone" });
    renderProvider();

    await startAndPollOnce();

    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError.mock.calls[0][0]).toContain("Failed to start download (500)");
    expect(val("downloading")).toBe("false");
    expect(statusCalls()).toHaveLength(0); // no interval was ever armed
  });
});

describe("DownloadModalContext user cancel", () => {
  it("POSTs the cancel endpoint for the running job when the user cancels", async () => {
    // Keep the job 'running' so the poll never completes it out from under us.
    statusResponder = () => ({
      ok: true,
      json: async () => ({ status: "running", progress: 10, time_left: 60 }),
    });
    renderProvider();

    // Start; jobId is captured from the POST response.
    await act(async () => {
      fireEvent.click(screen.getByText("OPEN"));
    });
    await act(async () => {
      fireEvent.click(screen.getByText("CONFIRM"));
      await vi.advanceTimersByTimeAsync(0); // flush POST -> setJobId, no poll yet
    });

    // Reveal the expanded bar (default is collapsed) to expose the Cancel control.
    await act(async () => {
      fireEvent.click(screen.getByLabelText("Expand"));
    });
    await act(async () => {
      fireEvent.click(screen.getByLabelText("Cancel"));
    });

    const cancel = tracedFetchMock.mock.calls.find(
      ([u, o]) => o?.method === "POST" && String(u).endsWith("/cancel")
    );
    expect(cancel).toBeTruthy();
    expect(String(cancel[0])).toMatch(/\/llms\/downloads\/job1\/cancel$/);
  });

  // Once the backend acknowledges the cancel the job is gone: any further
  // status poll can only be answered with a not-found warning in the backend
  // log. The widget must settle locally on the acknowledgement instead.
  it("stops polling the status as soon as the cancel is acknowledged", async () => {
    statusResponder = () => ({
      ok: true,
      json: async () => ({ status: "running", progress: 10, time_left: 60 }),
    });
    renderProvider();

    await act(async () => {
      fireEvent.click(screen.getByText("OPEN"));
    });
    await act(async () => {
      fireEvent.click(screen.getByText("CONFIRM"));
      await vi.advanceTimersByTimeAsync(2000); // one poll while running
    });
    expect(statusCalls().length).toBe(1);

    // The bar auto-expands after 2s, so the Cancel control is already exposed.
    await act(async () => {
      fireEvent.click(screen.getByLabelText("Cancel"));
      await vi.advanceTimersByTimeAsync(0); // cancel POST acknowledged
    });

    await act(async () => {
      await vi.advanceTimersByTimeAsync(3 * 2000);
    });
    expect(statusCalls().length).toBe(1);
    expect(val("downloading")).toBe("false");
    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError).toHaveBeenCalledWith(DOWNLOAD_CANCELLED);
    expect(onComplete).not.toHaveBeenCalled();
  });
});

describe("DownloadModalContext cancel refused while the download completes", () => {
  it("shows no error and lets the status conclude it as installed", async () => {
    let status = "running";
    statusResponder = () => ({
      ok: true,
      json: async () => ({ status, progress: status === "running" ? 99 : 100, time_left: 0 }),
    });
    // The download is finalizing: the backend refuses the cancel (409).
    cancelResponder = () => {
      status = "completed";
      return { ok: false, status: 409, statusText: "Conflict" };
    };
    renderProvider();

    await act(async () => {
      fireEvent.click(screen.getByText("OPEN"));
    });
    await act(async () => {
      fireEvent.click(screen.getByText("CONFIRM"));
      await vi.advanceTimersByTimeAsync(2000);
    });
    await act(async () => {
      fireEvent.click(screen.getByLabelText("Cancel"));
      await vi.advanceTimersByTimeAsync(0);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });

    expect(onError).not.toHaveBeenCalled();
    expect(onComplete).toHaveBeenCalledTimes(1);
  });

  it("resolves the cancel only once the completing download has ended", async () => {
    // The engine switch awaits the cancel and then restarts the backend: a
    // refused cancel must not hand back while the download still finalizes.
    let status = "running";
    statusResponder = () => ({
      ok: true,
      json: async () => ({ status, progress: 100, time_left: 0 }),
    });
    cancelResponder = () => ({ ok: false, status: 409, statusText: "Conflict" });
    renderProvider();

    await act(async () => {
      fireEvent.click(screen.getByText("OPEN"));
    });
    await act(async () => {
      fireEvent.click(screen.getByText("CONFIRM"));
      await vi.advanceTimersByTimeAsync(0);
    });
    let resolved = false;
    let pending;
    await act(async () => {
      pending = seenContext.cancelDownload().then(() => {
        resolved = true;
      });
      await vi.advanceTimersByTimeAsync(5000);
    });
    expect(resolved).toBe(false);

    status = "completed";
    await act(async () => {
      await vi.advanceTimersByTimeAsync(3000);
      await pending;
    });
    expect(resolved).toBe(true);
    expect(onComplete).toHaveBeenCalledTimes(1);
  });
});

describe("DownloadModalContext wait after a refused cancel", () => {
  const startAndRefuse = async () => {
    statusResponder = () => ({
      ok: true,
      json: async () => ({ status: "running", progress: 100, time_left: 0 }),
    });
    cancelResponder = () => ({ ok: false, status: 409, statusText: "Conflict" });
    renderProvider();
    await act(async () => {
      fireEvent.click(screen.getByText("OPEN"));
    });
    await act(async () => {
      fireEvent.click(screen.getByText("CONFIRM"));
      await vi.advanceTimersByTimeAsync(0);
    });
  };

  it("owns the polling while it waits and says it is waiting", async () => {
    await startAndRefuse();
    let pending;
    await act(async () => {
      pending = seenContext.cancelDownload();
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(seenContext.isCompleting).toBe(true);
    const before = statusCalls().length;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10000);
    });
    // One poll every 2 s, not the interval's on top of the wait's.
    expect(statusCalls().length - before).toBeLessThanOrEqual(5);

    statusResponder = () => ({
      ok: true,
      json: async () => ({ status: "completed", progress: 100, time_left: 0 }),
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
      await pending;
    });
    expect(seenContext.isCompleting).toBe(false);
  });

  it("ends at the first failed poll with one error", async () => {
    await startAndRefuse();
    let pending;
    await act(async () => {
      pending = seenContext.cancelDownload();
      await vi.advanceTimersByTimeAsync(0);
    });
    // The backend is gone: every poll now fails.
    statusResponder = () => {
      throw new Error("connection refused");
    };
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10000);
      await pending;
    });
    expect(onError).toHaveBeenCalledTimes(1);
    expect(seenContext.isCompleting).toBe(false);
  });
});

describe("DownloadModalContext cancel concluded by a poll first", () => {
  it("reports the cancel once when a poll sees it before the cancel answers", async () => {
    // The cancel commits at once and then waits for the transfer to stop:
    // a status poll meanwhile already reads `cancelled`.
    let status = "running";
    statusResponder = () => ({
      ok: true,
      json: async () => ({ status, progress: 10, time_left: 60 }),
    });
    let answerCancel;
    cancelResponder = () => {
      status = "cancelled";
      return new Promise((resolve) => {
        answerCancel = () => resolve({ ok: true });
      });
    };
    renderProvider();

    await act(async () => {
      fireEvent.click(screen.getByText("OPEN"));
    });
    await act(async () => {
      fireEvent.click(screen.getByText("CONFIRM"));
      await vi.advanceTimersByTimeAsync(2000);
    });
    let pending;
    await act(async () => {
      pending = seenContext.cancelDownload();
      await vi.advanceTimersByTimeAsync(2000); // a poll reads cancelled
    });
    await act(async () => {
      answerCancel();
      await pending;
    });

    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError).toHaveBeenCalledWith(DOWNLOAD_CANCELLED);
  });
});

// Switching the inference engine restarts the backend, which would kill a
// running download and leave it "interrupted" at the next boot. The engine
// switch cancels it first, through the same path as the widget's Cancel.
describe("DownloadModalContext cancel from elsewhere in the app", () => {
  it("cancels the running job through the endpoint and settles as cancelled", async () => {
    statusResponder = () => ({
      ok: true,
      json: async () => ({ status: "running", progress: 10, time_left: 60 }),
    });
    renderProvider();
    await startAndPollOnce();
    expect(val("downloading")).toBe("true");

    await act(async () => {
      fireEvent.click(screen.getByText("CANCEL_FROM_APP"));
      await vi.advanceTimersByTimeAsync(0);
    });

    const cancel = tracedFetchMock.mock.calls.find(
      ([u, o]) => o?.method === "POST" && String(u).endsWith("/cancel")
    );
    expect(String(cancel[0])).toMatch(/\/llms\/downloads\/job1\/cancel$/);
    expect(val("downloading")).toBe("false");
    expect(screen.getByText("Download cancelled")).toBeTruthy();
    expect(onError).toHaveBeenCalledWith(DOWNLOAD_CANCELLED);
  });

  it("does not turn a status poll already in flight into a failure", async () => {
    // The backend goes down right after the cancel (the engine restart): the
    // poll that was on the wire fails, and must be ignored, not reported.
    const errors = vi.spyOn(console, "error").mockImplementation(() => {});
    let failPoll;
    renderProvider();
    statusResponder = () => ({
      ok: true,
      json: async () => ({ status: "running", progress: 10, time_left: 60 }),
    });
    await startAndPollOnce();
    statusResponder = () =>
      new Promise((_resolve, reject) => {
        failPoll = () => reject(new TypeError("Failed to fetch"));
      });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000); // a second poll, left pending
    });

    await act(async () => {
      fireEvent.click(screen.getByText("CANCEL_FROM_APP"));
      await vi.advanceTimersByTimeAsync(0);
    });
    await act(async () => {
      failPoll();
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(screen.getByText("Download cancelled")).toBeTruthy();
    expect(screen.queryByText(/An error occured during download/)).toBeNull();
    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError).toHaveBeenCalledWith(DOWNLOAD_CANCELLED);
    expect(errors.mock.calls.map((c) => String(c[0])).join("\n")).not.toMatch(/Status check/);
    errors.mockRestore();
  });
});

describe("DownloadModalContext without a provider", () => {
  it("reports no download and offers a harmless cancel", async () => {
    const seen = {};
    function Probe() {
      Object.assign(seen, useDownloadModal());
      return null;
    }
    render(<Probe />);
    expect(seen.isDownloading).toBe(false);
    await expect(seen.cancelDownload()).resolves.toBeUndefined();
    // Starting a download needs the provider: no silent no-op stands in.
    expect(seen.open).toBeUndefined();
  });
});

describe("DownloadModalContext cancel while the download is starting", () => {
  it("cancels the job on the backend once its start answers, and never polls it", async () => {
    // The engine switch can cancel before the start POST has returned a job
    // id. Settling locally alone would let the POST start a poll against the
    // backend the switch is about to restart.
    let answerStart;
    postResponder = () =>
      new Promise((resolve) => {
        answerStart = () => resolve({ ok: true, json: async () => ({ id: "job1" }) });
      });
    renderProvider();
    await act(async () => {
      fireEvent.click(screen.getByText("OPEN"));
    });
    await act(async () => {
      fireEvent.click(screen.getByText("CONFIRM"));
      await vi.advanceTimersByTimeAsync(0);
    });

    let cancelled = false;
    let pendingCancel;
    await act(async () => {
      pendingCancel = seenContext.cancelDownload().then(() => {
        cancelled = true;
      });
      await vi.advanceTimersByTimeAsync(0);
    });
    // Not done yet: the job the backend is creating still has to be cancelled.
    expect(cancelled).toBe(false);

    await act(async () => {
      answerStart();
      await pendingCancel;
      await vi.advanceTimersByTimeAsync(3 * 2000);
    });

    const cancel = tracedFetchMock.mock.calls.find(
      ([u, o]) => o?.method === "POST" && String(u).endsWith("/cancel")
    );
    expect(String(cancel?.[0])).toMatch(/\/llms\/downloads\/job1\/cancel$/);
    expect(statusCalls()).toHaveLength(0);
    expect(cancelled).toBe(true);
    expect(val("downloading")).toBe("false");
    expect(screen.getByText("Download cancelled")).toBeTruthy();
  });
});

describe("DownloadModalContext cancel of a download that has just finished", () => {
  it("lets the job conclude as completed instead of showing a cancel", async () => {
    // Between two polls the transfer ended: the backend refuses the cancel
    // ("Download ended - cannot be cancelled", 422). The model is there; the
    // widget must say so, not "cancelled" with a cancel failure.
    let finished = false;
    statusResponder = () => ({
      ok: true,
      json: async () =>
        finished
          ? { status: "completed", progress: 100, time_left: 0 }
          : { status: "running", progress: 90, time_left: 5 },
    });
    cancelResponder = () => ({
      ok: false,
      status: 422,
      statusText: "Unprocessable Entity",
      json: async () => ({
        success: false,
        error: {
          type: "INVALID_INPUT",
          message: "Download ended - cannot be cancelled, please delete the model",
        },
      }),
    });
    const errors = vi.spyOn(console, "error").mockImplementation(() => {});
    renderProvider();
    await startAndPollOnce();

    finished = true;
    await act(async () => {
      await seenContext.cancelDownload();
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(val("completions")).toBe("1");
    expect(val("downloading")).toBe("false");
    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(onError).not.toHaveBeenCalled();
    expect(screen.queryByText("Download cancelled")).toBeNull();
    expect(screen.queryByText(/Failed to cancel download/)).toBeNull();
    errors.mockRestore();
  });
});

// The poll had no cap: it stopped only on completed/failed/cancelled, so when
// #291 stranded a job at `running` with progress 100 it ran 800+ times over 25
// minutes with the widget frozen and the sidebar dimmed (#315).
describe("DownloadModalContext stall guard (#315)", () => {
  const POLL_MS = 2000;
  const STALL_TICKS = 90;

  // open -> confirm -> drive N poll ticks.
  const startAndPoll = async (ticks) => {
    await act(async () => {
      fireEvent.click(screen.getByText("OPEN"));
    });
    await act(async () => {
      fireEvent.click(screen.getByText("CONFIRM"));
      await vi.advanceTimersByTimeAsync(ticks * POLL_MS);
    });
  };

  it("gives up after the finalize threshold and hands the app back to the user", async () => {
    statusResponder = () => ({
      ok: true,
      json: async () => ({ status: "running", progress: 100, time_left: 0 }),
    });
    renderProvider();

    await startAndPoll(STALL_TICKS);

    // The poll stopped, and the dimmed-sidebar/spinner state is released.
    expect(val("downloading")).toBe("false");
    expect(onError).toHaveBeenCalledWith(DOWNLOAD_STALLED);
    // Not a completion: nothing finished, so consumers must not reload as if it did.
    expect(onComplete).not.toHaveBeenCalled();
    expect(val("completions")).toBe("0");

    // And it stays stopped: no further requests once it has given up.
    const afterGiveUp = statusCalls().length;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(20 * POLL_MS);
    });
    expect(statusCalls().length).toBe(afterGiveUp);
  });

  it("keeps polling while still under the threshold", async () => {
    statusResponder = () => ({
      ok: true,
      json: async () => ({ status: "running", progress: 100, time_left: 0 }),
    });
    renderProvider();

    await startAndPoll(STALL_TICKS - 1);

    expect(val("downloading")).toBe("true");
    expect(onError).not.toHaveBeenCalled();
  });

  it("never caps a transfer that is merely slow", async () => {
    // Below 100% the bytes are still moving; a 70 GB pull on a bad line
    // legitimately takes hours and must never be killed by a timer.
    statusResponder = () => ({
      ok: true,
      json: async () => ({ status: "running", progress: 42, time_left: 9999 }),
    });
    renderProvider();

    await startAndPoll(STALL_TICKS * 3);

    expect(val("downloading")).toBe("true");
    expect(onError).not.toHaveBeenCalled();
    expect(statusCalls().length).toBeGreaterThan(STALL_TICKS);
  });

  it("does not fire when the job finalizes normally after touching 100%", async () => {
    let polls = 0;
    statusResponder = () => {
      polls += 1;
      return {
        ok: true,
        json: async () =>
          polls < 5
            ? { status: "running", progress: 100, time_left: 0 }
            : { status: "completed", progress: 100, time_left: 0 },
      };
    };
    renderProvider();

    await startAndPoll(10);

    expect(onError).not.toHaveBeenCalled();
    expect(onComplete).toHaveBeenCalledTimes(1);
    expect(val("completions")).toBe("1");
  });
});
