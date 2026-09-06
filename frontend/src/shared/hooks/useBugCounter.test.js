// @vitest-environment jsdom
/**
 * The bug-icon counter hook (#485): wires the three Diagnostics sources
 * (backend, app log, session buffer) and the last-visit marker into a single
 * live count, polling gently for the first two and updating immediately for
 * the other two.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor, act, cleanup } from "@testing-library/react";

const { getMock } = vi.hoisted(() => ({ getMock: vi.fn() }));

// Mocked the same way the existing page tests mock it (e.g.
// pages/DiagnosticsPage.test.jsx): this hook uses the exact same
// `apiClient.get("/diagnostics/")` call DiagnosticsPanel makes, so it
// degrades the same way an unreachable backend already does there, and it
// stays hermetic under the many page tests that render the real Sidebar
// (which mounts this hook) without expecting any network activity.
vi.mock("../../services/api/client", () => ({
  default: { get: getMock },
  apiClient: { get: getMock },
}));

import useBugCounter from "./useBugCounter";
import { recordSessionError, resetSessionErrors, getSessionErrors } from "../../utils/errorCapture";
import { markDiagnosticsVisited, LAST_VISIT_STORAGE_KEY } from "../../utils/bugCounter";

function memoryStorage(initial = {}) {
  const store = new Map(Object.entries(initial));
  return {
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, String(v)),
    removeItem: (k) => store.delete(k),
    clear: () => store.clear(),
  };
}

function installStorage(value = memoryStorage()) {
  Object.defineProperty(window, "localStorage", {
    value,
    configurable: true,
    writable: true,
  });
}

let appLogTail;

beforeEach(() => {
  installStorage();
  resetSessionErrors();
  // The IPC answers {ok, records}: an unreadable log is not an empty one.
  appLogTail = vi.fn().mockResolvedValue({ ok: true, records: [] });
  window.diagnosticsAPI = { appLogTail };
  getMock.mockReset();
  getMock.mockResolvedValue({ recent_errors: [] });
});

afterEach(() => {
  cleanup();
  delete window.diagnosticsAPI;
  resetSessionErrors();
  vi.restoreAllMocks();
});

describe("useBugCounter", () => {
  it("starts at 0 with no errors recorded", async () => {
    const { result } = renderHook(() => useBugCounter({ pollMs: 20 }));
    await waitFor(() =>
      expect(getMock).toHaveBeenCalledWith("/diagnostics/", { silentFailure: true })
    );
    expect(result.current.count).toBe(0);
    expect(result.current.label).toBe("");
  });

  it("polls with silentFailure so its own failure is never logged as a countable ERROR (#485 fix)", async () => {
    renderHook(() => useBugCounter({ pollMs: 20 }));
    await waitFor(() => expect(getMock).toHaveBeenCalled());
    for (const [, options] of getMock.mock.calls) {
      expect(options).toEqual({ silentFailure: true });
    }
  });

  it("counts a real backend ERROR newer than app launch", async () => {
    const future = new Date(Date.now() + 60_000).toISOString();
    getMock.mockResolvedValue({
      recent_errors: [{ timestamp: future, level: "ERROR", request_id: null, message: "boom" }],
    });
    const { result } = renderHook(() => useBugCounter({ pollMs: 20 }));
    await waitFor(() => expect(result.current.count).toBe(1));
    expect(result.current.label).toBe("1");
  });

  it("does not count a WARNING", async () => {
    const future = new Date(Date.now() + 60_000).toISOString();
    getMock.mockResolvedValue({
      recent_errors: [
        { timestamp: future, level: "WARNING", request_id: null, message: "degraded" },
      ],
    });
    const { result } = renderHook(() => useBugCounter({ pollMs: 20 }));
    await waitFor(() => expect(getMock).toHaveBeenCalled());
    expect(result.current.count).toBe(0);
  });

  it("does not count an environmental ERROR", async () => {
    const future = new Date(Date.now() + 60_000).toISOString();
    getMock.mockResolvedValue({
      recent_errors: [
        {
          timestamp: future,
          level: "ERROR",
          request_id: null,
          message:
            "POST /erudi/llms/download -> 503 HUGGINGFACE_API_ERROR: you appear to be offline - check your connection and retry",
        },
      ],
    });
    const { result } = renderHook(() => useBugCounter({ pollMs: 20 }));
    await waitFor(() => expect(getMock).toHaveBeenCalled());
    expect(result.current.count).toBe(0);
  });

  it("reflects a session error immediately, without waiting for a poll", async () => {
    getMock.mockImplementation(() => new Promise(() => {})); // never resolves
    const { result } = renderHook(() => useBugCounter({ pollMs: 100_000 }));
    expect(result.current.count).toBe(0);

    act(() => {
      recordSessionError({ origin: "window.onerror", message: "kaboom" });
    });

    await waitFor(() => expect(result.current.count).toBe(1));
    expect(getSessionErrors()).toHaveLength(1);
  });

  it("clears immediately when the Diagnostics page is visited, before the next poll", async () => {
    const past = new Date(Date.now() - 5_000).toISOString();
    getMock.mockResolvedValue({
      recent_errors: [{ timestamp: past, level: "ERROR", request_id: null, message: "boom" }],
    });
    act(() => {
      recordSessionError({ origin: "window.onerror", message: "still fresh" });
    });
    const { result } = renderHook(() => useBugCounter({ pollMs: 100_000 }));
    await waitFor(() => expect(result.current.count).toBe(1));

    act(() => {
      markDiagnosticsVisited();
    });

    await waitFor(() => expect(result.current.count).toBe(0));
    expect(window.localStorage.getItem(LAST_VISIT_STORAGE_KEY)).not.toBeNull();
  });

  it("polls both the backend and the app log tail on an interval", async () => {
    const { unmount } = renderHook(() => useBugCounter({ pollMs: 20 }));
    await waitFor(() => expect(getMock.mock.calls.length).toBeGreaterThanOrEqual(2));
    await waitFor(() => expect(appLogTail.mock.calls.length).toBeGreaterThanOrEqual(2));
    unmount();
  });

  it("degrades to the app log alone when the backend does not answer", async () => {
    getMock.mockRejectedValue(new Error("backend unreachable"));
    const future = new Date(Date.now() + 60_000).toISOString();
    appLogTail = vi.fn().mockResolvedValue({
      ok: true,
      records: [
        { timestamp: future, level: "ERROR", message: "renderer:uncaught something broke" },
      ],
    });
    window.diagnosticsAPI = { appLogTail };
    const { result } = renderHook(() => useBugCounter({ pollMs: 20 }));
    await waitFor(() => expect(result.current.count).toBe(1));
  });

  it("keeps the count it had when the app log cannot be read", async () => {
    // Dropping to zero would say the errors went away; the Diagnostics page
    // is where an unreadable log is reported, and a badge must not invent
    // good news on its own.
    const future = new Date(Date.now() + 60_000).toISOString();
    const record = {
      timestamp: future,
      level: "ERROR",
      message: "renderer:uncaught something broke",
    };
    appLogTail = vi
      .fn()
      .mockResolvedValueOnce({ ok: true, records: [record] })
      .mockResolvedValue({ ok: false, records: [], reason: "EACCES" });
    window.diagnosticsAPI = { appLogTail };

    const { result } = renderHook(() => useBugCounter({ pollMs: 20 }));

    await waitFor(() => expect(result.current.count).toBe(1));
    await waitFor(() => expect(appLogTail.mock.calls.length).toBeGreaterThanOrEqual(3));
    expect(result.current.count).toBe(1);
  });

  it("stops polling after unmount", async () => {
    const { unmount } = renderHook(() => useBugCounter({ pollMs: 20 }));
    await waitFor(() => expect(getMock).toHaveBeenCalled());
    unmount();
    const callsAtUnmount = getMock.mock.calls.length;
    await new Promise((resolve) => setTimeout(resolve, 60));
    expect(getMock.mock.calls.length).toBe(callsAtUnmount);
  });
});
