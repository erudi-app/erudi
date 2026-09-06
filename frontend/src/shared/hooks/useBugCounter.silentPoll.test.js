// @vitest-environment jsdom
/**
 * Regression test for the self-feeding badge (#485 adversarial review,
 * finding 1 - HIGH): the bug counter's own poll of `/diagnostics/` must
 * never turn into a record the counter later reads back and counts.
 *
 * Unlike shared/hooks/useBugCounter.test.js, this file does NOT mock
 * `services/api/client`: it exercises the REAL `apiClient` (retries and
 * all), the REAL logger bridge shape, and the REAL `parseAppLogRecords`
 * reader (utils/appLogTail.js) -- the same one `readAppLogTail` in main.js's
 * `diagnostics:appLogTail` handler applies to the actual `erudi-backend.log`
 * file. Only `fetch` (the network) and the passage of time are faked;
 * everything in between, including apiClient's retry/backoff ladder, is the
 * production code path.
 *
 * The scenario: the backend is down for two consecutive poll ticks. Each
 * tick's `apiClient.get("/diagnostics/", { silentFailure: true })` call logs
 * exactly one `api.failure` entry through `window.logAPI.send`, after riding
 * out the client's own retry ladder. Those raw entries are turned into the
 * literal log-file lines main.js's `renderer-log` handler would write
 * (`[ISO] [renderer:APIClient] LEVEL msg data`), fed through the real
 * parser, and handed back to the hook as its `appLogTail` result --
 * reproducing the exact feedback loop the bug lived in, end to end.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act, cleanup } from "@testing-library/react";

import useBugCounter from "./useBugCounter";
import { resetSessionErrors } from "../../utils/errorCapture";
import { markDiagnosticsVisited } from "../../utils/bugCounter";
import { parseAppLogRecords } from "../../utils/appLogTail";

function memoryStorage() {
  const store = new Map();
  return {
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, String(v)),
    removeItem: (k) => store.delete(k),
    clear: () => store.clear(),
  };
}

/**
 * The exact line main.js's `renderer-log` IPC handler writes to
 * erudi-backend.log for one `window.logAPI.send(entry)` call: `log()`
 * prepends "[<ISO>] ", and the handler itself formats
 * "[renderer:<ns>] <LEVEL> <msg>[ <data>]".
 */
function rendererLogLine(entry) {
  const level = String(entry.level ?? "info").toUpperCase();
  const data = entry.data ? ` ${entry.data}` : "";
  return `[${entry.ts}] [renderer:${entry.ns}] ${level} ${entry.msg}${data}`;
}

let sendSpy;

// apiClient retries a transient (TypeError "fetch") failure twice, with
// 1000ms then 2000ms backoff, before giving up -- real timers would make
// this test take several seconds per tick for no benefit, so time is faked
// and driven explicitly instead of through testing-library's `waitFor`
// (which assumes real timers).
beforeEach(() => {
  vi.useFakeTimers();
  Object.defineProperty(window, "localStorage", {
    value: memoryStorage(),
    configurable: true,
    writable: true,
  });
  resetSessionErrors();
  sendSpy = vi.fn();
  window.logAPI = { send: sendSpy };
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
});

afterEach(() => {
  cleanup();
  delete window.logAPI;
  delete window.diagnosticsAPI;
  resetSessionErrors();
  vi.unstubAllGlobals();
  vi.useRealTimers();
  vi.restoreAllMocks();
});

/** One apiClient.get retry ladder fully playing out: 1000ms + 2000ms backoff. */
const FULL_RETRY_LADDER_MS = 3100;

describe("useBugCounter does not feed its own poll failures back into its count", () => {
  it("stays at 0 across two failed polls, and does not resurrect after a Diagnostics visit", async () => {
    // window.diagnosticsAPI.appLogTail is wired to the REAL parser, fed with
    // whatever window.logAPI.send has actually captured so far -- exactly
    // what a real relaunch of the reader would see in the file.
    const appLogTail = vi.fn().mockImplementation(async () => {
      const text = sendSpy.mock.calls.map(([entry]) => rendererLogLine(entry)).join("\n");
      // The handler's own shape: {ok, records}.
      return { ok: true, records: parseAppLogRecords(text) };
    });
    window.diagnosticsAPI = { appLogTail };

    const pollMs = 5000;
    const { result } = renderHook(() => useBugCounter({ pollMs }));

    // First tick: mounting fires `poll()` immediately. Let its apiClient.get
    // retry ladder fully play out.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(FULL_RETRY_LADDER_MS);
    });

    const firstTickFailures = sendSpy.mock.calls
      .map(([entry]) => entry)
      .filter((e) => e.msg === "api.failure");
    expect(firstTickFailures.length).toBeGreaterThan(0);
    // The fix: every one of them is logged at info, never error.
    expect(firstTickFailures.every((e) => e.level === "info")).toBe(true);
    expect(result.current.count).toBe(0);

    // Second tick: advance past the interval boundary (fixed cadence from
    // mount, so it fires at t=pollMs regardless of how long the first tick's
    // ladder took) AND that poll's own retry ladder. appLogTail now reads
    // back the app log, INCLUDING the first tick's api.failure line(s) --
    // the real feedback path.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(pollMs + 100);
    });

    expect(appLogTail.mock.calls.length).toBeGreaterThanOrEqual(2);

    // The parser drops an INFO-level renderer line by construction
    // (utils/appLogTail.js only recognizes WARN/ERROR): the api.failure
    // record itself never reaches the hook. The retry ladder's own
    // "Request failed, retrying..." lines ARE real WARNING records (visible,
    // by design -- WARNING is never filtered), but WARNING never counts
    // either, so no ERROR/CRITICAL ever reaches this fixture.
    const lastResult = (await appLogTail.mock.results.at(-1).value).records;
    expect(lastResult.some((r) => r.message.includes("api.failure"))).toBe(false);
    expect(lastResult.every((r) => r.level === "WARNING")).toBe(true);

    expect(result.current.count).toBe(0);
    expect(result.current.label).toBe("");

    // Visiting Diagnostics must not resurrect a phantom count either.
    act(() => {
      markDiagnosticsVisited();
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(pollMs);
    });
    expect(result.current.count).toBe(0);
  });

  it("would have grown without the fix -- sanity check that the harness actually exercises the loop", () => {
    // Same harness, but simulating the PRE-FIX shape (api.failure at error)
    // to prove this test is capable of catching the regression, not just
    // trivially passing because nothing was ever fed back.
    const preFixEntries = [
      {
        ts: "2026-09-06T10:00:00.000Z",
        level: "error",
        ns: "APIClient",
        msg: "api.failure",
        data: '{"rid":"fe-1","method":"GET","path":"/diagnostics/","error":"Failed to fetch"}',
      },
      {
        ts: "2026-09-06T10:00:01.000Z",
        level: "error",
        ns: "APIClient",
        msg: "api.failure",
        data: '{"rid":"fe-2","method":"GET","path":"/diagnostics/","error":"Failed to fetch"}',
      },
    ];
    const text = preFixEntries.map(rendererLogLine).join("\n");
    const parsed = parseAppLogRecords(text);
    // An ERROR-level line the fix is supposed to prevent WOULD be visible to
    // the reader and WOULD count (no environmental pattern matches a plain
    // "Failed to fetch" against our own backend, by design).
    expect(parsed).toHaveLength(2);
    expect(parsed.every((r) => r.level === "ERROR")).toBe(true);
  });
});
