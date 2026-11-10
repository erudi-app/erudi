/**
 * The backend supervisor: respawn policy, the wait for a data folder the
 * previous backend still holds, and one loop at a time.
 *
 * Seen: a Retry during the DATA_DIR_IN_USE wait started a second loop next to
 * the first; both shared the child and the wait, and the loser killed the
 * winner mid-boot. A quit during the wait respawned the backend anyway.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

import { createBackendSupervisor, WAITING_PREVIOUS_SESSION_PHASE } from "./backendSupervisor";
import { DATA_DIR_RETRY_MS, DATA_DIR_WAIT_MS } from "./backendRetry";

let clock;
let quitting;
let starts;
let io;

/** Each start resolves or rejects as the test says, when it says. */
function deferredStarts() {
  const pending = [];
  const startBackend = vi.fn(
    () =>
      new Promise((resolve, reject) => {
        pending.push({ resolve, reject });
      })
  );
  return { startBackend, pending };
}

function supervisor() {
  return createBackendSupervisor({
    startBackend: starts.startBackend,
    stopBackend: io.stopBackend,
    sleep: (ms) => {
      clock += ms;
      return Promise.resolve();
    },
    now: () => clock,
    isQuitting: () => quitting,
    log: io.log,
    logWarn: io.logWarn,
    logError: io.logError,
    onReady: io.onReady,
    onWaiting: io.onWaiting,
    onFailed: io.onFailed,
  });
}

const flush = () => new Promise((r) => setTimeout(r, 0));

beforeEach(() => {
  clock = 0;
  quitting = false;
  starts = deferredStarts();
  io = {
    stopBackend: vi.fn(),
    log: vi.fn(),
    logWarn: vi.fn(),
    logError: vi.fn(),
    onReady: vi.fn(),
    onWaiting: vi.fn(),
    onFailed: vi.fn(),
  };
});

describe("the wait for a data folder the previous backend still holds", () => {
  it("respawns every 2 s, tells the window it is waiting, and comes up without an error", async () => {
    const run = supervisor().start();
    for (let i = 0; i < 3; i++) {
      await flush();
      starts.pending[i].reject(new Error("DATA_DIR_IN_USE"));
    }
    await flush();
    starts.pending[3].resolve({ port: 27183 });
    await run;

    expect(starts.startBackend).toHaveBeenCalledTimes(4);
    expect(clock).toBe(3 * DATA_DIR_RETRY_MS);
    expect(io.onWaiting).toHaveBeenCalledTimes(3);
    expect(io.onReady).toHaveBeenCalledWith(27183);
    expect(io.onFailed).not.toHaveBeenCalled();
    expect(io.logError).not.toHaveBeenCalled();
    expect(WAITING_PREVIOUS_SESSION_PHASE).toBe("waiting_previous_session");
  });

  it("surfaces the error with one ERROR once the 30 s wait runs out", async () => {
    const run = supervisor().start();
    for (let i = 0; i <= DATA_DIR_WAIT_MS / DATA_DIR_RETRY_MS; i++) {
      await flush();
      starts.pending[i].reject(new Error("DATA_DIR_IN_USE"));
    }
    await run;

    expect(io.onFailed).toHaveBeenCalledTimes(1);
    expect(io.onFailed.mock.calls[0][0]).toBe("DATA_DIR_IN_USE");
    expect(io.logError).toHaveBeenCalledTimes(1);
  });

  it("never respawns once the app is quitting", async () => {
    const run = supervisor().start();
    await flush();
    quitting = true;
    starts.pending[0].reject(new Error("DATA_DIR_IN_USE"));
    await run;

    expect(starts.startBackend).toHaveBeenCalledTimes(1);
    expect(io.onFailed).not.toHaveBeenCalled();
  });
});

describe("one loop at a time", () => {
  it("lets a Retry during the wait supersede the running loop", async () => {
    const sup = supervisor();
    const first = sup.start();
    await flush();
    // The user presses Retry while the first loop waits on its child.
    const second = sup.start();
    await flush();
    // The first loop's child fails: that loop must stand down, not respawn.
    starts.pending[0].reject(new Error("DATA_DIR_IN_USE"));
    await first;
    expect(starts.startBackend).toHaveBeenCalledTimes(2);
    expect(io.stopBackend).not.toHaveBeenCalled();

    starts.pending[1].resolve({ port: 27182 });
    await second;
    expect(io.onReady).toHaveBeenCalledTimes(1);
    expect(io.onReady).toHaveBeenCalledWith(27182);
    expect(io.onFailed).not.toHaveBeenCalled();
  });

  it("ignores a superseded loop's success", async () => {
    const sup = supervisor();
    const first = sup.start();
    await flush();
    sup.start();
    starts.pending[0].resolve({ port: 27190 });
    await first;
    expect(io.onReady).not.toHaveBeenCalled();
  });
});

describe("other failures", () => {
  it("respawns a transient code twice, then surfaces it", async () => {
    const run = supervisor().start();
    for (let i = 0; i < 3; i++) {
      await flush();
      starts.pending[i].reject(new Error("NO_PORT_AVAILABLE"));
    }
    await run;
    expect(starts.startBackend).toHaveBeenCalledTimes(3);
    expect(io.onFailed).toHaveBeenCalledTimes(1);
    expect(io.logError).not.toHaveBeenCalled();
  });

  it("surfaces a deterministic code at once", async () => {
    const run = supervisor().start();
    await flush();
    starts.pending[0].reject(new Error("IMPORT_ERROR"));
    await run;
    expect(starts.startBackend).toHaveBeenCalledTimes(1);
    expect(io.onFailed.mock.calls[0][0]).toBe("IMPORT_ERROR");
  });
});
