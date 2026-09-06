// @vitest-environment jsdom
/**
 * Global capture of uncaught renderer errors.
 *
 * The risk this module exists to manage is not "a bug goes unlogged" — it is
 * the capture mechanism itself misbehaving. Three failure modes are tested
 * explicitly, because each turns a single defect into an unusable app:
 *
 *   - logging from inside the handler throws, and the throw reaches the
 *     handler again (recursion);
 *   - a failing poll or a render loop repeats the same error and floods the
 *     log file with thousands of identical lines;
 *   - the preload bridge is missing (a browser, a test, an early boot) and the
 *     handler throws on `window.logAPI.send`.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

import {
  installGlobalErrorCapture,
  recordSessionError,
  getSessionErrors,
  resetSessionErrors,
  subscribeSessionErrors,
  SESSION_ERROR_CAP,
} from "./errorCapture";

let uninstall = null;

beforeEach(() => {
  resetSessionErrors();
  window.logAPI = { send: vi.fn() };
});

afterEach(() => {
  if (uninstall) uninstall();
  uninstall = null;
  delete window.logAPI;
  vi.restoreAllMocks();
});

describe("recordSessionError", () => {
  it("keeps an entry with its message, stack and origin", () => {
    recordSessionError({ origin: "window.onerror", message: "boom", stack: "at x" });
    const [entry] = getSessionErrors();
    expect(entry.message).toBe("boom");
    expect(entry.stack).toBe("at x");
    expect(entry.origin).toBe("window.onerror");
    expect(entry.count).toBe(1);
    expect(typeof entry.timestamp).toBe("string");
  });

  it("counts a repeat instead of adding a second entry", () => {
    for (let i = 0; i < 500; i += 1) {
      recordSessionError({ origin: "window.onerror", message: "boom", stack: "at x" });
    }
    const errors = getSessionErrors();
    expect(errors).toHaveLength(1);
    expect(errors[0].count).toBe(500);
  });

  it("treats a different message as a different error", () => {
    recordSessionError({ origin: "window.onerror", message: "a" });
    recordSessionError({ origin: "window.onerror", message: "b" });
    expect(getSessionErrors()).toHaveLength(2);
  });

  it("caps the buffer and keeps the newest entries", () => {
    for (let i = 0; i < SESSION_ERROR_CAP + 20; i += 1) {
      recordSessionError({ origin: "window.onerror", message: `e${i}` });
    }
    const errors = getSessionErrors();
    expect(errors).toHaveLength(SESSION_ERROR_CAP);
    expect(errors[errors.length - 1].message).toBe(`e${SESSION_ERROR_CAP + 19}`);
  });

  it("never throws when the bridge is missing", () => {
    delete window.logAPI;
    expect(() => recordSessionError({ origin: "x", message: "boom" })).not.toThrow();
    expect(getSessionErrors()).toHaveLength(1);
  });

  it("never throws when the bridge itself throws", () => {
    window.logAPI = {
      send: () => {
        throw new Error("ipc is gone");
      },
    };
    expect(() => recordSessionError({ origin: "x", message: "boom" })).not.toThrow();
    expect(getSessionErrors()).toHaveLength(1);
  });
});

describe("installGlobalErrorCapture", () => {
  it("logs an uncaught error once through the bridge", () => {
    uninstall = installGlobalErrorCapture();
    window.dispatchEvent(
      new ErrorEvent("error", { message: "kaboom", error: new Error("kaboom") })
    );
    expect(window.logAPI.send).toHaveBeenCalledTimes(1);
    const [entry] = window.logAPI.send.mock.calls[0];
    expect(entry.ns).toBe("renderer:uncaught");
    expect(entry.level).toBe("error");
    expect(entry.msg).toContain("kaboom");
    expect(getSessionErrors()).toHaveLength(1);
  });

  it("logs a repeat of the same error only once and counts the rest", () => {
    uninstall = installGlobalErrorCapture();
    for (let i = 0; i < 1000; i += 1) {
      window.dispatchEvent(
        new ErrorEvent("error", { message: "kaboom", error: new Error("kaboom") })
      );
    }
    // One log line, not a thousand: a render loop must not fill the log file.
    expect(window.logAPI.send).toHaveBeenCalledTimes(1);
    expect(getSessionErrors()[0].count).toBe(1000);
  });

  it("captures an unhandled promise rejection", () => {
    uninstall = installGlobalErrorCapture();
    const event = new Event("unhandledrejection");
    event.reason = new Error("nope");
    window.dispatchEvent(event);
    expect(getSessionErrors()).toHaveLength(1);
    expect(getSessionErrors()[0].origin).toBe("unhandledrejection");
    expect(getSessionErrors()[0].message).toContain("nope");
  });

  it("does not recurse when the bridge throws inside the handler", () => {
    let calls = 0;
    window.logAPI = {
      send: () => {
        calls += 1;
        throw new Error("bridge exploded");
      },
    };
    uninstall = installGlobalErrorCapture();
    expect(() =>
      window.dispatchEvent(new ErrorEvent("error", { message: "kaboom" }))
    ).not.toThrow();
    expect(calls).toBe(1);
  });

  it("installs once even when called twice, and removes its listeners", () => {
    uninstall = installGlobalErrorCapture();
    // A second call must not double-log, and must not strand the listeners
    // either: it hands back the same uninstaller.
    const second = installGlobalErrorCapture();
    expect(second).toBe(uninstall);
    window.dispatchEvent(new ErrorEvent("error", { message: "kaboom" }));
    expect(window.logAPI.send).toHaveBeenCalledTimes(1);

    second();
    uninstall = null;
    window.logAPI.send.mockClear();
    resetSessionErrors();
    window.dispatchEvent(new ErrorEvent("error", { message: "kaboom again" }));
    expect(window.logAPI.send).not.toHaveBeenCalled();
    expect(getSessionErrors()).toHaveLength(0);
  });
});

describe("subscribeSessionErrors", () => {
  it("notifies a subscriber when a new error is recorded", () => {
    const listener = vi.fn();
    const unsubscribe = subscribeSessionErrors(listener);
    recordSessionError({ origin: "x", message: "boom" });
    expect(listener).toHaveBeenCalledTimes(1);
    unsubscribe();
  });

  it("notifies a subscriber on a repeat too, without adding a second entry", () => {
    const listener = vi.fn();
    const unsubscribe = subscribeSessionErrors(listener);
    recordSessionError({ origin: "x", message: "boom" });
    recordSessionError({ origin: "x", message: "boom" });
    expect(listener).toHaveBeenCalledTimes(2);
    expect(getSessionErrors()).toHaveLength(1);
    unsubscribe();
  });

  it("notifies a subscriber when the buffer is reset", () => {
    recordSessionError({ origin: "x", message: "boom" });
    const listener = vi.fn();
    const unsubscribe = subscribeSessionErrors(listener);
    resetSessionErrors();
    expect(listener).toHaveBeenCalledTimes(1);
    unsubscribe();
  });

  it("stops notifying once unsubscribed", () => {
    const listener = vi.fn();
    const unsubscribe = subscribeSessionErrors(listener);
    unsubscribe();
    recordSessionError({ origin: "x", message: "boom" });
    expect(listener).not.toHaveBeenCalled();
  });

  it("never throws, and does not break capture, when a listener itself throws", () => {
    const unsubscribe = subscribeSessionErrors(() => {
      throw new Error("listener exploded");
    });
    expect(() => recordSessionError({ origin: "x", message: "boom" })).not.toThrow();
    expect(getSessionErrors()).toHaveLength(1);
    unsubscribe();
  });
});
