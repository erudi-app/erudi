import { describe, it, expect } from "vitest";
import { describeBackendError, isStartupError, isBackendReady } from "./backendStatus";

describe("describeBackendError", () => {
  it("maps every known backend error code to a user-facing title/detail/hint", () => {
    const codes = [
      "PORT_IN_USE",
      "NO_PORT_AVAILABLE",
      "PORT_TIMEOUT",
      "CRASH_BEFORE_READY",
      "IMPORT_ERROR",
      "DATA_PREP_ERROR",
      "DATABASE_ERROR",
      "CUDA_RUNTIME_ERROR",
      "BACKEND_STARTUP_FAILED",
      "BACKEND_EXIT_ERROR",
      "BACKEND_SPAWN_FAILED",
      "BACKEND_UNREACHABLE",
    ];
    for (const code of codes) {
      const d = describeBackendError({ event: "startup_error", code });
      expect(d.code).toBe(code);
      expect(d.title, code).toBeTruthy();
      expect(d.detail, code).toBeTruthy();
      expect(d.hint, code).toBeTruthy();
    }
  });

  it("explains a data folder held by another Erudi backend in its own words", () => {
    const d = describeBackendError({ event: "startup_error", code: "DATA_DIR_IN_USE" });
    const unknown = describeBackendError({ event: "startup_error", code: "WAT_IS_THIS" });
    expect(d.title).toBeTruthy();
    expect(d.title).not.toBe(unknown.title);
    expect(d.hint).not.toBe(unknown.hint);
  });

  it("falls back to a generic descriptor for an unknown code", () => {
    const d = describeBackendError({ event: "startup_error", code: "WAT_IS_THIS" });
    expect(d.code).toBe("WAT_IS_THIS");
    expect(d.title).toBeTruthy();
    expect(d.detail).toBeTruthy();
    expect(d.hint).toBeTruthy();
  });

  it("carries the raw backend message through when present", () => {
    const d = describeBackendError({ code: "IMPORT_ERROR", message: "No module named 'foo'" });
    expect(d.raw).toBe("No module named 'foo'");
  });

  it("tolerates a missing/blank message and a null/undefined event", () => {
    expect(describeBackendError({ code: "PORT_IN_USE" }).raw).toBeNull();
    expect(describeBackendError({ code: "PORT_IN_USE", message: "   " }).raw).toBeNull();
    expect(describeBackendError(null).title).toBeTruthy();
    expect(describeBackendError(undefined).code).toBe("UNKNOWN");
  });
});

describe("event classifiers", () => {
  it("isStartupError is true only for startup_error events", () => {
    expect(isStartupError({ event: "startup_error", code: "PORT_IN_USE" })).toBe(true);
    expect(isStartupError({ event: "ready" })).toBe(false);
    expect(isStartupError(null)).toBe(false);
  });

  it("isBackendReady is true for ready / backend_ready events", () => {
    expect(isBackendReady({ event: "ready", port: 8765 })).toBe(true);
    expect(isBackendReady({ event: "backend_ready" })).toBe(true);
    expect(isBackendReady({ event: "starting" })).toBe(false);
    expect(isBackendReady(null)).toBe(false);
  });
});
