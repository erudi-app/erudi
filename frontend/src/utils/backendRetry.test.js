import { describe, it, expect } from "vitest";
import {
  classifyStartupError,
  DATA_DIR_RETRY_MS,
  DATA_DIR_WAIT_MS,
  shouldWaitForDataDir,
  shouldRetrySpawn,
  healthConfirmBackoffMs,
  HEALTH_CONFIRM_BACKOFF_MS,
} from "./backendRetry.js";

describe("classifyStartupError", () => {
  it("marks port-contention codes transient", () => {
    expect(classifyStartupError("NO_PORT_AVAILABLE")).toBe("transient");
    expect(classifyStartupError("PORT_TIMEOUT")).toBe("transient");
  });

  it("leaves a data folder held by the previous backend to its own wait", () => {
    // Not one of the two quick respawns: the previous backend can take
    // ~20 s to exit (uvicorn's drain, the download wait, its own stops).
    expect(classifyStartupError("DATA_DIR_IN_USE")).toBe("deterministic");
  });

  it("marks everything else deterministic", () => {
    for (const code of [
      "IMPORT_ERROR",
      "DATA_PREP_ERROR",
      "CRASH_BEFORE_READY",
      "POLLING_ERROR",
      "UNKNOWN",
    ]) {
      expect(classifyStartupError(code)).toBe("deterministic");
    }
  });
});

describe("shouldRetrySpawn", () => {
  it("respawns transient failures up to the cap", () => {
    expect(shouldRetrySpawn("NO_PORT_AVAILABLE", 0)).toBe(true);
    expect(shouldRetrySpawn("NO_PORT_AVAILABLE", 1)).toBe(true);
    expect(shouldRetrySpawn("NO_PORT_AVAILABLE", 2)).toBe(false); // cap reached
  });

  it("never respawns deterministic failures (fail fast, manual retry)", () => {
    expect(shouldRetrySpawn("IMPORT_ERROR", 0)).toBe(false);
    expect(shouldRetrySpawn("CRASH_BEFORE_READY", 0)).toBe(false);
  });
});

describe("shouldWaitForDataDir", () => {
  it("keeps respawning for about 30 s while the previous backend holds the folder", () => {
    expect(DATA_DIR_WAIT_MS).toBe(30000);
    expect(DATA_DIR_RETRY_MS).toBe(2000);
    expect(shouldWaitForDataDir("DATA_DIR_IN_USE", 0)).toBe(true);
    expect(shouldWaitForDataDir("DATA_DIR_IN_USE", 29999)).toBe(true);
    expect(shouldWaitForDataDir("DATA_DIR_IN_USE", 30000)).toBe(false);
  });

  it("never waits on any other code", () => {
    expect(shouldWaitForDataDir("NO_PORT_AVAILABLE", 0)).toBe(false);
    expect(shouldWaitForDataDir("CRASH_BEFORE_READY", 0)).toBe(false);
  });
});

describe("healthConfirmBackoffMs", () => {
  it("follows the 5s/10s/20s schedule and clamps", () => {
    expect(healthConfirmBackoffMs(0)).toBe(5000);
    expect(healthConfirmBackoffMs(1)).toBe(10000);
    expect(healthConfirmBackoffMs(2)).toBe(20000);
    expect(healthConfirmBackoffMs(9)).toBe(20000); // clamped to tail
    expect(HEALTH_CONFIRM_BACKOFF_MS).toEqual([5000, 10000, 20000]);
  });
});
