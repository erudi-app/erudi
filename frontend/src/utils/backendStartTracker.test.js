/**
 * One ERROR record per backend start failure, and no false crash on restart.
 *
 * The scenarios are the ones the log used to get wrong: a startup_error
 * followed by the exit it causes and by the supervisor's surfacing line wrote
 * three ERROR rows; a forced restart wrote a crash for the child that main
 * itself had just killed.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";

import {
  createBackendStartTracker,
  exitFailureCode,
  requestStop,
  stopRequested,
} from "./backendStartTracker";

let log;
let logError;
let onFail;
let onSucceed;

const tracker = () => createBackendStartTracker({ log, logError, onFail, onSucceed });

beforeEach(() => {
  log = vi.fn();
  logError = vi.fn();
  onFail = vi.fn();
  onSucceed = vi.fn();
});

describe("one record per start failure", () => {
  it("a startup_error then the exit it causes: one ERROR, one plain line", () => {
    const t = tracker();
    const proc = {};
    t.fail("CRASH_BEFORE_READY", "backend reported: Backend thread exited before binding");
    t.exited(proc, 1, null);
    expect(logError).toHaveBeenCalledTimes(1);
    expect(logError.mock.calls[0][0]).toBe(
      "Backend start failed (CRASH_BEFORE_READY): backend reported: Backend thread exited before binding"
    );
    expect(log).toHaveBeenCalledTimes(1);
    expect(log.mock.calls[0][0]).toMatch(/exited with code 1.*after the start failure/);
    expect(onFail).toHaveBeenCalledTimes(1);
    expect(onFail).toHaveBeenCalledWith("CRASH_BEFORE_READY");
  });

  it("a spawn failure carries the error object and settles once", () => {
    const t = tracker();
    const error = new Error("spawn ENOENT");
    t.fail("BACKEND_SPAWN_FAILED", "spawn ENOENT", error);
    t.fail("BACKEND_EXIT_ERROR", "a later cause");
    expect(logError).toHaveBeenCalledTimes(1);
    expect(logError.mock.calls[0][1]).toBe(error);
    expect(onFail).toHaveBeenCalledTimes(1);
  });

  it("a health timeout is one ERROR", () => {
    const t = tracker();
    t.fail("PORT_TIMEOUT", "did not report ready within the 330s safety cap");
    expect(logError).toHaveBeenCalledTimes(1);
    expect(t.settled()).toBe(true);
    expect(t.succeeded()).toBe(false);
  });

  it("an exit before readiness is itself the record, with the exit status", () => {
    const t = tracker();
    t.exited({}, 127, null);
    expect(logError).toHaveBeenCalledTimes(1);
    expect(logError.mock.calls[0][0]).toMatch(
      /^Backend start failed \(BACKEND_NOT_FOUND\): Backend process exited with code 127/
    );
    expect(onFail).toHaveBeenCalledWith("BACKEND_NOT_FOUND");
  });

  it("maps exit statuses to the supervisor's codes", () => {
    expect(exitFailureCode(127)).toBe("BACKEND_NOT_FOUND");
    expect(exitFailureCode(1)).toBe("BACKEND_EXIT_ERROR");
    expect(exitFailureCode(0)).toBe("CRASH_BEFORE_READY");
    expect(exitFailureCode(null)).toBe("CRASH_BEFORE_READY");
  });
});

describe("a running backend that dies", () => {
  it("is an ERROR when nobody asked it to stop", () => {
    const t = tracker();
    t.succeed();
    t.exited({}, null, "SIGSEGV");
    expect(onSucceed).toHaveBeenCalledTimes(1);
    expect(logError).toHaveBeenCalledTimes(1);
    expect(logError.mock.calls[0][0]).toMatch(/signal SIGSEGV.*not requested: the backend died/);
  });

  it("is a plain line when main asked it to stop", () => {
    const t = tracker();
    const proc = {};
    t.succeed();
    requestStop(proc);
    t.exited(proc, 0, null);
    expect(logError).not.toHaveBeenCalled();
    expect(log.mock.calls[0][0]).toMatch(/stop requested/);
  });
});

describe("a forced restart", () => {
  it("never records a crash for the child main killed, whatever the event order", () => {
    // Stop requested on the old child -> graceful shutdown times out -> hard
    // kill -> the replacement is spawned (its own tracker, its own process
    // object) -> only then does the old child's exit event arrive.
    const old = tracker();
    const oldProc = {};
    old.succeed();
    requestStop(oldProc);

    const replacementProc = {};
    const replacement = tracker();
    expect(stopRequested(replacementProc)).toBe(false);

    old.exited(oldProc, null, "SIGKILL");
    expect(logError).not.toHaveBeenCalled();
    expect(log.mock.calls[0][0]).toMatch(/signal SIGKILL \(stop requested\)/);

    // The replacement's own life is unaffected by the old child's flag.
    replacement.succeed();
    replacement.exited(replacementProc, 1, null);
    expect(logError).toHaveBeenCalledTimes(1);
    expect(logError.mock.calls[0][0]).toMatch(/the backend died/);
  });

  it("requestStop tolerates a missing process", () => {
    expect(() => requestStop(null)).not.toThrow();
    expect(stopRequested(null)).toBe(false);
  });
});
