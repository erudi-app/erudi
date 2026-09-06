// One record per backend start failure, and the right level for each exit.
//
// A backend that fails to start is seen from several places at once: the
// `startup_error` event on its stdout, the `error` event of a spawn that never
// happened, the `exit` event that follows either, the safety cap, and the
// supervisor that finally surfaces the failure to the window. Each of them
// knows something, and if each wrote an ERROR the Diagnostics panel would show
// one failure two or three times. The tracker is the single owner of that
// record: the first cause to arrive writes it, everything after it is an
// unlevelled line.
//
// The requested-stop state lives on the child process object, not in a global:
// a restart whose graceful shutdown times out hard-kills the old child, spawns
// the replacement, and only then receives the old child's `exit` event. A
// global flag reset at spawn time would call that a crash.
//
// Pure: main.js cannot be imported by the test runner, so the decisions live
// here and main.js only wires them to its child process and its promise.

/** Mark a child as asked to stop; its exit is then never a crash. */
export function requestStop(proc) {
  if (proc && typeof proc === "object") proc.erudiStopRequested = true;
}

/** Whether a child was asked to stop (false for a missing process). */
export function stopRequested(proc) {
  return Boolean(proc && proc.erudiStopRequested);
}

/**
 * The failure code an exit status maps to while the start is still pending.
 * @param {number|null} code - Exit code, null when killed by a signal.
 * @returns {string} A code the supervisor classifies (see backendRetry.js).
 */
export function exitFailureCode(code) {
  if (code === 127) return "BACKEND_NOT_FOUND";
  if (code !== 0 && code !== null) return "BACKEND_EXIT_ERROR";
  // A clean exit before readiness was confirmed is still a crash.
  return "CRASH_BEFORE_READY";
}

/**
 * Track one start attempt of the backend child.
 *
 * @param {object} io - Where records and outcomes go.
 * @param {(message: string) => void} io.log - Unlevelled line.
 * @param {(message: string, error?: *) => void} io.logError - ERROR record.
 * @param {(code: string) => void} io.onFail - Called once, on the first failure.
 * @param {() => void} io.onSucceed - Called once, when the backend is confirmed.
 * @returns {{fail: Function, succeed: Function, exited: Function, settled: () => boolean, succeeded: () => boolean}} The tracker.
 */
export function createBackendStartTracker({ log, logError, onFail, onSucceed }) {
  let settled = false;
  let succeeded = false;

  /**
   * Record the failure and settle the start. Only the first call writes: the
   * later ones are consequences of the same cause.
   * @param {string} code - Failure code for the supervisor.
   * @param {string} cause - What was observed (the backend's message, the
   *   exit status, the cap that fired).
   * @param {*} [error] - The error object, when there is one.
   */
  const fail = (code, cause, error) => {
    if (settled) return;
    settled = true;
    logError(`Backend start failed (${code}): ${cause}`, error);
    onFail(code);
  };

  const succeed = () => {
    if (settled) return;
    settled = true;
    succeeded = true;
    onSucceed();
  };

  /**
   * The child exited. The level depends on who asked and on when.
   * @param {object} proc - The child process object.
   * @param {number|null} code - Exit code.
   * @param {string|null} signal - Terminating signal.
   */
  const exited = (proc, code, signal) => {
    const outcome = `Backend process exited with code ${code}, signal ${signal}`;
    if (stopRequested(proc)) {
      log(`${outcome} (stop requested)`);
      return;
    }
    if (!settled) {
      // The exit IS the failure: it is the record.
      fail(exitFailureCode(code), outcome);
      return;
    }
    if (succeeded) {
      // It was running and nobody asked it to stop: it crashed, or something
      // killed it.
      logError(`${outcome} (not requested: the backend died)`);
      return;
    }
    // The start already failed and was recorded; this exit follows from it.
    log(`${outcome} (after the start failure recorded above)`);
  };

  return {
    fail,
    succeed,
    exited,
    settled: () => settled,
    succeeded: () => succeeded,
  };
}
