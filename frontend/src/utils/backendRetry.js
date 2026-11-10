// Retry policy for backend startup — two distinct kinds of retry:
//
//  1. Spawn respawn (on an explicit startup_error / early process exit): only
//     worth it for TRANSIENT codes. Deterministic failures (missing import,
//     missing DLL, data-dir error) reproduce identically, so we fail fast and
//     let the user retry manually. "Slow" is NOT a failure — a still-alive,
//     still-initializing backend is never respawned (that was the 30s-kill bug).
//
//  2. Post-`ready` health confirmation (backend stays up): retry the HTTP
//     request with backoff to ride out loopback/port warm-up.

// startup_error codes worth an automatic respawn. run.py already scans a port
// range internally, so even these rarely need it — kept minimal on purpose.
export const TRANSIENT_CODES = new Set(["NO_PORT_AVAILABLE", "PORT_TIMEOUT"]);

// DATA_DIR_IN_USE has its own, longer wait: the data folder is still held by
// the previous backend, which can take ~20 s to end after a quit (uvicorn's
// drain of a stuck request, the download wait, its own stops, the exit
// bound in backend/run.py). Main respawns every DATA_DIR_RETRY_MS for up to
// DATA_DIR_WAIT_MS before it shows the error screen.
export const DATA_DIR_WAIT_MS = 30000;
export const DATA_DIR_RETRY_MS = 2000;

/**
 * True for a startup_error main waits out: it is not forwarded to the window
 * (which would show the error screen over a wait that usually ends well).
 */
export function isWaitedOutStartupError(event) {
  return Boolean(event) && event.event === "startup_error" && event.code === "DATA_DIR_IN_USE";
}

/** Whether to respawn after DATA_DIR_IN_USE, `waitedMs` into the wait. */
export function shouldWaitForDataDir(code, waitedMs) {
  return code === "DATA_DIR_IN_USE" && waitedMs < DATA_DIR_WAIT_MS;
}

/** "transient" (respawn may help) vs "deterministic" (respawn won't). */
export function classifyStartupError(code) {
  return TRANSIENT_CODES.has(code) ? "transient" : "deterministic";
}

/** Whether to auto-respawn the backend after a startup_error. */
export function shouldRetrySpawn(code, attempt, maxAttempts = 2) {
  return classifyStartupError(code) === "transient" && attempt < maxAttempts;
}

// Backoff (ms) for the post-`ready` health-confirmation retries.
export const HEALTH_CONFIRM_BACKOFF_MS = [5000, 10000, 20000];

/** Backoff for confirmation attempt `attempt` (0-indexed), clamped to the tail. */
export function healthConfirmBackoffMs(attempt) {
  const i = Math.min(Math.max(attempt, 0), HEALTH_CONFIRM_BACKOFF_MS.length - 1);
  return HEALTH_CONFIRM_BACKOFF_MS[i];
}
