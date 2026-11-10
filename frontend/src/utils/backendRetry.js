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
// DATA_DIR_IN_USE: the data folder is still held by the previous backend,
// which ends within a few seconds of a quit (backend/run.py bounds its exit),
// so an immediate relaunch gets the folder on a respawn.
export const TRANSIENT_CODES = new Set(["NO_PORT_AVAILABLE", "PORT_TIMEOUT", "DATA_DIR_IN_USE"]);

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
