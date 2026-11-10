// Backend start supervision: which failures respawn, the wait for a data
// folder the previous backend still holds, and one loop at a time.
//
// Pure: main.js cannot be imported by the test runner, so the decisions live
// here and main.js only wires them to its child process and its window.
//
// One loop at a time. `start()` begins a new generation; a loop that is no
// longer the current one stands down at its next step (it neither respawns
// nor reports), so a Retry while a wait runs (backend:restart) replaces the
// loop instead of racing it over the same child. A quit stands every loop
// down the same way.

import {
  DATA_DIR_RETRY_MS,
  DATA_DIR_WAIT_MS,
  shouldRetrySpawn,
  shouldWaitForDataDir,
} from "./backendRetry";

// The phase main sends the window while it waits for the previous backend
// (the loader says the previous session is closing).
export const WAITING_PREVIOUS_SESSION_PHASE = "waiting_previous_session";

// Respawns of a transient failure (port contention), then a manual retry.
const MAX_SPAWN_ATTEMPTS = 2;
const TRANSIENT_RETRY_MS = 2000;

/**
 * @param {object} deps
 * @param {() => Promise<{port: number}>} deps.startBackend - One spawn; rejects with the code.
 * @param {() => void} deps.stopBackend - Stop the child of a failed attempt.
 * @param {(ms: number) => Promise<void>} deps.sleep
 * @param {() => number} deps.now
 * @param {() => boolean} deps.isQuitting - True once the app is quitting.
 * @param {(message: string) => void} deps.log - Unlevelled line.
 * @param {(message: string) => void} deps.logWarn
 * @param {(message: string) => void} deps.logError
 * @param {(port: number) => void} deps.onReady - The backend is confirmed.
 * @param {() => void} deps.onWaiting - A wait for the previous backend goes on.
 * @param {(code: string, error: *) => void} deps.onFailed - Surface the failure.
 * @returns {{start: () => Promise<void>}}
 */
export function createBackendSupervisor({
  startBackend,
  stopBackend,
  sleep,
  now,
  isQuitting,
  log,
  logWarn,
  logError,
  onReady,
  onWaiting,
  onFailed,
}) {
  let generation = 0;
  let dataDirWaitSince = null;

  const standsDown = (gen) => gen !== generation || isQuitting();

  async function run(gen, attempt) {
    let port;
    try {
      log(`Backend startup attempt ${attempt + 1}...`);
      ({ port } = await startBackend());
    } catch (error) {
      const code = (error && error.message) || "BACKEND_STARTUP_FAILED";
      log(`Backend start attempt ${attempt + 1} failed: ${code}`);
      if (standsDown(gen)) {
        log(`Backend start loop superseded or app quitting; not respawning after ${code}`);
        return;
      }
      if (code === "DATA_DIR_IN_USE") {
        // The previous backend is still ending (a quit followed by a quick
        // relaunch): respawn until it lets go of the data folder.
        if (dataDirWaitSince === null) {
          dataDirWaitSince = now();
          log("The data folder is still held by the previous backend; waiting for it to exit");
        }
        if (shouldWaitForDataDir(code, now() - dataDirWaitSince)) {
          onWaiting();
          stopBackend();
          await sleep(DATA_DIR_RETRY_MS);
          if (standsDown(gen)) return;
          return run(gen, attempt);
        }
        dataDirWaitSince = null;
        logError(
          `Backend start failed (DATA_DIR_IN_USE): the data folder stayed held for ${DATA_DIR_WAIT_MS / 1000}s`
        );
      } else if (shouldRetrySpawn(code, attempt, MAX_SPAWN_ATTEMPTS)) {
        logWarn(`Transient backend start failure (${code}); respawning`);
        stopBackend();
        await sleep(TRANSIENT_RETRY_MS);
        if (standsDown(gen)) return;
        return run(gen, attempt + 1);
      }
      // The failure itself is already recorded (one ERROR, by the start
      // tracker, which knows the cause); this line only says it reached the
      // user.
      log(`Backend startup failed (${code}); surfacing to the user`);
      onFailed(code, error);
      return;
    }
    if (gen !== generation) return; // a newer loop owns the backend now
    dataDirWaitSince = null;
    log(`Backend is ready on port ${port}.`);
    onReady(port);
  }

  return {
    /** Start a new supervision loop, superseding any loop still running. */
    start() {
      generation += 1;
      dataDirWaitSince = null;
      return run(generation, 0);
    },
  };
}
