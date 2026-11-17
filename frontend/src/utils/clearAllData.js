// Clear All Data (Help menu): empty everything Erudi stored, so that the next
// start is a first run.
//
// The order matters. The backend is stopped first, through the same graceful
// path a quit takes, so the embedded PostgreSQL closes its files and the
// backend its log; the window is then pointed at a blank page, so the
// interface writes nothing to its storage after it was cleared; the
// interface's storage is cleared through Chromium's session (deleting a
// profile folder under a running Chromium is not an option); the app log is
// silenced, since it is about to be deleted and a line written after would
// bring it back; and only then are the targets deleted (utils/dataLocations.js
// lists them).
//
// What is deleted is checked three times: the guard (never a file system
// root, the home folder or a root Erudi does not own), the ownership file a
// folder must carry (OWNERSHIP_SENTINEL: a folder Erudi did not create is
// skipped and named), and for a file another user owns, a refusal instead of
// a retry.
//
// A file another process still holds (on Windows: an antivirus scan, a
// backend child that is slow to let go) is retried by fs.rm itself. What is
// still held after that does not stop the rest: the marker file records
// exactly those paths, with the kind of run that wrote it (installed or dev,
// and its ERUDI_DATA_ROOT). The next start of the same kind of run deletes
// those paths -- only those still among its own targets -- before it opens a
// window or starts the backend, at most MAX_PENDING_ATTEMPTS times. Any other
// run ignores the marker and removes it: a dev run and the installed app
// share the user data folder on macOS and Windows.
//
// Pure apart from the file system: main.js passes the steps that need
// Electron.

import fs from "node:fs";
import path from "node:path";

export const PENDING_CLEAR_FILE = "clear-all-data-pending.json";
export const MAX_PENDING_ATTEMPTS = 3;
const MARKER_VERSION = 1;

// fs.rm retries EBUSY, EPERM, EMFILE, ENFILE and ENOTEMPTY, waiting
// retryDelay * attempt between tries: about 5 s in all.
const RM_MAX_RETRIES = 6;
const RM_RETRY_DELAY_MS = 250;

const PERMISSION_CODES = new Set(["EACCES", "EPERM"]);

/**
 * @param {{path: string, owner: string|null}[]} targets
 * @param {object} deps
 * @param {(target: string) => boolean} deps.isSafe Last guard before a delete.
 * @param {(dir: string) => boolean} deps.isOwned Whether a folder carries
 *   the ownership file.
 * @param {(target: string) => boolean} deps.isForeign Whether another user
 *   owns the target (a permission error on it is then final).
 * @param {Function} [deps.rm] `fs.promises.rm`, injectable for tests.
 * @returns {Promise<{removed: string[], pending: {path: string, code: string}[],
 *   refused: {path: string, why: "unsafe"|"foreign"}[], skipped: string[]}>}
 */
export async function removeTargets(targets, { isSafe, isOwned, isForeign, rm = fs.promises.rm }) {
  const removed = [];
  const pending = [];
  const refused = [];
  const skipped = [];
  for (const target of targets) {
    if (!isSafe(target.path)) {
      refused.push({ path: target.path, why: "unsafe" });
      continue;
    }
    if (target.owner && !isOwned(target.owner)) {
      // Nothing there is nothing to protect; something there is not Erudi's.
      if (fs.existsSync(target.path)) skipped.push(target.path);
      continue;
    }
    try {
      await rm(target.path, {
        recursive: true,
        force: true,
        maxRetries: RM_MAX_RETRIES,
        retryDelay: RM_RETRY_DELAY_MS,
      });
      removed.push(target.path);
    } catch (error) {
      const code = (error && error.code) || "UNKNOWN";
      if (PERMISSION_CODES.has(code) && isForeign(target.path)) {
        refused.push({ path: target.path, why: "foreign" });
      } else {
        pending.push({ path: target.path, code });
      }
    }
  }
  return { removed, pending, refused, skipped };
}

const sameContext = (a, b) =>
  Boolean(a) && a.isPackaged === b.isPackaged && (a.instanceRoot ?? null) === b.instanceRoot;

/**
 * @param {object} deps
 * @param {{isPackaged: boolean, instanceRoot: string|null}} deps.context
 * @param {boolean} deps.available False in a dev run without ERUDI_DATA_ROOT.
 * @param {boolean} deps.rootRefused ERUDI_DATA_ROOT is the home folder or above.
 * @param {() => Promise<boolean>} deps.confirm
 * @param {() => void} deps.markQuitting
 * @param {() => Promise<void>} deps.stopBackend
 * @param {() => Promise<void>} deps.quietRenderer
 * @param {() => Promise<void>} deps.clearStorage
 * @param {() => void} deps.silenceAppLog
 * @param {() => {path: string, owner: string|null}[]} deps.targets
 * @param {(target: string) => boolean} deps.isSafe
 * @param {(dir: string) => boolean} deps.isOwned
 * @param {(target: string) => boolean} deps.isForeign
 * @param {string} deps.markerPath
 * @param {(result: {complete: boolean, leftInPlace: string[]}) => Promise<void>} deps.showResult
 * @param {() => Promise<void>} deps.showRootRefused
 * @param {(error: Error) => Promise<void>} deps.showError
 * @param {() => void} deps.quit
 * @param {Function} [deps.rm]
 */
export function createClearAllDataFlow(deps) {
  const { context, log, logWarn, logError } = deps;
  const removeOptions = {
    isSafe: deps.isSafe,
    isOwned: deps.isOwned,
    isForeign: deps.isForeign,
    rm: deps.rm,
  };

  const writeMarker = (marker) => {
    fs.mkdirSync(path.dirname(deps.markerPath), { recursive: true });
    fs.writeFileSync(deps.markerPath, JSON.stringify(marker));
  };
  const dropMarker = () => fs.rmSync(deps.markerPath, { force: true });

  async function clearStorageSafely() {
    try {
      await deps.clearStorage();
      return null;
    } catch (error) {
      return error;
    }
  }

  async function clearFromMenu() {
    if (!deps.available) {
      logWarn("Clear All Data is not available in a dev run without ERUDI_DATA_ROOT");
      return;
    }
    if (deps.rootRefused) {
      logError(
        "Clear All Data refused: ERUDI_DATA_ROOT is the home folder, a folder above it or a file system root"
      );
      await deps.showRootRefused();
      return;
    }
    if (!(await deps.confirm())) {
      log("User cancelled data deletion");
      return;
    }
    log("User confirmed data deletion. Clearing all data...");
    // The app quits at the end, whatever happens: the backend supervisor
    // stands down instead of respawning the backend stopped below.
    deps.markQuitting();
    try {
      await deps.stopBackend();
      await deps.quietRenderer();
      const storageError = await clearStorageSafely();
      const targets = deps.targets();
      log(`Clear All Data: deleting ${targets.length} folders and files; the log ends here`);
      deps.silenceAppLog();
      const result = await removeTargets(targets, removeOptions);
      const complete = result.pending.length === 0 && storageError === null;
      if (complete) {
        dropMarker();
      } else {
        writeMarker({
          version: MARKER_VERSION,
          context,
          attempts: 0,
          storage: storageError !== null,
          targets: result.pending.map((p) => p.path),
        });
      }
      await deps.showResult({
        complete,
        pending: result.pending,
        leftInPlace: [...result.skipped, ...result.refused.map((r) => r.path)],
      });
    } catch (error) {
      logError("Failed to clear the data", error);
      await deps.showError(error);
    }
    deps.quit();
  }

  async function finishPending() {
    if (!fs.existsSync(deps.markerPath)) return;
    let marker;
    try {
      marker = JSON.parse(fs.readFileSync(deps.markerPath, "utf8"));
    } catch (error) {
      logWarn("Clear All Data: the pending-clear file is unreadable; dropped", error);
      dropMarker();
      return;
    }
    if (marker?.version !== MARKER_VERSION || !sameContext(marker.context, context)) {
      logWarn(
        "Clear All Data: the pending-clear file belongs to another kind of run " +
          "(installed or dev, or another ERUDI_DATA_ROOT); ignored and dropped"
      );
      dropMarker();
      return;
    }
    if (deps.rootRefused) {
      logWarn("Clear All Data: pending clear dropped, ERUDI_DATA_ROOT is now refused");
      dropMarker();
      return;
    }

    // Only the marker's paths, and only those still among this run's targets.
    const current = new Map(deps.targets().map((target) => [target.path, target]));
    const wanted = Array.isArray(marker.targets) ? marker.targets : [];
    const strangers = wanted.filter((p) => !current.has(p));
    for (const stranger of strangers) {
      logWarn(`Clear All Data: ${stranger} is no longer one of this run's targets; left in place`);
    }
    const targets = wanted.filter((p) => current.has(p)).map((p) => current.get(p));

    log("Finishing the Clear All Data of the previous session...");
    const storageError = marker.storage ? await clearStorageSafely() : null;
    const result = await removeTargets(targets, removeOptions);
    for (const { path: target, why } of result.refused) {
      logWarn(`Clear All Data: ${target} left in place (${why})`);
    }
    for (const target of result.skipped) {
      logWarn(`Clear All Data: ${target} left in place (no ownership file)`);
    }
    if (result.pending.length === 0 && storageError === null) {
      dropMarker();
      log("Clear All Data: finished");
      return;
    }
    const attempts = (Number.isInteger(marker.attempts) ? marker.attempts : 0) + 1;
    const still = result.pending.map((p) => `${p.path} (${p.code})`).join(", ");
    if (attempts >= MAX_PENDING_ATTEMPTS) {
      logWarn(
        `Clear All Data: giving up after ${attempts} starts; still in use: ${still || "storage"}`,
        ...(storageError ? [storageError] : [])
      );
      dropMarker();
      return;
    }
    logWarn(`Clear All Data: still in use, retried at the next start: ${still || "storage"}`);
    writeMarker({
      ...marker,
      attempts,
      storage: storageError !== null,
      targets: result.pending.map((p) => p.path),
    });
  }

  return { clearFromMenu, finishPending };
}
