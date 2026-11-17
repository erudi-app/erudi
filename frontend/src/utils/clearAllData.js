// Clear All Data (Help menu): empty everything Erudi stored, so that the next
// start is a first run.
//
// The order matters. The backend is stopped first, through the same graceful
// path a quit takes, so the embedded PostgreSQL closes its files and the
// backend its log; the window is then pointed at a blank page, so the
// interface writes nothing to its storage after it was cleared; the
// interface's storage is cleared through Chromium's session (deleting a
// profile folder under a running Chromium is not an option); and only then
// are the folders deleted (utils/dataLocations.js lists them).
//
// A file another process still holds (on Windows: an antivirus scan, a
// backend child that is slow to let go) is retried by fs.rm itself. Whatever
// is still held after that does not stop the rest: it is reported, and a
// marker in the Electron user data folder makes the next start run the
// clearing again before it opens a window or starts the backend.
//
// Pure apart from the file system: main.js passes the steps that need
// Electron.

import fs from "node:fs";
import path from "node:path";

export const PENDING_CLEAR_FILE = "clear-all-data-pending";

// fs.rm retries EBUSY, EPERM, EMFILE, ENFILE and ENOTEMPTY, waiting
// retryDelay * attempt between tries: about 5 s in all.
const RM_MAX_RETRIES = 6;
const RM_RETRY_DELAY_MS = 250;

/**
 * @param {string[]} targets Absolute paths, files or folders.
 * @param {object} deps
 * @param {(target: string) => boolean} deps.isSafe Last guard before a delete.
 * @param {Function} [deps.rm] `fs.promises.rm`, injectable for tests.
 * @returns {Promise<{removed: string[], pending: {target: string, code: string}[], refused: string[]}>}
 */
export async function removeTargets(targets, { isSafe, rm = fs.promises.rm }) {
  const removed = [];
  const pending = [];
  const refused = [];
  for (const target of targets) {
    if (!isSafe(target)) {
      refused.push(target);
      continue;
    }
    try {
      await rm(target, {
        recursive: true,
        force: true,
        maxRetries: RM_MAX_RETRIES,
        retryDelay: RM_RETRY_DELAY_MS,
      });
      removed.push(target);
    } catch (error) {
      pending.push({ target, code: (error && error.code) || "UNKNOWN" });
    }
  }
  return { removed, pending, refused };
}

/**
 * Run the clearing. Each Electron step is optional: the pass a start runs
 * from the marker has no backend to stop and no window yet.
 *
 * @param {object} deps
 * @param {(() => Promise<void>)|null} deps.stopBackend
 * @param {(() => Promise<void>)|null} deps.quietRenderer
 * @param {() => Promise<void>} deps.clearStorage
 * @param {string[]} deps.targets
 * @param {(target: string) => boolean} deps.isSafe
 * @param {string} deps.pendingMarker Path of the marker file.
 * @param {Function} [deps.rm]
 * @returns {Promise<{complete: boolean, pending: object[], refused: string[], storageError: Error|null}>}
 */
export async function clearAllData({
  stopBackend,
  quietRenderer,
  clearStorage,
  targets,
  isSafe,
  pendingMarker,
  rm,
}) {
  if (stopBackend) await stopBackend();
  if (quietRenderer) await quietRenderer();

  let storageError = null;
  try {
    await clearStorage();
  } catch (error) {
    // The files are still deleted; the marker below retries the storage.
    storageError = error;
  }

  const { pending, refused } = await removeTargets(targets, { isSafe, rm });
  const complete = pending.length === 0 && storageError === null;

  if (complete) {
    await fs.promises.rm(pendingMarker, { force: true });
  } else {
    await fs.promises.mkdir(path.dirname(pendingMarker), { recursive: true });
    await fs.promises.writeFile(pendingMarker, "");
  }
  return { complete, pending, refused, storageError };
}
