// What main.js decides about Clear All Data before the flow runs
// (utils/clearAllData.js): whether it is offered, whether the ERUDI_DATA_ROOT
// is acceptable, the guard every deletion goes through, whether a file is
// another user's, and the marking of the installed app's Electron logs.

import fs from "node:fs";
import { isSafeToDelete, isUnsafeInstanceRoot } from "./dataLocations";
import { claimFolder } from "./folderOwnership";

/**
 * Offered by the installed app only. A run from source spawns no backend (it
 * expects one started by hand), so it could not stop the backend whose data
 * it would delete; and on macOS and Windows it shares the interface storage
 * with the installed app. Testing Clear All Data takes a packaged build or
 * the unit tests.
 */
export function isClearAllDataAvailable({ isPackaged }) {
  return Boolean(isPackaged);
}

/**
 * An ERUDI_DATA_ROOT that is the home folder, above it, or a file system
 * root: Clear All Data refuses to run at all.
 */
export function isInstanceRootRefused({ instanceRoot, platform, home, realpath }) {
  return Boolean(instanceRoot) && isUnsafeInstanceRoot(instanceRoot, { platform, home, realpath });
}

/**
 * The last check before each deletion: never the home folder or above, never
 * the ERUDI_DATA_ROOT folder (a user may point it at a shared folder) or the
 * Electron user data folder itself (the running Chromium's profile); compared
 * on real paths, case-folded where the file system ignores case.
 */
export function createClearTargetGuard({ platform, home, instanceRoot, userData, realpath }) {
  return (target) =>
    isSafeToDelete(target, {
      platform,
      home,
      protectedDirs: [instanceRoot, userData],
      realpath,
    });
}

/**
 * Whether another user owns `target` (a shared /tmp/erudi-backend.log on a
 * multi-user Linux machine): a permission error on it is then final. Windows
 * has no user ids here (`getuid` is undefined); there a permission error is a
 * held file and is retried.
 */
export function isForeignFile(target, { getuid = process.getuid } = {}) {
  if (typeof getuid !== "function") return false;
  try {
    return fs.lstatSync(target).uid !== getuid();
  } catch {
    // Gone or unreadable: nothing to call foreign; the retry decides.
    return false;
  }
}

/**
 * The installed app's Electron logs folder (`Erudi` in the platform's log
 * location, or in its user data folder) is Erudi's by name: marked on every
 * start, existing or not. Under ERUDI_DATA_ROOT, setUpElectronPaths marked
 * the instance's folders before Electron created them; a run from source has
 * no Electron logs to clear.
 *
 * @returns {Error|null} The error, for main to log, instead of throwing.
 */
export function claimInstalledLogsFolder({
  isPackaged,
  electronDir,
  logsDir,
  claim = claimFolder,
}) {
  if (electronDir || !isPackaged) return null;
  try {
    claim(logsDir, { namespaced: true });
    return null;
  } catch (error) {
    return error;
  }
}
