// The ownership file Clear All Data checks before it deletes a folder
// (OWNERSHIP_SENTINEL, written by the backend for its data and log folders,
// src/launcher/runtime_paths.py, and here for the Electron side's own).
//
// A folder is claimed when Erudi creates it, or when its whole path is fixed
// by Erudi (`namespaced`: an `Erudi` folder in the platform's log location,
// say), so an install from before the file gets it on its next start. A
// folder that already exists under a root the user chose (ERUDI_DATA_ROOT)
// is never claimed: it may be the user's own.

import fs from "node:fs";
import path from "node:path";
import { OWNERSHIP_SENTINEL } from "./dataLocations";

const SENTINEL_TEXT =
  "Erudi created this folder and keeps its data or logs here.\n" +
  "Help > Clear All Data deletes it.\n";

export function isOwnedFolder(dir) {
  return fs.existsSync(path.join(dir, OWNERSHIP_SENTINEL));
}

/**
 * @param {string} dir
 * @param {{namespaced: boolean}} options
 * @returns {boolean} Whether the folder is Erudi's after the call. Throws
 *   when the folder or the file cannot be written.
 */
export function claimFolder(dir, { namespaced }) {
  if (isOwnedFolder(dir)) return true;
  const existed = fs.existsSync(dir);
  if (existed && !namespaced) return false;
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(path.join(dir, OWNERSHIP_SENTINEL), SENTINEL_TEXT);
  return true;
}
