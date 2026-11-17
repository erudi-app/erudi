// Where Erudi keeps what it stores, seen from the Electron main process.
//
// The backend decides where its data and logs live
// (backend/src/launcher/runtime_paths.py). Main needs the same folders when
// the backend cannot answer: Open data folder works with the backend down,
// and Clear All Data stops it before deleting anything. So the layout is
// computed here too, and both sides are checked against one table,
// backend/tests/fixtures/runtime_paths_layout.json: neither can move a folder
// without failing a test until the other follows.
//
// Pure: the platform, the environment and the home folder are arguments, so
// the tests can resolve every platform's layout from any machine.

import path from "node:path";

const APP_NAME = "erudi";

// Electron's own files that Erudi writes in the user data folder (main.js,
// "Updated to X.Y.Z"). The rest of that folder is Chromium's profile, which
// Clear All Data empties through the session, not by deleting files under a
// running Chromium.
export const LAST_SEEN_VERSION_FILE = "last-seen-version";
export const UPDATE_NOTES_FILE = "update-notes.json";
export const ELECTRON_OWN_FILES = [LAST_SEEN_VERSION_FILE, UPDATE_NOTES_FILE];

const pathFor = (platform) => (platform === "win32" ? path.win32 : path.posix);

/**
 * The ERUDI_DATA_ROOT instance root, resolved the way the backend resolves
 * it (`~` expanded, made absolute), or null when unset.
 */
export function instanceRootFrom(env, { platform, home }) {
  const raw = env.ERUDI_DATA_ROOT;
  if (!raw) return null;
  const p = pathFor(platform);
  const expanded = raw === "~" || raw.startsWith("~/") ? p.join(home, raw.slice(1)) : raw;
  return p.resolve(expanded);
}

/**
 * @param {object} input
 * @param {string} input.platform `process.platform`.
 * @param {object} input.env `process.env`.
 * @param {string} input.home `os.homedir()`.
 * @param {string} input.tmpdir `os.tmpdir()`.
 * @param {boolean} input.isPackaged `app.isPackaged`.
 * @param {string|null} input.devBackendRoot The repository's `backend/`
 *   folder, used by a dev run (the backend runs from source there).
 * @returns {{
 *   platform: string, dataDir: string, backendLogDir: string, appLogDir: string,
 *   pastedImagesDir: string, backendDirs: string[], legacyPastedImagesDir: string|null,
 *   instanceRoot: string|null, electronDir: string|null,
 * }}
 *   `dataDir` is the backend's data folder (Open data folder); `appLogDir`
 *   the folder of main's own log, erudi-backend.log (the system's temporary
 *   folder, or the instance's log folder under ERUDI_DATA_ROOT, so that an
 *   instance shares no file with the installed app); `backendDirs` the
 *   folders holding the data and the backend logs, which Clear All Data
 *   deletes whole.
 */
export function resolveDataLocations({ platform, env, home, tmpdir, isPackaged, devBackendRoot }) {
  const p = pathFor(platform);
  const instanceRoot = instanceRootFrom(env, { platform, home });

  let dataDir;
  let backendLogDir;
  let backendHome;
  let legacyPastedImagesDir = null;
  if (instanceRoot) {
    dataDir = p.join(instanceRoot, "data");
    backendLogDir = p.join(instanceRoot, "logs");
    backendHome = dataDir;
  } else if (!isPackaged) {
    dataDir = p.join(devBackendRoot, "data");
    backendLogDir = p.join(devBackendRoot, "logs");
    backendHome = dataDir;
  } else {
    let appRoot;
    if (platform === "darwin") {
      appRoot = p.join(home, "Library", "Application Support", APP_NAME);
      backendLogDir = p.join(home, "Library", "Logs", APP_NAME);
    } else if (platform === "win32") {
      const localAppData = env.LOCALAPPDATA || p.join(home, "AppData", "Local");
      appRoot = p.join(localAppData, APP_NAME);
      backendLogDir = p.join(appRoot, "logs");
    } else {
      const dataHome = env.XDG_DATA_HOME || p.join(home, ".local", "share");
      const stateHome = env.XDG_STATE_HOME || p.join(home, ".local", "state");
      appRoot = p.join(dataHome, APP_NAME);
      backendLogDir = p.join(stateHome, APP_NAME, "logs");
    }
    backendHome = p.join(appRoot, "backend");
    dataDir = p.join(backendHome, "prod", "data");
    // Before pasted images moved into the data folder they were written under
    // %LOCALAPPDATA%\erudi on Windows and under ~/Library/Application
    // Support/erudi everywhere else -- Linux included. Clear All Data still
    // removes what an earlier version left there.
    const legacyRoot =
      platform === "win32"
        ? p.join(env.LOCALAPPDATA || p.join(home, "AppData", "Local"), APP_NAME)
        : p.join(home, "Library", "Application Support", APP_NAME);
    legacyPastedImagesDir = p.join(legacyRoot, "pasted-images");
  }

  return {
    platform,
    dataDir,
    backendLogDir,
    appLogDir: instanceRoot ? backendLogDir : tmpdir,
    pastedImagesDir: p.join(dataDir, "pasted-images"),
    backendDirs: [backendHome, backendLogDir],
    legacyPastedImagesDir,
    instanceRoot,
    electronDir: instanceRoot ? p.join(instanceRoot, "electron") : null,
  };
}

/**
 * Everything Clear All Data deletes, in order: the backend's folders, the
 * pasted images an earlier version left elsewhere, the Electron logs, Erudi's
 * own files in the Electron user data folder, and the app log.
 *
 * @param {ReturnType<typeof resolveDataLocations>} locations
 * @param {object} electron
 * @param {string} electron.userData `app.getPath("userData")`.
 * @param {string|null} electron.electronLogsDir `app.getPath("logs")`, or
 *   null when electron-log writes nothing (a dev run without an instance
 *   root: there that folder is the installed app's).
 * @param {string[]} electron.appLogFiles The app log and its rotated copy.
 * @returns {string[]}
 */
export function listClearTargets(locations, { userData, electronLogsDir, appLogFiles }) {
  const p = pathFor(locations.platform);
  const targets = [...locations.backendDirs];
  if (locations.legacyPastedImagesDir) targets.push(locations.legacyPastedImagesDir);
  if (electronLogsDir) targets.push(electronLogsDir);
  for (const name of ELECTRON_OWN_FILES) targets.push(p.join(userData, name));
  targets.push(...appLogFiles);
  return [...new Set(targets)];
}

/**
 * Last check before a recursive delete: the target is absolute, and neither
 * a file system root, the home folder or one of its ancestors, nor a root
 * Erudi does not own (the ERUDI_DATA_ROOT folder, the Electron user data
 * folder) or one of their ancestors.
 */
export function isSafeToDelete(target, { platform, home, protectedDirs }) {
  const p = pathFor(platform);
  if (typeof target !== "string" || !target || !p.isAbsolute(target)) return false;
  const resolved = p.resolve(target);
  if (p.parse(resolved).root === resolved) return false;
  const containsOrIs = (dir) => {
    const relative = p.relative(resolved, p.resolve(dir));
    return relative === "" || (!relative.startsWith("..") && !p.isAbsolute(relative));
  };
  return ![home, ...protectedDirs].filter(Boolean).some(containsOrIs);
}
