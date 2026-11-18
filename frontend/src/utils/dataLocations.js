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
// Pure apart from `realpath`, which is injectable: the platform, the
// environment and the home folder are arguments, so the tests can resolve
// every platform's layout from any machine.

import fs from "node:fs";
import path from "node:path";

const APP_NAME = "erudi";

// The file that marks a folder as Erudi's (runtime_paths.OWNERSHIP_SENTINEL;
// the shared table pins the name). Clear All Data deletes a folder only when
// it carries this file.
export const OWNERSHIP_SENTINEL = ".erudi-owned";

// The folder, inside the data folder, where a pasted image is written
// (utils/pastedImage.js). The backend removes a conversation's pasted images
// from it, and only from it, when the conversation is deleted
// (runtime_paths.PASTED_IMAGES_FOLDER; both pinned by
// backend/tests/fixtures/runtime_paths_layout.json).
export const PASTED_IMAGES_FOLDER = "pasted-images";

// Electron's own files that Erudi writes in the user data folder (main.js,
// "Updated to X.Y.Z"). The rest of that folder is Chromium's profile, which
// Clear All Data empties through the session, not by deleting files under a
// running Chromium.
export const LAST_SEEN_VERSION_FILE = "last-seen-version";
export const UPDATE_NOTES_FILE = "update-notes.json";
export const ELECTRON_OWN_FILES = [LAST_SEEN_VERSION_FILE, UPDATE_NOTES_FILE];

const pathFor = (platform) => (platform === "win32" ? path.win32 : path.posix);

/**
 * `~` expanded the way Python's expanduser does: `~` and `~/x` (and `~\x` on
 * Windows) under the home folder. `~user` names a sibling of the home folder,
 * which is what Windows does and what the usual /home or /Users layout gives
 * on POSIX; main hands the backend the absolute result, so both sides always
 * use the same folder whatever a particular system's user database says.
 */
function expandHome(raw, { platform, home }) {
  if (!raw.startsWith("~")) return raw;
  const p = pathFor(platform);
  const separators = platform === "win32" ? ["/", "\\"] : ["/"];
  let end = raw.length;
  for (const sep of separators) {
    const at = raw.indexOf(sep);
    if (at !== -1 && at < end) end = at;
  }
  const user = raw.slice(1, end);
  const rest = raw.slice(end + 1);
  const base = user ? p.join(p.dirname(home), user) : home;
  return rest ? p.join(base, rest) : base;
}

/**
 * The ERUDI_DATA_ROOT instance root, `~` expanded and made absolute against
 * `cwd`, or null when unset or empty. main.js writes the result back into its
 * environment, so the backend it spawns receives this absolute path.
 */
export function instanceRootFrom(env, { platform, home, cwd }) {
  const raw = env.ERUDI_DATA_ROOT;
  if (!raw) return null;
  const p = pathFor(platform);
  return p.resolve(cwd, expandHome(raw, { platform, home }));
}

/**
 * @param {object} input
 * @param {string} input.platform `process.platform`.
 * @param {object} input.env `process.env`.
 * @param {string} input.home `os.homedir()`.
 * @param {string} input.tmpdir `os.tmpdir()`.
 * @param {string} input.cwd `process.cwd()`, for a relative ERUDI_DATA_ROOT.
 * @param {boolean} input.isPackaged `app.isPackaged`.
 * @param {string|null} input.devBackendRoot The repository's `backend/`
 *   folder, used by a dev run (the backend runs from source there).
 * @returns {{
 *   platform: string, dataDir: string, backendLogDir: string, appLogDir: string,
 *   pastedImagesDir: string, backendHome: string, legacyPastedImagesDir: string|null,
 *   instanceRoot: string|null, electronDir: string|null,
 * }}
 *   `dataDir` is the backend's data folder (Open data folder); `backendHome`
 *   the folder Clear All Data deletes for it (the data folder itself, or the
 *   `backend` folder above `prod/data` in an installed app); `appLogDir` the
 *   folder of main's own log, erudi-backend.log (the system's temporary
 *   folder, or the instance's log folder under ERUDI_DATA_ROOT, so that an
 *   instance shares no file with the installed app).
 */
export function resolveDataLocations({
  platform,
  env,
  home,
  tmpdir,
  cwd = "/",
  isPackaged,
  devBackendRoot,
}) {
  const p = pathFor(platform);
  const instanceRoot = instanceRootFrom(env, { platform, home, cwd });

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
    pastedImagesDir: p.join(dataDir, PASTED_IMAGES_FOLDER),
    backendHome,
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
 * Each target names its `owner`: the folder whose OWNERSHIP_SENTINEL must be
 * present for the target to be deleted, or null when the whole path is fixed
 * by Erudi with no part a user chose (an `erudi` folder in the platform's
 * app-data location, a file named erudi-backend.log in the temporary folder).
 * Under ERUDI_DATA_ROOT every target has an owner.
 *
 * @param {ReturnType<typeof resolveDataLocations>} locations
 * @param {object} electron
 * @param {string} electron.userData `app.getPath("userData")`.
 * @param {string|null} electron.electronLogsDir `app.getPath("logs")`, or
 *   null when electron-log writes nothing (a dev run without an instance
 *   root: there that folder is the installed app's).
 * @param {string[]} electron.appLogFiles The app log and its rotated copy.
 * @returns {{path: string, owner: string|null}[]}
 */
export function listClearTargets(locations, { userData, electronLogsDir, appLogFiles }) {
  const p = pathFor(locations.platform);
  const underRoot = Boolean(locations.instanceRoot);
  const targets = [
    { path: locations.backendHome, owner: locations.dataDir },
    { path: locations.backendLogDir, owner: locations.backendLogDir },
  ];
  if (locations.legacyPastedImagesDir) {
    targets.push({ path: locations.legacyPastedImagesDir, owner: null });
  }
  if (electronLogsDir) targets.push({ path: electronLogsDir, owner: electronLogsDir });
  for (const name of ELECTRON_OWN_FILES) {
    targets.push({ path: p.join(userData, name), owner: underRoot ? userData : null });
  }
  for (const file of appLogFiles) {
    targets.push({ path: file, owner: underRoot ? locations.appLogDir : null });
  }
  const seen = new Set();
  return targets.filter((target) => !seen.has(target.path) && seen.add(target.path));
}

/**
 * A path in a form two spellings of the same folder share: symlinks of its
 * longest existing part resolved, and case-folded where the file system
 * ignores case (macOS and Windows by default).
 */
export function canonicalPath(target, { platform, realpath = fs.realpathSync.native }) {
  const p = pathFor(platform);
  let head = p.resolve(target);
  const tail = [];
  for (;;) {
    try {
      head = realpath(head);
      break;
    } catch {
      // Not there (yet): resolve the part that exists.
      const parent = p.dirname(head);
      if (parent === head) break;
      tail.unshift(p.basename(head));
      head = parent;
    }
  }
  const joined = p.join(head, ...tail);
  return platform === "darwin" || platform === "win32" ? joined.toLowerCase() : joined;
}

function containsOrIs(container, inner, p) {
  const relative = p.relative(container, inner);
  return relative === "" || (!relative.startsWith("..") && !p.isAbsolute(relative));
}

const isFileSystemRoot = (resolved, p) => p.parse(resolved).root === resolved;

/**
 * Whether an ERUDI_DATA_ROOT is too broad for Clear All Data: a file system
 * or drive root, the home folder, or a folder above it.
 */
export function isUnsafeInstanceRoot(root, { platform, home, realpath }) {
  const p = pathFor(platform);
  const canonicalRoot = canonicalPath(root, { platform, realpath });
  if (isFileSystemRoot(canonicalRoot, p)) return true;
  return containsOrIs(canonicalRoot, canonicalPath(home, { platform, realpath }), p);
}

/**
 * Last check before a recursive delete, on canonical paths: the target is
 * absolute, and neither a file system root, the home folder or one of its
 * ancestors, nor a root Erudi does not own (the ERUDI_DATA_ROOT folder, the
 * Electron user data folder) or one of their ancestors.
 */
export function isSafeToDelete(target, { platform, home, protectedDirs, realpath }) {
  const p = pathFor(platform);
  if (typeof target !== "string" || !target || !p.isAbsolute(target)) return false;
  const canonicalTarget = canonicalPath(target, { platform, realpath });
  if (isFileSystemRoot(canonicalTarget, p)) return false;
  return ![home, ...protectedDirs]
    .filter(Boolean)
    .some((dir) => containsOrIs(canonicalTarget, canonicalPath(dir, { platform, realpath }), p));
}
