// The two start-up sequences whose order Clear All Data depends on, kept out
// of main.js so the order is tested on fakes rather than read from the source.

import path from "node:path";

/**
 * An ERUDI_DATA_ROOT instance keeps its Electron side under the same root
 * (<root>/electron: the Chromium profile, Erudi's files beside it, the
 * Electron logs), so a run next to the installed app shares nothing with it,
 * neither its storage nor its single-instance lock. Set before
 * electron-updater loads (it reads these paths), then the updater is loaded.
 * Without a root the logs folder is Electron's default, set explicitly so
 * that app.getPath("logs") is defined for Clear All Data.
 *
 * Under a root, both folders are claimed first (utils/folderOwnership.js):
 * setAppLogsPath creates its folder, and Chromium the user data folder, so a
 * later claim could not tell a folder Erudi created from one that was there.
 * The logs folder is claimed only inside an electron folder that is Erudi's.
 *
 * @param {object} input
 * @param {Electron.App} input.app
 * @param {string|null} input.electronDir
 * @param {(dir: string) => boolean} [input.claim] Claims a folder, answers
 *   whether it is Erudi's.
 * @param {() => *} input.loadUpdater
 * @returns {*} What loadUpdater returns.
 */
export function setUpElectronPaths({ app, electronDir, claim = () => false, loadUpdater }) {
  if (electronDir) {
    const logsDir = path.join(electronDir, "logs");
    if (claim(electronDir)) claim(logsDir);
    app.setPath("userData", electronDir);
    app.setAppLogsPath(logsDir);
  } else {
    app.setAppLogsPath(undefined);
  }
  return loadUpdater();
}

/**
 * At ready: finish a Clear All Data the previous session left unfinished
 * before the window opens and anything writes again, then the menu and the
 * window. A failure there never keeps the window from opening.
 */
export async function openFirstWindow({
  finishPendingClear,
  createApplicationMenu,
  createWindow,
  logError = () => {},
}) {
  try {
    await finishPendingClear();
  } catch (error) {
    logError("Could not finish the Clear All Data of the previous session", error);
  }
  createApplicationMenu();
  createWindow();
}
