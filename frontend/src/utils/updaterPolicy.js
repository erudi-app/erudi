/**
 * The update policy (#563): every decision the main process takes about
 * electron-updater, as pure functions main.js only wires.
 *
 * Three switches, persisted in the user settings and handed to main by the
 * renderer: Check for updates (notify only), Download updates automatically,
 * Install downloaded updates when I quit. The two opt-ins only act while the
 * check is on.
 *
 * electron-updater 6 holds two flags. `autoDownload` makes a check download
 * what it finds. `autoInstallOnAppQuit` is read at the moment a download
 * finishes, and that moment decides whether the install at quit is armed:
 * - Windows and Linux (`BaseUpdater`): the library registers its quit handler
 *   only when the flag is on at download. The handler reads the flag again at
 *   quit, so turning install off later is honoured. Turning it on after the
 *   download needs the handler registered by hand (`addQuitHandler`).
 * - macOS (`MacUpdater`): with the flag on at download, the update is handed to
 *   Squirrel.Mac (the system installer), which installs it at quit on its own
 *   and cannot be told otherwise afterwards. With the flag off, the library
 *   keeps the local server that serves the update to Squirrel listening, so
 *   turning install on later hands it over by asking Squirrel to check again.
 */

/** Nothing armed yet: no quit handler registered, nothing handed to Squirrel. */
export const NO_INSTALL_ARMED = Object.freeze({ quitHandler: false, handedOffVersion: null });

/**
 * The effective switches from the snapshot the renderer sent.
 *
 * A switch is `true`, `false`, or unknown (`null`, missing, anything else) when
 * the settings could not be read. Unknown download or install means off:
 * nothing is fetched or installed without a known yes. Unknown check means
 * on: the check only notifies, and an unreadable setting must not strand an
 * install on an old version without a word.
 *
 * @param {{check?: unknown, download?: unknown, install?: unknown} | null | undefined} snapshot
 * @returns {{check: boolean, download: boolean, install: boolean}}
 */
export function resolveUpdatePreferences(snapshot) {
  const check = snapshot?.check !== false;
  return {
    check,
    download: check && snapshot?.download === true,
    install: check && snapshot?.install === true,
  };
}

/**
 * What main does when a preference snapshot arrives.
 *
 * @param {object} input
 * @param {{check: boolean, download: boolean, install: boolean}} input.preferences
 *   The effective switches (`resolveUpdatePreferences`).
 * @param {string} input.platform `process.platform`.
 * @param {string | null} input.downloadedVersion The version this process
 *   downloaded, still on disk, or null.
 * @param {{quitHandler: boolean, handedOffVersion: string | null}} input.armed
 *   What is already armed to install at quit.
 * @param {boolean} input.polling Whether the periodic check is running.
 * @returns {{
 *   autoDownload: boolean,
 *   autoInstallOnAppQuit: boolean,
 *   startPolling: boolean,
 *   stopPolling: boolean,
 *   armInstall: "quit-handler" | "squirrel" | null,
 * }} `armInstall` names how to arm the install of the update already on
 *   disk: register the library's quit handler (Windows, Linux) or hand the
 *   update to Squirrel (macOS).
 */
export function planPreferenceChange({ preferences, platform, downloadedVersion, armed, polling }) {
  let armInstall = null;
  if (preferences.install && downloadedVersion) {
    if (platform === "darwin") {
      armInstall = armed.handedOffVersion === downloadedVersion ? null : "squirrel";
    } else {
      armInstall = armed.quitHandler ? null : "quit-handler";
    }
  }
  return {
    autoDownload: preferences.download,
    autoInstallOnAppQuit: preferences.install,
    startPolling: preferences.check && !polling,
    stopPolling: !preferences.check && polling,
    armInstall,
  };
}

/**
 * What the library armed by itself when a download finished, from the value
 * `autoInstallOnAppQuit` had at that moment.
 *
 * Windows and Linux: the quit handler, once registered, stays for the life of
 * the process. macOS: Squirrel keeps the update it was handed; a later
 * download made with install off hands nothing over and leaves the earlier
 * one in place.
 *
 * @param {object} input
 * @param {string} input.platform `process.platform`.
 * @param {string} input.version The version that finished downloading.
 * @param {boolean} input.autoInstallOnAppQuit The flag at download time.
 * @param {{quitHandler: boolean, handedOffVersion: string | null}} input.armed
 * @returns {{quitHandler: boolean, handedOffVersion: string | null}}
 */
export function installArmedAfterDownload({ platform, version, autoInstallOnAppQuit, armed }) {
  if (!autoInstallOnAppQuit) {
    return armed;
  }
  if (platform === "darwin") {
    return { ...armed, handedOffVersion: version };
  }
  return { ...armed, quitHandler: true };
}

/**
 * Whether Squirrel.Mac holds an update, which it installs at quit whatever
 * the switches say from then on. Only macOS hands updates over.
 *
 * @param {{quitHandler: boolean, handedOffVersion: string | null}} armed
 * @returns {boolean}
 */
export function isHandedOff(armed) {
  return armed.handedOffVersion !== null;
}

/**
 * Whether the Settings card says an update may still install at quit: it was
 * handed to Squirrel.Mac, and the effective install switch is now off (install
 * turned off, or Check turned off).
 *
 * @param {object} input
 * @param {boolean} input.handedOff `isHandedOff` as main reported it.
 * @param {{install: boolean}} input.preferences The effective switches.
 * @returns {boolean}
 */
export function installMayStillHappen({ handedOff, preferences }) {
  return handedOff && !preferences.install;
}

/**
 * The banner's next state for an updater event.
 *
 * States: null (nothing to show), `{phase: "available", version}` (found,
 * nothing downloading: the banner offers Download), `{phase: "downloading",
 * version, percent}`, `{phase: "ready", version}` (on disk: Install now).
 * `update-available` carries `downloading: true` when a download follows the
 * check on its own, so the banner never says "available" for what is already
 * downloading, nor "downloading" for what is not. A later check that finds the
 * version already on disk leaves the banner ready.
 *
 * @param {object | null} state The current banner state.
 * @param {{event: string, version?: string, percent?: number, downloading?: boolean}} payload
 * @returns {object | null} The next state (the same object when unchanged).
 */
export function bannerStateAfter(state, payload) {
  switch (payload.event) {
    case "update-available":
      if (state?.phase === "ready" && state.version === payload.version) {
        return state;
      }
      return payload.downloading
        ? { phase: "downloading", version: payload.version, percent: 0 }
        : { phase: "available", version: payload.version };
    case "download-progress":
      return { phase: "downloading", version: state?.version ?? "", percent: payload.percent };
    case "update-downloaded":
      return { phase: "ready", version: payload.version };
    default:
      return state;
  }
}
