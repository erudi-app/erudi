/**
 * The update policy (#563): every decision the main process takes about
 * electron-updater, as pure functions main.js only wires.
 *
 * Three switches, persisted in the user settings and handed to main by the
 * renderer: Check for updates (notify only), Download updates automatically,
 * Install downloaded updates when I quit. The two opt-ins only act while the
 * check is on. `updaterWiring.js` applies these decisions to the library.
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
 *   Squirrel says it holds the update with its own `update-downloaded`; until
 *   then the hand-off is only requested, and it can still fail.
 */

/**
 * Nothing armed yet. `quitHandler`: the library's quit handler is registered
 * (Windows, Linux). `handOffRequestedVersion`: Squirrel.Mac was asked to fetch
 * that version. `handedOffVersion`: Squirrel.Mac holds that version.
 */
export const NO_INSTALL_ARMED = Object.freeze({
  quitHandler: false,
  handOffRequestedVersion: null,
  handedOffVersion: null,
});

/**
 * The effective switches from the snapshot the renderer sent.
 *
 * A switch is `true`, `false`, or unknown (`null`, missing, anything else) when
 * the settings could not be read. Unknown means off for all three: no request
 * leaves without a known yes, and the next launch reads the settings again.
 *
 * @param {{check?: unknown, download?: unknown, install?: unknown} | null | undefined} snapshot
 * @returns {{check: boolean, download: boolean, install: boolean}}
 */
export function resolveUpdatePreferences(snapshot) {
  const check = snapshot?.check === true;
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
 * @param {boolean} input.automaticDownload Whether a download started by a
 *   check (Download updates automatically) is running.
 * @returns {{
 *   autoDownload: boolean,
 *   autoInstallOnAppQuit: boolean,
 *   startPolling: boolean,
 *   stopPolling: boolean,
 *   armInstall: "quit-handler" | "squirrel" | null,
 *   cancelDownload: boolean,
 * }} `armInstall` names how to arm the install of the update already on
 *   disk: register the library's quit handler (Windows, Linux) or hand the
 *   update to Squirrel (macOS). `cancelDownload`: the effective Download
 *   switch went off while an automatic download runs.
 */
export function planPreferenceChange({
  preferences,
  platform,
  downloadedVersion,
  armed,
  polling,
  automaticDownload,
}) {
  let armInstall = null;
  if (preferences.install && downloadedVersion) {
    if (platform === "darwin") {
      const asked =
        armed.handedOffVersion === downloadedVersion ||
        armed.handOffRequestedVersion === downloadedVersion;
      armInstall = asked ? null : "squirrel";
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
    cancelDownload: automaticDownload && !preferences.download,
  };
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
    case "error":
    case "download-cancelled":
      // A download that failed or was cancelled: the update is still there.
      return state?.phase === "downloading"
        ? { phase: "available", version: state.version }
        : state;
    default:
      return state;
  }
}

/**
 * The updater state main reports (`updater:get-state`) and the Update Erudi
 * card keeps, after an updater event.
 *
 * `{phase, version, percent, downloadedVersion}`; phases: "idle" |
 * "checking" | "available" | "up-to-date" | "downloading" | "downloaded" |
 * "error". Once a version is on disk (`downloadedVersion`), the periodic
 * check cannot take it back to "available" for that same version, nor to
 * "checking", "up-to-date" or "error": the update can still be installed.
 * A check during a download of that same version leaves it downloading (the
 * library joins the download in flight). A download that fails or is
 * cancelled goes back to "available".
 *
 * @param {{phase: string, version: string | null, percent: number,
 *   downloadedVersion: string | null}} state
 * @param {{event: string, version?: string, percent?: number}} payload
 * @returns {object} The next state (the same object when unchanged).
 */
export function updaterStateAfter(state, payload) {
  const onDisk = state.downloadedVersion ?? null;
  switch (payload.event) {
    case "checking-for-update":
      return onDisk || state.phase === "downloading" ? state : { ...state, phase: "checking" };
    case "update-available":
      if (state.phase === "downloading" && payload.version === state.version) {
        return state;
      }
      if (onDisk && payload.version === onDisk) {
        return state.phase === "downloaded"
          ? state
          : { ...state, phase: "downloaded", version: onDisk, percent: 100 };
      }
      return { ...state, phase: "available", version: payload.version ?? null, percent: 0 };
    case "update-not-available":
      return onDisk ? state : { ...state, phase: "up-to-date", version: null, percent: 0 };
    case "download-progress":
      return { ...state, phase: "downloading", percent: payload.percent ?? 0 };
    case "update-downloaded":
      return {
        ...state,
        phase: "downloaded",
        version: payload.version ?? null,
        percent: 100,
        downloadedVersion: payload.version ?? null,
      };
    case "error":
    case "download-cancelled":
      if (state.phase === "downloading") {
        return { ...state, phase: "available", percent: 0 };
      }
      if (payload.event === "download-cancelled" || onDisk) {
        return state;
      }
      return { ...state, phase: "error", percent: 0 };
    default:
      return state;
  }
}
