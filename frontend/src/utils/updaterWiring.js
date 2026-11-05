/**
 * The updater wiring (#563, #572): everything main.js does with
 * electron-updater, taking the updater as an argument so it runs, in tests,
 * against fakes that behave like the library (`updaterWiring.test.js`). The
 * decisions come from `updaterPolicy.js`; this module applies them in the order
 * the library needs and keeps the state main reports to the window.
 *
 * Library facts it relies on (electron-updater 6.8, read in its sources):
 * - `checkForUpdates()` downloads what it finds only while `autoDownload` is
 *   on, and returns that download's cancellation token (AppUpdater
 *   `doCheckForUpdates`).
 * - Windows and Linux (`BaseUpdater`): when a download finishes, the library
 *   registers its quit handler only while `autoInstallOnAppQuit` is on, and
 *   records it in `quitHandlerAdded`; `addQuitHandler()` does nothing while the
 *   flag is off, so the flags are always set before arming.
 * - macOS (`MacUpdater`): when a download finishes with the flag on, the
 *   library asks Squirrel.Mac (Electron's native `autoUpdater`) to fetch the
 *   update; Squirrel then emits its own `update-downloaded` and installs it at
 *   quit whatever the switches say from then on.
 */
import {
  NO_INSTALL_ARMED,
  isHandedOff,
  planPreferenceChange,
  resolveUpdatePreferences,
  updaterStateAfter,
} from "./updaterPolicy";

/**
 * The value sent as `x-user-staging-id`. The library sends a random id it keeps
 * per install (`userData/.updaterId`) on every request, for staged rollouts;
 * Erudi publishes no staged rollout, and the update request carries nothing
 * but the version and the platform.
 */
export const NEUTRAL_STAGING_ID = "00000000-0000-0000-0000-000000000000";

const CHECK_INTERVAL_MS = 4 * 60 * 60 * 1000;

/**
 * Set the updater up right after loading it, before anything can run: both
 * flags off (the library's defaults are on; the wiring raises them from a
 * known snapshot only), and the per-install id replaced by a constant (the
 * library applies `requestHeaders` last, over its own header).
 * @param {object} updater electron-updater's `autoUpdater`.
 */
export function configureUpdater(updater) {
  updater.autoDownload = false;
  updater.autoInstallOnAppQuit = false;
  updater.requestHeaders = { ...updater.requestHeaders, "x-user-staging-id": NEUTRAL_STAGING_ID };
}

/**
 * Wire the updater's events and the user's switches.
 *
 * @param {object} deps
 * @param {object} deps.updater electron-updater's `autoUpdater`, configured.
 * @param {object | null} deps.nativeUpdater Electron's `autoUpdater` (Squirrel.Mac) on macOS.
 * @param {string} deps.platform `process.platform`.
 * @param {(event: string, payload: object) => void} deps.send Forward an event to the window.
 * @param {(message: string) => void} deps.log
 * @param {(message: string, error?: unknown) => void} deps.logWarn
 * @param {(info: object) => void} deps.onDownloaded Called with the library's
 *   info when a download finishes (main keeps the release notes).
 * @param {{setInterval: Function, clearInterval: Function}} [deps.timers]
 * @returns {{
 *   setPreferences: (snapshot: object) => void,
 *   checkNow: () => Promise<{ok: boolean, reason?: string}>,
 *   downloadNow: () => Promise<{ok: boolean, reason?: string}>,
 *   installNow: () => void,
 *   getState: () => object,
 * }}
 */
export function createUpdaterWiring({
  updater,
  nativeUpdater,
  platform,
  send,
  log,
  logWarn,
  onDownloaded,
  timers = { setInterval, clearInterval },
}) {
  let preferences = null;
  let pollTimer = null;
  let state = { phase: "idle", version: null, percent: 0, downloadedVersion: null };
  let armed = NO_INSTALL_ARMED;
  // The token of a download a check started on its own (Download updates
  // automatically), so turning that switch off can stop it.
  let automaticDownload = null;

  const apply = (payload) => {
    state = updaterStateAfter(state, payload);
  };

  // Every event both records the state and is forwarded on one channel.
  updater.on("checking-for-update", () => {
    log("Updater: checking for update...");
    apply({ event: "checking-for-update" });
    send("checking-for-update", {});
  });

  updater.on("update-available", (info) => {
    log(`Updater: update available - v${info.version}`);
    apply({ event: "update-available", version: info.version });
    // Whether a download follows on its own, so the banner says "downloading"
    // or offers Download, never the wrong one.
    send("update-available", { version: info.version, downloading: updater.autoDownload });
  });

  updater.on("update-not-available", () => {
    log("Updater: already on latest version.");
    apply({ event: "update-not-available" });
    send("update-not-available", {});
  });

  updater.on("download-progress", (progress) => {
    const percent = Math.round(progress.percent);
    log(`Updater: downloading... ${percent}%`);
    apply({ event: "download-progress", percent });
    send("download-progress", { percent });
  });

  // The library arms the install at quit right after this event, from the
  // value autoInstallOnAppQuit holds now.
  updater.on("update-downloaded", (info) => {
    log(`Updater: v${info.version} downloaded, ready to install.`);
    apply({ event: "update-downloaded", version: info.version });
    if (updater.autoInstallOnAppQuit) {
      if (platform === "darwin") {
        armed = { ...armed, handOffRequestedVersion: info.version };
        log(`Updater: v${info.version} is being handed to the system installer`);
      } else {
        log(`Updater: v${info.version} will install when the app quits`);
      }
    }
    onDownloaded(info);
    send("update-downloaded", { version: info.version, handedOff: isHandedOff(armed) });
  });

  updater.on("error", (err) => {
    // Never crash the app over an update failure. The detail stays in the
    // log; the window is told only that it failed.
    logWarn("Updater error (non-fatal)", err);
    apply({ event: "error" });
    send("error", {});
  });

  // Squirrel.Mac holds the update: from now on it installs at quit whatever
  // the switches say. A refusal (signature, notarization) arrives as an
  // "error" from MacUpdater instead, and nothing is recorded.
  nativeUpdater?.on("update-downloaded", () => {
    const version = armed.handOffRequestedVersion ?? state.downloadedVersion;
    armed = { ...armed, handedOffVersion: version };
    log(`Updater: the system installer holds v${version}; it installs when the app quits`);
    send("install-handed-off", { version });
  });

  const trackAutomaticDownload = (result) => {
    // A check during a download joins it (AppUpdater `downloadUpdate` returns
    // the download in flight and ignores the new token): the first token is
    // still the one that stops it.
    if (!result?.downloadPromise || automaticDownload !== null) return;
    const token = result.cancellationToken;
    automaticDownload = token;
    result.downloadPromise.then(
      () => {
        if (automaticDownload === token) automaticDownload = null;
      },
      () => {
        if (automaticDownload === token) automaticDownload = null;
        if (token?.cancelled) {
          log("Updater: the automatic download was cancelled");
          apply({ event: "download-cancelled" });
          send("download-cancelled", { version: state.version });
        }
        // Any other failure was emitted as "error" by the library and
        // handled (logged, forwarded) there.
      }
    );
  };

  const runCheck = async (failure) => {
    try {
      trackAutomaticDownload(await updater.checkForUpdates());
      return { ok: true };
    } catch (err) {
      logWarn(failure, err);
      return { ok: false, reason: "error" };
    }
  };

  const armInstall = (how) => {
    const version = state.downloadedVersion;
    if (how === "quit-handler") {
      if (typeof updater.addQuitHandler !== "function") {
        logWarn(
          `Updater: this electron-updater has no addQuitHandler; v${version} will not install at quit`
        );
        return;
      }
      updater.addQuitHandler();
      if (updater.quitHandlerAdded) {
        log(`Updater: v${version} will install when the app quits`);
      }
      return;
    }
    // MacUpdater keeps serving the download to Squirrel.Mac when install was
    // off; asking Squirrel to check again fetches it from there.
    try {
      nativeUpdater.checkForUpdates();
    } catch (err) {
      logWarn(`Updater: could not hand v${version} to the system installer`, err);
      return;
    }
    armed = { ...armed, handOffRequestedVersion: version };
    log(`Updater: v${version} is being handed to the system installer`);
  };

  return {
    // Renderer -> main: the three switches, at boot and on every change.
    setPreferences(snapshot) {
      preferences = resolveUpdatePreferences(snapshot);
      const plan = planPreferenceChange({
        preferences,
        platform,
        downloadedVersion: state.downloadedVersion,
        armed: { ...armed, quitHandler: updater.quitHandlerAdded === true },
        polling: pollTimer !== null,
        automaticDownload: automaticDownload !== null,
      });
      // Flags first: the library reads them when arming.
      updater.autoDownload = plan.autoDownload;
      updater.autoInstallOnAppQuit = plan.autoInstallOnAppQuit;
      log(
        `Updater: switches check=${preferences.check} download=${preferences.download} ` +
          `install=${preferences.install}`
      );
      if (plan.cancelDownload) {
        automaticDownload.cancel();
      }
      if (plan.armInstall) {
        armInstall(plan.armInstall);
      }
      if (plan.stopPolling) {
        timers.clearInterval(pollTimer);
        pollTimer = null;
      }
      if (!preferences.check) {
        log("Updater: Check for updates is off; no check will run");
        return;
      }
      if (!plan.startPolling) {
        return; // already checking on its cadence
      }
      log("Updater: Check for updates is on; checking now, then every 4 hours");
      runCheck("Updater: initial check failed");
      pollTimer = timers.setInterval(
        () => runCheck("Updater: periodic check failed"),
        CHECK_INTERVAL_MS
      );
    },

    // A check the user asked for. It runs whatever Check for updates says;
    // what it does with a version it finds is the effective Download switch
    // (off, or not known yet: no download without a second click). It never
    // arms an install at quit.
    checkNow() {
      updater.autoDownload = preferences?.download === true;
      return runCheck("Updater: a check the user asked for failed");
    },

    // The second click: download the update a check found. Not cancelled by
    // the switches: the user asked for it.
    async downloadNow() {
      try {
        await updater.downloadUpdate();
        return { ok: true };
      } catch (err) {
        logWarn("Updater: a download the user asked for failed", err);
        return { ok: false, reason: "error" };
      }
    },

    installNow() {
      updater.quitAndInstall(false, true);
    },

    // What the updater last said, for a window that mounts after the events.
    // `handedOff`: Squirrel.Mac holds an update and installs it at quit.
    getState() {
      return {
        available: true,
        phase: state.phase,
        version: state.version,
        percent: state.percent,
        downloadedVersion: state.downloadedVersion,
        handedOff: isHandedOff(armed),
      };
    },
  };
}
