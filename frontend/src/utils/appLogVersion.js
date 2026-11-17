// The app log belongs to the app version that is running (#576).
//
// `erudi-backend.log` outlives updates and reinstalls, and the Diagnostics page
// lists its WARNING and ERROR records: left alone, it shows records an older
// version wrote about defects since fixed, as if they were today's. So at the
// first launch of another version, main moves the file to
// `erudi-backend.old.log` (the same single previous file its size cap uses) and
// starts a fresh one; the version of the last launch is kept beside it in
// `erudi-backend.log.version`, so a restart or a reinstall of the same version
// keeps its records. That file is rewritten at every launch: it lives in the
// temp folder like the log it describes, and temp cleaners remove files by
// age, so one written only on a switch would be removed before the log, and
// the next launch would then roll the running version's own log away. Kept
// beside the log rather than in the user data folder, the two go together.
//
// Only the packaged app that holds the single-instance lock switches
// (`appLogSwitchAllowed`): a second instance hands over and quits, and a dev
// run (`npm start`) shares the temp folder with an installed app. backend.log gets the same treatment from the backend's
// launcher (backend/src/launcher/log_version.py).
//
// The rename can fail (Windows: the file held by another program). The launch
// goes on, and the switch is recorded anyway: a switched log opens with a
// marker line, and the reader (appLogTail.js) drops every record above the last
// marker it sees.
//
// Pure apart from the `fs` it is handed: main.js cannot be imported by the test
// runner, so what can be tested lives here and main.js only wires it.

/**
 * The line that opens a version's app log. It carries no level, so it is never
 * listed itself; `parseAppLogRecords` cuts at it. The `[main] ` prefix keeps
 * the backend's own marker, echoed on its stdout, from matching.
 */
export const APP_LOG_VERSION_MARKER =
  "[main] Diagnostics starts here: the app log now belongs to app version";

/**
 * Give the app log to `version` when the last launch was another version.
 *
 * Never throws. Must run before main writes its first line.
 *
 * @param {object} options
 * @param {object} options.fs - `node:fs` (or a stand-in in tests).
 * @param {string} options.logFile - `erudi-backend.log`.
 * @param {string} options.oldLogFile - `erudi-backend.old.log`.
 * @param {string} options.versionFile - Where the last launch's version is kept.
 * @param {string} options.version - The app version now running.
 * @returns {{version: string, previous: string|null, rotated: boolean, error: string|null}|null}
 *   The switch, or null when there is none (no version, or the same version).
 */
export function switchAppLogForVersion({ fs, logFile, oldLogFile, versionFile, version }) {
  if (!version) return null;
  let previous = null;
  try {
    previous = fs.readFileSync(versionFile, "utf8").trim() || null;
  } catch {
    // Absent on a fresh install and before versioned logs existed; the log is
    // then rolled over, which is right for both.
  }
  if (previous === version) {
    try {
      fs.writeFileSync(versionFile, version);
    } catch {
      // Refreshing its age only; the file still says the right version.
    }
    return null;
  }

  let rotated = false;
  let error = null;
  try {
    if (fs.existsSync(logFile) && fs.statSync(logFile).size > 0) {
      fs.rmSync(oldLogFile, { force: true }); // rename() won't replace on Windows
      fs.renameSync(logFile, oldLogFile);
      rotated = true;
    }
  } catch (err) {
    error = String(err?.message ?? err);
  }
  try {
    fs.writeFileSync(versionFile, version);
  } catch (err) {
    // The next launch switches again; the marker still holds for this one.
    error = error ?? String(err?.message ?? err);
  }
  return { version, previous, rotated, error };
}

/**
 * Whether this launch may switch the app log: the packaged app holding the
 * single-instance lock, and nothing else.
 * @param {{isPackaged: boolean, hasSingleInstanceLock: boolean}} launch
 * @returns {boolean}
 */
export function appLogSwitchAllowed({ isPackaged, hasSingleInstanceLock }) {
  return Boolean(isPackaged && hasSingleInstanceLock);
}

/**
 * The lines main writes about its version once it can log: the marker and,
 * when the old file could not be moved, a WARN after it (so the page shows
 * it); then, on every launch, the version itself.
 *
 * @param {object|null} versionSwitch - What `switchAppLogForVersion` returned.
 * @param {string} version - The app version now running.
 * @returns {string[]} Lines for main's `log()`.
 */
export function appLogVersionLines(versionSwitch, version) {
  const lines = [];
  if (versionSwitch) {
    lines.push(
      `${APP_LOG_VERSION_MARKER} ${versionSwitch.version} (previous: ${versionSwitch.previous})`
    );
    if (versionSwitch.error) {
      lines.push(
        `[main] WARN Could not start a fresh app log for app version ${versionSwitch.version};` +
          ` the records of earlier versions stay above the marker: ${versionSwitch.error}`
      );
    }
  }
  if (version) lines.push(`Erudi starting, app version ${version}`);
  return lines;
}
