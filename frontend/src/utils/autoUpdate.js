/**
 * Bridge the persisted update switches to the Electron main process (#563).
 *
 * electron-updater lives in main, which has no access to the database; the
 * three switches (Check for updates, Download updates automatically, Install
 * downloaded updates when I quit) live in the user-settings singleton, which
 * only the renderer reads. This carries the full snapshot across: main holds
 * every check until it arrives, and decides what to do with it in
 * `updaterPolicy.js`.
 *
 * A switch that cannot be read travels as `null`. Main then downloads and
 * installs nothing, and keeps the notify-only check, so a backend hiccup
 * neither fetches an installer nobody agreed to nor hides a new version.
 */
import { createLogger } from "./logger";

const log = createLogger("autoUpdate");

const UNKNOWN = Object.freeze({ check: null, download: null, install: null });

const known = (value) => (typeof value === "boolean" ? value : null);

/**
 * The snapshot main expects, from a user-settings payload.
 * @param {object | null | undefined} settings - `GET /user_settings/` or a PUT answer.
 * @returns {{check: boolean | null, download: boolean | null, install: boolean | null}}
 */
export function updatePreferencesFrom(settings) {
  return {
    check: known(settings?.auto_update_enabled),
    download: known(settings?.auto_download_updates),
    install: known(settings?.auto_install_updates),
  };
}

/**
 * Read the persisted switches and push them to main.
 * @param {{get: Function}} client - API client (apiClient in the app).
 * @returns {Promise<object>} The snapshot that was sent.
 */
export async function syncUpdatePreferencesWithMain(client) {
  let snapshot = UNKNOWN;
  try {
    snapshot = updatePreferencesFrom(await client.get("/user_settings/"));
  } catch (error) {
    log.warn("Could not read the update settings; nothing will download or install", error);
  }
  notifyUpdatePreferences(snapshot);
  return snapshot;
}

/**
 * Push a snapshot to main without reading it back from the backend. Used by
 * the Settings card, which already holds the values it just persisted.
 * @param {{check: boolean | null, download: boolean | null, install: boolean | null}} snapshot
 */
export function notifyUpdatePreferences(snapshot) {
  window.updaterAPI?.setPreferences?.(snapshot);
}
