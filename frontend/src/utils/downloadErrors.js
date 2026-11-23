// The sentence the download widget shows for a failed download (#780).
//
// A failed job carries `error_code` (DOWNLOAD_ERROR_CODES in
// backend/src/domains/llms/download_errors.py) beside its English
// `error_message`. The English text is for the log and Diagnostics; the user
// reads the code's sentence in the interface language.

import i18n from "../i18n";

const GENERIC = "downloads:errors.failedUnexpectedly";

/** Every code the backend stores, and its sentence. */
export const DOWNLOAD_ERROR_KEYS = {
  offline: "downloads:errors.codes.offline",
  disk_full: "downloads:errors.codes.diskFull",
  gated: "downloads:errors.codes.gated",
  not_found: "downloads:errors.codes.notFound",
  rate_limited: "downloads:errors.codes.rateLimited",
  hub_unavailable: "downloads:errors.codes.hubUnavailable",
  interrupted: "downloads:errors.codes.interrupted",
  no_engine_artifact: "downloads:errors.codes.noEngineArtifact",
  raw_gguf_chunks: "downloads:errors.codes.rawGgufChunks",
  incomplete_split: "downloads:errors.codes.incompleteSplit",
  unsafe_file_name: "downloads:errors.codes.unsafeFileName",
  not_runnable: "downloads:errors.codes.notRunnable",
  integrity_failed: "downloads:errors.codes.integrityFailed",
  unexpected: GENERIC,
};

/**
 * The widget's sentence for a failed job's `error_code`, in the interface
 * language. A code this interface does not know (a newer backend) and a job
 * without one (failed before codes were stored) get the generic sentence,
 * never the English message.
 * @param {string|null|undefined} code
 * @returns {string}
 */
export function downloadFailureMessage(code) {
  return i18n.t(Object.hasOwn(DOWNLOAD_ERROR_KEYS, code) ? DOWNLOAD_ERROR_KEYS[code] : GENERIC);
}
