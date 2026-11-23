// Download lifecycle helpers shared by the download modal context and its consumers.
//
// A user-initiated cancel was being reported through the same `onError` callback as
// a real failure, so cancelling a download showed a misleading "Download failed"
// dialog (#133). Centralizing the status string + the error-vs-cancel decision here
// keeps the two paths distinct and testable.

import i18n from "../i18n";

/** The status a cancelled download job reports (backend + poll path). */
export const DOWNLOAD_CANCELLED = "cancelled";

/**
 * Client-side status for a job that stopped making progress in a NON-terminal
 * state (#315). Never reported by the backend: the poll gives up on it so the
 * UI cannot spin forever waiting for a response that may never come.
 */
export const DOWNLOAD_STALLED = "stalled";

/**
 * Shown when the transfer finished but the job never reached a terminal state.
 * Deliberately not phrased as a failure: the weights are on disk, only the
 * finalization bookkeeping did not complete (#291), so telling the user the
 * download failed would be wrong. Resolved at call time so it follows the
 * active app language (#385).
 */
export function downloadStalledMessage() {
  return i18n.t("downloads:errors.stalled");
}

/**
 * The phases the download widget renders (#292). `stalled` is the client-side
 * "finalization never came back" state (#315); every other phase maps to a
 * backend status, with `running` split at 100% into downloading / finalizing.
 */
export const DOWNLOAD_PHASES = [
  "queued",
  "downloading",
  "finalizing",
  "completed",
  "cancelled",
  "failed",
  "stalled",
];

/**
 * Derive the widget phase from the raw job state the context tracks.
 *
 * Order matters: a cancel that failed to reach the backend leaves a detail in
 * `errorMessage` but is still a cancellation, whereas an error on a job that is
 * not yet terminal (failed start, poll network error) IS a failure.
 */
export function deriveDownloadPhase({ status, progress = 0, errorMessage = "" }) {
  if (status === DOWNLOAD_STALLED) return "stalled";
  if (status === "failed") return "failed";
  if (status === DOWNLOAD_CANCELLED) return "cancelled";
  if (status === "completed") return "completed";
  if (errorMessage) return "failed";
  if (status === "running") return (progress ?? 0) >= 100 ? "finalizing" : "downloading";
  return "queued";
}
