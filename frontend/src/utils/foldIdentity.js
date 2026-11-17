// What makes two log records "the same" for the Diagnostics page's repeat
// count (#534). Shared by the app-log reader (main process, appLogTail.js) and
// the merge (renderer, diagnosticsReport.js), so both fold on one rule.
//
// Pure and dependency-free: the renderer bundle must not pull `node:fs`.

/**
 * The backend formatter's header inside an echoed stdout/stderr line:
 * `Backend stdout: [WARNING] 2026-09-13T01:45:36.652Z [be-1f2e3d4c] - ...`.
 * The timestamp and the request id differ on every occurrence of the same
 * record, so they are left out of its identity.
 */
const ECHOED_BACKEND_HEADER_RE =
  /^(Backend (?:stdout|stderr): \[[A-Z]+\]) \d{4}-\d{2}-\d{2}T[\d:.]+Z \[[^\]]*\]/;

/**
 * The text two records must share to fold into one: the message, minus the
 * per-occurrence header of a backend line echoed into the app log.
 * @param {string} message - A record's message.
 * @returns {string} Its identity text.
 */
export function messageIdentity(message) {
  return String(message ?? "").replace(ECHOED_BACKEND_HEADER_RE, "$1");
}
