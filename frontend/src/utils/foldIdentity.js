// What makes two log records "the same" for the Diagnostics page's repeat
// count (#534). Shared by the app-log reader (main process, appLogTail.js) and
// the merge (renderer, diagnosticsReport.js), so both fold on one rule.
//
// Pure and dependency-free: the renderer bundle must not pull `node:fs`.

/**
 * The backend formatter's header inside an echoed stdout/stderr line, as the
 * console formatter writes it (`[WARN]  2026-11-17T13:42:25.689Z [be-1] - ...`,
 * padded after `[WARN]`/`[INFO]`) or the file format does (`[WARNING] ...`).
 * The timestamp and the request id differ on every occurrence of the same
 * record, so they are left out of its identity.
 */
const ECHOED_BACKEND_HEADER_RE =
  /^(Backend (?:stdout|stderr): \[[A-Z]+\])\s+\d{4}-\d{2}-\d{2}T[\d:.]+Z \[[^\]]*\]/;

/** ANSI colour codes, present in an app log written before main stripped them. */
// eslint-disable-next-line no-control-regex -- the ESC of a colour code is what is matched
const ANSI_SGR_RE = /\x1b\[[0-9;]*m/g;

/**
 * The text two records must share to fold into one: the message, without
 * colour codes and without the per-occurrence header of a backend line echoed
 * into the app log.
 * @param {string} message - A record's message.
 * @returns {string} Its identity text.
 */
export function messageIdentity(message) {
  return String(message ?? "")
    .replace(ANSI_SGR_RE, "")
    .replace(ECHOED_BACKEND_HEADER_RE, "$1");
}
