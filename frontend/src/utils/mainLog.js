// Levelled records for the Electron main process's own log lines.
//
// The app log (`erudi-backend.log`) has no level column: a line is
// `[<ISO>] <text>`, and the Diagnostics page keeps a record only when its
// text states a level (see appLogTail.js). Main's own lines therefore need to
// say their level themselves, in the same shape the renderer bridge uses --
// `[main] ERROR <message>` -- or a backend that died, a renderer that crashed
// and a spawn that failed never reach the page that exists to report them.
//
// Pure: main.js cannot be imported by the test runner, so what can be tested
// lives here and main.js only wires it to its `log()`.

/** Characters kept of an attached error's text; a stack past this says nothing new. */
export const MAX_ERROR_CHARS = 4000;

/**
 * The text of whatever a handler was handed as the error: the stack when
 * there is one (it carries the message), else the message, else the value
 * itself. Never throws.
 * @param {*} error - An Error, a rejection reason, or nothing.
 * @returns {string} "" when there is nothing to say.
 */
export function errorText(error) {
  if (error === null || error === undefined) return "";
  let text;
  try {
    if (typeof error === "string") text = error;
    else if (typeof error.stack === "string" && error.stack) text = error.stack;
    else if (typeof error.message === "string" && error.message) text = error.message;
    else text = String(error);
  } catch {
    // A value whose String() throws still gets a record, just an anonymous one.
    return "[unprintable error]";
  }
  return text.length > MAX_ERROR_CHARS
    ? `${text.slice(0, MAX_ERROR_CHARS)}... [+${text.length - MAX_ERROR_CHARS}]`
    : text;
}

/**
 * One levelled main-process record: `[main] LEVEL message`, with the error's
 * text on the following lines when one is attached, so the reader that groups
 * continuation lines under their header keeps the stack with its record.
 * @param {"WARN"|"ERROR"} level - The level the record states.
 * @param {string} message - What failed, on what.
 * @param {*} [error] - The error object, when there is one.
 * @returns {string} The text to hand to main's `log()`.
 */
export function formatMainRecord(level, message, error) {
  const head = `[main] ${level} ${message}`;
  const detail = errorText(error);
  return detail ? `${head}\n${detail}` : head;
}

/** ANSI SGR sequences (colours, bold, reset). */
// eslint-disable-next-line no-control-regex -- the ESC of a colour code is what is matched
const ANSI_SGR_RE = /\x1b\[[0-9;]*m/g;

/**
 * The text without its terminal colour codes. The backend's console formatter
 * colours its lines even when stdout is a pipe; main strips them before
 * echoing the backend's stdout/stderr into the app log, which is read as text
 * and pasted into bug reports.
 * @param {string} text - A line as the backend wrote it.
 * @returns {string} The same line, uncoloured.
 */
export function stripAnsi(text) {
  return String(text ?? "").replace(ANSI_SGR_RE, "");
}

/**
 * Describe a Chromium process that is gone (`child-process-gone`,
 * `render-process-gone`), naming the kind, the reason and the exit code.
 * @param {string} kind - "renderer", or the `type` of a child process (GPU, Utility, ...).
 * @param {{reason?: string, exitCode?: number, name?: string, serviceName?: string}} details - Electron's details object.
 * @returns {string} One line.
 */
export function describeProcessGone(kind, details = {}) {
  const parts = [`${kind} process gone: ${details.reason ?? "unknown reason"}`];
  if (details.exitCode !== undefined && details.exitCode !== null) {
    parts.push(`exit code ${details.exitCode}`);
  }
  const name = details.name || details.serviceName;
  if (name) parts.push(name);
  return parts.join(", ");
}
