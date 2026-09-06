// Reading the tail of `erudi-backend.log` for the Diagnostics page.
//
// The app log is not the backend log. Its lines are `[<ISO>] <text>`, written
// by main.js, and `<text>` has three origins: main's own messages, the
// backend's stdout (each line already carrying the backend formatter's own
// `[LEVEL]`), and renderer entries forwarded over the `renderer-log` IPC as
// `[renderer:<ns>] <LEVEL> <msg>`. There is no level column of its own.
//
// So the level is read out of the text, and only when the text states one:
//   - `[WARNING]` / `[ERROR]` / `[CRITICAL]` — the backend's file format;
//   - `[renderer:ns] WARN|ERROR ...` — the renderer bridge's format;
//   - `[main] WARN|ERROR ...` — main's own failures (utils/mainLog.js): a
//     backend that exited, a spawn that failed, a renderer that crashed.
// A line that declares no level is dropped. That is deliberate and it is the
// conservative direction: the backend logs conversation content at INFO on
// purpose (docs/privacy.md), and this text is written to be pasted into a
// public issue. Default-exclude means a format we have not taught this parser
// about is left out rather than leaked; main's lifecycle chatter (`Creating
// main window...`) stays out the same way.
//
// Records span lines: an Electron stack trace or a dumped HTTP header block is
// appended to the file as one write with embedded newlines. A record therefore
// starts at a line beginning with a bracketed ISO timestamp and absorbs the
// lines after it. Continuations inherit their header's level, so those under a
// dropped record are dropped with it — including a leading fragment whose
// header fell outside the read window.

// `import` rather than `require`: this module uses ESM exports (main.js
// requires it the way it requires the other shared helpers, and webpack bridges
// the two), and mixing `require` into an ESM module is the one combination
// neither bundler nor test runner is obliged to support.
import fs from "node:fs";

/** Read window. Large enough for hundreds of records, small enough to be free. */
export const DEFAULT_TAIL_BYTES = 512 * 1024;

/** Records returned by default, newest last. */
export const DEFAULT_LIMIT = 200;

/** Per-record cap, so one header dump cannot dominate the payload. */
export const MAX_MESSAGE_CHARS = 4000;

/** Start of a record: `[2026-09-05T23:03:14.797Z] `. */
const RECORD_RE = /^\[(\d{4}-\d{2}-\d{2}T[\d:.]+Z)\]\s?([\s\S]*)$/;

/** The backend's own level marker, as it survives inside a stdout line. */
const BACKEND_LEVEL_RE = /\[(WARNING|ERROR|CRITICAL)\]/;

/**
 * The shape of the two levelled writers of this process: the renderer bridge
 * (`[renderer:ns] LEVEL message`) and main itself (`[main] LEVEL message`).
 */
const LEVELLED_PREFIX_RE = /^\[(?:renderer:[^\]]*|main)\]\s+(WARN|ERROR)\b/;

/**
 * The level a record declares, or null when it declares none.
 * @param {string} text - Record body, timestamp already stripped.
 * @returns {string|null} "WARNING", "ERROR" or "CRITICAL".
 */
export function detectLevel(text) {
  const levelled = LEVELLED_PREFIX_RE.exec(text);
  if (levelled) return levelled[1] === "WARN" ? "WARNING" : "ERROR";
  const backend = BACKEND_LEVEL_RE.exec(text);
  if (backend) return backend[1];
  return null;
}

/**
 * Group raw app-log text into records and keep the levelled ones.
 * @param {string} text - Raw log text.
 * @param {number} limit - Maximum records to return (newest kept).
 * @returns {Array<{timestamp: string, level: string, message: string}>} Oldest first.
 */
export function parseAppLogRecords(text, limit = DEFAULT_LIMIT) {
  const records = [];
  let current = null;

  // `split("\n")`, never a regex that also breaks on other separators: a
  // record boundary is a newline and nothing else, and the app log carries
  // user content. `log()` writes "\n" explicitly and the backend's stdout is
  // already split on /\r?\n/ and trimmed before it gets here, so this file
  // never contains CRLF the way the backend's own log does on Windows.
  const lines = String(text ?? "").split("\n");
  if (lines.length && lines[lines.length - 1] === "") {
    // The file's final newline is a terminator, not a blank line; leaving it
    // in would append a newline to the last record's message.
    lines.pop();
  }

  for (const line of lines) {
    const match = RECORD_RE.exec(line);
    if (match) {
      const level = detectLevel(match[2]);
      if (!level) {
        current = null;
        continue;
      }
      current = { timestamp: match[1], level, message: match[2] };
      records.push(current);
    } else if (current) {
      current.message += `\n${line}`;
    }
    // else: a continuation of a dropped record, or of one whose header fell
    // outside the window. Both are dropped.
  }

  for (const record of records) {
    if (record.message.length > MAX_MESSAGE_CHARS) {
      const dropped = record.message.length - MAX_MESSAGE_CHARS;
      record.message = `${record.message.slice(0, MAX_MESSAGE_CHARS)}... [+${dropped}]`;
    }
  }

  return limit ? records.slice(-limit) : records;
}

/**
 * Read at most the last `maxBytes` of a file, dropping the partial first line.
 * Never throws: an unreadable file reads as "".
 * @param {string} filePath - Absolute path to read.
 * @param {number} maxBytes - Size of the window.
 * @returns {string} Decoded window.
 */
export function readTailSync(filePath, maxBytes = DEFAULT_TAIL_BYTES) {
  let handle = null;
  try {
    const { size } = fs.statSync(filePath);
    const start = Math.max(0, size - maxBytes);
    const length = size - start;
    if (length <= 0) return "";
    const buffer = Buffer.allocUnsafe(length);
    handle = fs.openSync(filePath, "r");
    const read = fs.readSync(handle, buffer, 0, length, start);
    const text = buffer.subarray(0, read).toString("utf8");
    if (start === 0) return text;
    const cut = text.indexOf("\n");
    return cut === -1 ? "" : text.slice(cut + 1);
  } catch {
    // A missing, locked or rotating log is not an error here: the panel shows
    // what it can and says the rest is unavailable.
    return "";
  } finally {
    if (handle !== null) {
      try {
        fs.closeSync(handle);
      } catch {
        /* already closed */
      }
    }
  }
}

/**
 * The last `limit` levelled records of the app log.
 * @param {string} filePath - Absolute path to `erudi-backend.log`.
 * @param {number} limit - Maximum records to return.
 * @returns {Array<{timestamp: string, level: string, message: string}>} Oldest first.
 */
export function readAppLogTail(filePath, limit = DEFAULT_LIMIT) {
  return parseAppLogRecords(readTailSync(filePath), limit);
}
