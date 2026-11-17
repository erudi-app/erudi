// The default file name of an exported conversation (#430): its title made safe
// as a file name on macOS, Windows and Linux, then the date. The Electron main
// process applies it to whatever the renderer sends, so it holds for any input.

const MAX_NAME_CHARACTERS = 80;
const FALLBACK_NAME = "conversation";
// What no common file system accepts in a name -- control characters (C0 and
// C1) and the characters Windows reserves, the path separators among them --
// and what hides in a name: zero-width and bidirectional formatting characters
// (a name that reads one way and ends in another).
// Windows device names (CON, NUL, COM1...) need no check: the name always ends
// with a date, and a device name is only one when it is the whole base name.
/* eslint-disable no-control-regex -- the control characters are the point */
const FORBIDDEN =
  /[\u0000-\u001f\u007f-\u009f<>:"/\\|?*\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]/g;
/* eslint-enable no-control-regex */

/**
 * A title as a file name: forbidden characters become spaces, spaces fold,
 * dot-only words and leading/trailing dots go (no `..`, no hidden file, no
 * trailing dot Windows drops), and the length is capped without cutting a
 * character in two.
 *
 * @param {unknown} title
 * @returns {string} The name, "" when nothing usable is left.
 */
export function sanitizeFileName(title) {
  if (typeof title !== "string") {
    return "";
  }
  const words = title
    .replace(FORBIDDEN, " ")
    .split(/\s+/)
    .filter((word) => word && !/^\.+$/.test(word));
  const trimDots = (text) => text.replace(/^[.\s]+|[.\s]+$/g, "");
  const name = trimDots(words.join(" "));
  return trimDots([...name].slice(0, MAX_NAME_CHARACTERS).join(""));
}

const pad = (value) => String(value).padStart(2, "0");

/**
 * The name the save dialog proposes: "<title> <YYYY-MM-DD>.md", the date being
 * the local day of the export.
 *
 * @param {unknown} title
 * @param {Date} [now]
 * @returns {string}
 */
export function defaultExportName(title, now = new Date()) {
  const day = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
  return `${sanitizeFileName(title) || FALLBACK_NAME} ${day}.md`;
}
