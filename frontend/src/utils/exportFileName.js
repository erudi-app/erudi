// The default file name of an exported conversation (#430): its title made safe
// as a file name on macOS, Windows and Linux, then the date. The Electron main
// process applies it to whatever the renderer sends, so it holds for any input.

const MAX_NAME_CHARACTERS = 80;
const FALLBACK_NAME = "conversation";
// What no common file system accepts in a name: control characters and the
// characters Windows reserves, the path separators among them.
// eslint-disable-next-line no-control-regex -- the control characters are the point
const FORBIDDEN = /[\u0000-\u001f\u007f<>:"/\\|?*]/g;
// Windows opens these as devices, whatever their extension.
const RESERVED = /^(con|prn|aux|nul|com[1-9]|lpt[1-9])$/i;

/**
 * A title as a file name: forbidden characters become spaces, spaces fold,
 * dot-only words and leading/trailing dots go (no `..`, no hidden file, no
 * trailing dot Windows drops), a reserved device name gives nothing, and the
 * length is capped without cutting a character in two.
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
  if (RESERVED.test(name.split(".")[0])) {
    return "";
  }
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
