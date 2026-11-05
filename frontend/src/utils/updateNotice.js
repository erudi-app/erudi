/**
 * The "Updated to X.Y.Z" dialog (#572): whether it shows at startup, and what
 * it shows.
 *
 * Main keeps two files in the app's user data folder: `last-seen-version`
 * (the version this dialog last settled) and `update-notes.json` (the release
 * notes received with the last update downloaded, `{version, notes}`). At
 * startup the page reads both through main, asks the backend whether the
 * database holds the user's data when no version was ever recorded, and
 * decides:
 * - running version already seen: nothing;
 * - no version recorded, no user data (fresh install, "Clear all data"):
 *   record it silently;
 * - no version recorded, user data (the first launch of this version over an
 *   older one, which recorded nothing): the dialog, with the explanation that
 *   downloading and installing are now separate choices;
 * - an older version recorded: the dialog, with the notes when they are this
 *   version's (an update downloaded but never installed keeps notes for
 *   another version), else the version and the release page link;
 * - a newer version recorded (a downgrade): nothing, and the higher version
 *   stays recorded, so a dialog dismissed for it does not come back.
 * The version is recorded once the dialog is dismissed, or at once when there
 * is nothing to show. Nothing here makes a network request.
 */
import { createLogger } from "./logger";

const log = createLogger("updateNotice");

const RELEASES_TAG_URL = "https://github.com/erudi-app/erudi/releases/tag/v";

/** The release page of a version, the dialog's only link. */
export function releasePageUrl(version) {
  return `${RELEASES_TAG_URL}${encodeURIComponent(version)}`;
}

const VERSION = /^v?(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$/;

function compareIdentifiers(a, b) {
  const numA = /^\d+$/.test(a);
  const numB = /^\d+$/.test(b);
  if (numA && numB) return Number(a) - Number(b);
  if (numA) return -1;
  if (numB) return 1;
  return a < b ? -1 : a > b ? 1 : 0;
}

/**
 * Compare two semantic versions.
 * @returns {number | null} Negative, zero or positive like a sort comparator;
 *   null when either side is not a version.
 */
export function compareVersions(a, b) {
  const ma = VERSION.exec(String(a ?? "").trim());
  const mb = VERSION.exec(String(b ?? "").trim());
  if (!ma || !mb) return null;
  for (let i = 1; i <= 3; i += 1) {
    const diff = Number(ma[i]) - Number(mb[i]);
    if (diff !== 0) return diff;
  }
  const preA = ma[4];
  const preB = mb[4];
  if (!preA && !preB) return 0;
  if (!preA) return 1;
  if (!preB) return -1;
  const partsA = preA.split(".");
  const partsB = preB.split(".");
  for (let i = 0; i < Math.max(partsA.length, partsB.length); i += 1) {
    if (partsA[i] === undefined) return -1;
    if (partsB[i] === undefined) return 1;
    const diff = compareIdentifiers(partsA[i], partsB[i]);
    if (diff !== 0) return diff;
  }
  return 0;
}

// Notes come from this project's own releases, but they are read at launch on
// the window's thread: the work stays linear in their size, and their size is
// capped.
export const NOTES_MAX_CHARS = 64 * 1024;

const NAMED_ENTITIES = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " " };

function decodeEntities(text) {
  return text.replace(/&(#x[0-9a-f]{1,8}|#\d{1,8}|[a-z]{1,32});/gi, (match, body) => {
    if (body[0] === "#") {
      const code =
        body[1] === "x" || body[1] === "X" ? parseInt(body.slice(2), 16) : Number(body.slice(1));
      return Number.isFinite(code) && code > 0 && code <= 0x10ffff
        ? String.fromCodePoint(code)
        : match;
    }
    return NAMED_ENTITIES[body.toLowerCase()] ?? match;
  });
}

function cleanLines(text) {
  return text
    .split("\n")
    .map((line) => line.replace(/\s+/g, " ").trim())
    .filter((line) => line && line !== "•");
}

// Tags that end a line; `li` also starts one, with a bullet.
const BLOCK_TAGS = new Set([
  "p",
  "div",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "ul",
  "ol",
  "li",
  "br",
  "blockquote",
  "pre",
  "table",
  "tr",
  "hr",
]);
// Their closing tags, searched from a given index (sticky search, built once).
const CLOSING_TAG = new Map([
  ["script", /<\/script/gi],
  ["style", /<\/style/gi],
]);

const isLetter = (char) => (char >= "a" && char <= "z") || (char >= "A" && char <= "Z");

/**
 * HTML to text lines, without a DOM: nothing is parsed into elements, so
 * nothing can load. One pass, each character read a bounded number of times:
 * blocks and line breaks end a line, list items start one with a bullet,
 * scripts, styles and comments are dropped with their content (to the end
 * when unclosed), every other tag is dropped and its text kept, and a "<"
 * that opens no tag is text.
 */
function htmlToLines(html) {
  const parts = [];
  const text = (segment) => parts.push(segment.replace(/\s+/g, " "));
  let i = 0;
  while (i < html.length) {
    const lt = html.indexOf("<", i);
    if (lt === -1) {
      text(html.slice(i));
      break;
    }
    text(html.slice(i, lt));
    if (html.startsWith("<!--", lt)) {
      const end = html.indexOf("-->", lt + 4);
      if (end === -1) break;
      i = end + 3;
      continue;
    }
    const nameStart = html[lt + 1] === "/" ? lt + 2 : lt + 1;
    const markup = html[lt + 1] === "!" || html[lt + 1] === "?";
    if (!markup && !isLetter(html[nameStart] ?? "")) {
      parts.push("<");
      i = lt + 1;
      continue;
    }
    const gt = html.indexOf(">", lt + 1);
    if (gt === -1) {
      // An unterminated tag: what follows is not text either.
      break;
    }
    i = gt + 1;
    if (markup) continue;
    let nameEnd = nameStart;
    while (nameEnd < gt && /[a-zA-Z0-9]/.test(html[nameEnd])) nameEnd += 1;
    const name = html.slice(nameStart, nameEnd).toLowerCase();
    const closing = nameStart === lt + 2;
    if (!closing && CLOSING_TAG.has(name)) {
      const close = CLOSING_TAG.get(name);
      close.lastIndex = i;
      const found = close.exec(html);
      if (!found) break;
      const end = html.indexOf(">", found.index);
      if (end === -1) break;
      i = end + 1;
      continue;
    }
    if (name === "li" && !closing) {
      parts.push("\n• ");
    } else if (BLOCK_TAGS.has(name)) {
      parts.push("\n");
    }
  }
  return cleanLines(decodeEntities(parts.join("")));
}

function noteToLines(note) {
  if (typeof note !== "string") return [];
  const capped = note.slice(0, NOTES_MAX_CHARS);
  return /<[a-z!/]/i.test(capped) && capped.includes(">")
    ? htmlToLines(capped)
    : cleanLines(capped);
}

/**
 * Release notes as electron-updater delivers them, as plain text lines.
 *
 * @param {string | Array<{version: string, note: string | null}> | null | undefined} notes
 *   An HTML string (GitHub), plain text, or one entry per version when the
 *   full changelog is requested.
 * @returns {string[] | null} The lines, or null when there is nothing to show.
 */
export function releaseNotesToLines(notes) {
  let lines = [];
  if (typeof notes === "string") {
    lines = noteToLines(notes);
  } else if (Array.isArray(notes)) {
    let budget = NOTES_MAX_CHARS;
    for (const entry of notes) {
      if (budget <= 0) break;
      const note = typeof entry?.note === "string" ? entry.note.slice(0, budget) : null;
      budget -= note?.length ?? 0;
      const entryLines = noteToLines(note);
      if (entryLines.length > 0) {
        lines.push(String(entry.version ?? ""), ...entryLines);
      }
    }
    lines = lines.filter(Boolean);
  }
  return lines.length > 0 ? lines : null;
}

/**
 * The startup decision.
 *
 * @param {object} input
 * @param {string | null} input.runningVersion The version running now.
 * @param {string | null} input.lastSeenVersion The version recorded last, or null.
 * @param {{version: string, notes: unknown} | null} input.notes Kept from the last download.
 * @param {boolean | null} input.hasUserData Whether the database holds a
 *   conversation or a downloaded model; null when unknown.
 * @returns {{action: "none"} | {action: "mark"} |
 *   {action: "show", version: string, lines: string[] | null, explainOptIns: boolean}}
 */
export function decideUpdateNotice({ runningVersion, lastSeenVersion, notes, hasUserData }) {
  if (!runningVersion || lastSeenVersion === runningVersion) {
    return { action: "none" };
  }
  const order = lastSeenVersion ? compareVersions(lastSeenVersion, runningVersion) : null;
  if (order !== null && order > 0) {
    // A downgrade: keep the higher version recorded.
    return { action: "none" };
  }
  const known = order !== null;
  if (!known && hasUserData === null) {
    // Cannot tell an update from a fresh install: ask again next launch.
    return { action: "none" };
  }
  if (!known && !hasUserData) {
    return { action: "mark" };
  }
  if (known && order === 0) {
    return { action: "mark" };
  }
  const lines = notes?.version === runningVersion ? releaseNotesToLines(notes.notes) : null;
  return { action: "show", version: runningVersion, lines, explainOptIns: !known };
}

/**
 * Tell main to record the running version as seen. Main writes its own
 * version, never one the page names.
 */
export async function markVersionSeen() {
  try {
    await window.updaterAPI?.markVersionSeen?.();
  } catch (error) {
    // The dialog comes back next launch; nothing else depends on the record.
    log.warn("Could not record the version as seen", error);
  }
}

/**
 * Run the startup decision. Records the version at once when there is
 * nothing to show; otherwise returns the dialog for the page to render and
 * leaves the recording to its dismissal.
 *
 * @param {{get: Function}} client The API client.
 * @returns {Promise<{action: "show", version: string, lines: string[] | null,
 *   explainOptIns: boolean} | null>}
 */
export async function loadUpdateNotice(client) {
  const bridge = window.updaterAPI;
  if (!bridge?.getLaunchState) {
    return null;
  }
  let launch;
  try {
    launch = await bridge.getLaunchState();
  } catch (error) {
    log.warn("The app process did not answer the launch state request", error);
    return null;
  }
  if (!launch?.running_version || launch.running_version === launch.last_seen_version) {
    return null;
  }

  let hasUserData = null;
  if (compareVersions(launch.last_seen_version, launch.running_version) === null) {
    try {
      const state = await client.get("/startup/install_state");
      hasUserData = state?.has_user_data === true;
    } catch (error) {
      // Asked again at the next launch: nothing is recorded meanwhile.
      log.warn("Could not read the install state; the update notice waits", error);
    }
  }

  const decision = decideUpdateNotice({
    runningVersion: launch.running_version,
    lastSeenVersion: launch.last_seen_version ?? null,
    notes: launch.notes ?? null,
    hasUserData,
  });
  if (decision.action === "mark") {
    await markVersionSeen();
  }
  return decision.action === "show" ? decision : null;
}
