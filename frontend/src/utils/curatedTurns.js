import i18n from "../i18n";
import { formatNumber } from "../i18n/format";

// Curated turns (#690.2): the app's own sentence for a turn it could not
// answer normally (a prompt that does not fit the window or the machine's
// memory, a model that did not start answering in time or went silent, the
// tool-loop limit, a turn that ended during reasoning).
//
// The backend stores the English line as the message content (the model reads
// it back, and it is what a turn stored before this version shows) and carries
// {"t":"curated_error","kind","params"} in the turn's trace (and on the live
// `answer` or `error` event). The sentence shown is `chat:curated.<kind>` in
// the interface language, the numbers formatted for it, translated at display:
// it follows a language switch and a reload. The kinds live in
// backend/src/agents/curated_turns.py (CURATED_KINDS); a kind this interface
// has no key for keeps its stored English line.

const SENTINEL = "[ERROR_MESSAGE_SYSTEM]";
const KIND = /^[a-z][a-z0-9_]*$/;

const isCuratedError = (e) =>
  !!e && typeof e === "object" && e.t === "curated_error" && typeof e.kind === "string";

/** The curated kind of a turn: {"t":"curated_error","kind","params"}, or null. */
export function curatedErrorOf(trace) {
  return (Array.isArray(trace) && trace.find(isCuratedError)) || null;
}

/**
 * The curated turn's sentence in the interface language, or null when there
 * is none to show (no curated kind, or one this interface does not know).
 *
 * @param {{kind: string, params?: Object}|null} curated - The trace event.
 * @returns {string|null}
 */
export function curatedTurnText(curated) {
  if (!curated || typeof curated.kind !== "string" || !KIND.test(curated.kind)) {
    return null;
  }
  const key = `chat:curated.${curated.kind}`;
  if (!i18n.exists(key)) {
    return null;
  }
  const values = {};
  for (const [name, value] of Object.entries(curated.params || {})) {
    values[name] = typeof value === "number" ? formatNumber(value) : String(value);
  }
  return i18n.t(key, values);
}

/**
 * The stored content with its curated line replaced by `line`.
 *
 * The curated line is always the end of the content: a failed turn's red line
 * starts at the sentinel, below whatever streamed before it; an empty-answer
 * line is the whole answer, below the notices streamed ahead of it (each
 * ending with a blank line). Without `line`, the content is returned as stored.
 *
 * @param {string} content - Raw message content as stored/streamed.
 * @param {string|null|undefined} line - The translated sentence.
 * @returns {string}
 */
export function withCuratedLine(content, line) {
  const text = String(content || "");
  if (!line) {
    return text;
  }
  const at = text.lastIndexOf(SENTINEL);
  if (at >= 0) {
    return `${text.slice(0, at)}${SENTINEL} ${line}`;
  }
  const cut = text.lastIndexOf("\n\n");
  return `${cut >= 0 ? text.slice(0, cut + 2) : ""}${line}`;
}
