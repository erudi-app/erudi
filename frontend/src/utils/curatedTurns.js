import i18n from "../i18n";
import { formatNumber } from "../i18n/format";
import { curatedErrorOf } from "./traceEvents";

// Curated turns (#690.2): the app's own sentence for a turn it could not
// answer normally (a prompt that does not fit the window or the machine's
// memory, a model that did not start answering in time or went silent, the
// tool-loop limit, a turn that ended during reasoning).
//
// The backend stores the English line in the message content (the model reads
// it back, and it is what a turn stored before this version shows) and carries
// {"t":"curated_error","kind","params","text"} in the turn's trace (and on the
// live `answer` or `error` event), `text` being that English line. The
// sentence shown is `chat:curated.<kind>` in the interface language, the
// numbers formatted for it, translated at display: it follows a language
// switch and a reload. The kinds live in backend/src/agents/curated_turns.py
// (CURATED_KINDS); a kind this interface has no key for keeps its stored line.

const KIND = /^[a-z][a-z0-9_]*$/;

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
 * The stored content with its curated English line replaced by the sentence
 * in the interface language.
 *
 * Only that line is replaced: what streamed before it and whatever the same
 * turn added after it (a later failure's red line, an interrupted stream's
 * line, the Arena's request-failure marker) stay as they are. Without a
 * curated kind, without a sentence for it, or when the line is not found, the
 * content is returned as stored.
 *
 * @param {string} content - Raw message content as stored/streamed.
 * @param {Array|null|undefined} trace - The message's trace.
 * @returns {string}
 */
export function withCuratedTurn(content, trace) {
  const text = String(content || "");
  const curated = curatedErrorOf(trace);
  const english = typeof curated?.text === "string" ? curated.text : "";
  const line = english ? curatedTurnText(curated) : null;
  const at = line ? text.lastIndexOf(english) : -1;
  if (at < 0) {
    return text;
  }
  return `${text.slice(0, at)}${line}${text.slice(at + english.length)}`;
}
