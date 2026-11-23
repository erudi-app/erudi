import i18n from "../i18n";
import { formatList } from "../i18n/format";
import { answerNoticesOf } from "./traceEvents";

// Answer notices (#764): the notes that open an answer when the model did not
// get the question in full (an image it cannot see, an attached file it could
// not read, attached text shortened to fit). The model never sees them.
//
// The backend stores each note's English line, in italics, in the message
// content (what a message stored before this version, and a plain read of the
// database, show) and carries {"t":"notice","kind","params"} in the turn's
// trace (and as `notice` on the live `answer` event). The sentence shown is
// `chat:notices.<kind>` in the interface language, the file names joined as a
// list of that language, each unreadable file's reason from
// `common:fileReasons.<reason>` and the files past the first ones counted,
// translated at display: it follows a language switch and a reload. The same
// keys in English rebuild the stored line, which is how it is found in the
// content: the `en` catalog says exactly what the backend writes
// (backend/tests/fixtures/answer_notices.json holds both sides to it). The
// kinds live in backend/src/agents/answer_notices.py (NOTICE_KINDS); a kind or
// a reason this interface has no key for keeps its stored line.

const CODE = /^[a-z][a-z0-9_]*$/;

// One unreadable file: "scan.pdf (no extractable text...)", or the reason
// alone for an entry without a name (the file cap). Null for a reason this
// interface does not know.
function unreadableFile(file, lng) {
  if (!file || typeof file.reason !== "string" || !CODE.test(file.reason)) {
    return null;
  }
  const key = `common:fileReasons.${file.reason}`;
  if (!i18n.exists(key)) {
    return null;
  }
  const reason = i18n.t(key, { lng, limit: Number.isInteger(file.limit) ? file.limit : "" });
  const name = typeof file.name === "string" ? file.name : "";
  return name ? i18n.t("chat:noticeFile", { lng, name, reason }) : reason;
}

// The `files` value of a kind, or null when one of its entries cannot be said.
function filesValue(kind, params, lng) {
  const { files, more } = params;
  if (!Array.isArray(files) || files.length === 0) {
    return null;
  }
  const items =
    kind === "attachments_unreadable"
      ? files.map((file) => unreadableFile(file, lng))
      : files.map((name) => (typeof name === "string" && name ? name : null));
  if (items.some((item) => item === null)) {
    return null;
  }
  if (Number.isInteger(more) && more > 0) {
    items.push(i18n.t("knowledgeBase:page.form.moreFiles", { lng, count: more }));
  }
  return formatList(items, lng);
}

/**
 * The notice's sentence, in the interface language or in `lng`, or null when
 * there is none to say (not a notice, or a kind or reason this interface does
 * not know).
 *
 * @param {{kind: string, params?: Object}|null} notice - The trace event.
 * @param {string} [lng] - An app language; the active one by default.
 * @returns {string|null}
 */
export function answerNoticeText(notice, lng = i18n.language) {
  if (!notice || typeof notice.kind !== "string" || !CODE.test(notice.kind)) {
    return null;
  }
  const key = `chat:notices.${notice.kind}`;
  if (!i18n.exists(key)) {
    return null;
  }
  const params = notice.params || {};
  if (!Object.hasOwn(params, "files")) {
    return i18n.t(key, { lng });
  }
  const files = filesValue(notice.kind, params, lng);
  return files === null ? null : i18n.t(key, { lng, files });
}

/**
 * The stored content with each notice's English line replaced by its
 * sentence in the interface language.
 *
 * The notices are looked up in their order, each after the previous one
 * found, and only their lines are replaced: the answer around them stays as
 * stored. A notice without a sentence, or whose line the content no longer
 * holds, leaves its part of the content as stored.
 *
 * @param {string} content - Raw message content as stored/streamed.
 * @param {Array|null|undefined} trace - The message's trace.
 * @returns {string}
 */
export function withNotices(content, trace) {
  let text = String(content || "");
  let from = 0;
  for (const notice of answerNoticesOf(trace)) {
    const english = answerNoticeText(notice, "en");
    const at = english ? text.indexOf(english, from) : -1;
    if (at < 0) {
      continue;
    }
    const line = answerNoticeText(notice);
    text = `${text.slice(0, at)}${line}${text.slice(at + english.length)}`;
    from = at + line.length;
  }
  return text;
}
