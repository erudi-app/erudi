import i18n from "../i18n";
import { formatList, formatNumber } from "../i18n/format";
import { answerNoticesOf } from "./traceEvents";

// Answer notices (#764): the notes that open an answer when the model did not
// get the question in full (an image it cannot see, an attached file it could
// not read, attached text shortened to fit).
//
// The backend stores each note's English line, in italics, in the message
// content (the model reads it back, and it is what a message stored before
// this version shows) and carries {"t":"notice","kind","params","text"} in the
// turn's trace (and as `notice` on the live `answer` event), `text` being that
// English line. The sentence shown is `chat:notices.<kind>` in the interface
// language, the file names joined as a list of that language and each
// unreadable file's reason from `common:fileReasons.<reason>`, translated at
// display: it follows a language switch and a reload. The kinds live in
// backend/src/agents/answer_notices.py (NOTICE_KINDS); a kind or a reason this
// interface has no key for keeps its stored line.

const CODE = /^[a-z][a-z0-9_]*$/;

const known = (key) => i18n.exists(key);

// One unreadable file: "scan.pdf (no extractable text...)", or the reason
// alone for an entry without a name (the file cap). Null for a reason this
// interface does not know.
function unreadableFile(file) {
  if (!file || typeof file.reason !== "string" || !CODE.test(file.reason)) {
    return null;
  }
  const key = `common:fileReasons.${file.reason}`;
  if (!known(key)) {
    return null;
  }
  const reason = i18n.t(key, {
    limit: typeof file.limit === "number" ? formatNumber(file.limit) : "",
  });
  const name = typeof file.name === "string" ? file.name : "";
  return name ? i18n.t("chat:noticeFile", { name, reason }) : reason;
}

// The `files` value of a kind, or null when one of its entries cannot be said.
function filesValue(kind, files) {
  if (!Array.isArray(files) || files.length === 0) {
    return null;
  }
  const items =
    kind === "attachments_unreadable"
      ? files.map(unreadableFile)
      : files.map((name) => (typeof name === "string" && name ? name : null));
  return items.every((item) => item !== null) ? formatList(items) : null;
}

/**
 * The notice's sentence in the interface language, or null when there is
 * none to show (not a notice, or a kind or reason this interface does not
 * know).
 *
 * @param {{kind: string, params?: Object}|null} notice - The trace event.
 * @returns {string|null}
 */
export function answerNoticeText(notice) {
  if (!notice || typeof notice.kind !== "string" || !CODE.test(notice.kind)) {
    return null;
  }
  const key = `chat:notices.${notice.kind}`;
  if (!known(key)) {
    return null;
  }
  const params = notice.params || {};
  if (!Object.hasOwn(params, "files")) {
    return i18n.t(key);
  }
  const files = filesValue(notice.kind, params.files);
  return files === null ? null : i18n.t(key, { files });
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
    const english = typeof notice.text === "string" ? notice.text : "";
    const at = english ? text.indexOf(english, from) : -1;
    const line = at < 0 ? null : answerNoticeText(notice);
    if (line === null) {
      continue;
    }
    text = `${text.slice(0, at)}${line}${text.slice(at + english.length)}`;
    from = at + line.length;
  }
  return text;
}
