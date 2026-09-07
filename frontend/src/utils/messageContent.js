/**
 * Turn a stored message's raw content into the readable text shown in the chat.
 *
 * Stored user messages can carry internal attachment markers — [image] for
 * this-session attachments, [image_path:<path>] for persisted ones, and
 * [file_path:<path>] for attached documents (#492) — that the UI renders as
 * thumbnails, chips or placeholders, never as text. Error messages carry the
 * [ERROR_MESSAGE_SYSTEM] sentinel displayed as a ❌ prefix.
 *
 * Single source of truth used by BOTH the chat display and the
 * copy-to-clipboard action, so copying yields exactly what the user sees
 * (#136).
 *
 * @param {string} content - Raw message content as stored/streamed.
 * @returns {string} The human-readable text, markers removed.
 */
export function getDisplayContent(content) {
  if (content.includes("[ERROR_MESSAGE_SYSTEM]")) {
    return content.replace("[ERROR_MESSAGE_SYSTEM] ", "❌ ");
  }
  return content
    .replace(/\[image\]/g, "")
    .replace(/\[image_path:[^\]]*\]/g, "")
    .replace(/\[file_path:[^\]]*\]/g, "")
    .trim();
}

/**
 * Last segment of a filesystem path, POSIX or Windows separators alike.
 *
 * @param {string} path - Absolute or relative path.
 * @returns {string} The file or folder name, "" when there is none.
 */
export function baseName(path) {
  const segments = String(path || "").split(/[\\/]/);
  return segments.filter(Boolean).pop() || "";
}

/**
 * Names of the documents a stored user message records as attached (#492).
 *
 * The extracted text rides the live turn only, so a reloaded conversation
 * recovers WHAT was attached from these markers, not the content.
 *
 * @param {string} content - Raw message content as stored.
 * @returns {string[]} The attached file/folder names, in order.
 */
export function getAttachmentNames(content) {
  return [...String(content || "").matchAll(/\[file_path:([^\]]+)\]/g)].map((match) =>
    baseName(match[1])
  );
}
