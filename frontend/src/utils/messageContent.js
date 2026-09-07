/**
 * Turn a stored message's raw content into the readable text shown in the chat.
 *
 * Stored user messages can carry internal attachment markers — [image] for
 * this-session attachments, [image_path:<path>] for persisted ones, and
 * [file_path:<path>] for attached documents (#492) — that the UI renders as
 * thumbnails, chips or placeholders, never as text. Error messages carry the
 * [ERROR_MESSAGE_SYSTEM] sentinel displayed as a ❌ prefix.
 *
 * Marker paths are percent-encoded by the backend (`_encode_marker_path` in
 * backend/src/domains/conversations/services.py): "%" as %25 and "]" as %5D, so
 * a path holding a closing bracket cannot end its own marker. Every reader of a
 * marker path here goes through decodeMarkerPath.
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
 * Turn a stored marker path back into the real filesystem path.
 *
 * The order matters and mirrors the backend's encoding order in reverse: %5D
 * first, then %25. A path that really contains "%5D" was stored as "%255D", and
 * the first pass cannot match inside it, so it survives intact.
 *
 * @param {string} path - Path as written inside a marker.
 * @returns {string} The real path.
 */
export function decodeMarkerPath(path) {
  return String(path || "")
    .replace(/%5D/g, "]")
    .replace(/%25/g, "%");
}

/**
 * Real filesystem paths of the images a stored user message carries.
 *
 * Bare [image] markers (clipboard images with no file origin) yield nothing:
 * there is no path to reload them from.
 *
 * @param {string} content - Raw message content as stored.
 * @returns {string[]} The image paths, decoded, in order.
 */
export function getImagePaths(content) {
  return [...String(content || "").matchAll(/\[image_path:([^\]]+)\]/g)].map((match) =>
    decodeMarkerPath(match[1])
  );
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
    baseName(decodeMarkerPath(match[1]))
  );
}
