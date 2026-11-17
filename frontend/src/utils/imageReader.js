// Read an image attachment back from disk for the conversation thread
// (fs:readImageAsDataURL). The path comes from the window, which got it from
// a stored message, so it is checked before anything is read: the extension
// of the real path (symlinks resolved) must be an image one, the target a
// regular file, its size within MAX_IMAGE_BYTES. There is no list of allowed
// folders: an image attached from anywhere on the disk must keep showing.

import fs from "node:fs";
import path from "node:path";
import { IMAGE_EXTENSIONS, MAX_IMAGE_BYTES } from "./imageLimits";

const MIME_BY_EXTENSION = {
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".webp": "image/webp",
  ".gif": "image/gif",
  ".bmp": "image/bmp",
};

/**
 * @param {string} filePath
 * @returns {Promise<{ok: true, dataUrl: string} | {ok: false, reason: "not_image"|"too_large"|"unreadable", error?: Error}>}
 *   `error` is set for "unreadable", for the caller's log.
 */
export async function readImageAsDataUrl(filePath) {
  if (typeof filePath !== "string" || !filePath) {
    return { ok: false, reason: "not_image" };
  }
  try {
    const realPath = await fs.promises.realpath(filePath);
    const extension = path.extname(realPath).toLowerCase();
    if (!IMAGE_EXTENSIONS.includes(extension)) {
      return { ok: false, reason: "not_image" };
    }
    const stats = await fs.promises.stat(realPath);
    if (!stats.isFile()) {
      return { ok: false, reason: "not_image" };
    }
    if (stats.size > MAX_IMAGE_BYTES) {
      return { ok: false, reason: "too_large" };
    }
    const bytes = await fs.promises.readFile(realPath);
    // The file may have grown since it was measured.
    if (bytes.length > MAX_IMAGE_BYTES) {
      return { ok: false, reason: "too_large" };
    }
    return {
      ok: true,
      dataUrl: `data:${MIME_BY_EXTENSION[extension]};base64,${bytes.toString("base64")}`,
    };
  } catch (error) {
    return { ok: false, reason: "unreadable", error };
  }
}
