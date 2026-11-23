// Read an image attachment back from disk for the conversation thread
// (fs:readImageAsDataURL). The path comes from the window, which got it from
// a stored message, so it is checked before anything is returned: the
// extension of the real path (symlinks resolved) must be an image one, the
// file is opened once (O_NOFOLLOW where the system has it, so the checked
// path cannot be swapped for a symlink between the check and the read), and
// that handle must be a regular file within MAX_IMAGE_BYTES whose first bytes
// are a JPEG, PNG, GIF, WebP or BMP signature. The MIME type comes from those
// bytes. There is no list of allowed folders: an image attached from
// anywhere on the disk must keep showing.

import fs from "node:fs";
import path from "node:path";
import { IMAGE_EXTENSIONS, MAX_IMAGE_BYTES } from "./imageLimits";
import { sniffImageType } from "./imageSignature";

/**
 * @param {string} filePath
 * @returns {Promise<{ok: true, dataUrl: string} | {ok: false, reason: "not_image"|"too_large"|"unreadable", error?: Error}>}
 *   `error` is set for "unreadable", for the caller's log.
 */
export async function readImageAsDataUrl(filePath) {
  if (typeof filePath !== "string" || !filePath) {
    return { ok: false, reason: "not_image" };
  }
  let handle = null;
  try {
    const realPath = await fs.promises.realpath(filePath);
    if (!IMAGE_EXTENSIONS.includes(path.extname(realPath).toLowerCase())) {
      return { ok: false, reason: "not_image" };
    }
    // O_NOFOLLOW is undefined on Windows; the flag is then plain read-only.
    handle = await fs.promises.open(
      realPath,
      fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW ?? 0)
    );
    const stats = await handle.stat();
    if (!stats.isFile()) {
      return { ok: false, reason: "not_image" };
    }
    if (stats.size > MAX_IMAGE_BYTES) {
      return { ok: false, reason: "too_large" };
    }
    // One byte past the cap tells a file that grew since it was measured.
    const buffer = Buffer.alloc(Math.min(stats.size, MAX_IMAGE_BYTES) + 1);
    let length = 0;
    while (length < buffer.length) {
      const { bytesRead } = await handle.read(buffer, length, buffer.length - length, length);
      if (bytesRead === 0) break;
      length += bytesRead;
    }
    if (length > MAX_IMAGE_BYTES) {
      return { ok: false, reason: "too_large" };
    }
    const bytes = buffer.subarray(0, length);
    const mime = sniffImageType(bytes);
    if (!mime) {
      return { ok: false, reason: "not_image" };
    }
    return { ok: true, dataUrl: `data:${mime};base64,${bytes.toString("base64")}` };
  } catch (error) {
    // ELOOP: the path became a symlink after it was resolved.
    if (error && error.code === "ELOOP") {
      return { ok: false, reason: "not_image" };
    }
    return { ok: false, reason: "unreadable", error };
  } finally {
    await handle?.close();
  }
}
