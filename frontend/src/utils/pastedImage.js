// Persist a pasted (clipboard) image (image:savePasted). The renderer has no
// file system access, and a clipboard image has no source path, so it would
// otherwise be stored as a bare [image] placeholder and lost on reload (#136).
// Only the raster formats the composer accepts are written (SVG and the rest
// are refused there before they get here; refusing them here too keeps the
// main process from writing whatever a window sends).

import fs from "node:fs";
import path from "node:path";
import { SUPPORTED_IMAGE_TYPES } from "./imageLimits";

const EXTENSION_BY_TYPE = {
  "image/jpeg": "jpg",
  "image/png": "png",
  "image/webp": "webp",
  "image/gif": "gif",
  "image/bmp": "bmp",
};

/**
 * @param {string} dataUrl `data:image/<type>;base64,...`
 * @param {string} dir The pasted-images folder, created when needed.
 * @returns {string|null} The new file's path, or null for anything refused.
 *   Throws when the file cannot be written.
 */
export function savePastedImage(dataUrl, dir) {
  const match = /^data:(image\/[a-zA-Z0-9.+-]+);base64,(.*)$/.exec(
    typeof dataUrl === "string" ? dataUrl : ""
  );
  if (!match) return null;
  const type = match[1].toLowerCase();
  if (!SUPPORTED_IMAGE_TYPES.includes(type)) return null;
  fs.mkdirSync(dir, { recursive: true });
  const filePath = path.join(
    dir,
    `paste-${Date.now()}-${Math.random().toString(36).slice(2, 8)}.${EXTENSION_BY_TYPE[type]}`
  );
  fs.writeFileSync(filePath, Buffer.from(match[2], "base64"));
  return filePath;
}
