// What an image file's first bytes announce. Pure (Uint8Array in, no Node API),
// so the Electron main process (imageReader.js) and the renderer (the composer)
// judge a file by its content with the same code, never by its name or its
// declared type.

/** Bytes needed to tell every supported format apart. */
export const SIGNATURE_BYTES = 16;

const startsWith = (bytes, signature, offset = 0) =>
  bytes.length >= offset + signature.length &&
  signature.every((byte, i) => bytes[offset + i] === byte);

const ascii = (text) => Array.from(text, (char) => char.charCodeAt(0));

/** The MIME type the first bytes announce, or null for anything else. */
export function sniffImageType(bytes) {
  if (startsWith(bytes, [0xff, 0xd8, 0xff])) return "image/jpeg";
  if (startsWith(bytes, [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])) return "image/png";
  if (startsWith(bytes, ascii("GIF87a")) || startsWith(bytes, ascii("GIF89a"))) {
    return "image/gif";
  }
  if (startsWith(bytes, ascii("RIFF")) && startsWith(bytes, ascii("WEBP"), 8)) return "image/webp";
  if (startsWith(bytes, ascii("BM"))) return "image/bmp";
  return null;
}
