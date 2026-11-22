import { describe, it, expect } from "vitest";
import { sniffImageType, SIGNATURE_BYTES } from "./imageSignature.js";

const ascii = (text) => Array.from(text, (char) => char.charCodeAt(0));
const bytes = (...parts) => new Uint8Array(parts.flat());

describe("sniffImageType", () => {
  it.each([
    ["image/png", bytes([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 0])],
    ["image/jpeg", bytes([0xff, 0xd8, 0xff, 0xe0, 0, 0x10, 0x4a, 0x46, 0x49, 0x46, 0, 1])],
    ["image/gif", bytes(ascii("GIF89a"), [1, 0, 1, 0])],
    ["image/gif", bytes(ascii("GIF87a"), [1, 0, 1, 0])],
    ["image/webp", bytes(ascii("RIFF"), [0x24, 0, 0, 0], ascii("WEBPVP8 "))],
    ["image/bmp", bytes(ascii("BM"), new Array(10).fill(0))],
  ])("recognizes %s", (mime, data) => {
    expect(sniffImageType(data)).toBe(mime);
  });

  it("returns null for text, HTML, a truncated PNG header and empty input", () => {
    expect(sniffImageType(bytes(ascii("Note is not an image")))).toBeNull();
    expect(sniffImageType(bytes(ascii("<html><body>")))).toBeNull();
    expect(sniffImageType(bytes([0x89, 0x50, 0x4e, 0x47]))).toBeNull();
    expect(sniffImageType(new Uint8Array())).toBeNull();
  });

  it("reads no more than SIGNATURE_BYTES, enough for WebP", () => {
    expect(SIGNATURE_BYTES).toBeGreaterThanOrEqual(12);
  });
});
