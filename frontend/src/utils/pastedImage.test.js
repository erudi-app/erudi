import { describe, it, expect, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { savePastedImage } from "./pastedImage.js";

const PNG = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
const dataUrl = (mime, bytes) => `data:${mime};base64,${bytes.toString("base64")}`;

let dir;
beforeEach(() => {
  dir = path.join(fs.mkdtempSync(path.join(os.tmpdir(), "erudi-paste-")), "pasted-images");
});
afterEach(() => {
  fs.rmSync(path.dirname(dir), { recursive: true, force: true });
});

describe("savePastedImage", () => {
  it("writes the bytes in the folder, named by their format, and returns the path", () => {
    const saved = savePastedImage(dataUrl("image/png", PNG), dir);
    expect(path.dirname(saved)).toBe(dir);
    expect(saved.endsWith(".png")).toBe(true);
    expect(fs.readFileSync(saved).equals(PNG)).toBe(true);
  });

  it("names a JPEG .jpg", () => {
    expect(savePastedImage(dataUrl("image/jpeg", PNG), dir).endsWith(".jpg")).toBe(true);
  });

  it.each([
    ["SVG", "image/svg+xml"],
    ["TIFF", "image/tiff"],
    ["a non-image", "text/html"],
  ])("refuses %s, which the composer never accepts", (_name, mime) => {
    expect(savePastedImage(dataUrl(mime, Buffer.from("<svg/>")), dir)).toBeNull();
    expect(fs.existsSync(dir)).toBe(false);
  });

  it("refuses something that is not a data URL", () => {
    expect(savePastedImage("file:///etc/passwd", dir)).toBeNull();
    expect(savePastedImage(null, dir)).toBeNull();
  });
});
