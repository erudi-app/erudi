import { describe, it, expect, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { readImageAsDataUrl } from "./imageReader.js";
import { MAX_IMAGE_BYTES } from "./imageLimits.js";

// The first bytes of each accepted format, followed by a little padding.
const SIGNATURES = {
  png: Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 0]),
  jpeg: Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0, 0x10, 0x4a, 0x46, 0x49, 0x46, 0, 1]),
  gif: Buffer.from("GIF89a\x01\x00\x01\x00\x00\x00", "latin1"),
  webp: Buffer.concat([Buffer.from("RIFF"), Buffer.from([0x24, 0, 0, 0]), Buffer.from("WEBPVP8 ")]),
  bmp: Buffer.concat([Buffer.from("BM"), Buffer.alloc(10)]),
};

let dir;
beforeEach(() => {
  dir = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "erudi-image-")));
});
afterEach(() => {
  fs.rmSync(dir, { recursive: true, force: true });
});

const file = (name, bytes = SIGNATURES.png) => {
  const target = path.join(dir, name);
  fs.writeFileSync(target, bytes);
  return target;
};

// A file of `size` bytes that starts with a PNG signature (sparse past it).
const sized = (name, size) => {
  const target = path.join(dir, name);
  const fd = fs.openSync(target, "w");
  fs.writeSync(fd, SIGNATURES.png);
  fs.ftruncateSync(fd, size);
  fs.closeSync(fd);
  return target;
};

describe("readImageAsDataUrl", () => {
  it("reads an image into a data URL with its MIME type", async () => {
    const result = await readImageAsDataUrl(file("shot.png"));
    expect(result).toEqual({
      ok: true,
      dataUrl: `data:image/png;base64,${SIGNATURES.png.toString("base64")}`,
    });
  });

  it.each([
    ["photo.JPG", "jpeg", "image/jpeg"],
    ["photo.jpeg", "jpeg", "image/jpeg"],
    ["anim.gif", "gif", "image/gif"],
    ["pic.webp", "webp", "image/webp"],
    ["scan.bmp", "bmp", "image/bmp"],
  ])("reads %s as %s", async (name, format, mime) => {
    const result = await readImageAsDataUrl(file(name, SIGNATURES[format]));
    expect(result.ok).toBe(true);
    expect(result.dataUrl.startsWith(`data:${mime};base64,`)).toBe(true);
  });

  it("names the format from the bytes, not from the extension", async () => {
    const result = await readImageAsDataUrl(file("really-a-jpeg.png", SIGNATURES.jpeg));
    expect(result.dataUrl.startsWith("data:image/jpeg;base64,")).toBe(true);
  });

  it("refuses a text file named like an image", async () => {
    expect(await readImageAsDataUrl(file("notes.png", "not an image at all"))).toEqual({
      ok: false,
      reason: "not_image",
    });
  });

  it("refuses an empty file named like an image", async () => {
    expect(await readImageAsDataUrl(file("empty.png", ""))).toEqual({
      ok: false,
      reason: "not_image",
    });
  });

  it("refuses a RIFF file that is not WebP", async () => {
    const wav = Buffer.concat([
      Buffer.from("RIFF"),
      Buffer.from([0, 0, 0, 0]),
      Buffer.from("WAVE"),
    ]);
    expect((await readImageAsDataUrl(file("sound.webp", wav))).reason).toBe("not_image");
  });

  it("refuses a file whose extension is not an image one, even with image bytes", async () => {
    expect(await readImageAsDataUrl(file("notes.txt"))).toEqual({
      ok: false,
      reason: "not_image",
    });
  });

  it("judges the extension on the real path, not on a symlink's name", async () => {
    const secret = file("id_rsa", "-----BEGIN-----");
    const link = path.join(dir, "innocent.png");
    fs.symlinkSync(secret, link);
    expect(await readImageAsDataUrl(link)).toEqual({ ok: false, reason: "not_image" });
  });

  it("follows a symlink to a real image", async () => {
    const real = file("real.webp", SIGNATURES.webp);
    const link = path.join(dir, "alias.png");
    fs.symlinkSync(real, link);
    const result = await readImageAsDataUrl(link);
    expect(result.ok).toBe(true);
    expect(result.dataUrl.startsWith("data:image/webp;base64,")).toBe(true);
  });

  it("refuses a folder named like an image", async () => {
    fs.mkdirSync(path.join(dir, "album.png"));
    expect(await readImageAsDataUrl(path.join(dir, "album.png"))).toEqual({
      ok: false,
      reason: "not_image",
    });
  });

  it("refuses an image over the size cap", async () => {
    expect(await readImageAsDataUrl(sized("huge.png", MAX_IMAGE_BYTES + 1))).toEqual({
      ok: false,
      reason: "too_large",
    });
  });

  it("accepts an image exactly at the cap", async () => {
    expect((await readImageAsDataUrl(sized("edge.png", MAX_IMAGE_BYTES))).ok).toBe(true);
  });

  it("reports a missing file as unreadable", async () => {
    const result = await readImageAsDataUrl(path.join(dir, "gone.png"));
    expect(result).toMatchObject({ ok: false, reason: "unreadable" });
    expect(result.error.code).toBe("ENOENT");
  });

  it("refuses something that is not a path", async () => {
    expect(await readImageAsDataUrl(42)).toEqual({ ok: false, reason: "not_image" });
    expect(await readImageAsDataUrl("")).toEqual({ ok: false, reason: "not_image" });
  });
});
