import { describe, it, expect, beforeEach, afterEach } from "vitest";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { readImageAsDataUrl } from "./imageReader.js";
import { MAX_IMAGE_BYTES } from "./imageLimits.js";

const PNG_BYTES = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

let dir;
beforeEach(() => {
  dir = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "erudi-image-")));
});
afterEach(() => {
  fs.rmSync(dir, { recursive: true, force: true });
});

const file = (name, bytes = PNG_BYTES) => {
  const target = path.join(dir, name);
  fs.writeFileSync(target, bytes);
  return target;
};

describe("readImageAsDataUrl", () => {
  it("reads an image into a data URL with its MIME type", async () => {
    const result = await readImageAsDataUrl(file("shot.png"));
    expect(result).toEqual({
      ok: true,
      dataUrl: `data:image/png;base64,${PNG_BYTES.toString("base64")}`,
    });
  });

  it("names a .jpg file image/jpeg, whatever the case of its extension", async () => {
    const result = await readImageAsDataUrl(file("photo.JPG"));
    expect(result.ok).toBe(true);
    expect(result.dataUrl.startsWith("data:image/jpeg;base64,")).toBe(true);
  });

  it("refuses a file that is not an image", async () => {
    expect(await readImageAsDataUrl(file("notes.txt", "secret"))).toEqual({
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
    const real = file("real.webp");
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

  it("refuses an image over the size cap without reading it", async () => {
    const big = path.join(dir, "huge.png");
    const fd = fs.openSync(big, "w");
    fs.ftruncateSync(fd, MAX_IMAGE_BYTES + 1);
    fs.closeSync(fd);
    expect(await readImageAsDataUrl(big)).toEqual({ ok: false, reason: "too_large" });
  });

  it("accepts an image exactly at the cap", async () => {
    const edge = path.join(dir, "edge.png");
    const fd = fs.openSync(edge, "w");
    fs.ftruncateSync(fd, MAX_IMAGE_BYTES);
    fs.closeSync(fd);
    expect((await readImageAsDataUrl(edge)).ok).toBe(true);
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
