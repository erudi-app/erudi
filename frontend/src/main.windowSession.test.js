/**
 * The window's session before its first request (#746).
 *
 * On some launches the renderer's first requests to the local backend stall
 * for 15 s to 143 s inside Chromium's network stack. Two things main.js used to
 * put in front of them: a fire-and-forget `session.clearCache()` and the system
 * proxy lookup (PAC/WPAD) Chromium runs for every URL, loopback included.
 * main.js runs in the Electron main process and cannot be imported here, so the
 * wiring is read from the source text.
 */
import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const source = readFileSync(fileURLToPath(new URL("./main.js", import.meta.url)), "utf8");

const createWindowBody = () => {
  const start = source.indexOf("const createWindow = ");
  expect(start).toBeGreaterThan(-1);
  const end = source.indexOf("\n};\n", start);
  expect(end).toBeGreaterThan(start);
  return source.slice(start, end);
};

describe("main window session", () => {
  it("connects directly, awaited, before the window loads anything", () => {
    const body = createWindowBody();
    const proxy = body.search(
      /await\s+mainWindow\.webContents\.session\.setProxy\(\{ mode: "direct" \}\)/
    );
    expect(proxy).toBeGreaterThan(-1);
    expect(proxy).toBeLessThan(body.indexOf("mainWindow.loadFile("));
    expect(proxy).toBeLessThan(body.indexOf("mainWindow.loadURL(RENDERER_DEV_URL)"));
  });

  it("still loads the window when the proxy setting fails, and says so", () => {
    expect(createWindowBody()).toMatch(
      /try \{\s*await mainWindow\.webContents\.session\.setProxy\(\{ mode: "direct" \}\);\s*\} catch \(error\) \{\s*logWarn\(/
    );
  });

  it("does not wipe the cache at window creation", () => {
    expect(createWindowBody()).not.toMatch(/clearCache/);
  });

  it("still wipes the cache on Clear All Data", () => {
    expect(source).toMatch(/await session\.defaultSession\.clearCache\(\);/);
  });

  it("leaves the updater its own session", () => {
    // electron-updater downloads through session.fromPartition("electron-updater"),
    // so the window's direct connection never reaches it.
    const executor = readFileSync(
      fileURLToPath(
        new URL("../node_modules/electron-updater/out/electronHttpExecutor.js", import.meta.url)
      ),
      "utf8"
    );
    expect(executor).toMatch(/NET_SESSION_NAME = "electron-updater"/);
    expect(executor).toMatch(/session\.fromPartition\(exports\.NET_SESSION_NAME/);
  });
});
