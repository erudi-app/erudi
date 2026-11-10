/**
 * One Erudi at a time.
 *
 * A second launch would spawn a second backend on the same data folder while
 * the first still runs (#706). main.js takes Electron's single-instance lock:
 * the second launch hands over to the first (which focuses its window) and
 * quits before it creates a window or spawns anything. main.js runs in the
 * Electron main process and cannot be imported here, so the wiring is read
 * from the source text.
 */
import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const source = readFileSync(fileURLToPath(new URL("./main.js", import.meta.url)), "utf8");

describe("single instance", () => {
  it("takes the single-instance lock and quits when another instance holds it", () => {
    // Packaged only: dev and the installed app share the lock's key, so
    // `npm start` would quit while the installed app runs.
    expect(source).toMatch(
      /const hasSingleInstanceLock = !app\.isPackaged \|\| app\.requestSingleInstanceLock\(\);/
    );
    expect(source).toMatch(/if \(!hasSingleInstanceLock\) \{[^}]*app\.quit\(\);/s);
  });

  it("focuses the existing window when another launch is attempted", () => {
    const handler = source.match(/app\.on\("second-instance",[\s\S]*?\n\}\);/);
    expect(handler).not.toBeNull();
    expect(handler[0]).toMatch(/mainWindow\.restore\(\)/);
    expect(handler[0]).toMatch(/mainWindow\.focus\(\)/);
    expect(handler[0]).toMatch(/createWindow\(\)/);
  });

  it("waits for a data folder the previous backend still holds", () => {
    expect(source).toMatch(/quietCodes: new Set\(\["DATA_DIR_IN_USE"\]\)/);
    expect(source).toMatch(/shouldWaitForDataDir\(code, Date\.now\(\) - dataDirWaitSince\)/);
  });

  it("never starts a window or a backend without the lock", () => {
    const ready = source.match(
      /\.whenReady\(\)\s*\.then\(async \(\) => \{([\s\S]*?)createWindow\(\);/
    );
    expect(ready).not.toBeNull();
    expect(ready[1]).toMatch(/if \(!hasSingleInstanceLock\) return;/);
  });
});
