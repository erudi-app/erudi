/**
 * One backend restart at a time.
 *
 * `backend:restart` stops the backend and spawns a new one without waiting for
 * it. Two calls in a row (a double click on an engine switch) would each spawn
 * one, and the two backends would fight over the embedded PostgreSQL data
 * directory. The handler is single-flight: a call made while a restart runs
 * gets that restart's promise. main.js runs in the Electron main process and
 * cannot be imported here, so the wiring is read from the source text; the
 * behaviour is tested in utils/singleFlight.test.js.
 */
import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const source = readFileSync(fileURLToPath(new URL("./main.js", import.meta.url)), "utf8");

describe("backend:restart", () => {
  it("is registered through a single-flight wrapper", () => {
    expect(source).toMatch(/require\("\.\/utils\/singleFlight"\)/);
    expect(source).toMatch(/ipcMain\.handle\(\s*"backend:restart",\s*singleFlight\(/);
  });
});
