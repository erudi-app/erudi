/**
 * The app-log reader behind `diagnostics:appLogTail`.
 *
 * The properties under test are the same three the backend reader has: the
 * read is bounded, the filter is default-exclude (a line that declares no
 * level never reaches a panel meant to be pasted into a public issue), and a
 * missing file is an empty result rather than a throw.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  detectLevel,
  parseAppLogRecords,
  readTailSync,
  readAppLogTail,
  MAX_MESSAGE_CHARS,
} from "./appLogTail";

const line = (text, ts = "2026-09-05T23:03:14.797Z") => `[${ts}] ${text}`;

let dir;
beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "erudi-logtail-"));
});
afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

describe("detectLevel", () => {
  it("reads the backend formatter's level out of a stdout line", () => {
    expect(detectLevel("Backend stdout: [ERROR] 2026-09-05T23:03:15.036Z [-] - erudi - x")).toBe(
      "ERROR"
    );
    expect(detectLevel("Backend stdout: [WARNING] ...")).toBe("WARNING");
    expect(detectLevel("Backend stdout: [CRITICAL] ...")).toBe("CRITICAL");
  });

  it("reads the renderer bridge's level and normalises WARN", () => {
    expect(detectLevel("[renderer:App] ERROR Backend startup error")).toBe("ERROR");
    expect(detectLevel("[renderer:App] WARN slow")).toBe("WARNING");
  });

  it("reads the main process's own level", () => {
    expect(detectLevel("[main] ERROR Backend process exited with code 1, signal null")).toBe(
      "ERROR"
    );
    expect(detectLevel("[main] WARN Updater: periodic check failed - offline")).toBe("WARNING");
  });

  it("returns null for a line that declares no level", () => {
    expect(detectLevel("Backend stdout: [INFO] all good")).toBeNull();
    expect(detectLevel("[renderer:App] INFO Backend is ready")).toBeNull();
    expect(detectLevel("Creating main window...")).toBeNull();
    expect(detectLevel("[main] INFO Backend is ready on port 27182.")).toBeNull();
  });
});

describe("parseAppLogRecords", () => {
  it("keeps levelled records and drops the rest", () => {
    const text = [
      line("Creating main window..."),
      line("Backend stdout: [INFO] Engine chosen"),
      line("Backend stdout: [ERROR] boom"),
      line("[renderer:App] WARN slow"),
    ].join("\n");
    expect(parseAppLogRecords(text).map((r) => r.level)).toEqual(["ERROR", "WARNING"]);
  });

  it("keeps the continuation lines of a kept record", () => {
    const text = [line("Backend stdout: [ERROR] boom"), "  at frame one", "  at frame two"].join(
      "\n"
    );
    const [record] = parseAppLogRecords(text);
    expect(record.message).toContain("at frame one");
    expect(record.message).toContain("at frame two");
  });

  it("drops the continuation lines of a filtered record", () => {
    // The privacy direction: an unlevelled or INFO record's body never leaks
    // through its continuations either.
    const text = [
      line("[renderer:QuestionInput] INFO input SECRET-PROMPT-ALPHA"),
      "  more: SECRET-PROMPT-BRAVO",
      line("Backend stdout: [ERROR] boom"),
    ].join("\n");
    const records = parseAppLogRecords(text);
    const blob = JSON.stringify(records);
    expect(blob).not.toContain("SECRET-PROMPT-ALPHA");
    expect(blob).not.toContain("SECRET-PROMPT-BRAVO");
    expect(records).toHaveLength(1);
  });

  it("drops a leading orphan continuation", () => {
    const text = ["  SECRET-ORPHAN", line("Backend stdout: [ERROR] boom")].join("\n");
    expect(JSON.stringify(parseAppLogRecords(text))).not.toContain("SECRET-ORPHAN");
  });

  it("returns the newest records last and honours the limit", () => {
    const text = Array.from({ length: 10 }, (_, i) => line(`Backend stdout: [ERROR] e${i}`)).join(
      "\n"
    );
    expect(parseAppLogRecords(text, 3).map((r) => r.message.slice(-2))).toEqual(["e7", "e8", "e9"]);
  });

  it("truncates an enormous record", () => {
    const text = line(`Backend stdout: [ERROR] ${"z".repeat(20000)}`);
    const [record] = parseAppLogRecords(text);
    expect(record.message.length).toBeLessThanOrEqual(MAX_MESSAGE_CHARS + 32);
  });

  it("treats the file's final newline as a terminator, not a blank line", () => {
    const [record] = parseAppLogRecords(`${line("Backend stdout: [ERROR] boom")}\n`);
    expect(record.message).toBe("Backend stdout: [ERROR] boom");
  });

  it("survives null and empty input", () => {
    expect(parseAppLogRecords(null)).toEqual([]);
    expect(parseAppLogRecords("")).toEqual([]);
  });
});

describe("readTailSync", () => {
  it("returns a small file whole", () => {
    const file = join(dir, "app.log");
    writeFileSync(file, "a\nb\n");
    expect(readTailSync(file, 1024)).toBe("a\nb\n");
  });

  it("reads a bounded window at the end of a large file", () => {
    const file = join(dir, "app.log");
    writeFileSync(file, `${"X".repeat(5000)}\nTAIL\n`);
    const text = readTailSync(file, 100);
    expect(text).toContain("TAIL");
    expect(text.length).toBeLessThanOrEqual(100);
    expect(text).not.toContain("X".repeat(200));
  });

  it("returns an empty string for a missing file instead of throwing", () => {
    expect(readTailSync(join(dir, "nope.log"), 1024)).toBe("");
  });
});

describe("readAppLogTail", () => {
  it("reads levelled records from a real file", () => {
    const file = join(dir, "app.log");
    writeFileSync(file, [line("App ready."), line("Backend stdout: [ERROR] boom"), ""].join("\n"));
    expect(readAppLogTail(file).map((r) => r.level)).toEqual(["ERROR"]);
  });

  it("returns nothing for a missing file", () => {
    expect(readAppLogTail(join(dir, "nope.log"))).toEqual([]);
  });
});
