/**
 * The levelled records the main process writes for its own failures.
 *
 * They must parse as levelled records in appLogTail (so the Diagnostics panel
 * shows them), carry the error's stack under the header (so the reader keeps
 * it as a continuation), and never throw whatever they are handed.
 */
import { describe, it, expect } from "vitest";

import { describeProcessGone, errorText, formatMainRecord, MAX_ERROR_CHARS } from "./mainLog";
import { detectLevel, parseAppLogRecords } from "./appLogTail";

describe("errorText", () => {
  it("prefers the stack, which carries the message", () => {
    const error = new Error("boom");
    expect(errorText(error)).toBe(error.stack);
  });

  it("falls back to the message, then to the value", () => {
    expect(errorText({ message: "no stack" })).toBe("no stack");
    expect(errorText("plain reason")).toBe("plain reason");
    expect(errorText(42)).toBe("42");
    expect(errorText(null)).toBe("");
    expect(errorText(undefined)).toBe("");
  });

  it("truncates an enormous stack", () => {
    const text = errorText({ stack: "x".repeat(MAX_ERROR_CHARS + 500) });
    expect(text.length).toBeLessThanOrEqual(MAX_ERROR_CHARS + 32);
    expect(text).toContain("[+500]");
  });

  it("never throws", () => {
    const hostile = {
      toString() {
        throw new Error("nope");
      },
    };
    expect(errorText(hostile)).toBe("[unprintable error]");
  });
});

describe("formatMainRecord", () => {
  it("states the level in the shape appLogTail reads", () => {
    expect(detectLevel(formatMainRecord("ERROR", "Backend process exited with code 1"))).toBe(
      "ERROR"
    );
    expect(detectLevel(formatMainRecord("WARN", "Updater check failed"))).toBe("WARNING");
  });

  it("puts the error's stack under the header as continuation lines", () => {
    const error = new Error("spawn ENOENT");
    const text = `[2026-09-06T10:00:00.000Z] ${formatMainRecord("ERROR", "Failed to start backend", error)}`;
    const [record] = parseAppLogRecords(text);
    expect(record.level).toBe("ERROR");
    expect(record.message).toContain("[main] ERROR Failed to start backend");
    expect(record.message).toContain("spawn ENOENT");
  });

  it("writes a single line when there is no error", () => {
    expect(formatMainRecord("ERROR", "gone")).toBe("[main] ERROR gone");
  });
});

describe("describeProcessGone", () => {
  it("names the kind, the reason and the exit code", () => {
    expect(describeProcessGone("renderer", { reason: "crashed", exitCode: 5 })).toBe(
      "renderer process gone: crashed, exit code 5"
    );
  });

  it("adds the child's name when Electron gives one", () => {
    expect(
      describeProcessGone("Utility", { reason: "killed", exitCode: 9, name: "network.mojom" })
    ).toBe("Utility process gone: killed, exit code 9, network.mojom");
  });

  it("survives an empty details object", () => {
    expect(describeProcessGone("GPU")).toBe("GPU process gone: unknown reason");
  });
});
