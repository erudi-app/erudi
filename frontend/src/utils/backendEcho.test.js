/**
 * The backend's stdout/stderr as main echoes it into the app log (#534).
 *
 * The fixtures are REAL lines: the output of the backend console formatter,
 * `CustomFormatter().format(record)` in backend/src/core/logging.py, for a
 * WARNING, an ERROR and a CRITICAL record. That formatter colours its output
 * even on a pipe, writes `[WARN]` and `[CRIT]` rather than the file's
 * `[WARNING]` / `[CRITICAL]`, and pads `[WARN]` with two spaces.
 */
import { describe, it, expect } from "vitest";

import { stripAnsi } from "./mainLog";
import { detectLevel, parseAppLogRecords } from "./appLogTail";
import { messageIdentity } from "./foldIdentity";
import { mergeRecentErrors } from "./diagnosticsReport";

// Copied from CustomFormatter().format(...) output, request id be-1.
const REAL_WARN =
  "\x1b[33;21m[WARN]  2026-11-17T13:42:25.689Z [be-1] - erudi - backend/src/domains/llms/services.py:f:l3 - boom\x1b[0m";
const REAL_ERROR =
  "\x1b[31;21m[ERROR] 2026-11-17T13:42:25.689Z [be-1] - erudi - backend/src/domains/llms/services.py:f:l3 - boom\x1b[0m";
const REAL_CRIT =
  "\x1b[31;1m[CRIT] 2026-11-17T13:42:25.689Z [be-1] - erudi - backend/src/domains/llms/services.py:f:l3 - boom\x1b[0m";
const REAL_INFO_QUOTING_A_LEVEL =
  "\x1b[38;21m[INFO]  2026-11-17T13:42:25.689Z [be-1] - erudi - backend/src/agents/runner.py:turn:l9 - question: why does it say [ERROR] here?\x1b[0m";

/** What main writes for a stdout line: ANSI stripped, then the prefix. */
const echo = (line, rid = "be-1", ts = "2026-11-17T13:42:25.689Z") =>
  `Backend stdout: ${stripAnsi(line)}`
    .replace("[be-1]", `[${rid}]`)
    .replace("2026-11-17T13:42:25.689Z", ts);

describe("stripAnsi", () => {
  it("removes the colour codes the backend formatter writes", () => {
    expect(stripAnsi(REAL_ERROR)).toBe(
      "[ERROR] 2026-11-17T13:42:25.689Z [be-1] - erudi - backend/src/domains/llms/services.py:f:l3 - boom"
    );
  });

  it("leaves text without codes alone", () => {
    expect(stripAnsi('{"event": "ready", "port": 27182}')).toBe(
      '{"event": "ready", "port": 27182}'
    );
  });
});

describe("detectLevel on real echoed lines", () => {
  it("reads WARN, ERROR and CRIT, normalised to the file's level names", () => {
    expect(detectLevel(echo(REAL_WARN))).toBe("WARNING");
    expect(detectLevel(echo(REAL_ERROR))).toBe("ERROR");
    expect(detectLevel(echo(REAL_CRIT))).toBe("CRITICAL");
  });

  it("still reads a line written with its colour codes (an older app log)", () => {
    expect(detectLevel(`Backend stdout: ${REAL_WARN}`)).toBe("WARNING");
  });

  it("never promotes an INFO line that quotes a level in its text", () => {
    // Privacy: INFO carries conversation content and stays off the page.
    expect(detectLevel(echo(REAL_INFO_QUOTING_A_LEVEL))).toBeNull();
    expect(detectLevel("[renderer:Chat] INFO pasted text [ERROR] x")).toBeNull();
  });
});

describe("folding real echoes", () => {
  it("gives three echoes of one ERROR with different request ids one identity", () => {
    const identities = new Set(
      [
        echo(REAL_ERROR, "be-1", "2026-11-17T13:42:25.689Z"),
        echo(REAL_ERROR, "be-2", "2026-11-17T13:42:26.120Z"),
        echo(REAL_ERROR, "be-3", "2026-11-17T13:43:01.004Z"),
      ].map(messageIdentity)
    );
    expect(identities.size).toBe(1);
  });

  it("folds them to one row with count 3 in the reader and in the merge", () => {
    const lines = [
      ["2026-11-17T13:42:25.700Z", echo(REAL_ERROR, "be-1", "2026-11-17T13:42:25.689Z")],
      ["2026-11-17T13:42:26.130Z", echo(REAL_ERROR, "be-2", "2026-11-17T13:42:26.120Z")],
      ["2026-11-17T13:43:01.010Z", echo(REAL_ERROR, "be-3", "2026-11-17T13:43:01.004Z")],
    ];
    const text = lines.map(([ts, message]) => `[${ts}] ${message}`).join("\n");

    const records = parseAppLogRecords(text);
    expect(records).toHaveLength(1);
    expect(records[0].count).toBe(3);
    expect(records[0].message).toContain("[be-3]");

    // The merge folds the same way when the records arrive unfolded.
    const entries = mergeRecentErrors({
      backend: null,
      appLog: lines.map(([timestamp, message]) => ({ timestamp, level: "ERROR", message })),
      sessionErrors: [],
    });
    expect(entries).toHaveLength(1);
    expect(entries[0].count).toBe(3);
  });

  it("folds warnings too, padded header included", () => {
    const text = [
      `[2026-11-17T13:42:25.700Z] ${echo(REAL_WARN, "be-1", "2026-11-17T13:42:25.689Z")}`,
      `[2026-11-17T13:42:26.130Z] ${echo(REAL_WARN, "-", "2026-11-17T13:42:26.120Z")}`,
    ].join("\n");
    const records = parseAppLogRecords(text);
    expect(records).toHaveLength(1);
    expect(records[0]).toMatchObject({ level: "WARNING", count: 2 });
  });
});
