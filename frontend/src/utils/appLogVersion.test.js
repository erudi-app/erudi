/**
 * #576 — the app log belongs to the running app version, like backend.log.
 *
 * At the first launch of another version, `erudi-backend.log` becomes
 * `erudi-backend.old.log` and a fresh file starts; a marker line opens every
 * version's log, and the Diagnostics reader drops what precedes the last one
 * (the fallback when the rename failed).
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import fs, { mkdtempSync, readFileSync, rmSync, writeFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  APP_LOG_VERSION_MARKER,
  appLogSwitchAllowed,
  appLogVersionLines,
  switchAppLogForVersion,
} from "./appLogVersion.js";
import { parseAppLogRecords } from "./appLogTail.js";

const OLD_LINE = "[2026-09-03T10:00:00.000Z] [main] ERROR an older version's failure\n";

let dir;
let paths;
beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "erudi-applogversion-"));
  paths = {
    logFile: join(dir, "erudi-backend.log"),
    oldLogFile: join(dir, "erudi-backend.old.log"),
    versionFile: join(dir, "erudi-backend.log.version"),
  };
});
afterEach(() => rmSync(dir, { recursive: true, force: true }));

describe("switchAppLogForVersion", () => {
  it("moves an older version's log aside at the first launch of another version", () => {
    writeFileSync(paths.logFile, OLD_LINE);
    writeFileSync(paths.oldLogFile, "the one before\n");
    writeFileSync(paths.versionFile, "1.2.0");

    const result = switchAppLogForVersion({ fs, ...paths, version: "1.3.0" });

    expect(result).toEqual({ version: "1.3.0", previous: "1.2.0", rotated: true, error: null });
    expect(existsSync(paths.logFile)).toBe(false);
    expect(readFileSync(paths.oldLogFile, "utf8")).toBe(OLD_LINE);
    expect(readFileSync(paths.versionFile, "utf8")).toBe("1.3.0");
  });

  it("keeps the log of the same version (a restart, a reinstall)", () => {
    writeFileSync(paths.logFile, OLD_LINE);
    writeFileSync(paths.versionFile, "1.3.0");

    expect(switchAppLogForVersion({ fs, ...paths, version: "1.3.0" })).toBeNull();
    expect(readFileSync(paths.logFile, "utf8")).toBe(OLD_LINE);
    expect(existsSync(paths.oldLogFile)).toBe(false);
  });

  it("rolls over a log written before versioned logs existed", () => {
    writeFileSync(paths.logFile, OLD_LINE);

    const result = switchAppLogForVersion({ fs, ...paths, version: "1.3.0" });

    expect(result.previous).toBeNull();
    expect(result.rotated).toBe(true);
  });

  it("has nothing to move on a fresh install", () => {
    const result = switchAppLogForVersion({ fs, ...paths, version: "1.3.0" });

    expect(result).toEqual({ version: "1.3.0", previous: null, rotated: false, error: null });
    expect(readFileSync(paths.versionFile, "utf8")).toBe("1.3.0");
  });

  it("survives a log another process holds, and still records the version", () => {
    writeFileSync(paths.logFile, OLD_LINE);
    writeFileSync(paths.versionFile, "1.2.0");
    const heldFs = {
      ...fs,
      renameSync: () => {
        const error = new Error("EBUSY: resource busy or locked, rename");
        error.code = "EBUSY";
        throw error;
      },
    };

    const result = switchAppLogForVersion({ fs: heldFs, ...paths, version: "1.3.0" });

    expect(result.rotated).toBe(false);
    expect(result.error).toContain("EBUSY");
    expect(readFileSync(paths.logFile, "utf8")).toBe(OLD_LINE);
    expect(readFileSync(paths.versionFile, "utf8")).toBe("1.3.0");
  });

  it("rewrites the version file at every launch, so it lives as long as the log", () => {
    // Temp cleaners delete files by age: a version file written only on a
    // switch would go first, and the next launch would then roll the running
    // version's own log away.
    writeFileSync(paths.logFile, OLD_LINE);
    writeFileSync(paths.versionFile, "1.3.0");
    const writes = [];
    const recordingFs = {
      ...fs,
      writeFileSync: (file, data) => {
        writes.push([file, data]);
        fs.writeFileSync(file, data);
      },
    };

    expect(switchAppLogForVersion({ fs: recordingFs, ...paths, version: "1.3.0" })).toBeNull();
    expect(writes).toEqual([[paths.versionFile, "1.3.0"]]);
    expect(readFileSync(paths.logFile, "utf8")).toBe(OLD_LINE);
  });

  it("does nothing without a version", () => {
    writeFileSync(paths.logFile, OLD_LINE);
    expect(switchAppLogForVersion({ fs, ...paths, version: "" })).toBeNull();
    expect(existsSync(paths.versionFile)).toBe(false);
  });
});

describe("appLogSwitchAllowed", () => {
  it("switches in the packaged app that holds the single-instance lock", () => {
    expect(appLogSwitchAllowed({ isPackaged: true, hasSingleInstanceLock: true })).toBe(true);
  });

  it("never switches in a second instance, which hands over and quits", () => {
    expect(appLogSwitchAllowed({ isPackaged: true, hasSingleInstanceLock: false })).toBe(false);
  });

  it("never switches in a dev run, which shares the log with an installed app", () => {
    // `npm start` reports the package.json version and takes no lock; it
    // must not roll the installed app's log away.
    expect(appLogSwitchAllowed({ isPackaged: false, hasSingleInstanceLock: true })).toBe(false);
  });
});

describe("appLogVersionLines", () => {
  it("opens a switched log with the marker, then the failure, then the version", () => {
    const lines = appLogVersionLines(
      { version: "1.3.0", previous: "1.2.0", rotated: false, error: "EBUSY" },
      "1.3.0"
    );

    expect(lines[0].startsWith(APP_LOG_VERSION_MARKER)).toBe(true);
    expect(lines[0]).toContain("1.3.0");
    expect(lines[1]).toMatch(/^\[main\] WARN .*EBUSY/);
    expect(lines[lines.length - 1]).toContain("app version 1.3.0");
  });

  it("writes no marker when the version did not change", () => {
    const lines = appLogVersionLines(null, "1.3.0");
    expect(lines.some((line) => line.startsWith(APP_LOG_VERSION_MARKER))).toBe(false);
    expect(lines).toHaveLength(1);
  });
});

describe("parseAppLogRecords and the version marker", () => {
  const marker = (version) => `[2026-11-17T09:00:00.000Z] ${APP_LOG_VERSION_MARKER} ${version}\n`;
  const today = "[2026-11-17T09:00:05.000Z] [main] ERROR a failure of today\n";

  it("drops every record before the last marker", () => {
    const text = OLD_LINE + marker("1.2.0") + OLD_LINE + marker("1.3.0") + today;
    const records = parseAppLogRecords(text);
    expect(records.map((r) => r.message)).toEqual(["[main] ERROR a failure of today"]);
  });

  it("keeps everything when there is no marker", () => {
    expect(parseAppLogRecords(OLD_LINE + today)).toHaveLength(2);
  });

  it("is not fooled by the backend's own marker echoed on stdout", () => {
    const echoed =
      "[2026-11-17T09:00:01.000Z] Backend stdout: [INFO] 2026-11-17T09:00:01.000Z [-] - erudi" +
      " - run.py:1 - Diagnostics starts here: backend.log now belongs to app version 1.3.0\n";
    expect(parseAppLogRecords(OLD_LINE + echoed + today)).toHaveLength(2);
  });
});
