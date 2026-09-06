/**
 * The main process records what no `catch` sees.
 *
 * A crash of the renderer, of a Chromium child, or of the main process itself
 * used to leave nothing in `erudi-backend.log`: the window went blank or the
 * app vanished, and the Diagnostics panel had nothing to show for it. main.js
 * runs in the Electron main process and cannot be imported here, so the
 * handlers are read from the source text, the way the CSP test reads the
 * policy.
 */
import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const source = readFileSync(fileURLToPath(new URL("./main.js", import.meta.url)), "utf8");

describe("main-process crash handlers", () => {
  it("observes uncaught exceptions without replacing Electron's default handling", () => {
    expect(source).toMatch(/process\.on\("uncaughtExceptionMonitor",/);
    expect(source).not.toMatch(/process\.on\("uncaughtException",/);
  });

  it("records unhandled rejections, gone processes and a hung renderer at ERROR", () => {
    expect(source).toMatch(/process\.on\("unhandledRejection",[\s\S]{0,200}logError\(/);
    expect(source).toMatch(/app\.on\("child-process-gone",[\s\S]{0,200}logError\(/);
    expect(source).toMatch(/app\.on\("render-process-gone",[\s\S]{0,200}logError\(/);
    expect(source).toMatch(/webContents\.on\("unresponsive",[\s\S]{0,200}logError\(/);
  });

  it("introduces no dialog and no exit of its own", () => {
    const start = source.indexOf('process.on("uncaughtExceptionMonitor"');
    const lastHandler = source.indexOf('app.on("render-process-gone"');
    const handlers = source.slice(start, source.indexOf("});", lastHandler) + 3);
    expect(handlers).not.toMatch(/dialog\./);
    expect(handlers).not.toMatch(/app\.(quit|exit)\(/);
    expect(handlers).not.toMatch(/process\.exit\(/);
  });

  it("hands every backend exit to the start tracker, which owns the level", () => {
    const exitHandler = source.slice(
      source.indexOf('backendProcess.on("exit"'),
      source.indexOf('backendProcess.on("error"')
    );
    expect(exitHandler).toMatch(/tracker\.exited\(proc, code, signal\)/);
    expect(exitHandler).not.toMatch(/logError\(/);
  });

  it("records a start failure once: the tracker writes it, the supervisor only surfaces it", () => {
    // Inside startRealBackend every cause goes through failWith (the
    // tracker); no site writes its own ERROR next to it.
    const start = source.slice(
      source.indexOf("const startRealBackend"),
      source.indexOf("async function startBackendSupervised")
    );
    const errorCalls = start.match(/logError\(/g) || [];
    // The two allowed ERROR writers before the tracker exists: the missing
    // executable and the dev backend that does not answer.
    expect(errorCalls.length).toBeLessThanOrEqual(2);
    expect(start).not.toMatch(/logError\([^;]*startup_error/);
    const supervisor = source.slice(
      source.indexOf("async function startBackendSupervised"),
      source.indexOf("const createApplicationMenu")
    );
    expect(supervisor).toMatch(
      /log\(`Backend startup failed \(\$\{code\}\); surfacing to the user`\)/
    );
    expect(supervisor).not.toMatch(/logError\(/);
  });

  it("keeps the requested-stop state on the child process, never in a global", () => {
    expect(source).not.toMatch(/backendStopRequested/);
    expect(source).toMatch(/requestStop\(backendProcess\)/);
  });

  it("levels its own records in the shape the Diagnostics reader parses", () => {
    expect(source).toMatch(/const logWarn = .*formatMainRecord\("WARN"/);
    expect(source).toMatch(/const logError = .*formatMainRecord\("ERROR"/);
  });
});
