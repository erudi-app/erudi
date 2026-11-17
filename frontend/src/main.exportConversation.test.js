/**
 * Markdown export wiring (#430).
 *
 * The handler's behaviour (dialog, trust check, validation, write) is pinned in
 * `utils/conversationSave.test.js`, against fakes shaped like Electron's
 * `dialog`, `fs` and `app`. main.js and preload.js run in Electron and cannot be
 * imported here, so this file pins the last two links: the preload bridge is run
 * against a fake `electron` and shown to carry a title and a text only; main.js
 * is read for the one line that registers the handler.
 */
import { describe, it, expect, vi } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const read = (name) => readFileSync(fileURLToPath(new URL(name, import.meta.url)), "utf8");

// Runs preload.js the way Electron does, with a `require` that serves a fake
// `electron`, and returns what it exposed to the page.
function runPreload() {
  const exposed = {};
  const invoke = vi.fn(async () => ({ status: "saved" }));
  const fakeElectron = {
    contextBridge: { exposeInMainWorld: (name, api) => (exposed[name] = api) },
    ipcRenderer: { invoke, send: vi.fn(), on: vi.fn(), removeListener: vi.fn() },
    webUtils: {},
  };
  const fakeRequire = (id) => {
    if (id === "electron") return fakeElectron;
    throw new Error(`unexpected require(${id})`);
  };
  new Function("require", "console", read("./preload.js"))(fakeRequire, { log: () => {} });
  return { exposed, invoke };
}

describe("the export bridge (preload)", () => {
  it("exposes saveMarkdown and nothing else under exportAPI", () => {
    const { exposed } = runPreload();
    expect(Object.keys(exposed.exportAPI)).toEqual(["saveMarkdown"]);
  });

  it("sends the title and the text on the export channel and returns main's status", async () => {
    const { exposed, invoke } = runPreload();
    const result = await exposed.exportAPI.saveMarkdown({ title: "T", text: "body" });
    expect(invoke).toHaveBeenCalledWith("conversation:saveMarkdown", { title: "T", text: "body" });
    expect(result).toEqual({ status: "saved" });
  });

  it("drops every other field the page passes, a path included", async () => {
    const { exposed, invoke } = runPreload();
    await exposed.exportAPI.saveMarkdown({
      title: "T",
      text: "body",
      path: "/etc/passwd",
      defaultPath: "/etc/passwd",
    });
    expect(invoke.mock.calls[0][1]).toEqual({ title: "T", text: "body" });
  });
});

describe("the export handler (main)", () => {
  const main = read("./main.js");

  it("is registered once, on the export channel, from the Electron-shaped factory", () => {
    expect(main.match(/"conversation:saveMarkdown"/g)).toHaveLength(1);
    expect(main).toMatch(
      /ipcMain\.handle\(\s*"conversation:saveMarkdown",\s*createElectronConversationSave\(\{\s*dialog,\s*fs,\s*app,/
    );
  });
});
