/**
 * Markdown export wiring (#430).
 *
 * The handler's behaviour is pinned in `utils/conversationSave.test.js`, against
 * fakes. main.js and preload.js run in Electron and cannot be imported here, so
 * what this file pins is how they are wired, read from the source text: the
 * channel, the dialog being main's own, the trust check, and a bridge that
 * carries a title and a text and nothing else.
 */
import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const read = (name) => readFileSync(fileURLToPath(new URL(name, import.meta.url)), "utf8");
const main = read("./main.js");
const preload = read("./preload.js");

describe("conversation export wiring", () => {
  const handler = main.match(
    /ipcMain\.handle\(\s*"conversation:saveMarkdown",\s*createConversationSave\(\{[\s\S]*?\n\s*\}\)\s*\);/
  );

  it("registers one handler built by createConversationSave", () => {
    expect(handler).not.toBeNull();
    expect(main.match(/"conversation:saveMarkdown"/g)).toHaveLength(1);
  });

  it("opens the save dialog from the main window and writes with the main process's fs", () => {
    expect(handler[0]).toMatch(/dialog\.showSaveDialog\(mainWindow, options\)/);
    expect(handler[0]).toMatch(/fs\.promises\.writeFile\(file, data, encoding\)/);
  });

  it("answers only the app window", () => {
    expect(handler[0]).toMatch(/event\.sender === mainWindow\.webContents/);
  });

  it("opens the dialog in the user's documents folder, not in a renderer-given one", () => {
    expect(handler[0]).toMatch(/app\.getPath\("documents"\)/);
  });

  it("exposes a bridge that sends a title and a text, no path, and takes nothing back but a status", () => {
    const bridge = preload.match(/exposeInMainWorld\("exportAPI",\s*\{([\s\S]*?)\}\);/);
    expect(bridge).not.toBeNull();
    expect(bridge[1]).toMatch(/saveMarkdown: \(\{ title, text \}\) =>/);
    expect(bridge[1]).toMatch(/invoke\("conversation:saveMarkdown", \{ title, text \}\)/);
    expect(bridge[1]).not.toMatch(/path/i);
    expect((bridge[1].match(/:\s*\(/g) || []).length).toBe(1);
  });
});
