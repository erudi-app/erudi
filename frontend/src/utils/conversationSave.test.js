import path from "node:path";
import { describe, it, expect, vi } from "vitest";
import {
  createConversationSave,
  createElectronConversationSave,
  isAppWindowEvent,
  MAX_EXPORT_CHARS,
} from "./conversationSave";

// The main-process side of the Markdown export (#430). The renderer sends a
// title and the text; main asks the user where to save, with its own dialog,
// and writes there. These tests run the handler against fakes that behave like
// Electron's dialog and the filesystem.

const TRUSTED = { sender: "main-window" };

const enoent = () => Object.assign(new Error("ENOENT"), { code: "ENOENT" });

// A file system that knows its files: `files` maps a path to "file" | "symlink".
function setup({
  dialogResult = { canceled: false, filePath: "/chosen/out.md" },
  writeError,
  files = {},
} = {}) {
  const writes = [];
  const dialogCalls = [];
  const deps = {
    isTrustedSender: (event) => event === TRUSTED,
    showSaveDialog: vi.fn(async (options) => {
      dialogCalls.push(options);
      return dialogResult;
    }),
    lstat: vi.fn(async (file) => {
      if (!(file in files)) throw enoent();
      return { isSymbolicLink: () => files[file] === "symlink" };
    }),
    writeFile: vi.fn(async (file, data) => {
      if (writeError) throw writeError;
      writes.push({ file, data });
    }),
    getDefaultDirectory: () => "/home/me/Documents",
    now: () => new Date(2026, 10, 3),
    logWarn: vi.fn(),
    logError: vi.fn(),
    dialogTitle: () => "Export conversation",
    filterName: () => "Markdown",
  };
  return { handler: createConversationSave(deps), deps, writes, dialogCalls };
}

describe("conversation export handler", () => {
  it("opens the save dialog itself, proposes a safe name, and writes where the user chose", async () => {
    const { handler, writes, dialogCalls } = setup();
    const result = await handler(TRUSTED, { title: "Airships: a story", text: "# Airships\n" });

    expect(dialogCalls).toHaveLength(1);
    expect(dialogCalls[0].defaultPath).toBe(
      path.join("/home/me/Documents", "Airships a story 2026-11-03.md")
    );
    expect(dialogCalls[0].filters).toEqual([{ name: "Markdown", extensions: ["md"] }]);
    expect(writes).toEqual([{ file: "/chosen/out.md", data: "# Airships\n" }]);
    expect(result).toEqual({ status: "saved" });
  });

  it("asks the system to confirm before it overwrites a file", async () => {
    const { handler, dialogCalls } = setup();
    await handler(TRUSTED, { title: "T", text: "x" });
    expect(dialogCalls[0].properties).toContain("showOverwriteConfirmation");
  });

  it("writes nothing when the user cancels", async () => {
    const { handler, writes } = setup({ dialogResult: { canceled: true } });
    expect(await handler(TRUSTED, { title: "T", text: "x" })).toEqual({ status: "cancelled" });
    expect(writes).toHaveLength(0);
  });

  it("never reads a path from the renderer: extra fields are ignored", async () => {
    const { handler, writes, dialogCalls } = setup();
    await handler(TRUSTED, {
      title: "T",
      text: "x",
      path: "/etc/passwd",
      filePath: "/etc/passwd",
      defaultPath: "/etc/passwd",
    });
    expect(dialogCalls[0].defaultPath.startsWith(path.join("/home/me/Documents", "T "))).toBe(true);
    expect(writes[0].file).toBe("/chosen/out.md");
  });

  it("keeps a path-like title inside the default directory", async () => {
    const { handler, dialogCalls } = setup();
    await handler(TRUSTED, { title: "../../etc/passwd", text: "x" });
    expect(dialogCalls[0].defaultPath).toBe(
      path.join("/home/me/Documents", "etc passwd 2026-11-03.md")
    );
  });

  describe("the file name the user typed", () => {
    it("gets .md when it has no extension", async () => {
      const { handler, writes } = setup({
        dialogResult: { canceled: false, filePath: "/chosen/notes" },
      });
      expect(await handler(TRUSTED, { title: "T", text: "x" })).toEqual({ status: "saved" });
      expect(writes[0].file).toBe("/chosen/notes.md");
    });

    it("keeps an extension the user chose", async () => {
      const { handler, writes } = setup({
        dialogResult: { canceled: false, filePath: "/chosen/notes.txt" },
      });
      await handler(TRUSTED, { title: "T", text: "x" });
      expect(writes[0].file).toBe("/chosen/notes.txt");
    });

    it("refuses to overwrite the file the added .md lands on: the dialog never confirmed it", async () => {
      const { handler, writes, deps } = setup({
        dialogResult: { canceled: false, filePath: "/chosen/notes" },
        files: { "/chosen/notes.md": "file" },
      });
      expect(await handler(TRUSTED, { title: "T", text: "x" })).toEqual({ status: "error" });
      expect(writes).toHaveLength(0);
      expect(deps.logWarn).toHaveBeenCalledTimes(1);
    });
  });

  describe("the target already exists", () => {
    it("overwrites a regular file the dialog confirmed", async () => {
      const { handler, writes } = setup({ files: { "/chosen/out.md": "file" } });
      expect(await handler(TRUSTED, { title: "T", text: "x" })).toEqual({ status: "saved" });
      expect(writes).toHaveLength(1);
    });

    it("refuses to write through a symbolic link", async () => {
      const { handler, writes, deps } = setup({ files: { "/chosen/out.md": "symlink" } });
      expect(await handler(TRUSTED, { title: "T", text: "x" })).toEqual({ status: "error" });
      expect(writes).toHaveLength(0);
      expect(deps.logWarn).toHaveBeenCalledTimes(1);
    });

    it("refuses a symbolic link reached through the added .md too", async () => {
      const { handler, writes } = setup({
        dialogResult: { canceled: false, filePath: "/chosen/notes" },
        files: { "/chosen/notes.md": "symlink" },
      });
      expect(await handler(TRUSTED, { title: "T", text: "x" })).toEqual({ status: "error" });
      expect(writes).toHaveLength(0);
    });

    it("reports an unreadable target instead of writing blind", async () => {
      const { handler, writes, deps } = setup();
      deps.lstat.mockRejectedValueOnce(Object.assign(new Error("EACCES"), { code: "EACCES" }));
      expect(await handler(TRUSTED, { title: "T", text: "x" })).toEqual({ status: "error" });
      expect(writes).toHaveLength(0);
      expect(deps.logError).toHaveBeenCalledTimes(1);
    });
  });

  it.each([
    ["no payload", undefined],
    ["a string payload", "hello"],
    ["a non-string text", { title: "T", text: 42 }],
    ["an empty text", { title: "T", text: "" }],
    ["a non-string title", { title: { a: 1 }, text: "x" }],
    ["a text over the cap", { title: "T", text: "x".repeat(MAX_EXPORT_CHARS + 1) }],
  ])("refuses %s without opening a dialog or writing", async (_name, payload) => {
    const { handler, deps, writes } = setup();
    expect(await handler(TRUSTED, payload)).toEqual({ status: "invalid" });
    expect(deps.showSaveDialog).not.toHaveBeenCalled();
    expect(writes).toHaveLength(0);
  });

  it("accepts a text exactly at the cap", async () => {
    const { handler, writes } = setup();
    const result = await handler(TRUSTED, { title: "T", text: "x".repeat(MAX_EXPORT_CHARS) });
    expect(result).toEqual({ status: "saved" });
    expect(writes).toHaveLength(1);
  });

  it("refuses a sender that is not the app window", async () => {
    const { handler, deps, writes } = setup();
    expect(await handler({ sender: "someone-else" }, { title: "T", text: "x" })).toEqual({
      status: "invalid",
    });
    expect(deps.showSaveDialog).not.toHaveBeenCalled();
    expect(writes).toHaveLength(0);
    expect(deps.logWarn).toHaveBeenCalled();
  });

  it("reports a failed write and logs it once, without the text", async () => {
    const { handler, deps } = setup({ writeError: new Error("EACCES: permission denied") });
    const result = await handler(TRUSTED, { title: "T", text: "SECRET BODY" });
    expect(result).toEqual({ status: "error" });
    expect(deps.logError).toHaveBeenCalledTimes(1);
    expect(JSON.stringify(deps.logError.mock.calls)).not.toContain("SECRET BODY");
  });

  it("reports a failed dialog", async () => {
    const { handler, deps } = setup();
    deps.showSaveDialog.mockRejectedValueOnce(new Error("no window"));
    expect(await handler(TRUSTED, { title: "T", text: "x" })).toEqual({ status: "error" });
    expect(deps.logError).toHaveBeenCalledTimes(1);
  });
});

describe("isAppWindowEvent", () => {
  const frame = { id: "main-frame" };
  const makeWindow = (overrides = {}) => ({
    isDestroyed: () => false,
    webContents: { mainFrame: frame },
    ...overrides,
  });
  const eventFrom = (win, senderFrame = frame) => ({ sender: win.webContents, senderFrame });

  it("accepts the main frame of the app window", () => {
    const win = makeWindow();
    expect(isAppWindowEvent(eventFrom(win), win)).toBe(true);
  });

  it("refuses a sub-frame of the app window (an embedded page)", () => {
    const win = makeWindow();
    expect(isAppWindowEvent(eventFrom(win, { id: "iframe" }), win)).toBe(false);
  });

  it("refuses another web contents", () => {
    const win = makeWindow();
    expect(isAppWindowEvent({ sender: {}, senderFrame: frame }, win)).toBe(false);
  });

  it("refuses everything when there is no window or it is destroyed", () => {
    const win = makeWindow();
    expect(isAppWindowEvent(eventFrom(win), null)).toBe(false);
    expect(isAppWindowEvent(eventFrom(win), makeWindow({ isDestroyed: () => true }))).toBe(false);
  });
});

// The handler main registers, built on Electron-shaped fakes: what main.js does
// with `dialog`, `fs`, `app` and the window, without Electron.
describe("the handler main registers", () => {
  const frame = { id: "main-frame" };
  const O_NOFOLLOW = 0x100;

  function electronSetup() {
    const win = { isDestroyed: () => false, webContents: { mainFrame: frame } };
    const calls = { dialog: [], writes: [] };
    const dialog = {
      showSaveDialog: vi.fn(async (parent, options) => {
        calls.dialog.push({ parent, options });
        return { canceled: false, filePath: "/chosen/out.md" };
      }),
    };
    const fs = {
      constants: { O_WRONLY: 1, O_CREAT: 0x200, O_TRUNC: 0x400, O_NOFOLLOW },
      promises: {
        lstat: vi.fn(async () => {
          throw enoent();
        }),
        writeFile: vi.fn(async (file, data, options) => {
          calls.writes.push({ file, data, options });
        }),
      },
    };
    const app = { getPath: vi.fn((name) => `/home/me/${name}`) };
    const logWarn = vi.fn();
    const logError = vi.fn();
    const t = (key) => `t(${key})`;
    const handler = createElectronConversationSave({
      dialog,
      fs,
      app,
      getWindow: () => win,
      t,
      logWarn,
      logError,
    });
    return { handler, win, calls, app, fs, logWarn };
  }

  it("opens the dialog on the app window, in the documents folder, with translated labels", async () => {
    const { handler, win, calls, app } = electronSetup();
    await handler({ sender: win.webContents, senderFrame: frame }, { title: "T", text: "x" });
    expect(app.getPath).toHaveBeenCalledWith("documents");
    expect(calls.dialog[0].parent).toBe(win);
    expect(calls.dialog[0].options.title).toBe("t(dialogs.exportConversation.title)");
    expect(calls.dialog[0].options.filters[0].name).toBe("t(dialogs.exportConversation.filter)");
    expect(
      calls.dialog[0].options.defaultPath.startsWith(path.join("/home/me/documents", "T "))
    ).toBe(true);
  });

  it("writes UTF-8 text that cannot follow a symbolic link planted after the check", async () => {
    const { handler, win, calls } = electronSetup();
    const result = await handler(
      { sender: win.webContents, senderFrame: frame },
      { title: "T", text: "héllo" }
    );
    expect(result).toEqual({ status: "saved" });
    expect(calls.writes).toHaveLength(1);
    expect(calls.writes[0].file).toBe("/chosen/out.md");
    expect(calls.writes[0].data).toBe("héllo");
    expect(calls.writes[0].options.encoding).toBe("utf8");
    expect(calls.writes[0].options.flag & O_NOFOLLOW).toBe(O_NOFOLLOW);
  });

  it("answers nothing to a frame that is not the window's main frame", async () => {
    const { handler, win, calls } = electronSetup();
    const result = await handler(
      { sender: win.webContents, senderFrame: { id: "iframe" } },
      { title: "T", text: "x" }
    );
    expect(result).toEqual({ status: "invalid" });
    expect(calls.dialog).toHaveLength(0);
    expect(calls.writes).toHaveLength(0);
  });
});
