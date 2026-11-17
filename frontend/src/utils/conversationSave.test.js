import path from "node:path";
import { describe, it, expect, vi } from "vitest";
import { createConversationSave, MAX_EXPORT_CHARS } from "./conversationSave";

// The main-process side of the Markdown export (#430). The renderer sends a
// title and the text; main asks the user where to save, with its own dialog,
// and writes there. These tests run the handler against fakes that behave like
// Electron's dialog and the filesystem.

const TRUSTED = { sender: "main-window" };

function setup({
  dialogResult = { canceled: false, filePath: "/chosen/out.md" },
  writeError,
} = {}) {
  const writes = [];
  const dialogCalls = [];
  const deps = {
    isTrustedSender: (event) => event === TRUSTED,
    showSaveDialog: vi.fn(async (options) => {
      dialogCalls.push(options);
      return dialogResult;
    }),
    writeFile: vi.fn(async (file, data, encoding) => {
      if (writeError) throw writeError;
      writes.push({ file, data, encoding });
    }),
    getDefaultDirectory: () => "/home/me/Documents",
    now: () => new Date(2026, 10, 3),
    log: vi.fn(),
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
    expect(writes).toEqual([{ file: "/chosen/out.md", data: "# Airships\n", encoding: "utf8" }]);
    expect(result).toEqual({ status: "saved" });
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
