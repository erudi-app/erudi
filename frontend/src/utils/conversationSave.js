// The main-process half of "Export as Markdown" (#430).
//
// The renderer sends { title, text }. Main asks the user where to save with its
// own dialog and writes the text to the path the user chose: no path, option or
// dialog setting comes from the renderer, so a compromised page cannot write
// anywhere the user did not pick. Everything it touches is injected, so the
// behaviour is tested against fakes (`conversationSave.test.js`).
//
// The text is built in the renderer (the role headings and the compaction notes
// are the interface language's, which only the renderer holds) and arrives
// capped: a conversation of a few hundred thousand words is far under the cap.
import path from "node:path";
import { defaultExportName } from "./exportFileName";

/** Largest text main accepts, in UTF-16 code units (about 20 MB of prose). */
export const MAX_EXPORT_CHARS = 20_000_000;

const INVALID = { status: "invalid" };
const FAILED = { status: "error" };

/**
 * Whether an IPC event comes from the main frame of the app window: not from
 * another window, and not from a frame embedded in it.
 *
 * @param {{sender: object, senderFrame: object}} event
 * @param {{isDestroyed: () => boolean, webContents: object}|null} win
 */
export function isAppWindowEvent(event, win) {
  return (
    !!win &&
    !win.isDestroyed() &&
    event.sender === win.webContents &&
    event.senderFrame === win.webContents.mainFrame
  );
}

/**
 * @param {object} deps
 * @param {(event: object) => boolean} deps.isTrustedSender - true only for the app window
 * @param {(options: object) => Promise<{canceled: boolean, filePath?: string}>} deps.showSaveDialog
 * @param {(file: string) => Promise<{isSymbolicLink: () => boolean}>} deps.lstat - rejects
 *   with `code: "ENOENT"` when the file does not exist
 * @param {(file: string, text: string) => Promise<void>} deps.writeFile - UTF-8
 * @param {() => string} deps.getDefaultDirectory - where the dialog opens
 * @param {() => Date} deps.now
 * @param {() => string} deps.dialogTitle
 * @param {() => string} deps.filterName
 * @param {(message: string) => void} deps.logWarn
 * @param {(message: string, error?: unknown) => void} deps.logError
 * @returns {(event: object, payload: unknown) => Promise<{status: "saved"|"cancelled"|"invalid"|"error"}>}
 */
export function createConversationSave(deps) {
  // What lies at `file`: "none", "file" (or anything but a link), or "symlink".
  async function kindAt(file) {
    try {
      return (await deps.lstat(file)).isSymbolicLink() ? "symlink" : "file";
    } catch (error) {
      if (error && error.code === "ENOENT") {
        return "none";
      }
      throw error;
    }
  }

  return async function saveConversation(event, payload) {
    if (!deps.isTrustedSender(event)) {
      deps.logWarn("Refused an export request that did not come from the app window");
      return INVALID;
    }
    if (!payload || typeof payload !== "object") {
      return INVALID;
    }
    // Only these two fields are read; anything else the renderer adds is ignored.
    const { title, text } = payload;
    if (
      typeof title !== "string" ||
      typeof text !== "string" ||
      text.length === 0 ||
      text.length > MAX_EXPORT_CHARS
    ) {
      return INVALID;
    }
    try {
      const result = await deps.showSaveDialog({
        title: deps.dialogTitle(),
        defaultPath: path.join(deps.getDefaultDirectory(), defaultExportName(title, deps.now())),
        filters: [{ name: deps.filterName(), extensions: ["md"] }],
        properties: ["showOverwriteConfirmation"],
      });
      if (result.canceled || typeof result.filePath !== "string" || !result.filePath) {
        return { status: "cancelled" };
      }
      // A name typed without an extension gets `.md`. The dialog confirmed the
      // overwrite of the name the user typed, not of this one: a file already
      // there is not touched.
      const chosen = result.filePath;
      const target = path.extname(chosen) ? chosen : `${chosen}.md`;
      const found = await kindAt(target);
      if (found === "symlink" || (found !== "none" && target !== chosen)) {
        deps.logWarn(
          found === "symlink"
            ? "Refused to export a conversation through a symbolic link"
            : "Refused to export over a file the save dialog did not confirm"
        );
        return FAILED;
      }
      await deps.writeFile(target, text);
      return { status: "saved" };
    } catch (error) {
      // The export did not happen; the renderer tells the user. The text is
      // never logged: it is the conversation.
      deps.logError("Could not export a conversation to Markdown", error);
      return FAILED;
    }
  };
}

/**
 * The handler main.js registers, built from Electron's own objects (taken as
 * arguments so tests hand it fakes shaped like them).
 *
 * The write opens the file without following a symbolic link (`O_NOFOLLOW`
 * where the platform has it), so a link planted between the check and the write
 * does not redirect it.
 *
 * @param {object} electron
 * @param {{showSaveDialog: Function}} electron.dialog
 * @param {{promises: {lstat: Function, writeFile: Function}, constants: object}} electron.fs
 * @param {{getPath: (name: string) => string}} electron.app
 * @param {() => object|null} electron.getWindow - the app window
 * @param {(key: string) => string} electron.t - main's translator
 * @param {(message: string) => void} electron.logWarn
 * @param {(message: string, error?: unknown) => void} electron.logError
 */
export function createElectronConversationSave({
  dialog,
  fs,
  app,
  getWindow,
  t,
  logWarn,
  logError,
}) {
  const { O_WRONLY, O_CREAT, O_TRUNC, O_NOFOLLOW = 0 } = fs.constants;
  return createConversationSave({
    isTrustedSender: (event) => isAppWindowEvent(event, getWindow()),
    showSaveDialog: (options) => dialog.showSaveDialog(getWindow(), options),
    lstat: (file) => fs.promises.lstat(file),
    writeFile: (file, text) =>
      fs.promises.writeFile(file, text, {
        encoding: "utf8",
        flag: O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW,
      }),
    getDefaultDirectory: () => app.getPath("documents"),
    now: () => new Date(),
    dialogTitle: () => t("dialogs.exportConversation.title"),
    filterName: () => t("dialogs.exportConversation.filter"),
    logWarn,
    logError,
  });
}
