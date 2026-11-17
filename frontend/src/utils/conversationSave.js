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

/**
 * @param {object} deps
 * @param {(event: object) => boolean} deps.isTrustedSender - true only for the app window
 * @param {(options: object) => Promise<{canceled: boolean, filePath?: string}>} deps.showSaveDialog
 * @param {(file: string, data: string, encoding: string) => Promise<void>} deps.writeFile
 * @param {() => string} deps.getDefaultDirectory - where the dialog opens
 * @param {() => Date} deps.now
 * @param {() => string} deps.dialogTitle
 * @param {() => string} deps.filterName
 * @param {(message: string) => void} deps.logWarn
 * @param {(message: string, error?: unknown) => void} deps.logError
 * @returns {(event: object, payload: unknown) => Promise<{status: "saved"|"cancelled"|"invalid"|"error"}>}
 */
export function createConversationSave(deps) {
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
      });
      if (result.canceled || typeof result.filePath !== "string" || !result.filePath) {
        return { status: "cancelled" };
      }
      await deps.writeFile(result.filePath, text, "utf8");
      return { status: "saved" };
    } catch (error) {
      // The export did not happen; the renderer tells the user. The text is
      // never logged: it is the conversation.
      deps.logError("Could not export a conversation to Markdown", error);
      return { status: "error" };
    }
  };
}
