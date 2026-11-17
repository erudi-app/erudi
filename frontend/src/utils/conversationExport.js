import { API_BASE_URL } from "../config/api.js";
import { tracedFetch } from "../services/api/client";
import { formatDate } from "../i18n/format";
import { buildConversationMarkdown } from "./conversationMarkdown";

// The renderer's half of "Export as Markdown" (#430), shared by the history row
// and the conversation header. It fetches the conversation, writes its Markdown
// in the interface language and hands the title and the text to the main
// process (`window.exportAPI`), which opens the save dialog and writes the file.
// The renderer never names a path.

/** The labels of the Markdown, in the interface language of `t`. */
export function exportLabels(conversation, t) {
  const date = conversation.created_at ? formatDate(conversation.created_at) : "";
  return {
    untitled: t("chat:conversation.untitled"),
    user: t("chat:export.user"),
    assistant: t("chat:export.assistant"),
    attachment: t("chat:export.attachment"),
    image: t("chat:export.image"),
    created: date ? t("chat:export.created", { date }) : "",
    compaction: (count) =>
      count > 0 ? t("chat:export.compaction", { count }) : t("chat:export.compactionNoCount"),
  };
}

/**
 * Export one conversation to a Markdown file the user chooses.
 *
 * @param {number|string} id - The conversation to export.
 * @param {object} options
 * @param {Function} options.t - i18next `t`.
 * @param {Function} [options.fetcher] - `tracedFetch`, replaceable in tests.
 * @returns {Promise<{status: "saved"|"cancelled"}>}
 * @throws When the conversation cannot be read or is ephemeral, the bridge is
 *   missing, or the main process could not save.
 */
export async function exportConversation(id, { t, fetcher = tracedFetch }) {
  const save = window.exportAPI?.saveMarkdown;
  if (typeof save !== "function") {
    throw new Error("The export bridge is not available");
  }
  const response = await fetcher(`${API_BASE_URL}/conversations/${id}`);
  if (!response.ok) {
    throw new Error(`Could not read the conversation (HTTP ${response.status})`);
  }
  const conversation = await response.json();
  // An ephemeral conversation (#428) is never written to a file: the header
  // offers no export for it and its history row does not exist.
  if (conversation.ephemeral) {
    throw new Error("An ephemeral conversation is not exported");
  }
  const text = buildConversationMarkdown(conversation, exportLabels(conversation, t));
  const result = await save({ title: conversation.name, text });
  if (result?.status === "saved" || result?.status === "cancelled") {
    return { status: result.status };
  }
  throw new Error(`The export failed (${result?.status ?? "no answer"})`);
}
