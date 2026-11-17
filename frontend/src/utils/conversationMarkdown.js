import { ERROR_SENTINEL, baseName, decodeMarkerPath, getAttachmentNames } from "./messageContent";
import { manualCompactions, automaticCompactions } from "./traceEvents";

// One conversation as Markdown (#430): a pure function of what the backend
// returns for it and of the labels the caller translated.
//
// What it writes: the title, the messages in order under a role heading, a
// separator where a compaction took over, the names of the attachments and
// images. What it leaves out: the model's reasoning (the trace, never read
// except for its compactions), the failed turns, the summaries a compaction
// wrote (the model's working memory, not conversation), and any file path.

const DEFAULT_CONVERSATION_NAME = "New Conversation";

const SEPARATOR = "---";

const oneLine = (text) => String(text).replace(/\s+/g, " ").trim();

const compactionBlock = (event, labels) => {
  const count = Number.isInteger(event.summarized) && event.summarized > 0 ? event.summarized : 0;
  return `${SEPARATOR}\n\n*${labels.compaction(count)}*\n\n${SEPARATOR}`;
};

// The text a user message shows, minus the internal attachment markers.
const visibleText = (content) =>
  content
    .replace(/\[image\]/g, "")
    .replace(/\[image_path:[^\]]*\]/g, "")
    .replace(/\[file_path:[^\]]*\]/g, "")
    .trim();

const attachmentLines = (content, labels) => {
  const namedImages = [...content.matchAll(/\[image_path:([^\]]+)\]/g)].map(
    (match) => `> ${labels.image}: ${baseName(decodeMarkerPath(match[1]))}`
  );
  const unnamedImages = [...content.matchAll(/\[image\]/g)].map(() => `> ${labels.image}`);
  const files = getAttachmentNames(content).map((name) => `> ${labels.attachment}: ${name}`);
  return [...namedImages, ...unnamedImages, ...files];
};

/**
 * @param {object|null} conversation - `GET /conversations/{id}`: `name`,
 *   `messages[]` of `{ sender: "user"|"llm", content, trace }`.
 * @param {object} labels - Translated by the caller: `untitled`, `user`,
 *   `assistant`, `attachment`, `image`, `created` (the line under the title,
 *   "" for none) and `compaction(count)` (the separator's sentence; `count` is
 *   0 when the number of summarized messages is unknown).
 * @returns {string} The Markdown text, "" for no conversation.
 */
export function buildConversationMarkdown(conversation, labels) {
  if (!conversation) {
    return "";
  }
  const title =
    !conversation.name || conversation.name === DEFAULT_CONVERSATION_NAME
      ? labels.untitled
      : oneLine(conversation.name);
  const blocks = [`# ${title}`];
  if (labels.created) {
    blocks.push(labels.created);
  }

  for (const message of Array.isArray(conversation.messages) ? conversation.messages : []) {
    const isUser = message.sender === "user";
    const content = typeof message.content === "string" ? message.content : "";
    const trace = isUser ? null : message.trace;
    for (const event of automaticCompactions(trace)) {
      blocks.push(compactionBlock(event, labels));
    }
    if (!content.includes(ERROR_SENTINEL)) {
      const text = visibleText(content);
      const notes = isUser ? attachmentLines(content, labels) : [];
      if (text || notes.length > 0) {
        const body = [text, notes.join("\n")].filter(Boolean).join("\n\n");
        blocks.push(`## ${isUser ? labels.user : labels.assistant}\n\n${body}`);
      }
    }
    for (const event of manualCompactions(trace)) {
      blocks.push(compactionBlock(event, labels));
    }
  }
  return `${blocks.join("\n\n")}\n`;
}
