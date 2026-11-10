import { getApiBaseUrl } from "../config/api.js";
import { tracedFetch } from "./api/client";
import { createLogger } from "../utils/logger";

const log = createLogger("arenaService");

export async function askArena({
  question,
  images = [],
  attachments = [],
  llmId,
  temperature,
  topP,
  quantize,
  customPrompt,
  signal,
  onStreamEvent,
}) {
  // Image-only asks are valid vision-model turns (#136 C); document-only asks
  // are valid "what does this say?" turns (#492).
  if (!question.trim() && images.length === 0 && attachments.length === 0) {
    throw new Error("Question is empty");
  }

  const res = await tracedFetch(`${getApiBaseUrl()}/arena/${llmId}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question: question,
      images: images,
      attachments: attachments,
      temperature: temperature,
      top_p: topP,
      quantize: quantize,
      custom_prompt: customPrompt,
    }),
    // Caller-owned AbortController: stopping a comparison aborts the stream (#136 H).
    signal,
  });
  if (!res.ok) {
    throw new Error("Arena query failed");
  }

  // NDJSON, one event per line: `answer` (text) and, for a failed turn, one
  // `error` shaped like a conversation's (its red line, and a curated turn's
  // `curated_error`). A line can span chunks; a chunk can hold several lines.
  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let answerText = "";
  let parseWarned = false;

  const handleLine = (rawLine) => {
    const text = rawLine.trim();
    if (!text) {
      return;
    }
    let event;
    try {
      event = JSON.parse(text);
    } catch (parseError) {
      // The line is skipped and the stream goes on: degraded, once per ask so
      // a garbled stream cannot flood the log.
      if (!parseWarned) {
        parseWarned = true;
        log.warn(`Skipping an unparseable Arena stream line (${text.length} chars)`, parseError);
      }
      return;
    }
    if (event?.t === "answer") {
      answerText += event.text || "";
    }
    onStreamEvent?.(event);
  };

  let streamDone = false;
  while (!streamDone) {
    const { done, value } = await reader.read();
    streamDone = done;
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    let newline;
    while ((newline = buffer.indexOf("\n")) !== -1) {
      handleLine(buffer.slice(0, newline));
      buffer = buffer.slice(newline + 1);
    }
  }
  handleLine(buffer + decoder.decode());

  return answerText.trim();
}
