import { getApiBaseUrl } from "../config/api.js";
import { tracedFetch } from "./api/client";
export async function askArena({
  question,
  images = [],
  attachments = [],
  llmId,
  temperature,
  topP,
  maxNewTokens,
  quantize,
  customPrompt,
  signal,
  onStreamChunk,
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
      max_new_tokens: maxNewTokens,
      quantize: quantize,
      custom_prompt: customPrompt,
    }),
    // Caller-owned AbortController: stopping a comparison aborts the stream (#136 H).
    signal,
  });
  if (!res.ok) {
    throw new Error("Arena query failed");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let fullText = "";

  let streamDone = false;
  while (!streamDone) {
    const { done, value } = await reader.read();
    streamDone = done;
    if (done) {
      break;
    }
    const chunk = decoder.decode(value, { stream: true });
    fullText += chunk;
    onStreamChunk?.(chunk);
  }

  return fullText.trim();
}
