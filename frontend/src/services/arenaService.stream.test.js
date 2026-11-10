import { describe, it, expect, vi, beforeEach } from "vitest";

// Streaming path of askArena: the backend streams NDJSON (`answer` events and,
// for a failed turn, one `error` event shaped like a conversation's). Each
// event is forwarded to onStreamEvent as it arrives, whatever the chunking,
// and the trimmed answer text is returned. A non-OK response must reject
// before any body read.

const { tracedFetchMock } = vi.hoisted(() => ({ tracedFetchMock: vi.fn() }));

vi.mock("./api/client", () => ({ tracedFetch: tracedFetchMock }));

import { askArena } from "./arenaService.js";

const encoder = new TextEncoder();

/** Streaming Response stub that yields the given raw chunks then finishes. */
const streamOf = (...chunks) => {
  const queue = chunks.map((text) => ({ done: false, value: encoder.encode(text) }));
  queue.push({ done: true, value: undefined });
  return {
    ok: true,
    body: { getReader: () => ({ read: async () => queue.shift() }) },
  };
};

const line = (event) => `${JSON.stringify(event)}\n`;

beforeEach(() => {
  tracedFetchMock.mockReset();
});

describe("askArena streaming", () => {
  it("forwards each event to onStreamEvent and returns the trimmed answer text", async () => {
    tracedFetchMock.mockResolvedValue(
      streamOf(
        line({ t: "answer", text: "Hello " }),
        line({ t: "answer", text: "world" }),
        line({ t: "answer", text: "  " })
      )
    );
    const onStreamEvent = vi.fn();

    const result = await askArena({ question: "hi", llmId: 1, onStreamEvent });

    expect(onStreamEvent.mock.calls.map(([e]) => e.text)).toEqual(["Hello ", "world", "  "]);
    expect(result).toBe("Hello world");
  });

  it("reassembles an event split across chunks, and several events in one chunk", async () => {
    const error = {
      t: "error",
      text: "[ERROR_MESSAGE_SYSTEM] Too long.",
      curated: true,
      curated_error: { kind: "context_overflow_unsized", params: {} },
    };
    const whole = line({ t: "answer", text: "Part" }) + line(error);
    tracedFetchMock.mockResolvedValue(
      streamOf(whole.slice(0, 7), whole.slice(7, 40), whole.slice(40))
    );
    const onStreamEvent = vi.fn();

    await askArena({ question: "hi", llmId: 1, onStreamEvent });

    expect(onStreamEvent.mock.calls.map(([e]) => e)).toEqual([
      { t: "answer", text: "Part" },
      error,
    ]);
  });

  it("works without an onStreamEvent callback", async () => {
    tracedFetchMock.mockResolvedValue(streamOf(line({ t: "answer", text: "plain answer" })));

    await expect(askArena({ question: "hi", llmId: 1 })).resolves.toBe("plain answer");
  });

  it("rejects when the backend answers non-OK, before reading any body", async () => {
    const getReader = vi.fn();
    tracedFetchMock.mockResolvedValue({ ok: false, status: 500, body: { getReader } });

    await expect(askArena({ question: "hi", llmId: 1 })).rejects.toThrow("Arena query failed");
    expect(getReader).not.toHaveBeenCalled();
  });
});
