import { describe, it, expect, vi, beforeEach } from "vitest";

// Arena requests must carry attached images (#136 C) and honor an
// AbortController signal so the user can stop a running comparison (#136 H).
// tracedFetch is a drop-in fetch wrapper: options are forwarded untouched, so
// asserting on the options object is equivalent to asserting on fetch itself.

const { tracedFetchMock } = vi.hoisted(() => ({ tracedFetchMock: vi.fn() }));

vi.mock("./api/client", () => ({ tracedFetch: tracedFetchMock }));

import { askArena } from "./arenaService.js";

const DATA_URL = "data:image/png;base64,AAA";

/** Minimal streaming Response stub: an already-finished, empty body. */
const doneStream = () => ({
  ok: true,
  body: { getReader: () => ({ read: async () => ({ done: true, value: undefined }) }) },
});

beforeEach(() => {
  tracedFetchMock.mockReset();
  tracedFetchMock.mockResolvedValue(doneStream());
});

describe("askArena images (#136 C)", () => {
  it("includes attached images in the request payload", async () => {
    await askArena({ question: "what is this?", images: [DATA_URL], llmId: 7 });

    expect(tracedFetchMock).toHaveBeenCalledTimes(1);
    const [url, options] = tracedFetchMock.mock.calls[0];
    expect(String(url)).toContain("/arena/7/query");
    expect(JSON.parse(options.body).images).toEqual([DATA_URL]);
  });

  it("allows image-only asks (empty question with an attachment)", async () => {
    await expect(askArena({ question: "", images: [DATA_URL], llmId: 7 })).resolves.toBe("");
    expect(JSON.parse(tracedFetchMock.mock.calls[0][1].body).images).toEqual([DATA_URL]);
  });

  it("still rejects an empty ask with no images", async () => {
    await expect(askArena({ question: "  ", images: [], llmId: 7 })).rejects.toThrow(
      "Question is empty"
    );
    expect(tracedFetchMock).not.toHaveBeenCalled();
  });
});

describe("askArena abort (#136 H)", () => {
  it("forwards the caller's signal to tracedFetch", async () => {
    const controller = new AbortController();
    await askArena({ question: "hi", llmId: 3, signal: controller.signal });

    const [, options] = tracedFetchMock.mock.calls[0];
    expect(options.signal).toBe(controller.signal);
  });

  it("rejects with AbortError when the signal aborts mid-flight", async () => {
    tracedFetchMock.mockImplementation(
      (url, options) =>
        new Promise((resolve, reject) => {
          options.signal?.addEventListener("abort", () =>
            reject(Object.assign(new Error("aborted"), { name: "AbortError" }))
          );
        })
    );

    const controller = new AbortController();
    const pending = askArena({ question: "hi", llmId: 3, signal: controller.signal });
    controller.abort();

    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
  });
});
