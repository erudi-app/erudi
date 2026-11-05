// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act, cleanup } from "@testing-library/react";

// The settings owner under races (#609 review): a save of conversation A
// never blocks or blanks B, a load older than a change is dropped, a failed
// save shows what the server holds, the draft's model switch keeps what the
// user sets meanwhile, and /config global never saves values it has not read.

const { patchMock, putMock, getMock } = vi.hoisted(() => ({
  patchMock: vi.fn(),
  putMock: vi.fn(),
  getMock: vi.fn(),
}));

vi.mock("../../services/api/client", () => ({
  default: { patch: patchMock, put: putMock, get: getMock },
}));

import useConversationSettings, { DRAFT, SAVE_FAILED_NOTE_MS } from "./useConversationSettings";

const row = (id, overrides = {}) => ({
  id,
  llm_id: 1,
  reasoning_effort: "medium",
  temperature: 0.7,
  top_p: 0.8,
  compaction_threshold: 80,
  web_search_enabled: false,
  custom_prompt: "",
  ...overrides,
});

const deferred = () => {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
};

const resolved = (overrides = {}) => ({
  reasoning_effort: { value: "medium", source: "settings" },
  temperature: { value: 0.6, source: "model" },
  top_p: { value: 0.95, source: "model" },
  compaction_threshold: { value: 80, source: "settings" },
  web_search_enabled: { value: false, source: "settings" },
  custom_prompt: { value: "", source: "fallback" },
  ...overrides,
});

beforeEach(() => {
  patchMock.mockReset();
  putMock.mockReset();
  getMock.mockReset();
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe("switching conversation while a save of the previous one is on its way", () => {
  it("adopts every value of B, even before A's save comes back", async () => {
    vi.useFakeTimers();
    const slowA = deferred();
    patchMock.mockReturnValueOnce(slowA.promise);
    const { result, rerender } = renderHook(({ id }) => useConversationSettings(id), {
      initialProps: { id: "7" },
    });
    act(() => result.current.adopt(row(7, { custom_prompt: "A's" })));
    act(() => result.current.setField("temperature", 1.4)); // slider moved in A

    rerender({ id: "8" }); // switch within the 300 ms
    await act(async () => {});
    // A's slider save goes to A, at once.
    expect(patchMock).toHaveBeenCalledWith("/conversations/7", { temperature: 1.4 });

    act(() => {
      result.current.adopt(row(8, { temperature: 0.3, top_p: 0.5, custom_prompt: "B's" }));
    });
    expect(result.current.values).toMatchObject({
      temperature: 0.3,
      top_p: 0.5,
      custom_prompt: "B's",
    });

    await act(async () => slowA.resolve(row(7)));
    expect(result.current.values.temperature).toBe(0.3);
    expect(result.current.saveFailed).toBe(false);
  });

  it("never reverts B's controls when A's save fails after the switch", async () => {
    const slowA = deferred();
    patchMock.mockReturnValueOnce(slowA.promise);
    getMock.mockResolvedValue(row(7));
    const { result, rerender } = renderHook(({ id }) => useConversationSettings(id), {
      initialProps: { id: "7" },
    });
    act(() => result.current.adopt(row(7)));
    act(() => result.current.setField("reasoning_effort", "high"));
    rerender({ id: "8" });
    act(() => result.current.adopt(row(8, { reasoning_effort: "low" })));
    await act(async () => slowA.reject(new Error("offline")));
    await act(async () => result.current.flush());
    expect(result.current.values.reasoning_effort).toBe("low");
    expect(result.current.saveFailed).toBe(false);
  });
});

describe("a load of another conversation", () => {
  it("is never adopted into the conversation the page moved to", () => {
    const { result, rerender } = renderHook(({ id }) => useConversationSettings(id), {
      initialProps: { id: "7" },
    });
    const stamp = result.current.loadStamp(); // A's load starts
    rerender({ id: "8" });
    act(() => result.current.adopt(row(8, { custom_prompt: "B's" }), result.current.loadStamp()));
    let taken;
    act(() => {
      taken = result.current.adopt(row(7, { llm_id: 3, custom_prompt: "A's" }), stamp);
    });
    expect(taken).toBe(false);
    expect(result.current.values).toMatchObject({ llm_id: 1, custom_prompt: "B's" });
  });
});

describe("one save chain per conversation", () => {
  it("a save of A still on its way never delays a send in B", async () => {
    const slowA = deferred();
    patchMock.mockReturnValueOnce(slowA.promise);
    const { result, rerender } = renderHook(({ id }) => useConversationSettings(id), {
      initialProps: { id: "7" },
    });
    act(() => result.current.adopt(row(7)));
    act(() => result.current.setField("reasoning_effort", "high"));
    rerender({ id: "8" });
    act(() => result.current.adopt(row(8)));

    let flushed = false;
    await act(async () => {
      result.current.flush().then(() => {
        flushed = true;
      });
    });
    expect(flushed).toBe(true);

    patchMock.mockResolvedValueOnce(row(8, { compaction_threshold: 60 }));
    let savedInB = false;
    await act(async () => {
      result.current.setField("compaction_threshold", 60);
      result.current.flush().then(() => {
        savedInB = true;
      });
    });
    expect(patchMock).toHaveBeenLastCalledWith("/conversations/8", { compaction_threshold: 60 });
    expect(savedInB).toBe(true);
    await act(async () => slowA.reject(new Error("offline")));
  });
});

describe("a load older than a change", () => {
  it("is dropped when a save was made after it started", async () => {
    patchMock.mockImplementation(async (_p, body) => row(7, body));
    const { result } = renderHook(() => useConversationSettings("7"));
    act(() => result.current.adopt(row(7)));
    const stamp = result.current.loadStamp();
    await act(async () => {
      result.current.setField("compaction_threshold", 60);
      await result.current.flush();
    });
    let taken;
    act(() => {
      taken = result.current.adopt(row(7, { compaction_threshold: 80 }), stamp);
    });
    expect(taken).toBe(false);
    expect(result.current.values.compaction_threshold).toBe(60);
  });

  it("is dropped when /model switched the conversation after it started", async () => {
    patchMock.mockResolvedValue(row(7, { llm_id: 2, temperature: 0.3 }));
    const { result } = renderHook(() => useConversationSettings("7"));
    act(() => result.current.adopt(row(7)));
    const stamp = result.current.loadStamp();
    await act(async () => {
      await result.current.changeModel(2);
    });
    act(() => {
      result.current.adopt(row(7, { llm_id: 1 }), stamp);
    });
    expect(result.current.values.llm_id).toBe(2);
    expect(result.current.values.temperature).toBe(0.3);
  });

  it("keeps the model while a switch is on its way", async () => {
    const slow = deferred();
    patchMock.mockReturnValueOnce(slow.promise);
    const { result } = renderHook(() => useConversationSettings("7"));
    act(() => result.current.adopt(row(7)));
    let switching;
    act(() => {
      switching = result.current.changeModel(2);
    });
    const stamp = result.current.loadStamp();
    act(() => {
      result.current.adopt(row(7, { llm_id: 1, reasoning_effort: "low" }), stamp);
    });
    expect(result.current.values.reasoning_effort).toBe("low");
    await act(async () => {
      slow.resolve(row(7, { llm_id: 2 }));
      await switching;
    });
    expect(result.current.values.llm_id).toBe(2);
  });
});

describe("a failed save or switch", () => {
  it("shows what the server holds when it applied the change all the same", async () => {
    patchMock.mockRejectedValueOnce(new Error("timeout"));
    getMock.mockResolvedValueOnce(row(7, { reasoning_effort: "xhigh" }));
    const { result } = renderHook(() => useConversationSettings("7"));
    act(() => result.current.adopt(row(7)));
    await act(async () => {
      result.current.setField("reasoning_effort", "xhigh");
      await result.current.flush();
    });
    expect(getMock).toHaveBeenCalledWith("/conversations/7");
    expect(result.current.values.reasoning_effort).toBe("xhigh");
    expect(result.current.saveFailed).toBe(true);
  });

  it("goes back to the last saved value when the server cannot be read either", async () => {
    patchMock.mockRejectedValueOnce(new Error("offline"));
    getMock.mockRejectedValueOnce(new Error("offline"));
    const { result } = renderHook(() => useConversationSettings("7"));
    act(() => result.current.adopt(row(7)));
    await act(async () => {
      result.current.setField("reasoning_effort", "xhigh");
      await result.current.flush();
    });
    expect(result.current.values.reasoning_effort).toBe("medium");
    expect(result.current.saveFailed).toBe(true);
  });

  it("adopts the server's conversation after a failed switch", async () => {
    patchMock.mockRejectedValueOnce(new Error("timeout"));
    getMock.mockResolvedValueOnce(row(7, { llm_id: 2, temperature: 0.3 }));
    const { result } = renderHook(() => useConversationSettings("7"));
    act(() => result.current.adopt(row(7)));
    await act(async () => {
      await result.current.changeModel(2);
    });
    expect(result.current.values).toMatchObject({ llm_id: 2, temperature: 0.3 });
    expect(result.current.saveFailed).toBe(true);
  });
});

describe("the could-not-be-saved note", () => {
  const fail = async (result) => {
    patchMock.mockRejectedValueOnce(new Error("offline"));
    getMock.mockRejectedValueOnce(new Error("offline"));
    await act(async () => {
      result.current.setField("reasoning_effort", "xhigh");
      await result.current.flush();
    });
  };

  it("clears after a few seconds", async () => {
    vi.useFakeTimers();
    const { result } = renderHook(() => useConversationSettings("7"));
    act(() => result.current.adopt(row(7)));
    await fail(result);
    expect(result.current.saveFailed).toBe(true);
    act(() => vi.advanceTimersByTime(SAVE_FAILED_NOTE_MS));
    expect(result.current.saveFailed).toBe(false);
  });

  it("clears on a conversation switch", async () => {
    const { result, rerender } = renderHook(({ id }) => useConversationSettings(id), {
      initialProps: { id: "7" },
    });
    act(() => result.current.adopt(row(7)));
    await fail(result);
    rerender({ id: "8" });
    expect(result.current.saveFailed).toBe(false);
  });

  it("clears on the next successful save", async () => {
    const { result } = renderHook(() => useConversationSettings("7"));
    act(() => result.current.adopt(row(7)));
    await fail(result);
    patchMock.mockResolvedValueOnce(row(7));
    await act(async () => {
      result.current.setField("compaction_threshold", 55);
      await result.current.flush();
    });
    expect(result.current.saveFailed).toBe(false);
  });
});

describe("the draft while a model switch is on its way", () => {
  const MODEL_A = { id: 1, name: "A", sampling_defaults: { temperature: 0.7, top_p: 0.8 } };
  const MODEL_B = { id: 2, name: "B", sampling_defaults: { temperature: 0.3, top_p: 0.9 } };

  const switchingToB = async (bAnswer) => {
    getMock.mockResolvedValueOnce(resolved());
    getMock.mockReturnValueOnce(bAnswer.promise);
    const hook = renderHook(() => useConversationSettings(DRAFT));
    await act(async () => hook.result.current.selectModel(MODEL_A));
    act(() => {
      hook.result.current.setField("temperature", 1.5);
      hook.result.current.setField("reasoning_effort", "xhigh");
    });
    act(() => {
      hook.result.current.selectModel(MODEL_B);
    });
    return hook;
  };

  it("never sends the previous model's creativity to the new one, even before its defaults arrive", async () => {
    const bAnswer = deferred();
    const { result } = await switchingToB(bAnswer);
    expect(result.current.createPayload()).toEqual({ llm_id: 2, reasoning_effort: "xhigh" });
  });

  it("keeps a value set after the switch started, whatever the new model saved", async () => {
    const bAnswer = deferred();
    const { result } = await switchingToB(bAnswer);
    act(() => result.current.setField("compaction_threshold", 55));
    await act(async () =>
      bAnswer.resolve(
        resolved({
          compaction_threshold: { value: 70, source: "saved" },
          reasoning_effort: { value: "low", source: "saved" },
        })
      )
    );
    expect(result.current.values.compaction_threshold).toBe(55);
    // Set before the switch, saved for B: replaced (the /model rule).
    expect(result.current.values.reasoning_effort).toBe("low");
    expect(result.current.createPayload()).toEqual({ llm_id: 2, compaction_threshold: 55 });
  });

  it("ready() waits for the new model's defaults", async () => {
    const bAnswer = deferred();
    const { result } = await switchingToB(bAnswer);
    let isReady = false;
    act(() => {
      result.current.ready().then(() => {
        isReady = true;
      });
    });
    await act(async () => {});
    expect(isReady).toBe(false);
    await act(async () =>
      bAnswer.resolve(resolved({ temperature: { value: 0.3, source: "model" } }))
    );
    expect(isReady).toBe(true);
    expect(result.current.values.temperature).toBe(0.3);
  });

  it("/config global refuses while the new model's defaults are on their way", async () => {
    const bAnswer = deferred();
    const { result } = await switchingToB(bAnswer);
    await expect(result.current.saveAsDefaults()).rejects.toThrow(/not loaded yet/);
    expect(putMock).not.toHaveBeenCalled();
  });
});

describe("the draft after a failed defaults read", () => {
  const MODEL = { id: 1, name: "A", sampling_defaults: { temperature: 0.7, top_p: 0.8 } };

  it("/config global reads them again, and refuses when that fails too", async () => {
    getMock.mockRejectedValue(new Error("offline"));
    const { result } = renderHook(() => useConversationSettings(DRAFT));
    await act(async () => result.current.selectModel(MODEL));
    await act(async () => {
      await expect(result.current.saveAsDefaults()).rejects.toThrow(/not loaded yet/);
    });
    expect(getMock).toHaveBeenCalledTimes(2);
    expect(putMock).not.toHaveBeenCalled();
  });

  it("/config global saves the values once a new read succeeds", async () => {
    getMock.mockRejectedValueOnce(new Error("offline"));
    getMock.mockResolvedValueOnce(
      resolved({ custom_prompt: { value: "Saved before", source: "saved" } })
    );
    putMock.mockResolvedValue(resolved());
    const { result } = renderHook(() => useConversationSettings(DRAFT));
    await act(async () => result.current.selectModel(MODEL));
    await act(async () => {
      await result.current.saveAsDefaults();
    });
    expect(putMock).toHaveBeenCalledWith(
      "/llms/1/conversation_defaults",
      expect.objectContaining({ custom_prompt: "Saved before", temperature: 0.6 })
    );
  });

  it("a value the user set is real, not unread", async () => {
    getMock.mockRejectedValueOnce(new Error("offline"));
    const { result } = renderHook(() => useConversationSettings(DRAFT));
    await act(async () => result.current.selectModel(MODEL));
    act(() => {
      for (const [field, value] of [
        ["reasoning_effort", "low"],
        ["temperature", 0.5],
        ["top_p", 0.5],
        ["compaction_threshold", 60],
        ["web_search_enabled", false],
        ["custom_prompt", ""],
      ]) {
        result.current.setField(field, value);
      }
    });
    putMock.mockResolvedValue(resolved());
    await act(async () => {
      await result.current.saveAsDefaults();
    });
    expect(getMock).toHaveBeenCalledTimes(1);
    expect(putMock).toHaveBeenCalledTimes(1);
  });
});
