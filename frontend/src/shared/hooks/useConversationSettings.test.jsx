// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act, cleanup } from "@testing-library/react";

// One owner for a conversation's settings (#609): the header panel, /config
// and /effort all read and write through this hook. In a conversation it
// saves every change (discrete ones at once, the sliders 300 ms after they
// stop), keeps the saves in order, puts a control back when its save fails,
// and lets the send path and /model wait for what is pending. On the
// new-conversation page (the draft) it resolves the picked model's defaults
// and remembers which values the user set.

const { patchMock, putMock, getMock } = vi.hoisted(() => ({
  patchMock: vi.fn(),
  putMock: vi.fn(),
  getMock: vi.fn(),
}));

vi.mock("../../services/api/client", () => ({
  default: { patch: patchMock, put: putMock, get: getMock },
}));

import useConversationSettings, { DRAFT } from "./useConversationSettings";

const CONVERSATION = {
  id: 7,
  llm_id: 1,
  reasoning_effort: "medium",
  temperature: 0.7,
  top_p: 0.8,
  compaction_threshold: 80,
  web_search_enabled: false,
  custom_prompt: "Be brief",
};

const deferred = () => {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
};

const resolved = (overrides = {}) => {
  const base = {
    reasoning_effort: { value: "medium", source: "settings" },
    temperature: { value: 0.6, source: "model" },
    top_p: { value: 0.95, source: "model" },
    compaction_threshold: { value: 80, source: "settings" },
    web_search_enabled: { value: false, source: "settings" },
    custom_prompt: { value: "", source: "fallback" },
  };
  return { ...base, ...overrides };
};

beforeEach(() => {
  patchMock.mockReset();
  putMock.mockReset();
  getMock.mockReset();
  patchMock.mockImplementation(async (_path, body) => ({ ...CONVERSATION, ...body }));
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

const renderConversation = () => {
  const hook = renderHook(() => useConversationSettings(7));
  act(() => hook.result.current.adopt(CONVERSATION));
  return hook;
};

describe("in a conversation", () => {
  it("adopts the conversation row", () => {
    const { result } = renderConversation();
    expect(result.current.loaded).toBe(true);
    expect(result.current.values).toMatchObject({
      llm_id: 1,
      reasoning_effort: "medium",
      temperature: 0.7,
      top_p: 0.8,
      compaction_threshold: 80,
      web_search_enabled: false,
      custom_prompt: "Be brief",
    });
  });

  it.each([
    ["reasoning_effort", "high"],
    ["web_search_enabled", true],
    ["compaction_threshold", 65],
    ["custom_prompt", ""],
  ])("saves %s at once, in a one-field PATCH", async (field, value) => {
    const { result } = renderConversation();
    await act(async () => {
      result.current.setField(field, value);
      await result.current.flush();
    });
    expect(patchMock).toHaveBeenCalledTimes(1);
    expect(patchMock).toHaveBeenCalledWith("/conversations/7", { [field]: value });
    expect(result.current.values[field]).toBe(value);
  });

  it("saves the sliders 300 ms after they stop, in one PATCH", async () => {
    vi.useFakeTimers();
    const { result } = renderConversation();
    act(() => {
      result.current.setField("temperature", 0.9);
      result.current.setField("temperature", 1.1);
      result.current.setField("top_p", 0.5);
    });
    expect(result.current.values.temperature).toBe(1.1);
    act(() => vi.advanceTimersByTime(299));
    expect(patchMock).not.toHaveBeenCalled();
    await act(async () => {
      vi.advanceTimersByTime(1);
    });
    expect(patchMock).toHaveBeenCalledTimes(1);
    expect(patchMock).toHaveBeenCalledWith("/conversations/7", { temperature: 1.1, top_p: 0.5 });
  });

  it("flush() sends a pending slider save at once and waits for it", async () => {
    vi.useFakeTimers();
    const { result } = renderConversation();
    const pending = deferred();
    patchMock.mockReturnValueOnce(pending.promise);
    act(() => result.current.setField("temperature", 1.4));

    let flushed = false;
    act(() => {
      result.current.flush().then(() => {
        flushed = true;
      });
    });
    await act(async () => {});
    // Sent without waiting for the 300 ms delay, and still awaited.
    expect(patchMock).toHaveBeenCalledWith("/conversations/7", { temperature: 1.4 });
    expect(flushed).toBe(false);
    await act(async () => pending.resolve({ ...CONVERSATION, temperature: 1.4 }));
    expect(flushed).toBe(true);
  });

  it("sends the saves one after another, in the order they were made", async () => {
    const { result } = renderConversation();
    const first = deferred();
    patchMock.mockReturnValueOnce(first.promise);
    act(() => {
      result.current.setField("reasoning_effort", "low");
      result.current.setField("compaction_threshold", 55);
    });
    await act(async () => {});
    expect(patchMock).toHaveBeenCalledTimes(1);
    await act(async () => first.resolve(CONVERSATION));
    await act(async () => result.current.flush());
    expect(patchMock).toHaveBeenCalledTimes(2);
    expect(patchMock.mock.calls[1][1]).toEqual({ compaction_threshold: 55 });
  });

  it("puts the control back and reports the failure when a save fails", async () => {
    const { result } = renderConversation();
    patchMock.mockRejectedValueOnce(new Error("offline"));
    await act(async () => {
      result.current.setField("reasoning_effort", "xhigh");
      await result.current.flush();
    });
    expect(result.current.values.reasoning_effort).toBe("medium");
    expect(result.current.saveFailed).toBe(true);

    await act(async () => {
      result.current.setField("reasoning_effort", "low");
      await result.current.flush();
    });
    expect(result.current.saveFailed).toBe(false);
    expect(result.current.values.reasoning_effort).toBe("low");
  });

  it("does not undo a later change when an earlier save of the same field fails", async () => {
    const { result } = renderConversation();
    const first = deferred();
    patchMock.mockReturnValueOnce(first.promise);
    act(() => {
      result.current.setField("reasoning_effort", "high");
      result.current.setField("reasoning_effort", "low");
    });
    await act(async () => first.reject(new Error("offline")));
    await act(async () => result.current.flush());
    expect(result.current.values.reasoning_effort).toBe("low");
  });

  it("changeModel waits for pending saves, sends only llm_id, and adopts the answer", async () => {
    vi.useFakeTimers();
    const { result } = renderConversation();
    act(() => result.current.setField("temperature", 1.2));
    const switched = {
      ...CONVERSATION,
      llm_id: 2,
      temperature: 0.6,
      top_p: 0.95,
      custom_prompt: "Saved",
    };
    patchMock.mockImplementation(async (_p, body) => (body.llm_id ? switched : CONVERSATION));

    await act(async () => {
      await result.current.changeModel(2);
    });
    expect(patchMock.mock.calls.map((c) => c[1])).toEqual([{ temperature: 1.2 }, { llm_id: 2 }]);
    expect(result.current.values).toMatchObject({
      llm_id: 2,
      temperature: 0.6,
      top_p: 0.95,
      custom_prompt: "Saved",
    });
  });

  it("keeps the current model when the switch fails", async () => {
    const { result } = renderConversation();
    patchMock.mockRejectedValueOnce(new Error("offline"));
    await act(async () => {
      await result.current.changeModel(2);
    });
    expect(result.current.values.llm_id).toBe(1);
    expect(result.current.saveFailed).toBe(true);
  });

  it("saveAsDefaults PUTs the six current values for the current model", async () => {
    const { result } = renderConversation();
    putMock.mockResolvedValue(resolved());
    act(() => result.current.setField("custom_prompt", ""));
    await act(async () => {
      await result.current.saveAsDefaults();
    });
    expect(putMock).toHaveBeenCalledWith("/llms/1/conversation_defaults", {
      reasoning_effort: "medium",
      temperature: 0.7,
      top_p: 0.8,
      compaction_threshold: 80,
      web_search_enabled: false,
      custom_prompt: "",
    });
  });

  it("getValues() reads the latest values synchronously, after adopt", () => {
    const { result } = renderHook(() => useConversationSettings(7));
    let read;
    act(() => {
      result.current.adopt({ ...CONVERSATION, custom_prompt: "Fresh" });
      read = result.current.getValues();
    });
    expect(read.custom_prompt).toBe("Fresh");
  });
});

describe("on the new-conversation page (draft)", () => {
  const MODEL_A = { id: 1, name: "A", sampling_defaults: { temperature: 0.7, top_p: 0.8 } };
  const MODEL_B = { id: 2, name: "B", sampling_defaults: { temperature: 0.3, top_p: 0.9 } };

  const renderDraft = () => renderHook(() => useConversationSettings(DRAFT));

  it("takes the picked model's resolved values and sends none it was not asked to", async () => {
    getMock.mockResolvedValue(resolved());
    const { result } = renderDraft();
    await act(async () => result.current.selectModel(MODEL_A));
    expect(getMock).toHaveBeenCalledWith("/llms/1/conversation_defaults");
    expect(result.current.loaded).toBe(true);
    expect(result.current.values).toMatchObject({ temperature: 0.6, reasoning_effort: "medium" });
    expect(result.current.createPayload()).toEqual({ llm_id: 1 });
    expect(patchMock).not.toHaveBeenCalled();
  });

  it("sends exactly the values the user set", async () => {
    getMock.mockResolvedValue(resolved());
    const { result } = renderDraft();
    await act(async () => result.current.selectModel(MODEL_A));
    act(() => {
      result.current.setField("reasoning_effort", "high");
      result.current.setField("custom_prompt", "Pirate");
    });
    expect(result.current.createPayload()).toEqual({
      llm_id: 1,
      reasoning_effort: "high",
      custom_prompt: "Pirate",
    });
  });

  it("keeps a value set before the defaults arrive", async () => {
    const pending = deferred();
    getMock.mockReturnValueOnce(pending.promise);
    const { result } = renderDraft();
    act(() => {
      result.current.selectModel(MODEL_A);
      result.current.setField("compaction_threshold", 60);
    });
    await act(async () => pending.resolve(resolved()));
    expect(result.current.values.compaction_threshold).toBe(60);
    expect(result.current.createPayload()).toEqual({ llm_id: 1, compaction_threshold: 60 });
  });

  it("on a model switch, untouched values follow the new model", async () => {
    getMock.mockResolvedValueOnce(resolved());
    getMock.mockResolvedValueOnce(
      resolved({
        reasoning_effort: { value: "low", source: "saved" },
        temperature: { value: 0.3, source: "model" },
      })
    );
    const { result } = renderDraft();
    await act(async () => result.current.selectModel(MODEL_A));
    await act(async () => result.current.selectModel(MODEL_B));
    expect(result.current.values).toMatchObject({
      llm_id: 2,
      reasoning_effort: "low",
      temperature: 0.3,
    });
    expect(result.current.createPayload()).toEqual({ llm_id: 2 });
  });

  it("on a model switch, touched values follow the /model rule", async () => {
    getMock.mockResolvedValueOnce(resolved());
    getMock.mockResolvedValueOnce(
      resolved({
        reasoning_effort: { value: "low", source: "saved" },
        temperature: { value: 0.3, source: "model" },
        web_search_enabled: { value: false, source: "settings" },
      })
    );
    const { result } = renderDraft();
    await act(async () => result.current.selectModel(MODEL_A));
    act(() => {
      result.current.setField("reasoning_effort", "xhigh"); // B saved one: replaced
      result.current.setField("temperature", 1.5); // sampling: B's recommendation replaces it
      result.current.setField("web_search_enabled", true); // B saved nothing: stays
    });
    await act(async () => result.current.selectModel(MODEL_B));
    expect(result.current.values).toMatchObject({
      reasoning_effort: "low",
      temperature: 0.3,
      web_search_enabled: true,
    });
    expect(result.current.createPayload()).toEqual({ llm_id: 2, web_search_enabled: true });
  });

  it("ignores a late answer for the previous model", async () => {
    const slowA = deferred();
    getMock.mockReturnValueOnce(slowA.promise);
    getMock.mockResolvedValueOnce(resolved({ temperature: { value: 0.3, source: "model" } }));
    const { result } = renderDraft();
    act(() => {
      result.current.selectModel(MODEL_A);
    });
    await act(async () => result.current.selectModel(MODEL_B));
    await act(async () =>
      slowA.resolve(resolved({ temperature: { value: 1.9, source: "saved" } }))
    );
    expect(result.current.values.llm_id).toBe(2);
    expect(result.current.values.temperature).toBe(0.3);
  });

  it("shows the model's recommended sampling when its defaults cannot be read", async () => {
    getMock.mockRejectedValue(new Error("offline"));
    const { result } = renderDraft();
    await act(async () => result.current.selectModel(MODEL_A));
    expect(result.current.loaded).toBe(true);
    expect(result.current.values).toMatchObject({
      temperature: 0.7,
      top_p: 0.8,
      reasoning_effort: "medium",
      compaction_threshold: 80,
      web_search_enabled: false,
      custom_prompt: "",
    });
    expect(result.current.createPayload()).toEqual({ llm_id: 1 });
  });

  it("refuses to save defaults while the values are not known yet (null would clear them)", async () => {
    getMock.mockReturnValueOnce(new Promise(() => {}));
    const { result } = renderDraft();
    act(() => {
      result.current.selectModel(MODEL_A);
    });
    await expect(result.current.saveAsDefaults()).rejects.toThrow(/not loaded yet/);
    expect(putMock).not.toHaveBeenCalled();
  });

  it("saveAsDefaults PUTs the draft values for the picked model", async () => {
    getMock.mockResolvedValue(resolved());
    putMock.mockResolvedValue(resolved());
    const { result } = renderDraft();
    await act(async () => result.current.selectModel(MODEL_A));
    act(() => result.current.setField("reasoning_effort", "none"));
    await act(async () => {
      await result.current.saveAsDefaults();
    });
    expect(putMock).toHaveBeenCalledWith("/llms/1/conversation_defaults", {
      reasoning_effort: "none",
      temperature: 0.6,
      top_p: 0.95,
      compaction_threshold: 80,
      web_search_enabled: false,
      custom_prompt: "",
    });
  });
});
