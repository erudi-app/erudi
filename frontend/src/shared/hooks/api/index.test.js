// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor, act } from "@testing-library/react";

vi.mock("../../../services/api/client", () => ({
  apiClient: { get: vi.fn() },
}));
vi.mock("../../../utils/logger", () => ({
  createLogger: () => ({ error: vi.fn(), warn: vi.fn(), info: vi.fn(), debug: vi.fn() }),
}));

import { apiClient } from "../../../services/api/client";
import {
  useLLMs,
  useConversation,
  useConversationMessages,
  useConversations,
  useAppStartupInfo,
  useBackendHealth,
} from "./index";

beforeEach(() => {
  vi.clearAllMocks();
});

describe("useLLMs", () => {
  it("starts loading, then exposes the fetched list from /llms/local", async () => {
    const llms = [{ id: 1, name: "qwen" }];
    apiClient.get.mockResolvedValueOnce(llms);

    const { result } = renderHook(() => useLLMs());
    expect(result.current).toEqual({ llms: [], loading: true, error: null });

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(apiClient.get).toHaveBeenCalledWith("/llms/local");
    expect(result.current.llms).toEqual(llms);
    expect(result.current.error).toBeNull();
  });

  it("normalizes a non-array payload to an empty list", async () => {
    apiClient.get.mockResolvedValueOnce({ unexpected: true });

    const { result } = renderHook(() => useLLMs());
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.llms).toEqual([]);
    expect(result.current.error).toBeNull();
  });

  it("exposes the error and stops loading when the request fails", async () => {
    const failure = new Error("boom");
    apiClient.get.mockRejectedValueOnce(failure);

    const { result } = renderHook(() => useLLMs());
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.error).toBe(failure);
    expect(result.current.llms).toEqual([]);
  });
});

describe("useConversation", () => {
  it("fetches the conversation by id", async () => {
    const conversation = { id: "c1", title: "hello" };
    apiClient.get.mockResolvedValueOnce(conversation);

    const { result } = renderHook(() => useConversation("c1"));
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(apiClient.get).toHaveBeenCalledWith("/conversations/c1");
    expect(result.current.conversation).toEqual(conversation);
    expect(result.current.error).toBeNull();
  });

  it("does not fetch without an id and stays in the initial loading state", () => {
    const { result } = renderHook(() => useConversation(undefined));
    expect(apiClient.get).not.toHaveBeenCalled();
    expect(result.current).toEqual({ conversation: null, loading: true, error: null });
  });

  it("refetches when the id changes", async () => {
    apiClient.get.mockResolvedValue({ id: "any" });

    const { result, rerender } = renderHook(({ id }) => useConversation(id), {
      initialProps: { id: "c1" },
    });
    await waitFor(() => expect(result.current.loading).toBe(false));

    rerender({ id: "c2" });
    await waitFor(() => expect(apiClient.get).toHaveBeenCalledTimes(2));
    expect(apiClient.get).toHaveBeenLastCalledWith("/conversations/c2");
  });

  it("exposes the error when the request fails", async () => {
    const failure = new Error("404");
    apiClient.get.mockRejectedValueOnce(failure);

    const { result } = renderHook(() => useConversation("missing"));
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.error).toBe(failure);
    expect(result.current.conversation).toBeNull();
  });
});

describe("useConversationMessages", () => {
  it("fetches messages from the fetch_messages endpoint", async () => {
    const messages = [{ id: 1, content: "hi" }];
    apiClient.get.mockResolvedValueOnce(messages);

    const { result } = renderHook(() => useConversationMessages("c9"));
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(apiClient.get).toHaveBeenCalledWith("/conversations/c9/fetch_messages");
    expect(result.current.messages).toEqual(messages);
    expect(result.current.error).toBeNull();
  });

  it("normalizes a non-array payload to an empty list", async () => {
    apiClient.get.mockResolvedValueOnce("nope");

    const { result } = renderHook(() => useConversationMessages("c9"));
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.messages).toEqual([]);
  });

  it("does not fetch without an id", () => {
    const { result } = renderHook(() => useConversationMessages(null));
    expect(apiClient.get).not.toHaveBeenCalled();
    expect(result.current).toEqual({ messages: [], loading: true, error: null });
  });

  it("exposes the error when the request fails", async () => {
    const failure = new Error("network");
    apiClient.get.mockRejectedValueOnce(failure);

    const { result } = renderHook(() => useConversationMessages("c9"));
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.error).toBe(failure);
    expect(result.current.messages).toEqual([]);
  });
});

describe("useConversations", () => {
  it("fetches the conversation list on mount", async () => {
    const conversations = [{ id: "c1" }, { id: "c2" }];
    apiClient.get.mockResolvedValueOnce(conversations);

    const { result } = renderHook(() => useConversations());
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(apiClient.get).toHaveBeenCalledWith("/conversations/");
    expect(result.current.conversations).toEqual(conversations);
    expect(result.current.error).toBeNull();
  });

  it("refetch() reloads the list and clears a previous error", async () => {
    const failure = new Error("first load down");
    apiClient.get.mockRejectedValueOnce(failure);

    const { result } = renderHook(() => useConversations());
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.error).toBe(failure);

    const conversations = [{ id: "c3" }];
    apiClient.get.mockResolvedValueOnce(conversations);
    await result.current.refetch();

    await waitFor(() => expect(result.current.conversations).toEqual(conversations));
    expect(result.current.error).toBeNull();
    expect(apiClient.get).toHaveBeenCalledTimes(2);
  });

  it("normalizes a non-array payload to an empty list", async () => {
    apiClient.get.mockResolvedValueOnce(null);

    const { result } = renderHook(() => useConversations());
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.conversations).toEqual([]);
  });
});

describe("useAppStartupInfo", () => {
  it("fetches hardware startup info", async () => {
    const info = { recommended_param_min: 1, recommended_param_max: 8 };
    apiClient.get.mockResolvedValueOnce(info);

    const { result } = renderHook(() => useAppStartupInfo());
    expect(result.current).toMatchObject({
      startupInfo: null,
      loading: true,
      error: null,
      refetchesAfterReady: 0,
    });

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(apiClient.get).toHaveBeenCalledWith("/hardware/app_startup");
    expect(result.current.startupInfo).toEqual(info);
    expect(result.current.error).toBeNull();
  });

  it("exposes the error when the request fails", async () => {
    const failure = new Error("startup down");
    apiClient.get.mockRejectedValueOnce(failure);

    const { result } = renderHook(() => useAppStartupInfo());
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.error).toBe(failure);
    expect(result.current.startupInfo).toBeNull();
  });
});

/**
 * The startup info names the engine that RUNS, so it goes stale the moment the
 * backend restarts on another engine (the Inference engine setting). The hook
 * re-reads it when the main process says the restarted backend is ready.
 */
describe("useAppStartupInfo — after a backend restart", () => {
  let listeners;
  let unsubscribe;

  beforeEach(() => {
    listeners = [];
    unsubscribe = vi.fn();
    window.backendAPI = {
      onBackendEvent: vi.fn((cb) => {
        listeners.push(cb);
        return unsubscribe;
      }),
    };
  });

  afterEach(() => {
    delete window.backendAPI;
  });

  const emit = (evt) => act(() => listeners.forEach((cb) => cb(evt)));

  it("fetches again on backend_ready and exposes the fresh engine", async () => {
    apiClient.get
      .mockResolvedValueOnce({ backend_type: "cpu" })
      .mockResolvedValueOnce({ backend_type: "cuda" });

    const { result } = renderHook(() => useAppStartupInfo());
    await waitFor(() => expect(result.current.startupInfo).toEqual({ backend_type: "cpu" }));

    emit({ event: "backend_ready", port: 27183 });

    await waitFor(() => expect(result.current.startupInfo).toEqual({ backend_type: "cuda" }));
    expect(apiClient.get).toHaveBeenCalledTimes(2);
    expect(apiClient.get).toHaveBeenNthCalledWith(2, "/hardware/app_startup");
    expect(result.current.refetchesAfterReady).toBe(1);
  });

  it("ignores the other lifecycle events", async () => {
    apiClient.get.mockResolvedValue({ backend_type: "cuda" });

    const { result } = renderHook(() => useAppStartupInfo());
    await waitFor(() => expect(result.current.loading).toBe(false));

    emit({ event: "starting", first_run: false });
    emit({ event: "phase", phase: "database" });
    // run.py's own `ready`, relayed raw by main.js before its health check:
    // main.js then sends `backend_ready`, the one this hook acts on, so
    // reacting to both would read the engine twice per restart.
    emit({ event: "ready", port: 27183 });

    expect(apiClient.get).toHaveBeenCalledTimes(1);
    expect(result.current.refetchesAfterReady).toBe(0);
  });

  it("counts a failed re-read as settled, with the error exposed", async () => {
    const failure = new Error("still booting");
    apiClient.get.mockResolvedValueOnce({ backend_type: "cuda" }).mockRejectedValueOnce(failure);

    const { result } = renderHook(() => useAppStartupInfo());
    await waitFor(() => expect(result.current.loading).toBe(false));

    emit({ event: "backend_ready", port: 27183 });

    await waitFor(() => expect(result.current.refetchesAfterReady).toBe(1));
    expect(result.current.error).toBe(failure);
  });

  it("unsubscribes from the backend events on unmount", async () => {
    apiClient.get.mockResolvedValue({ backend_type: "cuda" });

    const { unmount } = renderHook(() => useAppStartupInfo());
    await waitFor(() => expect(window.backendAPI.onBackendEvent).toHaveBeenCalledTimes(1));

    unmount();

    expect(unsubscribe).toHaveBeenCalledTimes(1);
  });
});

describe("useBackendHealth", () => {
  it("reports healthy when /health/ responds", async () => {
    apiClient.get.mockResolvedValueOnce({ status: "ok" });

    const { result } = renderHook(() => useBackendHealth());
    expect(result.current).toEqual({ isHealthy: false, loading: true, error: null });

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(apiClient.get).toHaveBeenCalledWith("/health/");
    expect(result.current.isHealthy).toBe(true);
    expect(result.current.error).toBeNull();
  });

  it("reports unhealthy with the error when the check fails", async () => {
    const failure = new Error("down");
    apiClient.get.mockRejectedValueOnce(failure);

    const { result } = renderHook(() => useBackendHealth());
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.isHealthy).toBe(false);
    expect(result.current.error).toBe(failure);
  });
});
