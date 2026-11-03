// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useEngineSwitch } from "./useEngineSwitch";

vi.mock("../../services/api/client", () => ({
  apiClient: { get: vi.fn(), put: vi.fn() },
  tracedFetch: vi.fn(),
}));

let order;

beforeEach(() => {
  order = [];
  window.backendAPI = {
    restartBackend: vi.fn(async () => {
      order.push("restart");
    }),
  };
});

afterEach(() => {
  delete window.backendAPI;
});

describe("useEngineSwitch — one switch at a time", () => {
  it("joins a second request made while a switch is running instead of starting another", async () => {
    let release;
    const persist = vi.fn(
      () =>
        new Promise((resolve) => {
          order.push("put");
          release = resolve;
        })
    );
    const { result } = renderHook(() => useEngineSwitch({ persist }));

    let first;
    let second;
    act(() => {
      first = result.current.requestSwitch("cpu");
      second = result.current.requestSwitch("cpu");
    });
    await act(async () => {
      release();
      await first;
    });

    expect(persist).toHaveBeenCalledTimes(1);
    expect(window.backendAPI.restartBackend).toHaveBeenCalledTimes(1);
    expect(order).toEqual(["put", "restart"]);
    await expect(second).resolves.toBe("restarted");
  });

  it("ignores a confirmation that no question is waiting for", async () => {
    const persist = vi.fn(async () => {});
    const { result } = renderHook(() => useEngineSwitch({ persist }));

    await act(async () => {
      await result.current.confirm();
    });

    expect(persist).not.toHaveBeenCalled();
    expect(window.backendAPI.restartBackend).not.toHaveBeenCalled();
  });
});
