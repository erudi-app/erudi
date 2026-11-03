// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, act, cleanup, waitFor } from "@testing-library/react";

// App readiness paths not exercised by App.test.jsx: the no-preload fallback
// (poll /health directly, give up after the timeout), the getInfo() race
// (readiness resolved before the event listener attached), the startup phase
// label, and the Quit action on the error screen.

vi.mock("./pages/LandingPage", () => ({ default: () => <div>MODELS_PAGE</div> }));
vi.mock("./pages/ChatPage", () => ({ default: () => <div>CHAT</div> }));
vi.mock("./pages/ConversationPage", () => ({ default: () => <div>CONV</div> }));
vi.mock("./pages/ArenaPage", () => ({ default: () => <div>ARENA</div> }));
vi.mock("./pages/KnowledgeBasePage", () => ({ default: () => <div>KB</div> }));
vi.mock("./components/UpdateBanner", () => ({ default: () => null }));
vi.mock("./components/InteractionLogger", () => ({ default: () => null }));
vi.mock("./contexts/DownloadModalContext", () => ({
  DownloadModalProvider: ({ children }) => <>{children}</>,
  // Read by the engine dialog's switch (no download in these tests).
  useDownloadModal: () => ({ isDownloading: false, cancelDownload: async () => {} }),
}));
vi.mock("./contexts/KnowledgeBaseContext", () => ({
  KnowledgeBaseProvider: ({ children }) => <>{children}</>,
}));
// The language sync (#385) and the automatic-update sync also call the client
// once ready; keep the health poll counts below about the health poll only.
vi.mock("./i18n/sync", () => ({ syncLanguageWithBackend: vi.fn() }));
vi.mock("./utils/autoUpdate", () => ({ syncAutoUpdateWithMain: vi.fn() }));
vi.mock("./services/api/client", () => ({
  apiClient: { get: vi.fn() },
  default: { get: vi.fn() },
}));

import App from "./App.jsx";
import { apiClient } from "./services/api/client";

afterEach(() => {
  cleanup();
  delete window.backendAPI;
  delete window.__ERUDI_BACKEND_TIMEOUT_MS__;
  vi.useRealTimers();
});
beforeEach(() => {
  apiClient.get.mockReset();
});

describe("App fallback health polling (no preload bridge)", () => {
  it("becomes ready when /health/ answers", async () => {
    apiClient.get.mockResolvedValue({ status: "ok" });

    render(<App />);

    await waitFor(() => expect(screen.getByText("MODELS_PAGE")).toBeTruthy());
    expect(apiClient.get).toHaveBeenCalledWith("/health/");
  });

  it("retries every 2s until the backend answers", async () => {
    vi.useFakeTimers();
    window.__ERUDI_BACKEND_TIMEOUT_MS__ = 60000;
    apiClient.get.mockRejectedValueOnce(new Error("down")).mockResolvedValue({ status: "ok" });

    render(<App />);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(0);
    });
    expect(screen.queryByText("MODELS_PAGE")).toBeNull();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    expect(screen.getByText("MODELS_PAGE")).toBeTruthy();
    expect(apiClient.get).toHaveBeenCalledTimes(2);
  });

  // The three tests below wait for the BackendErrorScreen, and all three used to
  // flake (#323). App.jsx gives up only when `Date.now() - start >= timeoutMs`,
  // evaluated in the catch of the very first probe. With the timeout at 1ms that
  // subtraction is usually 0, because the mocked rejection settles inside the
  // same millisecond, so the give-up is deferred by one 2s retry -- past the 1s
  // default of findByText/waitFor. Whether the assertion won the race came down
  // to whether a millisecond happened to tick over, i.e. machine speed.
  //
  // 0 is not usable as the timeout (`Number(0) || FALLBACK_HEALTH_TIMEOUT_MS`
  // falls back to 90s), so instead of racing the clock these drive it, the same
  // way "retries every 2s" above already does. Deterministic, and instant in
  // real time.

  it("gives up with the unreachable error screen after the timeout", async () => {
    vi.useFakeTimers();
    window.__ERUDI_BACKEND_TIMEOUT_MS__ = 1;
    apiClient.get.mockRejectedValue(new Error("down"));

    render(<App />);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });

    expect(screen.getByText("Retry")).toBeTruthy();
  });

  it("recovers through Retry once the backend answers again", async () => {
    vi.useFakeTimers();
    window.__ERUDI_BACKEND_TIMEOUT_MS__ = 1;
    apiClient.get.mockRejectedValue(new Error("down"));

    render(<App />);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    const retry = screen.getByText("Retry");

    apiClient.get.mockResolvedValue({ status: "ok" });
    await act(async () => {
      retry.click();
      await vi.advanceTimersByTimeAsync(0);
    });

    expect(screen.getByText("MODELS_PAGE")).toBeTruthy();
  });

  it("Quit closes the window", async () => {
    vi.useFakeTimers();
    window.__ERUDI_BACKEND_TIMEOUT_MS__ = 1;
    apiClient.get.mockRejectedValue(new Error("down"));
    const closeSpy = vi.spyOn(window, "close").mockImplementation(() => {});

    render(<App />);
    await act(async () => {
      await vi.advanceTimersByTimeAsync(2000);
    });
    const quit = screen.getByText("Quit");
    await act(async () => {
      quit.click();
    });

    expect(closeSpy).toHaveBeenCalledTimes(1);
    closeSpy.mockRestore();
  });
});

describe("App bridge extras", () => {
  it("adopts readiness from getInfo() when the ready event fired before mount", async () => {
    window.backendAPI = {
      onBackendEvent: () => () => {},
      getInfo: vi.fn().mockResolvedValue({ port: 27183, ready: true }),
    };

    render(<App />);

    await waitFor(() => expect(screen.getByText("MODELS_PAGE")).toBeTruthy());
  });

  it("shows the human label of the reported startup phase", async () => {
    let emit;
    window.backendAPI = {
      onBackendEvent: (cb) => {
        emit = cb;
        return () => {};
      },
      getInfo: vi.fn().mockResolvedValue({ port: null, ready: false }),
    };

    render(<App />);
    await act(async () => {
      emit({ event: "phase", phase: "preparing_database" });
    });

    expect(screen.getByText(/Preparing the database/)).toBeTruthy();
  });
});
