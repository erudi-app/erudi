// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, act, cleanup, waitFor } from "@testing-library/react";

// Stub the heavy pages/contexts so App's readiness logic renders in isolation.
vi.mock("./pages/LandingPage", () => ({ default: () => <div>MODELS_PAGE</div> }));
vi.mock("./pages/ChatPage", () => ({ default: () => <div>CHAT</div> }));
vi.mock("./pages/ConversationPage", () => ({ default: () => <div>CONV</div> }));
vi.mock("./pages/ArenaPage", () => ({ default: () => <div>ARENA</div> }));
vi.mock("./pages/KnowledgeBasePage", () => ({ default: () => <div>KB</div> }));
vi.mock("./pages/SettingsPage", () => ({ default: () => <div>SETTINGS</div> }));
vi.mock("./components/UpdateBanner", () => ({ default: () => null }));
vi.mock("./contexts/DownloadModalContext", () => ({
  DownloadModalProvider: ({ children }) => <>{children}</>,
}));
vi.mock("./contexts/KnowledgeBaseContext", () => ({
  KnowledgeBaseProvider: ({ children }) => <>{children}</>,
}));
const { syncLanguageMock } = vi.hoisted(() => ({ syncLanguageMock: vi.fn() }));
vi.mock("./i18n/sync", () => ({ syncLanguageWithBackend: syncLanguageMock }));

import App from "./App.jsx";
import { getApiBaseUrl } from "./config/api.js";
import { apiClient } from "./services/api/client";

let emit;
beforeEach(() => {
  emit = null;
  window.backendAPI = {
    onBackendEvent: (cb) => {
      emit = cb;
      return () => {
        emit = null;
      };
    },
    getInfo: vi.fn().mockResolvedValue({ port: null, ready: false }),
    restartBackend: vi.fn().mockResolvedValue({ ok: true }),
    getLogPath: vi.fn().mockResolvedValue("/tmp/erudi-backend.log"),
  };
});
afterEach(() => {
  cleanup();
  delete window.backendAPI;
  syncLanguageMock.mockReset();
});

describe("App readiness", () => {
  it("reconciles the interface language with the backend once ready (#385)", async () => {
    render(<App />);
    expect(syncLanguageMock).not.toHaveBeenCalled();
    await act(async () => {
      emit({ event: "ready", port: 8766 });
    });
    await waitFor(() => expect(syncLanguageMock).toHaveBeenCalledWith(apiClient));
  });

  it("shows the loader until a ready event, then the models page", async () => {
    render(<App />);
    expect(screen.getByText(/AI with you, for you/i)).toBeTruthy(); // loading screen
    await act(async () => {
      emit({ event: "ready", port: 8766 });
    });
    await waitFor(() => expect(screen.getByText("MODELS_PAGE")).toBeTruthy());
  });

  it("adopts the backend's resolved port from the starting event", async () => {
    render(<App />);
    await act(async () => {
      emit({ event: "starting", port: 8791, first_run: true });
    });
    expect(getApiBaseUrl()).toContain("8791");
  });

  it("shows the error screen on a startup_error event", async () => {
    render(<App />);
    await act(async () => {
      emit({ event: "startup_error", code: "IMPORT_ERROR" });
    });
    await waitFor(() => expect(screen.getByText(/Backend failed to load/i)).toBeTruthy());
  });

  it("re-spawns the backend when Retry is clicked", async () => {
    render(<App />);
    await act(async () => {
      emit({ event: "startup_error", code: "CRASH_BEFORE_READY" });
    });
    const retry = await screen.findByText("Retry");
    await act(async () => {
      retry.click();
    });
    expect(window.backendAPI.restartBackend).toHaveBeenCalled();
  });
});

describe("App engine notice", () => {
  const NOTICE = {
    event: "engine_notice",
    code: "CUDA_DRIVER_TOO_OLD",
    gpu_name: "NVIDIA GeForce GTX 1080",
    compute_capability: "6.1",
    driver_cuda_version: "12.1",
    required_cuda_version: "12.8",
    raw: "driver reports CUDA 12.1",
  };

  it("holds the notice until the app is ready, then shows the decision dialog", async () => {
    render(<App />);
    await act(async () => {
      emit(NOTICE);
    });
    // Still on the loader: the notice is not a startup failure and must not
    // replace the boot screen or the error screen.
    expect(screen.queryByText(/driver is too old/i)).toBeNull();
    expect(screen.queryByText(/Backend failed/i)).toBeNull();

    await act(async () => {
      emit({ event: "ready", port: 8766 });
    });
    await waitFor(() => expect(screen.getByText(/driver is too old/i)).toBeTruthy());
    // Over the real UI, not instead of it.
    expect(screen.getByText("MODELS_PAGE")).toBeTruthy();
  });

  it("closes on Not now without persisting anything", async () => {
    render(<App />);
    await act(async () => {
      emit({ event: "ready", port: 8766 });
      emit(NOTICE);
    });
    const notNow = await screen.findByText("Not now");
    await act(async () => {
      notNow.click();
    });
    await waitFor(() => expect(screen.queryByText(/driver is too old/i)).toBeNull());
  });

  it("does not treat the notice as a port or readiness event", async () => {
    render(<App />);
    await act(async () => {
      emit(NOTICE);
    });
    // A notice carries no port and never makes the app think it is ready.
    expect(screen.queryByText("MODELS_PAGE")).toBeNull();
  });
});
