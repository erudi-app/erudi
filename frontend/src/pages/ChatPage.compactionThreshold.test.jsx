// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

// #607 — the pre-conversation panel's Compaction threshold control: seeded
// from the global default (`GET /user_settings/`), and sent explicitly on the
// creation POST once known (mirrors the Reasoning effort control).

const { tracedFetchMock, navigateMock } = vi.hoisted(() => ({
  tracedFetchMock: vi.fn(),
  navigateMock: vi.fn(),
}));

vi.mock("../services/api/client", () => ({
  default: { get: vi.fn() },
  apiClient: { get: vi.fn() },
  tracedFetch: tracedFetchMock,
}));

vi.mock("react-router-dom", async (importOriginal) => ({
  ...(await importOriginal()),
  useNavigate: () => navigateMock,
}));

vi.mock("../contexts/DownloadModalContext", () => ({
  useDownloadModal: () => ({ open: vi.fn(), completionCount: 0 }),
}));

vi.mock("../components/Sidebar", () => ({ default: () => null }));
vi.mock("../components/ChatCollapsibleSection", () => ({ default: () => null }));
vi.mock("../components/GradientBox", () => ({ default: ({ children }) => <div>{children}</div> }));
vi.mock("../components/modals/CustomizePromptModal", () => ({ default: () => null }));
vi.mock("../components/modals/ErrorModal", () => ({ default: () => null }));
vi.mock("../components/QuestionInput", () => ({
  default: ({ onSend }) => <button onClick={() => onSend("hello", [], [])}>send-question</button>,
}));

import apiClient from "../services/api/client";
import ChatPage from "./ChatPage.jsx";

const QWEN3 = {
  id: 7,
  name: "Qwen3 0.6B",
  sampling_defaults: {
    temperature: 0.6,
    top_p: 0.95,
    max_tokens: 1024,
    max_tokens_cap: 8192,
    top_k: 20,
    source: "base_generation_config",
  },
};

const renderPage = () =>
  render(
    <MemoryRouter initialEntries={["/erudi/chat"]}>
      <ChatPage />
    </MemoryRouter>
  );

const openSettings = () => fireEvent.click(screen.getByLabelText("Toggle settings"));
const thresholdSelect = () => screen.getByRole("combobox", { name: "Compaction threshold" });

const creationBody = () => {
  const [, opts] = tracedFetchMock.mock.calls.find(([url]) =>
    String(url).endsWith("/conversations/")
  );
  return JSON.parse(opts.body);
};

beforeEach(() => {
  tracedFetchMock.mockReset();
  tracedFetchMock.mockResolvedValue({ ok: true, json: async () => ({ id: 99 }) });
  apiClient.get.mockReset();
  apiClient.get.mockImplementation(async (path) => {
    if (path === "/llms/local") return [QWEN3];
    if (path === "/user_settings/")
      return {
        web_search_enabled: false,
        default_reasoning_effort: "medium",
        default_compaction_threshold: 60,
      };
    return [];
  });
});
afterEach(() => {
  cleanup();
});

describe("ChatPage compaction threshold control (#607)", () => {
  it("seeds the select from the global default", async () => {
    renderPage();
    await screen.findByTitle("Qwen3 0.6B");
    openSettings();

    await waitFor(() => expect(thresholdSelect().value).toBe("60"));
    expect(thresholdSelect().disabled).toBe(false);
  });

  it("sends the seeded value explicitly on the creation POST", async () => {
    renderPage();
    await screen.findByTitle("Qwen3 0.6B");
    openSettings();
    await waitFor(() => expect(thresholdSelect().value).toBe("60"));

    fireEvent.click(screen.getByText("send-question"));

    await waitFor(() => expect(tracedFetchMock).toHaveBeenCalled());
    expect(creationBody().compaction_threshold).toBe(60);
  });

  it("sends a manually changed value, as a number, on the creation POST", async () => {
    renderPage();
    await screen.findByTitle("Qwen3 0.6B");
    openSettings();
    await waitFor(() => expect(thresholdSelect().value).toBe("60"));

    fireEvent.change(thresholdSelect(), { target: { value: "90" } });
    fireEvent.click(screen.getByText("send-question"));

    await waitFor(() => expect(tracedFetchMock).toHaveBeenCalled());
    expect(creationBody().compaction_threshold).toBe(90);
  });

  describe("when the global default cannot be fetched", () => {
    beforeEach(() => {
      apiClient.get.mockImplementation(async (path) => {
        if (path === "/llms/local") return [QWEN3];
        if (path === "/user_settings/") throw new Error("backend unreachable");
        return [];
      });
    });

    it("shows 80 %, enabled, and leaves the field to the backend", async () => {
      renderPage();
      await screen.findByTitle("Qwen3 0.6B");
      openSettings();

      await waitFor(() => expect(thresholdSelect().disabled).toBe(false));
      expect(thresholdSelect().value).toBe("80");
      fireEvent.click(screen.getByText("send-question"));

      await waitFor(() => expect(tracedFetchMock).toHaveBeenCalled());
      expect(creationBody()).not.toHaveProperty("compaction_threshold");
    });

    it("sends a value the user picked", async () => {
      renderPage();
      await screen.findByTitle("Qwen3 0.6B");
      openSettings();
      await waitFor(() => expect(thresholdSelect().disabled).toBe(false));

      fireEvent.change(thresholdSelect(), { target: { value: "60" } });
      fireEvent.click(screen.getByText("send-question"));

      await waitFor(() => expect(tracedFetchMock).toHaveBeenCalled());
      expect(creationBody().compaction_threshold).toBe(60);
    });
  });

  it("is disabled and omitted from the creation POST while the global default has not loaded", async () => {
    apiClient.get.mockImplementation(async (path) => {
      if (path === "/llms/local") return [QWEN3];
      // /user_settings/ never resolves in this test.
      if (path === "/user_settings/") return new Promise(() => {});
      return [];
    });
    renderPage();
    await screen.findByTitle("Qwen3 0.6B");
    openSettings();
    expect(thresholdSelect().disabled).toBe(true);
    fireEvent.click(screen.getByText("send-question"));

    await waitFor(() => expect(tracedFetchMock).toHaveBeenCalled());
    expect(creationBody()).not.toHaveProperty("compaction_threshold");
  });
});
