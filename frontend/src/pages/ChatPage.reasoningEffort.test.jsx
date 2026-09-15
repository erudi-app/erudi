// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

// PR-D2 — the pre-conversation panel's Reasoning effort control: seeded from
// the global default (`GET /user_settings/`), and sent explicitly on the
// creation POST once known (mirrors the Web search toggle, #310).

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
const reasoningSelect = () => screen.getByRole("combobox", { name: "Reasoning effort" });

beforeEach(() => {
  tracedFetchMock.mockReset();
  tracedFetchMock.mockResolvedValue({ ok: true, json: async () => ({ id: 99 }) });
  apiClient.get.mockReset();
  apiClient.get.mockImplementation(async (path) => {
    if (path === "/llms/local") return [QWEN3];
    if (path === "/user_settings/")
      return { web_search_enabled: false, default_reasoning_effort: "high" };
    return [];
  });
});
afterEach(() => {
  cleanup();
});

describe("ChatPage reasoning effort control (PR-D2)", () => {
  it("seeds the select from the global default", async () => {
    renderPage();
    await screen.findByTitle("Qwen3 0.6B");
    openSettings();

    await waitFor(() => expect(reasoningSelect().value).toBe("high"));
  });

  it("sends the seeded value explicitly on the creation POST", async () => {
    renderPage();
    await screen.findByTitle("Qwen3 0.6B");
    openSettings();
    await waitFor(() => expect(reasoningSelect().value).toBe("high"));

    fireEvent.click(screen.getByText("send-question"));

    await waitFor(() => expect(tracedFetchMock).toHaveBeenCalled());
    const [, opts] = tracedFetchMock.mock.calls.find(([url]) =>
      String(url).endsWith("/conversations/")
    );
    const body = JSON.parse(opts.body);
    expect(body.reasoning_effort).toBe("high");
  });

  it("sends a manually flipped value on the creation POST", async () => {
    renderPage();
    await screen.findByTitle("Qwen3 0.6B");
    openSettings();
    await waitFor(() => expect(reasoningSelect().value).toBe("high"));

    fireEvent.change(reasoningSelect(), { target: { value: "none" } });
    fireEvent.click(screen.getByText("send-question"));

    await waitFor(() => expect(tracedFetchMock).toHaveBeenCalled());
    const [, opts] = tracedFetchMock.mock.calls.find(([url]) =>
      String(url).endsWith("/conversations/")
    );
    const body = JSON.parse(opts.body);
    expect(body.reasoning_effort).toBe("none");
  });

  it("omits reasoning_effort from the creation POST while the global default has not loaded", async () => {
    apiClient.get.mockImplementation(async (path) => {
      if (path === "/llms/local") return [QWEN3];
      // /user_settings/ never resolves in this test.
      if (path === "/user_settings/") return new Promise(() => {});
      return [];
    });
    renderPage();
    await screen.findByTitle("Qwen3 0.6B");
    fireEvent.click(screen.getByText("send-question"));

    await waitFor(() => expect(tracedFetchMock).toHaveBeenCalled());
    const [, opts] = tracedFetchMock.mock.calls.find(([url]) =>
      String(url).endsWith("/conversations/")
    );
    const body = JSON.parse(opts.body);
    expect(body).not.toHaveProperty("reasoning_effort");
  });
});
