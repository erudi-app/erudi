// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

// The ephemeral toggle of the new-conversation page (#428): off by default,
// and when switched on the conversation is created with `ephemeral: true` and
// remembered as ephemeral, so leaving it deletes it.

const { tracedFetchMock, navigateMock } = vi.hoisted(() => ({
  tracedFetchMock: vi.fn(),
  navigateMock: vi.fn(),
}));

vi.mock("../services/api/client", () => ({
  default: { get: vi.fn() },
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
vi.mock("../components/QuestionInput", () => ({
  default: ({ onSend }) => <button onClick={() => onSend("Secret plans", [], [])}>SEND</button>,
}));

import apiClient from "../services/api/client";
import ChatPage from "./ChatPage.jsx";
import { isEphemeral, resetEphemeralConversations } from "../utils/ephemeralConversations";

const MODELS = [{ id: 7, name: "Alpha Model" }];

const defaults = {
  reasoning_effort: { value: "medium", source: "settings" },
  temperature: { value: 0.2, source: "fallback" },
  top_p: { value: 0.95, source: "fallback" },
  compaction_threshold: { value: 80, source: "settings" },
  web_search_enabled: { value: false, source: "settings" },
  custom_prompt: { value: "", source: "fallback" },
};

const renderPage = () =>
  render(
    <MemoryRouter initialEntries={["/erudi/chat"]}>
      <ChatPage />
    </MemoryRouter>
  );

const created = () => JSON.parse(tracedFetchMock.mock.calls[0][1].body);

beforeEach(() => {
  resetEphemeralConversations();
  navigateMock.mockReset();
  tracedFetchMock.mockReset();
  apiClient.get.mockReset();
  apiClient.get.mockImplementation(async (path) => {
    if (path === "/llms/local") return MODELS;
    if (path.endsWith("/conversation_defaults")) return defaults;
    return [];
  });
});
afterEach(cleanup);

describe("ChatPage ephemeral toggle", () => {
  it("is off by default and creates a regular conversation", async () => {
    tracedFetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({ id: 123, ephemeral: false }),
    });
    renderPage();
    const toggle = await screen.findByRole("switch", { name: "Ephemeral conversation" });
    expect(toggle.getAttribute("aria-checked")).toBe("false");
    await waitFor(() =>
      expect(apiClient.get).toHaveBeenCalledWith("/llms/7/conversation_defaults")
    );

    fireEvent.click(screen.getByText("SEND"));
    await waitFor(() => expect(navigateMock).toHaveBeenCalled());

    expect(created()).toEqual({ llm_id: 7 });
    expect(isEphemeral(123)).toBe(false);
  });

  it("creates an ephemeral conversation when switched on, and says what that means", async () => {
    tracedFetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({ id: 124, ephemeral: true }),
    });
    renderPage();
    const toggle = await screen.findByRole("switch", { name: "Ephemeral conversation" });
    await waitFor(() =>
      expect(apiClient.get).toHaveBeenCalledWith("/llms/7/conversation_defaults")
    );

    fireEvent.click(toggle);
    expect(toggle.getAttribute("aria-checked")).toBe("true");
    expect(screen.getByText("This conversation is deleted when you leave it.")).toBeTruthy();

    fireEvent.click(screen.getByText("SEND"));
    await waitFor(() => expect(navigateMock).toHaveBeenCalled());

    expect(created()).toEqual({ llm_id: 7, ephemeral: true });
    expect(isEphemeral(124)).toBe(true);
    expect(navigateMock.mock.calls[0][0]).toBe("/erudi/conversations/124");
  });
});
