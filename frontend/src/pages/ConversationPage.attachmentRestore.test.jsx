// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, waitFor, act } from "@testing-library/react";

// #492 — an attached document is persisted as a [file_path:...] marker, not as
// its extracted text. On reload the page turns each marker back into a named
// chip, so a revisited conversation still shows what the turn carried.

const { tracedFetchMock, navigateMock, locationMock } = vi.hoisted(() => ({
  tracedFetchMock: vi.fn(),
  navigateMock: vi.fn(),
  locationMock: { pathname: "/conversation/7", state: null },
}));

vi.mock("../services/api/client", () => ({
  default: { get: vi.fn(async () => []) },
  apiClient: { get: vi.fn(async () => []) },
  tracedFetch: tracedFetchMock,
}));

vi.mock("react-router-dom", () => ({
  useParams: () => ({ id: "7" }),
  useNavigate: () => navigateMock,
  useLocation: () => locationMock,
}));

vi.mock("../components/Sidebar", () => ({ default: () => null }));
vi.mock("../components/ChatCollapsibleSection", () => ({ default: () => null }));
vi.mock("../components/QuestionInput", () => ({ default: () => null }));
vi.mock("../components/HeaderBar", () => ({ default: () => null }));
vi.mock("../components/TypingIndicator", () => ({ default: () => null }));
vi.mock("../components/MarkdownRenderer", () => ({ default: () => null }));
vi.mock("../components/modals/CustomizePromptModal", () => ({ default: () => null }));

import ConversationPage from "./ConversationPage.jsx";
import apiClient from "../services/api/client";

const messages = [
  {
    id: 101,
    sender: "user",
    content: "What does it say? [file_path:/Users/me/docs/report.pdf]",
    starred: false,
  },
];

const conversationDetail = {
  id: 7,
  llm_id: 1,
  temperature: 0.7,
  top_p: 0.9,
  max_tokens: 512,
  custom_prompt: "",
};

beforeEach(() => {
  Element.prototype.scrollTo = () => {};
  apiClient.get.mockReset();
  apiClient.get.mockImplementation(async () => messages);
  tracedFetchMock.mockReset();
  tracedFetchMock.mockImplementation(async (url) => {
    const u = String(url);
    if (u.endsWith("/conversations/7")) return { ok: true, json: async () => conversationDetail };
    return { ok: true, json: async () => [] };
  });
});

afterEach(() => {
  cleanup();
});

describe("ConversationPage attachment restore (#492)", () => {
  it("renders a stored [file_path:...] marker as a named chip, not as text", async () => {
    render(<ConversationPage />);
    await waitFor(() => expect(apiClient.get).toHaveBeenCalled());
    await act(async () => {});

    // The file name shows as a chip...
    expect(await screen.findByText("report.pdf")).toBeTruthy();
    // ...and the raw marker never leaks into the readable text.
    expect(screen.queryByText(/file_path/)).toBeNull();
    expect(screen.getByText("What does it say?")).toBeTruthy();
  });

  it("shows the real name when the stored path carries ] or %", async () => {
    apiClient.get.mockImplementation(async () => [
      {
        id: 102,
        sender: "user",
        content: "Read it [file_path:/docs/[2026%5D 100%25/report v2%5D.pdf]",
        starred: false,
      },
    ]);

    render(<ConversationPage />);
    await waitFor(() => expect(apiClient.get).toHaveBeenCalled());
    await act(async () => {});

    expect(await screen.findByText("report v2].pdf")).toBeTruthy();
    expect(screen.getByText("Read it")).toBeTruthy();
  });
});
