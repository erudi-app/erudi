// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, act } from "@testing-library/react";

// Export from the conversation header (#430): the page gives its header an
// `onExport` that exports the OPEN conversation through the shared export, and
// tells the user when it fails (a cancelled save dialog is not a failure).

const { tracedFetchMock, navigateMock, locationMock, exportMock } = vi.hoisted(() => ({
  tracedFetchMock: vi.fn(),
  navigateMock: vi.fn(),
  locationMock: { pathname: "/conversation/7", state: null },
  exportMock: vi.fn(),
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

vi.mock("../utils/conversationExport", () => ({ exportConversation: exportMock }));
vi.mock("../components/ChatCollapsibleSection", () => ({ default: () => null }));
vi.mock("../components/Sidebar", () => ({ default: () => null }));
vi.mock("../components/QuestionInput", () => ({ default: () => null }));
vi.mock("../components/TypingIndicator", () => ({ default: () => null }));
vi.mock("../components/MarkdownRenderer", () => ({ default: () => null }));
vi.mock("../components/ConversationHeader", () => ({
  default: ({ onExport, exportDisabled }) => (
    <button onClick={onExport} disabled={exportDisabled}>
      HEADER_EXPORT
    </button>
  ),
}));

import ConversationPage from "./ConversationPage.jsx";
import apiClient from "../services/api/client";

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
  tracedFetchMock.mockReset();
  tracedFetchMock.mockImplementation(async (url) => ({
    ok: true,
    json: async () => (String(url).endsWith("/conversations/7") ? conversationDetail : []),
  }));
  apiClient.get.mockReset();
  apiClient.get.mockImplementation(async () => []);
  exportMock.mockReset();
  exportMock.mockResolvedValue({ status: "saved" });
  vi.spyOn(window, "alert").mockImplementation(() => {});
});
afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const renderAndSettle = async () => {
  render(<ConversationPage />);
  await waitFor(() => expect(apiClient.get).toHaveBeenCalled());
  await act(async () => {});
};

describe("ConversationPage export", () => {
  it("exports the open conversation from the header", async () => {
    await renderAndSettle();
    fireEvent.click(screen.getByText("HEADER_EXPORT"));
    await waitFor(() => expect(exportMock).toHaveBeenCalledTimes(1));
    expect(String(exportMock.mock.calls[0][0])).toBe("7");
    expect(window.alert).not.toHaveBeenCalled();
  });

  it("tells the user in the error dialog when the export fails, not in a native alert", async () => {
    exportMock.mockRejectedValue(new Error("disk full"));
    await renderAndSettle();
    fireEvent.click(screen.getByText("HEADER_EXPORT"));
    expect(await screen.findByText("Could not export the conversation: disk full")).toBeTruthy();
    expect(window.alert).not.toHaveBeenCalled();
    fireEvent.click(screen.getAllByRole("button", { name: "Close" })[0]);
    await waitFor(() =>
      expect(screen.queryByText("Could not export the conversation: disk full")).toBeNull()
    );
  });

  it("leaves the header export enabled when no turn is running", async () => {
    await renderAndSettle();
    expect(screen.getByText("HEADER_EXPORT").disabled).toBe(false);
  });
});
