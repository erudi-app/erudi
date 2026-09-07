// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

// Main send flow of ChatPage: asking a question creates a conversation with
// the composer's parameters (POST payload pinned) and hands the question over
// to ConversationPage through router state. Failures surface in the error
// modal instead of navigating. Also covers the history sidebar wiring
// (sorting, rename/delete/refresh) and the model dropdown.

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
vi.mock("../components/GradientBox", () => ({ default: ({ children }) => <div>{children}</div> }));

// The composer is stubbed with a trigger that sends a fixed ask; the page's
// handleAsk (payload + navigation) is what this suite pins.
vi.mock("../components/QuestionInput", () => ({
  default: ({ onSend }) => (
    <button onClick={() => onSend("What is the sun?", ["data:img"], ["/tmp/img.png"])}>
      send-question
    </button>
  ),
}));

// The prompt modal is stubbed with a trigger that saves a known custom prompt.
vi.mock("../components/modals/CustomizePromptModal", () => ({
  default: ({ isOpen, onSave }) =>
    isOpen ? <button onClick={() => onSave("Answer like a pirate")}>save-prompt</button> : null,
}));

// History section stub exposing the page callbacks as buttons.
vi.mock("../components/ChatCollapsibleSection", () => ({
  default: ({ items, onItemClick, onRename, onDelete, onRefresh }) => (
    <div>
      <ul data-testid="history">
        {items.map((c) => (
          <li key={c.id}>
            <button onClick={() => onItemClick(c.id)}>{c.name}</button>
          </li>
        ))}
      </ul>
      <button onClick={() => onRename(1, "Renamed chat")}>rename-first</button>
      <button onClick={() => onDelete(1)}>delete-first</button>
      <button onClick={() => onRefresh()}>refresh</button>
    </div>
  ),
}));

import apiClient from "../services/api/client";
import ChatPage from "./ChatPage.jsx";

const MODELS = [
  { id: 7, name: "Alpha Model" },
  { id: 42, name: "Beta Model" },
];

const CONVERSATIONS = [
  { id: 1, name: "Older chat", last_message_time: "2026-01-01T10:00:00Z" },
  { id: 2, name: "Newer chat", last_message_time: "2026-02-01T10:00:00Z" },
];

const renderPage = () =>
  render(
    <MemoryRouter initialEntries={["/erudi/chat"]}>
      <ChatPage />
    </MemoryRouter>
  );

beforeEach(() => {
  navigateMock.mockReset();
  tracedFetchMock.mockReset();
  tracedFetchMock.mockResolvedValue({ ok: true, json: async () => ({ id: 123 }) });
  apiClient.get.mockReset();
  apiClient.get.mockImplementation(async (path) =>
    path === "/llms/local"
      ? MODELS
      : path === "/user_settings/"
        ? { web_search_enabled: false }
        : CONVERSATIONS
  );
});
afterEach(() => {
  cleanup();
});

describe("ChatPage send flow", () => {
  it("creates a conversation with the default settings and navigates with the ask in state", async () => {
    renderPage();
    fireEvent.click(await screen.findByText("send-question"));

    await waitFor(() => expect(navigateMock).toHaveBeenCalledTimes(1));

    expect(tracedFetchMock).toHaveBeenCalledTimes(1);
    const [url, options] = tracedFetchMock.mock.calls[0];
    expect(String(url)).toContain("/conversations/");
    expect(options.method).toBe("POST");
    expect(JSON.parse(options.body)).toEqual({
      llm_id: 7, // first local model is the default selection
      temperature: 0.2,
      top_p: 0.95,
      max_tokens: 1024,
      custom_prompt: "",
      web_search_enabled: false, // inherited from the global setting (#310)
    });

    const [path, { state }] = navigateMock.mock.calls[0];
    expect(path).toBe("/erudi/conversations/123");
    expect(state).toEqual({
      initialQuestion: "What is the sun?",
      initialImages: ["data:img"],
      initialImagePaths: ["/tmp/img.png"],
      // Documents attached to the very first question ride the same state (#492).
      initialAttachments: [],
      // Models without hints seed from the backend fallback, cap included (#388).
      initialSettings: { temperature: 0.2, topP: 0.95, maxTokens: 1024, maxTokensCap: 32768 },
      initialCustomPrompt: "",
    });
  });

  it("sends the tuned settings, selected model and saved custom prompt", async () => {
    const { container } = renderPage();
    await screen.findByText("send-question");

    // Pick the second model from the dropdown.
    fireEvent.click(screen.getByTitle("Alpha Model"));
    fireEvent.click(screen.getByText("Beta Model"));

    // Open the settings panel and tune every control.
    fireEvent.click(screen.getByLabelText("Toggle settings"));
    const [temperature, topP] = container.querySelectorAll('input[type="range"]');
    fireEvent.change(temperature, { target: { value: "0.8" } });
    fireEvent.change(topP, { target: { value: "0.5" } });
    fireEvent.change(container.querySelector('input[type="number"]'), {
      target: { value: "512" },
    });

    // Save a custom prompt through the modal.
    fireEvent.click(screen.getByText("Customize Prompt"));
    fireEvent.click(screen.getByText("save-prompt"));

    fireEvent.click(screen.getByText("send-question"));
    await waitFor(() => expect(navigateMock).toHaveBeenCalled());

    expect(JSON.parse(tracedFetchMock.mock.calls[0][1].body)).toEqual({
      llm_id: 42,
      temperature: 0.8,
      top_p: 0.5,
      max_tokens: 512,
      custom_prompt: "Answer like a pirate",
      web_search_enabled: false,
    });
    expect(navigateMock.mock.calls[0][1].state.initialCustomPrompt).toBe("Answer like a pirate");
    expect(navigateMock.mock.calls[0][1].state.initialSettings).toEqual({
      temperature: 0.8,
      topP: 0.5,
      maxTokens: 512,
      maxTokensCap: 32768,
    });
  });

  it("inherits an enabled global web-search default and lets the user flip it (#310)", async () => {
    apiClient.get.mockImplementation(async (path) =>
      path === "/llms/local"
        ? MODELS
        : path === "/user_settings/"
          ? { web_search_enabled: true }
          : CONVERSATIONS
    );
    renderPage();
    await screen.findByText("send-question");

    // The pre-conversation panel shows the inherited value.
    fireEvent.click(screen.getByLabelText("Toggle settings"));
    const toggle = await screen.findByRole("switch", { name: "Web search" });
    await waitFor(() => expect(toggle.getAttribute("aria-checked")).toBe("true"));

    // Untouched: the creation payload carries the inherited value.
    fireEvent.click(screen.getByText("send-question"));
    await waitFor(() => expect(navigateMock).toHaveBeenCalled());
    expect(JSON.parse(tracedFetchMock.mock.calls[0][1].body).web_search_enabled).toBe(true);
  });

  it("sends the flipped web-search value on creation (#310)", async () => {
    renderPage();
    await screen.findByText("send-question");

    fireEvent.click(screen.getByLabelText("Toggle settings"));
    const toggle = await screen.findByRole("switch", { name: "Web search" });
    await waitFor(() => expect(toggle.getAttribute("aria-checked")).toBe("false"));
    fireEvent.click(toggle);

    fireEvent.click(screen.getByText("send-question"));
    await waitFor(() => expect(navigateMock).toHaveBeenCalled());
    expect(JSON.parse(tracedFetchMock.mock.calls[0][1].body).web_search_enabled).toBe(true);
  });

  it("shows the error modal and does not navigate when the creation fails", async () => {
    tracedFetchMock.mockResolvedValue({ ok: false, status: 500 });
    renderPage();

    fireEvent.click(await screen.findByText("send-question"));

    expect(
      await screen.findByText(/Failed to start conversation: Failed to create conversation/)
    ).toBeTruthy();
    expect(navigateMock).not.toHaveBeenCalled();
  });
});

describe("ChatPage data loading errors", () => {
  it("surfaces a models fetch failure in the error modal", async () => {
    apiClient.get.mockImplementation(async (path) => {
      if (path === "/llms/local") throw new Error("boom");
      return [];
    });
    renderPage();

    expect(await screen.findByText(/Failed to load models: boom/)).toBeTruthy();
  });

  it("surfaces a conversations fetch failure in the error modal", async () => {
    apiClient.get.mockImplementation(async (path) => {
      if (path === "/conversations/") throw new Error("down");
      return MODELS;
    });
    renderPage();

    expect(await screen.findByText(/Failed to load conversations: down/)).toBeTruthy();
  });
});

describe("ChatPage history sidebar", () => {
  it("lists conversations newest-first and navigates on click", async () => {
    renderPage();

    const history = await screen.findByTestId("history");
    await waitFor(() => expect(within(history).getAllByRole("button")).toHaveLength(2));
    expect(
      within(history)
        .getAllByRole("button")
        .map((b) => b.textContent)
    ).toEqual(["Newer chat", "Older chat"]);

    fireEvent.click(within(history).getByText("Older chat"));
    expect(navigateMock).toHaveBeenCalledWith("/erudi/conversations/1");
  });

  it("applies rename and delete locally", async () => {
    renderPage();
    await screen.findByTestId("history");
    await waitFor(() => expect(screen.queryByText("Older chat")).toBeTruthy());

    fireEvent.click(screen.getByText("rename-first"));
    expect(await screen.findByText("Renamed chat")).toBeTruthy();

    fireEvent.click(screen.getByText("delete-first"));
    await waitFor(() => expect(screen.queryByText("Renamed chat")).toBeNull());
    expect(screen.getByText("Newer chat")).toBeTruthy();
  });

  it("re-fetches and re-sorts on refresh, and surfaces refresh failures", async () => {
    renderPage();
    await screen.findByTestId("history");

    apiClient.get.mockImplementation(async () => [
      { id: 3, name: "Fresh chat", last_message_time: "2026-03-01T10:00:00Z" },
    ]);
    fireEvent.click(screen.getByText("refresh"));
    expect(await screen.findByText("Fresh chat")).toBeTruthy();

    apiClient.get.mockImplementation(async () => {
      throw new Error("offline");
    });
    fireEvent.click(screen.getByText("refresh"));
    expect(await screen.findByText(/Failed to refresh conversations: offline/)).toBeTruthy();
  });
});

describe("ChatPage composer chrome", () => {
  it("closes the model dropdown on an outside mousedown", async () => {
    renderPage();
    await screen.findByText("send-question");

    fireEvent.click(screen.getByTitle("Alpha Model"));
    expect(screen.getByText("Beta Model")).toBeTruthy();

    fireEvent.mouseDown(document.body);
    await waitFor(() => expect(screen.queryByText("Beta Model")).toBeNull());
  });

  it("expands the language note on click", async () => {
    renderPage();
    await screen.findByText("send-question");

    expect(screen.queryByText(/massively trained on English data/)).toBeNull();
    fireEvent.click(screen.getByText("Note on Language"));
    expect(await screen.findByText(/massively trained on English data/)).toBeTruthy();
  });
});
