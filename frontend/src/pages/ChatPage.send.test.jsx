// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

// Main send flow of ChatPage: asking a question creates a conversation from
// the draft (the picked model and the values the user set; POST payload
// pinned) and hands the question over to ConversationPage through router
// state. Failures surface in the error modal instead of navigating. Also
// covers the history sidebar wiring (sorting, rename/delete/refresh).

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

// What GET /llms/{id}/conversation_defaults answers for both models.
const resolvedDefaults = ({ webSearch = false } = {}) => ({
  reasoning_effort: { value: "medium", source: "settings" },
  temperature: { value: 0.2, source: "fallback" },
  top_p: { value: 0.95, source: "fallback" },
  compaction_threshold: { value: 80, source: "settings" },
  web_search_enabled: { value: webSearch, source: "settings" },
  custom_prompt: { value: "", source: "fallback" },
});

const routeGet =
  ({ webSearch } = {}) =>
  async (path) => {
    if (path === "/llms/local") return MODELS;
    if (path.endsWith("/conversation_defaults")) return resolvedDefaults({ webSearch });
    return CONVERSATIONS;
  };

const openCommand = async (label) => {
  fireEvent.click(screen.getByRole("button", { name: "Conversation settings" }));
  fireEvent.click(screen.getByRole("button", { name: new RegExp(`^${label}`) }));
  return screen.findByRole("dialog");
};

const closeDialog = async (dialog) => {
  fireEvent.keyDown(dialog, { key: "Escape" });
  await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
};

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
  apiClient.get.mockImplementation(routeGet());
});
afterEach(() => {
  cleanup();
});

describe("ChatPage send flow", () => {
  it("creates a conversation from the model alone and navigates with the ask in state", async () => {
    renderPage();
    await waitFor(() =>
      expect(apiClient.get).toHaveBeenCalledWith("/llms/7/conversation_defaults")
    );
    fireEvent.click(await screen.findByText("send-question"));

    await waitFor(() => expect(navigateMock).toHaveBeenCalledTimes(1));

    expect(tracedFetchMock).toHaveBeenCalledTimes(1);
    const [url, options] = tracedFetchMock.mock.calls[0];
    expect(String(url)).toContain("/conversations/");
    expect(options.method).toBe("POST");
    // The first local model is the default selection; every value the user
    // did not set is resolved by the backend for that model (#609).
    expect(JSON.parse(options.body)).toEqual({ llm_id: 7 });

    const [path, { state }] = navigateMock.mock.calls[0];
    expect(path).toBe("/erudi/conversations/123");
    // The settings do not ride along: the conversation page reads the
    // created conversation and sends the first turn on its values.
    expect(state).toEqual({
      initialQuestion: "What is the sun?",
      initialImages: ["data:img"],
      initialImagePaths: ["/tmp/img.png"],
      // Documents attached to the very first question ride the same state (#492).
      initialAttachments: [],
    });
  });

  it("sends the picked model and the values set through the header's commands", async () => {
    renderPage();
    await screen.findByText("send-question");

    // Pick the second model through the Model command.
    const models = await openCommand("Model");
    fireEvent.click(within(models).getByRole("button", { name: /Beta Model/ }));
    await waitFor(() =>
      expect(apiClient.get).toHaveBeenCalledWith("/llms/42/conversation_defaults")
    );
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());

    // Tune the sliders and save instructions through the Config command.
    const config = await openCommand("Config");
    await waitFor(() =>
      expect(within(config).getByRole("slider", { name: "Creativity" }).disabled).toBe(false)
    );
    fireEvent.change(within(config).getByRole("slider", { name: "Creativity" }), {
      target: { value: "0.8" },
    });
    fireEvent.change(within(config).getByRole("slider", { name: "Diversity" }), {
      target: { value: "0.5" },
    });
    fireEvent.change(within(config).getByRole("textbox", { name: "Custom instructions" }), {
      target: { value: "Answer like a pirate" },
    });
    fireEvent.click(within(config).getByRole("button", { name: "Save instructions" }));
    await closeDialog(config);

    fireEvent.click(screen.getByText("send-question"));
    await waitFor(() => expect(navigateMock).toHaveBeenCalled());

    expect(JSON.parse(tracedFetchMock.mock.calls[0][1].body)).toEqual({
      llm_id: 42,
      temperature: 0.8,
      top_p: 0.5,
      custom_prompt: "Answer like a pirate",
    });
  });

  it("shows the model's resolved web search and leaves it to the backend when untouched (#310)", async () => {
    apiClient.get.mockImplementation(routeGet({ webSearch: true }));
    renderPage();
    await screen.findByText("send-question");

    const config = await openCommand("Config");
    const toggle = within(config).getByRole("switch", { name: "Web search" });
    await waitFor(() => expect(toggle.getAttribute("aria-checked")).toBe("true"));
    await closeDialog(config);

    fireEvent.click(screen.getByText("send-question"));
    await waitFor(() => expect(navigateMock).toHaveBeenCalled());
    expect(JSON.parse(tracedFetchMock.mock.calls[0][1].body)).toEqual({ llm_id: 7 });
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
  it("expands the language note on click", async () => {
    renderPage();
    await screen.findByText("send-question");

    expect(screen.queryByText(/massively trained on English data/)).toBeNull();
    fireEvent.click(screen.getByText("Note on Language"));
    expect(await screen.findByText(/massively trained on English data/)).toBeTruthy();
  });
});
