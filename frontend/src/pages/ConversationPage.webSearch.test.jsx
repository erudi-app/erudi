// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, act } from "@testing-library/react";

// #310 — the per-conversation Web Search toggle: hydrated from the
// conversation GET (web_search_enabled), shown in the HeaderBar settings
// panel, and persisted through a one-field PATCH the moment it is flipped
// (like the model picker), so the NEXT turn already honors it.

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
vi.mock("../components/TypingIndicator", () => ({ default: () => null }));
vi.mock("../components/MarkdownRenderer", () => ({ default: () => null }));
vi.mock("../components/modals/CustomizePromptModal", () => ({ default: () => null }));

// HeaderBar probe: exposes the web-search wiring as testable elements.
vi.mock("../components/HeaderBar", () => ({
  default: ({ showWebSearch, initialWebSearch, webSearchDisabled, onWebSearchChange }) => (
    <div>
      <div data-testid="show-web-search">{String(showWebSearch)}</div>
      <div data-testid="initial-web-search">{String(initialWebSearch)}</div>
      <div data-testid="web-search-disabled">{String(webSearchDisabled)}</div>
      <button onClick={() => onWebSearchChange(!initialWebSearch)}>FLIP_WEB_SEARCH</button>
    </div>
  ),
}));

import ConversationPage from "./ConversationPage.jsx";
import apiClient from "../services/api/client";

beforeEach(() => {
  Element.prototype.scrollTo = () => {};
});

const conversationDetail = {
  id: 7,
  llm_id: 1,
  temperature: 0.7,
  top_p: 0.9,
  max_tokens: 512,
  custom_prompt: "",
  web_search_enabled: true,
};

const routeFetch = async (url, opts = {}) => {
  const u = String(url);
  if (opts.method === "PATCH") return { ok: true, json: async () => ({}) };
  if (u.endsWith("/conversations/7")) return { ok: true, json: async () => conversationDetail };
  return { ok: true, json: async () => [] };
};

const patchCalls = () =>
  tracedFetchMock.mock.calls.filter(
    ([url, opts]) => opts?.method === "PATCH" && String(url).includes("/conversations/7")
  );

// Same routing as `routeFetch`, but `/llms/local` returns the given models
// list instead of an empty array, so the assigned model's tool flags reach
// ConversationPage (#570).
const routeFetchWithModels =
  (models) =>
  async (url, opts = {}) => {
    const u = String(url);
    if (opts.method === "PATCH") return { ok: true, json: async () => ({}) };
    if (u.endsWith("/conversations/7")) return { ok: true, json: async () => conversationDetail };
    if (u.endsWith("/llms/local")) return { ok: true, json: async () => models };
    return { ok: true, json: async () => [] };
  };

const renderAndSettle = async () => {
  render(<ConversationPage />);
  await waitFor(() => expect(apiClient.get).toHaveBeenCalled());
  await act(async () => {});
};

beforeEach(() => {
  tracedFetchMock.mockClear();
  apiClient.get.mockClear();
  tracedFetchMock.mockImplementation(routeFetch);
});
afterEach(() => {
  cleanup();
});

describe("ConversationPage web search toggle (#310)", () => {
  it("opts the HeaderBar into the toggle and hydrates it from the GET", async () => {
    await renderAndSettle();
    expect(screen.getByTestId("show-web-search").textContent).toBe("true");
    await waitFor(() => expect(screen.getByTestId("initial-web-search").textContent).toBe("true"));
  });

  it("PATCHes web_search_enabled immediately when flipped", async () => {
    await renderAndSettle();
    await waitFor(() => expect(screen.getByTestId("initial-web-search").textContent).toBe("true"));

    fireEvent.click(screen.getByText("FLIP_WEB_SEARCH"));

    await waitFor(() => expect(patchCalls().length).toBe(1));
    const body = JSON.parse(patchCalls()[0][1].body);
    expect(body.web_search_enabled).toBe(false);
    // Local state follows optimistically.
    await waitFor(() => expect(screen.getByTestId("initial-web-search").textContent).toBe("false"));
  });
});

describe("ConversationPage web search toggle disabled for tool-incapable models (#570)", () => {
  it("disables the toggle when the assigned model cannot execute tools", async () => {
    tracedFetchMock.mockImplementation(
      routeFetchWithModels([
        { id: 1, name: "no-tools-model", supports_tools: false, supports_tools_wire: null },
      ])
    );
    await renderAndSettle();
    await waitFor(() => expect(screen.getByTestId("web-search-disabled").textContent).toBe("true"));
  });

  it("disables the toggle when the wire is known unreliable even if supports_tools is true", async () => {
    tracedFetchMock.mockImplementation(
      routeFetchWithModels([
        { id: 1, name: "unreliable-wire-model", supports_tools: true, supports_tools_wire: false },
      ])
    );
    await renderAndSettle();
    await waitFor(() => expect(screen.getByTestId("web-search-disabled").textContent).toBe("true"));
  });

  it("keeps the toggle enabled when the model can execute tools", async () => {
    tracedFetchMock.mockImplementation(
      routeFetchWithModels([
        { id: 1, name: "tool-model", supports_tools: true, supports_tools_wire: true },
      ])
    );
    await renderAndSettle();
    await waitFor(() =>
      expect(screen.getByTestId("web-search-disabled").textContent).toBe("false")
    );
  });

  it("keeps the toggle enabled when the flags are unknown (not yet detected)", async () => {
    tracedFetchMock.mockImplementation(routeFetchWithModels([{ id: 1, name: "unknown-model" }]));
    await renderAndSettle();
    await waitFor(() =>
      expect(screen.getByTestId("web-search-disabled").textContent).toBe("false")
    );
  });

  it("keeps the toggle enabled when the assigned model is orphaned/unknown", async () => {
    tracedFetchMock.mockImplementation(routeFetchWithModels([]));
    await renderAndSettle();
    await waitFor(() =>
      expect(screen.getByTestId("web-search-disabled").textContent).toBe("false")
    );
  });
});
