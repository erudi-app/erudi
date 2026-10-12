// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, act } from "@testing-library/react";

// #607 — the per-conversation Compaction threshold control: hydrated from the
// conversation GET (compaction_threshold), shown in the HeaderBar settings
// panel, and persisted through a one-field PATCH the moment it changes (like
// the reasoning effort), so the NEXT turn already compacts at it.

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

// HeaderBar probe: exposes the compaction-threshold wiring as testable elements.
vi.mock("../components/HeaderBar", () => ({
  default: ({
    showCompactionThreshold,
    initialCompactionThreshold,
    onCompactionThresholdChange,
  }) => (
    <div>
      <div data-testid="show-compaction-threshold">{String(showCompactionThreshold)}</div>
      <div data-testid="initial-compaction-threshold">{String(initialCompactionThreshold)}</div>
      <button onClick={() => onCompactionThresholdChange(55)}>SET_THRESHOLD_55</button>
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
  web_search_enabled: false,
  reasoning_effort: "medium",
  compaction_threshold: 65,
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

describe("ConversationPage compaction threshold control (#607)", () => {
  it("opts the HeaderBar into the control and hydrates it from the GET", async () => {
    await renderAndSettle();
    expect(screen.getByTestId("show-compaction-threshold").textContent).toBe("true");
    await waitFor(() =>
      expect(screen.getByTestId("initial-compaction-threshold").textContent).toBe("65")
    );
  });

  it("PATCHes compaction_threshold alone, as a number, when changed", async () => {
    await renderAndSettle();
    await waitFor(() =>
      expect(screen.getByTestId("initial-compaction-threshold").textContent).toBe("65")
    );

    fireEvent.click(screen.getByText("SET_THRESHOLD_55"));

    await waitFor(() => expect(patchCalls().length).toBe(1));
    const body = JSON.parse(patchCalls()[0][1].body);
    expect(body).toEqual({ compaction_threshold: 55 });
    // Local state follows optimistically.
    await waitFor(() =>
      expect(screen.getByTestId("initial-compaction-threshold").textContent).toBe("55")
    );
  });
});
