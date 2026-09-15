// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, act } from "@testing-library/react";

// Saving a prompt in CustomizePromptModal must persist it immediately (#136).
// Before the fix, onSave only updated local state and the prompt was silently
// lost on reload unless the user also clicked Apply in the HeaderBar.

const { tracedFetchMock, navigateMock, locationMock } = vi.hoisted(() => ({
  tracedFetchMock: vi.fn(),
  navigateMock: vi.fn(),
  // Stable identities: the page's mount effect lists navigate/location values
  // in its dependency array, so fresh mocks per render would loop it forever.
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

// Stub the heavy children; the modal stub exposes buttons that drive onSave.
vi.mock("../components/Sidebar", () => ({ default: () => null }));
vi.mock("../components/ChatCollapsibleSection", () => ({ default: () => null }));
vi.mock("../components/QuestionInput", () => ({ default: () => null }));
vi.mock("../components/HeaderBar", () => ({ default: () => null }));
vi.mock("../components/TypingIndicator", () => ({ default: () => null }));
vi.mock("../components/MarkdownRenderer", () => ({ default: () => null }));
vi.mock("../components/modals/CustomizePromptModal", () => ({
  default: ({ onSave }) => (
    <div>
      <button onClick={() => onSave("NEW PROMPT")}>SAVE_PROMPT</button>
      <button onClick={() => onSave("")}>SAVE_EMPTY</button>
    </div>
  ),
}));

import ConversationPage from "./ConversationPage.jsx";
import apiClient from "../services/api/client";

// jsdom does not implement Element.scrollTo (used by the auto-scroll effect).
beforeEach(() => {
  Element.prototype.scrollTo = () => {};
});

const conversationDetail = {
  id: 7,
  llm_id: 1,
  temperature: 0.7,
  top_p: 0.9,
  max_tokens: 512,
  custom_prompt: "OLD PROMPT",
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
  // The mount effect ends with apiClient.get after the conversation detail
  // (settings + custom_prompt) state updates have been queued.
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

describe("ConversationPage prompt persistence (#136)", () => {
  it("persists the prompt immediately when saved in the modal, without Apply", async () => {
    await renderAndSettle();
    expect(patchCalls().length).toBe(0);

    fireEvent.click(screen.getByText("SAVE_PROMPT"));

    await waitFor(() => expect(patchCalls().length).toBe(1));
    const body = JSON.parse(patchCalls()[0][1].body);
    expect(body.custom_prompt).toBe("NEW PROMPT");
    // Current settings (loaded from the conversation) ride along unchanged.
    expect(body.temperature).toBe(0.7);
    expect(body.top_p).toBe(0.9);
    // The output budget is not among them: it is derived backend-side.
    expect(body).not.toHaveProperty("max_tokens");
  });

  it("persists an empty prompt when the user clears it in the modal", async () => {
    await renderAndSettle();

    fireEvent.click(screen.getByText("SAVE_EMPTY"));

    await waitFor(() => expect(patchCalls().length).toBe(1));
    const body = JSON.parse(patchCalls()[0][1].body);
    // "" must not be swallowed by a || fallback onto the previous prompt.
    expect(body.custom_prompt).toBe("");
  });
});
