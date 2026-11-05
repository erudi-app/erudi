// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, act } from "@testing-library/react";

// #228 — regression guard for the delete double-fire.
// ChatCollapsibleSection (the child) OWNS the DELETE request; ConversationPage's
// handleDelete must only do post-delete UI (optimistic filter, server refresh,
// navigate away from a now-deleted open conversation) and must NOT issue its own
// DELETE. This test drives the child's onDelete and asserts the parent makes
// ZERO DELETE requests. It FAILS on the pre-fix code, where handleDelete repeated
// the DELETE ~10ms after the child's, sending two mutations with distinct ids.

const { tracedFetchMock, navigateMock, locationMock } = vi.hoisted(() => ({
  tracedFetchMock: vi.fn(),
  navigateMock: vi.fn(),
  // Stable identities: the page's mount effect lists navigate/location values in
  // its dependency array, so fresh mocks per render would loop it forever.
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

// The child stub exposes a button that fires onDelete, simulating the child
// having already issued (and completed) its own single DELETE request.
vi.mock("../components/ChatCollapsibleSection", () => ({
  default: ({ onDelete }) => <button onClick={() => onDelete(7)}>CHILD_ON_DELETE</button>,
}));
vi.mock("../components/Sidebar", () => ({ default: () => null }));
vi.mock("../components/QuestionInput", () => ({ default: () => null }));
vi.mock("../components/ConversationHeader", () => ({ default: () => null }));
vi.mock("../components/TypingIndicator", () => ({ default: () => null }));
vi.mock("../components/MarkdownRenderer", () => ({ default: () => null }));

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

const routeFetch = async (url) => {
  const u = String(url);
  if (u.endsWith("/conversations/7")) return { ok: true, json: async () => conversationDetail };
  return { ok: true, json: async () => [] };
};

const deleteCalls = () =>
  tracedFetchMock.mock.calls.filter(([, opts]) => opts?.method === "DELETE");

const renderAndSettle = async () => {
  render(<ConversationPage />);
  await waitFor(() => expect(apiClient.get).toHaveBeenCalled());
  await act(async () => {});
};

beforeEach(() => {
  // jsdom does not implement Element.scrollTo (used by the auto-scroll effect).
  Element.prototype.scrollTo = () => {};
  tracedFetchMock.mockReset();
  apiClient.get.mockReset();
  apiClient.get.mockImplementation(async () => []);
  tracedFetchMock.mockImplementation(routeFetch);
});
afterEach(() => {
  cleanup();
});

describe("ConversationPage delete ownership (#228)", () => {
  it("handleDelete issues no DELETE of its own — the child owns the mutation", async () => {
    await renderAndSettle();
    expect(deleteCalls()).toHaveLength(0);

    fireEvent.click(screen.getByText("CHILD_ON_DELETE"));

    // Post-delete UI settles: refresh runs, then navigation away from the
    // now-deleted open conversation (id 7 === the routed id).
    await act(async () => {});
    await waitFor(() => expect(navigateMock).toHaveBeenCalledWith("/erudi/chat"));

    // The parent must NOT have repeated the DELETE the child already sent.
    expect(deleteCalls()).toHaveLength(0);
  });
});
