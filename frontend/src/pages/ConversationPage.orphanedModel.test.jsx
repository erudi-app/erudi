// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, waitFor, fireEvent } from "@testing-library/react";

// A conversation whose model was deleted SURVIVES with llm_id null (#225):
// sending is blocked, the header picker gets a red attention state with a
// "Please select a model" prompt, and only an explicit switch (the existing
// PATCH llm_id flow) unblocks the composer — there is no auto-fallback.

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

vi.mock("../components/Sidebar", () => ({ default: () => null }));
vi.mock("../components/ChatCollapsibleSection", () => ({ default: () => null }));
vi.mock("../components/TypingIndicator", () => ({ default: () => null }));
vi.mock("../components/MarkdownRenderer", () => ({ default: () => null }));
vi.mock("../components/modals/CustomizePromptModal", () => ({ default: () => null }));
// Surfaces the composer's blocked state.
vi.mock("../components/QuestionInput", () => ({
  default: ({ disabled }) => (
    <div data-testid="composer" data-disabled={disabled ? "true" : "false"} />
  ),
}));
// Surfaces the picker attention wiring and lets the test switch models.
vi.mock("../components/HeaderBar", () => ({
  default: ({ currentModel, pickerAttention, pickerAttentionMessage, onModelChange }) => (
    <div>
      <div data-testid="model-picker" data-attention={pickerAttention ? "true" : "false"}>
        {currentModel || "Select model..."}
      </div>
      {pickerAttention && <span>{pickerAttentionMessage}</span>}
      <button onClick={() => onModelChange("llama-3.2-1b-instruct")}>PICK_MODEL</button>
    </div>
  ),
}));

import ConversationPage from "./ConversationPage.jsx";
import apiClient from "../services/api/client";

const models = [
  { id: 1, name: "llama-3.2-1b-instruct", weights_available: true },
  { id: 2, name: "qwen2.5-3b-instruct", weights_available: true },
];

const makeDetail = (llmId) => ({
  id: 7,
  llm_id: llmId,
  temperature: 0.7,
  top_p: 0.9,
  max_tokens: 512,
  custom_prompt: "",
});

const jsonResponse = (payload) => ({ ok: true, json: async () => payload });

const makeRouteFetch =
  (detail) =>
  async (url, opts = {}) => {
    const u = String(url);
    if (opts.method === "PATCH") return jsonResponse({});
    if (u.endsWith("/llms/local")) return jsonResponse(models);
    if (u.endsWith("/conversations/7")) return jsonResponse(detail);
    return jsonResponse([]);
  };

const composer = () => screen.getByTestId("composer");
const picker = () => screen.getByTestId("model-picker");

beforeEach(() => {
  // jsdom does not implement Element.scrollTo (used by the auto-scroll effect).
  Element.prototype.scrollTo = () => {};
  tracedFetchMock.mockClear();
  apiClient.get.mockClear();
});
afterEach(() => {
  cleanup();
});

describe("ConversationPage orphaned conversation (#225)", () => {
  it("blocks sending and flags the picker when llm_id is null", async () => {
    tracedFetchMock.mockImplementation(makeRouteFetch(makeDetail(null)));
    render(<ConversationPage />);

    await waitFor(() => expect(composer().dataset.disabled).toBe("true"));
    expect(picker().dataset.attention).toBe("true");
    expect(screen.getByText("Please select a model")).toBeTruthy();
  });

  it("blocks when the assigned model is missing from the installed list", async () => {
    tracedFetchMock.mockImplementation(makeRouteFetch(makeDetail(99)));
    render(<ConversationPage />);

    await waitFor(() => expect(composer().dataset.disabled).toBe("true"));
    expect(picker().dataset.attention).toBe("true");
  });

  it("switching models PATCHes llm_id and unblocks the composer", async () => {
    tracedFetchMock.mockImplementation(makeRouteFetch(makeDetail(null)));
    render(<ConversationPage />);
    await waitFor(() => expect(composer().dataset.disabled).toBe("true"));

    fireEvent.click(screen.getByText("PICK_MODEL"));

    await waitFor(() => expect(composer().dataset.disabled).toBe("false"));
    expect(picker().dataset.attention).toBe("false");
    const patches = tracedFetchMock.mock.calls.filter(([, opts]) => opts?.method === "PATCH");
    expect(patches).toHaveLength(1);
    expect(String(patches[0][0])).toContain("/conversations/7");
    // The switch also re-defaults the sampling to the new model's values
    // (#388); this model has no hints, so the backend fallback rides along.
    expect(JSON.parse(patches[0][1].body)).toEqual({
      llm_id: 1,
      temperature: 0.2,
      top_p: 0.95,
    });
  });

  it("does not block a healthy conversation", async () => {
    tracedFetchMock.mockImplementation(makeRouteFetch(makeDetail(2)));
    render(<ConversationPage />);

    await waitFor(() => expect(picker().textContent).toBe("qwen2.5-3b-instruct"));
    expect(composer().dataset.disabled).toBe("false");
    expect(picker().dataset.attention).toBe("false");
  });
});
