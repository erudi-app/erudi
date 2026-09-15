// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, waitFor, fireEvent } from "@testing-library/react";

// Mid-conversation model switch (#388, maintainer decision 1): the sampling
// re-defaults to the NEW model's values whether or not the user touched the
// sliders, and the switch persists llm_id AND the new sampling in one PATCH
// so the next turn already runs on them. The header's max-tokens ceiling
// follows the conversation's model.

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
vi.mock("../components/HeaderBar", () => ({
  default: ({ currentModel, initialTemperature, initialTopP, onModelChange, onApply }) => (
    <div>
      <div data-testid="header">{`${currentModel}|${initialTemperature}|${initialTopP}`}</div>
      <button onClick={() => onModelChange("qwen3")}>PICK_QWEN3</button>
      <button onClick={() => onApply({ temperature: 1.3, topP: 0.9 })}>TOUCH</button>
    </div>
  ),
}));

import ConversationPage from "./ConversationPage.jsx";

const models = [
  {
    id: 1,
    name: "qwen3",
    weights_available: true,
    sampling_defaults: {
      temperature: 0.6,
      top_p: 0.95,
      max_tokens: 1024,
      max_tokens_cap: 8192,
      top_k: 20,
      source: "base_generation_config",
    },
  },
  { id: 2, name: "plain", weights_available: true },
];

const detail = {
  id: 7,
  llm_id: 2,
  temperature: 0.7,
  top_p: 0.9,
  max_tokens: 512,
  custom_prompt: "",
};

const jsonResponse = (payload) => ({ ok: true, json: async () => payload });

const routeFetch = async (url, opts = {}) => {
  const u = String(url);
  if (opts.method === "PATCH") return jsonResponse({});
  if (u.endsWith("/llms/local")) return jsonResponse(models);
  if (u.endsWith("/conversations/7")) return jsonResponse(detail);
  return jsonResponse([]);
};

const header = () => screen.getByTestId("header").textContent;
const patchBodies = () =>
  tracedFetchMock.mock.calls
    .filter(([, opts]) => opts?.method === "PATCH")
    .map(([, opts]) => JSON.parse(opts.body));

beforeEach(() => {
  Element.prototype.scrollTo = () => {};
  tracedFetchMock.mockReset();
  tracedFetchMock.mockImplementation(routeFetch);
});
afterEach(() => {
  cleanup();
});

describe("ConversationPage sampling on model switch (#388)", () => {
  it("hydrates from the conversation row", async () => {
    render(<ConversationPage />);
    await waitFor(() => expect(header()).toBe("plain|0.7|0.9"));
  });

  it("re-defaults the sampling to the new model's values and persists them with llm_id", async () => {
    render(<ConversationPage />);
    await waitFor(() => expect(header()).toBe("plain|0.7|0.9"));

    fireEvent.click(screen.getByText("PICK_QWEN3"));

    await waitFor(() => expect(header()).toBe("qwen3|0.6|0.95"));
    await waitFor(() => expect(patchBodies()).toHaveLength(1));
    // No max_tokens: the row's value is only the backend's fallback budget and
    // nothing in the interface writes it.
    expect(patchBodies()[0]).toEqual({ llm_id: 1, temperature: 0.6, top_p: 0.95 });
  });

  it("re-defaults even after the user touched the sliders", async () => {
    render(<ConversationPage />);
    await waitFor(() => expect(header()).toBe("plain|0.7|0.9"));

    fireEvent.click(screen.getByText("TOUCH"));
    await waitFor(() => expect(header()).toBe("plain|1.3|0.9"));

    fireEvent.click(screen.getByText("PICK_QWEN3"));
    await waitFor(() => expect(header()).toBe("qwen3|0.6|0.95"));
  });
});
