// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, act } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

// Deep links into the chat page carry a `?model=` parameter (issue #223).
// LandingPage passes the model NAME, but consumers may also link by numeric
// id. The page must preselect the referenced model in both cases, fall back
// to the default (first local model) when the reference is unknown, and
// apply the selection even when the models list resolves after mount.

const { tracedFetchMock } = vi.hoisted(() => ({ tracedFetchMock: vi.fn() }));

vi.mock("../services/api/client", () => ({
  default: { get: vi.fn() },
  tracedFetch: tracedFetchMock,
}));

// ChatPage reads completionCount from the download context (#303); this suite
// exercises only the ?model= preselection, so an inert context is enough.
vi.mock("../contexts/DownloadModalContext", () => ({
  useDownloadModal: () => ({ open: vi.fn(), completionCount: 0 }),
}));

// Stub the heavy children; the header naming the model stays real.
vi.mock("../components/Sidebar", () => ({ default: () => null }));
vi.mock("../components/ChatCollapsibleSection", () => ({ default: () => null }));
vi.mock("../components/GradientBox", () => ({ default: ({ children }) => <div>{children}</div> }));
vi.mock("../components/QuestionInput", () => ({ default: () => null }));
vi.mock("../components/modals/ErrorModal", () => ({ default: () => null }));

import apiClient from "../services/api/client";
import ChatPage from "./ChatPage.jsx";

const MODELS = [
  { id: 7, name: "Alpha Model" },
  { id: 42, name: "Beta Model" },
];

const renderAt = (search) =>
  render(
    <MemoryRouter initialEntries={[`/erudi/chat${search}`]}>
      <ChatPage />
    </MemoryRouter>
  );

/** The header names the selected model, titled with its name. */
const selectedModelNode = (name) => screen.findByTitle(name);

beforeEach(() => {
  apiClient.get.mockReset();
  apiClient.get.mockImplementation(async (path) => (path === "/llms/local" ? MODELS : []));
});
afterEach(() => {
  cleanup();
});

describe("ChatPage ?model= preselection (#223)", () => {
  it("preselects the model referenced by name (URL-encoded)", async () => {
    renderAt(`?model=${encodeURIComponent("Beta Model")}`);

    const selected = await selectedModelNode("Beta Model");
    expect(selected.textContent).toBe("Beta Model");
  });

  it("preselects the model referenced by numeric id", async () => {
    renderAt("?model=42");

    const selected = await selectedModelNode("Beta Model");
    expect(selected.textContent).toBe("Beta Model");
  });

  it("keeps the default selection when the referenced model is unknown", async () => {
    renderAt(`?model=${encodeURIComponent("No Such Model")}`);

    const selected = await selectedModelNode("Alpha Model");
    expect(selected.textContent).toBe("Alpha Model");
  });

  it("applies the selection when the models list arrives after mount", async () => {
    let resolveModels;
    apiClient.get.mockImplementation((path) => {
      if (path === "/llms/local") return new Promise((resolve) => (resolveModels = resolve));
      return Promise.resolve([]);
    });

    renderAt("?model=42");

    // Before the list resolves the page shows its empty state…
    expect(screen.getByText(/No current local models found/)).toBeTruthy();

    // …and once it does, the deep-linked model is selected, not the default.
    await act(async () => resolveModels(MODELS));
    const selected = await selectedModelNode("Beta Model");
    expect(selected.textContent).toBe("Beta Model");
  });
});
