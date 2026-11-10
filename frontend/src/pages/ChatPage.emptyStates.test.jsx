// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";
import { MemoryRouter, Routes, Route } from "react-router-dom";

// The Chat page has three states: no installed model at all, only KB
// assistants whose base model was deleted (nothing can answer: guidance, not
// a header asking for a pick with nothing to pick), and a selectable model
// (#690). The list may also become empty after a refresh.

const { downloadState } = vi.hoisted(() => ({ downloadState: { completionCount: 0 } }));

vi.mock("../services/api/client", () => ({
  default: { get: vi.fn() },
  tracedFetch: vi.fn(),
}));

vi.mock("../contexts/DownloadModalContext", () => ({
  useDownloadModal: () => ({ open: vi.fn(), completionCount: downloadState.completionCount }),
}));

vi.mock("../components/Sidebar", () => ({ default: () => null }));
vi.mock("../components/ChatCollapsibleSection", () => ({ default: () => null }));
vi.mock("../components/GradientBox", () => ({ default: ({ children }) => <div>{children}</div> }));
vi.mock("../components/QuestionInput", () => ({ default: () => <div data-testid="composer" /> }));
vi.mock("../components/modals/ErrorModal", () => ({ default: () => null }));

import apiClient from "../services/api/client";
import ChatPage from "./ChatPage.jsx";

const ORPHAN = {
  id: 9,
  name: "Nimbus Assistant",
  kb_id: 3,
  is_attached_to_kb: true,
  weights_available: false,
};
const HEALTHY = { id: 42, name: "Beta Model", weights_available: true };

const useModels = (models) => {
  apiClient.get.mockImplementation(async (path) => (path === "/llms/local" ? models : []));
};

const tree = () => (
  <MemoryRouter initialEntries={["/erudi/chat"]}>
    <Routes>
      <Route path="/erudi/chat" element={<ChatPage />} />
      <Route path="/erudi/models" element={<div>models page</div>} />
    </Routes>
  </MemoryRouter>
);

beforeEach(() => {
  apiClient.get.mockReset();
  downloadState.completionCount = 0;
});
afterEach(() => {
  cleanup();
});

describe("ChatPage empty states (#690)", () => {
  it("shows the no-models state when nothing is installed", async () => {
    useModels([]);
    render(tree());
    expect(await screen.findByText(/No current local models found/)).toBeTruthy();
    expect(screen.queryByTestId("composer")).toBeNull();
  });

  it("shows guidance, not the red picker, when only orphaned assistants exist", async () => {
    useModels([ORPHAN]);
    render(tree());
    expect(await screen.findByText(/was deleted/)).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Please select a model" })).toBeNull();
    expect(screen.queryByTestId("composer")).toBeNull();
    expect(screen.getByRole("button", { name: "Download a model" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Re-bind the assistant" })).toBeNull();
  });

  it("falls back to the no-models text when a plain model lost its weights", async () => {
    useModels([{ id: 5, name: "Gone Model", weights_available: false }]);
    render(tree());
    expect(await screen.findByText(/No current local models found/)).toBeTruthy();
    expect(screen.queryByText(/was deleted/)).toBeNull();
    expect(screen.queryByRole("button", { name: "Please select a model" })).toBeNull();
  });

  it("falls back to the no-models text when a plain model and an assistant both lack weights", async () => {
    useModels([ORPHAN, { id: 5, name: "Gone Model", weights_available: false }]);
    render(tree());
    expect(await screen.findByText(/No current local models found/)).toBeTruthy();
  });

  it("opens the Models page from the download button", async () => {
    useModels([ORPHAN]);
    render(tree());
    fireEvent.click(await screen.findByRole("button", { name: "Download a model" }));
    expect(await screen.findByText("models page")).toBeTruthy();
  });

  it("shows today's page when a model is selectable", async () => {
    useModels([ORPHAN, HEALTHY]);
    render(tree());
    expect(await screen.findByTestId("composer")).toBeTruthy();
    expect(screen.queryByText(/was deleted/)).toBeNull();
  });

  it("drops the stale list when a refresh returns an empty one", async () => {
    useModels([HEALTHY]);
    const view = render(tree());
    expect(await screen.findByTestId("composer")).toBeTruthy();

    useModels([]);
    downloadState.completionCount = 1;
    view.rerender(tree());
    await waitFor(() => expect(screen.getByText(/No current local models found/)).toBeTruthy());
    expect(screen.queryByTestId("composer")).toBeNull();
  });
});
