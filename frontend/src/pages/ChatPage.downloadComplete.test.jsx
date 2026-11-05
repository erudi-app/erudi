// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, cleanup, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

// A completed download is surfaced by the DownloadModalContext as an
// incrementing completionCount (context STATE). A user sitting on the Chat
// page during a download must see the model list refresh when that counter
// ticks — otherwise the page keeps showing "No current local models found"
// until they navigate away and back (#303). Mirrors the LandingPage
// downloadComplete coverage (#205).

const { ctx } = vi.hoisted(() => ({ ctx: { completionCount: 0 } }));

// The context value carries completionCount; the test drives it via `ctx`.
vi.mock("../contexts/DownloadModalContext", () => ({
  useDownloadModal: () => ({ open: vi.fn(), completionCount: ctx.completionCount }),
}));

vi.mock("../services/api/client", () => ({
  default: { get: vi.fn() },
  apiClient: { get: vi.fn() },
  tracedFetch: vi.fn(),
}));

// Stub the heavy children — none are needed to observe the refresh fetch.
vi.mock("../components/Sidebar", () => ({ default: () => null }));
vi.mock("../components/ChatCollapsibleSection", () => ({ default: () => null }));
vi.mock("../components/GradientBox", () => ({ default: ({ children }) => <div>{children}</div> }));
vi.mock("../components/QuestionInput", () => ({ default: () => null }));
vi.mock("../components/modals/ErrorModal", () => ({ default: () => null }));

import apiClient from "../services/api/client";
import ChatPage from "./ChatPage.jsx";

const MODELS = [{ id: 7, name: "Alpha Model" }];

const localCalls = () =>
  apiClient.get.mock.calls.filter(([path]) => String(path).includes("/llms/local"));

const renderPage = () =>
  render(
    <MemoryRouter initialEntries={["/erudi/chat"]}>
      <ChatPage />
    </MemoryRouter>
  );

beforeEach(() => {
  ctx.completionCount = 0;
  apiClient.get.mockReset();
  apiClient.get.mockImplementation(async (path) => (path === "/llms/local" ? [] : []));
});
afterEach(() => {
  cleanup();
});

describe("ChatPage refresh on download completion (#303)", () => {
  it("re-fetches the local models when the context completion counter ticks", async () => {
    const { rerender } = renderPage();

    // Mount fetches the local list once; no model installed yet.
    await waitFor(() => expect(localCalls().length).toBeGreaterThan(0));
    const before = localCalls().length;

    // The download completes -> the model exists now -> context bumps the counter.
    apiClient.get.mockImplementation(async (path) => (path === "/llms/local" ? MODELS : []));
    ctx.completionCount = 1;
    rerender(
      <MemoryRouter initialEntries={["/erudi/chat"]}>
        <ChatPage />
      </MemoryRouter>
    );

    await waitFor(() => expect(localCalls().length).toBeGreaterThan(before));
  });

  it("replaces the empty state with the composer once the refreshed list has models", async () => {
    const { rerender, queryByText } = renderPage();

    await waitFor(() => expect(queryByText(/No current local models found/)).toBeTruthy());

    apiClient.get.mockImplementation(async (path) => (path === "/llms/local" ? MODELS : []));
    ctx.completionCount = 1;
    rerender(
      <MemoryRouter initialEntries={["/erudi/chat"]}>
        <ChatPage />
      </MemoryRouter>
    );

    await waitFor(() => expect(queryByText(/No current local models found/)).toBeNull());
  });

  it("does not refresh on mount when no download has completed", async () => {
    renderPage();
    await waitFor(() => expect(localCalls().length).toBeGreaterThan(0));
    // The mount fetch hits /llms/local exactly once; the completion effect
    // stays inert while completionCount is 0.
    expect(localCalls().length).toBe(1);
  });
});
