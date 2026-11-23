// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, cleanup, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

// A list that cannot be loaded says why in the interface language: a request
// that got no answer is the runtime's English "Failed to fetch" otherwise (#780).

vi.mock("../contexts/DownloadModalContext", () => ({
  useDownloadModal: () => ({ open: vi.fn(), completionCount: 0 }),
}));

vi.mock("../services/api/client", () => ({
  default: { get: vi.fn() },
  apiClient: { get: vi.fn() },
  tracedFetch: vi.fn(),
}));

vi.mock("../components/Sidebar", () => ({ default: () => null }));
vi.mock("../components/ChatCollapsibleSection", () => ({ default: () => null }));
vi.mock("../components/GradientBox", () => ({ default: ({ children }) => <div>{children}</div> }));
vi.mock("../components/QuestionInput", () => ({ default: () => null }));
vi.mock("../components/modals/ErrorModal", () => ({
  default: ({ errorMessage }) => (errorMessage ? <p role="alert">{errorMessage}</p> : null),
}));

import apiClient from "../services/api/client";
import ChatPage from "./ChatPage.jsx";

beforeEach(() => {
  apiClient.get.mockReset();
});
afterEach(() => {
  cleanup();
});

describe("ChatPage load failures (#780)", () => {
  it("says the backend is not responding, not the runtime's Failed to fetch", async () => {
    apiClient.get.mockImplementation(async (path) => {
      if (path === "/llms/local") throw new TypeError("Failed to fetch");
      return [];
    });
    render(
      <MemoryRouter initialEntries={["/erudi/chat"]}>
        <ChatPage />
      </MemoryRouter>
    );

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toBe("Failed to load models: The local backend is not responding");
    expect(alert.textContent).not.toContain("Failed to fetch");
  });
});
