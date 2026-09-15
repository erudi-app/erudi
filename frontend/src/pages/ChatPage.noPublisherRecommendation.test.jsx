// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

// The pre-conversation settings panel tells the user when the selected model's
// publisher gives no sampling recommendation (#388, `sampling_defaults.source
// === "none"`), and stays silent when one exists. The note follows the model
// picker.

const { tracedFetchMock, navigateMock } = vi.hoisted(() => ({
  tracedFetchMock: vi.fn(),
  navigateMock: vi.fn(),
}));

vi.mock("../services/api/client", () => ({
  default: { get: vi.fn() },
  apiClient: { get: vi.fn() },
  tracedFetch: tracedFetchMock,
}));

vi.mock("react-router-dom", async (importOriginal) => ({
  ...(await importOriginal()),
  useNavigate: () => navigateMock,
}));

vi.mock("../contexts/DownloadModalContext", () => ({
  useDownloadModal: () => ({ open: vi.fn(), completionCount: 0 }),
}));

vi.mock("../components/Sidebar", () => ({ default: () => null }));
vi.mock("../components/ChatCollapsibleSection", () => ({ default: () => null }));
vi.mock("../components/GradientBox", () => ({ default: ({ children }) => <div>{children}</div> }));
vi.mock("../components/modals/CustomizePromptModal", () => ({ default: () => null }));
vi.mock("../components/modals/ErrorModal", () => ({ default: () => null }));
vi.mock("../components/QuestionInput", () => ({ default: () => null }));

import apiClient from "../services/api/client";
import ChatPage from "./ChatPage.jsx";

const NOTE = "No sampling recommendation from this model's publisher; neutral defaults applied.";

const QWEN3 = {
  id: 7,
  name: "Qwen3 0.6B",
  sampling_defaults: {
    temperature: 0.6,
    top_p: 0.95,
    max_tokens: 1024,
    max_tokens_cap: 8192,
    top_k: 20,
    source: "base_generation_config",
  },
};
const LLAMA = {
  id: 42,
  name: "Llama 3.2 1B",
  sampling_defaults: {
    temperature: 0.2,
    top_p: 0.95,
    max_tokens: 1024,
    max_tokens_cap: 32768,
    source: "none",
  },
};

const renderPage = () =>
  render(
    <MemoryRouter initialEntries={["/erudi/chat"]}>
      <ChatPage />
    </MemoryRouter>
  );

const openSettings = () => fireEvent.click(screen.getByLabelText("Toggle settings"));
const pickModel = async (name) => {
  fireEvent.click(await screen.findByTitle(/./));
  fireEvent.click(await screen.findByText(name, { selector: "div.px-3" }));
};

beforeEach(() => {
  tracedFetchMock.mockReset();
  apiClient.get.mockReset();
});
afterEach(() => {
  cleanup();
});

describe("ChatPage publisher recommendation note (#388)", () => {
  it("shows the note under the sliders when the selected model has none", async () => {
    apiClient.get.mockImplementation(async (path) =>
      path === "/llms/local" ? [LLAMA, QWEN3] : []
    );
    renderPage();
    await screen.findByTitle("Llama 3.2 1B");
    openSettings();
    const note = await screen.findByTestId("no-publisher-recommendation");
    expect(note.textContent).toBe(NOTE);
  });

  it("shows nothing when the selected model has a recommendation", async () => {
    apiClient.get.mockImplementation(async (path) =>
      path === "/llms/local" ? [QWEN3, LLAMA] : []
    );
    renderPage();
    await screen.findByTitle("Qwen3 0.6B");
    openSettings();
    await screen.findAllByRole("slider");
    expect(screen.queryByTestId("no-publisher-recommendation")).toBeNull();
    expect(screen.queryByText(NOTE)).toBeNull();
  });

  it("follows the model picker", async () => {
    apiClient.get.mockImplementation(async (path) =>
      path === "/llms/local" ? [QWEN3, LLAMA] : []
    );
    renderPage();
    await screen.findByTitle("Qwen3 0.6B");
    openSettings();
    await screen.findAllByRole("slider");
    expect(screen.queryByTestId("no-publisher-recommendation")).toBeNull();

    await pickModel("Llama 3.2 1B");
    await screen.findByTestId("no-publisher-recommendation");

    await pickModel("Qwen3 0.6B");
    await waitFor(() => expect(screen.queryByTestId("no-publisher-recommendation")).toBeNull());
  });
});
