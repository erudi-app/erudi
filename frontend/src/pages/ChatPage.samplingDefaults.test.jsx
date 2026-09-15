// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

// Per-model sampling defaults on the pre-conversation panel (#388): the
// sliders seed from the selected model's `sampling_defaults`, re-default on
// every model switch whether or not they were touched (maintainer decision
// 1), keep a touched value while the model stays the same, and the creation
// POST sends what the panel shows. There is no output-budget control: the
// backend derives it from the context window the model runs in, so the panel
// neither shows it nor sends it.

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
vi.mock("../components/QuestionInput", () => ({
  default: ({ onSend }) => <button onClick={() => onSend("hello", [], [])}>send-question</button>,
}));

import apiClient from "../services/api/client";
import ChatPage from "./ChatPage.jsx";

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
const PLAIN = {
  id: 42,
  name: "Plain Model",
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
const sliders = () => screen.getAllByRole("slider");
const temperatureSlider = () => sliders()[0];

beforeEach(() => {
  tracedFetchMock.mockReset();
  tracedFetchMock.mockResolvedValue({ ok: true, json: async () => ({ id: 99 }) });
  apiClient.get.mockReset();
  apiClient.get.mockImplementation(async (path) => (path === "/llms/local" ? [QWEN3, PLAIN] : []));
});
afterEach(() => {
  cleanup();
});

describe("ChatPage per-model sampling defaults (#388)", () => {
  it("seeds the panel from the selected model's defaults", async () => {
    renderPage();
    await screen.findByTitle("Qwen3 0.6B");
    openSettings();

    await waitFor(() => expect(temperatureSlider().value).toBe("0.6"));
    expect(screen.getByText("0.60")).toBeTruthy();
  });

  it("caps temperature on the backend's own bound and offers no output budget", async () => {
    renderPage();
    await screen.findByTitle("Qwen3 0.6B");
    openSettings();

    expect(temperatureSlider().getAttribute("max")).toBe("2");
    expect(screen.queryByRole("spinbutton")).toBeNull();
  });

  it("follows the model on switch while the sliders are untouched", async () => {
    renderPage();
    await screen.findByTitle("Qwen3 0.6B");
    openSettings();
    await waitFor(() => expect(temperatureSlider().value).toBe("0.6"));

    await pickModel("Plain Model");
    await waitFor(() => expect(temperatureSlider().value).toBe("0.2"));

    await pickModel("Qwen3 0.6B");
    await waitFor(() => expect(temperatureSlider().value).toBe("0.6"));
  });

  it("re-defaults on a model switch even after the user touched a slider", async () => {
    renderPage();
    await screen.findByTitle("Qwen3 0.6B");
    openSettings();
    await waitFor(() => expect(temperatureSlider().value).toBe("0.6"));

    fireEvent.change(temperatureSlider(), { target: { value: "1.3" } });
    expect(temperatureSlider().value).toBe("1.3");

    await pickModel("Plain Model");
    await screen.findByTitle("Plain Model");
    await waitFor(() => expect(temperatureSlider().value).toBe("0.2"));
  });

  it("keeps a touched value while the model stays the same and sends it at creation", async () => {
    renderPage();
    await screen.findByTitle("Qwen3 0.6B");
    openSettings();
    await waitFor(() => expect(temperatureSlider().value).toBe("0.6"));

    fireEvent.change(temperatureSlider(), { target: { value: "1.3" } });
    fireEvent.click(screen.getByText("send-question"));

    await waitFor(() => expect(tracedFetchMock).toHaveBeenCalled());
    const [, opts] = tracedFetchMock.mock.calls.find(([url]) =>
      String(url).endsWith("/conversations/")
    );
    const body = JSON.parse(opts.body);
    expect([body.temperature, body.top_p]).toEqual([1.3, 0.95]);
    // The conversation is created without an output budget: the backend
    // resolves the fallback column from the model's own defaults.
    expect(body).not.toHaveProperty("max_tokens");
  });

  it("creates the conversation with the model's defaults when untouched", async () => {
    renderPage();
    await screen.findByTitle("Qwen3 0.6B");
    openSettings();
    await waitFor(() => expect(temperatureSlider().value).toBe("0.6"));

    fireEvent.click(screen.getByText("send-question"));

    await waitFor(() => expect(tracedFetchMock).toHaveBeenCalled());
    const [, opts] = tracedFetchMock.mock.calls.find(([url]) =>
      String(url).endsWith("/conversations/")
    );
    const body = JSON.parse(opts.body);
    expect(body.llm_id).toBe(7);
    expect([body.temperature, body.top_p]).toEqual([0.6, 0.95]);
  });
});
