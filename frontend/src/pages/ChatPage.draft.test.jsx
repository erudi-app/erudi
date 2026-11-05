// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";

// The new-conversation page (#585, #609): the same header and commands as a
// conversation, editing a draft. The picked model's defaults are resolved by
// the backend; the creation sends only the values the user set, and the first
// turn reads the conversation as created (nothing rides in the router state).

const { tracedFetchMock, navigateMock, getMock, putMock } = vi.hoisted(() => ({
  tracedFetchMock: vi.fn(),
  navigateMock: vi.fn(),
  getMock: vi.fn(),
  putMock: vi.fn(),
}));

vi.mock("../services/api/client", () => ({
  default: { get: getMock, put: putMock, patch: vi.fn() },
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

import ChatPage from "./ChatPage.jsx";

// Both run tools (web search can be switched on for them).
const TOOLS = { supports_tools: true, supports_tools_wire: true };
const MODELS = [
  { id: 7, name: "Alpha Model", sampling_defaults: { temperature: 0.7, top_p: 0.8 }, ...TOOLS },
  { id: 42, name: "Beta Model", sampling_defaults: { temperature: 0.3, top_p: 0.9 }, ...TOOLS },
];

const resolved = (overrides = {}) => ({
  llm_id: 7,
  reasoning_effort: { value: "medium", source: "settings" },
  temperature: { value: 0.7, source: "model" },
  top_p: { value: 0.8, source: "model" },
  compaction_threshold: { value: 80, source: "settings" },
  web_search_enabled: { value: false, source: "settings" },
  custom_prompt: { value: "", source: "fallback" },
  ...overrides,
});

let defaultsByModel;
let models;

const composer = () => document.querySelector("textarea");
const typeAndEnter = (value) => {
  fireEvent.change(composer(), { target: { value } });
  fireEvent.keyDown(composer(), { key: "Enter" });
};
const header = () => screen.getByRole("heading", { level: 3 });
const createCalls = () => tracedFetchMock.mock.calls.filter(([, opts]) => opts?.method === "POST");
const createBody = () => JSON.parse(createCalls()[0][1].body);

const renderPage = async () => {
  render(
    <MemoryRouter initialEntries={["/erudi/chat"]}>
      <ChatPage />
    </MemoryRouter>
  );
  await waitFor(() => expect(header().textContent).toContain("Alpha Model"));
  await waitFor(() => expect(getMock).toHaveBeenCalledWith("/llms/7/conversation_defaults"));
};

const openModelPopupAndPick = async (name) => {
  fireEvent.click(screen.getByRole("button", { name: "Conversation settings" }));
  fireEvent.click(screen.getByRole("button", { name: /^Model/ }));
  const dialog = await screen.findByRole("dialog");
  fireEvent.click(within(dialog).getByRole("button", { name: new RegExp(name) }));
};

beforeEach(() => {
  Element.prototype.scrollTo = () => {};
  models = MODELS;
  defaultsByModel = { 7: resolved(), 42: resolved({ llm_id: 42 }) };
  navigateMock.mockReset();
  tracedFetchMock.mockReset();
  tracedFetchMock.mockResolvedValue({ ok: true, json: async () => ({ id: 123 }) });
  putMock.mockReset();
  getMock.mockReset();
  getMock.mockImplementation(async (path) => {
    if (path === "/llms/local") return models;
    const match = path.match(/^\/llms\/(\d+)\/conversation_defaults$/);
    if (match) return defaultsByModel[match[1]];
    return [];
  });
});

afterEach(cleanup);

describe("the new-conversation page", () => {
  it("carries the header naming the picked model, and no inline settings block", async () => {
    await renderPage();
    expect(header().textContent).toBe("You are chatting with Alpha Model");
    expect(screen.queryByRole("slider")).toBeNull();
    expect(screen.queryByText("Customize Prompt")).toBeNull();
  });

  it("creates the conversation with the model alone when nothing was set", async () => {
    await renderPage();
    typeAndEnter("What is the sun?");
    await waitFor(() => expect(navigateMock).toHaveBeenCalledTimes(1));
    expect(createBody()).toEqual({ llm_id: 7 });
    const [path, { state }] = navigateMock.mock.calls[0];
    expect(path).toContain("123");
    expect(state).toEqual({
      initialQuestion: "What is the sun?",
      initialImages: [],
      initialImagePaths: [],
      initialAttachments: [],
    });
  });

  it("creates the conversation with the values set through /effort and /config", async () => {
    await renderPage();
    typeAndEnter("/effort");
    let dialog = await screen.findByRole("dialog");
    fireEvent.change(within(dialog).getByRole("slider", { name: "Reasoning effort" }), {
      target: { value: "3" },
    });
    fireEvent.keyDown(dialog, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());

    typeAndEnter("/config");
    dialog = await screen.findByRole("dialog");
    fireEvent.change(within(dialog).getByRole("textbox", { name: "Custom instructions" }), {
      target: { value: "Answer like a pirate" },
    });
    fireEvent.click(within(dialog).getByRole("button", { name: "Save instructions" }));
    fireEvent.keyDown(dialog, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());

    typeAndEnter("Ahoy?");
    await waitFor(() => expect(createCalls()).toHaveLength(1));
    expect(createBody()).toEqual({
      llm_id: 7,
      reasoning_effort: "high",
      custom_prompt: "Answer like a pirate",
    });
  });

  it("on a model switch, the new model brings its values and a set switch stays", async () => {
    defaultsByModel[42] = resolved({
      llm_id: 42,
      reasoning_effort: { value: "low", source: "saved" },
      temperature: { value: 0.3, source: "model" },
    });
    await renderPage();
    typeAndEnter("/config");
    const dialog = await screen.findByRole("dialog");
    fireEvent.click(within(dialog).getByRole("switch", { name: "Web search" }));
    fireEvent.keyDown(dialog, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());

    await openModelPopupAndPick("Beta Model");
    await waitFor(() => expect(header().textContent).toBe("You are chatting with Beta Model"));
    await waitFor(() => expect(getMock).toHaveBeenCalledWith("/llms/42/conversation_defaults"));

    typeAndEnter("/config");
    const reopened = await screen.findByRole("dialog");
    expect(within(reopened).getByRole("slider", { name: "Creativity" }).value).toBe("0.3");
    expect(
      within(reopened)
        .getByRole("slider", { name: "Reasoning effort" })
        .getAttribute("aria-valuetext")
    ).toBe("Low");
    fireEvent.keyDown(reopened, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());

    typeAndEnter("Hi");
    await waitFor(() => expect(createCalls()).toHaveLength(1));
    expect(createBody()).toEqual({ llm_id: 42, web_search_enabled: true });
  });

  it("a send right after a model switch waits for the new model's defaults", async () => {
    let releaseBeta;
    defaultsByModel[42] = new Promise((resolve) => {
      releaseBeta = () => resolve(resolved({ llm_id: 42 }));
    });
    await renderPage();
    typeAndEnter("/config");
    const dialog = await screen.findByRole("dialog");
    fireEvent.change(within(dialog).getByRole("slider", { name: "Creativity" }), {
      target: { value: "1.5" },
    });
    fireEvent.keyDown(dialog, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());

    await openModelPopupAndPick("Beta Model");
    await waitFor(() => expect(header().textContent).toBe("You are chatting with Beta Model"));
    typeAndEnter("Now");
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(createCalls()).toHaveLength(0);

    releaseBeta();
    await waitFor(() => expect(createCalls()).toHaveLength(1));
    // Alpha's creativity is never sent for Beta: Beta's own applies.
    expect(createBody()).toEqual({ llm_id: 42 });
  });

  it("/config global saves the draft values for the picked model", async () => {
    putMock.mockResolvedValue(resolved());
    await renderPage();
    typeAndEnter("/config global");
    await screen.findByText("Saved as defaults for Alpha Model.");
    expect(putMock).toHaveBeenCalledWith("/llms/7/conversation_defaults", {
      reasoning_effort: "medium",
      temperature: 0.7,
      top_p: 0.8,
      compaction_threshold: 80,
      web_search_enabled: false,
      custom_prompt: "",
    });
    expect(createCalls()).toHaveLength(0);
  });
});

describe("a model whose weights are missing (#376)", () => {
  const ORPHAN = { id: 9, name: "Orphan assistant", weights_available: false };

  it("is never offered by /model nor auto-selected", async () => {
    models = [ORPHAN, ...MODELS];
    await renderPage();
    typeAndEnter("/model");
    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).queryByText("Orphan assistant")).toBeNull();
    expect(within(dialog).getByRole("button", { name: /Alpha Model/ })).toBeTruthy();
  });

  it("leaves nothing selected and the composer disabled when it is the only model", async () => {
    models = [ORPHAN];
    render(
      <MemoryRouter initialEntries={["/erudi/chat"]}>
        <ChatPage />
      </MemoryRouter>
    );
    await waitFor(() =>
      expect(screen.getByRole("alert").textContent).toBe("Please select a model")
    );
    expect(composer().disabled).toBe(true);
  });
});
