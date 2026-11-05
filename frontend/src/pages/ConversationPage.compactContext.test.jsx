// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, within } from "@testing-library/react";

// /compact and /context on a conversation (#585, #608). /compact asks first,
// then summarizes through POST /compact with no client timeout: a status row
// at the end of the thread while it runs (sending, the header buttons and the
// commands disabled), the conversation re-read so its separator shows below
// the last answer, a notice for each refusal or failure. /context reads GET
// /context and draws a panel in the thread, kept apart from the messages and
// gone at the next re-read. The real header, composer and popups render; only
// the network is faked.

const { tracedFetchMock, navigateMock, locationMock, getMock, patchMock } = vi.hoisted(() => ({
  tracedFetchMock: vi.fn(),
  navigateMock: vi.fn(),
  locationMock: { pathname: "/conversation/7", state: null },
  getMock: vi.fn(),
  patchMock: vi.fn(),
}));

const paramsMock = { id: "7" };

vi.mock("../services/api/client", () => ({
  default: { get: getMock, patch: patchMock, put: vi.fn() },
  tracedFetch: tracedFetchMock,
}));

vi.mock("react-router-dom", () => ({
  useParams: () => paramsMock,
  useNavigate: () => navigateMock,
  useLocation: () => locationMock,
}));

vi.mock("../components/Sidebar", () => ({ default: () => null }));
vi.mock("../components/ChatCollapsibleSection", () => ({ default: () => null }));
vi.mock("../components/TypingIndicator", () => ({ default: () => null }));
vi.mock("../components/MarkdownRenderer", () => ({
  default: ({ content }) => <div>{content}</div>,
}));
vi.mock("../components/modals/EngineFailureModal", () => ({ default: () => null }));

import ConversationPage from "./ConversationPage.jsx";

const MODELS = [
  { id: 1, name: "Qwen3-4B", supports_tools: true, supports_tools_wire: true },
  { id: 3, name: "Orphan assistant", weights_available: false },
];

const DETAIL = {
  id: 7,
  llm_id: 1,
  reasoning_effort: "medium",
  temperature: 0.7,
  top_p: 0.8,
  max_tokens: 512,
  compaction_threshold: 80,
  web_search_enabled: false,
  custom_prompt: "",
};

const automatic = {
  t: "compaction",
  phase: "done",
  summarized: 6,
  reason: "length",
  summary: "Automatic summary.",
};
const manual = {
  t: "compaction",
  phase: "done",
  manual: true,
  summarized: 2,
  reason: "length",
  summary: "Ada lives in Lyon.",
};

const HISTORY = [
  { id: 1, sender: "user", content: "My name is Ada." },
  { id: 2, sender: "llm", content: "Hello Ada.", trace: null },
  { id: 3, sender: "user", content: "I live in Lyon." },
  { id: 4, sender: "llm", content: "Lyon is lovely.", trace: null },
];
const withSeparator = () => [...HISTORY.slice(0, 3), { ...HISTORY[3], trace: [manual] }];

const USAGE = {
  categories: {
    system_prompt: 412,
    tools: 0,
    messages: 2310,
    summary: 0,
    free: 3831,
    autocompaction_buffer: 1639,
  },
  window: {
    working: 8192,
    allocated: null,
    trained: 8192,
    reduced_by_memory: false,
    model_loaded: false,
  },
  kb_excerpts_excluded: false,
  over_window: false,
};

const encoder = new TextEncoder();
const answerStream = () => {
  const queue = [
    { done: false, value: encoder.encode('{"t":"answer","text":"ok"}\n{"t":"done"}\n') },
    { done: true, value: undefined },
  ];
  return { ok: true, body: { getReader: () => ({ read: async () => queue.shift() }) } };
};
const emptyStream = () => ({
  ok: true,
  body: { getReader: () => ({ read: async () => ({ done: true, value: undefined }) }) },
});
const deferred = () => {
  let resolve;
  const promise = new Promise((res) => {
    resolve = res;
  });
  return { promise, resolve };
};
const jsonResponse = (status, body) => ({
  ok: status >= 200 && status < 300,
  status,
  json: async () => body,
});
const appError = (status, type, message, detail) =>
  jsonResponse(status, {
    success: false,
    error: { type, message, ...(detail ? { detail } : {}) },
  });

let detail;
let messages;
// What POST /compact answers (a function of nothing, so a test can gate it).
let compactAnswer;

const routeFetch = async (url) => {
  const u = String(url);
  if (u.endsWith("/compact")) return compactAnswer();
  if (u.includes("generate_title")) return emptyStream();
  if (u.includes("/query")) return answerStream();
  if (u.endsWith("/llms/local")) return { ok: true, json: async () => MODELS };
  if (u.endsWith("/fetch_messages")) return { ok: true, json: async () => messages };
  if (/\/conversations\/\d+$/.test(u)) return { ok: true, json: async () => detail };
  return { ok: true, json: async () => [] };
};

const composer = () => document.querySelector("textarea");
const typeAndEnter = (value) => {
  fireEvent.change(composer(), { target: { value } });
  fireEvent.keyDown(composer(), { key: "Enter" });
};
const header = () => screen.getByRole("heading", { level: 3 });
const compactCalls = () =>
  tracedFetchMock.mock.calls.filter(([url]) => String(url).endsWith("/compact"));
const messageReads = () =>
  tracedFetchMock.mock.calls.filter(([url]) => String(url).endsWith("/fetch_messages")).length;
const contextReads = () => getMock.mock.calls.filter(([path]) => path.endsWith("/context"));
const openSettings = () =>
  fireEvent.click(screen.getByRole("button", { name: "Conversation settings" }));
const contextPanel = () => screen.queryByRole("region", { name: "Context usage" });
/** Whether `a` comes before `b` in the document. */
const before = (a, b) => Boolean(a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING);

const renderPage = async () => {
  render(<ConversationPage />);
  await waitFor(() => expect(header().textContent).toContain("Qwen3-4B"));
  await screen.findByText("Lyon is lovely.");
  await waitFor(() => expect(composer().disabled).toBe(false));
};

const confirmCompact = async () => {
  typeAndEnter("/compact");
  const dialog = await screen.findByRole("dialog", { name: "Summarize earlier messages?" });
  fireEvent.click(within(dialog).getByRole("button", { name: "Summarize" }));
  await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
};

beforeEach(() => {
  Element.prototype.scrollTo = () => {};
  detail = { ...DETAIL };
  messages = HISTORY;
  compactAnswer = async () => jsonResponse(200, { summarized: 0 });
  locationMock.state = null;
  paramsMock.id = "7";
  getMock.mockReset();
  getMock.mockImplementation(async (path) => {
    if (path.endsWith("/fetch_messages")) return messages;
    if (path.endsWith("/context")) return USAGE;
    return [];
  });
  tracedFetchMock.mockReset();
  tracedFetchMock.mockImplementation(routeFetch);
  patchMock.mockReset();
  patchMock.mockImplementation(async (_path, body) => ({ ...detail, ...body }));
});

afterEach(() => {
  cleanup();
});

describe("/compact asks first", () => {
  it("explains what it does, and dismissing it compacts nothing", async () => {
    await renderPage();
    typeAndEnter("/compact");
    const dialog = await screen.findByRole("dialog", { name: "Summarize earlier messages?" });
    expect(within(dialog).getByText(/replaced, for the model, by a summary/)).toBeTruthy();
    fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(compactCalls()).toHaveLength(0);
    expect(tracedFetchMock.mock.calls.some(([u]) => String(u).includes("/query"))).toBe(false);
  });

  it("opens from the header's Compact button too", async () => {
    await renderPage();
    openSettings();
    fireEvent.click(screen.getByRole("button", { name: /^Compact/ }));
    expect(await screen.findByRole("dialog", { name: "Summarize earlier messages?" })).toBeTruthy();
  });

  it("posts once confirmed, with no client-side timeout", async () => {
    await renderPage();
    await confirmCompact();
    await waitFor(() => expect(compactCalls()).toHaveLength(1));
    const [url, opts] = compactCalls()[0];
    expect(String(url)).toMatch(/\/conversations\/7\/compact$/);
    expect(opts.method).toBe("POST");
    expect(opts.signal).toBeUndefined();
  });
});

describe("while /compact runs", () => {
  it("shows a status row at the end of the thread and disables sending, the header buttons and the commands", async () => {
    const gate = deferred();
    compactAnswer = async () => {
      await gate.promise;
      return jsonResponse(200, { summarized: 2, reason: "length", summary: "S." });
    };
    await renderPage();
    await confirmCompact();

    const status = await screen.findByRole("status");
    expect(status.textContent).toBe("Summarizing earlier messages to make room…");
    expect(before(screen.getByText("Lyon is lovely."), status)).toBe(true);
    expect(composer().disabled).toBe(true);
    openSettings();
    for (const label of ["Model", "Config", "Effort", "Compact", "Context"]) {
      expect(screen.getByRole("button", { name: new RegExp(`^${label}`) }).disabled).toBe(true);
    }

    messages = withSeparator();
    gate.resolve();
    await waitFor(() => expect(screen.queryByRole("status")).toBeNull());
    expect(composer().disabled).toBe(false);
    expect(screen.getByRole("button", { name: /^Compact/ }).disabled).toBe(false);
  });
});

describe("what /compact answers", () => {
  it("re-reads the conversation on success, the separator showing below the last answer", async () => {
    compactAnswer = async () => {
      messages = withSeparator();
      return jsonResponse(200, { summarized: 2, reason: "length", summary: "Ada lives in Lyon." });
    };
    await renderPage();
    const reads = messageReads();
    await confirmCompact();

    const separator = await screen.findByText("You summarized 2 earlier messages.");
    expect(messageReads()).toBeGreaterThan(reads);
    expect(before(screen.getByText("Lyon is lovely."), separator)).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Show summary" }));
    expect(screen.getByText("Ada lives in Lyon.")).toBeTruthy();
  });

  it("says there is nothing to compact yet, and re-reads nothing", async () => {
    await renderPage();
    const reads = messageReads();
    await confirmCompact();
    expect(await screen.findByText("Nothing to compact yet.")).toBeTruthy();
    expect(messageReads()).toBe(reads);
  });

  it.each([
    [
      "the model is busy",
      () => appError(409, "ENGINE_BUSY", "The model is busy; try again in a moment."),
      "The model is busy; try again in a moment.",
    ],
    [
      "the conversation has no model",
      () => appError(409, "STATE_CONFLICT", "no model", { reason: "no_model" }),
      "This conversation has no model: pick one with Model first.",
    ],
    [
      "the last question has no answer",
      () => appError(409, "STATE_CONFLICT", "unanswered", { reason: "unanswered_question" }),
      "The last question has no answer yet; send it again or ask something new first.",
    ],
    [
      "the conversation was deleted",
      () => appError(404, "CONVERSATION_NOT_FOUND", "Conversation with ID 7 not found"),
      "This conversation no longer exists.",
    ],
    [
      "the summary could not be written",
      () => appError(503, "COMPACTION_FAILED", "The summary could not be written."),
      "The summary could not be written; nothing was changed.",
    ],
    [
      "the separator could not be saved",
      () => appError(500, "COMPACTION_SEPARATOR_NOT_SAVED", "separator"),
      "The earlier messages were summarized, but the separator could not be saved.",
    ],
    [
      "the answer is not one it knows",
      () => jsonResponse(502, null),
      "The summary did not come back; the conversation shows where it stands.",
    ],
    [
      "the request fails",
      async () => {
        throw new TypeError("Failed to fetch");
      },
      "The summary did not come back; the conversation shows where it stands.",
    ],
  ])("says so when %s, and re-reads the conversation", async (_label, answer, copy) => {
    compactAnswer = answer;
    await renderPage();
    const reads = messageReads();
    await confirmCompact();
    const notice = await screen.findByText(copy);
    expect(notice.closest('[role="alert"]')).toBeTruthy();
    await waitFor(() => expect(messageReads()).toBeGreaterThan(reads));
    expect(screen.queryByRole("status")).toBeNull();
    expect(composer().disabled).toBe(false);
  });

  it("shows a separator the server saved before its answer was lost", async () => {
    compactAnswer = async () => {
      messages = withSeparator();
      throw new TypeError("Failed to fetch");
    };
    await renderPage();
    await confirmCompact();
    expect(await screen.findByText("You summarized 2 earlier messages.")).toBeTruthy();
  });

  it("clears its notice at the next command", async () => {
    await renderPage();
    await confirmCompact();
    await screen.findByText("Nothing to compact yet.");
    typeAndEnter("/config");
    await screen.findByRole("dialog");
    expect(screen.queryByText("Nothing to compact yet.")).toBeNull();
  });
});

describe("the separators of a reloaded conversation", () => {
  it("draws a /compact separator below its answer and an automatic one above", async () => {
    messages = [...HISTORY.slice(0, 3), { ...HISTORY[3], trace: [automatic, manual] }];
    await renderPage();
    const answer = screen.getByText("Lyon is lovely.");
    const above = screen.getByText(/6 earlier messages were summarized/);
    const below = screen.getByText("You summarized 2 earlier messages.");
    expect(before(above, answer)).toBe(true);
    expect(before(answer, below)).toBe(true);
  });

  it("draws a /compact separator below a failed turn", async () => {
    messages = [
      ...HISTORY.slice(0, 3),
      { id: 4, sender: "llm", content: "[ERROR_MESSAGE_SYSTEM] Timed out.", trace: [manual] },
    ];
    render(<ConversationPage />);
    const answer = await screen.findByText(/Timed out\./);
    expect(before(answer, screen.getByText("You summarized 2 earlier messages."))).toBe(true);
  });
});

describe("/context", () => {
  it("draws its panel in the thread, below the last answer, from GET /context", async () => {
    await renderPage();
    typeAndEnter("/context");
    const panel = await screen.findByRole("region", { name: "Context usage" });
    expect(contextReads()).toHaveLength(1);
    expect(contextReads()[0][0]).toBe("/conversations/7/context");
    expect(before(screen.getByText("Lyon is lovely."), panel)).toBe(true);
    expect(within(panel).getByRole("img", { name: /Context usage/ })).toBeTruthy();
    expect(within(panel).getByText(/The model is not loaded/)).toBeTruthy();
    // Never a message: no new bubble, no copy or star action of its own.
    expect(screen.getAllByTitle("Copy message")).toHaveLength(HISTORY.length);
    expect(compactCalls()).toHaveLength(0);
    expect(tracedFetchMock.mock.calls.some(([u]) => String(u).includes("/query"))).toBe(false);
  });

  it("opens from the header's Context button too", async () => {
    await renderPage();
    openSettings();
    fireEvent.click(screen.getByRole("button", { name: /^Context/ }));
    expect(await screen.findByRole("region", { name: "Context usage" })).toBeTruthy();
  });

  it("is gone once the conversation is re-read after the next turn", async () => {
    await renderPage();
    typeAndEnter("/context");
    await screen.findByRole("region", { name: "Context usage" });
    typeAndEnter("And now?");
    await waitFor(() => expect(contextPanel()).toBeNull());
    expect(screen.getByText("And now?")).toBeTruthy();
  });

  it("is gone when another conversation opens", async () => {
    const { rerender } = render(<ConversationPage />);
    await screen.findByText("Lyon is lovely.");
    await waitFor(() => expect(composer().disabled).toBe(false));
    typeAndEnter("/context");
    await screen.findByRole("region", { name: "Context usage" });
    paramsMock.id = "8";
    rerender(<ConversationPage />);
    await waitFor(() => expect(contextPanel()).toBeNull());
  });

  it("closes from its own button", async () => {
    await renderPage();
    typeAndEnter("/context");
    const panel = await screen.findByRole("region", { name: "Context usage" });
    fireEvent.click(within(panel).getByRole("button", { name: "Close context usage" }));
    expect(contextPanel()).toBeNull();
  });

  it("says to pick a model first when the backend refuses for that", async () => {
    getMock.mockImplementation(async (path) => {
      if (path.endsWith("/context")) {
        const error = new Error("This conversation has no model; pick a model first.");
        error.status = 409;
        error.data = {
          success: false,
          error: { type: "STATE_CONFLICT", message: error.message, detail: { reason: "no_model" } },
        };
        throw error;
      }
      return path.endsWith("/fetch_messages") ? messages : [];
    });
    await renderPage();
    typeAndEnter("/context");
    expect(
      await screen.findByText("This conversation has no model: pick one with Model first.")
    ).toBeTruthy();
    expect(contextPanel()).toBeNull();
  });

  it("says the usage could not be read on any other failure", async () => {
    getMock.mockImplementation(async (path) => {
      if (path.endsWith("/context")) throw new Error("Request timeout");
      return path.endsWith("/fetch_messages") ? messages : [];
    });
    await renderPage();
    typeAndEnter("/context");
    expect(await screen.findByText("The context usage could not be read.")).toBeTruthy();
  });
});

describe("an orphaned conversation", () => {
  it.each([
    ["deleted", null],
    ["an assistant whose weights are missing", 3],
  ])(
    "says to pick a model first for /compact and /context when its model is %s, asking nothing",
    async (_label, llmId) => {
      detail = { ...DETAIL, llm_id: llmId };
      render(<ConversationPage />);
      await waitFor(() =>
        expect(screen.getByRole("alert").textContent).toBe("Please select a model")
      );
      openSettings();
      fireEvent.click(screen.getByRole("button", { name: /^Context/ }));
      await screen.findByText("This conversation has no model: pick one with Model first.");
      fireEvent.click(screen.getByRole("button", { name: /^Compact/ }));
      expect(screen.queryByRole("dialog")).toBeNull();
      expect(
        screen.getByText("This conversation has no model: pick one with Model first.")
      ).toBeTruthy();
      expect(contextReads()).toHaveLength(0);
      expect(compactCalls()).toHaveLength(0);
      expect(patchMock).not.toHaveBeenCalled();
    }
  );
});
