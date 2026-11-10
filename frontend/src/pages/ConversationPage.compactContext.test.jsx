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
// The history list, reduced to its names and a trigger for deleting another
// conversation.
vi.mock("../components/ChatCollapsibleSection", () => ({
  default: ({ items, onDelete }) => (
    <div>
      <ul aria-label="history">
        {items.map((c) => (
          <li key={c.id}>{c.name}</li>
        ))}
      </ul>
      <button type="button" onClick={() => onDelete(99)}>
        delete-other-conversation
      </button>
    </div>
  ),
}));
vi.mock("../components/TypingIndicator", () => ({ default: () => null }));
vi.mock("../components/MarkdownRenderer", () => ({
  default: ({ content }) => <div>{content}</div>,
}));
vi.mock("../components/modals/EngineFailureModal", () => ({ default: () => null }));

import ConversationPage from "./ConversationPage.jsx";
import { resetCompacting } from "../shared/compactingConversations";
import { GENERIC_ERROR_LINE } from "../utils/messageContent";

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
// What GET /conversations/ answers (the sidebar's list).
let conversationList;

const routeFetch = async (url, opts = {}) => {
  const u = String(url);
  if (u.endsWith("/compact")) return compactAnswer(opts);
  if (u.includes("generate_title")) return emptyStream();
  if (u.includes("/query")) return answerStream();
  if (u.endsWith("/llms/local")) return { ok: true, json: async () => MODELS };
  if (u.endsWith("/fetch_messages")) return { ok: true, json: async () => messages };
  if (/\/conversations\/\d+$/.test(u)) return { ok: true, json: async () => detail };
  if (u.endsWith("/conversations/")) return { ok: true, json: async () => conversationList };
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
  conversationList = [];
  navigateMock.mockReset();
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
  // A test that leaves a run unfinished never locks a conversation for the next.
  resetCompacting();
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

  it("posts once confirmed, through tracedFetch with its own ceiling, not the 30 s one", async () => {
    await renderPage();
    await confirmCompact();
    await waitFor(() => expect(compactCalls()).toHaveLength(1));
    const [url, opts] = compactCalls()[0];
    expect(String(url)).toMatch(/\/conversations\/7\/compact$/);
    expect(opts.method).toBe("POST");
    // Its own signal (the 10-minute ceiling, see below), not the API client's.
    expect(opts.signal).toBeInstanceOf(AbortSignal);
    expect(opts.signal.aborted).toBe(false);
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
      "the server failed in a way it does not map",
      () => appError(500, "MODEL_LOADING_ERROR", "The model could not be loaded."),
      "The summary failed: The model could not be loaded.",
    ],
    [
      "the server answered without an error body",
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

  it("keeps the automatic one above and the /compact one below a failure that carries its Details", async () => {
    const cause = {
      t: "error_detail",
      text: "ReadTimeout: model did not answer",
      request_id: "be-9",
    };
    messages = [
      ...HISTORY.slice(0, 3),
      { id: 4, sender: "llm", content: GENERIC_ERROR_LINE, trace: [automatic, cause, manual] },
    ];
    render(<ConversationPage />);
    const details = await screen.findByRole("button", { name: "Details" });
    const above = screen.getByText(/6 earlier messages were summarized/);
    const below = screen.getByText("You summarized 2 earlier messages.");
    expect(before(above, details)).toBe(true);
    expect(before(details, below)).toBe(true);
    fireEvent.click(details);
    expect(screen.getByText("ReadTimeout: model did not answer")).toBeTruthy();
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
        expect(screen.getByRole("button", { name: "Please select a model" }).textContent).toBe(
          "Please select a model"
        )
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

// Review round (#687): races, stale answers and edge payloads.

/** A streaming answer that stays open until end(); with `started: false`
 * it sends nothing at all before then (the model has not begun answering). */
function openStream({ started = true } = {}) {
  let finish;
  const done = new Promise((resolve) => {
    finish = resolve;
  });
  let sent = !started;
  return {
    response: {
      ok: true,
      body: {
        getReader: () => ({
          read: async () => {
            if (!sent) {
              sent = true;
              return { done: false, value: encoder.encode('{"t":"answer","text":"Long"}\n') };
            }
            await done;
            return { done: true, value: undefined };
          },
        }),
      },
    },
    end: () => finish(),
  };
}

const switchTo = (rerender, conversationId) => {
  paramsMock.id = conversationId;
  rerender(<ConversationPage />);
};
const readsOf = (conversationId) =>
  tracedFetchMock.mock.calls.filter(([u]) =>
    String(u).endsWith(`/conversations/${conversationId}/fetch_messages`)
  ).length;

describe("a /compact already running", () => {
  it("sends one POST for a double click on Summarize, its status row staying until it ends", async () => {
    const gate = deferred();
    compactAnswer = async () => {
      await gate.promise;
      return jsonResponse(200, { summarized: 0 });
    };
    await renderPage();
    typeAndEnter("/compact");
    const dialog = await screen.findByRole("dialog", { name: "Summarize earlier messages?" });
    const confirm = within(dialog).getByRole("button", { name: "Summarize" });
    fireEvent.click(confirm);
    fireEvent.click(confirm);
    await screen.findByRole("status");
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(compactCalls()).toHaveLength(1);
    expect(screen.getByRole("status").textContent).toContain("Summarizing");

    gate.resolve();
    await screen.findByText("Nothing to compact yet.");
    expect(screen.queryByText(/Summarizing/)).toBeNull();
  });

  it("stays locked after leaving the page and coming back while it runs", async () => {
    const gate = deferred();
    compactAnswer = async () => {
      await gate.promise;
      messages = withSeparator();
      return jsonResponse(200, { summarized: 2, reason: "length", summary: "S." });
    };
    const first = render(<ConversationPage />);
    await screen.findByText("Lyon is lovely.");
    await waitFor(() => expect(composer().disabled).toBe(false));
    await confirmCompact();
    await screen.findByRole("status");
    first.unmount();

    render(<ConversationPage />);
    await screen.findByText("Lyon is lovely.");
    expect(screen.getByRole("status").textContent).toContain("Summarizing");
    expect(composer().disabled).toBe(true);

    gate.resolve();
    expect(await screen.findByText("You summarized 2 earlier messages.")).toBeTruthy();
    await waitFor(() => expect(screen.queryByRole("status")).toBeNull());
    expect(composer().disabled).toBe(false);
  });

  it("leaves Compact usable while an answer streams, so the backend can say it is busy", async () => {
    const stream = openStream();
    tracedFetchMock.mockImplementation(async (url, opts) =>
      String(url).includes("/query") ? stream.response : routeFetch(url, opts)
    );
    compactAnswer = async () =>
      appError(409, "ENGINE_BUSY", "The model is busy; try again in a moment.");
    await renderPage();
    typeAndEnter("A long answer please");
    await waitFor(() => expect(composer().disabled).toBe(true));
    openSettings();
    const compact = screen.getByRole("button", { name: /^Compact/ });
    expect(compact.disabled).toBe(false);
    fireEvent.click(compact);
    const dialog = await screen.findByRole("dialog", { name: "Summarize earlier messages?" });
    fireEvent.click(within(dialog).getByRole("button", { name: "Summarize" }));
    expect(await screen.findByText("The model is busy; try again in a moment.")).toBeTruthy();
    stream.end();
    await waitFor(() => expect(composer().disabled).toBe(false));
  });
});

describe("switching conversation while a command runs", () => {
  it("never brings A's /compact outcome or rows into B, and B is not locked", async () => {
    const gate = deferred();
    compactAnswer = async () => {
      await gate.promise;
      return appError(503, "COMPACTION_FAILED", "The summary could not be written.");
    };
    const { rerender } = render(<ConversationPage />);
    await screen.findByText("Lyon is lovely.");
    await waitFor(() => expect(composer().disabled).toBe(false));
    await confirmCompact();
    await screen.findByRole("status");

    const readsOfA = readsOf(7);
    messages = [{ id: 20, sender: "user", content: "B's own question" }];
    switchTo(rerender, "8");
    await screen.findByText("B's own question");
    await waitFor(() => expect(composer().disabled).toBe(false));
    expect(screen.queryByRole("status")).toBeNull();

    gate.resolve();
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(screen.queryByText(/could not be written/)).toBeNull();
    expect(screen.getByText("B's own question")).toBeTruthy();
    expect(readsOf(7)).toBe(readsOfA);
  });

  it("drops A's /context answer once B is shown", async () => {
    const gate = deferred();
    getMock.mockImplementation(async (path) => {
      if (path.endsWith("/context")) {
        await gate.promise;
        return USAGE;
      }
      return path.endsWith("/fetch_messages") ? messages : [];
    });
    const { rerender } = render(<ConversationPage />);
    await screen.findByText("Lyon is lovely.");
    await waitFor(() => expect(composer().disabled).toBe(false));
    typeAndEnter("/context");
    await waitFor(() => expect(contextReads()).toHaveLength(1));
    switchTo(rerender, "8");
    gate.resolve();
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(contextPanel()).toBeNull();
  });

  it("drops a re-read of A that lands once B is shown", async () => {
    const gate = deferred();
    compactAnswer = async () => appError(503, "COMPACTION_FAILED", "failed");
    const { rerender } = render(<ConversationPage />);
    await screen.findByText("Lyon is lovely.");
    await waitFor(() => expect(composer().disabled).toBe(false));
    // A's re-read after the failure waits; B opens meanwhile.
    tracedFetchMock.mockImplementation(async (url, opts) => {
      if (String(url).endsWith("/conversations/7/fetch_messages")) {
        await gate.promise;
      }
      return routeFetch(url, opts);
    });
    await confirmCompact();
    await waitFor(() => expect(compactCalls()).toHaveLength(1));
    const bRows = [{ id: 20, sender: "user", content: "B's own question" }];
    messages = bRows;
    switchTo(rerender, "8");
    await screen.findByText("B's own question");

    messages = HISTORY;
    gate.resolve();
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(screen.queryByText("Lyon is lovely.")).toBeNull();
    expect(screen.getByText("B's own question")).toBeTruthy();
    expect(screen.queryByText(/could not be written/)).toBeNull();
  });
});

describe("a /context answer that lands late", () => {
  it("is dropped when a message was sent meanwhile", async () => {
    const gate = deferred();
    getMock.mockImplementation(async (path) => {
      if (path.endsWith("/context")) {
        await gate.promise;
        return USAGE;
      }
      return path.endsWith("/fetch_messages") ? messages : [];
    });
    await renderPage();
    typeAndEnter("/context");
    await waitFor(() => expect(contextReads()).toHaveLength(1));
    typeAndEnter("And now?");
    gate.resolve();
    await screen.findByText("And now?");
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(contextPanel()).toBeNull();
  });
});

describe("panels and re-reads", () => {
  it("are cleared by the re-read after a successful /compact", async () => {
    compactAnswer = async () => {
      messages = withSeparator();
      return jsonResponse(200, { summarized: 2, reason: "length", summary: "S." });
    };
    await renderPage();
    typeAndEnter("/context");
    await screen.findByRole("region", { name: "Context usage" });
    await confirmCompact();
    await screen.findByText("You summarized 2 earlier messages.");
    await waitFor(() => expect(contextPanel()).toBeNull());
  });

  it("are kept when another conversation is deleted from the history", async () => {
    await renderPage();
    typeAndEnter("/context");
    await screen.findByRole("region", { name: "Context usage" });
    fireEvent.click(screen.getByRole("button", { name: "delete-other-conversation" }));
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(contextPanel()).toBeTruthy();
    expect(navigateMock).not.toHaveBeenCalled();
  });

  it("scroll the thread when one is added, never when one is closed", async () => {
    const scrollTo = vi.fn();
    Element.prototype.scrollTo = scrollTo;
    await renderPage();
    typeAndEnter("/context");
    await screen.findByRole("region", { name: "Context usage" });
    typeAndEnter("/context");
    await waitFor(() =>
      expect(screen.getAllByRole("region", { name: "Context usage" })).toHaveLength(2)
    );
    await new Promise((resolve) => setTimeout(resolve, 50));
    const calls = scrollTo.mock.calls.length;
    expect(calls).toBeGreaterThan(0);
    fireEvent.click(screen.getAllByRole("button", { name: "Close context usage" })[0]);
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(screen.getAllByRole("region", { name: "Context usage" })).toHaveLength(1);
    expect(scrollTo.mock.calls.length).toBe(calls);
  });
});

describe("a /compact refusal that changes the page", () => {
  it("reads the whole conversation again when it has no model any more", async () => {
    compactAnswer = async () => {
      detail = { ...DETAIL, llm_id: null };
      return appError(409, "STATE_CONFLICT", "no model", { reason: "no_model" });
    };
    await renderPage();
    await confirmCompact();
    await screen.findByText("This conversation has no model: pick one with Model first.");
    await waitFor(() => expect(header().textContent).toBe("Please select a model"));
  });

  it("does the same for /context", async () => {
    getMock.mockImplementation(async (path) => {
      if (path.endsWith("/context")) {
        detail = { ...DETAIL, llm_id: null };
        const error = new Error("no model");
        error.status = 409;
        error.data = { error: { type: "STATE_CONFLICT", detail: { reason: "no_model" } } };
        throw error;
      }
      return path.endsWith("/fetch_messages") ? messages : [];
    });
    await renderPage();
    typeAndEnter("/context");
    await waitFor(() => expect(header().textContent).toBe("Please select a model"));
  });

  it("leaves a conversation deleted meanwhile for the new-conversation page, with the notice", async () => {
    compactAnswer = async () =>
      appError(404, "CONVERSATION_NOT_FOUND", "Conversation with ID 7 not found");
    await renderPage();
    const reads = messageReads();
    await confirmCompact();
    await waitFor(() =>
      expect(navigateMock).toHaveBeenCalledWith("/erudi/chat", {
        state: { notice: { tone: "error", text: "This conversation no longer exists." } },
      })
    );
    expect(messageReads()).toBe(reads);
  });
});

describe("a /compact that never answers", () => {
  it("gives up after ten minutes, says the summary did not come back, and re-reads", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"], shouldAdvanceTime: true });
    try {
      compactAnswer = (opts) =>
        new Promise((_resolve, reject) => {
          opts.signal.addEventListener("abort", () =>
            reject(new DOMException("The operation was aborted.", "AbortError"))
          );
        });
      await renderPage();
      const reads = messageReads();
      await confirmCompact();
      await screen.findByRole("status");
      vi.advanceTimersByTime(9 * 60 * 1000);
      await new Promise((resolve) => setTimeout(resolve, 20));
      expect(screen.getByRole("status")).toBeTruthy();
      vi.advanceTimersByTime(60 * 1000);
      expect(
        await screen.findByText(
          "The summary did not come back; the conversation shows where it stands."
        )
      ).toBeTruthy();
      await waitFor(() => expect(messageReads()).toBeGreaterThan(reads));
      expect(screen.queryByRole("status")).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("an answer that ends after a switch to another conversation", () => {
  const streamingOnA = async () => {
    const stream = openStream();
    tracedFetchMock.mockImplementation(async (url, opts) =>
      String(url).includes("/query") ? stream.response : routeFetch(url, opts)
    );
    const view = render(<ConversationPage />);
    await screen.findByText("Lyon is lovely.");
    await waitFor(() => expect(composer().disabled).toBe(false));
    return { stream, ...view };
  };

  it("refreshes the history list and loads the conversation now shown", async () => {
    const { stream, rerender } = await streamingOnA();
    typeAndEnter("A long answer please");
    await waitFor(() => expect(composer().disabled).toBe(true));

    switchTo(rerender, "8");
    conversationList = [
      { id: 8, name: "B conversation", last_message_time: "2026-11-05T10:00:00Z" },
      { id: 7, name: "Trip Title", last_message_time: "2026-11-05T11:00:00Z" },
    ];
    messages = [{ id: 20, sender: "user", content: "B's own question" }];
    stream.end();

    await screen.findByText("B's own question");
    expect(screen.queryByText("Lyon is lovely.")).toBeNull();
    const history = screen.getByRole("list", { name: "history" });
    expect(within(history).getByText("Trip Title")).toBeTruthy();
    await waitFor(() => expect(composer().disabled).toBe(false));
  });

  it("never sends the user back to the first question's conversation", async () => {
    locationMock.state = { initialQuestion: "First question" };
    const stream = openStream();
    tracedFetchMock.mockImplementation(async (url, opts) =>
      String(url).includes("/query") ? stream.response : routeFetch(url, opts)
    );
    const { rerender } = render(<ConversationPage />);
    await waitFor(() =>
      expect(tracedFetchMock.mock.calls.some(([u]) => String(u).includes("/query"))).toBe(true)
    );
    switchTo(rerender, "8");
    stream.end();
    await new Promise((resolve) => setTimeout(resolve, 100));
    expect(navigateMock).not.toHaveBeenCalled();
  });

  it("consumes the first question from its own entry when the page stayed", async () => {
    locationMock.state = { initialQuestion: "First question" };
    render(<ConversationPage />);
    await waitFor(() =>
      expect(navigateMock).toHaveBeenCalledWith("/conversation/7", { replace: true, state: {} })
    );
  });
});

describe("/context while an answer streams", () => {
  it("waits, like Model, until the answer starts", async () => {
    const stream = openStream({ started: false });
    tracedFetchMock.mockImplementation(async (url, opts) =>
      String(url).includes("/query") ? stream.response : routeFetch(url, opts)
    );
    await renderPage();
    typeAndEnter("A long answer please");
    await waitFor(() => expect(composer().disabled).toBe(true));
    openSettings();
    expect(screen.getByRole("button", { name: /^Context/ }).disabled).toBe(true);
    stream.end();
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /^Context/ }).disabled).toBe(false)
    );
  });

  it("drops a snapshot taken mid-answer once the turn ends", async () => {
    const stream = openStream();
    tracedFetchMock.mockImplementation(async (url, opts) =>
      String(url).includes("/query") ? stream.response : routeFetch(url, opts)
    );
    await renderPage();
    typeAndEnter("A long answer please");
    await screen.findByText("Long");
    openSettings();
    fireEvent.click(screen.getByRole("button", { name: /^Context/ }));
    await screen.findByRole("region", { name: "Context usage" });
    stream.end();
    await waitFor(() => expect(contextPanel()).toBeNull());
  });
});

describe("the re-read after /compact", () => {
  it("is waited for at most a minute: the conversation is unlocked either way", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"], shouldAdvanceTime: true });
    try {
      compactAnswer = async () => {
        // The re-read that follows never answers.
        tracedFetchMock.mockImplementation(async (url, opts) =>
          String(url).endsWith("/fetch_messages") ? new Promise(() => {}) : routeFetch(url, opts)
        );
        return appError(503, "COMPACTION_FAILED", "The summary could not be written.");
      };
      await renderPage();
      await confirmCompact();
      await screen.findByRole("status");
      vi.advanceTimersByTime(59 * 1000);
      await new Promise((resolve) => setTimeout(resolve, 20));
      expect(screen.getByRole("status")).toBeTruthy();
      vi.advanceTimersByTime(1000);
      await waitFor(() => expect(screen.queryByRole("status")).toBeNull());
      expect(
        screen.getByText("The summary could not be written; nothing was changed.")
      ).toBeTruthy();
      expect(composer().disabled).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });
});
