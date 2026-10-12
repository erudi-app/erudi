// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, act } from "@testing-library/react";

// A compaction is visible in the thread (#614):
//  - `compaction` `start` shows a transient state in the pending answer (the
//    slot the reload state uses) until the model speaks (answer, thinking,
//    tool), an error, or the end of the turn -- `done` does not end it, the
//    model still re-reads the shortened conversation;
//  - `compaction` `done` lands in the turn's trace and renders a separator
//    above the assistant bubble, kept on an error turn and after a reload;
//  - a trace holding only compactions mounts no reasoning strip.

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

// TraceStrip and CompactionMarker are the real components.
vi.mock("../components/Sidebar", () => ({ default: () => null }));
vi.mock("../components/ChatCollapsibleSection", () => ({ default: () => null }));
vi.mock("../components/HeaderBar", () => ({ default: () => null }));
vi.mock("../components/TypingIndicator", () => ({
  default: () => <span data-testid="typing" />,
}));
vi.mock("../components/modals/CustomizePromptModal", () => ({ default: () => null }));
vi.mock("../components/MarkdownRenderer", () => ({
  default: ({ content }) => <div data-testid="answer">{content}</div>,
}));
vi.mock("../components/QuestionInput", () => ({
  default: ({ onSend }) => <button onClick={() => onSend("hi", [], [])}>SEND</button>,
}));

import ConversationPage from "./ConversationPage.jsx";
import apiClient from "../services/api/client";

const conversationDetail = {
  id: 7,
  llm_id: 1,
  temperature: 0.7,
  top_p: 0.9,
  max_tokens: 512,
  custom_prompt: "",
};

/** A streaming Response whose body reader is fed chunks on demand. */
function makeControlledStream() {
  const enc = new TextEncoder();
  const queue = [];
  let waiting = null;
  let ended = false;
  const push = (str) => {
    const chunk = { done: false, value: enc.encode(str) };
    if (waiting) {
      const resolve = waiting;
      waiting = null;
      resolve(chunk);
    } else {
      queue.push(chunk);
    }
  };
  const end = () => {
    ended = true;
    if (waiting) {
      const resolve = waiting;
      waiting = null;
      resolve({ done: true, value: undefined });
    }
  };
  const read = () =>
    new Promise((resolve) => {
      if (queue.length) {
        resolve(queue.shift());
      } else if (ended) {
        resolve({ done: true, value: undefined });
      } else {
        waiting = resolve;
      }
    });
  return { response: { ok: true, body: { getReader: () => ({ read }) } }, push, end };
}

const doneStream = () => ({
  ok: true,
  body: { getReader: () => ({ read: async () => ({ done: true, value: undefined }) }) },
});

const makeRoute =
  ({ stream, messages = [] } = {}) =>
  async (url) => {
    const u = String(url);
    if (u.includes("/query")) return stream.response;
    if (u.includes("generate_title")) return doneStream();
    if (u.endsWith("/conversations/7")) return { ok: true, json: async () => conversationDetail };
    if (u.includes("fetch_messages")) return { ok: true, json: async () => messages };
    return { ok: true, json: async () => [] };
  };

const renderAndSettle = async () => {
  render(<ConversationPage />);
  await waitFor(() => expect(apiClient.get).toHaveBeenCalled());
  await act(async () => {});
  await screen.findByText("SEND");
};

const push = async (stream, line) => {
  await act(async () => {
    stream.push(line);
  });
};

const runTurn = async (stream, lines) => {
  await act(async () => {
    for (const line of lines) {
      stream.push(line);
    }
    stream.end();
  });
};

const COMPACTING = "Summarizing earlier messages to make room…";
const RELOAD = "Memory was freed on your computer; reloading the model to resume the conversation…";
const START = '{"t":"compaction","phase":"start"}\n';
const doneLine = (fields = {}) =>
  `${JSON.stringify({
    t: "compaction",
    phase: "done",
    summarized: 12,
    reason: "length",
    summary: "Ada lives in Lyon.",
    summary_lost: false,
    ...fields,
  })}\n`;
const LENGTH_12 =
  "12 earlier messages were summarized to keep the conversation within the model's context";

beforeEach(() => {
  vi.clearAllMocks();
  Element.prototype.scrollTo = () => {};
  apiClient.get.mockImplementation(async () => []);
});

afterEach(() => {
  cleanup();
});

describe("ConversationPage compaction: while it runs", () => {
  it("shows the transient state from start and ends it at the first answer", async () => {
    const turn = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute({ stream: turn }));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await push(turn, START);
    expect(await screen.findByText(COMPACTING)).toBeTruthy();
    // The typing indicator of the transient state, and no reasoning strip.
    expect(screen.getAllByTestId("typing").length).toBe(1);

    await push(turn, '{"t":"answer","text":"Sure."}\n');
    await waitFor(() => expect(screen.queryByText(COMPACTING)).toBeNull());
    await act(async () => {
      turn.end();
    });
  });

  it("stays through done -- the model still re-reads the conversation -- until it speaks", async () => {
    const turn = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute({ stream: turn }));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await push(turn, START);
    await push(turn, doneLine());
    expect(await screen.findByText(COMPACTING)).toBeTruthy();
    // A trace holding only the compaction mounts no reasoning strip, and the
    // pending answer keeps its indicator.
    expect(screen.queryByRole("button", { name: /Reasoning/ })).toBeNull();
    expect(screen.getAllByTestId("typing").length).toBe(1);

    await push(turn, '{"t":"thinking","text":"hm"}\n');
    await waitFor(() => expect(screen.queryByText(COMPACTING)).toBeNull());
    await act(async () => {
      turn.end();
    });
  });

  it.each([
    ["a tool call", '{"t":"tool_call","name":"calculator","args":{}}\n'],
    ["an error", '{"t":"error","text":"[ERROR_MESSAGE_SYSTEM] The model could not load."}\n'],
    ["the end of the turn", '{"t":"done"}\n'],
  ])("ends on %s", async (_label, line) => {
    const turn = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute({ stream: turn }));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await push(turn, START);
    expect(await screen.findByText(COMPACTING)).toBeTruthy();

    await push(turn, line);
    await waitFor(() => expect(screen.queryByText(COMPACTING)).toBeNull());
    await act(async () => {
      turn.end();
    });
  });

  it("takes over from the reload state", async () => {
    const turn = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute({ stream: turn }));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await push(turn, '{"t":"memory_reload"}\n');
    expect(await screen.findByText(RELOAD)).toBeTruthy();

    await push(turn, START);
    expect(await screen.findByText(COMPACTING)).toBeTruthy();
    expect(screen.queryByText(RELOAD)).toBeNull();
    await act(async () => {
      turn.end();
    });
  });

  it("never lands in the message content", async () => {
    const turn = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute({ stream: turn }));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await runTurn(turn, [START, doneLine(), '{"t":"answer","text":"Sure."}\n', '{"t":"done"}\n']);

    await waitFor(() => expect(screen.getByTestId("answer").textContent).toBe("Sure."));
    expect(screen.queryByText(COMPACTING)).toBeNull();
  });
});

describe("ConversationPage compaction: the marker", () => {
  it("renders the separator above the answer of the turn", async () => {
    const turn = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute({ stream: turn }));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await runTurn(turn, [START, doneLine(), '{"t":"answer","text":"Sure."}\n', '{"t":"done"}\n']);

    const marker = await screen.findByText(LENGTH_12);
    const answer = screen.getByTestId("answer");
    expect(marker.compareDocumentPosition(answer) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Show summary" }));
    expect(screen.getByText("Ada lives in Lyon.")).toBeTruthy();
  });

  it("stacks two compactions of one turn above the reasoning strip", async () => {
    const turn = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute({ stream: turn }));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await runTurn(turn, [
      START,
      doneLine({ summarized: 4 }),
      '{"t":"tool_call","name":"calculator","args":{"expression":"2 + 2"}}\n',
      '{"t":"tool_result","name":"calculator","text":"4"}\n',
      START,
      doneLine({ summarized: 2, reason: "memory" }),
      '{"t":"answer","text":"It is 4."}\n',
      '{"t":"done"}\n',
    ]);

    const first = await screen.findByText(
      "4 earlier messages were summarized to keep the conversation within the model's context"
    );
    const second = screen.getByText(
      "2 earlier messages were summarized to fit this computer's memory"
    );
    const strip = screen.getByRole("button", { name: /Reasoning/ });
    expect(first.compareDocumentPosition(second) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(second.compareDocumentPosition(strip) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("stays on an error turn", async () => {
    const turn = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute({ stream: turn }));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await runTurn(turn, [
      START,
      doneLine(),
      '{"t":"thinking","text":"hm"}\n',
      '{"t":"error","text":"[ERROR_MESSAGE_SYSTEM] The model failed."}\n',
      '{"t":"done"}\n',
    ]);

    expect(await screen.findByText(LENGTH_12)).toBeTruthy();
    expect(screen.getByText(/The model failed\./)).toBeTruthy();
    // The error turn shows no reasoning strip, as before.
    expect(screen.queryByRole("button", { name: /Reasoning/ })).toBeNull();
  });

  it("replays from a persisted trace after a reload", async () => {
    const persisted = [
      { id: 11, sender: "user", content: "and then?", starred: false },
      {
        id: 12,
        sender: "llm",
        content: "Then Lyon.",
        starred: false,
        trace: [
          {
            t: "compaction",
            phase: "done",
            summarized: 1,
            reason: "memory",
            summary: "Ada lives in Lyon.",
            summary_lost: false,
          },
        ],
      },
    ];
    tracedFetchMock.mockImplementation(makeRoute({ messages: persisted }));
    apiClient.get.mockImplementation(async () => persisted);

    render(<ConversationPage />);
    await waitFor(() => expect(apiClient.get).toHaveBeenCalled());
    await act(async () => {});

    expect(
      await screen.findByText("1 earlier message was summarized to fit this computer's memory")
    ).toBeTruthy();
    expect(screen.queryByRole("button", { name: /Reasoning/ })).toBeNull();
    expect(screen.getByTestId("answer").textContent).toBe("Then Lyon.");
    fireEvent.click(screen.getByRole("button", { name: "Show summary" }));
    expect(screen.getByText("Ada lives in Lyon.")).toBeTruthy();
  });
});
