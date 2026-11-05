// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, act } from "@testing-library/react";

// #90 — NDJSON streaming + live/persisted reasoning trace panel:
//  - the /query reader parses NDJSON (one JSON event per line) with partial-line
//    carry across reads; answer events stream into the bubble as before;
//  - a turn with no thinking/tool events mounts NO trace strip;
//  - a thinking+tool turn shows the strip live (expanded) then auto-collapses on
//    the first answer event;
//  - a persisted message with a `trace` array replays the collapsed strip;
//  - an `error` event renders the SAME red error bubble the sentinel produces.

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

// Stub heavy/irrelevant children. MarkdownRenderer echoes content so the answer
// bubble text is assertable. TraceStrip is the real component under test.
vi.mock("../components/Sidebar", () => ({ default: () => null }));
vi.mock("../components/ChatCollapsibleSection", () => ({ default: () => null }));
vi.mock("../components/ConversationHeader", () => ({ default: () => null }));
vi.mock("../components/TypingIndicator", () => ({ default: () => null }));
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

/** An already-finished plain-text stream (used for the title-gen reader). */
const doneStream = () => ({
  ok: true,
  body: { getReader: () => ({ read: async () => ({ done: true, value: undefined }) }) },
});

const makeRoute =
  ({ queryResponse, messages = [] } = {}) =>
  async (url) => {
    const u = String(url);
    if (u.includes("/query")) return queryResponse;
    if (u.includes("generate_title")) return doneStream();
    if (u.endsWith("/conversations/7")) return { ok: true, json: async () => conversationDetail };
    if (u.includes("fetch_messages")) return { ok: true, json: async () => messages };
    if (u.includes("store_error_message")) return { ok: true };
    return { ok: true, json: async () => [] };
  };

const renderAndSettle = async () => {
  render(<ConversationPage />);
  await waitFor(() => expect(apiClient.get).toHaveBeenCalled());
  await act(async () => {});
  await screen.findByText("SEND");
};

// Resolve a queued read and let the reader loop drain + re-render.
const settle = async () => {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
};
const pushAndSettle = async (ctrl, str) => {
  ctrl.push(str);
  await settle();
};

beforeEach(() => {
  Element.prototype.scrollTo = () => {};
  tracedFetchMock.mockReset();
  apiClient.get.mockReset();
  apiClient.get.mockImplementation(async () => []);
});
afterEach(() => {
  cleanup();
});

describe("NDJSON reader (#90)", () => {
  it("streams answer text (partial line carried across reads) with no strip for answer-only turns", async () => {
    const ctrl = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute({ queryResponse: ctrl.response }));

    await renderAndSettle();
    fireEvent.click(screen.getByText("SEND"));
    await settle();

    // A single answer line delivered as two reads (split mid-JSON).
    await pushAndSettle(ctrl, '{"t":"ans');
    expect(screen.queryByText(/Hello/)).toBeNull(); // incomplete line: nothing yet

    await pushAndSettle(ctrl, 'wer","text":"Hello "}\n{"t":"ans');
    expect(screen.getByTestId("answer").textContent).toBe("Hello ");

    await pushAndSettle(ctrl, 'wer","text":"world"}\n');
    expect(screen.getByTestId("answer").textContent).toBe("Hello world");

    // Answer-only turn: no reasoning strip is ever mounted.
    expect(screen.queryByText(/Reasoning/)).toBeNull();

    await act(async () => {
      ctrl.push('{"t":"done"}\n');
      ctrl.end();
      await Promise.resolve();
    });
    await settle();
  });

  it("shows the trace strip live (expanded) then auto-collapses on the first answer event", async () => {
    const ctrl = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute({ queryResponse: ctrl.response }));

    await renderAndSettle();
    fireEvent.click(screen.getByText("SEND"));
    await settle();

    // Thinking streams first -> strip is the only content, expanded by default.
    await pushAndSettle(ctrl, '{"t":"thinking","text":"I should add the numbers."}\n');
    expect(screen.getByText(/I should add the numbers\./)).toBeTruthy();

    // Tool call + result appear as formatted rows (never raw JSON).
    await pushAndSettle(
      ctrl,
      '{"t":"tool_call","name":"calculator","args":{"expression":"2 + 2"}}\n' +
        '{"t":"tool_result","name":"calculator","text":"4"}\n'
    );
    expect(screen.getByText(/calculator\(2 \+ 2\)/)).toBeTruthy();

    // First answer event -> strip auto-collapses (rows hidden), answer streams
    // into the bubble; the collapsed summary counts 2 steps.
    await pushAndSettle(ctrl, '{"t":"answer","text":"The sum is 4."}\n');
    expect(screen.queryByText(/I should add the numbers\./)).toBeNull();
    expect(screen.getByTestId("answer").textContent).toBe("The sum is 4.");
    expect(screen.getByRole("button", { name: /Reasoning/ }).textContent).toMatch(
      /Reasoning.*2 steps/
    );

    await act(async () => {
      ctrl.push('{"t":"done"}\n');
      ctrl.end();
      await Promise.resolve();
    });
    await settle();
  });

  it("maps an error event to the red error bubble and mounts no strip", async () => {
    const ctrl = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute({ queryResponse: ctrl.response }));

    await renderAndSettle();
    fireEvent.click(screen.getByText("SEND"));
    await settle();

    // Even if the turn produced thinking, an error event ends it as a red bubble.
    await pushAndSettle(ctrl, '{"t":"thinking","text":"trying..."}\n');
    await pushAndSettle(ctrl, '{"t":"error","text":"[ERROR_MESSAGE_SYSTEM] boom"}\n');

    const errorNode = screen.getByText(/boom/);
    expect(errorNode.closest(".text-red-400")).toBeTruthy();
    // The sentinel is rendered as the "❌ " prefix, not shown raw.
    expect(errorNode.textContent).not.toContain("[ERROR_MESSAGE_SYSTEM]");
    // Error turns show no reasoning strip (matches persisted history).
    expect(screen.queryByText(/Reasoning/)).toBeNull();

    await act(async () => {
      ctrl.push('{"t":"done"}\n');
      ctrl.end();
      await Promise.resolve();
    });
    await settle();
  });
});

describe("persisted trace replay (#90)", () => {
  const persisted = [
    { id: 11, sender: "user", content: "add them", starred: false },
    {
      id: 12,
      sender: "llm",
      content: "The sum is 4.",
      starred: false,
      trace: [
        { t: "thinking", text: "I should add the numbers." },
        { t: "tool_call", name: "calculator", args: { expression: "2 + 2" } },
        { t: "tool_result", name: "calculator", text: "4" },
      ],
    },
  ];

  it("renders a collapsed, expandable strip above an assistant message that carries a trace", async () => {
    tracedFetchMock.mockImplementation(makeRoute({ messages: persisted }));
    apiClient.get.mockImplementation(async () => persisted);

    render(<ConversationPage />);
    await waitFor(() => expect(apiClient.get).toHaveBeenCalled());
    await act(async () => {});

    // Collapsed summary present, rows hidden until the user expands.
    const strip = await screen.findByRole("button", { name: /Reasoning/ });
    expect(strip.textContent).toMatch(/Reasoning.*2 steps/);
    expect(screen.queryByText(/I should add the numbers\./)).toBeNull();

    fireEvent.click(strip);
    expect(screen.getByText(/I should add the numbers\./)).toBeTruthy();
    expect(screen.getByText(/calculator\(2 \+ 2\)/)).toBeTruthy();
  });

  it("shows the (earlier steps elided) note when the trace begins with a truncated marker", async () => {
    const truncated = [
      persisted[0],
      { ...persisted[1], trace: [{ t: "truncated" }, ...persisted[1].trace] },
    ];
    tracedFetchMock.mockImplementation(makeRoute({ messages: truncated }));
    apiClient.get.mockImplementation(async () => truncated);

    render(<ConversationPage />);
    await waitFor(() => expect(apiClient.get).toHaveBeenCalled());
    await act(async () => {});

    fireEvent.click(await screen.findByRole("button", { name: /Reasoning/ }));
    expect(screen.getByText(/earlier steps elided/)).toBeTruthy();
  });
});
