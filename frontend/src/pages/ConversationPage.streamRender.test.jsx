// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, act } from "@testing-library/react";

// #618 — while an answer streams, only the message being written re-renders.
// react-markdown is the expensive part (remark, remark-gfm, remark-math and
// rehype-katex parse the whole text), so it is replaced by a render counter:
// every call is one parse of one message. MarkdownRenderer itself stays real,
// which is what the memo boundary is tested on.
//
// The same file is the benchmark: it renders a conversation of 20 and of 100
// messages (markdown and math), streams 200 tokens into the last one and logs
// the renders it counted.

const { tracedFetchMock, renders, navigateMock, locationMock } = vi.hoisted(() => ({
  tracedFetchMock: vi.fn(),
  renders: [],
  navigateMock: vi.fn(),
  locationMock: { pathname: "/conversation/7", state: null },
}));

vi.mock("react-markdown", () => ({
  default: ({ children }) => {
    renders.push(String(children));
    return <div data-testid="md">{children}</div>;
  },
}));
vi.mock("remark-gfm", () => ({ default: () => {} }));
vi.mock("remark-math", () => ({ default: () => {} }));
vi.mock("rehype-katex", () => ({ default: () => {} }));
vi.mock("katex/dist/katex.min.css", () => ({}));

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
vi.mock("../components/Sidebar", () => ({ default: () => null }));
vi.mock("../components/ChatCollapsibleSection", () => ({ default: () => null }));
vi.mock("../components/ConversationHeader", () => ({ default: () => null }));
vi.mock("../components/TypingIndicator", () => ({ default: () => null }));
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

const oldRows = (count) =>
  Array.from({ length: count }, (_, i) => ({
    id: 1000 + i,
    sender: i % 2 === 0 ? "user" : "llm",
    content:
      i % 2 === 0
        ? `OLD question ${i}`
        : `OLD answer ${i}\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nInline $x^2 + y^2$ and\n\n$$\\int_0^1 x\\,dx = \\tfrac12$$`,
  }));

function makeControlledStream() {
  const enc = new TextEncoder();
  const queue = [];
  let waiting = null;
  let ended = false;
  let failure = null;
  const read = () =>
    new Promise((resolve, reject) => {
      if (queue.length) resolve(queue.shift());
      else if (failure) reject(failure);
      else if (ended) resolve({ done: true, value: undefined });
      else waiting = { resolve, reject };
    });
  return {
    response: { ok: true, body: { getReader: () => ({ read }) } },
    push: (str) => {
      const chunk = { done: false, value: enc.encode(str) };
      if (waiting) {
        const { resolve } = waiting;
        waiting = null;
        resolve(chunk);
      } else queue.push(chunk);
    },
    end: () => {
      ended = true;
      if (waiting) {
        const { resolve } = waiting;
        waiting = null;
        resolve({ done: true, value: undefined });
      }
    },
    fail: (err) => {
      failure = err;
      if (waiting) {
        const { reject } = waiting;
        waiting = null;
        reject(err);
      }
    },
  };
}

const doneStream = () => ({
  ok: true,
  body: { getReader: () => ({ read: async () => ({ done: true, value: undefined }) }) },
});

const settle = async () => {
  await act(async () => {
    // Enough turns of the microtask queue for the stream reader to hand
    // every pushed line to the handler before the next token is pushed.
    for (let i = 0; i < 20; i++) {
      await Promise.resolve();
    }
  });
};
const token = (ctrl, text) => ctrl.push(`${JSON.stringify({ t: "answer", text })}\n`);
// Let the coalescing window elapse, then the render it releases.
const tick = async (ms = 40) => {
  await act(async () => {
    vi.advanceTimersByTime(ms);
    await Promise.resolve();
  });
};

// Sending runs the settings flush before the chat request: its timers are faked.
const send = async () => {
  fireEvent.click(screen.getByText("SEND"));
  await tick(500);
  await settle();
};

let ctrl;
let rowsAfterTurn;
const mountWith = async (count) => {
  ctrl = makeControlledStream();
  rowsAfterTurn = oldRows(count);
  apiClient.get.mockImplementation(async (path) =>
    String(path).includes("fetch_messages") ? rowsAfterTurn : []
  );
  tracedFetchMock.mockImplementation(async (url) => {
    const u = String(url);
    if (u.includes("/query")) return ctrl.response;
    if (u.includes("generate_title")) return doneStream();
    if (u.endsWith("/conversations/7")) return { ok: true, json: async () => conversationDetail };
    if (u.includes("fetch_messages")) return { ok: true, json: async () => rowsAfterTurn };
    return { ok: true, json: async () => [] };
  });
  render(<ConversationPage />);
  await waitFor(() => expect(apiClient.get).toHaveBeenCalled());
  await settle();
  await screen.findByText("SEND");
  await waitFor(() => expect(screen.getAllByTestId("md").length).toBeGreaterThan(0));
  // Testing Library's polling needs real timers; the coalescing window is
  // driven by hand from here on.
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
};

beforeEach(() => {
  Element.prototype.scrollTo = () => {};
  renders.length = 0;
  tracedFetchMock.mockReset();
  apiClient.get.mockReset();
  apiClient.get.mockImplementation(async () => []);
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

const streamedText = () => {
  const all = screen.getAllByTestId("md");
  return all[all.length - 1].textContent;
};
const oldRendersSince = (mark) => renders.slice(mark).filter((c) => c.startsWith("OLD")).length;

describe("streaming renders only the message being written (#618)", () => {
  for (const count of [20, 100]) {
    it(`benchmark: ${count} messages, 200 tokens into the last one`, async () => {
      await mountWith(count);
      await send();
      const mark = renders.length;
      for (let i = 0; i < 200; i++) {
        token(ctrl, `tok${i} `);
        await settle();
        if (i % 3 === 2) await tick(35);
      }
      const streamedRenders = renders.slice(mark).filter((c) => !c.startsWith("OLD")).length;
      const old = oldRendersSince(mark);
      // The benchmark's report is its output (run with --disable-console-intercept).
      // eslint-disable-next-line no-console
      console.log(
        `[bench] ${count} messages: old-message renders=${old}, streamed-message renders=${streamedRenders}, tokens=200`
      );
      expect(old).toBe(0);
    }, 120_000);
  }

  it("shows every token in order and the exact final text after the end of the turn", async () => {
    await mountWith(6);
    await send();
    const tokens = Array.from({ length: 40 }, (_, i) => `w${i} `);
    let expected = "";
    for (const tk of tokens) {
      token(ctrl, tk);
      expected += tk;
      await settle();
      await tick();
      expect(streamedText().trim()).toBe(expected.trim());
    }
    rowsAfterTurn = [
      ...oldRows(6),
      { id: 2000, sender: "user", content: "hi" },
      { id: 2001, sender: "llm", content: expected },
    ];
    await act(async () => {
      ctrl.push('{"t":"done"}\n');
      ctrl.end();
    });
    await settle();
    await settle();
    expect(streamedText().trim()).toBe(expected.trim());
  });

  it("flushes the text received when the turn ends before the window elapses", async () => {
    await mountWith(6);
    await send();
    token(ctrl, "first ");
    await settle();
    token(ctrl, "second ");
    token(ctrl, "third");
    await settle();
    // No timer advance: the end of the turn must not leave "second third" behind.
    rowsAfterTurn = [...oldRows(6), { id: 2000, sender: "user", content: "hi" }];
    await act(async () => {
      ctrl.push('{"t":"done"}\n');
      ctrl.end();
    });
    await settle();
    expect(streamedText()).toContain("first second third");
  });

  it("flushes what was received when an error event arrives", async () => {
    await mountWith(6);
    await send();
    token(ctrl, "partial ");
    await settle();
    token(ctrl, "text");
    ctrl.push('{"t":"error","text":"[ERROR] boom"}\n');
    await settle();
    await settle();
    expect(document.body.textContent).toContain("partial text");
  });

  it("flushes what was received when the connection drops mid-stream", async () => {
    await mountWith(6);
    await send();
    token(ctrl, "before ");
    await settle();
    token(ctrl, "the drop");
    await settle();
    await act(async () => {
      ctrl.fail(new Error("socket closed"));
    });
    await settle();
    await settle();
    expect(document.body.textContent).toContain("before the drop");
  });

  it("re-renders a message whose content changes outside streaming", async () => {
    await mountWith(6);
    expect(screen.getAllByTestId("md")[0].textContent).toContain("OLD answer 1");
    const mark = renders.length;
    rowsAfterTurn = oldRows(6).map((r) =>
      r.id === 1001 ? { ...r, content: "EDITED answer one" } : r
    );
    await send();
    await act(async () => {
      ctrl.push('{"t":"done"}\n');
      ctrl.end();
    });
    await settle();
    await settle();
    expect(screen.getAllByTestId("md")[0].textContent).toBe("EDITED answer one");
    expect(renders.slice(mark)).toContain("EDITED answer one");
  });
});
