// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, act } from "@testing-library/react";

// #303 — a message-list refetch must never erase streamed content.
// Observed live on the packaged build: a turn's streamed answer sometimes never
// rendered (backend fine, chunks read by the renderer, DB row present). Two
// interleavings reproduce it deterministically:
//  A. the end-of-turn refetch races the DB insert and returns rows WITHOUT the
//     just-streamed answer; the blind list replacement then drops the bubble;
//  B. a stale refetch (kicked off around an earlier turn) resolves while a NEW
//     turn is already streaming; the replacement drops the in-flight optimistic
//     bubble, and every later flush (matched by message id) silently no-ops.
// The fix reconciles instead of replacing: fetched rows win, but trailing local
// messages the fetch doesn't cover yet (streaming, or newer than the fetch) are
// kept, and get matched away by content on a later refetch.

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

const doneStream = () => ({
  ok: true,
  body: { getReader: () => ({ read: async () => ({ done: true, value: undefined }) }) },
});

const renderAndSettle = async () => {
  render(<ConversationPage />);
  await waitFor(() => expect(apiClient.get).toHaveBeenCalled());
  await act(async () => {});
  await screen.findByText("SEND");
};

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

describe("refetch vs stream races (#303)", () => {
  it("A: keeps the streamed answer when the end-of-turn refetch misses the DB row", async () => {
    const ctrl = makeControlledStream();
    // fetch_messages knows only the user row — the assistant insert "lost" the
    // race. The streamed bubble must survive the replacement.
    const staleRows = [{ id: 101, sender: "user", content: "hi" }];
    tracedFetchMock.mockImplementation(async (url) => {
      const u = String(url);
      if (u.includes("/query")) return ctrl.response;
      if (u.includes("generate_title")) return doneStream();
      if (u.endsWith("/conversations/7")) return { ok: true, json: async () => conversationDetail };
      if (u.includes("fetch_messages")) return { ok: true, json: async () => staleRows };
      return { ok: true, json: async () => [] };
    });

    await renderAndSettle();
    fireEvent.click(screen.getByText("SEND"));
    await settle();

    await pushAndSettle(ctrl, '{"t":"answer","text":"The streamed answer"}\n');
    expect(screen.getByTestId("answer").textContent).toBe("The streamed answer");

    await act(async () => {
      ctrl.push('{"t":"done"}\n');
      ctrl.end();
      await Promise.resolve();
    });
    // Let the end-of-turn refetch resolve and re-render.
    await settle();
    await settle();

    expect(screen.getByTestId("answer").textContent).toBe("The streamed answer");
  });

  it("B: turn 1's late refetch resolving during turn 2 does not kill turn 2's bubble", async () => {
    // The live pattern: turn N renders, turn N+1 never does. Turn 1's
    // end-of-turn refetch is slow; it resolves while turn 2 is streaming,
    // carrying rows that predate turn 2 — the replacement must not drop the
    // in-flight bubble (whose later flushes match by message id).
    const ctrl1 = makeControlledStream();
    const ctrl2 = makeControlledStream();
    let queryCount = 0;
    let fetchCount = 0;
    let resolveStaleFetch;
    const staleFetch = new Promise((r) => {
      resolveStaleFetch = r;
    });
    const turn1Rows = [
      { id: 101, sender: "user", content: "hi" },
      { id: 102, sender: "llm", content: "Turn one answer" },
    ];
    tracedFetchMock.mockImplementation(async (url) => {
      const u = String(url);
      if (u.includes("/query")) {
        queryCount += 1;
        return queryCount === 1 ? ctrl1.response : ctrl2.response;
      }
      if (u.includes("generate_title")) return doneStream();
      if (u.endsWith("/conversations/7")) return { ok: true, json: async () => conversationDetail };
      if (u.includes("fetch_messages")) {
        fetchCount += 1;
        if (fetchCount === 2) {
          // Turn 1's end-of-turn refetch: held until mid-turn-2.
          await staleFetch;
        }
        return { ok: true, json: async () => turn1Rows };
      }
      return { ok: true, json: async () => [] };
    });

    await renderAndSettle();

    // Turn 1 streams and completes; its refetch stays in flight.
    fireEvent.click(screen.getByText("SEND"));
    await settle();
    await pushAndSettle(ctrl1, '{"t":"answer","text":"Turn one answer"}\n');
    await act(async () => {
      ctrl1.push('{"t":"done"}\n');
      ctrl1.end();
      await Promise.resolve();
    });
    await settle();

    // Turn 2 starts streaming.
    fireEvent.click(screen.getByText("SEND"));
    await settle();
    await pushAndSettle(ctrl2, '{"t":"answer","text":"Turn two "}\n');
    expect(screen.getAllByTestId("answer").some((n) => n.textContent === "Turn two ")).toBe(true);

    // Turn 1's stale refetch lands NOW, mid-turn-2.
    await act(async () => {
      resolveStaleFetch();
      await Promise.resolve();
      await Promise.resolve();
    });

    // Later turn-2 chunks must still paint.
    await pushAndSettle(ctrl2, '{"t":"answer","text":"streams on"}\n');
    expect(
      screen.getAllByTestId("answer").some((n) => n.textContent === "Turn two streams on")
    ).toBe(true);

    await act(async () => {
      ctrl2.push('{"t":"done"}\n');
      ctrl2.end();
      await Promise.resolve();
    });
    await settle();
    await settle();
    expect(
      screen.getAllByTestId("answer").some((n) => n.textContent === "Turn two streams on")
    ).toBe(true);
  });
});
