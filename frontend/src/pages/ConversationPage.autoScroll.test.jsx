// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, act } from "@testing-library/react";

// Auto-scroll: once the user has scrolled up into a long history, the view
// must NOT follow the stream; sending a message is the user's explicit return
// to the bottom, so the view jumps there and follows the answer again.

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

const longHistory = Array.from({ length: 6 }, (_, i) => ({
  id: 200 + i,
  sender: i % 2 === 0 ? "user" : "llm",
  content: `message ${i}`,
}));

const HISTORY_HEIGHT = 2000;
const VIEW_HEIGHT = 300;

/** A streaming Response whose body reader is fed chunks on demand. */
function makeControlledStream() {
  const enc = new TextEncoder();
  const queue = [];
  let waiting = null;
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
  const read = () =>
    new Promise((resolve) => {
      if (queue.length) {
        resolve(queue.shift());
      } else {
        waiting = resolve;
      }
    });
  return { response: { ok: true, body: { getReader: () => ({ read }) } }, push };
}

const doneStream = () => ({
  ok: true,
  body: { getReader: () => ({ read: async () => ({ done: true, value: undefined }) }) },
});

const settle = async () => {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
};

const waitMs = async (ms) => {
  await act(async () => {
    await new Promise((r) => setTimeout(r, ms));
  });
};

/** The scrolling message list: its geometry is fixed so the scroll math is real. */
const scrollPane = (container) => {
  const pane = container.querySelector(".flex-1.overflow-y-auto");
  let top = HISTORY_HEIGHT - VIEW_HEIGHT;
  Object.defineProperty(pane, "scrollHeight", { configurable: true, get: () => HISTORY_HEIGHT });
  Object.defineProperty(pane, "clientHeight", { configurable: true, get: () => VIEW_HEIGHT });
  Object.defineProperty(pane, "scrollTop", {
    configurable: true,
    get: () => top,
    set: (v) => {
      top = v;
    },
  });
  return pane;
};

// The handler only sees a scroll UP relative to the last event, so the pane
// first reports the bottom, then the user scrolls up from it.
const scrollUp = (pane) => {
  pane.scrollTop = HISTORY_HEIGHT - VIEW_HEIGHT;
  fireEvent.scroll(pane);
  pane.scrollTop = 300;
  fireEvent.scroll(pane);
};

let scrollToMock;

beforeEach(() => {
  scrollToMock = vi.fn();
  Element.prototype.scrollTo = scrollToMock;
  tracedFetchMock.mockReset();
  // The page reloads its history through apiClient on the first load; it must
  // keep the same long history, or the refetch would empty the list.
  apiClient.get.mockReset();
  apiClient.get.mockImplementation(async (path) =>
    String(path).includes("fetch_messages") ? longHistory : []
  );
  tracedFetchMock.mockImplementation(async (url) => {
    const u = String(url);
    if (u.includes("generate_title")) return doneStream();
    if (u.endsWith("/conversations/7")) return { ok: true, json: async () => conversationDetail };
    if (u.includes("fetch_messages")) return { ok: true, json: async () => longHistory };
    return { ok: true, json: async () => [] };
  });
});
afterEach(() => {
  cleanup();
  delete Element.prototype.scrollTo;
});

describe("auto-scroll after the user scrolled up (Conversation spec)", () => {
  it("jumps to the bottom on send and follows the answer as it streams", async () => {
    const ctrl = makeControlledStream();
    tracedFetchMock.mockImplementation(async (url) => {
      const u = String(url);
      if (u.includes("/query")) return ctrl.response;
      if (u.includes("generate_title")) return doneStream();
      if (u.endsWith("/conversations/7")) return { ok: true, json: async () => conversationDetail };
      if (u.includes("fetch_messages")) return { ok: true, json: async () => longHistory };
      return { ok: true, json: async () => [] };
    });

    const { container } = render(<ConversationPage />);
    await waitFor(() => expect(apiClient.get).toHaveBeenCalled());
    await screen.findByText("message 5");
    const pane = scrollPane(container);

    scrollUp(pane);
    await waitMs(150);
    scrollToMock.mockClear();

    await act(async () => {
      fireEvent.click(screen.getByText("SEND"));
    });
    await settle();

    expect(scrollToMock).toHaveBeenCalledWith(
      expect.objectContaining({ top: HISTORY_HEIGHT, behavior: "smooth" })
    );

    // Following continues: a streamed chunk after the send keeps the view at the bottom.
    await waitMs(150);
    scrollToMock.mockClear();
    await act(async () => {
      ctrl.push('{"t":"answer","text":"The streamed answer"}\n');
    });
    await settle();
    expect(screen.getByText("The streamed answer")).toBeTruthy();
    expect(scrollToMock).toHaveBeenCalledWith(expect.objectContaining({ top: HISTORY_HEIGHT }));
  });

  it("cancels a pending scroll re-check on send, so the view keeps following", async () => {
    const ctrl = makeControlledStream();
    tracedFetchMock.mockImplementation(async (url) => {
      const u = String(url);
      if (u.includes("/query")) return ctrl.response;
      if (u.includes("generate_title")) return doneStream();
      if (u.endsWith("/conversations/7")) return { ok: true, json: async () => conversationDetail };
      if (u.includes("fetch_messages")) return { ok: true, json: async () => longHistory };
      return { ok: true, json: async () => [] };
    });

    const { container } = render(<ConversationPage />);
    await waitFor(() => expect(apiClient.get).toHaveBeenCalled());
    await screen.findByText("message 5");
    const pane = scrollPane(container);

    // The scroll re-check is still pending (under 100 ms) when the user sends.
    scrollUp(pane);
    await act(async () => {
      fireEvent.click(screen.getByText("SEND"));
    });
    // Let the debounce would-be re-check fire: it must not re-mark the view as scrolled up.
    await waitMs(150);
    scrollToMock.mockClear();

    await act(async () => {
      ctrl.push('{"t":"answer","text":"Following"}\n');
    });
    await settle();
    expect(scrollToMock).toHaveBeenCalledWith(expect.objectContaining({ top: HISTORY_HEIGHT }));
  });
});
