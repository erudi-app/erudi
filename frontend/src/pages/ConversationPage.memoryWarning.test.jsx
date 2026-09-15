// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, act } from "@testing-library/react";

// 1.1.2 — the amber memory warning:
//  - a `memory_warning` stream event renders an amber alert near the composer
//    with the localized copy and the conversation's approximate memory size;
//  - the warning is transient state only: it never lands in the answer bubble
//    (so it cannot be copied with the message) and a LATER turn that carries
//    no warning clears it.

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
vi.mock("../components/HeaderBar", () => ({ default: () => null }));
vi.mock("../components/TypingIndicator", () => ({ default: () => null }));
vi.mock("../components/TraceStrip", () => ({ default: () => null }));
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

/** Routes /query calls to a queue of controlled streams (one per turn). */
const makeRoute = (queryStreams) => {
  const pending = [...queryStreams];
  return async (url) => {
    const u = String(url);
    if (u.includes("/query")) return pending.shift().response;
    if (u.includes("generate_title")) return doneStream();
    if (u.endsWith("/conversations/7")) return { ok: true, json: async () => conversationDetail };
    if (u.includes("fetch_messages")) return { ok: true, json: async () => [] };
    return { ok: true, json: async () => [] };
  };
};

const renderAndSettle = async () => {
  render(<ConversationPage />);
  await waitFor(() => expect(apiClient.get).toHaveBeenCalled());
  await act(async () => {});
  await screen.findByText("SEND");
};

const runTurn = async (stream, lines) => {
  await act(async () => {
    for (const line of lines) {
      stream.push(line);
    }
    stream.end();
  });
};

beforeEach(() => {
  vi.clearAllMocks();
  Element.prototype.scrollTo = () => {};
});

afterEach(() => {
  cleanup();
});

describe("ConversationPage memory warning", () => {
  it("renders the amber warning with the conversation size, out of the bubble", async () => {
    const turn = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute([turn]));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await runTurn(turn, [
      '{"t":"answer","text":"Fine."}\n',
      '{"t":"memory_warning","used_fraction":0.91,"conversation_bytes":4294967296}\n',
      '{"t":"done"}\n',
    ]);

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("4.0 GB");
    expect(alert.textContent).toContain("memory");
    // Never part of the message content (so never in the copy path).
    for (const bubble of screen.getAllByTestId("answer")) {
      expect(bubble.textContent).not.toContain("4.0 GB");
      expect(bubble.textContent).toBe("Fine.");
    }
  });

  it("clears when a later turn carries no warning", async () => {
    const first = makeControlledStream();
    const second = makeControlledStream();
    tracedFetchMock.mockImplementation(makeRoute([first, second]));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await runTurn(first, [
      '{"t":"answer","text":"Long."}\n',
      '{"t":"memory_warning","used_fraction":0.9,"conversation_bytes":1048576}\n',
      '{"t":"done"}\n',
    ]);
    await screen.findByRole("alert");

    fireEvent.click(screen.getByText("SEND"));
    await runTurn(second, ['{"t":"answer","text":"Compacted."}\n', '{"t":"done"}\n']);

    await waitFor(() => {
      expect(screen.queryByRole("alert")).toBeNull();
    });
  });
});
