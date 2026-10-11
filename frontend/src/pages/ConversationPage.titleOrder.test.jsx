// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, act } from "@testing-library/react";

// On the first message the title request runs FIRST and the chat request starts
// only once the title stream has ended. llama.cpp runs one slot with no prompt
// cache, so a title that ran after the first chat turn would evict the
// conversation's prefix and the second message would re-read the whole
// conversation. The chat never waits on a title that failed: an HTTP error or a
// rejected fetch also lets it start. Later messages issue no title request.

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
vi.mock("../components/ChatCollapsibleSection", () => ({
  default: ({ items }) => (
    <ul data-testid="history">
      {items.map((c) => (
        <li key={c.id}>{c.name}</li>
      ))}
    </ul>
  ),
}));
vi.mock("../components/HeaderBar", () => ({ default: () => null }));
vi.mock("../components/TypingIndicator", () => ({ default: () => null }));
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

const convRow = { id: 7, name: "New Conversation", last_message_time: "2026-01-01" };

const encoder = new TextEncoder();

/** A streaming Response whose body reader is fed chunks on demand. */
function makeControlledStream() {
  const queue = [];
  let waiting = null;
  let ended = false;
  const push = (str) => {
    const chunk = { done: false, value: encoder.encode(str) };
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
      if (queue.length) resolve(queue.shift());
      else if (ended) resolve({ done: true, value: undefined });
      else waiting = resolve;
    });
  return { response: { ok: true, body: { getReader: () => ({ read }) } }, push, end };
}

const emptyStream = () => ({
  ok: true,
  body: { getReader: () => ({ read: async () => ({ done: true, value: undefined }) }) },
});

/** Finite NDJSON stream for the chat answer. */
const answerStream = () => {
  const queue = [
    { done: false, value: encoder.encode('{"t":"answer","text":"ok"}\n') },
    { done: false, value: encoder.encode('{"t":"done"}\n') },
    { done: true, value: undefined },
  ];
  return { ok: true, body: { getReader: () => ({ read: async () => queue.shift() }) } };
};

const settle = async () => {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
  });
};

/** Routes every call; `title` answers generate_title. */
const routeFetch = (title) => async (url) => {
  const u = String(url);
  if (u.includes("generate_title")) return title(u);
  if (u.includes("/query")) return answerStream();
  if (u.endsWith("/conversations/7")) return { ok: true, json: async () => conversationDetail };
  if (u.includes("fetch_messages")) return { ok: true, json: async () => [] };
  // The history list must contain conversation 7 so the live retitle is visible.
  return { ok: true, json: async () => [convRow] };
};

const calls = (needle) =>
  tracedFetchMock.mock.calls.filter(([u]) => String(u).includes(needle)).length;

const renderAndSettle = async () => {
  render(<ConversationPage />);
  await waitFor(() => expect(apiClient.get).toHaveBeenCalled());
  await act(async () => {});
  await screen.findByText("SEND");
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

describe("first message: title before chat", () => {
  it("starts the chat request only once the title stream has ended", async () => {
    const title = makeControlledStream();
    tracedFetchMock.mockImplementation(routeFetch(() => title.response));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await settle();

    expect(calls("generate_title")).toBe(1);
    expect(calls("/query")).toBe(0);

    // The name updates live while the title streams; the chat still waits.
    title.push("Quantum");
    await settle();
    expect(screen.getByTestId("history").textContent).toContain("Quantum");
    expect(calls("/query")).toBe(0);

    title.end();
    await waitFor(() => expect(calls("/query")).toBe(1));
    const order = tracedFetchMock.mock.calls.map(([u]) => String(u));
    expect(order.findIndex((u) => u.includes("generate_title"))).toBeLessThan(
      order.findIndex((u) => u.includes("/query"))
    );
  });

  it("still starts the chat when the title request returns an HTTP error", async () => {
    tracedFetchMock.mockImplementation(routeFetch(() => ({ ok: false, status: 500 })));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));

    await waitFor(() => expect(calls("/query")).toBe(1));
    expect(calls("generate_title")).toBe(1);
  });

  it("still starts the chat when the title fetch rejects", async () => {
    tracedFetchMock.mockImplementation(
      routeFetch(() => {
        throw new Error("network down");
      })
    );
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));

    await waitFor(() => expect(calls("/query")).toBe(1));
    expect(calls("generate_title")).toBe(1);
  });

  it("issues no title request for a later message", async () => {
    tracedFetchMock.mockImplementation(routeFetch(emptyStream));
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await waitFor(() => expect(calls("/query")).toBe(1));
    await screen.findByText("ok");

    fireEvent.click(screen.getByText("SEND"));
    await waitFor(() => expect(calls("/query")).toBe(2));
    expect(calls("generate_title")).toBe(1);
  });
});
