// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, act } from "@testing-library/react";

// Failure paths of the send flow, plus the first-message title stream and the
// post-delete navigation contract:
//  - a non-OK /query response stores an error row server-side and renders the
//    apology bubble;
//  - a stream that dies mid-answer appends the "Connection interrupted"
//    sentinel and drops the trace;
//  - a wire error event replaces the answer with the backend's sentinel text;
//  - the first message streams a conversation title into the history list;
//  - deleting the OPEN conversation navigates back to /erudi/chat (#228: the
//    parent handler does post-delete UI only).

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
  default: ({ items, onDelete }) => (
    <div>
      <ul data-testid="history">
        {items.map((c) => (
          <li key={c.id}>{c.name}</li>
        ))}
      </ul>
      <button onClick={() => onDelete(7)}>delete-open</button>
    </div>
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

const doneStream = () => ({
  ok: true,
  body: { getReader: () => ({ read: async () => ({ done: true, value: undefined }) }) },
});

const encoder = new TextEncoder();
/** Finite stream of text chunks. */
const streamOf = (...chunks) => {
  const queue = chunks.map((text) => ({ done: false, value: encoder.encode(text) }));
  queue.push({ done: true, value: undefined });
  return { ok: true, body: { getReader: () => ({ read: async () => queue.shift() }) } };
};

/** Base URL routing shared by the suite; individual tests override /query. */
const routeFetch =
  (overrides = {}) =>
  async (url) => {
    const u = String(url);
    for (const [needle, handler] of Object.entries(overrides)) {
      if (u.includes(needle)) return handler(u);
    }
    if (u.includes("generate_title")) return doneStream();
    if (u.endsWith("/conversations/7")) return { ok: true, json: async () => conversationDetail };
    if (u.includes("fetch_messages")) return { ok: true, json: async () => [] };
    return { ok: true, json: async () => [] };
  };

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

beforeEach(() => {
  Element.prototype.scrollTo = () => {};
  navigateMock.mockReset();
  tracedFetchMock.mockReset();
  apiClient.get.mockReset();
  apiClient.get.mockImplementation(async () => []);
});
afterEach(() => {
  cleanup();
});

describe("ConversationPage send failure paths", () => {
  it("stores an error row and shows the apology bubble on a non-OK /query", async () => {
    tracedFetchMock.mockImplementation(
      routeFetch({ "/query": () => ({ ok: false, status: 500 }) })
    );
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));

    await screen.findByText(/I apologize, but I encountered an error/);
    await waitFor(() =>
      expect(
        tracedFetchMock.mock.calls.some(([u]) => String(u).includes("store_error_message"))
      ).toBe(true)
    );
    const storeCall = tracedFetchMock.mock.calls.find(([u]) =>
      String(u).includes("store_error_message")
    );
    expect(storeCall[1].method).toBe("POST");
  });

  it("appends the connection-interrupted sentinel when the stream dies mid-answer", async () => {
    const read = vi
      .fn()
      .mockResolvedValueOnce({
        done: false,
        value: encoder.encode('{"t":"answer","text":"Par"}\n'),
      })
      .mockRejectedValueOnce(new Error("socket reset"));
    tracedFetchMock.mockImplementation(
      routeFetch({
        "/query": () => ({ ok: true, body: { getReader: () => ({ read }) } }),
      })
    );
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));

    // The partial answer is kept in front of the sentinel (error bubbles
    // render as plain text, not through MarkdownRenderer).
    const bubble = await screen.findByText(/Connection interrupted while generating response/);
    expect(bubble.textContent).toContain("Par");
  });

  it("renders the backend's sentinel text on a wire error event", async () => {
    tracedFetchMock.mockImplementation(
      routeFetch({
        "/query": () =>
          streamOf(
            '{"t":"error","text":"[ERROR_MESSAGE_SYSTEM] model exploded"}\n',
            '{"t":"done"}\n'
          ),
      })
    );
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));

    await screen.findByText(/model exploded/);
  });

  it("opens the decision dialog when the error event carries an engine code", async () => {
    // The engine identified the cause: the red bubble is still there, and the
    // dialog offers the remedy on top of it.
    tracedFetchMock.mockImplementation(
      routeFetch({
        "/query": () =>
          streamOf(
            JSON.stringify({
              t: "error",
              text: "[ERROR_MESSAGE_SYSTEM] the inference server exited before becoming ready",
              code: "CUDA_COMPUTE_CAPABILITY_TOO_LOW",
              raw: "CUDA error: no kernel image is available for execution on the device",
            }) + "\n",
            '{"t":"done"}\n'
          ),
      })
    );
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));

    await screen.findByText(/too old for GPU mode/i);
    expect(screen.getByText(/no kernel image is available/)).toBeTruthy();
    expect(screen.getByText(/exited before becoming ready/)).toBeTruthy();
  });

  it("leaves an untyped error as a red bubble and nothing else", async () => {
    tracedFetchMock.mockImplementation(
      routeFetch({
        "/query": () =>
          streamOf(
            '{"t":"error","text":"[ERROR_MESSAGE_SYSTEM] model exploded"}\n',
            '{"t":"done"}\n'
          ),
      })
    );
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));

    await screen.findByText(/model exploded/);
    expect(screen.queryByText(/Switch to processor mode/i)).toBeNull();
  });
});

describe("ConversationPage first-message title stream", () => {
  it("streams the generated title into the conversation history entry", async () => {
    // The history list must contain conversation 7 so the retitle is visible.
    // The /query stream is held open: the end-of-turn refetch would otherwise
    // overwrite the client-side retitle with the mocked (stale) list.
    const convRow = { id: 7, name: "New Conversation", last_message_time: "2026-01-01" };
    tracedFetchMock.mockImplementation(async (url) => {
      const u = String(url);
      if (u.includes("generate_title")) return streamOf("Sun ", "facts");
      if (u.includes("/query")) return new Promise(() => {});
      if (u.endsWith("/conversations/7")) return { ok: true, json: async () => conversationDetail };
      if (u.includes("fetch_messages")) return { ok: true, json: async () => [] };
      return { ok: true, json: async () => [convRow] };
    });
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    await settle();
    await settle();

    // The streamed title lands in the history list entry (trimmed).
    await waitFor(() => expect(screen.getByTestId("history").textContent).toContain("Sun facts"));
    // And the title endpoint got the question it should summarize.
    const titleCall = tracedFetchMock.mock.calls.find(([u]) =>
      String(u).includes("generate_title")
    );
    expect(JSON.parse(titleCall[1].body)).toEqual({ question: "hi" });
  });
});

describe("ConversationPage post-delete navigation (#228)", () => {
  it("navigates back to the chat hub when the open conversation is deleted", async () => {
    tracedFetchMock.mockImplementation(routeFetch({}));
    await renderAndSettle();

    fireEvent.click(screen.getByText("delete-open"));

    await waitFor(() => expect(navigateMock).toHaveBeenCalledWith("/erudi/chat"));
  });
});
