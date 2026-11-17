// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, act } from "@testing-library/react";

// An ephemeral conversation (#428) on the conversation page: the header shows
// the badge and offers no export, the first message asks for no title, and
// the answer streaming when the user leaves is aborted -- without a red
// bubble or an error record, since the user asked for it.

const { tracedFetchMock, navigateMock, locationMock, logMock } = vi.hoisted(() => ({
  tracedFetchMock: vi.fn(),
  navigateMock: vi.fn(),
  locationMock: { pathname: "/erudi/conversations/7", state: null },
  logMock: { error: vi.fn(), warn: vi.fn(), info: vi.fn(), log: vi.fn(), debug: vi.fn() },
}));

vi.mock("../services/api/client", () => ({
  default: { get: vi.fn(async () => []) },
  apiClient: { get: vi.fn(async () => []) },
  tracedFetch: tracedFetchMock,
}));

vi.mock("react-router-dom", async (importOriginal) => ({
  ...(await importOriginal()),
  useParams: () => ({ id: "7" }),
  useNavigate: () => navigateMock,
  useLocation: () => locationMock,
}));

vi.mock("../utils/logger", () => ({ createLogger: () => logMock }));
vi.mock("../components/Sidebar", () => ({ default: () => null }));
vi.mock("../components/ChatCollapsibleSection", () => ({ default: () => null }));
vi.mock("../components/TypingIndicator", () => ({ default: () => null }));
vi.mock("../components/MarkdownRenderer", () => ({
  default: ({ content }) => <div data-testid="answer">{content}</div>,
}));
vi.mock("../components/ConversationHeader", () => ({
  default: ({ ephemeral, onExport }) => (
    <div>
      {ephemeral ? <span>HEADER_EPHEMERAL</span> : null}
      {onExport ? <span>HEADER_EXPORT</span> : null}
    </div>
  ),
}));
vi.mock("../components/QuestionInput", () => ({
  default: ({ onSend }) => <button onClick={() => onSend("hi", [], [])}>SEND</button>,
}));

import ConversationPage from "./ConversationPage.jsx";
import apiClient from "../services/api/client";
import {
  isEphemeral,
  leaveConversation,
  resetEphemeralConversations,
} from "../utils/ephemeralConversations";
import { abortTurns } from "../utils/conversationTurns";

const detail = (ephemeral) => ({
  id: 7,
  llm_id: 1,
  temperature: 0.7,
  top_p: 0.9,
  max_tokens: 512,
  custom_prompt: "",
  ephemeral,
});

const encoder = new TextEncoder();

/** A /query answer that streams one chunk, then waits until aborted. */
const hangingAnswer = (signal) => {
  let sent = false;
  const read = () => {
    if (!sent) {
      sent = true;
      return Promise.resolve({
        done: false,
        value: encoder.encode('{"t":"answer","text":"Partial"}\n'),
      });
    }
    return new Promise((_resolve, reject) => {
      const fail = () => reject(new DOMException("The user aborted a request.", "AbortError"));
      if (signal?.aborted) fail();
      else signal?.addEventListener("abort", fail);
    });
  };
  return { ok: true, body: { getReader: () => ({ read }) } };
};

const routeFetch =
  ({ ephemeral }) =>
  async (url, options = {}) => {
    const u = String(url);
    if (u.includes("generate_title")) return { ok: false, status: 500 };
    if (u.includes("/query")) return hangingAnswer(options.signal);
    if (options.method === "DELETE") return { ok: true, status: 200 };
    if (u.endsWith("/conversations/7")) return { ok: true, json: async () => detail(ephemeral) };
    return { ok: true, json: async () => [] };
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
  resetEphemeralConversations();
  tracedFetchMock.mockReset();
  apiClient.get.mockReset();
  apiClient.get.mockImplementation(async () => []);
  Object.values(logMock).forEach((fn) => fn.mockReset());
});
afterEach(() => {
  cleanup();
});

describe("ConversationPage, ephemeral conversation", () => {
  it("shows the badge, offers no export and remembers the conversation as ephemeral", async () => {
    tracedFetchMock.mockImplementation(routeFetch({ ephemeral: true }));
    await renderAndSettle();

    expect(await screen.findByText("HEADER_EPHEMERAL")).toBeTruthy();
    expect(screen.queryByText("HEADER_EXPORT")).toBeNull();
    expect(isEphemeral("7")).toBe(true);
  });

  it("keeps a regular conversation exportable and unbadged", async () => {
    tracedFetchMock.mockImplementation(routeFetch({ ephemeral: false }));
    await renderAndSettle();

    expect(await screen.findByText("HEADER_EXPORT")).toBeTruthy();
    expect(screen.queryByText("HEADER_EPHEMERAL")).toBeNull();
    expect(isEphemeral("7")).toBe(false);
  });

  it("asks for no title on the first message", async () => {
    tracedFetchMock.mockImplementation(routeFetch({ ephemeral: true }));
    await renderAndSettle();
    await screen.findByText("HEADER_EPHEMERAL");

    fireEvent.click(screen.getByText("SEND"));

    await waitFor(() => expect(calls("/query")).toBe(1));
    expect(calls("generate_title")).toBe(0);
  });

  it("aborts the streaming answer on leave, with no red bubble and no error record", async () => {
    tracedFetchMock.mockImplementation(routeFetch({ ephemeral: true }));
    await renderAndSettle();
    await screen.findByText("HEADER_EPHEMERAL");

    fireEvent.click(screen.getByText("SEND"));
    await waitFor(() => expect(screen.getAllByTestId("answer").at(-1).textContent).toBe("Partial"));
    const querySignal = tracedFetchMock.mock.calls.find(([u]) => String(u).includes("/query"))[1]
      .signal;
    expect(querySignal.aborted).toBe(false);

    await act(async () => {
      await leaveConversation("7");
    });

    expect(querySignal.aborted).toBe(true);
    expect(calls("store_error_message")).toBe(0);
    await act(async () => {});
    const answers = screen.getAllByTestId("answer").map((a) => a.textContent);
    expect(answers.join(" ")).not.toContain("ERROR");
    expect(logMock.error).not.toHaveBeenCalled();
  });

  it("lets a delete abort the streaming answer of a regular conversation, silently", async () => {
    tracedFetchMock.mockImplementation(routeFetch({ ephemeral: false }));
    await renderAndSettle();
    await screen.findByText("HEADER_EXPORT");

    fireEvent.click(screen.getByText("SEND"));
    await waitFor(() => expect(screen.getAllByTestId("answer").at(-1).textContent).toBe("Partial"));
    const querySignal = tracedFetchMock.mock.calls.find(([u]) => String(u).includes("/query"))[1]
      .signal;

    // What the history row does before its DELETE.
    await act(async () => {
      abortTurns("7");
    });

    expect(querySignal.aborted).toBe(true);
    await act(async () => {});
    expect(calls("store_error_message")).toBe(0);
    expect(logMock.error).not.toHaveBeenCalled();
  });
});
