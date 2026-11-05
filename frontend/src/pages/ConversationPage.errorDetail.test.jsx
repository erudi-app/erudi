// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, act } from "@testing-library/react";

// The exact cause behind a generic failure (#615):
//  - a generic failure keeps its English apology as stored content and shows
//    the headline in the interface language, live, after a reload and after a
//    language switch;
//  - a "Details" control under it shows the cleaned cause and the request id
//    (live from the wire `error` event, after a reload from the trace's
//    `error_detail`); a turn stored before it existed has the headline only;
//  - a curated turn reads as written, without Details;
//  - the page logs a curated failure at WARNING and a generic one at ERROR;
//  - text streamed before the failure stays above the headline.

const { tracedFetchMock, navigateMock, locationMock, loggerMock } = vi.hoisted(() => ({
  tracedFetchMock: vi.fn(),
  navigateMock: vi.fn(),
  locationMock: { pathname: "/conversation/7", state: null },
  loggerMock: { log: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn() },
}));

vi.mock("../services/api/client", () => ({
  default: { get: vi.fn(async () => []) },
  apiClient: { get: vi.fn(async () => []) },
  tracedFetch: tracedFetchMock,
}));

vi.mock("../utils/logger", async (importOriginal) => {
  const mod = await importOriginal();
  return { ...mod, createLogger: () => loggerMock };
});

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
import i18n from "../i18n";
import { GENERIC_ERROR_LINE } from "../utils/messageContent";

const APOLOGY_EN =
  "I apologize, but I encountered an error while generating a response. Please try asking your question again.";
const APOLOGY_FR =
  "Je suis désolé, une erreur est survenue lors de la génération de la réponse. Veuillez reposer votre question.";
const CAUSE = "Message content too long (max 32K chars)";
const REQUEST_ID = "be-1f2e3d4c";

const conversationDetail = {
  id: 7,
  llm_id: 1,
  temperature: 0.7,
  top_p: 0.9,
  max_tokens: 512,
  custom_prompt: "",
};

const encoder = new TextEncoder();
/** Finite stream of NDJSON lines. */
const streamOf = (...events) => {
  const queue = events.map((e) => ({
    done: false,
    value: encoder.encode(`${typeof e === "string" ? e : JSON.stringify(e)}\n`),
  }));
  queue.push({ done: true, value: undefined });
  return { ok: true, body: { getReader: () => ({ read: async () => queue.shift() }) } };
};
const emptyStream = () => streamOf();

const routeFetch =
  ({ query = emptyStream, messages = [] } = {}) =>
  async (url) => {
    const u = String(url);
    if (u.includes("/query")) return query();
    if (u.includes("generate_title")) return emptyStream();
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

const send = async () => {
  fireEvent.click(screen.getByText("SEND"));
};

const failedTurn = (trace) => [
  { id: 11, sender: "user", content: "hi", starred: false },
  { id: 12, sender: "llm", content: GENERIC_ERROR_LINE, starred: false, trace },
];

const reloadWith = async (rows) => {
  tracedFetchMock.mockImplementation(routeFetch({ messages: rows }));
  apiClient.get.mockImplementation(async () => rows);
  await renderAndSettle();
};

const genericWire = {
  t: "error",
  text: GENERIC_ERROR_LINE,
  detail: CAUSE,
  request_id: REQUEST_ID,
};

beforeEach(() => {
  vi.clearAllMocks();
  Element.prototype.scrollTo = () => {};
  apiClient.get.mockImplementation(async () => []);
});

afterEach(async () => {
  cleanup();
  await i18n.changeLanguage("en");
});

describe("a generic failure, live", () => {
  it("shows the headline and a Details control with the cause and the request id", async () => {
    tracedFetchMock.mockImplementation(
      routeFetch({ query: () => streamOf(genericWire, { t: "done" }) })
    );
    await renderAndSettle();

    await send();

    expect(await screen.findByText(`❌ ${APOLOGY_EN}`)).toBeTruthy();
    expect(screen.queryByText(CAUSE)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Details" }));
    expect(screen.getByText(CAUSE)).toBeTruthy();
    expect(screen.getByText(REQUEST_ID)).toBeTruthy();
  });

  it("keeps the partial reply above the headline", async () => {
    tracedFetchMock.mockImplementation(
      routeFetch({
        query: () => streamOf({ t: "answer", text: "Half an answer" }, genericWire, { t: "done" }),
      })
    );
    await renderAndSettle();

    await send();

    const bubble = await screen.findByText(/Half an answer/);
    expect(bubble.textContent).toBe(`Half an answer\n\n❌ ${APOLOGY_EN}`);
    expect(screen.getByRole("button", { name: "Details" })).toBeTruthy();
  });

  it("logs a generic failure at ERROR", async () => {
    tracedFetchMock.mockImplementation(
      routeFetch({ query: () => streamOf(genericWire, { t: "done" }) })
    );
    await renderAndSettle();

    await send();
    await screen.findByText(`❌ ${APOLOGY_EN}`);

    const failures = (fn) => fn.mock.calls.filter(([m]) => /Generation failed/.test(m));
    expect(failures(loggerMock.error)).toHaveLength(1);
    expect(failures(loggerMock.warn)).toHaveLength(0);
  });
});

describe("a curated failure", () => {
  const curated = {
    t: "error",
    text: "[ERROR_MESSAGE_SYSTEM] The model did not start answering in time.",
    curated: true,
  };

  it("reads as written, without Details, and is logged at WARNING", async () => {
    tracedFetchMock.mockImplementation(
      routeFetch({ query: () => streamOf(curated, { t: "done" }) })
    );
    await renderAndSettle();

    await send();

    expect(await screen.findByText("❌ The model did not start answering in time.")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Details" })).toBeNull();
    const failures = (fn) => fn.mock.calls.filter(([m]) => /Generation failed/.test(m));
    expect(failures(loggerMock.warn)).toHaveLength(1);
    expect(failures(loggerMock.error)).toHaveLength(0);
  });
});

describe("a generic failure, reloaded", () => {
  it("shows the headline and Details from the persisted trace", async () => {
    await reloadWith(failedTurn([{ t: "error_detail", text: CAUSE, request_id: REQUEST_ID }]));

    expect(await screen.findByText(`❌ ${APOLOGY_EN}`)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Details" }));
    expect(screen.getByText(CAUSE)).toBeTruthy();
    expect(screen.getByText(REQUEST_ID)).toBeTruthy();
  });

  it("follows a language switch and keeps the cause untranslated", async () => {
    await reloadWith(failedTurn([{ t: "error_detail", text: CAUSE, request_id: REQUEST_ID }]));
    await screen.findByText(`❌ ${APOLOGY_EN}`);

    await act(async () => {
      await i18n.changeLanguage("fr");
    });

    expect(await screen.findByText(`❌ ${APOLOGY_FR}`)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Détails" }));
    expect(screen.getByText(CAUSE)).toBeTruthy();
  });

  it("translates the headline of a turn stored before 1.1.4, with no Details", async () => {
    await reloadWith(failedTurn(null));
    await act(async () => {
      await i18n.changeLanguage("fr");
    });

    expect(await screen.findByText(`❌ ${APOLOGY_FR}`)).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Détails" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Details" })).toBeNull();
  });

  it("keeps a stored curated turn as written", async () => {
    await reloadWith([
      { id: 11, sender: "user", content: "hi", starred: false },
      {
        id: 12,
        sender: "llm",
        content: "[ERROR_MESSAGE_SYSTEM] The tool loop hit its limit.",
        starred: false,
        trace: null,
      },
    ]);
    await act(async () => {
      await i18n.changeLanguage("fr");
    });

    expect(await screen.findByText("❌ The tool loop hit its limit.")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Détails" })).toBeNull();
  });
});
