// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor } from "@testing-library/react";

// #690.2: an Arena panel shows a curated turn the way a conversation does --
// the sentence in the interface language, from the kind and the numbers the
// stream's event carries -- and a generic failure with its translated
// headline, in red; never the raw sentinel line.

const { tracedFetchMock } = vi.hoisted(() => ({ tracedFetchMock: vi.fn() }));

vi.mock("../services/api/client", () => ({
  default: { get: vi.fn() },
  apiClient: { get: vi.fn() },
  tracedFetch: tracedFetchMock,
}));

vi.mock("../components/Sidebar", () => ({ default: () => null }));
vi.mock("../components/GradientBox", () => ({ default: ({ children }) => <div>{children}</div> }));
vi.mock("../components/MarkdownRenderer", () => ({
  default: ({ content }) => <div data-testid="answer">{content}</div>,
}));
vi.mock("../components/modals/CustomizePromptModal", () => ({ default: () => null }));
vi.mock("../components/HeaderBar", () => ({ default: () => null }));
vi.mock("../components/QuestionInput", () => ({
  default: ({ onSend }) => <button onClick={() => onSend("plain question", [], [])}>SEND</button>,
}));

import apiClient from "../services/api/client";
import i18n from "../i18n";
import ArenaPage from "./ArenaPage.jsx";
import { GENERIC_ERROR_LINE } from "../utils/messageContent";

const SENTINEL = "[ERROR_MESSAGE_SYSTEM]";
const WINDOW_EN =
  "This conversation no longer fits the model's context window (the request needs about 9030 tokens; the window holds 8192). Start a new conversation or send less at once.";
const WINDOW_ES =
  "Esta conversación ya no cabe en la ventana de contexto del modelo (la solicitud necesita unos 9030 tokens; la ventana admite 8192). Empieza una conversación nueva o envía menos texto a la vez.";
const STOP_EN =
  "The model finished its turn without writing an answer. Send a follow-up asking it to continue.";
const STOP_ZH = "模型结束了本轮，但没有写出回答。请发送后续消息，请它继续。";
const APOLOGY_FR =
  "Je suis désolé, une erreur est survenue lors de la génération de la réponse. Veuillez reposer votre question.";

const encoder = new TextEncoder();
const streamOf = (...events) => {
  const queue = events.map((e) => ({
    done: false,
    value: encoder.encode(`${JSON.stringify(e)}\n`),
  }));
  queue.push({ done: true, value: undefined });
  return { ok: true, body: { getReader: () => ({ read: async () => queue.shift() }) } };
};

const sendWith = async (...events) => {
  tracedFetchMock.mockImplementation(async () => streamOf(...events));
  apiClient.get.mockResolvedValue([{ id: 1, name: "m1", supports_vision: false }]);
  render(<ArenaPage />);
  fireEvent.click(await screen.findByText("SEND"));
};

beforeEach(() => {
  tracedFetchMock.mockReset();
  apiClient.get.mockReset();
});

afterEach(async () => {
  cleanup();
  await i18n.changeLanguage("en");
});

describe("an Arena panel", () => {
  it("shows a curated failure in the interface language, in red, below the partial answer", async () => {
    await i18n.changeLanguage("es");
    await sendWith(
      { t: "answer", text: "Media respuesta" },
      {
        t: "error",
        text: `${SENTINEL} ${WINDOW_EN}`,
        curated: true,
        curated_error: {
          kind: "context_overflow",
          params: { prompt_tokens: 9030, context_tokens: 8192 },
          text: WINDOW_EN,
        },
      }
    );

    const bubbles = await screen.findAllByText(`Media respuesta ❌ ${WINDOW_ES}`);
    expect(bubbles).toHaveLength(2); // both panels run the same model
    expect(bubbles[0].textContent).toBe(`Media respuesta\n\n❌ ${WINDOW_ES}`);
    expect(bubbles[0].closest(".text-red-400")).toBeTruthy();
    expect(screen.queryByText(/ERROR_MESSAGE_SYSTEM/)).toBeNull();
  });

  it("translates an empty-answer line as a normal answer", async () => {
    await i18n.changeLanguage("zh");
    await sendWith({
      t: "answer",
      text: STOP_EN,
      curated_error: { kind: "empty_answer_stop", params: {}, text: STOP_EN },
    });

    const answers = await screen.findAllByText(STOP_ZH);
    expect(answers[0].dataset.testid).toBe("answer");
  });

  it("keeps the request-failure marker that followed an empty-answer line", async () => {
    await i18n.changeLanguage("zh");
    // One reader per panel: the line arrives, then the connection drops.
    tracedFetchMock.mockImplementation(async () => {
      const read = vi.fn();
      read.mockResolvedValueOnce({
        done: false,
        value: encoder.encode(
          `${JSON.stringify({
            t: "answer",
            text: STOP_EN,
            curated_error: { kind: "empty_answer_stop", params: {}, text: STOP_EN },
          })}\n`
        ),
      });
      read.mockRejectedValueOnce(new Error("socket reset"));
      return { ok: true, body: { getReader: () => ({ read }) } };
    });
    apiClient.get.mockResolvedValue([{ id: 1, name: "m1", supports_vision: false }]);
    render(<ArenaPage />);
    fireEvent.click(await screen.findByText("SEND"));

    const bubbles = await screen.findAllByText(`${STOP_ZH}${i18n.t("arena:panel.error")}`);
    expect(bubbles).toHaveLength(2);
  });

  it("shows what one network read carried at once, and ends the run with it", async () => {
    // 40 answer events in a single read: the panel must not type them out one
    // per tick (20 per second), which also held the run open for two seconds.
    const words = Array.from({ length: 40 }, (_, i) => `w${i} `);
    const body = words.map((text) => `${JSON.stringify({ t: "answer", text })}\n`).join("");
    tracedFetchMock.mockImplementation(async () => {
      const queue = [
        { done: false, value: encoder.encode(body) },
        { done: true, value: undefined },
      ];
      return { ok: true, body: { getReader: () => ({ read: async () => queue.shift() }) } };
    });
    apiClient.get.mockResolvedValue([{ id: 1, name: "m1", supports_vision: false }]);
    render(<ArenaPage />);
    fireEvent.click(await screen.findByText("SEND"));

    const answers = await screen.findAllByText(words.join("").trim(), {}, { timeout: 400 });
    expect(answers).toHaveLength(2);
    await waitFor(
      () => expect(screen.queryByRole("button", { name: "Stop generation" })).toBeNull(),
      {
        timeout: 400,
      }
    );
  });

  it("shows a generic failure with its translated headline", async () => {
    await i18n.changeLanguage("fr");
    await sendWith({ t: "error", text: GENERIC_ERROR_LINE, detail: "boom", request_id: "be-1" });

    expect((await screen.findAllByText(`❌ ${APOLOGY_FR}`)).length).toBe(2);
  });
});
