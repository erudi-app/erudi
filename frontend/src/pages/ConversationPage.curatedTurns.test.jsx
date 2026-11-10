// @vitest-environment jsdom
import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent, waitFor, act } from "@testing-library/react";

// Curated turns in the interface language (#690.2) and the death of the
// inference process behind Details (#690.1):
//  - a curated turn shows `chat:curated.<kind>` in the interface language,
//    numbers in its format, live (from the wire event's `curated_error`) and
//    after a reload (from the trace), and follows a language switch;
//  - an empty-answer line is translated the same way, as a normal answer;
//  - a turn stored before this version (no `curated_error`) shows as stored;
//  - a generic failure whose inference process died shows that cause under
//    Details.

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
vi.mock("../components/TraceStrip", () => ({ default: () => null }));
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

const SENTINEL = "[ERROR_MESSAGE_SYSTEM]";
const MEMORY_EN =
  "This conversation no longer fits what your machine's memory can hold for this model (the request needs about 13531 tokens; memory allows 8584). Start a new conversation, send less at once, or switch to a smaller model.";
const MEMORY_FR =
  "Cette conversation dépasse ce que la mémoire de votre machine peut contenir pour ce modèle (la requête demande environ 13 531 tokens ; la mémoire en permet 8 584). Démarrez une nouvelle conversation, envoyez moins de texte à la fois ou passez à un modèle plus petit.";
const MEMORY_ES =
  "Esta conversación ya no cabe en lo que la memoria de tu equipo puede contener para este modelo (la solicitud necesita unos 13.531 tokens; la memoria permite 8584). Empieza una conversación nueva, envía menos texto a la vez o cambia a un modelo más pequeño.";
const MEMORY_ZH =
  "此对话已超出本机内存可为该模型容纳的范围（请求约需 13,531 个 token；内存允许 8,584 个）。请开始新对话、一次少发送一些内容，或改用更小的模型。";
const STOP_EN =
  "The model finished its turn without writing an answer. Send a follow-up asking it to continue.";
const STOP_FR =
  "Le modèle a terminé son tour sans écrire de réponse. Envoyez un message lui demandant de continuer.";
const LOOP_ZH = "无法为此请求得出最终回答（模型一直在重试却没有进展）。请尝试换一种方式提问。";
const CHILD_DEATH = "mlx_vlm.server child is dead (pid 61112, port 27300, exit code -9)";

const LOOP_EN = "I couldn't reach a final answer for this request.";
const APOLOGY_FR =
  "Je suis désolé, une erreur est survenue lors de la génération de la réponse. Veuillez reposer votre question.";

// The trace carries the stored English line itself (`text`): only that line
// is replaced at display.
const memoryKind = {
  kind: "memory_overflow",
  params: { prompt_tokens: 13531, context_tokens: 8584 },
  text: MEMORY_EN,
};
const stopKind = { kind: "empty_answer_stop", params: {}, text: STOP_EN };

const conversationDetail = {
  id: 7,
  llm_id: 1,
  temperature: 0.7,
  top_p: 0.9,
  max_tokens: 512,
  custom_prompt: "",
};

const encoder = new TextEncoder();
const streamOf = (...events) => {
  const queue = events.map((e) => ({
    done: false,
    value: encoder.encode(`${JSON.stringify(e)}\n`),
  }));
  queue.push({ done: true, value: undefined });
  return { ok: true, body: { getReader: () => ({ read: async () => queue.shift() }) } };
};

const routeFetch =
  ({ query = () => streamOf(), messages = [] } = {}) =>
  async (url) => {
    const u = String(url);
    if (u.includes("/query")) return query();
    if (u.includes("generate_title")) return streamOf();
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

const reloadWith = async (rows) => {
  tracedFetchMock.mockImplementation(routeFetch({ messages: rows }));
  apiClient.get.mockImplementation(async () => rows);
  await renderAndSettle();
};

const turn = (content, trace) => [
  { id: 11, sender: "user", content: "hi", starred: false },
  { id: 12, sender: "llm", content, starred: false, trace },
];

beforeEach(() => {
  vi.clearAllMocks();
  Element.prototype.scrollTo = () => {};
  apiClient.get.mockImplementation(async () => []);
});

afterEach(async () => {
  cleanup();
  await i18n.changeLanguage("en");
});

describe("a curated failure, live", () => {
  it("shows the sentence in the interface language with its numbers, without Details", async () => {
    await i18n.changeLanguage("fr");
    tracedFetchMock.mockImplementation(
      routeFetch({
        query: () =>
          streamOf(
            {
              t: "error",
              text: `${SENTINEL} ${MEMORY_EN}`,
              curated: true,
              curated_error: memoryKind,
            },
            { t: "done" }
          ),
      })
    );
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));

    // The matcher compares against the whitespace-normalized text: the French
    // group separator (a narrow no-break space) reads as a space there.
    const bubble = await screen.findByText(`❌ ${MEMORY_FR.replace(/\s+/g, " ")}`);
    expect(bubble.textContent).toBe(`❌ ${MEMORY_FR}`);
    expect(screen.queryByText(/no longer fits/)).toBeNull();
    expect(screen.queryByRole("button", { name: "Détails" })).toBeNull();
    const failures = (fn) => fn.mock.calls.filter(([m]) => /Generation failed/.test(m));
    expect(failures(loggerMock.warn)).toHaveLength(1);
    expect(failures(loggerMock.error)).toHaveLength(0);
  });

  it("keeps what streamed before it above the translated line", async () => {
    await i18n.changeLanguage("es");
    tracedFetchMock.mockImplementation(
      routeFetch({
        query: () =>
          streamOf(
            { t: "answer", text: "Media respuesta" },
            {
              t: "error",
              text: `${SENTINEL} ${MEMORY_EN}`,
              curated: true,
              curated_error: memoryKind,
            },
            { t: "done" }
          ),
      })
    );
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));

    const bubble = await screen.findByText(/Media respuesta/);
    expect(bubble.textContent).toBe(`Media respuesta\n\n❌ ${MEMORY_ES}`);
  });
});

describe("an empty-answer line, live", () => {
  it("is translated as a normal answer", async () => {
    await i18n.changeLanguage("fr");
    tracedFetchMock.mockImplementation(
      routeFetch({
        query: () =>
          streamOf(
            { t: "thinking", text: "hmm" },
            { t: "answer", text: STOP_EN, curated_error: stopKind },
            { t: "done" }
          ),
      })
    );
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));

    expect((await screen.findByTestId("answer")).textContent).toBe(STOP_FR);
  });

  it("keeps a failure that followed it in the same turn", async () => {
    await i18n.changeLanguage("fr");
    tracedFetchMock.mockImplementation(
      routeFetch({
        query: () =>
          streamOf(
            { t: "answer", text: STOP_EN, curated_error: stopKind },
            { t: "error", text: GENERIC_ERROR_LINE, detail: "boom", request_id: "be-1" },
            { t: "done" }
          ),
      })
    );
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));

    const bubble = await screen.findByText(/Le modèle a terminé son tour/);
    expect(bubble.textContent).toBe(`${STOP_FR}\n\n❌ ${APOLOGY_FR}`);
  });

  it("keeps the line of a stream dropped after it", async () => {
    await i18n.changeLanguage("fr");
    const read = vi.fn();
    read.mockResolvedValueOnce({
      done: false,
      value: encoder.encode(
        `${JSON.stringify({ t: "answer", text: STOP_EN, curated_error: stopKind })}\n`
      ),
    });
    read.mockRejectedValueOnce(new Error("socket reset"));
    tracedFetchMock.mockImplementation(
      routeFetch({ query: () => ({ ok: true, body: { getReader: () => ({ read }) } }) })
    );
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));

    const bubble = await screen.findByText(/Le modèle a terminé son tour/);
    expect(bubble.textContent).toBe(
      `${STOP_FR}\n\n❌ Connexion interrompue pendant la génération de la réponse.`
    );
  });
});

describe("a curated turn, reloaded", () => {
  it("is translated from its trace and follows a language switch", async () => {
    await i18n.changeLanguage("es");
    await reloadWith(turn(`${SENTINEL} ${MEMORY_EN}`, [{ t: "curated_error", ...memoryKind }]));

    expect(await screen.findByText(`❌ ${MEMORY_ES}`)).toBeTruthy();

    await act(async () => {
      await i18n.changeLanguage("zh");
    });

    expect(await screen.findByText(`❌ ${MEMORY_ZH}`)).toBeTruthy();
  });

  it("translates a kind without numbers", async () => {
    await i18n.changeLanguage("zh");
    await reloadWith(
      turn(`${SENTINEL} ${LOOP_EN}`, [
        { t: "curated_error", kind: "loop_limit", params: {}, text: LOOP_EN },
      ])
    );

    expect(await screen.findByText(`❌ ${LOOP_ZH}`)).toBeTruthy();
  });

  it("translates an empty-answer line below its notice", async () => {
    await i18n.changeLanguage("fr");
    await reloadWith(
      turn(`*A notice.*\n\n${STOP_EN}`, [
        { t: "thinking", text: "hmm" },
        { t: "curated_error", ...stopKind },
      ])
    );

    expect((await screen.findByTestId("answer")).textContent).toBe(`*A notice.*\n\n${STOP_FR}`);
  });

  it("translates an empty-answer line and keeps the failure stored after it", async () => {
    await i18n.changeLanguage("fr");
    await reloadWith(
      turn(`${STOP_EN}\n\n${GENERIC_ERROR_LINE}`, [
        { t: "curated_error", ...stopKind },
        { t: "error_detail", text: "boom", request_id: "be-1" },
      ])
    );

    const bubble = await screen.findByText(/Le modèle a terminé son tour/);
    expect(bubble.textContent).toBe(`${STOP_FR}\n\n❌ ${APOLOGY_FR}`);
  });

  it("shows the content as stored when its line is not in it", async () => {
    await i18n.changeLanguage("fr");
    await reloadWith(
      turn(`${SENTINEL} Another sentence.`, [{ t: "curated_error", ...memoryKind }])
    );

    expect(await screen.findByText("❌ Another sentence.")).toBeTruthy();
  });

  it("shows a turn stored before this version as stored", async () => {
    await i18n.changeLanguage("fr");
    await reloadWith(turn(`${SENTINEL} ${MEMORY_EN}`, null));

    expect(await screen.findByText(`❌ ${MEMORY_EN}`)).toBeTruthy();
  });

  it("shows a kind this interface does not know as stored", async () => {
    await i18n.changeLanguage("fr");
    await reloadWith(
      turn(`${SENTINEL} A sentence from a newer version.`, [
        {
          t: "curated_error",
          kind: "a_future_kind",
          params: {},
          text: "A sentence from a newer version.",
        },
      ])
    );

    expect(await screen.findByText("❌ A sentence from a newer version.")).toBeTruthy();
  });
});

describe("an inference process that died", () => {
  it("is the cause under Details", async () => {
    tracedFetchMock.mockImplementation(
      routeFetch({
        query: () =>
          streamOf(
            { t: "answer", text: "Half" },
            {
              t: "error",
              text: GENERIC_ERROR_LINE,
              detail: CHILD_DEATH,
              request_id: "be-0a1b2c3d",
            },
            { t: "done" }
          ),
      })
    );
    await renderAndSettle();

    fireEvent.click(screen.getByText("SEND"));
    fireEvent.click(await screen.findByRole("button", { name: "Details" }));

    expect(screen.getByText(CHILD_DEATH)).toBeTruthy();
  });
});
