import { describe, it, expect, afterEach } from "vitest";
import i18n from "../i18n";
import { curatedTurnText, withCuratedTurn } from "./curatedTurns";
import { curatedErrorOf } from "./traceEvents";
import { ERROR_SENTINEL as SENTINEL } from "./messageContent";

// #690.2: a curated turn is translated at display from the trace's
// {"t":"curated_error","kind","params","text"}, numbers in the interface
// language's format, replacing exactly the stored English line (`text`); a
// kind the interface does not know, or a line not found, keeps the content as
// stored.

const memory = {
  t: "curated_error",
  kind: "memory_overflow",
  params: { prompt_tokens: 13531, context_tokens: 8584 },
};

afterEach(async () => {
  await i18n.changeLanguage("en");
});

describe("curatedErrorOf", () => {
  it("finds the curated event of a trace", () => {
    expect(curatedErrorOf([{ t: "thinking", text: "x" }, memory])).toBe(memory);
  });

  it("is null for a turn without one (an older turn, an ordinary answer)", () => {
    expect(curatedErrorOf(null)).toBeNull();
    expect(curatedErrorOf([])).toBeNull();
    expect(curatedErrorOf([{ t: "error_detail", text: "boom" }])).toBeNull();
  });
});

describe("curatedTurnText", () => {
  it.each([
    [
      "en",
      "This conversation no longer fits what your machine's memory can hold for this model (the request needs about 13,531 tokens; memory allows 8,584). Start a new conversation, send less at once, or switch to a smaller model.",
    ],
    [
      "fr",
      "Cette conversation dépasse ce que la mémoire de votre machine peut contenir pour ce modèle (la requête demande environ 13 531 tokens ; la mémoire en permet 8 584). Démarrez une nouvelle conversation, envoyez moins de texte à la fois ou passez à un modèle plus petit.",
    ],
    [
      "es",
      "Esta conversación ya no cabe en lo que la memoria de tu equipo puede contener para este modelo (la solicitud necesita unos 13.531 tokens; la memoria permite 8584). Empieza una conversación nueva, envía menos texto a la vez o cambia a un modelo más pequeño.",
    ],
    [
      "zh",
      "此对话已超出本机内存可为该模型容纳的范围（请求约需 13,531 个 token；内存允许 8,584 个）。请开始新对话、一次少发送一些内容，或改用更小的模型。",
    ],
  ])("writes the memory overflow in %s with its numbers formatted", async (lang, expected) => {
    await i18n.changeLanguage(lang);
    expect(curatedTurnText(memory)).toBe(expected);
  });

  it.each(["en", "fr", "es", "zh"])("has a sentence for every kind in %s", async (lang) => {
    await i18n.changeLanguage(lang);
    const kinds = {
      context_overflow: { prompt_tokens: 9030, context_tokens: 8192 },
      context_overflow_unsized: {},
      memory_overflow: { prompt_tokens: 13531, context_tokens: 8584 },
      prefill_timeout: { minutes: 5 },
      decode_timeout: { seconds: 120 },
      loop_limit: {},
      empty_answer_length: {},
      empty_answer_stop: {},
    };
    for (const [kind, params] of Object.entries(kinds)) {
      const text = curatedTurnText({ t: "curated_error", kind, params });
      expect(text, kind).toBeTruthy();
      expect(text, kind).not.toMatch(/\{\{|\}\}/);
      expect(text, kind).not.toContain(SENTINEL);
    }
  });

  it("formats the minutes of a first-token timeout", async () => {
    await i18n.changeLanguage("fr");
    expect(
      curatedTurnText({ t: "curated_error", kind: "prefill_timeout", params: { minutes: 5 } })
    ).toMatch(/^Ce modèle n'a pas commencé à répondre en 5 min sur cette machine\./);
  });

  it("is null for a kind the interface does not know", () => {
    expect(curatedTurnText({ t: "curated_error", kind: "a_future_kind", params: {} })).toBeNull();
    expect(curatedTurnText({ t: "curated_error", kind: "errors.network", params: {} })).toBeNull();
    expect(curatedTurnText(null)).toBeNull();
  });
});

describe("withCuratedTurn", () => {
  const STOP_EN =
    "The model finished its turn without writing an answer. Send a follow-up asking it to continue.";
  const STOP_FR =
    "Le modèle a terminé son tour sans écrire de réponse. Envoyez un message lui demandant de continuer.";
  const LOOP_EN = "I couldn't reach a final answer for this request.";
  const stop = [{ t: "curated_error", kind: "empty_answer_stop", params: {}, text: STOP_EN }];
  const loop = [{ t: "curated_error", kind: "loop_limit", params: {}, text: LOOP_EN }];
  const APOLOGY_EN =
    "[ERROR_MESSAGE_SYSTEM] I apologize, but I encountered an error while generating a response. Please try asking your question again.";

  it("replaces the stored red line, keeping what streamed above it", async () => {
    await i18n.changeLanguage("zh");
    expect(withCuratedTurn(`Half an answer\n\n${SENTINEL} ${LOOP_EN}`, loop)).toBe(
      `Half an answer\n\n${SENTINEL} 无法为此请求得出最终回答（模型一直在重试却没有进展）。请尝试换一种方式提问。`
    );
  });

  it("replaces an empty-answer line below its notice", async () => {
    await i18n.changeLanguage("fr");
    expect(withCuratedTurn(`*A notice.*\n\n${STOP_EN}`, stop)).toBe(`*A notice.*\n\n${STOP_FR}`);
  });

  it.each([
    ["the generic apology of a later failure", `\n\n${APOLOGY_EN}`],
    [
      "the line of a dropped stream",
      `\n\n${SENTINEL} Connection interrupted while generating response.`,
    ],
    ["the Arena's request failure", "[Erreur]"],
  ])("keeps %s that followed the line", async (_label, after) => {
    await i18n.changeLanguage("fr");
    expect(withCuratedTurn(`${STOP_EN}${after}`, stop)).toBe(`${STOP_FR}${after}`);
  });

  it("shows the content as stored when the line is not in it", async () => {
    await i18n.changeLanguage("fr");
    expect(withCuratedTurn("Something else entirely.", stop)).toBe("Something else entirely.");
  });

  it("shows the content as stored without a usable trace", async () => {
    await i18n.changeLanguage("fr");
    const stored = `${SENTINEL} ${LOOP_EN}`;
    expect(withCuratedTurn(stored, null)).toBe(stored);
    expect(withCuratedTurn(stored, [{ t: "curated_error", kind: "loop_limit", params: {} }])).toBe(
      stored
    );
    expect(
      withCuratedTurn(stored, [
        { t: "curated_error", kind: "a_future_kind", params: {}, text: LOOP_EN },
      ])
    ).toBe(stored);
  });
});
