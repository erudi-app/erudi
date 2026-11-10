import { describe, it, expect, afterEach } from "vitest";
import i18n from "../i18n";
import { curatedErrorOf, curatedTurnText, withCuratedLine } from "./curatedTurns";

// #690.2: a curated turn is translated at display from the trace's
// {"t":"curated_error","kind","params"}, numbers in the interface language's
// format; a kind the interface does not know keeps the stored English line.

const SENTINEL = "[ERROR_MESSAGE_SYSTEM]";

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

describe("withCuratedLine", () => {
  const english = `${SENTINEL} This conversation no longer fits.`;

  it("replaces the stored red line, keeping what streamed above it", () => {
    expect(withCuratedLine(`Half an answer\n\n${english}`, "Traduit.")).toBe(
      `Half an answer\n\n${SENTINEL} Traduit.`
    );
    expect(withCuratedLine(english, "Traduit.")).toBe(`${SENTINEL} Traduit.`);
  });

  it("replaces an empty-answer line, keeping the notices above it", () => {
    expect(withCuratedLine("*A notice.*\n\nThe model finished its turn.", "Traduit.")).toBe(
      "*A notice.*\n\nTraduit."
    );
    expect(withCuratedLine("The model finished its turn.", "Traduit.")).toBe("Traduit.");
  });

  it("leaves the content as stored without a translated line", () => {
    expect(withCuratedLine(english, null)).toBe(english);
  });
});
