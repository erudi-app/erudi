import { describe, it, expect, afterEach } from "vitest";
import i18n from "../i18n";
import { answerNoticeText, withNotices } from "./answerNotices";
import { errorTurnTrace } from "./traceEvents";

// #764: the notes that open an answer (an image the model cannot see, an
// attachment it could not read or got shortened) are translated at display
// from the trace's {"t":"notice","kind","params","text"}, replacing exactly
// the stored English line (`text`); a kind the interface does not know, or a
// line no longer in the content, keeps the content as stored.

const SCAN_EN = "Some attachments could not be read: scan.pdf (it holds no extractable text).";
const scan = {
  t: "notice",
  kind: "attachments_unreadable",
  params: { files: [{ name: "scan.pdf", reason: "pending_vision" }] },
  text: SCAN_EN,
};
const BIG_EN = "Attached content was shortened to fit this model's context: big.txt, huge.txt.";
const big = {
  t: "notice",
  kind: "attachments_truncated",
  params: { files: ["big.txt", "huge.txt"] },
  text: BIG_EN,
};
const IMAGES_EN = "This model doesn't support images — your image was ignored.";
const images = { t: "notice", kind: "images_ignored", params: {}, text: IMAGES_EN };

afterEach(async () => {
  await i18n.changeLanguage("en");
});

describe("answerNoticeText", () => {
  it("names each file with its reason, in the interface language", async () => {
    await i18n.changeLanguage("fr");
    expect(answerNoticeText(scan)).toBe(
      "Certaines pièces jointes n'ont pas pu être lues : scan.pdf (aucun texte extractible (un scan ou une image), et la reconnaissance de texte n'est pas encore incluse)."
    );
  });

  it("joins the names as a list of the interface language", async () => {
    await i18n.changeLanguage("fr");
    expect(answerNoticeText(big)).toBe(
      "Le contenu joint a été raccourci pour tenir dans le contexte de ce modèle : big.txt et huge.txt."
    );
  });

  it("states the file cap with its limit, without a name", () => {
    const capped = {
      ...scan,
      params: { files: [{ name: "", reason: "too_many", limit: 50 }] },
    };
    expect(answerNoticeText(capped)).toBe(
      "Some attachments could not be read: only the first 50 files were attached."
    );
  });

  it("is null for a kind or a reason this interface does not know", () => {
    expect(answerNoticeText({ ...scan, kind: "future_kind" })).toBeNull();
    expect(
      answerNoticeText({ ...scan, params: { files: [{ name: "a.pdf", reason: "later" }] } })
    ).toBeNull();
    expect(answerNoticeText(null)).toBeNull();
  });
});

describe("withNotices", () => {
  it("replaces the English line by the French sentence, keeping the italics", async () => {
    await i18n.changeLanguage("fr");
    const content = `*${IMAGES_EN}*\n\nUn carré rouge.`;
    expect(withNotices(content, [images])).toBe(
      "*Ce modèle ne prend pas en charge les images : votre image a été ignorée.*\n\nUn carré rouge."
    );
  });

  it("translates every note of a message", async () => {
    await i18n.changeLanguage("fr");
    const content = `*${SCAN_EN}*\n\n*${BIG_EN}*\n\nRéponse.`;
    const shown = withNotices(content, [scan, { t: "thinking", text: "x" }, big]);
    expect(shown).not.toContain(SCAN_EN);
    expect(shown).not.toContain(BIG_EN);
    expect(shown).toContain("*Certaines pièces jointes n'ont pas pu être lues : scan.pdf");
    expect(shown).toContain("big.txt et huge.txt.*\n\nRéponse.");
  });

  it("keeps the stored line of a kind this interface does not know", async () => {
    await i18n.changeLanguage("fr");
    const content = "*A future note.*\n\nRéponse.";
    const future = { t: "notice", kind: "future_kind", params: {}, text: "A future note." };
    expect(withNotices(content, [future])).toBe(content);
  });

  it("keeps a content that no longer holds the line as it is", async () => {
    await i18n.changeLanguage("fr");
    const edited = "Réponse retouchée.";
    expect(withNotices(edited, [scan])).toBe(edited);
  });

  it("returns the content as stored without a notice", () => {
    expect(withNotices("Hello.", null)).toBe("Hello.");
    expect(withNotices("Hello.", [{ t: "thinking", text: "x" }])).toBe("Hello.");
  });
});

describe("errorTurnTrace", () => {
  it("keeps the notices of a failed turn", () => {
    expect(errorTurnTrace([images, { t: "thinking", text: "x" }])).toEqual([images]);
  });
});
