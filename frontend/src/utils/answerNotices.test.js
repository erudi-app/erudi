import fs from "node:fs";
import { describe, it, expect, afterEach } from "vitest";
import i18n from "../i18n";
import { answerNoticeText, withNotices } from "./answerNotices";
import { errorTurnTrace } from "./traceEvents";

// #764: the notes that open an answer (an image the model cannot see, an
// attachment it could not read or got shortened) are translated at display
// from the trace's {"t":"notice","kind","params"}: the English line, rebuilt
// from the en catalog, is found in the stored content and replaced; a kind the
// interface does not know, or a line no longer in the content, keeps the
// content as stored.

// The lines the backend stores (backend/tests/fixtures/answer_notices.json):
// the en catalog must say exactly them, or the line is never found.
const SHARED = JSON.parse(
  fs.readFileSync(
    new URL("../../../backend/tests/fixtures/answer_notices.json", import.meta.url),
    "utf-8"
  )
);

const notice = (kind, params = {}) => ({ t: "notice", kind, params });
const scan = notice("attachments_unreadable", {
  files: [{ name: "scan.pdf", reason: "pending_vision" }],
});
const SCAN_EN =
  "Some attachments could not be read: scan.pdf (no extractable text (a scan or an image), and text recognition is not bundled yet).";
const big = notice("attachments_truncated", { files: ["big.txt", "huge.txt"] });
const BIG_EN = "Attached content was shortened to fit this model's context: big.txt and huge.txt.";
const images = notice("images_ignored");
const IMAGES_EN = "This model doesn't support images — your image was ignored.";

afterEach(async () => {
  await i18n.changeLanguage("en");
});

describe("answerNoticeText", () => {
  it.each(SHARED.cases.map((c) => [c.english, c]))(
    "says in English exactly what the backend stores: %s",
    async (english, { kind, params }) => {
      await i18n.changeLanguage("fr");
      expect(answerNoticeText({ kind, params }, "en")).toBe(english);
    }
  );

  it("names each file with its reason, in the interface language", async () => {
    await i18n.changeLanguage("fr");
    expect(answerNoticeText(scan)).toBe(
      "Certaines pièces jointes n'ont pas pu être lues : scan.pdf (aucun texte extractible (un scan ou une image), et la reconnaissance de texte n'est pas encore incluse)."
    );
  });

  it("joins the names as a list of the interface language and counts the rest", async () => {
    await i18n.changeLanguage("fr");
    expect(answerNoticeText(big)).toBe(
      "Le contenu joint a été raccourci pour tenir dans le contexte de ce modèle : big.txt et huge.txt."
    );
    expect(answerNoticeText(notice("attachments_truncated", { files: ["a.txt"], more: 3 }))).toBe(
      "Le contenu joint a été raccourci pour tenir dans le contexte de ce modèle : a.txt et 3 autres."
    );
  });

  it("is null for a kind or a reason this interface does not know", () => {
    expect(answerNoticeText({ ...scan, kind: "future_kind" })).toBeNull();
    expect(
      answerNoticeText(
        notice("attachments_unreadable", { files: [{ name: "a", reason: "later" }] })
      )
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

  it("shows the stored line itself in English", () => {
    const content = `*${SCAN_EN}*\n\nAnswer.`;
    expect(withNotices(content, [scan])).toBe(content);
  });

  it("keeps the stored line of a kind this interface does not know", async () => {
    await i18n.changeLanguage("fr");
    const content = "*A future note.*\n\nRéponse.";
    expect(withNotices(content, [notice("future_kind")])).toBe(content);
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
