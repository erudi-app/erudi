import { describe, it, expect, afterEach } from "vitest";
import i18n from "../i18n";
import { publishedLabel } from "./publishedDate";

// #561: a card says when the publisher released the model, as a localized
// month and year, and says nothing when the date is unknown.

afterEach(async () => {
  await i18n.changeLanguage("en");
});

describe("publishedLabel", () => {
  const cases = [
    ["en", "Published March 2026"],
    ["fr", "Publié en mars 2026"],
    ["es", "Publicado en marzo de 2026"],
    ["zh", "发布于 2026年3月"],
  ];

  it.each(cases)("renders the month and year in %s", async (lang, expected) => {
    await i18n.changeLanguage(lang);
    expect(publishedLabel("2026-03-04T10:00:00+00:00")).toBe(expected);
  });

  it("does not shift the month with the local time zone", () => {
    expect(publishedLabel("2026-03-01T00:00:00+00:00")).toBe("Published March 2026");
    expect(publishedLabel("2026-03-31T23:59:59+00:00")).toBe("Published March 2026");
  });

  it.each([null, undefined, "", "Unknown", "not a date"])("is null for %s", (value) => {
    expect(publishedLabel(value)).toBeNull();
  });
});
