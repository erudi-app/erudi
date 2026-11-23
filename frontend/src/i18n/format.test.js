import { describe, it, expect, beforeEach } from "vitest";
import i18n from "./index";
import {
  formatNumber,
  formatFileSize,
  formatDate,
  formatPercent,
  formatList,
  currentLocale,
} from "./format";
import { displayModelSize } from "../utils/modelSize";

describe("format helpers follow the active language", () => {
  beforeEach(async () => {
    await i18n.changeLanguage("en");
  });

  it("maps the app language to an Intl locale", async () => {
    expect(currentLocale()).toBe("en-US");
    await i18n.changeLanguage("fr");
    expect(currentLocale()).toBe("fr-FR");
    await i18n.changeLanguage("es");
    expect(currentLocale()).toBe("es-ES");
    await i18n.changeLanguage("zh");
    expect(currentLocale()).toBe("zh-CN");
  });

  it("formats numbers with the locale's separators", async () => {
    expect(formatNumber(1234.5)).toBe("1,234.5");
    await i18n.changeLanguage("fr");
    // French uses a narrow no-break space as thousands separator and a comma.
    expect(formatNumber(1234.5)).toBe("1 234,5");
  });

  it("formats sizes in GB/MB with locale decimals and translated units", async () => {
    expect(formatFileSize(1_500_000_000)).toBe("1.5 GB");
    expect(formatFileSize(512_000_000)).toBe("512 MB");
    expect(formatFileSize(0)).toBe("0 B");
    await i18n.changeLanguage("fr");
    expect(formatFileSize(1_500_000_000)).toBe("1,5 Go");
  });

  it("counts files and transfers in decimal units, like the model card (#753)", async () => {
    expect(formatFileSize(2_279_000_000)).toBe("2.3 GB");
    await i18n.changeLanguage("fr");
    expect(formatFileSize(2_279_000_000)).toBe("2,3 Go");
  });

  it.each(["en", "fr", "es", "zh"])(
    "gives the card's size for the same bytes in %s (#753)",
    async (lang) => {
      await i18n.changeLanguage(lang);
      const bytes = 2_279_000_000;
      expect(formatFileSize(bytes)).toBe(displayModelSize({ artifact_size_bytes: bytes }));
    }
  );

  it("formats percentages", async () => {
    expect(formatPercent(42.256)).toBe("42.3%");
    await i18n.changeLanguage("fr");
    expect(formatPercent(42.256)).toBe("42,3 %");
  });

  it("formats dates in the active locale", async () => {
    const d = new Date(2026, 0, 31, 14, 5);
    expect(formatDate(d, { dateStyle: "medium" })).toBe("Jan 31, 2026");
    await i18n.changeLanguage("fr");
    expect(formatDate(d, { dateStyle: "medium" })).toBe("31 janv. 2026");
  });

  it("is safe on non-numeric input", () => {
    expect(formatNumber(undefined)).toBe("");
    expect(formatFileSize(null)).toBe("");
    expect(formatDate("not a date")).toBe("");
  });

  it("joins a list with the active language's separators", async () => {
    const files = ["a.pdf", "b.txt", "c.docx"];
    expect(formatList(files)).toBe("a.pdf, b.txt, and c.docx");
    await i18n.changeLanguage("fr");
    expect(formatList(files)).toBe("a.pdf, b.txt et c.docx");
    await i18n.changeLanguage("zh");
    expect(formatList(files)).toBe("a.pdf、b.txt和c.docx");
    expect(formatList(["a.pdf"])).toBe("a.pdf");
  });
});
