import { describe, it, expect } from "vitest";
import { defaultExportName, sanitizeFileName } from "./exportFileName";

// The default name of an exported conversation (#430): the title made safe as a
// file name on every OS, then the date. Main applies it to whatever the renderer
// sends, so it must hold for any input.

describe("sanitizeFileName", () => {
  it("keeps a plain title, accents and CJK included", () => {
    expect(sanitizeFileName("Résumé du projet")).toBe("Résumé du projet");
    expect(sanitizeFileName("量子计算")).toBe("量子计算");
  });

  it("drops the characters a file name cannot hold and folds spaces", () => {
    expect(sanitizeFileName('a/b\\c:d*e?f"g<h>i|j')).toBe("a b c d e f g h i j");
    expect(sanitizeFileName("  many   spaces \n here ")).toBe("many spaces here");
  });

  it("drops control characters", () => {
    expect(sanitizeFileName("a\u0000b\u001fc\u007fd")).toBe("a b c d");
  });

  it("never yields a path, a dot name or trailing dots", () => {
    expect(sanitizeFileName("../../etc/passwd")).toBe("etc passwd");
    expect(sanitizeFileName("..")).toBe("");
    expect(sanitizeFileName("notes...")).toBe("notes");
  });

  it("refuses the names Windows reserves", () => {
    expect(sanitizeFileName("CON")).toBe("");
    expect(sanitizeFileName("lpt1")).toBe("");
    expect(sanitizeFileName("com3.txt")).toBe("");
    expect(sanitizeFileName("console")).toBe("console");
  });

  it("caps the length without cutting a character in two", () => {
    const out = sanitizeFileName("😀".repeat(200));
    expect([...out].length).toBeLessThanOrEqual(80);
    expect(out).toMatch(/^(😀)+$/u);
  });

  it("answers an empty string for anything that is not text", () => {
    expect(sanitizeFileName(undefined)).toBe("");
    expect(sanitizeFileName(42)).toBe("");
    expect(sanitizeFileName({})).toBe("");
  });
});

describe("defaultExportName", () => {
  const day = new Date(2026, 10, 3, 23, 59);

  it("is the sanitized title, the date and .md", () => {
    expect(defaultExportName("Airships: a story", day)).toBe("Airships a story 2026-11-03.md");
  });

  it("falls back to a neutral name when the title leaves nothing", () => {
    expect(defaultExportName("///", day)).toBe("conversation 2026-11-03.md");
    expect(defaultExportName(null, day)).toBe("conversation 2026-11-03.md");
  });
});
