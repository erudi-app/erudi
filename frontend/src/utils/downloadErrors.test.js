import { describe, it, expect, beforeAll, afterAll } from "vitest";
import i18n from "../i18n";
import fr from "../locales/fr/downloads.json";
import { DOWNLOAD_ERROR_KEYS, downloadFailureMessage } from "./downloadErrors";

// The codes the backend stores on a failed download job (#780),
// DOWNLOAD_ERROR_CODES in backend/src/domains/llms/download_errors.py.
const BACKEND_CODES = [
  "offline",
  "disk_full",
  "gated",
  "not_found",
  "rate_limited",
  "hub_unavailable",
  "interrupted",
  "no_engine_artifact",
  "raw_gguf_chunks",
  "incomplete_split",
  "unsafe_file_name",
  "not_runnable",
  "integrity_failed",
  "unexpected",
];

const lookup = (catalog, key) =>
  key
    .replace(/^downloads:/, "")
    .split(".")
    .reduce((node, part) => node?.[part], catalog);

describe("downloadFailureMessage", () => {
  beforeAll(async () => {
    await i18n.changeLanguage("fr");
  });
  afterAll(async () => {
    await i18n.changeLanguage("en");
  });

  it("has a sentence for every code the backend stores", () => {
    expect(Object.keys(DOWNLOAD_ERROR_KEYS).sort()).toEqual([...BACKEND_CODES].sort());
  });

  it.each(BACKEND_CODES)("says %s in the interface language", (code) => {
    const expected = lookup(fr, DOWNLOAD_ERROR_KEYS[code]);
    expect(typeof expected).toBe("string");
    expect(downloadFailureMessage(code)).toBe(expected);
  });

  it("says the generic sentence for a code it does not know", () => {
    expect(downloadFailureMessage("a_code_from_a_newer_backend")).toBe(
      fr.errors.failedUnexpectedly
    );
  });

  it("says the generic sentence for a job without a code", () => {
    expect(downloadFailureMessage(null)).toBe(fr.errors.failedUnexpectedly);
    expect(downloadFailureMessage(undefined)).toBe(fr.errors.failedUnexpectedly);
  });
});
